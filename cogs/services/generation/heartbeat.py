
import time
import asyncio
import discord
from typing import Optional

from ...utils.constants import PLACEHOLDER_EMOJI


class HeartbeatMixin:
    """Placeholder message lifecycle and the heartbeat-driven generation call
    that keeps a live status update in Discord while the model responds.
    """

    async def _safe_delete_placeholder(self, channel, message_id, bot_id=None):
        if not message_id: return

        if bot_id:
            try:
                await self.cog.manager_queue.put({
                    "action": "send_to_child", "bot_id": bot_id,
                    "payload": {
                        "action": "delete_message",
                        "channel_id": channel.id,
                        "message_id": message_id
                    }
                })
            except Exception:
                pass

        max_retries = 3
        backoff = 1.0

        for attempt in range(max_retries):
            try:
                webhook = await self.cog.server_manager._get_or_create_webhook(channel)
                if webhook:
                    await webhook.delete_message(message_id)
                    return
            except Exception:
                # If the message was sent by a Child Bot, the webhook will throw an error (usually NotFound).
                # We must pass here to ensure we fall through to the standard client deletion below.
                pass

            try:
                msg = await channel.fetch_message(message_id)
                await msg.delete()
                return
            except discord.NotFound:
                return # The message genuinely no longer exists in the channel
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to delete placeholder {message_id} after {max_retries} attempts: {e}")
                    return

            await asyncio.sleep(backoff)
            backoff *= 2.0

    async def _send_child_bot_placeholder(self, bot_id: str, channel_id: int, custom_emoji: str) -> Optional[int]:
        sent = await self.cog.child_bot_manager.execute_send(bot_id, {
            "channel_id": channel_id,
            "content": custom_emoji,
            "realistic_typing": False
        })
        if sent:
            return sent[-1].id
        return None

    async def _generate_with_heartbeat(self, model, contents, gen_config, channel, participant, msg_a_id, is_fallback=False, app_name='Bot', app_avatar=None, existing_state=None, message_type="text"):
        # Hard Limits: 4 minutes for Main, 3 minutes for Fallback
        hard_timeout = 180.0 if is_fallback else 240.0

        state_container = existing_state or {}
        state_container.setdefault('msg_a_id', msg_a_id)
        state_container.setdefault('msg_b_id', None)
        state_container.setdefault('app_name', app_name)
        state_container.setdefault('app_avatar', app_avatar)
        state_container.setdefault('message_type', message_type)
        state_container.setdefault('custom_emoji', PLACEHOLDER_EMOJI)

        gen_task = asyncio.create_task(model.generate_content_async(contents, generation_config=gen_config))
        start_time = time.time()
        last_interval = 0

        try:
            # The Absolute Watchdog Loop
            while not gen_task.done():
                elapsed = time.time() - start_time

                if elapsed >= hard_timeout:
                    gen_task.cancel()
                    try:
                        await gen_task
                    except (Exception, asyncio.CancelledError):
                        pass # Ignore exceptions from the forcibly killed task
                    err = TimeoutError(f"Generation timed out after {hard_timeout} seconds")
                    err.state_container = state_container
                    raise err

                # Periodic typing pulse for Child Bots without a placeholder (Discord typing expires in 9s)
                if participant and participant.get('method') == 'child_bot':
                    bot_id = participant.get('bot_id')
                    current_msg_a_id = state_container.get('msg_a_id')
                    if not current_msg_a_id and int(elapsed) > 0 and int(elapsed) % 5 == 0:
                        await self.cog.manager_queue.put({
                            "action": "send_to_child", "bot_id": bot_id,
                            "payload": {"action": "start_typing", "channel_id": channel.id}
                        })

                # Every 10 seconds, update or create the placeholder message
                current_interval = int(elapsed // 10)
                if current_interval > last_interval:
                    last_interval = current_interval

                    mins = int(elapsed) // 60
                    secs = int(elapsed) % 60
                    time_str = f"{mins}:{secs:02d}"

                    base_text = "Using fallback model" if is_fallback else "Still generating"
                    text = f"-# {base_text}... ({time_str})"

                    msg_a_id = state_container.get('msg_a_id')
                    custom_emoji = state_container.get('custom_emoji', PLACEHOLDER_EMOJI)

                    try:
                        if participant and participant.get('method') == 'child_bot':
                            bot_id = participant.get('bot_id')
                            if msg_a_id:
                                await self.cog.manager_queue.put({
                                    "action": "send_to_child", "bot_id": bot_id,
                                    "payload": {
                                        "action": "regenerate_message", "channel_id": channel.id,
                                        "message_id": msg_a_id, "content": f"{custom_emoji}\n\n{text}"
                                    }
                                })
                            else:
                                # Only spawn a placeholder if generation is actively still running
                                if not gen_task.done():
                                    new_msg_id = await self._send_child_bot_placeholder(bot_id, channel.id, f"{custom_emoji}\n\n{text}")
                                    if new_msg_id:
                                        if gen_task.done():
                                            # If generation completed while awaiting placeholder creation, delete immediately
                                            await self._safe_delete_placeholder(channel, new_msg_id, bot_id=bot_id)
                                        else:
                                            state_container['msg_a_id'] = new_msg_id
                                            msg_a_id = new_msg_id
                        else:
                            if state_container.get('message_type') == "embed" and state_container.get('placeholder_msg'):
                                try:
                                    msg_obj = state_container['placeholder_msg']
                                    if msg_obj and msg_obj.embeds:
                                        embed = msg_obj.embeds[0]
                                        embed.description = f"{custom_emoji}\n\n{text}"
                                        await msg_obj.edit(embed=embed)
                                except Exception: pass
                            else:
                                # Webhooks
                                webhook = await self.cog.server_manager._get_or_create_webhook(channel)
                                if webhook and msg_a_id:
                                    await webhook.edit_message(msg_a_id, content=f"{custom_emoji}\n\n{text}")
                    except Exception:
                        pass

                # Check exactly every 1 second
                await asyncio.sleep(1)

            return gen_task.result(), state_container

        except asyncio.CancelledError:
            gen_task.cancel()
            try:
                await gen_task
            except Exception:
                pass
            raise

    async def _update_sending_placeholder(self, channel, participant_method, bot_id, state_container,
                                          start_time_mono, spawn_after=None):
        """Keeps the placeholder ticking through everything that happens after generation.

        `state_container['phase_label']` names the phase and may be changed while this is
        running, so speech synthesis reads as itself rather than as a "Sending..." that
        sits there for half a minute.

        `spawn_after` is the number of seconds after which this will create a placeholder
        that does not exist yet, and is only passed by paths that can be slow here. The
        original rule was never to spawn one during sending, which was right when sending
        meant one upload: a placeholder created and deleted inside the same second is a
        flicker and nothing else. TTS broke the premise rather than the rule -- synthesis
        runs ten to thirty seconds -- so the rule now carries the threshold it always
        implied. Left None, the behaviour is exactly as before.
        """
        if not state_container: return

        has_target = bool(state_container.get('msg_a_id')) or bool(
            state_container.get('message_type') == 'embed' and state_container.get('placeholder_msg'))
        if not has_target and spawn_after is None:
            return

        async def heartbeat_loop():
            try:
                # Immediate initial update
                elapsed = time.monotonic() - start_time_mono
                mins = int(elapsed) // 60
                secs = int(elapsed) % 60
                time_str = f"{mins}:{secs:02d}"
                label = state_container.get('phase_label') or "Sending"
                sending_text = f"-# {label}... ({time_str})"

                async def do_update(text):
                    if state_container.get('message_type') == 'embed' and state_container.get('placeholder_msg'):
                        try:
                            msg_obj = state_container['placeholder_msg']
                            if msg_obj and msg_obj.embeds:
                                embed = msg_obj.embeds[0]
                                embed.description = f"{state_container.get('custom_emoji', PLACEHOLDER_EMOJI)}\n\n{text}"
                                await msg_obj.edit(embed=embed)
                        except Exception: pass
                        return

                    target_msg_id = state_container.get('msg_a_id')
                    if not target_msg_id:
                        return

                    full_text = f"{state_container.get('custom_emoji', PLACEHOLDER_EMOJI)}\n\n{text}"

                    try:
                        if participant_method == 'child_bot' and bot_id:
                            await self.cog.manager_queue.put({
                                "action": "send_to_child", "bot_id": bot_id,
                                "payload": {
                                    "action": "regenerate_message", "channel_id": channel.id,
                                    "message_id": target_msg_id, "content": full_text
                                }
                            })
                        else:
                            wh = await self.cog.server_manager._get_or_create_webhook(channel)
                            if wh:
                                await wh.edit_message(target_msg_id, content=full_text)
                    except Exception:
                        pass

                await do_update(sending_text)

                # Fixed 10-second heartbeat loop
                while True:
                    await asyncio.sleep(10)

                    elapsed = time.monotonic() - start_time_mono
                    mins = int(elapsed) // 60
                    secs = int(elapsed) % 60
                    time_str = f"{mins}:{secs:02d}"
                    label = state_container.get('phase_label') or "Sending"
                    sending_text = f"-# {label}... ({time_str})"

                    if spawn_after is not None and elapsed >= spawn_after:
                        await maybe_spawn(f"{state_container.get('custom_emoji', PLACEHOLDER_EMOJI)}\n\n{sending_text}")

                    await do_update(sending_text)

            except asyncio.CancelledError:
                pass

        async def maybe_spawn(content):
            """Creates the placeholder this phase is trying to edit, if there is none.

            Only child bots reach this: a webhook turn always has one by now, and
            spawning a webhook message here would need the appearance data that only the
            caller holds.

            The send runs as its own shielded task, recorded in the state container. A
            round can end while this is in flight, and by then the child bot has already
            posted the message -- so cancelling the heartbeat must not take the id down
            with it, or that message stays in the channel with nothing able to address
            it. `_stop_sending_heartbeat` collects the id from the task instead.
            """
            if state_container.get('msg_a_id') or participant_method != 'child_bot' or not bot_id:
                return
            if state_container.get('spawn_task'):
                return
            task = asyncio.create_task(self._send_child_bot_placeholder(bot_id, channel.id, content))
            state_container['spawn_task'] = task
            try:
                new_id = await asyncio.shield(task)
            except Exception:
                return
            if new_id:
                state_container['msg_a_id'] = new_id

        state_container['sending_task'] = asyncio.create_task(heartbeat_loop())

    async def _stop_sending_heartbeat(self, state_container):
        """Cancels the post-generation heartbeat and waits for it to actually stop.

        The wait is what makes `maybe_spawn` safe: cancel() only schedules the
        CancelledError, so a caller that cancelled and immediately read `msg_a_id` could
        read it before a placeholder created microseconds earlier was recorded, and
        leave that message stranded in the channel.
        """
        if not state_container:
            return

        task = state_container.get('sending_task')
        if task:
            task.cancel()
            try:
                await task
            except (Exception, asyncio.CancelledError):
                pass
            state_container['sending_task'] = None

        # Shielded, so cancelling the loop above left it running. Collect what it
        # produced: the message is already in the channel either way, and the id is the
        # only thing that can delete it.
        spawn_task = state_container.get('spawn_task')
        if spawn_task:
            try:
                new_id = await spawn_task
                if new_id and not state_container.get('msg_a_id'):
                    state_container['msg_a_id'] = new_id
            except (Exception, asyncio.CancelledError):
                pass
            state_container['spawn_task'] = None
