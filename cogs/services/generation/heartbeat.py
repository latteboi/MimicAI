import time
import uuid
import asyncio
import discord
from typing import Optional

from ...utils.constants import PLACEHOLDER_EMOJI


import time
import uuid
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
        corr_id = str(uuid.uuid4())
        conf_event = asyncio.Event()
        conf_data = {"event": conf_event, "type": "heartbeat_placeholder"}
        self.cog.pending_child_confirmations[corr_id] = conf_data

        await self.cog.manager_queue.put({
            "action": "send_to_child", "bot_id": bot_id,
            "payload": {
                "action": "send_message", "channel_id": channel_id,
                "content": custom_emoji, "realistic_typing": False, "correlation_id": corr_id
            }
        })
        try:
            await asyncio.wait_for(conf_event.wait(), timeout=5.0)
            msg_ids = conf_data.get("message_ids", [])
            if msg_ids: return msg_ids[-1]
        except asyncio.TimeoutError: pass
        finally: self.cog.pending_child_confirmations.pop(corr_id, None)
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
                                new_msg_id = await self._send_child_bot_placeholder(bot_id, channel.id, f"{custom_emoji}\n\n{text}")
                                if new_msg_id:
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

    async def _update_sending_placeholder(self, channel, participant_method, bot_id, state_container, start_time_mono):
        if not state_container: return

        async def heartbeat_loop():
            try:
                # Immediate initial update
                elapsed = time.monotonic() - start_time_mono
                mins = int(elapsed) // 60
                secs = int(elapsed) % 60
                time_str = f"{mins}:{secs:02d}"
                sending_text = f"-# Sending... ({time_str})"

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
                    full_text = f"{state_container.get('custom_emoji', PLACEHOLDER_EMOJI)}\n\n{text}"

                    try:
                        if participant_method == 'child_bot' and bot_id:
                            if target_msg_id:
                                await self.cog.manager_queue.put({
                                    "action": "send_to_child", "bot_id": bot_id,
                                    "payload": {
                                        "action": "regenerate_message", "channel_id": channel.id,
                                        "message_id": target_msg_id, "content": full_text
                                    }
                                })
                            else:
                                new_msg_id = await self._send_child_bot_placeholder(bot_id, channel.id, full_text)
                                if new_msg_id:
                                    state_container['msg_a_id'] = new_msg_id
                        else:
                            if target_msg_id:
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
                    sending_text = f"-# Sending... ({time_str})"

                    await do_update(sending_text)

            except asyncio.CancelledError:
                pass

        state_container['sending_task'] = asyncio.create_task(heartbeat_loop())
