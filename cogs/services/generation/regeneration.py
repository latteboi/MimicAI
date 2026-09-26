import time
import re
import asyncio
import discord
import datetime
from typing import Dict, List, Optional

from ...utils.constants import (
    PLACEHOLDER_EMOJI,
    DEFAULT_KICKSTART_START, DEFAULT_IMAGE_PRESENT, DEFAULT_WHISPER_RECAP,
)
from ...utils.helpers import (_format_history_entry, image_command_prefix,
                             is_citation_subtext, kickstart_note,
                             turn_posted_at)
from ...utils.attachment_limits import over_attachment_limit
from .reply import reply_gen_config, reply_meta
from .triggers import referenced_message, reply_record
from .tool_loop import functions_for, ltm_auto_threshold


def _is_image(attachment) -> bool:
    return bool(attachment.content_type and attachment.content_type.startswith("image/"))


def _add_to_last_user_turn(history: List[Dict], parts: List) -> None:
    for turn in reversed(history):
        if turn.get('role') == 'user':
            turn['parts'].extend(parts)
            return


class RegenerationMixin:
    """Re-runs generation for a single existing turn (triggered by the regenerate
    reaction), editing the message in place instead of sending a new one.

    The generation itself -- model, fallback, text, warnings and trace -- is ReplyMixin's,
    shared with the round worker.
    """

    async def _restore_regenerated_message(self, channel, participant, message_id,
                                           content, attachments):
        """Puts the message back to what it said before the regeneration started.

        Every exit that produces no new text has to come through here. The message was
        overwritten with the placeholder emoji before any work began, so returning
        early -- a model that would not construct, a cancellation -- used to leave the
        turn showing a bare emoji with nothing able to fix it but another regeneration.
        """
        if not content:
            return

        async def _restore():
            try:
                if participant.get('method') == 'child_bot':
                    await self.cog.manager_queue.put({
                        "action": "send_to_child", "bot_id": participant['bot_id'],
                        "payload": {
                            "action": "regenerate_message", "channel_id": channel.id,
                            "message_id": message_id, "content": content
                        }
                    })
                    return
                await self.cog.server_manager.run_webhook(
                    channel, "edit_message", message_id, content=content,
                    attachments=attachments or [])
            except Exception:
                pass

        # Detached, then shielded. Every caller is already unwinding a cancellation, and
        # a second one is ordinary -- /cancel allows two in ten seconds, and /suspend and
        # /purge reach the same task. Interrupted here, the restore never lands and the
        # turn keeps the placeholder emoji with no path back but another regeneration.
        task = asyncio.ensure_future(_restore())
        self.cog.background_tasks.add(task)
        task.add_done_callback(self.cog.background_tasks.discard)
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    async def _drop_regenerated_follow_ups(self, channel, turn: Dict, message_id: int) -> List[int]:
        """Deletes the follow-ups the new text replaces. Returns the turn's ids that stay.

        Source lines and image follow-ups are kept -- the regenerated text is edited onto
        the turn's own message and these still belong to it. What survives has to keep
        its id too, or the rebind orphans it: the message stays in the channel with
        nothing left able to address it. A follow-up that cannot be read is dropped.
        """
        others = [mid for mid in turn.get("message_ids", []) if mid != message_id]
        fetched = dict(zip(others, await asyncio.gather(
            *(channel.fetch_message(mid) for mid in others), return_exceptions=True)))

        kept, doomed = [], []
        for mid in turn.get("message_ids", []):
            msg = fetched.get(mid)
            if mid == message_id:
                kept.append(mid)
            elif msg is None or isinstance(msg, BaseException):
                continue
            elif is_citation_subtext(msg.content) or any(_is_image(a) for a in msg.attachments):
                kept.append(mid)
            else:
                doomed.append(mid)
        if message_id not in kept:
            kept.insert(0, message_id)

        # Through the turn-deletion path: it marks the ids as the bot's own for the delete
        # listeners, and without Manage Messages still removes a webhook's message through
        # the webhook. A bare delete() failed there and left the follow-up behind.
        await self._delete_channel_messages(channel, doomed)
        return kept

    async def _recover_regeneration_media(self, channel, original_attachments: List,
                                          last_user_turn: Optional[Dict]) -> List:
        """The media the original reply was answering, which the log does not keep.

        A reply carrying a generated image is re-presented that image and the prompt that
        made it. Otherwise the last user message's media, and what it replied to.
        """
        if original_attachments:
            prompt_text = ""
            if last_user_turn:
                lines = last_user_turn.get("content", "").split('\n')
                body = "\n".join(l for l in lines
                                 if not re.match(r'^<.+> \[[^\]]+\]:', l) and not l.startswith("</")).strip()
                prefix = image_command_prefix(body)
                prompt_text = body[len(prefix):].strip() if prefix else body
            template = self.cog.global_prompts.get("IMAGE_PRESENT", DEFAULT_IMAGE_PRESENT)
            return [template.format(prompt=prompt_text)] + [
                {"url": a.url, "mime_type": a.content_type} for a in original_attachments]

        def readable(attachment) -> bool:
            # A file over the limit was never read into the turn, so it is not
            # recovered into the regeneration either.
            kind = attachment.content_type or ""
            return kind.startswith(("image/", "audio/", "video/")) and not over_attachment_limit(attachment)

        user_msg_ids = (last_user_turn or {}).get("message_ids", [])
        if not user_msg_ids:
            return []
        recovered_media_parts = []
        try:
            target_msg = await channel.fetch_message(user_msg_ids[-1])
            recovered_media_parts.extend({"url": a.url, "mime_type": a.content_type}
                                         for a in target_msg.attachments if readable(a))
            # What the round sent for the reply, so the regeneration is answering the same.
            _, reply_media = reply_record([], await referenced_message(target_msg))
            recovered_media_parts.extend(reply_media)
        except Exception as e:
            print(f"Failed to recover media for regeneration: {e}")
        return recovered_media_parts

    async def _execute_regeneration(self, payload: discord.RawReactionActionEvent, session: Dict, turn_id: str, participant: Dict):
        channel = self.cog.bot.get_channel(payload.channel_id)
        if not channel: return

        session['is_regenerating'] = True
        session_type = session.get("type", "multi")
        owner_id, profile_name = participant['owner_id'], participant['profile_name']
        bot_id = participant.get('bot_id')
        # Set once the new text is on the message. Everything after that point --
        # notably the session flush -- must not roll the channel back to the old reply,
        # because the log already holds the new one.
        delivered = False
        original_content = None
        original_attachments = []
        state_container = None

        async def put_back():
            # The heartbeat first: left running, its next tick writes a status line over
            # the text just put back.
            await self._stop_sending_heartbeat(state_container)
            await self._restore_regenerated_message(
                channel, participant, payload.message_id, original_content, original_attachments)

        # Everything after the flag runs inside the try whose handlers put the
        # message back and whose finally clears is_regenerating. The fetch, the emoji
        # edit and the log save used to sit above it, so a cancel landing on any of them
        # -- the user pulling the reaction back off, or /cancel -- left the message
        # showing the emoji for good and is_regenerating set, which parks the channel's
        # worker until somebody runs /cancel.
        try:
            log = session.get("unified_log", [])
            turn_index = next((i for i, t in enumerate(log) if t.get("turn_id") == turn_id), None)
            if turn_index is None:
                return
            target_turn = log[turn_index]

            # The one refusal that must stop, and checked before anything changes on
            # screen. It used to come after the emoji edit and the follow-up cleanup, so a
            # refused regeneration of a long reply deleted its continuation messages and
            # put back only the first.
            if not self.cog.profile_manager._check_unrestricted_safety_policy(owner_id, profile_name, channel):
                return

            bot_pid = self.cog.profile_manager._get_pid_from_name_any(owner_id, profile_name)
            p_index = self.cog.profile_manager._get_user_index(owner_id)
            p_settings = self.cog.profile_manager._get_profile_config(
                owner_id, profile_name, profile_name in p_index.get("borrowed", [])) or {}
            custom_emoji = p_settings.get("placeholder_emoji") or PLACEHOLDER_EMOJI

            # The message being regenerated is edited in place, so its current text is the
            # only copy of what to put back if this cannot run or is cancelled. Read once,
            # here, because every path below has already overwritten it with the emoji. Its
            # images are kept through every edit, and re-presented to the model.
            try:
                original = await channel.fetch_message(payload.message_id)
                original_content = original.content
                original_attachments = [a for a in original.attachments if _is_image(a)]
            except Exception:
                pass

            # Visual feedback before the context gathering.
            if participant.get('method') == 'child_bot':
                child_emoji = await self.cog.child_bot_manager.resolve_emoji_for_child(bot_id, custom_emoji)
                await self.cog.manager_queue.put({
                    "action": "send_to_child", "bot_id": bot_id,
                    "payload": {
                        "action": "regenerate_message", "channel_id": channel.id,
                        "message_id": payload.message_id, "content": child_emoji
                    }
                })
            else:
                try:
                    await self.cog.server_manager.run_webhook(
                        channel, "edit_message", payload.message_id, content=custom_emoji,
                        attachments=original_attachments)
                except Exception: pass

            surviving_message_ids = await self._drop_regenerated_follow_ups(
                channel, target_turn, payload.message_id)

            # The history as it stood when the turn was first written.
            earlier = log[:turn_index]
            last_user_index = next(
                (i for i in range(len(earlier) - 1, -1, -1) if earlier[i].get("is_user") is True), None)
            last_user_turn = earlier[last_user_index] if last_user_index is not None else None

            # The turns since the last user message are what this regeneration is
            # answering; STM governs how far back it remembers, not those. See
            # SessionManager._build_history_for_participant.
            history = self.cog.session_manager._build_history_for_participant(
                earlier, bot_pid, p_settings, reserved_tail=len(earlier) - (last_user_index or 0),
                hide_folded=self.cog.session_manager.compaction_enabled(session),
            )
            pending_whispers = self.cog.session_manager._get_pending_whispers_for_participant(earlier, bot_pid)

            # Pseudo-turn injection to ensure history ends with a 'user' role. The same
            # note a live round would get for the same silence -- `kickstart_note` owns
            # which one that is, so a regenerate cannot answer it differently.
            follow_up = kickstart_note(history, self.cog.global_prompts)
            if follow_up:
                history.append({'role': 'user', 'parts': [follow_up]})
            elif not history:
                history.append({'role': 'user', 'parts': [self.cog.global_prompts.get("KICKSTART_START", DEFAULT_KICKSTART_START)]})

            media = await self._recover_regeneration_media(channel, original_attachments, last_user_turn)
            if media:
                _add_to_last_user_turn(history, media)
            if pending_whispers:
                recap_template = self.cog.global_prompts.get("WHISPER_RECAP", DEFAULT_WHISPER_RECAP)
                _add_to_last_user_turn(history, [recap_template.format(whispers="\n---\n".join(pending_whispers))])

            trigger_content = last_user_turn.get("content", "") if last_user_turn else ""
            ltm_recall_text, training_examples = await asyncio.gather(
                self.cog.memory_manager._get_relevant_ltm_for_prompt(
                    (channel.id, owner_id, profile_name), history, owner_id, profile_name,
                    trigger_content, "User", channel.guild.id, payload.user_id,
                    threshold=ltm_auto_threshold(p_settings)),
                self.cog.memory_manager._get_relevant_training_examples(
                    owner_id, profile_name, trigger_content, channel.guild.id),
            )
            # One tuple for the prompt and both models -- see tool_loop.
            functions = functions_for(p_settings)
            (system_instruction, _, _, temp, top_p, top_k,
             primary_model, fallback_model_name) = await asyncio.to_thread(
                self._construct_system_instructions,
                owner_id, profile_name, channel.id, is_multi_profile=True,
                training_examples_list=training_examples, recalled_ltm=ltm_recall_text,
                functions=functions,
            )

            app_name, app_avatar = self._resolve_appearance_data(owner_id, profile_name)
            state_container = {
                'msg_a_id': payload.message_id,
                'msg_b_id': None,
                'app_name': app_name,
                'app_avatar': app_avatar,
                'message_type': "text",
                'custom_emoji': custom_emoji,
                'bot_id': bot_id,
                # msg_a_id here is the turn's own message, not a placeholder.
                # _abandon_state_container must never delete it -- this path puts the
                # original text back instead, through _restore_regenerated_message.
                'placeholder_owned': False,
            }
            # Published so /cancel can tell "still generating, safe to undo" from
            # "applying, too late". Released in the finally.
            self.cog.session_manager.register_in_flight(session, state_container)

            participant_names = self._participant_names(session)
            t_start = time.monotonic()
            attempt = await self._attempt_reply(
                channel=channel, participant=participant, p_settings=p_settings, owner_id=owner_id,
                user_id=payload.user_id, system_instruction=system_instruction,
                primary_model=primary_model, fallback_model_name=fallback_model_name,
                history=history, gen_config=reply_gen_config(p_settings, temp, top_p, top_k),
                msg_a_id=payload.message_id, app_name=app_name, app_avatar=app_avatar,
                state_container=state_container, participant_names=participant_names,
                functions=functions)

            # The post-generation phase, as the worker and global chat both run it, so the
            # message does not sit frozen on the last "Still generating" tick through the
            # rest of this.
            state_container['phase_label'] = "Applying"
            await self._update_sending_placeholder(
                channel, participant.get('method', 'webhook'), bot_id, state_container, time.monotonic())

            reply = self._reply_text(attempt, p_settings, owner_id, profile_name, participant_names)
            # The warnings are shown, never recorded: the model would read them back as
            # something the character said.
            display_text = reply.text
            if reply.warnings:
                display_text += "\n\n" + "\n".join(f"-# {i+1}. {w}" for i, w in enumerate(reply.warnings))

            await self._safe_delete_placeholder(channel, state_container.get('msg_b_id'), bot_id=bot_id)
            state_container['msg_b_id'] = None

            sent_timestamp = datetime.datetime.now(datetime.timezone.utc)
            # A regeneration rewrites what the turn says, not when it was said. The turn
            # keeps its original moment -- the message is edited in place, so it is still
            # sitting where it was in the channel -- and records the regeneration
            # separately as `edited_at`. Stamping the turn "now" put an older turn's
            # clock ahead of every turn after it, in a transcript the model reads back
            # and the session viewer shows.
            original_timestamp = turn_posted_at(target_turn) or sent_timestamp
            # Through the accessor, not the raw cache. _get_user_appearance resolves the
            # effective profile first and populates the entry from config when it is
            # cold; reading self.cog.user_appearances directly did neither, so a
            # borrowed profile -- or any profile whose appearance had not been cached
            # yet -- came back empty and the regenerated turn was rewritten under the
            # bare profile name instead of the character's display name.
            app_data = self.cog.profile_manager._get_user_appearance(owner_id, profile_name)
            new_history_line = _format_history_entry(
                app_data.get("custom_display_name") or profile_name, original_timestamp, reply.text,
                p_settings.get("timezone", "UTC"),
                entity_id=self.cog.profile_manager._get_profile_id(owner_id, profile_name))

            final_target_turn = next((t for t in session.get("unified_log", []) if t.get("turn_id") == turn_id), None)
            if not final_target_turn:
                final_target_turn = target_turn
                session.setdefault("unified_log", []).append(final_target_turn)

            final_target_turn["content"] = new_history_line
            # Written out rather than left to the message id it was read from, so the
            # turn's moment survives its messages being deleted from under it.
            final_target_turn["timestamp"] = original_timestamp.isoformat()
            final_target_turn["edited_at"] = sent_timestamp.isoformat()
            final_target_turn["message_ids"] = list(surviving_message_ids)
            final_target_turn["meta"] = reply_meta(
                attempt, duration=time.monotonic() - t_start, training_examples=training_examples,
                ltm_recall_text=ltm_recall_text, sources=reply.sources, neuro_state=reply.neuro_state)
            # Clean up legacy signatures from the turn if they exist
            final_target_turn.pop('thought_signature', None)

            await self._stop_sending_heartbeat(state_container)

            # Truncate text strictly for Discord's 2000 character limit on edits
            safe_text = display_text if len(display_text) <= 2000 else display_text[:1997] + "..."
            if participant.get('method') == 'child_bot':
                await self.cog.manager_queue.put({
                    "action": "send_to_child", "bot_id": bot_id,
                    "payload": {
                        "action": "regenerate_message", "channel_id": channel.id,
                        "message_id": payload.message_id, "content": safe_text
                    }
                })
            else:
                try:
                    await self.cog.server_manager.run_webhook(
                        channel, "edit_message", payload.message_id,
                        content=safe_text, attachments=original_attachments)
                except Exception: pass

            delivered = True

            # A tail write when the turn sits in the unsealed tail -- the usual case, a
            # recent reply. An edit to a sealed turn needs the full rewrite. This used to
            # rewrite the whole log twice per regeneration: once before generating, when
            # nothing had changed yet, which also sealed the tail and so forced the second.
            log = session.get("unified_log") or []
            index = next((i for i, t in enumerate(log) if t is final_target_turn), 0)
            await self.cog.session_manager.flush_session(
                (channel.id, None, None), session_type,
                structural=index < session.get("_log_cold_len", 0))

            # No rebuild: a regenerated turn only changes its own content, which no
            # derived session state reads. See _recompute_pending_whispers.

        except asyncio.CancelledError:
            # /cancel, the reaction pulled back off, or a shutdown: the message is still
            # showing the placeholder emoji, so it goes back before the cancellation does.
            if not delivered:
                await put_back()
            raise
        except Exception as e:
            print(f"Regeneration failed: {e}")
            # Otherwise the message keeps the placeholder emoji this function wrote over
            # it, with no path back but another regeneration.
            if not delivered:
                await put_back()
        finally:
            # The same teardown the round worker uses. It stops the heartbeat, deletes
            # the "still sending" placeholder and releases the in-flight entry, and
            # leaves msg_a_id alone because this container is not placeholder_owned.
            await self._abandon_state_container(channel, state_container, session=session, bot_id=bot_id)
            session['is_regenerating'] = False
