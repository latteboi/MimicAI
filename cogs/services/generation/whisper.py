import uuid
import asyncio
import discord
import datetime
from typing import Dict, Optional

from ...utils.constants import (
    PLACEHOLDER_EMOJI, DEFAULT_WHISPER_INJECTION,
    SESSION_BUSY_FLAGS, WHISPER_BUSY_WAIT_TIMEOUT_SECONDS, WHISPER_WAITING_NOTICE,
)
from ...utils.helpers import _format_citation_subtext, _format_history_entry, _get_user_hash
from . import tool_loop
from .reply import ReplyText, reply_gen_config
from ...gui.gui_sessions import WhisperActionView
from ...managers.session_manager import intern_turn


class WhisperMixin:
    """Private (ephemeral) whisper exchanges with a single multi-profile
    participant, their regeneration, and reply-context resolution for replies
    to a previous message.
    """

    async def _execute_whisper(self, interaction: discord.Interaction, target_participant: Dict, whisper_message: str):
        """Gate, claim, run, release.

        A whisper is a blocking private turn. It never generates on top of a live round --
        it would be answering a conversation that is still changing underneath it, and its
        own turns would land in unified_log in the middle of one. So it waits for the
        channel to go idle, claims it with is_whispering, and everything else queues behind
        it until it lands: _multi_profile_worker holds before its model call (after posting
        the placeholder, so the queued profile still shows feedback and waits silently), and
        the regeneration reaction already queues on the same flags.

        The claim is released in a finally. is_whispering is read by the worker, /purge, the
        regeneration reaction and the eviction sweep, so leaking it True blocks all four for
        the life of the process.
        """
        session = self.cog.multi_profile_channels.get(interaction.channel_id)

        # [NEW] Force hydration if session exists or might exist on disk
        if not session or not session.get("is_hydrated"):
            session = await self.cog.session_manager._ensure_session_hydrated(interaction.channel_id, session.get("type", "multi") if session else "multi")

        if not session: 
            await interaction.followup.send("Session not found.", ephemeral=True)
            return

        owner_id = target_participant['owner_id']
        profile_name = target_participant['profile_name']
        participant_key = (owner_id, profile_name)

        # Ensure session is hydrated to get history
        if not session.get("is_hydrated"):
            session = await self.cog.session_manager._ensure_session_hydrated(interaction.channel_id, session.get("type", "multi"))

        participant_keys = {(p['owner_id'], p['profile_name']) for p in session.get("profiles", [])}
        if participant_key not in participant_keys:
            await interaction.followup.send("An error occurred: Could not find that participant in this session.", ephemeral=True)
            return

        # Announce the intent before waiting for the channel. The worker checks this
        # counter before it claims a round, so a whisper cannot be starved by a channel
        # that never goes quiet -- polling for an idle instant at 2 Hz would lose that race
        # every time against a worker that re-arms is_running the moment its queue refills.
        session['whisper_waiting'] = session.get('whisper_waiting', 0) + 1
        waiting_msg = None
        try:
            # Tell the user *before* the wait, not after it -- an ephemeral that sits on
            # "thinking" for a whole round is indistinguishable from a dead command. The
            # notice is then edited into the placeholder rather than replaced, so this costs
            # no extra message when the channel was busy and none at all when it was not.
            if any(session.get(flag) for flag in SESSION_BUSY_FLAGS):
                waiting_msg = await interaction.followup.send(WHISPER_WAITING_NOTICE, ephemeral=True, wait=True)

            if not await self.cog.session_manager._wait_for_session_flags(
                session, SESSION_BUSY_FLAGS, WHISPER_BUSY_WAIT_TIMEOUT_SECONDS
            ):
                timed_out = (f"The session is still busy after {int(WHISPER_BUSY_WAIT_TIMEOUT_SECONDS)}s. "
                             "Your whisper was not sent \u2014 try again in a moment.")
                if waiting_msg:
                    await waiting_msg.edit(content=timed_out)
                else:
                    await interaction.followup.send(timed_out, ephemeral=True)
                return

            # No await between the clear return above and this claim, which is what makes
            # check-then-set atomic on a single-threaded loop.
            session['is_whispering'] = True
        finally:
            # Released on the claim, not on completion -- from here the channel is held by
            # is_whispering, and a worker still parked on this counter would deadlock.
            session['whisper_waiting'] = max(0, session.get('whisper_waiting', 1) - 1)

        try:
            await self._run_whisper_turn(interaction, session, target_participant, whisper_message, waiting_msg)
        finally:
            session['is_whispering'] = False

    async def _whisper_reply(self, interaction: discord.Interaction, session: Dict,
                             target_participant: Dict, p_settings: Dict, history,
                             placeholder_msg, custom_emoji: str, display_name: str, avatar_url,
                             log_context: str) -> ReplyText:
        """One whispered reply, made the way a session reply is: `_attempt_reply`'s primary,
        fallback race and Final Fallback, then `_reply_text`'s text, sources and warnings.

        Whispers used to build one model and call it: no fallback at all, though they read
        one, and a block reason of their own wording. Private by construction still -- the
        placeholder is the whisperer's ephemeral embed, and the caller puts the warnings in
        it rather than in the channel.
        """
        owner_id, profile_name = target_participant['owner_id'], target_participant['profile_name']
        if not self.cog.profile_manager._check_unrestricted_safety_policy(owner_id, profile_name, interaction.channel):
            return ReplyText(text="This character's content rating is Adult 18+, which only runs in "
                                  "age-restricted channels.", blocked=True)

        # One tuple for the prompt and both models -- see tool_loop.
        functions = tool_loop.functions_for(p_settings)
        (system_instruction, _, _, temp, top_p, top_k,
         primary_model, fallback_model_name) = await asyncio.to_thread(
            self._construct_system_instructions, owner_id, profile_name, interaction.channel_id,
            is_multi_profile=True, functions=functions)

        # Every name the history wraps a turn in: the whisperer, and each seated profile under
        # both the name its private turns are logged as and the one it is shown as.
        names = [interaction.user.name, *(p['profile_name'] for p in session.get("profiles", [])),
                 *self._participant_names(session)]
        state_container = {"custom_emoji": custom_emoji, "placeholder_msg": placeholder_msg,
                           "message_type": "embed"}
        try:
            attempt = await self._attempt_reply(
                channel=interaction.channel,
                # No `method`: the placeholder is this embed, never a child bot's message.
                participant={"owner_id": owner_id, "profile_name": profile_name},
                p_settings=p_settings, owner_id=owner_id, user_id=interaction.user.id,
                system_instruction=system_instruction, primary_model=primary_model,
                fallback_model_name=fallback_model_name, history=history,
                gen_config=reply_gen_config(p_settings, temp, top_p, top_k),
                msg_a_id=placeholder_msg.id, app_name=display_name, app_avatar=avatar_url,
                state_container=state_container, participant_names=names,
                log_context=log_context, functions=functions, private=True)
        finally:
            await self._safe_delete_placeholder(interaction.channel, state_container.get('msg_b_id'))
        return self._reply_text(attempt, p_settings, owner_id, profile_name, names)

    @staticmethod
    def _whisper_display(reply: ReplyText) -> str:
        """The reply as the whisperer's embed shows it: the text, its sources, the warnings."""
        text = reply.text
        sources = _format_citation_subtext(reply.sources or [])
        if sources:
            text += "\n\n" + "\n".join(sources)
        if reply.warnings:
            text += "\n\n" + "\n".join(f"-# {i+1}. {w}" for i, w in enumerate(reply.warnings))
        return text[:4096]

    def _whisper_appearance(self, owner_id: int, profile_name: str, p_settings: Dict):
        """(effective owner, effective name, display name, avatar url, placeholder emoji)."""
        effective_owner_id, effective_profile_name = self.cog.profile_manager._resolve_effective_profile(owner_id, profile_name)
        display_name = effective_profile_name
        avatar_url = self.cog.bot.user.display_avatar.url
        appearance = self.cog.profile_manager._get_user_appearance(effective_owner_id, effective_profile_name)
        if appearance:
            display_name = appearance.get("custom_display_name") or display_name
            avatar_url = appearance.get("custom_avatar_url") or avatar_url
        custom_emoji = p_settings.get("placeholder_emoji") or PLACEHOLDER_EMOJI
        return effective_owner_id, effective_profile_name, display_name, avatar_url, custom_emoji

    def _whisper_embed(self, interaction: discord.Interaction, description: str, display_name: str,
                       avatar_url, whisper_message: str) -> discord.Embed:
        embed = discord.Embed(description=description, color=discord.Color.dark_grey())
        embed.set_author(name=display_name, icon_url=avatar_url)
        embed.set_footer(text=f"{whisper_message}"[:1000], icon_url=interaction.user.display_avatar.url)
        return embed

    async def _run_whisper_turn(self, interaction: discord.Interaction, session: Dict, target_participant: Dict, whisper_message: str, waiting_msg: Optional[discord.Message] = None):
        """The whisper turn proper. Only called with the channel already claimed by
        _execute_whisper -- never call this directly."""
        owner_id = target_participant['owner_id']
        profile_name = target_participant['profile_name']
        participant_key = (owner_id, profile_name)

        user_index = self.cog.profile_manager._get_user_index(owner_id)
        is_borrowed = profile_name in user_index.get("borrowed", [])
        p_settings = self.cog.profile_manager._get_profile_config(owner_id, profile_name, is_borrowed) or {}
        effective_owner_id, effective_profile_name, display_name, avatar_url, custom_emoji = \
            self._whisper_appearance(owner_id, profile_name, p_settings)

        user_hash = _get_user_hash(interaction.user.id)
        whisper_content = _format_history_entry(interaction.user.name, interaction.created_at, whisper_message, entity_id=user_hash)
        api_whisper_prompt = self.cog.global_prompts.get("WHISPER_INJECTION", DEFAULT_WHISPER_INJECTION).format(whisper_content=whisper_content.strip())

        # Derived from unified_log, the single source of truth, rather than a shadow copy
        # maintained by incremental appends. _build_history_for_participant already rewrites
        # this participant's own whispers and private responses into their XML tags, and
        # hides other participants' — so the privacy boundary is enforced in one place.
        bot_pid = self.cog.profile_manager._get_pid_from_name_any(owner_id, profile_name)
        contents_for_api_call = self.cog.session_manager._build_history_for_participant(
            session.get("unified_log", []), bot_pid, p_settings,
            hide_folded=self.cog.session_manager.compaction_enabled(session),
        )

        # Ensure alternating roles by appending to the last user turn if present
        if contents_for_api_call and contents_for_api_call[-1].get('role', 'user') == 'user':
            contents_for_api_call[-1]['parts'].append(api_whisper_prompt)
        else:
            contents_for_api_call.append({'role': 'user', 'parts': [api_whisper_prompt]})

        placeholder_embed = self._whisper_embed(interaction, custom_emoji, display_name, avatar_url, whisper_message)
        if waiting_msg is not None:
            placeholder_msg = await waiting_msg.edit(content=None, embed=placeholder_embed)
        else:
            placeholder_msg = await interaction.followup.send(embed=placeholder_embed, ephemeral=True, wait=True)

        try:
            reply = await self._whisper_reply(
                interaction, session, target_participant, p_settings, contents_for_api_call,
                placeholder_msg, custom_emoji, display_name, avatar_url, "whisper")
        except asyncio.CancelledError:
            return

        if reply.blocked:
            # Nothing is logged: the character said nothing, and the whisper is theirs to send again.
            await placeholder_msg.edit(embed=self._whisper_embed(
                interaction, self._whisper_display(reply), display_name, avatar_url, whisper_message))
            return

        # Recorded with its citations, as a session turn is; the warnings are shown, never
        # recorded -- the model would read them back as something the character said.
        response_text = reply.text
        sources = _format_citation_subtext(reply.sources or [])
        if sources:
            response_text += "\n\n" + "\n".join(sources)

        whisper_turn_id = str(uuid.uuid4())
        target_pid = bot_pid

        session.setdefault("unified_log", []).append(intern_turn({
            "turn_id": whisper_turn_id, "type": "whisper",
            "is_user": True, "speaker_pid": str(interaction.user.id), "target_pid": target_pid,
            "message_ids":[],
            "content": whisper_content,
            "timestamp": interaction.created_at.isoformat()
        }))

        response_turn_id = str(uuid.uuid4())
        profile_id = self.cog.profile_manager._get_profile_id(effective_owner_id, effective_profile_name)
        response_content = _format_history_entry(profile_name, datetime.datetime.now(datetime.timezone.utc), response_text, entity_id=profile_id)

        resp_log = {
            "turn_id": response_turn_id, "type": "private_response",
            "is_user": False, "speaker_pid": target_pid, "target_id": interaction.user.id,
            "message_ids":[],
            "content": response_content,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

        session.setdefault("unified_log", []).append(intern_turn(resp_log))

        # Both turns are already in unified_log above; _build_history_for_participant
        # wraps them in <private_whisper> / <private_response> when it derives this
        # participant's history, so there is no second copy to maintain here.

        # Add to pending whispers to be injected into the next public turn
        session.setdefault("pending_whispers", {}).setdefault(participant_key, []).append(whisper_content)

        # [NEW] Immediate persistence for private whisper turns
        session_type = session.get("type", "multi")
        await self.cog.session_manager.flush_session((interaction.channel_id, None, None), session_type)

        # self.cog, not self. WhisperActionView reaches for cog.multi_profile_channels and
        # cog.generation_service; handed the GenerationService instead, every button press
        # raised AttributeError inside the callback and surfaced as "This interaction
        # failed" -- which is why Delete and Regenerate did nothing at all.
        view = WhisperActionView(self.cog, interaction, whisper_turn_id, response_turn_id, target_participant, whisper_message)
        resp_msg = await placeholder_msg.edit(
            embed=self._whisper_embed(interaction, self._whisper_display(reply), display_name, avatar_url, whisper_message),
            view=view)

        # Inject the message ID back into the log turn
        if resp_msg:
            resp_log["message_ids"] = [resp_msg.id]
            await self.cog.session_manager.flush_session((interaction.channel_id, None, None), session_type)

    async def _execute_whisper_regeneration(self, interaction: discord.Interaction, whisper_turn_id: str, response_turn_id: str, target_participant: Dict, whisper_message: str):
        """Gate, claim, run, release -- the same contract as _execute_whisper.

        This is a component interaction, so the 3-second response deadline is what decides
        the shape: the waiting notice has to *be* the response, and the placeholder then
        edits it, rather than the other way round.
        """
        session = self.cog.multi_profile_channels.get(interaction.channel_id)

        # [NEW] Force hydration if session exists or might exist on disk
        if not session or not session.get("is_hydrated"):
            session = await self.cog.session_manager._ensure_session_hydrated(interaction.channel_id, session.get("type", "multi") if session else "multi")

        if not session:
            await interaction.response.send_message("Session not found.", ephemeral=True)
            return

        session['whisper_waiting'] = session.get('whisper_waiting', 0) + 1
        answered = False
        try:
            if any(session.get(flag) for flag in SESSION_BUSY_FLAGS):
                await interaction.response.edit_message(content=WHISPER_WAITING_NOTICE, embed=None, view=None)
                answered = True

            if not await self.cog.session_manager._wait_for_session_flags(
                session, SESSION_BUSY_FLAGS, WHISPER_BUSY_WAIT_TIMEOUT_SECONDS
            ):
                timed_out = (f"The session is still busy after {int(WHISPER_BUSY_WAIT_TIMEOUT_SECONDS)}s. "
                             "Nothing was regenerated \u2014 try again in a moment.")
                if answered:
                    await interaction.edit_original_response(content=timed_out, embed=None, view=None)
                else:
                    await interaction.response.send_message(timed_out, ephemeral=True)
                return

            session['is_whispering'] = True
        finally:
            session['whisper_waiting'] = max(0, session.get('whisper_waiting', 1) - 1)

        try:
            await self._run_whisper_regeneration(interaction, session, whisper_turn_id, response_turn_id, target_participant, whisper_message, answered)
        finally:
            session['is_whispering'] = False

    async def _run_whisper_regeneration(self, interaction: discord.Interaction, session: Dict, whisper_turn_id: str, response_turn_id: str, target_participant: Dict, whisper_message: str, answered: bool = False):
        """Only called with the channel already claimed by _execute_whisper_regeneration."""
        owner_id = target_participant['owner_id']
        profile_name = target_participant['profile_name']

        user_index = self.cog.profile_manager._get_user_index(owner_id)
        is_borrowed = profile_name in user_index.get("borrowed", [])
        p_settings = self.cog.profile_manager._get_profile_config(owner_id, profile_name, is_borrowed) or {}
        effective_owner_id, effective_profile_name, display_name, avatar_url, custom_emoji = \
            self._whisper_appearance(owner_id, profile_name, p_settings)

        placeholder_embed = self._whisper_embed(interaction, custom_emoji, display_name, avatar_url, whisper_message)
        if answered:
            # The waiting notice already consumed the interaction response.
            await interaction.edit_original_response(content=None, embed=placeholder_embed, view=None)
        else:
            await interaction.response.edit_message(embed=placeholder_embed, view=None)
        placeholder_msg = await interaction.original_response()

        # Reconstruct context for AI
        log = session.get("unified_log", [])
        try:
            old_resp_index = next(i for i, t in enumerate(log) if t.get("turn_id") == response_turn_id)
            sliced_log = log[:old_resp_index]
        except StopIteration:
            await interaction.followup.send("Original response not found in log.", ephemeral=True)
            return

        # [NEW] Hybrid STM for Whisper Regeneration
        batch_start_index = 0
        for i in range(len(sliced_log) - 1, -1, -1):
            if sliced_log[i].get("is_user") is True:
                batch_start_index = i
                break

        bot_pid = self.cog.profile_manager._get_pid_from_name_any(owner_id, profile_name)
        participant_history = self.cog.session_manager._build_history_for_participant(
            sliced_log, bot_pid, p_settings,
            reserved_tail=len(sliced_log) - batch_start_index,
            hide_folded=self.cog.session_manager.compaction_enabled(session),
        )

        try:
            reply = await self._whisper_reply(
                interaction, session, target_participant, p_settings, participant_history,
                placeholder_msg, custom_emoji, display_name, avatar_url, "whisper_regen")
        except asyncio.CancelledError:
            return

        if reply.blocked:
            # The old reply stays in the log; this one only says why it was not replaced.
            await interaction.edit_original_response(embed=self._whisper_embed(
                interaction, ("Regeneration failed.\n\n" + self._whisper_display(reply))[:4096],
                display_name, avatar_url, whisper_message))
            return

        response_text = reply.text
        sources = _format_citation_subtext(reply.sources or [])
        if sources:
            response_text += "\n\n" + "\n".join(sources)

        # Update log
        profile_id = self.cog.profile_manager._get_profile_id(effective_owner_id, effective_profile_name)
        new_content = _format_history_entry(profile_name, datetime.datetime.now(datetime.timezone.utc), response_text, entity_id=profile_id)

        for turn in log:
            if turn.get("turn_id") == response_turn_id:
                turn["content"] = new_content
                turn.pop('thought_signature', None) # Clean up legacy signature
                break

        await self.cog.session_manager._save_session_to_disk((interaction.channel_id, None, None), session.get("type", "multi"), log)
        # No is_hydrated=False here: this rewrites one turn's content, which nothing
        # derives from, and clearing the flag without rehydrating left the session
        # holding a full unified_log that _evict_inactive_sessions then skipped --
        # its entire body is gated on is_hydrated, so the log never got released.

        view = WhisperActionView(self.cog, interaction, whisper_turn_id, response_turn_id, target_participant, whisper_message)
        await interaction.edit_original_response(
            embed=self._whisper_embed(interaction, self._whisper_display(reply), display_name, avatar_url, whisper_message),
            view=view)

    async def _resolve_reply_context(self, message: discord.Message) -> Optional[str]:
        if not message.reference or not message.reference.message_id:
            return None

        try:
            referenced_message = await message.channel.fetch_message(message.reference.message_id)
            author_name = referenced_message.author.display_name
            content = referenced_message.clean_content
            if len(content) > 150:
                content = content[:150] + "..."
            return f"<reply_context author='{author_name}'>\n{content}\n</reply_context>"
        except (discord.NotFound, discord.Forbidden):
            return "<reply_context author='Unknown'>\n[Message could not be loaded]\n</reply_context>"
        except Exception as e:
            print(f"Error resolving reply context: {e}")
            return None
