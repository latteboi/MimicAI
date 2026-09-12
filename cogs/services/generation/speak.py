import time
import uuid
import asyncio
import discord
from typing import Any, Dict, Optional

from ...utils.constants import (
    DEFAULT_SPEAK_REWRITE_STRICT, DEFAULT_SPEAK_REWRITE_LOOSE,
    SPEAK_REWRITE_HISTORY_TURNS, SPEAK_REWRITE_MAX_INPUT_CHARS,
    PLACEHOLDER_EMOJI,
)
from ...utils.helpers import (
    _format_history_entry, _resolve_safety_settings, _scrub_response_text,
    resolve_native_tools, resolve_thinking_params,
)
from ...managers.session_manager import intern_turn


class SpeakAsMixin:
    """The `/speak` one-off message injection, verbatim or re-voiced in character."""

    async def _execute_speak_as(self, interaction_to_respond: discord.Interaction, channel: discord.abc.Messageable, author: discord.User, profile_name: str, message: str, method: str, style: str = 'verbatim', fidelity: str = 'strict'):
        ctx = await self._resolve_speak_target(interaction_to_respond, channel, author, profile_name, method)
        if ctx is None:
            return

        if style != 'in_character':
            if await self._deliver_speak_as(interaction_to_respond, ctx, message):
                await interaction_to_respond.followup.send("Message sent.", ephemeral=True)
            else:
                await interaction_to_respond.followup.send(
                    "That profile is no longer seated in this channel's session.", ephemeral=True)
            return

        if len(message) > SPEAK_REWRITE_MAX_INPUT_CHARS:
            await interaction_to_respond.followup.send(
                f"That line is {len(message):,} characters. In-character delivery takes at most "
                f"{SPEAK_REWRITE_MAX_INPUT_CHARS:,} — shorten it, or send it verbatim.",
                ephemeral=True)
            return

        from ...gui.gui_sessions import SpeakPreviewView, build_speak_placeholder_embed

        # The card goes up before the wait, not after it. A rewrite is a full model call
        # against the character's own prompt, and until it lands the author has an
        # ephemeral that says nothing and a channel where nothing is happening -- the
        # same dead-command problem WHISPER_WAITING_NOTICE solves. It is then edited into
        # the preview rather than replaced, so this costs no extra message.
        placeholder = await interaction_to_respond.followup.send(
            embed=build_speak_placeholder_embed(self.cog, ctx, message, fidelity),
            ephemeral=True, wait=True)

        rewritten, error = await self._rewrite_in_character(ctx, message, fidelity, placeholder)
        if not rewritten:
            # Never silently fall back to posting the raw text: the author asked for the
            # character's voice, and delivering their own words under the character's
            # face is a different message than the one they approved.
            await placeholder.edit(
                content=f"{ctx['speaker_display_name']} could not deliver that line.\n-# {error}",
                embed=None)
            return

        view = SpeakPreviewView(self.cog, interaction_to_respond, ctx, message, rewritten, fidelity)
        await placeholder.edit(embed=view.build_embed(), view=view)

    # --- Resolution -----------------------------------------------------------

    async def _resolve_speak_target(self, interaction_to_respond: discord.Interaction, channel: discord.abc.Messageable, author: discord.User, profile_name: str, method: str) -> Optional[Dict[str, Any]]:
        """Every gate `/speak` applies, resolved once into the context both styles need.

        Split out of _execute_speak_as so the in-character path can resolve, generate,
        preview and only then deliver -- three entry points into the same gate rather
        than a second copy of it.
        """
        user_id = author.id
        index = self.cog.profile_manager._get_user_index(user_id)

        is_borrowed = profile_name in index.get("borrowed", [])
        is_personal = profile_name in index.get("personal", [])

        if not is_borrowed and not is_personal:
            await interaction_to_respond.followup.send(f"You do not have a profile named '{profile_name}'.", ephemeral=True)
            return None

        if isinstance(channel, (discord.TextChannel, discord.Thread)):
            if not channel.permissions_for(channel.guild.me).send_messages:
                await interaction_to_respond.followup.send("I do not have permission to send messages in that channel.", ephemeral=True)
                return None

        effective_owner_id = user_id
        effective_profile_name = profile_name
        profile_data_source = {}

        if is_borrowed:
            borrowed_data = self.cog.profile_manager._get_profile_config(user_id, profile_name, True) or {}
            effective_owner_id = int(borrowed_data.get("original_owner_id", user_id))
            effective_profile_name = borrowed_data.get("original_profile_name", profile_name)
            profile_data_source = self.cog.profile_manager._get_profile_config(effective_owner_id, effective_profile_name, False) or {}
            own_config = borrowed_data
        else:
            profile_data_source = self.cog.profile_manager._get_profile_config(user_id, profile_name, False) or {}
            own_config = profile_data_source

        if not self.cog.profile_manager._check_unrestricted_safety_policy(effective_owner_id, effective_profile_name, channel):
            await interaction_to_respond.followup.send("This profile is rated 'Adult 18+' and cannot speak in this channel because it is not marked as Age-Restricted.", ephemeral=True)
            return None

        delivery_method = 'webhook'
        child_bot_id = None

        session = self.cog.multi_profile_channels.get(channel.id)
        
        # [NEW] Force hydration if session exists or might exist on disk
        if not session or not session.get("is_hydrated"):
            session = await self.cog.session_manager._ensure_session_hydrated(channel.id, "multi")

        if session:
            participant_data = next((p for p in session.get("profiles", []) if p.get("owner_id") == user_id and p.get("profile_name") == profile_name), None)
            if not participant_data:
                await interaction_to_respond.followup.send(f"The profile '{profile_name}' is not an active participant in this channel's multi-profile session.", ephemeral=True)
                return None

            session_method = participant_data.get("method", "webhook")
            child_bot_id = participant_data.get("bot_id")

            if method == 'auto':
                delivery_method = session_method
            elif method == 'child_bot':
                if session_method == 'child_bot' and child_bot_id:
                    delivery_method = 'child_bot'
                else:
                    await interaction_to_respond.followup.send(f"This profile is not configured to use a Child Bot in this session. Cannot use 'child_bot' method.", ephemeral=True)
                    return None
        else:
            linked_bot_id = next((bot_id for bot_id, data in self.cog.child_bots.items() if data.get("owner_id") == effective_owner_id and data.get("profile_name") == effective_profile_name), None)
            if linked_bot_id and channel.guild.get_member(int(linked_bot_id)):
                child_bot_id = linked_bot_id

            if method == 'auto':
                if child_bot_id:
                    delivery_method = 'child_bot'
            elif method == 'child_bot':
                if child_bot_id:
                    delivery_method = 'child_bot'
                else:
                    await interaction_to_respond.followup.send(f"The profile '{effective_profile_name}' is not linked to a Child Bot, or that bot is not present in this server. Cannot use 'child_bot' method.", ephemeral=True)
                    return None

        speaker_display_name = effective_profile_name
        appearance_data = self.cog.profile_manager._get_user_appearance(effective_owner_id, effective_profile_name)
        if appearance_data.get("custom_display_name"):
            speaker_display_name = appearance_data["custom_display_name"]

        return {
            "channel": channel,
            "author": author,
            "user_id": user_id,
            # The invoker's handle on the profile -- a borrow keeps its own config, so
            # this is what the persona, the model slots and the thinking level resolve
            # through, matching _multi_profile_worker.
            "profile_name": profile_name,
            "is_borrowed": is_borrowed,
            # The original, for appearance, safety policy and the child-bot link.
            "effective_owner_id": effective_owner_id,
            "effective_profile_name": effective_profile_name,
            "profile_data_source": profile_data_source,
            "delivery_method": delivery_method,
            "child_bot_id": child_bot_id,
            "speaker_display_name": speaker_display_name,
            "profile_id": self.cog.profile_manager._get_profile_id(effective_owner_id, effective_profile_name),
            # Read off the invoker's handle, like every other config value here: a borrow
            # owns its own placeholder emoji. Carried in ctx so the preview and the
            # waiting card it grows out of are built from one resolution.
            "placeholder_emoji": own_config.get("placeholder_emoji") or PLACEHOLDER_EMOJI,
        }

    # --- In-character rewrite -------------------------------------------------

    async def _rewrite_in_character(self, ctx: Dict[str, Any], source_text: str, fidelity: str, placeholder_msg: Optional[discord.Message] = None):
        """Re-voice `source_text` as the character. Returns (text, error_message).

        `placeholder_msg` is the ephemeral card to tick while the model works. Absent,
        the generation runs exactly as before and nothing is edited.

        The prompt is the one the profile would get for an ordinary turn --
        _construct_system_instructions unchanged, history from
        _build_history_for_participant unchanged -- with the directive appended as the
        last part of the final user turn.

        That position is the feature. Every block in the system instruction says "you
        are mid-scene, continue the conversation", and <context_rules> closes with
        "Always respond as yourself"; a rewrite directive placed anywhere earlier loses
        to them and the model answers the transcript instead of re-voicing the line.

        The history tail is SPEAK_REWRITE_HISTORY_TURNS rather than the profile's STM
        window for the same reason: enough to catch the tone the scene is in, not so
        much that the pull to continue it wins anyway.
        """
        channel = ctx["channel"]
        owner_id = ctx["user_id"]
        profile_name = ctx["profile_name"]
        guild_id = channel.guild.id if getattr(channel, "guild", None) else 0

        p_settings = self.cog.profile_manager._get_profile_config(
            owner_id, profile_name, ctx["is_borrowed"]) or {}

        template = (DEFAULT_SPEAK_REWRITE_LOOSE if fidelity == 'loose'
                    else DEFAULT_SPEAK_REWRITE_STRICT)
        key = "SPEAK_REWRITE_LOOSE" if fidelity == 'loose' else "SPEAK_REWRITE_STRICT"
        directive = self.cog.global_prompts.get(key, template).format(source_text=source_text.strip())

        session = self.cog.multi_profile_channels.get(channel.id)
        contents = []
        if session:
            bot_pid = self.cog.profile_manager._get_pid_from_name_any(owner_id, profile_name)
            contents = self.cog.session_manager._build_history_for_participant(
                session.get("unified_log", []), bot_pid,
                {**p_settings, "stm_length": SPEAK_REWRITE_HISTORY_TURNS},
            )

        # Its own final user turn where the history allows one. Merged into a trailing
        # user turn only when there is one, which is what the adapters' role handling
        # everywhere else in the codebase expects.
        if contents and contents[-1].get('role') == 'user':
            contents[-1]['parts'].append(directive)
        else:
            contents.append({'role': 'user', 'parts': [directive]})

        try:
            system_instruction, _, _, temp, top_p, top_k, primary_model, fallback_model = await asyncio.to_thread(
                self._construct_system_instructions,
                owner_id, profile_name, channel.id, is_multi_profile=bool(session),
            )
        except Exception as e:
            return None, f"Could not build the character's prompt: {e}"

        safety_settings = _resolve_safety_settings(channel, p_settings)
        tools = resolve_native_tools(p_settings)

        gen_config = {"temperature": temp, "top_p": top_p, "top_k": top_k}
        # Mutated in place by every attempt, so a fallback keeps ticking the card the
        # primary was already ticking rather than opening a second one beside it.
        state_container = {
            "custom_emoji": ctx.get("placeholder_emoji") or PLACEHOLDER_EMOJI,
            "placeholder_msg": placeholder_msg,
        }

        async def _attempt(model_name, is_fallback):
            # The response slot, not `utility`: this is a visible in-character line and
            # gets what a real one gets. `utility` is for internal passes with no
            # profile setting to hang off, and its shipped default is low/512.
            model = self.cog.api_service._instantiate_model(
                model_name, guild_id, owner_id, system_instruction, safety_settings,
                resolve_thinking_params(p_settings, "response",
                                        "fallback" if is_fallback else "primary"),
                tools, p_settings, config_owner_id=owner_id,
            )
            return await self._generate_with_heartbeat(
                model, contents, gen_config, channel, None, None,
                is_fallback=is_fallback, message_type="embed",
                existing_state=state_container)

        status = "api_error"
        try:
            result, model_used, _was_fallback = await self.cog.api_service.run_with_fallback(
                primary_model, fallback_model, _attempt, label="Speak rewrite")
            response, _state = result
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.cog._log_api_call(user_id=owner_id, guild_id=guild_id, context="speak_rewrite",
                                   model_used=primary_model, status=status)
            return None, str(e)

        if not response or not response.candidates:
            self.cog._log_api_call(user_id=owner_id, guild_id=guild_id, context="speak_rewrite",
                                   model_used=model_used, status="blocked_by_safety")
            return None, "The response was empty or blocked by a safety filter."

        self.cog._log_api_call(user_id=owner_id, guild_id=guild_id, context="speak_rewrite",
                               model_used=model_used, status="success")

        text = (getattr(response, 'text', "") or "").strip()
        text, _ = self._extract_and_apply_neuro_state(text, owner_id, profile_name)

        # The model routinely echoes the tag it was addressed in. Both are in
        # SYSTEM_XML_TAGS, so PATTERN_SYSTEM_XML_BLOCKS would delete the entire reply
        # rather than the wrapper -- the same trap _run_whisper_turn strips for
        # <private_response>. Unwrap before scrubbing, not after.
        for tag in ("rewrite_request", "source_text"):
            text = text.replace(f"<{tag}>", "").replace(f"</{tag}>", "")

        names = [ctx["speaker_display_name"], ctx["effective_profile_name"], profile_name]
        if session:
            for p in session.get("profiles", []):
                names.append(p['profile_name'])
                other_name, _ = self._resolve_appearance_data(p['owner_id'], p['profile_name'])
                names.append(other_name)

        text = _scrub_response_text(text, participant_names=names).strip()
        if not text:
            return None, "The response was empty after cleanup."
        return text, None

    # --- Delivery -------------------------------------------------------------

    async def _deliver_speak_as(self, interaction_to_respond: discord.Interaction, ctx: Dict[str, Any], text: str, *, style: str = 'verbatim') -> bool:
        """Log the turn and send it. Returns False if the session moved out from under us.

        The session is re-read here rather than carried in `ctx`: the in-character path
        puts a preview and an author's attention span between resolution and delivery,
        and in that gap the session can be suspended, purged, or reseated without this
        profile.
        """
        channel = ctx["channel"]
        author = ctx["author"]

        session = self.cog.multi_profile_channels.get(channel.id)
        if session:
            if not session.get("is_hydrated"):
                session = await self.cog.session_manager._ensure_session_hydrated(channel.id, session.get("type", "multi"))
            if session and not any(p.get("owner_id") == ctx["user_id"] and p.get("profile_name") == ctx["profile_name"]
                                   for p in session.get("profiles", [])):
                return False

        history_line = _format_history_entry(
            ctx["speaker_display_name"], interaction_to_respond.created_at, text,
            entity_id=ctx["profile_id"])

        display_message = f"{text}\n\n||-# {'Directed by' if style == 'in_character' else 'Authored by'} {author.mention} ({author.id}).||"

        turn_object = None
        if session:
            turn_id = str(uuid.uuid4())
            turn_object = {
                "turn_id": turn_id,
                "is_user": False,
                "speaker_pid": self.cog.profile_manager._get_pid_from_name_any(ctx["user_id"], ctx["profile_name"]),
                "owner_id": ctx["effective_owner_id"],
                "profile_name": ctx["effective_profile_name"],
                "message_ids": [],
                "content": history_line,
                # Who put these words in the character's mouth. The attribution used to
                # live only in the Discord message body, so a reloaded log could not tell
                # a puppeted turn from a generated one -- not for the critic, not for
                # compaction, and not for whoever is reading the log afterwards.
                "authored_by": author.id,
                "speak_style": style,
            }
            session.setdefault("unified_log", []).append(intern_turn(turn_object))

            session_type = session.get("type", "multi")
            self.cog.session_last_accessed[channel.id] = time.time()
            # message_ids land below once the send returns, and that flush persists both.
            self.cog.session_manager.mark_session_dirty((channel.id, None, None), session_type)

        sent_messages = []
        if ctx["delivery_method"] == 'child_bot' and ctx["child_bot_id"]:
            profile_data_source = ctx["profile_data_source"]
            payload = {
                "channel_id": channel.id,
                "content": display_message,
                "realistic_typing": profile_data_source.get("realistic_typing_enabled", False),
                "typing_cps": profile_data_source.get("typing_cps", 30.0),
                "typing_max_delay": profile_data_source.get("typing_max_delay", 2.5),
                "typing_mode": profile_data_source.get("typing_mode", "sentence")
            }
            sent_messages = await self.cog.child_bot_manager.execute_send(ctx["child_bot_id"], payload)
        else:
            sent_messages = await self._send_channel_message(
                channel, display_message,
                profile_owner_id_for_appearance=ctx["effective_owner_id"],
                profile_name_for_appearance=ctx["effective_profile_name"]
            )

        if sent_messages and turn_object and session:
            for msg in sent_messages:
                turn_object.setdefault("message_ids", []).append(msg.id)
            session['last_bot_message_id'] = sent_messages[-1].id
            await self.cog.session_manager.flush_session((channel.id, None, None), session.get("type", "multi"))

        return True
