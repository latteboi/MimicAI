import re
import time
import uuid
import base64
import asyncio
import discord
import datetime
from typing import Dict, Optional

from ...utils.constants import defaultConfig, PLACEHOLDER_EMOJI, DEFAULT_WHISPER_INJECTION
from ...utils.helpers import (
    _add_inline_citations, _format_citation_subtext, _format_history_entry,
    _get_user_hash, _scrub_response_text,
)
from ...gui.gui_sessions import WhisperActionView


class WhisperMixin:
    """Private (ephemeral) whisper exchanges with a single multi-profile
    participant, their regeneration, and reply-context resolution for replies
    to a previous message.
    """

    async def _execute_whisper(self, interaction: discord.Interaction, target_participant: Dict, whisper_message: str):
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

        user_index = self.cog.profile_manager._get_user_index(owner_id)
        is_borrowed = profile_name in user_index.get("borrowed", [])
        p_settings = self.cog.profile_manager._get_profile_config(owner_id, profile_name, is_borrowed) or {}

        # Get model and settings for the target profile
        model, _, temp, top_p, top_k, _, fallback_model_name = await self.cog.api_service._get_or_create_model_for_channel(
            interaction.channel_id, owner_id, interaction.guild.id,
            profile_owner_override=owner_id, profile_name_override=profile_name
        )
        if not model:
            await interaction.followup.send("Could not initialize the AI model for that profile.", ephemeral=True)
            return

        # Construct the prompt for the private response
        user_hash = _get_user_hash(interaction.user.id)
        whisper_content = _format_history_entry(interaction.user.name, interaction.created_at, whisper_message, entity_id=user_hash)

        api_whisper_prompt = self.cog.global_prompts.get("WHISPER_INJECTION", DEFAULT_WHISPER_INJECTION).format(whisper_content=whisper_content.strip())

        # Derived from unified_log, the single source of truth, rather than a shadow copy
        # maintained by incremental appends. _build_history_for_participant already rewrites
        # this participant's own whispers and private responses into their XML tags, and
        # hides other participants' — so the privacy boundary is enforced in one place.
        bot_pid = self.cog.profile_manager._get_pid_from_name_any(owner_id, profile_name)
        contents_for_api_call = self.cog.session_manager._build_history_for_participant(
            session.get("unified_log", []), bot_pid, p_settings
        )

        # Ensure alternating roles by appending to the last user turn if present
        if contents_for_api_call and contents_for_api_call[-1].get('role', 'user') == 'user':
            contents_for_api_call[-1]['parts'].append(api_whisper_prompt)
        else:
            contents_for_api_call.append({'role': 'user', 'parts': [api_whisper_prompt]})

        # Enable internal thoughts but keep summary display off (UI ignores it)
        gen_config = {
            "temperature": temp, "top_p": top_p, "top_k": top_k,
            "thinking_config": {"include_thoughts": True}
        }

        status = "api_error"
        response = None

        # Resolve appearance and identity for placeholder
        effective_owner_id, effective_profile_name = self.cog.profile_manager._resolve_effective_profile(owner_id, profile_name)

        custom_emoji = p_settings.get("placeholder_emoji") or PLACEHOLDER_EMOJI

        display_name = effective_profile_name
        appearance = self.cog.user_appearances.get(str(effective_owner_id), {}).get(effective_profile_name)
        avatar_url = self.cog.bot.user.display_avatar.url
        if appearance:
            display_name = appearance.get("custom_display_name") or display_name
            avatar_url = appearance.get("custom_avatar_url") or avatar_url

        # --- SEND PLACEHOLDER EMBED ---
        placeholder_embed = discord.Embed(description=f"{custom_emoji}", color=discord.Color.dark_grey())
        placeholder_embed.set_author(name=display_name, icon_url=avatar_url)
        placeholder_embed.set_footer(text=f"{whisper_message}"[:1000], icon_url=interaction.user.display_avatar.url)
        placeholder_msg = await interaction.followup.send(embed=placeholder_embed, ephemeral=True, wait=True)

        try:
            if not model:
                raise ValueError("Could not initialize the AI model.")

            t_start = time.time()
            response, state_container = await self._generate_with_heartbeat(
                model, contents_for_api_call, gen_config, interaction.channel, None, placeholder_msg.id, is_fallback=False, message_type="embed", existing_state={"custom_emoji": custom_emoji, "placeholder_msg": placeholder_msg}
            )
            status = "blocked_by_safety" if not response or not response.candidates else "success"
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"Whisper generation error: {e}")
            status = "api_error"
        finally:
            if 'state_container' in locals() and state_container:
                await self._safe_delete_placeholder(interaction.channel, state_container.get('msg_b_id'))
            self.cog._log_api_call(user_id=interaction.user.id, guild_id=interaction.guild.id, context="whisper", model_used=model, status=status)

        response_text = "..."
        if not response or not response.candidates:
            reason = "Safety Filter"
            if response and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                reason = response.prompt_feedback.block_reason.name.replace('_', ' ').title()

            p_index = self.cog.profile_manager._get_user_index(owner_id)
            p_is_borrowed = profile_name in p_index.get("borrowed",[])
            p_settings = self.cog.profile_manager._get_profile_config(owner_id, profile_name, p_is_borrowed) or {}

            custom_main = p_settings.get("error_response", "An error has occurred.")

            err_embed = placeholder_msg.embeds[0]
            err_embed.description = f"{custom_main}\n\n-# Blocked due to: **{reason}**."
            await placeholder_msg.edit(embed=err_embed)
            return
        elif response.candidates:
            response_text = getattr(response, 'text', "...")
            if hasattr(response, 'raw') and response.raw.candidates and hasattr(response.raw.candidates[0], 'grounding_metadata'):
                response_text = _add_inline_citations(response_text, response.raw.candidates[0].grounding_metadata)
            response_text = response_text.strip()

        response_text, _ = self._extract_and_apply_neuro_state(response_text, owner_id, profile_name)

        # PREVENT GLOBAL XML SCRUBBER FROM DELETING THE RESPONSE
        response_text = re.sub(r'</?private_response>', '', response_text, flags=re.IGNORECASE)
        response_text = re.sub(r'</?whisper_context>', '', response_text, flags=re.IGNORECASE)
        response_text = re.sub(r'</?private_context>', '', response_text, flags=re.IGNORECASE)

        response_text = _scrub_response_text(response_text, participant_names=[display_name])

        # [NEW] Safety Fallback for empty responses
        if not response_text or not response_text.strip():
            p_index = self.cog.profile_manager._get_user_index(owner_id)
            p_is_borrowed = profile_name in p_index.get("borrowed", [])
            p_settings = self.cog.profile_manager._get_profile_config(owner_id, profile_name, p_is_borrowed) or {}
            response_text = p_settings.get("error_response", "...")

        if response_text:
            grounding_sources = []
            if hasattr(response, 'raw') and response.raw.candidates:
                if hasattr(response.raw.candidates[0], 'grounding_metadata'):
                    metadata = response.raw.candidates[0].grounding_metadata
                    if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks is not None:
                        for chunk in metadata.grounding_chunks:
                            if hasattr(chunk, 'web'):
                                grounding_sources.append({'uri': chunk.web.uri, 'title': chunk.web.title})

                if hasattr(response.raw.candidates[0], 'url_context_metadata'):
                    url_metadata = response.raw.candidates[0].url_context_metadata
                    if hasattr(url_metadata, 'url_metadata') and url_metadata.url_metadata is not None:
                        for u in url_metadata.url_metadata:
                            if hasattr(u, 'retrieved_url') and u.retrieved_url:
                                grounding_sources.append({'uri': u.retrieved_url, 'title': 'URL Context'})

            sources_text_list = _format_citation_subtext(grounding_sources)
            if sources_text_list:
                response_text += "\n\n" + "\n".join(sources_text_list)

        whisper_turn_id = str(uuid.uuid4())
        user_hash = _get_user_hash(interaction.user.id)
        whisper_content = _format_history_entry(interaction.user.name, interaction.created_at, whisper_message, entity_id=user_hash)

        target_pid = self.cog.profile_manager._get_pid_from_name_any(owner_id, profile_name)

        session.setdefault("unified_log", []).append({
            "turn_id": whisper_turn_id, "type": "whisper",
            "is_user": True, "speaker_pid": str(interaction.user.id), "target_pid": target_pid,
            "message_ids":[],
            "content": whisper_content,
            "timestamp": interaction.created_at.isoformat()
        })

        response_turn_id = str(uuid.uuid4())
        profile_id = self.cog.profile_manager._get_profile_id(effective_owner_id, effective_profile_name)
        response_content = _format_history_entry(profile_name, datetime.datetime.now(datetime.timezone.utc), response_text, entity_id=profile_id)

        p_index = self.cog.profile_manager._get_user_index(owner_id)
        p_is_borrowed = profile_name in p_index.get("borrowed",[])
        p_settings = self.cog.profile_manager._get_profile_config(owner_id, profile_name, p_is_borrowed) or {}

        resp_log = {
            "turn_id": response_turn_id, "type": "private_response",
            "is_user": False, "speaker_pid": target_pid, "target_id": interaction.user.id,
            "message_ids":[],
            "content": response_content,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

        session.setdefault("unified_log", []).append(resp_log)

        # Both turns are already in unified_log above; _build_history_for_participant
        # wraps them in <private_whisper> / <private_response> when it derives this
        # participant's history, so there is no second copy to maintain here.

        # Add to pending whispers to be injected into the next public turn
        session.setdefault("pending_whispers", {}).setdefault(participant_key, []).append(whisper_content)

        # [NEW] Immediate persistence for private whisper turns
        session_type = session.get("type", "multi")
        await self.cog.session_manager._save_session_to_disk((interaction.channel_id, None, None), session_type, session["unified_log"])

        # Send the private response to the user
        embed = discord.Embed(description=response_text, color=discord.Color.dark_grey())
        embed.set_author(name=display_name, icon_url=avatar_url)
        embed.set_footer(text=f"{whisper_message}"[:1000], icon_url=interaction.user.display_avatar.url)

        view = WhisperActionView(self, interaction, whisper_turn_id, response_turn_id, target_participant, whisper_message)
        resp_msg = await placeholder_msg.edit(embed=embed, view=view)

        # Inject the message ID back into the log turn
        if resp_msg:
            resp_log["message_ids"] = [resp_msg.id]
            await self.cog.session_manager._save_session_to_disk((interaction.channel_id, None, None), session_type, session["unified_log"])

    async def _execute_whisper_regeneration(self, interaction: discord.Interaction, whisper_turn_id: str, response_turn_id: str, target_participant: Dict, whisper_message: str):
        session = self.cog.multi_profile_channels.get(interaction.channel_id)

        # [NEW] Force hydration if session exists or might exist on disk
        if not session or not session.get("is_hydrated"):
            session = await self.cog.session_manager._ensure_session_hydrated(interaction.channel_id, session.get("type", "multi") if session else "multi")

        if not session:
            await interaction.response.send_message("Session not found.", ephemeral=True)
            return

        owner_id = target_participant['owner_id']
        profile_name = target_participant['profile_name']
        participant_key = (owner_id, profile_name)

        if not session.get("is_hydrated"):
            session = await self.cog.session_manager._ensure_session_hydrated(interaction.channel_id, session.get("type", "multi"))

        model, _, temp, top_p, top_k, _, fallback_model_name = await self.cog.api_service._get_or_create_model_for_channel(
            interaction.channel_id, interaction.user.id, interaction.guild.id,
            profile_owner_override=owner_id, profile_name_override=profile_name
        )

        # Resolve appearance
        effective_owner_id, effective_profile_name = self.cog.profile_manager._resolve_effective_profile(owner_id, profile_name)

        user_index = self.cog.profile_manager._get_user_index(owner_id)
        is_borrowed = profile_name in user_index.get("borrowed", [])
        p_settings = self.cog.profile_manager._get_profile_config(owner_id, profile_name, is_borrowed) or {}
        custom_emoji = p_settings.get("placeholder_emoji") or PLACEHOLDER_EMOJI

        display_name = effective_profile_name
        appearance = self.cog.user_appearances.get(str(effective_owner_id), {}).get(effective_profile_name)
        avatar_url = self.cog.bot.user.display_avatar.url
        if appearance:
            display_name = appearance.get("custom_display_name") or display_name
            avatar_url = appearance.get("custom_avatar_url") or avatar_url

        # --- IMMEDIATE EDIT TO PLACEHOLDER ---
        placeholder_embed = discord.Embed(description=f"{custom_emoji}", color=discord.Color.dark_grey())
        placeholder_embed.set_author(name=display_name, icon_url=avatar_url)
        placeholder_embed.set_footer(text=f"{whisper_message}"[:1000], icon_url=interaction.user.display_avatar.url)

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

        past_log = sliced_log[:batch_start_index]
        current_batch_log = sliced_log[batch_start_index:]

        stm_length = int(p_settings.get("stm_length", defaultConfig.CHATBOT_MEMORY_LENGTH))
        if stm_length > 0:
            past_log = past_log[-stm_length:]
        else:
            past_log = []

        combined_log = past_log + current_batch_log

        bot_pid = self.cog.profile_manager._get_pid_from_name_any(owner_id, profile_name)
        participant_history = self.cog.session_manager._build_history_for_participant(combined_log, bot_pid, p_settings)

        gen_config = {"temperature": temp, "top_p": top_p, "top_k": top_k, "thinking_config": {"include_thoughts": True}}

        status = "api_error"
        response = None
        t_start = time.time()
        try:
            response, state_container = await self._generate_with_heartbeat(
                model, participant_history, gen_config, interaction.channel, None, placeholder_msg.id, is_fallback=False, message_type="embed", existing_state={"custom_emoji": custom_emoji, "placeholder_msg": placeholder_msg}
            )
            status = "success"
        except asyncio.CancelledError: return
        except Exception: status = "api_error"
        finally:
            if 'state_container' in locals() and state_container:
                await self._safe_delete_placeholder(interaction.channel, state_container.get('msg_b_id'))
            self.cog._log_api_call(user_id=interaction.user.id, guild_id=interaction.guild.id, context="whisper_regen", model_used=model, status=status)

        if not response or not response.candidates:
            err_embed = placeholder_embed.copy()
            err_embed.description = "Regeneration failed."
            await interaction.edit_original_response(embed=err_embed)
            return

        response_text = getattr(response, 'text', "...").strip()
        response_text, _ = self._extract_and_apply_neuro_state(response_text, owner_id, profile_name)

        response_text = re.sub(r'</?private_response>', '', response_text, flags=re.IGNORECASE)
        response_text = _scrub_response_text(response_text, participant_names=[display_name])

        # [NEW] Safety Fallback for empty responses
        if not response_text or not response_text.strip():
            p_index = self.cog.profile_manager._get_user_index(owner_id)
            p_is_borrowed = profile_name in p_index.get("borrowed", [])
            p_settings = self.cog.profile_manager._get_profile_config(owner_id, profile_name, p_is_borrowed) or {}
            response_text = p_settings.get("error_response", "...")

        # Update log
        profile_id = self.cog.profile_manager._get_profile_id(effective_owner_id, effective_profile_name)
        new_content = _format_history_entry(profile_name, datetime.datetime.now(datetime.timezone.utc), response_text, entity_id=profile_id)

        for turn in log:
            if turn.get("turn_id") == response_turn_id:
                turn["content"] = new_content
                turn.pop('thought_signature', None) # Clean up legacy signature
                break

        await self.cog.session_manager._save_session_to_disk((interaction.channel_id, None, None), session.get("type", "multi"), log)
        session["is_hydrated"] = False

        # Final Embed Update
        final_embed = placeholder_embed.copy()
        final_embed.description = response_text
        view = WhisperActionView(self, interaction, whisper_turn_id, response_turn_id, target_participant, whisper_message)
        await interaction.edit_original_response(embed=final_embed, view=view)

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
