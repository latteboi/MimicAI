import time
import re
import base64
import asyncio
import discord
import datetime
from typing import Dict

from ...utils.constants import (
    defaultConfig, PRIMARY_MODEL_NAME, FALLBACK_MODEL_NAME, PLACEHOLDER_EMOJI,
    ERR_GENERAL_ERROR, ERR_RATE_LIMIT, ERR_REASON_EMPTY_RESPONSE, ERR_REASON_TIMEOUT_BOTH, ERR_SAFETY_BLOCK,
    WARN_BOTH_MODELS_FAILED, WARN_FALLBACK_USED, WARN_MAIN_MODEL_FAILED,
    DEFAULT_KICKSTART_START, DEFAULT_KICKSTART_IDLE, DEFAULT_IMAGE_PRESENT, DEFAULT_WHISPER_RECAP,
)
from ...utils.helpers import _add_inline_citations, _format_api_error, _format_history_entry, _scrub_response_text
from ._shared import _resolve_safety_settings, _strip_neuro_update_and_scrub


class RegenerationMixin:
    """Re-runs generation for a single existing turn (triggered by the regenerate
    reaction), editing the message in place instead of sending a new one.
    """

    async def _execute_regeneration(self, payload: discord.RawReactionActionEvent, session: Dict, turn_id: str, participant: Dict):
        channel = self.cog.bot.get_channel(payload.channel_id)
        if not channel: return

        # 1. State Locking & Pre-emptive Persistence
        session['is_regenerating'] = True

        actual_turn_index = -1
        target_turn = None
        for i, t in enumerate(session.get("unified_log", [])):
            if t.get("turn_id") == turn_id:
                actual_turn_index = i
                target_turn = t
                break

        if not target_turn:
            session['is_regenerating'] = False
            return

        p_owner_id = participant['owner_id']
        p_name = participant['profile_name']
        p_key = (p_owner_id, p_name)
        bot_pid = self.cog.profile_manager._get_pid_from_name_any(p_owner_id, p_name)

        p_index = self.cog.profile_manager._get_user_index(p_owner_id)
        p_is_borrowed = p_name in p_index.get("borrowed", [])
        p_profile = self.cog.profile_manager._get_profile_config(p_owner_id, p_name, p_is_borrowed) or {}

        custom_emoji = p_profile.get("placeholder_emoji") or PLACEHOLDER_EMOJI

        # Pre-emptive visual feedback before disk I/O and context gathering
        if participant.get('method') == 'child_bot':
            await self.cog.manager_queue.put({
                "action": "send_to_child", "bot_id": participant['bot_id'],
                "payload": {
                    "action": "regenerate_message", "channel_id": channel.id,
                    "message_id": payload.message_id, "content": custom_emoji
                }
            })
        else:
            wh = await self.cog.server_manager._get_or_create_webhook(channel)
            if wh:
                try:
                    msg = await channel.fetch_message(payload.message_id)
                    kept_atts = [a for a in msg.attachments if a.content_type and a.content_type.startswith("image/")]
                    await wh.edit_message(payload.message_id, content=custom_emoji, attachments=kept_atts)
                except Exception: pass

        session_type = session.get("type", "multi")
        dummy_key = (channel.id, None, None)

        # Flush session state to disk in parallel with initial cleanups
        self.cog.session_manager._save_multi_profile_sessions()
        await self.cog.session_manager._save_session_to_disk(dummy_key, session_type, session["unified_log"])

        try:
            message_ids_to_check = target_turn.get("message_ids", [])

            # 2. Cleanup follow-up messages
            for msg_id in message_ids_to_check:
                if msg_id == payload.message_id: continue
                try:
                    msg = await channel.fetch_message(msg_id)
                    if not msg: continue
                    is_sources = "Sources:" in msg.content
                    has_image = any(a.content_type and a.content_type.startswith("image/") for a in msg.attachments)
                    if not is_sources and not has_image:
                        self.cog.purged_message_ids[msg_id] = True
                        await msg.delete()
                except Exception: pass

            # 3. History Slicing (Time Travel)
            sliced_unified_log = session["unified_log"][:actual_turn_index]

            # [NEW] Hybrid STM for Regeneration
            batch_start_index = 0
            for i in range(len(sliced_unified_log) - 1, -1, -1):
                if sliced_unified_log[i].get("is_user") is True:
                    batch_start_index = i
                    break

            past_log = sliced_unified_log[:batch_start_index]
            current_batch_log = sliced_unified_log[batch_start_index:]

            stm_length = int(p_profile.get("stm_length", defaultConfig.CHATBOT_MEMORY_LENGTH))
            if stm_length > 0:
                past_log = past_log[-stm_length:]
            else:
                past_log = []

            combined_log = past_log + current_batch_log

            participant_history = self.cog.session_manager._build_history_for_participant(combined_log, bot_pid, p_profile)
            pending_whispers_for_regen = self.cog.session_manager._get_pending_whispers_for_participant(combined_log, bot_pid)

            # Pseudo-turn injection to ensure history ends with a 'user' role
            if participant_history and participant_history[-1].get('role', 'user') == 'model':
                participant_history.append({'role': 'user', 'parts': [self.cog.global_prompts.get("KICKSTART_IDLE", DEFAULT_KICKSTART_IDLE)]})
            elif not participant_history:
                participant_history.append({'role': 'user', 'parts': [self.cog.global_prompts.get("KICKSTART_START", DEFAULT_KICKSTART_START)]})

            # 4. Re-run Generation
            model, _, temp, top_p, top_k, _, fallback_model_name = await self.cog.api_service._get_or_create_model_for_channel(
                channel.id, payload.user_id, channel.guild.id,
                profile_owner_override=p_owner_id, profile_name_override=p_name
            )
            if not model: return

            last_user_turn = next((t for t in reversed(sliced_unified_log) if t.get("is_user") is True), None)
            trigger_content = last_user_turn.get("content", "") if last_user_turn else ""

            # Media Recovery Logic for Regeneration (Handles both User & Bot Generated Media)
            recovered_media_parts = []
            
            # 1. Check if the bot message being regenerated has an attached generated image
            regen_msg = None
            try:
                regen_msg = await channel.fetch_message(payload.message_id)
            except Exception:
                pass

            is_generated_image_turn = False
            if regen_msg and regen_msg.attachments:
                bot_image_attachments = [a for a in regen_msg.attachments if a.content_type and a.content_type.startswith("image/")]
                if bot_image_attachments:
                    is_generated_image_turn = True
                    
                    # Extract original prompt from user turn
                    prompt_text = ""
                    if last_user_turn:
                        raw_u_content = last_user_turn.get("content", "")
                        lines = raw_u_content.split('\n')
                        body_lines = [l for l in lines if not re.match(r'^<.+> \[[^\]]+\]:', l) and not l.startswith("</")]
                        clean_body = "\n".join(body_lines).strip()
                        for prefix in ["!image", "!imagine"]:
                            if clean_body.lower().startswith(prefix):
                                clean_body = clean_body[len(prefix):].strip()
                                break
                        prompt_text = clean_body

                    present_template = self.cog.global_prompts.get("IMAGE_PRESENT", DEFAULT_IMAGE_PRESENT)
                    system_note = present_template.format(prompt=prompt_text)
                    recovered_media_parts.append(system_note)
                    for a in bot_image_attachments:
                        recovered_media_parts.append({"url": a.url, "mime_type": a.content_type})

            # 2. If not a bot image generation, recover standard user attachments & replied-to media
            if not is_generated_image_turn and last_user_turn:
                user_msg_ids = last_user_turn.get("message_ids", [])
                if user_msg_ids:
                    target_msg_id = user_msg_ids[-1]
                    try:
                        target_msg = await channel.fetch_message(target_msg_id)
                        attachments = [a for a in target_msg.attachments if a.content_type and (a.content_type.startswith("image/") or a.content_type.startswith("audio/") or a.content_type.startswith("video/"))]
                        if attachments:
                            for attachment in attachments:
                                recovered_media_parts.append({"url": attachment.url, "mime_type": attachment.content_type})

                        if target_msg.reference and target_msg.reference.message_id:
                            ref_msg = target_msg.reference.resolved
                            if not ref_msg:
                                r_ch = self.cog.bot.get_channel(target_msg.reference.channel_id)
                                if r_ch: ref_msg = await r_ch.fetch_message(target_msg.reference.message_id)
                            if ref_msg and ref_msg.attachments:
                                ref_media = next((a for a in ref_msg.attachments if a.content_type and (a.content_type.startswith("image/") or a.content_type.startswith("audio/") or a.content_type.startswith("video/"))), None)
                                if ref_media:
                                    recovered_media_parts.append({"url": ref_media.url, "mime_type": ref_media.content_type})
                    except Exception as e:
                        print(f"Failed to recover media for regeneration: {e}")

            # Inject recovered media into the history right before generation
            if recovered_media_parts and participant_history:
                for h_turn in reversed(participant_history):
                    if h_turn.get('role') == 'user':
                        h_turn['parts'].extend(recovered_media_parts)
                        break

            # Inject pending whispers if any
            if pending_whispers_for_regen and participant_history:
                recap_template = self.cog.global_prompts.get("WHISPER_RECAP", DEFAULT_WHISPER_RECAP)
                whisper_context = recap_template.format(whispers="\n---\n".join(pending_whispers_for_regen))

                for h_turn in reversed(participant_history):
                    if h_turn.get('role') == 'user':
                        h_turn['parts'].append(whisper_context)
                        break

            ltm_recall_text = await self.cog.memory_manager._get_relevant_ltm_for_prompt((channel.id, p_owner_id, p_name), participant_history, p_owner_id, p_name, trigger_content, "User", channel.guild.id, payload.user_id)
            training_examples = await self.cog.memory_manager._get_relevant_training_examples(p_owner_id, p_name, trigger_content, channel.guild.id)
            full_system_instruction, _, _, _, _, _, _, _ = await asyncio.to_thread(
                self._construct_system_instructions,
                p_owner_id, p_name, channel.id, is_multi_profile=True, training_examples_list=training_examples, recalled_ltm=ltm_recall_text
            )

            if hasattr(model, 'system_instruction'):
                model.system_instruction = full_system_instruction

            # [NEW] Advanced Params Injection for Regeneration
            index = self.cog.profile_manager._get_user_index(p_owner_id)
            is_borrowed = p_name in index.get("borrowed", [])
            p_profile = self.cog.profile_manager._get_profile_config(p_owner_id, p_name, is_borrowed) or {}

            adv_params = {
                "frequency_penalty": p_profile.get("frequency_penalty"),
                "presence_penalty": p_profile.get("presence_penalty"),
                "repetition_penalty": p_profile.get("repetition_penalty"),
                "min_p": p_profile.get("min_p"),
                "top_a": p_profile.get("top_a")
            }
            adv_params = {k: v for k, v in adv_params.items() if v is not None}

            gen_config = {"temperature": temp, "top_p": top_p, "top_k": top_k, "_advanced_params": adv_params}

            # [FIXED] Define missing variables for Fallback logic
            primary_model = p_profile.get("primary_model", PRIMARY_MODEL_NAME)
            fallback_model_name = p_profile.get("fallback_model", FALLBACK_MODEL_NAME)

            safety_level_str = p_profile.get('safety_level', 'low')
            dynamic_safety_settings = _resolve_safety_settings(safety_level_str)

            t_params_worker = {
                "thinking_summary_visible": p_profile.get("thinking_summary_visible", "off"),
                "thinking_level": p_profile.get("thinking_level", "high"),
                "thinking_budget": p_profile.get("thinking_budget", -1)
            }

            app_name, app_avatar = self._resolve_appearance_data(p_owner_id, p_name)
            state_container = None
            status = "api_error"
            was_blocked = False
            turn_warnings = []
            api_error_reason = None
            main_api_error = None
            response = None
            fallback_used = False

            all_participant_names = []
            for p_data_temp in session.get("profiles", []):
                app_name_temp, _ = self._resolve_appearance_data(p_data_temp['owner_id'], p_data_temp['profile_name'])
                all_participant_names.append(app_name_temp)

            t_start_regen = time.monotonic()
            try:
                response, state_container = await self._generate_with_heartbeat(
                    model, participant_history, gen_config, channel, participant, payload.message_id, is_fallback=False, app_name=app_name, app_avatar=app_avatar, existing_state={"custom_emoji": custom_emoji}
                )

                if not response or not response.candidates:
                    raise ValueError("Response blocked or empty")

                raw_text_check = getattr(response, 'text', "").strip()
                temp_scrubbed = _strip_neuro_update_and_scrub(raw_text_check, all_participant_names)

                if not temp_scrubbed:
                    raise ValueError("Empty Response (AI produced no text content)")

            except asyncio.CancelledError:
                if state_container:
                    if state_container.get('sending_task'): state_container['sending_task'].cancel()
                    await self._safe_delete_placeholder(channel, state_container.get('msg_a_id'))
                    await self._safe_delete_placeholder(channel, state_container.get('msg_b_id'))
                session['is_regenerating'] = False
                return
            except Exception as e:
                is_timeout_main = isinstance(e, TimeoutError)
                main_api_error = _format_api_error(e)
                if hasattr(e, 'state_container'): state_container = e.state_container

                if not fallback_model_name or primary_model == fallback_model_name:
                    api_error_reason = main_api_error
                else:
                    try:
                        grounding_mode_native = p_profile.get("grounding_mode", "off")
                        if isinstance(grounding_mode_native, bool): grounding_mode_native = "rag" if grounding_mode_native else "off"
                        elif grounding_mode_native in ["on", "on+"]: grounding_mode_native = "rag"

                        url_mode_native = p_profile.get("url_mode", "off")
                        if "url_mode" not in p_profile:
                            url_mode_native = "rag" if p_profile.get("url_fetching_enabled", False) else "off"

                        model_tools_list = []
                        if grounding_mode_native == "native":
                            model_tools_list.append({"google_search": {}})
                        if url_mode_native == "native":
                            model_tools_list.append({"url_context": {}})

                        model_tools = model_tools_list if model_tools_list else None

                        fallback_instance = self.cog.api_service._instantiate_model(
                            fallback_model_name, channel.guild.id, payload.user_id, full_system_instruction, dynamic_safety_settings, t_params_worker, model_tools, p_profile,
                            openrouter_key_error="No OpenRouter key for fallback", use_broad_openrouter_heuristic=False
                        )

                        response, state_container = await self._generate_with_heartbeat(
                            fallback_instance, participant_history, gen_config, channel, participant, payload.message_id, is_fallback=True, app_name=app_name, app_avatar=app_avatar, existing_state=state_container
                        )

                        if not response or not response.candidates:
                            raise ValueError("Response blocked or empty")

                        fb_raw_check = getattr(response, 'text', "").strip()
                        temp_scrubbed = _strip_neuro_update_and_scrub(fb_raw_check, all_participant_names)

                        if not temp_scrubbed:
                            raise ValueError("Empty Response (AI produced no text content)")

                        fallback_used = True
                    except asyncio.CancelledError:
                        if state_container:
                            if state_container.get('sending_task'): state_container['sending_task'].cancel()
                            await self._safe_delete_placeholder(channel, state_container.get('msg_a_id'))
                            await self._safe_delete_placeholder(channel, state_container.get('msg_b_id'))
                        session['is_regenerating'] = False
                        return
                    except Exception as retry_e:
                        is_timeout_fallback = isinstance(retry_e, TimeoutError)
                        if hasattr(retry_e, 'state_container'): state_container = retry_e.state_container

                        if is_timeout_main and is_timeout_fallback:
                            api_error_reason = ERR_REASON_TIMEOUT_BOTH
                        else:
                            api_error_reason = _format_api_error(retry_e)

            if not response or not response.candidates:
                new_text = p_profile.get("error_response", ERR_GENERAL_ERROR)
                was_blocked = True

                reason = api_error_reason or "Unknown Error"
                is_safety = False
                if response and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                    reason = response.prompt_feedback.block_reason.name.replace('_', ' ').title()
                    is_safety = True

                if is_safety:
                    turn_warnings.append(ERR_SAFETY_BLOCK.format(reason=reason))
                elif "Rate Limit" in reason:
                    turn_warnings.append(ERR_RATE_LIMIT)
                else:
                    if fallback_model_name and primary_model != fallback_model_name:
                        turn_warnings.append(WARN_BOTH_MODELS_FAILED.format(reason=reason))
                    else:
                        turn_warnings.append(WARN_MAIN_MODEL_FAILED.format(reason=reason))
            else:
                raw_text = getattr(response, 'text', "")
                if hasattr(response, 'raw') and response.raw.candidates and hasattr(response.raw.candidates[0], 'grounding_metadata'):
                    raw_text = _add_inline_citations(raw_text, response.raw.candidates[0].grounding_metadata)
                raw_text = raw_text.strip()

                raw_text, parsed_neuro_state = self._extract_and_apply_neuro_state(raw_text, p_owner_id, p_name)

                new_text = _scrub_response_text(raw_text, participant_names=all_participant_names)

                if not new_text:
                    new_text = p_profile.get("error_response", ERR_GENERAL_ERROR)
                    warn_tmp = WARN_BOTH_MODELS_FAILED if fallback_used else WARN_MAIN_MODEL_FAILED
                    turn_warnings.append(warn_tmp.format(reason=ERR_REASON_EMPTY_RESPONSE))
                    was_blocked = True

            if fallback_used and p_profile.get("show_fallback_indicator", True):
                turn_warnings.append(WARN_FALLBACK_USED)
                turn_warnings.append(WARN_MAIN_MODEL_FAILED.format(reason=main_api_error))

            # [FIX] Separate display text from history text to prevent AI hallucinations
            display_text = new_text
            if turn_warnings:
                warning_str = "\n\n" + "\n".join([f"-# {i+1}. {w}" for i, w in enumerate(turn_warnings)])
                display_text += warning_str

            if state_container:
                await self._safe_delete_placeholder(channel, state_container.get('msg_b_id'))
                state_container['msg_b_id'] = None

            # 5. Apply Changes
            sent_timestamp = datetime.datetime.now(datetime.timezone.utc)
            p_index = self.cog.profile_manager._get_user_index(p_owner_id)
            p_is_borrowed = p_name in p_index.get("borrowed", [])
            p_profile = self.cog.profile_manager._get_profile_config(p_owner_id, p_name, p_is_borrowed) or {}
            tz_str = p_profile.get("timezone", "UTC")

            sp_name = p_name
            app_data = self.cog.user_appearances.get(str(p_owner_id), {}).get(p_name, {})
            if app_data.get("custom_display_name"): sp_name = app_data["custom_display_name"]

            profile_id = self.cog.profile_manager._get_profile_id(p_owner_id, p_name)
            
            # [FIX] Save new_text (WITHOUT warnings) to history
            new_history_line = _format_history_entry(sp_name, sent_timestamp, new_text, tz_str, entity_id=profile_id)

            final_target_turn = next((t for t in session.get("unified_log", []) if t.get("turn_id") == turn_id), None)
            if not final_target_turn:
                final_target_turn = target_turn
                session.setdefault("unified_log", []).append(final_target_turn)

            final_target_turn["content"] = new_history_line
            final_target_turn["timestamp"] = sent_timestamp.isoformat()
            final_target_turn["message_ids"] = [payload.message_id]

            regen_grounding_sources = []
            if response and hasattr(response, 'raw') and response.raw.candidates:
                if hasattr(response.raw.candidates[0], 'grounding_metadata'):
                    metadata = response.raw.candidates[0].grounding_metadata
                    if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks is not None:
                        for chunk in metadata.grounding_chunks:
                            if hasattr(chunk, 'web'):
                                regen_grounding_sources.append(chunk.web.uri)

            # [NEW] Meta collection for regeneration
            meta = {
                "duration": round(time.monotonic() - t_start_regen, 2) if 't_start_regen' in locals() else 0.0,
                "model": model.model_name.replace("models/", "").replace("OPENROUTER/", "").replace("GOOGLE/", "") if hasattr(model, 'model_name') else fallback_model_name,
                "fallback": fallback_used,
                "input_tokens": getattr(response, 'input_tokens', 0) if response else 0,
                "output_tokens": getattr(response, 'output_tokens', 0) if response else 0,
                "reasoning_tokens": getattr(response, 'reasoning_tokens', 0) if response else 0,
                "training_recalled": len(training_examples) if 'training_examples' in locals() and training_examples else 0,
                "grounding_sources": regen_grounding_sources,
                "ltms_recalled": []
            }
            if 'ltm_recall_text' in locals() and ltm_recall_text:
                lines = ltm_recall_text.split('\n')
                clean_lines = [l.strip() for l in lines if l.strip() and not l.startswith("<")]
                meta["ltms_recalled"] = [l[:100] + "..." if len(l) > 100 else l for l in clean_lines]

            if 'parsed_neuro_state' in locals() and parsed_neuro_state:
                meta["neuro_state"] = parsed_neuro_state

            final_target_turn["meta"] = meta

            # Clean up legacy signatures from the turn if they exist
            final_target_turn.pop('thought_signature', None)

            if state_container and state_container.get('sending_task'):
                state_container['sending_task'].cancel()

            # Truncate text strictly for Discord's 2000 character limit on edits
            safe_text = display_text
            if len(safe_text) > 2000:
                safe_text = safe_text[:1997] + "..."

            if participant.get('method') == 'child_bot':
                await self.cog.manager_queue.put({
                    "action": "send_to_child", "bot_id": participant['bot_id'],
                    "payload": {
                        "action": "regenerate_message", "channel_id": channel.id,
                        "message_id": payload.message_id, "content": safe_text
                    }
                })
            else:
                wh = await self.cog.server_manager._get_or_create_webhook(channel)
                if wh:
                    try:
                        msg = await channel.fetch_message(payload.message_id)
                        kept_atts = [a for a in msg.attachments if a.content_type and a.content_type.startswith("image/")]
                        await wh.edit_message(payload.message_id, content=safe_text, attachments=kept_atts)
                    except: pass

            # Re-sync memory and save final state
            await self.cog.session_manager._save_session_to_disk(dummy_key, session_type, session["unified_log"])

            # No rebuild: a regenerated turn only changes its own content, which no
            # derived session state reads. See _recompute_pending_whispers.

        except Exception as e:
            print(f"Regeneration failed: {e}")
        finally:
            session['is_regenerating'] = False
