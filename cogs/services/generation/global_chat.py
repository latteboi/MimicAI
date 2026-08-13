import time
import uuid
import base64
import asyncio
import discord
import traceback
import datetime
from typing import Dict, List

from ...utils.constants import (
    defaultConfig, PRIMARY_MODEL_NAME, OLLAMA_LOCAL_URL, STM_LIMIT_MAX, PLACEHOLDER_EMOJI,
    ERR_GENERAL_ERROR, ERR_REASON_TIMEOUT_BOTH, ERR_SAFETY_BLOCK,
    WARN_BOTH_MODELS_FAILED, WARN_FALLBACK_USED, WARN_MAIN_MODEL_FAILED,
)
from ...utils.helpers import (
    _add_inline_citations, _format_api_error, _format_citation_subtext, _format_history_entry,
    _get_user_hash, _scrub_response_text,
)
from ..api_service import GoogleGenAIModel, GoogleGenAIChatSession, OllamaModel, OpenRouterModel
from ._shared import _resolve_safety_settings, _strip_neuro_update_and_scrub


class GlobalChatMixin:
    """The `/profile global_chat` DM/cross-server chat session (its own model-caching 
    and history path, separate from the multi-profile worker).
    """

    async def _execute_global_chat(self, interaction: discord.Interaction, host_user_id: int, profile_name: str, queued_turns: List[Dict]):
        t1_start_mono = time.monotonic()
        t1_start_utc = datetime.datetime.now(datetime.timezone.utc)

        index = self.cog.profile_manager._get_user_index(host_user_id)
        is_borrowed = profile_name in index.get("borrowed", [])

        source_owner_id = host_user_id
        source_profile_name = profile_name
        if is_borrowed:
            borrowed_data = self.cog.profile_manager._get_profile_config(host_user_id, profile_name, True) or {}
            source_owner_id = int(borrowed_data.get("original_owner_id", host_user_id))
            source_profile_name = borrowed_data.get("original_profile_name", profile_name)

        profile_data = self.cog.profile_manager._get_profile_config(source_owner_id, source_profile_name, False)

        if not profile_data:
            await interaction.followup.send(f"The source for '{profile_name}' could not be found.", ephemeral=True)
            return

        safety_level = profile_data.get("safety_level", "low")
        if safety_level == "unrestricted":
            await interaction.followup.send("For safety reasons, profiles with an 'Unrestricted 18+' safety level cannot be used with `/profile global_chat`. Please set the safety level to 'Low', 'Medium', or 'High'.", ephemeral=True)
            return

        user_api_key = self.cog.storage_manager._get_api_key_for_user(host_user_id, "gemini")
        or_key = self.cog.storage_manager._get_api_key_for_user(host_user_id, "openrouter")
        has_ollama = profile_data.get("primary_model", "").upper().startswith("OLLAMA/")

        if not user_api_key and not or_key and not has_ollama:
            await interaction.followup.send("The host of this session needs to submit a personal API key using `/settings` to use this feature.", ephemeral=True)
            return

        model_cache_key = ('global', host_user_id, profile_name)

        try:
            model, temp, top_p, top_k, warning_message, fallback_model_name = await self.cog.api_service._get_or_create_model_for_global_chat(host_user_id, profile_name)

            # Resolve primary model and safety settings
            profile_data = self.cog.profile_manager._get_profile_config(source_owner_id, source_profile_name, False) or {}
            primary_model = profile_data.get("primary_model", PRIMARY_MODEL_NAME)

            if warning_message:
                try: await interaction.user.send(warning_message)
                except discord.Forbidden: pass

            if not model:
                error_msg = warning_message or "Could not initialize the AI model for this profile."
                await interaction.followup.send(error_msg, ephemeral=True)
                return

            session_data = self.cog.global_chat_sessions.get(model_cache_key)
            if not session_data:
                session_data = await self.cog.session_manager._load_session_from_disk(model_cache_key, 'global_chat')

            if not session_data:
                chat = GoogleGenAIChatSession(history=[])
                session_data = {'chat_session': chat, 'unified_log': []}

            self.cog.global_chat_sessions[model_cache_key] = session_data

            # FIX: Reconstruct Chat Session if missing from ram cache
            chat = session_data.get('chat_session')
            if not chat:
                rebuilt_history = []
                for t in session_data.get('unified_log', []):
                    role = 'model' if t.get('is_user') is False else 'user'
                    rebuilt_history.append({'role': role, 'parts': [t.get('content')]})
                chat = GoogleGenAIChatSession(history=rebuilt_history)
                session_data['chat_session'] = chat

            self.cog.session_last_accessed[model_cache_key] = time.time()

            rebuilt_history = []
            for t in session_data.get('unified_log', []):
                t_role = t.get('role')
                parts = [t.get('content')]

                if t_role == 'user':
                    if t.get('url_context') and profile_data.get('url_fetching_enabled', False):
                        parts.append(f"\n<document_context>\n{t.get('url_context')}\n</document_context>")
                    if t.get('grounding_context') and profile_data.get('grounding_mode', 'off') != 'off':
                        parts.append(f"\n{t.get('grounding_context')}")

                content_obj = {'role': t_role, 'parts': parts}
                rebuilt_history.append(content_obj)

            chat.history = rebuilt_history

            if len(chat.history) > STM_LIMIT_MAX * 2:
                chat.history = chat.history[-(STM_LIMIT_MAX * 2):]
            session_data['unified_log'] = session_data['unified_log'][-(STM_LIMIT_MAX * 2):]

            combined_prompt_text = "\n\n".join([f"{t['display_name']}: {t['content']}" for t in queued_turns])
            combined_footer_text = " ".join([f"{t['display_name']}: {t['content']}" for t in queued_turns])

            source_owner_id, source_profile_name = self.cog.profile_manager._resolve_effective_profile(host_user_id, profile_name)

            bot_display_name = source_profile_name
            appearance = self.cog.user_appearances.get(str(source_owner_id), {}).get(source_profile_name)
            if appearance and appearance.get("custom_display_name"):
                bot_display_name = appearance.get("custom_display_name")

            contents_for_api_call =[]

            user_tz = profile_data.get("timezone", "UTC")
            final_user_parts = []
            turn_warnings =[]

            url_mode = profile_data.get('url_mode', 'off')
            if 'url_mode' not in profile_data:
                url_mode = 'rag' if profile_data.get('url_fetching_enabled', False) else 'off'

            if url_mode == 'rag':
                u_text, _, u_warn = await self.cog.tools_service._process_urls_in_content(combined_prompt_text, 0, {"url_fetching_enabled": True})
                turn_warnings.extend(u_warn)
                if u_text:
                    final_user_parts.append(f"<document_context>\n" + "\n".join(u_text) + "\n</document_context>")

            # [NEW] RAG Grounding for Global Chat
            grounding_mode = profile_data.get('grounding_mode', 'off')
            if isinstance(grounding_mode, bool): grounding_mode = "rag" if grounding_mode else "off"
            elif grounding_mode in ["on", "on+"]: grounding_mode = "rag"

            global_rag_sources = []
            if grounding_mode == "rag":
                g_hist = []
                stm_length = int(profile_data.get("stm_length", defaultConfig.CHATBOT_MEMORY_LENGTH))
                g_stm_capped = min(10, stm_length)
                if g_stm_capped > 0:
                    g_hist = chat.history[-(g_stm_capped * 2):]

                d_safe = _resolve_safety_settings(safety_level)

                g_res = await self.cog.tools_service._get_hybrid_grounding_context(combined_prompt_text, 0, g_hist, ('global_chat', host_user_id), safety_settings=d_safe)
                if g_res:
                    g_ctx, g_srcs, _, g_warn = g_res
                    if g_warn: turn_warnings.append(g_warn)
                    if g_ctx:
                        final_user_parts.append(g_ctx)
                        global_rag_sources.extend(g_srcs)

            for turn in queued_turns:
                user_hash = _get_user_hash(turn["user_id"])
                user_line = _format_history_entry(turn["display_name"], turn["timestamp"], turn["content"], user_tz, entity_id=user_hash)
                final_user_parts.append(user_line)

            user_content_obj_for_turn = {'role': 'user', 'parts': final_user_parts}

            stm_length = int(profile_data.get("stm_length", defaultConfig.CHATBOT_MEMORY_LENGTH))
            if stm_length > 0:
                history_slice = chat.history[-stm_length:]
                contents_for_api_call.extend(history_slice)

            if contents_for_api_call and contents_for_api_call[-1].get('role') == 'user':
                contents_for_api_call[-1]['parts'].extend(user_content_obj_for_turn['parts'])
            else:
                contents_for_api_call.append(user_content_obj_for_turn)

            gen_config = {
                "temperature": temp, "top_p": top_p, "top_k": top_k,
                "thinking_config": {"include_thoughts": True}
            }

            status = "api_error"
            response = None
            fallback_used = False
            api_error_reason = None
            main_api_error = None
            state_container = None

            custom_emoji = profile_data.get("placeholder_emoji") or PLACEHOLDER_EMOJI

            # --- EDIT ORIGINAL RESPONSE ---
            placeholder_embed = discord.Embed(description=f"{custom_emoji}", color=discord.Color.dark_grey())
            placeholder_embed.set_author(name=bot_display_name, icon_url=appearance.get("custom_avatar_url") if appearance else self.cog.bot.user.display_avatar.url)
            placeholder_embed.set_footer(text=combined_footer_text[:1000], icon_url=interaction.user.display_avatar.url)

            await interaction.edit_original_response(embed=placeholder_embed, view=None)
            placeholder_msg = await interaction.original_response()

            app_name, app_avatar = self._resolve_appearance_data(host_user_id, profile_name)

            try:
                response, state_container = await self._generate_with_heartbeat(
                    model, contents_for_api_call, gen_config, interaction.channel, None, placeholder_msg.id, is_fallback=False, app_name=app_name, app_avatar=app_avatar, message_type="embed", existing_state={"custom_emoji": custom_emoji, "placeholder_msg": placeholder_msg}
                )
                if not response or not response.candidates:
                    raise ValueError("Response blocked or empty")

                raw_text_check = getattr(response, 'text', "").strip()
                temp_scrubbed = _strip_neuro_update_and_scrub(raw_text_check, [app_name])

                if not temp_scrubbed:
                    raise ValueError("Empty Response (AI produced no text content)")

                status = "success"
            except asyncio.CancelledError:
                if state_container and state_container.get('sending_task'):
                    state_container['sending_task'].cancel()
                msg_b_to_delete = state_container.get('msg_b_id') if state_container else None
                await self._safe_delete_placeholder(interaction.channel, msg_b_to_delete)
                return
            except Exception as e:
                is_timeout_main = isinstance(e, TimeoutError)
                main_api_error = _format_api_error(e)
                if hasattr(e, 'state_container'): state_container = e.state_container

                if not fallback_model_name or primary_model == fallback_model_name:
                    api_error_reason = main_api_error
                else:
                    try:
                        sys_instr, _, _, _, _, _, _, _ = await asyncio.to_thread(self._construct_system_instructions, host_user_id, profile_name, 0)

                        user_index_f = self.cog.profile_manager._get_user_index(host_user_id)
                        is_borrowed_f = profile_name in user_index_f.get("borrowed", [])
                        source_id_f = host_user_id
                        source_name_f = profile_name
                        if is_borrowed_f:
                            bd = self.cog.profile_manager._get_profile_config(host_user_id, profile_name, True) or {}
                            source_id_f = int(bd.get("original_owner_id", host_user_id))
                            source_name_f = bd.get("original_profile_name", profile_name)

                        p_data_f = self.cog.profile_manager._get_profile_config(source_id_f, source_name_f, False) or {}
                        safe_lvl = p_data_f.get("safety_level", "low")

                        d_safe = _resolve_safety_settings(safe_lvl)

                        fb_name = fallback_model_name
                        fb_is_or = False
                        fb_is_ollama = False

                        if fb_name.upper().startswith("OPENROUTER/"):
                            fb_name = fb_name[11:]
                            fb_is_or = True
                        elif fb_name.upper().startswith("GOOGLE/"):
                            fb_name = fb_name[7:]
                        elif fb_name.upper().startswith("OLLAMA/"):
                            fb_name = fb_name[7:]
                            fb_is_ollama = True
                        elif "/" in fb_name:
                            fb_is_or = True

                        if fb_is_or:
                            or_key = self.cog.storage_manager._get_api_key_for_user(host_user_id, provider="openrouter")
                            if or_key:
                                fallback_instance = OpenRouterModel(fb_name, api_key=or_key, system_instruction=sys_instr, thinking_params={})
                            else:
                                raise ValueError("No OR key for fallback")
                        elif fb_is_ollama:
                            ollama_host = p_data_f.get("ollama_host_url", OLLAMA_LOCAL_URL)
                            fallback_instance = OllamaModel(fb_name, api_url=ollama_host, system_instruction=sys_instr, thinking_params={})
                        else:
                            user_key = self.cog.storage_manager._get_api_key_for_user(host_user_id)
                            if user_key:
                                t_params_f = {
                                    "thinking_summary_visible": p_data_f.get("thinking_summary_visible", "off"),
                                    "thinking_level": p_data_f.get("thinking_level", "high"),
                                    "thinking_budget": p_data_f.get("thinking_budget", -1)
                                }

                                grounding_mode_native = p_data_f.get("grounding_mode", "off")
                                if isinstance(grounding_mode_native, bool): grounding_mode_native = "rag" if grounding_mode_native else "off"
                                elif grounding_mode_native in ["on", "on+"]: grounding_mode_native = "rag"

                                url_mode_native = p_data_f.get("url_mode", "off")
                                if "url_mode" not in p_data_f:
                                    url_mode_native = "rag" if p_data_f.get("url_fetching_enabled", False) else "off"

                                model_tools_list = []
                                if grounding_mode_native == "native":
                                    model_tools_list.append({"google_search": {}})
                                if url_mode_native == "native":
                                    model_tools_list.append({"url_context": {}})

                                model_tools = model_tools_list if model_tools_list else None

                                fallback_instance = GoogleGenAIModel(
                                    api_key=user_key,
                                    model_name=fb_name,
                                    system_instruction=sys_instr,
                                    safety_settings=d_safe,
                                    thinking_params=t_params_f,
                                    tools=model_tools
                                )
                            else:
                                raise ValueError("No Google key for fallback")

                            response, state_container = await self._generate_with_heartbeat(
                                fallback_instance, contents_for_api_call, gen_config, interaction.channel, None, placeholder_msg.id, is_fallback=True, app_name=app_name, app_avatar=app_avatar, existing_state=state_container, message_type="embed"
                            )
                            status = "blocked_by_safety" if not response or not response.candidates else "success"
                            if status == "success":
                                fb_raw_check = getattr(response, 'text', "").strip()
                                temp_scrubbed = _strip_neuro_update_and_scrub(fb_raw_check, [app_name])

                                if not temp_scrubbed:
                                    raise ValueError("Empty Response (AI produced no text content)")

                                fallback_used = True
                                self.cog._log_api_call(user_id=host_user_id, guild_id=None, context="global_chat_fallback", model_used=fb_name, status="success")
                    except asyncio.CancelledError:
                        return
                    except Exception as retry_e:
                        print(f"Global Chat fallback retry failed: {retry_e}")
                        is_timeout_fallback = isinstance(retry_e, TimeoutError)
                        if hasattr(retry_e, 'state_container'): state_container = retry_e.state_container

                        if is_timeout_main and is_timeout_fallback:
                            api_error_reason = ERR_REASON_TIMEOUT_BOTH
                        else:
                            api_error_reason = _format_api_error(retry_e)
                        status = "api_error"
            finally:
                self.cog._log_api_call(user_id=host_user_id, guild_id=None, context="global_chat", model_used=model, status=status)

            if not response or not response.candidates:
                reason = api_error_reason or "Unknown Error"
                is_safety = False
                if response and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                    reason = response.prompt_feedback.block_reason.name.replace('_', ' ').title()
                    is_safety = True

                custom_main = profile_data.get("error_response", ERR_GENERAL_ERROR)

                if is_safety:
                    turn_warnings.append(ERR_SAFETY_BLOCK.format(reason=reason))
                elif "Rate Limit" in reason:
                    turn_warnings.append(reason)
                else:
                    if fallback_model_name and primary_model != fallback_model_name:
                        turn_warnings.append(WARN_BOTH_MODELS_FAILED.format(reason=reason))
                    else:
                        turn_warnings.append(WARN_MAIN_MODEL_FAILED.format(reason=reason))

                err_embed = placeholder_msg.embeds[0]
                err_embed.description = custom_main
                await placeholder_msg.edit(embed=err_embed)

                if state_container and state_container.get('sending_task'):
                    state_container['sending_task'].cancel()
                msg_b_to_delete = state_container.get('msg_b_id') if state_container else None
                await self._safe_delete_placeholder(interaction.channel, msg_b_to_delete)
                if state_container:
                    state_container['msg_b_id'] = None

                await self._dispatch_warnings(interaction.channel, 'webhook', None, turn_warnings, host_user_id, profile_name)
                return

            raw_text = getattr(response, 'text', "")
            if hasattr(response, 'raw') and response.raw.candidates and hasattr(response.raw.candidates[0], 'grounding_metadata'):
                raw_text = _add_inline_citations(raw_text, response.raw.candidates[0].grounding_metadata)
            raw_text = raw_text.strip()

            raw_text, _ = self._extract_and_apply_neuro_state(raw_text, host_user_id, profile_name)

            # Apply filters
            response_text = _scrub_response_text(raw_text, participant_names=[app_name])

            if response_text:
                grounding_sources = []
                grounding_sources.extend(global_rag_sources)
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

            model_content_obj_for_turn = {'role': 'model', 'parts': [response_text]}
            chat.history.extend([user_content_obj_for_turn, model_content_obj_for_turn])

            if len(chat.history) > STM_LIMIT_MAX * 2:
                chat.history = chat.history[-(STM_LIMIT_MAX * 2):]

            current_log = session_data.get('unified_log', [])
            if len(current_log) > STM_LIMIT_MAX * 2:
                session_data['unified_log'] = current_log[-(STM_LIMIT_MAX * 2):]

            # --- Turn Logging ---
            for turn in queued_turns:
                user_turn_id = str(uuid.uuid4())
                session_data.setdefault('unified_log', []).append({
                    "turn_id": user_turn_id, "role": "user", "content": turn["content"], "timestamp": turn["timestamp"], "user_id": turn["user_id"], "display_name": turn["display_name"]
                })

            model_turn_id = str(uuid.uuid4())
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            model_log = {
                "turn_id": model_turn_id, "role": "model", "content": response_text, "timestamp": timestamp
            }
            
            session_data.setdefault('unified_log', []).append(model_log)

            text_for_embed = response_text

            if fallback_used and profile_data.get("show_fallback_indicator", True):
                turn_warnings.append(WARN_FALLBACK_USED)
                turn_warnings.append(WARN_MAIN_MODEL_FAILED.format(reason=main_api_error))

            if state_container:
                await self._safe_delete_placeholder(interaction.channel, state_container.get('msg_b_id'))
                state_container['msg_b_id'] = None

            await self._update_sending_placeholder(interaction.channel, 'webhook', None, state_container, t1_start_mono)

            embed = discord.Embed(description=text_for_embed, color=discord.Color.blue())
            embed.set_author(name=app_name, icon_url=app_avatar or self.cog.bot.user.display_avatar.url)
            embed.set_footer(text=combined_footer_text[:1000], icon_url=interaction.user.display_avatar.url)

            if state_container and state_container.get('sending_task'):
                state_container['sending_task'].cancel()

            response_message = await placeholder_msg.edit(embed=embed)
            await self._dispatch_warnings(interaction.channel, 'webhook', None, turn_warnings, host_user_id, profile_name)

            t2_end_mono = time.monotonic()
            duration = t2_end_mono - t1_start_mono

            timezone_str = profile_data.get("timezone", "UTC")
            profile_id = self.cog.profile_manager._get_profile_id(source_owner_id, source_profile_name)
            main_history_line = _format_history_entry(app_name, response_message.created_at, response_text, timezone_str, entity_id=profile_id)

            bot_response_formatted = main_history_line

            if chat.history and chat.history[-1].get('role', 'user') == 'model':
                old_turn = chat.history[-1]
                new_turn = {'role': 'model', 'parts': [bot_response_formatted]}
                chat.history[-1] = new_turn

            await self.cog.session_manager._save_session_to_disk(model_cache_key, 'global_chat', session_data)

        except Exception as e:
            await interaction.followup.send(f"An error occurred during the global chat: {e}", ephemeral=True)
            traceback.print_exc()