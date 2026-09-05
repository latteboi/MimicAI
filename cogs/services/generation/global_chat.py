import time
import uuid
import asyncio
import discord
import traceback
import datetime
from typing import Dict, List, Optional

from ...utils.constants import (
    defaultConfig, PRIMARY_MODEL_NAME, STM_LIMIT_MAX, PLACEHOLDER_EMOJI,
    ERR_GENERAL_ERROR, ERR_REASON_TIMEOUT_BOTH, ERR_SAFETY_BLOCK,
    WARN_BOTH_MODELS_FAILED, WARN_FALLBACK_USED, WARN_MAIN_MODEL_FAILED,
)
from ...utils.helpers import (
    _add_inline_citations, _format_api_error, _format_citation_subtext, _format_history_entry,
    _get_user_hash, _resolve_safety_settings, _scrub_response_text, default_profile_avatar_url,
    resolve_thinking_params,
)
from ._shared import _strip_neuro_update_and_scrub

#: Speakers named on the card before the rest collapse into "+n". Four fits the footer
#: and the field title at any sensible display-name length.
MAX_NAMED_SPEAKERS = 4


def _name_list(names: List[str], limit: int = MAX_NAMED_SPEAKERS) -> str:
    """`Alice, Bob and Carol`, or `Alice, Bob, Carol, Dave +3` past the limit."""
    names = [n for n in names if n]
    if not names:
        return "someone"
    if len(names) <= limit:
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return ", ".join(names[:limit]) + f" +{len(names) - limit}"


def _incoming_from_log(log: List[Dict]) -> List[Dict]:
    """The user turns that produced the newest reply.

    Walked backwards from the last model turn and stopped at the first non-user turn --
    a multi-user round appends one user turn per speaker before the single model turn,
    so the run is the whole round. Backwards rather than `log[::-1]`, which copies the
    entire log on every render.
    """
    model_index = None
    for i in range(len(log) - 1, -1, -1):
        if log[i].get("role") == "model":
            model_index = i
            break
    if model_index is None:
        return []

    incoming: List[Dict] = []
    i = model_index - 1
    while i >= 0 and log[i].get("role") == "user":
        incoming.append(log[i])
        i -= 1
    incoming.reverse()
    return incoming


def build_global_chat_embed(cog, host_user_id: int, profile_name: str,
                            session_data: Optional[Dict], *,
                            description: Optional[str] = None,
                            incoming: Optional[List[Dict]] = None,
                            footer: Optional[str] = None,
                            colour: Optional[discord.Colour] = None) -> discord.Embed:
    """The Global Chat card, drawn one way wherever it is drawn.

    The view and the generator had each grown their own version and they disagreed: one
    drew grey with a `You: ...` footer under the *host's* avatar, the other blue with
    every speaker's line run together on a single line under the avatar of whoever
    pressed Play. With more than one person in the session both were simply false --
    "You" named whoever happened to be reading, and neither avatar belonged to a
    speaker. So speakers are named in a field, and the footer carries the state the
    other people in a channel could not see at all: who may reply, and who is waiting.

    `description` and `incoming` override what the log says, for the placeholder and for
    the round being generated -- its turns are not in the log yet. `footer` replaces the
    status line, which means nothing on a history browser looking at an old round.
    """
    session_data = session_data or {}
    log = session_data.get("unified_log") or []

    eff_owner, eff_name = cog.profile_manager._resolve_effective_profile(host_user_id, profile_name)
    appearance = cog.profile_manager._get_user_appearance(eff_owner, eff_name) or {}
    display_name = appearance.get("custom_display_name") or profile_name
    avatar_url = appearance.get("custom_avatar_url") or default_profile_avatar_url(eff_name)

    last_model = next((t for t in reversed(log) if t.get("role") == "model"), None)
    body = description if description is not None else (last_model or {}).get("content")
    has_reply = bool(body)
    if not has_reply:
        body = "Nothing said here yet. Press **Reply** to start."

    embed = discord.Embed(
        description=body[:4096],
        colour=colour or (discord.Colour.blue() if has_reply else discord.Colour.dark_grey()),
    )
    embed.set_author(name=display_name[:256], icon_url=avatar_url)

    if incoming is None:
        incoming = _incoming_from_log(log)

    if incoming:
        # Budgeted per line, not truncated as a whole: one long message from the first
        # speaker would otherwise push everyone else's out of the field entirely.
        per_line = max(60, 1000 // len(incoming))
        lines = []
        for turn in incoming[:MAX_NAMED_SPEAKERS * 2]:
            speaker = (turn.get("display_name") or "Someone")[:32]
            text = " ".join((turn.get("content") or "").split()) or "—"
            if len(text) > per_line:
                text = text[:per_line - 1] + "…"
            lines.append(f"**{speaker}** {text}")
        title = ("In reply to" if len(incoming) == 1
                 else f"In reply to {_name_list([t.get('display_name') for t in incoming])}")
        embed.add_field(name=title[:256], value="\n".join(lines)[:1024], inline=False)

    if footer is not None:
        embed.set_footer(text=footer[:2048])
        return embed

    queue = session_data.get("pending_queue") or {}
    locked = session_data.get("is_locked", True)
    status = ["🔒 Only the host can reply" if locked else "🔓 Anyone here can reply"]
    if queue:
        status.append("waiting on " + _name_list(
            [q.get("display_name") for q in queue.values()]))
    embed.set_footer(text=" · ".join(status)[:2048])
    return embed


class GlobalChatMixin:
    """The `/profile global_chat` session (its own model-caching and history path,
    separate from the multi-profile worker).

    Not a DM and not a channel of messages: the conversation is one embed, posted
    wherever the command was run and edited in place by GlobalChatPlayView. History
    is keyed on (host, profile) rather than on a channel, which is what makes it
    "global" -- the same conversation continues in any server, or in a DM.
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

        # The embed goes wherever the command was run, and nothing constrains that
        # to an age-restricted channel, so an adult-rated profile has nowhere
        # compliant to run here. Read through the resolver rather than off the
        # config, so a borrowed profile is judged by its source's rating.
        allowed, deny_reason = self.cog.profile_manager.content_capability(
            host_user_id, profile_name, "global_chat")
        if not allowed:
            await interaction.followup.send(
                f"**'{profile_name}' cannot be used in Global Chat.**\n{deny_reason}", ephemeral=True)
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
                session_data = {'unified_log': []}

            self.cog.global_chat_sessions[model_cache_key] = session_data

            self.cog.session_last_accessed[model_cache_key] = time.time()

            # Derived from unified_log on every turn. This used to be written onto a
            # GoogleGenAIChatSession held on session_data, which was then overwritten from
            # the same log a few lines below — the object never carried state between turns.
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

            if len(rebuilt_history) > STM_LIMIT_MAX * 2:
                rebuilt_history = rebuilt_history[-(STM_LIMIT_MAX * 2):]
            session_data['unified_log'] = session_data['unified_log'][-(STM_LIMIT_MAX * 2):]

            combined_prompt_text = "\n\n".join([f"{t['display_name']}: {t['content']}" for t in queued_turns])

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
                    g_hist = rebuilt_history[-(g_stm_capped * 2):]

                # Always a DM: not age-restricted, so BLOCK_ONLY_HIGH. (`safety_level`
                # was never defined in this scope -- this line raised NameError for
                # any global-chat profile with RAG grounding enabled.)
                d_safe = _resolve_safety_settings(None, profile_data)

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
                history_slice = rebuilt_history[-stm_length:]
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
            # `incoming` is passed explicitly: this round's turns are only appended to
            # unified_log once the reply lands, so the card would otherwise show the
            # previous round's speakers while answering this one.
            placeholder_embed = build_global_chat_embed(
                self.cog, host_user_id, profile_name, session_data,
                description=custom_emoji, incoming=queued_turns,
                colour=discord.Colour.dark_grey())

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
                temp_scrubbed = _strip_neuro_update_and_scrub(raw_text_check, [app_name] + [t['display_name'] for t in queued_turns])

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
                        # channel_id 0 resolves to no channel, so the builder takes the
                        # not-age-restricted branch and always injects <content_policy>.
                        # That is the intended answer, not an accident of the sentinel: a
                        # Global Chat card can be opened in any channel and none is
                        # guaranteed age-restricted, which is the same reason
                        # content_capability refuses an Adult profile here.
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

                        d_safe = _resolve_safety_settings(None, p_data_f)

                        # Only ever handed to the fallback instance below, so it
                        # resolves as the fallback role.
                        t_params_f = resolve_thinking_params(p_data_f, "response", "fallback")

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

                        # One factory call in place of three hand-rolled provider branches.
                        # Those branches also left the generation call below nested inside the
                        # Google branch, so an OpenRouter or Ollama fallback model was built and
                        # then never used: the fallback silently did nothing and the turn
                        # reported that both models had failed. Collapsing the branches puts
                        # the call back on the single path every provider reaches.
                        fb_name = fallback_model_name
                        fallback_instance = self.cog.api_service._instantiate_model(
                            fb_name, None, host_user_id,
                            sys_instr, d_safe, t_params_f, model_tools, p_data_f,
                            openrouter_key_error="No OR key for fallback",
                            google_key_error="No Google key for fallback",
                        )

                        response, state_container = await self._generate_with_heartbeat(
                            fallback_instance, contents_for_api_call, gen_config, interaction.channel, None, placeholder_msg.id, is_fallback=True, app_name=app_name, app_avatar=app_avatar, existing_state=state_container, message_type="embed"
                        )
                        status = "blocked_by_safety" if not response or not response.candidates else "success"
                        if status == "success":
                            fb_raw_check = getattr(response, 'text', "").strip()
                            temp_scrubbed = _strip_neuro_update_and_scrub(fb_raw_check, [app_name] + [t['display_name'] for t in queued_turns])

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

            # Apply filters. Every speaker's name is XML-tag-wrapped in the history the
            # model sees (_format_history_entry), so a hallucinated continuation as one of
            # the queued users leaves a bare closing tag that only the name scrubber can
            # catch -- scrubbing just the bot's own name left every other name's closing
            # tag in the clear.
            chat_participant_names = [app_name] + [t['display_name'] for t in queued_turns]
            response_text = _scrub_response_text(raw_text, participant_names=chat_participant_names)

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

            # Same builder the view uses, so the refresh that follows in play_callback
            # redraws an identical card instead of flipping colour and footer.
            embed = build_global_chat_embed(
                self.cog, host_user_id, profile_name, session_data,
                description=text_for_embed, incoming=queued_turns)

            if state_container and state_container.get('sending_task'):
                state_container['sending_task'].cancel()

            await placeholder_msg.edit(embed=embed)
            await self._dispatch_warnings(interaction.channel, 'webhook', None, turn_warnings, host_user_id, profile_name)

            t2_end_mono = time.monotonic()
            duration = t2_end_mono - t1_start_mono

            await self.cog.session_manager._save_session_to_disk(model_cache_key, 'global_chat', session_data)

        except Exception as e:
            await interaction.followup.send(f"An error occurred during the global chat: {e}", ephemeral=True)
            traceback.print_exc()