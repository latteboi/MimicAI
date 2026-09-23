import time
import uuid
import asyncio
import discord
import traceback
import datetime
from typing import Dict, List, Optional

from ...utils.constants import (
    defaultConfig, STM_LIMIT_MAX, PLACEHOLDER_EMOJI,
    GEMINI_FREE_TIER_BLOCKED, STATUS_SEARCHING_WEB,
)
from ...utils.helpers import (
    _format_citation_subtext, _format_history_entry, _get_user_hash, _resolve_safety_settings,
    default_profile_avatar_url, resolve_grounding_mode, suppress_link_previews,
)
from . import tool_loop
from .reply import _merge_sources, reply_gen_config

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

        source_owner_id, source_profile_name = \
            self.cog.profile_manager._resolve_effective_profile(host_user_id, profile_name)

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

        # The host's own copy from here on, as a session reads a seated borrow's: a borrow
        # owns its config, and its models and sampling are what the host set on it.
        is_borrowed = profile_name in self.cog.profile_manager._get_user_index(host_user_id).get("borrowed", [])
        profile_data = self.cog.profile_manager._get_profile_config(host_user_id, profile_name, is_borrowed) or {}

        user_api_key = self.cog.storage_manager._get_api_key_for_user(host_user_id, "gemini")
        or_key = self.cog.storage_manager._get_api_key_for_user(host_user_id, "openrouter")
        has_ollama = (str(profile_data.get("primary_model") or "").upper().startswith("OLLAMA/")
                      and self.cog.profile_manager.may_use_ollama(host_user_id))

        # The host's own key, carrying a conversation others can be let into wherever the
        # card was opened: a free Gemini tier answers to that server's policy, and is
        # refused with no server at all. See cogs/utils/data_policy.
        gemini_blocked = bool(user_api_key) and not self.cog.storage_manager.personal_gemini_allowed_in_conversation(
            host_user_id, interaction.guild_id)
        if gemini_blocked:
            user_api_key = None

        if not user_api_key and not or_key and not has_ollama:
            await interaction.followup.send(
                GEMINI_FREE_TIER_BLOCKED if gemini_blocked else
                "The host of this session needs to submit a personal API key using `/settings` to use this feature.",
                ephemeral=True)
            return

        model_cache_key = ('global', host_user_id, profile_name)

        try:
            # The people writing in this round: whose birthdays the character may know.
            present_users = [(t["user_id"], t["display_name"]) for t in queued_turns]
            # No server, so no `recall`: memories are filed per server. One tuple for the
            # prompt and every model -- see tool_loop.
            functions = tool_loop.functions_for(profile_data, has_server=False,
                                                can_search=bool(user_api_key))
            # Channel 0 resolves to no channel, so the builder takes the not-age-restricted
            # branch and always injects <content_policy>: the card can be opened in any
            # channel, which is why content_capability refuses an Adult profile here.
            (system_instruction, _, _, temp, top_p, top_k,
             primary_model, fallback_model_name) = await asyncio.to_thread(
                self._construct_system_instructions, host_user_id, profile_name, 0,
                present_users=present_users, functions=functions)

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

            custom_emoji = profile_data.get("placeholder_emoji") or PLACEHOLDER_EMOJI

            # --- EDIT ORIGINAL RESPONSE ---
            # `incoming` is passed explicitly: this round's turns are only appended to
            # unified_log once the reply lands, so the card would otherwise show the
            # previous round's speakers while answering this one.
            #
            # Before link reading and the web search, not after: the search names itself
            # on this placeholder when it runs long. GlobalChatPlayView rebuilds the card
            # once this returns, so a failure in either still leaves a working card.
            placeholder_embed = build_global_chat_embed(
                self.cog, host_user_id, profile_name, session_data,
                description=custom_emoji, incoming=queued_turns,
                colour=discord.Colour.dark_grey())

            await interaction.edit_original_response(embed=placeholder_embed, view=None)
            placeholder_msg = await interaction.original_response()

            url_mode = profile_data.get('url_mode', 'off')
            if 'url_mode' not in profile_data:
                url_mode = 'rag' if profile_data.get('url_fetching_enabled', False) else 'off'

            if url_mode == 'rag':
                u_text, _, u_warn = await self.cog.tools_service._process_urls_in_content(combined_prompt_text, 0, {"url_fetching_enabled": True})
                turn_warnings.extend(u_warn)
                if u_text:
                    final_user_parts.append(f"<document_context>\n" + "\n".join(u_text) + "\n</document_context>")

            # [NEW] RAG Grounding for Global Chat
            grounding_mode = resolve_grounding_mode(profile_data)

            global_rag_sources = []
            # Legacy RAG only. RAG ("tool") is the character's own `search_web`, which
            # this path declares and answers through `tool_loop.run` like every other --
            # running the gate here as well would search twice.
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

                g_res = await self._await_with_status(
                    self.cog.tools_service._get_hybrid_grounding_context(combined_prompt_text, 0, g_hist, ('global_chat', host_user_id), safety_settings=d_safe),
                    STATUS_SEARCHING_WEB, interaction.channel, None,
                    {"custom_emoji": custom_emoji, "placeholder_msg": placeholder_msg, "message_type": "embed"})
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

            app_name, app_avatar = self._resolve_appearance_data(host_user_id, profile_name)
            # No server: `recall` is not offered here, and a search bills the host's key.
            fn_ctx = tool_loop.FunctionContext(
                owner_id=host_user_id, profile_name=profile_name, author_dn=app_name,
                guild_id=None,
                triggering_user_id=queued_turns[-1]["user_id"] if queued_turns else host_user_id,
                safety_settings=_resolve_safety_settings(None, profile_data),
                search_key=user_api_key)
            chat_participant_names = [app_name] + [t['display_name'] for t in queued_turns]
            state_container = {"custom_emoji": custom_emoji, "placeholder_msg": placeholder_msg,
                               "message_type": "embed"}

            # The session reply's own path -- primary, fallback race, Final Fallback, the
            # warnings -- on the host's key. It used to be a copy of it with the fallback
            # retried by hand and no Final, and it drifted the way copies do.
            try:
                attempt = await self._attempt_reply(
                    channel=interaction.channel,
                    # No `method`: the placeholder is the card, never a child bot's message.
                    participant={"owner_id": host_user_id, "profile_name": profile_name},
                    p_settings=profile_data, owner_id=host_user_id, user_id=host_user_id,
                    system_instruction=system_instruction, primary_model=primary_model,
                    fallback_model_name=fallback_model_name, history=contents_for_api_call,
                    gen_config=reply_gen_config(profile_data, temp, top_p, top_k),
                    msg_a_id=placeholder_msg.id, app_name=app_name, app_avatar=app_avatar,
                    state_container=state_container, participant_names=chat_participant_names,
                    log_context="global_chat", functions=functions, conversation=True,
                    function_context=fn_ctx)
            except asyncio.CancelledError:
                if state_container.get('sending_task'):
                    state_container['sending_task'].cancel()
                await self._safe_delete_placeholder(interaction.channel, state_container.get('msg_b_id'))
                return

            reply = self._reply_text(attempt, profile_data, host_user_id, profile_name,
                                     chat_participant_names)
            turn_warnings.extend(reply.warnings)

            if reply.blocked:
                err_embed = placeholder_msg.embeds[0]
                err_embed.description = reply.text
                await placeholder_msg.edit(embed=err_embed)

                if state_container.get('sending_task'):
                    state_container['sending_task'].cancel()
                await self._safe_delete_placeholder(interaction.channel, state_container.get('msg_b_id'))
                state_container['msg_b_id'] = None

                await self._dispatch_warnings(interaction.channel, 'webhook', None, turn_warnings, host_user_id, profile_name)
                return

            response_text = reply.text
            sources_text_list = _format_citation_subtext(_merge_sources(global_rag_sources, reply.sources))
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

            await self._safe_delete_placeholder(interaction.channel, state_container.get('msg_b_id'))
            state_container['msg_b_id'] = None

            await self._update_sending_placeholder(interaction.channel, 'webhook', None, state_container, t1_start_mono)

            # Same builder the view uses, so the refresh that follows in play_callback
            # redraws an identical card instead of flipping colour and footer.
            embed = build_global_chat_embed(
                self.cog, host_user_id, profile_name, session_data,
                description=text_for_embed, incoming=queued_turns)

            if state_container.get('sending_task'):
                state_container['sending_task'].cancel()

            await placeholder_msg.edit(embed=embed)
            await self._dispatch_warnings(interaction.channel, 'webhook', None, turn_warnings, host_user_id, profile_name)

            t2_end_mono = time.monotonic()
            duration = t2_end_mono - t1_start_mono

            await self.cog.session_manager._save_session_to_disk(model_cache_key, 'global_chat', session_data)

        except Exception as e:
            await interaction.followup.send(f"An error occurred during the global chat: {suppress_link_previews(str(e))}", ephemeral=True)
            traceback.print_exc()