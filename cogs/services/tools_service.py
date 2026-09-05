import re
import html
import asyncio
import httpx
import discord
from typing import List, Dict, Any, Optional, Tuple

from ..utils.constants import (
    FALLBACK_MODEL_NAME, MAX_URL_CONTEXT_CHARACTERS, MAX_URL_FETCH_BYTES, WARN_URL_FETCHING_FAILED,
    WARN_GROUNDING_FAILED, DEFAULT_WEB_GROUNDING_VISUAL,
    DEFAULT_WEB_GROUNDING_TEXT, PATTERN_HTML_CONTAINERS, PATTERN_HTML_TAGS,
    PATTERN_HTML_BLANKLINES, DEFAULT_GROUNDING_RAG_PAYLOAD,
)
from ..utils.helpers import (_format_api_error, _truncate_text_by_char, is_real_model,
                            resolve_thinking_params)
from ..utils.net_guard import UnsafeURL, safe_stream
from .api_service import GoogleGenAIModel


# One httpx.AsyncClient for URL context fetching instead of one per call. Building a
# client costs ~14 ms and ~0.8 MB, and the header set here is constant, so there is
# nothing per-call to vary. Created lazily so it binds to the running event loop, and
# closed from MimicCog.cog_unload.
_URL_FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

_url_fetch_client: Optional[httpx.AsyncClient] = None


def get_url_fetch_client() -> httpx.AsyncClient:
    global _url_fetch_client
    if _url_fetch_client is None or _url_fetch_client.is_closed:
        _url_fetch_client = httpx.AsyncClient(headers=_URL_FETCH_HEADERS, follow_redirects=False)
    return _url_fetch_client


async def close_url_fetch_client():
    global _url_fetch_client
    if _url_fetch_client is not None and not _url_fetch_client.is_closed:
        await _url_fetch_client.aclose()
    _url_fetch_client = None


class ToolsService:
    """Owns RAG web-search grounding, URL fetching/scraping context extraction,
    and the Anti-Repetition Critic pass.

    Holds a back-reference to the parent cog for state/logic not yet migrated
    (profile/session lookups, model instantiation, and API-call logging),
    per the transitional Dependency Injection pattern in CLAUDE.md.
    """

    def __init__(self, cog):
        self.cog = cog

    async def _run_critic(self, history: list, char_name: str, guild_id: int,
                          p_config: Optional[Dict[str, Any]] = None,
                          session_transcript: Optional[List[str]] = None,
                          instructions: Optional[str] = None
                          ) -> Tuple[Optional[str], Optional[str]]:
        """Finds linguistic loops in recent output and returns a negative constraint.

        Returns `(constraint, source)`. `source` is "lexical" when the free in-process
        scan caught it, "model" when the critic model wrote it, and None when nothing
        was produced -- which is what `/session audit` reports per turn. It matters
        because a profile in "full" mode still short-circuits on the lexical scan, so
        "full" in the config does not mean a model call was actually paid for.

        `p_config` is the speaking profile's config, passed in by the caller. It used to
        be hunted for here by scanning `cog.multi_profile_channels` for a session whose
        participant matched `char_name` -- but that dict is keyed by channel id and the
        only caller passed `channel.guild.id`, so the lookup never hit. `critic_model`
        and `critic_fallback_model` were configurable in three places and had never once
        been read; every critic call ran on FALLBACK_MODEL_NAME with an empty config,
        which also cost Ollama-hosted profiles their host URL.

        `session_transcript` is what "session" scope supplies: every participant's recent
        lines rather than this profile's own. Passed in for the same reason -- the log
        belongs to the caller, and reaching back into the cog for it is what broke this.
        """
        from ..utils.helpers import (_fast_repetition_scan, resolve_critic_settings,
                                     strip_history_envelope)

        p_config = p_config or {}
        settings = resolve_critic_settings(p_config)
        if not settings["enabled"]:
            return None, None

        # Both sources arrive as *stored* turns -- `<Name> [ID: pid] [timestamp]:` around
        # the dialogue, and sometimes a Duration line under it. That envelope is byte
        # for byte identical on every turn a profile speaks, so handing it to a
        # repetition detector flags every session after three turns on the scaffolding
        # alone: `_fast_repetition_scan` compares the first five words of each turn, and
        # for one speaker those words are its name, its pid and the date. The strip is
        # on this transcript only; the log and the generating model keep the envelope.
        if session_transcript is not None:
            candidates = session_transcript
        else:
            candidates = []
            for turn in reversed(history):
                if isinstance(turn, dict) and turn.get('role') == 'model':
                    parts = [p if isinstance(p, str) else p.get('text', '') for p in turn.get('parts', [])]
                    if parts:
                        candidates.append(" ".join(parts))
                        # Counted before the strip, so a turn that scrubs down to nothing
                        # does not silently pull an older one into the window.
                        if len(candidates) >= settings["lookback"]:
                            break
            candidates.reverse()

        chronological_turns = [c for c in (strip_history_envelope(t) for t in candidates) if c]
        chronological_turns = chronological_turns[-settings["lookback"]:]

        if len(chronological_turns) < 2:
            return None, None

        # 1. Ultra-fast local lexical scan (0ms latency fast-path)
        is_repetitive, reason = _fast_repetition_scan(chronological_turns, settings["min_gram"])
        if is_repetitive and reason:
            return f"DO NOT repeat phrasing or structure. {reason}.", "lexical"

        # "lexical" stops here by design: the scan is free and in-process, so it is the
        # mode that costs a profile nothing per turn. Only "full" buys the model pass.
        if settings["mode"] != "full":
            return None, None

        if len(chronological_turns) < 3:
            return None, None

        transcript = "\n---\n".join(chronological_turns)

        # Resolved by the caller (profile prompt, then /mod's instance-wide override,
        # then the shipped default), for the same reason p_config is: the prompt lives
        # encrypted in the profile's `prompts`, which is the caller's to read.
        # str.format on user-authored text, so an unknown or stray brace is a ValueError
        # or a KeyError rather than a crashed turn.
        instructions = instructions or self.cog.profile_manager._default_critic_instructions()
        try:
            system_instruction = instructions.format(char_name=char_name)
        except (KeyError, IndexError, ValueError):
            system_instruction = instructions

        critic_model_raw = p_config.get("critic_model") or FALLBACK_MODEL_NAME

        try:
            critic_cfg = {"temperature": 0.1, "top_p": 0.95}

            async def _attempt(model_name, is_fallback):
                # Resolved inside the attempt so the retry gets the fallback slot's own
                # effort rather than the primary's.
                t_params = resolve_thinking_params(
                    p_config, "critic", "fallback" if is_fallback else "primary")
                model = self.cog.api_service._instantiate_model(
                    model_name, guild_id, None, system_instruction, None, t_params, None, p_config)
                return await model.generate_content_async(
                    [f"Transcript:\n{transcript}"], generation_config=critic_cfg)

            resp, _used, _was_fallback = await self.cog.api_service.run_with_fallback(
                critic_model_raw, p_config.get("critic_fallback_model"), _attempt,
                label="Anti-repetition critic")

            if resp.text:
                if "PASS" not in resp.text.upper():
                    return resp.text.strip(), "model"
        except Exception as e:
            print(f"Critic error: {e}")
        return None, None

    async def _process_urls_in_content(self, content: str, guild_id: int, profile_settings: Dict[str, Any], warning_channel: Optional[discord.abc.Messageable] = None) -> Tuple[List[str], List[Dict], List[str]]:
        warnings = []
        if not profile_settings.get("url_fetching_enabled", False):
            return [], [], warnings

        text_contexts = []
        media_parts = []
        url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        found_urls = re.findall(url_pattern, content)
        if not found_urls:
            return [], [], warnings

        client = get_url_fetch_client()
        for url in found_urls[:2]:
            try:
                if not url.startswith(('http://', 'https://')):
                    url = 'http://' + url

                async with safe_stream(client, "HEAD", url, timeout=5.0) as head_response:
                    head_response.raise_for_status()
                    content_type = head_response.headers.get('content-type', '').lower()

                # Strictly handle images and text
                if content_type.startswith('image/'):
                    media_parts.append({"url": url, "mime_type": content_type})

                elif 'text/html' in content_type:
                    # Streamed with a hard byte cap. Reading .text on an unbounded body
                    # made peak RSS a function of whatever page the user linked.
                    chunks, total = [], 0
                    async with safe_stream(client, "GET", url, timeout=10.0) as get_response:
                        get_response.raise_for_status()
                        async for chunk in get_response.aiter_bytes(65536):
                            chunks.append(chunk)
                            total += len(chunk)
                            if total >= MAX_URL_FETCH_BYTES:
                                break
                        encoding = get_response.encoding or 'utf-8'
                    page_content = b"".join(chunks)[:MAX_URL_FETCH_BYTES].decode(encoding, errors='replace')
                    del chunks

                    def _sync_scrub_html():
                        # Two full-string rewrites, not four: the container tags share one
                        # alternation with a backreference, so style/script/head/nav/... are
                        # stripped in a single pass before the generic tag strip.
                        clean_content = PATTERN_HTML_CONTAINERS.sub('', page_content)
                        clean_content = PATTERN_HTML_TAGS.sub('', clean_content)

                        clean_content = html.unescape(clean_content)
                        clean_content = "\n".join([line.strip() for line in clean_content.splitlines() if line.strip()])
                        clean_content = PATTERN_HTML_BLANKLINES.sub('\n\n', clean_content)
                        return clean_content

                    try:
                        # Offloaded to a worker thread (regex/unescape work on full page bodies is CPU-bound);
                        # signal.alarm-based Timeout can't be used off the main thread, so the timeout is
                        # enforced here via wait_for instead.
                        clean_content = await asyncio.wait_for(asyncio.to_thread(_sync_scrub_html), timeout=3.0)
                        truncated_content = _truncate_text_by_char(clean_content, MAX_URL_CONTEXT_CHARACTERS)
                        url_context = f"Source URL: {url}\nExtracted Content:\n{truncated_content}"
                        text_contexts.append(url_context)
                    except TimeoutError:
                        warnings.append(WARN_URL_FETCHING_FAILED.format(reason="HTML parsing timed out"))
                        continue

            except UnsafeURL:
                # Deliberately unspecific: the reason reaches the channel, and
                # distinguishing "refused" from "timed out" would tell whoever
                # posted the link which internal hosts exist.
                warnings.append(WARN_URL_FETCHING_FAILED.format(reason="destination not permitted"))
            except Exception as e:
                warnings.append(WARN_URL_FETCHING_FAILED.format(reason=_format_api_error(e)))

        return text_contexts, media_parts, warnings

    async def _get_hybrid_grounding_context(self, user_query: str, guild_id: int, conversation_history: List, mapping_key: Any, safety_settings: Optional[Dict] = None, is_for_image: bool = False, warning_channel: Optional[discord.abc.Messageable] = None) -> Optional[Tuple[str, List[Dict], bool, Optional[str]]]:
        effective_guild_id = guild_id or 0
        api_key = self.cog.storage_manager._get_api_key_for_guild(effective_guild_id)
        if not api_key:
            return None, [], False, None

        status = "api_error"
        warning_str = None
        # Provisional, for the finally-block log if we fail before resolving the user's
        # configured grounding model. Reassigned to the model actually used below --
        # this used to stay hardcoded, so anyone who changed grounding_rag_model had
        # every grounding call attributed to flash-lite in their usage stats.
        model_name = FALLBACK_MODEL_NAME
        try:
            history_for_decision = conversation_history

            # [UPDATED] Standardize history for the Grounding Model
            # Omit technical metadata and recalled memories, but ALLOW previous search summaries
            clean_history_lines = []
            for turn in history_for_decision:
                parts = turn.get('parts', [])
                if not parts: continue

                raw_text = "".join(p if isinstance(p, str) else p.get('text', '') for p in parts)
                if not raw_text: continue

                # 1. Strip technical metadata line
                text = re.sub(r'\(\s*Thought Initiated:.*?\)\s*\n?', '', raw_text).strip()

                # 2. Selective Block Filtering (Recognize new XML tags)
                lines = text.split('\n')
                filtered_lines = []
                skip_block = False
                for line in lines:
                    l_strip = line.strip()

                    if any(l_strip.startswith(prefix) for prefix in [
                        "<document_context>",
                        "<archive_context>",
                        "<internal_note>",
                        "<image_context>"
                    ]):
                        skip_block = True
                        continue

                    if skip_block:
                        if l_strip.startswith(("</document_context>", "</archive_context>", "</internal_note>", "</image_context>")):
                            skip_block = False
                        continue

                    filtered_lines.append(line)

                final_turn_text = "\n".join(filtered_lines).strip()
                if final_turn_text:
                    clean_history_lines.append(final_turn_text)

            history_transcript = "\n\n".join(clean_history_lines)

            # New combined system instruction
            if is_for_image:
                system_instruction = self.cog.global_prompts.get("WEB_GROUNDING_VISUAL", DEFAULT_WEB_GROUNDING_VISUAL)
            else:
                # The original system instruction for text-based queries
                system_instruction = self.cog.global_prompts.get("WEB_GROUNDING_TEXT", DEFAULT_WEB_GROUNDING_TEXT)

            # [FIXED] Use XML structure for the data payload
            payload_template = self.cog.global_prompts.get("GROUNDING_RAG_PAYLOAD", DEFAULT_GROUNDING_RAG_PAYLOAD)
            user_prompt = payload_template.format(transcript=history_transcript, query=user_query)

            # [FIXED] Use universal dict configuration for Google GenAI v2 Tools
            grounding_tool = {"google_search": {}}

            # [NEW] Utility Routing Logic for Grounding RAG
            rag_model_raw = FALLBACK_MODEL_NAME
            rag_fallback_raw = None
            # Bound up front rather than only inside the session branch: the grounding
            # phase runs without a resolvable session often enough, and an empty config
            # is exactly the "use the slot default" case the resolver is built for.
            p_cfg: Dict[str, Any] = {}
            session_id = mapping_key[1] if isinstance(mapping_key, tuple) else None
            if session_id:
                session = self.cog.multi_profile_channels.get(session_id)
                if session:
                    # Just grab the first profile's settings for the RAG model to keep it simple
                    first_p = session.get("profiles", [])[0] if session.get("profiles") else None
                    if first_p:
                        p_idx = self.cog.profile_manager._get_user_index(first_p["owner_id"])
                        is_b = first_p["profile_name"] in p_idx.get("borrowed", [])
                        p_cfg = self.cog.profile_manager._get_profile_config(first_p["owner_id"], first_p["profile_name"], is_b) or {}
                        rag_model_raw = p_cfg.get("grounding_rag_model", FALLBACK_MODEL_NAME)
                        rag_fallback_raw = p_cfg.get("grounding_rag_fallback_model")

            gen_config = {"temperature": 0.1, "top_p": 0.95}

            def _is_rerouted(raw: str) -> bool:
                """True when `raw` names a model this phase cannot honour.

                The native Google Search tool has no OpenRouter equivalent in our
                adapter, so anything routed there -- or to any other provider -- cannot
                serve this phase and is answered by the standard Google fallback.
                """
                return bool(raw) and "/" in raw and not raw.upper().startswith("GOOGLE/")

            def _resolve_google_name(raw: str) -> str:
                """The bare Google model id this raw name resolves to.

                GoogleGenAIModel is constructed directly here rather than through
                _instantiate_model, so the provider prefix has to come off on every
                branch -- including the reroute one, whose default carries one.
                """
                name = FALLBACK_MODEL_NAME if (not raw or _is_rerouted(raw)) else raw
                return name[7:] if name.upper().startswith("GOOGLE/") else name

            # Resolved before the retry rather than inside it, so run_with_fallback's
            # "skip a fallback equal to the primary" rule sees the model that will
            # actually be called. Two different OpenRouter ids both answer as the Google
            # default, and retrying that is one more call to be refused the same way.
            rag_primary = _resolve_google_name(rag_model_raw)
            rag_fallback = _resolve_google_name(rag_fallback_raw) if is_real_model(rag_fallback_raw) else None

            for raw, resolved, slot in ((rag_model_raw, rag_primary, "primary"),
                                        (rag_fallback_raw, rag_fallback, "fallback")):
                if resolved and _is_rerouted(raw):
                    print(f"Grounding summariser: {slot} '{raw}' cannot serve the native "
                          f"search tool; using '{resolved}' instead.")

            # Set inside the attempt so the log names the model the call was actually
            # made against -- including which of the two it ended up on.
            model_name = rag_primary
            model = None

            async def _attempt(name, is_fallback):
                nonlocal model_name, model
                model_name = name
                # Was a hardcoded low/512 literal, duplicated at the critic call site.
                # Same default, now a per-profile, per-role setting resolved inside the
                # attempt so the retry gets the fallback's own effort.
                model = GoogleGenAIModel(
                    api_key=api_key,
                    model_name=model_name,
                    system_instruction=system_instruction,
                    safety_settings=safety_settings,
                    thinking_params=resolve_thinking_params(
                        p_cfg, "grounding", "fallback" if is_fallback else "primary"),
                    tools=[grounding_tool]
                )
                return await model.generate_content_async([user_prompt], generation_config=gen_config)

            grounding_response, _used, _was_fallback = await self.cog.api_service.run_with_fallback(
                rag_primary, rag_fallback, _attempt, label="Grounding summariser")
            status = "success"

            if not grounding_response.text:
                return None, [], False, None

            rag_text = grounding_response.text

            lines = rag_text.strip().split('\n')
            decision = lines[0].strip().lower()

            if decision != 'yes':
                return None, [], False, None

            summary = "\n".join(lines[1:]).strip()
            if not summary:
                return None, [], False, None

            truncated_summary = _truncate_text_by_char(summary, MAX_URL_CONTEXT_CHARACTERS)

            if is_for_image:
                summary_context = f"<external_context>\n{truncated_summary}\n</external_context>"
            else:
                summary_context = (
                    f"<external_context>\n"
                    f"DO NOT include footnotes, citation markers, or URLs in your text.\n"
                    f"{truncated_summary}\n"
                    f"</external_context>"
                )

            sources = []
            if grounding_response.candidates and hasattr(grounding_response.raw.candidates[0], 'grounding_metadata'):
                metadata = grounding_response.raw.candidates[0].grounding_metadata
                if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks is not None:
                    for chunk in metadata.grounding_chunks:
                        if hasattr(chunk, 'web'):
                            sources.append({'uri': chunk.web.uri, 'title': chunk.web.title})

            if not sources:
                warning_str = WARN_GROUNDING_FAILED.format(reason="The AI hallucinated a response without retrieving valid web citations.")
                return None, [], False, warning_str

            return summary_context, sources, True, None

        except Exception as e:
            status = "api_error"
            warning_str = WARN_GROUNDING_FAILED.format(reason=_format_api_error(e))
            return None, [], False, warning_str
        finally:
            self.cog._log_api_call(user_id=0, guild_id=guild_id, context="grounding_combined", model_used=model_name, status=status)
