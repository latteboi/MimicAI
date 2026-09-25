import re
import html
import asyncio
import httpx
import discord
from typing import List, Dict, Any, Optional, Tuple

from ..utils.constants import (
    GREEDY_SAMPLING,
    MAX_GROUNDING_SUMMARY_CHARACTERS, MAX_URL_CONTEXT_CHARACTERS,
    MAX_URL_FETCH_BYTES, WARN_URL_FETCHING_FAILED,
    WARN_GROUNDING_FAILED, DEFAULT_WEB_GROUNDING_VISUAL,
    DEFAULT_WEB_GROUNDING_TEXT, DEFAULT_WEB_SEARCH_RESEARCH,
    PATTERN_HTML_CONTAINERS, PATTERN_HTML_TAGS,
    PATTERN_HTML_BLANKLINES, DEFAULT_GROUNDING_RAG_PAYLOAD,
)
from ..utils.helpers import (_format_api_error, _truncate_text_by_char,
                            resolve_thinking_params)
from ..utils.net_guard import UnsafeURL, safe_stream
from .api_service import MissingKeyError


#: The researcher's verdict, read off its first line. "yes" has to be the first word
#: there, which is what the prompt asks for; anything the model decorates it with --
#: a full stop, bold markers, an em dash and a comment -- no longer discards a search
#: that has already been run and billed. A line that merely contains "yes" further
#: along ("no, yes would need a search") is not a verdict and must not match.
_GROUNDING_DECISION = re.compile(r'^[\s*_`"\'>-]*yes\b', re.IGNORECASE)


#: The search tool, in Google's spelling. One object, because it is constant and both
#: grounding paths send exactly it; on OpenRouter `_instantiate_model` sends that
#: provider's own search tool in its place (OPENROUTER_SERVER_TOOLS).
_GOOGLE_SEARCH_TOOL = {"google_search": {}}

#: What the researcher is sampled at. Low, in both paths: this call reports what a
#: search returned, and creativity here is the failure mode, not the feature.
_GROUNDING_GEN_CONFIG = {"temperature": 0.1, "top_p": 0.95}


def grounding_sources(response) -> List[Dict[str, str]]:
    """The web pages a grounded response cites, as {'uri', 'title'}.

    An empty list is the signal both callers act on: a search that cited nothing was
    answered from the model's own weights, which is the one thing this whole phase
    exists to avoid.
    """
    sources: List[Dict[str, str]] = []
    candidates = getattr(getattr(response, "raw", None), "candidates", None) or []
    if not candidates or not getattr(response, "candidates", None):
        return sources
    metadata = getattr(candidates[0], "grounding_metadata", None)
    for chunk in (getattr(metadata, "grounding_chunks", None) or ()):
        web = getattr(chunk, "web", None)
        if web is not None:
            sources.append({"uri": web.uri, "title": web.title})
    return sources


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
                          instructions: Optional[str] = None,
                          config_owner_id: Optional[int] = None
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

        critic_model_raw, critic_fallbacks = self.cog.api_service.model_chain(
            p_config, "critic_model", config_owner_id)

        try:
            critic_cfg = dict(GREEDY_SAMPLING)

            async def _attempt(model_name, is_fallback):
                # Resolved inside the attempt so the retry gets the fallback slot's own
                # effort rather than the primary's.
                t_params = resolve_thinking_params(
                    p_config, "critic", "fallback" if is_fallback else "primary")
                model = self.cog.api_service._instantiate_model(
                    model_name, guild_id, None, system_instruction, None, t_params, None, p_config,
                    config_owner_id=config_owner_id)
                return await model.generate_content_async(
                    [f"Transcript:\n{transcript}"], generation_config=critic_cfg)

            resp, _used, _was_fallback = await self.cog.api_service.run_with_fallback(
                critic_model_raw, critic_fallbacks, _attempt,
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
            except httpx.HTTPStatusError as e:
                # The site answered with an error. Its status is the whole story -- the link
                # is the one they just posted -- and _format_api_error reads statuses as a
                # model API's, where a 404 is "Model Not Found".
                status = e.response
                warnings.append(WARN_URL_FETCHING_FAILED.format(
                    reason=f"HTTP {status.status_code} {status.reason_phrase}".strip()))
            except Exception as e:
                warnings.append(WARN_URL_FETCHING_FAILED.format(reason=_format_api_error(e)))

        return text_contexts, media_parts, warnings

    def _researcher(self, name: str, is_fallback: bool, p_cfg: Dict[str, Any],
                    owner_id: Optional[int], guild_id: Optional[int], instruction: str,
                    safety_settings: Optional[Dict], policy: Optional[Dict[str, Any]] = None):
        """The grounding slot's model `name`, carrying the search tool.

        Built by `_instantiate_model` like every other slot's, so the key, the data policy
        and the provider come from the one place that decides them -- and the slot may
        hold an OpenRouter model, whose search runs on OpenRouter. Thinking is resolved
        per attempt so the retry gets the fallback's own effort.
        """
        return self.cog.api_service._instantiate_model(
            name, guild_id, owner_id, instruction, safety_settings,
            resolve_thinking_params(p_cfg, "grounding", "fallback" if is_fallback else "primary"),
            [_GOOGLE_SEARCH_TOOL], p_cfg, config_owner_id=owner_id, **(policy or {}))

    async def _get_hybrid_grounding_context(self, user_query: str, guild_id: int, conversation_history: List, mapping_key: Any, safety_settings: Optional[Dict] = None, is_for_image: bool = False, warning_channel: Optional[discord.abc.Messageable] = None) -> Optional[Tuple[str, List[Dict], bool, Optional[str]]]:
        status = "api_error"
        warning_str = None
        # Provisional, for the finally-block log if we fail before resolving the slot.
        # Reassigned to the model actually called below.
        model_name = None
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

            # [NEW] Utility Routing Logic for Grounding RAG
            # Bound up front rather than only inside the session branch: the grounding
            # phase runs without a resolvable session often enough, and an empty config
            # is exactly the "use the slot default" case the resolver is built for.
            p_cfg: Dict[str, Any] = {}
            owner_id = None
            session_id = mapping_key[1] if isinstance(mapping_key, tuple) else None
            if session_id:
                session = self.cog.multi_profile_channels.get(session_id)
                if session:
                    # Just grab the first profile's settings for the RAG model to keep it simple
                    first_p = session.get("profiles", [])[0] if session.get("profiles") else None
                    if first_p:
                        owner_id = first_p["owner_id"]
                        p_idx = self.cog.profile_manager._get_user_index(first_p["owner_id"])
                        is_b = first_p["profile_name"] in p_idx.get("borrowed", [])
                        p_cfg = self.cog.profile_manager._get_profile_config(first_p["owner_id"], first_p["profile_name"], is_b) or {}

            rag_primary, rag_fallbacks = self.cog.api_service.model_chain(
                p_cfg, "grounding_rag_model", owner_id)

            # Set inside the attempt so the log names the model the call was actually
            # made against -- including which of the chain it ended up on.
            model_name = rag_primary

            async def _attempt(name, is_fallback):
                nonlocal model_name
                model_name = name
                model = self._researcher(name, is_fallback, p_cfg, owner_id, guild_id,
                                         system_instruction, safety_settings)
                return await model.generate_content_async(
                    [user_prompt], generation_config=_GROUNDING_GEN_CONFIG)

            grounding_response, _used, _was_fallback = await self.cog.api_service.run_with_fallback(
                rag_primary, rag_fallbacks, _attempt, label="Grounding summariser")
            status = "success"

            if not grounding_response.text:
                return None, [], False, None

            rag_text = grounding_response.text

            lines = rag_text.strip().split('\n')
            # Normalised, not compared raw. The search has already been made and paid
            # for by the time this line is read, and an exact `== 'yes'` threw the whole
            # result away over a full stop, a bold marker, or "yes -- searching now".
            decision = _GROUNDING_DECISION.match(lines[0])

            if not decision:
                return None, [], False, None

            summary = "\n".join(lines[1:]).strip()
            if not summary:
                return None, [], False, None

            # The grounding budget, not the URL-context one. This used to truncate at
            # MAX_URL_CONTEXT_CHARACTERS -- 16000, so never -- while the constant written
            # for this call sat unread, and a thousand-word block landed in a roleplay
            # turn and flattened the character's voice under it.
            truncated_summary = _truncate_text_by_char(summary, MAX_GROUNDING_SUMMARY_CHARACTERS)

            if is_for_image:
                summary_context = f"<external_context>\n{truncated_summary}\n</external_context>"
            else:
                summary_context = (
                    f"<external_context>\n"
                    f"DO NOT include footnotes, citation markers, or URLs in your text.\n"
                    f"{truncated_summary}\n"
                    f"</external_context>"
                )

            sources = grounding_sources(grounding_response)

            if not sources:
                warning_str = WARN_GROUNDING_FAILED.format(reason="The AI hallucinated a response without retrieving valid web citations.")
                return None, [], False, warning_str

            return summary_context, sources, True, None

        except MissingKeyError:
            # No key here for any model in the chain: grounding is off, as it always
            # was on a server without one, not a failure to warn every round about.
            return None, [], False, None
        except Exception as e:
            status = "api_error"
            warning_str = WARN_GROUNDING_FAILED.format(reason=_format_api_error(e))
            return None, [], False, warning_str
        finally:
            self.cog._log_api_call(user_id=0, guild_id=guild_id, context="grounding_combined", model_used=model_name, status=status)

    async def search_for_tool(self, query: str, *, guild_id: Optional[int],
                              owner_id: Optional[int], profile_name: str,
                              safety_settings: Optional[Dict] = None,
                              policy: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """The `search_web` function's answer, as the object handed back to the model.

        The same slot, the same search tool and the same settings as
        `_get_hybrid_grounding_context` -- and none of its transcript. The character has
        already decided what it wants to know and written the query, so this call sends
        one line where the legacy mode sends the conversation, and the research prompt
        may not re-open a decision the turn has already paid for.

        Every failure is answered rather than raised. The model is mid-reply waiting on
        this, and "the search failed" is something a character can work around while a
        dropped turn is not -- the same rule `recall` follows, for the same reason.

        `policy` is Global Chat's `{"policy_guild_id", "conversation"}`, as its replies
        are built with: the host's own keys, policed where the card is open. Otherwise the
        server's keys.
        """
        if not query:
            return {"error": "search_web needs a `query` string."}

        index = self.cog.profile_manager._get_user_index(owner_id) if owner_id else {}
        is_borrowed = profile_name in (index.get("borrowed") or [])
        p_cfg = (self.cog.profile_manager._get_profile_config(
            owner_id, profile_name, is_borrowed) or {}) if owner_id else {}

        primary, fallbacks = self.cog.api_service.model_chain(p_cfg, "grounding_rag_model", owner_id)
        instructions = self.cog.global_prompts.get("WEB_SEARCH_RESEARCH",
                                                   DEFAULT_WEB_SEARCH_RESEARCH)
        status = "api_error"
        model_name = primary
        try:
            async def _attempt(name, is_fallback):
                nonlocal model_name
                model_name = name
                model = self._researcher(name, is_fallback, p_cfg, owner_id, guild_id,
                                         instructions, safety_settings, policy)
                return await model.generate_content_async(
                    [query], generation_config=_GROUNDING_GEN_CONFIG)

            response, _used, _was_fallback = await self.cog.api_service.run_with_fallback(
                primary, fallbacks, _attempt, label="Web search")
            status = "success"

            sources = grounding_sources(response)
            summary = (getattr(response, "text", "") or "").strip()
            if not summary:
                return {"note": "That search returned nothing."}
            # No sources means the model answered from its own weights without
            # searching, which is the one outcome this whole phase exists to prevent --
            # and handing it back as a result would launder a guess into a fact the
            # character then states with a citation-shaped confidence. The `note` is
            # deliberately something the character can act on.
            #
            # Printed, not silent. "Answered without searching" and "searched and found
            # nothing" are the same empty result to every caller, and they want opposite
            # responses -- the first is a model or prompt problem on a slot the operator
            # chose, the second is the truth about the query. The legacy path surfaces
            # the same condition as a channel warning; this one has a character waiting,
            # so it goes to the log instead.
            if not sources:
                print(f"Web search: {model_name} answered without citing a search "
                      f"({query[:60]!r}) -- dropped: {summary[:100]!r}")
                return {"note": "That search found nothing on the web. Say you do not "
                                "know rather than guessing."}
            return {"summary": _truncate_text_by_char(summary, MAX_GROUNDING_SUMMARY_CHARACTERS),
                    "sources": sources}
        except MissingKeyError:
            return {"error": "Web search is unavailable: no API key here for its model."}
        except Exception as e:  # noqa: BLE001 -- see docstring
            print(f"search_web failed: {_format_api_error(e)}")
            return {"error": "That search failed."}
        finally:
            self.cog._log_api_call(user_id=0, guild_id=guild_id or 0,
                                   context="grounding_search_tool",
                                   model_used=model_name, status=status)
