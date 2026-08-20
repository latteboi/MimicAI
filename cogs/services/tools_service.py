import re
import html
import asyncio
import httpx
import discord
from typing import List, Dict, Any, Optional, Tuple

from ..utils.constants import (
    FALLBACK_MODEL_NAME, MAX_URL_CONTEXT_CHARACTERS, MAX_URL_FETCH_BYTES, WARN_URL_FETCHING_FAILED,
    WARN_GROUNDING_FAILED, DEFAULT_ANTI_REPETITION_PROMPT, DEFAULT_WEB_GROUNDING_VISUAL,
    DEFAULT_WEB_GROUNDING_TEXT, PATTERN_HTML_CONTAINERS, PATTERN_HTML_TAGS,
    PATTERN_HTML_BLANKLINES, DEFAULT_GROUNDING_RAG_PAYLOAD,
)
from ..utils.helpers import _add_inline_citations, _format_api_error, _truncate_text_by_char
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
        _url_fetch_client = httpx.AsyncClient(headers=_URL_FETCH_HEADERS, follow_redirects=True)
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

    async def _run_critic(self, history: list, char_name: str, guild_id: int) -> Optional[str]:
        """Uses fast lexical heuristics and a reasoning model to find linguistic loops and robotic staleness."""
        from ..utils.helpers import _fast_repetition_scan

        recent_turns = []
        count = 0
        for turn in reversed(history):
            if isinstance(turn, dict) and turn.get('role') == 'model':
                parts = [p if isinstance(p, str) else p.get('text', '') for p in turn.get('parts', [])]
                if parts:
                    recent_turns.append(" ".join(parts))
                    count += 1
            if count >= 4: break

        if len(recent_turns) < 2: return None

        chronological_turns = list(reversed(recent_turns))

        # 1. Ultra-fast local lexical scan (0ms latency fast-path)
        is_repetitive, reason = _fast_repetition_scan(chronological_turns)
        if is_repetitive and reason:
            return f"DO NOT repeat phrasing or structure. {reason}."

        if len(recent_turns) < 3: return None

        transcript = "\n---\n".join(chronological_turns)

        system_instruction = self.cog.global_prompts.get("ANTI_REPETITION", DEFAULT_ANTI_REPETITION_PROMPT).format(char_name=char_name)

        # [NEW] Route to profile's defined critic model
        # We need the profile to fetch the setting, pass via kwargs or fetch via guild context.
        # For simplicity since this is called mid-generation, we will rely on the Fallback model if we can't find the profile context easily here,
        # but we should pass profile data.

        # Since _run_critic signature is (_run_critic(self, history: list, char_name: str, guild_id: int)),
        # we will use the fallback model for now to keep the signature clean, or fetch it via active session.
        # To perfectly align, let's fetch it via the active session in this guild.
        critic_model_raw = FALLBACK_MODEL_NAME
        session = self.cog.multi_profile_channels.get(guild_id) # guild_id is actually channel_id in the _multi_profile_worker call
        if session:
            for p in session.get("profiles", []):
                if p["profile_name"] == char_name or self.cog.user_appearances.get(str(p["owner_id"]), {}).get(p["profile_name"], {}).get("custom_display_name") == char_name:
                    p_index = self.cog.profile_manager._get_user_index(p["owner_id"])
                    p_is_b = p["profile_name"] in p_index.get("borrowed", [])
                    p_config = self.cog.profile_manager._get_profile_config(p["owner_id"], p["profile_name"], p_is_b) or {}
                    critic_model_raw = p_config.get("critic_model", FALLBACK_MODEL_NAME)
                    break

        try:
            t_params = {"thinking_budget": 512, "thinking_summary_visible": "off", "thinking_level": "low"}
            critic_cfg = {"temperature": 0.1, "top_p": 0.95}

            model = self.cog.api_service._instantiate_model(critic_model_raw, guild_id, None, system_instruction, None, t_params, None, p_config if 'p_config' in locals() else {})

            resp = await model.generate_content_async([f"Transcript:\n{transcript}"], generation_config=critic_cfg)

            if resp.text:
                if "PASS" not in resp.text.upper():
                    return resp.text.strip()
        except Exception as e:
            print(f"Critic error: {e}")
        return None

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

                async with client.stream("HEAD", url, timeout=5.0) as head_response:
                    head_response.raise_for_status()
                    content_type = head_response.headers.get('content-type', '').lower()

                # Strictly handle images and text
                if content_type.startswith('image/'):
                    media_parts.append({"url": url, "mime_type": content_type})

                elif 'text/html' in content_type:
                    # Streamed with a hard byte cap. Reading .text on an unbounded body
                    # made peak RSS a function of whatever page the user linked.
                    chunks, total = [], 0
                    async with client.stream("GET", url, timeout=10.0) as get_response:
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
        model_name = 'gemini-2.5-flash-lite' # Use a single, tool-capable model
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

            is_or = False
            actual_model_name = rag_model_raw
            if rag_model_raw.upper().startswith("OPENROUTER/"):
                actual_model_name = rag_model_raw[11:]
                is_or = True
            elif rag_model_raw.upper().startswith("GOOGLE/"):
                actual_model_name = rag_model_raw[7:]
                is_or = False
            elif "/" in rag_model_raw:
                is_or = True

            t_params = {"thinking_budget": 512, "thinking_summary_visible": "off", "thinking_level": "low"}

            if is_or:
                # OpenRouter doesn't support the Google Search Tool natively yet in our adapter
                # We will fall back to Google for the RAG phase if they attempt to route grounding to OpenRouter
                actual_model_name = FALLBACK_MODEL_NAME

            model = GoogleGenAIModel(
                api_key=api_key,
                model_name=actual_model_name,
                system_instruction=system_instruction,
                safety_settings=safety_settings,
                thinking_params=t_params,
                tools=[grounding_tool]
            )

            gen_config = {"temperature": 0.1, "top_p": 0.95}

            grounding_response = await model.generate_content_async([user_prompt], generation_config=gen_config)
            status = "success"

            if not grounding_response.text:
                return None, [], False, None

            # [NEW] Apply inline citations to the RAG model's text before passing it to the profile
            rag_text = grounding_response.text
            if hasattr(grounding_response, 'raw') and grounding_response.raw.candidates and hasattr(grounding_response.raw.candidates[0], 'grounding_metadata'):
                rag_text = _add_inline_citations(rag_text, grounding_response.raw.candidates[0].grounding_metadata)

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
                    f"FOOTNOTES (e.g. **[1]** **[2]**) MUST BE INCLUDED IN YOUR TEXT; DO NOT INCLUDE URLS.\n"
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
