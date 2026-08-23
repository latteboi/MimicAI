import re
import asyncio
import platform
import discord
import signal
import functools
import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict, Tuple, Any, Optional, Union
import orjson as json
from .constants import (
    DISCORD_MAX_MESSAGE_LENGTH, API_ERROR_MAPPINGS,
    HARM_CATEGORIES, HarmBlockThreshold, HarmCategory,
    PATTERN_SYSTEM_XML_BLOCKS, PATTERN_SYSTEM_XML_ORPHANS,
    PATTERN_REASONING_BLOCKS, PATTERN_REASONING_ORPHANS, PATTERN_SYSTEM_HEADER,
    PATTERN_TIMESTAMP_HEADER, PATTERN_METADATA, PATTERN_MESSAGE_LINK,
    PATTERN_WHITESPACE_CLEANUP,
)


def _channel_is_age_restricted(channel: Any) -> bool:
    """True only when a resolved channel object is flagged age-restricted.

    DMs, group channels, and anything the gateway cache could not resolve count
    as not age-restricted -- the same direction
    ProfileManager._check_unrestricted_safety_policy already fails in, so the
    placement gate and the provider thresholds agree on every channel type.
    """
    if not isinstance(channel, (discord.TextChannel, discord.Thread, discord.VoiceChannel)):
        return False
    try:
        return channel.is_nsfw()
    except Exception:
        return False


def _resolve_safety_settings(channel: Any, profile_config: Optional[Dict[str, Any]] = None) -> Dict[HarmCategory, HarmBlockThreshold]:
    """Maps the *destination channel* onto the provider harm thresholds.

    This used to key off the profile's own safety_level, which diverged from the
    placement gate in the worst possible direction: a profile the classifier
    ruled 'adult' was confined to an age-restricted channel and *still* sent
    BLOCK_ONLY_HIGH, so the provider filtered content the channel had already
    been cleared for -- surfacing as empty candidates and a generic generation
    failure. Keying off the channel makes the filter agree with the gate by
    construction, and puts it on the same axis as the <content_policy> block
    that prompt_builder injects for non-age-restricted channels.

    An age-restricted channel only ever receives profiles the gate has already
    cleared for it, so BLOCK_NONE re-litigates nothing. Everything else keeps
    BLOCK_ONLY_HIGH, which is what the old 'low' default resolved to.

    `profile_config` carries the one carve-out: a profile the bot owner marked
    exempt runs unfiltered wherever it runs. Callers holding a borrowed
    profile's local copy may not see the exemption, since it is only ever
    written at the source -- that fails towards the stricter threshold, which is
    the right direction.
    """
    rating = (profile_config or {}).get("content_rating") or {}
    exempt = rating.get("verdict") == "exempt"

    threshold = (
        HarmBlockThreshold.BLOCK_NONE
        if exempt or _channel_is_age_restricted(channel)
        else HarmBlockThreshold.BLOCK_ONLY_HIGH
    )
    return {cat: threshold for cat in HARM_CATEGORIES}


def _split_into_sentences_with_abbreviations(text: str) -> List[str]:
    abbreviations = {
        'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'rev.', 'hon.', 'st.', 'sr.', 'jr.', 'capt.', 'sgt.', 'col.', 'gen.',
        'etc.', 'vs.', 'i.e.', 'e.g.', 'cf.', 'et al.', 'viz.',
        'ave.', 'blvd.', 'rd.',
        'a.m.', 'p.m.', 'in.', 'ft.', 'yd.', 'mi.',
        'approx.', 'apt.', 'assn.', 'asst.', 'bldg.', 'co.', 'corp.', 'dept.', 'est.', 'inc.', 'ltd.', 'mfg.', 'vol.'
    }

    potential_sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    if not potential_sentences:
        return []

    merged_sentences = []
    for s in potential_sentences:
        if not merged_sentences:
            merged_sentences.append(s)
            continue

        last_sentence = merged_sentences[-1]
        words = last_sentence.split()
        if words and words[-1].lower() in abbreviations:
            merged_sentences[-1] += " " + s
        else:
            merged_sentences.append(s)

    return merged_sentences

def _yield_message_chunks(content: str, max_length: int = DISCORD_MAX_MESSAGE_LENGTH):
    """Generator that splits strings precisely to fit Discord limits without breaking paragraphs/sentences."""
    remaining = content
    while remaining:
        if len(remaining) <= max_length:
            yield remaining
            break

        split_pos = -1
        para_break = remaining.rfind('\n\n', 0, max_length)
        if para_break != -1:
            split_pos = para_break + 2
        else:
            sent_break = remaining.rfind('. ', 0, max_length)
            if sent_break != -1:
                split_pos = sent_break + 2
            else:
                split_pos = max_length

        yield remaining[:split_pos]
        remaining = remaining[split_pos:]

# Try importing native Rust/C extension if compiled into the environment
try:
    import mimic_core  # type: ignore
    _HAS_NATIVE_CORE = True
except ImportError:
    _HAS_NATIVE_CORE = False

def _estimate_text_tokens(text: str) -> int:
    """High-throughput token estimation with native BPE fast-path."""
    if not text: return 0
    
    if _HAS_NATIVE_CORE and hasattr(mimic_core, "count_tokens"):
        return mimic_core.count_tokens(text)

    # Optimised heuristic based on cl100k / gemini average token byte lengths
    length = len(text)
    if length < 16:
        return max(1, len(text.split()))
    return int(length / 3.75) + 1

def _fast_repetition_scan(recent_turns: List[str], min_gram: int = 4, max_gram: int = 8) -> Tuple[bool, Optional[str]]:
    """Zero-allocation rolling n-gram and sentence overlap scanner.
    Quickly detects repetitive phrases and linguistic loops across conversation turns.
    """
    if len(recent_turns) < 2:
        return False, None

    if _HAS_NATIVE_CORE and hasattr(mimic_core, "scan_repetition"):
        return mimic_core.scan_repetition(recent_turns, min_gram, max_gram)

    def extract_ngrams(words: List[str], n: int) -> set:
        return set(" ".join(words[i:i+n]) for i in range(len(words) - n + 1))

    tokenised_turns = []
    for turn in recent_turns:
        clean = re.sub(r'[^\w\s]', '', turn.lower()).split()
        if clean:
            tokenised_turns.append(clean)

    if len(tokenised_turns) < 2:
        return False, None

    # 1. Check for consecutive identical opening structures
    if len(tokenised_turns) >= 3:
        openings = [" ".join(t[:5]) for t in tokenised_turns if len(t) >= 5]
        if len(openings) >= 3 and len(set(openings)) == 1:
            return True, f"Repetitive opening phrase detected: '{openings[0]}...'"

    # 2. Check rolling N-gram intersection across recent turns
    latest_words = tokenised_turns[-1]
    if len(latest_words) >= min_gram:
        latest_ngrams = extract_ngrams(latest_words, min_gram)
        for prev_words in tokenised_turns[:-1]:
            if len(prev_words) >= min_gram:
                prev_ngrams = extract_ngrams(prev_words, min_gram)
                overlap = latest_ngrams.intersection(prev_ngrams)
                if len(overlap) >= 3:
                    sample = next(iter(overlap))
                    return True, f"Severe repetition overlap on phrase: '{sample}'"

    return False, None

def _truncate_text_by_char(text: str, max_chars: int) -> str:
    if len(text) > max_chars:
        return text[:max_chars]
    return text

def _is_history_effectively_empty(history: list) -> bool:
    # A session is effectively empty if it contains NO model turns.
    # System notes and director prompts are injected as 'user' turns.
    # Real conversation requires a 'model' response. If none exist, no real conversation is left.
    for turn in history:
        if isinstance(turn, dict) and turn.get('role') == 'model':
            return False
    return True

def _sanitise_filename(name: str) -> str:
    """Removes any special characters or directory traversal dots/slashes."""
    return re.sub(r'[^a-zA-Z0-9_-]', '', name)

def _pf(val): return float(val) if val and val.strip() else None
def _pi(val): return int(val) if val and val.strip() else None
def _ps(val): return val.strip() if val and val.strip() else None
def _pb(val): return val.strip().lower() == "on"

def _get_user_hash(user_id: int) -> str:
    import hashlib
    # Prefix with 'U' and return 15 hex characters for a total 16-character PID
    return "A" + hashlib.sha256(str(user_id).encode()).hexdigest()[:15].upper()

class Timeout:
    def __init__(self, seconds=2, error_message='Function call timed out'):
        self.seconds = seconds
        self.error_message = error_message
        self.is_windows = platform.system() == "Windows"

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        if not self.is_windows:
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        if not self.is_windows:
            signal.alarm(0)

@functools.lru_cache(maxsize=128)
def _compile_name_scrub_patterns(escaped_names: Tuple[str, ...]) -> Tuple[re.Pattern, re.Pattern]:
    names_pattern_part = "|".join(escaped_names)
    pattern_name_prefix = re.compile(rf'(?:^|\n)(?:<\s*(?:{names_pattern_part})\s*>|{names_pattern_part})\s*:\s*', flags=re.IGNORECASE)
    pattern_name_xml = re.compile(rf'</?\s*(?:{names_pattern_part})\s*>', flags=re.IGNORECASE)
    return pattern_name_prefix, pattern_name_xml


def _scrub_response_text(text: str, participant_names: Optional[List[str]] = None) -> str:
    """Hard-coded filter to remove any leaked script formatting or specific XML tags from the AI's response."""
    if not text or not text.strip():
        return ""

    raw_original = text.strip()

    try:
        with Timeout(seconds=2, error_message="Scrubbing timed out due to complex regex."):
            scrubbed_text = raw_original.replace("&#x20;", " ")

            scrubbed_text = PATTERN_SYSTEM_XML_BLOCKS.sub('', scrubbed_text)
            scrubbed_text = PATTERN_SYSTEM_XML_ORPHANS.sub('', scrubbed_text)
            scrubbed_text = PATTERN_REASONING_BLOCKS.sub('', scrubbed_text)
            scrubbed_text = PATTERN_REASONING_ORPHANS.sub('', scrubbed_text)
            scrubbed_text = PATTERN_SYSTEM_HEADER.sub('', scrubbed_text)
            scrubbed_text = PATTERN_TIMESTAMP_HEADER.sub('', scrubbed_text)
            scrubbed_text = PATTERN_METADATA.sub('', scrubbed_text)

            if participant_names:
                escaped_names = tuple(re.escape(name.strip()) for name in participant_names if name and name.strip())
                if escaped_names:
                    pattern_name_prefix, pattern_name_xml = _compile_name_scrub_patterns(escaped_names)
                    scrubbed_text = pattern_name_prefix.sub('', scrubbed_text).strip()
                    scrubbed_text = pattern_name_xml.sub('', scrubbed_text).strip()

            scrubbed_text = PATTERN_MESSAGE_LINK.sub('', scrubbed_text).strip()
            scrubbed_text = PATTERN_WHITESPACE_CLEANUP.sub('\n\n', scrubbed_text).strip()

            # Diagnostic Safeguard: If scrubbing wiped out non-empty content, log and recover
            if not scrubbed_text and raw_original:
                print(f"[SCRUBBER DIAGNOSTIC] Warning: Aggressive scrubbing deleted response text. Falling back to sanitized raw text.")
                fallback_text = PATTERN_SYSTEM_XML_BLOCKS.sub('', raw_original)
                fallback_text = PATTERN_SYSTEM_XML_ORPHANS.sub('', fallback_text).strip()
                return fallback_text if fallback_text else raw_original

            return scrubbed_text
    except TimeoutError as e:
        print(f"Warning: {e}. Returning original text.")
        return raw_original

TIMEZONE_ALIASES: Dict[str, str] = {
    "AEST": "Australia/Sydney",
    "AEDT": "Australia/Sydney",
    "ACST": "Australia/Adelaide",
    "ACDT": "Australia/Adelaide",
    "AWST": "Australia/Perth",
    "PST": "America/Los_Angeles",
    "PDT": "America/Los_Angeles",
    "MST": "America/Denver",
    "MDT": "America/Denver",
    "CST": "America/Chicago",
    "CDT": "America/Chicago",
    "EST": "America/New_York",
    "EDT": "America/New_York",
    "AKST": "America/Anchorage",
    "HST": "Pacific/Honolulu",
    "JST": "Asia/Tokyo",
    "KST": "Asia/Seoul",
    "CST_CHINA": "Asia/Shanghai",
    "SGT": "Asia/Singapore",
    "HKT": "Asia/Hong_Kong",
    "IST": "Asia/Kolkata",
    "PKT": "Asia/Karachi",
    "BST": "Europe/London",
    "GMT": "Europe/London",
    "CET": "Europe/Berlin",
    "CEST": "Europe/Berlin",
    "EET": "Europe/Athens",
    "EEST": "Europe/Athens",
    "MSK": "Europe/Moscow",
    "NZST": "Pacific/Auckland",
    "NZDT": "Pacific/Auckland"
}

def _resolve_zoneinfo(tz_str: Optional[str]) -> Tuple[ZoneInfo, str]:
    """Resolves arbitrary timezone input or acronym into a valid IANA ZoneInfo instance."""
    if not tz_str or not tz_str.strip():
        return ZoneInfo("UTC"), "UTC"
    
    clean_tz = tz_str.strip()
    upper_tz = clean_tz.upper()

    if upper_tz in TIMEZONE_ALIASES:
        canonical = TIMEZONE_ALIASES[upper_tz]
        return ZoneInfo(canonical), canonical

    try:
        return ZoneInfo(clean_tz), clean_tz
    except Exception:
        # Check case-insensitive match against aliases
        for alias, canonical in TIMEZONE_ALIASES.items():
            if clean_tz.lower() == alias.lower() or clean_tz.lower() == canonical.lower():
                return ZoneInfo(canonical), canonical
        return ZoneInfo("UTC"), "UTC"

def _format_history_entry(display_name: str, timestamp: Union[datetime.datetime, str], content: str, timezone_str: str = "UTC", entity_id: str = "00000000") -> str:
    # Convert string timestamp to datetime object if necessary
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.datetime.fromisoformat(timestamp)
        except ValueError:
            timestamp = datetime.datetime.now(datetime.timezone.utc)

    try:
        target_tz, _ = _resolve_zoneinfo(timezone_str)
        local_time = timestamp.astimezone(target_tz)
        time_str = local_time.strftime("[%a, %d %b %Y, %I:%M %p %Z]")
    except Exception:
        time_str = timestamp.strftime("[%a, %d %b %Y, %I:%M %p UTC]")

    return f"<{display_name}> [ID: {entity_id}] {time_str}:\n{content}\n</{display_name}>\n\n"

def _add_inline_citations(text: str, grounding_metadata) -> str:
    if not grounding_metadata: return text
    supports = getattr(grounding_metadata, 'grounding_supports', None)
    chunks = getattr(grounding_metadata, 'grounding_chunks', None)
    if not supports or not chunks: return text

    # Sort descending to avoid shifting indices when inserting text
    sorted_supports = sorted(supports, key=lambda s: getattr(s.segment, 'end_index', 0), reverse=True)

    for support in sorted_supports:
        end_index = getattr(support.segment, 'end_index', None)
        indices = getattr(support, 'grounding_chunk_indices', [])
        if end_index is None or not indices: continue

        citation_links = []
        for i in indices:
            if i < len(chunks):
                citation_links.append(f"**[{i + 1}]**")

        if citation_links:
            citation_string = " " + ", ".join(citation_links)
            text = text[:end_index] + citation_string + text[end_index:]
    return text

def _format_citation_subtext(grounding_sources: List[Dict]) -> List[str]:
    if not grounding_sources: return []
    source_links = []

    # Deduplicate by URI to prevent redundant footnotes
    seen_uris = set()
    deduped_sources = []
    for s in grounding_sources:
        uri = s.get('uri')
        if uri and uri not in seen_uris:
            seen_uris.add(uri)
            deduped_sources.append(s)

    for i, source in enumerate(deduped_sources):
        domain = source.get('title')
        if not domain or domain == 'URL Context' or domain == 'User Provided Link':
            try:
                from urllib.parse import urlparse
                domain = urlparse(source['uri']).netloc
                if domain.startswith('www.'): domain = domain[4:]
            except Exception:
                domain = "source"
        domain = re.sub(r'\[|\]', '', domain)
        domain = re.sub(r'\s+', ' ', domain).strip()
        source_links.append(f"**[{i+1}]** [{domain}](<{source['uri']}>)")

    links_per_line = 5
    chunked_links = [source_links[i:i + links_per_line] for i in range(0, len(source_links), links_per_line)]

    messages = []
    for i, chunk in enumerate(chunked_links):
        if i == 0:
            messages.append(f"> -# Sources:  {'  '.join(chunk)}")
        else:
            messages.append(f"> -# {'  '.join(chunk)}")

    return messages

def _get_sanitized_history_and_author(history: List[str], user_id_map: Dict[int, str], primary_author_id: int) -> Tuple[List[str], str]:
    primary_author_name = user_id_map.get(primary_author_id, "A user")
    return history, primary_author_name

def _serialize_content_for_debug(content: Any) -> Optional[Dict]:
    if not content or not hasattr(content, 'role') or not hasattr(content, 'parts'):
        return None

    parts_list = []
    for part in content.parts:
        if hasattr(part, 'text'):
            parts_list.append({'text': part.text})
        elif hasattr(part, 'inline_data') and part.inline_data:
            # Redact image data for brevity in debug logs
            parts_list.append({'inline_data': {'mime_type': part.inline_data.mime_type, 'data': '[IMAGE_DATA]'}})
        elif hasattr(part, 'file_data') and part.file_data:
            parts_list.append({'file_data': {'file_uri': part.file_data.file_uri}})

    if not parts_list:
        return None

    return {'role': content.role, 'parts': parts_list}

def _format_debug_prompt(turns_for_debug: List[Any]) -> str:
    serialized_turns = []
    for turn in turns_for_debug:
        serialized = _serialize_content_for_debug(turn)
        if serialized:
            serialized_turns.append(serialized)

    if not serialized_turns:
        return "```json\n[]\n```"

    json_string = json.dumps(serialized_turns, option=json.OPT_INDENT_2).decode('utf-8')

    if len(json_string) > 1980: # Add buffer for markdown
        json_string = json_string[:1977] + "..."

    return f"```json\n{json_string}```"

def _format_and_chunk_thought_summary(thought_text: str) -> List[str]:
    if not thought_text:
        return []

    header = "> -# Thoughts\n"
    wrapper_start = "||```\n"
    wrapper_end = "\n```||"

    # Max length for the raw text inside the block, accounting for wrappers
    max_len_first = 2000 - len(header) - len(wrapper_start) - len(wrapper_end)
    max_len_subsequent = 2000 - len(wrapper_start) - len(wrapper_end)

    chunks = []
    remaining_text = thought_text

    # Handle the first chunk which includes the header
    if remaining_text:
        chunk = remaining_text[:max_len_first]
        remaining_text = remaining_text[max_len_first:]
        chunks.append(f"{header}{wrapper_start}{chunk}{wrapper_end}")

    # Handle any subsequent chunks without the header
    while remaining_text:
        chunk = remaining_text[:max_len_subsequent]
        remaining_text = remaining_text[max_len_subsequent:]
        chunks.append(f"{wrapper_start}{chunk}{wrapper_end}")

    return chunks

def _format_api_error(error: Exception) -> str:
    """Analyses API exceptions to provide specific, user-friendly diagnostic strings."""
    if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        return "Generation Stalled (No data received for 20s)" if "Generation stalled or timed out" in str(error) else "Response Timed-out (Took longer than 2 minutes)"

    error_str = str(error)

    if "Ollama API Error" in error_str:
        return f"Ollama Error: {error_str.split(':', 1)[-1].strip()}"

    if "OpenRouter API Error" in error_str:
        try:
            err_data = json.loads(error_str[error_str.find("{"):])
            msg = err_data.get("error", {}).get("message", "")
            return "Provider Error" if msg == "Provider returned error" else f"OpenRouter: {msg}"
        except Exception: pass

    error_str_clean = re.sub(r'https?://[^\s]+', '', error_str).lower()

    if "429" in error_str_clean or "resource_exhausted" in error_str_clean:
        return "**OpenRouter Rate Limit:** Add credits to your OpenRouter account for increased RPM & RPD." if "openrouter" in error_str.lower() or "sk-or" in error_str.lower() else "**Gemini Rate Limit:** Set up billing in Google AI Studio for increased RPM & RPD (Paid Tier 1+)."

    for keys, error_msg in API_ERROR_MAPPINGS.items():
        if any(k in error_str_clean for k in keys):
            return error_msg

    clean_err = error_str.replace('"', "'").replace('{', '').replace('}', '').replace('\n', ' ')
    return clean_err[:80] + "..." if len(clean_err) > 80 else clean_err
