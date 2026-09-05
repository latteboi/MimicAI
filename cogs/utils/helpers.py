import re
import zlib
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
    PATTERN_SPEAKER_CLOSE,
    PATTERN_WHITESPACE_CLEANUP, NO_FALLBACK,
    IMAGE_MODEL_CAPS, IMAGE_MODEL_CAPS_DEFAULT, IMAGE_THINKING_LEVELS,
    IMAGE_GROUNDING_MODES, DEFAULT_TYPING_CURSOR,
    CRITIC_MODES, CRITIC_SCOPES, CRITIC_STRICTNESS_LEVELS, CRITIC_STRICTNESS_MIN_GRAM,
    DEFAULT_CRITIC_MODE, DEFAULT_CRITIC_SCOPE, DEFAULT_CRITIC_STRICTNESS,
    DEFAULT_CRITIC_LOOKBACK, DEFAULT_CRITIC_PERSISTENCE,
    CRITIC_LOOKBACK_MIN, CRITIC_LOOKBACK_MAX,
    CRITIC_PERSISTENCE_MIN, CRITIC_PERSISTENCE_MAX,
    THINKING_LEVELS, THINKING_SLOT_KEYS, THINKING_SLOT_DEFAULTS,
    THINKING_LEVELS_TO_GOOGLE, THINKING_LEVELS_TO_GOOGLE_BINARY,
    MEDIA_RESOLUTION_VALUES, MEDIA_RESOLUTION_TO_OPENROUTER_DETAIL,
)


#: Discord serves six default avatars at this path. A profile with no avatar of its
#: own gets one of them rather than the bot's face, so an unconfigured character still
#: reads as its own speaker in a channel full of them.
DEFAULT_AVATAR_COUNT = 6


def default_profile_avatar_url(name: str) -> str:
    """A stable default avatar for a profile that has none of its own.

    `crc32`, not `hash()`: str hashing is salted per interpreter, so the previous
    `hash(name) % 6` handed the same character a different face after every restart
    -- and every child bot and webhook in a channel restarted together, so a whole
    cast reshuffled at once.
    """
    index = zlib.crc32(str(name).encode("utf-8")) % DEFAULT_AVATAR_COUNT
    return f"https://cdn.discordapp.com/embed/avatars/{index}.png"


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
    cleared for it, so standing the filter down re-litigates nothing. Everything else
    keeps BLOCK_ONLY_HIGH, which is what the old 'low' default resolved to.

    The permissive branch sends OFF rather than BLOCK_NONE. Google documents them as
    different states -- OFF disables the filter, BLOCK_NONE leaves the classifier
    running and never blocks on it -- and on Gemini 2.5 and 3 the *unset* default is
    already OFF, so BLOCK_NONE was quietly asking for a filter this branch exists to
    stand down. Note that community reports disagree with the documentation about
    which of the two is looser in practice; if an age-restricted session starts
    returning empty candidates where it did not before, this constant is the first
    thing to put back. The strict branch is unaffected either way.

    Neither value reaches Google's non-configurable protections -- core harms such as
    child safety are always blocked -- so no setting here makes a model unfiltered.

    `profile_config` carries the one carve-out: a profile the bot owner marked
    exempt runs unfiltered wherever it runs. Callers holding a borrowed
    profile's local copy may not see the exemption, since it is only ever
    written at the source -- that fails towards the stricter threshold, which is
    the right direction.
    """
    rating = (profile_config or {}).get("content_rating") or {}
    exempt = rating.get("verdict") == "exempt"

    threshold = (
        HarmBlockThreshold.OFF
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


def strip_history_envelope(text: str) -> str:
    """A stored turn reduced to what the character actually wrote.

    `_format_history_entry` wraps every turn in `<Name> [ID: pid] [timestamp]:` ...
    `</Name>`, and some paths add a `(Thought Initiated: ... | Duration: 1.23s)` line.
    That envelope is load-bearing for the model -- it is how a participant knows who
    said what and when -- so it stays in `unified_log` and in the history handed to the
    generating model. It must not reach anything that measures the *character's* prose,
    because it is identical on every turn by construction.

    Built from the patterns `_scrub_response_text` already uses, and deliberately not
    from that function: this runs over a whole lookback window at once and must not take
    `Timeout`'s `signal.alarm` with it.

    XML markers are dropped without their contents. A `<private_response>` body is the
    character writing; the tag around it is not.
    """
    if not text:
        return ""
    cleaned = PATTERN_SYSTEM_XML_ORPHANS.sub('', text)
    cleaned = PATTERN_SYSTEM_HEADER.sub('', cleaned)
    cleaned = PATTERN_TIMESTAMP_HEADER.sub('', cleaned)
    cleaned = PATTERN_SPEAKER_CLOSE.sub('', cleaned)
    cleaned = PATTERN_METADATA.sub('', cleaned)
    return PATTERN_WHITESPACE_CLEANUP.sub('\n\n', cleaned).strip()


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

def _format_history_entry(display_name: str, timestamp: Union[datetime.datetime, str], content: str, timezone_str: str = "UTC", *, entity_id: str) -> str:
    """One turn's stored form: the identity header the model reads, plus the content.

    `entity_id` is required and keyword-only. It used to default to "00000000", which
    is not a marker of anything -- it is a plausible-looking id that reads as real. Two
    call sites had quietly been taking it: editing a Discord message rewrote the user's
    turn with it in place of their stable hash, and the debug prompt dump showed it
    instead of the speaking profile's PID. Both were invisible because the output still
    looked well-formed. A missing id is now a TypeError at the call site.
    """
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


def image_model_caps(raw_name: Optional[str]) -> dict:
    """What image options `raw_name` will honour: allowed ratios, sizes, thinkingLevel.

    Shared by the picker, which uses it to decide what to offer, and by
    MediaService.resolve_image_output_params, which uses it to decide what to send. If
    those two ever answered differently the dropdown would be offering settings the
    request path then silently dropped.

    An unrecognised id -- a model newer than this table, or a typo -- gets the
    conservative default rather than the full set.
    """
    name = raw_name or ""
    if name.upper().startswith("GOOGLE/"):
        name = name[7:]
    return IMAGE_MODEL_CAPS.get(name.lower(), IMAGE_MODEL_CAPS_DEFAULT)


def resolve_image_output_params(image_config, raw_name: Optional[str]) -> dict:
    """The aspect ratio, resolution and thinking level `raw_name` will actually take.

    Resolved per model rather than once per request, because the four image models do
    not carry the same options: the two 3.1 models take the extreme banner ratios, 2.5
    Flash has one fixed resolution and rejects imageSize outright, and only the 3.x
    models take a thinkingLevel. A fallback onto a different model therefore needs its
    own answer, which is why the image paths resolve inside the attempt rather than
    beside it.

    An option the chosen model does not carry is dropped rather than sent and 400'd.
    That is deliberate: a profile set to 2K on 3.1 Flash keeps its stored preference
    when its owner switches to 2.5 Flash for an afternoon, instead of having it
    silently rewritten to something the previous model would not honour.

    Lives here rather than on MediaService because /profile manage reports the resolved
    settings and a manager importing a service to do it would be the wrong direction.
    """
    caps = image_model_caps(raw_name)
    cfg = image_config or {}
    out = {}

    # Not a stored preference: it is what this model must be asked to return. Resolved
    # here anyway because it varies per model exactly as the other three do, and this
    # is already the one place that knows which model the request is going to.
    if caps["modalities"]:
        out["modalities"] = caps["modalities"]

    ratio = cfg.get("image_aspect_ratio") or ""
    if ratio in caps["ratios"]:
        out["aspect_ratio"] = ratio

    size = cfg.get("image_size") or ""
    if size in caps["sizes"]:
        out["image_size"] = size

    level = (cfg.get("image_thinking_level") or "").upper()
    if caps["thinking"] and level in IMAGE_THINKING_LEVELS:
        out["thinking_level"] = level

    # Sampling. Carried through the same per-model filter as everything else even
    # though no image model rejects these outright, so that one call -- and one
    # stored profile -- decides the whole request. Absent stays absent: an image
    # model with no temperature on the wire uses its own, which for the Gemini 3
    # family is the value Google asks you not to move.
    for stored, wire in (("image_temperature", "temperature"),
                         ("image_top_p", "top_p"),
                         ("image_top_k", "top_k")):
        value = cfg.get(stored)
        if value is None or value == "":
            continue
        try:
            out[wire] = int(value) if wire == "top_k" else float(value)
        except (TypeError, ValueError):
            continue

    return out


def google_thinking_caps(model_name: Optional[str]) -> Dict[str, Any]:
    """What a Google model will actually honour on the thinking config.

    Three families, three answers, and this is the only place that decides which:

    * Gemini 3 takes `thinkingLevel`. 3 Pro collapses the six levels to two -- it
      publishes LOW and HIGH and nothing between -- so `levels` says which mapping the
      caller should use.
    * Gemini 2.5 takes `thinkingBudget`, except Flash Lite, which takes neither. 2.5
      Pro refuses a budget under 128 while still allowing -1 (dynamic), hence
      `budget_floor` rather than a plain clamp.
    * Image, TTS and embedding models take nothing. An image model does accept a
      thinking level, but as an *output* control resolved by
      `resolve_image_output_params` -- not from the text profile's keys, which is why
      it is `None` here.

    The name arrives bare or prefixed depending on the call site, so both are handled.
    An unrecognised model gets `None`: sending a field a model has never heard of is a
    400, and a custom id is likelier to be a new model than a typo.
    """
    lowered = (model_name or "").lower()
    for prefix in ("google/", "openrouter/", "ollama/"):
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix):]
            break

    if any(suffix in lowered for suffix in ("-image", "-tts", "-embedding")):
        return {"mode": None, "levels": "full", "budget_floor": 0}
    if "gemini-3" in lowered:
        return {"mode": "level",
                "levels": "binary" if "pro" in lowered else "full",
                "budget_floor": 0}
    if "gemini-2.5" in lowered:
        if "lite" in lowered:
            return {"mode": None, "levels": "full", "budget_floor": 0}
        return {"mode": "budget", "levels": "full",
                "budget_floor": 128 if "pro" in lowered else 0}
    return {"mode": None, "levels": "full", "budget_floor": 0}


def resolve_thinking_params(config: Optional[Dict[str, Any]],
                            slot: str = "response",
                            role: str = "primary") -> Dict[str, Any]:
    """The thinking parameters one model of one slot of one profile runs at.

    Every generation path builds its `thinking_params` through here, which is what
    stops the slots drifting into separate opinions -- they already had. The response
    slot read the profile; the critic and the grounding summariser hardcoded the same
    literal at two call sites; the LTM summariser and the session-synopsis compactor
    passed `{}` and so inherited the adapters' own `"high"` default, paying for a full
    reasoning pass to compress a transcript.

    `role` is "primary" or "fallback", and they are genuinely different questions. The
    usual fallback is a cheap standby behind an expensive primary, so one shared effort
    either wasted the money the standby was chosen to save or under-thought a request
    the primary was configured for.

    **An unset fallback inherits the primary's resolved values**, not the slot default.
    That is what keeps a fallback a drop-in replacement for anyone who never opens the
    third dropdown: raise the response primary to Max and the standby follows, until
    the moment you say otherwise.

    Storage is sparse and must stay sparse: an absent key means "inherit", not "high".
    Writing defaults out at profile-creation time would freeze today's value onto every
    profile ever made, which is the trap `index.json["defaults"]` avoids.

    `thinking_summary_visible` is response-only by construction. A utility slot's
    thoughts reach no user -- the critic's verdict is parsed, the summariser's output is
    stored -- so asking for them buys billed tokens nobody reads.
    """
    config = config or {}
    if slot not in THINKING_SLOT_DEFAULTS:
        slot = "response"
    defaults = THINKING_SLOT_DEFAULTS[slot]
    # A slot with no entry here is one nothing can configure -- `compaction` -- and
    # resolves to its default alone.
    roles = THINKING_SLOT_KEYS.get(slot, {})

    def _read(keys) -> Dict[str, Any]:
        """The level and budget stored under one (level, budget) pair, or None each."""
        if not keys:
            return {"level": None, "budget": None}
        level_key, budget_key = keys
        level = str(config.get(level_key) or "").lower()
        if level not in THINKING_LEVELS:
            level = None
        raw = config.get(budget_key)
        try:
            budget = int(raw)
        except (TypeError, ValueError):
            budget = None
        if budget is not None and budget < -1:
            budget = None
        return {"level": level, "budget": budget}

    primary = _read(roles.get("primary"))
    level = primary["level"] or defaults["level"]
    budget = primary["budget"] if primary["budget"] is not None else defaults["budget"]

    if role == "fallback":
        # Resolved against the primary rather than the slot default, so "unset" reads
        # as "same as the model in front of me".
        secondary = _read(roles.get("fallback"))
        level = secondary["level"] or level
        budget = secondary["budget"] if secondary["budget"] is not None else budget

    summary = "off"
    if slot == "response":
        summary = "on" if str(config.get("thinking_summary_visible", "off")).lower() == "on" else "off"

    return {"thinking_level": level, "thinking_budget": budget,
            "thinking_summary_visible": summary}


def resolve_media_resolution(config: Optional[Dict[str, Any]]) -> str:
    """The stored `media_input_resolution`, or "" for "send nothing".

    Validated rather than trusted: the value reaches the wire as a protobuf enum name,
    and an imported profile or an older shard can carry anything at all.
    """
    value = str((config or {}).get("media_input_resolution") or "").upper()
    return value if value in MEDIA_RESOLUTION_VALUES else ""


def resolve_openrouter_image_detail(config: Optional[Dict[str, Any]]) -> Optional[str]:
    """The same setting as OpenRouter's per-part `detail` hint, or None.

    OpenRouter carries no request-level media-resolution field; what it forwards is the
    OpenAI-compatible `detail` on an `image_url` part, which has two useful values
    against Google's four. Folding four onto two loses precision, but the alternative
    is a setting that silently does nothing on one of the two providers that can
    actually read images.
    """
    return MEDIA_RESOLUTION_TO_OPENROUTER_DETAIL.get(resolve_media_resolution(config))


def resolve_typing_cursor(config: Optional[Dict[str, Any]], fallback_emoji: str) -> Tuple[str, str]:
    """`(mode, emoji)` for the still-typing marker on one profile's replies.

    Absent reads as the default rather than as "off" -- see TYPING_CURSOR_MODES -- so
    profiles saved before the setting existed get the effect. The emoji is the same
    `placeholder_emoji` the profile already shows while a reply is generating; the
    caller passes the global PLACEHOLDER_EMOJI as the fallback so this module does not
    have to reach for defaultConfig.

    Shared by the webhook path (DeliveryMixin) and the child-bot path
    (ChildBotManager), which run the same edit loop over two different message APIs.
    """
    cfg = config or {}
    mode = str(cfg.get("typing_cursor") or DEFAULT_TYPING_CURSOR).lower()
    if mode not in ("prefix", "below"):
        mode = "off"
    return mode, (cfg.get("placeholder_emoji") or fallback_emoji or "")


def apply_typing_cursor(text: str, mode: str, emoji: str) -> str:
    """`text` with the still-typing marker attached, or unchanged when it is off.

    Never called for the final chunk: the last edit writes the bare text, which is
    what removes the marker.
    """
    if not emoji or not text or mode not in ("prefix", "below"):
        return text
    return f"{emoji} {text}" if mode == "prefix" else f"{text}\n{emoji}"


def typing_cursor_cost(mode: str, emoji: str) -> int:
    """How much room the marker needs, so the 2000-character chunker leaves it some.

    Without this a chunk sized exactly to the limit would produce a decorated body
    Discord rejects, and the edit that carries the marker would be the one that fails.
    """
    if not emoji or mode not in ("prefix", "below"):
        return 0
    return len(emoji) + 1


def resolve_image_tools(image_config, raw_name: Optional[str]) -> Optional[list]:
    """The native search tool `raw_name` will actually take on an image request.

    Separate from resolve_image_output_params because a tool is not a generationConfig
    field -- it rides in `tools` at the top of the payload -- but it is resolved the
    same way and for the same reason: a mode the chosen model does not carry is
    dropped rather than sent and 400'd, and a fallback onto a different image model
    needs its own answer.

    Returns the snake_case declaration shape the adapters already take
    (`_build_tools` camelCases the outer key), or None for no tool at all.
    """
    caps = image_model_caps(raw_name)
    mode = ((image_config or {}).get("image_grounding_mode") or "off")
    if mode not in IMAGE_GROUNDING_MODES or mode == "off":
        return None
    if not caps["grounding"]:
        return None
    if mode == "web_images" and caps["image_search"]:
        # Nested inside the one google_search tool, not a second tool beside it.
        # `searchTypes` stays camelCase here: _build_tools only maps the outer key.
        return [{"google_search": {"searchTypes": {"webSearch": {}, "imageSearch": {}}}}]
    return [{"google_search": {}}]


def is_real_model(name: Optional[str]) -> bool:
    """False for the empty, missing and explicit "no fallback" values.

    The utility fallback dropdowns offer a None option and an unset key reads back as
    absent or "", so the pickers, the apply paths and `run_with_fallback` all need the
    same three-way answer to "is there a second model to try".
    """
    if not name:
        return False
    text = str(name).strip()
    if not text:
        return False
    for prefix in ("GOOGLE/", "OPENROUTER/", "OLLAMA/"):
        if text.upper().startswith(prefix):
            text = text[len(prefix):]
            break
    return text.upper() != NO_FALLBACK


def resolve_critic_settings(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The Anti-Repetition Critic's effective settings for one profile.

    Every critic call site reads through here so the legacy flag is folded in exactly
    once. `critic_enabled` is what older profiles carry and what the boolean toggle
    wrote; an absent `critic_mode` reads off it, and True means "full", because the
    model pass is the only thing that boolean ever selected. Both keys are written
    together by the dashboard, so a profile edited on one version still reads correctly
    on the other.

    Values out of range are clamped rather than rejected: these arrive from a Discord
    modal, and a critic that silently runs at a sane lookback beats a turn that fails
    because someone typed 400.
    """
    config = config or {}

    mode = str(config.get("critic_mode") or "").strip().lower()
    if mode not in CRITIC_MODES:
        mode = "full" if config.get("critic_enabled", False) else DEFAULT_CRITIC_MODE

    scope = str(config.get("critic_scope") or "").strip().lower()
    if scope not in CRITIC_SCOPES:
        scope = DEFAULT_CRITIC_SCOPE

    strictness = str(config.get("critic_strictness") or "").strip().lower()
    if strictness not in CRITIC_STRICTNESS_LEVELS:
        strictness = DEFAULT_CRITIC_STRICTNESS

    def _clamp(key, default, low, high):
        try:
            value = int(config.get(key, default))
        except (TypeError, ValueError):
            return default
        return max(low, min(high, value))

    return {
        "mode": mode,
        "enabled": mode != "off",
        "scope": scope,
        "strictness": strictness,
        "min_gram": CRITIC_STRICTNESS_MIN_GRAM[strictness],
        "lookback": _clamp("critic_lookback", DEFAULT_CRITIC_LOOKBACK,
                           CRITIC_LOOKBACK_MIN, CRITIC_LOOKBACK_MAX),
        "persistence": _clamp("critic_persistence", DEFAULT_CRITIC_PERSISTENCE,
                              CRITIC_PERSISTENCE_MIN, CRITIC_PERSISTENCE_MAX),
    }
