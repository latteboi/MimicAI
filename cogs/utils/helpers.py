import os
import re
import zlib
import hashlib
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
    ATTACHMENT_TAG, ATTACHMENT_TAG_DEFAULT, ATTACHMENT_TAG_KINDS,
    DOCUMENT_MIME_TYPES, TEXT_ATTACHMENT_EXTENSIONS,
    DISCORD_MAX_MESSAGE_LENGTH, API_ERROR_MAPPINGS, VOICE_SAMPLE_SLOTS, VOICE_SAMPLE_TYPES,
    HARM_CATEGORIES, HarmBlockThreshold, HarmCategory,
    PATTERN_SYSTEM_XML_BLOCKS, PATTERN_SYSTEM_XML_ORPHANS,
    PATTERN_REASONING_BLOCKS, PATTERN_REASONING_ORPHANS, PATTERN_SYSTEM_HEADER,
    PATTERN_TIMESTAMP_HEADER, PATTERN_METADATA, PATTERN_MESSAGE_LINK,
    PATTERN_SPEAKER_CLOSE,
    PATTERN_WHITESPACE_CLEANUP, NO_FALLBACK, SYSTEM_MODEL_DEFAULTS,
    SYSTEM_MODEL_DEFAULTS_BY_PROVIDER,
    IMAGE_COMMAND_PREFIXES, IMAGE_MODEL_CAPS, IMAGE_MODEL_CAPS_DEFAULT, IMAGE_THINKING_LEVELS,
    OPENROUTER_IMAGE_CAPS_UNKNOWN, IMAGE_MIME_SUFFIXES, IMAGE_SUFFIX_MIMES,
    IMAGE_GROUNDING_TOOL_MODES, DEFAULT_TYPING_CURSOR,
    CRITIC_MODES, CRITIC_SCOPES, CRITIC_STRICTNESS_LEVELS, CRITIC_STRICTNESS_MIN_GRAM,
    DEFAULT_CRITIC_MODE, DEFAULT_CRITIC_SCOPE, DEFAULT_CRITIC_STRICTNESS,
    DEFAULT_CRITIC_LOOKBACK, DEFAULT_CRITIC_PERSISTENCE,
    CRITIC_LOOKBACK_MIN, CRITIC_LOOKBACK_MAX,
    CRITIC_PERSISTENCE_MIN, CRITIC_PERSISTENCE_MAX,
    THINKING_LEVELS, THINKING_SLOT_KEYS, THINKING_SLOT_DEFAULTS,
    THINKING_LEVELS_TO_GOOGLE, THINKING_LEVELS_TO_GOOGLE_BINARY,
    MEDIA_RESOLUTION_VALUES, MEDIA_RESOLUTION_TO_OPENROUTER_DETAIL,
    GROUNDING_MODE_LABELS,
    DEFAULT_KICKSTART_CONTINUE, DEFAULT_KICKSTART_IDLE, DEFAULT_WHISPER_RECAP,
    SUPERSEDED_LTM_SUMMARIZATION_HASHES,
    OPENROUTER_SERVICE_TIER_VALUES,
    UNREADABLE_MEDIA_DEFAULT, UNREADABLE_MEDIA_KEYS, UNREADABLE_MEDIA_LABELS,
    UNREADABLE_MEDIA_VALUES,
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

#: How many of a character's own turns in a row are answered with "Continue" before the
#: note turns to "Idle", and how long the whole cycle is: one Continue, then two Idles,
#: repeating. See `kickstart_note`.
KICKSTART_FOLLOW_UP_CYCLE = 3


def kickstart_note(history: List[Dict[str, Any]],
                   global_prompts: Optional[Dict[str, str]] = None) -> Optional[str]:
    """The pseudo-user turn to append when a history ends on the character's own turn.

    An adapter needs the last turn to be the user's, so a character speaking into
    silence -- a proactive round, a regenerate, a cast that has run past everyone --
    gets one written for it. Which one is what the character is told about that silence,
    and the wording of all three is the operator's (`/mod` -> Kickstart).

    Every follow-up used to be **Idle** ("no response from anyone, or no user is
    present"), which is a true sentence and a terrible note to receive twice: told it is
    alone, a character writes *about* being alone, and three turns later the room has
    become the subject. **Continue** ("continue the public conversation") is the same
    invitation without the premise. So the run alternates -- the first unanswered turn
    is a lull in a conversation that is still going, and the two after it are the
    silence it actually is -- and it cycles rather than latching, so a character left
    alone all evening is periodically asked to carry on rather than told forty times
    over that nobody is there.

    The run length is `len(parts)` on the trailing model entry: `_build_history_for_participant`
    merges consecutive same-role turns and contributes exactly one part per turn, so it
    is how many times in a row this character has spoken with nobody -- no user, no
    other participant -- answering. A turn from anyone else ends the entry, and the
    history no longer ends on a model role at all.

    Returns None when the history does not end on a model turn, which is the caller's
    "append nothing".
    """
    if not history or history[-1].get('role', 'user') != 'model':
        return None
    prompts = global_prompts or {}
    parts = history[-1].get('parts') or []
    text = "".join(p if isinstance(p, str) else p.get('text', '') if isinstance(p, dict) else ''
                   for p in parts)
    # A private reply nobody else saw leaves the public conversation exactly where it
    # was, however long the run: what it is owed is Continue, not a note about silence.
    if "<private_response>" in text or len(parts) % KICKSTART_FOLLOW_UP_CYCLE == 1:
        return prompts.get("KICKSTART_CONTINUE", DEFAULT_KICKSTART_CONTINUE)
    return prompts.get("KICKSTART_IDLE", DEFAULT_KICKSTART_IDLE)


def whisper_recap(pending: List[Dict[str, Any]], timezone: Optional[str],
                  global_prompts: Optional[Dict[str, str]] = None) -> Optional[str]:
    """The recap of whispers a character has not yet spoken since, on its own clock.

    `pending` is whisper turns, as `_get_pending_whispers_for_participant` returns them.
    They used to be the stored text, on whatever clock it was stamped in, so a character
    in AEST read a whisper stamped in UTC a minute ago as ten hours old. None when there
    are none.
    """
    if not pending:
        return None
    clock, _ = _resolve_zoneinfo(timezone)
    template = (global_prompts or {}).get("WHISPER_RECAP", DEFAULT_WHISPER_RECAP)
    return template.format(whispers="\n---\n".join(
        restamp_turn(t.get("content") or "", turn_posted_at(t), clock) for t in pending))


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

#: How a time is written in a turn's header and in `<current_time>` alike.
#: Seconds, not finer: enough to tell two messages in one minute apart, and every extra
#: digit is paid for again in every header of every prompt.
TURN_TIME_FORMAT = "%a, %d %b %Y, %I:%M:%S %p %Z"


def _format_history_entry(display_name: str, timestamp: Union[datetime.datetime, str], content: str, timezone_str: str = "UTC", *, entity_id: str) -> str:
    """One turn's stored form: the identity header the model reads, plus the content.

    `entity_id` is required and keyword-only. It used to default to "00000000", which
    is not a marker of anything -- it is a plausible-looking id that reads as real.
    Editing a Discord message had quietly been taking it, rewriting the user's turn with
    it in place of their stable hash, invisibly because the output still looked
    well-formed. A missing id is now a TypeError at the call site.
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
        time_str = f"[{local_time.strftime(TURN_TIME_FORMAT)}]"
    except Exception:
        time_str = timestamp.strftime(f"[{TURN_TIME_FORMAT.replace('%Z', 'UTC')}]")

    return f"<{display_name}> [ID: {entity_id}] {time_str}:\n{content}\n</{display_name}>\n\n"


#: A stored turn's opening header, capturing its time: `<Name> [ID: x] [time]:`.
_TURN_HEADER_TIME = re.compile(r'<[^>\r\n]+> \[ID: [^\]\r\n]+\] \[([^\]\r\n]+)\]:')


def restamp_turn(content: str, moment: Optional[datetime.datetime], tz) -> str:
    """`content` with its header's time shown on `tz`'s clock.

    A turn is stamped once, in its speaker's zone, when it is stored -- so a transcript
    mixes clocks, and a character in AEST read a UTC speaker as ten hours out. Every
    reader restamps to its own clock. `moment` is `turn_posted_at`; with none, or a
    header that is not the stored shape, the content is returned as it was.
    """
    if moment is None:
        return content
    match = _TURN_HEADER_TIME.match(content)
    if not match:
        return content
    stamp = moment.astimezone(tz).strftime(TURN_TIME_FORMAT)
    if match.group(1) == stamp:
        return content
    return f"{content[:match.start(1)]}{stamp}{content[match.end(1):]}"


def turn_posted_at(turn: Dict[str, Any]) -> Optional[datetime.datetime]:
    """When a turn happened, in UTC, or None when nothing in it says.

    The single answer to that question: the log's order, the session viewer and a
    regeneration all have to agree on it. A turn records a `timestamp` only when
    something other than a Discord message decided its moment -- a system note, a
    whisper, the synopsis. A delivered turn carries the ids of the messages it posted
    instead, and a Discord id encodes the moment its message went up, which is exactly
    the order the channel shows. The first id, not the last: a reply split across
    several messages happened when it started arriving.
    """
    stamp = turn.get("timestamp")
    if stamp:
        try:
            moment = datetime.datetime.fromisoformat(str(stamp))
            # A naive stamp is UTC: compared with an aware one it would raise.
            return moment if moment.tzinfo else moment.replace(tzinfo=datetime.timezone.utc)
        except (TypeError, ValueError):
            pass
    ids = turn.get("message_ids")
    if ids:
        try:
            return discord.utils.snowflake_time(int(ids[0]))
        except (TypeError, ValueError, OverflowError, OSError):
            pass
    return None


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

# Only the first line _format_citation_subtext emits carries the "Sources:" label;
# every line after it opens straight on a numbered link. Anchored, so the numbered
# form cannot match a model's own text further into a message.
_CITATION_SUBTEXT_RE = re.compile(r'^>\s*-#\s+(?:Sources:|\*\*\[\d+\]\*\*)')

def is_citation_subtext(content: str) -> bool:
    """True for any message _format_citation_subtext produced.

    Regeneration keeps a turn's source lines and deletes its other follow-ups. It
    used to decide with `"Sources:" in content`, which is false for every
    continuation line, so a turn citing more than five sources lost all but the
    first line of them each time it was regenerated.
    """
    return bool(content) and bool(_CITATION_SUBTEXT_RE.match(content))

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

#: A bare link: anchored on its scheme, and stopped at whitespace, a quote or an angle
#: bracket, since error text quotes the URL it failed on as 'https://...'.
_BARE_LINK = re.compile(r"https?://[^\s<>'\"]+")
#: A link already wrapped for Discord, which a length cut must not split.
_WRAPPED_LINK = re.compile(r"<https?://[^\s<>]*>")
_LINK_OPENERS = {")": "(", "]": "[", "}": "{"}


def suppress_link_previews(text: str) -> str:
    """`text` with every bare link wrapped in `<...>`, so Discord previews none of them.

    For anything that puts error text in front of a user. An exception's message carries
    the URL it failed on, and posted bare it unfurled into an embed of that page -- or of
    whatever half of a link survived a length cut. Wrapped, the link stays clickable. A
    link already wrapped is left alone, so running this twice changes nothing.
    """
    def wrap(match: "re.Match[str]") -> str:
        link = match.group(0)
        if match.start() and text[match.start() - 1] == "<":
            return link
        end = len(link)
        # Sentence punctuation after a link is not part of it. A closing bracket is only
        # when the link opened one, as a Wikipedia title does.
        while end:
            last = link[end - 1]
            if last in ".,;:!?":
                end -= 1
            elif last in _LINK_OPENERS and link.count(last, 0, end) > link.count(_LINK_OPENERS[last], 0, end):
                end -= 1
            else:
                break
        return f"<{link[:end]}>{link[end:]}"
    return _BARE_LINK.sub(wrap, text)


def _cut_outside_links(text: str, limit: int) -> str:
    """`text` cut to `limit` characters and an ellipsis, stopping short of a wrapped link
    rather than cutting through it -- half a link is a broken one, and unwrapped."""
    if len(text) <= limit:
        return text
    cut = limit
    for match in _WRAPPED_LINK.finditer(text):
        if match.start() < cut < match.end():
            cut = match.start()
            break
    return text[:cut].rstrip() + "..."


def _format_api_error(error: Exception) -> str:
    """Analyses API exceptions to provide specific, user-friendly diagnostic strings.

    Whichever branch answers, the answer leaves through suppress_link_previews: several
    pass a provider's own message through, and those carry links.
    """
    return suppress_link_previews(_describe_api_error(error))


def _describe_api_error(error: Exception) -> str:
    # An exception that arrives already phrased for the user keeps its phrasing.
    formatted = getattr(error, "formatted_reason", None)
    if formatted:
        return formatted
    if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        return "Generation Stalled (No data received for 20s)" if "Generation stalled or timed out" in str(error) else "Response Timed-out (Took longer than 2 minutes)"

    error_str = str(error)

    if "Ollama API Error" in error_str:
        return f"Ollama Error: {error_str.split(':', 1)[-1].strip()}"

    if "OpenRouter API Error" in error_str:
        try:
            err_data = json.loads(error_str[error_str.find("{"):])
            err = err_data.get("error", {})
            msg = err.get("message", "")
            if err.get("code") == 429:
                # "Provider returned error" is all a host's own 429 says at the top level,
                # which read as a fault rather than as traffic that will clear.
                if msg == "Provider returned error":
                    host = (err.get("metadata") or {}).get("provider_name") or "The model's host"
                    return f"**OpenRouter Rate Limit:** {host} is turning requests away. Try again shortly."
                return f"**OpenRouter Rate Limit:** {msg}"
            # Collapsed to one line: a schema error arrives as a pretty-printed JSON
            # document inside `message`, which is a wall in the journal and a wall in
            # the channel. Cut wide, since the rest is a host's own sentence.
            msg = " ".join(msg.split())
            return ("Provider Error" if msg == "Provider returned error"
                    else _cut_outside_links(f"OpenRouter: {msg}", 300))
        except Exception: pass

    error_str_clean = re.sub(r'https?://[^\s]+', '', error_str).lower()

    if "429" in error_str_clean or "resource_exhausted" in error_str_clean:
        return "**OpenRouter Rate Limit:** Add credits to your OpenRouter account for increased RPM & RPD." if "openrouter" in error_str.lower() or "sk-or" in error_str.lower() else "**Gemini Rate Limit:** Set up billing in Google AI Studio for increased RPM & RPD (Paid Tier 1+)."

    for keys, error_msg in API_ERROR_MAPPINGS.items():
        if any(k in error_str_clean for k in keys):
            return error_msg

    clean_err = suppress_link_previews(
        error_str.replace('"', "'").replace('{', '').replace('}', '').replace('\n', ' '))
    # Wrapped before the cut, so the cut can stop short of a link instead of splitting it.
    return _cut_outside_links(clean_err, 80)


def upload_too_large(error: BaseException) -> bool:
    """Whether Discord refused a send for what it attached: HTTP 413, error code 40005.

    The message itself was fine, so a sender posts it again without the file rather than
    lose both. No other refusal says anything about the attachment.
    """
    return isinstance(error, discord.HTTPException) and (error.status == 413 or error.code == 40005)


def attachment_mime(attachment: Any) -> str:
    """An attachment's type, lowercased and without its parameters.

    Takes a `discord.Attachment` or the dict a child bot's payload carries, because every
    caller here is reached from both and each used to unpack the two shapes by hand.
    Discord sends `text/plain; charset=utf-8`, so the parameters have to come off before
    anything compares the type.
    """
    raw = attachment.get("content_type") if isinstance(attachment, dict) else getattr(attachment, "content_type", None)
    return (raw or "").split(";", 1)[0].strip().lower()


def attachment_filename(attachment: Any) -> str:
    name = attachment.get("filename") if isinstance(attachment, dict) else getattr(attachment, "filename", None)
    return name or "attachment"


def is_media_attachment(attachment: Any) -> bool:
    """Whether this goes to a model as a media part rather than being read as text.

    The single test, because the intake, the child-bot payload builder and the reply
    scanner all ask it: a type one of them forwards and another drops is a file the
    character is told about and never shown.
    """
    mime = attachment_mime(attachment)
    return mime.startswith(("image/", "audio/", "video/")) or mime in DOCUMENT_MIME_TYPES


def is_text_attachment(attachment: Any) -> bool:
    """Whether this is read as text and folded into the turn that carried it."""
    mime = attachment_mime(attachment)
    if mime in DOCUMENT_MIME_TYPES:
        # A PDF is `application/`, never `text/`, but say so rather than rely on that:
        # it goes to the model whole, and reading it as text as well would send it twice.
        return False
    return mime.startswith("text/") or attachment_filename(attachment).lower().endswith(TEXT_ATTACHMENT_EXTENSIONS)


def attachment_tag(attachment: Any) -> str:
    """What a turn says to announce the file it carried: `[Attached Audio: note.ogg]`.

    The kind is read from the mime rather than assumed. Every attachment was announced as
    an Image, so a character sent a voice message was told it had been sent a picture --
    and on a model that could not hear it, that tag was the whole of what it got.
    """
    kind = ATTACHMENT_TAG_KINDS.get(attachment_mime(attachment).split("/", 1)[0], ATTACHMENT_TAG_DEFAULT)
    return ATTACHMENT_TAG.format(kind=kind, filename=attachment_filename(attachment))


def voice_sample_mime_type(content_type: Optional[str], filename: Optional[str]) -> Optional[str]:
    """The audio type an uploaded voice sample is stored and sent as, or None if it is not audio.

    Discord's own label wins; an upload it did not label is judged by its suffix.
    """
    declared = (content_type or "").split(";", 1)[0].strip().lower()
    if declared.startswith("audio/"):
        return declared
    return VOICE_SAMPLE_TYPES.get(os.path.splitext(filename or "")[1].lower())


def describe_voice_samples(summary: Optional[Tuple[int, int, bool]]) -> Optional[str]:
    """A dashboard's line for `ProfileManager.voice_sample_summary`, or None when no slot is filled."""
    filled, slot, selected_filled = summary or (0, 1, False)
    if not filled:
        return None
    return (f"Slot {slot}" + ("" if selected_filled else " (empty)")
            + f" \u00b7 {filled} of {VOICE_SAMPLE_SLOTS} saved")


#: model id -> `image_model_caps` for OpenRouter's image models, installed whole by the image
#: catalogue each time it loads or syncs. A registry rather than a catalogue lookup because
#: the pickers, the request path and /profile manage all ask `image_model_caps`, and a utils
#: function reaching into a service for it would be the wrong direction. Bounded by the image
#: models OpenRouter lists.
_OPENROUTER_IMAGE_CAPS: Dict[str, dict] = {}


def install_openrouter_image_caps(caps: Dict[str, dict]) -> None:
    """Replaces the OpenRouter image caps in one assignment, so no reader sees half a table."""
    global _OPENROUTER_IMAGE_CAPS
    _OPENROUTER_IMAGE_CAPS = dict(caps)


def openrouter_image_ratios() -> Tuple[str, ...]:
    """Every aspect ratio some listed OpenRouter image model takes, in first-seen order."""
    seen: Dict[str, None] = {}
    for caps in _OPENROUTER_IMAGE_CAPS.values():
        for ratio in caps["ratios"]:
            seen.setdefault(ratio, None)
    return tuple(seen)


def image_model_caps(raw_name: Optional[str]) -> dict:
    """What image options `raw_name` will honour: ratios, sizes, quality, thinkingLevel.

    Shared by the picker, which uses it to decide what to offer, and by
    MediaService.resolve_image_output_params, which uses it to decide what to send. If
    those two ever answered differently the dropdown would be offering settings the
    request path then silently dropped.

    An unrecognised Google id -- a model newer than this table, or a typo -- gets the
    conservative default rather than the full set. An OpenRouter id the image catalogue
    does not list gets nothing at all: see OPENROUTER_IMAGE_CAPS_UNKNOWN.
    """
    name = raw_name or ""
    # Case-sensitive, as every routing prefix is -- see APIService._instantiate_model.
    if name.startswith("OPENROUTER/"):
        return _OPENROUTER_IMAGE_CAPS.get(name[len("OPENROUTER/"):], OPENROUTER_IMAGE_CAPS_UNKNOWN)
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

    quality = (cfg.get("image_quality") or "").lower()
    if quality in caps["quality"]:
        out["quality"] = quality

    # Not preferences either, like `modalities`: how this model is asked to encode, and how
    # many references it takes. PNG where the model offers a choice, because the 2K cap and
    # Discord's preview were both reasoned about in PNG; a model with no choice sends its own.
    if "png" in caps["formats"]:
        out["output_format"] = "png"
    if caps["max_refs"]:
        out["max_refs"] = caps["max_refs"]

    # Sampling. Carried through the same per-model filter as everything else even
    # though no image model rejects these outright, so that one call -- and one
    # stored profile -- decides the whole request. Absent stays absent: an image
    # model with no temperature on the wire uses its own, which for the Gemini 3
    # family is the value Google asks you not to move. OpenRouter's Image API takes
    # none of the three, so its models skip this outright.
    if caps["sampling"]:
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


def image_command_prefix(content: Optional[str]) -> Optional[str]:
    """Which IMAGE_COMMAND_PREFIXES prefix `content` opens with, or None for neither."""
    lowered = (content or "").lower()
    return next((p for p in IMAGE_COMMAND_PREFIXES if lowered.startswith(p)), None)


def image_command_prompt(content: Optional[str]) -> Optional[str]:
    """What an image command asks to be drawn, or None when there is nothing to draw.

    Nothing to draw is None rather than "": a bare `!image` is a slip, not a request,
    and an empty prompt is refused by OpenRouter's Image API as a schema error and then
    sent to the fallback model unchanged, so it costs two calls to answer nobody. Every
    entry point asks this one question, so none of them can be the one that forgets.

    A session round may still fold an attached text file in afterwards
    (`_extend_image_prompt`), which is why it asks again after that rather than here.
    """
    prefix = image_command_prefix(content)
    if prefix is None:
        return None
    return (content or "")[len(prefix):].strip() or None


def image_suffix_for_mime(mime_type: Optional[str]) -> str:
    """The suffix a generated image is saved under. PNG when the type is missing or unknown,
    which is what every generated image was before a provider could answer in another."""
    return IMAGE_MIME_SUFFIXES.get(str(mime_type or "").split(";", 1)[0].strip().lower(), ".png")


def generated_image_attachment(path: Optional[str]) -> Tuple[str, str]:
    """(filename, mime type) for a generated image, read back off the suffix it was saved under.

    The suffix is the one place the type survives from the response to the send: a path is
    all that moves between the image workers, the round and the child bots.
    """
    mime_type = IMAGE_SUFFIX_MIMES.get(os.path.splitext(path or "")[1].lower(), "image/png")
    return f"generated_image{IMAGE_MIME_SUFFIXES[mime_type]}", mime_type


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


def resolve_unreadable_media_mode(config: Optional[Dict[str, Any]]) -> str:
    """What this profile does with an attachment none of its models can read.

    Validated rather than trusted, like every other stored mode: an imported profile or
    a shard written before this setting existed can carry anything, and the answer
    decides whether a paid describe pass runs.
    """
    value = str((config or {}).get("unreadable_media_mode") or "").lower()
    return value if value in UNREADABLE_MEDIA_VALUES else UNREADABLE_MEDIA_DEFAULT


def clean_model_name(name: Optional[str]) -> str:
    """A model id as the UI shows it: without the routing prefix that picked its provider.

    One spelling of this, because three places show a model to a user -- the turn's
    trace, the warning under a reply, and the describe line -- and a model named one way
    in one of them and another way in the next reads as two different models.
    """
    return (name or "").replace("models/", "").replace("OPENROUTER/", "").replace("GOOGLE/", "")


def refuse_unreadable_modality(mime_type: str) -> None:
    """Raises the refusal a provider would have raised, for a file an adapter cannot send.

    OpenRouter has no content part for video and Ollama has none for anything but images,
    so for those there is no request to make and no gateway message to read: this has to
    be the refusal. The text carries the phrase `UNREADABLE_MEDIA_KEYS` matches, because
    that is what `unreadable_media_modality` reads to decide the media-free retry, and
    what `_format_api_error` turns into "Unsupported File Format" for the reader. A part
    dropped without one is a file the character is never told it missed -- it keeps the
    `[Attached ...]` tag and answers as though it had read the thing.
    """
    modality = mime_type.split("/", 1)[0]
    # 'image' for anything else, because the reply path knows only these three labels and
    # the recovery is identical for all of them -- `_retry_without_media` strips every
    # attachment whichever is named. Unreachable from the intake, which admits only
    # image, audio, video and the document types an adapter handles by name.
    key = UNREADABLE_MEDIA_KEYS.get(modality, UNREADABLE_MEDIA_KEYS['image'])[0]
    raise Exception(f"No endpoints found that support {key} for {mime_type}")


def unreadable_media_modality(error: Exception) -> Optional[str]:
    """'image', 'audio' or 'video' when `error` is a model refusing to read that, else None.

    Normalised the way `_describe_api_error` normalises before its own table lookup --
    lowercased, with URLs stripped -- because these are the same substrings read for the
    same provider messages, and a refusal the user is told is a vision problem has to be
    the one the retry recognises as droppable.
    """
    text = re.sub(r'https?://[^\s]+', '', str(error)).lower()
    for modality, keys in UNREADABLE_MEDIA_KEYS.items():
        if any(key in text for key in keys):
            return modality
    return None


def split_media_parts(history: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """`history` with every attachment taken out, and the attachments that were in it.

    A copy: the caller's history belongs to a request that has already been made, and a
    retry that mutated it would leave the trace and any later attempt looking at
    something neither of them sent. Only the `parts` lists are rebuilt -- the parts
    themselves are shared, since a media part is read and never written.

    A part is an attachment when it is a dict carrying a `mime_type` (the `{url,
    mime_type}` shape the round builds and the `{mime_type, data}` shape a legacy path
    still hands over) or an object with `inline_data`, which is what every adapter
    tests for when it converts them onto the wire.
    """
    def is_media(part: Any) -> bool:
        if isinstance(part, dict):
            return "mime_type" in part
        return hasattr(part, "inline_data")

    stripped, removed = [], []
    for turn in history:
        parts = turn.get("parts") or []
        kept = [p for p in parts if not is_media(p)]
        removed.extend(p for p in parts if is_media(p))
        stripped.append({**turn, "parts": kept} if len(kept) != len(parts) else turn)
    return stripped, removed


def media_kinds(parts: List[Any]) -> str:
    """How a batch of attachments reads in a warning: "images", "images and audio"."""
    kinds = []
    for part in parts:
        mime = part.get("mime_type", "") if isinstance(part, dict) else getattr(
            getattr(part, "inline_data", None), "mime_type", "")
        label = UNREADABLE_MEDIA_LABELS.get(str(mime).split("/")[0])
        if label and label not in kinds:
            kinds.append(label)
    if not kinds:
        return "attachments"
    return kinds[0] if len(kinds) == 1 else ", ".join(kinds[:-1]) + f" and {kinds[-1]}"


def resolve_openrouter_image_detail(config: Optional[Dict[str, Any]]) -> Optional[str]:
    """The same setting as OpenRouter's per-part `detail` hint, or None.

    OpenRouter carries no request-level media-resolution field; what it forwards is the
    OpenAI-compatible `detail` on an `image_url` part, which has two useful values
    against Google's four. Folding four onto two loses precision, but the alternative
    is a setting that silently does nothing on one of the two providers that can
    actually read images.
    """
    return MEDIA_RESOLUTION_TO_OPENROUTER_DETAIL.get(resolve_media_resolution(config))


def resolve_openrouter_service_tier(config: Optional[Dict[str, Any]]) -> Optional[str]:
    """The profile's OpenRouter service tier, or None for "let OpenRouter route".

    Validated rather than trusted: the value goes straight onto the wire as
    `service_tier`, and an imported profile or a shard written before this existed can
    carry anything at all. "" and absent both resolve to None, which is the tier the
    adapter has always used.
    """
    value = str((config or {}).get("openrouter_service_tier") or "").lower()
    return value if value in OPENROUTER_SERVICE_TIER_VALUES else None


#: What a pinned OpenRouter endpoint tag may look like before it goes on the wire: a
#: host slug and up to four variant segments (`google-vertex/global/priority`,
#: `deepinfra/fp4`). Always used with fullmatch.
OPENROUTER_ENDPOINT_TAG = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}(?:/[a-z0-9][a-z0-9._-]{0,63}){0,4}")


def resolve_openrouter_endpoint(config: Optional[Dict[str, Any]], model_id: Optional[str]) -> Optional[str]:
    """The endpoint this profile pins `model_id` to, or None for "let OpenRouter route".

    Pins live in `openrouter_endpoints`, keyed by model id rather than by slot:
    `_instantiate_model` is handed a model name and a config, never the slot it fills,
    so a pin follows its model into every slot that holds it and stops applying the
    moment a slot moves to another model. Validated like the tier, for the same reason.
    """
    pins = (config or {}).get("openrouter_endpoints")
    if not isinstance(pins, dict) or not isinstance(model_id, str):
        return None
    tag = pins.get(model_id)
    return tag if isinstance(tag, str) and OPENROUTER_ENDPOINT_TAG.fullmatch(tag) else None


def prune_openrouter_endpoints(config: Dict[str, Any]) -> None:
    """Drops pins for models no slot of `config` holds, in place; the key goes when empty.

    An orphaned pin is inert, but without this they would pile up one per model a
    profile was ever pinned on. Matched against every string value rather than a list of
    slot keys, so a slot added later needs nothing here.
    """
    pins = config.get("openrouter_endpoints")
    if not isinstance(pins, dict):
        config.pop("openrouter_endpoints", None)
        return
    held = set()
    for value in config.values():
        if isinstance(value, str):
            held.add(value[len("OPENROUTER/"):] if value.startswith("OPENROUTER/") else value)
    kept = {model: tag for model, tag in pins.items() if model in held}
    if kept:
        config["openrouter_endpoints"] = kept
    else:
        config.pop("openrouter_endpoints", None)


def record_billed_usage(meta: Dict[str, Any], response) -> None:
    """Copy what the provider reports about billing onto a turn's `meta`: its cost,
    served tier and host, and any thinking tokens it counted apart from the reply.

    Written by every path that records a turn, read by `/session audit`. Only
    OpenRouter reports a cost, tier or host: it returns what it actually charged, the one
    figure that survives a flex discount, a `:floor` route or a cached-prompt rebate.
    The rate table `_calculate_turn_cost` reads is keyed on the listed model id and
    knows about none of them, so it is the estimate and this is the invoice.

    Each key stays absent when the provider sent nothing, so the audit can tell a
    billed figure from an estimated one instead of showing 0.00 as if it were free.
    """
    cost = getattr(response, "billed_cost", None)
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        meta["cost"] = float(cost)
    tier = getattr(response, "service_tier", None)
    if tier:
        meta["service_tier"] = str(tier)
    host = getattr(response, "served_by", None)
    if isinstance(host, str) and host:
        meta["served_by"] = host[:64]
    # Google's alone: OpenRouter's completion count already includes reasoning, so a
    # key of its own there would bill the same tokens twice. Sparse, like the rest.
    thinking = getattr(response, "thinking_tokens", None)
    if isinstance(thinking, int) and not isinstance(thinking, bool) and thinking > 0:
        meta["thinking_tokens"] = thinking


def billable_output_tokens(meta: Dict[str, Any]) -> int:
    """The output tokens a turn was billed for: the reply plus thinking counted apart from it.

    The one reading of a turn's output for pricing, so the audit's totals, its per-turn
    estimate and its projection cannot disagree about whether thinking counts.
    """
    total = 0
    for key in ("output_tokens", "thinking_tokens"):
        value = meta.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total += int(value)
    return total


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
    # "rag" is retrieval this side of the call (image_rag_enabled), never a declared tool.
    if mode not in IMAGE_GROUNDING_TOOL_MODES:
        return None
    if not caps["grounding"]:
        return None
    if mode == "web_images" and caps["image_search"]:
        # Nested inside the one google_search tool, not a second tool beside it.
        # `searchTypes` stays camelCase here: _build_tools only maps the outer key.
        return [{"google_search": {"searchTypes": {"webSearch": {}, "imageSearch": {}}}}]
    return [{"google_search": {}}]


def image_rag_enabled(image_config) -> bool:
    """Whether an image is researched before it is drawn: the grounding summariser searches,
    and its visual summary is written into the prompt whichever model then draws.

    Read off `image_grounding_mode`, never the profile's chat `grounding_mode`. The image
    paths used to follow that one, so a profile could not ground its replies without every
    picture also paying for a search and drawing from its summary.
    """
    return (image_config or {}).get("image_grounding_mode") == "rag"


def is_shipped_ltm_prompt(text: str, live_default: str) -> bool:
    """Whether `text` is a default nobody authored, rather than an owner's own prompt.

    True for the wording in force right now, and for any default this project has
    shipped before -- profiles created while the default was seeded carry an encrypted
    copy of whichever one was current that day, and a copy is not an authorship claim.
    Hashes rather than the old strings themselves, so retiring a prompt costs one line
    and never leaves a superseded default sitting in the file to be edited by mistake.
    """
    stripped = (text or "").strip()
    if not stripped:
        return True
    if stripped == (live_default or "").strip():
        return True
    return hashlib.sha256(stripped.encode("utf-8")).hexdigest() in SUPERSEDED_LTM_SUMMARIZATION_HASHES


#: A sentence end: its mark, any closing quote or bracket, then a space or the end of the
#: text -- so "18.5" and "v0.6.1" are not ends.
_SENTENCE_END = re.compile(r'[.!?]+["\'”’)\]]*(?=\s|$)')
#: A full stop after one of these belongs to a name: "Mr. C" is not a sentence end, and
#: taking it for one stored a memory ending "to which Mr.".
_TITLES = frozenset({"mr", "mrs", "ms", "mx", "dr", "prof", "st", "jr", "sr", "vs"})


def clip_to_sentence(text: str, limit: int) -> str:
    """`text` cut to `limit` characters, at a sentence end where there is one.

    A stored memory is embedded whole, so a hard cut mid-clause embeds half a thought
    and matches accordingly. Falls back to the hard cut only when nothing in range ends
    a sentence.
    """
    stripped = (text or "").strip()
    if len(stripped) <= limit:
        return stripped
    end = 0
    for match in _SENTENCE_END.finditer(stripped):
        if match.end() > limit:
            break
        word = stripped[:match.start()].rsplit(None, 1)
        if word and word[-1].lower() not in _TITLES:
            end = match.end()
    # Half the limit, so a single runaway sentence is not cut down to its first clause.
    if end >= limit // 2:
        return stripped[:end]
    return stripped[:limit].rstrip()


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


def system_model(cog, key: str, provider: Optional[str] = None) -> str:
    """A model no profile chooses: the operator's `/mod` override, else the shipped model.

    `provider` is the preference whose chain a SYSTEM_MODEL_DEFAULTS_BY_PROVIDER key is
    read from -- the profile owner's `effective_provider` -- and unset is Google, as an
    unchosen preference is everywhere. SYSTEM_MODEL_DEFAULTS' keys ignore it.

    Asked at the moment of use, never held: an override takes effect on the next call.
    """
    stored = getattr(cog, "system_models", None) or {}
    if key in SYSTEM_MODEL_DEFAULTS:
        return stored.get(key) or SYSTEM_MODEL_DEFAULTS[key]
    side = provider if provider in SYSTEM_MODEL_DEFAULTS_BY_PROVIDER else "gemini"
    return (stored.get(side) or {}).get(key) or SYSTEM_MODEL_DEFAULTS_BY_PROVIDER[side][key]


def is_gateway_shutdown(exc: BaseException) -> bool:
    """Whether this exception is aiohttp's "the gateway went away", not a real fault.

    Every long-lived worker loop has to tell the two apart: on shutdown the session
    closes under whatever was in flight, and a worker that treats that as an error
    prints a traceback per iteration for as long as it takes the process to die. The
    three workers each carried their own copy of this string test, so a fourth added
    without it would look exactly like a crash loop on every restart.
    """
    return isinstance(exc, RuntimeError) and "Session is closed" in str(exc)


def resolve_grounding_mode(config: Optional[Dict[str, Any]]) -> str:
    """One of "off", "rag", "tool", "native" -- the only values the rest of the code sees.

    "tool" is the dashboard's **RAG**: the character calls `search_web` when it wants
    one. "rag" is **Legacy RAG**: a gate model reads the transcript before every round
    and decides. The stored spelling is the older one because renaming it on disk would
    have to walk every profile, so the display names and the stored names differ here
    and only here -- `_grounding_display` in gui_profiles is the other side of it.

    The setting has had three encodings: a bool, then "on"/"on+", then today's
    RAG/NATIVE. Nothing normalises it on write, so all three are still on disk and
    every reader has to fold them in. Doing that inline is what let two vocabularies
    drift apart -- `child_bot_manager` and the image-grounding path tested
    `in ("on", "on+")`, which no profile written since the rename matches, so a
    profile set to RAG silently got no grounding at all on those paths.
    """
    raw = (config or {}).get("grounding_mode", "off")
    if isinstance(raw, bool):
        return "rag" if raw else "off"
    if raw in ("on", "on+"):
        return "rag"
    return raw if raw in ("off", "rag", "tool", "native") else "off"


def grounding_mode_display(config: Optional[Dict[str, Any]]) -> str:
    """A profile's grounding mode as the dashboard shows it, markdown and all.

    One function for every surface, because the stored name and the shown name differ
    for two of the four modes and a second copy of that mapping is how "RAG" came to
    mean two things at once.
    """
    label = GROUNDING_MODE_LABELS.get(resolve_grounding_mode(config), "Off")
    return "`OFF`" if label == "Off" else f"**`{label.upper()}`**"


def resolve_url_mode(config: Optional[Dict[str, Any]]) -> str:
    """One of "off", "rag", "native", reading off the older `url_fetching_enabled`
    flag when `url_mode` is absent entirely."""
    config = config or {}
    if "url_mode" not in config:
        return "rag" if config.get("url_fetching_enabled", False) else "off"
    raw = config.get("url_mode", "off")
    return raw if raw in ("off", "rag", "native") else "off"


def resolve_native_tools(config: Optional[Dict[str, Any]]) -> Optional[List[Dict]]:
    """The `tools` list an instantiated model needs for native grounding/URL context.

    Google-only by construction: the caller decides whether the slot routes to Google
    before asking, because these declarations only exist there.
    """
    tools = []
    if resolve_grounding_mode(config) == "native":
        tools.append({"google_search": {}})
    if resolve_url_mode(config) == "native":
        tools.append({"url_context": {}})
    return tools or None


def provider_takes_functions(*raw_model_names: Optional[str]) -> bool:
    """Whether every model named can carry a function declaration.

    Every one, not any: the prompt describing a function is written once for the turn,
    and the fallback may be the model that answers it.

    Ollama is the only provider excluded, and only because its adapter streams -- a call
    arrives split across chunks with no accumulator written for it. An empty name is
    the "use the slot default" case, which resolves to a Google model.
    """
    return not any((name or "").upper().startswith("OLLAMA/") for name in raw_model_names)


def ltm_auto_recall_enabled(config: Optional[Dict[str, Any]]) -> bool:
    """Whether the automatic long-term memory pass runs for this profile at all.

    Absent means **on**, and that is the whole point of reading it through a function.
    Recall has been unconditional for the life of the setting, so every profile written
    before the toggle existed has no key -- and defaulting those to off would make a
    character that has been remembering things for months quietly stop, with nothing
    on screen saying why. New profiles are created with an explicit False instead, so
    "off by default" is a property of the template rather than of absence.

    The `recall` tool is a separate setting and is not consulted here: pull-only
    retrieval -- the automatic pass off, Memory Search on -- is a legitimate and much
    cheaper shape, and folding the two together would make it unreachable.
    """
    raw = (config or {}).get("ltm_recall_enabled")
    return True if raw is None else bool(raw)


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
