"""Follow-ups a character chose to set: `<follow_up for="ID" in="40h">note</follow_up>` at the
end of a reply, to come back to someone once that long has passed: when they are next back,
or by itself once the channel has had a minute's quiet, Proactivity on or off (`fire`).

A session switch, `proactivity["follow_ups"]`, off by default. The log is the only record:
a follow-up is `meta["follow_up"]` on the turn that set it, and the turn that raised it
records the setter's `turn_id` as `meta["follow_up_answered"]`. Nothing is edited in place,
so regenerating, deleting, branching, `/purge` and the log trim need no code of their own,
and what is pending is read back off the log each time (it holds at most a few hundred turns).

People are known by the ID in their turn header (`_get_user_hash`), read off the header
itself, never by a Discord ID: that is the ID a character sees and writes back.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from .constants import (DEFAULT_FOLLOW_UP_DUE, DEFAULT_FOLLOW_UP_RULES, FOLLOW_UP_EXPIRY_HOURS,
                        FOLLOW_UP_MAX_HOURS, FOLLOW_UP_NOTE_MAX, FOLLOW_UP_PER_CHARACTER)
from .helpers import turn_posted_at

TAG = re.compile(r'<follow_up\b([^>]*)>(.*?)</follow_up>', re.IGNORECASE | re.DOTALL)
_ATTR = re.compile(r'(\w+)\s*=\s*["\']([^"\']*)["\']')
# `s` is not in the rules but is read, floored to a minute: asked for "30 seconds", models
# wrote in="30s", and a tag that failed to parse was a promise silently dropped.
_HOURS = re.compile(r'^\s*(\d+(?:\.\d+)?)\s*([smhd])\s*$', re.IGNORECASE)
_UNIT_HOURS = {"s": 1 / 3600, "m": 1 / 60, "h": 1, "d": 24}
_HEADER = re.compile(r'<([^>\r\n]+)> \[ID: ([^\]\r\n]+)\]')

Pending = Tuple[str, Dict[str, Any]]


def enabled(session: Optional[Dict[str, Any]]) -> bool:
    return bool(((session or {}).get("proactivity") or {}).get("follow_ups"))


def take(raw_text: str) -> Tuple[str, Optional[Dict[str, str]]]:
    """The reply without any follow-up tag, and the first one's attributes and note.

    Always removed, honoured or not, like `take_memory_flag`: the tag is a protocol marker
    and must never reach the channel or the log, whichever path the reply came down.
    """
    match = TAG.search(raw_text)
    if not match:
        return raw_text, None
    found = {k.lower(): v for k, v in _ATTR.findall(match.group(1))}
    found["note"] = match.group(2).strip()
    return TAG.sub('', raw_text).strip(), found


def _person(turn: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """(header ID, name) of a person's turn, or None for anything else. A typed turn is a
    whisper or a note, which nobody but its reader saw."""
    if not turn.get("is_user") or turn.get("type"):
        return None
    match = _HEADER.search(turn.get("content") or "")
    return (match.group(2).strip().upper(), match.group(1)) if match else None


def people(log: List[Dict[str, Any]]) -> Dict[str, str]:
    """{header ID: name} for everyone who has spoken in the log, their latest name winning."""
    out: Dict[str, str] = {}
    for turn in log:
        person = _person(turn)
        if person:
            out[person[0]] = person[1]
    return out


def resolve(found: Optional[Dict[str, str]], log: List[Dict[str, Any]], now: float) -> Optional[Dict[str, Any]]:
    """The record to store for a tag, or None when it names nobody here, no time or no note.

    `for=` is the header ID, or a name only one person here has: models wrote a name a few
    times in testing. `in=` is "not before", cut to FOLLOW_UP_MAX_HOURS.
    """
    if not found:
        return None
    known = people(log)
    who = (found.get("for") or "").strip()
    target = who.upper() if who.upper() in known else None
    if target is None:
        named = [pid for pid, name in known.items() if name.strip().lower() == who.lower()]
        target = named[0] if len(named) == 1 else None
    hours = _HOURS.match(found.get("in") or "")
    note = (found.get("note") or "").strip()[:FOLLOW_UP_NOTE_MAX]
    if not (target and hours and note):
        return None
    # A minute at least: the rules' floor, and the quiet a fire waits for anyway.
    span = max(float(hours.group(1)) * _UNIT_HOURS[hours.group(2).lower()], 1 / 60)
    # The name is a snapshot for `/session audit`, which has the turn and not the log.
    return {"for": target, "name": known[target], "due": int(now + min(span, FOLLOW_UP_MAX_HOURS) * 3600),
            "note": note}


def pending(log: List[Dict[str, Any]], pid: str, owner_id: int, now: float) -> List[Pending]:
    """This character's follow-ups still live, as (setter's turn_id, record), oldest due first.

    Walked newest first, so every later turn's `follow_up_answered` is seen before the turn
    it answers. A newer one for the same person replaces theirs, answered or not; past
    FOLLOW_UP_PER_CHARACTER the oldest go; one unanswered FOLLOW_UP_EXPIRY_HOURS past due is gone.
    """
    answered, seen, live = set(), set(), []
    for turn in reversed(log):
        meta = turn.get("meta") or {}
        if meta.get("follow_up_answered"):
            answered.add(meta["follow_up_answered"])
        record = meta.get("follow_up")
        if (not isinstance(record, dict) or turn.get("is_user") or turn.get("speaker_pid") != pid
                or turn.get("owner_id") != owner_id or record.get("for") in seen):
            continue
        seen.add(record.get("for"))
        if turn.get("turn_id") in answered or now > record.get("due", 0) + FOLLOW_UP_EXPIRY_HOURS * 3600:
            continue
        live.append((turn.get("turn_id"), record))
        if len(live) >= FOLLOW_UP_PER_CHARACTER:
            break
    return sorted(live, key=lambda item: item[1].get("due", 0))


def due(log: List[Dict[str, Any]], pid: str, owner_id: int, now: float) -> Optional[Pending]:
    """The oldest follow-up of this character's that is due and whose person has spoken
    since it fell due, or None. "They are back" is what the rules promise the character."""
    ready = [item for item in pending(log, pid, owner_id, now) if item[1].get("due", 0) <= now]
    if not ready:
        return None
    last_spoke: Dict[str, float] = {}
    for turn in reversed(log):
        person = _person(turn)
        if person and person[0] not in last_spoke:
            at = turn_posted_at(turn)
            if at:
                last_spoke[person[0]] = at.timestamp()
    return next((item for item in ready if last_spoke.get(item[1].get("for", ""), 0) > item[1].get("due", 0)), None)


def fire(log: List[Dict[str, Any]], seats: List[Tuple[Dict[str, Any], str]],
         now: float) -> Tuple[Optional[Tuple[Dict[str, Any], Pending]], Optional[int]]:
    """For a proactive round: (seat, follow-up) for the oldest due across `seats`, its person
    back or not, else None; and the due time of the next one after it, else None.

    `seats` is (seat, pid) for each. The due time is the session's hint,
    `proactivity["follow_up_at"]`: a quiet session's log is not in memory, so the roll reads
    that and not this. The one fired is left out of it, so a turn that fails to raise it
    waits for its person to speak rather than firing again every cooldown.
    """
    items = sorted(((seat, item) for seat, pid in seats for item in pending(log, pid, seat["owner_id"], now)),
                   key=lambda s: s[1][1].get("due", 0))
    first = items[0] if items and items[0][1][1].get("due", 0) <= now else None
    rest = items[1:] if first else items
    return first, (rest[0][1][1].get("due") if rest else None)


def note_set(session: Dict[str, Any], record: Dict[str, Any]) -> None:
    """Moves the session's hint (`fire`) up to a newly set follow-up's due time."""
    pro = session.setdefault("proactivity", {})
    pro["follow_up_at"] = min(pro.get("follow_up_at") or record["due"], record["due"])


def prompt_block(global_prompts: Dict[str, str], raised: Optional[Pending],
                 log: List[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
    """(the rules, for the system instruction; the reminder for `raised`, for the turn
    context, or None)."""
    rules = global_prompts.get("FOLLOW_UP_RULES", DEFAULT_FOLLOW_UP_RULES)
    if not raised:
        return rules, None
    record = raised[1]
    return rules, global_prompts.get("FOLLOW_UP_DUE", DEFAULT_FOLLOW_UP_DUE).format(
        name=people(log).get(record.get("for", ""), record.get("name") or "them"), id=record.get("for", ""),
        note=record.get("note", ""))
