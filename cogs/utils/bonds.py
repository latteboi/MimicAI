"""Bonds: what a character feels about each person it talks to, beside the neuro engine.

Two scores per person, affection and resentment, 0-100 and independent. Stored in the
profile's own shard (`bonds.json.gz`) as `{ID: [affection, resentment]}`, keyed by the ID
that person's turns carry in the transcript -- a user's hash, a profile's PID -- so the
model names someone the way a reply names a turn, and nothing is matched by name.
Insertion order is recency: an update re-inserts, and past LIMIT_BONDS the oldest goes.
"""
import re
from typing import Any, Dict, Iterable, List, Tuple

from .constants import BOND_BANDS, BOND_FIRST_MEETING, BOND_ROSTER_SIZE, BOND_STEP_MAX, LIMIT_BONDS

#: The report, as DEFAULT_BOND_INSTRUCTION asks for it, or cut off at the end of a reply
#: that ran out of tokens. One pattern for the parser and the scrub, as the neuro engine's.
BOND_TAG_PATTERN = re.compile(r'<bond_update>\s*(.*?)\s*(?:</bond_update>|\Z)',
                              re.IGNORECASE | re.DOTALL)
#: An ID as the transcript writes it. Case-sensitive: sixteen capitals and digits, which no
#: word a model writes beside it is.
_ID = re.compile(r'\b[A-Z0-9]{16}\b')
_AXIS = re.compile(r'\b(aff(?:ection)?|res(?:entment)?)\s*[:=]?\s*(\d{1,3})\b', re.IGNORECASE)


def take_updates(text: str) -> Tuple[str, Dict[str, Dict[str, int]]]:
    """The reply without any `<bond_update>`, and what it reported: ID -> {"a"/"r": level}.

    Always removed, honoured or not. Each ID owns the levels written after it, up to the
    next; an axis left out did not move. At most BOND_ROSTER_SIZE people, the number a
    reply is shown.

    ponytail: any well-formed ID is taken, not only those the reply was shown -- checking
    meant handing the list through every reply path. A stray one is an entry that ages out.
    """
    updates: Dict[str, Dict[str, int]] = {}
    for body in BOND_TAG_PATTERN.findall(text):
        ids = list(_ID.finditer(body))
        for m, nxt in zip(ids, ids[1:] + [None]):
            chunk = body[m.end():nxt.start() if nxt else len(body)]
            axes = {key[0].lower(): min(100, int(value)) for key, value in _AXIS.findall(chunk)}
            if axes and (m.group() in updates or len(updates) < BOND_ROSTER_SIZE):
                updates.setdefault(m.group(), {}).update(axes)
    clean = BOND_TAG_PATTERN.sub('', text)
    return (clean.strip() if clean != text else text), updates


def apply_updates(store: Dict[str, Any], updates: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    """`updates` applied to `store`, in place: a first impression as reported, a known
    person moved at most BOND_STEP_MAX an axis, each re-inserted as the newest, and the
    oldest past LIMIT_BONDS dropped."""
    for bond_id, axes in updates.items():
        old = store.pop(bond_id, None)
        if isinstance(old, list) and len(old) == 2:
            store[bond_id] = [max(was - BOND_STEP_MAX, min(was + BOND_STEP_MAX, axes.get(key, was)))
                              for was, key in zip(old, "ar")]
        else:
            store[bond_id] = [axes.get(key, start) for key, start in zip("ar", BOND_FIRST_MEETING)]
    for stale in list(store)[:max(0, len(store) - LIMIT_BONDS)]:
        del store[stale]
    return store


def render(people: Iterable[Tuple[str, str]], store: Dict[str, Any]) -> str:
    """`<bonds>`'s lines: each (ID, name) with where the character stands with them."""
    lines: List[str] = []
    for bond_id, name in people:
        held = store.get(bond_id)
        state = (f"AFF {held[0]} | RES {held[1]}" if isinstance(held, list) and len(held) == 2
                 else "first meeting: set both")
        lines.append(f"{name} [ID: {bond_id}]: {state}")
    return "\n".join(lines)


def describe(held: Any, numbers: bool = False) -> str:
    """Where a character stands with one person, in BOND_BANDS' words, for that person to
    read: "Affection: **warm** · Resentment: **friendly**". `numbers` adds each level."""
    if not (isinstance(held, list) and len(held) == 2):
        return "Not met yet: the first conversation sets both."
    return " · ".join(
        f"{axis}: **{next(band[col] for band in BOND_BANDS if level <= band[0])}**"
        + (f" (`{level}`)" if numbers else "")
        for axis, level, col in (("Affection", held[0], 1), ("Resentment", held[1], 2)))
