"""A character's lorebook: what it knows, handed to it when a keyword comes up.

Stored as `prompts["lorebook"]`, a list of `{"id", "keys", "content"}`, beside the persona,
so exports, clones and a borrow's source carry it with no shard of its own. An entry with
no keys is always in mind and rides the system instruction; one with keys is matched
against the last few public turns and rides the turn (`prompt_builder`), where it costs
no cache.

Keys are literal text, matched whole-word and case-insensitively -- never a regex. A
card's book can carry `/pattern/` keys, and compiling a stranger's pattern on a 0.25 vCPU
box invites one that never finishes.

Every write goes through `normalise`; a read trusts the shape it wrote (`entries`).
"""
import uuid
from typing import Any, Dict, Iterable, List

from .constants import LIMIT_LORE_ENTRIES, LORE_ENTRY_MAX_CHARS, LORE_KEYS_MAX_CHARS, LORE_PER_TURN
from .helpers import clean_imported_text


def new_id() -> str:
    return "L" + uuid.uuid4().hex[:8].upper()


def parse_keys(value: Any) -> List[str]:
    """Keys from a list or a comma-separated string: cleaned, deduplicated ignoring case,
    and as many as fit in LORE_KEYS_MAX_CHARS together."""
    raw = value.split(",") if isinstance(value, str) else value if isinstance(value, list) else []
    keys, seen, total = [], set(), 0
    for item in raw:
        key = " ".join(clean_imported_text(item, LORE_KEYS_MAX_CHARS).split())
        if not key or key.casefold() in seen:
            continue
        total += len(key) + 2
        if total > LORE_KEYS_MAX_CHARS:
            break
        seen.add(key.casefold())
        keys.append(key)
    return keys


def normalise(raw: Any) -> List[Dict[str, Any]]:
    """The entries worth storing out of anything: each one cleaned and cut to the editor's
    limits, empty ones dropped, ids kept when unique and minted when not, at most
    LIMIT_LORE_ENTRIES."""
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        content = clean_imported_text(item.get("content"), LORE_ENTRY_MAX_CHARS)
        if not content:
            continue
        entry_id = item.get("id")
        if not isinstance(entry_id, str) or not entry_id or len(entry_id) > 16 or entry_id in seen:
            entry_id = new_id()
        seen.add(entry_id)
        out.append({"id": entry_id, "keys": parse_keys(item.get("keys")), "content": content})
        if len(out) >= LIMIT_LORE_ENTRIES:
            break
    return out


def entries(raw: Any) -> List[Dict[str, Any]]:
    """A stored lorebook, read: `normalise` wrote it, so only the shape is checked."""
    return [e for e in raw if isinstance(e, dict) and isinstance(e.get("content"), str)
            and isinstance(e.get("keys", []), list)] if isinstance(raw, list) else []


def constant(book: Iterable[Dict[str, Any]]) -> List[str]:
    """The always-in-mind entries: those with no keys."""
    return [e["content"] for e in book if not e.get("keys")]


def _word_char(c: str) -> bool:
    return c.isalnum() or c == "_"


def _has_word(text: str, key: str) -> bool:
    """`key` in `text` with no word character either side. Both already casefolded."""
    start = text.find(key)
    while start != -1:
        end = start + len(key)
        if ((start == 0 or not _word_char(text[start - 1]))
                and (end == len(text) or not _word_char(text[end]))):
            return True
        start = text.find(key, start + 1)
    return False


def triggered(book: Iterable[Dict[str, Any]], text: str, limit: int = LORE_PER_TURN) -> List[str]:
    """The keyed entries one of whose keys `text` mentions, in stored order, at most `limit`."""
    folded = (text or "").casefold()
    if not folded:
        return []
    hits = []
    for e in book:
        if any(_has_word(folded, k.casefold()) for k in e.get("keys") or [] if k):
            hits.append(e["content"])
            if len(hits) >= limit:
                break
    return hits


def render(contents: List[str]) -> str:
    return "<lorebook>\n" + "\n\n".join(contents) + "\n</lorebook>"
