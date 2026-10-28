"""A character's fixed examples: how it speaks, handed to it with every reply.

Stored as `prompts["fixed_examples"]`, a list of `{"id", "said", "reply"}`, beside the
persona as the lorebook is, so exports, clones and a borrow's source carry them with no
shard of their own. Training examples (`training.json.gz`) are matched to each turn by
embedding what was said; these are never embedded, so `said` may be empty -- a reply on
its own is a sample of the voice. Always sent, they ride the system instruction
(`prompt_builder`) and are cached with it.

Every write goes through `normalise`; a read trusts the shape it wrote (`entries`).
"""
import uuid
from typing import Any, Dict, Iterable, List

from .constants import EXAMPLE_REPLY_MAX_CHARS, EXAMPLE_SAID_MAX_CHARS, LIMIT_FIXED_EXAMPLES
from .helpers import clean_imported_text


def new_id() -> str:
    return "X" + uuid.uuid4().hex[:8].upper()


def normalise(raw: Any) -> List[Dict[str, str]]:
    """The examples worth storing out of anything: each side cleaned and cut to the
    editor's caps, any with no reply dropped, ids kept when unique and minted when not, at
    most LIMIT_FIXED_EXAMPLES."""
    out: List[Dict[str, str]] = []
    seen = set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        reply = clean_imported_text(item.get("reply"), EXAMPLE_REPLY_MAX_CHARS).strip()
        if not reply:
            continue
        entry_id = item.get("id")
        if not isinstance(entry_id, str) or not entry_id or len(entry_id) > 16 or entry_id in seen:
            entry_id = new_id()
        seen.add(entry_id)
        out.append({"id": entry_id,
                    "said": clean_imported_text(item.get("said"), EXAMPLE_SAID_MAX_CHARS).strip(),
                    "reply": reply})
        if len(out) >= LIMIT_FIXED_EXAMPLES:
            break
    return out


def entries(raw: Any) -> List[Dict[str, str]]:
    """Stored fixed examples, read: `normalise` wrote them, so only the shape is checked."""
    return [e for e in raw if isinstance(e, dict) and isinstance(e.get("reply"), str)
            and isinstance(e.get("said", ""), str)] if isinstance(raw, list) else []


def block(said: str, reply: str) -> str:
    """One example as the character reads it, a training example's included."""
    lines = ([f"User: {said}"] if said else []) + [f"You: {reply}"]
    return "<example>\n" + "\n".join(lines) + "\n</example>"


def render(examples: Iterable[Dict[str, str]]) -> str:
    """The `{examples_block}` of TRAINING_DATA_INJECTION, as training examples fill it."""
    return "\n---\n".join(block(e.get("said", ""), e["reply"]) for e in examples)
