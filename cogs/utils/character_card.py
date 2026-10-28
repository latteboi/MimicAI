"""Character cards (SillyTavern, Chub, RisuAI; spec V1 to V3), read for `/import`.

A card is text someone else wrote, and it becomes only what its importer could have typed:
persona text, fixed examples and a lorebook, each through `clean_imported_text` at the
editor's own caps. Nothing in a card sets a config key or a model, nothing in it is a URL
the bot fetches (a V3 card's `assets` are ignored, bar the main icon a CHARX carries inside
itself), and the only rating it carries is the strictest: a card tagged NSFW is declared
18+ on save. `system_prompt` and `post_history_instructions` are dropped -- they are
written for another engine's prompt layout, and are where a card's jailbreak lives.

The PNG is walked chunk by chunk and its image is never decoded or copied: only the
`tEXt` chunk holding the card is read, so a file built to explode when decoded costs the
walk alone. Compressed text chunks are not read at all, so there is nothing to inflate.
A CHARX (V3's ZIP: `card.json` at the root, its assets beside it) is opened only once its
central directory is a card's size, and each entry is read to a cap, never to its stated size.

Pure functions over bytes, run off the event loop by the caller.
"""
import base64
import binascii
import io
import lzma
import re
import zipfile
import zlib
from typing import Any, Callable, Dict, List, Optional, Tuple

import orjson

from . import fixed_examples, lorebook
from .constants import (CARD_TEXT_MAX_BYTES, LIMIT_FIXED_EXAMPLES, PERSONA_TEXT_INPUT_MAX_LENGTH,
                        LORE_ENTRY_MAX_CHARS)
from .helpers import clean_imported_text

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
#: A 25 MB image in 8 KB IDAT chunks is ~3,200; past this the file is not a card.
_MAX_CHUNKS = 8192
#: The newest spec first: a V3 card carries a V2 copy under `chara` for older readers.
_KEYWORDS = (b"ccv3", b"chara")
#: Base64 of the cap, plus padding.
_MAX_ENCODED = CARD_TEXT_MAX_BYTES * 4 // 3 + 4

ZIP_SIGNATURE = b"PK\x03\x04"
#: Bytes of central directory a CHARX may have. Opening a ZIP reads it whole and builds a
#: ZipInfo of ~480 bytes for every 46-byte entry: a 25 MB file of empty entries kept ~150 MB
#: and held the GIL for seconds. This is ~5,000 entries at most; a card with every sprite
#: lists a few hundred.
_MAX_CHARX_DIRECTORY_BYTES = 256 * 1024
#: What a damaged or hostile archive raises from zipfile, by codec. RuntimeError covers an
#: encrypted entry and an unknown compression method.
_ZIP_ERRORS = (zipfile.BadZipFile, RuntimeError, EOFError, OSError, ValueError,
               zlib.error, lzma.LZMAError)

#: What a description may bring to the model that sorts it: room to fill every persona
#: section, and no more.
DESCRIPTION_MAX_CHARS = 3 * PERSONA_TEXT_INPUT_MAX_LENGTH

_CHAR_MACRO = re.compile(r"\{\{\s*char\s*\}\}|<bot>", re.IGNORECASE)
_USER_MACRO = re.compile(r"\{\{\s*user\s*\}\}|<user>", re.IGNORECASE)
_EXAMPLE_START = re.compile(r"<start>", re.IGNORECASE)
_EXAMPLE_SPEAKER = re.compile(r"\s*(?:\{\{\s*(user|char)\s*\}\}|<(user|bot)>)\s*:\s*", re.IGNORECASE)
_NSFW_TAGS = {"nsfw", "nsfl"}


LOREBOOK_NOT_CARD = ("This is a lorebook, not a character card. To add it to one of your characters, "
                     "run `/import` again with that character as `profile_name`.")


class CardError(ValueError):
    """The file is not a card this bot reads. The message is written for the user."""


def _png_card_text(data: bytes) -> bytes:
    """The base64 card a PNG's `ccv3` or `chara` tEXt chunk holds."""
    found: Dict[bytes, bytes] = {}
    pos, chunks = len(PNG_SIGNATURE), 0
    while pos + 12 <= len(data):
        chunks += 1
        if chunks > _MAX_CHUNKS:
            raise CardError("This PNG has too many chunks to be a character card.")
        length = int.from_bytes(data[pos:pos + 4], "big")
        kind = data[pos + 4:pos + 8]
        end = pos + 12 + length
        if end > len(data):
            raise CardError("This PNG is damaged: a chunk runs past the end of the file.")
        if kind == b"tEXt":
            # The keyword is at most 79 bytes, so it is read before the chunk is copied.
            keyword = data[pos + 8:pos + 8 + min(length, 80)].partition(b"\0")[0]
            if keyword in _KEYWORDS and keyword not in found:
                if length > _MAX_ENCODED:
                    raise CardError("This card holds more text than any character needs.")
                found[keyword] = data[pos + 8 + len(keyword) + 1:pos + 8 + length]
        elif kind == b"IEND":
            break
        pos = end
    encoded = next((found[k] for k in _KEYWORDS if k in found), None)
    if encoded is None:
        raise CardError("This PNG holds no character card.")
    try:
        return base64.b64decode(encoded)
    except (binascii.Error, ValueError):
        raise CardError("This card's text is damaged.")


def _open_charx(data: bytes) -> zipfile.ZipFile:
    """A CHARX opened, once its central directory is known to be a card's size."""
    fp = io.BytesIO(data)
    try:
        # ZipFile's own reading of the end record, ZIP64 included, so the size checked is the
        # size it then reads. Private: a rename fails loudly in tests/test_character_cards.py.
        end = zipfile._EndRecData(fp)
    except (OSError, zipfile.BadZipFile):
        end = None
    if not end:
        raise CardError("This CHARX is damaged.")
    if end[zipfile._ECD_SIZE] > _MAX_CHARX_DIRECTORY_BYTES:
        raise CardError("This CHARX lists more files than a character card holds.")
    try:
        return zipfile.ZipFile(fp)
    except _ZIP_ERRORS:
        raise CardError("This CHARX is damaged.")


def _charx_member(archive: zipfile.ZipFile, name: str, n: int) -> bytes:
    """At most `n` bytes of `name`; KeyError when there is none. An entry's stated size is
    the archive's claim, so the read is what bounds it, inflated or not."""
    try:
        with archive.open(name) as f:
            return f.read(n)
    except _ZIP_ERRORS:
        raise CardError("This CHARX is damaged.")


def _charx_card_text(data: bytes) -> bytes:
    """A CHARX's `card.json`, read one byte past the cap so that the cap refuses it."""
    with _open_charx(data) as archive:
        try:
            return _charx_member(archive, "card.json", CARD_TEXT_MAX_BYTES + 1)
        except KeyError:
            raise CardError("This CHARX holds no card.json.")


def _image_suffix(head: bytes) -> Optional[str]:
    """The suffix of an image Discord shows, by its first bytes: a name is only a claim."""
    if head.startswith(PNG_SIGNATURE):
        return ".png"
    if head.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return ".webp"
    return None


def card_portrait(data: bytes, max_bytes: int) -> Tuple[Optional[bytes], str]:
    """The picture a card brings for its avatar and its suffix, or (None, ""). A PNG card is
    its own; a CHARX's is the main icon it lists as inside the file (`embeded://`, the
    spec's spelling), and at most `max_bytes` + 1 of that is read, so a caller testing
    `len(...) > max_bytes` refuses one too large without it ever being read whole."""
    if data.startswith(PNG_SIGNATURE):
        return data, ".png"
    if not data.startswith(ZIP_SIGNATURE):
        return None, ""
    assets = _card_object(data, "").get("assets")
    icons = [a for a in (assets if isinstance(assets, list) else [])
             if isinstance(a, dict) and a.get("type") == "icon"]
    icon = next((a for a in icons if a.get("name") == "main"), icons[0] if icons else {})
    scheme, _, path = str(icon.get("uri") or "").partition("://")
    if scheme not in ("embeded", "embedded"):
        return None, ""
    with _open_charx(data) as archive:
        try:
            image = _charx_member(archive, path, max_bytes + 1)
        except KeyError:
            return None, ""
    suffix = _image_suffix(image)
    return (image, suffix) if suffix else (None, "")


def _macros(text: str, name: str) -> str:
    return _USER_MACRO.sub("the user", _CHAR_MACRO.sub(lambda _m: name, text))


def _text(data: Dict[str, Any], key: str, name: str, limit: int) -> str:
    value = data.get(key)
    return clean_imported_text(_macros(value, name), limit) if isinstance(value, str) else ""


def _examples(raw: Any, name: str) -> List[Dict[str, str]]:
    """`mes_example` as fixed examples, which SillyTavern sends whole on every reply too:
    each `{{char}}:` line with the `{{user}}:` line straight before it as what was said,
    or nothing -- a character line on its own is a sample of the voice. Lines with no
    speaker continue the one before."""
    if not isinstance(raw, str):
        return []
    pairs = []
    for block in _EXAMPLE_START.split(raw[:DESCRIPTION_MAX_CHARS]):
        turns: List[List[str]] = []
        for line in block.splitlines():
            m = _EXAMPLE_SPEAKER.match(line)
            if m:
                who = (m.group(1) or m.group(2)).lower()
                turns.append(["user" if who == "user" else "char", line[m.end():]])
            elif turns:
                turns[-1][1] += "\n" + line
        for before, (who, answer) in zip([None] + turns, turns):
            if who == "char":
                said = before[1] if before and before[0] == "user" else ""
                pairs.append({"said": _macros(said, name), "reply": _macros(answer, name)})
        if len(pairs) >= LIMIT_FIXED_EXAMPLES:
            break
    return fixed_examples.normalise(pairs)


def _lore(book: Any, name: str) -> List[Dict[str, Any]]:
    """A card's `character_book`, or a SillyTavern World Info file, as lorebook entries.
    World Info keeps its entries in a dict, under `key` and `disable`. Disabled entries
    are dropped, a `constant` one keeps no keys (always in mind), and a keyed one with no
    keys left would never trigger, so it is dropped too. Only primary keys: secondary keys
    narrow a match this lorebook does not narrow."""
    raw = book.get("entries") if isinstance(book, dict) else None
    out = []
    for e in raw.values() if isinstance(raw, dict) else raw if isinstance(raw, list) else []:
        if not isinstance(e, dict) or e.get("enabled") is False or e.get("disable") is True:
            continue
        content = e.get("content")
        if not isinstance(content, str):
            continue
        keys = [] if e.get("constant") else [_macros(k, name) for k in e.get("keys") or e.get("key") or []
                                             if isinstance(k, str)]
        if not e.get("constant") and not lorebook.parse_keys(keys):
            continue
        out.append({"keys": keys, "content": _macros(content[:LORE_ENTRY_MAX_CHARS], name)})
    return lorebook.normalise(out)


def _card_object(data: bytes, filename: str) -> Dict[str, Any]:
    """The JSON object a card PNG, CHARX or JSON file holds: V2 and V3 nest the character
    under `data`, V1 is the character itself, and so is a lorebook file its book."""
    if data.startswith(PNG_SIGNATURE):
        raw = _png_card_text(data)
    elif data.startswith(ZIP_SIGNATURE):
        raw = _charx_card_text(data)
    elif filename.lower().endswith(".json"):
        raw = data
    else:
        raise CardError("That is not a character card: a card PNG, a CHARX, or a card's JSON.")
    if len(raw) > CARD_TEXT_MAX_BYTES:
        raise CardError("This card holds more text than any character needs.")
    try:
        card = orjson.loads(raw)
    except orjson.JSONDecodeError:
        raise CardError("This card's text is not readable JSON.")
    if not isinstance(card, dict):
        raise CardError("This card's text is not a character.")
    nested = card.get("data")
    return nested if isinstance(nested, dict) else card


def read_card(data: bytes, filename: str = "") -> Dict[str, Any]:
    """A card's usable fields, cleaned. Raises CardError for anything else."""
    data_obj = _card_object(data, filename)
    name = " ".join(clean_imported_text(data_obj.get("name"), 100).split())
    if not name and "entries" in data_obj:
        raise CardError(LOREBOOK_NOT_CARD)
    if not name:
        raise CardError("This card has no character name.")
    tags = data_obj.get("tags") if isinstance(data_obj.get("tags"), list) else []
    return {
        "name": name,
        "description": _text(data_obj, "description", name, DESCRIPTION_MAX_CHARS),
        "personality": _text(data_obj, "personality", name, PERSONA_TEXT_INPUT_MAX_LENGTH),
        "scenario": _text(data_obj, "scenario", name, PERSONA_TEXT_INPUT_MAX_LENGTH),
        "examples": _examples(data_obj.get("mes_example"), name),
        "lore": _lore(data_obj.get("character_book"), name),
        "adult": any(isinstance(t, str) and t.strip().casefold() in _NSFW_TAGS for t in tags),
    }


def read_lorebook(data: bytes, filename: str, name: str) -> List[Dict[str, Any]]:
    """A lorebook file's entries (SillyTavern World Info, or a book on its own), or the
    book inside a card, for `/import profile_name:`. `{{char}}` becomes `name`."""
    data_obj = _card_object(data, filename)
    book = _lore(data_obj.get("character_book", data_obj), name)
    if not book:
        raise CardError("This file holds no lorebook entries.")
    return book


def prompt_text(card: Dict[str, Any]) -> str:
    """The card as the sorting model reads it. Its tag is in SYSTEM_XML_TAGS, which
    `clean_imported_text` strips from the fields, so no field can close it."""
    parts = [f"name: {card['name']}"] + [f"{key}:\n{card[key]}" for key in
                                         ("description", "personality", "scenario") if card[key]]
    return "<character_card>\n" + "\n\n".join(parts) + "\n</character_card>"


def profile_name_for(name: str, is_valid: Callable[[str], Tuple[bool, str]]) -> str:
    """An internal profile name from a card's character name: lower case, letters, digits,
    `_` and `-`, at most 20. A reserved word gets `-card` on the end."""
    slug = re.sub(r"[^a-z0-9_-]+", "-", name.lower()).strip("-_")[:20].rstrip("-_") or "character"
    return slug if is_valid(slug)[0] else f"{slug[:15]}-card"
