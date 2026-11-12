"""Message signatures: an invisible token at the end of every message a profile speaks,
naming the account that ran the character and the profile it was. MESSAGE_SIGNATURES.md
has the design and why.

One AES-256 block, its key HKDF'd from the master key per server, written two bits a
character in four invisible operators: 64 characters, plus the space before them.
Trust comes from who posted the message, never from the token: a wrong key reads as a
plausible answer about 2% of the time, which the trace settles by asking Discord whether
the user ID is an account (`gui_mod.trace_signature`).
"""

import re
import time
from typing import Optional, Tuple

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .constants import PID_CLASS_PREFIXES, SIGNATURE_ROOM

#: U+2061 to U+2064. Not U+200B (sent as a spacer, and models pad with it) and not
#: U+200C to U+200F, which real writing needs (`helpers._KEPT_FORMAT_CHARS`).
ALPHABET = "⁡⁢⁣⁤"
SIG_LEN = 64
assert SIGNATURE_ROOM == SIG_LEN + 1

_BYTE = ["".join(ALPHABET[(b >> s) & 3] for s in (6, 4, 2, 0)) for b in range(256)]
_VALUE = {c: i for i, c in enumerate(ALPHABET)}
_STRIP = str.maketrans("", "", ALPHABET)
_TRAILING = re.compile(f"[{ALPHABET}]{{{SIG_LEN}}}\\Z")
_SIGNED_TAIL = re.compile(f" ?[{ALPHABET}]{{{SIG_LEN}}}\\Z")
_CLASSES = "AXBC"
assert set(_CLASSES) == set(PID_CLASS_PREFIXES)
_VERSION = 0
_DISCORD_EPOCH_MS = 1420070400000


def _aes(root: bytes, guild_id: int) -> Cipher:
    info = b"mimicai:signature:v1:" + int(guild_id or 0).to_bytes(8, "big")
    key = HKDF(hashes.SHA256(), 32, None, info).derive(root)
    return Cipher(algorithms.AES(key), modes.ECB())  # exactly one block: the bare cipher


def sign(root: bytes, guild_id: int, user_id: int, pid: str) -> str:
    """The token for `user_id` running `pid` in this server (0 outside one)."""
    low = (_CLASSES.index(pid[0]) << 62) | (int(pid[1:], 16) << 2) | _VERSION
    enc = _aes(root, guild_id).encryptor()
    block = enc.update(int(user_id).to_bytes(8, "big") + low.to_bytes(8, "big")) + enc.finalize()
    return "".join(_BYTE[b] for b in block)


def strip(text: str) -> str:
    """`text` with every alphabet character gone: a pasted token, or a forged one."""
    return text.translate(_STRIP)


def unsign(text: str) -> str:
    """A posted message's text as it was before `attach`."""
    return strip(_SIGNED_TAIL.sub("", text))


def attach(text: str, token: str) -> str:
    """`text` as it goes out: stripped first, so the only token in it is ours, then a
    space and the token, at the end where it breaks no `-# `, `> ` or list marker."""
    text = strip(text).rstrip()
    if not token:
        return text
    return f"{text} {token}" if text else token


def kept(sent: str, stored: str) -> bool:
    """Whether Discord stored the token `sent` ended with. True when it carried none.

    Everything here rests on Discord keeping four characters it has never promised to
    keep. Stripped or trimmed, every later trace fails with nothing else to show for it,
    so the wire asks once per message it gets back (`ServerManager.run_webhook`).
    """
    m = _TRAILING.search(sent or "")
    return not m or (stored or "").endswith(m.group())


def read(root: bytes, guild_id: int, content: str) -> Optional[Tuple[int, str]]:
    """(user ID, PID) from a message's trailing token, or None. Only for a message whose
    author is already known to be ours."""
    m = _TRAILING.search(content or "")
    if not m:
        return None
    t = m.group()
    raw = bytes(sum(_VALUE[t[i + j]] << (6 - 2 * j) for j in range(4)) for i in range(0, SIG_LEN, 4))
    dec = _aes(root, guild_id).decryptor()
    block = dec.update(raw) + dec.finalize()
    user_id, low = int.from_bytes(block[:8], "big"), int.from_bytes(block[8:], "big")
    made_ms = (user_id >> 22) + _DISCORD_EPOCH_MS
    if low & 3 != _VERSION or not _DISCORD_EPOCH_MS < made_ms < time.time() * 1000:
        return None
    return user_id, f"{_CLASSES[low >> 62]}{(low >> 2) & (2**60 - 1):015X}"
