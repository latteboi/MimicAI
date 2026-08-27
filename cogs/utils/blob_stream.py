"""Streaming extraction of large base64 blobs from a JSON response body.

Gemini returns generated images and synthesised audio as base64 `inlineData`
inside the `generateContent` response — there is no Files API destination and no
bucket to write to, so the bytes come back over the wire whatever else we do.
Reading that body the obvious way costs about 3.7x the image:

    json.loads(response.content)

    1. httpx accumulates the body in a list of chunks and `b"".join`s them, so
       the wire form alone peaks at twice its own size (~1.33x the image, twice).
    2. orjson builds a str for the base64 field while the raw body is still held
       by the response object: 1.33x alongside 1.33x.
    3. `_RestView` base64-decodes on first read: another 1x.

On the e2-micro that is ~22 MB of transient for one 2K PNG, and two image workers
can be inside that window at once. Nulling the locals afterwards -- which the call
sites do -- bounds how *long* the copies live, but nothing a caller does can touch
the peak, because the peak happens inside `json.loads` before any caller sees the
response.

This module removes the peak instead. It scans the body as it streams off the
socket and diverts any oversized `data` value straight into a file, base64-decoding
as it goes, leaving a small JSON skeleton in which that value has been replaced by
`"@blob:/path/to/file"`. Peak cost is one chunk plus the skeleton, whatever the
image weighs.

Three properties of the wire format make a hand-written scan the right tool here
rather than an incremental JSON parser:

- **Base64 contains no backslash.** Once a value is known to be a blob, finding
  its end is a bare `find(b'"')` -- no escape handling, no ambiguity. Escape-aware
  scanning is still used for every *other* string, because those are text.
- **Only key tracking is needed**, not a parse tree: remember the last string that
  was immediately followed by a colon.
- **The threshold makes it backwards compatible.** A `data` value under
  `DEFAULT_BLOB_THRESHOLD` is left in the skeleton as the base64 it arrived as, so
  small inline parts take exactly the path they always did, and a text part that
  happens to contain the literal `"data":` costs nothing because it never gets
  that far.
"""

import binascii
import os
import tempfile
from typing import List, Optional, Tuple

#: Only a `data` value longer than this is diverted to disk. Comfortably above the
#: largest base64 field the bot sees that is not media -- thought signatures run to
#: a few KB -- and comfortably below the smallest generated image. It is also the
#: peak: a value that turns out to be shorter has to be re-emitted verbatim, so this
#: much of it is held in RAM before the decision can be made.
DEFAULT_BLOB_THRESHOLD = 64 * 1024

#: Prefix marking a diverted value in the skeleton. `@` is not in the base64
#: alphabet, so no genuine inline value can be mistaken for one.
BLOB_SENTINEL = "@blob:"

_BACKSLASH = 0x5C
_QUOTE = 0x22
_WHITESPACE = frozenset(b" \t\r\n")

# States.
_OUTSIDE = 0      # structural JSON, between strings
_IN_STRING = 1    # inside a string that is kept in the skeleton
_AWAIT_COLON = 2  # a string just closed; a colon here makes it a key
_IN_BLOB = 3      # inside the value of a `data` key


class TruncatedJSONError(ValueError):
    """The stream ended mid-value. Raised rather than returning a half skeleton,
    which would surface as an unrelated orjson error one frame later."""


class InlineBlobExtractor:
    """Feed it response chunks; get back a skeleton and the blob paths.

    Not thread-safe and not reusable: one instance per response.
    """

    __slots__ = ("skeleton", "blob_paths", "_key", "_threshold", "_suffix", "_dir",
                 "_buf", "_state", "_scan_from", "_pending_key", "_blob_buf",
                 "_blob_file", "_blob_path", "_carry")

    def __init__(self, key: bytes = b"data", threshold: int = DEFAULT_BLOB_THRESHOLD,
                 suffix: str = ".bin", dir: Optional[str] = None):
        self.skeleton = bytearray()
        self.blob_paths: List[str] = []
        self._key = key
        self._threshold = threshold
        self._suffix = suffix
        self._dir = dir

        self._buf = bytearray()
        self._state = _OUTSIDE
        self._scan_from = 0
        self._pending_key: Optional[bytes] = None

        self._blob_buf: Optional[bytearray] = None
        self._blob_file = None
        self._blob_path: Optional[str] = None
        self._carry = b""

    # -- public ---------------------------------------------------------------

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._buf += chunk
        self._run()

    def finish(self) -> Tuple[bytes, List[str]]:
        """Returns (skeleton_json, blob_paths). The caller owns the files from here."""
        if self._state != _OUTSIDE:
            self.cleanup()
            raise TruncatedJSONError(
                f"response ended mid-value (state {self._state}, "
                f"{len(self._buf)} bytes unconsumed)")
        self.skeleton += self._buf
        self._buf.clear()
        return bytes(self.skeleton), list(self.blob_paths)

    def cleanup(self) -> None:
        """Unlinks everything already written. For the error paths -- a partial
        response has no owner to hand the files to."""
        if self._blob_file is not None:
            try:
                self._blob_file.close()
            except Exception:
                pass
            self._blob_file = None
        paths = list(self.blob_paths)
        if self._blob_path:
            paths.append(self._blob_path)
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                pass
        self.blob_paths = []
        self._blob_path = None

    # -- scanner --------------------------------------------------------------

    def _run(self) -> None:
        while True:
            if self._state == _OUTSIDE:
                if not self._step_outside():
                    return
            elif self._state == _IN_STRING:
                if not self._step_in_string():
                    return
            elif self._state == _AWAIT_COLON:
                if not self._step_await_colon():
                    return
            else:
                if not self._step_in_blob():
                    return

    def _step_outside(self) -> bool:
        if self._pending_key == self._key:
            # `"data": null` is legal and `"data": 12` would be too. Without this
            # the search below would run past the non-string value and divert the
            # *next* string it found instead.
            i = 0
            n = len(self._buf)
            while i < n and self._buf[i] in _WHITESPACE:
                i += 1
            self.skeleton += self._buf[:i]
            del self._buf[:i]
            if not self._buf:
                return False
            if self._buf[0] != _QUOTE:
                self._pending_key = None

        i = self._buf.find(b'"')
        if i < 0:
            # Structural JSON has no long runs; copying it wholesale is what keeps
            # the skeleton small enough to parse normally at the end.
            self.skeleton += self._buf
            self._buf.clear()
            return False

        self.skeleton += self._buf[:i]
        is_blob = self._pending_key == self._key
        self._pending_key = None
        if is_blob:
            # The opening quote is deliberately *not* emitted here: _close_blob
            # writes the whole quoted string, so a path needing escapes is still
            # legal JSON.
            del self._buf[:i + 1]
            self._state = _IN_BLOB
            self._blob_buf = bytearray()
            self._blob_file = None
            self._blob_path = None
            self._carry = b""
        else:
            self.skeleton += b'"'
            del self._buf[:i + 1]
            self._state = _IN_STRING
            self._scan_from = 0
        return True

    def _step_in_string(self) -> bool:
        j = self._find_unescaped_quote()
        if j < 0:
            return False
        content = bytes(self._buf[:j])
        self.skeleton += self._buf[:j + 1]
        del self._buf[:j + 1]
        self._pending_key = content
        self._state = _AWAIT_COLON
        return True

    def _step_await_colon(self) -> bool:
        i = 0
        n = len(self._buf)
        while i < n and self._buf[i] in _WHITESPACE:
            i += 1
        self.skeleton += self._buf[:i]
        del self._buf[:i]
        if not self._buf:
            return False
        if self._buf[0] == 0x3A:  # ':' -- the string just closed was a key
            self.skeleton += b':'
            del self._buf[:1]
        else:
            # It was a value. Nothing to remember.
            self._pending_key = None
        self._state = _OUTSIDE
        return True

    def _step_in_blob(self) -> bool:
        i = self._buf.find(b'"')
        if i < 0:
            self._absorb(self._buf)
            self._buf.clear()
            return False
        self._absorb(self._buf[:i])
        del self._buf[:i + 1]
        self._close_blob()
        self._state = _OUTSIDE
        return True

    def _find_unescaped_quote(self) -> int:
        buf = self._buf
        i = self._scan_from
        while True:
            j = buf.find(b'"', i)
            if j < 0:
                self._scan_from = len(buf)
                return -1
            k = j - 1
            backslashes = 0
            while k >= 0 and buf[k] == _BACKSLASH:
                backslashes += 1
                k -= 1
            if backslashes % 2 == 0:
                return j
            i = j + 1

    # -- blob sink ------------------------------------------------------------

    def _absorb(self, data) -> None:
        pending = self._blob_buf
        if pending is not None:
            pending += data
            if len(pending) < self._threshold:
                return
            fd, self._blob_path = tempfile.mkstemp(suffix=self._suffix, dir=self._dir)
            self._blob_file = os.fdopen(fd, 'wb')
            self._blob_buf = None
            self._decode_into_file(pending)
        else:
            self._decode_into_file(data)

    def _decode_into_file(self, data) -> None:
        # a2b_base64 wants whole 4-character groups; whatever does not divide
        # evenly is carried into the next chunk. Padding only ever appears at the
        # very end of the run, which _close_blob handles.
        #
        # Concatenating only when there is a carry, and slicing through a
        # memoryview, keeps this to zero full-size copies -- at these sizes the
        # copies *are* the peak.
        buf = (self._carry + bytes(data)) if self._carry else data
        n = (len(buf) // 4) * 4
        sink = self._blob_file
        assert sink is not None
        if n:
            sink.write(binascii.a2b_base64(memoryview(buf)[:n]))
        self._carry = bytes(buf[n:])

    def _close_blob(self) -> None:
        sink = self._blob_file
        if sink is None:
            # Under the threshold: put it back exactly as it arrived and let the
            # normal decode path have it.
            self.skeleton += b'"'
            self.skeleton += self._blob_buf or b""
            self.skeleton += b'"'
            self._blob_buf = None
            return

        if self._carry:
            sink.write(binascii.a2b_base64(self._carry))
            self._carry = b""
        sink.close()
        self._blob_file = None

        path = self._blob_path or ""
        self.skeleton += _json_quote(BLOB_SENTINEL + path)
        self.blob_paths.append(path)
        self._blob_path = None


def _json_quote(value: str) -> bytes:
    """Minimal JSON string encoder for a filesystem path. mkstemp names are tame,
    but `dir=` is caller-supplied and a path is not guaranteed to be."""
    out = bytearray(b'"')
    for ch in value:
        if ch == '"':
            out += b'\\"'
        elif ch == '\\':
            out += b'\\\\'
        elif ch < ' ':
            out += f'\\u{ord(ch):04x}'.encode('ascii')
        else:
            out += ch.encode('utf-8')
    out += b'"'
    return bytes(out)


def is_blob_sentinel(value) -> bool:
    return isinstance(value, str) and value.startswith(BLOB_SENTINEL)


def sentinel_path(value: str) -> str:
    return value[len(BLOB_SENTINEL):]
