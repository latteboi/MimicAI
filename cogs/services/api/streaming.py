"""Streaming a request body that carries files, without ever holding one whole.

The adapters emit `_FILE_BLOB_TOKEN` where a base64 blob would go; `_plan_streamed_body`
turns the payload plus its file list into segments and an exact byte count, and
`_aiter_streamed_body` splices the files in as the body goes out. The explicit
Content-Length that count buys is what keeps httpx off chunked encoding.
"""

import base64
import os
import orjson as json
from typing import Any, List, Tuple

from ...utils.net_guard import safe_stream



# 256 KB per read. Large enough that a 6 MB attachment costs ~24 reads rather
# than ~750, small enough that each one stays well under the ~10 ms event-loop
# budget — a read of this size from the page cache (and the file was written
# moments earlier, so it is hot) costs tens of microseconds.
_UPLOAD_CHUNK_BYTES = 256 * 1024

# Download chunk for the streaming fetch. 8 KB meant ~750 allocation/append
# cycles for a single phone photo; 64 KB cuts that by a factor of eight without
# meaningfully raising the peak.
_DOWNLOAD_CHUNK_BYTES = 64 * 1024


async def _aiter_file_bytes(path: str, chunk_size: int = _UPLOAD_CHUNK_BYTES):
    """Yields `path` in chunks for use as an httpx request body.

    Must be an *async* generator: httpx refuses a sync iterable body on an
    AsyncClient. httpx would normally pair an async body with
    `Transfer-Encoding: chunked`, but `Request._prepare` skips that default when
    Content-Length is already set explicitly — which the caller does, from
    `os.path.getsize`. So the request goes out byte-identical to the buffered
    version it replaces, and the resumable protocol sees no difference.
    """
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


#: Read size for base64-encoding a file into a request body. A multiple of 3, so
#: every block encodes to a whole number of base64 quads and padding can only ever
#: appear on the final short read -- which is what makes the concatenation of the
#: blocks identical to encoding the file in one go.
_B64_READ_BYTES = 192 * 1024

#: Marks the spot in a serialised payload where a file's base64 belongs. The token
#: has to survive JSON encoding unchanged and never collide with real content;
#: `@` is outside the base64 alphabet and the rest is not a substring of anything
#: the bot sends.
_FILE_BLOB_TOKEN = "@@MIMIC_FILE_BLOB_{}@@"


def _b64_encoded_length(byte_length: int) -> int:
    """Length of the base64 of `byte_length` bytes, padding included."""
    return 4 * ((byte_length + 2) // 3)


def _plan_streamed_body(payload: dict, file_paths: List[str]) -> Tuple[List[Any], int]:
    """Splits a serialised payload around its file placeholders.

    Returns (segments, content_length), where a segment is either `bytes` to send
    as-is or an open file handle whose base64 goes in its place. Nothing here reads
    the file contents — the length comes from `fstat`, so a payload carrying a 6 MB
    image costs the size of the *rest* of the JSON, not the image.

    The handles are opened *now*, not while the body streams, and that is
    load-bearing: Content-Length is already on the wire by then, so a file that has
    since been deleted -- a round tears its generated image down in a `finally` --
    would abort a request mid-flight instead of failing cleanly before it starts.
    An open descriptor still reads fine after the path is unlinked.

    Used by the OpenRouter and Ollama adapters, which unlike Google have to send
    image bytes inline: the provider takes a data URI, not a file handle. Building
    that the obvious way -- read, b64encode, decode, interpolate, serialise -- put
    about five copies of the image in the heap at once, per participant per round,
    with nothing shared between participants.

    The caller must send the segments (`_aiter_streamed_body` closes the handles) or
    close them itself.
    """
    body = json.dumps(payload)
    segments: List[Any] = []
    total = 0
    rest = body
    try:
        for idx, path in enumerate(file_paths):
            token = _FILE_BLOB_TOKEN.format(idx).encode('ascii')
            head, sep, rest = rest.partition(token)
            if not sep:
                raise ValueError(f"payload lost its placeholder for {path}")
            segments.append(head)
            total += len(head)
            handle = open(path, "rb")
            segments.append(handle)
            total += _b64_encoded_length(os.fstat(handle.fileno()).st_size)
    except BaseException:
        _close_body_segments(segments)
        raise
    segments.append(rest)
    total += len(rest)
    return segments, total


def _close_body_segments(segments: List[Any]) -> None:
    for segment in segments:
        if not isinstance(segment, bytes):
            try:
                segment.close()
            except Exception:
                pass


async def _aiter_streamed_body(segments: List[Any]):
    """Yields the request body planned by `_plan_streamed_body`, closing its handles.

    Async for the reason `_aiter_file_bytes` is, and paired with an explicit
    Content-Length for the same reason: it keeps httpx off `Transfer-Encoding:
    chunked`, so the request on the wire is identical to the buffered one this
    replaces.
    """
    try:
        for segment in segments:
            if isinstance(segment, bytes):
                yield segment
                continue
            while True:
                block = segment.read(_B64_READ_BYTES)
                if not block:
                    break
                yield base64.b64encode(block)
    finally:
        # Covers the abandoned-generator case too: httpx never finishing the body
        # would otherwise leak a descriptor per image.
        _close_body_segments(segments)


async def _stream_to_tempfile(url: str, client, timeout: float = 15.0) -> str:
    """Downloads `url` to a temp file without ever holding it in RAM. Returns the
    path; the caller owns it."""
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".tmp")
    try:
        # Both callers pass a URL a Discord user supplied, so the destination is
        # validated on every redirect hop rather than trusted once.
        async with safe_stream(client, "GET", url, timeout=timeout) as resp:
            resp.raise_for_status()
            with os.fdopen(fd, 'wb') as f:
                async for chunk in resp.aiter_bytes(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                    f.write(chunk)
    except BaseException:
        try:
            os.remove(path)
        except OSError:
            pass
        raise
    return path
