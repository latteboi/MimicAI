"""One shared httpx.AsyncClient for the call sites that used to build their own.

Constructing an `httpx.AsyncClient` is not cheap on the deployment target: it
builds a fresh `ssl.SSLContext` and parses the certifi CA bundle into OpenSSL
X509 objects — roughly 14 ms and ~0.8 MB of native allocation, torn down again
moments later. Ten-odd call sites were doing that per request, which cost RSS
twice over: the transient buffers themselves, and the heap fragmentation left
behind by allocating and freeing them at that rate (see
`cogs/utils/memory_tuning.py`). It also threw away connection reuse, so every
OpenRouter or Ollama turn paid for a fresh TLS handshake.

This is the same pattern as `api_service.get_google_rest_client` and
`tools_service.get_url_fetch_client`, generalised for everything else. Created
lazily so it binds to the running event loop, and closed from
`MimicCog.cog_unload`.

Per-call behaviour that genuinely varies — timeouts, `follow_redirects`, a
spoofed User-Agent — belongs on the individual request, not on the client, so
nothing here is baked in beyond a conservative default timeout.
"""

from typing import NamedTuple, Optional

import httpx

# Deliberately below httpx's defaults (100/20). Each idle keepalive connection
# holds a TLS session and its buffers, and the bot never has enough concurrent
# work on 0.25 vCPU to profit from a larger pool.
_LIMITS = httpx.Limits(
    max_connections=20,
    max_keepalive_connections=5,
    keepalive_expiry=30.0,
)

_DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

_shared_client: Optional[httpx.AsyncClient] = None


def get_shared_client() -> httpx.AsyncClient:
    """Returns the process-wide client. Pass `timeout=` per request where the
    default is wrong — the previous per-site clients all did exactly that."""
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT, limits=_LIMITS)
    return _shared_client


async def close_shared_client():
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        await _shared_client.aclose()
    _shared_client = None


#: Read size for `get_capped`, as `streaming._DOWNLOAD_CHUNK_BYTES`.
_CAPPED_CHUNK_BYTES = 64 * 1024


class CappedBody(NamedTuple):
    status_code: int
    headers: httpx.Headers
    #: None for an answer other than 200, and for a body over the cap not asked to be cut.
    content: Optional[bytes]


async def get_capped(client: httpx.AsyncClient, url: str, max_bytes: int, *, timeout,
                     headers: Optional[dict] = None, truncate: bool = False) -> CappedBody:
    """GETs `url`, holding no more than `max_bytes` of its body.

    `(await client.get(url)).content` has buffered the whole body before any size test can
    run, so a limit checked afterwards bounds nothing: an avatar URL can serve a gigabyte.
    The body is read in chunks and the read stops at the cap. Past it `content` is None, or
    with `truncate` the first `max_bytes`. A stated Content-Length over the cap is refused
    before a byte is read, and a body under any status but 200 is not read at all.
    """
    async with client.stream("GET", url, follow_redirects=True, timeout=timeout,
                             headers=headers) as response:
        if response.status_code != 200:
            return CappedBody(response.status_code, response.headers, None)
        declared = response.headers.get("content-length", "")
        if not truncate and declared.isdigit() and int(declared) > max_bytes:
            return CappedBody(response.status_code, response.headers, None)
        chunks, total = [], 0
        async for chunk in response.aiter_bytes(_CAPPED_CHUNK_BYTES):
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                if not truncate:
                    return CappedBody(response.status_code, response.headers, None)
                break
        return CappedBody(response.status_code, response.headers, b"".join(chunks)[:max_bytes])
