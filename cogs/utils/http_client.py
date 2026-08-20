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

from typing import Optional

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
