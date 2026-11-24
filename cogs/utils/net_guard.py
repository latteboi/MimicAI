"""Destination validation for URLs the bot fetches on a user's behalf.

RAG URL context and media resolution both fetch links a Discord user typed, from
inside the host process, and hand the body to a model that anyone in the channel
can then ask to read back. That makes every service the host can reach readable
from chat: an Ollama server on 127.0.0.1:11434, a VPC neighbour on 10.x, the
metadata endpoint on 169.254.169.254 -- and, more mundanely, a page that echoes
the requester's address, which is how the host's own public IP leaks.

`ip_address(...).is_global` is the whole test. It is False for loopback, private,
link-local, CGNAT, multicast, reserved and unspecified in one call, so there is no
range list here to fall out of date. IPv4-mapped IPv6 is unmapped first: before
Python 3.13 the whole of ::ffff:0:0/96 read as private, which fails closed but
would refuse legitimate v4-mapped answers.

Checking the typed URL is not enough -- one 302 defeats it -- so redirects are
walked here with `follow_redirects=False` and every hop is revalidated. Use
`safe_stream` rather than open-coding that loop.

This does not close the DNS-rebinding window between the check and the connect;
doing that means pinning the connection to the validated address, which needs a
custom transport. The exposure is a few milliseconds against an attacker who
controls the authoritative nameserver, and is worth revisiting only if the rest
is in place.
"""

import asyncio
import ipaddress
import socket
from contextlib import asynccontextmanager
from urllib.parse import urljoin, urlsplit

import httpx

#: Hops allowed before a redirect chain is treated as hostile.
MAX_REDIRECTS = 5

#: How long one name may take to resolve. A resolver's own timeout is several seconds a try.
DNS_SECONDS = 5.0

#: Asked of every server, and a body sent compressed anyway is refused unread. httpx
#: decodes a whole network chunk before any caller's byte cap sees it: 8 KB of zstd came
#: out as 768 MiB, an OOM kill on the 1 GB host. `tests/test_attachment_limits.py`.
_IDENTITY = {"Accept-Encoding": "identity"}


class UnsafeURL(Exception):
    """A URL that resolves somewhere the bot must not fetch from."""


async def assert_public_url(url: str) -> None:
    """Raises UnsafeURL unless every address `url`'s host resolves to is public."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise UnsafeURL("unsupported scheme")

    host = parts.hostname
    if not host:
        raise UnsafeURL("no host")

    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        raise UnsafeURL("invalid port")

    # A literal address needs no resolver; getaddrinfo would accept it anyway, but
    # this keeps the common attack -- a bare http://127.0.0.1:11434 -- off the DNS
    # path entirely.
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        # The loop's own resolver, never `to_thread`: that is the default executor, two
        # workers (BotManager) which every session flush, profile read and lock refresh
        # shares, and a name whose nameserver never answers held one for the resolver's
        # whole timeout -- three links in one round held both. Under uvloop this runs on
        # libuv's pool instead, and the bound stops the wait as well. `tests/test_net_guard.py`.
        # ponytail: plain asyncio (MIMIC_UVLOOP=0) still uses that executor, so there the
        # bound frees the caller and not the thread; a dedicated DNS executor if that is ever prod.
        try:
            infos = await asyncio.wait_for(
                asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM),
                DNS_SECONDS)
        except (socket.gaierror, UnicodeError, asyncio.TimeoutError):
            raise UnsafeURL("host does not resolve")
        addresses = [ipaddress.ip_address(info[4][0]) for info in infos]

    for addr in addresses:
        mapped = getattr(addr, "ipv4_mapped", None)
        if mapped is not None:
            addr = mapped
        # Every answer must be public, not just the first. A name returning one
        # routable address and one loopback address is the standard rebinding
        # setup, and the connect is free to pick either.
        if not addr.is_global:
            raise UnsafeURL("resolves to a non-public address")


@asynccontextmanager
async def safe_stream(client: httpx.AsyncClient, method: str, url: str, *,
                      timeout: float, max_redirects: int = MAX_REDIRECTS):
    """`client.stream`, with every hop of the redirect chain validated.

    Yields the first non-redirect response. The caller streams from it as usual;
    the response is closed on exit exactly as `client.stream` would close it.
    """
    for _ in range(max_redirects + 1):
        await assert_public_url(url)
        async with client.stream(method, url, follow_redirects=False,
                                 timeout=timeout, headers=_IDENTITY) as response:
            if response.has_redirect_location:
                # Relative Locations are legal and common, so join against the URL
                # actually requested rather than trusting the header alone.
                url = urljoin(str(response.url), response.headers["location"])
                continue
            if response.headers.get("content-encoding", "").strip().lower() not in ("", "identity"):
                raise httpx.DecodingError("compressed response refused", request=response.request)
            yield response
            return
    raise UnsafeURL("too many redirects")
