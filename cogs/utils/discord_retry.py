"""A 503 from Discord, retried where retrying is safe.

discord.py retries 500, 502, 504 and 524 itself, inside `HTTPClient.request`, and raises
DiscordServerError on a 503 at once. Discord's edge answers 503 when it cannot reach the
service behind it ("upstream connect error or disconnect/reset before headers"), which
is as transient as a 502 -- and one cost a regeneration its edit: the new reply stored,
the old one left on screen.

Patched on the class, once, so the main bot, every child bot and every webhook in the
process are covered by the one fix rather than a retry at each call site.

Only methods that mean the same thing sent twice. A 503'd POST may still have landed,
and sending it again posts the message twice.
"""

import asyncio

import discord
from discord.http import HTTPClient

_IDEMPOTENT = frozenset({"GET", "PUT", "PATCH", "DELETE"})
_TRIES = 3
_installed = False


def install_503_retry() -> None:
    # ponytail: wraps a discord.py internal by its 2.x signature; drop it once
    # discord.py retries 503 itself.
    global _installed
    if _installed:
        return
    _installed = True
    original = HTTPClient.request

    async def request(self, route, *, files=None, form=None, **kwargs):
        for tries in range(_TRIES):
            try:
                return await original(self, route, files=files, form=form, **kwargs)
            except discord.DiscordServerError as e:
                if e.status != 503 or route.method not in _IDEMPOTENT or tries == _TRIES - 1:
                    raise
            # discord.py rewinds files only on its own retries; a fresh call starts at
            # tries 0, which it reads as "do not seek", and would upload zero bytes.
            for f in files or ():
                f.reset(seek=True)
            await asyncio.sleep(1 + tries * 2)

    HTTPClient.request = request
