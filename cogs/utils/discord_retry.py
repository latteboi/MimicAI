"""Transient Discord failures, retried where retrying is safe.

A 503: discord.py retries 500, 502, 504 and 524 itself, inside `HTTPClient.request`, and
raises DiscordServerError on a 503 at once. Discord's edge answers 503 when it cannot
reach the service behind it ("upstream connect error or disconnect/reset before
headers"), which is as transient as a 502 -- and one cost a regeneration its edit: the
new reply stored, the old one left on screen.

A connection reset on an interaction's callback (`defer`, `send_message`): aiohttp
resends a request that meets a reset keep-alive connection only for idempotent methods,
and the callback is a POST; discord.py's own reset retry tests BSD's and Windows' errno
(54, 10054), never Linux's 104. So on prod a reset failed the command before its body
ran, where dev (macOS) retried it unseen.

Patched on the class, once, so the main bot, every child bot and every webhook in the
process are covered by the one fix rather than a retry at each call site.

Only requests that mean the same thing sent twice. A 503'd or reset POST may still have
landed, and sending it again posts the message twice -- bar the callback: Discord takes
one per interaction, so a second can never answer twice.
"""

import asyncio
import errno

import discord
from discord.http import HTTPClient
from discord.webhook.async_ import AsyncWebhookAdapter

_IDEMPOTENT = frozenset({"GET", "PUT", "PATCH", "DELETE"})
_TRIES = 3
_installed = False


def install_retries() -> None:
    # ponytail: wraps discord.py internals by their 2.x signatures; drop each once
    # discord.py retries it itself.
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

    original_webhook = AsyncWebhookAdapter.request

    async def webhook_request(self, route, session, *, files=None, **kwargs):
        try:
            return await original_webhook(self, route, session, files=files, **kwargs)
        except OSError as e:
            if e.errno != errno.ECONNRESET or not route.path.endswith("/callback"):
                raise
        # At once, on a fresh connection: the interaction has three seconds.
        for f in files or ():
            f.reset(seek=True)
        return await original_webhook(self, route, session, files=files, **kwargs)

    AsyncWebhookAdapter.request = webhook_request
