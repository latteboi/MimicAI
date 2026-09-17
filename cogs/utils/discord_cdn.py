"""Discord attachment links, which stop working a day after they are handed out.

An attachment URL is signed: `ex` is when it expires (hex epoch seconds), `is` when it
was issued, `hm` the signature. Past `ex`, anything outside Discord gets a 404 -- the
bot's own downloads included. A saved avatar is kept exactly as it was typed, and one of
these runs on the way out, chosen by who reads the link:

- **Discord** -- an embed's image or icon, a webhook's `avatar_url`. Handed the link with
  no query at all, Discord signs a fresh one whenever it draws it. That is the API
  reference's own answer ("Signed Attachment CDN URLs"), and costs no request:
  `unsigned_attachment_url`.
- **The bot** -- the content classifier and a child bot's avatar download the image, so
  they need a signature that is still good: `signed_attachment_url`.
"""
import asyncio
import re
import time
from typing import Any, Optional
from urllib.parse import parse_qs, urlsplit

from discord.http import Route

#: The attachment's path, on the CDN or the media proxy. Anchored at both ends, so a
#: look-alike on another host, or a longer path, is left as it is.
_ATTACHMENT = re.compile(
    r"^https://(?:cdn\.discordapp\.com|media\.discordapp\.net)"
    r"(/attachments/\d+/\d+/[^/?#\s]+)(?:[?#]\S*)?$",
    re.IGNORECASE)
_CDN = "https://cdn.discordapp.com"
#: A signature with less than this left is refreshed before downloading on it.
SIGNATURE_MARGIN_SECONDS = 60


def _attachment_path(url: Optional[str]) -> Optional[str]:
    # The substring test first: this runs for every appearance lookup, autocomplete
    # included, and almost no avatar is a Discord attachment.
    if not url or "/attachments/" not in url:
        return None
    match = _ATTACHMENT.match(url.strip())
    return match.group(1) if match else None


def unsigned_attachment_url(url: Optional[str]) -> Optional[str]:
    """`url` for Discord to draw: a Discord attachment on the CDN with no query, anything
    else untouched. The media proxy's `format`/`width` go with the signature."""
    path = _attachment_path(url)
    return _CDN + path if path else url


def attachment_expiry(url: str) -> Optional[float]:
    """When a signed attachment link stops working, or None if it carries no `ex`."""
    try:
        return float(int(parse_qs(urlsplit(url).query)["ex"][0], 16))
    except (KeyError, IndexError, ValueError):
        return None


async def signed_attachment_url(http: Any, url: str, timeout: float = 10.0) -> str:
    """`url`, or a freshly signed copy when it is a Discord attachment with no signature
    left, for the bot to download. `http` is a client's `discord.http.HTTPClient`.

    `POST /attachments/refresh-urls` is what Discord's own client calls, and the API
    reference does not list it. So any failure hands `url` back unchanged, and the
    download that follows fails the way it always did.
    """
    path = _attachment_path(url)
    if path is None:
        return url
    expiry = attachment_expiry(url)
    if expiry is not None and expiry - time.time() > SIGNATURE_MARGIN_SECONDS:
        return url
    try:
        data = await asyncio.wait_for(
            http.request(Route("POST", "/attachments/refresh-urls"),
                         json={"attachment_urls": [_CDN + path]}),
            timeout=timeout)
    except Exception as e:
        print(f"Could not refresh a Discord attachment link ({type(e).__name__}); "
              "downloading it as saved.")
        return url
    refreshed = data.get("refreshed_urls") if isinstance(data, dict) else None
    for item in refreshed if isinstance(refreshed, list) else []:
        fresh = item.get("refreshed") if isinstance(item, dict) else None
        if isinstance(fresh, str) and _attachment_path(fresh) and attachment_expiry(fresh):
            return fresh
    return url
