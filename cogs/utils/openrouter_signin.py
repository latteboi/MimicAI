"""Sign in with OpenRouter: its OAuth PKCE flow, which hands back a key the user never
copies. https://openrouter.ai/docs/guides/overview/auth/oauth

OpenRouter sends the browser back to `OPENROUTER_SIGNIN_URL` with `?code=&state=`. That
URL is a Cloudflare Worker (`cloudflare/openrouter_signin.js`), not this process, so the
VM opens no port: the Worker posts `"<state> <code>"` through a webhook into a channel
the bot can read, and the bot, already on the gateway, takes it with `wait_for` and deletes
it in `on_message`. A code
is worthless without its verifier, which never leaves this process, works once, and dies
ten minutes after it is issued.
"""

import asyncio
import base64
import hashlib
import secrets
from typing import Optional, Tuple
from urllib.parse import urlencode

import discord
import httpx

from .constants import defaultConfig
from .http_client import get_shared_client

AUTH_URL = "https://openrouter.ai/auth"
EXCHANGE_URL = "https://openrouter.ai/api/v1/auth/keys"


class SignInFailed(Exception):
    """A sign-in that came back with no key, worded for the user."""


def enabled() -> bool:
    return bool(defaultConfig.OPENROUTER_SIGNIN_URL and defaultConfig.OPENROUTER_RELAY_WEBHOOK_ID)


def challenge_for(verifier: str) -> str:
    """PKCE's S256: the verifier's SHA-256, base64url without padding."""
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()


def start() -> Tuple[str, str, str]:
    """A new sign-in: (the link to OpenRouter, its state, its verifier)."""
    verifier, state = secrets.token_urlsafe(48), secrets.token_urlsafe(32)
    query = urlencode({"callback_url": defaultConfig.OPENROUTER_SIGNIN_URL,
                       "code_challenge": challenge_for(verifier), "code_challenge_method": "S256",
                       "key_label": "MimicAI", "state": state})
    return f"{AUTH_URL}?{query}", state, verifier


async def receive_code(bot: discord.Client, state: str, timeout: float) -> Optional[str]:
    """The code the Worker relays for `state`, or None if none arrives in `timeout`.

    Matched on the relay webhook's id, never the channel alone, so nobody else posting
    there can answer a sign-in. The post is deleted by `clear_relay_post`, not here.
    """
    relay, prefix = defaultConfig.OPENROUTER_RELAY_WEBHOOK_ID, f"{state} "
    try:
        message = await bot.wait_for(
            "message", timeout=timeout,
            check=lambda m: m.webhook_id == relay and m.content.startswith(prefix))
    except asyncio.TimeoutError:
        return None
    return message.content[len(prefix):].strip()


async def clear_relay_post(message: discord.Message) -> bool:
    """Deletes every relay post on arrival, whether or not a sign-in waits for it. A post
    for a wait that is gone (the wizard timed out, a restart, the link opened twice) is
    matched by nobody, and deleted only after a match it would stay in the channel for good."""
    if message.webhook_id != defaultConfig.OPENROUTER_RELAY_WEBHOOK_ID:
        return False
    try:
        await message.delete()
    except discord.NotFound:
        pass
    except discord.HTTPException as e:
        # 403: the bot lacks Manage Messages in the relay channel.
        print(f"OpenRouter sign-in: could not delete relay post {message.id}: {e}")
    return True


async def exchange(code: str, verifier: str) -> str:
    """Swaps a code for the user's new key. Raises SignInFailed."""
    try:
        response = await get_shared_client().post(EXCHANGE_URL, timeout=10.0, json={
            "code": code, "code_verifier": verifier, "code_challenge_method": "S256"})
    except httpx.RequestError as e:
        raise SignInFailed(f"Could not reach OpenRouter ({type(e).__name__}). Try again in a minute.") from e
    if response.status_code == 403:
        raise SignInFailed("That sign-in expired or was already used. Press **Sign in with OpenRouter** again.")
    if response.status_code != 200:
        raise SignInFailed(f"OpenRouter refused the sign-in (status {response.status_code}).")
    try:
        key = response.json().get("key")
    except ValueError:
        key = None
    if not key:
        raise SignInFailed("OpenRouter answered the sign-in without a key.")
    return key
