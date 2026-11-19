"""OpenRouter's management API: the keys the MimicAI subscription mints on the operator's account.

Stateless calls on the shared client; `cogs/managers/subscriptions` owns what they mean. Every
call raises `KeyApiError` on anything but success, so a caller can never read "could not list"
as "there are none". The management key is never logged, and nothing identifying a subscriber
is sent: every key an instance makes has the same name (`SubscriptionStore.key_name`).
"""
from typing import Any, Dict, List, Optional

import httpx

from ...utils.constants import defaultConfig
from ...utils.http_client import get_shared_client

API = "https://openrouter.ai/api/v1"
#: What OpenRouter returns per page of `GET /keys` (its guide, not its schema).
PAGE = 100


class KeyApiError(Exception):
    """A management call that did not succeed. `status` is the HTTP status, or None."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


async def _call(method: str, path: str, **kwargs) -> Dict[str, Any]:
    key = defaultConfig.OPENROUTER_MANAGEMENT_KEY
    if not key:
        raise KeyApiError("no management key")
    try:
        response = await get_shared_client().request(
            method, API + path, timeout=15.0, headers={"Authorization": f"Bearer {key}"}, **kwargs)
    except httpx.HTTPError as e:
        raise KeyApiError(f"{method} {path}: {type(e).__name__}") from None
    if response.status_code >= 400:
        raise KeyApiError(f"{method} {path}: HTTP {response.status_code}", response.status_code)
    try:
        return response.json()
    except ValueError:
        raise KeyApiError(f"{method} {path}: not JSON") from None


def _in_workspace(fields: Dict[str, Any]) -> Dict[str, Any]:
    """`fields`, naming the subscription's workspace when one is set. `create` and the listing
    must both go through this: OpenRouter lists one workspace at a time (Default unless named),
    so a key made in another is invisible to the sweep and its spend never syncs."""
    workspace = defaultConfig.OPENROUTER_SUBSCRIPTION_WORKSPACE
    return {**fields, "workspace_id": workspace} if workspace else fields


async def create(name: str, limit: float) -> Dict[str, Any]:
    """A new key called `name`, capped at `limit` US$: its object, with the key under "key"."""
    body = await _call("POST", "/keys", json=_in_workspace(
        {"name": name, "limit": limit, "include_byok_in_limit": True}))
    data = dict(body.get("data") or {})
    data["key"] = body.get("key")
    if not data.get("key") or not data.get("hash"):
        raise KeyApiError("POST /keys: no key in the answer")
    return data


async def set_limit(key_hash: str, limit: float) -> None:
    await _call("PATCH", f"/keys/{key_hash}", json={"limit": limit})


async def usage(key_hash: str) -> float:
    """What a key has spent, in US$, now rather than as of the last sweep."""
    data = (await _call("GET", f"/keys/{key_hash}")).get("data") or {}
    return float(data.get("usage") or 0)


async def delete(key_hash: str) -> None:
    """Deletes a key. One already gone counts as deleted."""
    try:
        await _call("DELETE", f"/keys/{key_hash}")
    except KeyApiError as e:
        if e.status != 404:
            raise


async def list_subscription_keys(name: str) -> List[Dict[str, Any]]:
    """Every key on the account called `name`, disabled ones included."""
    found, offset = [], 0
    while True:
        page = (await _call("GET", "/keys", params=_in_workspace(
            {"offset": offset, "include_disabled": "true"}))).get("data")
        if not isinstance(page, list):
            raise KeyApiError("GET /keys: no list in the answer")
        found += [k for k in page if k.get("name") == name]
        if len(page) < PAGE:
            return found
        offset += len(page)


async def credit() -> float:
    """What is left on the account, in US$: purchased less used."""
    data = (await _call("GET", "/credits")).get("data") or {}
    return float(data.get("total_credits") or 0) - float(data.get("total_usage") or 0)
