"""Stripe's API, which sells the MimicAI subscription: Checkout, and the subscriptions
`reconcile` reads. Manage is the Customer Portal's login link (`STRIPE_PORTAL_URL`), no call. https://docs.stripe.com/payments/managed-payments

Stateless calls on the shared client; `cogs/managers/subscriptions` owns what they mean. The
version is pinned, so no field moves under us (a period's start is on the subscription *item*
since 2025). Every call raises `StripeError` on anything but success, so a caller can never read
"could not list" as "nobody subscribes". The key is never logged.
"""
from typing import Any, Dict, Iterable, List, Optional

import httpx

from ...utils.constants import SUBSCRIPTION_RETURN_URL, defaultConfig
from ...utils.http_client import get_shared_client

API = "https://api.stripe.com/v1"
VERSION = "2026-10-28.endive"
#: Paid, or in Stripe's retry window after a failed charge.
ENTITLED = ("active", "past_due")


class StripeError(Exception):
    """A Stripe call that did not succeed. `status` is the HTTP status, or None."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


def enabled() -> bool:
    return bool(defaultConfig.STRIPE_SECRET_KEY)


async def _call(method: str, path: str, **kwargs) -> Dict[str, Any]:
    key = defaultConfig.STRIPE_SECRET_KEY
    if not key:
        raise StripeError("no Stripe key")
    try:
        response = await get_shared_client().request(
            method, API + path, timeout=15.0,
            headers={"Authorization": f"Bearer {key}", "Stripe-Version": VERSION}, **kwargs)
    except httpx.HTTPError as e:
        raise StripeError(f"{method} {path}: {type(e).__name__}") from None
    try:
        body = response.json()
    except ValueError:
        raise StripeError(f"{method} {path}: HTTP {response.status_code}, not JSON",
                          response.status_code) from None
    if response.status_code >= 400:
        # Stripe's own words name what to fix ("No such price", a missing Terms URL).
        said = ((body.get("error") or {}).get("message") or "")[:300]
        raise StripeError(f"{method} {path}: HTTP {response.status_code} {said}".strip(),
                          response.status_code)
    return body


async def prices(tiers: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """Each tier's active Price, by its lookup key: tier -> Price. A tier with none is absent."""
    body = await _call("GET", "/prices", params=[("active", "true"), ("limit", "100")]
                       + [("lookup_keys[]", t) for t in tiers])
    return {p["lookup_key"]: p for p in body.get("data") or [] if p.get("lookup_key")}


async def checkout(price_id: str, discord_id: str, customer: Optional[str] = None) -> Dict[str, Any]:
    """A Checkout Session (`url`, `id`) selling `price_id` to this Discord user, through Managed
    Payments (Stripe is the seller). The ID rides on the subscription, which is how reconcile
    finds its owner."""
    data = {"mode": "subscription", "line_items[0][price]": price_id,
            "line_items[0][quantity]": "1", "managed_payments[enabled]": "true",
            "client_reference_id": discord_id,
            "subscription_data[metadata][discord_id]": discord_id,
            "consent_collection[terms_of_service]": "required",
            "success_url": SUBSCRIPTION_RETURN_URL}
    if customer:
        data["customer"] = customer
    return await _call("POST", "/checkout/sessions", data=data)


async def checkout_status(session_id: str) -> str:
    """"open", "complete" (paid) or "expired": one cheap read, where reconcile lists everyone."""
    return (await _call("GET", f"/checkout/sessions/{session_id}")).get("status") or "open"


async def expire_checkout(session_id: str) -> None:
    """Ends an unpaid Checkout, so a link left open in a browser can no longer charge. Raises
    once it is paid (Stripe expires only an open one)."""
    await _call("POST", f"/checkout/sessions/{session_id}/expire")


async def subscriptions() -> List[Dict[str, Any]]:
    """Every subscription on the account not yet cancelled, whatever its status."""
    found, after = [], None
    while True:
        params = {"limit": "100", **({"starting_after": after} if after else {})}
        page = await _call("GET", "/subscriptions", params=params)
        data = page.get("data")
        if not isinstance(data, list):
            raise StripeError("GET /subscriptions: no list in the answer")
        found += data
        if not page.get("has_more") or not data:
            return found
        after = data[-1]["id"]


async def cancel(subscription_id: str) -> None:
    """Cancels at once, not at the period's end. One already gone counts as cancelled."""
    try:
        await _call("DELETE", f"/subscriptions/{subscription_id}")
    except StripeError as e:
        if e.status != 404:
            raise
