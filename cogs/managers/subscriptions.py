"""The MimicAI subscription: who is entitled to which tier, and the OpenRouter key each one holds.

A payment source only answers *who is entitled*: owner grants from `/mod`, and Stripe.
`reconcile` turns that answer into capped OpenRouter keys on the operator's account
(`services/api/openrouter_keys`) and this record. A turn only reads the record, from memory
(`key_for`), so neither OpenRouter's management API nor a payment provider is ever on the turn
path. The design is in tests/MIMIC_SUBSCRIPTION.md, **Architecture**.

`data/subscriptions.json.gz`, sealed, outside every user's folder because `/data`'s export
dumps that folder decrypted. One row per Discord ID:

  tier, source        a SUBSCRIPTION_TIERS key; "grant" or "stripe"
  ref                 Stripe's customer and subscription IDs; kept once lapsed, so a
                      returning subscriber is the same customer
  key, hash           the OpenRouter key and its management handle; absent once lapsed
  limit, usage, synced  the key's cap and spend at the last reconcile, in US$
  period              the start of the paid period the cap was last set for
  base                the key's usage when that period began (`period_usage`): limit and
                      usage are the key's lifetime figures
  ends                when a cancelled Stripe subscription stops; absent while it renews
  renews              when this period ends and the next allowance begins
  status, since       "active" or "lapsed", and since when

and `grants`: Discord ID -> {tier, since}, the owner's grants, and `paused`: the owner stopped new
sales from `/mod` (`sells` is False, `billing` still True).

On the official instance, or a dev one that sets MIMIC_SUBSCRIPTION_DEV (`enabled`). The sweep
deletes every key with this instance's name that no row holds, so each instance names its keys
apart (`key_name`): two sharing one OpenRouter account and one name would delete each other's.
"""
import asyncio
import os
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

from discord.ext import tasks

from ..services.api import openrouter_keys, stripe
from ..services.api.openrouter_keys import KeyApiError
from ..services.api.stripe import StripeError
from ..utils.constants import (OFFICIAL_BOT_ID, SUBSCRIPTION_FILE_PATH, SUBSCRIPTION_KEY_NAME,
                               SUBSCRIPTION_TIERS, defaultConfig)
from .storage_manager import IOManager

ACTIVE, LAPSED = "active", "lapsed"
OFF = "The subscription is off on this instance."
UNREADABLE = "The subscription record would not decode, so nothing was changed."

#: Every write of the record. `drop_user` runs in worker threads.
_LOCK = threading.Lock()
#: A grant has no billing date, so its allowance renews every 30 days from the grant.
GRANT_PERIOD = 30 * 86400


def carried_limit(limit: float, usage: float, allowance: float) -> float:
    """A key's cap for a new paid period (Open questions 2, C): this period's allowance, plus
    what the last left unspent up to one more allowance. So nobody is owed more than two."""
    return usage + allowance + min(max(0.0, limit - usage), allowance)


def tier_name(tier: Optional[str]) -> str:
    """How a tier reads to people: `tier_1` is "Tier 1"."""
    return str(tier or "").replace("_", " ").capitalize()


def period_usage(row: Dict[str, Any]) -> Tuple[float, float]:
    """This period's spend, and what it had to spend (its allowance plus any carried), in US$."""
    base = row.get("base", 0.0)
    return max(0.0, row.get("usage", 0.0) - base), row.get("limit", 0.0) - base


def used_share(row: Dict[str, Any]) -> float:
    """How much of this period's allowance a row has spent, 0 to 1."""
    used, available = period_usage(row)
    return min(1.0, used / available) if available > 0 else 0.0


def runs_out_at(row: Dict[str, Any], now: Optional[float] = None) -> Optional[int]:
    """When this period's allowance runs out at the pace so far, if that is before the period
    ends. None while on pace, used up already, or too early to tell: under 3 days in and under
    20% used, a day-one burst would read as "runs out tomorrow"."""
    now = time.time() if now is None else now
    start, end, share = row.get("period"), row.get("ends") or row.get("renews"), used_share(row)
    if not start or not end or now <= start or not 0 < share < 1:
        return None
    if now - start < 3 * 86400 and share < 0.2:
        return None
    out = start + (now - start) / share
    return int(out) if out < end else None


def carried_over(row: Dict[str, Any]) -> bool:
    """Whether this period's allowance includes some the last one left unspent."""
    allowance = SUBSCRIPTION_TIERS.get(row.get("tier") or "")
    return allowance is not None and period_usage(row)[1] > allowance + 0.005


def row_summary(row: Dict[str, Any]) -> str:
    """A row as one plain line for the owner: tier, status, this period's use."""
    grant = " (grant)" if row.get("source") == "grant" else ""
    used = f" · {used_share(row):.0%} used" if row.get("status") == ACTIVE else ""
    return f"{tier_name(row.get('tier'))} · {row.get('status', '?')}{grant}{used}"


class SubscriptionStore:
    def __init__(self, cog):
        self.cog = cog
        self._data: Dict[str, Any] = {"rows": {}, "grants": {}}
        #: Active key -> Discord ID, for `is_key`. Rebuilt on every write.
        self._keys: Dict[str, str] = {}
        #: The file exists and would not decode. Nothing reconciles until it is fixed: an
        #: empty record would read every key as an orphan and delete it.
        self._unreadable = False
        self._reconciling = asyncio.Lock()
        #: The last full run's findings, for the owner's `/mod` tab.
        self.credit: Optional[float] = None
        self.unused: Optional[float] = None
        self.last_run: Optional[float] = None
        self.last_error: Optional[str] = None
        self._load()

    # --- the record -----------------------------------------------------------------

    def _load(self) -> None:
        # Asked before the read: one that will not decode moves the file aside (quarantine).
        existed = os.path.exists(SUBSCRIPTION_FILE_PATH)
        data = IOManager.read_json_gzip(SUBSCRIPTION_FILE_PATH, self.cog.fernet, encrypted=True)
        if isinstance(data, dict) and isinstance(data.get("rows"), dict):
            data.setdefault("grants", {})
            self._data = data
        elif existed:
            self._unreadable = True
            print("Subscriptions: the record would not decode; nothing is written or reconciled "
                  "until it is restored from its .corrupt copy.")
        self._reindex()

    def _reindex(self) -> None:
        self._keys = {row["key"]: uid for uid, row in self._data["rows"].items()
                      if row.get("status") == ACTIVE and row.get("key")}

    def _write(self, change: Callable[[Dict[str, Any]], Any]) -> None:
        """Applies `change` to the record and saves it. Sync: run it through `to_thread`."""
        if self._unreadable:
            # A fresh file here would hold no rows, and the next boot's sweep would read every
            # key on the account as an orphan.
            raise RuntimeError(UNREADABLE)
        with _LOCK:
            change(self._data)
            IOManager.write_json_gzip(self._data, SUBSCRIPTION_FILE_PATH, self.cog.fernet,
                                      encrypted=True)
            self._reindex()

    async def _commit(self, user_id: str, **fields) -> None:
        def change(data):
            row = data["rows"].setdefault(user_id, {})
            row.update(fields)
            for name in [k for k, v in row.items() if v is None]:
                del row[name]
        await asyncio.to_thread(self._write, change)

    # --- reads: the turn path, a dict lookup each -----------------------------------

    @property
    def enabled(self) -> bool:
        user = self.cog.bot.user
        return bool(defaultConfig.OPENROUTER_MANAGEMENT_KEY and user
                    and (user.id == OFFICIAL_BOT_ID or defaultConfig.MIMIC_SUBSCRIPTION_DEV))

    @property
    def billing(self) -> bool:
        """Stripe is wired: Manage, `/subscribe`, the ticket category and the cancelling that account
        deletion does. What anyone already paying needs, so a pause never touches it."""
        return self.enabled and stripe.enabled()

    @property
    def sells(self) -> bool:
        """Subscribe is offered to someone not yet paying: billing, and the owner hasn't paused sales.
        Gate offers on this and service on `billing`: a pause must not strand a subscriber."""
        return self.billing and not self.paused

    @property
    def paused(self) -> bool:
        return bool(self._data.get("paused"))

    async def set_paused(self, paused: bool) -> Optional[str]:
        """The owner's pause on new sales, from `/mod`. None when saved, else why not."""
        if self._unreadable:
            return UNREADABLE
        await asyncio.to_thread(self._write, lambda d: d.__setitem__("paused", paused))
        return None

    @property
    def key_name(self) -> str:
        """What this instance names its keys, and so the only keys its sweep may delete. Any
        bot but the official one adds its own ID, so a dev instance on the same OpenRouter
        account never deletes the official instance's keys, nor they its."""
        user_id = self.cog.bot.user.id
        return SUBSCRIPTION_KEY_NAME if user_id == OFFICIAL_BOT_ID else f"{SUBSCRIPTION_KEY_NAME} (dev {user_id})"

    def row(self, user_id: int) -> Optional[Dict[str, Any]]:
        row = self._data["rows"].get(str(user_id))
        return dict(row) if row else None

    def everyone(self) -> Dict[str, Dict[str, Any]]:
        """Discord ID -> a copy of their row, active or lapsed, for the owner's list."""
        return {uid: dict(row) for uid, row in self._data["rows"].items()}

    def key_for(self, user_id: int) -> Optional[str]:
        """The user's subscription key while it is active, else None."""
        if not self.enabled:
            return None
        row = self._data["rows"].get(str(user_id))
        return row.get("key") if row and row.get("status") == ACTIVE else None

    def is_key(self, api_key: Optional[str]) -> bool:
        return bool(api_key) and api_key in self._keys

    def active(self, user_id: int) -> bool:
        return self.key_for(user_id) is not None

    def grant_of(self, user_id: int) -> Optional[Dict[str, Any]]:
        grant = self._data["grants"].get(str(user_id))
        return dict(grant) if grant else None

    def export_row(self, user_id: int) -> Optional[Dict[str, Any]]:
        """What `/data`'s export carries: never the key or its hash."""
        row = self.row(user_id)
        return {k: row[k] for k in ("tier", "source", "status", "since", "ends", "renews", "limit", "usage",
                                    "base", "synced") if k in row} if row else None

    # --- the owner's grants ---------------------------------------------------------

    async def grant(self, user_id: int, tier: str) -> Optional[str]:
        """Grants `tier` and reconciles that user. None when done, else why not."""
        if not self.enabled or self._unreadable:
            return UNREADABLE if self._unreadable else OFF
        if tier not in SUBSCRIPTION_TIERS:
            return f"No tier named {tier}."
        await asyncio.to_thread(self._write, lambda d: d["grants"].__setitem__(
            str(user_id), {"tier": tier, "since": int(time.time())}))
        return await self.reconcile(user_id)

    async def revoke(self, user_id: int) -> Optional[str]:
        if not self.enabled or self._unreadable:
            return UNREADABLE if self._unreadable else OFF
        await asyncio.to_thread(self._write, lambda d: d["grants"].pop(str(user_id), None))
        return await self.reconcile(user_id)

    def drop_user(self, user_id: int) -> None:
        """Forgets the user: their row and grant. Their key, which nothing names any more, goes
        on the next full reconcile's sweep. Sync, beside each `tickets.drop_user`."""
        uid = str(user_id)
        if self._unreadable or (uid not in self._data["rows"] and uid not in self._data["grants"]):
            return
        self._write(lambda d: (d["rows"].pop(uid, None), d["grants"].pop(uid, None)))

    async def cancel_payments(self, user_id: int) -> None:
        """Cancels each of the user's Stripe subscriptions at once, so a deleted account is
        never billed again: found by the Discord ID they carry, so one paid for but not yet
        reconciled goes too. Raises StripeError, and deletion stops rather than leave one."""
        if not self.billing:
            return
        for sub in await stripe.subscriptions():
            if (sub.get("metadata") or {}).get("discord_id") == str(user_id):
                await stripe.cancel(sub["id"])

    # --- reconcile --------------------------------------------------------------------

    async def reconcile(self, user_id: Optional[int] = None) -> Optional[str]:
        """Makes the keys on OpenRouter match who is entitled: one user, or (None) everyone,
        with the orphan sweep and the float check. None when done, else why it stopped.

        Level-triggered, so it is safe to run twice or after a crash. A source or call that
        fails stops the run where it is: nothing is read as "nobody is entitled".
        """
        if not self.enabled:
            return OFF
        if self._unreadable:
            return UNREADABLE
        async with self._reconciling:
            try:
                await self._reconcile(user_id)
            except (KeyApiError, StripeError) as e:
                self.last_error = str(e)
                print(f"Subscriptions: reconcile stopped: {e}")
                who = "Stripe" if isinstance(e, StripeError) else "OpenRouter"
                return f"{who} didn't answer ({e}). Nothing else was changed; try again."
        return None

    async def _entitled(self, now: int) -> Dict[str, Dict[str, Any]]:
        """Discord ID -> {tier, source, period, renews, ref, ends}, from every source; where one person is
        entitled twice, the larger allowance. Raises StripeError if Stripe can't be listed."""
        found: Dict[str, Dict[str, Any]] = {}

        def offer(uid: str, claim: Dict[str, Any]) -> None:
            held = found.get(uid)
            if held:
                print(f"Subscriptions: {uid} is entitled twice; the larger allowance counts.")
            if not held or SUBSCRIPTION_TIERS[claim["tier"]] > SUBSCRIPTION_TIERS[held["tier"]]:
                found[uid] = claim

        for uid, g in self._data["grants"].items():
            if g.get("tier") in SUBSCRIPTION_TIERS:
                since = int(g.get("since") or now)
                period = since + (now - since) // GRANT_PERIOD * GRANT_PERIOD
                offer(uid, {"tier": g["tier"], "source": "grant", "period": period,
                            "renews": period + GRANT_PERIOD})
        if self.billing:  # never `sells`: a pause would lapse every paying subscriber
            # ponytail: lists every subscription even for one user's Check; fine to a few
            # hundred subscribers, then fetch that user's by the ID in their row.
            for sub in await stripe.subscriptions():
                uid = (sub.get("metadata") or {}).get("discord_id")
                item = ((sub.get("items") or {}).get("data") or [{}])[0]
                tier = (item.get("price") or {}).get("lookup_key")
                if sub.get("status") in stripe.ENTITLED and uid and tier in SUBSCRIPTION_TIERS:
                    # Cancelled in the Portal, it stays paid for, and active, until this.
                    ends = sub.get("cancel_at") or (
                        item.get("current_period_end") if sub.get("cancel_at_period_end") else None)
                    offer(uid, {"tier": tier, "source": "stripe",
                                "period": int(item.get("current_period_start") or 0),
                                "renews": int(item.get("current_period_end") or 0) or None,
                                "ref": {"customer": sub.get("customer"), "subscription": sub["id"]},
                                "ends": int(ends) if ends else None})
        return found

    async def _reconcile(self, user_id: Optional[int]) -> None:
        now = int(time.time())
        entitled = await self._entitled(now)
        rows = self._data["rows"]
        scope = {str(user_id)} if user_id is not None else set(entitled) | set(rows)
        for uid in sorted(scope):
            got = entitled.get(uid)
            row = rows.get(uid) or {}
            if got and not await asyncio.to_thread(self.cog.user_settings.is_registered, int(uid)):
                print(f"Subscriptions: {uid} is entitled but has no account, so has no key.")
                got = None
            tier = got["tier"] if got else None
            terms = {k: got.get(k) for k in ("tier", "source", "period", "renews", "ref", "ends")} if got else {}
            if got and not row.get("hash"):
                # Key first, row second: a crash orphans a key, which the sweep removes,
                # and never records one that does not exist.
                made = await openrouter_keys.create(self.key_name, SUBSCRIPTION_TIERS[tier])
                usage = float(made.get("usage") or 0)
                await self._commit(uid, **terms, key=made["key"], hash=made["hash"],
                                   limit=SUBSCRIPTION_TIERS[tier], usage=usage, base=usage,
                                   synced=now, status=ACTIVE, since=now)
            elif got and any(row.get(k) != v for k, v in terms.items()):
                limit, fresh = row.get("limit", 0.0), {}
                if row.get("period") != got["period"] or row.get("tier") != tier:
                    # The cap is set outright from the row's, never added to the key's, so a
                    # crash before the commit sets the same cap again on the next run.
                    usage = await openrouter_keys.usage(row["hash"])
                    fresh = {"usage": usage, "synced": now}
                    if row.get("period") != got["period"]:
                        limit = carried_limit(limit, usage, SUBSCRIPTION_TIERS[tier])
                        fresh["base"] = usage
                    else:  # a tier change within a period: the difference, now
                        limit = max(usage, limit + SUBSCRIPTION_TIERS[tier]
                                    - SUBSCRIPTION_TIERS.get(row.get("tier"), 0.0))
                    await openrouter_keys.set_limit(row["hash"], limit)
                await self._commit(uid, **terms, limit=limit, **fresh)
            elif not got and row.get("hash"):
                await openrouter_keys.delete(row["hash"])
                await self._commit(uid, key=None, hash=None, status=LAPSED, since=now)
        if user_id is None:
            await self._sweep(now)

    async def _sweep(self, now: int) -> None:
        """Refreshes every row's spend, deletes keys no row holds, and checks the float."""
        listed = await openrouter_keys.list_subscription_keys(self.key_name)
        held = {row["hash"]: uid for uid, row in self._data["rows"].items() if row.get("hash")}
        spend = {}
        for key in listed:
            uid = held.get(key.get("hash"))
            if uid is None:
                await openrouter_keys.delete(key["hash"])
                print("Subscriptions: deleted an orphaned key.")
            else:
                spend[uid] = (float(key.get("usage") or 0), key.get("limit"))

        def change(data):
            for uid, (usage, limit) in spend.items():
                row = data["rows"].get(uid)
                if row:
                    row.update(usage=usage, synced=now)
                    if limit is not None:
                        row["limit"] = float(limit)
        if spend:
            await asyncio.to_thread(self._write, change)

        # Every subscriber spends from the operator's one balance: if it runs out, all stop.
        credit = await openrouter_keys.credit()
        unused = sum(max(0.0, row.get("limit", 0) - row.get("usage", 0))
                     for row in self._data["rows"].values() if row.get("status") == ACTIVE)
        if credit < unused:
            print(f"Subscriptions: the account holds US${credit:.2f}, below the "
                  f"US${unused:.2f} of allowance not yet spent.")
        self.credit, self.unused, self.last_run, self.last_error = credit, unused, now, None

    @tasks.loop(hours=1.0)
    async def hourly_reconcile_task(self):
        if self.cog.has_lock and self.enabled:
            await self.reconcile()
