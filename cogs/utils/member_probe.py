"""Member-cache footprint probe for the e2-micro deployment target.

`intents.members` with discord.py's default `chunk_guilds_at_startup` means the
bot requests every member of every guild at connect and keeps them all. Nothing in
this codebase listens to a member event -- the intent exists solely to populate
that cache, which is read by a handful of `guild.get_member()` point lookups.

Whether that trade is worth reversing depends entirely on a number nobody has:
how many members are actually cached. At roughly 550 bytes each, 15,000 members is
an 8 MB cache not worth touching and 400,000 is a fifth of the box. The estimate
is worthless; the measurement decides.

So this reports the count and calibrates the per-member cost in-process, on the
running architecture and interpreter, rather than trusting a constant measured
somewhere else. Calibration builds throwaway Members against a private state
object -- the bot's own user and member caches are never written to.

Off unless MIMIC_MEMBER_PROBE is set. It samples once and stops.

    MIMIC_MEMBER_PROBE=1            enable
    MIMIC_MEMBER_PROBE_DELAY=30     seconds after on_ready before sampling
    MIMIC_MEMBER_PROBE_SAMPLE=4000  members built to calibrate per-member cost

Output goes to stdout, so it lands in
`journalctl -u mimicai | grep member-probe`.
"""

import asyncio
import gc
import os
import resource
import sys
from typing import Any, Dict, Optional

import discord

from .loop_probe import rss_bytes
from .memory_tuning import trim_malloc

# Bytes of dict machinery per entry in guild._members, which sys.getsizeof cannot
# attribute to the Member the entry points at. A CPython dict at typical load
# reserves an index slot plus a three-word entry; the structural floor below is a
# floor precisely because small change like this is invisible to getsizeof.
_DICT_SLOT_BYTES = 104

_started = False


def _truthy(val: Optional[str]) -> bool:
    return bool(val) and val.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except (TypeError, ValueError):
        return default


def _rss() -> tuple:
    """(bytes, is_current). /proc reports the current set; getrusage a peak.

    The distinction is not pedantry: the calibration below infers a per-member cost
    from an RSS delta, and a high-water mark does not move when a process that has
    already peaked higher allocates a few more megabytes. On Darwin the calibration
    is therefore expected to read zero and hand over to the structural floor.
    """
    value = rss_bytes()
    if value:
        return value, True
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, Darwin and the BSDs report bytes.
    return (raw if sys.platform == "darwin" else raw * 1024), False


def _mb(n: float) -> str:
    return f"{n / (1024 * 1024):.1f} MB"


class _CalibrationState:
    """Enough of ConnectionState to construct a Member, and nothing shared.

    Deliberately not the bot's own state: `store_user` writes into the real user
    cache, and a probe must not leave synthetic users in it even briefly.
    """

    def __init__(self) -> None:
        self._users: Dict[int, Any] = {}
        self.member_cache_flags = discord.MemberCacheFlags.all()

    def store_user(self, data, cache: bool = True):
        user_id = int(data["id"])
        user = self._users.get(user_id)
        if user is None:
            user = discord.User(state=self, data=data)
            self._users[user_id] = user
        return user

    def get_user(self, user_id):
        return self._users.get(user_id)

    def _get_guild(self, guild_id):
        return None


def _average_role_count(bot, ceiling: int = 2000) -> float:
    """Roles per cached member, so calibration builds members the shape of yours."""
    seen = total = 0
    for guild in bot.guilds:
        for member in guild._members.values():
            total += len(getattr(member, "_roles", ()) or ())
            seen += 1
            if seen >= ceiling:
                return total / seen
    return total / seen if seen else 3.0


def _structural_bytes(bot, ceiling: int = 2000) -> float:
    """A floor for per-member cost, from sys.getsizeof over real cached members.

    Used when the RSS calibration is unreadable, which is the normal case on
    Darwin: getrusage reports a high-water mark there, so allocating a few
    megabytes into a process that has already peaked higher moves nothing.

    A floor, and reported as one. getsizeof cannot see allocator overhead, and the
    User behind a Member is shared with every other guild that member is in, so
    counting it per membership would inflate the one figure here that has to be
    trustworthy. Prefer the measurement; fall back to this rather than to a guess.
    """
    seen = total = 0
    for guild in bot.guilds:
        for member in guild._members.values():
            total += sys.getsizeof(member) + _DICT_SLOT_BYTES
            for slot in ("_roles", "nick", "_avatar"):
                value = getattr(member, slot, None)
                if value is not None:
                    total += sys.getsizeof(value)
            user = getattr(member, "_user", None)
            if user is not None:
                total += sys.getsizeof(user)
            seen += 1
            if seen >= ceiling:
                return total / seen
    return total / seen if seen else 0.0


def _calibrate(guild, sample: int, roles_each: int) -> float:
    """Bytes of RSS per Member, measured by building `sample` of them."""
    state = _CalibrationState()
    role_ids = [str(1_000_000_000_000_000 + i) for i in range(max(0, roles_each))]

    gc.collect()
    trim_malloc()
    before, _ = _rss()

    held = []
    try:
        for i in range(sample):
            held.append(discord.Member(
                state=state,
                guild=guild,
                data={
                    "user": {
                        "id": str(700_000_000_000_000_000 + i),
                        "username": f"probeuser{i}",
                        "discriminator": "0",
                        "global_name": f"Probe User {i}",
                        "avatar": "0" * 32,
                        "public_flags": 0,
                    },
                    "roles": role_ids,
                    "joined_at": "2023-01-01T00:00:00.000000+00:00",
                    "deaf": False, "mute": False, "flags": 0,
                    "nick": None, "avatar": None, "premium_since": None, "pending": False,
                },
            ))
        gc.collect()
        after, _ = _rss()
    finally:
        del held
        del state
        gc.collect()
        trim_malloc()

    if not before or not after or after <= before:
        return 0.0
    return (after - before) / sample


async def _report(bot, delay: int, sample: int) -> None:
    await asyncio.sleep(delay)

    guilds = list(bot.guilds)
    if not guilds:
        print("[member-probe] no guilds; nothing to measure.")
        return

    reported = sum(g.member_count or 0 for g in guilds)
    cached = sum(len(g._members) for g in guilds)
    unique = len({uid for g in guilds for uid in g._members})
    largest = max(guilds, key=lambda g: len(g._members))
    roles_each = _average_role_count(bot)

    rss, rss_is_current = _rss()

    # Only calibrate where RSS is the current set. On a high-water counter the delta
    # is whatever the allocation happened to push past the previous peak, which is
    # some arbitrary fraction of the real cost -- a Darwin run reported 160 B/member
    # against a true ~700 B, and reported it as measured. Understating by 4x with
    # confidence is worse than declining to answer, so this declines.
    per_member = _calibrate(guilds[0], min(sample, 20000), round(roles_each)) if rss_is_current else 0.0
    measured = per_member > 0
    if not measured:
        per_member = _structural_bytes(bot)

    footprint = cached * per_member
    share = (footprint / rss * 100) if rss else 0.0
    if measured:
        basis = "measured here"
    elif rss_is_current:
        basis = "structural floor, real cost is higher"
    else:
        basis = "structural floor -- no current-RSS source to calibrate against"
    bound = "" if measured else "at least "

    print("[member-probe] ----------------------------------------------")
    print(f"[member-probe] guilds                {len(guilds)}")
    print(f"[member-probe] members reported      {reported:,}   (what Discord says the guilds hold)")
    print(f"[member-probe] members cached        {cached:,}   (what this process is holding)")
    print(f"[member-probe] unique users          {unique:,}")
    print(f"[member-probe] largest guild cache   {len(largest._members):,}")
    print(f"[member-probe] roles per member      {roles_each:.1f}")
    print(f"[member-probe] bytes per member      {per_member:.0f}   ({basis})")
    print(f"[member-probe] member cache          {bound}~{_mb(footprint)}")
    print(f"[member-probe] process RSS           {_mb(rss)}"
          f"{'' if rss_is_current else '   (peak, not current -- no /proc here)'}")
    print(f"[member-probe] cache is              ~{share:.0f}% of resident memory")
    print("[member-probe] ----------------------------------------------")

    if cached < 500:
        print(f"[member-probe] NOTE: only {cached} members cached across {len(guilds)} guild(s). "
              "That is a dev instance,")
        print("[member-probe]       not a population. The verdict below describes THIS "
              "process and says")
        print("[member-probe]       nothing about production -- rerun there for the "
              "answer that decides.")

    # The decision this exists to answer. Thresholds are deliberately wide: the
    # question is order of magnitude, and the work it gates is a re-architecture
    # of every `guild.get_member()` call, not a config change.
    if footprint < 25 * 1024 * 1024:
        verdict = ("NOT WORTH IT -- the cache is smaller than the work to remove it. "
                   "Keep chunking on.")
    elif footprint < 75 * 1024 * 1024:
        verdict = ("MARGINAL -- worth doing only if RSS is already the binding "
                   "constraint. Check the loop probe for swap or OOM pressure first.")
    else:
        verdict = ("WORTH IT -- this is the largest single reclaimable allocation in "
                   "the process. See the staged plan before touching chunking: "
                   "get_member() reads None for everyone once it is off.")
    print(f"[member-probe] verdict: {verdict}")


def start_member_probe(bot) -> Optional[asyncio.Task]:
    """Starts the one-shot probe if MIMIC_MEMBER_PROBE is set. Returns the task.

    on_ready fires again on every resume, so this refuses to start twice. The
    caller keeps the reference: an unreferenced task can be collected mid-flight.
    """
    global _started
    if _started or not _truthy(os.getenv("MIMIC_MEMBER_PROBE")):
        return None

    _started = True
    delay = max(0, _env_int("MIMIC_MEMBER_PROBE_DELAY", 30))
    sample = max(100, _env_int("MIMIC_MEMBER_PROBE_SAMPLE", 4000))
    print(f"[member-probe] enabled; sampling in {delay}s with {sample} calibration members.")
    return asyncio.create_task(_report(bot, delay, sample))
