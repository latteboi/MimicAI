"""Event-loop lag and RSS probe for the e2-micro deployment target.

CPU percentage does not answer the question this codebase actually asks. The bot
can average 0.05% CPU across a day and still block the loop for 300 ms while a
1000-turn log is re-serialised, re-compressed and re-encrypted -- and it is the
300 ms that drops a gateway heartbeat, not the average. htop cannot see that.
This measures it directly: sleep a fixed interval, and report how much longer than
that it actually took.

The overshoot is the whole signal. `await asyncio.sleep(0.1)` returning 0.4 s
later means something held the loop thread for ~300 ms, and the only things that
can are synchronous work on it. RSS is sampled alongside, on stalls and at each
summary, so a peak in one can be read against the other.

It reports danger, not activity. A probe that prints every quarter-second hiccup
trains you to skim past it, and the line that mattered goes by in the same colour
as three hundred that did not -- so the thresholds here are pinned to what
discord.py actually does about a blocked loop, rather than to a round number:

    >= 1 s      user-visible. Every message in flight waits this long.
    >= 10 s     discord.py logs `heartbeat blocked for more than 10 seconds`, and
                `Can't keep up, websocket is N.Ns behind`, and is now working
                against you rather than with you.
    >= 60 s     the default `heartbeat_timeout`: the gateway connection is
                declared dead, dropped and resumed. Every child bot shares this
                loop, so they all go together.

A window with no stall in it prints nothing at all. Silence is the report.

Off unless MIMIC_LOOP_PROBE is set. The point of the deployment target is not to
pay for what is not being used, and a permanent 10 Hz wakeup is a cost even when
it is a small one.

    MIMIC_LOOP_PROBE=1                 enable
    MIMIC_LOOP_PROBE_INTERVAL=0.1      seconds between samples
    MIMIC_LOOP_PROBE_THRESHOLD=1.0     log a stall at or above this
    MIMIC_LOOP_PROBE_CRITICAL=10.0     escalate at or above this
    MIMIC_LOOP_PROBE_SUMMARY=300       seconds between summary lines
    MIMIC_LOOP_PROBE_ALWAYS=0          summarise quiet windows too, for baselining

Output goes to stdout, which systemd captures, so it lands in
`journalctl -u mimicai | grep loop-probe`.
"""

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

# discord.py's KeepAliveHandler waits on the loop thread with `f.result(10)` and
# logs its blocked-heartbeat warning per 10 s elapsed; ConnectionState defaults
# `heartbeat_timeout` to 60.0, past which the socket is resumed. Both are read off
# the installed library in discord.py 2.5.2 -- restated here rather than imported
# because neither is exported as a constant.
_HEARTBEAT_BLOCK_WARNING = 10.0
_HEARTBEAT_TIMEOUT = 60.0

_PAGE_SIZE = 4096
try:
    _PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
except (ValueError, OSError, AttributeError):
    pass

_state: Dict[str, Any] = {
    "enabled": False,
    "samples": 0,
    "stalls": 0,
    "critical": 0,
    "max_lag": 0.0,
    "max_lag_at": 0.0,
    "rss_peak": 0,
    "window": [],
}


def _truthy(val: Optional[str]) -> bool:
    return bool(val) and val.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except (TypeError, ValueError):
        return default


def rss_bytes() -> int:
    """Current resident set, or 0 where /proc is not available.

    statm rather than status: two fields to parse instead of fifty, and this is
    read on every stall. Field 1 is resident pages.
    """
    try:
        with open("/proc/self/statm", "rb") as f:
            return int(f.read().split()[1]) * _PAGE_SIZE
    except Exception:
        return 0


def _mb(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB"


def _ms(seconds: float) -> str:
    return f"{seconds * 1000:.0f} ms"


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(round((pct / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[max(0, min(idx, len(sorted_vals) - 1))]


def snapshot() -> Dict[str, Any]:
    """The counters so far, for anything that wants to report them in-process."""
    return {
        "enabled": _state["enabled"],
        "samples": _state["samples"],
        "stalls": _state["stalls"],
        "critical": _state["critical"],
        "max_lag": _state["max_lag"],
        "max_lag_at": _state["max_lag_at"],
        "rss": rss_bytes(),
        "rss_peak": _state["rss_peak"],
    }


async def _probe(interval: float, threshold: float, critical: float,
                 summary_every: float, always: bool) -> None:
    window_started = time.monotonic()
    window_stalls = 0

    while True:
        t0 = time.perf_counter()
        await asyncio.sleep(interval)
        lag = time.perf_counter() - t0 - interval
        if lag < 0.0:
            lag = 0.0

        _state["samples"] += 1
        _state["window"].append(lag)

        if lag > _state["max_lag"]:
            _state["max_lag"] = lag
            _state["max_lag_at"] = time.time()

        if lag >= threshold:
            _state["stalls"] += 1
            window_stalls += 1
            rss = rss_bytes()
            if rss > _state["rss_peak"]:
                _state["rss_peak"] = rss
            # rss_bytes answers 0 off /proc. Printing "0.0 MB" would read as a
            # measurement; leaving the field out reads as what it is.
            suffix = f"  rss {_mb(rss)}" if rss else ""
            if lag >= critical:
                _state["critical"] += 1
                print(f"[loop-probe] CRITICAL stall {_ms(lag)}{suffix} -- discord.py is "
                      f"logging a blocked heartbeat by now; the gateway is dropped and "
                      f"resumed at {_HEARTBEAT_TIMEOUT:.0f}s, taking every child bot with it",
                      flush=True)
            else:
                print(f"[loop-probe] stall {_ms(lag)}{suffix}", flush=True)

        now = time.monotonic()
        if now - window_started < summary_every:
            continue

        window = sorted(_state["window"])
        _state["window"] = []
        elapsed = now - window_started
        window_started = now
        stalled = window_stalls
        window_stalls = 0
        if not window:
            continue

        # The whole point. A window that held nothing above the threshold is the
        # expected state on a healthy bot, and printing p50/p99 for it every five
        # minutes is how the one line that matters gets skimmed past. MIMIC_LOOP_
        # PROBE_ALWAYS brings the periodic line back for establishing a baseline.
        if not stalled and not always:
            continue

        rss = rss_bytes()
        if rss > _state["rss_peak"]:
            _state["rss_peak"] = rss
        mem = f", rss {_mb(rss)} (peak {_mb(_state['rss_peak'])})" if rss else ""
        # p50 alongside p99 because they answer different questions: p50 drifting
        # up is steady load, p99 alone spiking is one long synchronous block.
        print(
            f"[loop-probe] {elapsed / 60:.0f}m: "
            f"p50 {_ms(_percentile(window, 50))}, "
            f"p99 {_ms(_percentile(window, 99))}, "
            f"max {_ms(window[-1])}, "
            f"stalls>={_ms(threshold)} {stalled} here, {_state['stalls']} total"
            f"{', ' + str(_state['critical']) + ' CRITICAL' if _state['critical'] else ''}"
            f"{mem}",
            flush=True,
        )


def start_loop_probe() -> Optional[asyncio.Task]:
    """Starts the probe if MIMIC_LOOP_PROBE is set. Returns the task, or None.

    Must be called with a running loop. The caller keeps the reference: an
    unreferenced task can be collected mid-flight.
    """
    if not _truthy(os.getenv("MIMIC_LOOP_PROBE")):
        return None

    interval = max(0.01, _env_float("MIMIC_LOOP_PROBE_INTERVAL", 0.1))
    threshold = max(0.0, _env_float("MIMIC_LOOP_PROBE_THRESHOLD", 1.0))
    critical = max(threshold, _env_float("MIMIC_LOOP_PROBE_CRITICAL", _HEARTBEAT_BLOCK_WARNING))
    summary_every = max(interval, _env_float("MIMIC_LOOP_PROBE_SUMMARY", 300.0))
    always = _truthy(os.getenv("MIMIC_LOOP_PROBE_ALWAYS"))

    _state["enabled"] = True
    _state["rss_peak"] = rss_bytes()
    print(
        f"[loop-probe] on: sampling every {_ms(interval)}, "
        f"stalls >= {_ms(threshold)}, critical >= {_ms(critical)}, "
        f"{'summary every ' + format(summary_every / 60, '.0f') + 'm' if always else 'silent unless something stalls'}",
        flush=True,
    )
    return asyncio.create_task(_probe(interval, threshold, critical, summary_every, always))
