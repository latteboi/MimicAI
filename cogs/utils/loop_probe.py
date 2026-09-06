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

Off unless MIMIC_LOOP_PROBE is set. The point of the deployment target is not to
pay for what is not being used, and a permanent 10 Hz wakeup is a cost even when
it is a small one.

    MIMIC_LOOP_PROBE=1                 enable
    MIMIC_LOOP_PROBE_INTERVAL=0.1      seconds between samples
    MIMIC_LOOP_PROBE_THRESHOLD=0.25    log a single stall at or above this
    MIMIC_LOOP_PROBE_SUMMARY=300       seconds between summary lines

Output goes to stdout, which systemd captures, so it lands in
`journalctl -u mimicai | grep loop-probe`.
"""

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

_PAGE_SIZE = 4096
try:
    _PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
except (ValueError, OSError, AttributeError):
    pass

_state: Dict[str, Any] = {
    "enabled": False,
    "samples": 0,
    "stalls": 0,
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
        "max_lag": _state["max_lag"],
        "max_lag_at": _state["max_lag_at"],
        "rss": rss_bytes(),
        "rss_peak": _state["rss_peak"],
    }


async def _probe(interval: float, threshold: float, summary_every: float) -> None:
    window_started = time.monotonic()

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
            rss = rss_bytes()
            if rss > _state["rss_peak"]:
                _state["rss_peak"] = rss
            # rss_bytes answers 0 off /proc. Printing "0.0 MB" would read as a
            # measurement; leaving the field out reads as what it is.
            suffix = f"  rss {_mb(rss)}" if rss else ""
            print(f"[loop-probe] stall {_ms(lag)}{suffix}", flush=True)

        now = time.monotonic()
        if now - window_started < summary_every:
            continue

        window = sorted(_state["window"])
        _state["window"] = []
        elapsed = now - window_started
        window_started = now
        if not window:
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
            f"stalls>={_ms(threshold)} {_state['stalls']} total"
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
    threshold = max(0.0, _env_float("MIMIC_LOOP_PROBE_THRESHOLD", 0.25))
    summary_every = max(interval, _env_float("MIMIC_LOOP_PROBE_SUMMARY", 300.0))

    _state["enabled"] = True
    _state["rss_peak"] = rss_bytes()
    print(
        f"[loop-probe] on: sampling every {_ms(interval)}, "
        f"reporting stalls >= {_ms(threshold)}, summary every {summary_every / 60:.0f}m",
        flush=True,
    )
    return asyncio.create_task(_probe(interval, threshold, summary_every))
