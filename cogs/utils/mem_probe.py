"""Opt-in memory attribution for the paths that spike.

Switched on with `MIMIC_MEM_PROBE=1` (phase totals) or `MIMIC_MEM_PROBE=full`
(phase totals plus the top allocation sites for each phase). Off, every function
here is a branch on a module-level bool and nothing else, so the probes can live
permanently at the phase boundaries rather than being added and removed.

It reports two numbers per phase, and the gap between them is the point:

- **Python heap**, from `tracemalloc`: what the interpreter allocated, attributable
  to a source line.
- **RSS**, from the OS: what the process actually holds. Native allocations --
  OpenSSL buffers, zlib/zstd, a first-time numpy or Pillow import, glibc arenas
  that were freed but not returned -- appear here and nowhere else.

A phase where RSS jumps and the Python heap does not is *not* a Python problem, and
no amount of dropping references will fix it. That distinction is why both numbers
are here; chasing the wrong one costs days.

Note for anyone profiling on macOS or musl: `memory_tuning` is a no-op off glibc.
`M_MMAP_THRESHOLD`, `M_ARENA_MAX` and `malloc_trim` are what keep the resident set
from ratcheting on the deployment target, and none of them exist on a Mac. RSS
measured on a dev box is not a prediction of RSS on the e2-micro.
"""

import ctypes
import os
import platform
import sys
import tracemalloc
from contextlib import contextmanager
from typing import Optional

_MODE = (os.environ.get("MIMIC_MEM_PROBE") or "").strip().lower()
ENABLED = _MODE in ("1", "true", "yes", "on", "full")
_FULL = _MODE == "full"

#: Allocation sites to list per phase in `full` mode.
_TOP_N = 12

_started = False
_libproc = None


def _rss_bytes() -> Optional[int]:
    """Current resident set size, or None where it cannot be read cheaply."""
    if sys.platform.startswith("linux"):
        try:
            # statm field 2 is resident pages. Cheaper than /proc/self/status,
            # which formats a dozen fields we do not want.
            with open("/proc/self/statm", "rb") as f:
                return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except Exception:
            return None

    if sys.platform == "darwin":
        # ru_maxrss is a high-water mark, so per-phase deltas would read zero after
        # the first spike. libproc gives the current value.
        global _libproc
        try:
            if _libproc is None:
                lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
                # Declared explicitly: the third argument is a uint64 and the
                # default int marshalling gets the call wrong on arm64.
                lib.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int,
                                             ctypes.c_uint64, ctypes.c_void_p,
                                             ctypes.c_int]
                lib.proc_pidinfo.restype = ctypes.c_int
                _libproc = lib
            # 256, not sizeof(struct proc_taskinfo): the call refuses anything
            # under 128 outright and only writes the 96 bytes it has.
            buf = ctypes.create_string_buffer(256)
            # PROC_PIDTASKINFO == 4; pti_virtual_size then pti_resident_size, both
            # uint64, are the first two fields of struct proc_taskinfo.
            n = _libproc.proc_pidinfo(os.getpid(), 4, 0, buf, 256)
            if n >= 16:
                return int.from_bytes(buf.raw[8:16], sys.byteorder)
        except Exception:
            return None
    return None


def start() -> None:
    """Begins tracing. Call once at startup; safe to call again."""
    global _started
    if not ENABLED or _started:
        return
    # 8 frames: enough to tell _build_history_for_participant's allocations from
    # _construct_system_instructions', which one frame cannot.
    tracemalloc.start(8 if _FULL else 1)
    _started = True
    print(f"[mem] probe enabled ({_MODE}); platform={platform.system()} "
          f"allocator tuning={'glibc' if platform.libc_ver()[0] == 'glibc' else 'unavailable'}")


def _fmt(n: Optional[float]) -> str:
    if n is None:
        return "  n/a  "
    sign = "+" if n >= 0 else "-"
    return f"{sign}{abs(n) / 1e6:7.2f} MB"


@contextmanager
def probe(label: str, peak: bool = True):
    """Wraps one phase. Reports the Python-heap peak *within* the phase, the heap
    it kept, and the RSS it kept.

    `peak=False` for a phase that contains other probes. `tracemalloc.reset_peak()`
    is process-global, so a nested probe silently truncates its parent's peak to
    "everything after the child finished" -- which reads as a small number and hides
    exactly what is being looked for. An enclosing probe therefore measures what it
    *kept* and leaves the peaks to the phases inside it.
    """
    if not ENABLED:
        yield
        return

    start()
    rss_before = _rss_bytes()
    heap_before, _ = tracemalloc.get_traced_memory()
    if peak:
        tracemalloc.reset_peak()
    snap_before = tracemalloc.take_snapshot() if _FULL else None
    try:
        yield
    finally:
        heap_after, heap_peak = tracemalloc.get_traced_memory()
        rss_after = _rss_bytes()
        kept_rss = None if (rss_before is None or rss_after is None) else rss_after - rss_before
        peak_col = _fmt(heap_peak - heap_before) if peak else "    --   "
        print(f"[mem] {label:<34} heap peak {peak_col}"
              f"   heap kept {_fmt(heap_after - heap_before)}"
              f"   rss kept {_fmt(kept_rss)}")

        if _FULL and snap_before is not None:
            snap_after = tracemalloc.take_snapshot()
            diff = snap_after.compare_to(snap_before, 'lineno')
            shown = 0
            for stat in diff:
                if stat.size_diff <= 0:
                    continue
                frame = stat.traceback[0]
                path = frame.filename
                # Trim to something readable without losing which package it is in.
                for marker in ("/cogs/", "/site-packages/", "/lib/"):
                    idx = path.find(marker)
                    if idx != -1:
                        path = path[idx + 1:]
                        break
                print(f"[mem]     {stat.size_diff / 1e6:7.2f} MB  {path}:{frame.lineno}")
                shown += 1
                if shown >= _TOP_N:
                    break
            del snap_after, diff

async def reporter(interval: float = 5.0):
    """Prints a heap/RSS line every `interval` seconds, and in `full` mode the
    allocation sites that grew since the previous tick.

    The phase probes above only see the boundaries someone thought to instrument.
    This sees everything, which is what is wanted when the phase that grows is not
    yet known -- and its ticks line up with what Activity Monitor or `top` is
    showing, so a spike on screen can be matched to a source line.
    """
    if not ENABLED:
        return
    import asyncio

    start()
    prev_snapshot = tracemalloc.take_snapshot() if _FULL else None
    prev_rss = _rss_bytes()
    prev_heap, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()

    while True:
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return

        heap, heap_peak = tracemalloc.get_traced_memory()
        rss = _rss_bytes()
        d_rss = None if (rss is None or prev_rss is None) else rss - prev_rss
        print(f"[mem] tick  heap {heap / 1e6:7.2f} MB (peak {heap_peak / 1e6:7.2f} MB, "
              f"{_fmt(heap - prev_heap)})   rss "
              f"{'  n/a  ' if rss is None else f'{rss / 1e6:7.2f} MB'} ({_fmt(d_rss)})")

        if _FULL and prev_snapshot is not None:
            snapshot = tracemalloc.take_snapshot()
            shown = 0
            for stat in snapshot.compare_to(prev_snapshot, 'lineno'):
                if stat.size_diff <= 262144:  # 256 KB: below this it is noise
                    break
                frame = stat.traceback[0]
                path = frame.filename
                for marker in ("/cogs/", "/site-packages/", "/lib/"):
                    idx = path.find(marker)
                    if idx != -1:
                        path = path[idx + 1:]
                        break
                print(f"[mem]     grew {stat.size_diff / 1e6:7.2f} MB  {path}:{frame.lineno}")
                shown += 1
                if shown >= _TOP_N:
                    break
            prev_snapshot = snapshot

        prev_rss, prev_heap = rss, heap
        tracemalloc.reset_peak()
