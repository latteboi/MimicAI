"""glibc allocator tuning for the e2-micro deployment target.

The bot's resident set ratchets upward after any round that touches a
multi-megabyte attachment and never comes back down, even though Python has
long since freed the buffers. That is not a leak — it is glibc declining to
return the pages:

1.  **The dynamic mmap threshold.** glibc services allocations above
    `M_MMAP_THRESHOLD` (128 KB by default) with `mmap`, and `munmap`s them on
    free, so they cost nothing once released. But the *first* time such a block
    is freed, glibc raises the threshold to that block's size (capped at 32 MB)
    and the trim threshold to twice it, on the theory that a workload which
    allocates 6 MB once will do it again and should be spared the syscalls. Every
    subsequent large buffer then comes from the arena via `brk`, below a trim
    threshold it will never reach. One 6 MB image is enough to arm this.

2.  **Per-thread arenas.** `IOManager` reads/writes and the File API upload all
    run under `asyncio.to_thread`, and glibc hands each thread its own arena —
    up to `8 * ncores` of them. An arena is only trimmed from its top, so
    interleaved allocations strand freed space below the high-water mark in
    several arenas at once.

`mallopt` fixes both, and setting `M_MMAP_THRESHOLD` explicitly has the useful
side effect of switching off the dynamic adjustment in (1) permanently
(`no_dyn_threshold`). Prefer the `MALLOC_*` environment variables in the systemd
unit where possible — those apply from process start, before the interpreter has
allocated anything — this module is the fallback that also covers existing
installs and non-systemd runs.

Every function here is a no-op off glibc, so macOS and musl development boxes are
unaffected.
"""

import ctypes
import ctypes.util
import os
import platform
import time
from typing import Optional

# From glibc's malloc.h. Negative values, hence the explicit constants.
_M_TRIM_THRESHOLD = -1
_M_MMAP_THRESHOLD = -3
_M_ARENA_MAX = -8

# 128 KB: glibc's own default, pinned so the dynamic adjustment cannot raise it.
_MMAP_THRESHOLD_BYTES = 128 * 1024
_TRIM_THRESHOLD_BYTES = 128 * 1024

# 0.25 vCPU baseline. More arenas buy contention relief the bot cannot use and
# cost fragmentation it cannot afford.
_ARENA_MAX = 2

_libc: Optional[ctypes.CDLL] = None
_tuned = False
_last_trim = 0.0


def _get_libc() -> Optional[ctypes.CDLL]:
    """Returns a handle to glibc, or None if this is not a glibc platform."""
    global _libc
    if _libc is not None:
        return _libc

    try:
        if platform.system() != "Linux":
            return None
        # libc_ver() reports ('glibc', '2.36') on glibc and ('', '') on musl.
        if platform.libc_ver()[0] != "glibc":
            return None
        name = ctypes.util.find_library("c") or "libc.so.6"
        candidate = ctypes.CDLL(name, use_errno=True)
        if not hasattr(candidate, "mallopt"):
            return None
        candidate.mallopt.argtypes = [ctypes.c_int, ctypes.c_int]
        candidate.mallopt.restype = ctypes.c_int
        candidate.malloc_trim.argtypes = [ctypes.c_size_t]
        candidate.malloc_trim.restype = ctypes.c_int
    except Exception:
        return None

    _libc = candidate
    return _libc


def tune_allocator() -> bool:
    """Applies the mallopt settings. Call once, as early in startup as possible
    and before any worker threads exist — `M_ARENA_MAX` only governs arenas not
    yet created. Returns True if the settings were applied.
    """
    global _tuned
    if _tuned:
        return True

    libc = _get_libc()
    if libc is None:
        return False

    try:
        # An env var, if the operator set one, is authoritative and was applied
        # before the interpreter started; mallopt would silently override it.
        if not os.environ.get("MALLOC_MMAP_THRESHOLD_"):
            libc.mallopt(_M_MMAP_THRESHOLD, _MMAP_THRESHOLD_BYTES)
        if not os.environ.get("MALLOC_TRIM_THRESHOLD_"):
            libc.mallopt(_M_TRIM_THRESHOLD, _TRIM_THRESHOLD_BYTES)
        if not os.environ.get("MALLOC_ARENA_MAX"):
            libc.mallopt(_M_ARENA_MAX, _ARENA_MAX)
    except Exception as e:
        print(f"[memory] Allocator tuning unavailable: {e}")
        return False

    _tuned = True
    return True


def trim_malloc() -> bool:
    """Releases free memory at the top of every arena back to the OS.

    Cheap in a way `gc.collect()` is not: it walks the allocator's own free
    lists, never Python's object graph, so there is no per-object cost and no
    generational traversal. Still holds the GIL for its duration, so it belongs
    at the end of a large transfer, not in a per-message path.
    """
    libc = _get_libc()
    if libc is None:
        return False
    try:
        libc.malloc_trim(0)
        return True
    except Exception:
        return False


def maybe_trim_malloc(min_interval: float = 30.0) -> bool:
    """Rate-limited `trim_malloc`, for call sites that fire several times in a
    round — a four-participant turn sharing one attachment should trim once, not
    four times."""
    global _last_trim
    now = time.monotonic()
    if now - _last_trim < min_interval:
        return False
    _last_trim = now
    return trim_malloc()
