"""Which event loop the bot runs on: uvloop where it is installed, asyncio otherwise.

uvloop is libuv under the asyncio API. The bot's own Python costs the same on either;
what it takes off the loop thread is the loop's share -- selecting on sockets, waking
tasks, moving bytes through transports -- which on 0.25 vCPU with a gateway connection
per child bot is not nothing. Nothing in the codebase depends on which loop it is:
`add_signal_handler`, `to_thread` and `helpers.Timeout`'s SIGALRM all behave the same
(tests/test_event_loop.py).

Optional, never required: absent, or on Windows, the bot runs on asyncio as it always
did. MIMIC_UVLOOP=0 turns it off without uninstalling it, for telling whether a
problem is the loop's.

Falling back is silent by nature -- the bot runs either way -- so the boot line says
which reason it was. A box that has been running since before uvloop was a dependency,
and one carrying a distribution package too old to use, both read as "asyncio" and are
fixed by different commands.
"""

import asyncio
import os
import platform
import sys
from typing import Any, Coroutine, Optional, Tuple

_OFF = ("0", "false", "no", "off")


def _uvloop_choice() -> Tuple[Optional[Any], str]:
    """The uvloop module to run on and why, or None and why not, in the words the
    boot line prints."""
    if (os.getenv("MIMIC_UVLOOP") or "1").strip().lower() in _OFF:
        return None, "turned off by MIMIC_UVLOOP"
    if platform.system() == "Windows":
        return None, "no uvloop build exists for Windows"
    try:
        import uvloop  # type: ignore
    except ImportError:
        return None, "uvloop is not installed (pip install -r requirements.txt)"
    version = getattr(uvloop, "__version__", "?")
    if not hasattr(uvloop, "run"):
        # uvloop.run arrived in 0.18; an older install is treated as absent rather than
        # half-used through the deprecated policy API. Debian's own package is one of
        # them, so this is what a distribution install reads as.
        return None, f"uvloop {version} is older than 0.18 (pip install -U uvloop)"
    return uvloop, f"uvloop {version}"


def _uvloop() -> Optional[Any]:
    return _uvloop_choice()[0]


def loop_name() -> str:
    return "uvloop" if _uvloop() else "asyncio"


def run(main: Coroutine) -> Any:
    """asyncio.run, on uvloop when it is available. Prints what it chose and why, and
    the Python it chose it on, so the journal answers all three without a shell on the
    box."""
    uvloop, reason = _uvloop_choice()
    print(f"Python {platform.python_version()} ({sys.implementation.name}), "
          f"event loop: {reason if uvloop else 'asyncio -- ' + reason}", flush=True)
    if uvloop:
        return uvloop.run(main)
    return asyncio.run(main)
