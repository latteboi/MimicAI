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
"""

import asyncio
import os
import platform
import sys
from typing import Any, Coroutine, Optional

_OFF = ("0", "false", "no", "off")


def _uvloop() -> Optional[Any]:
    if (os.getenv("MIMIC_UVLOOP") or "1").strip().lower() in _OFF:
        return None
    if platform.system() == "Windows":
        return None
    try:
        import uvloop  # type: ignore
    except ImportError:
        return None
    # uvloop.run arrived in 0.18; an older install is treated as absent rather than
    # half-used through the deprecated policy API.
    return uvloop if hasattr(uvloop, "run") else None


def loop_name() -> str:
    return "uvloop" if _uvloop() else "asyncio"


def run(main: Coroutine) -> Any:
    """asyncio.run, on uvloop when it is available. Prints what it chose, and the
    Python it chose it on, so the journal answers both without a shell on the box."""
    uvloop = _uvloop()
    print(f"Python {platform.python_version()} ({sys.implementation.name}), "
          f"event loop: {'uvloop ' + uvloop.__version__ if uvloop else 'asyncio'}", flush=True)
    if uvloop:
        return uvloop.run(main)
    return asyncio.run(main)
