"""People who asked, in /data, that the bot ignore their messages: data/ignored_users.json.

Every path that takes in what a person does in a channel tests `IGNORED`: a message, its
edit, typing, a reaction a session reads, and someone else's reply quoting them (one test
in `referenced_message` covers all its callers). Commands and buttons are deliberate
input and stay open.

A list of its own rather than a key in a user's index: anyone in a session channel may
ask, registered or not, and nothing else is stored for the unregistered. Delete My Account
leaves the entry, so erasing an account never starts reading that person again. Plaintext:
Discord IDs and nothing else.
"""
import asyncio

from ..utils.constants import IGNORED_USERS_FILE
from .storage_manager import IOManager

#: Mutated in place, never rebound: importers hold this same set.
IGNORED: set = set()
_WRITE_LOCK = asyncio.Lock()


def load() -> None:
    IGNORED.clear()
    IGNORED.update(int(uid) for uid in IOManager.read_json(IGNORED_USERS_FILE) or ())


async def set_ignored(user_id: int, ignored: bool) -> None:
    """The only writer. Snapshots inside the lock, so the last write holds every change."""
    (IGNORED.add if ignored else IGNORED.discard)(user_id)
    async with _WRITE_LOCK:
        await asyncio.to_thread(IOManager.write_json, sorted(IGNORED), IGNORED_USERS_FILE)
