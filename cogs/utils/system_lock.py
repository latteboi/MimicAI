"""cogs/data/system_lock.json: SHA-256 digests of the credentials a data directory was sealed with.

A changed DISCORD_OWNER_ID or ENCRYPTION_KEY stops the bot before it writes anything the old
key cannot read. The bot token is recorded but not enforced: it encrypts nothing and can be
reset.

`verify_system_lock` is the only writer, and nothing reaches the lock before it is shown to
work -- a key Fernet accepts, an owner id the bot can `int()`. A lock written from a typo
fails every later start with a mismatch, and only deleting it by hand gets out of that.
"""
import hashlib
import os
from typing import Optional

import orjson as json
from cryptography.fernet import Fernet

from ..managers.storage_manager import _atomic_write


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def verify_system_lock(path: str, token: str, owner_id: str, key: str) -> Optional[str]:
    """None to start; otherwise the reason not to.

    A first start writes the lock. A later one compares the owner id and key against it, and
    refreshes the token digest when only the token changed.
    """
    try:
        Fernet(key)
    except Exception as e:
        return f"Failed to load encryption key: {e}"
    digests = {"sdk_hash": _digest(token), "owner_hash": _digest(owner_id), "key_hash": _digest(key)}

    if not os.path.exists(path):
        # Admin gates read `int(DISCORD_OWNER_ID)`, so a blank one is no owner at all.
        if not owner_id.isdigit():
            return "DISCORD_OWNER_ID must be your numeric Discord user ID."
        try:
            _atomic_write(path, json.dumps(digests))
        except Exception as e:
            return f"Failed to create system lock file: {e}"
        return None

    try:
        with open(path, "rb") as f:
            lock = json.loads(f.read())
        mismatches = [name for name, field in (("DISCORD_OWNER_ID", "owner_hash"), ("ENCRYPTION_KEY", "key_hash"))
                      if lock.get(field) != digests[field]]
        if mismatches:
            return (f"Authentication mismatch in: {', '.join(mismatches)}\n"
                    "The provided credentials do not match the original setup. "
                    "Bot startup aborted to prevent data corruption.")
        if lock.get("sdk_hash") != digests["sdk_hash"]:
            lock["sdk_hash"] = digests["sdk_hash"]
            try:
                _atomic_write(path, json.dumps(lock))
            except Exception as e:
                print(f"Warning: Failed to update system lock file with new SDK hash: {e}")
    except Exception as e:
        return f"Failed to read system lock file: {e}"
    return None
