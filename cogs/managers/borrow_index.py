"""The borrow reverse index, data/borrows.json: which borrows point at each source profile."""

import os
from typing import Dict, List, Any, Optional, Tuple
from ..utils.constants import USERS_DIR, BORROW_INDEX_FILE
from .storage_manager import IOManager
from .profile_manager import _UNKNOWN_SOURCE_PID



class BorrowIndex:
    """The borrow reverse index, data/borrows.json: which borrows point at each source
    profile. Reached as `cog.borrows`."""

    def __init__(self, cog):
        self.cog = cog

    # --- Borrow reverse index --------------------------------------------------
    #
    # "Who borrows this profile?" used to be answered by walking every user
    # directory and decrypting every borrowed profile config -- once per
    # is_profile_distributed() call, and again per profile deletion. Both are
    # O(all users x their borrows) with a decrypt each, and both reached
    # _get_user_index per user against an LRU, so past its capacity every
    # iteration was also a fresh disk read plus a consistency check.
    #
    # borrows.json is the mapping in the other direction:
    #
    #   {"version": 2,
    #    "sources": {"<owner_id>:<source_pid>": {"<borrower_id>": ["<borrow_pid>", ...]}}}
    #
    # **PIDs on both ends.** Version 1 keyed the borrower side by the borrower's
    # local name for the borrow, which made a rename -- an operation that moves no
    # file and changes no relationship -- look like a delete plus a create: the
    # reconcile below had to decrypt the "new" borrow's shard to re-derive a source
    # it already knew, _cascade_delete_borrowed_profiles resolved a name back to a
    # PID to find the directory to remove (falling back to the name itself, so a
    # miss rmtree'd a name-shaped path), and a removal keyed by a name the borrower
    # had since changed silently did nothing and leaked the entry.
    #
    # A borrow's PID is minted once and never moves. Keyed by it, the reconcile is
    # a set comparison over immutable values that reads no files at all, and renames
    # are free because there is nothing about a rename for this index to record. The
    # local name is display, and is read from the borrower's own index when a screen
    # needs to show it.
    #
    # Source keys are PIDs for the same reason: one key form, `<owner_id>:<pid>`.
    #
    # The inverse map (borrower -> {borrow PID: source key}) is rebuilt in memory
    # on load rather than persisted, so there is one invariant on disk, not two.

    #: Bumped when the on-disk shape changes. A file older than this is rebuilt from
    #: a scan, which is also the migration -- see _load_borrow_index.
    BORROW_INDEX_VERSION = 2

    @staticmethod
    def _borrow_pid_key(owner_id: Any, pid: str) -> str:
        return f"{owner_id}:{pid}"

    def _borrow_key_for_config(self, config: Dict[str, Any]) -> Optional[str]:
        """The source key a borrow's own config points at, or None if it points nowhere.

        PID only. A borrow whose source PID is unknown is unresolvable and is deleted by
        _validate_and_clean_borrowed_profiles; there is nothing to index.
        """
        owner = config.get("original_owner_id")
        if not owner:
            return None
        pid = config.get("original_pid")
        if pid and pid != _UNKNOWN_SOURCE_PID:
            return self._borrow_pid_key(owner, pid)
        return None

    def _load_borrow_index(self):
        """Read borrows.json, or build it from a full scan when it is missing or stale.

        The one-off scan is the migration, in both directions it has ever been needed:
        an install predating the index has borrows on disk and no file describing them,
        and an install on version 1 has a file keyed by local name that this build
        cannot read. An empty index would report every distributed profile as private,
        so neither may be treated as "no borrows".
        """
        data = IOManager.read_json(BORROW_INDEX_FILE)
        if (isinstance(data, dict) and isinstance(data.get("sources"), dict)
                and data.get("version") == self.BORROW_INDEX_VERSION):
            self.cog.borrow_index = data["sources"]
            self._rebuild_borrow_inverse()
            return

        count = self._rebuild_borrow_index()
        print(f"Borrow index: built from scan ({count} borrow"
              f"{'' if count == 1 else 's'} across "
              f"{len(self.cog.borrow_index)} source profile"
              f"{'' if len(self.cog.borrow_index) == 1 else 's'}).")

    def _rebuild_borrow_inverse(self):
        """Derive borrower -> {borrow PID: source key} from the persisted sources map."""
        inverse: Dict[str, Dict[str, str]] = {}
        for source_key, borrowers in self.cog.borrow_index.items():
            if not isinstance(borrowers, dict):
                continue
            for borrower_str, borrow_pids in borrowers.items():
                slot = inverse.setdefault(str(borrower_str), {})
                for borrow_pid in (borrow_pids or []):
                    slot[borrow_pid] = source_key
        self.cog.borrow_index_inverse = inverse

    def _save_borrow_index(self):
        try:
            IOManager.write_json({"version": self.BORROW_INDEX_VERSION,
                                  "sources": self.cog.borrow_index},
                                 BORROW_INDEX_FILE)
        except Exception as e:
            print(f"Error saving borrow index: {e}")

    def _rebuild_borrow_index(self) -> int:
        """Full scan of every user's borrows. Blocking -- boot and manual repair only.

        This is the cost the index exists to stop paying, so nothing on a user-facing
        path may call it. Reads profile shards directly by PID rather than through
        _get_profile_config, because _repair_user_index reaches this while a user's
        index is still being reconstructed and the resolving path would recurse.
        """
        sources: Dict[str, Dict[str, List[str]]] = {}
        total = 0
        if os.path.isdir(USERS_DIR):
            for user_id_str in os.listdir(USERS_DIR):
                if not user_id_str.isdigit():
                    continue
                try:
                    index = IOManager.read_json(
                        os.path.join(USERS_DIR, user_id_str, "index.json")) or {}
                    for borrow_pid in (index.get("borrowed") or {}).values():
                        data = self.cog.profile_manager._get_profile_by_pid(int(user_id_str), borrow_pid) or {}
                        key = self._borrow_key_for_config(data.get("config") or {})
                        if not key:
                            continue
                        sources.setdefault(key, {}).setdefault(user_id_str, [])
                        if borrow_pid not in sources[key][user_id_str]:
                            sources[key][user_id_str].append(borrow_pid)
                            total += 1
                except Exception as e:
                    print(f"Borrow index: skipped user {user_id_str} ({e})")

        self.cog.borrow_index = sources
        self._rebuild_borrow_inverse()
        self._save_borrow_index()
        return total

    def _borrow_index_add(self, borrower_str: str, borrow_pid: str,
                          config: Dict[str, Any]) -> bool:
        key = self._borrow_key_for_config(config)
        if not key:
            return False
        pids = self.cog.borrow_index.setdefault(key, {}).setdefault(borrower_str, [])
        if borrow_pid not in pids:
            pids.append(borrow_pid)
        self.cog.borrow_index_inverse.setdefault(borrower_str, {})[borrow_pid] = key
        return True

    def _borrow_index_remove(self, borrower_str: str, borrow_pid: str) -> bool:
        key = self.cog.borrow_index_inverse.get(borrower_str, {}).pop(borrow_pid, None)
        if not self.cog.borrow_index_inverse.get(borrower_str):
            self.cog.borrow_index_inverse.pop(borrower_str, None)
        if not key:
            return False
        borrowers = self.cog.borrow_index.get(key) or {}
        pids = borrowers.get(borrower_str) or []
        if borrow_pid in pids:
            pids.remove(borrow_pid)
        if not pids:
            borrowers.pop(borrower_str, None)
        if not borrowers:
            self.cog.borrow_index.pop(key, None)
        return True

    def _borrow_index_drop_user(self, user_id: int):
        """Forget a user entirely, as both borrower and source.

        Account deletion and the daily ghost-directory sweep rmtree a user's whole
        tree without touching their index.json, so neither reaches the reconcile
        below. Their borrows are gone and so are the profiles others borrowed from
        them -- the cascade has already run, but a crash mid-delete would leave the
        source keys behind, and a source key naming a directory that no longer
        exists would report a deleted profile as still distributed.
        """
        user_str = str(user_id)
        changed = False
        for borrow_pid in list(self.cog.borrow_index_inverse.get(user_str, {})):
            changed = self._borrow_index_remove(user_str, borrow_pid) or changed

        prefix = f"{user_str}:"
        for key in [k for k in self.cog.borrow_index if k.startswith(prefix)]:
            self.cog.borrow_index.pop(key, None)
            changed = True
        if changed:
            self._rebuild_borrow_inverse()
            self._save_borrow_index()

    def _reconcile_borrow_index(self, user_id: int, index: Dict[str, Any]):
        """Bring one user's borrow entries in line with their index. Called on every save.

        Hooking _save_user_index rather than the seven places that add or remove a
        borrow is what makes this maintenance-free: every one of them saves the index
        afterwards, so none can be forgotten.

        Compares PIDs, so it sees exactly the two events that concern it -- a borrow
        gained and a borrow lost. A rename changes only the key it is filed under in
        the user's own index and does not reach here at all; keyed by name this was a
        remove plus an add, and the add decrypted the shard to re-derive a source that
        had not changed.

        The common case is a personal-profile save with no borrows touched, which
        costs one set comparison. Only a genuinely new borrow reads a file, and it
        reads the shard directly by PID -- _get_profile_config would resolve through
        _get_user_index, which is what called us during a repair.
        """
        user_str = str(user_id)
        borrowed = index.get("borrowed") or {}
        known = set(self.cog.borrow_index_inverse.get(user_str, {}))
        current = set(borrowed.values())
        if known == current:
            return

        changed = False
        for borrow_pid in known - current:
            changed = self._borrow_index_remove(user_str, borrow_pid) or changed
        for borrow_pid in current - known:
            data = self.cog.profile_manager._get_profile_by_pid(user_id, borrow_pid) or {}
            changed = self._borrow_index_add(
                user_str, borrow_pid, data.get("config") or {}) or changed

        if changed:
            self._save_borrow_index()

    def _borrowers_of(self, owner_id: int, pid: Optional[str]) -> List[Tuple[int, str]]:
        """Every (borrower_id, their borrow's PID) currently holding this source profile.

        One key form, one lookup -- see the note above _borrow_pid_key for why the
        name-keyed form no longer exists. The owner is excluded: a profile the owner
        also holds a borrow of is not distributed to anybody.
        """
        if not pid:
            return []

        owner_str = str(owner_id)
        found: List[Tuple[int, str]] = []
        for borrower_str, borrow_pids in (
                self.cog.borrow_index.get(self._borrow_pid_key(owner_id, pid)) or {}).items():
            if borrower_str == owner_str or not borrower_str.isdigit():
                continue
            for borrow_pid in (borrow_pids or []):
                found.append((int(borrower_str), borrow_pid))
        return found
