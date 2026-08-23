import os
import io
import time
import hashlib
import re
import uuid
import shutil
import base64
import gzip
import zstandard as zstd
import datetime
import asyncio
import traceback
import httpx
from discord.ext import tasks
import discord
from typing import Dict, List, Set, Any, Optional, Tuple
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from ..utils.constants import (
    USERS_DIR, PUBLIC_PROFILES_DIR, defaultConfig,
    PRIMARY_MODEL_NAME, FALLBACK_MODEL_NAME, DEFAULT_LTM_SUMMARIZATION_INSTRUCTIONS,
    DEFAULT_SAFETY_SETTINGS,
    CONTENT_RATING_LABELS, CHANNEL_ACCESS_LABELS,
    CONTENT_RATING_REASON_LABELS, CONTENT_RATING_REASON_FALLBACK,
    DEFAULT_CONTENT_CLASSIFIER_PROMPT,
    CONTENT_RATING_UNRATED, CONTENT_RATING_PENDING, CONTENT_RATING_GENERAL,
    CONTENT_RATING_ADULT, CONTENT_RATING_EXEMPT,
    CONTENT_RATING_CAPABILITIES, CONTENT_CAPABILITY_DENIALS,
    CONTENT_RATING_EMOJI,
)
from ..utils.http_client import get_shared_client
from .storage_manager import IOManager
from ..services.api_service import OpenRouterModel, GoogleGenAIModel

try:
    import orjson as json
except ImportError:
    import json


class ProfileManager:
    """Owns profile CRUD, personal/borrowed inheritance resolution, share codes, and cloning logic.

    Holds a back-reference to the parent cog for state/logic not yet migrated
    (fernet, the generic shard system, and shared instance caches),
    per the transitional Dependency Injection pattern in CLAUDE.md.
    """

    def __init__(self, cog):
        self.cog = cog

    # Name resolution across the profile classes checks the user's OWN profiles
    # before the global System ones. It used to be the other way around, which meant
    # a user who created a personal profile sharing a name with any System profile
    # -- 'mimicguide' being one that ships -- had their own profile shadowed
    # everywhere a name was resolved: edits landed on it, but generation, prompts and
    # appearance all read the System profile instead. No creation path checks the
    # System index for collisions, so the name was accepted and the profile then
    # quietly did not exist. System stays the fallback, so a name with no personal
    # profile behind it still resolves to it exactly as before.

    def _get_pid_from_name(self, user_id: int, profile_name: str, is_borrowed: bool = False) -> str:
        index = self._get_user_index(user_id)
        if not is_borrowed:
            personal = index.get("personal", {})
            if isinstance(personal, dict) and profile_name in personal:
                return personal[profile_name]
            if isinstance(index.get("system"), dict) and profile_name in index.get("system", {}):
                return index["system"][profile_name]
            mapping = personal
        else:
            mapping = index.get("borrowed", {})
        if isinstance(mapping, dict):
            return mapping.get(profile_name, profile_name)
        return profile_name

    def _get_pid_from_name_any(self, user_id: int, profile_name: str) -> str:
        index = self._get_user_index(user_id)
        if isinstance(index.get("personal"), dict) and profile_name in index["personal"]:
            return index["personal"][profile_name]
        if isinstance(index.get("borrowed"), dict) and profile_name in index["borrowed"]:
            return index["borrowed"][profile_name]
        if isinstance(index.get("system"), dict) and profile_name in index["system"]:
            return index["system"][profile_name]
        return profile_name

    def _get_name_from_pid(self, user_id: int, target_pid: str) -> Optional[str]:
        """The local name a PID maps to, across the classes the owner can share.

        System profiles are included because _accept_share_request resolves the
        shared profile's current name through here: searching only 'personal' meant
        a shared System profile fell through to the requester's cached name, which
        is stale the moment the bot owner renames it. Borrowed profiles are
        deliberately excluded -- a borrow is not the borrower's to re-share.
        """
        index = self._get_user_index(user_id)
        for category in ("personal", "system"):
            mapping = index.get(category, {})
            if isinstance(mapping, dict):
                for name, pid in mapping.items():
                    if pid == target_pid: return name
        return None

    def _is_valid_profile_name(self, name: str) -> tuple[bool, str]:
        if not name or not name.strip():
            return False, "Profile name cannot be empty."
        if len(name) > 20:
            return False, "Profile name must be 20 characters or fewer."
        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            return False, "Profile name can only contain letters, numbers, underscores, and hyphens (no spaces)."
        reserved = [
            "clyde", "system", "user", "none", "all", "everyone", "here",
            "discord", "null", "undefined", "true", "false", "root",
            "mimic", "mimicai", "mimica", "bot", "admin", "mod", "help",
            "global", "bulk"
        ]
        if name.lower() in reserved:
            return False, "This name is a reserved system keyword and cannot be used."
        return True, ""

    def _get_profile_id(self, user_id: int, profile_name: str) -> str:
        index = self._get_user_index(user_id)
        is_borrowed = profile_name in index.get("borrowed", [])
        if is_borrowed:
            config = self._get_profile_config(user_id, profile_name, True) or {}
            orig_owner = config.get("original_owner_id", user_id)
            orig_name = config.get("original_profile_name", profile_name)
            return self._get_pid_from_name_any(int(orig_owner), orig_name)
        else:
            return self._get_pid_from_name_any(user_id, profile_name)

    def _load_public_profiles(self):
        self.cog.public_profiles = {}
        index_path = os.path.join(PUBLIC_PROFILES_DIR, "index.json")
        data = IOManager.read_json(index_path)
        if data:
            self.cog.public_profiles = data

    def _save_public_index(self):
        index_path = os.path.join(PUBLIC_PROFILES_DIR, "index.json")
        IOManager.write_json(self.cog.public_profiles, index_path)

    def _load_profile_shares(self):
        self.cog.profile_shares = {}
        if not os.path.isdir(USERS_DIR):
            return
        for user_id_str in os.listdir(USERS_DIR):
            if not user_id_str.isdigit(): continue
            file_path = os.path.join(USERS_DIR, user_id_str, "shares.json.gz")
            if os.path.exists(file_path):
                data = IOManager.read_json_gzip(file_path, self.cog.fernet)
                if data:
                    self.cog.profile_shares[user_id_str] = data

    def _save_profile_share_shard(self, recipient_id_str: str, data: List):
        if not data:
            self.cog.storage_manager._delete_shard("profile_shares", recipient_id_str)
        else:
            self.cog.storage_manager._save_shard("profile_shares", recipient_id_str, data)

    def _describe_public_entry(self, entry_id: str, p_info: Any) -> Optional[Dict[str, Any]]:
        """Normalises one public-index entry into a single shape.

        The index holds two formats. Publishing writes a plain "owner_id:pid"
        string; older entries are a dict carrying the display name and timestamp.
        Four separate readers branched on the type themselves, and every string
        branch looked the profile name up in a `profiles/<pid>/name.txt` that
        nothing in the codebase has ever written -- so a string entry always
        rendered as "Unknown" and, worse, silently vanished from the
        already-published set that apply_public diffs against. The name is
        resolved from the owner's index here, which is where it actually lives.

        Returns None for a malformed entry so callers can skip it.
        """
        name = None
        published_at = 0
        status = "active"

        if isinstance(p_info, str) and ":" in p_info:
            owner_str, pid = p_info.split(":", 1)
        elif isinstance(p_info, dict):
            owner_str = str(p_info.get("owner_id"))
            pid = p_info.get("original_pid") or p_info.get("original_profile_id")
            name = p_info.get("original_profile_name")
            published_at = p_info.get("published_at") or 0
            status = p_info.get("status", "active")
        else:
            return None

        if not pid or not owner_str:
            return None
        try:
            owner_id = int(owner_str)
        except (TypeError, ValueError):
            return None

        # The owner's index is the authority on whether the profile still exists.
        # A pid that is not in it was deleted, unpublished by hand, or belonged to a
        # user whose data was wiped -- the entry is a tombstone. It used to render
        # as the literal string "Unknown" and sit in the hub forever.
        resolved = self._get_name_from_pid(owner_id, pid)
        orphaned = resolved is None
        if not name or name == "Unknown":
            name = resolved

        if not published_at:
            # Sorting fallback for entries that predate published_at. The profile
            # file is the one artefact guaranteed to exist for a live entry.
            try:
                published_at = os.path.getmtime(
                    os.path.join(USERS_DIR, str(owner_id), "profiles", pid, "profile.json.gz"))
            except OSError:
                published_at = 0

        return {
            "id": entry_id,
            "owner_id": owner_id,
            "original_pid": pid,
            "profile_name": name or "Unknown",
            "published_at": published_at,
            "status": status,
            "orphaned": orphaned,
        }

    def _iter_public_entries(self, owner_id: Optional[int] = None, include_orphaned: bool = False):
        """Yields normalised public entries, optionally filtered to one owner.

        Orphans are withheld by default so a tombstone can never be listed,
        borrowed, or counted, even between prunes.
        """
        for entry_id, p_info in list(self.cog.public_profiles.items()):
            desc = self._describe_public_entry(entry_id, p_info)
            if not desc:
                continue
            if desc["orphaned"] and not include_orphaned:
                continue
            if owner_id is None or desc["owner_id"] == int(owner_id):
                yield desc

    def _prune_public_index(self) -> List[Dict[str, Any]]:
        """Drops public entries whose source profile no longer exists.

        Two independent checks, because either can fail on its own: the pid must
        still be in the owner's personal index, and the profile file must still be
        on disk. Deletion normally cascades through
        _cascade_delete_borrowed_profiles, so anything reaching here survived a
        crash, a manual edit, or a delete path that predates that cascade.

        Returns the removed entries so a caller can report them.
        """
        removed = []
        for entry_id, p_info in list(self.cog.public_profiles.items()):
            desc = self._describe_public_entry(entry_id, p_info)

            if desc is None:
                # Unparseable entry -- neither a pointer string nor a dict.
                removed.append({"id": entry_id, "profile_name": "<malformed>", "owner_id": None})
                self.cog.public_profiles.pop(entry_id, None)
                continue

            if not desc["orphaned"]:
                profile_path = os.path.join(
                    USERS_DIR, str(desc["owner_id"]), "profiles", desc["original_pid"], "profile.json.gz")
                if os.path.exists(profile_path):
                    continue

            removed.append(desc)
            self.cog.public_profiles.pop(entry_id, None)

        if removed:
            self._save_public_index()
        return removed

    def _find_public_entry_id(self, owner_id: int, pid: str) -> Optional[str]:
        """The public-index key for a given owner/pid, or None if not published."""
        for desc in self._iter_public_entries(owner_id):
            if desc["original_pid"] == pid:
                return desc["id"]
        return None

    def _is_profile_public(self, user_id: int, profile_name: str) -> bool:
        index = self._get_user_index(user_id)
        is_borrowed = profile_name in index.get("borrowed", [])

        effective_owner_id = user_id
        effective_pid = self._get_pid_from_name_any(user_id, profile_name)

        if is_borrowed:
            b_config = self._get_profile_config(user_id, profile_name, True) or {}
            effective_owner_id = int(b_config.get("original_owner_id", user_id))
            effective_pid = b_config.get("original_pid") or b_config.get("original_profile_id")

        target_pointer = f"{effective_owner_id}:{effective_pid}"

        for p_info in self.cog.public_profiles.values():
            if isinstance(p_info, str):
                if p_info == target_pointer:
                    return True
            elif isinstance(p_info, dict):
                if str(p_info.get("owner_id")) == str(effective_owner_id) and p_info.get("original_pid") == effective_pid:
                    return True
        return False

    def _resolve_borrowed_pointer(self, pointer: str) -> Optional[Tuple[int, str]]:
        """A borrow's pointer resolved to (owner_id, pid), or None if it is broken.

        A pointer has one of two shapes: "<owner_id>:<pid>" for a private share-code
        borrow, or a key into the public index for a Public Library borrow. The
        public form used to be recognised by an allowlist of leading characters --
        "pub_" or "A" -- which was the source profile's own PID class showing
        through, since publishing keys the index by the profile's PID. Any other
        class fell through both branches and resolved to None, orphaning the borrow,
        and the list would have needed a new case for every class added since.

        The colon is what actually separates the two shapes, so that is what is
        tested. No PID class is named here and none needs to be.
        """
        if not pointer:
            return None

        if ":" not in pointer:
            if not self.cog.public_profiles:
                self._load_public_profiles()
            target = self.cog.public_profiles.get(pointer)
            if isinstance(target, dict):
                owner_id = target.get("owner_id")
                pid = target.get("original_pid") or target.get("original_profile_id")
                return (int(owner_id), pid) if owner_id and pid else None
            if not isinstance(target, str):
                return None
            # Indirection into the "<owner_id>:<pid>" form, resolved below.
            pointer = target

        try:
            owner_id_str, pid = pointer.split(":", 1)
            return int(owner_id_str), pid
        except ValueError:
            return None

    def _get_profile(self, user_id: int, profile_name: str, is_borrowed: bool = False) -> Optional[Dict[str, Any]]:
        if not profile_name:
            return None
        pid = self._get_pid_from_name(user_id, profile_name, is_borrowed)
        if not pid:
            return None
        return self._get_profile_by_pid(user_id, pid)

    def _save_profile(self, user_id: int, profile_name: str, data: Dict[str, Any], is_borrowed: bool = False):
        if not profile_name:
            return
        pid = self._get_pid_from_name(user_id, profile_name, is_borrowed)
        if not pid:
            return
        self._save_profile_by_pid(user_id, pid, data)

    def _get_profile_by_pid(self, user_id: int, pid: str) -> Optional[Dict[str, Any]]:
        if not pid:
            return None
        path = os.path.join(USERS_DIR, str(user_id), "profiles", pid, "profile.json.gz")
        return IOManager.read_json_gzip(path, self.cog.fernet)

    def _save_profile_by_pid(self, user_id: int, pid: str, data: Dict[str, Any]):
        if not pid:
            return
        p_dir = os.path.join(USERS_DIR, str(user_id), "profiles", pid)
        os.makedirs(p_dir, exist_ok=True)
        path = os.path.join(p_dir, "profile.json.gz")
        IOManager.write_json_gzip(data, path, self.cog.fernet)

    def _set_child_bot_config(self, user_id: int, profile_name: str, bot_config: Dict[str, Any]):
        p_data = self._get_profile(user_id, profile_name, is_borrowed=False)
        if p_data is not None:
            p_data["child_bot"] = bot_config
            self._save_profile(user_id, profile_name, p_data, is_borrowed=False)

    def _delete_child_bot_config(self, user_id: int, profile_name: str):
        p_data = self._get_profile(user_id, profile_name, is_borrowed=False)
        if p_data is not None and p_data.get("child_bot"):
            p_data["child_bot"] = None
            self._save_profile(user_id, profile_name, p_data, is_borrowed=False)

    def _update_child_bot_presence(self, user_id: int, profile_name: str, presence_update: Dict[str, Any]) -> Dict[str, Any]:
        p_data = self._get_profile(user_id, profile_name, is_borrowed=False)
        if not p_data:
            return {}
        bot_cfg = p_data.get("child_bot") or {}
        current_presence = bot_cfg.get("presence", {})
        current_presence.update(presence_update)
        bot_cfg["presence"] = current_presence
        p_data["child_bot"] = bot_cfg
        self._save_profile(user_id, profile_name, p_data, is_borrowed=False)
        return current_presence

    def _rename_profile(self, user_id: int, old_name: str, new_name: str, is_borrowed: bool = False) -> bool:
        user_index = self._get_user_index(user_id)
        list_key = "borrowed" if is_borrowed else "personal"
        
        if old_name not in user_index.get(list_key, {}):
            return False
            
        pid = user_index[list_key].pop(old_name)
        user_index[list_key][new_name] = pid
        self._save_user_index(user_id, user_index)
        
        p_data = self._get_profile_by_pid(user_id, pid)
        if p_data:
            p_data["name"] = new_name
            self._save_profile_by_pid(user_id, pid, p_data)

        # The name is part of the classified surface, and the cache is keyed by it,
        # so both entries must go -- the stale key would otherwise linger until the
        # LRU evicted it.
        self._invalidate_content_rating(user_id, old_name)
        return True

    def _duplicate_profile(self, user_id: int, source_name: str, target_name: str) -> Tuple[bool, str]:
        user_index = self._get_user_index(user_id)
        if len(user_index.get("personal", {})) >= defaultConfig.LIMIT_PROFILES:
            return False, "Limit reached."

        source_pid = self._get_pid_from_name_any(user_id, source_name)
        source_data = self._get_profile_by_pid(user_id, source_pid)
        if not source_data:
            return False, "Source profile data not found."

        new_pid = f"A{uuid.uuid4().hex[:15].upper()}"
        new_data = {
            "name": target_name,
            "config": source_data.get("config", {}).copy(),
            "prompts": source_data.get("prompts", {}).copy(),
            "child_bot": None
        }
        # The PID assigned above, not a fresh id. This used to mint an unrelated
        # 8-character value, so a duplicated profile's config disagreed with its own
        # folder name -- and every other write site (import, clone, convert) sets
        # config["profile_id"] to the PID. See _get_profile_config for what the
        # mismatch broke.
        new_data["config"]["profile_id"] = new_pid
        new_data["config"]["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        self._save_profile_by_pid(user_id, new_pid, new_data)

        if not isinstance(user_index.get("personal"), dict):
            legacy_personal = user_index.get("personal", [])
            user_index["personal"] = {}
            if isinstance(legacy_personal, list):
                for p_name in legacy_personal:
                    user_index["personal"][p_name] = p_name

        user_index["personal"][target_name] = new_pid
        self._save_user_index(user_id, user_index)

        self.cog.memory_manager._copy_ltm_shard(str(user_id), source_name, target_name)
        self.cog.memory_manager._copy_training_shard(str(user_id), source_name, target_name)
        return True, f"Duplicated to '{target_name}'."

    def _repair_user_index(self, user_id: int) -> Dict[str, Any]:
        """Scans the user's profile directory to reconstruct a missing or corrupted index.json."""
        user_id_str = str(user_id)
        old_index = IOManager.read_json(os.path.join(USERS_DIR, user_id_str, "index.json")) or {}
        index = {"personal": {}, "borrowed": {}, "system": old_index.get("system", {})}
        profiles_dir = os.path.join(USERS_DIR, user_id_str, "profiles")

        if os.path.isdir(profiles_dir):
            for pid_folder in os.listdir(profiles_dir):
                p_dir = os.path.join(profiles_dir, pid_folder)
                if not os.path.isdir(p_dir):
                    continue

                profile_path = os.path.join(p_dir, "profile.json.gz")
                if not os.path.exists(profile_path):
                    continue

                profile_data = IOManager.read_json_gzip(profile_path, self.cog.fernet) or {}
                p_name = profile_data.get("name") or pid_folder
                config_data = profile_data.get("config", {})

                is_borrowed = bool(config_data.get("original_owner_id"))

                if is_borrowed:
                    index["borrowed"][p_name] = pid_folder
                elif pid_folder.startswith("X"):
                    if "system" not in index:
                        index["system"] = {}
                    index["system"][p_name] = pid_folder
                else:
                    index["personal"][p_name] = pid_folder

            keys_path = os.path.join(USERS_DIR, user_id_str, "keys.json.gz")
            index["has_personal_key"] = False
            if os.path.exists(keys_path):
                keys_data = IOManager.read_json_gzip(keys_path, self.cog.fernet)
                if keys_data and (keys_data.get("key") or keys_data.get("slots")):
                    index["has_personal_key"] = True

            self._save_user_index(user_id, index)
        else:
            keys_path = os.path.join(USERS_DIR, user_id_str, "keys.json.gz")
            index["has_personal_key"] = False
            if os.path.exists(keys_path):
                keys_data = IOManager.read_json_gzip(keys_path, self.cog.fernet)
                if keys_data and (keys_data.get("key") or keys_data.get("slots")):
                    index["has_personal_key"] = True

            self._save_user_index(user_id, index)

        return index

    def _index_is_consistent(self, user_id_str: str) -> bool:
        """Cheap pre-check for _repair_all_user_indices.

        A repair opens, Fernet-decrypts and zstd-decompresses every profile shard the
        user owns. That is the right cost when the index is actually wrong, and pure
        waste when it is not -- and this runs on every boot AND hourly, so the common
        case is "nothing has changed since the last one".

        Everything checked here is a directory listing plus one small file, so the
        check is orders of magnitude cheaper than the repair it avoids. It compares:
          - the index parses and has the expected shape
          - it names exactly as many profiles as there are shards on disk
          - its has_personal_key flag still matches the key file

        Deliberately conservative: anything unexpected returns False and the full
        repair runs, so a false negative only costs the work we do today.
        """
        try:
            index = IOManager.read_json(os.path.join(USERS_DIR, user_id_str, "index.json"))
            if not index or not isinstance(index.get("personal"), dict) or not isinstance(index.get("borrowed"), dict):
                return False

            profiles_dir = os.path.join(USERS_DIR, user_id_str, "profiles")
            shards_on_disk = 0
            if os.path.isdir(profiles_dir):
                for pid_folder in os.listdir(profiles_dir):
                    if os.path.exists(os.path.join(profiles_dir, pid_folder, "profile.json.gz")):
                        shards_on_disk += 1

            named_in_index = (len(index.get("personal", {}))
                              + len(index.get("borrowed", {}))
                              + len(index.get("system", {})))
            if named_in_index != shards_on_disk:
                return False

            # Nothing clears has_personal_key when a key is deleted -- only a repair
            # does -- so it has to be verified here or a stale True would survive.
            keys_path = os.path.join(USERS_DIR, user_id_str, "keys.json.gz")
            has_key = False
            if os.path.exists(keys_path):
                keys_data = IOManager.read_json_gzip(keys_path, self.cog.fernet)
                has_key = bool(keys_data and (keys_data.get("key") or keys_data.get("slots")))
            return bool(index.get("has_personal_key")) == has_key
        except Exception:
            return False

    def _repair_all_user_indices(self, force: bool = False):
        """Scans the USERS_DIR for user folders and runs self-repair on each user's index.json.

        Skips users whose index already checks out (see _index_is_consistent) unless
        force=True, because this runs on every boot and then hourly thereafter.
        """
        if not os.path.isdir(USERS_DIR):
            return
        for user_id_str in os.listdir(USERS_DIR):
            if user_id_str.isdigit():
                try:
                    if not force and self._index_is_consistent(user_id_str):
                        continue
                    user_id = int(user_id_str)
                    self._repair_user_index(user_id)
                except Exception as e:
                    print(f"Error repairing index for user {user_id_str}: {e}")

    def _get_user_index(self, user_id: int) -> Dict[str, Any]:
        user_id_str = str(user_id)
        if user_id_str in self.cog.user_indices: return self.cog.user_indices[user_id_str]

        path = os.path.join(USERS_DIR, user_id_str, "index.json")
        index = IOManager.read_json(path)

        if not index or not isinstance(index.get("personal"), dict) or not isinstance(index.get("borrowed"), dict):
            index = self._repair_user_index(user_id)

        self.cog.user_indices[user_id_str] = index
        return index

    def _save_user_index(self, user_id: int, data: Dict[str, Any]):
        user_id_str = str(user_id)
        path = os.path.join(USERS_DIR, user_id_str, "index.json")
        IOManager.write_json(data, path)
        self.cog.user_indices[user_id_str] = data

    def _get_profile_config(self, user_id: int, profile_name: str, is_borrowed: bool = False) -> Optional[Dict[str, Any]]:
        p_data = self._get_profile(user_id, profile_name, is_borrowed)
        if p_data is not None:
            config = p_data.get("config", {})
            if not is_borrowed:
                # config["profile_id"] is meant to equal the PID the profile is
                # stored under, and import, clone and convert all set it that way.
                # Two paths did not: a missing value was filled with a fresh
                # 8-character id, and _duplicate_profile minted one that matched
                # nothing at all. Either way the config claimed an identity the rest
                # of the system could not resolve -- _accept_share_request falls back
                # to this field when it has no PID to work from and stores it as the
                # borrow's original_pid, so _find_public_entry_id could not link a
                # public borrow to its listing and _is_profile_public reported every
                # such borrow as unpublished.
                #
                # Reconciled rather than only backfilled, so profiles already
                # carrying a stray id are repaired on their next read. The write
                # happens once per affected profile; afterwards the values agree and
                # this is a string comparison.
                pid = self._get_pid_from_name(user_id, profile_name, is_borrowed)
                # A name that does not map -- mid-repair, or one that never mapped --
                # comes back as the name itself. Leave the field alone in that case
                # rather than writing an id that would outlive the confusion.
                if pid and pid != profile_name and config.get("profile_id") != pid:
                    config["profile_id"] = pid
                    p_data["config"] = config
                    self._save_profile(user_id, profile_name, p_data, is_borrowed)
            return config
        return None
    def _save_profile_config(self, user_id: int, profile_name: str, data: Dict[str, Any], is_borrowed: bool = False):
        p_data = self._get_profile(user_id, profile_name, is_borrowed)
        if p_data is None:
            p_data = {
                "name": profile_name,
                "config": data,
                "prompts": {},
                "child_bot": None
            }
        else:
            p_data["name"] = profile_name
            p_data["config"] = data
        self._save_profile(user_id, profile_name, p_data, is_borrowed)

    def _resolve_effective_profile(self, user_id: int, profile_name: str) -> Tuple[int, str]:
        index = self._get_user_index(user_id)

        # Own profiles first, System as the fallback -- see the note above
        # _get_pid_from_name. The System branch used to run before this one, so a
        # user's personal profile sharing a name with a System profile resolved to
        # the System profile and never ran.
        if profile_name in index.get("borrowed", []):
            b_config = self._get_profile_config(user_id, profile_name, True) or {}
            eff_owner = int(b_config.get("original_owner_id", user_id))
            eff_name = b_config.get("original_profile_name", profile_name)
            return eff_owner, eff_name
        if profile_name in index.get("personal", []):
            return user_id, profile_name

        owner_id = int(defaultConfig.DISCORD_OWNER_ID)
        if user_id != owner_id:
            owner_idx = self._get_user_index(owner_id)
            if profile_name in owner_idx.get("system", {}):
                return owner_id, profile_name

        return user_id, profile_name

    def _get_user_appearance(self, owner_id: int, profile_name: str) -> Dict[str, Optional[str]]:
        eff_owner_id, eff_name = self._resolve_effective_profile(owner_id, profile_name)
        owner_id_str = str(eff_owner_id)
        if owner_id_str in self.cog.user_appearances and eff_name in self.cog.user_appearances[owner_id_str]:
            return self.cog.user_appearances[owner_id_str][eff_name]

        config = self._get_profile_config(eff_owner_id, eff_name, False) or {}
        disp = config.get("custom_display_name")
        ava = config.get("custom_avatar_url")

        data = {"custom_display_name": disp, "custom_avatar_url": ava}
        self.cog.user_appearances.setdefault(owner_id_str, {})[eff_name] = data
        return data

    def _get_profile_prompts(self, user_id: int, profile_name: str) -> Optional[Dict[str, Any]]:
        pid = self._get_pid_from_name_any(user_id, profile_name)
        if not pid: return None
        p_data = self._get_profile_by_pid(user_id, pid)
        if p_data is not None:
            return p_data.get("prompts", {})
        return None

    def _save_profile_prompts(self, user_id: int, profile_name: str, data: Dict[str, Any]):
        """Persists prompts and invalidates any cached content rating.

        Every persona, instruction and image-prompt write funnels through here.

        Deliberately does NOT classify. This hook used to fire the classifier on
        every persona and instruction save and was the dominant source of classifier
        calls -- a profile could be judged dozens of times over its life without its
        owner ever asking for a verdict. The edit now only invalidates: the caller
        decides what to do about the rating going stale, interactively where there
        is a user to ask, and via resolve_stale_rating everywhere else.
        """
        pid = self._get_pid_from_name_any(user_id, profile_name)
        if not pid: return
        p_data = self._get_profile_by_pid(user_id, pid)
        if p_data is None:
            p_data = {
                "name": profile_name,
                "config": {},
                "prompts": data,
                "child_bot": None
            }
        else:
            p_data["prompts"] = data
        self._save_profile_by_pid(user_id, pid, p_data)
        self._invalidate_content_rating(user_id, profile_name)

    def _get_or_create_user_profile(self, user_id: int, profile_name: str) -> Optional[Dict[str, Any]]:
        profile_name = profile_name.lower().strip()

        index = self._get_user_index(user_id)

        if profile_name not in index.get("personal", []):
            if len(index.get("personal", [])) >= defaultConfig.LIMIT_PROFILES:
                return None

            if isinstance(index.get("personal"), dict):
                pid = f"A{uuid.uuid4().hex[:15].upper()}"
                index["personal"][profile_name] = pid
            else:
                index.setdefault("personal", []).append(profile_name)

            self._save_user_index(user_id, index)

            config = {
                "grounding_enabled": False, "stm_length": defaultConfig.CHATBOT_MEMORY_LENGTH,
                "temperature": defaultConfig.GEMINI_TEMPERATURE, "top_p": defaultConfig.GEMINI_TOP_P,
                "top_k": defaultConfig.GEMINI_TOP_K, "training_context_size": defaultConfig.TRAINING_CONTEXT_SIZE,
                "training_relevance_threshold": defaultConfig.TRAINING_RELEVANCE_THRESHOLD,
                "ltm_context_size": 3, "ltm_relevance_threshold": 0.75, "ltm_creation_interval": 10,
                "ltm_summarization_context": 10,
                "primary_model": PRIMARY_MODEL_NAME, "fallback_model": FALLBACK_MODEL_NAME,
                "time_tracking_enabled": True, "timezone": "UTC",
                "realistic_typing_enabled": False, "ltm_creation_enabled": False,
                "image_generation_enabled": False, "image_generation_model": "GOOGLE/gemini-2.5-flash-image",
                "url_fetching_enabled": False, "response_mode": "regular", "thinking_summary_visible": "off",
                "thinking_level": "low", "thinking_budget": -1,
                "error_response": "An error has occurred.", "speech_tts_enabled": False, "speech_voice": "Aoede",
                "speech_model": "GOOGLE/gemini-2.5-flash-preview-tts", "speech_temperature": 1.0,
                "neuro_engine_enabled": False, "neuro_state": {"dopamine": 50, "cortisol": 20, "oxytocin": 50, "adrenaline": 20},
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }

            prompts = {
                "persona": {}, "ai_instructions": ["", "", "", ""], "image_generation_prompt": None,
                "ltm_summarization_instructions": self.cog.storage_manager._encrypt_data(self._default_ltm_summarization_instructions())
            }

            unified_profile = {
                "name": profile_name,
                "config": config,
                "prompts": prompts,
                "child_bot": None
            }
            self._save_profile(user_id, profile_name, unified_profile, False)
            return {"config": config, "prompts": prompts}

        return {"config": self._get_profile_config(user_id, profile_name), "prompts": self._get_profile_prompts(user_id, profile_name)}

    def _get_or_create_system_profile(self, profile_name: str) -> Optional[Dict[str, Any]]:
        user_id = int(defaultConfig.DISCORD_OWNER_ID)
        profile_name = profile_name.lower().strip()

        index = self._get_user_index(user_id)
        if "system" not in index:
            index["system"] = {}

        if profile_name not in index["system"]:
            pid = f"X{uuid.uuid4().hex[:15].upper()}"
            index["system"][profile_name] = pid
            self._save_user_index(user_id, index)

            config = {
                "grounding_enabled": False, "stm_length": defaultConfig.CHATBOT_MEMORY_LENGTH,
                "temperature": 0.2, "top_p": 0.9,
                "top_k": 40, "training_context_size": 0,
                "training_relevance_threshold": 0.0,
                "ltm_context_size": 0, "ltm_relevance_threshold": 1.0, "ltm_creation_interval": 100,
                "ltm_summarization_context": 10,
                "primary_model": "GOOGLE/gemini-2.5-flash-lite", "fallback_model": "GOOGLE/gemini-2.5-flash-lite",
                "time_tracking_enabled": True, "timezone": "UTC", "generation_metadata_enabled": False,
                "realistic_typing_enabled": False, "ltm_creation_enabled": False,
                "image_generation_enabled": False, "image_generation_model": "GOOGLE/gemini-2.5-flash-image",
                "url_fetching_enabled": False, "response_mode": "regular", "thinking_summary_visible": "off",
                "thinking_level": "low", "thinking_budget": -1,
                "error_response": "An error has occurred.", "speech_tts_enabled": False, "speech_voice": "Aoede",
                "speech_model": "GOOGLE/gemini-2.5-flash-preview-tts", "speech_temperature": 1.0,
                "neuro_engine_enabled": False, "neuro_state": {"dopamine": 50, "cortisol": 20, "oxytocin": 50, "adrenaline": 20},
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "help_mode_enabled": False
            }

            prompts = {
                "persona": {
                    "backstory": [self.cog.storage_manager._encrypt_data("You are MimicGuide, the official technical assistant for the MimicAI Discord Bot.")],
                    "personality_traits": [self.cog.storage_manager._encrypt_data("Helpful, precise, and highly technical.")],
                    "likes": [], "dislikes": [], "appearance": []
                },
                "ai_instructions": [self.cog.storage_manager._encrypt_data("Answer questions concisely using the provided documentation. If you do not know the answer, state that you do not know."), "", "", ""],
                "image_generation_prompt": None,
                "ltm_summarization_instructions": self.cog.storage_manager._encrypt_data(self._default_ltm_summarization_instructions())
            }

            unified_profile = {
                "name": profile_name,
                "config": config,
                "prompts": prompts,
                "child_bot": None
            }
            self._save_profile_by_pid(user_id, pid, unified_profile)

        return {"config": self._get_profile_config(user_id, profile_name), "prompts": self._get_profile_prompts(user_id, profile_name)}

    def _moderated_surface(self, owner_id: int, profile_name: str) -> str:
        """The profile text the classifier judges, and the text the hash covers.

        Built from the same decrypted fields prompt_builder assembles, so the
        verdict is about what the model will actually be told to be. Deliberately
        excludes LTM, session logs and training examples: the first two are
        conversation-derived, and training examples -- though the strongest signal
        of what a profile really outputs -- change often enough that hashing them
        would invalidate the verdict constantly. Revisit them with their own hash
        and a debounce.
        """
        eff_owner, eff_name = self._resolve_effective_profile(owner_id, profile_name)
        prompts = self._get_profile_prompts(eff_owner, eff_name) or {}
        config = self._get_profile_config(eff_owner, eff_name, False) or {}

        parts = [f"name: {eff_name}"]

        display = config.get("custom_display_name")
        if display:
            parts.append(f"display_name: {display}")
        avatar = config.get("custom_avatar_url")
        if avatar:
            parts.append(f"avatar_url: {avatar}")

        persona = prompts.get("persona", {}) or {}
        for key in self.cog.persona_modal_sections_order:
            lines = persona.get(key) or []
            decrypted = [self.cog.storage_manager._decrypt_data(l).strip()
                         for l in lines if isinstance(l, str) and l.strip()]
            body = "\n".join(d for d in decrypted if d)
            if body:
                parts.append(f"{key}:\n{body}")

        instr = prompts.get("ai_instructions", "")
        instr_list = instr if isinstance(instr, list) else [instr]
        decrypted_instr = [self.cog.storage_manager._decrypt_data(i).strip()
                           for i in instr_list if isinstance(i, str) and i.strip()]
        body = "\n\n".join(d for d in decrypted_instr if d)
        if body:
            parts.append(f"instructions:\n{body}")

        img_prompt = prompts.get("image_generation_prompt")
        if isinstance(img_prompt, str) and img_prompt.strip():
            parts.append(f"image_prompt:\n{self.cog.storage_manager._decrypt_data(img_prompt).strip()}")

        return "\n\n".join(parts)

    def _surface_hash(self, surface: str) -> str:
        return hashlib.sha256(surface.encode("utf-8", "replace")).hexdigest()[:16]


    def reset_all_content_ratings(self) -> Dict[str, Any]:
        """Operator maintenance: reset every profile on the instance to Unrated.

        A single deliberate action, run once by the operator, rather than a lazy
        per-read migration or a boot sweep. That is why nothing here is versioned:
        with no automatic pass to guard against, there is no need to record which
        profiles have already been through it, and no schema marker to carry
        forward in every rating record for the rest of the bot's life.

        **Unconditional.** Every profile, including Adult and Exempt ones. The point
        is a clean baseline: a rating now means "the owner asked for this and got an
        answer", and no verdict predating that rule qualifies, however it was
        reached. Sparing Adult profiles would leave verdicts nobody consented to,
        and sparing Exempt ones would leave grants whose original justification is
        no longer visible. The operator can re-grant an exemption in one click, and
        an owner can re-declare 18+ in one click with no API call.

        Also strips the retired `safety_level`, which is the last of the pre-rating
        fields.

        Borrowed profiles (B/C PIDs) are skipped: their rating resolves through the
        source profile, so their local snapshot is not the authority and rewriting
        it would only create a second answer.

        Blocking -- walks and rewrites every profile file. Call it in a thread.
        """
        counts = {"scanned": 0, "reset": 0, "skipped_borrowed": 0,
                  "legacy_fields": 0, "errors": 0, "delisted": []}
        if not os.path.isdir(USERS_DIR):
            return counts

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        for user_id_str in os.listdir(USERS_DIR):
            if not user_id_str.isdigit():
                continue
            profiles_dir = os.path.join(USERS_DIR, user_id_str, "profiles")
            if not os.path.isdir(profiles_dir):
                continue

            for pid in os.listdir(profiles_dir):
                path = os.path.join(profiles_dir, pid, "profile.json.gz")
                if not os.path.exists(path):
                    continue
                counts["scanned"] += 1

                if pid.startswith(("B", "C")):
                    counts["skipped_borrowed"] += 1
                    continue

                try:
                    p_data = IOManager.read_json_gzip(path, self.cog.fernet)
                    if not isinstance(p_data, dict):
                        continue
                    config = p_data.get("config")
                    if not isinstance(config, dict):
                        continue

                    if config.pop("safety_level", None) is not None:
                        counts["legacy_fields"] += 1

                    config["content_rating"] = {
                        "verdict": CONTENT_RATING_UNRATED,
                        "hash": None,
                        "model": None,
                        "at": now,
                        "reason": None,
                        "source": "reset",
                    }
                    p_data["config"] = config
                    IOManager.write_json_gzip(p_data, path, self.cog.fernet)
                    counts["reset"] += 1
                except Exception as e:
                    counts["errors"] += 1
                    print(f"Content rating reset failed for {user_id_str}/{pid}: "
                          f"{type(e).__name__}({e})")

        # Everything is Unrated now, and an Unrated profile may not be published, so
        # the entire public index is stale by construction. Clearing it is not a side
        # effect of the reset -- it is the same decision applied to the index.
        # Existing borrows are untouched: a borrow reads the source profile, never
        # this index.
        for entry_id, p_info in list(self.cog.public_profiles.items()):
            desc = self._describe_public_entry(entry_id, p_info)
            label = entry_id
            if desc:
                label = f"{desc['profile_name']} (owner {desc['owner_id']})"
            self.cog.public_profiles.pop(entry_id, None)
            counts["delisted"].append(label)
        if counts["delisted"]:
            self._save_public_index()

        self.cog.content_rating_cache.clear()
        return counts

    def _verdict_of(self, config: Dict[str, Any]) -> str:
        """The rating verdict for a config, normalised to a known state.

        Anything unrecognised -- an absent record, the retired "unclassified", a
        value from a build that has since been removed -- reads as Unrated. That is
        the correct failure direction now: Unrated runs normally but cannot be
        distributed, so an unreadable rating costs the owner a submission rather
        than either silencing the profile or letting it out into the library.
        """
        rating = config.get("content_rating") or {}
        verdict = rating.get("verdict")
        if verdict in CONTENT_RATING_CAPABILITIES:
            return verdict
        return CONTENT_RATING_UNRATED

    def _effective_safety_level(self, config: Dict[str, Any]) -> str:
        """Where this profile may run, derived from its content rating alone.

        Only Adult restricts placement. Unrated, Pending and General are identical
        at runtime -- the rating governs distribution, not execution -- so a profile
        whose owner has never submitted it is not silenced, it is merely undistributable.
        """
        capabilities = CONTENT_RATING_CAPABILITIES[self._verdict_of(config)]
        return "unrestricted" if capabilities["age_restricted_only"] else "restricted"

    def _content_rating_state(self, owner_id: int, profile_name: str) -> Tuple[str, Optional[str]]:
        """(verdict, reason) as last recorded, for display and for the UI gates.

        Deliberately does NOT recompute the surface hash. This runs on every render
        of the profile Home tab, and hashing means decrypting the whole persona;
        staleness is detected instead by rating_is_stale, which is async and
        runs on dashboard open and session setup. The consequence is conservative in
        the right direction: a stale 'adult' verdict keeps the toggle locked until
        the recheck lands, rather than briefly unlocking it.
        """
        eff_owner, eff_name = self._resolve_effective_profile(owner_id, profile_name)
        config = self._get_profile_config(eff_owner, eff_name, False) or {}
        return self._verdict_of(config), (config.get("content_rating") or {}).get("reason")

    def content_capability(self, owner_id: int, profile_name: str, capability: str) -> Tuple[bool, Optional[str]]:
        """(allowed, reason_if_not) for one distribution capability.

        Single source of truth for the share, publish and global-chat gates. Each
        used to test verdicts inline in its own module -- the hub checked one set of
        values, the global chat command checked publication instead of rating, and
        the turn gate checked a third thing -- so the same profile could be accepted
        by one and refused by another with no shared wording to explain it.
        """
        verdict, _ = self._content_rating_state(owner_id, profile_name)
        allowed = CONTENT_RATING_CAPABILITIES[verdict][capability]
        if allowed:
            return True, None
        reason = CONTENT_CAPABILITY_DENIALS.get(
            (capability, verdict),
            f"Not available for a profile rated {CONTENT_RATING_LABELS[verdict]}.")
        return False, reason

    def _resolve_enforced_safety_level(self, profile_owner_id: int, profile_name: str) -> str:
        """The level the gate enforces for this profile, cached per (owner, name).

        The uncached path reads, decrypts and decompresses up to two profile files
        and hashes the persona. The gate runs on the turn path, so the result is
        memoised and invalidated by _invalidate_content_rating from every edit,
        classification and bulk write.
        """
        cache_key = (int(profile_owner_id), profile_name)
        cached = self.cog.content_rating_cache.get(cache_key)
        if cached is not None:
            return cached

        index = self._get_user_index(profile_owner_id)
        is_borrowed = profile_name in index.get("borrowed", [])
        config = self._get_profile_config(profile_owner_id, profile_name, is_borrowed) or {}

        level = self._effective_safety_level(config)

        # A borrowed profile carries its own local copy of the config, and
        # _store_content_rating refuses to write one -- the authoritative rating
        # only ever exists at the source, where the persona the verdict was formed
        # from actually lives. The local copy can still hold a stale rating, or a
        # legacy 18+ declaration frozen in at borrow time, so take whichever of the
        # two is stricter: the borrow can only ever escalate, never relax.
        if is_borrowed and level != "unrestricted":
            src_owner, src_name = self._resolve_effective_profile(profile_owner_id, profile_name)
            if (src_owner, src_name) != (profile_owner_id, profile_name):
                src_config = self._get_profile_config(src_owner, src_name, False) or {}
                level = self._effective_safety_level(src_config)

        self.cog.content_rating_cache[cache_key] = level
        return level

    def _invalidate_content_rating(self, owner_id: int, profile_name: str):
        """Drops the cached level for a profile and for every borrow of it.

        A borrowed entry is keyed by the borrower's local name, so a change at the
        source cannot be located by key -- the borrow map is scanned instead. It is
        small (one entry per live borrow in the cache) and this runs on edits, not
        on the turn path.
        """
        self.cog.content_rating_cache.pop((int(owner_id), profile_name), None)
        for key in [k for k in list(self.cog.content_rating_cache.keys())
                    if k != (int(owner_id), profile_name)]:
            k_owner, k_name = key
            try:
                src = self._resolve_effective_profile(k_owner, k_name)
            except Exception:
                continue
            if src == (int(owner_id), profile_name):
                self.cog.content_rating_cache.pop(key, None)

    def _check_unrestricted_safety_policy(self, profile_owner_id: int, profile_name: str, channel: discord.abc.Messageable) -> bool:
        if self._resolve_enforced_safety_level(profile_owner_id, profile_name) == "unrestricted":
            if not isinstance(channel, (discord.TextChannel, discord.Thread, discord.VoiceChannel)):
                return False
            return channel.is_nsfw()

        return True

    async def _validate_and_clean_borrowed_profiles(self, user_id: int) -> int:
        """
        Scans a user's borrowed profiles. If the source profile no longer exists
        (deleted by owner), it removes the borrowed entry.
        Returns the number of profiles removed.
        """
        index = self._get_user_index(user_id)
        borrowed = index.get("borrowed", [])
        if not borrowed:
            return 0

        # Group by owner to minimize I/O
        profiles_by_owner = {}
        ids_to_remove = []
        for local_name in borrowed:
            b_config = self._get_profile_config(user_id, local_name, True)
            if not b_config:
                # Index entry with no readable profile behind it -- the directory was
                # removed, the file is corrupt, or a save was interrupted. This used
                # to be skipped silently, so the dead name stayed in the index
                # forever and kept passing the session-hydration existence check,
                # which only looks the name up in the index.
                ids_to_remove.append(local_name)
                continue

            o_id = b_config.get("original_owner_id")
            o_pid = b_config.get("original_pid")
            o_name = b_config.get("original_profile_name")
            if o_id and (o_pid or o_name):
                profiles_by_owner.setdefault(str(o_id),[]).append((local_name, o_pid, o_name))
            else:
                # A borrowed profile with no pointer back to a source cannot be
                # resolved or refreshed; nothing can ever validate it again.
                ids_to_remove.append(local_name)

        removed_count = 0

        for owner_id_str, items in profiles_by_owner.items():
            owner_index = self._get_user_index(int(owner_id_str))
            owner_personal = owner_index.get("personal", {})
            valid_pids = list(owner_personal.values()) if isinstance(owner_personal, dict) else[]
            valid_names = list(owner_personal.keys()) if isinstance(owner_personal, dict) else owner_personal

            for local_name, o_pid, o_name in items:
                if o_pid:
                    if o_pid not in valid_pids:
                        ids_to_remove.append(local_name)
                else:
                    if o_name not in valid_names:
                        ids_to_remove.append(local_name)

        if ids_to_remove:
            if isinstance(borrowed, dict):
                for local_name in ids_to_remove:
                    pid = index["borrowed"].pop(local_name, local_name)
                    p_dir = os.path.join(USERS_DIR, str(user_id), "profiles", pid)
                    shutil.rmtree(p_dir, ignore_errors=True)
            else:
                index["borrowed"] = [b for b in borrowed if b not in ids_to_remove]
                for local_name in ids_to_remove:
                    p_dir = os.path.join(USERS_DIR, str(user_id), "profiles", local_name)
                    shutil.rmtree(p_dir, ignore_errors=True)

            self._save_user_index(user_id, index)
            removed_count = len(ids_to_remove)

        return removed_count

    def _cascade_delete_borrowed_profiles(self, original_owner_id: int, deleted_pid: str, original_profile_name: str):
        """Instantly removes all borrowed variants linked to a deleted personal profile across the entire system."""
        owner_str = str(original_owner_id)

        # Proactively remove the deleted profile from the global public database index
        public_ids_to_del = []
        for pub_id, info in list(self.cog.public_profiles.items()):
            if isinstance(info, str) and ":" in info:
                if info == f"{owner_str}:{deleted_pid}":
                    public_ids_to_del.append(pub_id)
            elif isinstance(info, dict) and str(info.get("owner_id")) == owner_str:
                if info.get("original_pid") == deleted_pid or info.get("original_profile_name") == original_profile_name:
                    public_ids_to_del.append(pub_id)

        if public_ids_to_del:
            for pub_id in public_ids_to_del:
                self.cog.public_profiles.pop(pub_id, None)
            self._save_public_index()

        if not os.path.isdir(USERS_DIR): return

        for user_id_str in os.listdir(USERS_DIR):
            if not user_id_str.isdigit(): continue
            try:
                uid = int(user_id_str)
                index = self._get_user_index(uid)
                borrowed = index.get("borrowed", {})

                to_delete =[]
                for b_name in list(borrowed):
                    b_config = self._get_profile_config(uid, b_name, True)
                    if b_config and str(b_config.get("original_owner_id")) == owner_str:
                        b_pid = b_config.get("original_pid")
                        if b_pid and b_pid == deleted_pid:
                            to_delete.append(b_name)
                        elif not b_pid and b_config.get("original_profile_name") == original_profile_name:
                            to_delete.append(b_name)

                if to_delete:
                    if isinstance(borrowed, dict):
                        for b_name in to_delete:
                            pid = index["borrowed"].pop(b_name, b_name)
                            p_dir = os.path.join(USERS_DIR, user_id_str, "profiles", pid)
                            shutil.rmtree(p_dir, ignore_errors=True)
                    else:
                        index["borrowed"] = [b for b in borrowed if b not in to_delete]
                        for b_name in to_delete:
                            p_dir = os.path.join(USERS_DIR, user_id_str, "profiles", b_name)
                            shutil.rmtree(p_dir, ignore_errors=True)
                    self._save_user_index(uid, index)
            except Exception as e:
                print(f"Error in cascade delete for user {user_id_str}: {e}")

    def _classifier_api_key(self, owner_id: int, provider: str) -> Optional[str]:
        """Personal -> instance-owner key for the classifier.

        The auto-moderator only ever tried the personal key, which was fine while
        the only caller was publishing. Classification runs for every profile, so a
        user with no key of their own would otherwise never get a verdict. Guild
        keys are deliberately not used: a profile is not owned by a guild, and
        billing someone's server for another member's profile edit is surprising.
        """
        key = self.cog.storage_manager._get_api_key_for_user(owner_id, provider)
        if key:
            return key
        bot_owner = int(defaultConfig.DISCORD_OWNER_ID)
        if int(owner_id) != bot_owner:
            return self.cog.storage_manager._get_api_key_for_user(bot_owner, provider)
        return None

    async def _fetch_avatar_part(self, owner_id: int, profile_name: str) -> Optional[Dict[str, Any]]:
        """The profile's avatar as an inline image part, or None.

        Absorbed from the retired auto-moderator, with one behavioural change that
        was a standing bug: a download failure is no longer fatal.

        The auto-moderator refused to publish when it could not fetch the avatar,
        which conflated "this image is unacceptable" with "this host would not serve
        *us*". Those are not the same, and the difference is visible in production:
        the webhook path never downloads anything -- it hands the URL to Discord and
        Discord's servers fetch it -- so an image the bot could not retrieve from a
        datacentre IP still rendered perfectly in chat, while publishing the same
        profile failed with a URL error. Now the image is one signal among several:
        if it cannot be had, the text is judged alone.
        """
        eff_owner, eff_name = self._resolve_effective_profile(owner_id, profile_name)
        config = self._get_profile_config(eff_owner, eff_name, False) or {}
        avatar_url = config.get("custom_avatar_url")
        if not avatar_url:
            return None

        try:
            # The spoofed User-Agent rides on the request, not the client, so this
            # can share the process-wide pool with everything else.
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                                     "Chrome/120.0.0.0 Safari/537.36"}
            response = await get_shared_client().get(
                avatar_url, follow_redirects=True, timeout=10.0, headers=headers)
            response.raise_for_status()
        except Exception as e:
            print(f"Classifier could not fetch the avatar for {eff_owner}/{eff_name} "
                  f"({type(e).__name__}); judging the text alone.")
            return None

        data = response.content
        if not data or len(data) > defaultConfig.CONTENT_CLASSIFY_MAX_IMAGE_BYTES:
            return None

        content_type = (response.headers.get("Content-Type") or "image/png").split(";")[0].strip()
        if not content_type.startswith("image/"):
            # A URL that serves an HTML error page is not an avatar. Sending it as
            # an image part is a guaranteed provider error.
            return None

        return {"mime_type": content_type, "data": data}

    async def _classify_profile_content(self, owner_id: int, profile_name: str) -> Optional[Dict[str, Any]]:
        """Runs the classifier over one profile and returns a content_rating record.

        Returns None when no verdict could be reached -- no key, or every provider
        failed -- so the caller can leave the profile unclassified and retry rather
        than recording a wrong verdict.
        """
        surface = await asyncio.to_thread(self._moderated_surface, owner_id, profile_name)
        surface_hash = self._surface_hash(surface)

        max_chars = defaultConfig.CONTENT_CLASSIFY_MAX_CHARS
        truncated = surface[:max_chars]
        if len(surface) > max_chars:
            truncated += "\n[...truncated]"

        prompt_text = self.cog.global_prompts.get(
            "CONTENT_CLASSIFIER", DEFAULT_CONTENT_CLASSIFIER_PROMPT)

        parts = [f"<target_profile>\n{truncated}\n</target_profile>"]
        avatar_part = await self._fetch_avatar_part(owner_id, profile_name)
        if avatar_part:
            parts.append(avatar_part)

        payload = [{"role": "user", "parts": parts}]
        gen_cfg = {"temperature": 0.0, "top_k": 1, "top_p": 0.9}

        attempts = [
            ("openrouter", "amazon/nova-lite-v1", OpenRouterModel),
            ("gemini", "gemini-2.5-flash-lite", GoogleGenAIModel),
        ]

        raw = None
        used_model = None
        status = None
        model_name = None
        failures = []
        for provider, model_name, model_cls in attempts:
            key = self._classifier_api_key(owner_id, provider)
            if not key:
                # _get_api_key_for_user returns None both when no key is assigned for
                # this provider AND when the assigned key is inside its rate-limit
                # cooldown window. Both used to `continue` in silence, so a run that
                # never reached a provider looked identical to one that failed at it.
                failures.append(f"{provider}: no usable key (unassigned or rate-limit cooldown)")
                continue
            status = "api_error"
            try:
                kwargs = {"api_key": key, "model_name": model_name, "system_instruction": prompt_text}
                if model_cls is GoogleGenAIModel:
                    kwargs["safety_settings"] = DEFAULT_SAFETY_SETTINGS
                model = model_cls(**kwargs)
                response = await model.generate_content_async(payload, generation_config=gen_cfg)
                if response and response.candidates:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        raw = "".join(p.text for p in candidate.content.parts
                                      if hasattr(p, "text")).strip()
                        status = "success"
                else:
                    status = "blocked_by_safety"
            except Exception as e:
                failures.append(f"{provider}: {e}")
            finally:
                self.cog._log_api_call(user_id=0, guild_id=None, context="content_classification",
                                       model_used=model_name, status=status)
            if raw:
                used_model = model_name
                break

        if not raw:
            if status == "blocked_by_safety" and not failures:
                failures.append(f"{model_name}: provider returned no candidates")
            self._last_classify_failure = "; ".join(failures) or "unknown"
            # No provider was even reachable because none had a usable key. Retrying
            # cannot help within the backoff window -- a key is not going to appear
            # in fifteen seconds -- so tell the caller not to bother.
            self._last_classify_retryable = not all("no usable key" in f for f in failures) if failures else True
            return None

        head, _, tail = raw.partition(":")
        verdict = "adult" if head.strip().upper().startswith("ADULT") else "general"

        # Only a recognised category code survives. The prompt asks for one, but a
        # model that answers in prose anyway must not have that prose stored and
        # shown back to the owner -- it would be describing their persona to them,
        # in the classifier's words. An unusable reason is dropped, and the embed
        # renders the generic label in its place.
        code = "".join(tail.split()).upper().strip(".")
        reason = code if code in CONTENT_RATING_REASON_LABELS else None

        return {
            "verdict": verdict,
            "hash": surface_hash,
            "model": used_model,
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "reason": reason,
            "source": "classifier",
        }

    def _store_content_rating(self, owner_id: int, profile_name: str, rating: Dict[str, Any]) -> bool:
        """Persists a verdict, unless a moderator override is pinned for this content."""
        index = self._get_user_index(owner_id)
        is_borrowed = profile_name in index.get("borrowed", [])
        if is_borrowed:
            return False

        config = self._get_profile_config(owner_id, profile_name, False)
        if config is None:
            return False

        existing = config.get("content_rating") or {}
        if (existing.get("source") == "owner_override"
                and existing.get("hash") == rating.get("hash")):
            # A moderator cleared this exact content; re-running the classifier over
            # it must not silently undo that. A later edit changes the hash and the
            # override lapses with the content it was granted for.
            return False

        if (existing.get("verdict") == "adult"
                and existing.get("source") == "owner_declared"
                and rating.get("verdict") != "adult"):
            # The owner volunteered 18+. The classifier escalates, it does not
            # relax, so a 'general' verdict here changes nothing -- but the hash is
            # stamped anyway, so rating_is_stale stops seeing this content as
            # unclassified and re-queueing it on every dashboard render. Normally
            # unreachable: schedule_content_classification skips declared profiles.
            # This catches a job already in flight when the declaration landed.
            # Rebuilt rather than mutated: _duplicate_profile shallow-copies the
            # config, so a duplicate can share this dict with its source.
            config["content_rating"] = {**existing, "hash": rating.get("hash")}
            self._save_profile_config(owner_id, profile_name, config, False)
            self._invalidate_content_rating(owner_id, profile_name)
            return False

        config["content_rating"] = rating
        self._save_profile_config(owner_id, profile_name, config, False)
        self._invalidate_content_rating(owner_id, profile_name)
        return True

    async def rating_is_stale(self, owner_id: int, profile_name: str) -> bool:
        """True when the profile's text has changed since its verdict was recorded.

        Hashes the moderated surface, which decrypts the whole persona -- so this
        belongs on deliberate paths only, never on the turn path. States that cannot
        go stale are answered before the hash is computed rather than after it.
        """
        eff_owner, eff_name = self._resolve_effective_profile(owner_id, profile_name)
        config = self._get_profile_config(eff_owner, eff_name, False) or {}
        rating = config.get("content_rating") or {}
        verdict = self._verdict_of(config)

        # Unrated has nothing to be stale against; Pending is mid-flight; Exempt is
        # the operator's standing decision; an owner declaration is about the profile
        # rather than a revision of its text, and Adult is already the strictest
        # verdict reachable.
        if verdict in (CONTENT_RATING_UNRATED, CONTENT_RATING_PENDING, CONTENT_RATING_EXEMPT):
            return False
        if verdict == CONTENT_RATING_ADULT and rating.get("source") == "owner_declared":
            return False
        retry_after = rating.get("retry_after")
        if retry_after and time.time() < retry_after:
            return False

        stored = rating.get("hash")
        if not stored:
            return False

        current = await asyncio.to_thread(
            lambda: self._surface_hash(self._moderated_surface(eff_owner, eff_name)))
        return stored != current

    async def resolve_stale_rating(self, owner_id: int, profile_name: str) -> Optional[str]:
        """Handles a rating whose profile has been edited since it was judged.

        Returns what happened, for the caller to report, or None if nothing needed
        doing. This is the backstop for edits that do not go through the interactive
        prompt -- an import, a bulk apply, a restored backup, a future edit path.

        The rule is the whole point of the redesign:

        * **Undistributed** -- drop to Unrated. Free, no API call. The owner keeps
          using the profile exactly as before and re-submits when they next want to
          share it.
        * **Distributed** -- re-classify. Other people are running this profile on
          the strength of its verdict, so the verdict has to keep up. This is the
          only path that spends a call without being asked to, and it is bounded by
          the number of profiles that are actually shared.
        """
        if not await self.rating_is_stale(owner_id, profile_name):
            return None

        eff_owner, eff_name = self._resolve_effective_profile(owner_id, profile_name)

        distributed = await asyncio.to_thread(
            self.is_profile_distributed, eff_owner, eff_name)
        if distributed:
            self.schedule_content_classification(eff_owner, eff_name)
            return "reclassify"

        await asyncio.to_thread(self.drop_to_unrated, eff_owner, eff_name)
        return "unrated"

    def _record_classification_failure(self, owner_id: int, profile_name: str, reason: str):
        """Stamps a retry-after on a profile the classifier could not judge."""
        config = self._get_profile_config(owner_id, profile_name, False)
        if config is None:
            return
        if self._verdict_of(config) in (CONTENT_RATING_ADULT, CONTENT_RATING_GENERAL,
                                       CONTENT_RATING_EXEMPT):
            return  # a real verdict already stands; a later failure must not erase it
        config["content_rating"] = {
            # Stays Pending rather than dropping to Unrated: the owner did ask for a
            # verdict, and the dashboard shows the stored reason and the retry time
            # against this state. Dropping to Unrated would silently discard their
            # submission because a key happened to be missing.
            "verdict": CONTENT_RATING_PENDING,
            "hash": None,
            "model": None,
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "reason": reason[:200],
            "source": "failed",
            "retry_after": time.time() + defaultConfig.CONTENT_CLASSIFY_RETRY_AFTER,
        }
        self._save_profile_config(owner_id, profile_name, config, False)
        self._invalidate_content_rating(owner_id, profile_name)

    def _classification_on_cooldown(self, owner_id: int, profile_name: str) -> bool:
        config = self._get_profile_config(owner_id, profile_name, False) or {}
        rating = config.get("content_rating") or {}
        retry_after = rating.get("retry_after")
        return bool(retry_after) and time.time() < retry_after

    def set_classification_exempt(self, owner_id: int, profile_name: str, exempt: bool) -> bool:
        """Bot-owner switch: exempt a profile from classification entirely.

        An exempt profile is treated as 'general' for placement -- it runs in any
        channel -- and no classifier job is ever queued for it. It is also the one
        carve-out in _resolve_safety_settings: an exemption is a standing statement
        that this profile does not need filtering, so it sends BLOCK_NONE even in a
        general channel, where every other profile sends BLOCK_ONLY_HIGH. The
        <content_policy> block is all that shapes it there.

        Unlike a moderator override, the exemption is NOT tied to a content hash:
        it is a standing decision about the profile, so editing the persona does
        not revoke it. That is the point of it, and also its risk.
        """
        config = self._get_profile_config(owner_id, profile_name, False)
        if config is None:
            return False
        if exempt:
            config["content_rating"] = {
                "verdict": "exempt",
                "hash": None,
                "model": None,
                "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "reason": "Exempted from classification by the bot owner",
                "source": "owner_exempt",
            }
        else:
            # Back to Unrated rather than straight to the classifier: removing an
            # exemption returns the profile to its owner's hands, and submitting is
            # the owner's call to make.
            config["content_rating"] = {
                "verdict": CONTENT_RATING_UNRATED,
                "hash": None,
                "model": None,
                "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "reason": None,
                "source": "exemption_removed",
            }
        self._save_profile_config(owner_id, profile_name, config, False)
        self._invalidate_content_rating(owner_id, profile_name)
        return True

    def clear_adult_verdict(self, owner_id: int, profile_name: str, moderator_id: int) -> bool:
        """Operator appeal: pins General for exactly the content that was flagged.

        Keyed to the current surface hash, so the override lapses the moment the
        persona changes -- a cleared profile cannot be edited into adult content and
        keep its clearance. Lifted out of the dashboard view so the Content Safety
        page and any future caller share one implementation.
        """
        config = self._get_profile_config(owner_id, profile_name, False)
        if config is None:
            return False
        if self._verdict_of(config) != CONTENT_RATING_ADULT:
            return False
        if self._is_owner_declared_adult(owner_id, profile_name):
            # An owner's own declaration is theirs to withdraw, not a moderator's to
            # overrule -- clearing it would silently un-declare their profile.
            return False

        surface = self._moderated_surface(owner_id, profile_name)
        config["content_rating"] = {
            "verdict": CONTENT_RATING_GENERAL,
            "hash": self._surface_hash(surface),
            "model": None,
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "reason": f"Cleared by moderator {moderator_id}",
            "source": "owner_override",
        }
        self._save_profile_config(owner_id, profile_name, config, False)
        self._invalidate_content_rating(owner_id, profile_name)
        return True

    def _is_classification_exempt(self, owner_id: int, profile_name: str) -> bool:
        eff_owner, eff_name = self._resolve_effective_profile(owner_id, profile_name)
        config = self._get_profile_config(eff_owner, eff_name, False) or {}
        return (config.get("content_rating") or {}).get("verdict") == "exempt"

    def _is_owner_declared_adult(self, owner_id: int, profile_name: str) -> bool:
        """True when the adult verdict came from the owner rather than the classifier.

        The two are stored identically and enforced identically; they differ only
        in who may move them. An owner may take their own declaration back; a
        classifier verdict is appealed to a moderator via _handle_clear_verdict.
        """
        eff_owner, eff_name = self._resolve_effective_profile(owner_id, profile_name)
        config = self._get_profile_config(eff_owner, eff_name, False) or {}
        rating = config.get("content_rating") or {}
        return rating.get("verdict") == "adult" and rating.get("source") == "owner_declared"

    def set_owner_adult_declaration(self, owner_id: int, profile_name: str, declared: bool) -> bool:
        """The owner's own 18+ declaration, recorded as a content_rating verdict.

        This is what the retired safety_level toggle became. Declaring costs no API
        call and needs no classifier agreement, which is the point: an owner who
        wants the age-restricted lane should not have to write a persona
        inflammatory enough to trip a flash-lite model into agreeing with them.

        Refuses rather than silently doing nothing where the verdict is not the
        owner's to move -- a classifier 'adult' (appealed via /mod), an exemption
        (the bot owner's), or a borrowed profile (not the borrower's content).
        Returns True only when the stored rating now matches what was asked for.
        """
        index = self._get_user_index(owner_id)
        if profile_name in index.get("borrowed", []):
            return False

        config = self._get_profile_config(owner_id, profile_name, False)
        if config is None:
            return False

        rating = config.get("content_rating") or {}
        verdict, source = rating.get("verdict"), rating.get("source")

        if verdict == "exempt":
            return False
        if verdict == "adult" and source != "owner_declared":
            return False

        if declared:
            if verdict == "adult":
                return True
            config["content_rating"] = {
                "verdict": "adult",
                # No hash. A declaration is about the profile, not about one
                # revision of its text, and rating_is_stale returns on the
                # declaration before it hashes anything -- so a hash would protect
                # against nothing while costing a full persona decrypt inside a GUI
                # callback, once per profile in the bulk path.
                "hash": None,
                "model": None,
                "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "reason": "Declared 18+ by the profile owner",
                "source": "owner_declared",
            }
        else:
            if source != "owner_declared":
                return False
            # Dropped rather than rewritten to 'general': the owner withdrawing a
            # declaration is not a judgement that the content is general, it is a
            # Withdrawing a declaration is not a judgement that the content is
            # general, it is a request to stop asserting otherwise. The profile
            # returns to Unrated and its owner submits it when they want a verdict.
            config["content_rating"] = {
                "verdict": CONTENT_RATING_UNRATED,
                "hash": None,
                "model": None,
                "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "reason": None,
                "source": "declaration_withdrawn",
            }

        self._save_profile_config(owner_id, profile_name, config, False)
        self._invalidate_content_rating(owner_id, profile_name)
        return True

    def is_profile_distributed(self, owner_id: int, profile_name: str) -> bool:
        """True when someone other than the owner currently holds this profile.

        Published to the library, or borrowed by at least one user. This is the line
        the whole scheme turns on: an undistributed profile is nobody else's problem
        and is never classified without being asked for, while a distributed one has
        to keep its verdict honest because other people are running it.

        Scans every user's borrow index, so it belongs on deliberate paths -- an
        edit, a dashboard render -- and never on the turn path.
        """
        if self._is_profile_public(owner_id, profile_name):
            return True

        eff_owner, eff_name = self._resolve_effective_profile(owner_id, profile_name)
        owner_str = str(eff_owner)
        if not os.path.isdir(USERS_DIR):
            return False

        for user_id_str in os.listdir(USERS_DIR):
            if not user_id_str.isdigit() or user_id_str == owner_str:
                continue
            try:
                index = self._get_user_index(int(user_id_str))
                for b_name in index.get("borrowed", []) or []:
                    b_config = self._get_profile_config(int(user_id_str), b_name, True)
                    if not b_config:
                        continue
                    if str(b_config.get("original_owner_id")) != owner_str:
                        continue
                    if b_config.get("original_profile_name") == eff_name:
                        return True
            except Exception:
                # A single unreadable index must not decide that a shared profile is
                # private -- that is the direction that skips a needed reclassification.
                continue
        return False

    async def submit_for_rating(self, owner_id: int, profile_name: str) -> Tuple[bool, str]:
        """Owner-initiated: move a profile to Pending and queue the classifier.

        The only path that starts a classification from nothing. Everything else
        either re-runs an existing verdict for a distributed profile, or drops the
        profile back to Unrated and waits to be asked again.

        **Async on purpose.** schedule_content_classification needs a running event
        loop to create its task, and returns silently when there is not one. An
        earlier version of this was a plain sync method that callers reached through
        asyncio.to_thread for the file writes -- which meant the schedule call landed
        on a worker thread every single time, hit that guard, and did nothing. The
        profile was written Pending, the user was told it had been submitted, and no
        job existed to ever move it off Pending. The blocking work is threaded
        individually here so the scheduling stays on the loop.
        """
        index = await asyncio.to_thread(self._get_user_index, owner_id)
        if profile_name in index.get("borrowed", []):
            return False, "A borrowed profile is rated by its owner, not by you."

        verdict, _ = await asyncio.to_thread(self._content_rating_state, owner_id, profile_name)
        if verdict == CONTENT_RATING_EXEMPT:
            return False, "This profile is exempt from classification."
        if verdict == CONTENT_RATING_PENDING:
            # A Pending profile is either genuinely in flight or a submission that
            # failed and stamped a retry-after. Refusing both alike strands the
            # second kind permanently: the internal retries are exhausted, nothing
            # re-queues it, and the owner is told to wait for a verdict that will
            # never arrive. Once the cooldown has passed, let them try again.
            eff_owner, eff_name = await asyncio.to_thread(
                self._resolve_effective_profile, owner_id, profile_name)
            config = await asyncio.to_thread(self._get_profile_config, eff_owner, eff_name, False)
            rating = (config or {}).get("content_rating") or {}
            retry_after = rating.get("retry_after")
            if not retry_after:
                return False, "This profile is already awaiting a verdict."
            if time.time() < retry_after:
                return False, (f"The last attempt failed and it will retry "
                               f"<t:{int(retry_after)}:R>. This usually means no API key "
                               f"was available -- check `/settings`.")
        if verdict == CONTENT_RATING_ADULT and await asyncio.to_thread(
                self._is_owner_declared_adult, owner_id, profile_name):
            return False, ("You have declared this profile 18+. Withdraw the declaration "
                           "first if you want it classified instead.")

        config = await asyncio.to_thread(self._get_profile_config, owner_id, profile_name, False)
        if config is None:
            return False, "Profile not found."

        config["content_rating"] = {
            "verdict": CONTENT_RATING_PENDING,
            "hash": None,
            "model": None,
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "reason": None,
            "source": "submitted",
        }
        await asyncio.to_thread(self._save_profile_config, owner_id, profile_name, config, False)
        self._invalidate_content_rating(owner_id, profile_name)

        # On the loop, not in a thread -- see the note above.
        self.schedule_content_classification(owner_id, profile_name)
        return True, "Submitted. The verdict usually lands within a few seconds."

    def drop_to_unrated(self, owner_id: int, profile_name: str) -> bool:
        """Returns a profile to Unrated without spending a call.

        What an edit to an undistributed profile does. The owner keeps using it
        exactly as before -- Unrated and General are identical at runtime -- and
        re-submits whenever they next want to share it.
        """
        config = self._get_profile_config(owner_id, profile_name, False)
        if config is None:
            return False
        if self._verdict_of(config) == CONTENT_RATING_EXEMPT:
            return False

        config["content_rating"] = {
            "verdict": CONTENT_RATING_UNRATED,
            "hash": None,
            "model": None,
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "reason": None,
            "source": "edited",
        }
        self._save_profile_config(owner_id, profile_name, config, False)
        self._invalidate_content_rating(owner_id, profile_name)
        return True

    def schedule_content_classification(self, owner_id: int, profile_name: str):
        """Queues a profile for classification. Never awaited.

        No longer called from the edit paths. Classification now starts in exactly
        two places: an owner submitting via the Content Safety dashboard, and the
        re-check of a *distributed* profile whose text has changed underneath its
        borrowers. Every other edit either drops the profile to Unrated or, for a
        distributed one, asks the owner which they want.
        """
        index = self._get_user_index(owner_id)
        if profile_name in index.get("borrowed", []):
            return
        if self._is_classification_exempt(owner_id, profile_name):
            return
        if self._is_owner_declared_adult(owner_id, profile_name):
            # Nothing a verdict could add: adult is already the strictest outcome
            # and the classifier cannot relax one. Skipping here is what keeps a
            # declared profile from ever spending an API call.
            return
        if self._classification_on_cooldown(owner_id, profile_name):
            return

        key = (int(owner_id), profile_name)
        self._invalidate_content_rating(owner_id, profile_name)
        if key in self.cog.pending_classifications:
            return

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # Called from a worker thread with no loop, so no task can be created.
            # Loud, because a silent return here is what left submitted profiles
            # stuck on Pending forever: the caller had already written the state and
            # told the user it was in progress, and nothing existed to finish it.
            # The invalidation above still stands, and resolve_stale_rating picks the
            # profile up next time the dashboard or a session touches it.
            print(f"Classification for {owner_id}/{profile_name} was scheduled from a "
                  f"worker thread with no event loop and has been dropped. Call this "
                  f"from the loop.")
            self.cog.pending_classifications.pop(key, None)
            return

        self.cog.pending_classifications[key] = 0
        asyncio.create_task(self._run_classification_job(owner_id, profile_name))

    async def _run_classification_job(self, owner_id: int, profile_name: str):
        key = (int(owner_id), profile_name)
        try:
            for attempt in range(defaultConfig.CONTENT_CLASSIFY_MAX_ATTEMPTS):
                self.cog.pending_classifications[key] = attempt + 1
                try:
                    rating = await self._classify_profile_content(owner_id, profile_name)
                except Exception as e:
                    print(f"Classification job errored for {owner_id}/{profile_name}: {e}")
                    rating = None

                if rating:
                    await asyncio.to_thread(self._store_content_rating, owner_id, profile_name, rating)
                    return

                if not getattr(self, "_last_classify_retryable", True):
                    break

                if attempt < defaultConfig.CONTENT_CLASSIFY_MAX_ATTEMPTS - 1:
                    await asyncio.sleep(5 * (2 ** attempt))

            # Out of attempts. Record why and when, so the next dashboard render or
            # session setup does not immediately re-queue the same doomed job -- that
            # loop is what turned one unclassifiable profile into an endless stream of
            # give-up lines. The profile stays Pending, which runs normally but
            # cannot be shared until a verdict is actually reached.
            reason = getattr(self, "_last_classify_failure", None) or "unknown"
            print(f"Classification gave up for {owner_id}/{profile_name}: {reason}. "
                  f"Retrying no sooner than {defaultConfig.CONTENT_CLASSIFY_RETRY_AFTER}s from now.")
            await asyncio.to_thread(self._record_classification_failure, owner_id, profile_name, reason)
        finally:
            self.cog.pending_classifications.pop(key, None)

    @tasks.loop(hours=1.0)
    async def hourly_self_repair_task(self):
        if self.cog.has_lock:
            await asyncio.to_thread(self._repair_all_user_indices)

    async def _execute_export(self, interaction: discord.Interaction, profile_names: List[str], filters: Set[str], passphrase: Optional[str] = None):
        user_id = interaction.user.id
        user_id_str = str(user_id)
        
        raw_export_data = {
            "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "profiles": {}
        }

        for name in profile_names:
            pid = self._get_pid_from_name_any(user_id, name)
            p_dir = os.path.join(self.cog.USERS_DIR, user_id_str, "profiles", pid)
            if not os.path.exists(p_dir): continue

            p_data = self._get_profile_by_pid(user_id, pid) or {}
            config = p_data.get("config", {}).copy()
            prompts = p_data.get("prompts", {}).copy()
            config.pop("profile_id", None)

            p_entry = {
                "pid": pid,
                "config": config,
                "prompts": prompts,
                "ltm": [],
                "training": []
            }

            if "ltm" in filters:
                ltm_data = self.cog.storage_manager._load_json_gzip(os.path.join(p_dir, "ltm.json.gz"))
                if ltm_data:
                    p_entry["ltm"] = ltm_data.get("guild", [])

            if "training" in filters:
                training_data = self.cog.storage_manager._load_json_gzip(os.path.join(p_dir, "training.json.gz"))
                if training_data:
                    p_entry["training"] = training_data

            raw_export_data["profiles"][name] = p_entry

        raw_json_bytes = json.dumps(raw_export_data)
        compressed_payload = zstd.ZstdCompressor(level=1).compress(raw_json_bytes)
        export_container = {
            "mimic_version": "3.0",
        }

        if passphrase:
            salt = os.urandom(16)
            kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=480000)
            derived_key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode('utf-8')))
            temp_fernet = Fernet(derived_key)
            
            encrypted_payload = temp_fernet.encrypt(compressed_payload)
            export_container["auth_mode"] = "passphrase"
            export_container["salt"] = base64.b64encode(salt).decode('utf-8')
            export_container["payload"] = encrypted_payload.decode('utf-8')
        else:
            encrypted_payload = self.cog.fernet.encrypt(compressed_payload)
            export_container["auth_mode"] = "master"
            export_container["payload"] = encrypted_payload.decode('utf-8')

        file_data = json.dumps(export_container, option=json.OPT_INDENT_2)
        filename = f"mimic_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.mimic"
        
        buffer = io.BytesIO(file_data)
        discord_file = discord.File(buffer, filename=filename)
        
        msg = "✅ Export complete."
        if passphrase:
            msg += " Your data has been securely encrypted with your passphrase for self-hosted migration."
        else:
            msg += " Your data is securely encrypted with the official instance master key."
            
        await interaction.followup.send(msg, file=discord_file, ephemeral=True)

    async def _execute_import(self, interaction: discord.Interaction, file_bytes: bytes, passphrase: Optional[str] = None):
        try:
            try:
                container = json.loads(file_bytes)
            except json.JSONDecodeError:
                raise ValueError("The provided file is not a valid MimicAI 3.0 export container (Invalid JSON).")

            if container.get("mimic_version") != "3.0" or "payload" not in container:
                raise ValueError("Plaintext or legacy v2.0 exports are rejected by the official instance for security and anti-injection compliance. Please use a valid v3.0 encrypted `.mimic` file.")

            auth_mode = container.get("auth_mode")
            encrypted_payload = container["payload"].encode('utf-8')
            raw_json_bytes = None

            if auth_mode == "passphrase":
                if not passphrase:
                    raise ValueError("This file is encrypted with a passphrase. Please use the import command properly to enter it.")
                
                salt_b64 = container.get("salt")
                if not salt_b64:
                    raise ValueError("Corrupted passphrase export: missing cryptographic salt.")
                    
                salt = base64.b64decode(salt_b64)
                kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=480000)
                derived_key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode('utf-8')))
                temp_fernet = Fernet(derived_key)
                
                try:
                    decrypted_bytes = temp_fernet.decrypt(encrypted_payload)
                except InvalidToken:
                    raise ValueError("Decryption failed. The passphrase provided is incorrect.")
            else:
                try:
                    decrypted_bytes = self.cog.fernet.decrypt(encrypted_payload)
                except InvalidToken:
                    raise ValueError("Master key decryption failed. This file belongs to a different MimicAI instance and cannot be imported here without a passphrase migration export.")

            try:
                raw_json_bytes = zstd.ZstdDecompressor().decompress(decrypted_bytes)
            except zstd.ZstdError:
                try:
                    raw_json_bytes = gzip.decompress(decrypted_bytes)
                except gzip.BadGzipFile:
                    # Backward compatibility fallback for uncompressed legacy v3 exports
                    raw_json_bytes = decrypted_bytes

            data = json.loads(raw_json_bytes)
            
            if "profiles" not in data:
                raise ValueError("Decrypted payload is missing the profiles object.")

            user_id = interaction.user.id
            user_id_str = str(user_id)
            index = self._get_user_index(user_id)
            
            import_log = []
            for name, p_data in data["profiles"].items():
                local_name = name
                if local_name in index.get("personal", []):
                    local_name = f"{name}_imported_{uuid.uuid4().hex[:4]}"
                
                new_pid = f"A{uuid.uuid4().hex[:15].upper()}"
                recip_dir = os.path.join(self.cog.USERS_DIR, user_id_str, "profiles", new_pid)
                os.makedirs(recip_dir, exist_ok=True)

                config = p_data.get("config", {})
                prompts = p_data.get("prompts", {})

                config["profile_id"] = new_pid
                config["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

                unified_profile = {
                    "name": local_name,
                    "config": config,
                    "prompts": prompts,
                    "child_bot": None
                }
                self._save_profile_by_pid(user_id, new_pid, unified_profile)

                ltm_list = p_data.get("ltm", [])
                if ltm_list:
                    self.cog.storage_manager._atomic_json_save_gzip({"guild": ltm_list}, os.path.join(recip_dir, "ltm.json.gz"))

                training_list = p_data.get("training", [])
                if training_list:
                    self.cog.storage_manager._atomic_json_save_gzip(training_list, os.path.join(recip_dir, "training.json.gz"))

                if config.get("custom_display_name") or config.get("custom_avatar_url"):
                    self.cog.user_appearances.setdefault(user_id_str, {})[local_name] = {
                        "custom_display_name": config.get("custom_display_name"),
                        "custom_avatar_url": config.get("custom_avatar_url")
                    }

                if isinstance(index.get("personal"), dict):
                    index["personal"][local_name] = new_pid
                else:
                    index.setdefault("personal", []).append(local_name)

                import_log.append(f"- `{local_name}`")

            self._save_user_index(user_id, index)
            
            await interaction.followup.send(f"### 📥 Import Successful\nThe following profiles have been securely decrypted and added to your vault:\n" + "\n".join(import_log), ephemeral=True)

        except ValueError as ve:
            await interaction.followup.send(f"❌ **Import Rejected:** {ve}", ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"❌ **Import Failed:** An unexpected error occurred: {e}", ephemeral=True)

    async def _execute_privacy_export(self, user_id: int, interaction: discord.Interaction):
        user_id_str = str(user_id)
        user_dir = os.path.join(self.cog.USERS_DIR, user_id_str)
        
        if not os.path.exists(user_dir):
            await interaction.followup.send("No data found for your account.", ephemeral=True)
            return
            
        import tempfile

        def decrypt_payload(data):
            if isinstance(data, dict):
                return {k: decrypt_payload(v) for k, v in data.items()}
            elif isinstance(data, list):
                return[decrypt_payload(i) for i in data]
            elif isinstance(data, str):
                if data.startswith("gAAAAA"):
                    return self.cog.storage_manager._decrypt_data(data)
                return data
            else:
                return data

        def make_zip():
            temp_d = tempfile.mkdtemp()
            export_base = os.path.join(temp_d, f"mimicai_data_{user_id_str}")
            os.makedirs(export_base)
            
            for root, dirs, files in os.walk(user_dir):
                rel_path = os.path.relpath(root, user_dir)
                target_dir = os.path.join(export_base, rel_path) if rel_path != '.' else export_base
                os.makedirs(target_dir, exist_ok=True)
                
                for file in files:
                    src_file = os.path.join(root, file)
                    
                    if file.endswith('.json.gz'):
                        is_encrypted = file != 'child_bot.json.gz'
                        data = self.cog.storage_manager._load_json_gzip(src_file, encrypted=is_encrypted)
                        
                        if data is not None:
                            decrypted_data = decrypt_payload(data)
                            target_file = os.path.join(target_dir, file[:-3]) 
                            with open(target_file, 'wb') as f:
                                f.write(json.dumps(decrypted_data, option=json.OPT_INDENT_2))
                    
                    elif file.endswith('.json') or file.endswith('.txt'):
                        target_file = os.path.join(target_dir, file)
                        shutil.copy2(src_file, target_file)
            
            zip_base = os.path.join(temp_d, f"mimicai_data_{user_id_str}")
            shutil.make_archive(zip_base, 'zip', export_base)
            return temp_d, f"{zip_base}.zip"
            
        temp_dir, zip_path = await asyncio.to_thread(make_zip)
        
        try:
            file = discord.File(zip_path, filename=f"privacy_export_{user_id_str}.zip")
            await interaction.followup.send("Here is your complete data export. This archive contains your profiles, API keys, and memory data in unencrypted, uncompressed JSON format.", file=file, ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"Failed to send export file: {e}", ephemeral=True)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def _execute_account_deletion(self, user_id: int, interaction: discord.Interaction):
        user_id_str = str(user_id)
        
        # 1. Shutdown Child Bots
        child_bots_to_kill = [bot_id for bot_id, data in self.cog.child_bots.items() if str(data.get("owner_id")) == user_id_str]
        for bot_id in child_bots_to_kill:
            await self.cog.manager_queue.put({"action": "shutdown_bot", "bot_id": bot_id})
            self.cog.child_bots.pop(bot_id, None)
            
        # 2. Scrub from Public Hub
        pids_to_unpublish = []
        for pub_id, info in self.cog.public_profiles.items():
            if isinstance(info, str) and ":" in info:
                if info.startswith(user_id_str + ":"):
                    pids_to_unpublish.append(pub_id)
            elif isinstance(info, dict) and str(info.get("owner_id")) == user_id_str:
                pids_to_unpublish.append(pub_id)

        for pid in pids_to_unpublish:
            self.cog.public_profiles.pop(pid, None)
        if pids_to_unpublish:
            self._save_public_index()
            
        # 3. Scrub from Server Pointers
        for g in self.cog.bot.guilds:
            idx = self.cog.server_manager._get_server_index(str(g.id))
            changed = False
            for prov in list(idx.get("assigned_keys", {}).keys()):
                if idx["assigned_keys"][prov].get("user_id") == user_id:
                    del idx["assigned_keys"][prov]
                    self.cog.server_key_pointers.pop((g.id, prov), None)
                    changed = True
            if changed:
                self.cog.server_manager._save_server_index(str(g.id), idx)
                
        # 4. Delete Core User Directory
        user_dir = os.path.join(self.cog.USERS_DIR, user_id_str)
        if os.path.exists(user_dir):
            shutil.rmtree(user_dir, ignore_errors=True)
            
        # 5. Remove from In-Memory Dicts
        self.cog.user_indices.pop(user_id_str, None)
        self.cog.user_appearances.pop(user_id_str, None)
        self.cog.profile_shares.pop(user_id_str, None)
        
        keys_to_del_cache = [k for k in self.cog.decrypted_key_cache.keys() if k[0] == user_id]
        for k in keys_to_del_cache:
            self.cog.decrypted_key_cache.pop(k, None)
        
        # 6. Cancel/Delete active global chat sessions
        keys_to_del = [k for k in self.cog.global_chat_sessions.keys() if isinstance(k, tuple) and len(k) == 3 and k[0] == 'global' and k[1] == user_id]
        for k in keys_to_del:
            self.cog.global_chat_sessions.pop(k, None)
            self.cog.session_last_accessed.pop(k, None)
            self.cog.ltm_recall_history.pop(k, None)

        await interaction.followup.send("Account Deleted. All your profiles, memories, and settings have been permanently erased from this instance.", ephemeral=True)

    def _generate_unique_local_name(self, user_id: int, original_name: str, sharer_name: str) -> str:
        index = self._get_user_index(user_id)
        all_profile_names = set(index.get("personal", [])) | set(index.get("borrowed", []))
        
        base_name = f"{original_name}-{sharer_name}".lower().strip()
        if base_name not in all_profile_names:
            return base_name
        
        counter = 2
        while True:
            new_name = f"{base_name}-{counter}"
            if new_name not in all_profile_names:
                return new_name
            counter += 1

    async def _build_participant_embed(self, participant: Dict, channel_id: int) -> discord.Embed:
        owner_id = participant['owner_id']
        profile_name = participant['profile_name']
        
        _, _, _, temp, topp, topk, training_ctx, training_rel, prim_model, fall_model = await asyncio.to_thread(
            self.cog.session_manager._get_user_profile_for_model, owner_id, channel_id, profile_name
        )

        effective_owner_id, effective_profile_name = self._resolve_effective_profile(owner_id, profile_name)
        
        index = self._get_user_index(owner_id)
        is_borrowed = profile_name in index.get("borrowed", [])
        
        profile_data = self._get_profile_config(owner_id, profile_name, is_borrowed) or {}

        appearance = self._get_user_appearance(effective_owner_id, effective_profile_name)
        display_name = appearance.get("custom_display_name") or effective_profile_name
        appearance_text = f"`{effective_profile_name}`" if appearance else "None"
        
        ltm_shard = self.cog.memory_manager._load_ltm_shard(str(owner_id), profile_name)
        ltm_count = len(ltm_shard.get("guild", [])) if ltm_shard else 0
        training_shard = self.cog.memory_manager._load_training_shard(str(effective_owner_id), effective_profile_name)
        training_count = len(training_shard) if training_shard else 0

        realistic_typing = profile_data.get("realistic_typing_enabled", False)
        timezone_str = profile_data.get("timezone", "UTC")
        
        grounding_mode = profile_data.get("grounding_mode", "off")
        if isinstance(grounding_mode, bool): grounding_mode = "on" if grounding_mode else "off"
        grounding_display = {"off": "`OFF`", "native": "**`NATIVE`**", "rag": "**`RAG`**"}.get(grounding_mode, "OFF")

        stm_length = profile_data.get("stm_length", defaultConfig.CHATBOT_MEMORY_LENGTH)
        ltm_ctx = profile_data.get("ltm_context_size", 3)
        ltm_rel = profile_data.get("ltm_relevance_threshold", 0.75)
        access_display = CHANNEL_ACCESS_LABELS[self._resolve_enforced_safety_level(owner_id, profile_name)]
        ltm_creation_status = "**`ON`**" if profile_data.get("ltm_creation_enabled", False) else "`OFF`"

        created_str = profile_data.get('created_at')
        created_display = "Unknown"
        if created_str:
            try:
                dt = datetime.datetime.fromisoformat(created_str)
                ts = int(dt.timestamp())
                created_display = f"<t:{ts}:D>\n(<t:{ts}:R>)"
            except: pass

        profile_type = "Personal"
        if is_borrowed:
            owner_user = self.cog.bot.get_user(effective_owner_id)
            owner_name = owner_user.name if owner_user else "Unknown User"
            profile_type = f"Borrowed (from {owner_name})"

        embed = discord.Embed(title=f"Participant: {display_name}", color=discord.Color.blue())
        
        embed.add_field(name="Profile Type", value=f"`{profile_type}`", inline=True)
        embed.add_field(name="Created", value=created_display, inline=True)
        embed.add_field(name="Display Name", value=f"`{display_name}`", inline=True)
        
        # Same column discipline as the profile dashboard: identity on the left,
        # Public Library in the middle, channel access on the right. Borrowed
        # profiles carry a second identity field and fill the row on their own.
        is_public = self._is_profile_public(owner_id, profile_name)
        library_display = "🌐 `Published`" if is_public else "`Not published`"

        if is_borrowed:
            borrowed_config = self._get_profile_config(owner_id, profile_name, True) or {}
            # Not "A class" and "B class": the source of a borrow can be a personal
            # profile (A) or a System one (X), and the local PID is B or C depending
            # on whether the borrow came through a share code or the Public Library.
            source_pid = borrowed_config.get("original_profile_id", "Unknown")
            local_pid = self._get_pid_from_name_any(owner_id, profile_name)
            embed.add_field(name="Profile ID (Source)", value=f"`{source_pid}`", inline=True)
            embed.add_field(name="Profile ID (Local)", value=f"`{local_pid}`", inline=True)
            embed.add_field(name="\u200b", value="\u200b", inline=True)
            embed.add_field(name="\u200b", value="\u200b", inline=True)
        else:
            profile_id = self._get_profile_id(effective_owner_id, effective_profile_name)
            embed.add_field(name="Profile ID (PID)", value=f"`{profile_id}`", inline=True)

        embed.add_field(name="Public Library", value=library_display, inline=True)
        embed.add_field(name="Channel Access", value=f"`{access_display}`", inline=True)

        embed.add_field(name="\u200b", value="**Core Settings**", inline=False)
        embed.add_field(name="Primary Model", value=f"`{prim_model}`", inline=True)
        embed.add_field(name="Fallback Model", value=f"`{fall_model}`", inline=True)
        embed.add_field(name="Appearance", value=appearance_text, inline=True)
        
        embed.add_field(name="Grounding", value=grounding_display, inline=True)
        embed.add_field(name="Realistic Typing", value="**`ON`**" if realistic_typing else "`OFF`", inline=True)
        embed.add_field(name="Timezone", value=f"`{timezone_str}`", inline=True)

        embed.add_field(name="\u200b", value="**Generation Parameters**", inline=False)
        embed.add_field(name="Temperature", value=f"`{temp}`", inline=True)
        embed.add_field(name="Top P", value=f"`{topp}`", inline=True)
        embed.add_field(name="Top K", value=f"`{topk}`", inline=True)
        embed.add_field(name="STM Length", value=f"`{stm_length}`", inline=True)

        embed.add_field(name="\u200b", value="**Training & Memory**", inline=False)
        embed.add_field(name="Train Ctx", value=f"`{training_ctx}`", inline=True)
        embed.add_field(name="Train Rel", value=f"`{training_rel}`", inline=True)
        embed.add_field(name="Train Count", value=f"`{training_count}`", inline=True)
        embed.add_field(name="LTM Ctx", value=f"`{ltm_ctx}`", inline=True)
        embed.add_field(name="LTM Rel", value=f"`{ltm_rel}`", inline=True)
        embed.add_field(name="LTM Info", value=f"Count: `{ltm_count}`\nAuto-Creation: {ltm_creation_status}", inline=True)
        
        if appearance.get("custom_avatar_url"):
            embed.set_thumbnail(url=appearance["custom_avatar_url"])
            
        return embed
    

    async def _build_profile_embed(self, user_id: int, profile_name: str, channel_id: int) -> discord.Embed:
        index = self._get_user_index(user_id)
        is_borrowed = profile_name in index.get("borrowed", [])
        
        embed = discord.Embed(title=f"Profile Dashboard: '{profile_name}'", color=discord.Color.blue())
        
        effective_owner_id, effective_profile_name = self._resolve_effective_profile(user_id, profile_name)
        profile_type = "Personal"

        if is_borrowed:
            owner_user = self.cog.bot.get_user(effective_owner_id)
            profile_type = f"Borrowed (from {owner_user.name if owner_user else 'Unknown User'})"

        _, _, _, temp, top_p, top_k, train_ctx, train_rel, prim_model, fall_model = await asyncio.to_thread(
            self.cog.session_manager._get_user_profile_for_model, user_id, channel_id, profile_name
        )

        config = self._get_profile_config(user_id, profile_name, is_borrowed) or {}
        
        ltm_shard = self.cog.memory_manager._load_ltm_shard(str(user_id), profile_name)
        ltm_count = len(ltm_shard.get("guild", [])) if ltm_shard else 0
        training_shard = self.cog.memory_manager._load_training_shard(str(effective_owner_id), effective_profile_name)
        train_count = len(training_shard) if training_shard else 0

        created_str = config.get('created_at')
        created_display = "Unknown"
        if created_str:
            try:
                dt = datetime.datetime.fromisoformat(created_str)
                ts = int(dt.timestamp())
                created_display = f"<t:{ts}:D>"
            except: pass
        
        appearance_data = self._get_user_appearance(effective_owner_id, effective_profile_name)
        display_name = appearance_data.get("custom_display_name") or effective_profile_name

        embed.add_field(name="Profile Type", value=f"`{profile_type}`", inline=True)
        embed.add_field(name="Created", value=created_display, inline=True)
        embed.add_field(name="Display Name", value=f"`{display_name}`", inline=True)
        
        enforced = self._resolve_enforced_safety_level(user_id, profile_name)
        verdict, verdict_reason = self._content_rating_state(user_id, profile_name)
        declared = self._is_owner_declared_adult(user_id, profile_name)
        is_public = self._is_profile_public(user_id, profile_name)

        rating_display = (f"{CONTENT_RATING_EMOJI[verdict]} "
                          f"`{CONTENT_RATING_LABELS[verdict]}`")
        # The padlock marks a verdict the owner cannot move themselves, which is
        # every adult verdict except their own declaration.
        if verdict == CONTENT_RATING_ADULT and not declared:
            rating_display += " 🔒"

        # Discord packs inline fields three to a row, so these are emitted as two
        # deliberate rows rather than in the order they were computed: identity in
        # the left column, content state in the middle, consequence on the right.
        # Content Rating and Public Library sharing the middle column is the point --
        # read downwards, they are the two fields that say where the profile may go.
        # The zero-width fields are the padding that holds the grid square.
        if is_borrowed:
            borrowed_config = self._get_profile_config(user_id, profile_name, True) or {}
            source_pid = borrowed_config.get("original_profile_id", "Unknown")
            local_pid = self._get_pid_from_name_any(user_id, profile_name)
            left_column = [("Profile ID (Source)", f"`{source_pid}`"),
                           ("Profile ID (Local)", f"`{local_pid}`")]
        else:
            profile_id = self._get_profile_id(effective_owner_id, effective_profile_name)
            left_column = [("Profile ID (PID)", f"`{profile_id}`"),
                           ("\u200b", "\u200b")]

            if profile_id.startswith("X"):
                embed.description = f"⚠️ **System Profile.** Global settings managed by Bot Admin.\n\n" + (embed.description or "")
                profile_type = "System"

        embed.add_field(name=left_column[0][0], value=left_column[0][1], inline=True)
        embed.add_field(name="Content Rating", value=rating_display, inline=True)
        embed.add_field(name="Channel Access",
                        value=f"`{CHANNEL_ACCESS_LABELS[enforced]}`", inline=True)

        embed.add_field(name=left_column[1][0], value=left_column[1][1], inline=True)
        embed.add_field(name="Public Library",
                        value="🌐 `Published`" if is_public else "`Not published`", inline=True)
        embed.add_field(name="\u200b", value="\u200b", inline=True)

        if verdict == CONTENT_RATING_PENDING and verdict_reason:
            # Surface why a profile is stuck on Pending. A silent Pending was
            # indistinguishable from one still in flight. This reason describes an
            # operational failure rather than the persona, so it is shown as stored.
            embed.description = ((embed.description or "") +
                                 f"\n⏳ **Classification unavailable:** {verdict_reason}").strip()
        if verdict == "adult" and declared:
            embed.description = ((embed.description or "") +
                                 "\n🔞 **Declared as adult content by you.** This profile is "
                                 "limited to age-restricted channels and cannot be published. "
                                 "Withdraw the declaration from Content Safety to have it "
                                 "classified normally.").strip()
        elif verdict == "adult":
            # A category, never the classifier's own words about the persona.
            embed.description = ((embed.description or "") +
                                 "\n🔞 **Classified as adult content: "
                                 + CONTENT_RATING_REASON_LABELS.get(
                                     verdict_reason, CONTENT_RATING_REASON_FALLBACK) +
                                 ".** This profile is limited to age-restricted channels. "
                                 "Edit the persona and submit it again from Content Safety, "
                                 "or contact the bot operator to dispute the "
                                 "result.").strip()

        # Re-check off the turn path: the edit hooks cover the normal cases, this
        # catches imports and anything they miss.
        asyncio.create_task(self.resolve_stale_rating(user_id, profile_name))

        if is_public:
            embed.description = ((embed.description or "") +
                                 "\n🌐 **Published to the Public Library.** The 18+ declaration is "
                                 "withheld while listed, and bulk rating changes skip this profile. "
                                 "Unpublish via `/profile hub` to change it.").strip()

        embed.add_field(name="\u200b", value="\u200b", inline=False)

        def clean_m(m_str):
            if not m_str: return "None"
            return str(m_str)

        img_model = config.get("image_generation_model", "gemini-2.5-flash-image")
        aud_model = config.get("speech_model", "gemini-2.5-flash-preview-tts")
        grd_model = config.get("grounding_rag_model", FALLBACK_MODEL_NAME)
        crt_model = config.get("critic_model", FALLBACK_MODEL_NAME)
        ltm_model = config.get("ltm_model", FALLBACK_MODEL_NAME)

        models_val = (
            f"Primary: `{clean_m(prim_model)}`\n"
            f"Fallback: `{clean_m(fall_model)}`\n"
            f"Image: `{clean_m(img_model)}`\n"
            f"Audio: `{clean_m(aud_model)}`\n"
            f"Grounding: `{clean_m(grd_model)}`\n"
            f"Critic: `{clean_m(crt_model)}`\n"
            f"LTM: `{clean_m(ltm_model)}`"
        )
        embed.add_field(name="Models", value=models_val, inline=False)

        stm_length = config.get("stm_length", defaultConfig.CHATBOT_MEMORY_LENGTH)
        gen_val = (
            f"Temp: `{temp}`\n"
            f"Top P: `{top_p}`\n"
            f"Top K: `{top_k}`\n"
            f"STM Length: `{stm_length}`"
        )
        embed.add_field(name="\u200bGeneration Parameters", value=gen_val, inline=True)

        freq_p = config.get("frequency_penalty", 0.0)
        pres_p = config.get("presence_penalty", 0.0)
        rep_p = config.get("repetition_penalty", 0.0)
        min_p = config.get("min_p", 0.0)
        top_a = config.get("top_a", 0.0)

        adv_val = (
            f"Freq P: `{freq_p}`\n"
            f"Pres P: `{pres_p}`\n"
            f"Rep P: `{rep_p}`\n"
            f"Min P: `{min_p}`\n"
            f"Top A: `{top_a}`"
        )
        embed.add_field(name="Advanced (OpenRouter Only)", value=adv_val, inline=True)

        t_summary_raw = config.get("thinking_summary_visible", "off").lower()
        t_summary = "**`ON`**" if t_summary_raw == "on" else "`OFF`"
        t_level = config.get("thinking_level", "high").title()
        t_budget = config.get("thinking_budget", -1)
        budget_display = "Dynamic (-1)" if t_budget == -1 else f"{t_budget}"

        thinking_val = (
            f"Summary: {t_summary}\n"
            f"Effort: `{t_level}`\n"
            f"Budget: `{budget_display}`"
        )
        embed.add_field(name="Thinking/Reasoning", value=thinking_val, inline=True)

        img_gen = "**`ON`**" if config.get("image_generation_enabled", False) else "`OFF`"
        
        raw_ground_mode = config.get("grounding_mode", "off")
        if isinstance(raw_ground_mode, bool): raw_ground_mode = "rag" if raw_ground_mode else "off"
        elif raw_ground_mode in ["on", "on+"]: raw_ground_mode = "rag"
        grounding_display = {"off": "`OFF`", "native": "**`NATIVE`**", "rag": "**`RAG`**"}.get(raw_ground_mode, "`OFF`")
        
        raw_url_mode = config.get("url_mode", "off")
        if "url_mode" not in config:
            raw_url_mode = "rag" if config.get("url_fetching_enabled", False) else "off"
        url_ctx = {"off": "`OFF`", "native": "**`NATIVE`**", "rag": "**`RAG`**"}.get(raw_url_mode, "`OFF`")
        
        timezone = config.get("timezone", "UTC")
        typing = "**`ON`**" if config.get("realistic_typing_enabled", False) else "`OFF`"
        critic = "**`ON`**" if config.get("critic_enabled", False) else "`OFF`"
        help_mode = "**`ON`**" if config.get("help_mode_enabled", False) else "`OFF`"
        resp_mode = config.get("response_mode", "regular").replace('_', ' ').title()

        ph_text = f"{config.get('placeholder_emoji') or 'Default'}"
        
        tools_val = (
            f"Image Gen: {img_gen}\n"
            f"Grounding: {grounding_display}\n"
            f"URL Context: {url_ctx}\n"
            f"Response Mode: `{resp_mode}`\n"
            f"Timezone: `{timezone}`\n"
            f"Realistic Typing: {typing}\n"
            f"Critic: {critic}\n"
            f"Help Mode: {help_mode}\n"
            f"Placeholder: {ph_text}"
        )
        embed.add_field(name="Tools", value=tools_val, inline=True)

        neuro_status = "**`ON`**" if config.get("neuro_engine_enabled", False) else "`OFF`"
        neuro_state = config.get("neuro_state", {"dopamine": 50, "cortisol": 20, "oxytocin": 50, "adrenaline": 20})
        neuro_val = (
            f"Status: {neuro_status}\n"
            f"Dopamine: `{neuro_state.get('dopamine', 50)}`\n"
            f"Cortisol: `{neuro_state.get('cortisol', 20)}`\n"
            f"Oxytocin: `{neuro_state.get('oxytocin', 50)}`\n"
            f"Adrenaline: `{neuro_state.get('adrenaline', 20)}`"
        )
        embed.add_field(name="Neuro Engine", value=neuro_val, inline=True)

        s_voice = config.get("speech_voice", "Aoede")
        s_model = config.get("speech_model", "gemini-2.5-flash-preview-tts")
        s_temp = config.get("speech_temperature", 1.0)
        s_enabled = "**`ON`**" if config.get("speech_tts_enabled", False) else "`OFF`"
        
        speech_val = (
            f"Enabled: {s_enabled}\n"
            f"Voice: `{s_voice}`\n"
            f"Temperature: `{s_temp}`"
        )
        embed.add_field(name="Speech TTS", value=speech_val, inline=True)

        train_val = (
            f"Count: `{train_count}`\n"
            f"Context Size: `{train_ctx}`\n"
            f"Relevance Threshold: `{train_rel}`"
        )
        embed.add_field(name="Training Examples", value=train_val, inline=True)

        ltm_ctx = config.get("ltm_context_size", 3)
        ltm_rel = config.get("ltm_relevance_threshold", 0.75)
        ltm_status = "**`ON`**" if config.get("ltm_creation_enabled", False) else "`OFF`"
        ltm_inv = config.get("ltm_creation_interval", 10)
        ltm_s_ctx = config.get("ltm_summarization_context", 10)

        ltm_val = (
            f"Auto-Creation: {ltm_status}\n"
            f"Count: `{ltm_count}`\n"
            f"Creation Interval: `{ltm_inv}`\n"
            f"Summ Context: `{ltm_s_ctx}`\n"
            f"Context Size: `{ltm_ctx}`\n"
            f"Relevance Threshold: `{ltm_rel}`"
        )
        embed.add_field(name="Long-Term Memories", value=ltm_val, inline=True)

        if appearance_data.get("custom_avatar_url"):
            embed.set_thumbnail(url=appearance_data["custom_avatar_url"])

        return embed
    

    def _default_ltm_summarization_instructions(self) -> str:
        """The LTM summariser prompt a profile gets when it has none of its own.

        /mod's "LTM Summarization" entry writes global_prompts, which nothing read
        before -- summarisation resolves the prompt per-profile. This makes the
        global the source of that per-profile default, so it governs newly created
        profiles, the GUI's reset-to-default, and any profile whose stored value is
        blank. A profile that already has its own text keeps it; changing the global
        does not rewrite existing profiles.
        """
        return self.cog.global_prompts.get(
            "LTM_SUMMARIZATION_INSTRUCTIONS", DEFAULT_LTM_SUMMARIZATION_INSTRUCTIONS)

    async def _build_profile_manage_embed(self, interaction: discord.Interaction, profile_name: str, target_user_id: Optional[int] = None) -> discord.Embed:
        return await self._build_profile_embed(target_user_id or interaction.user.id, profile_name, interaction.channel_id)
    

    def _invalidate_channel_model_cache(self, key: Tuple[int, int]):
        if key in self.cog.channel_models: del self.cog.channel_models[key]
        self.cog.channel_model_last_profile_key.pop(key, None)

    async def update_profile_advanced_params(self, user_id: int, profile_name: str, params: Dict[str, Any], channel_id_context: int, is_borrowed: bool) -> bool:
        if not self.cog.has_lock: return False

        def _sync_update():
            target = self._get_profile_config(user_id, profile_name, is_borrowed)
            if not target: return None

            for k, v in params.items():
                if v is None:
                    if k in target: del target[k] # Reset to default (remove override)
                else:
                    target[k] = v

            self._save_profile_config(user_id, profile_name, target, is_borrowed)
            return self.cog.session_manager._get_active_user_profile_name_for_channel(user_id, channel_id_context) == profile_name

        is_active = await asyncio.to_thread(_sync_update)
        if is_active is None: return False

        if is_active:
            self._invalidate_channel_model_cache((channel_id_context, user_id))

        return True
    

    async def update_user_profile_persona(self, user_id: int, profile_name: str, persona_data: Dict[str, List[str]], channel_id_context: int) -> bool:
        if not self.cog.has_lock: return False

        def _sync_update():
            index = self._get_user_index(user_id)
            if profile_name not in index.get("personal", []): return None

            prompts = self._get_profile_prompts(user_id, profile_name) or {}
            prompts["persona"] = persona_data
            self._save_profile_prompts(user_id, profile_name, prompts)

            return self.cog.session_manager._get_active_user_profile_name_for_channel(user_id, channel_id_context) == profile_name

        is_active = await asyncio.to_thread(_sync_update)
        if is_active is None: return False

        if is_active:
            self._invalidate_channel_model_cache((channel_id_context, user_id))
        return True

    async def update_user_profile_ai_instructions(self, user_id: int, profile_name: str, instructions: str, channel_id_context: int) -> bool:
        if not self.cog.has_lock: return False

        def _sync_update():
            index = self._get_user_index(user_id)
            if profile_name not in index.get("personal", []): return None

            prompts = self._get_profile_prompts(user_id, profile_name) or {}
            prompts["ai_instructions"] = instructions
            self._save_profile_prompts(user_id, profile_name, prompts)

            return self.cog.session_manager._get_active_user_profile_name_for_channel(user_id, channel_id_context) == profile_name

        is_active = await asyncio.to_thread(_sync_update)
        if is_active is None: return False

        if is_active:
            self._invalidate_channel_model_cache((channel_id_context, user_id))
        return True
        

    async def update_profile_generation_params(self, user_id: int, profile_name: str, params: Dict[str, Any], channel_id_context: int, is_borrowed: bool) -> bool:
        if not self.cog.has_lock: return False

        def _sync_update():
            profile = self._get_profile_config(user_id, profile_name, is_borrowed)
            if not profile: return None

            if "temperature" in params: profile["temperature"] = params["temperature"]
            if "top_p" in params: profile["top_p"] = params["top_p"]
            if "top_k" in params: profile["top_k"] = params["top_k"]
            if "stm_length" in params: profile["stm_length"] = params["stm_length"]

            self._save_profile_config(user_id, profile_name, profile, is_borrowed)
            return self.cog.session_manager._get_active_user_profile_name_for_channel(user_id, channel_id_context) == profile_name

        is_active = await asyncio.to_thread(_sync_update)
        if is_active is None: return False

        if is_active:
            self._invalidate_channel_model_cache((channel_id_context, user_id))
        return True

    async def update_profile_training_params(self, user_id: int, profile_name: str, params: Dict[str, Any]) -> bool:
        if not self.cog.has_lock: return False

        def _sync_update():
            profile = self._get_profile_config(user_id, profile_name, False)
            if not profile: return False

            if "training_context_size" in params: profile["training_context_size"] = params["training_context_size"]
            if "training_relevance_threshold" in params: profile["training_relevance_threshold"] = params["training_relevance_threshold"]
            self._save_profile_config(user_id, profile_name, profile, False)
            return True

        return await asyncio.to_thread(_sync_update)


    async def update_profile_ltm_params(self, user_id: int, profile_name: str, params: Dict[str, Any]) -> bool:
        if not self.cog.has_lock: return False

        def _sync_update():
            index = self._get_user_index(user_id)
            is_borrowed = profile_name in index.get("borrowed", [])
            profile = self._get_profile_config(user_id, profile_name, is_borrowed)
            if not profile: return False

            if "ltm_context_size" in params: profile["ltm_context_size"] = params["ltm_context_size"]
            if "ltm_relevance_threshold" in params: profile["ltm_relevance_threshold"] = params["ltm_relevance_threshold"]
            if "ltm_creation_interval" in params: profile["ltm_creation_interval"] = params["ltm_creation_interval"]
            if "ltm_summarization_context" in params: profile["ltm_summarization_context"] = params["ltm_summarization_context"]

            self._save_profile_config(user_id, profile_name, profile, is_borrowed)
            return True

        return await asyncio.to_thread(_sync_update)


    async def update_profile_models(self, user_id: int, profile_name: str, primary_model: Optional[str], fallback_model: Optional[str], is_borrowed: bool, channel_id_context: int, show_fallback_indicator: Optional[bool] = None) -> bool:
        if not self.cog.has_lock: return False

        def _sync_update():
            profile = self._get_profile_config(user_id, profile_name, is_borrowed)
            if not profile: return False

            if primary_model: profile["primary_model"] = primary_model
            if fallback_model: profile["fallback_model"] = fallback_model
            if show_fallback_indicator is not None: profile["show_fallback_indicator"] = show_fallback_indicator

            self._save_profile_config(user_id, profile_name, profile, is_borrowed)
            return True

        if not await asyncio.to_thread(_sync_update):
            return False

        keys_to_delete = []
        for key in list(self.cog.channel_models.keys()):
            key_user_id = None
            if isinstance(key, tuple) and len(key) == 2:
                key_user_id = key[1]
            
            if key_user_id == user_id:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            self._invalidate_channel_model_cache(key)

        return True
    

    async def _accept_share_request(self, interaction: discord.Interaction, sharer_id: int, target_pid: Optional[str], fallback_name: str, desired_name: str, is_public_borrow: bool = False):
        def _sync_prepare():
            current_name = self._get_name_from_pid(sharer_id, target_pid) if target_pid else fallback_name
            if not current_name: current_name = fallback_name

            owner_profile_data = self._get_profile_config(sharer_id, current_name, False)
            if not owner_profile_data:
                return None

            index = self._get_user_index(interaction.user.id)
            current_borrowed = len(index.get("borrowed", {})) if isinstance(index.get("borrowed"), dict) else len(index.get("borrowed", []))
            return current_name, owner_profile_data, index, current_borrowed

        prep = await asyncio.to_thread(_sync_prepare)
        if prep is None:
            await interaction.followup.send("The shared profile seems to no longer exist.", ephemeral=True)
            return
        current_name, owner_profile_data, index, current_borrowed = prep

        limit = defaultConfig.LIMIT_BORROWED
        if current_borrowed >= limit:
            await interaction.followup.send(f"Limit Reached. You have {current_borrowed}/{limit} borrowed profiles.", ephemeral=True)
            return

        def _sync_save():
            target_original_pid = target_pid or owner_profile_data.get("profile_id", "00000000")
            source_pointer = f"{sharer_id}:{target_original_pid}"
            if is_public_borrow:
                # Was an equality test against the raw value, which only ever matched
                # the "owner:pid" string form -- a dict entry fell through to the raw
                # pointer and the borrow lost its link to the public listing.
                public_key = self._find_public_entry_id(sharer_id, target_original_pid)
                pointer_value = public_key if public_key else source_pointer
            else:
                pointer_value = source_pointer

            snapshot_data = owner_profile_data.copy()
            snapshot_data.update({
                "original_owner_id": str(sharer_id),
                "original_pid": target_original_pid,
                "original_profile_name": current_name,
                "original_profile_id": target_original_pid,
                "pointer": pointer_value,
                "borrowed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "ltm_creation_enabled": False
            })

            # The class letter records provenance, not current state: 'C' means the
            # borrow arrived through the Public Library, 'B' through a private share
            # code. A PID is the on-disk folder name and never changes, so the owner
            # later unpublishing does not turn a C back into a B -- the pointer above
            # is what tracks that.
            #
            # Nothing branches on the letter, and nothing may start to. Every borrow
            # created before this existed is a 'B' whatever its origin, so code that
            # needs to know whether a borrow is public must read `pointer` or ask
            # _find_public_entry_id. Renaming the old folders to match would mean
            # moving directories and rewriting index entries for cosmetics, with
            # orphaned profiles as the failure mode; the mixed population stays.
            pid = f"{'C' if is_public_borrow else 'B'}{uuid.uuid4().hex[:15].upper()}"

            if "borrowed" not in index or not isinstance(index["borrowed"], dict):
                index["borrowed"] = {}

            index["borrowed"][desired_name] = pid
            self._save_user_index(interaction.user.id, index)

            unified_borrowed = {
                "name": desired_name,
                "config": snapshot_data,
                "prompts": {},
                "child_bot": None
            }
            self._save_profile_by_pid(interaction.user.id, pid, unified_borrowed)

        await asyncio.to_thread(_sync_save)

        if not is_public_borrow:
            await self._reject_share_request(interaction, sharer_id, target_pid, fallback_name, notify_sharer=True, accepted=True)

    async def _reject_share_request(self, interaction: discord.Interaction, sharer_id: int, target_pid: Optional[str], fallback_name: str, notify_sharer: bool = True, accepted: bool = False):
        recipient_id_str = str(interaction.user.id)
        if recipient_id_str in self.cog.profile_shares:
            if target_pid:
                updated_shares = [s for s in self.cog.profile_shares[recipient_id_str] if not (s['sharer_id'] == sharer_id and s.get('original_pid') == target_pid)]
            else:
                updated_shares =[s for s in self.cog.profile_shares[recipient_id_str] if not (s['sharer_id'] == sharer_id and s['profile_name'] == fallback_name)]
                
            if not updated_shares:
                del self.cog.profile_shares[recipient_id_str]
            else:
                self.cog.profile_shares[recipient_id_str] = updated_shares
            self._save_profile_share_shard(recipient_id_str, updated_shares)

        if notify_sharer:
            sharer = self.cog.bot.get_user(sharer_id)
            if sharer:
                status = "accepted" if accepted else "rejected"
                try:
                    await sharer.send(f"Your share request for '{fallback_name}' to **{interaction.user.name}** was **{status}**.")
                except discord.Forbidden:
                    pass

    async def _validate_active_profile(self, user_id: int, channel: discord.abc.Messageable) -> bool:
        index = self._get_user_index(user_id)
        active_profile_name = self.cog.session_manager._get_active_user_profile_name_for_channel(user_id, channel.id)
        if not active_profile_name: return True

        if active_profile_name in index.get("borrowed", []):
            borrowed_data = self._get_profile_config(user_id, active_profile_name, True) or {}
            owner_id = int(borrowed_data.get("original_owner_id", 0))
            owner_profile_name = borrowed_data.get("original_profile_name", active_profile_name)
            
            owner_index = self._get_user_index(owner_id)
            if owner_profile_name not in owner_index.get("personal", []):
                if isinstance(index["borrowed"], dict):
                    pid = index["borrowed"].pop(active_profile_name, active_profile_name)
                else:
                    index["borrowed"].remove(active_profile_name)
                    pid = active_profile_name
                
                self._save_user_index(user_id, index)
                
                channel_obj = self.cog.bot.get_channel(channel.id) if hasattr(channel, 'id') else channel
                server_id_str = str(channel_obj.guild.id) if channel_obj and getattr(channel_obj, 'guild', None) else "dm"
                server_index = self.cog.server_manager._get_server_index(server_id_str)
                if str(channel.id) in server_index.setdefault("user_active_profiles", {}).setdefault(str(user_id), {}):
                    del server_index["user_active_profiles"][str(user_id)][str(channel.id)]
                self.cog.server_manager._save_server_index(server_id_str, server_index)
                
                try:
                    p_dir = os.path.join(self.cog.USERS_DIR, str(user_id), "profiles", pid)
                    shutil.rmtree(p_dir, ignore_errors=True)
                    await channel.send(f"<@{user_id}>, the borrowed profile '{active_profile_name}' is broken because the original was deleted or renamed. It has been removed from your list and your active profile in this channel has been reset.")
                except discord.Forbidden:
                    pass
                return False
        return True

    async def _execute_clone_handshake(self, owner_id: int, source_pid: str, recipient_id: int, desired_name: str) -> Tuple[bool, str]:
        def _sync_clone():
            owner_id_str = str(owner_id)
            recip_id_str = str(recipient_id)

            src_dir = os.path.join(self.cog.USERS_DIR, owner_id_str, "profiles", source_pid)

            if not os.path.exists(src_dir):
                return False, "Source profile data no longer exists."

            index = self._get_user_index(recipient_id)
            if len(index.get("personal", [])) >= defaultConfig.LIMIT_PROFILES:
                return False, "You have reached your personal profile limit."

            new_pid = f"A{uuid.uuid4().hex[:15].upper()}"
            recip_dir = os.path.join(self.cog.USERS_DIR, recip_id_str, "profiles", new_pid)

            try:
                src_data = self._get_profile_by_pid(owner_id, source_pid) or {}
                config_data = src_data.get("config", {}).copy()
                config_data["profile_id"] = new_pid
                config_data["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

                new_data = {
                    "name": desired_name,
                    "config": config_data,
                    "prompts": src_data.get("prompts", {}).copy(),
                    "child_bot": None
                }
                self._save_profile_by_pid(recipient_id, new_pid, new_data)

                disp = config_data.get("custom_display_name")
                ava = config_data.get("custom_avatar_url")
                if disp or ava:
                    self.cog.user_appearances.setdefault(recip_id_str, {})[desired_name] = {
                        "custom_display_name": disp,
                        "custom_avatar_url": ava
                    }

                if not isinstance(index.get("personal"), dict):
                    legacy_personal = index.get("personal", [])
                    index["personal"] = {}
                    if isinstance(legacy_personal, list):
                        for p_name in legacy_personal:
                            index["personal"][p_name] = p_name

                index["personal"][desired_name] = new_pid
                self._save_user_index(recipient_id, index)
                return True, f"✅ Profile cloned successfully as '{desired_name}'!"

            except Exception as e:
                print(f"Error executing clone handshake: {e}")
                traceback.print_exc()
                shutil.rmtree(recip_dir, ignore_errors=True)
                return False, f"An unexpected error occurred during cloning: {e}"

        return await asyncio.to_thread(_sync_clone)


    async def _convert_copy_profile(self, user_id: int, source_name: str, target_name: str, to_system: bool) -> Tuple[bool, str]:
        owner_id = int(defaultConfig.DISCORD_OWNER_ID)
        if user_id != owner_id:
            return False, "Only the Bot Owner can perform system profile conversions."

        is_valid, err_msg = self._is_valid_profile_name(target_name)
        if not is_valid:
            return False, err_msg

        def _sync_convert():
            owner_index = self._get_user_index(owner_id)

            if to_system:
                if source_name not in owner_index.get("personal", {}):
                    return False, f"Source personal profile '{source_name}' not found."
                if target_name in owner_index.get("system", {}):
                    return False, f"A system profile named '{target_name}' already exists."
                source_pid = owner_index["personal"][source_name]
                target_pid = f"X{uuid.uuid4().hex[:15].upper()}"
                target_category = "system"
            else:
                if source_name not in owner_index.get("system", {}):
                    return False, f"Source system profile '{source_name}' not found."
                if target_name in owner_index.get("personal", {}) or target_name in owner_index.get("borrowed", {}):
                    return False, f"A personal profile named '{target_name}' already exists."
                source_pid = owner_index["system"][source_name]
                target_pid = f"A{uuid.uuid4().hex[:15].upper()}"
                target_category = "personal"

            source_dir = os.path.join(self.cog.USERS_DIR, str(owner_id), "profiles", source_pid)
            target_dir = os.path.join(self.cog.USERS_DIR, str(owner_id), "profiles", target_pid)

            if not os.path.exists(source_dir):
                return False, "Source profile directory does not exist on disk."

            try:
                os.makedirs(target_dir, exist_ok=True)
                src_data = self._get_profile_by_pid(owner_id, source_pid) or {}
                config_data = src_data.get("config", {}).copy()
                config_data["profile_id"] = target_pid
                config_data["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

                target_data = {
                    "name": target_name,
                    "config": config_data,
                    "prompts": src_data.get("prompts", {}).copy(),
                    "child_bot": None
                }
                self._save_profile_by_pid(owner_id, target_pid, target_data)

                for item in ["ltm.json.gz", "training.json.gz"]:
                    src_file = os.path.join(source_dir, item)
                    if os.path.exists(src_file):
                        shutil.copy2(src_file, os.path.join(target_dir, item))

                if target_category not in owner_index or not isinstance(owner_index[target_category], dict):
                    owner_index[target_category] = {}

                owner_index[target_category][target_name] = target_pid
                self._save_user_index(owner_id, owner_index)

                return True, f"Successfully copied '{source_name}' to {target_category.title()} Profile '{target_name}'."
            except Exception as e:
                shutil.rmtree(target_dir, ignore_errors=True)
                return False, f"An unexpected error occurred during copy: {e}"

        return await asyncio.to_thread(_sync_convert)
    
