import os
import gzip
import threading
import zstandard as zstd
import pathlib
import time
import datetime
import shutil
import asyncio
from discord.ext import tasks
from typing import Any, Dict, Optional
from cryptography.fernet import Fernet, InvalidToken
import orjson as json

from ..utils.constants import SERVERS_DIR

# zstandard's ZstdCompressor / ZstdDecompressor are NOT thread-safe: each owns a
# native ZSTD_CCtx / ZSTD_DCtx, and the C backend releases the GIL while working on
# it. Two threads sharing one instance therefore run libzstd on the same context
# concurrently and corrupt it.
#
# These used to be module-level singletons, and every IOManager read/write reaches
# them through asyncio.to_thread — so any two concurrent shard operations raced.
# In production that showed up two ways: bursts of "IOManager Read Error ...
# ZstdError" on valid files, and a hard SIGSEGV inside
# backend_c.cpython-311-x86_64-linux-gnu.so when the trampled context dereferenced
# an unmapped pointer.
#
# Thread-local rather than per-call: one context per worker thread, built once and
# reused, so the hot path keeps its allocation-free property on the e2-micro while
# no context is ever touched by two threads.
_ZSTD_LOCAL = threading.local()


def _get_compressor() -> "zstd.ZstdCompressor":
    compressor = getattr(_ZSTD_LOCAL, "compressor", None)
    if compressor is None:
        compressor = _ZSTD_LOCAL.compressor = zstd.ZstdCompressor(level=1)
    return compressor


def _get_decompressor() -> "zstd.ZstdDecompressor":
    decompressor = getattr(_ZSTD_LOCAL, "decompressor", None)
    if decompressor is None:
        decompressor = _ZSTD_LOCAL.decompressor = zstd.ZstdDecompressor()
    return decompressor


def _delete_file_shard(file_path: str):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except OSError as e:
        print(f"Error deleting file shard {file_path}: {e}")

class IOManager:
    """Centralised I/O Helper Block for MimicAI Data Ops."""

    @staticmethod
    def read_json(file_path: str) -> Optional[Any]:
        if not os.path.exists(file_path): return None
        try:
            with open(file_path, 'rb') as f:
                return json.loads(f.read())
        except Exception as e:
            print(f"IOManager read_json Error ({file_path}): {e}")
            return None

    @staticmethod
    def write_json(data: Any, file_path: str):
        temp = file_path + ".tmp"
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(temp, 'wb') as f:
                f.write(json.dumps(data))
            os.replace(temp, file_path)
        except Exception as e:
            print(f"IOManager write_json Error ({file_path}): {e}")
            if os.path.exists(temp): os.remove(temp)
            raise

    @staticmethod
    def read_json_gzip(file_path: str, fernet: Optional[Fernet] = None, encrypted: bool = True) -> Optional[Any]:
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, 'rb') as f:
                file_bytes = f.read()

            if encrypted and fernet:
                file_bytes = fernet.decrypt(file_bytes)

            try:
                decompressed_bytes = _get_decompressor().decompress(file_bytes)
            except zstd.ZstdError:
                # Automatic fallback for legacy gzip files on disk
                decompressed_bytes = gzip.decompress(file_bytes)

            return json.loads(decompressed_bytes)
        except (IOError, json.JSONDecodeError, gzip.BadGzipFile, InvalidToken, zstd.ZstdError) as e:
            print(f"IOManager Read Error ({file_path}): {e}")
            return None

    @staticmethod
    def write_json_gzip(data: Any, file_path: str, fernet: Optional[Fernet] = None, encrypted: bool = True):
        temp_file_path = file_path + ".tmp"
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            json_bytes = json.dumps(data)
            compressed_bytes = _get_compressor().compress(json_bytes)

            bytes_to_write = compressed_bytes
            if encrypted and fernet:
                bytes_to_write = fernet.encrypt(compressed_bytes)

            with open(temp_file_path, 'wb') as f:
                f.write(bytes_to_write)
            os.replace(temp_file_path, file_path)
        except Exception as e:
            print(f"IOManager Write Error ({file_path}): {e}")
            if os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except OSError: pass
            raise

class StorageManager:
    """Owns Fernet-keyed encryption, atomic .json/.json.gz persistence primitives, generic entity
    shard IO, API key persistence, and legacy filesystem migration utilities.

    Holds a back-reference to the parent cog for shared instance caches and cross-manager lookups,
    per the transitional Dependency Injection pattern in CLAUDE.md.
    """

    def __init__(self, cog=None, fernet: Optional[Fernet] = None):
        self.cog = cog
        self.fernet = fernet if fernet is not None else (cog.fernet if cog is not None else None)

    def _encrypt_data(self, plaintext: str) -> str:
        # Value-level encryption is deprecated to prevent CPU overhead.
        # Files are already Fernet-encrypted natively at the shard-level.
        return plaintext

    def _decrypt_data(self, encrypted_text: str) -> str:
        if not self.fernet or not encrypted_text:
            return encrypted_text
        # Fast prefix guard: Fernet tokens always begin with 'gAAAAA' in Base64
        if not isinstance(encrypted_text, str) or not encrypted_text.startswith("gAAAAA"):
            return encrypted_text
        try:
            return self.fernet.decrypt(encrypted_text.encode()).decode()
        except Exception:
            return encrypted_text

    def _atomic_json_save_gzip(self, data: Any, file_path: str, encrypted: bool = True):
        IOManager.write_json_gzip(data, file_path, self.fernet, encrypted)

    def _load_json_gzip(self, file_path: str, encrypted: bool = True) -> Optional[Any]:
        return IOManager.read_json_gzip(file_path, self.fernet, encrypted)

    def _get_shard_path(self, shard_type: str, entity_id: str, sub_key: Optional[str] = None) -> str:
        if shard_type in ["ltm", "training"]:
            pid = self.cog.profile_manager._get_pid_from_name_any(int(entity_id), sub_key)
            return os.path.join(self.cog.USERS_DIR, str(entity_id), "profiles", pid, f"{shard_type}.json.gz")
        elif shard_type == "personal_keys":
            return os.path.join(self.cog.USERS_DIR, str(entity_id), "keys.json.gz")
        elif shard_type == "profile_shares":
            return os.path.join(self.cog.USERS_DIR, str(entity_id), "shares.json.gz")
        elif shard_type == "server_keys":
            return os.path.join(self.cog.SERVERS_DIR, str(entity_id), "api_keys.json.gz")
        raise ValueError(f"Unknown shard type: {shard_type}")

    def _load_shard(self, shard_type: str, entity_id: str, sub_key: Optional[str] = None) -> Optional[Any]:
        path = self._get_shard_path(shard_type, entity_id, sub_key)
        return IOManager.read_json_gzip(path, self.fernet)

    def _save_shard(self, shard_type: str, entity_id: str, data: Any, sub_key: Optional[str] = None):
        path = self._get_shard_path(shard_type, entity_id, sub_key)
        IOManager.write_json_gzip(data, path, self.fernet)

    def _delete_shard(self, shard_type: str, entity_id: str, sub_key: Optional[str] = None):
        path = self._get_shard_path(shard_type, entity_id, sub_key)
        _delete_file_shard(path)

    def _purge_legacy_default_profile(self):
        users_path = pathlib.Path(self.cog.USERS_DIR)
        if not users_path.exists(): return

        for user_dir in users_path.iterdir():
            if not user_dir.is_dir() or not user_dir.name.isdigit(): continue

            profiles_dir = user_dir / "profiles"
            if profiles_dir.exists():
                for p_dir in list(profiles_dir.iterdir()):
                    if p_dir.is_dir() and p_dir.name.lower() == "mimic":
                        shutil.rmtree(str(p_dir), ignore_errors=True)

            index_path = user_dir / "index.json"
            if index_path.exists():
                index = IOManager.read_json(str(index_path))
                if index:
                    changed = False
                    for key in ["personal", "borrowed"]:
                        mapping = index.get(key, {})
                        to_remove = []
                        if isinstance(mapping, dict):
                            for k, v in mapping.items():
                                if k.lower() == "mimic":
                                    to_remove.append(k)
                            for k in to_remove:
                                pid = mapping.pop(k)
                                changed = True
                                shutil.rmtree(str(profiles_dir / pid), ignore_errors=True)
                        elif isinstance(mapping, list):
                            new_mapping = []
                            for k in mapping:
                                if k.lower() == "mimic":
                                    changed = True
                                else:
                                    new_mapping.append(k)
                            index[key] = new_mapping

                    if changed:
                        IOManager.write_json(index, str(index_path))

            # --- Cleanup Empty Folders ---
            has_profiles = False
            if profiles_dir.exists():
                has_profiles = any(p.is_dir() for p in profiles_dir.iterdir())

            if not has_profiles:
                # If they have no valid profiles, keys, or shares, nuke the entire user ID folder
                has_keys = (user_dir / "keys.json.gz").exists()
                has_shares = (user_dir / "shares.json.gz").exists()
                if not has_keys and not has_shares:
                    shutil.rmtree(str(user_dir), ignore_errors=True)

    def _load_server_api_keys(self):
        self.cog.server_api_keys = {}
        servers_dir = self.cog.SERVERS_DIR
        if not os.path.isdir(servers_dir):
            return

        for server_id_str in os.listdir(servers_dir):
            server_path = os.path.join(servers_dir, server_id_str)
            if os.path.isdir(server_path) and server_id_str.isdigit():
                api_keys_file = os.path.join(server_path, "api_keys.json.gz")
                if os.path.exists(api_keys_file):
                    server_keys_data = self._load_json_gzip(api_keys_file)
                    if server_keys_data and server_keys_data.get("primary"):
                        # Primary key data might now contain 'openrouter_key'
                        self.cog.server_api_keys[server_id_str] = server_keys_data.get("primary")

    def _save_server_api_key_shard(self, server_id_str: str, primary_key_data: Optional[Dict]):
        self._save_shard("server_keys", server_id_str, {
            "primary": primary_key_data
        })

    def _load_personal_api_keys(self):
        self.cog.personal_api_keys = {}
        if not os.path.isdir(self.cog.USERS_DIR):
            return
        for user_id_str in os.listdir(self.cog.USERS_DIR):
            if not user_id_str.isdigit(): continue
            file_path = os.path.join(self.cog.USERS_DIR, user_id_str, "keys.json.gz")
            if os.path.exists(file_path):
                data = IOManager.read_json_gzip(file_path, self.fernet)
                if data and isinstance(data, dict) and "key" in data:
                    self.cog.personal_api_keys[user_id_str] = data["key"]

    def _save_personal_api_key_shard(self, user_id_str: str, encrypted_key: Optional[str]):
        if not encrypted_key:
            self._delete_shard("personal_keys", user_id_str)
        else:
            self._save_shard("personal_keys", user_id_str, {"key": encrypted_key})

    def _get_user_keys_data(self, user_id: int) -> Dict[str, Any]:
        path = os.path.join(self.cog.USERS_DIR, str(user_id), "keys.json.gz")
        if not os.path.exists(path):
            return {"slots": {}, "personal_assignments": {}}

        data = IOManager.read_json_gzip(path, self.fernet, encrypted=True)

        # Auto-purge legacy/corrupted files (e.g., the old b'gA' format)
        if not data or "slots" not in data:
            try:
                os.remove(path)
            except OSError:
                pass
            return {"slots": {}, "personal_assignments": {}}

        return data

    def _save_user_keys_data(self, user_id: int, data: Dict[str, Any]):
        path = os.path.join(self.cog.USERS_DIR, str(user_id), "keys.json.gz")
        IOManager.write_json_gzip(data, path, self.fernet, encrypted=True)

    def _get_api_key_for_guild(self, guild_id: int, provider: str = "gemini") -> Optional[str]:
        if not self.fernet: return None
        guild_id_str = str(guild_id)
        now = time.time()

        cache_key = (guild_id, provider)
        pointer = self.cog.server_key_pointers.get(cache_key)

        if not pointer:
            server_index = self.cog.server_manager._get_server_index(guild_id_str)
            assigned = server_index.get("assigned_keys", {}).get(provider)
            if assigned:
                pointer = (assigned["user_id"], assigned["slot"])
                self.cog.server_key_pointers[cache_key] = pointer

        if pointer:
            user_id, slot_id = pointer

            guild = self.cog.bot.get_guild(guild_id)
            if not guild or not guild.get_member(user_id):
                self.cog.server_key_pointers.pop(cache_key, None)
                return None

            decrypted_key = self.cog.decrypted_key_cache.get((user_id, slot_id))
            if decrypted_key:
                if decrypted_key not in self.cog.api_key_cooldowns or now > self.cog.api_key_cooldowns[decrypted_key]:
                    return decrypted_key

            user_data = self._get_user_keys_data(user_id)
            slot_data = user_data.get("slots", {}).get(slot_id)
            if slot_data and slot_data.get("key"):
                raw_key = slot_data["key"]
                self.cog.decrypted_key_cache[(user_id, slot_id)] = raw_key
                if raw_key not in self.cog.api_key_cooldowns or now > self.cog.api_key_cooldowns[raw_key]:
                    return raw_key

        return None

    def _get_api_key_for_user(self, user_id: int, provider: str = "gemini") -> Optional[str]:
        if not self.fernet: return None
        now = time.time()

        user_data = self._get_user_keys_data(user_id)
        slot_id = user_data.get("personal_assignments", {}).get(provider)

        if slot_id:
            decrypted_key = self.cog.decrypted_key_cache.get((user_id, slot_id))
            if decrypted_key:
                if decrypted_key not in self.cog.api_key_cooldowns or now > self.cog.api_key_cooldowns[decrypted_key]:
                    return decrypted_key

            slot_data = user_data.get("slots", {}).get(slot_id)
            if slot_data and slot_data.get("key"):
                raw_key = slot_data["key"]
                self.cog.decrypted_key_cache[(user_id, slot_id)] = raw_key
                if raw_key not in self.cog.api_key_cooldowns or now > self.cog.api_key_cooldowns[raw_key]:
                    return raw_key

        return None

    async def _perform_data_cleanup(self):
        await asyncio.to_thread(self._sync_perform_data_cleanup)

    def _sync_perform_data_cleanup(self):
        log = ["Starting Automatic Daily Data Cleanup..."]
        bot_guild_ids = {g.id for g in self.cog.bot.guilds}
        all_bot_member_ids = {str(m.id) for g in self.cog.bot.guilds for m in g.members}
        all_bot_channel_ids = {c.id for g in self.cog.bot.guilds for c in g.channels}

        # `all_bot_member_ids` decides four irreversible things below: deleting a
        # user's whole directory (profiles, LTM, training, keys), deleting their
        # session directories, dropping their profile shares, and dropping server
        # key assignments. It is derived from discord.py's member cache, which is
        # only populated if the members intent is on AND guild chunking actually
        # completed. If chunking was disabled, failed, or is still in flight when
        # the daily task fires, this set is empty or badly short -- and an empty
        # set means "nobody is a member of anything", which deletes everything.
        #
        # A cache miss and a genuinely empty server are indistinguishable here, so
        # refuse to run rather than guess. Nothing is lost by skipping a day.
        if self.cog.bot.guilds and not all_bot_member_ids:
            print(
                "[Cleanup] Aborted: the member cache is empty across "
                f"{len(self.cog.bot.guilds)} guild(s). This is a cache problem, not "
                "an empty server -- refusing to treat every user as departed. Check "
                "that the members intent is enabled and guild chunking completed."
            )
            return

        # --- 1. Expired Share Codes ---
        cleaned_codes = 0
        now = time.time()
        for code, data in list(self.cog.share_codes.items()):
            if now > data.get("expires_at", 0):
                del self.cog.share_codes[code]
                cleaned_codes += 1
        if cleaned_codes > 0:
            log.append(f"🧹 Removed {cleaned_codes} expired share codes.")

        # --- 2. Stale/Broken Profile Shares ---
        cleaned_shares = 0
        for recipient_id_str, shares in list(self.cog.profile_shares.items()):
            if recipient_id_str not in all_bot_member_ids:
                cleaned_shares += len(self.cog.profile_shares.pop(recipient_id_str, []))
                self.cog.profile_manager._save_profile_share_shard(recipient_id_str, None)
                continue
            
            original_len = len(shares)
            valid_shares = []
            for share in shares:
                sharer_id_str = str(share.get("sharer_id"))
                profile_name = share.get("profile_name")
                if sharer_id_str in all_bot_member_ids:
                    sharer_index = self.cog.profile_manager._get_user_index(int(sharer_id_str))
                    if profile_name in sharer_index.get("personal", []):
                        valid_shares.append(share)
            
            if len(valid_shares) < original_len:
                self.cog.profile_shares[recipient_id_str] = valid_shares
                cleaned_shares += original_len - len(valid_shares)
                self.cog.profile_manager._save_profile_share_shard(recipient_id_str, valid_shares)
        if cleaned_shares > 0:
            log.append(f"🧹 Removed {cleaned_shares} stale or broken profile share requests.")

        # --- 3. Orphaned Server Pointers ---
        cleaned_pointers = 0
        for g in self.cog.bot.guilds:
            idx = self.cog.server_manager._get_server_index(str(g.id))
            changed = False
            for prov in list(idx.get("assigned_keys", {}).keys()):
                uid = idx["assigned_keys"][prov].get("user_id")
                if uid and str(uid) not in all_bot_member_ids:
                    del idx["assigned_keys"][prov]
                    self.cog.server_key_pointers.pop((g.id, prov), None)
                    changed = True
                    cleaned_pointers += 1
            if changed:
                self.cog.server_manager._save_server_index(str(g.id), idx)
        if cleaned_pointers > 0:
            log.append(f"🧹 Removed {cleaned_pointers} orphaned server key assignments.")

        # --- 4. Orphaned Channel Webhooks ---
        cleaned_webhooks = 0
        for ch_id in list(self.cog.channel_webhooks.keys()):
            if ch_id not in all_bot_channel_ids:
                del self.cog.channel_webhooks[ch_id]
                cleaned_webhooks += 1
        if cleaned_webhooks > 0:
            self.cog.server_manager._save_channel_webhooks()
            log.append(f"🧹 Removed {cleaned_webhooks} orphaned channel webhooks.")

        # --- 5. Orphaned Server-Level Files ---
        cleaned_server_files = 0
        servers_path = pathlib.Path(SERVERS_DIR)
        if servers_path.is_dir():
            for server_dir in list(servers_path.iterdir()):
                try:
                    if server_dir.is_dir() and int(server_dir.name) not in bot_guild_ids:
                        shutil.rmtree(server_dir, ignore_errors=True)
                        cleaned_server_files += 1
                except ValueError:
                    continue
        if cleaned_server_files > 0:
            log.append(f"🧹 Removed {cleaned_server_files} orphaned server-level data directories/files.")

        # --- 6. Full User Data Cleanup (Ghost Directories & Missing Users) ---
        cleaned_users_count = 0
        if os.path.isdir(self.cog.USERS_DIR):
            for user_id_str in os.listdir(self.cog.USERS_DIR):
                if not user_id_str.isdigit(): continue
                
                user_dir = os.path.join(self.cog.USERS_DIR, user_id_str)
                is_missing = user_id_str not in all_bot_member_ids
                
                # Check for ghost directory (no profiles, no keys, no shares)
                is_ghost = False
                try:
                    uid = int(user_id_str)
                    index = self.cog.profile_manager._get_user_index(uid)
                    has_personal = bool(index.get("personal"))
                    has_borrowed = bool(index.get("borrowed"))
                    has_system = bool(index.get("system"))
                    has_keys = os.path.exists(os.path.join(user_dir, "keys.json.gz"))
                    has_shares = os.path.exists(os.path.join(user_dir, "shares.json.gz"))
                    
                    if not (has_personal or has_borrowed or has_system or has_keys or has_shares):
                        is_ghost = True
                except Exception:
                    pass

                if is_missing or is_ghost:
                    shutil.rmtree(user_dir, ignore_errors=True)
                    self.cog.user_appearances.pop(user_id_str, None)
                    self.cog.profile_shares.pop(user_id_str, None)
                    self.cog.user_indices.pop(user_id_str, None)
                    cleaned_users_count += 1
        
        if cleaned_users_count > 0:
            log.append(f"🧹 Removed {cleaned_users_count} ghost user directories or users no longer sharing a server.")

        # --- 7. Detailed Per-User & Per-Server Integrity Check ---
        cleaned_borrows = 0
        cleaned_session_files, cleaned_child_bots = 0, 0
        
        users_path = pathlib.Path(self.cog.USERS_DIR)
        user_dirs = [d.name for d in users_path.iterdir() if d.is_dir() and d.name.isdigit()] if users_path.exists() else []
        
        for user_id_str in user_dirs:
            uid = int(user_id_str)
            index = self.cog.profile_manager._get_user_index(uid)
            if not index: continue

            user_profiles = set(index.get("personal", []))
            borrowed_profiles = set(index.get("borrowed", []))
            all_valid_profiles = user_profiles | borrowed_profiles
            data_changed = False

            # Borrowed profile cleanup
            for borrowed_name in list(borrowed_profiles):
                b_config = self.cog.profile_manager._get_profile_config(uid, borrowed_name, True)
                if b_config:
                    owner_id = b_config.get("original_owner_id")
                    original_name = b_config.get("original_profile_name")
                    if owner_id and original_name:
                        owner_index = self.cog.profile_manager._get_user_index(int(owner_id))
                        if original_name not in owner_index.get("personal", []):
                            if isinstance(index["borrowed"], dict):
                                pid = index["borrowed"].pop(borrowed_name, borrowed_name)
                            else:
                                index["borrowed"].remove(borrowed_name)
                                pid = borrowed_name
                            shutil.rmtree(str(users_path / user_id_str / "profiles" / pid), ignore_errors=True)
                            cleaned_borrows += 1
                            data_changed = True
            
            # Session file cleanup
            global_session_dir = pathlib.Path(self.cog.SESSIONS_GLOBAL_DIR) / user_id_str
            if global_session_dir.is_dir():
                for session_file in global_session_dir.iterdir():
                    if session_file.name.endswith(".json.gz"):
                        profile_name = session_file.name[:-len(".json.gz")]
                        if profile_name not in all_valid_profiles:
                            _delete_file_shard(str(session_file))
                            cleaned_session_files += 1
            
            if data_changed: self.cog.profile_manager._save_user_index(uid, index)

        # Child Bot cleanup
        child_bots_changed = False
        for user_id_str in user_dirs:
            profiles_dir = os.path.join(self.cog.USERS_DIR, user_id_str, "profiles")
            if not os.path.isdir(profiles_dir): continue
            index = self.cog.profile_manager._get_user_index(int(user_id_str))
            
            valid_pids = set()
            personal_entry = index.get("personal", {})
            if isinstance(personal_entry, dict):
                valid_pids.update(personal_entry.values())
            else:
                valid_pids.update(personal_entry)
            
            for pid_folder in os.listdir(profiles_dir):
                profile_file = os.path.join(profiles_dir, pid_folder, "profile.json.gz")
                if os.path.exists(profile_file) and pid_folder not in valid_pids:
                    profile_data = IOManager.read_json_gzip(profile_file, self.fernet)
                    if profile_data and profile_data.get("child_bot"):
                        profile_data["child_bot"] = None
                        IOManager.write_json_gzip(profile_data, profile_file, self.fernet)
                        cleaned_child_bots += 1
                        child_bots_changed = True
                    
        if child_bots_changed: self.cog.child_bot_manager._load_child_bots()

        if cleaned_borrows > 0: log.append(f"🧹 Removed {cleaned_borrows} broken borrowed profiles.")
        if cleaned_session_files > 0: log.append(f"🧹 Removed {cleaned_session_files} orphaned session files for deleted profiles.")
        if cleaned_child_bots > 0: log.append(f"🧹 Removed {cleaned_child_bots} orphaned child bot configurations.")

        # --- 8. Channel & User Session Directory Cleanup ---
        cleaned_channel_dirs, cleaned_user_session_dirs = 0, 0
        servers_path = pathlib.Path(SERVERS_DIR)
        if servers_path.is_dir():
            for server_dir in list(servers_path.iterdir()):
                if not server_dir.is_dir(): continue
                try:
                    server_id_int = int(server_dir.name)
                    guild = self.cog.bot.get_guild(server_id_int)
                    
                    sessions_dir = server_dir / "sessions"
                    if not sessions_dir.is_dir(): continue
                    
                    for channel_dir in list(sessions_dir.iterdir()):
                        if not channel_dir.is_dir(): continue
                        try:
                            channel_id = int(channel_dir.name)
                            # Remove if channel is gone or if directory is empty of actual data
                            is_deleted = guild and channel_id not in {c.id for c in guild.channels}
                            has_files = any(f.is_file() for f in channel_dir.rglob('*') if not f.name.startswith('.'))
                            
                            if is_deleted or not has_files:
                                shutil.rmtree(channel_dir, ignore_errors=True)
                                cleaned_channel_dirs += 1
                                continue
                            
                            # Deep check for orphaned user subdirectories in single-profile sessions
                            single_user_path = channel_dir / "single"
                            if single_user_path.is_dir():
                                current_member_ids = {str(m.id) for m in guild.members} if guild else set()
                                for user_dir in list(single_user_path.iterdir()):
                                    if not user_dir.is_dir(): continue
                                    is_orphaned = guild and user_dir.name not in current_member_ids
                                    is_empty = not any(user_dir.iterdir())
                                    if is_orphaned or is_empty:
                                        shutil.rmtree(user_dir, ignore_errors=True)
                                        cleaned_user_session_dirs += 1
                        except (ValueError, OSError): continue
                    
                    # Remove server dir if empty
                    if not any(server_dir.iterdir()):
                        server_dir.rmdir()
                except (ValueError, OSError): continue
        
        if cleaned_channel_dirs > 0: log.append(f"🧹 Removed {cleaned_channel_dirs} session directories for deleted channels.")
        if cleaned_user_session_dirs > 0: log.append(f"🧹 Removed {cleaned_user_session_dirs} user session directories for users no longer in the server.")

        # --- 9. Final Config File Cleanup ---
        cleaned_channel_settings = 0
        for ch_id in list(self.cog.multi_profile_channels.keys()):
            if ch_id not in all_bot_channel_ids:
                del self.cog.multi_profile_channels[ch_id]
                cleaned_channel_settings += 1
        
        if cleaned_channel_settings > 0:
            self.cog.session_manager._save_multi_profile_sessions()
            log.append(f"🧹 Removed settings for {cleaned_channel_settings} deleted channels from config files.")

        # --- 10. Inactive Session File Cleanup (30-Day TTL) ---
        cleaned_session_files_ttl = 0
        thirty_days_ago = time.time() - (30 * 86400)
        
        # Check Server Sessions
        if servers_path.is_dir():
            for server_dir in list(servers_path.iterdir()):
                if not server_dir.is_dir(): continue
                sessions_dir = server_dir / "sessions"
                if not sessions_dir.is_dir(): continue
                for channel_dir in list(sessions_dir.iterdir()):
                    if not channel_dir.is_dir(): continue
                    for session_type in ["multi", "freewill"]:
                        type_dir = channel_dir / session_type
                        log_file = type_dir / "session_log.json.gz"
                        if log_file.exists() and log_file.stat().st_mtime < thirty_days_ago:
                            _delete_file_shard(str(log_file))
                            cleaned_session_files_ttl += 1
        
        # Check Global Sessions
        if users_path.is_dir():
            for user_dir in list(users_path.iterdir()):
                if not user_dir.is_dir() or not user_dir.name.isdigit(): continue
                profiles_dir = user_dir / "profiles"
                if not profiles_dir.is_dir(): continue
                for pid_dir in list(profiles_dir.iterdir()):
                    if not pid_dir.is_dir(): continue
                    gc_file = pid_dir / "global_chat.json.gz"
                    if gc_file.exists() and gc_file.stat().st_mtime < thirty_days_ago:
                        _delete_file_shard(str(gc_file))
                        cleaned_session_files_ttl += 1

        if cleaned_session_files_ttl > 0:
            log.append(f"🧹 Removed {cleaned_session_files_ttl} inactive session logs (30-day TTL expired).")

        log.append("Cleanup complete.")
        print("\n".join(log).replace("**", ""))

    @tasks.loop(time=datetime.time(hour=17, minute=0, tzinfo=datetime.timezone.utc)) # 17:00 UTC = 3:00 AM AEST
    async def daily_cleanup_task(self):
        if self.cog.has_lock:
            print("Starting daily data cleanup...")
            await self._perform_data_cleanup()
            print("Daily data cleanup finished.")

    async def _has_api_key_access(self, user_id: int, guild_id: Optional[int] = None) -> bool:
        def _sync_check():
            keys_data = self._get_user_keys_data(user_id)
            if keys_data.get("slots"):
                return True

            if guild_id:
                idx = self.cog.server_manager._get_server_index(str(guild_id))
                if idx.get("assigned_keys"):
                    return True

            return False

        return await asyncio.to_thread(_sync_check)

