import os
import gzip
import threading
import zstandard as zstd
import pathlib
import time
import datetime
import shutil
import asyncio
from collections import OrderedDict
from discord.ext import tasks
from typing import Any, Dict, Optional
from cryptography.exceptions import InvalidTag
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
import base64
import orjson as json

from ..utils.constants import SERVERS_DIR, CLEANUP_STATE_FILE

# The member cache may legitimately shrink between runs -- people do leave servers --
# so the guard has to allow a real decline while refusing a collapse. A run that sees
# less than this fraction of the highest count ever recorded is treated as a cache
# problem, not as an exodus.
_CLEANUP_MEMBER_FLOOR_RATIO = 0.5

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
# --- Blob encryption -------------------------------------------------------------
#
# Files on disk are AES-256-GCM. Fernet is a *token* format: it base64s its output so
# the result can be a URL-safe string, and for a binary file that is 33% wasted bytes
# plus -- measured on a 9 KB payload -- more time than the AES and the HMAC together
# (17 us of 38, because urlsafe_b64encode is an encode pass and then a translate()
# pass). It also runs AES-CBC, which is serial by construction, followed by a separate
# HMAC pass; GCM authenticates in the same pass and parallelises. End to end, including
# the disk:
#
#     17 KB profile    read 1.8x   write 1.3x   file 25% smaller
#     250-turn log     read 2.1x   write 1.5x   file 25% smaller
#
# Both formats are read; only the new one is written, so a file converts the next time
# it is saved and there is no migration pass and no flag day. Telling them apart is
# exact, not heuristic: a Fernet token is base64 of a leading 0x80 version byte, so it
# always begins 'gAAAAA', and BLOB_MAGIC is not something base64 can emit.
BLOB_MAGIC = b"MAI1"

# GCM's one hard rule: a nonce must never repeat under a given key. 12 random bytes is
# what the mode wants, and os.urandom is the whole story at this scale -- the birthday
# bound on a 96-bit random nonce is ~2**32 writes per key, which a bot writing once a
# second reaches in 136 years. Nothing here may switch to a counter without also
# solving where the counter is persisted across restarts.
_NONCE_BYTES = 12


class MasterCipher:
    """The bot's key material, in both of the forms it gets used in.

    `cog.fernet` holds one of these rather than a bare Fernet. Every caller that
    reaches for `.encrypt` / `.decrypt` directly is a *text* site -- an API key or a
    bot token stored as a string inside JSON, an export payload that travels between
    installs -- where base64 is exactly what is wanted, so those keep Fernet
    behaviour byte for byte. `aead` is the binary-file path and nothing else uses it.

    Both come from the one ENCRYPTION_KEY, so no deployment changes: HKDF gives the
    AEAD its own independent key from the same 32 bytes rather than borrowing one of
    Fernet's two halves for a second purpose.
    """

    __slots__ = ("fernet", "aead")

    def __init__(self, key):
        if isinstance(key, str):
            key = key.encode()
        self.fernet = Fernet(key)
        self.aead = AESGCM(HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"mimicai:blob:aes256gcm:v1",
        ).derive(base64.urlsafe_b64decode(key)))

    def encrypt(self, data: bytes) -> bytes:
        return self.fernet.encrypt(data)

    def decrypt(self, token: bytes) -> bytes:
        return self.fernet.decrypt(token)


def seal_blob(payload: bytes, cipher) -> bytes:
    """Encrypt one already-compressed blob for disk.

    A bare Fernet (tests, and any construction site not yet passing a MasterCipher)
    is still correct here -- it just writes the old format, which every reader
    handles. That fallback is deliberate: an unbound key can cost speed, never data.
    """
    aead = getattr(cipher, "aead", None)
    if aead is None:
        return cipher.encrypt(payload)
    nonce = os.urandom(_NONCE_BYTES)
    return BLOB_MAGIC + nonce + aead.encrypt(nonce, payload, None)


def unseal_blob(blob: bytes, cipher) -> bytes:
    """Decrypt one blob, in whichever of the two formats it was written.

    Raises InvalidToken for a failed GCM tag as well as a failed Fernet HMAC. The
    translation is what lets every existing `except InvalidToken` handler keep
    working -- a corrupt file has to read as None the same way it always did, and
    InvalidTag reaching those call sites would surface as an unhandled exception.
    """
    if blob.startswith(BLOB_MAGIC):
        aead = getattr(cipher, "aead", None)
        if aead is None:
            raise InvalidToken("AES-GCM blob, but the cipher has no AEAD key bound")
        body = memoryview(blob)[len(BLOB_MAGIC):]
        try:
            return aead.decrypt(bytes(body[:_NONCE_BYTES]), bytes(body[_NONCE_BYTES:]), None)
        except InvalidTag as e:
            raise InvalidToken("AES-GCM authentication failed") from e
    return cipher.decrypt(blob)


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


# --- Decrypted shard cache -----------------------------------------------------
#
# Reading a profile shard is 43 us for a 17 KB profile (79 us before the move off
# Fernet), and only 10 us of that is parsing. _get_profile_config has no cache of its
# own and a single turn reaches it several times, so the same file was decrypted from
# scratch over and over. Cheaper crypto shortens that read; it does not remove it.
#
# What is cached is the *plaintext bytes*, never the parsed object. Every caller
# still parses its own dict, which is the property that makes this safe to add:
# profile configs are handed out by reference and mutated in place (see the
# profile_id reconciliation in _get_profile_config), so caching the object would
# alias the cache into whatever the caller did to it next.
#
# The stat stamp is the entire invalidation story, and it is exact rather than
# probabilistic: write_json_gzip builds a temp file and os.replace()s it into
# position, so a changed file has a new inode, not merely a new mtime.
#
# Budgeted in bytes rather than entries because callers opt in per path -- this is
# for the small shards a turn re-reads, not for session logs.
#
# What must NOT opt in: one-shot passes over every profile on disk. The daily orphan
# sweep, the boot share scan and the content-rating reset each touch every shard once
# and never look again, so caching them would evict the working set the turn path
# depends on in order to hold data nobody will read again. Opting in is for a path
# that re-reads the same file; everything else is scan pollution.
#
# Size is handled separately, by the per-entry ceiling below, so no caller has to
# guess how big its shard will turn out to be.
_SHARD_CACHE_MAX_BYTES = 8 * 1024 * 1024

# No single file may take more than an eighth of the budget. Without this, one large
# shard is allowed to be the entire cache and evicts everything the turn path wants:
# an LTM shard at LIMIT_LTM is 5000 entries carrying an embedding each, which is most
# of the budget on its own. With it, the common small shard is cached and the outlier
# is simply read from disk every time, which is what it would have done anyway. This
# is why opting a path in is a question about *frequency* only -- size answers itself.
_SHARD_CACHE_MAX_ENTRY_BYTES = _SHARD_CACHE_MAX_BYTES // 8
_shard_cache: "OrderedDict[str, tuple]" = OrderedDict()
_shard_cache_bytes = 0
_shard_cache_lock = threading.Lock()


def _shard_stamp(file_path: str):
    """(mtime_ns, size, inode), or None if the file is not there."""
    try:
        st = os.stat(file_path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def _shard_cache_store(file_path: str, stamp, plaintext: bytes):
    global _shard_cache_bytes
    size = len(plaintext)
    if size > _SHARD_CACHE_MAX_ENTRY_BYTES:
        return
    with _shard_cache_lock:
        previous = _shard_cache.pop(file_path, None)
        if previous is not None:
            _shard_cache_bytes -= len(previous[1])
        _shard_cache[file_path] = (stamp, plaintext)
        _shard_cache_bytes += size
        while _shard_cache_bytes > _SHARD_CACHE_MAX_BYTES and _shard_cache:
            _, evicted = _shard_cache.popitem(last=False)
            _shard_cache_bytes -= len(evicted[1])


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
                file_bytes = unseal_blob(file_bytes, fernet)

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
    def read_json_gzip_cached(file_path: str, fernet: Optional[Fernet] = None, encrypted: bool = True) -> Optional[Any]:
        """read_json_gzip with the decrypted plaintext cached per path.

        Opt-in: see the shard cache notes above for what belongs here and what does
        not. Returns a freshly parsed object every call, never a shared one.
        """
        stamp = _shard_stamp(file_path)
        if stamp is None:
            return None

        with _shard_cache_lock:
            entry = _shard_cache.get(file_path)
            if entry is not None and entry[0] == stamp:
                _shard_cache.move_to_end(file_path)
                plaintext = entry[1]
            else:
                plaintext = None

        if plaintext is None:
            try:
                with open(file_path, 'rb') as f:
                    file_bytes = f.read()

                if encrypted and fernet:
                    file_bytes = unseal_blob(file_bytes, fernet)

                try:
                    plaintext = _get_decompressor().decompress(file_bytes)
                except zstd.ZstdError:
                    plaintext = gzip.decompress(file_bytes)
            except (IOError, OSError, gzip.BadGzipFile, InvalidToken, zstd.ZstdError) as e:
                print(f"IOManager Read Error ({file_path}): {e}")
                return None

            # Stamped from before the read on purpose. A write landing mid-read leaves
            # this entry carrying new bytes under the old stamp, so the next call stats
            # a different inode and misses -- the failure mode is one wasted read, and
            # never a stale hit.
            _shard_cache_store(file_path, stamp, plaintext)

        try:
            return json.loads(plaintext)
        except json.JSONDecodeError as e:
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
                bytes_to_write = seal_blob(compressed_bytes, fernet)

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
    """Owns key-derived encryption, atomic .json/.json.gz persistence primitives, generic entity
    shard IO, API key persistence, and legacy filesystem migration utilities.

    Holds a back-reference to the parent cog for shared instance caches and cross-manager lookups,
    per the transitional Dependency Injection pattern in CLAUDE.md.
    """

    def __init__(self, cog=None, fernet: Optional[Fernet] = None):
        self.cog = cog
        self.fernet = fernet if fernet is not None else (cog.fernet if cog is not None else None)

    def _encrypt_data(self, plaintext: str) -> str:
        # Value-level encryption is deprecated to prevent CPU overhead.
        # Files are already encrypted natively at the shard-level.
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

    def _get_shard_path(self, shard_type: str, entity_id: str, sub_key: Optional[str] = None,
                        strict: bool = False) -> Optional[str]:
        """Where a per-profile side shard lives, resolved through the owner's index.

        `strict` picks which resolver answers the name. The soft one substitutes the
        name itself on a miss, which is survivable for a read -- the file will not be
        there and the caller gets None -- but not for a write: write_json_gzip creates
        the directory, so a stale name mints `profiles/<Name>/ltm.json.gz`, a folder
        with no profile.json.gz in it. Nothing can ever find that again. Rebuilds skip
        directories without a shard, the consistency check does not count them, and the
        orphan sweeps require one, so it sits there holding a user's memories forever.

        Returns None under `strict` when the name resolves to no PID, and the write
        callers below turn that into a no-op rather than a phantom directory.
        """
        if shard_type in ["ltm", "training"]:
            pm = self.cog.profile_manager
            pid = (pm._get_pid_from_name(int(entity_id), sub_key) if strict
                   else pm._get_pid_from_name_any(int(entity_id), sub_key))
            if not pid:
                return None
            return os.path.join(self.cog.USERS_DIR, str(entity_id), "profiles", pid, f"{shard_type}.json.gz")
        elif shard_type == "profile_shares":
            return os.path.join(self.cog.USERS_DIR, str(entity_id), "shares.json.gz")
        raise ValueError(f"Unknown shard type: {shard_type}")

    def _load_shard(self, shard_type: str, entity_id: str, sub_key: Optional[str] = None) -> Optional[Any]:
        # Cached: an LTM shard is read in full on every retrieval turn -- the vector
        # cache above it holds the built matrix, not the file -- and a training shard
        # on every generation that uses examples. Both re-read the same path over and
        # over, which is the whole test for opting in; the per-entry ceiling decides
        # what actually gets kept, so a huge LTM shard falls back to a plain read.
        path = self._get_shard_path(shard_type, entity_id, sub_key)
        return IOManager.read_json_gzip_cached(path, self.fernet) if path else None

    def _save_shard(self, shard_type: str, entity_id: str, data: Any, sub_key: Optional[str] = None):
        path = self._get_shard_path(shard_type, entity_id, sub_key, strict=True)
        if not path:
            print(f"Shard save skipped: '{sub_key}' resolves to no profile for {entity_id} ({shard_type}).")
            return
        IOManager.write_json_gzip(data, path, self.fernet)

    def _delete_shard(self, shard_type: str, entity_id: str, sub_key: Optional[str] = None):
        path = self._get_shard_path(shard_type, entity_id, sub_key)
        if path:
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

    def _get_user_keys_data(self, user_id: int) -> Dict[str, Any]:
        path = os.path.join(self.cog.USERS_DIR, str(user_id), "keys.json.gz")
        if not os.path.exists(path):
            return {"slots": {}, "personal_assignments": {}}

        # Cached: every key resolution and every /settings screen lands here, several
        # times per interaction across the nine call sites, for a file measured in
        # hundreds of bytes.
        data = IOManager.read_json_gzip_cached(path, self.fernet, encrypted=True)

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

        # Every key add, reassignment and deletion lands here, so this is where the
        # user index's has_personal_key flag is kept true -- and, with it, the stat
        # stamp _index_is_consistent verifies the flag against. Without the stamp
        # being refreshed alongside the file, the hourly self-repair would see a
        # changed key file, assume the index was stale and rebuild it from a full
        # directory scan. Deleting a key never updated the flag at all before this.
        try:
            pm = self.cog.profile_manager
            index = pm._get_user_index(user_id)
            pm._refresh_key_flag(str(user_id), index)
            pm._save_user_index(user_id, index)
        except Exception as e:
            print(f"Could not refresh key flag for user {user_id}: {e}")

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

    def _embedding_api_key(self, guild_id: Optional[int], owner_id: Optional[int] = None) -> Optional[str]:
        """The key an embedding should be billed to: the guild's, else the owner's own.

        `_get_api_key_for_guild` alone was the whole resolver, which made every
        embedding fail wherever there is no guild -- and `/profile` is not
        `guild_only`, so managing a profile's LTM or training examples from a DM
        passed guild_id=None, looked up a server index named "None", found nothing
        and reported it as "failed to generate embedding". In a server it failed the
        same way whenever no key was assigned there, or the key's donor had left.

        `owner_id` is opt-in, and only the profile-management paths pass it: turn-time
        retrieval stays guild-billed, because a server pays for its own conversations.
        There is deliberately no instance-owner fallback here -- unlike
        `_classifier_api_key`, this runs because a user asked for it, so "add a key"
        is an answerable error rather than a silent bill to whoever hosts the bot.
        """
        if guild_id:
            key = self._get_api_key_for_guild(guild_id)
            if key:
                return key
        if owner_id:
            return self._get_api_key_for_user(owner_id)
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

        # An empty cache is the obvious failure and the check above catches it. The
        # dangerous one is a cache that is merely *short* -- chunking still in flight,
        # a partial reconnect, or member caching turned down -- because then this set
        # is non-empty, every check below passes, and every user who happens to be
        # missing from it has their entire directory deleted: profiles, LTM, training
        # and keys. Nothing above can tell that apart from a genuine departure.
        #
        # So compare against the largest count ever recorded rather than against zero.
        # A collapse skips the run and leaves the high-water mark intact, so a bot that
        # comes up under-chunked never cleans until it is properly populated again.
        member_count = len(all_bot_member_ids)
        cleanup_state = IOManager.read_json(CLEANUP_STATE_FILE) or {}
        high_water = int(cleanup_state.get("member_high_water", 0))
        if high_water and member_count < high_water * _CLEANUP_MEMBER_FLOOR_RATIO:
            print(
                f"[Cleanup] Aborted: member cache holds {member_count} across "
                f"{len(self.cog.bot.guilds)} guild(s), against a high-water mark of "
                f"{high_water}. Too steep a drop to be departures -- refusing to treat "
                "the difference as departed users. Check that guild chunking completed."
            )
            return

        if member_count > high_water:
            try:
                IOManager.write_json({"member_high_water": member_count}, CLEANUP_STATE_FILE)
            except Exception as e:
                print(f"[Cleanup] Could not record member high-water mark: {e}")

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
                    # Their index.json went with the tree, so the reconcile hooked
                    # into _save_user_index can never fire for them again.
                    self.cog.profile_manager._borrow_index_drop_user(int(user_id_str))
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

            # Borrowed profile cleanup.
            #
            # By source PID, not source name. Keyed by name, an owner renaming a
            # profile deleted every borrow of it on the next run -- the borrow stores
            # the name it was taken under, and that snapshot goes stale the moment the
            # owner edits it. The PID never moves.
            #
            # "system" is checked alongside "personal" because System profiles are
            # shareable (see _get_name_from_pid), so a borrow of one was validated
            # against a map it could never appear in and was deleted daily.
            for borrowed_name in list(borrowed_profiles):
                b_config = self.cog.profile_manager._get_profile_config(uid, borrowed_name, True)
                if not b_config:
                    continue
                owner_id = b_config.get("original_owner_id")
                source_pid = b_config.get("original_pid") or b_config.get("original_profile_id")
                if not (owner_id and source_pid):
                    continue

                owner_index = self.cog.profile_manager._get_user_index(int(owner_id))
                live_pids = set()
                for category in ("personal", "system"):
                    mapping = owner_index.get(category)
                    if isinstance(mapping, dict):
                        live_pids.update(mapping.values())

                if source_pid not in live_pids:
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
            
            # Every class, not just "personal". A System profile's X-prefixed PID is
            # never in the personal map, so the owner's System profiles all looked
            # orphaned here and had their child bot silently unconfigured on every
            # daily run. Borrowed is included for the same reason.
            valid_pids = set()
            for category in ("personal", "borrowed", "system"):
                entry = index.get(category, {})
                if isinstance(entry, dict):
                    valid_pids.update(entry.values())
                else:
                    valid_pids.update(entry)
            
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
                        hot_file = type_dir / "session_log.hot.json.gz"
                        if not log_file.exists():
                            continue

                        # Both halves, and the newer of the two mtimes.
                        #
                        # A live session appends only to the tail and reseals into the
                        # cold segment once every SESSION_HOT_TAIL_MAX turns, so the
                        # cold file's mtime says nothing about whether the session is
                        # in use -- a quiet channel could pass thirty days between
                        # reseals and have its sealed history deleted underneath it.
                        #
                        # And deleting the cold file alone stranded the tail: the
                        # loader returns early when the cold path is missing, so it was
                        # never read and never deleted, and its presence kept the whole
                        # channel directory alive past the emptiness check above.
                        last_touched = log_file.stat().st_mtime
                        if hot_file.exists():
                            last_touched = max(last_touched, hot_file.stat().st_mtime)

                        if last_touched < thirty_days_ago:
                            _delete_file_shard(str(log_file))
                            if hot_file.exists():
                                _delete_file_shard(str(hot_file))
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
            # A key sitting unassigned in a slot isn't usable -- generation only ever
            # reads personal_assignments (or a server's assigned_keys), so that's what
            # actually has to be non-empty for "you have a way to use profiles" to hold.
            if keys_data.get("personal_assignments"):
                return True

            if guild_id:
                idx = self.cog.server_manager._get_server_index(str(guild_id))
                if idx.get("assigned_keys"):
                    return True

            return False

        return await asyncio.to_thread(_sync_check)

