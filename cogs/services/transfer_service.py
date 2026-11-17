"""Moving a user's data in and out: .mimic export and import, the privacy export, and account
deletion."""

import os
import io
import shutil
import base64
import zlib
import zstandard as zstd
import datetime
import asyncio
import discord
from typing import Dict, List, Set, Any, Optional, Tuple
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from ..utils.constants import (
    USERS_DIR, defaultConfig, CONTENT_RATING_UNRATED, CONTENT_RATING_GENERAL, CONTENT_RATING_ADULT, CONTENT_RATING_REFUSED, PERSONA_TEXT_INPUT_MAX_LENGTH)
from ..utils.helpers import suppress_link_previews, clean_imported_text
from ..utils import fixed_examples, lorebook
from ..managers.storage_manager import IOManager, _get_compressor, _get_decompressor
from ..managers.memory_manager import read_ltm_dir, write_imported_ltm
from ..gui.base_components import text_embed
try:
    import orjson as json
except ImportError:
    import json
from ..managers.profile_manager import _carried_hold, _unchecked_declaration

#: The most a .mimic payload may inflate to. A real export is at most one Discord
#: upload -- 10 MiB, a few times that inflated -- and every inflated byte costs several
#: more once parsed, on a 1 GB box.
_IMPORT_MAX_INFLATED_BYTES = 64 * 1024 * 1024
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_GZIP_MAGIC = b"\x1f\x8b"

#: A borrow's pointer at its source. Written only by a borrow being made, so an import --
#: always a personal profile -- that carries one is carrying a claim on someone else's
#: profile, which the borrow index would read. The retired keys are stripped with them.
_BORROW_POINTER_KEYS = ("original_owner_id", "original_pid", "original_profile_id",
                        "original_profile_name", "pointer")


def _clean_imported_prompts(value: Any) -> Any:
    """A .mimic profile's prompts as a person could have typed them: every string through
    `clean_imported_text` at the editor's cap, and the lorebook and fixed examples
    normalised. A passphrase file is anyone's to write."""
    if isinstance(value, str):
        return clean_imported_text(value, PERSONA_TEXT_INPUT_MAX_LENGTH)
    if isinstance(value, list):
        return [_clean_imported_prompts(v) for v in value]
    if isinstance(value, dict):
        return {k: lorebook.normalise(v) if k == "lorebook"
                else fixed_examples.normalise(v) if k == "fixed_examples"
                else _clean_imported_prompts(v) for k, v in value.items()}
    return value


def _passphrase_fernet(passphrase: str, salt: bytes) -> Fernet:
    """The key an Export for Self-Hosted is sealed with. Seconds of CPU on an e2-micro:
    call it off the loop (it releases the GIL, so a thread is enough)."""
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=480000)
    return Fernet(base64.urlsafe_b64encode(kdf.derive(passphrase.encode('utf-8'))))


def _inflate_export(blob: bytes) -> bytes:
    """A decrypted .mimic payload inflated: zstd, gzip, or neither, as each build wrote it.

    Bounded, because a passphrase file is anyone's to write. A zstd frame states its own
    size and `decompress` allocates that on trust -- 6 KB of zeros declares 200 MB --
    so the declared size is refused before anything is allocated.
    """
    damaged = ValueError("The export is damaged, or inflates past the "
                         f"{_IMPORT_MAX_INFLATED_BYTES // (1024 * 1024)} MB an import may hold.")
    if blob[:4] == _ZSTD_MAGIC:
        try:
            # -1 when the frame does not say; max_output_size then caps it.
            if zstd.frame_content_size(blob) <= _IMPORT_MAX_INFLATED_BYTES:
                return _get_decompressor().decompress(blob, max_output_size=_IMPORT_MAX_INFLATED_BYTES)
        except zstd.ZstdError:
            pass
        raise damaged
    if blob[:2] == _GZIP_MAGIC:
        try:
            out = zlib.decompressobj(zlib.MAX_WBITS | 16).decompress(blob, _IMPORT_MAX_INFLATED_BYTES + 1)
        except zlib.error:
            raise damaged
        if len(out) > _IMPORT_MAX_INFLATED_BYTES:
            raise damaged
        return out
    return blob


class TransferService:
    """Moving a user's data in and out: .mimic export and import, the privacy export, and
    account deletion. Reached as `cog.transfer`."""

    def __init__(self, cog):
        self.cog = cog

    async def _execute_export(self, interaction: discord.Interaction, profile_names: List[str], filters: Set[str], passphrase: Optional[str] = None):
        user_id = interaction.user.id
        # Names resolve here, on the loop, against the cog's index; the shard reads, the
        # seal and the passphrase's key derivation all run off it.
        pids = {name: self.cog.profile_manager._get_pid_from_name(user_id, name) for name in profile_names}
        file_data = await asyncio.to_thread(self._build_export, user_id, pids, filters, passphrase)

        # Discord's default upload: what every channel takes, a DM or an unboosted server.
        limit = discord.utils.DEFAULT_FILE_SIZE_LIMIT_BYTES
        if len(file_data) > limit:
            await interaction.followup.send(
                embed=text_embed(f"❌ That export comes to {len(file_data) / (1024 * 1024):.1f} MB, over the "
                f"{limit // (1024 * 1024)} MB Discord lets me send. Export fewer profiles at a "
                "time, or leave out Long-Term Memories."), ephemeral=True)
            return

        filename = f"mimic_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.mimic"
        discord_file = discord.File(io.BytesIO(file_data), filename=filename)

        msg = "✅ Export complete."
        if passphrase:
            msg += " Your data has been securely encrypted with your passphrase for self-hosted migration."
        else:
            msg += " Your data is encrypted with this instance's master key, so only this instance can import it."

        await interaction.followup.send(embed=text_embed(msg), file=discord_file, ephemeral=True)

    def _build_export(self, user_id: int, pids: Dict[str, Optional[str]], filters: Set[str],
                      passphrase: Optional[str]) -> bytes:
        """The .mimic file for `pids` (name -> PID), as bytes. Blocking: run it in a thread."""
        user_dir = os.path.join(USERS_DIR, str(user_id), "profiles")
        decrypt = self.cog.storage_manager._decrypt_data

        def unseal(value):
            # Values sealed one at a time, before shards were encrypted whole, open only
            # with this instance's key -- which the file's reader need not hold.
            if isinstance(value, str):
                return decrypt(value)
            if isinstance(value, list):
                return [unseal(v) for v in value]
            if isinstance(value, dict):
                return {k: unseal(v) for k, v in value.items()}
            return value

        profiles = {}
        for name, pid in pids.items():
            p_dir = os.path.join(user_dir, pid) if pid else None
            if not p_dir or not os.path.isdir(p_dir):
                continue
            p_data = self.cog.profile_manager._get_profile_by_pid(user_id, pid) or {}
            config = dict(p_data.get("config") or {})
            # Retired: older shards still carry it, and a PID is the folder, never config.
            config.pop("profile_id", None)
            entry = {"pid": pid, "config": config, "prompts": p_data.get("prompts") or {},
                     "ltm": [], "training": []}
            if "ltm" in filters:
                # One list, every server, as a .mimic has always carried it: the per-server
                # files are this instance's layout, and any version reads a list.
                entry["ltm"] = read_ltm_dir(p_dir, self.cog.fernet)
            if "training" in filters:
                entry["training"] = IOManager.read_json_gzip(
                    os.path.join(p_dir, "training.json.gz"), self.cog.fernet) or []
            profiles[name] = unseal(entry)

        raw = json.dumps({"exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          "profiles": profiles})
        # Fernet, deliberately, where every file on disk is AES-GCM: the payload below
        # is stored as a *string* inside a plaintext JSON container that travels between
        # installs, so base64 is the point rather than the overhead it is on disk, and
        # any build that can import a .mimic must keep being able to import this one.
        compressed = _get_compressor().compress(raw)
        container = {"mimic_version": "3.0"}
        if passphrase:
            salt = os.urandom(16)
            container["auth_mode"] = "passphrase"
            container["salt"] = base64.b64encode(salt).decode('utf-8')
            container["payload"] = _passphrase_fernet(passphrase, salt).encrypt(compressed).decode('utf-8')
        else:
            container["auth_mode"] = "master"
            container["payload"] = self.cog.fernet.encrypt(compressed).decode('utf-8')
        return json.dumps(container, option=json.OPT_INDENT_2)

    def _open_export(self, file_bytes: bytes, passphrase: Optional[str]) -> Dict[str, Any]:
        """A .mimic file decrypted, inflated and parsed. Blocking: run it in a thread.

        Raises ValueError carrying the message the user is shown.
        """
        try:
            container = json.loads(file_bytes)
        except json.JSONDecodeError:
            raise ValueError("The provided file is not a valid MimicAI 3.0 export container (Invalid JSON).")
        if (not isinstance(container, dict) or container.get("mimic_version") != "3.0"
                or not isinstance(container.get("payload"), str)):
            raise ValueError("Plaintext and legacy v2.0 exports cannot be imported. "
                             "Please use a valid v3.0 encrypted `.mimic` file.")

        payload = container["payload"].encode('utf-8')
        if container.get("auth_mode") == "passphrase":
            if not passphrase:
                raise ValueError("This file is encrypted with a passphrase. Please use the import command properly to enter it.")
            salt_b64 = container.get("salt")
            if not salt_b64:
                raise ValueError("Corrupted passphrase export: missing cryptographic salt.")
            try:
                decrypted = _passphrase_fernet(passphrase, base64.b64decode(salt_b64)).decrypt(payload)
            except InvalidToken:
                raise ValueError("Decryption failed. The passphrase provided is incorrect.")
        else:
            try:
                decrypted = self.cog.fernet.decrypt(payload)
            except InvalidToken:
                raise ValueError("Master key decryption failed. This file belongs to a different MimicAI instance and cannot be imported here without a passphrase migration export.")

        data = json.loads(_inflate_export(decrypted))
        if not isinstance(data, dict) or not isinstance(data.get("profiles"), dict):
            raise ValueError("Decrypted payload is missing the profiles object.")
        # Here, off the loop: cleaning is a scrub over every string the file holds.
        for p_data in data["profiles"].values():
            if not isinstance(p_data, dict):
                continue
            if isinstance(p_data.get("config"), dict):
                for key in _BORROW_POINTER_KEYS:
                    p_data["config"].pop(key, None)
            if isinstance(p_data.get("prompts"), dict):
                p_data["prompts"] = _clean_imported_prompts(p_data["prompts"])
        return data

    async def _execute_import(self, interaction: discord.Interaction, file_bytes: bytes, passphrase: Optional[str] = None):
        try:
            data = await asyncio.to_thread(self._open_export, file_bytes, passphrase)

            user_id = interaction.user.id
            index = self.cog.profile_manager._get_user_index(user_id)
            personal = index.setdefault("personal", {})
            profiles = data["profiles"]
            room = defaultConfig.LIMIT_PROFILES - len(personal)
            if len(profiles) > room:
                raise ValueError(f"This file holds {len(profiles)} profiles and you have room for "
                                 f"{max(room, 0)} more (the limit is {defaultConfig.LIMIT_PROFILES}). "
                                 "Delete some first, or export fewer.")

            # Planned on the loop, with no await between the limit check and the last
            # name reserved; the shards are written after, off it, and the index saved last.
            jobs, skipped, unrated, declared = [], [], [], []
            for name, p_data in profiles.items():
                local_name = self._generate_unique_local_name(user_id, name)
                valid, reason = self.cog.profile_manager._is_valid_profile_name(local_name)
                if not valid or not isinstance(p_data, dict):
                    skipped.append(f"- `{name[:40]}`: {reason or 'unreadable entry'}")
                    continue

                config = p_data.get("config") if isinstance(p_data.get("config"), dict) else {}
                prompts = p_data.get("prompts") if isinstance(p_data.get("prompts"), dict) else {}
                new_pid = self.cog.profile_manager._mint_pid(user_id, "A")
                config["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                # Adult and Refused alone cross: dropping either would let the profile run
                # where it may not, and an export-import round trip must not launder a
                # refusal. Refused crosses as written, since claiming it gains nothing. Adult
                # crosses as an unchecked declaration: a passphrase file is anyone's to
                # write, and a verdict it claimed -- hash and all, which the file's own text
                # computes -- ran without the check for what Refused catches. Any other was
                # made about another copy, perhaps on another instance. Unrated runs the
                # same and waits to be submitted before sharing. A provider's hold crosses
                # with whatever it rode on, for the same reason Refused does.
                rating = config.pop("content_rating", None)
                verdict = rating.get("verdict") if isinstance(rating, dict) else None
                hold = _carried_hold(rating) if isinstance(rating, dict) else {}
                if verdict == CONTENT_RATING_REFUSED:
                    config["content_rating"] = rating
                elif verdict == CONTENT_RATING_ADULT:
                    config["content_rating"] = {**_unchecked_declaration(), **hold}
                    declared.append(local_name)
                elif hold:
                    config["content_rating"] = {"verdict": CONTENT_RATING_UNRATED, **hold}
                elif verdict == CONTENT_RATING_GENERAL:
                    unrated.append(local_name)

                ltm = p_data.get("ltm") if isinstance(p_data.get("ltm"), list) else []
                training = p_data.get("training") if isinstance(p_data.get("training"), list) else []
                # Newest last in both, as they are written. Memories are bounded per server
                # as they are filed (write_imported_ltm).
                jobs.append((new_pid, {"name": local_name, "config": config, "prompts": prompts, "child_bot": None},
                             ltm, training[-defaultConfig.LIMIT_TRAINING:]))
                personal[local_name] = new_pid

            try:
                await asyncio.to_thread(self._write_imported_profiles, user_id, jobs)
            except BaseException:
                for _, profile, _, _ in jobs:
                    personal.pop(profile["name"], None)
                raise
            if jobs:
                self.cog.profile_manager._save_user_index(user_id, index)

            if jobs:
                lines = ["### 📥 Import Successful\nThe following profiles have been securely decrypted and added to your vault:"]
                lines += [f"- `{profile['name']}`" for _, profile, _, _ in jobs]
            else:
                lines = ["Nothing was imported."]
            if unrated:
                lines.append(f"-# Content ratings other than Adult 18+ do not carry over: submit "
                             f"{', '.join(f'`{n}`' for n in unrated)} again before sharing.")
            if declared:
                lines.append(f"-# {', '.join(f'`{n}`' for n in declared)} came in as Adult 18+, and "
                             "runs once the content check has read it.")
            if skipped:
                lines.append("**Not imported:**")
                lines += skipped
            await interaction.followup.send(embed=text_embed("\n".join(lines)), ephemeral=True)

        except ValueError as ve:
            await interaction.followup.send(embed=text_embed(f"❌ **Import Rejected:** {ve}"), ephemeral=True)
        except Exception as e:
            await interaction.followup.send(embed=text_embed(f"❌ **Import Failed:** An unexpected error occurred: {suppress_link_previews(str(e))}"), ephemeral=True)

    def _write_imported_profiles(self, user_id: int, jobs: List[Tuple[str, Dict[str, Any], list, list]]):
        """Writes each (pid, profile, ltm, training). Blocking: run it in a thread.

        All or nothing: a failure removes what this call wrote, which a rebuild would
        otherwise adopt as profiles nobody asked for.
        """
        written = []
        try:
            for pid, profile, ltm, training in jobs:
                p_dir = os.path.join(USERS_DIR, str(user_id), "profiles", pid)
                written.append(p_dir)
                self.cog.profile_manager._save_profile_by_pid(user_id, pid, profile)
                if ltm:
                    write_imported_ltm(p_dir, ltm, self.cog.fernet)
                if training:
                    IOManager.write_json_gzip(training, os.path.join(p_dir, "training.json.gz"), self.cog.fernet)
        except BaseException:
            for p_dir in written:
                shutil.rmtree(p_dir, ignore_errors=True)
            raise

    async def _execute_privacy_export(self, user_id: int, interaction: discord.Interaction):
        user_id_str = str(user_id)
        user_dir = os.path.join(self.cog.USERS_DIR, user_id_str)
        
        if not os.path.exists(user_dir):
            await interaction.followup.send(embed=text_embed("No data found for your account."), ephemeral=True)
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

                    elif file.endswith('.bin'):
                        # Voice recordings and spoken consent, sealed audio. Exported as the
                        # plain file under its own name; the record beside it gives the type.
                        audio = IOManager.read_blob(src_file, self.cog.fernet)
                        if audio is not None:
                            with open(os.path.join(target_dir, file), 'wb') as f:
                                f.write(audio)
            
            zip_base = os.path.join(temp_d, f"mimicai_data_{user_id_str}")
            shutil.make_archive(zip_base, 'zip', export_base)
            return temp_d, f"{zip_base}.zip"
            
        temp_dir, zip_path = await asyncio.to_thread(make_zip)
        
        try:
            file = discord.File(zip_path, filename=f"privacy_export_{user_id_str}.zip")
            await interaction.followup.send(embed=text_embed("Here is your complete data export. This archive contains your profiles, API keys and memory data in unencrypted, uncompressed JSON format, and your voice recordings as plain audio (`.bin`). Keep it safe: your API keys are in it unencrypted."), file=file, ephemeral=True)
        except Exception as e:
            await interaction.followup.send(embed=text_embed(f"Failed to send export file: {suppress_link_previews(str(e))}"), ephemeral=True)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def _execute_account_deletion(self, user_id: int, interaction: discord.Interaction):
        user_id_str = str(user_id)
        
        # 1. Shutdown Child Bots
        child_bots_to_kill = [bot_id for bot_id, data in self.cog.child_bots.items() if str(data.get("owner_id")) == user_id_str]
        for bot_id in child_bots_to_kill:
            await self.cog.manager_queue.put({"action": "shutdown_bot", "bot_id": bot_id})
            bot = self.cog.child_bots.pop(bot_id, None) or {}
            self.cog.child_bots_by_owner_profile.pop((user_id, bot.get("profile_name")), None)

        guild_ids = [g.id for g in self.cog.bot.guilds]

        def _scrub():  # Steps 2-4b: every one writes a file, the rmtree a whole tree.
            # 2. Scrub from Public Hub
            pids_to_unpublish = [d["id"] for d in self.cog.library._iter_public_entries(user_id, include_orphaned=True)]
            for pid in pids_to_unpublish:
                self.cog.public_profiles.pop(pid, None)
            if pids_to_unpublish:
                self.cog.library._save_public_index()

            # 3. Scrub from Server Pointers
            for gid in guild_ids:
                idx = self.cog.server_manager._get_server_index(str(gid))
                changed = False
                for prov in list(idx.get("assigned_keys", {}).keys()):
                    if idx["assigned_keys"][prov].get("user_id") == user_id:
                        del idx["assigned_keys"][prov]
                        self.cog.server_key_pointers.pop((gid, prov), None)
                        changed = True
                if changed:
                    self.cog.server_manager._save_server_index(str(gid), idx)

            # 4. Delete Core User Directory
            user_dir = os.path.join(self.cog.USERS_DIR, user_id_str)
            if os.path.exists(user_dir):
                shutil.rmtree(user_dir, ignore_errors=True)

            # 4b. Forget them in the borrow index, as borrower and as source. The rmtree
            # above never writes their index.json, so the reconcile in _save_user_index
            # cannot fire and both directions would otherwise survive the deletion.
            self.cog.borrows._borrow_index_drop_user(user_id)
        await asyncio.to_thread(_scrub)

        # Deliberately not touched: data/mod/blacklist.json. An enforcement record is
        # about the account, not owned by it, and a ban that a user can lift by deleting
        # and re-creating their profiles is not a ban. Privacy Policy 5 says so.

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

        await interaction.followup.send(embed=text_embed("Account Deleted. All your profiles, memories, and settings have been permanently erased from this instance."), ephemeral=True)

    def _generate_unique_local_name(self, user_id: int, original_name: str) -> str:
        """The source's own name, else the first free `name-2`, `name-3`..., cut to
        `_is_valid_profile_name`'s 20 characters."""
        index = self.cog.profile_manager._get_user_index(user_id)
        taken = set(index.get("personal", [])) | set(index.get("borrowed", []))
        base = original_name.lower().strip()[:20]
        name, counter = base, 1
        while name in taken:
            counter += 1
            suffix = f"-{counter}"
            name = base[:20 - len(suffix)] + suffix
        return name
