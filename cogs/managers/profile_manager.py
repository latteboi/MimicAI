import os
import io
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
    DEFAULT_AUTO_MODERATOR_PROMPT, DEFAULT_SAFETY_SETTINGS,
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

    def _get_pid_from_name(self, user_id: int, profile_name: str, is_borrowed: bool = False) -> str:
        index = self._get_user_index(user_id)
        if not is_borrowed:
            if isinstance(index.get("system"), dict) and profile_name in index.get("system", {}):
                return index["system"][profile_name]
            mapping = index.get("personal", {})
        else:
            mapping = index.get("borrowed", {})
        if isinstance(mapping, dict):
            return mapping.get(profile_name, profile_name)
        return profile_name

    def _get_pid_from_name_any(self, user_id: int, profile_name: str) -> str:
        index = self._get_user_index(user_id)
        if isinstance(index.get("system"), dict) and profile_name in index["system"]:
            return index["system"][profile_name]
        if isinstance(index.get("personal"), dict) and profile_name in index["personal"]:
            return index["personal"][profile_name]
        if isinstance(index.get("borrowed"), dict) and profile_name in index["borrowed"]:
            return index["borrowed"][profile_name]
        return profile_name

    def _get_name_from_pid(self, user_id: int, target_pid: str) -> Optional[str]:
        index = self._get_user_index(user_id)
        personal = index.get("personal", {})
        if isinstance(personal, dict):
            for name, pid in personal.items():
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
        if not pointer:
            return None
        if pointer.startswith("pub_") or pointer.startswith("A"):
            if not self.cog.public_profiles:
                self._load_public_profiles()
            target = self.cog.public_profiles.get(pointer)
            if target:
                if isinstance(target, str) and ":" in target:
                    pointer = target
                elif isinstance(target, dict):
                    owner_id = target.get("owner_id")
                    pid = target.get("original_pid") or target.get("original_profile_id")
                    if owner_id and pid:
                        return int(owner_id), pid

        if ":" in pointer:
            try:
                owner_id_str, pid = pointer.split(":", 1)
                return int(owner_id_str), pid
            except ValueError:
                pass
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
        new_data["config"]["profile_id"] = str(uuid.uuid4().hex[:8].upper())
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
            if not is_borrowed and "profile_id" not in config:
                config["profile_id"] = str(uuid.uuid4().hex[:8].upper())
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
        owner_id = int(defaultConfig.DISCORD_OWNER_ID)
        if user_id != owner_id:
            owner_idx = self._get_user_index(owner_id)
            if profile_name in owner_idx.get("system", {}):
                return owner_id, profile_name

        index = self._get_user_index(user_id)
        if profile_name in index.get("borrowed", []):
            b_config = self._get_profile_config(user_id, profile_name, True) or {}
            eff_owner = int(b_config.get("original_owner_id", user_id))
            eff_name = b_config.get("original_profile_name", profile_name)
            return eff_owner, eff_name
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
                "ltm_summarization_context": 10, "ltm_scope": "server", "safety_level": "low",
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
                "ltm_summarization_context": 10, "ltm_scope": "server", "safety_level": "low",
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

    def _check_unrestricted_safety_policy(self, profile_owner_id: int, profile_name: str, channel: discord.abc.Messageable) -> bool:
        index = self._get_user_index(profile_owner_id)
        is_borrowed = profile_name in index.get("borrowed", [])
        config = self._get_profile_config(profile_owner_id, profile_name, is_borrowed) or {}

        safety_level = config.get("safety_level", "low")

        if safety_level == "unrestricted":
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
        for local_name in borrowed:
            b_config = self._get_profile_config(user_id, local_name, True)
            if b_config:
                o_id = b_config.get("original_owner_id")
                o_pid = b_config.get("original_pid")
                o_name = b_config.get("original_profile_name")
                if o_id and (o_pid or o_name):
                    profiles_by_owner.setdefault(str(o_id),[]).append((local_name, o_pid, o_name))

        removed_count = 0
        ids_to_remove =[]

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

    async def _is_profile_content_safe(self, user_id: int, profile_name: str, display_name: str, avatar_url: Optional[str]) -> Tuple[bool, str]:
        prompt_text = self.cog.global_prompts.get("AUTO_MODERATOR", DEFAULT_AUTO_MODERATOR_PROMPT)
        
        image_data = None
        content_type = "image/png"
        
        if avatar_url:
            try:
                # The spoofed User-Agent rides on the request, not the client, so
                # this can share the process-wide pool with everything else.
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
                response = await get_shared_client().get(avatar_url, follow_redirects=True, timeout=10.0, headers=headers)
                response.raise_for_status()
                image_data = response.content
                content_type = response.headers.get("Content-Type", "image/png")
            except httpx.RequestError as e:
                return False, f"Could not download the avatar image from the provided URL: {e}"
            except Exception as e:
                return False, f"An error occurred while processing the avatar URL: {e}"

        try:
            or_key = self.cog.storage_manager._get_api_key_for_user(user_id, "openrouter")
            g_key = self.cog.storage_manager._get_api_key_for_user(user_id, "gemini")
            
            if not or_key and not g_key:
                return False, "A Personal API Key (OpenRouter or Google) is required to perform safety analysis for public profiles. Please configure one via the `/settings` command."
            
            parts_list = [
                f"<target_content>\nProfile Name: {profile_name}\nDisplay Name: {display_name}\n</target_content>"
            ]
            if image_data:
                parts_list.append({"mime_type": content_type, "data": image_data})

            eval_payload = [{"role": "user", "parts": parts_list}]
            gen_cfg = {"temperature": 0.0, "top_k": 1, "top_p": 0.9}
            
            response = None
            status = "api_error"
            or_error = None
            g_error = None
            
            if or_key:
                used_model = "amazon/nova-lite-v1"
                try:
                    model = OpenRouterModel(
                        api_key=or_key,
                        model_name=used_model, 
                        system_instruction=prompt_text
                    )
                    response = await model.generate_content_async(eval_payload, generation_config=gen_cfg)
                    status = "blocked_by_safety" if not response.candidates else "success"
                except Exception as e:
                    or_error = str(e)
                    print(f"OpenRouter Auto-Mod failed: {e}")
                    response = None
                finally:
                    self.cog._log_api_call(user_id=0, guild_id=None, context="moderation_check", model_used=used_model, status=status)

            if not response and g_key:
                status = "api_error"
                used_model = "gemini-2.5-flash-lite"
                try:
                    model = GoogleGenAIModel(
                        api_key=g_key,
                        model_name=used_model,
                        system_instruction=prompt_text,
                        safety_settings=DEFAULT_SAFETY_SETTINGS
                    )
                    response = await model.generate_content_async(eval_payload, generation_config=gen_cfg)
                    status = "blocked_by_safety" if not response.candidates else "success"
                except Exception as e:
                    g_error = str(e)
                    print(f"Google Auto-Mod failed: {e}")
                    response = None
                finally:
                    self.cog._log_api_call(user_id=0, guild_id=None, context="moderation_check_fallback" if or_key else "moderation_check", model_used=used_model, status=status)

            if not response or not response.candidates:
                reason = "Unknown"
                if response and hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                    reason = response.prompt_feedback.block_reason.name
                
                print(f"Auto-moderation check failed. OpenRouter Error: {or_error} | Google Error: {g_error} | Block Reason: {reason}")
                
                err_msg = "Content was flagged as unsafe or validation failed."
                if or_error and g_error:
                    err_msg += f" Primary model failed ({self.cog._format_api_error(Exception(or_error))}). Fallback model failed ({self.cog._format_api_error(Exception(g_error))})."
                elif or_error:
                    err_msg += f" Model failed ({self.cog._format_api_error(Exception(or_error))})."
                elif g_error:
                    err_msg += f" Model failed ({self.cog._format_api_error(Exception(g_error))})."
                elif reason != "Unknown":
                    err_msg += f" Block reason: {reason}."
                
                return False, err_msg

            result = ""
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    result = "".join(p.text for p in candidate.content.parts if hasattr(p, 'text')).strip().upper()

            if "SAFE" in result and "UNSAFE" not in result:
                return True, "Content is safe."
            elif result == "SAFE":
                return True, "Content is safe."
            else:
                return False, "Content was flagged as unsafe by the AI moderator."
        except Exception as e:
            print(f"Auto-moderation check failed: {e}")
            traceback.print_exc()
            return False, "An error occurred during the moderation check."
        

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
        safety_level = profile_data.get("safety_level", "low").title()
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
        
        if is_borrowed:
            borrowed_config = self._get_profile_config(owner_id, profile_name, True) or {}
            a_class_pid = borrowed_config.get("original_profile_id", "Unknown")
            b_class_pid = self._get_pid_from_name_any(owner_id, profile_name)
            embed.add_field(name="Profile ID (Source)", value=f"`{a_class_pid}`", inline=True)
            embed.add_field(name="Profile ID (Local)", value=f"`{b_class_pid}`", inline=True)
        else:
            profile_id = self._get_profile_id(effective_owner_id, effective_profile_name)
            embed.add_field(name="Profile ID (PID)", value=f"`{profile_id}`", inline=True)
            embed.add_field(name="\u200b", value="\u200b", inline=True)

        embed.add_field(name="Safety Level", value=f"`{safety_level}`", inline=True)
        embed.add_field(name="\u200b", value="\u200b", inline=True) # Spacer for alignment

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
        safety_level = config.get("safety_level", "low").title()

        embed.add_field(name="Profile Type", value=f"`{profile_type}`", inline=True)
        embed.add_field(name="Created", value=created_display, inline=True)
        embed.add_field(name="Display Name", value=f"`{display_name}`", inline=True)
        
        if is_borrowed:
            borrowed_config = self._get_profile_config(user_id, profile_name, True) or {}
            a_class_pid = borrowed_config.get("original_profile_id", "Unknown")
            b_class_pid = self._get_pid_from_name_any(user_id, profile_name)
            embed.add_field(name="Profile ID (Source)", value=f"`{a_class_pid}`", inline=True)
            embed.add_field(name="Profile ID (Local)", value=f"`{b_class_pid}`", inline=True)
        else:
            profile_id = self._get_profile_id(effective_owner_id, effective_profile_name)
            embed.add_field(name="Profile ID (PID)", value=f"`{profile_id}`", inline=True)
            embed.add_field(name="\u200b", value="\u200b", inline=True)
            
            if profile_id.startswith("X"):
                embed.description = f"⚠️ **System Profile.** Global settings managed by Bot Admin.\n\n" + (embed.description or "")
                profile_type = "System"

        embed.add_field(name="Safety Level", value=f"`{safety_level}`", inline=True)

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
                public_key = next((k for k, v in self.cog.public_profiles.items() if v == source_pointer), None)
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
                "ltm_creation_enabled": False,
                "ltm_scope": "server"
            })

            pid = f"B{uuid.uuid4().hex[:15].upper()}"

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
    
