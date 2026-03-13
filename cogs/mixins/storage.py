import os
import re
import gzip
import pathlib
import shutil
import discord
import uuid
import datetime
import orjson as json
from typing import Dict, List, Any, Optional, Literal, Union, Tuple
from cryptography.fernet import Fernet, InvalidToken
import asyncio
import traceback
import time
import random
import numpy as np
from .constants import *

def _delete_file_shard(file_path: str):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except OSError as e:
        print(f"Error deleting file shard {file_path}: {e}")

def _atomic_json_save(data: Any, file_path: str):
    temp_file_path = file_path + ".tmp"
    try:
        with open(temp_file_path, 'wb') as f:
            f.write(json.dumps(data, option=json.OPT_INDENT_2))
        os.replace(temp_file_path, file_path) 
    except IOError as e:
        print(f"Error saving data to {file_path}: {e}")
        if os.path.exists(temp_file_path):
            try: os.remove(temp_file_path)
            except OSError: pass 
        raise 
    except Exception as e: 
        print(f"Unexpected error saving data to {file_path}: {e}")
        if os.path.exists(temp_file_path):
            try: os.remove(temp_file_path)
            except OSError: pass
        raise 

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    np_vec1=np.array(vec1); np_vec2=np.array(vec2); dot=np.dot(np_vec1,np_vec2); n1=np.linalg.norm(np_vec1); n2=np.linalg.norm(np_vec2)
    return 0.0 if n1==0 or n2==0 else float(dot/(n1*n2))

def _quantize_embedding(embedding: List[float]) -> List[float]:
    if not embedding: return []
    return np.array(embedding, dtype=np.float32).astype(np.float16).tolist()

def _dequantize_embedding(quantized_embedding: List[float]) -> List[float]:
    if not quantized_embedding: return []
    return np.array(quantized_embedding, dtype=np.float16).astype(np.float32).tolist()

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
                f.write(json.dumps(data, option=json.OPT_INDENT_2))
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
                decrypted_bytes = fernet.decrypt(file_bytes)
                decompressed_bytes = gzip.decompress(decrypted_bytes)
            else: 
                decompressed_bytes = gzip.decompress(file_bytes)
            
            return json.loads(decompressed_bytes)
        except (IOError, json.JSONDecodeError, gzip.BadGzipFile, InvalidToken) as e:
            print(f"IOManager Read Error ({file_path}): {e}")
            return None
            
    @staticmethod
    def write_json_gzip(data: Any, file_path: str, fernet: Optional[Fernet] = None, encrypted: bool = True):
        temp_file_path = file_path + ".tmp"
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            json_bytes = json.dumps(data, option=json.OPT_INDENT_2)
            compressed_bytes = gzip.compress(json_bytes)
            
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

class StorageMixin:

    def _migrate_to_pid_v2(self):
        users_dir = pathlib.Path(self.USERS_DIR)
        if not users_dir.exists(): return
        print("Phase 4 Migration: Upgrading to A/B 16-Hex PID Architecture...")
        import uuid
        migrated_any = False
        
        for user_dir in users_dir.iterdir():
            if not user_dir.is_dir() or not user_dir.name.isdigit(): continue
            index_path = user_dir / "index.json"
            if not index_path.exists(): continue
            
            index = IOManager.read_json(str(index_path))
            if not index: continue
            
            migrated = False
            for p_type, prefix in [("personal", "A"), ("borrowed", "B")]:
                p_data = index.get(p_type)
                if isinstance(p_data, list):
                    new_dict = {}
                    for pname in p_data:
                        if pname == DEFAULT_PROFILE_NAME:
                            new_dict[pname] = pname
                            continue
                        
                        pid = f"{prefix}{uuid.uuid4().hex[:15].upper()}"
                        new_dict[pname] = pid
                        
                        old_p_dir = user_dir / "profiles" / pname
                        new_p_dir = user_dir / "profiles" / pid
                        if old_p_dir.exists() and old_p_dir.is_dir():
                            old_p_dir.rename(new_p_dir)
                            with open(new_p_dir / "name.txt", "w", encoding="utf-8") as f:
                                f.write(pname)
                    index[p_type] = new_dict
                    migrated = True
                    migrated_any = True
            
            if migrated:
                IOManager.write_json(index, str(index_path))
                
        if migrated_any:
            print("Phase 4 Migration complete. Immutable PID directories created.")

    def _get_pid_from_name(self, user_id: int, profile_name: str, is_borrowed: bool = False) -> str:
        if profile_name == DEFAULT_PROFILE_NAME: return DEFAULT_PROFILE_NAME
        index = self._get_user_index(user_id)
        key = "borrowed" if is_borrowed else "personal"
        mapping = index.get(key, {})
        if isinstance(mapping, dict):
            return mapping.get(profile_name, profile_name)
        return profile_name

    def _get_pid_from_name_any(self, user_id: int, profile_name: str) -> str:
        if profile_name == DEFAULT_PROFILE_NAME: return DEFAULT_PROFILE_NAME
        index = self._get_user_index(user_id)
        if isinstance(index.get("personal"), dict) and profile_name in index["personal"]:
            return index["personal"][profile_name]
        if isinstance(index.get("borrowed"), dict) and profile_name in index["borrowed"]:
            return index["borrowed"][profile_name]
        return profile_name

    def _get_name_from_pid(self, user_id: int, target_pid: str) -> Optional[str]:
        if target_pid == DEFAULT_PROFILE_NAME: return DEFAULT_PROFILE_NAME
        index = self._get_user_index(user_id)
        personal = index.get("personal", {})
        if isinstance(personal, dict):
            for name, pid in personal.items():
                if pid == target_pid: return name
        return None

    def _is_valid_profile_name(self, name: str) -> tuple[bool, str]:
        if not name or not name.strip():
            return False, "Profile name cannot be empty."
        if len(name) > 30:
            return False, "Profile name must be 30 characters or fewer."
        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            return False, "Profile name can only contain letters, numbers, underscores, and hyphens (no spaces)."
        if name.lower() in ["mimic", "clyde", "system", "user", "none", "all"]:
            return False, "This name is a reserved system keyword and cannot be used."
        return True, ""

    def _get_user_hash(self, user_id: int) -> str:
        import hashlib
        # Prefix with 'U' and return 15 hex characters for a total 16-character PID
        return "A" + hashlib.sha256(str(user_id).encode()).hexdigest()[:15].upper()

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

    def _encrypt_data(self, plaintext: str) -> str:
        if not self.fernet or not plaintext:
            return plaintext
        try:
            return self.fernet.encrypt(plaintext.encode()).decode()
        except Exception as e:
            print(f"Encryption failed: {e}")
            return plaintext

    def _decrypt_data(self, encrypted_text: str) -> str:
        if not self.fernet or not encrypted_text:
            return encrypted_text
        try:
            return self.fernet.decrypt(encrypted_text.encode()).decode()
        except Exception:
            # If decryption fails (e.g. text is already plain), return as is
            return encrypted_text

    def _atomic_json_save_gzip(self, data: Any, file_path: str, encrypted: bool = True):
        IOManager.write_json_gzip(data, file_path, self.fernet, encrypted)

    def _load_json_gzip(self, file_path: str, encrypted: bool = True) -> Optional[Any]:
        return IOManager.read_json_gzip(file_path, self.fernet, encrypted)
        
    def _load_ltm_shard(self, user_id: str, profile_name: str) -> Optional[Dict[str, List[Dict]]]:
        pid = self._get_pid_from_name_any(int(user_id), profile_name)
        file_path = os.path.join(self.USERS_DIR, str(user_id), "profiles", pid, "ltm.json.gz")
        return self._load_json_gzip(file_path)

    def _save_ltm_shard(self, user_id: str, profile_name: str, data: Dict[str, List[Dict]]):
        pid = self._get_pid_from_name_any(int(user_id), profile_name)
        file_path = os.path.join(self.USERS_DIR, str(user_id), "profiles", pid, "ltm.json.gz")
        self._atomic_json_save_gzip(data, file_path)

    def _delete_ltm_shard(self, user_id: str, profile_name: str):
        pid = self._get_pid_from_name_any(int(user_id), profile_name)
        file_path = os.path.join(self.USERS_DIR, str(user_id), "profiles", pid, "ltm.json.gz")
        _delete_file_shard(file_path)

    def _rename_ltm_shards(self, user_id: str, old_profile_name: str, new_profile_name: str):
        pass # Obsolete. Data moves seamlessly when string changes in index.json

    def _copy_ltm_shard(self, user_id: str, source_profile_name: str, new_profile_name: str):
        src_pid = self._get_pid_from_name_any(int(user_id), source_profile_name)
        new_pid = self._get_pid_from_name_any(int(user_id), new_profile_name)
        source_path = os.path.join(self.USERS_DIR, str(user_id), "profiles", src_pid, "ltm.json.gz")
        if os.path.exists(source_path):
            new_path = os.path.join(self.USERS_DIR, str(user_id), "profiles", new_pid, "ltm.json.gz")
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            import shutil
            shutil.copy2(source_path, new_path)

    def _add_ltm(self, profile_owner_id: int, profile_name: str, summary: str, summary_embedding: List[float], guild_id: Optional[int], triggering_user_id: int, user_dn: Optional[str] = None, force_user_scope: bool = False):
        owner_id_str = str(profile_owner_id)
        
        index = self._get_user_index(profile_owner_id)
        is_borrowed = profile_name in index.get("borrowed", [])
        profile_settings = self._get_profile_config(profile_owner_id, profile_name, is_borrowed) or {}
        
        ltm_scope = profile_settings.get('ltm_scope', 'server')
        
        if force_user_scope:
            ltm_scope = 'user'

        context_id = None
        if ltm_scope == 'server':
            context_id = guild_id
        elif ltm_scope == 'user':
            context_id = profile_owner_id

        ltm_data = self._load_ltm_shard(owner_id_str, profile_name)
        if ltm_data is None:
            ltm_data = {"guild": [], "dm": []}
        
        context_type = "guild"
        ltm_list = ltm_data.get(context_type, [])

        # [NEW] Non-Destructive Rolling Window
        is_premium = self.is_user_premium(profile_owner_id)
        limit = defaultConfig.LIMIT_LTM_PREMIUM if is_premium else defaultConfig.LIMIT_LTM_FREE
        
        # Sort oldest -> newest to ensure we pop the correct one
        ltm_list.sort(key=lambda x: x.get('created_ts', x.get('ts')))
        
        # CRITICAL FIX: Use 'if' instead of 'while'. 
        # This ensures we only remove ONE memory to make room for the NEW one.
        # This preserves legacy data (e.g., if user has 250/50, they stay at 250).
        if len(ltm_list) >= limit:
            ltm_list.pop(0)

        now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        entry = {
            "id": str(uuid.uuid4())[:8], 
            "created_ts": now_ts,
            "modified_ts": now_ts,
            "sum": summary.strip(),
            "s_emb": summary_embedding,
            "usr": user_dn,
            "scope": ltm_scope,
            "context_id": str(context_id) if context_id else None
        }
        ltm_list.append(entry)
        
        # Ensure we don't accidentally truncate with the global max clamp either
        # We use the larger of (current count) or (premium max) to ensure data safety
        max_safe_clamp = max(len(ltm_list), defaultConfig.LIMIT_LTM_PREMIUM)
        
        ltm_data[context_type] = sorted(ltm_list, key=lambda x: x.get('created_ts', x.get('ts')))[-max_safe_clamp:]
        self._save_ltm_shard(owner_id_str, profile_name, ltm_data)

    def update_ltm(self, profile_owner_id: int, profile_name: str, ltm_id: str, new_summary: str, new_embedding: List[float]) -> bool:
        owner_id_str = str(profile_owner_id)
        ltm_data = self._load_ltm_shard(owner_id_str, profile_name)
        if not ltm_data:
            return False

        for context_type in ["guild", "dm"]:
            ltm_list = ltm_data.get(context_type, [])
            for i, ltm_entry in enumerate(ltm_list):
                if ltm_entry.get("id") == ltm_id:
                    ltm_data[context_type][i]["sum"] = new_summary.strip()
                    ltm_data[context_type][i]["s_emb"] = new_embedding
                    ltm_data[context_type][i]["modified_ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    if "kw" in ltm_data[context_type][i]:
                        del ltm_data[context_type][i]["kw"]
                    self._save_ltm_shard(owner_id_str, profile_name, ltm_data)
                    return True
        return False
    
    def _load_training_shard(self, user_id: str, profile_name: str) -> Optional[List[Dict]]:
        pid = self._get_pid_from_name_any(int(user_id), profile_name)
        file_path = os.path.join(self.USERS_DIR, str(user_id), "profiles", pid, "training.json.gz")
        return self._load_json_gzip(file_path)

    def _save_training_shard(self, user_id: str, profile_name: str, data: List[Dict]):
        pid = self._get_pid_from_name_any(int(user_id), profile_name)
        file_path = os.path.join(self.USERS_DIR, str(user_id), "profiles", pid, "training.json.gz")
        self._atomic_json_save_gzip(data, file_path)

    def _delete_training_shard(self, user_id: str, profile_name: str):
        pid = self._get_pid_from_name_any(int(user_id), profile_name)
        file_path = os.path.join(self.USERS_DIR, str(user_id), "profiles", pid, "training.json.gz")
        _delete_file_shard(file_path)

    def _rename_training_shards(self, user_id: str, old_profile_name: str, new_profile_name: str):
        pass # Obsolete. Data moves seamlessly when string changes in index.json

    def _copy_training_shard(self, user_id: str, source_profile_name: str, new_profile_name: str):
        src_pid = self._get_pid_from_name_any(int(user_id), source_profile_name)
        new_pid = self._get_pid_from_name_any(int(user_id), new_profile_name)
        source_path = os.path.join(self.USERS_DIR, str(user_id), "profiles", src_pid, "training.json.gz")
        if os.path.exists(source_path):
            new_path = os.path.join(self.USERS_DIR, str(user_id), "profiles", new_pid, "training.json.gz")
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            import shutil
            shutil.copy2(source_path, new_path)

    def _get_session_dir_path(self, session_key: Any, session_type: str) -> pathlib.Path:
        if session_type == 'global_chat':
            _, user_id, _ = session_key
            return pathlib.Path(SESSIONS_GLOBAL_DIR) / str(user_id)
        
        # Phase 1: Consolidated Server Directory
        channel_id, _, _ = session_key
        channel = self.bot.get_channel(channel_id)
        server_id = channel.guild.id if channel and channel.guild else "dm"
        return pathlib.Path(SERVERS_DIR) / str(server_id) / "sessions" / str(channel_id) / session_type

    def _migrate_servers_directory(self):
        old_sessions_dir = pathlib.Path(SESSIONS_SERVERS_DIR)
        new_servers_dir = pathlib.Path(SERVERS_DIR)
        
        if not old_sessions_dir.exists():
            return
            
        print("Phase 1 Migration: Consolidating Server Sessions...")
        try:
            for server_dir in old_sessions_dir.iterdir():
                if not server_dir.is_dir():
                    continue
                    
                target_sessions_dir = new_servers_dir / server_dir.name / "sessions"
                target_sessions_dir.mkdir(parents=True, exist_ok=True)
                
                for channel_dir in server_dir.iterdir():
                    if channel_dir.is_dir():
                        target_channel_dir = target_sessions_dir / channel_dir.name
                        if not target_channel_dir.exists():
                            import shutil
                            shutil.move(str(channel_dir), str(target_channel_dir))
            
            import shutil
            shutil.rmtree(str(old_sessions_dir), ignore_errors=True)
            print("Phase 1 Migration complete. Old sessions directory removed.")
        except Exception as e:
            print(f"Error during Phase 1 Migration: {e}")

    def _migrate_users_root_directory(self):
        legacy_shares = pathlib.Path(LEGACY_SHARES_DIR)
        legacy_keys = pathlib.Path(LEGACY_PERSONAL_KEYS_DIR)
        users_dir = pathlib.Path(self.USERS_DIR)
        
        migrated_anything = False
        
        if legacy_keys.exists():
            print("Phase 2 Migration: Consolidating Personal API Keys...")
            for file_path in legacy_keys.iterdir():
                if file_path.name.endswith(".json.gz"):
                    user_id_str = file_path.name[:-len(".json.gz")]
                    target_dir = users_dir / user_id_str
                    target_dir.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.move(str(file_path), str(target_dir / "keys.json.gz"))
            import shutil
            shutil.rmtree(str(legacy_keys), ignore_errors=True)
            migrated_anything = True

        if legacy_shares.exists():
            print("Phase 2 Migration: Consolidating Profile Shares...")
            for file_path in legacy_shares.iterdir():
                if file_path.name.endswith(".json.gz"):
                    user_id_str = file_path.name[:-len(".json.gz")]
                    target_dir = users_dir / user_id_str
                    target_dir.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.move(str(file_path), str(target_dir / "shares.json.gz"))
            import shutil
            shutil.rmtree(str(legacy_shares), ignore_errors=True)
            migrated_anything = True
            
        if migrated_anything:
                print("Phase 2 Migration complete.")

    def _migrate_profiles_directory(self):
        legacy_profiles = pathlib.Path(LEGACY_PROFILES_DIR)
        if not legacy_profiles.exists(): return
        
        print("Phase 3 Migration: Splitting Monolithic Profiles & Consolidating Data...")
        users_dir = pathlib.Path(self.USERS_DIR)
        
        try:
            for profile_file in legacy_profiles.iterdir():
                if not profile_file.name.endswith(".json.gz"): continue
                user_id_str = profile_file.name[:-len(".json.gz")]
                user_target_dir = users_dir / user_id_str
                user_target_dir.mkdir(parents=True, exist_ok=True)
                
                # Load monoliths
                user_data = self._load_json_gzip(str(profile_file))
                if not user_data: continue
                
                app_data = IOManager.read_json_gzip(os.path.join(self.APPEARANCES_DIR, f"{user_id_str}.json.gz"), self.fernet) or {}
                bot_data = IOManager.read_json_gzip(os.path.join(LEGACY_CHILD_BOTS_DIR, f"{user_id_str}.json.gz"), encrypted=False) or {}
                
                # 1. Create Lightweight Index
                index_data = {
                    "personal": list(user_data.get("profiles", {}).keys()),
                    "borrowed": list(user_data.get("borrowed_profiles", {}).keys()),
                    "channel_active_profiles": user_data.get("channel_active_profiles", {})
                }
                IOManager.write_json(index_data, str(user_target_dir / "index.json"))
                
                # 2. Split Personal Profiles
                for pname, pdata in user_data.get("profiles", {}).items():
                    p_dir = user_target_dir / "profiles" / pname
                    p_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Separate Prompts
                    prompts = {
                        "persona": pdata.pop("persona", {}),
                        "ai_instructions": pdata.pop("ai_instructions", ["", "", "", ""]),
                        "image_generation_prompt": pdata.pop("image_generation_prompt", None),
                        "ltm_summarization_instructions": pdata.pop("ltm_summarization_instructions", self._encrypt_data(DEFAULT_LTM_SUMMARIZATION_INSTRUCTIONS))
                    }
                    IOManager.write_json_gzip(prompts, str(p_dir / "prompts.json.gz"), self.fernet)
                    
                    # Merge Appearance into Config
                    if pname in app_data: pdata.update(app_data[pname])
                    IOManager.write_json_gzip(pdata, str(p_dir / "config.json.gz"), self.fernet)
                    
                    # Move LTM, Training, Global Chat
                    import shutil
                    ltm_path = pathlib.Path(LEGACY_LTM_DIR) / user_id_str / f"{pname}.json.gz"
                    if ltm_path.exists(): shutil.move(str(ltm_path), str(p_dir / "ltm.json.gz"))
                    
                    train_path = pathlib.Path(LEGACY_TRAINING_DIR) / user_id_str / f"{pname}.json.gz"
                    if train_path.exists(): shutil.move(str(train_path), str(p_dir / "training.json.gz"))
                    
                    gc_path = pathlib.Path(LEGACY_GLOBAL_CHAT_DIR) / user_id_str / f"{pname}.json.gz"
                    if gc_path.exists(): shutil.move(str(gc_path), str(p_dir / "global_chat.json.gz"))
                    
                    # Move Child Bot Token
                    for bid, bconfig in bot_data.items():
                        if bconfig.get("profile_name") == pname:
                            bconfig["bot_id"] = bid
                            IOManager.write_json_gzip(bconfig, str(p_dir / "child_bot.json.gz"), encrypted=False)
                            break

                # 3. Split Borrowed Profiles
                for bname, bdata in user_data.get("borrowed_profiles", {}).items():
                    p_dir = user_target_dir / "profiles" / bname
                    p_dir.mkdir(parents=True, exist_ok=True)
                    IOManager.write_json_gzip(bdata, str(p_dir / "borrowed_config.json.gz"), self.fernet)
                    
                    gc_path = pathlib.Path(LEGACY_GLOBAL_CHAT_DIR) / user_id_str / f"{bname}.json.gz"
                    if gc_path.exists(): 
                        import shutil
                        shutil.move(str(gc_path), str(p_dir / "global_chat.json.gz"))

            # 4. Clean up legacy roots
            import shutil
            shutil.rmtree(str(legacy_profiles), ignore_errors=True)
            shutil.rmtree(self.APPEARANCES_DIR, ignore_errors=True)
            shutil.rmtree(LEGACY_LTM_DIR, ignore_errors=True)
            shutil.rmtree(LEGACY_TRAINING_DIR, ignore_errors=True)
            shutil.rmtree(LEGACY_CHILD_BOTS_DIR, ignore_errors=True)
            shutil.rmtree(LEGACY_GLOBAL_CHAT_DIR, ignore_errors=True)
            
            print("Phase 3 Migration complete. Profiles successfully split and consolidated into entity folders.")
        except Exception as e:
            print(f"Error during Phase 3 Migration: {e}")

    def _get_session_path(self, session_key: Any, session_type: str) -> pathlib.Path:
        if session_type == 'global_chat':
            _, user_id, profile_name = session_key
            pid = self._get_pid_from_name_any(user_id, profile_name)
            return pathlib.Path(self.USERS_DIR) / str(user_id) / "profiles" / pid / "global_chat.json.gz"
        
        dir_path = self._get_session_dir_path(session_key, session_type)
        # All other session types use a unified log
        return dir_path / "session_log.json.gz"
    
    def _save_session_to_disk(self, session_key: Any, session_type: str, session_data: Union[GoogleGenAIChatSession, List[Dict], Dict]):
        if not session_data:
            self._delete_session_from_disk(session_key, session_type)
            return
        
        data_to_save = session_data

        if session_type == 'global_chat':
            # If it's the hydrated dict {chat_session, unified_log}, extract the log
            if isinstance(session_data, dict) and 'unified_log' in session_data:
                if not session_data['unified_log']:
                    self._delete_session_from_disk(session_key, session_type)
                    return
                data_to_save = session_data['unified_log']
            elif hasattr(session_data, 'history'):
                 # Convert chat session to log format on save if needed (fallback)
                 log = []
                 for content in session_data.history:
                     parts_text = "".join(p if isinstance(p, str) else p.get('text', '') for p in content.get('parts', []))
                     log.append({
                         "turn_id": str(uuid.uuid4()), "role": content.get('role', 'user'), "content": parts_text,
                         "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                     })
                 data_to_save = log

        if hasattr(data_to_save, 'history') and not data_to_save.history:
             self._delete_session_from_disk(session_key, session_type)
             return

        try:
            path = self._get_session_path(session_key, session_type)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Always save as JSON list (unified log)
            serialized_bytes = json.dumps(data_to_save)

            compressed_bytes = gzip.compress(serialized_bytes)
            encrypted_compressed_bytes = self.fernet.encrypt(compressed_bytes)

            temp_path = path.with_suffix(path.suffix + '.tmp')
            with open(temp_path, 'wb') as f:
                f.write(encrypted_compressed_bytes)
            os.replace(temp_path, path)
        except Exception as e:
            print(f"Error saving session for key {session_key}: {e}")

    def _load_session_from_disk(self, session_key: Any, session_type: str) -> Optional[Union[GoogleGenAIChatSession, List[Dict], Dict]]:
        try:
            path = self._get_session_path(session_key, session_type)
            if not path.exists():
                return None
            
            with open(path, 'rb') as f:
                encrypted_compressed_bytes = f.read()

            decrypted_compressed_bytes = self.fernet.decrypt(encrypted_compressed_bytes)
            json_bytes = gzip.decompress(decrypted_compressed_bytes)
            
            if not json_bytes:
                _delete_file_shard(str(path))
                return None

            data = json.loads(json_bytes)

            if session_type == 'global_chat':
                if data and isinstance(data, list) and 'parts' in data[0]:
                    # Convert old format to new unified_log format
                    unified_log = []
                    for item in data:
                        role = item.get('role')
                        parts = item.get('parts', [])
                        content = "".join(p.get('text', '') for p in parts)
                        log_item = {
                            "turn_id": str(uuid.uuid4()),
                            "role": role,
                            "content": content,
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                        }
                        if item.get('thought_signature'):
                            log_item['thought_signature'] = item.get('thought_signature')
                        unified_log.append(log_item)
                    
                    history = []
                    for t in unified_log:
                        obj = {'role': t['role'], 'parts': [t['content']]}
                        if t.get('thought_signature'):
                            obj['thought_signature'] = t.get('thought_signature')
                        history.append(obj)
                    chat_session = GoogleGenAIChatSession(history=history)
                    
                    return {'chat_session': chat_session, 'unified_log': unified_log}
                
                elif data and isinstance(data, list) and 'turn_id' in data[0]:
                    history = []
                    for t in data:
                        obj = {'role': t['role'], 'parts': [t['content']]}
                        if t.get('thought_signature'):
                            obj['thought_signature'] = t.get('thought_signature')
                        history.append(obj)
                    chat_session = GoogleGenAIChatSession(history=history)
                    return {'chat_session': chat_session, 'unified_log': data}
                
                return None

            else: # Multi/Freewill use raw unified log
                return data
        except (gzip.BadGzipFile, json.JSONDecodeError, InvalidToken):
             print(f"Warning: Corrupted or old-format session file for key {session_key}. Deleting file.")
             self._delete_session_from_disk(session_key, session_type)
        except Exception as e:
            print(f"Error loading session for key {session_key}: {e}")
        return None

    def _delete_session_from_disk(self, session_key: Any, session_type: str):
        try:
            path = self._get_session_path(session_key, session_type)
            _delete_file_shard(str(path))
        except Exception as e:
            print(f"Error deleting session file for key {session_key}: {e}")

    def _get_mapping_path(self, mapping_key: Any) -> pathlib.Path:
        session_type, key_id = mapping_key
        if session_type == 'global_chat':
            user_id = key_id
            dummy_session_key = (None, user_id, None)
            dir_path = self._get_session_dir_path(dummy_session_key, session_type)
        elif session_type in ['multi', 'freewill']:
            channel_id = key_id
            dummy_session_key = (channel_id, None, None)
            dir_path = self._get_session_dir_path(dummy_session_key, session_type)
        else:
            raise ValueError(f"Unknown mapping key type: {session_type}")
        
        return dir_path / "_mappings.json.gz"

    def _load_mapping_from_disk(self, mapping_key: Any) -> Dict[str, Any]:
        try:
            path = self._get_mapping_path(mapping_key)
            if path.exists():
                data = self._load_json_gzip(str(path)) or {}
                return data
        except Exception as e:
            print(f"Error loading mapping for key {mapping_key}: {e}")
        return {}

    def _save_mapping_to_disk(self, mapping_key: Any, data: Dict[str, Any]):
        try:
            path = self._get_mapping_path(mapping_key)
            if not data:
                _delete_file_shard(str(path))
            else:
                self._atomic_json_save_gzip(data, str(path))
        except Exception as e:
            print(f"Error saving mapping for key {mapping_key}: {e}")

    def _load_user_appearances(self):
        self.user_appearances = {}
        if not os.path.isdir(self.USERS_DIR):
            return
        for user_id_str in os.listdir(self.USERS_DIR):
            if not user_id_str.isdigit(): continue
            index = self._get_user_index(int(user_id_str))
            all_profiles = list(index.get("personal", {})) + list(index.get("borrowed", {}))
            
            for profile_name in all_profiles:
                pid = self._get_pid_from_name_any(int(user_id_str), profile_name)
                config_path = os.path.join(self.USERS_DIR, user_id_str, "profiles", pid, "config.json.gz")
                borrowed_path = os.path.join(self.USERS_DIR, user_id_str, "profiles", pid, "borrowed_config.json.gz")
                
                target_path = None
                if os.path.exists(config_path):
                    target_path = config_path
                elif os.path.exists(borrowed_path):
                    target_path = borrowed_path
                    
                if target_path:
                    data = self._load_json_gzip(target_path)
                    if data and (data.get("custom_display_name") or data.get("custom_avatar_url")):
                        self.user_appearances.setdefault(user_id_str, {})[profile_name] = {
                            "custom_display_name": data.get("custom_display_name"),
                            "custom_avatar_url": data.get("custom_avatar_url")
                        }

    def _save_user_appearance_shard(self, user_id_str: str, data: Dict):
        pass # Deprecated in Phase 3. Handled directly via _save_profile_config.

    def _load_channel_webhooks(self):
        self.channel_webhooks = {}
        servers_dir = self.FREEWILL_SERVERS_DIR # This is cogs/data/servers
        if not os.path.isdir(servers_dir):
            return
        
        for server_id_str in os.listdir(servers_dir):
            server_path = os.path.join(servers_dir, server_id_str)
            if os.path.isdir(server_path):
                webhooks_file = os.path.join(server_path, "webhooks.json.gz")
                if os.path.exists(webhooks_file):
                    server_webhooks_data = self._load_json_gzip(webhooks_file)
                    if server_webhooks_data:
                        # The keys in the file are channel_ids as strings, need to convert to int
                        for ch_id_str, wh_data in server_webhooks_data.items():
                            try:
                                self.channel_webhooks[int(ch_id_str)] = wh_data
                            except ValueError:
                                print(f"Warning: Found non-integer channel ID '{ch_id_str}' in webhook file for server {server_id_str}")

    def _save_channel_webhooks(self):
        try:
            # Group webhooks by server_id
            server_grouped_webhooks = {}
            for channel_id, webhook_data in self.channel_webhooks.items():
                channel = self.bot.get_channel(channel_id)
                if channel and hasattr(channel, 'guild'):
                    server_id = channel.guild.id
                    if server_id not in server_grouped_webhooks:
                        server_grouped_webhooks[server_id] = {}
                    # Store channel_id as string for JSON compatibility
                    server_grouped_webhooks[server_id][str(channel_id)] = webhook_data

            # Save each server's webhooks to its own file
            servers_dir = self.FREEWILL_SERVERS_DIR
            for server_id, webhooks_for_server in server_grouped_webhooks.items():
                server_path = os.path.join(servers_dir, str(server_id))
                os.makedirs(server_path, exist_ok=True)
                file_path = os.path.join(server_path, "webhooks.json.gz")
                self._atomic_json_save_gzip(webhooks_for_server, file_path)
        except Exception as e:
            print(f"Error saving sharded channel webhook configurations: {e}"); traceback.print_exc()

    def _load_blacklist(self):
        try:
            if os.path.exists(BLACKLIST_FILE_PATH):
                with open(BLACKLIST_FILE_PATH, 'rb') as f:
                    user_ids = json.loads(f.read())
                    self.global_blacklist = set(user_ids)
            else:
                self.global_blacklist = set()
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading global blacklist: {e}")
            self.global_blacklist = set()

    def _save_blacklist(self):
        try:
            _atomic_json_save(list(self.global_blacklist), BLACKLIST_FILE_PATH)
        except Exception as e:
            print(f"Error saving global blacklist: {e}")

    def _load_public_profiles(self):
        self.public_profiles = {}
        index_path = os.path.join(self.PUBLIC_PROFILES_DIR, "index.json.gz")
        if os.path.exists(index_path):
            data = self._load_json_gzip(index_path)
            if data: self.public_profiles = data

    def _save_public_index(self):
        index_path = os.path.join(self.PUBLIC_PROFILES_DIR, "index.json.gz")
        self._atomic_json_save_gzip(self.public_profiles, index_path)

    def _load_child_bots(self):
        self.child_bots = {}
        if not os.path.isdir(self.USERS_DIR):
            return
        
        for user_id_str in os.listdir(self.USERS_DIR):
            if not user_id_str.isdigit(): continue
            profiles_dir = os.path.join(self.USERS_DIR, user_id_str, "profiles")
            if not os.path.isdir(profiles_dir): continue
            
            for pid_folder in os.listdir(profiles_dir):
                bot_file = os.path.join(profiles_dir, pid_folder, "child_bot.json.gz")
                if os.path.exists(bot_file):
                    bot_data = IOManager.read_json_gzip(bot_file, encrypted=False)
                    if bot_data and "bot_id" in bot_data:
                        # Dynamically resolve the profile name from the directory's name.txt
                        name_file = os.path.join(profiles_dir, pid_folder, "name.txt")
                        p_name = None
                        if os.path.exists(name_file):
                            with open(name_file, 'r', encoding='utf-8') as nf:
                                p_name = nf.read().strip()
                        
                        if p_name:
                            bot_data["owner_id"] = int(user_id_str)
                            bot_data["profile_name"] = p_name
                            bot_data["pid"] = pid_folder
                            self.child_bots[bot_data["bot_id"]] = bot_data

    # Helper methods deprecated by Phase 3 structure, retained as no-ops to prevent crashes
    def _get_user_child_bot_shard(self, owner_id: int) -> Dict[str, Any]:
        return {}
    def _get_all_child_bot_shards(self) -> Dict[str, Dict[str, Any]]:
        return {}
    def _save_user_child_bot_shard(self, owner_id: int, data: Dict[str, Any]):
        pass

    def _load_server_api_keys(self):
        self.server_api_keys = {}
        servers_dir = self.FREEWILL_SERVERS_DIR
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
                        self.server_api_keys[server_id_str] = server_keys_data.get("primary")

    def _save_server_api_key_shard(self, server_id_str: str, primary_key_data: Optional[Dict], submissions_data: List):
        server_path = os.path.join(self.FREEWILL_SERVERS_DIR, server_id_str)
        os.makedirs(server_path, exist_ok=True)
        file_path = os.path.join(server_path, "api_keys.json.gz")
        
        full_data = {
            "primary": primary_key_data,
            "submissions": submissions_data
        }
        
        IOManager.write_json_gzip(full_data, file_path, self.fernet)

    def _load_personal_api_keys(self):
        self.personal_api_keys = {}
        if not os.path.isdir(self.USERS_DIR):
            return
        for user_id_str in os.listdir(self.USERS_DIR):
            if not user_id_str.isdigit(): continue
            file_path = os.path.join(self.USERS_DIR, user_id_str, "keys.json.gz")
            if os.path.exists(file_path):
                data = IOManager.read_json_gzip(file_path, self.fernet)
                if data and isinstance(data, dict) and "key" in data:
                    self.personal_api_keys[user_id_str] = data["key"]

    def _save_personal_api_key_shard(self, user_id_str: str, encrypted_key: Optional[str]):
        file_path = os.path.join(self.USERS_DIR, user_id_str, "keys.json.gz")
        if not encrypted_key:
            _delete_file_shard(file_path)
        else:
            data_to_save = {"key": encrypted_key}
            IOManager.write_json_gzip(data_to_save, file_path, self.fernet)

    def _load_key_submissions(self):
        self.key_submissions = {}
        servers_dir = self.FREEWILL_SERVERS_DIR # This is cogs/data/servers
        if not os.path.isdir(servers_dir):
            return
        
        for server_id_str in os.listdir(servers_dir):
            server_path = os.path.join(servers_dir, server_id_str)
            if os.path.isdir(server_path) and server_id_str.isdigit():
                api_keys_file = os.path.join(server_path, "api_keys.json.gz")
                if os.path.exists(api_keys_file):
                    server_keys_data = self._load_json_gzip(api_keys_file)
                    if server_keys_data and server_keys_data.get("submissions"):
                        self.key_submissions[server_id_str] = server_keys_data.get("submissions")

    def _save_key_submissions_shard(self, server_id_str: str, submissions_data: List):
        primary_key_data = self.server_api_keys.get(server_id_str)
        self._save_server_api_key_shard(server_id_str, primary_key_data, submissions_data)

    def _load_profile_shares(self):
        self.profile_shares = {}
        if not os.path.isdir(self.USERS_DIR):
            return
        for user_id_str in os.listdir(self.USERS_DIR):
            if not user_id_str.isdigit(): continue
            file_path = os.path.join(self.USERS_DIR, user_id_str, "shares.json.gz")
            if os.path.exists(file_path):
                data = IOManager.read_json_gzip(file_path, self.fernet)
                if data:
                    self.profile_shares[user_id_str] = data

    def _save_profile_share_shard(self, recipient_id_str: str, data: List):
        file_path = os.path.join(self.USERS_DIR, recipient_id_str, "shares.json.gz")
        if not data:
            _delete_file_shard(file_path)
        else:
            IOManager.write_json_gzip(data, file_path, self.fernet)

    async def _load_multi_profile_sessions(self):
        await self.bot.wait_until_ready()
        
        servers_dir = self.FREEWILL_SERVERS_DIR # This is cogs/data/servers
        if not os.path.isdir(servers_dir):
            return

        for server_id_str in os.listdir(servers_dir):
            if not server_id_str.isdigit():
                continue
            
            server_path = os.path.join(servers_dir, server_id_str)
            sessions_file = os.path.join(server_path, "sessions.json.gz")

            if os.path.exists(sessions_file):
                try:
                    saved_sessions = self._load_json_gzip(sessions_file)
                    if not saved_sessions: continue

                    for ch_id_str, session_data in saved_sessions.items():
                        channel_id = int(ch_id_str)
                        channel = self.bot.get_channel(channel_id)
                        if not channel or not channel.guild: continue
                        
                        owner_id = session_data.get("owner_id")
                        profiles_data = session_data.get("profiles", [])

                        if not owner_id or not profiles_data:
                            continue
                        
                        if profiles_data and isinstance(profiles_data[0], list):
                            profiles_data = [{"owner_id": p[0], "profile_name": p[1], "method": "webhook"} for p in profiles_data]

                        chat_sessions = {}
                        for p_data in profiles_data:
                            p_key = (p_data['owner_id'], p_data['profile_name'])
                            chat_sessions[p_key] = None # Create placeholder, do not load yet

                        self.multi_profile_channels[channel_id] = {
                            "profiles": profiles_data,
                            "chat_sessions": chat_sessions,
                            "is_hydrated": False, 
                            "last_bot_message_id": None,
                            "owner_id": owner_id,
                            "is_running": False,
                            "task_queue": asyncio.Queue(),
                            "worker_task": None,
                            "turns_since_last_ltm": 0,
                            "session_prompt": session_data.get("session_prompt"),
                            "session_mode": session_data.get("session_mode", "sequential"),
                            "audio_mode": session_data.get("audio_mode", "text-only"),
                            "type": session_data.get("type"),
                            "freewill_mode": session_data.get("freewill_mode")
                        }
                except Exception as e:
                    print(f"Unexpected error reloading multi-profile sessions for server {server_id_str}: {e}")

    def _save_multi_profile_sessions(self):
        try:
            # Group sessions by server_id
            server_grouped_sessions = {}
            for channel_id, session_data in self.multi_profile_channels.items():
                channel = self.bot.get_channel(channel_id)
                if channel and hasattr(channel, 'guild'):
                    server_id = channel.guild.id
                    if server_id not in server_grouped_sessions:
                        server_grouped_sessions[server_id] = {}
                    
                    # Store only the blueprint, not the live state
                    server_grouped_sessions[server_id][str(channel_id)] = {
                        "owner_id": session_data["owner_id"],
                        "profiles": session_data["profiles"],
                        "session_prompt": session_data.get("session_prompt"),
                        "session_mode": session_data.get("session_mode", "sequential"),
                        "audio_mode": session_data.get("audio_mode", "text-only"),
                        "type": session_data.get("type", "multi"),
                        "freewill_mode": session_data.get("freewill_mode")
                    }

            # Save each server's sessions to its own file
            servers_dir = self.FREEWILL_SERVERS_DIR
            for server_id, sessions_for_server in server_grouped_sessions.items():
                server_path = os.path.join(servers_dir, str(server_id))
                os.makedirs(server_path, exist_ok=True)
                file_path = os.path.join(server_path, "sessions.json.gz")
                self._atomic_json_save_gzip(sessions_for_server, file_path)
        except Exception as e:
            print(f"Error saving sharded multi-profile sessions: {e}")

    def _load_freewill_config(self):
        self.freewill_config = {}
        servers_dir = self.FREEWILL_SERVERS_DIR # This is cogs/data/servers
        if not os.path.isdir(servers_dir):
            return

        for server_id_str in os.listdir(servers_dir):
            if not server_id_str.isdigit():
                continue
            
            server_path = os.path.join(servers_dir, server_id_str)
            settings_file = os.path.join(server_path, "settings.json.gz")

            if os.path.exists(settings_file):
                data = self._load_json_gzip(settings_file)
                if data:
                    # Extract only the freewill-related keys
                    fw_data = {
                        "enabled": data.get("freewill_enabled", False),
                        "living_channel_ids": data.get("freewill_living_channel_ids", []),
                        "lurking_channel_ids": data.get("freewill_lurking_channel_ids", []),
                        "event_chance": data.get("freewill_event_chance", "off"), # Legacy fallback
                        "event_cooldown": data.get("freewill_event_cooldown", 300), # Legacy fallback
                        "channel_settings": data.get("freewill_channel_settings", {}) # [ADDED]
                    }
                    self.freewill_config[server_id_str] = fw_data

    def _get_freewill_server_file_path(self, guild_id: int) -> str:
        return os.path.join(self.FREEWILL_SERVERS_DIR, f"{guild_id}.json.gz")

    def _load_freewill_participation(self):
        self.freewill_participation = {}
        if not os.path.isdir(self.FREEWILL_SERVERS_DIR):
            return
        for filename in os.listdir(self.FREEWILL_SERVERS_DIR):
            if filename.endswith(".json.gz"):
                guild_id_str = filename[:-len(".json.gz")]
                file_path = os.path.join(self.FREEWILL_SERVERS_DIR, filename)
                data = self._load_json_gzip(file_path)
                if data:
                    self.freewill_participation[guild_id_str] = data

    def _save_freewill_for_server(self, guild_id: int):
        guild_id_str = str(guild_id)
        server_data = self.freewill_participation.get(guild_id_str)
        file_path = self._get_freewill_server_file_path(guild_id)
        if not server_data:
            _delete_file_shard(file_path)
        else:
            self._atomic_json_save_gzip(server_data, file_path)

    def _load_channel_settings(self):
        # Initialize only Freewill settings
        servers_dir = self.FREEWILL_SERVERS_DIR
        if not os.path.isdir(servers_dir):
            return

        for server_id_str in os.listdir(servers_dir):
            try:
                if not server_id_str.isdigit():
                    continue
                server_id = int(server_id_str)
                server_path = os.path.join(servers_dir, server_id_str)
                if os.path.isdir(server_path):
                    settings_file = os.path.join(server_path, "settings.json.gz")
                    if os.path.exists(settings_file):
                        # Use existing load logic but don't populate removed dicts
                        pass
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not process settings for server directory '{server_id_str}': {e}")

    def _save_channel_settings(self):
        try:
            # Group settings by server ID (Only Freewill configs remain)
            all_server_data = {}
            # Use guild list as source of truth for saving configs that exist in memory
            for guild in self.bot.guilds:
                fw_config = self.freewill_config.get(str(guild.id), {})
                if fw_config:
                    settings_for_server = {
                        "freewill_enabled": fw_config.get("enabled", False),
                        "freewill_living_channel_ids": fw_config.get("living_channel_ids", []),
                        "freewill_lurking_channel_ids": fw_config.get("lurking_channel_ids", []),
                        "freewill_channel_settings": fw_config.get("channel_settings", {}),
                    }
                    all_server_data[str(guild.id)] = settings_for_server

            # Save each server's settings to its own file
            servers_dir = self.FREEWILL_SERVERS_DIR
            for server_id_str, settings_data in all_server_data.items():
                server_path = os.path.join(servers_dir, server_id_str)
                os.makedirs(server_path, exist_ok=True)
                file_path = os.path.join(server_path, "settings.json.gz")
                self._atomic_json_save_gzip(settings_data, file_path)

        except Exception as e:
            print(f"Error saving sharded channel settings: {e}"); traceback.print_exc()

    def _get_api_key_for_guild(self, guild_id: int, provider: str = "gemini") -> Optional[str]:
        if not self.fernet: return None
        guild_id_str = str(guild_id)
        now = time.time()
        
        # 1. Prioritize Primary Key
        primary_key_data = self.server_api_keys.get(guild_id_str)
        if primary_key_data and isinstance(primary_key_data, dict):
            field = 'openrouter_key' if provider == "openrouter" else 'key'
            encrypted_key = primary_key_data.get(field)
            if encrypted_key:
                try:
                    raw_key = self.fernet.decrypt(encrypted_key.encode()).decode()
                    # Check if this specific key is currently on rate-limit cooldown
                    if raw_key not in self.api_key_cooldowns or now > self.api_key_cooldowns[raw_key]:
                        return raw_key
                except Exception: pass 

        # 2. Failover to PAID Tier Keys in Pool
        if guild_id_str in self.key_submissions:
            pool_candidates = []
            for submission in self.key_submissions[guild_id_str]:
                if submission.get("status") == "active" and submission.get("tier") == "paid":
                    sub_provider = submission.get("provider", "gemini")
                    if sub_provider == provider:
                        try:
                            raw_pool_key = self.fernet.decrypt(submission["encrypted_key"].encode()).decode()
                            # Only add if not on cooldown
                            if raw_pool_key not in self.api_key_cooldowns or now > self.api_key_cooldowns[raw_pool_key]:
                                pool_candidates.append(raw_pool_key)
                        except Exception: pass
            
            if pool_candidates:
                return random.choice(pool_candidates)

        return None

    def _get_api_key_for_user(self, user_id: int, provider: str = "gemini") -> Optional[str]:
        if not self.fernet: return None
        user_id_str = str(user_id)

        file_path = os.path.join(self.USERS_DIR, user_id_str, "keys.json.gz")
        data = IOManager.read_json_gzip(file_path, self.fernet)
        
        if not data:
            return None

        encrypted_key = None
        if provider == "openrouter":
            encrypted_key = data.get("openrouter_key")
        else:
            encrypted_key = data.get("key")

        if not encrypted_key:
            return None

        try:
            return self.fernet.decrypt(encrypted_key.encode()).decode()
        except Exception:
            # Fallback if the key was saved in plain text or only value-level encrypted during migration
            return encrypted_key

    def _is_profile_public(self, user_id: int, profile_name: str) -> bool:
        user_id_str = str(user_id)
        for p_info in self.public_profiles.values():
            if str(p_info.get("owner_id")) == user_id_str and p_info.get("original_profile_name") == profile_name:
                return True
        return False

    def _get_user_profile_for_model(self, user_id: int, channel_id: int, profile_name_override: Optional[str] = None) -> Tuple[Dict[str, List[str]], str, bool, float, float, int, int, float, str, str]:
        active_profile_name = profile_name_override if profile_name_override else self._get_active_user_profile_name_for_channel(user_id, channel_id)
        owner_id = int(defaultConfig.DISCORD_OWNER_ID)

        if active_profile_name == DEFAULT_PROFILE_NAME and user_id != owner_id:
            owner_config = self._get_profile_config(owner_id, DEFAULT_PROFILE_NAME)
            if not owner_config:
                self._get_or_create_user_profile(owner_id, DEFAULT_PROFILE_NAME)
                owner_config = self._get_profile_config(owner_id, DEFAULT_PROFILE_NAME) or {}
            
            owner_prompts = self._get_profile_prompts(owner_id, DEFAULT_PROFILE_NAME) or {}

            persona = owner_prompts.get("persona", {})
            ai_instructions = owner_prompts.get("ai_instructions", "")
            grounding_enabled = owner_config.get("grounding_enabled", False)
            temperature = owner_config.get("temperature", defaultConfig.GEMINI_TEMPERATURE)
            top_p = owner_config.get("top_p", defaultConfig.GEMINI_TOP_P)
            top_k = owner_config.get("top_k", defaultConfig.GEMINI_TOP_K)
            training_context_size = owner_config.get("training_context_size", defaultConfig.TRAINING_CONTEXT_SIZE)
            training_relevance_threshold = owner_config.get("training_relevance_threshold", defaultConfig.TRAINING_RELEVANCE_THRESHOLD)
            primary_model = owner_config.get("primary_model", PRIMARY_MODEL_NAME)
            fallback_model = owner_config.get("fallback_model", FALLBACK_MODEL_NAME)
            
            return (persona, ai_instructions, grounding_enabled, float(temperature), float(top_p), int(top_k), int(training_context_size), float(training_relevance_threshold), primary_model, fallback_model)

        index = self._get_user_index(user_id)
        is_borrowed = active_profile_name in index.get("borrowed", [])
        
        config = self._get_profile_config(user_id, active_profile_name, is_borrowed)
        if not config:
            return {}, "", False, defaultConfig.GEMINI_TEMPERATURE, defaultConfig.GEMINI_TOP_P, defaultConfig.GEMINI_TOP_K, defaultConfig.TRAINING_CONTEXT_SIZE, defaultConfig.TRAINING_RELEVANCE_THRESHOLD, PRIMARY_MODEL_NAME, FALLBACK_MODEL_NAME

        if is_borrowed:
            source_owner_id = int(config.get("original_owner_id", owner_id))
            source_profile_name = config.get("original_profile_name", DEFAULT_PROFILE_NAME)
            source_config = self._get_profile_config(source_owner_id, source_profile_name, False) or {}
            prompts = self._get_profile_prompts(source_owner_id, source_profile_name) or {}
        else:
            source_config = config
            prompts = self._get_profile_prompts(user_id, active_profile_name) or {}
            
        persona = prompts.get("persona", {})
        ai_instructions = prompts.get("ai_instructions", "")
        
        # Local overrides first, then source, then default
        training_context_size = config.get("training_context_size", source_config.get("training_context_size", defaultConfig.TRAINING_CONTEXT_SIZE))
        training_relevance_threshold = config.get("training_relevance_threshold", source_config.get("training_relevance_threshold", defaultConfig.TRAINING_RELEVANCE_THRESHOLD))
        temperature = config.get("temperature", source_config.get("temperature", defaultConfig.GEMINI_TEMPERATURE))
        top_p = config.get("top_p", source_config.get("top_p", defaultConfig.GEMINI_TOP_P))
        top_k = config.get("top_k", source_config.get("top_k", defaultConfig.GEMINI_TOP_K))
        primary_model = config.get("primary_model", source_config.get("primary_model", PRIMARY_MODEL_NAME))
        fallback_model = config.get("fallback_model", source_config.get("fallback_model", FALLBACK_MODEL_NAME))
        grounding_enabled = config.get("grounding_enabled", source_config.get("grounding_enabled", False))

        return (persona, ai_instructions, grounding_enabled, float(temperature), float(top_p), int(top_k), int(training_context_size), float(training_relevance_threshold), primary_model, fallback_model)
    
    def _get_or_create_user_profile(self, user_id: int, profile_name: str) -> Optional[Dict[str, Any]]:
        profile_name = profile_name.lower().strip()
        owner_id = int(defaultConfig.DISCORD_OWNER_ID)

        if profile_name == DEFAULT_PROFILE_NAME and user_id != owner_id:
            return None

        index = self._get_user_index(user_id)
        
        if profile_name not in index.get("personal", []):
            if len(index.get("personal", [])) >= MAX_USER_PROFILES and profile_name != DEFAULT_PROFILE_NAME:
                return None 
            
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
                "time_tracking_enabled": True, "timezone": "UTC", "generation_metadata_enabled": False,
                "realistic_typing_enabled": False, "ltm_creation_enabled": False,
                "image_generation_enabled": False, "image_generation_model": "gemini-2.5-flash-image",
                "url_fetching_enabled": False, "response_mode": "regular", "thinking_summary_visible": "off",
                "thinking_level": "high", "thinking_budget": -1, "thinking_signatures_enabled": "off",
                "error_response": "An error has occurred.", "speech_voice": "Aoede",
                "speech_model": "gemini-2.5-flash-preview-tts", "speech_temperature": 1.0,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            
            prompts = {
                "persona": {}, "ai_instructions": ["", "", "", ""], "image_generation_prompt": None,
                "ltm_summarization_instructions": self._encrypt_data(DEFAULT_LTM_SUMMARIZATION_INSTRUCTIONS)
            }
            
            self._save_profile_config(user_id, profile_name, config)
            self._save_profile_prompts(user_id, profile_name, prompts)
            return {"config": config, "prompts": prompts}
            
        return {"config": self._get_profile_config(user_id, profile_name), "prompts": self._get_profile_prompts(user_id, profile_name)}
    
    def _get_active_user_profile_name_for_channel(self, user_id: int, channel_id: int) -> str:
        index = self._get_user_index(user_id)
        return index.get("channel_active_profiles", {}).get(str(channel_id), DEFAULT_PROFILE_NAME)

    def _get_active_user_profile_data(self, user_id: int, channel_id: int) -> Optional[Dict[str, Any]]:
        active_profile_name = self._get_active_user_profile_name_for_channel(user_id, channel_id)
        index = self._get_user_index(user_id)
        
        is_borrowed = active_profile_name in index.get("borrowed", [])
        config = self._get_profile_config(user_id, active_profile_name, is_borrowed)
        
        if not config and active_profile_name != DEFAULT_PROFILE_NAME: 
            config = self._get_profile_config(user_id, DEFAULT_PROFILE_NAME, False)
        
        if config: 
            config.setdefault("grounding_enabled", False)
            config.setdefault("temperature", defaultConfig.GEMINI_TEMPERATURE)
            config.setdefault("top_p", defaultConfig.GEMINI_TOP_P)
            config.setdefault("top_k", defaultConfig.GEMINI_TOP_K)
            config.setdefault("training_context_size", defaultConfig.TRAINING_CONTEXT_SIZE)
            config.setdefault("training_relevance_threshold", defaultConfig.TRAINING_RELEVANCE_THRESHOLD)
            config.setdefault("primary_model", PRIMARY_MODEL_NAME)
            config.setdefault("fallback_model", FALLBACK_MODEL_NAME)
        return config

    def _get_user_index(self, user_id: int) -> Dict[str, Any]:
        user_id_str = str(user_id)
        if user_id_str in self.user_indices: return self.user_indices[user_id_str]
        
        path = os.path.join(self.USERS_DIR, user_id_str, "index.json")
        index = IOManager.read_json(path)
        
        if not index:
            index = {"personal": {}, "borrowed": {}, "channel_active_profiles": {}}
            owner_id = int(defaultConfig.DISCORD_OWNER_ID)
            
            if user_id != owner_id:
                index["borrowed"][DEFAULT_PROFILE_NAME] = DEFAULT_PROFILE_NAME
                self._save_user_index(user_id, index)
                
                b_config = self._get_profile_config(user_id, DEFAULT_PROFILE_NAME, True)
                if not b_config:
                    owner_config = self._get_profile_config(owner_id, DEFAULT_PROFILE_NAME, False) or {}
                    self._save_profile_config(user_id, DEFAULT_PROFILE_NAME, {
                        "original_owner_id": str(owner_id),
                        "original_profile_name": DEFAULT_PROFILE_NAME,
                        "borrowed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "grounding_enabled": owner_config.get("grounding_enabled", False),
                        "realistic_typing_enabled": owner_config.get("realistic_typing_enabled", False),
                        "time_tracking_enabled": owner_config.get("time_tracking_enabled", False),
                        "timezone": owner_config.get("timezone", "UTC")
                    }, is_borrowed=True)
                    
        self.user_indices[user_id_str] = index
        return index

    def _save_user_index(self, user_id: int, data: Dict[str, Any]):
        user_id_str = str(user_id)
        path = os.path.join(self.USERS_DIR, user_id_str, "index.json")
        IOManager.write_json(data, path)
        self.user_indices[user_id_str] = data

    def _get_profile_config(self, user_id: int, profile_name: str, is_borrowed: bool = False) -> Optional[Dict[str, Any]]:
        cache_key = f"{user_id}:{profile_name}:{'b' if is_borrowed else 'p'}"
        if cache_key in self.profile_configs: return self.profile_configs[cache_key]
        
        pid = self._get_pid_from_name(user_id, profile_name, is_borrowed)
        filename = "borrowed_config.json.gz" if is_borrowed else "config.json.gz"
        path = os.path.join(self.USERS_DIR, str(user_id), "profiles", pid, filename)
        data = IOManager.read_json_gzip(path, self.fernet)
        
        if data is not None:
            if not is_borrowed and "profile_id" not in data:
                import uuid
                data["profile_id"] = str(uuid.uuid4().hex[:8].upper())
                self._save_profile_config(user_id, profile_name, data, False)
            self.profile_configs[cache_key] = data
        return data

    def _save_profile_config(self, user_id: int, profile_name: str, data: Dict[str, Any], is_borrowed: bool = False):
        cache_key = f"{user_id}:{profile_name}:{'b' if is_borrowed else 'p'}"
        pid = self._get_pid_from_name(user_id, profile_name, is_borrowed)
        filename = "borrowed_config.json.gz" if is_borrowed else "config.json.gz"
        path = os.path.join(self.USERS_DIR, str(user_id), "profiles", pid, filename)
        IOManager.write_json_gzip(data, path, self.fernet)
        self.profile_configs[cache_key] = data

    def _get_profile_prompts(self, user_id: int, profile_name: str) -> Optional[Dict[str, Any]]:
        cache_key = f"{user_id}:{profile_name}"
        if cache_key in self.profile_prompts: return self.profile_prompts[cache_key]
        
        pid = self._get_pid_from_name_any(user_id, profile_name)
        path = os.path.join(self.USERS_DIR, str(user_id), "profiles", pid, "prompts.json.gz")
        data = IOManager.read_json_gzip(path, self.fernet)
        
        if data is not None:
            self.profile_prompts[cache_key] = data
        return data

    def _save_profile_prompts(self, user_id: int, profile_name: str, data: Dict[str, Any]):
        cache_key = f"{user_id}:{profile_name}"
        pid = self._get_pid_from_name_any(user_id, profile_name)
        path = os.path.join(self.USERS_DIR, str(user_id), "profiles", pid, "prompts.json.gz")
        IOManager.write_json_gzip(data, path, self.fernet)
        self.profile_prompts[cache_key] = data

    def _get_or_create_user_profile(self, user_id: int, profile_name: str) -> Optional[Dict[str, Any]]:
        profile_name = profile_name.lower().strip()
        owner_id = int(defaultConfig.DISCORD_OWNER_ID)

        if profile_name == DEFAULT_PROFILE_NAME and user_id != owner_id:
            return None

        index = self._get_user_index(user_id)
        
        if profile_name not in index.get("personal", []):
            if len(index.get("personal", [])) >= MAX_USER_PROFILES and profile_name != DEFAULT_PROFILE_NAME:
                return None 
            
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
                "time_tracking_enabled": True, "timezone": "UTC", "generation_metadata_enabled": False,
                "realistic_typing_enabled": False, "ltm_creation_enabled": False,
                "image_generation_enabled": False, "image_generation_model": "gemini-2.5-flash-image",
                "url_fetching_enabled": False, "response_mode": "regular", "thinking_summary_visible": "off",
                "thinking_level": "high", "thinking_budget": -1, "thinking_signatures_enabled": "off",
                "error_response": "An error has occurred.", "speech_voice": "Aoede",
                "speech_model": "gemini-2.5-flash-preview-tts", "speech_temperature": 1.0,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            
            prompts = {
                "persona": {}, "ai_instructions": ["", "", "", ""], "image_generation_prompt": None,
                "ltm_summarization_instructions": self._encrypt_data(DEFAULT_LTM_SUMMARIZATION_INSTRUCTIONS)
            }
            
            self._save_profile_config(user_id, profile_name, config)
            self._save_profile_prompts(user_id, profile_name, prompts)
            return {"config": config, "prompts": prompts}
            
        return {"config": self._get_profile_config(user_id, profile_name), "prompts": self._get_profile_prompts(user_id, profile_name)}
    
    def _get_active_user_profile_name_for_channel(self, user_id: int, channel_id: int) -> str:
        index = self._get_user_index(user_id)
        return index.get("channel_active_profiles", {}).get(str(channel_id), DEFAULT_PROFILE_NAME)

    async def _set_active_user_profile_for_channel(self, user_id: int, channel_id: int, profile_name: str, interaction_for_feedback: Optional[discord.Interaction] = None) -> bool:
        index = self._get_user_index(user_id)
        profile_name = profile_name.lower().strip()
        is_borrowed = profile_name in index.get("borrowed", [])

        if not is_borrowed and profile_name != DEFAULT_PROFILE_NAME and profile_name not in index.get("personal", []): 
            if interaction_for_feedback:
                await interaction_for_feedback.followup.send(f"Your profile '{profile_name}' not found. Cannot activate.", ephemeral=True)
            return False
        
        old_profile_name = self._get_active_user_profile_name_for_channel(user_id, channel_id)
        index.setdefault("channel_active_profiles", {})[str(channel_id)] = profile_name
        self._save_user_index(user_id, index)
        
        if interaction_for_feedback and interaction_for_feedback.guild:
            warning_key_to_clear = (user_id, interaction_for_feedback.guild.id, old_profile_name)
            self.model_override_warnings_sent.discard(warning_key_to_clear)

        model_cache_key = (channel_id, user_id)

        if model_cache_key in self.channel_models: del self.channel_models[model_cache_key]
        if model_cache_key in self.chat_sessions: self.chat_sessions.pop(model_cache_key, None)
        self.channel_model_last_profile_key.pop(model_cache_key, None) 
        
        if interaction_for_feedback:
            channel_mention = f"<#{channel_id}>" if interaction_for_feedback.guild else "this DM"
            
            embed_title = f"Your Profile Preference Swapped to: '{profile_name}'"
            embed_desc = f"Your individual preferred profile in {channel_mention} is now '{profile_name}'."
            if profile_name == DEFAULT_PROFILE_NAME:
                embed_title = "Your Profile Preference Reverted to Default"
                embed_desc = f"Your individual preferred profile in {channel_mention} has been reverted to your '{DEFAULT_PROFILE_NAME}' profile."

            embed = discord.Embed(title=embed_title, description=embed_desc, color=discord.Color.green())
            
            effective_owner_id = user_id
            effective_profile_name = profile_name
            if is_borrowed:
                b_config = self._get_profile_config(user_id, profile_name, True)
                if b_config:
                    effective_owner_id = int(b_config.get("original_owner_id", user_id))
                    effective_profile_name = b_config.get("original_profile_name", profile_name)
            
            active_appearance = None
            owner_config = self._get_profile_config(effective_owner_id, effective_profile_name)
            if owner_config and (owner_config.get("custom_display_name") or owner_config.get("custom_avatar_url")):
                active_appearance = owner_config

            app_name = self.bot.user.name if self.bot.user else "Bot"
            app_avatar_url = self.bot.user.display_avatar.url if self.bot.user else None

            if active_appearance:
                app_name = active_appearance.get("custom_display_name") or app_name
                app_avatar_url = active_appearance.get("custom_avatar_url") or app_avatar_url
            
            embed.add_field(name="Linked Appearance", value=f"Name: {app_name}", inline=False)
            if app_avatar_url:
                embed.set_thumbnail(url=app_avatar_url)
            
            await interaction_for_feedback.followup.send(embed=embed, ephemeral=True)
        return True

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
                    import shutil
                    p_dir = os.path.join(self.USERS_DIR, str(user_id), "profiles", pid)
                    shutil.rmtree(p_dir, ignore_errors=True)
            else:
                index["borrowed"] = [b for b in borrowed if b not in ids_to_remove]
                for local_name in ids_to_remove:
                    import shutil
                    p_dir = os.path.join(self.USERS_DIR, str(user_id), "profiles", local_name)
                    shutil.rmtree(p_dir, ignore_errors=True)
                
            self._save_user_index(user_id, index)
            removed_count = len(ids_to_remove)
            
        return removed_count

    def _cascade_delete_borrowed_profiles(self, original_owner_id: int, deleted_pid: str, original_profile_name: str):
        """Instantly removes all borrowed variants linked to a deleted personal profile across the entire system."""
        owner_str = str(original_owner_id)
        if not os.path.isdir(self.USERS_DIR): return
        
        for user_id_str in os.listdir(self.USERS_DIR):
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
                            import shutil
                            p_dir = os.path.join(self.USERS_DIR, user_id_str, "profiles", pid)
                            shutil.rmtree(p_dir, ignore_errors=True)
                    else:
                        index["borrowed"] = [b for b in borrowed if b not in to_delete]
                        for b_name in to_delete:
                            import shutil
                            p_dir = os.path.join(self.USERS_DIR, user_id_str, "profiles", b_name)
                            shutil.rmtree(p_dir, ignore_errors=True)
                    self._save_user_index(uid, index)
            except Exception as e:
                print(f"Error in cascade delete for user {user_id_str}: {e}")
    
    def _get_mapping_key_for_session(self, session_key: Any, session_type: str) -> Any:
        if session_type == 'global_chat':
            _, user_id, _ = session_key
            return ('global_chat', user_id)
        
        # All other session types are keyed by channel_id
        elif session_type in ['multi', 'freewill']:
            channel_id, _, _ = session_key
            return (session_type, channel_id)
        return None

    def _save_key_submissions(self):
        # This method now saves ALL shards, used for cleanup/migration.
        # For single-server updates, _save_key_submissions_shard is used.
        try:
            for server_id_str, submissions_data in self.key_submissions.items():
                self._save_key_submissions_shard(server_id_str, submissions_data)
        except Exception as e:
            print(f"Error saving all key submission shards: {e}")