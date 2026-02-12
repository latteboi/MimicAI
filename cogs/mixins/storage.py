import os
import gzip
import pathlib
import shutil
import discord
import uuid
import datetime
import orjson as json
from typing import Dict, List, Any, Optional, Literal, Union, Tuple
from cryptography.fernet import Fernet, InvalidToken
import google.generativeai as genai
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

class StorageMixin:

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
        temp_file_path = file_path + ".tmp"
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            json_bytes = json.dumps(data, option=json.OPT_INDENT_2)
            compressed_bytes = gzip.compress(json_bytes)
            
            bytes_to_write = compressed_bytes
            if encrypted and self.fernet:
                bytes_to_write = self.fernet.encrypt(compressed_bytes)

            with open(temp_file_path, 'wb') as f:
                f.write(bytes_to_write)
            os.replace(temp_file_path, file_path)
        except Exception as e:
            print(f"Error saving gzipped data to {file_path}: {e}")
            if os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except OSError: pass
            raise

    def _load_json_gzip(self, file_path: str, encrypted: bool = True) -> Optional[Any]:
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, 'rb') as f:
                file_bytes = f.read()

            if encrypted and self.fernet:
                decrypted_bytes = self.fernet.decrypt(file_bytes)
                decompressed_bytes = gzip.decompress(decrypted_bytes)
            else: # Not encrypted, just compressed
                decompressed_bytes = gzip.decompress(file_bytes)
            
            return json.loads(decompressed_bytes)
        except (IOError, json.JSONDecodeError, gzip.BadGzipFile, InvalidToken) as e:
            print(f"Error loading gzipped data from {file_path}: {e}")
            return None
        
    def _load_ltm_shard(self, user_id: str, profile_name: str) -> Optional[Dict[str, List[Dict]]]:
        file_path = os.path.join(self.LTM_DIR, user_id, f"{profile_name}.json.gz")
        return self._load_json_gzip(file_path)

    def _save_ltm_shard(self, user_id: str, profile_name: str, data: Dict[str, List[Dict]]):
        file_path = os.path.join(self.LTM_DIR, user_id, f"{profile_name}.json.gz")
        self._atomic_json_save_gzip(data, file_path)

    def _delete_ltm_shard(self, user_id: str, profile_name: str):
        file_path = os.path.join(self.LTM_DIR, user_id, f"{profile_name}.json.gz")
        _delete_file_shard(file_path)

    def _rename_ltm_shards(self, user_id: str, old_profile_name: str, new_profile_name: str):
        old_path = os.path.join(self.LTM_DIR, user_id, f"{old_profile_name}.json.gz")
        if os.path.exists(old_path):
            new_path = os.path.join(self.LTM_DIR, user_id, f"{new_profile_name}.json.gz")
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            os.rename(old_path, new_path)

    def _copy_ltm_shard(self, user_id: str, source_profile_name: str, new_profile_name: str):
        source_path = os.path.join(self.LTM_DIR, user_id, f"{source_profile_name}.json.gz")
        if os.path.exists(source_path):
            new_path = os.path.join(self.LTM_DIR, user_id, f"{new_profile_name}.json.gz")
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            import shutil
            shutil.copy2(source_path, new_path)

    def _add_ltm(self, profile_owner_id: int, profile_name: str, summary: str, summary_embedding: List[float], guild_id: Optional[int], triggering_user_id: int, user_dn: Optional[str] = None, force_user_scope: bool = False):
        owner_id_str = str(profile_owner_id)
        
        user_data = self._get_user_data_entry(profile_owner_id)
        is_borrowed = profile_name in user_data.get("borrowed_profiles", {})
        profile_settings = user_data.get("borrowed_profiles" if is_borrowed else "profiles", {}).get(profile_name, {})
        
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
        file_path = os.path.join(self.TRAINING_DIR, user_id, f"{profile_name}.json.gz")
        return self._load_json_gzip(file_path)

    def _save_training_shard(self, user_id: str, profile_name: str, data: List[Dict]):
        file_path = os.path.join(self.TRAINING_DIR, user_id, f"{profile_name}.json.gz")
        self._atomic_json_save_gzip(data, file_path)

    def _delete_training_shard(self, user_id: str, profile_name: str):
        file_path = os.path.join(self.TRAINING_DIR, user_id, f"{profile_name}.json.gz")
        _delete_file_shard(file_path)

    def _rename_training_shards(self, user_id: str, old_profile_name: str, new_profile_name: str):
        old_path = os.path.join(self.TRAINING_DIR, user_id, f"{old_profile_name}.json.gz")
        if os.path.exists(old_path):
            new_path = os.path.join(self.TRAINING_DIR, user_id, f"{new_profile_name}.json.gz")
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            os.rename(old_path, new_path)

    def _copy_training_shard(self, user_id: str, source_profile_name: str, new_profile_name: str):
        source_path = os.path.join(self.TRAINING_DIR, user_id, f"{source_profile_name}.json.gz")
        if os.path.exists(source_path):
            new_path = os.path.join(self.TRAINING_DIR, user_id, f"{new_profile_name}.json.gz")
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            import shutil
            shutil.copy2(source_path, new_path)

    def _get_session_dir_path(self, session_key: Any, session_type: str) -> pathlib.Path:
        if session_type == 'global_chat':
            _, user_id, _ = session_key
            return pathlib.Path(SESSIONS_GLOBAL_DIR) / str(user_id)
        
        # All other session types are now guild-based and use the same structure
        channel_id, _, _ = session_key
        channel = self.bot.get_channel(channel_id)
        server_id = channel.guild.id if channel and channel.guild else "dm"
        return pathlib.Path(SESSIONS_SERVERS_DIR) / str(server_id) / str(channel_id) / session_type

    def _get_session_path(self, session_key: Any, session_type: str) -> pathlib.Path:
        dir_path = self._get_session_dir_path(session_key, session_type)
        if session_type == 'global_chat':
            _, _, profile_name = session_key
            return dir_path / f"{profile_name}.json.gz"
        
        # All other session types use a unified log
        return dir_path / "session_log.json.gz"
    
    def _save_session_to_disk(self, session_key: Any, session_type: str, session_data: Union[genai.ChatSession, List[Dict], Dict]):
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
            # Fallback for potential legacy/transition states (shouldn't happen with new load logic)
            elif isinstance(session_data, genai.ChatSession):
                 # Convert chat session to log format on save if needed (fallback)
                 log = []
                 for content in session_data.history:
                     parts_text = "".join(p.text for p in content.parts if hasattr(p, 'text'))
                     log.append({
                         "turn_id": str(uuid.uuid4()), "role": content.role, "content": parts_text,
                         "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                     })
                 data_to_save = log

        if isinstance(data_to_save, genai.ChatSession) and not data_to_save.history:
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

    def _load_session_from_disk(self, session_key: Any, session_type: str, model: genai.GenerativeModel) -> Optional[Union[genai.ChatSession, List[Dict], Dict]]:
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
                # Migration Logic: Old format is list of ChatSession parts (list of dicts with 'role' and 'parts')
                # New format is list of turn objects (dicts with 'turn_id', 'role', 'content', 'timestamp')
                if data and isinstance(data, list) and 'parts' in data[0]:
                    # Convert old format to new unified_log format
                    unified_log = []
                    for item in data:
                        role = item.get('role')
                        parts = item.get('parts', [])
                        content = "".join(p.get('text', '') for p in parts)
                        unified_log.append({
                            "turn_id": str(uuid.uuid4()),
                            "role": role,
                            "content": content,
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat() # Approx time
                        })
                    
                    # Hydrate ChatSession from new log
                    from google.generativeai.types import content_types
                    history = [content_types.to_content({'role': t['role'], 'parts': [t['content']]}) for t in unified_log]
                    chat_session = model.start_chat(history=history)
                    
                    return {'chat_session': chat_session, 'unified_log': unified_log}
                
                # New format is just the unified_log list
                elif data and isinstance(data, list) and 'turn_id' in data[0]:
                    from google.generativeai.types import content_types
                    history = [content_types.to_content({'role': t['role'], 'parts': [t['content']]}) for t in data]
                    chat_session = model.start_chat(history=history)
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
        if not os.path.isdir(self.APPEARANCES_DIR):
            return
        for filename in os.listdir(self.APPEARANCES_DIR):
            if filename.endswith(".json.gz"):
                user_id_str = filename[:-len(".json.gz")]
                file_path = os.path.join(self.APPEARANCES_DIR, filename)
                data = self._load_json_gzip(file_path)
                if data:
                    self.user_appearances[user_id_str] = data

    def _save_user_appearance_shard(self, user_id_str: str, data: Dict):
        file_path = os.path.join(self.APPEARANCES_DIR, f"{user_id_str}.json.gz")
        if not data:
            _delete_file_shard(file_path)
        else:
            self._atomic_json_save_gzip(data, file_path)

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
        if not os.path.isdir(self.CHILD_BOTS_DIR):
            return
        for filename in os.listdir(self.CHILD_BOTS_DIR):
            if filename.endswith(".json.gz"):
                owner_id = filename[:-len(".json.gz")]
                file_path = os.path.join(self.CHILD_BOTS_DIR, filename)
                user_bot_data = self._load_json_gzip(file_path, encrypted=False)
                if user_bot_data:
                    for bot_id, bot_config in user_bot_data.items():
                        # Add owner_id for easier access later
                        bot_config['owner_id'] = int(owner_id)
                        self.child_bots[bot_id] = bot_config

    def _get_user_child_bot_shard(self, owner_id: int) -> Dict[str, Any]:
        file_path = os.path.join(self.CHILD_BOTS_DIR, f"{owner_id}.json.gz")
        return self._load_json_gzip(file_path, encrypted=False) or {}

    def _get_all_child_bot_shards(self) -> Dict[str, Dict[str, Any]]:
        all_shards = {}
        if not os.path.isdir(self.CHILD_BOTS_DIR):
            return {}
        for filename in os.listdir(self.CHILD_BOTS_DIR):
            if filename.endswith(".json.gz"):
                owner_id_str = filename[:-len(".json.gz")]
                file_path = os.path.join(self.CHILD_BOTS_DIR, filename)
                user_bot_data = self._load_json_gzip(file_path, encrypted=False)
                if user_bot_data:
                    all_shards[owner_id_str] = user_bot_data
        return all_shards

    def _save_user_child_bot_shard(self, owner_id: int, data: Dict[str, Any]):
        file_path = os.path.join(self.CHILD_BOTS_DIR, f"{owner_id}.json.gz")
        if not data:
            _delete_file_shard(file_path)
        else:
            self._atomic_json_save_gzip(data, file_path, encrypted=False)

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
        
        self._atomic_json_save_gzip(full_data, file_path)

    def _load_personal_api_keys(self):
        self.personal_api_keys = {}
        if not os.path.isdir(self.PERSONAL_KEYS_DIR):
            return
        for filename in os.listdir(self.PERSONAL_KEYS_DIR):
            if filename.endswith(".json.gz"):
                user_id_str = filename[:-len(".json.gz")]
                file_path = os.path.join(self.PERSONAL_KEYS_DIR, filename)
                data = self._load_json_gzip(file_path)
                if data and isinstance(data, dict) and "key" in data:
                    self.personal_api_keys[user_id_str] = data["key"]

    def _save_personal_api_key_shard(self, user_id_str: str, encrypted_key: Optional[str]):
        file_path = os.path.join(self.PERSONAL_KEYS_DIR, f"{user_id_str}.json.gz")
        if not encrypted_key:
            _delete_file_shard(file_path)
        else:
            data_to_save = {"key": encrypted_key}
            self._atomic_json_save_gzip(data_to_save, file_path)

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
        if not os.path.isdir(self.SHARES_DIR):
            return
        for filename in os.listdir(self.SHARES_DIR):
            if filename.endswith(".json.gz"):
                recipient_id_str = filename[:-len(".json.gz")]
                file_path = os.path.join(self.SHARES_DIR, filename)
                data = self._load_json_gzip(file_path)
                if data:
                    self.profile_shares[recipient_id_str] = data

    def _save_profile_share_shard(self, recipient_id_str: str, data: List):
        file_path = os.path.join(self.SHARES_DIR, f"{recipient_id_str}.json.gz")
        if not data:
            _delete_file_shard(file_path)
        else:
            self._atomic_json_save_gzip(data, file_path)

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

        file_path = os.path.join(self.PERSONAL_KEYS_DIR, f"{user_id_str}.json.gz")
        data = self._load_json_gzip(file_path)
        
        if not data:
            # Backward compatibility check for old single-string file format (though _load_json_gzip might fail on string)
            # Assuming migration or clean start. If it returns dict:
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
        except Exception as e:
            print(f"Failed to decrypt a personal API key for user {user_id}: {e}")
            return None

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
            owner_user_data = self._get_user_data_entry(owner_id)
            profile_data = owner_user_data.get("profiles", {}).get(DEFAULT_PROFILE_NAME)
            if not profile_data:
                self._get_or_create_user_profile(owner_id, DEFAULT_PROFILE_NAME)
                profile_data = owner_user_data.get("profiles", {}).get(DEFAULT_PROFILE_NAME, {})

            persona = profile_data.get("persona", {})
            ai_instructions = profile_data.get("ai_instructions", "")
            grounding_enabled = profile_data.get("grounding_enabled", False)
            temperature = profile_data.get("temperature", defaultConfig.GEMINI_TEMPERATURE)
            top_p = profile_data.get("top_p", defaultConfig.GEMINI_TOP_P)
            top_k = profile_data.get("top_k", defaultConfig.GEMINI_TOP_K)
            training_context_size = profile_data.get("training_context_size", defaultConfig.TRAINING_CONTEXT_SIZE)
            training_relevance_threshold = profile_data.get("training_relevance_threshold", defaultConfig.TRAINING_RELEVANCE_THRESHOLD)
            primary_model = profile_data.get("primary_model", PRIMARY_MODEL_NAME)
            fallback_model = profile_data.get("fallback_model", FALLBACK_MODEL_NAME)
            
            return (persona, ai_instructions, grounding_enabled, float(temperature), float(top_p), int(top_k), int(training_context_size), float(training_relevance_threshold), primary_model, fallback_model)

        user_data = self._get_user_data_entry(user_id)
        is_borrowed = active_profile_name in user_data.get("borrowed_profiles", {})
        
        if is_borrowed:
            borrowed_data = user_data["borrowed_profiles"][active_profile_name]
            source_owner_id = int(borrowed_data["original_owner_id"])
            source_profile_name = borrowed_data["original_profile_name"]
            source_owner_user_data = self._get_user_data_entry(source_owner_id)
            owner_profile_data = source_owner_user_data.get("profiles", {}).get(source_profile_name, {})
            
            persona = owner_profile_data.get("persona", {})
            ai_instructions = owner_profile_data.get("ai_instructions", "")
            training_context_size = owner_profile_data.get("training_context_size", defaultConfig.TRAINING_CONTEXT_SIZE)
            training_relevance_threshold = owner_profile_data.get("training_relevance_threshold", defaultConfig.TRAINING_RELEVANCE_THRESHOLD)
            
            # [UPDATED] Check local borrowed_data first for overrides, fallback to owner_profile_data
            temperature = borrowed_data.get("temperature", owner_profile_data.get("temperature", defaultConfig.GEMINI_TEMPERATURE))
            top_p = borrowed_data.get("top_p", owner_profile_data.get("top_p", defaultConfig.GEMINI_TOP_P))
            top_k = borrowed_data.get("top_k", owner_profile_data.get("top_k", defaultConfig.GEMINI_TOP_K))
            
            primary_model = borrowed_data.get("primary_model", owner_profile_data.get("primary_model", PRIMARY_MODEL_NAME))
            fallback_model = borrowed_data.get("fallback_model", owner_profile_data.get("fallback_model", FALLBACK_MODEL_NAME))

            grounding_enabled = borrowed_data.get("grounding_enabled", False)
        else:
            profile_data = user_data.get("profiles", {}).get(active_profile_name)
            
            if not profile_data:
                # This can happen if a user's active personal profile was deleted.
                # We return empty data, which will cause a graceful failure message to be sent.
                return {}, "", False, defaultConfig.GEMINI_TEMPERATURE, defaultConfig.GEMINI_TOP_P, defaultConfig.GEMINI_TOP_K, defaultConfig.TRAINING_CONTEXT_SIZE, defaultConfig.TRAINING_RELEVANCE_THRESHOLD, PRIMARY_MODEL_NAME, FALLBACK_MODEL_NAME
            
            persona = profile_data.get("persona", {})
            ai_instructions = profile_data.get("ai_instructions", "")
            grounding_enabled = profile_data.get("grounding_enabled", False)
            temperature = profile_data.get("temperature", defaultConfig.GEMINI_TEMPERATURE)
            top_p = profile_data.get("top_p", defaultConfig.GEMINI_TOP_P)
            top_k = profile_data.get("top_k", defaultConfig.GEMINI_TOP_K)
            training_context_size = profile_data.get("training_context_size", defaultConfig.TRAINING_CONTEXT_SIZE)
            training_relevance_threshold = profile_data.get("training_relevance_threshold", defaultConfig.TRAINING_RELEVANCE_THRESHOLD)
            primary_model = profile_data.get("primary_model", PRIMARY_MODEL_NAME)
            fallback_model = profile_data.get("fallback_model", FALLBACK_MODEL_NAME)
            
        return (persona, ai_instructions, grounding_enabled, float(temperature), float(top_p), int(top_k), int(training_context_size), float(training_relevance_threshold), primary_model, fallback_model)
    
    def _get_user_data_entry(self, user_id: int) -> Dict[str, Any]:
        user_id_str = str(user_id)
        if user_id_str in self.user_profiles:
            return self.user_profiles[user_id_str]

        file_path = os.path.join(self.PROFILES_DIR, f"{user_id_str}.json.gz")
        user_data = self._load_json_gzip(file_path)

        if user_data is None:
            user_data = {
                "profiles": {},
                "borrowed_profiles": {},
                "channel_active_profiles": {},
            }

        if "profiles" not in user_data: user_data["profiles"] = {}
        if "borrowed_profiles" not in user_data: user_data["borrowed_profiles"] = {}
        if "channel_active_profiles" not in user_data: user_data["channel_active_profiles"] = {}

        # --- Data Migration: Fix for "personality traits" key ---
        data_was_migrated = False
        for pname, pdata in user_data.get("profiles", {}).items():
            if "persona" in pdata and "personality traits" in pdata["persona"]:
                pdata["persona"]["personality_traits"] = pdata["persona"].pop("personality traits")
                data_was_migrated = True
        if data_was_migrated:
            self._save_user_data_entry(user_id, user_data)
        # --- End Migration ---

        for pname, pdata in user_data.get("profiles", {}).items():
            pdata.setdefault("grounding_enabled", False)
            pdata.setdefault("top_p", defaultConfig.GEMINI_TOP_P)
            pdata.setdefault("top_k", defaultConfig.GEMINI_TOP_K)
            pdata.setdefault("training_context_size", defaultConfig.TRAINING_CONTEXT_SIZE)
            pdata.setdefault("training_relevance_threshold", defaultConfig.TRAINING_RELEVANCE_THRESHOLD)
            pdata.setdefault("primary_model", PRIMARY_MODEL_NAME)
            pdata.setdefault("fallback_model", FALLBACK_MODEL_NAME)
            pdata.setdefault("time_tracking_enabled", True)
            pdata.setdefault("timezone", "UTC")
            pdata.setdefault("realistic_typing_enabled", False)
            pdata.setdefault("freewill_enabled", False)
            pdata.setdefault("wakewords", [])
            pdata.setdefault("safety_level", "low")
            pdata.setdefault("ltm_creation_enabled", False)
            pdata.setdefault("speech_voice", "Aoede")
            pdata.setdefault("speech_model", "gemini-2.5-flash-preview-tts")
            pdata.setdefault("speech_temperature", 1.0)
            pdata.setdefault("speech_archetype", "")
            pdata.setdefault("speech_accent", "")
            pdata.setdefault("speech_dynamics", "")
            pdata.setdefault("speech_style", "")
            pdata.setdefault("speech_pacing", "")

        for pname, pdata in user_data.get("borrowed_profiles", {}).items():
            pdata.setdefault("ltm_creation_enabled", False)
            pdata.setdefault("thinking_summary_visible", "off")
            pdata.setdefault("thinking_level", "high")
            pdata.setdefault("thinking_budget", -1)

        owner_id = int(defaultConfig.DISCORD_OWNER_ID)
        if user_id != owner_id and DEFAULT_PROFILE_NAME not in user_data.get("borrowed_profiles", {}):
            owner_user_data = self._get_user_data_entry(owner_id)
            
            owner_profile_data = owner_user_data.get("profiles", {}).get(DEFAULT_PROFILE_NAME)
            if not owner_profile_data:
                owner_profile_data = self._get_or_create_user_profile(owner_id, DEFAULT_PROFILE_NAME)

            user_data.setdefault("borrowed_profiles", {})[DEFAULT_PROFILE_NAME] = {
                "original_owner_id": str(owner_id),
                "original_profile_name": DEFAULT_PROFILE_NAME,
                "borrowed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "grounding_enabled": owner_profile_data.get("grounding_enabled", False),
                "realistic_typing_enabled": owner_profile_data.get("realistic_typing_enabled", False),
                "time_tracking_enabled": owner_profile_data.get("time_tracking_enabled", False),
                "timezone": owner_profile_data.get("timezone", "UTC")
            }

        self.user_profiles[user_id_str] = user_data
        return user_data
    
    def _save_user_data_entry(self, user_id: int, data: Dict[str, Any]):
        user_id_str = str(user_id)
        file_path = os.path.join(self.PROFILES_DIR, f"{user_id_str}.json.gz")
        try:
            self._atomic_json_save_gzip(data, file_path)
            if user_id_str in self.user_profiles:
                self.user_profiles[user_id_str] = data
        except Exception as e:
            print(f"Error saving user profile shard for {user_id_str}: {e}")

    def _get_or_create_user_profile(self, user_id: int, profile_name: str) -> Optional[Dict[str, Any]]:
        user_data = self._get_user_data_entry(user_id)
        profile_name = profile_name.lower().strip()
        owner_id = int(defaultConfig.DISCORD_OWNER_ID)

        if profile_name == DEFAULT_PROFILE_NAME and user_id != owner_id:
            return None

        if profile_name not in user_data["profiles"]:
            if len(user_data["profiles"]) >= MAX_USER_PROFILES and profile_name != DEFAULT_PROFILE_NAME:
                return None 
            user_data["profiles"][profile_name] = {
                "persona": {},
                "ai_instructions": ["", "", "", ""], # [NEW] 4 slots for expanded instructions
                "grounding_enabled": False,
                "stm_length": defaultConfig.CHATBOT_MEMORY_LENGTH,
                "temperature": defaultConfig.GEMINI_TEMPERATURE,
                "top_p": defaultConfig.GEMINI_TOP_P,
                "top_k": defaultConfig.GEMINI_TOP_K,
                "training_context_size": defaultConfig.TRAINING_CONTEXT_SIZE,
                "training_relevance_threshold": defaultConfig.TRAINING_RELEVANCE_THRESHOLD,
                "ltm_context_size": 3,
                "ltm_relevance_threshold": 0.75,
                "ltm_creation_interval": 10,
                "ltm_summarization_context": 10,
                "ltm_scope": "server",
                "ltm_summarization_instructions": self._encrypt_data(DEFAULT_LTM_SUMMARIZATION_INSTRUCTIONS),
                "safety_level": "low",
                "primary_model": PRIMARY_MODEL_NAME,
                "fallback_model": FALLBACK_MODEL_NAME,
                "time_tracking_enabled": True,
                "timezone": "UTC",
                "realistic_typing_enabled": False,
                "ltm_creation_enabled": False,
                "image_generation_enabled": False,
                "url_fetching_enabled": False,
                "response_mode": "regular",
                "thinking_summary_visible": "off",
                "thinking_level": "high",
                "thinking_budget": -1,
                "error_response": "An error has occurred."
            }
        else: 
            profile = user_data["profiles"][profile_name]
            profile.setdefault("error_response", "An error has occurred.")
            profile.setdefault("grounding_enabled", False)
            profile.setdefault("stm_length", defaultConfig.CHATBOT_MEMORY_LENGTH)
            profile.setdefault("temperature", defaultConfig.GEMINI_TEMPERATURE)
            profile.setdefault("top_p", defaultConfig.GEMINI_TOP_P)
            profile.setdefault("top_k", defaultConfig.GEMINI_TOP_K)
            profile.setdefault("training_context_size", defaultConfig.TRAINING_CONTEXT_SIZE)
            profile.setdefault("training_relevance_threshold", defaultConfig.TRAINING_RELEVANCE_THRESHOLD)
            profile.setdefault("ltm_context_size", 3)
            profile.setdefault("ltm_relevance_threshold", 0.75)
            profile.setdefault("ltm_scope", "server")
            profile.setdefault("ltm_summarization_instructions", DEFAULT_LTM_SUMMARIZATION_INSTRUCTIONS)
            profile.setdefault("safety_level", "low")
            profile.setdefault("primary_model", PRIMARY_MODEL_NAME)
            profile.setdefault("fallback_model", FALLBACK_MODEL_NAME)
            profile.setdefault("time_tracking_enabled", True)
            profile.setdefault("timezone", "UTC")
            profile.setdefault("realistic_typing_enabled", False)
            profile.setdefault("ltm_creation_enabled", False)
            profile.setdefault("image_generation_prompt", None)
            profile.setdefault("thinking_summary_visible", "off")
            profile.setdefault("thinking_level", "medium")
            profile.setdefault("thinking_budget", -1)
            profile.setdefault("speech_voice", "Aoede")
            profile.setdefault("speech_model", "gemini-2.5-flash-preview-tts")
            profile.setdefault("speech_temperature", 1.0)
            profile.setdefault("speech_archetype", "")
            profile.setdefault("speech_accent", "")
            profile.setdefault("speech_dynamics", "")
            profile.setdefault("speech_style", "")
            profile.setdefault("speech_pacing", "")
        return user_data["profiles"][profile_name]
    
    def _get_active_user_profile_name_for_channel(self, user_id: int, channel_id: int) -> str:
        user_data = self._get_user_data_entry(user_id)
        return user_data.get("channel_active_profiles", {}).get(str(channel_id), DEFAULT_PROFILE_NAME)

    def _get_active_user_profile_data(self, user_id: int, channel_id: int) -> Optional[Dict[str, Any]]:
        user_data = self._get_user_data_entry(user_id)
        active_profile_name = self._get_active_user_profile_name_for_channel(user_id, channel_id)
        
        profile = user_data.get("profiles", {}).get(active_profile_name)
        if not profile and active_profile_name != DEFAULT_PROFILE_NAME: 
            profile = user_data.get("profiles", {}).get(DEFAULT_PROFILE_NAME)
        
        if profile: 
            profile.setdefault("grounding_enabled", False)
            profile.setdefault("temperature", defaultConfig.GEMINI_TEMPERATURE)
            profile.setdefault("top_p", defaultConfig.GEMINI_TOP_P)
            profile.setdefault("top_k", defaultConfig.GEMINI_TOP_K)
            profile.setdefault("training_context_size", defaultConfig.TRAINING_CONTEXT_SIZE)
            profile.setdefault("training_relevance_threshold", defaultConfig.TRAINING_RELEVANCE_THRESHOLD)
            profile.setdefault("primary_model", PRIMARY_MODEL_NAME)
            profile.setdefault("fallback_model", FALLBACK_MODEL_NAME)
        return profile

    async def _set_active_user_profile_for_channel(self, user_id: int, channel_id: int, profile_name: str, interaction_for_feedback: Optional[discord.Interaction] = None) -> bool:
        user_data = self._get_user_data_entry(user_id)
        profile_name = profile_name.lower().strip()
        is_borrowed = profile_name in user_data.get("borrowed_profiles", {})

        if not is_borrowed and profile_name != DEFAULT_PROFILE_NAME and profile_name not in user_data.get("profiles", {}): 
            if interaction_for_feedback:
                await interaction_for_feedback.followup.send(f"Your profile '{profile_name}' not found. Cannot activate.", ephemeral=True)
            return False
        
        old_profile_name = self._get_active_user_profile_name_for_channel(user_id, channel_id)
        user_data.setdefault("channel_active_profiles", {})[str(channel_id)] = profile_name
        self._save_user_data_entry(user_id, user_data)
        
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
                borrowed_data = user_data["borrowed_profiles"][profile_name]
                effective_owner_id = int(borrowed_data["original_owner_id"])
                effective_profile_name = borrowed_data["original_profile_name"]
            
            active_appearance = None
            owner_id_str = str(effective_owner_id)
            if owner_id_str in self.user_appearances and effective_profile_name in self.user_appearances[owner_id_str]:
                active_appearance = self.user_appearances[owner_id_str][effective_profile_name]

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
    
    async def _validate_and_clean_borrowed_profiles(self, user_id: int) -> int:
        """
        Scans a user's borrowed profiles. If the source profile no longer exists
        (deleted by owner), it removes the borrowed entry.
        Returns the number of profiles removed.
        """
        user_data = self._get_user_data_entry(user_id)
        borrowed = user_data.get("borrowed_profiles", {})
        if not borrowed:
            return 0

        # Group by owner to minimize I/O
        profiles_by_owner = {}
        for local_name, data in borrowed.items():
            o_id = data.get("original_owner_id")
            o_name = data.get("original_profile_name")
            if o_id and o_name:
                profiles_by_owner.setdefault(str(o_id), []).append((local_name, o_name))

        removed_count = 0
        ids_to_remove = []

        for owner_id_str, items in profiles_by_owner.items():
            # Load owner data once per owner
            owner_data = self._get_user_data_entry(int(owner_id_str))
            owner_profiles = owner_data.get("profiles", {})
            
            for local_name, original_name in items:
                if original_name not in owner_profiles:
                    ids_to_remove.append(local_name)

        if ids_to_remove:
            for local_name in ids_to_remove:
                del user_data["borrowed_profiles"][local_name]
            self._save_user_data_entry(user_id, user_data)
            removed_count = len(ids_to_remove)
            
        return removed_count
    
    def _serialize_chat_session(self, chat_session: genai.ChatSession) -> bytes:
        serialized = []
        for content in chat_session.history:
            parts_list = []
            for part in content.parts:
                # For now, we only serialize text parts to avoid complexity with images/other data.
                if hasattr(part, 'text'):
                    parts_list.append({'text': part.text})
            if parts_list:
                serialized.append({'role': content.role, 'parts': parts_list})
        return json.dumps(serialized)

    def _deserialize_chat_session(self, data_bytes: bytes, model: genai.GenerativeModel) -> genai.ChatSession:
        from google.generativeai.types import content_types
        data = json.loads(data_bytes)
        history = [content_types.to_content(item) for item in data]
        return model.start_chat(history=history)
    
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