import os
import gzip
import zstandard as zstd
import uuid
import heapq
import asyncio
import random
import time
from discord.ext import tasks
import pathlib
import datetime
import collections
import discord
from typing import Dict, List, Any, Optional, Union, Tuple
from cryptography.fernet import InvalidToken
import orjson as json

from ..utils.constants import (
    USERS_DIR, SERVERS_DIR, SESSIONS_GLOBAL_DIR, defaultConfig,
    PRIMARY_MODEL_NAME, FALLBACK_MODEL_NAME,
)
from .storage_manager import IOManager, _delete_file_shard


class SessionManager:
    """Owns chat session state: history hydration/dehydration to disk, multi-profile
    channel memory tracking, and the inactive-session eviction sweep.

    Holds a back-reference to the parent cog for state/logic not yet migrated
    (bot, manager_queue, in-memory caches, and profile/server-index lookups),
    per the transitional Dependency Injection pattern in CLAUDE.md.
    """

    def __init__(self, cog):
        self.cog = cog

    def _get_session_dir_path(self, session_key: Any, session_type: str) -> pathlib.Path:
        if session_type == 'global_chat':
            _, user_id, _ = session_key
            return pathlib.Path(SESSIONS_GLOBAL_DIR) / str(user_id)

        channel_id, _, _ = session_key
        channel = self.cog.bot.get_channel(channel_id)
        server_id = channel.guild.id if channel and getattr(channel, 'guild', None) else None
        if not server_id:
            raise ValueError("Sessions are not supported in Direct Messages.")
        return pathlib.Path(SERVERS_DIR) / str(server_id) / "sessions" / str(channel_id) / session_type

    def _get_session_path(self, session_key: Any, session_type: str) -> pathlib.Path:
        if session_type == 'global_chat':
            _, user_id, profile_name = session_key
            pid = self.cog.profile_manager._get_pid_from_name_any(user_id, profile_name)
            return pathlib.Path(USERS_DIR) / str(user_id) / "profiles" / pid / "global_chat.json.gz"

        dir_path = self._get_session_dir_path(session_key, session_type)
        # All other session types use a unified log
        return dir_path / "session_log.json.gz"

    async def _save_session_to_disk(self, session_key: Any, session_type: str, session_data: Union[List[Dict], Dict]):
        if not session_data:
            await self._delete_session_from_disk(session_key, session_type)
            return

        if session_type != 'global_chat':
            channel_id, _, _ = session_key
            channel = self.cog.bot.get_channel(channel_id)
            if not channel or not getattr(channel, 'guild', None):
                return

        data_to_save = session_data

        if session_type == 'global_chat':
            if isinstance(session_data, dict):
                if 'unified_log' in session_data:
                    if not session_data['unified_log']:
                        await self._delete_session_from_disk(session_key, session_type)
                        return
                    data_to_save = session_data['unified_log']
                else:
                    return
            elif hasattr(session_data, 'history'):
                 log = []
                 for content in session_data.history:
                     parts_text = "".join(p if isinstance(p, str) else p.get('text', '') for p in content.get('parts', []))
                     log.append({
                         "turn_id": str(uuid.uuid4()), "role": content.get('role', 'user'), "content": parts_text,
                         "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                     })
                 data_to_save = log

        if hasattr(data_to_save, 'history') and not data_to_save.history:
             await self._delete_session_from_disk(session_key, session_type)
             return

        try:
            path = self._get_session_path(session_key, session_type)
            data_copy = list(data_to_save) if isinstance(data_to_save, list) else data_to_save.copy()

            def _thread_save():
                path.parent.mkdir(parents=True, exist_ok=True)
                serialized_bytes = json.dumps(data_copy, option=json.OPT_SERIALIZE_NUMPY | json.OPT_NON_STR_KEYS)
                compressed_bytes = zstd.ZstdCompressor(level=1).compress(serialized_bytes)
                encrypted_compressed_bytes = self.cog.fernet.encrypt(compressed_bytes)

                temp_path = path.with_suffix(path.suffix + '.tmp')
                with open(temp_path, 'wb') as f:
                    f.write(encrypted_compressed_bytes)
                
                try:
                    os.replace(temp_path, path)
                except FileNotFoundError:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(temp_path, path)

            await asyncio.to_thread(_thread_save)
        except Exception as e:
            print(f"Error saving session for key {session_key}: {e}")

    async def _load_session_from_disk(self, session_key: Any, session_type: str) -> Optional[Union[List[Dict], Dict]]:
        try:
            path = self._get_session_path(session_key, session_type)
            if not path.exists():
                return None

            def _thread_load():
                with open(path, 'rb') as f:
                    encrypted_compressed_bytes = f.read()
                decrypted_compressed_bytes = self.cog.fernet.decrypt(encrypted_compressed_bytes)
                try:
                    json_bytes = zstd.ZstdDecompressor().decompress(decrypted_compressed_bytes)
                except zstd.ZstdError:
                    json_bytes = gzip.decompress(decrypted_compressed_bytes)
                if not json_bytes: return None
                return json.loads(json_bytes)

            data = await asyncio.to_thread(_thread_load)

            if not data:
                await self._delete_session_from_disk(session_key, session_type)
                return None

            if session_type == 'global_chat':
                if data and isinstance(data, list) and 'parts' in data[0]:
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
                        unified_log.append(log_item)

                    return {'unified_log': unified_log}

                elif data and isinstance(data, list) and 'turn_id' in data[0]:
                    return {'unified_log': data}

                return None

            else:
                return data
        except (gzip.BadGzipFile, zstd.ZstdError, json.JSONDecodeError, InvalidToken):
             print(f"Warning: Corrupted or old-format session file for key {session_key}. Deleting file.")
             await self._delete_session_from_disk(session_key, session_type)
        except Exception as e:
            print(f"Error loading session for key {session_key}: {e}")
        return None

    async def _delete_session_from_disk(self, session_key: Any, session_type: str):
        try:
            path = self._get_session_path(session_key, session_type)
            await asyncio.to_thread(_delete_file_shard, str(path))
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
                data = IOManager.read_json_gzip(str(path), self.cog.fernet) or {}
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
                IOManager.write_json_gzip(data, str(path), self.cog.fernet)
        except Exception as e:
            print(f"Error saving mapping for key {mapping_key}: {e}")

    def _get_mapping_key_for_session(self, session_key: Any, session_type: str) -> Any:
        if session_type == 'global_chat':
            _, user_id, _ = session_key
            return ('global_chat', user_id)

        # All other session types are keyed by channel_id
        elif session_type in ['multi']:
            channel_id, _, _ = session_key
            return (session_type, channel_id)
        return None

    async def _load_multi_profile_sessions(self):
        await self.cog.bot.wait_until_ready()

        servers_dir = SERVERS_DIR
        if not os.path.isdir(servers_dir):
            return

        for server_id_str in os.listdir(servers_dir):
            if not server_id_str.isdigit():
                continue

            server_index = self.cog.server_manager._get_server_index(server_id_str)
            active_sessions = server_index.get("active_sessions", {})
            if isinstance(active_sessions, list):
                continue

            # Combine regular and legacy freewill for memory loading
            all_sessions = {**active_sessions.get("regular", {}), **active_sessions.get("freewill", {})}

            for ch_id_str, session_data in all_sessions.items():
                try:
                    channel_id = int(ch_id_str)
                    channel = self.cog.bot.get_channel(channel_id)
                    if not channel or not channel.guild: continue

                    owner_id = session_data.get("owner_id")
                    profiles_data = session_data.get("profiles",[])

                    if not owner_id or not profiles_data:
                        continue

                    self.cog.multi_profile_channels[channel_id] = {
                        "profiles": profiles_data,
                        "is_hydrated": False,
                        "last_bot_message_id": None,
                        "owner_id": owner_id,
                        "is_running": False,
                        "task_queue": asyncio.Queue(),
                        "worker_task": None,
                        "turns_since_last_ltm": 0,
                        "session_prompt": session_data.get("session_prompt"),
                        "session_mode": session_data.get("session_mode", "sequential"),
                        "type": "multi",
                        "proactivity": session_data.get("proactivity", {"enabled": False, "chance": 20, "cooldown": 300, "director_model": "off", "director_instructions": "You are an AI Director for a roleplay session. Introduce a sudden event, an environmental change, or a question to spark conversation among the cast. Keep it brief (1-2 sentences)."})
                    }
                except Exception as e:
                    print(f"Unexpected error reloading multi-profile sessions for server {server_id_str}, channel {ch_id_str}: {e}")

    def _save_multi_profile_sessions(self):
        try:
            current_server_sessions = collections.defaultdict(lambda: {"regular": {}})

            for channel_id, session_data in self.cog.multi_profile_channels.items():
                channel = self.cog.bot.get_channel(channel_id)
                server_id_str = str(channel.guild.id) if channel and getattr(channel, 'guild', None) else "dm"

                category = "regular"

                profiles_to_save =[]
                for p in session_data.get("profiles",[]):
                    pid = self.cog.profile_manager._get_pid_from_name_any(p["owner_id"], p["profile_name"])
                    profiles_to_save.append({
                        "pid": pid,
                        "profile_name": p["profile_name"],
                        "owner_id": p["owner_id"],
                        "method": p.get("method", "webhook"),
                        "bot_id": p.get("bot_id"),
                        "ephemeral": p.get("ephemeral", False),
                        "chance": p.get("chance", 100),
                        "wakewords": p.get("wakewords",[])
                    })

                blueprint = {
                    "owner_id": session_data.get("owner_id"),
                    "profiles": profiles_to_save,
                    "session_prompt": session_data.get("session_prompt"),
                    "session_mode": session_data.get("session_mode", "sequential"),
                    "type": "multi",
                    "proactivity": session_data.get("proactivity", {"enabled": False, "chance": 10, "cooldown": 300, "director_model": "off", "director_instructions": "You are an AI Director for a roleplay session. Introduce a sudden event, an environmental change, or a question to spark conversation among the cast. Keep it brief (1-2 sentences)."})
                }

                current_server_sessions[server_id_str][category][str(channel_id)] = blueprint

            servers_dir = SERVERS_DIR
            if os.path.exists(servers_dir):
                for s_dir in os.listdir(servers_dir):
                    if os.path.isdir(os.path.join(servers_dir, s_dir)) and s_dir not in current_server_sessions:
                        current_server_sessions[s_dir] = {"regular": {}}

            for server_id_str, sessions_for_server in current_server_sessions.items():
                server_index = self.cog.server_manager._get_server_index(server_id_str)
                server_index["active_sessions"] = sessions_for_server
                self.cog.server_manager._save_server_index(server_id_str, server_index)

                # Housekeeping: Delete legacy file if it exists
                old_file = os.path.join(servers_dir, server_id_str, "sessions.json.gz")
                if os.path.exists(old_file):
                    _delete_file_shard(old_file)

        except Exception as e:
            print(f"Error saving sharded multi-profile sessions to index: {e}")

    def _get_user_profile_for_model(self, user_id: int, channel_id: int, profile_name_override: Optional[str] = None) -> Tuple[Dict[str, List[str]], str, bool, float, float, int, int, float, str, str]:
        active_profile_name = profile_name_override if profile_name_override else self._get_active_user_profile_name_for_channel(user_id, channel_id)

        if not active_profile_name:
            return {}, "", False, defaultConfig.GEMINI_TEMPERATURE, defaultConfig.GEMINI_TOP_P, defaultConfig.GEMINI_TOP_K, defaultConfig.TRAINING_CONTEXT_SIZE, defaultConfig.TRAINING_RELEVANCE_THRESHOLD, PRIMARY_MODEL_NAME, FALLBACK_MODEL_NAME

        index = self.cog.profile_manager._get_user_index(user_id)
        is_borrowed = active_profile_name in index.get("borrowed", [])

        config = self.cog.profile_manager._get_profile_config(user_id, active_profile_name, is_borrowed)
        if not config:
            return {}, "", False, defaultConfig.GEMINI_TEMPERATURE, defaultConfig.GEMINI_TOP_P, defaultConfig.GEMINI_TOP_K, defaultConfig.TRAINING_CONTEXT_SIZE, defaultConfig.TRAINING_RELEVANCE_THRESHOLD, PRIMARY_MODEL_NAME, FALLBACK_MODEL_NAME

        source_owner_id, source_profile_name = self.cog.profile_manager._resolve_effective_profile(user_id, active_profile_name)

        prompts = self.cog.profile_manager._get_profile_prompts(source_owner_id, source_profile_name) or {}

        persona = prompts.get("persona", {})
        ai_instructions = prompts.get("ai_instructions", "")

        training_context_size = config.get("training_context_size", defaultConfig.TRAINING_CONTEXT_SIZE)
        training_relevance_threshold = config.get("training_relevance_threshold", defaultConfig.TRAINING_RELEVANCE_THRESHOLD)
        temperature = config.get("temperature", defaultConfig.GEMINI_TEMPERATURE)
        top_p = config.get("top_p", defaultConfig.GEMINI_TOP_P)
        top_k = config.get("top_k", defaultConfig.GEMINI_TOP_K)
        primary_model = config.get("primary_model", PRIMARY_MODEL_NAME)
        fallback_model = config.get("fallback_model", FALLBACK_MODEL_NAME)
        grounding_enabled = config.get("grounding_enabled", False)

        return (persona, ai_instructions, grounding_enabled, float(temperature), float(top_p), int(top_k), int(training_context_size), float(training_relevance_threshold), primary_model, fallback_model)

    def _get_active_user_profile_name_for_channel(self, user_id: int, channel_id: int) -> Optional[str]:
        channel = self.cog.bot.get_channel(channel_id)
        server_id_str = str(channel.guild.id) if channel and getattr(channel, 'guild', None) else "dm"
        server_index = self.cog.server_manager._get_server_index(server_id_str)
        active = server_index.get("user_active_profiles", {}).get(str(user_id), {}).get(str(channel_id))
        if active: return active

        index = self.cog.profile_manager._get_user_index(user_id)
        personal = index.get("personal", [])
        if personal:
            return next(iter(personal))
        borrowed = index.get("borrowed", [])
        if borrowed:
            return next(iter(borrowed))
        return None

    def _get_active_user_profile_data(self, user_id: int, channel_id: int) -> Optional[Dict[str, Any]]:
        active_profile_name = self._get_active_user_profile_name_for_channel(user_id, channel_id)
        if not active_profile_name: return None
        index = self.cog.profile_manager._get_user_index(user_id)

        is_borrowed = active_profile_name in index.get("borrowed", [])
        config = self.cog.profile_manager._get_profile_config(user_id, active_profile_name, is_borrowed)

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

    async def _set_active_user_profile_for_channel(self, user_id: int, channel_id: int, profile_name: str, interaction_for_feedback=None) -> bool:
        profile_name = profile_name.lower().strip()
        channel = self.cog.bot.get_channel(channel_id)
        server_id_str = str(channel.guild.id) if channel and getattr(channel, 'guild', None) else "dm"

        def _sync_switch():
            index = self.cog.profile_manager._get_user_index(user_id)
            is_borrowed = profile_name in index.get("borrowed", [])

            if not is_borrowed and profile_name not in index.get("personal", []):
                return None

            server_index = self.cog.server_manager._get_server_index(server_id_str)
            old_profile_name = server_index.get("user_active_profiles", {}).get(str(user_id), {}).get(str(channel_id))
            server_index.setdefault("user_active_profiles", {}).setdefault(str(user_id), {})[str(channel_id)] = profile_name
            self.cog.server_manager._save_server_index(server_id_str, server_index)

            effective_owner_id = user_id
            effective_profile_name = profile_name
            active_appearance = None
            if interaction_for_feedback:
                if is_borrowed:
                    b_config = self.cog.profile_manager._get_profile_config(user_id, profile_name, True)
                    if b_config:
                        effective_owner_id = int(b_config.get("original_owner_id", user_id))
                        effective_profile_name = b_config.get("original_profile_name", profile_name)

                owner_config = self.cog.profile_manager._get_profile_config(effective_owner_id, effective_profile_name)
                if owner_config and (owner_config.get("custom_display_name") or owner_config.get("custom_avatar_url")):
                    active_appearance = owner_config

            return old_profile_name, active_appearance

        result = await asyncio.to_thread(_sync_switch)
        if result is None:
            if interaction_for_feedback:
                await interaction_for_feedback.followup.send(f"Your profile '{profile_name}' not found. Cannot activate.", ephemeral=True)
            return False

        old_profile_name, active_appearance = result

        if interaction_for_feedback and interaction_for_feedback.guild:
            warning_key_to_clear = (user_id, interaction_for_feedback.guild.id, old_profile_name)
            self.cog.model_override_warnings_sent.discard(warning_key_to_clear)

        # channel_models is keyed (channel_id, profile_owner_id, profile_name) — see
        # APIService.get_or_create_model. This used to build a 2-tuple, so all three
        # evictions silently matched nothing and a profile switch left the previous
        # profile's model cached. Drop every entry for this (channel, user) pair, which
        # covers both the old and the new profile name.
        stale_keys = [
            k for k in self.cog.channel_models
            if isinstance(k, tuple) and len(k) == 3 and k[0] == channel_id and k[1] == user_id
        ]
        for k in stale_keys:
            self.cog.channel_models.pop(k, None)
            self.cog.channel_model_last_profile_key.pop(k, None)

        if interaction_for_feedback:
            channel_mention = f"<#{channel_id}>" if interaction_for_feedback.guild else "this DM"

            embed_title = f"Your Profile Preference Swapped to: '{profile_name}'"
            embed_desc = f"Your individual preferred profile in {channel_mention} is now '{profile_name}'."

            embed = discord.Embed(title=embed_title, description=embed_desc, color=discord.Color.green())

            app_name = self.cog.bot.user.name if self.cog.bot.user else "Bot"
            app_avatar_url = self.cog.bot.user.display_avatar.url if self.cog.bot.user else None

            if active_appearance:
                app_name = active_appearance.get("custom_display_name") or app_name
                app_avatar_url = active_appearance.get("custom_avatar_url") or app_avatar_url

            embed.add_field(name="Linked Appearance", value=f"Name: {app_name}", inline=False)
            if app_avatar_url:
                embed.set_thumbnail(url=app_avatar_url)

            await interaction_for_feedback.followup.send(embed=embed, ephemeral=True)
        return True

    async def _ensure_session_hydrated(self, channel_id: int, session_type: str) -> Optional[Dict]:
        """Checks if a session is in memory and hydrated. If not, loads it from disk."""
        session = self.cog.multi_profile_channels.get(channel_id)

        # Lazily validate and clean up all session participants if the session is engaged
        if session and session.get("profiles"):
            cleaned_any = False
            valid_profiles = []

            # Once per distinct owner, not once per participant. This scan reads and
            # decrypts every borrowed profile the owner has, so a five-participant
            # session sharing one owner ran that whole sweep five times per
            # hydration, on the event loop.
            for owner_id in {p["owner_id"] for p in session["profiles"]}:
                await self.cog.profile_manager._validate_and_clean_borrowed_profiles(owner_id)

            for p in list(session["profiles"]):
                p_owner_id = p["owner_id"]
                p_name = p["profile_name"]

                # Verify if the profile continues to exist for the owner
                p_index = self.cog.profile_manager._get_user_index(p_owner_id)
                exists = (p_name in p_index.get("personal", {})) or (p_name in p_index.get("borrowed", {})) or (p_name in p_index.get("system", {}))

                if exists:
                    valid_profiles.append(p)
                else:
                    cleaned_any = True
                    if p.get("method") == "child_bot" and p.get("bot_id"):
                        await self.cog.manager_queue.put({
                            "action": "send_to_child", "bot_id": p["bot_id"],
                            "payload": {"action": "session_update_remove", "channel_id": channel_id}
                        })

            if cleaned_any:
                session["profiles"] = valid_profiles
                self._save_multi_profile_sessions()
                if not session["profiles"]:
                    self.cog.multi_profile_channels.pop(channel_id, None)
                    dummy_session_key = (channel_id, None, None)
                    await self._delete_session_from_disk(dummy_session_key, session_type)
                    return None

        if session and session.get("is_hydrated"):
            return session

        # Resolve Guild ID for pathing
        guild_id_str = None
        channel = self.cog.bot.get_channel(channel_id)
        if channel and hasattr(channel, 'guild'):
            guild_id_str = str(channel.guild.id)
        else:
            for guild in self.cog.bot.guilds:
                if guild.get_channel(channel_id):
                    guild_id_str = str(guild.id); break

        if not guild_id_str: return None

        if not session or not session.get("profiles"):
            server_index = self.cog.server_manager._get_server_index(guild_id_str)
            active_sessions = server_index.get("active_sessions", {})
            if isinstance(active_sessions, dict):
                session_config = active_sessions.get("regular", {}).get(str(channel_id))
                if not session_config:
                    session_config = active_sessions.get("freewill", {}).get(str(channel_id))

                if session_config:
                    profiles = session_config.get("profiles", [])
                    for p in profiles: p.setdefault('ltm_counter', 0)

                    if not session:
                        session = {
                            "profiles": profiles,
                            "owner_id": session_config.get("owner_id"),
                            "session_prompt": session_config.get("session_prompt"),
                            "session_mode": session_config.get("session_mode", "sequential"),
                            "type": "multi",
                            "proactivity": session_config.get("proactivity", {"enabled": False, "chance": 10, "cooldown": 300, "director_model": "off", "director_instructions": "You are an AI Director for a roleplay session. Introduce a sudden event, an environmental change, or a question to spark conversation among the cast. Keep it brief (1-2 sentences)."}),
                            "task_queue": asyncio.Queue(),
                            "is_running": False
                        }
                        self.cog.multi_profile_channels[channel_id] = session
                    else:
                        # Update the existing dehydrated shell with index data
                        session["profiles"] = profiles
                        session["owner_id"] = session_config.get("owner_id")
                        session["session_prompt"] = session_config.get("session_prompt")
                        session["session_mode"] = session_config.get("session_mode", "sequential")
                        session["type"] = "multi"
                        session["proactivity"] = session_config.get("proactivity", {"enabled": False, "chance": 10, "cooldown": 300, "director_model": "off", "director_instructions": "You are an AI Director for a roleplay session. Introduce a sudden event, an environmental change, or a question to spark conversation among the cast. Keep it brief (1-2 sentences)."})

        if not session: return None

        # 2. Load History Log
        dummy_session_key = (channel_id, None, None)
        disk_log = await self._load_session_from_disk(dummy_session_key, session_type) or []

        current_mem_log = session.get("unified_log", [])
        if not disk_log and current_mem_log:
            await self._save_session_to_disk(dummy_session_key, session_type, current_mem_log)
            unified_log = current_mem_log
        elif current_mem_log and len(current_mem_log) >= len(disk_log):
            unified_log = current_mem_log
        else:
            unified_log = disk_log

        session["unified_log"] = unified_log

        # 3. Synchronise Memory for all current participants
        self._recompute_pending_whispers(session, force=True)

        session["is_hydrated"] = True
        return session

    def _recompute_pending_whispers(self, session: Dict, force: bool = False) -> None:
        """Re-derives session['pending_whispers'] from unified_log, in place.

        This is the *only* state a session rebuild derives. Everything else in
        _ensure_session_hydrated is either unconditional (the participant
        validation above the is_hydrated check) or dead for an engaged session
        (the server-index branch, gated on an empty profile list). So the paths
        that mutate unified_log and then flush it call this directly, instead of
        forcing an is_hydrated=False + rehydrate round-trip. That round-trip
        re-read, decrypted, decompressed and re-parsed the whole session log to
        arrive back at the log it already held correctly in memory -- measured at
        2.4 ms and ~15 MB of immediately-discarded allocation for a 1000-turn log,
        GIL-held inside orjson -- and its "no disk log, but a memory log" branch
        re-wrote session files the caller had just deleted.

        No-op while a round is in flight, on purpose. _multi_profile_worker pops a
        participant's whispers when it builds that participant's prompt, but the
        bot's own turn -- which this derivation reads as the signal that a whisper
        has been answered -- does not reach unified_log until the response lands.
        Recomputing inside that window marks consumed whispers pending again and
        re-injects them on the participant's next turn. Deferring is safe, and is
        also what recovers them: a round cancelled mid-flight never writes its
        turn, so the whispers stay derivable from the log and come back on the
        next rebuild. Hydration passes force=True -- it is establishing initial
        state, and at that point nothing has been popped.
        """
        if not force and (session.get('is_running') or session.get('is_regenerating')):
            return

        unified_log = session.get("unified_log", [])
        pending_map = {}

        for p_data in session.get("profiles", []):
            p_key = (p_data['owner_id'], p_data['profile_name'])
            bot_pid = self.cog.profile_manager._get_pid_from_name_any(p_data['owner_id'], p_data['profile_name'])

            pending = self._get_pending_whispers_for_participant(unified_log, bot_pid)
            if pending:
                pending_map[p_key] = pending

        session["pending_whispers"] = pending_map

    async def _wait_for_session_flags(self, session: Dict, flags: Tuple[str, ...], timeout: float) -> bool:
        """Spin until none of `flags` is set on the session. True if it cleared, False on timeout.

        Bounded on purpose. Every one of these flags is cleared in a finally, but a worker
        that dies in the wrong place still leaks one, and an unbounded spin then blocks the
        channel for the life of the process -- which is exactly what /purge's wait used to
        do (see PURGE_BUSY_WAIT_TIMEOUT_SECONDS). A caller that times out must decide for
        itself whether to proceed or abandon; this only reports which happened.

        A caller claiming the channel must assign its own flag *synchronously* on the True
        return, with no await in between. The event loop is single-threaded and this returns
        without yielding, so check-then-set is atomic only for as long as that holds.
        """
        deadline = time.monotonic() + timeout
        while any(session.get(flag) for flag in flags):
            if time.monotonic() > deadline:
                return False
            await asyncio.sleep(0.5)
        return True

    def _mark_session_accessed(self, key):
        ts = time.time()
        self.cog.session_last_accessed[key] = ts
        heapq.heappush(self.cog.eviction_heap, (ts, key))

    async def _evict_inactive_sessions(self):
        now = time.time()
        inactive_threshold = 600  # 10 minute strict dehydration policy

        keys_to_evict = set()

        # O(1) Heap Peek: Only evaluate sessions that have officially expired
        while self.cog.eviction_heap and now - self.cog.eviction_heap[0][0] > inactive_threshold:
            ts, key = heapq.heappop(self.cog.eviction_heap)

            # Lazy Deletion: Ensure this wasn't updated recently
            if self.cog.session_last_accessed.get(key) == ts:
                if isinstance(key, int):
                    session = self.cog.multi_profile_channels.get(key)
                    # whisper_waiting counts as busy as well: a queued whisper holds a
                    # reference to this session dict and reads its unified_log the moment it
                    # claims the channel, and dehydrating underneath it would have the
                    # profile answer with no history at all.
                    if session and (session.get('is_running') or session.get('is_regenerating')
                                    or session.get('is_purging') or session.get('is_whispering')
                                    or session.get('whisper_waiting')):
                        continue
                keys_to_evict.add(key)

        for key in keys_to_evict:
            # Handle multi-profile sessions (key is channel_id int)
            if isinstance(key, int):
                session_to_evict = self.cog.multi_profile_channels.get(key)
                if session_to_evict and session_to_evict.get("is_hydrated"):
                    session_type = session_to_evict.get("type", "multi")
                    unified_log = session_to_evict.get("unified_log")

                    if unified_log is not None:
                        dummy_session_key = (key, None, None)
                        await self._save_session_to_disk(dummy_session_key, session_type, unified_log)

                    # Use safe cancel to prevent "Task destroyed but pending"
                    if session_to_evict.get('worker_task'):
                        self._safe_cancel_task(session_to_evict['worker_task'])
                        session_to_evict['worker_task'] = None

                    # Aggressively dehydrate the session to release memory.
                    session_to_evict['is_hydrated'] = False

                    # Clear the large data structures from the in-memory dictionary.
                    if 'unified_log' in session_to_evict:
                        del session_to_evict['unified_log']
                    # Both are re-derived on the next hydration (pending_whispers from
                    # the log, cancellations only matter for a queued trigger), and
                    # dropping the cancellation set is what bounds it for a session
                    # that is abandoned before its worker next drains the queue.
                    session_to_evict.pop('cancelled_reaction_triggers', None)
                    session_to_evict.pop('pending_whispers', None)
                    # Clean up LTM recall history for participants in this channel to prevent RAM leak
                    for p in session_to_evict.get("profiles", []):
                        full_key = (key, p.get("owner_id"), p.get("profile_name"))
                        self.cog.ltm_recall_history.pop(full_key, None)

                    # If it's a completely empty dormant session, pop it entirely to save memory.
                    if not session_to_evict.get("profiles"):
                        self.cog.multi_profile_channels.pop(key, None)

            # Handle global chat sessions (key is a tuple)
            elif isinstance(key, tuple):
                session_to_save = self.cog.global_chat_sessions.get(key)
                if session_to_save:
                    await self._save_session_to_disk(key, 'global_chat', session_to_save)

                self.cog.global_chat_sessions.pop(key, None)
                self.cog.ltm_recall_history.pop(key, None)

            # Remove from tracking dict after processing
            self.cog.session_last_accessed.pop(key, None)

        # Prune stale child bot edit cooldown timestamps
        for bot_id in list(self.cog.child_bot_edit_cooldowns.keys()):
            timestamps = [ts for ts in self.cog.child_bot_edit_cooldowns[bot_id] if now - ts < 600]
            if timestamps:
                self.cog.child_bot_edit_cooldowns[bot_id] = timestamps
            else:
                self.cog.child_bot_edit_cooldowns.pop(bot_id, None)

    @tasks.loop(seconds=60.0)
    async def proactive_session_task(self):
        if not self.cog.has_lock: return
        now = time.time()
        for channel_id, session in list(self.cog.multi_profile_channels.items()):
            pro = session.get("proactivity", {})
            if not pro.get("enabled"): continue
            
            last_event = session.setdefault("last_proactive_event", 0)
            cooldown = pro.get("cooldown", 300)
            if now - last_event < cooldown: continue
            
            chance = pro.get("chance", 10) / 100.0
            if random.random() > chance: continue
            
            session["last_proactive_event"] = now
            
            await session['task_queue'].put(('proactive_trigger', None))
            if not session.get('is_running') and (not session.get('worker_task') or session['worker_task'].done()):
                task = self.cog.bot.loop.create_task(self.cog.generation_service._multi_profile_worker(channel_id))
                session['worker_task'] = task
                self.cog.background_tasks.add(task)

    @tasks.loop(seconds=60.0)
    async def evict_inactive_sessions_task(self):
        await self._evict_inactive_sessions()

    def _get_pending_whispers_for_participant(self, log_list: List[Dict], bot_pid: str) -> List[str]:
        pending = []
        for turn in log_list:
            if turn.get("is_hidden", False): continue
            turn_type = turn.get("type")
            if not turn_type:
                if turn.get("speaker_pid") == bot_pid:
                    pending.clear()
            elif turn_type == "whisper":
                if turn.get("target_pid") == bot_pid:
                    pending.append(turn.get("content"))
        return pending

    def _build_history_for_participant(self, full_log: List[Dict], bot_pid: str, p_settings: Dict[str, Any], num_participants: int = 1) -> List[Dict]:
        stm_length = int(p_settings.get("stm_length", defaultConfig.CHATBOT_MEMORY_LENGTH))
        effective_stm = max(stm_length, num_participants) if stm_length > 0 else 0
        log_slice = full_log[-effective_stm:] if effective_stm > 0 else []
        
        participant_history = []
        for turn in log_slice:
            if turn.get("is_hidden"): continue
            
            turn_type = turn.get("type")
            if not turn_type:
                role = 'model' if turn.get("speaker_pid") == bot_pid else 'user'
                parts = [turn.get("content")]
                
                if role == 'user':
                    if turn.get("url_context") and p_settings.get("url_fetching_enabled", False):
                        parts.append(f"\n<document_context>\n{turn.get('url_context')}\n</document_context>")
                    if turn.get("grounding_context") and p_settings.get("grounding_mode", "off") != "off":
                        parts.append(f"\n{turn.get('grounding_context')}")
                
                if participant_history and participant_history[-1]['role'] == role:
                    participant_history[-1]['parts'].extend(parts)
                else:
                    content_obj = {'role': role, 'parts': parts}
                    participant_history.append(content_obj)
                    
            elif turn_type == "whisper":
                if turn.get("target_pid") == bot_pid:
                    clean_content = turn.get("content")
                    header, body = clean_content.split('\n', 1) if '\n' in clean_content else ("", clean_content)
                    wrapped = f"{header}\n<private_whisper>\n{body.strip()}\n</private_whisper>\n"
                    if participant_history and participant_history[-1]['role'] == 'user':
                        participant_history[-1]['parts'].append(wrapped)
                    else:
                        participant_history.append({'role': 'user', 'parts': [wrapped]})
                        
            elif turn_type == "private_response":
                if turn.get("speaker_pid") == bot_pid:
                    clean_content = turn.get("content")
                    header, body = clean_content.split('\n', 1) if '\n' in clean_content else ("", clean_content)
                    wrapped = f"{header}\n<private_response>\n{body.strip()}\n</private_response>\n"
                    if participant_history and participant_history[-1]['role'] == 'model':
                        participant_history[-1]['parts'].append(wrapped)
                    else:
                        obj = {'role': 'model', 'parts': [wrapped]}
                        participant_history.append(obj)
                        
        return participant_history

    async def setup_multi_profile_session(self, interaction: discord.Interaction, participants: List[Dict], session_prompt: Optional[str], session_mode: str, as_admin_scope: bool = False, audio_mode: str = "off"):
        user_id = interaction.user.id
        is_update = interaction.channel_id in self.cog.multi_profile_channels

        # Verify each participant's content rating is still current before the
        # session runs. Session setup is user-initiated and infrequent, which is why
        # the staleness check lives here and not in the per-turn gate -- it decrypts
        # the whole persona to hash it. Fire-and-forget: a stale verdict only ever
        # errs toward the previous, stricter answer while the recheck lands.
        for _p in {(p["owner_id"], p["profile_name"]) for p in participants}:
            asyncio.create_task(
                self.cog.profile_manager.verify_content_rating(_p[0], _p[1]))

        if is_update:
            session = self.cog.multi_profile_channels[interaction.channel_id]
            if not session.get("is_hydrated"):
                session = await self._ensure_session_hydrated(interaction.channel_id, session.get("type", "multi"))

        else:
            for p in participants:
                p['ltm_counter'] = 0

            session = {
                "type": "multi",
                "unified_log": [],
                "is_hydrated": False,
                "last_bot_message_id": None,
                "owner_id": interaction.user.id,
                "is_running": False,
                "task_queue": asyncio.Queue(),
                "worker_task": None,
                "turns_since_last_ltm": 0,
                "session_prompt": None,
                "session_mode": "sequential",
                "pending_image_gen_data": None,
                "pending_whispers": {},
                "audio_mode": "off"
            }
            self.cog.multi_profile_channels[interaction.channel_id] = session

        session["type"] = "multi"
        session["session_prompt"] = session_prompt
        session["profiles"] = participants
        session["session_mode"] = session_mode
        session["audio_mode"] = audio_mode
        
        for p_data in participants:
            if p_data.get('method') == 'child_bot':
                await self.cog.manager_queue.put({
                    "action": "send_to_child", "bot_id": p_data['bot_id'],
                    "payload": {"action": "session_update_add", "channel_id": interaction.channel_id}
                })
        
        self._save_multi_profile_sessions()

        profile_list_str = []
        for p_data in participants:
            if p_data.get('method') == 'child_bot':
                bot_user = self.cog.bot.get_user(int(p_data['bot_id']))
                profile_list_str.append(f"`{bot_user.name if bot_user else 'Unknown Bot'}`")
            else:
                profile_list_str.append(f"`{p_data['profile_name']}`")

        action_str = "updated" if is_update else "activated"
        msg = f"Regular session {action_str} with participants: {', '.join(profile_list_str)}."
        if as_admin_scope:
            msg = f"Regular session is now active for all users with profiles: {', '.join(profile_list_str)}."
        
        await interaction.edit_original_response(content=msg, view=None)

    def _safe_cancel_task(self, task: asyncio.Task):
        if task and not task.done():
            task.cancel()
            self.cog.background_tasks.add(task)
            task.add_done_callback(self.cog.background_tasks.discard)

    async def cancel_channel_operations(self, channel_id: int):
        """Forcefully cancels all active generation, workers, typing, and queued tasks for a channel."""
        session = self.cog.multi_profile_channels.get(channel_id)
        if session:
            session['is_running'] = False
            session['is_regenerating'] = False
            session['is_purging'] = False
            session['is_whispering'] = False

            # Drain task queue completely
            q = session.get('task_queue')
            if q:
                while not q.empty():
                    try:
                        q.get_nowait()
                        q.task_done()
                    except (asyncio.QueueEmpty, ValueError):
                        break

            # Cancel running session worker task
            worker_task = session.get('worker_task')
            if worker_task and not worker_task.done():
                self._safe_cancel_task(worker_task)
                session['worker_task'] = None

            # Cancel any in-flight regeneration tasks
            for msg_id, r_task in list(session.get('regen_tasks', {}).items()):
                if r_task and not r_task.done():
                    r_task.cancel()
            session.get('regen_tasks', {}).clear()

        # Stop active child bot typing tasks in this channel
        for bot_id in list(self.cog.child_bots.keys()):
            await self.cog.child_bot_manager.stop_typing(bot_id, channel_id)

        # Purge pending image generation requests
        self.cog.media_service._purge_channel_image_requests(channel_id)
