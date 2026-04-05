from .mixins.storage import *
from .mixins.storage import StorageMixin, _delete_file_shard, _atomic_json_save, _quantize_embedding, _dequantize_embedding, cosine_similarity
from .mixins.services import *
from .mixins.constants import *
from .mixins.core import *
from .mixins.gui import *

import configs.DefaultConfig as defaultConfig
from discord.ext import commands, tasks
import discord
from discord import Interaction, app_commands, ui

from cryptography.fernet import Fernet, InvalidToken
import asyncio
import os
import orjson as json
import datetime
import random
import uuid
from typing import List, Dict, Tuple, Set, Literal, Any, Optional, Union, get_args
import numpy as np
import copy 
import traceback
import time
from collections import OrderedDict
import collections
import re
import httpx
import sqlite3
from zoneinfo import ZoneInfo, available_timezones
import functools
import gzip
from PIL import Image
import io
import gc
import pathlib
import signal
import platform
from urllib.parse import urlparse
import base64

class LRUCache(OrderedDict):
    def __init__(self, max_size, *args, **kwargs):
        self.max_size = max_size
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            oldest = next(iter(self))
            del self[oldest]

class GeminiAgent(commands.Cog, StorageMixin, ServicesMixin, CoreMixin):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.manager_queue = bot.manager_queue
        self.cog_id = str(uuid.uuid4()) 
        self.has_lock = False
        
        # [NEW] Client placeholder for session-specific usage
        self.client = None
        
        self._try_acquire_lock()

        if self.has_lock:
            print(f"GeminiAgent Cog {self.cog_id} acquired lock and is ACTIVE.")
            self.refresh_lock_task.start()
        else:
            print(f"GeminiAgent Cog {self.cog_id} DID NOT acquire lock. Will run in INACTIVE mode.")
        
        print(f"GeminiAgent Init. Models: Primary='{PRIMARY_MODEL_NAME}', Fallback='{FALLBACK_MODEL_NAME}'.")

        self.PROFILES_DIR = LEGACY_PROFILES_DIR
        self.LTM_DIR = LEGACY_LTM_DIR
        self.TRAINING_DIR = LEGACY_TRAINING_DIR
        self.PUBLIC_PROFILES_DIR = PUBLIC_PROFILES_DIR
        self.CHILD_BOTS_DIR = LEGACY_CHILD_BOTS_DIR
        self.USERS_DIR = USERS_DIR
        self.APPEARANCES_DIR = APPEARANCES_DIR
        self.SHARES_DIR = LEGACY_SHARES_DIR
        self.PERSONAL_KEYS_DIR = LEGACY_PERSONAL_KEYS_DIR
        self.DATA_DIR = DATA_DIR
        self.MOD_DATA_DIR = MOD_DATA_DIR
        self.FREEWILL_SERVERS_DIR = FREEWILL_SERVERS_DIR
        self.SESSIONS_GLOBAL_DIR = LEGACY_GLOBAL_CHAT_DIR
        self.SERVERS_DIR = SERVERS_DIR
        
        # Only create the active Phase 3 directories to prevent ghost folders on boot
        active_dirs = [self.USERS_DIR, self.DATA_DIR, self.PUBLIC_PROFILES_DIR, self.SERVERS_DIR, self.MOD_DATA_DIR]
        
        for d in active_dirs:
            os.makedirs(d, exist_ok=True)

        self.global_prompts: Dict[str, str] = {}
        self._load_global_prompts()

        try:
            self.fernet = Fernet(defaultConfig.ENCRYPTION_KEY)
        except Exception as e:
            print(f"CRITICAL: Failed to initialize encryption. Ensure ENCRYPTION_KEY is set in defaultConfig.py. Error: {e}")
            self.fernet = None

        self.persona_modal_sections_order = ['backstory', 'personality_traits', 'likes', 'dislikes', 'appearance'] 
        
        self.user_indices: LRUCache = LRUCache(max_size=20)
        self.server_indices: LRUCache = LRUCache(max_size=50)
        self.profile_configs: LRUCache = LRUCache(max_size=50)
        self.profile_prompts: LRUCache = LRUCache(max_size=20)
        
        self.user_appearances: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {}
        self._load_user_appearances()
        self.channel_webhooks: Dict[int, Dict[str, Any]] = {}
        self._load_channel_webhooks()

        self.share_codes: Dict[str, Dict[str, Any]] = {}

        self.multi_profile_channels: Dict[int, Dict[str, Any]] = {}
        self.sessions_loaded = False
        self.server_api_keys: Dict[int, str] = {}
        self._load_server_api_keys()
        self.personal_api_keys: Dict[str, str] = {}
        self._load_personal_api_keys()
        self.key_submissions: Dict[str, List[Dict[str, Any]]] = {}
        self._load_key_submissions()
        self.profile_shares: Dict[str, List[Dict[str, Any]]] = {}
        self._load_profile_shares()
        self.public_profiles: Dict[str, Dict[str, Any]] = {}
        self._load_public_profiles()
        self.freewill_config: Dict[str, Dict[str, Any]] = {}
        self._load_freewill_config()
        self.freewill_participation: Dict[str, Dict[str, Any]] = {}
        self._load_freewill_participation()
        self.child_bots: Dict[str, Dict[str, Any]] = {}
        self._load_child_bots()
        self.last_freewill_event: Dict[int, float] = {}
        self.last_freewill_message_info: Dict[int, Dict[str, Any]] = {}

        self.channel_models: Dict[Any, Tuple[genai.GenerativeModel, bool, str]] = {} 
        self.channel_model_last_profile_key: Dict[Any, Tuple[Optional[int], str]] = {} 

        self.chat_sessions: LRUCache = LRUCache(max_size=CHAT_SESSION_CACHE_MAX_SIZE)
        self.max_history_items = defaultConfig.CHATBOT_MEMORY_LENGTH
        
        # Server settings (Freewill only)
        self._load_channel_settings()

        self.message_counters_for_ltm: Dict[Tuple[int, str, Literal["guild", "dm"]], int] = {}
        
        self.model_override_warnings_sent: Set[Tuple[int, int, str]] = set()
        self.debug_users: Set[int] = set()
        self.global_chat_sessions: LRUCache = LRUCache(max_size=10)
        self.freewill_busy_profiles: Set[Tuple[int, int, str]] = set()
        self.purged_message_ids: Set[int] = set()
        self.pending_child_confirmations: Dict[str, Any] = {}
        self.global_blacklist: Set[int] = set()
        self._load_blacklist()
        self.session_last_accessed = {}
        self.evict_inactive_sessions_task.start()
        self.message_cooldown = commands.CooldownMapping.from_cooldown(5, 60.0, commands.BucketType.user)
        self.processed_child_messages: LRUCache = LRUCache(max_size=25)
        self.all_bot_ids: Set[int] = set()
        self.image_gen_semaphore = asyncio.Semaphore(1)
        self.ltm_recall_history: Dict[Any, Dict[str, Tuple[int, float]]] = {}
        self.child_bot_edit_cooldowns: Dict[str, List[float]] = {}

        # FIXED: Priority Queues for Premium Fast-Lane
        self.image_request_queue = asyncio.PriorityQueue(maxsize=10)
        self.text_request_queue = asyncio.PriorityQueue()
        
        self.image_gen_workers = []
        self.image_finisher_worker_task = None
        self.active_session_config_views: Dict[int, ui.View] = {}
        self.background_tasks = set()
        self.child_bot_single_sessions = {}
        
        # [NEW] API Key Health & Tier Tracking
        self.api_key_cooldowns: Dict[str, float] = {} # {key: expiry_timestamp}
        
        # Model Stats Initialization
        self.MODELS_DATA_DIR = os.path.join(DATA_DIR, "models")
        os.makedirs(self.MODELS_DATA_DIR, exist_ok=True)

    profile_group = app_commands.Group(name="profile", description="Manage your personal bot profiles (persona, instructions).")

    @profile_group.command(name="create", description="Creates a new, blank personal profile.")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    @app_commands.describe(profile_name="The name for your new profile. Must be unique.")
    async def create_profile_slash(self, interaction: discord.Interaction, profile_name: str):
        await interaction.response.defer(ephemeral=True)

        has_access = await self._has_api_key_access(interaction.user.id)
        if not has_access:
            error_msg = (
                "**Cannot Create Profile**\n"
                "To create profiles, you must have a way to use them. Please do one of the following:\n\n"
                "1. **Join a Server:** Be in a server where an administrator has already configured an API key for MimicAI.\n"
                "2. **Configure Your Server:** If you are a server administrator, use the `/settings` command in a Direct Message with me to add a server-wide API key.\n"
                "3. **Provide a Personal Key:** Use the `/settings` command in a Direct Message with me to add your own personal Google Gemini API key for private use."
            )
            await interaction.followup.send(error_msg, ephemeral=True)
            return

        profile_name = profile_name.lower().strip()
        
        is_valid, err_msg = self._is_valid_profile_name(profile_name)
        if not is_valid:
            await interaction.followup.send(f"❌ **Invalid Name:** {err_msg}", ephemeral=True)
            return

        index = self._get_user_index(interaction.user.id)
        
        if profile_name in index.get("personal", []) or profile_name in index.get("borrowed", []):
            await interaction.followup.send(f"A profile with the name '{profile_name}' already exists.", ephemeral=True)
            return

        # FIXED: Dynamic Limit Check
        current_count = len(index.get("personal", []))
        is_premium = self.is_user_premium(interaction.user.id)
        limit = defaultConfig.LIMIT_PROFILES_PREMIUM if is_premium else defaultConfig.LIMIT_PROFILES_FREE
        
        if current_count >= limit:
            msg = f"**Limit Reached.**\n\n"
            if is_premium:
                msg += f"You have reached the maximum of **{limit}** profiles allowed for Premium users."
            else:
                msg += f"Free tier is limited to **{limit}** personal profiles. You currently have **{current_count}**.\nUpgrade to Premium via `/subscription` to increase this limit to **{defaultConfig.LIMIT_PROFILES_PREMIUM}**."
            
            await interaction.followup.send(msg, ephemeral=True)
            return

        new_profile = self._get_or_create_user_profile(interaction.user.id, profile_name)
        if new_profile:
            # [NEW] Add Creation Timestamp
            config = new_profile.get('config', {})
            config['created_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            self._save_profile_config(interaction.user.id, profile_name, config)
        
        await interaction.followup.send(f"Successfully created new profile '{profile_name}'.\nUse `/profile manage profile_name:{profile_name}` to start editing it.", ephemeral=True)

    @profile_group.command(name="generate", description="Uses AI to generate a new profile from a concept.")
    @app_commands.checks.cooldown(1, 60.0, key=lambda i: i.user.id)
    @app_commands.describe(
        prompt="The character concept (e.g., 'A cynical noir detective').",
        profile_name="The unique internal name for the new profile."
    )
    async def profile_generate_slash(self, interaction: discord.Interaction, prompt: str, profile_name: str):
        await interaction.response.defer(ephemeral=True, thinking=True)
        profile_name = profile_name.lower().strip()

        is_valid, err_msg = self._is_valid_profile_name(profile_name)
        if not is_valid:
            await interaction.followup.send(f"❌ **Invalid Name:** {err_msg}", ephemeral=True)
            return
            
        if not prompt.strip():
            await interaction.followup.send("Prompt cannot be empty.", ephemeral=True)
            return

        index = self._get_user_index(interaction.user.id)
        if profile_name in index.get("personal", []) or profile_name in index.get("borrowed", []):
            await interaction.followup.send(f"A profile with the name '{profile_name}' already exists.", ephemeral=True)
            return

        is_premium = self.is_user_premium(interaction.user.id)
        limit = defaultConfig.LIMIT_PROFILES_PREMIUM if is_premium else defaultConfig.LIMIT_PROFILES_FREE
        
        if len(index.get("personal", [])) >= limit:
            tier_name = "Premium" if is_premium else "Free"
            await interaction.followup.send(f"You have reached the maximum of {limit} personal profiles ({tier_name} tier).", ephemeral=True)
            return

        api_key = self._get_api_key_for_guild(interaction.guild_id)
        if not api_key:
            await interaction.followup.send("This server's API key is not configured, so I cannot generate a profile.", ephemeral=True)
            return

        generation_prompt = self.global_prompts.get("PROFILE_GENERATOR", DEFAULT_PROFILE_GENERATOR_PROMPT).format(prompt=prompt)

        model_name = 'gemini-flash-lite-latest'
        status = "api_error"
        try:
            model = GoogleGenAIModel(api_key=api_key, model_name=model_name, safety_settings=DEFAULT_SAFETY_SETTINGS)
            gen_config = {"temperature": 0.3}
            response = await model.generate_content_async([generation_prompt], generation_config=gen_config)
            
            if not response or not response.candidates:
                raise ValueError("AI returned an empty response, possibly due to a safety filter.")

            response_text = getattr(response, 'text', "").strip()
            
            # Parse the text using the custom delimiters
            sections = re.split(r'\[SECTION:([\w_]+)\]', response_text)
            
            parsed_data = {}
            # Start from index 1 to get the first key, then step by 2
            for i in range(1, len(sections), 2):
                key = sections[i]
                value = sections[i+1].strip()
                parsed_data[key] = value

            generated_data = {
                "persona": {
                    "backstory": parsed_data.get("persona_backstory", ""),
                    "personality_traits": parsed_data.get("persona_personality_traits", ""),
                    "likes": parsed_data.get("persona_likes", ""),
                    "dislikes": parsed_data.get("persona_dislikes", "")
                },
                "ai_instructions": parsed_data.get("ai_instructions", "")
            }

            if not generated_data["persona"]["personality_traits"]:
                 raise ValueError("AI failed to generate content for the 'personality traits' section.")

            new_profile = self._get_or_create_user_profile(interaction.user.id, profile_name)
            if not new_profile:
                await interaction.followup.send("Failed to create the profile structure.", ephemeral=True)
                return

            # Encrypt and save the generated data
            encrypted_persona = {key: [self._encrypt_data(line) for line in value.splitlines()] for key, value in generated_data['persona'].items()}
            encrypted_instructions = self._encrypt_data(generated_data['ai_instructions'])

            prompts = new_profile.get('prompts', {})
            prompts['persona'] = encrypted_persona
            
            if not isinstance(prompts.get('ai_instructions'), list):
                prompts['ai_instructions'] = ["", "", "", ""]
            prompts['ai_instructions'][0] = encrypted_instructions
            
            self._save_profile_prompts(interaction.user.id, profile_name, prompts)

            await interaction.followup.send(f"✅ Successfully generated and created new profile '{profile_name}'.\nUse `/profile manage profile_name:{profile_name}` to view or edit it.", ephemeral=True)

        except json.JSONDecodeError:
            await interaction.followup.send("❌ **Generation Failed:** The AI returned an invalid data format. Please try again.", ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"❌ **Generation Failed:** An error occurred: {e}", ephemeral=True)

    @profile_group.command(name="manage", description="Manage all settings for a specific profile from a unified dashboard.")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    @app_commands.describe(profile_name="The name of the personal or borrowed profile to manage.")
    @app_commands.autocomplete(profile_name=CoreMixin.profile_autocomplete)
    async def manage_profile_slash(self, interaction: discord.Interaction, profile_name: str):
        await interaction.response.defer(ephemeral=True)

        index = self._get_user_index(interaction.user.id)
        is_personal = profile_name in index.get("personal", [])
        is_borrowed = profile_name in index.get("borrowed", [])
        is_default = profile_name == DEFAULT_PROFILE_NAME

        if not is_personal and not is_borrowed and not is_default:
            await interaction.followup.send(f"Profile '{profile_name}' not found.", ephemeral=True)
            return
            
        is_view_borrowed = is_borrowed
        if is_default and interaction.user.id != int(defaultConfig.DISCORD_OWNER_ID):
            is_view_borrowed = True
        
        embed = await self._build_profile_manage_embed(interaction, profile_name)
        view = ProfileManageView(self, interaction, profile_name, is_view_borrowed)

        await interaction.followup.send(embed=embed, view=view, ephemeral=True)

    bulk_group = app_commands.Group(name="bulk", description="Perform actions on multiple profiles at once.", parent=profile_group)

    @bulk_group.command(name="manage", description="Open the dashboard to perform bulk actions.")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    async def bulk_manage_slash(self, interaction: discord.Interaction):
        view = BulkManageView(self, interaction)
        await interaction.response.send_message("Choose a bulk action to perform from the dropdown below.", view=view, ephemeral=True)

    @profile_group.command(name="list", description="Lists all of your saved profile names.")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    async def list_profiles_slash(self, interaction: discord.Interaction): 
        if not self.has_lock and interaction.guild : return
        await interaction.response.defer(ephemeral=True)
        
        index = self._get_user_index(interaction.user.id)
        if not index.get("personal") and not index.get("borrowed"):
            await interaction.followup.send("You have no saved profiles yet.", ephemeral=True)
            return
        
        personal_list_raw = index.get("personal", [])
        borrowed_list_raw = index.get("borrowed", [])
        
        embed = discord.Embed(title=f"Your Profiles", color=discord.Color.purple())
        
        active_in_current_channel = self._get_active_user_profile_name_for_channel(interaction.user.id, interaction.channel_id)
        channel_type_str = "this channel" if interaction.guild else "this DM"
        
        # Helper function to split a list for two columns
        def split_list_for_columns(data_list):
            midpoint = (len(data_list) + 1) // 2
            return data_list[:midpoint], data_list[midpoint:]

        # Process Personal Profiles
        personal_list = []
        for name in sorted(personal_list_raw):
            marker = f" (Active)" if name == active_in_current_channel else ""
            personal_list.append(f"- `{name}`{marker}")
        
        if personal_list:
            col1, col2 = split_list_for_columns(personal_list)
            embed.add_field(name="Personal Profiles", value="\n".join(col1) if col1 else "\u200b", inline=True)
            embed.add_field(name="\u200b", value="\n".join(col2) if col2 else "\u200b", inline=True)

        # Process Borrowed Profiles
        borrowed_list = []
        for name in sorted(borrowed_list_raw):
            b_config = self._get_profile_config(interaction.user.id, name, True) or {}
            owner_id = int(b_config.get("original_owner_id", 0))
            owner = self.bot.get_user(owner_id) or await self.bot.fetch_user(owner_id)
            owner_name = owner.display_name if owner else "Unknown"
            marker = f" (Active)" if name == active_in_current_channel else ""
            borrowed_list.append(f"- `{name}` (from {owner_name}){marker}")

        if borrowed_list:
            # Add a separator if there were personal profiles
            if personal_list:
                embed.add_field(name="\u200b", value="\u200b", inline=False)

            col1, col2 = split_list_for_columns(borrowed_list)
            embed.add_field(name="Borrowed Profiles", value="\n".join(col1) if col1 else "\u200b", inline=True)
            embed.add_field(name="\u200b", value="\n".join(col2) if col2 else "\u200b", inline=True)

        if not personal_list and not borrowed_list:
            embed.description = "You have no personal or borrowed profiles."
        else:
            embed.set_footer(text=f"The (Active) tag shows your active profile in {channel_type_str}.")
            
        await interaction.followup.send(embed=embed, ephemeral=True)

    @app_commands.command(name="export", description="Export selected profiles and memories to a plaintext file (DM Only).")
    @app_commands.checks.cooldown(1, 60.0, key=lambda i: i.user.id)
    @app_commands.dm_only()
    async def export_command(self, interaction: discord.Interaction):
        view = BulkExportView(self, interaction.user.id)
        await interaction.response.send_message("### 📤 Profile Export\nSelect profiles and components to export. **Warning:** The file will contain decrypted data.", view=view, ephemeral=True)

    @app_commands.command(name="import", description="Import profiles and memories from a MimicAI export file (DM Only).")
    @app_commands.checks.cooldown(1, 30.0, key=lambda i: i.user.id)
    @app_commands.dm_only()
    @app_commands.describe(file="The .json or .mimic file exported from a MimicAI instance.")
    async def import_command(self, interaction: discord.Interaction, file: discord.Attachment):
        if not file.filename.endswith(('.json', '.mimic')):
            await interaction.response.send_message("❌ Invalid file type. Please upload a `.json` or `.mimic` file.", ephemeral=True)
            return
        
        await interaction.response.defer(ephemeral=True, thinking=True)
        await self._execute_import(interaction, file)

    @app_commands.command(name="debug", description="Toggle debug mode to see the bot's context for all scopes in your DMs (DM-Only).")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    @app_commands.dm_only()
    @app_commands.describe(mode="Turn the debug view on or off.")
    async def debug_slash(self, interaction: discord.Interaction, mode: Literal['on', 'off']):
        if mode == 'on':
            self.debug_users.add(interaction.user.id)
            await interaction.response.send_message("Universal debug mode **ENABLED**. You will now receive the bot's context in your DMs when you trigger a turn in any scope.", ephemeral=True)
        else:
            self.debug_users.discard(interaction.user.id)
            await interaction.response.send_message("Universal debug mode **DISABLED**.", ephemeral=True)

    data_group = app_commands.Group(name="data", description="Manage LTM and Training data for your profiles.", parent=profile_group)

    @data_group.command(name="manage", description="Manage LTM (all profiles) and Training Examples (personal profiles).")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    @app_commands.describe(profile_name="The name of the profile to manage data for.")
    @app_commands.autocomplete(profile_name=CoreMixin.profile_autocomplete)
    async def data_manage_slash(self, interaction: discord.Interaction, profile_name: str):
        if not self.has_lock: return
        
        profile_name = profile_name.lower().strip()
        index = self._get_user_index(interaction.user.id)
        owner_id = int(defaultConfig.DISCORD_OWNER_ID)

        is_borrowed = profile_name in index.get("borrowed", [])
        is_personal = profile_name in index.get("personal", [])
        
        can_manage = is_personal or is_borrowed or profile_name == DEFAULT_PROFILE_NAME
        if not can_manage:
            await interaction.response.send_message(f"You do not have a profile named '{profile_name}'.", ephemeral=True)
            return

        is_view_borrowed = is_borrowed
        if profile_name == DEFAULT_PROFILE_NAME and interaction.user.id != owner_id:
            is_view_borrowed = True

        effective_owner_id = interaction.user.id
        if is_view_borrowed:
            if profile_name == DEFAULT_PROFILE_NAME:
                effective_owner_id = owner_id
            else:
                b_config = self._get_profile_config(interaction.user.id, profile_name, True) or {}
                effective_owner_id = int(b_config.get("original_owner_id", 0))

        await interaction.response.defer(ephemeral=True)
        view = DataManageView(self, interaction, profile_name, is_borrowed=is_view_borrowed, effective_owner_id=effective_owner_id)
        await view.start()

    @profile_group.command(name="hub", description="The unified dashboard for managing profiles, sharing, and the public library.")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    async def hub_slash(self, interaction: discord.Interaction):
        # [FIX] Defer immediately to prevent interaction timeout (Error 10062)
        await interaction.response.defer(ephemeral=True)
        
        # Lazy Deletion Check (Disk I/O) happens safely after defer
        removed = await self._validate_and_clean_borrowed_profiles(interaction.user.id)

        view = HubHomeView(self, interaction)
        await view.update_display()

        if removed > 0:
            # Use followup because response was already deferred/used
            await interaction.followup.send(f"ℹ️ Notice: {removed} borrowed profiles were removed because their original creators deleted them.", ephemeral=True)

    session_group = app_commands.Group(name="session", description="Manage chat sessions.", guild_only=True)

    @session_group.command(name="config", description="Configure a chat session (Regular or Freewill).")
    @app_commands.describe(mode="The type of session interface to open.")
    @app_commands.choices(mode=[
        app_commands.Choice(name="Regular", value="regular"),
        app_commands.Choice(name="Freewill", value="freewill")
    ])
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    async def session_config_slash(self, interaction: discord.Interaction, mode: str):
        if not self.has_lock: return
        
        # [NEW] Global Admin Check for all session config modes
        if not (interaction.user.guild_permissions.administrator or interaction.user.id == int(defaultConfig.DISCORD_OWNER_ID)):
            await interaction.response.send_message("You must be a server administrator to configure sessions.", ephemeral=True)
            return

        # Invalidate previous session configuration view for this user
        if interaction.user.id in self.active_session_config_views:
            try:
                self.active_session_config_views[interaction.user.id].stop()
            except Exception:
                pass

        if mode == "freewill":
            # [NEW] Immediate Premium Gate
            if not self.is_user_premium(interaction.user.id):
                await interaction.response.send_message(
                    "**Premium Required.**\n\n"
                    "The Freewill System allows profiles to act autonomously and react to server events.\n"
                    "This is a Premium feature. Use `/subscription` to upgrade.",
                    ephemeral=True
                )
                return

            # Teardown any existing regular (multi) session
            session = self.multi_profile_channels.get(interaction.channel_id)
            if session and session.get("type") == "multi":
                for participant in session.get("profiles", []):
                    if participant.get("method") == "child_bot":
                        bot_id = participant.get("bot_id")
                        if bot_id:
                            await self.manager_queue.put({
                                "action": "send_to_child", "bot_id": bot_id,
                                "payload": {"action": "session_update_remove", "channel_id": interaction.channel_id}
                            })
                            await self.manager_queue.put({
                                "action": "send_to_child", "bot_id": bot_id,
                                "payload": {"action": "stop_typing", "channel_id": interaction.channel_id}
                            })
                if session.get('worker_task'): self._safe_cancel_task(session['worker_task'])
                self.multi_profile_channels.pop(interaction.channel_id, None)
                self._save_multi_profile_sessions()

            try:
                import shutil
                dummy_key = (interaction.channel_id, None, None)
                multi_path = self._get_session_dir_path(dummy_key, "multi")
                if multi_path.exists():
                    shutil.rmtree(multi_path)
            except Exception as e:
                print(f"Error cleaning up multi session dir: {e}")

            # Initialize an empty Freewill session immediately to block child bot interactions
            self.multi_profile_channels[interaction.channel_id] = {
                "type": "freewill", "freewill_mode": "reactive", "chat_sessions": {},
                "unified_log": [], "is_hydrated": True, "owner_id": interaction.user.id,
                "is_running": False, "task_queue": asyncio.Queue(), "worker_task": None,
                "profiles": [],
                "turns_since_last_ltm": 0
            }

            # Default to Reactive if not already in a list
            config = self.freewill_config.setdefault(str(interaction.guild.id), {})
            lid = config.setdefault("living_channel_ids", [])
            lud = config.setdefault("lurking_channel_ids", [])
            
            if interaction.channel_id not in lid and interaction.channel_id not in lud:
                lud.append(interaction.channel_id)
                self._save_channel_settings()

            await interaction.response.defer(ephemeral=True)
            view = FreewillHomeView(self, interaction)
            self.active_session_config_views[interaction.user.id] = view
            await view.update_display()
            return
        
        if mode == "regular":
            try:
                import shutil
                dummy_key = (interaction.channel_id, None, None)
                fw_path = self._get_session_dir_path(dummy_key, "freewill")
                if fw_path.exists():
                    shutil.rmtree(fw_path)
            except Exception as e:
                print(f"Error cleaning up freewill session dir: {e}")

            index = self._get_user_index(interaction.user.id)
            personal_profiles = index.get("personal", [])
            borrowed_profiles = index.get("borrowed", [])
            all_selectable_profiles = list(personal_profiles) + list(borrowed_profiles)

            if len(all_selectable_profiles) < 1:
                await interaction.response.send_message("You must have at least one personal or borrowed profile to start a session.", ephemeral=True)
                return
            
            ch_id = interaction.channel_id
            guild_id_str = str(interaction.guild.id)

            # Teardown any existing freewill session with child bot notification
            session = self.multi_profile_channels.get(ch_id)
            if session and session.get("type") == "freewill":
                for participant in session.get("profiles", []):
                    if participant.get("method") == "child_bot":
                        bot_id = participant.get("bot_id")
                        if bot_id:
                            await self.manager_queue.put({
                                "action": "send_to_child", "bot_id": bot_id,
                                "payload": {"action": "session_update_remove", "channel_id": ch_id}
                            })
                self._cleanup_freewill_session(ch_id)

            self.last_freewill_event.pop(ch_id, None)
            self.last_freewill_message_info.pop(ch_id, None)
            
            fw_config = self.freewill_config.get(guild_id_str, {})
            fw_changed = False
            if ch_id in fw_config.get("living_channel_ids", []):
                fw_config["living_channel_ids"].remove(ch_id)
                fw_changed = True
            if ch_id in fw_config.get("lurking_channel_ids", []):
                fw_config["lurking_channel_ids"].remove(ch_id)
                fw_changed = True
            
            if fw_changed:
                self._save_channel_settings()

            current_profiles = []
            current_prompt = None
            current_mode = 'sequential'
            current_audio_mode = 'text-only'
            
            if ch_id in self.multi_profile_channels:
                session = self.multi_profile_channels[ch_id]
                if session.get("type") in ["multi", None]:
                    current_profiles = list(session.get("profiles", []))
                    current_prompt = session.get("session_prompt")
                    current_mode = session.get("session_mode", "sequential")
                    current_audio_mode = session.get("audio_mode", "text-only")

            view = MultiProfileSelectView(self, interaction.user.id, as_admin_scope=True, current_profiles=current_profiles, current_prompt=current_prompt, current_mode=current_mode, current_audio_mode=current_audio_mode)
            self.active_session_config_views[interaction.user.id] = view
            
            await interaction.response.send_message(view.get_ordered_list_message(), view=view, ephemeral=True)

    @session_group.command(name="swap", description="Swaps, adds, or removes a profile from the current session.")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    @app_commands.guild_only()
    @is_admin_or_owner_check()
    @app_commands.autocomplete(profile_name=CoreMixin.profile_autocomplete)
    @app_commands.describe(
        profile_name="The profile to swap or add. Leave blank to remove a profile.",
        use_child_bot="Whether to use the linked Child Bot (if available). Defaults to True.",
        slot="The participant number (1-10) to affect. See '/session swap' with no options for a list."
    )
    async def swap_session_slash(self, interaction: discord.Interaction, profile_name: Optional[str] = None, use_child_bot: Optional[bool] = None, slot: Optional[app_commands.Range[int, 1, 10]] = None):
        await interaction.response.defer(ephemeral=True)
        session = self.multi_profile_channels.get(interaction.channel_id)

        if session and session.get("type") == "freewill":
            await interaction.followup.send("This command is disabled for sessions in Freewill mode. Manage participants via the Freewill dashboard.", ephemeral=True)
            return

        if not profile_name and not slot:
            server_id_str = str(interaction.guild_id) if interaction.guild_id else "dm"
            server_index = self._get_server_index(server_id_str)
            channel_str = str(interaction.channel_id)
            
            active_sessions = server_index.get("active_sessions", {})
            session_data_idx = {}
            if isinstance(active_sessions, dict):
                session_data_idx = active_sessions.get("regular", {}).get(channel_str)
                if not session_data_idx:
                    session_data_idx = active_sessions.get("freewill", {}).get(channel_str)

            if not session_data_idx or not session_data_idx.get("profiles"):
                await interaction.followup.send("There is no active session in this channel.", ephemeral=True)
                return
            
            profile_list = []
            for i, p_data in enumerate(session_data_idx["profiles"]):
                p_name = p_data.get('profile_name')
                pid = p_data.get('pid', 'Unknown PID')
                
                if p_data.get('method') == 'child_bot':
                    display = f"Child Bot ({p_name})"
                else:
                    display = p_name
                profile_list.append(f"**{i+1}.** `{display}` [PID: {pid}]")
            
            owner_user = self.bot.get_user(session_data_idx.get('owner_id'))
            admin_name = owner_user.name if owner_user else "Unknown Admin"
            
            msg = f"**Session Admin:** {admin_name}\n\n**Current Participants:**\n" + "\n".join(profile_list)
            await interaction.followup.send(msg, ephemeral=True)
            return

        # If no session exists, we must have a profile name to start one
        if not session and not profile_name:
            await interaction.followup.send("There is no active session to modify. Provide a profile name to start one.", ephemeral=True)
            return

        # If session exists, check ownership
        if session and session.get("owner_id") != interaction.user.id:
            await interaction.followup.send("You are not the owner of this session and cannot modify its participants.", ephemeral=True)
            return

        # Logic for removing a participant (Session must exist here)
        if not profile_name and slot:
            if len(session["profiles"]) <= 1:
                await interaction.followup.send("You cannot remove the last participant. Use `/suspend` to end the session instead.", ephemeral=True)
                return
            if slot > len(session["profiles"]):
                await interaction.followup.send(f"Invalid slot. There are only {len(session['profiles'])} participants.", ephemeral=True)
                return
            
            removed_participant = session["profiles"].pop(slot - 1)
            session["chat_sessions"].pop((removed_participant['owner_id'], removed_participant['profile_name']), None)
            
            # If removing a child bot, send update
            if removed_participant.get('method') == 'child_bot':
                bot_id = removed_participant.get('bot_id')
                if bot_id:
                    await self.manager_queue.put({
                        "action": "send_to_child", "bot_id": bot_id,
                        "payload": {"action": "session_update_remove", "channel_id": interaction.channel_id}
                    })
                    await self.manager_queue.put({
                        "action": "send_to_child", "bot_id": bot_id,
                        "payload": {"action": "stop_typing", "channel_id": interaction.channel_id}
                    })

            self._save_multi_profile_sessions()
            await interaction.followup.send(f"Removed `{removed_participant['profile_name']}` from the session.", ephemeral=True)
            return

        if profile_name:
            index = self._get_user_index(interaction.user.id)
            is_personal = profile_name in index.get("personal", [])
            is_borrowed = profile_name in index.get("borrowed", [])
            if not is_personal and not is_borrowed:
                await interaction.followup.send(f"You do not have a profile named '{profile_name}'.", ephemeral=True)
                return

            # Resolve Method (Child Bot vs Webhook)
            effective_owner_id = interaction.user.id
            effective_profile_name = profile_name
            if is_borrowed:
                borrowed_data = self._get_profile_config(interaction.user.id, profile_name, True) or {}
                effective_owner_id = int(borrowed_data.get("original_owner_id", interaction.user.id))
                effective_profile_name = borrowed_data.get("original_profile_name", profile_name)

            linked_bot_id = next((bot_id for bot_id, data in self.child_bots.items() if data.get("owner_id") == effective_owner_id and data.get("profile_name") == effective_profile_name), None)
            is_bot_in_guild = linked_bot_id and interaction.guild.get_member(int(linked_bot_id))

            method = "webhook"
            bot_id_to_use = None
            
            # Logic: Default to child bot if available, unless explicitly set to False
            should_use_bot = True if use_child_bot is None else use_child_bot
            
            if should_use_bot and is_bot_in_guild:
                method = "child_bot"
                bot_id_to_use = linked_bot_id
            
            # Create new participant object for comparison
            new_participant = {
                "owner_id": interaction.user.id, "profile_name": profile_name,
                "method": method, "ephemeral": False
            }
            if bot_id_to_use:
                new_participant["bot_id"] = bot_id_to_use

            # Duplicate Guard: Check if this identity is already in the session
            for p in (session["profiles"] if session else []):
                if p["owner_id"] == new_participant["owner_id"] and p["profile_name"] == new_participant["profile_name"]:
                    await interaction.followup.send(f"Profile `{profile_name}` is already a participant in this session.", ephemeral=True)
                    return

            # Create new session if none exists
            if not session:
                chat_sessions = {(interaction.user.id, profile_name): None}
                session = {
                    "type": "multi", "profiles": [new_participant], "chat_sessions": chat_sessions,
                    "unified_log": [], "is_hydrated": False, "last_bot_message_id": None,
                    "owner_id": interaction.user.id, "is_running": False,
                    "task_queue": asyncio.Queue(),
                    "worker_task": None, "turns_since_last_ltm": 0, "session_prompt": None,
                    "session_mode": "sequential", "pending_image_gen_data": None, "pending_whispers": {}
                }
                self.multi_profile_channels[interaction.channel_id] = session
                
                # Hydrate immediately for the single participant logic below
                session["chat_sessions"][(interaction.user.id, profile_name)] = GoogleGenAIChatSession(history=[])
                session["is_hydrated"] = True
                
                if method == "child_bot":
                    await self.manager_queue.put({
                        "action": "send_to_child", "bot_id": bot_id_to_use,
                        "payload": {"action": "session_update_add", "channel_id": interaction.channel_id}
                    })

                self._save_multi_profile_sessions()
                await interaction.followup.send(f"Started a new regular session with `{profile_name}`.", ephemeral=True)
                return

            # Existing Session Logic (Swap/Add)
            action_description = ""
            old_participant_to_remove = None

            if not slot:
                if len(session["profiles"]) == 1:
                    old_participant = session["profiles"][0]
                    session["profiles"][0] = new_participant
                    old_participant_to_remove = old_participant
                    action_description = f"Swapped session profile to `{profile_name}`."
                elif len(session["profiles"]) >= MAX_MULTI_PROFILES:
                    await interaction.followup.send(f"Session is full ({MAX_MULTI_PROFILES} participants). Please specify a slot to replace.", ephemeral=True)
                    return
                else:
                    session["profiles"].append(new_participant)
                    action_description = f"Added `{profile_name}` to the session."
            else:
                target_index = min(slot - 1, len(session["profiles"]))
                is_insertion = (target_index == len(session["profiles"]))

                if is_insertion:
                    if len(session["profiles"]) >= MAX_MULTI_PROFILES:
                        await interaction.followup.send(f"Session is full ({MAX_MULTI_PROFILES} participants). Cannot add another.", ephemeral=True)
                        return
                    session["profiles"].insert(target_index, new_participant)
                    action_description = f"Added `{profile_name}` to session slot {target_index + 1}."
                else:
                    old_participant = session["profiles"][target_index]
                    session["profiles"][target_index] = new_participant
                    old_participant_to_remove = old_participant
                    action_description = f"Replaced slot {target_index + 1} with `{profile_name}`."

            # Handle Child Bot Updates (Add New)
            if method == "child_bot":
                await self.manager_queue.put({
                    "action": "send_to_child", "bot_id": bot_id_to_use,
                    "payload": {"action": "session_update_add", "channel_id": interaction.channel_id}
                })

            # Handle Child Bot Updates (Remove Old)
            if old_participant_to_remove:
                # Remove old chat session from memory
                session["chat_sessions"].pop((old_participant_to_remove['owner_id'], old_participant_to_remove['profile_name']), None)
                
                if old_participant_to_remove.get('method') == 'child_bot':
                    old_bot_id = old_participant_to_remove.get('bot_id')
                    if old_bot_id:
                        await self.manager_queue.put({
                            "action": "send_to_child", "bot_id": old_bot_id,
                            "payload": {"action": "session_update_remove", "channel_id": interaction.channel_id}
                        })
                        await self.manager_queue.put({
                            "action": "send_to_child", "bot_id": old_bot_id,
                            "payload": {"action": "stop_typing", "channel_id": interaction.channel_id}
                        })

            # Initialize chat session for new participant
            new_participant_key = (new_participant['owner_id'], new_participant['profile_name'])
            
            # Rebuild history from unified log for the new participant
            participant_history = []
            if session.get("is_hydrated"):
                for turn in session.get("unified_log", []):
                    speaker_key = tuple(turn.get("speaker_key", []))
                    role = 'model' if speaker_key == new_participant_key else 'user'
                    participant_history.append({'role': role, 'parts': [turn.get("content")]})
            
            session["chat_sessions"][new_participant_key] = GoogleGenAIChatSession(history=participant_history)

            self._save_multi_profile_sessions()
            await interaction.followup.send(action_description, ephemeral=True)
            return

    @session_group.command(name="trigger", description="Manually triggers a new round of the current regular session.")
    @app_commands.checks.cooldown(2, 10.0, key=lambda i: i.user.id)
    @app_commands.guild_only()
    async def trigger_session_slash(self, interaction: discord.Interaction):
        if not self.has_lock: return
        await interaction.response.defer(ephemeral=True)

        session = self.multi_profile_channels.get(interaction.channel_id)
        if not session:
            await interaction.followup.send("No active session found in this channel.", ephemeral=True)
            return

        if session.get("type") == "freewill":
            await interaction.followup.send("Manual triggers are disabled for Freewill sessions.", ephemeral=True)
            return

        if session.get("owner_id") != interaction.user.id and not interaction.user.guild_permissions.administrator:
            await interaction.followup.send("Only the session owner or a server administrator can trigger a round.", ephemeral=True)
            return

        # Trigger logic: push a null trigger to simulate automated continuation
        await session['task_queue'].put(None)
        if not session.get('worker_task') or session['worker_task'].done():
            task = self.bot.loop.create_task(self._multi_profile_worker(interaction.channel_id))
            session['worker_task'] = task
            self.background_tasks.add(task)

        await interaction.followup.send("Round triggered.", ephemeral=True)

    @session_group.command(name="view", description="View details of the current session and its participants.")
    @app_commands.checks.cooldown(5, 10.0, key=lambda i: i.user.id)
    async def session_view_slash(self, interaction: discord.Interaction):
        if not self.has_lock: return
        
        server_id_str = str(interaction.guild_id) if interaction.guild_id else "dm"
        server_index = self._get_server_index(server_id_str)
        channel_str = str(interaction.channel_id)
        
        active_sessions = server_index.get("active_sessions", {})
        session_data_idx = None
        is_freewill = False
        
        if isinstance(active_sessions, dict):
            if channel_str in active_sessions.get("freewill", {}):
                session_data_idx = active_sessions["freewill"][channel_str]
                is_freewill = True
            elif channel_str in active_sessions.get("regular", {}):
                session_data_idx = active_sessions["regular"][channel_str]

        if not session_data_idx:
            await interaction.response.send_message("No active session in this channel.", ephemeral=True)
            return

        type_display = "Freewill (Proactive/Reactive)" if is_freewill else "Regular (Multi-Profile)"
        
        owner_id = session_data_idx.get("owner_id")
        owner = self.bot.get_user(owner_id)
        owner_name = owner.name if owner else f"ID: {owner_id}"
        
        profiles_for_display = []
        
        if is_freewill:
            # For Freewill, show all opted-in profiles regardless of current scene status
            server_data = self.freewill_participation.get(str(interaction.guild.id), {})
            channel_data = server_data.get(str(interaction.channel.id), {})
            
            for user_id_str, user_profiles in channel_data.items():
                member = interaction.guild.get_member(int(user_id_str))
                if not member: continue

                for profile_name, settings in user_profiles.items():
                    if settings.get("personality", "off") != "off":
                        # Pass the channel object to the helper
                        p_dict = self._build_freewill_participant_dict(int(user_id_str), profile_name, interaction.channel)
                        if p_dict:
                            p_dict['pid'] = self._get_pid_from_name_any(int(user_id_str), profile_name)
                            profiles_for_display.append(p_dict)
        else:
            # For Regular, use the session's profile list from the index
            profiles_for_display = session_data_idx.get("profiles", [])

        participant_count = len(profiles_for_display)
        
        embed = discord.Embed(title=f"Session Info: #{interaction.channel.name}", color=discord.Color.gold())
        embed.add_field(name="Session Type", value=type_display, inline=True)
        embed.add_field(name="Session Admin", value=owner_name, inline=True)
        embed.add_field(name="Participants", value=str(participant_count), inline=True)
        
        if session_data_idx.get("session_prompt"):
            prompt_val = session_data_idx["session_prompt"]
            embed.add_field(name="Prompt", value=prompt_val[:200] + "..." if len(prompt_val) > 200 else prompt_val, inline=False)

        # Create a temporary session dict for the View to use
        session_view_data = {
            "type": "freewill" if is_freewill else "multi",
            "owner_id": owner_id,
            "profiles": profiles_for_display
        }

        view = SessionView(self, interaction, session_view_data)
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

    @app_commands.command(name="refresh", description="Clears the bot's short-term memory for the current context.")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    @app_commands.guild_only()
    async def refresh_slash(self, interaction: discord.Interaction):
        if not self.has_lock: return
        await interaction.response.defer(ephemeral=True)

        ch_id = interaction.channel_id
        session = self.multi_profile_channels.get(ch_id)

        if not session:
            await interaction.followup.send("There is no active session in this channel to refresh.", ephemeral=True)
            return

        if session.get("owner_id") != interaction.user.id and not interaction.user.guild_permissions.administrator:
            await interaction.followup.send("You must be the session owner or a server administrator to refresh the memory.", ephemeral=True)
            return
        
        for participant in session.get("profiles", []):
            if participant.get("method") == "child_bot":
                bot_id = participant.get("bot_id")
                if bot_id:
                    await self.manager_queue.put({
                        "action": "send_to_child", "bot_id": bot_id,
                        "payload": {"action": "stop_typing", "channel_id": ch_id}
                    })

        session_type = session.get("type", "multi")
        dummy_session_key = (ch_id, None, None)
        await self._delete_session_from_disk(dummy_session_key, session_type)

        for p_key in session["chat_sessions"].keys():
            session["chat_sessions"][p_key] = GoogleGenAIChatSession(history=[])
        
        # [NEW] Reset counters for all participants
        for p in session.get('profiles', []):
            p['ltm_counter'] = 0
            # Also reset the global counter for this profile in this guild
            ltm_counter_key = (p['owner_id'], p['profile_name'], "guild")
            self.message_counters_for_ltm.pop(ltm_counter_key, None)
        
        # [NEW] Reset LTM recall history (penalty system) for this channel
        for p_key in session.get("chat_sessions", {}).keys():
            owner_id, profile_name = p_key
            full_session_key = (ch_id, owner_id, profile_name)
            self.ltm_recall_history.pop(full_session_key, None)

        session['is_hydrated'] = True
        session['unified_log'] = []

        if session.get('worker_task'):
            self._safe_cancel_task(session['worker_task'])
            session['worker_task'] = None
            
        while not session['task_queue'].empty():
            try:
                session['task_queue'].get_nowait()
                session['task_queue'].task_done()
            except asyncio.QueueEmpty:
                break
        
        await interaction.followup.send("The session memory for this channel has been cleared. The conversation will start from scratch.", ephemeral=True)

    @app_commands.command(name="shutdown", description="Gracefully shuts down this bot instance (Bot Owner Only).")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    @app_commands.dm_only()
    @is_owner_in_dm_check()
    async def shutdown_slash(self, interaction: discord.Interaction):
        if interaction.user.id != int(defaultConfig.DISCORD_OWNER_ID):
            await interaction.response.send_message("This command is restricted to the Bot Owner.", ephemeral=True)
            return

        view = ShutdownConfirmView(self)
        await interaction.response.send_message("Are you sure you want to shut down this bot instance?", view=view, ephemeral=True)

    @app_commands.command(name="suspend", description="Suspends the bot in this channel, clearing the session (Admin Only).")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    @app_commands.guild_only()
    @is_admin_or_owner_check()
    async def suspend_slash(self, interaction: discord.Interaction):
        if not self.has_lock: return
        await interaction.response.defer(ephemeral=True)
        
        ch_id = interaction.channel_id
        session = self.multi_profile_channels.pop(ch_id, None)
        
        # Also remove from Freewill config to prevent auto-restart
        guild_id_str = str(interaction.guild.id)
        fw_config = self.freewill_config.get(guild_id_str, {})
        fw_changed = False
        if ch_id in fw_config.get("living_channel_ids", []):
            fw_config["living_channel_ids"].remove(ch_id)
            fw_changed = True
        if ch_id in fw_config.get("lurking_channel_ids", []):
            fw_config["lurking_channel_ids"].remove(ch_id)
            fw_changed = True
        if fw_changed:
            self._save_channel_settings()

        if not session:
            if fw_changed:
                await interaction.followup.send("No active session was found, but Freewill has been disabled for this channel.", ephemeral=True)
            else:
                await interaction.followup.send("There is no active session in this channel to suspend.", ephemeral=True)
            return

        # [NEW] Robust Counter Cleanup
        for participant in session.get("profiles", []):
            p_oid = participant.get("owner_id")
            p_name = participant.get("profile_name")
            if p_oid and p_name:
                # Clear the round counter for this specific profile
                self.message_counters_for_ltm.pop((p_oid, p_name, "guild"), None)
                
                # [NEW] Reset LTM recall history (penalty system) for this channel
                full_session_key = (ch_id, p_oid, p_name)
                self.ltm_recall_history.pop(full_session_key, None)

            if participant.get("method") == "child_bot":
                bot_id = participant.get("bot_id")
                if bot_id:
                    await self.manager_queue.put({
                        "action": "send_to_child", "bot_id": bot_id,
                        "payload": {"action": "session_update_remove", "channel_id": ch_id}
                    })
                    await self.manager_queue.put({
                        "action": "send_to_child", "bot_id": bot_id,
                        "payload": {"action": "stop_typing", "channel_id": ch_id}
                    })

        session_type = session.get("type", "multi")
        dummy_session_key = (ch_id, None, None)
        await self._delete_session_from_disk(dummy_session_key, session_type)

        if session.get('worker_task'):
            self._safe_cancel_task(session['worker_task'])
        
        self.session_last_accessed.pop(ch_id, None)
        self._save_multi_profile_sessions()

        await interaction.followup.send(f"Session suspended for {interaction.channel.mention} and Freewill triggers disabled. The bot will be silent until mentioned or configured again.", ephemeral=True)

    @app_commands.command(name="purge", description="Purges messages and memory, or gracefully dehydrates the session (Admin Only).")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    @app_commands.guild_only()
    @is_admin_or_owner_check()
    @app_commands.describe(amount="Messages to delete (1-100). Leave blank to gracefully dehydrate the session.")
    async def purge_slash(self, interaction: discord.Interaction, amount: Optional[app_commands.Range[int,1,100]] = None):
        if not self.has_lock : return
        await interaction.response.defer(ephemeral=True)

        if not isinstance(interaction.channel, (discord.TextChannel, discord.Thread)):
            await interaction.followup.send("Purge command not supported in this channel type.", ephemeral=True)
            return
            
        app_perms = interaction.app_permissions
        if not app_perms or not app_perms.manage_messages:
            await interaction.followup.send("I lack 'Manage Messages' permission.", ephemeral=True); return

        if amount is None:
            if interaction.channel_id in self.multi_profile_channels:
                self.session_last_accessed[interaction.channel_id] = 0
                await interaction.followup.send("Session marked for graceful dehydration. It will be unloaded from RAM when inactive.", ephemeral=True)
            else:
                await interaction.followup.send("No active session found in RAM to dehydrate.", ephemeral=True)
            return

        def check_and_track(m):
            self.purged_message_ids.add(m.id)
            return True

        session_lock = self.multi_profile_channels.get(interaction.channel_id)
        if session_lock:
            # Re-hydrate immediately before executing the Discord API purge
            session_type = session_lock.get("type", "multi")
            if not session_lock.get("is_hydrated"):
                session_lock = await self._ensure_session_hydrated(interaction.channel_id, session_type)
                
            session_lock['is_purging'] = True
            while session_lock.get('is_running') or session_lock.get('is_regenerating'):
                await asyncio.sleep(0.5)

        try:
            messages_to_delete = await interaction.channel.purge(limit=amount, check=check_and_track, before=interaction.created_at, reason=f"Purge by {interaction.user}")
            
            progress_message = await interaction.followup.send(f"Deleted {len(messages_to_delete)} message(s). Now cleaning them from my memory...", ephemeral=True)

            session = self.multi_profile_channels.get(interaction.channel_id)
            cleaned_turns_count = 0

            if session:
                session_type = session.get("type", "multi")
                
                deleted_msg_ids = {m.id for m in messages_to_delete}
                turn_ids_to_delete = set()

                # Find which turns contain the deleted message IDs
                for turn in session.get("unified_log", []):
                    turn_msg_ids = turn.get("message_ids", [])
                    if any(mid in deleted_msg_ids for mid in turn_msg_ids):
                        turn_ids_to_delete.add(turn.get("turn_id"))

                if turn_ids_to_delete:
                    cleaned_turns_count = len(turn_ids_to_delete)
                    
                    # Decrement Individual Counters
                    for t_id in turn_ids_to_delete:
                        turn_obj = next((turn for turn in session.get("unified_log", []) if turn.get("turn_id") == t_id), None)
                        if turn_obj and turn_obj.get("is_user") is False:
                            bot_pid = turn_obj.get("speaker_pid")
                            for p in session.get('profiles', []):
                                if self._get_pid_from_name_any(p['owner_id'], p['profile_name']) == bot_pid:
                                    p['ltm_counter'] = max(0, p.get('ltm_counter', 0) - 1)
                                    break

                    original_log_len = len(session.get("unified_log", []))
                    session["unified_log"] = [
                        turn for turn in session.get("unified_log", [])
                        if turn.get("turn_id") not in turn_ids_to_delete
                    ]

                    if len(session["unified_log"]) < original_log_len:
                        is_effectively_empty = not session.get("unified_log") or all(
                            turn.get("type") in ["whisper", "private_response"] for turn in session.get("unified_log", [])
                        )
                        
                        dummy_session_key = (interaction.channel_id, None, None)
                        if is_effectively_empty:
                            await self._delete_session_from_disk(dummy_session_key, session_type)
                            for p_key in session.get("chat_sessions", {}).keys():
                                owner_id, profile_name = p_key
                                full_session_key = (interaction.channel_id, owner_id, profile_name)
                                self.ltm_recall_history.pop(full_session_key, None)
                        else:
                            await self._save_session_to_disk(dummy_session_key, session_type, session["unified_log"])

                        # Now that the correct state is on disk, force a re-read and rebuild
                        session["is_hydrated"] = False
                        await self._ensure_session_hydrated(interaction.channel_id, session_type)

                    self.session_last_accessed[interaction.channel_id] = time.time()

            await progress_message.edit(content=f"Deleted {len(messages_to_delete)} message(s) and cleaned {cleaned_turns_count} turn(s) from memory.")

        except Exception as e:
            await interaction.followup.send(f"An error occurred during purge: {e}", ephemeral=True)
            traceback.print_exc()
        finally:
            if session_lock:
                session_lock['is_purging'] = False

    @profile_group.command(name="global_chat", description="Have a persistent, private conversation with a profile.")
    @app_commands.checks.cooldown(5, 60.0, key=lambda i: i.user.id)
    @app_commands.autocomplete(profile_name=CoreMixin.global_chat_profile_autocomplete)
    @app_commands.describe(
        profile_name="The profile to chat with. Leave blank to view private history.",
        refresh="Set to True to clear your conversation history with this profile.",
        suspend="Set to True to permanently delete ALL global chat histories for every profile."
    )
    async def global_chat_slash(self, interaction: discord.Interaction, profile_name: Optional[str] = None, refresh: Optional[bool] = False, suspend: Optional[bool] = False):
        user_id = interaction.user.id
        
        if suspend:
            await interaction.response.defer(ephemeral=True)
            import shutil
            
            try:
                keys_to_del = [k for k in self.global_chat_sessions.keys() if isinstance(k, tuple) and len(k) == 3 and k[0] == 'global' and k[1] == user_id]
                for k in keys_to_del:
                    self.global_chat_sessions.pop(k, None)
                    self.session_last_accessed.pop(k, None)
                    self.ltm_recall_history.pop(k, None)

                dir_path = pathlib.Path(self.USERS_DIR) / str(user_id) / "profiles"
                if dir_path.is_dir():
                    for p_dir in dir_path.iterdir():
                        if p_dir.is_dir():
                            gc_file = p_dir / "global_chat.json.gz"
                            if gc_file.exists():
                                try: gc_file.unlink()
                                except: pass
            except Exception as e:
                print(f"Error suspending global chat for {user_id}: {e}")

            await interaction.followup.send("✅ All global conversation histories have been permanently deleted.", ephemeral=True)
            return

        profile_name_lower = profile_name.lower().strip() if profile_name else None

        if refresh and profile_name_lower:
            await interaction.response.defer(ephemeral=True)
            session_key = ('global', user_id, profile_name_lower)
            
            self.global_chat_sessions.pop(session_key, None)
            self.session_last_accessed.pop(session_key, None)
            self.ltm_recall_history.pop(session_key, None)

            await self._delete_session_from_disk(session_key, 'global_chat')
            
            await interaction.followup.send(f"Your global chat history with '{profile_name_lower}' has been cleared.", ephemeral=True)
            return

        if not profile_name_lower:
            await interaction.response.defer(ephemeral=True)
            view = GlobalChatHistoryView(self, interaction, user_id)
            await view.initialize()
            if not view.available_profiles:
                await interaction.followup.send("You have no active global chat histories.", ephemeral=True)
            else:
                await interaction.followup.send(embed=view.get_embed(), view=view, ephemeral=True)
            return

        index = self._get_user_index(user_id)
        is_personal = profile_name_lower in index.get("personal", [])
        is_borrowed = profile_name_lower in index.get("borrowed", [])
        if not is_personal and not is_borrowed:
            await interaction.response.send_message(f"Profile '{profile_name_lower}' not found.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=False)

        model_cache_key = ('global', user_id, profile_name_lower)
        session_data = self.global_chat_sessions.get(model_cache_key)
        if not session_data:
            session_data = await self._load_session_from_disk(model_cache_key, 'global_chat')
            if not session_data:
                chat = GoogleGenAIChatSession(history=[])
                session_data = {'chat_session': chat, 'unified_log': []}
            self.global_chat_sessions[model_cache_key] = session_data

        view = GlobalChatPlayView(self, interaction, user_id, profile_name_lower)
        await view.initialize()
        await interaction.followup.send(embed=view.get_embed(), view=view)

    @app_commands.command(name="clear", description="Clears all of the bot's messages from this DM channel.")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    @app_commands.dm_only()
    async def clear_dm_slash(self, interaction: discord.Interaction):
        # Defer immediately to acknowledge the interaction within Discord's 3-second window.
        await interaction.response.defer(ephemeral=True)
        
        dm_channel = interaction.channel
        deleted_count = 0
        
        try:
            while True:
                messages_to_delete = []
                async for message in dm_channel.history(limit=100):
                    if message.author.id == self.bot.user.id:
                        messages_to_delete.append(message)
                
                if not messages_to_delete:
                    break # No more messages to delete

                # Delete messages in bulk (as much as the API allows for DMs)
                # This is faster than one by one with a sleep
                delete_tasks = [msg.delete() for msg in messages_to_delete]
                results = await asyncio.gather(*delete_tasks, return_exceptions=True)
                
                for result in results:
                    if not isinstance(result, Exception):
                        deleted_count += 1

                if len(messages_to_delete) < 100:
                    break # Reached the end of the history
            
            await interaction.followup.send(f"Successfully deleted {deleted_count} of my messages from this DM.", ephemeral=True)

        except Exception as e:
            print(f"Error during DM clear for user {interaction.user.id}: {e}")
            traceback.print_exc()
            if not interaction.response.is_done():
                await interaction.followup.send("An unexpected error occurred while trying to clear my messages.", ephemeral=True)
            
    @app_commands.command(name="viewavatar", description="Displays the avatar of a specified user.")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    @app_commands.describe(user="The user whose avatar you want to view.")
    async def viewavatar_slash(self, interaction: discord.Interaction, user: discord.User):
        await interaction.response.defer(ephemeral=True)
        
        embed = discord.Embed(title=f"Avatar for {user.display_name}")
        if user.display_avatar:
            embed.set_image(url=user.display_avatar.url)
            embed.description = f"[Link to Avatar]({user.display_avatar.url})"
        else:
            embed.description = "This user does not have a displayable avatar."
            
        await interaction.followup.send(embed=embed, ephemeral=True)

    @app_commands.command(name="documentation", description="Interactive guide for prompt engineering, intelligence, and generation parameters.")
    @app_commands.checks.cooldown(5, 60.0, key=lambda i: i.user.id)
    async def documentation_slash(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        from .mixins.content import DOC_CATEGORIES
        view = DropdownContentView(DOC_CATEGORIES, "MimicAI Advanced Documentation")
        await interaction.followup.send(embed=view.get_embed(), view=view, ephemeral=True)

    @app_commands.command(name="transparency", description="Displays info about background AI models and their API costs.")
    @app_commands.checks.cooldown(1, 30.0, key=lambda i: i.user.id)
    async def transparency_slash(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        from .mixins.content import TRANSPARENCY_CATEGORIES
        view = DropdownContentView(TRANSPARENCY_CATEGORIES, "Transparency & Pricing", link_button_label="Official Google Pricing Page", link_button_url="https://ai.google.dev/gemini-api/docs/pricing")
        await interaction.followup.send(embed=view.get_embed(), view=view, ephemeral=True)

    @app_commands.command(name="help", description="Displays detailed documentation about the bot's features and commands.")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    async def help_slash(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        from .mixins.content import HELP_CATEGORIES
        view = DropdownContentView(HELP_CATEGORIES, "MimicAI Help & Documentation")
        await interaction.followup.send(embed=view.get_embed(), view=view, ephemeral=True)

    @app_commands.command(name="terms", description="View the MimicAI Terms of Service and Privacy Policy.")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    async def terms_slash(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        from .mixins.content import TERMS_CATEGORIES_OFFICIAL, TERMS_CATEGORIES_SELF_HOSTED
        
        OFFICIAL_BOT_ID = 1376696185947164854
        
        if self.bot.user and self.bot.user.id == OFFICIAL_BOT_ID:
            categories_to_use = TERMS_CATEGORIES_OFFICIAL
        else:
            categories_to_use = TERMS_CATEGORIES_SELF_HOSTED
            
        view = DropdownContentView(categories_to_use, "Terms & Privacy", link_button_label="Open Official Website", link_button_url="https://mimic-ai.org/")
        await interaction.followup.send(embed=view.get_embed(), view=view, ephemeral=True)

    @app_commands.command(name="settings", description="Manage API keys and your personal child bots (DM-Only).")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    @app_commands.dm_only()
    async def settings_slash(self, interaction: discord.Interaction):
        if not self.fernet:
            await interaction.response.send_message("Error: The bot's encryption service is not configured.", ephemeral=True)
            return
            
        await interaction.response.defer(ephemeral=True)
        view = SettingsHomeView(self, interaction)
        await view.update_display()

    def _record_model_usage(self, model_name: str, provider: str):
        if not model_name or provider == "google": return
        
        filename = "openrouter_models.json"
        path = os.path.join(self.MODELS_DATA_DIR, filename)
        
        try:
            data = {}
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        content = f.read()
                        if content.strip():
                            data = json.loads(content)
                except Exception:
                    data = {}
            
            clean_name = model_name.replace("OPENROUTER/", "")
            data[clean_name] = data.get(clean_name, 0) + 1
            
            with open(path, 'wb') as f:
                f.write(json.dumps(data))
        except Exception as e:
            print(f"Error recording model usage: {e}")

    def _log_api_call(self, user_id: int, guild_id: Optional[int], context: str, model_used: str, status: str):
        if status == "success":
            allowed_contexts = [
                'multi_profile', 'global_chat', 'freewill',
                'multi_profile_fallback', 'global_chat_fallback', 'freewill_fallback'
            ]
            if context not in allowed_contexts:
                return

            # Robust provider detection for recording popularity
            is_google = False
            if model_used in get_args(ALLOWED_MODELS):
                is_google = True
            elif model_used.startswith("models/"):
                is_google = True
            elif "gemini" in model_used.lower():
                if "/" in model_used and not model_used.startswith("models/"):
                    is_google = False
                else:
                    is_google = True
            
            if not is_google:
                self._record_model_usage(model_used, "openrouter")

    async def session_participant_autocomplete(self, interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
        server_id_str = str(interaction.guild_id) if interaction.guild_id else "dm"
        server_index = self._get_server_index(server_id_str)
        channel_str = str(interaction.channel_id)
        
        active_sessions = server_index.get("active_sessions", {})
        session_data = {}
        if isinstance(active_sessions, dict):
            session_data = active_sessions.get("regular", {}).get(channel_str)
            if not session_data:
                session_data = active_sessions.get("freewill", {}).get(channel_str)

        if not session_data:
            return []

        choices = []
        current_lower = current.lower()
        for participant in session_data.get("profiles", []):
            owner_id = participant.get("owner_id")
            profile_name = participant.get("profile_name")
            
            # If it's the speak command, only show profiles owned by the command user (unless bot owner)
            if interaction.command.name == "speak":
                if owner_id != interaction.user.id and interaction.user.id != int(defaultConfig.DISCORD_OWNER_ID):
                    continue
            
            value = f"{owner_id}:{profile_name}"

            if current_lower in profile_name.lower():
                choices.append(app_commands.Choice(name=profile_name, value=value))
        
        return choices[:25]
    
    @app_commands.command(name="speak", description="Anonymously speak as one of your profiles (Admin Only).")
    @app_commands.checks.cooldown(2, 10.0, key=lambda i: i.user.id)
    @app_commands.guild_only()
    @is_admin_or_owner_check()
    @app_commands.autocomplete(profile_name=session_participant_autocomplete, method=CoreMixin.speak_method_autocomplete)
    @app_commands.describe(
        profile_name="Your active session profile to speak as.",
        message="The message to send. If omitted, a multi-line input box will appear.",
        method="The method to send the message with. Defaults to 'auto'."
    )
    async def speak_slash(self, interaction: discord.Interaction, profile_name: str, method: Literal['auto', 'webhook', 'child_bot'] = 'auto', message: Optional[str] = None):
        session = self.multi_profile_channels.get(interaction.channel_id)
        if not session:
            await interaction.response.send_message("There is no active session in this channel. The profile must be active in a session.", ephemeral=True)
            return

        try:
            p_owner_id_str, p_name = profile_name.split(":", 1)
            p_owner_id = int(p_owner_id_str)
        except ValueError:
            p_owner_id = interaction.user.id
            p_name = profile_name

        participant = next((p for p in session.get("profiles", []) if p.get("profile_name") == p_name and p.get("owner_id") == p_owner_id), None)
        if not participant:
            await interaction.response.send_message(f"The profile '{p_name}' is not active in this session. It must be added first.", ephemeral=True)
            return

        if p_owner_id != interaction.user.id and interaction.user.id != int(defaultConfig.DISCORD_OWNER_ID):
            await interaction.response.send_message("You can only speak as your own personal or borrowed profiles.", ephemeral=True)
            return

        if message:
            await interaction.response.defer(ephemeral=True)
            await self._execute_speak_as(
                interaction_to_respond=interaction,
                channel=interaction.channel,
                author=interaction.user,
                profile_name=p_name,
                message=message,
                method=method
            )
        else:
            async def modal_callback(modal_interaction: discord.Interaction, message_text: str):
                await modal_interaction.response.defer(ephemeral=True)
                await self._execute_speak_as(
                    interaction_to_respond=interaction,
                    channel=interaction.channel,
                    author=interaction.user,
                    profile_name=p_name,
                    message=message_text,
                    method=method
                )
            
            modal = ActionTextInputModal(
                title=f"Speak as '{p_name}'",
                label="Message Content",
                placeholder="Enter the message to send...",
                on_submit_callback=modal_callback
            )
            await interaction.response.send_modal(modal)

    @app_commands.command(name="whisper", description="Send a private message to a profile in an active multi-profile session.")
    @app_commands.checks.cooldown(3, 30.0, key=lambda i: i.user.id)
    @app_commands.guild_only()
    @app_commands.autocomplete(profile=session_participant_autocomplete)
    @app_commands.describe(
        profile="The participant to whisper to. Leave blank to view history.",
        message="The private message to send. Leave blank to view history."
    )
    async def whisper_slash(self, interaction: discord.Interaction, profile: Optional[str] = None, message: Optional[str] = None):
        session = self.multi_profile_channels.get(interaction.channel_id)
        if not session or session.get("type") not in ["multi", "freewill"]:
            await interaction.response.send_message("This command can only be used in an active multi-profile or freewill session.", ephemeral=True)
            return

        target_participant = None
        if profile:
            try:
                p_owner_id_str, p_name = profile.split(":", 1)
                p_owner_id = int(p_owner_id_str)
            except ValueError:
                p_owner_id = None
                p_name = profile

            target_participant = next((p for p in session.get("profiles", []) if p.get("profile_name") == p_name and (p_owner_id is None or p.get("owner_id") == p_owner_id)), None)

            if not target_participant:
                await interaction.response.send_message(f"The profile '{p_name}' is not an active participant in this session.", ephemeral=True)
                return

        if profile and message:
            await interaction.response.defer(ephemeral=True, thinking=True)
            await self._execute_whisper(interaction, target_participant, message)
        elif profile and not message:
            async def modal_callback(modal_interaction: discord.Interaction, message_text: str):
                await modal_interaction.response.defer(ephemeral=True, thinking=True)
                await self._execute_whisper(modal_interaction, target_participant, message_text)

            modal = ActionTextInputModal(
                title=f"Whisper to {target_participant['profile_name']}",
                label="Whisper Message",
                placeholder="Enter your private message...",
                on_submit_callback=modal_callback
            )
            await interaction.response.send_modal(modal)
        elif not profile and not message:
            await interaction.response.defer(ephemeral=True)
            await self._show_whisper_history(interaction)
        else:
            await interaction.response.send_message("To send a whisper, you must provide both a profile and a message. To view history, provide neither.", ephemeral=True)

    async def _show_whisper_history(self, interaction: discord.Interaction):
        session = self.multi_profile_channels.get(interaction.channel_id)
        if not session: 
            await interaction.followup.send("Session not found.", ephemeral=True)
            return

        if not session.get("is_hydrated"):
            session = await self._ensure_session_hydrated(interaction.channel_id, session.get("type", "multi"))

        user_id = interaction.user.id
        
        # [FIXED] Changed whisperer_id to speaker_pid to match the new unified_log format
        whisper_turns = {turn['turn_id']: turn for turn in session.get("unified_log", []) if turn.get("type") == "whisper" and turn.get("speaker_pid") == str(user_id)}
        
        paired_whispers = []
        log = session.get("unified_log", [])
        for i, turn in enumerate(log):
            if turn.get("turn_id") in whisper_turns:
                # Search forward from the whisper to find the first corresponding private response
                for j in range(i + 1, len(log)):
                    next_turn = log[j]
                    if next_turn.get("type") == "private_response" and turn.get("target_pid") == next_turn.get("speaker_pid"):
                        paired_whispers.append((turn, next_turn))
                        break # Found the pair, stop searching for this whisper

        if not paired_whispers:
            await interaction.followup.send("You have no whisper history in this session.", ephemeral=True)
            return

        # Note: History view regeneration is disabled because the target_participant dict 
        # is not easily reconstructed from old logs without PID-to-Owner mapping.
        view = WhisperHistoryView(self, interaction, paired_whispers)
        await interaction.followup.send(embed=view._get_current_embed(), view=view, ephemeral=True)

    @app_commands.command(name="mod", description="Moderation Dashboard (Bot Owner Only).")
    @app_commands.dm_only()
    @is_owner_in_dm_check()
    async def mod_slash(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        view = ModStatsView(self, interaction)
        await view.update_display()

async def setup(bot: commands.Bot):
    await bot.add_cog(GeminiAgent(bot))
# --- End of GeminiCog.py ---