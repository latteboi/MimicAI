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

        self.PROFILES_DIR = PROFILES_DIR
        self.LTM_DIR = LTM_DIR
        self.TRAINING_DIR = TRAINING_DIR
        self.PUBLIC_PROFILES_DIR = PUBLIC_PROFILES_DIR
        self.CHILD_BOTS_DIR = CHILD_BOTS_DIR
        self.USERS_DIR = USERS_DIR
        self.APPEARANCES_DIR = APPEARANCES_DIR
        self.SHARES_DIR = SHARES_DIR
        self.PERSONAL_KEYS_DIR = PERSONAL_KEYS_DIR
        self.DATA_DIR = DATA_DIR
        self.FREEWILL_SERVERS_DIR = FREEWILL_SERVERS_DIR
        self.SESSIONS_GLOBAL_DIR = SESSIONS_GLOBAL_DIR
        self.SESSIONS_SERVERS_DIR = SESSIONS_SERVERS_DIR
        for d in [PROFILES_DIR, LTM_DIR, TRAINING_DIR, PUBLIC_PROFILES_DIR, CHILD_BOTS_DIR, USERS_DIR, APPEARANCES_DIR, SHARES_DIR, PERSONAL_KEYS_DIR, DATA_DIR, FREEWILL_SERVERS_DIR, SESSIONS_GLOBAL_DIR, SESSIONS_SERVERS_DIR]:
            os.makedirs(d, exist_ok=True)

        try:
            self.fernet = Fernet(defaultConfig.ENCRYPTION_KEY)
        except Exception as e:
            print(f"CRITICAL: Failed to initialize encryption. Ensure ENCRYPTION_KEY is set in defaultConfig.py. Error: {e}")
            self.fernet = None

        self.persona_modal_sections_order = ['backstory', 'personality_traits', 'likes', 'dislikes', 'appearance'] 
        
        self.user_profiles: LRUCache = LRUCache(max_size=10)
        
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
        
        self.message_id_to_original_prompt: LRUCache = LRUCache(max_size=PROMPT_CACHE_MAX_SIZE)
        self.message_to_history_turn: LRUCache = LRUCache(max_size=PROMPT_CACHE_MAX_SIZE * 4) 
        self.model_override_warnings_sent: Set[Tuple[int, int, str]] = set()
        self.debug_users: Set[int] = set()
        self.global_chat_sessions: LRUCache = LRUCache(max_size=10)
        self.freewill_busy_profiles: Set[Tuple[int, int, str]] = set()
        self.purged_message_ids: Set[int] = set()
        self.pending_child_confirmations: Dict[str, Any] = {}
        self.global_blacklist: Set[int] = set()
        self._load_blacklist()
        self.session_last_accessed = {}
        self.mapping_caches: LRUCache = LRUCache(max_size=5) # Cache for on-disk mappings
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
        
        if not profile_name:
            await interaction.followup.send("Profile name cannot be empty.", ephemeral=True)
            return

        if profile_name == DEFAULT_PROFILE_NAME or profile_name.lower() == 'clyde':
            await interaction.followup.send(f"The name '{profile_name}' is reserved and cannot be used.", ephemeral=True)
            return

        user_data = self._get_user_data_entry(interaction.user.id)
        
        if profile_name in user_data.get("profiles", {}) or profile_name in user_data.get("borrowed_profiles", {}):
            await interaction.followup.send(f"A profile with the name '{profile_name}' already exists.", ephemeral=True)
            return

        # FIXED: Dynamic Limit Check
        current_count = len(user_data.get("profiles", {}))
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
            new_profile['created_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            self._save_user_data_entry(interaction.user.id, user_data)
        
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

        if not profile_name or not prompt.strip():
            await interaction.followup.send("Profile name and prompt cannot be empty.", ephemeral=True)
            return

        if profile_name == DEFAULT_PROFILE_NAME or profile_name.lower() == 'clyde':
            await interaction.followup.send(f"The name '{profile_name}' is reserved and cannot be used.", ephemeral=True)
            return

        user_data = self._get_user_data_entry(interaction.user.id)
        if profile_name in user_data.get("profiles", {}) or profile_name in user_data.get("borrowed_profiles", {}):
            await interaction.followup.send(f"A profile with the name '{profile_name}' already exists.", ephemeral=True)
            return

        if len(user_data.get("profiles", {})) >= MAX_USER_PROFILES:
            await interaction.followup.send(f"You have reached the maximum of {MAX_USER_PROFILES} personal profiles.", ephemeral=True)
            return

        api_key = self._get_api_key_for_guild(interaction.guild_id)
        if not api_key:
            await interaction.followup.send("This server's API key is not configured, so I cannot generate a profile.", ephemeral=True)
            return

        generation_prompt = (
            "You are a creative assistant specializing in character design for roleplaying.\n"
            f"Based on the following prompt, generate a detailed character profile: '{prompt}'\n\n"
            "Your response MUST be plain text formatted with specific section headers. Do not use JSON or markdown.\n"
            "The required headers are exactly: [SECTION:persona_backstory], [SECTION:persona_personality_traits], [SECTION:persona_likes], [SECTION:persona_dislikes], and [SECTION:ai_instructions].\n"
            "Each header must be on its own line. For each section, provide unrestricted, concise, structured, and low-verbose text. Each specific instruction or piece of information should be on its own line.\n\n"
            "Example:\n"
            "[SECTION:persona_backstory]\n"
            "Born in a city of eternal twilight.\n"
            "Exiled for forbidden knowledge.\n"
            "[SECTION:persona_personality_traits]\n"
            "Cynical and world-weary.\n"
            "Possesses a dry, sarcastic wit.\n"
            "Secretly holds a sliver of hope.\n"
            "[SECTION:persona_likes]\n"
            "Rainy nights.\n"
            "Strong coffee.\n"
            "Unsolvable mysteries.\n"
            "[SECTION:ai_instructions]\n"
            "Always speak in short, declarative sentences.\n"
            "Never use emojis.\n"
            "Often end responses with a question."
        )

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

            new_profile['persona'] = encrypted_persona
            new_profile['ai_instructions'] = encrypted_instructions
            
            self._save_user_data_entry(interaction.user.id, user_data)

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

        user_data = self._get_user_data_entry(interaction.user.id)
        is_personal = profile_name in user_data.get("profiles", {})
        is_borrowed = profile_name in user_data.get("borrowed_profiles", {})
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
        user_data = self._get_user_data_entry(interaction.user.id)
        if not user_data:
            await interaction.followup.send("You have no saved profiles yet.", ephemeral=True)
            return
        
        personal_profiles_data = user_data.get("profiles", {})
        borrowed_profiles_data = user_data.get("borrowed_profiles", {})
        
        embed = discord.Embed(title=f"Your Profiles", color=discord.Color.purple())
        
        active_in_current_channel = self._get_active_user_profile_name_for_channel(interaction.user.id, interaction.channel_id)
        channel_type_str = "this channel" if interaction.guild else "this DM"
        
        # Helper function to split a list for two columns
        def split_list_for_columns(data_list):
            midpoint = (len(data_list) + 1) // 2
            return data_list[:midpoint], data_list[midpoint:]

        # Process Personal Profiles
        personal_list = []
        for name in sorted(personal_profiles_data.keys()):
            marker = f" (Active)" if name == active_in_current_channel else ""
            personal_list.append(f"- `{name}`{marker}")
        
        if personal_list:
            col1, col2 = split_list_for_columns(personal_list)
            embed.add_field(name="Personal Profiles", value="\n".join(col1) if col1 else "\u200b", inline=True)
            embed.add_field(name="\u200b", value="\n".join(col2) if col2 else "\u200b", inline=True)

        # Process Borrowed Profiles
        borrowed_list = []
        for name, data in sorted(borrowed_profiles_data.items()):
            owner_id = int(data["original_owner_id"])
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
        user_data = self._get_user_data_entry(interaction.user.id)
        owner_id = int(defaultConfig.DISCORD_OWNER_ID)

        is_borrowed = profile_name in user_data.get("borrowed_profiles", {})
        is_personal = profile_name in user_data.get("profiles", {})
        
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
                borrow_data = user_data["borrowed_profiles"][profile_name]
                effective_owner_id = int(borrow_data["original_owner_id"])

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

            user_data = self._get_user_data_entry(interaction.user.id)
            personal_profiles = user_data.get("profiles", {}).keys()
            borrowed_profiles = user_data.get("borrowed_profiles", {}).keys()
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
            if not session or not session.get("profiles"):
                await interaction.followup.send("There is no active session in this channel.", ephemeral=True)
                return
            
            profile_list = []
            for i, p_data in enumerate(session["profiles"]):
                p_owner_id = p_data['owner_id']
                p_name = p_data['profile_name']
                p_method = p_data.get('method', 'webhook')
                
                if p_method == 'child_bot':
                    bot_user = self.bot.get_user(int(p_data['bot_id']))
                    display = bot_user.name if bot_user else p_name
                else:
                    display = p_name
                profile_list.append(f"**{i+1}.** `{display}`")
            
            await interaction.followup.send("Current session participants:\n" + "\n".join(profile_list), ephemeral=True)
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
            user_data = self._get_user_data_entry(interaction.user.id)
            is_personal = profile_name in user_data.get("profiles", {})
            is_borrowed = profile_name in user_data.get("borrowed_profiles", {})
            if not is_personal and not is_borrowed:
                await interaction.followup.send(f"You do not have a profile named '{profile_name}'.", ephemeral=True)
                return

            # Resolve Method (Child Bot vs Webhook)
            effective_owner_id = interaction.user.id
            effective_profile_name = profile_name
            if is_borrowed:
                borrowed_data = user_data["borrowed_profiles"][profile_name]
                effective_owner_id = int(borrowed_data["original_owner_id"])
                effective_profile_name = borrowed_data["original_profile_name"]

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
        session = self.multi_profile_channels.get(interaction.channel_id)
        
        if not session:
            await interaction.response.send_message("No active session in this channel.", ephemeral=True)
            return

        session_type = session.get("type", "multi")
        type_display = "Regular (Multi-Profile)" if session_type == "multi" else "Freewill (Proactive/Reactive)"
        
        owner = self.bot.get_user(session.get("owner_id"))
        owner_name = owner.name if owner else f"ID: {session.get('owner_id')}"
        
        # Logic to gather profiles for display
        profiles_for_display = []
        
        if session_type == "freewill":
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
                        if p_dict: profiles_for_display.append(p_dict)
        else:
            # For Regular, use the session's profile list
            profiles_for_display = session.get("profiles", [])

        participant_count = len(profiles_for_display)
        
        embed = discord.Embed(title=f"Session Info: #{interaction.channel.name}", color=discord.Color.gold())
        embed.add_field(name="Session Type", value=type_display, inline=True)
        embed.add_field(name="Session Owner", value=owner_name, inline=True)
        embed.add_field(name="Participants", value=str(participant_count), inline=True)
        
        if session.get("session_prompt"):
            embed.add_field(name="Prompt", value=session["session_prompt"][:200] + "..." if len(session["session_prompt"]) > 200 else session["session_prompt"], inline=False)

        # Create a temporary session dict for the View to use
        session_view_data = session.copy()
        session_view_data["profiles"] = profiles_for_display

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
        self._delete_session_from_disk(dummy_session_key, session_type)
        
        mapping_key = (session_type, ch_id)
        self.mapping_caches.pop(mapping_key, None)
        path = self._get_mapping_path(mapping_key)
        _delete_file_shard(str(path))

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
        self._delete_session_from_disk(dummy_session_key, session_type)
        
        mapping_key = (session_type, ch_id)
        self.mapping_caches.pop(mapping_key, None)
        path = self._get_mapping_path(mapping_key)
        _delete_file_shard(str(path))

        if session.get('worker_task'):
            self._safe_cancel_task(session['worker_task'])
        
        self.session_last_accessed.pop(ch_id, None)
        self._save_multi_profile_sessions()

        await interaction.followup.send(f"Session suspended for {interaction.channel.mention} and Freewill triggers disabled. The bot will be silent until mentioned or configured again.", ephemeral=True)

    @app_commands.command(name="purge", description="Purges messages and their associated memory (Admin Only).")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    @app_commands.guild_only()
    @is_admin_or_owner_check()
    @app_commands.describe(amount="Number of messages to delete (1-100).", user="[Optional] Target a specific user's messages.")
    async def purge_slash(self, interaction: discord.Interaction, amount: app_commands.Range[int,1,100], user: Optional[discord.Member]=None):
        if not self.has_lock : return
        await interaction.response.defer(ephemeral=True)

        if not isinstance(interaction.channel, (discord.TextChannel, discord.Thread)):
            await interaction.followup.send("Purge command not supported in this channel type.", ephemeral=True)
            return
            
        app_perms = interaction.app_permissions
        if not app_perms or not app_perms.manage_messages:
            await interaction.followup.send("I lack 'Manage Messages' permission.", ephemeral=True); return

        check_fn = (lambda m: m.author == user) if user else (lambda m: True)
        
        # [FIXED] Intercept message IDs during the filter check to prevent on_raw_message_delete race condition
        def check_and_track(m):
            valid = check_fn(m)
            if valid:
                self.purged_message_ids.add(m.id)
            return valid

        try:
            messages_to_delete = await interaction.channel.purge(limit=amount, check=check_and_track, before=interaction.created_at, reason=f"Purge by {interaction.user}")
            
            progress_message = await interaction.followup.send(f"Deleted {len(messages_to_delete)} message(s). Now cleaning them from my memory...", ephemeral=True)

            # [UPDATED] Removed legacy 'single' session data structures
            multi_turns_to_remove = {}  # { (channel_id, session_type): {turn_ids} }
            
            session_type_multi = self.multi_profile_channels.get(interaction.channel_id, {}).get("type", "multi")
            mapping_key_multi = (session_type_multi, interaction.channel_id)
            if mapping_key_multi not in self.mapping_caches:
                self.mapping_caches[mapping_key_multi] = self._load_mapping_from_disk(mapping_key_multi)

            for msg in messages_to_delete:
                turn_info = self.message_to_history_turn.pop(msg.id, None)
                # Fallback to multi mapping check
                if not turn_info and mapping_key_multi in self.mapping_caches:
                    turn_info = self.mapping_caches[mapping_key_multi].pop(str(msg.id), None)

                if turn_info:
                    if isinstance(turn_info[0], list): turn_info[0] = tuple(turn_info[0])

                    # Modern Multi/Freewill Session Mapping
                    if isinstance(turn_info[0], int):
                        try:
                            channel_id, session_type, turn_id = turn_info
                            session_id = (channel_id, session_type)
                            if session_id not in multi_turns_to_remove:
                                multi_turns_to_remove[session_id] = set()
                            multi_turns_to_remove[session_id].add(turn_id)
                        except ValueError:
                            continue

            # --- Clean mapping files before cleaning history ---
            for (channel_id, session_type), lines_to_delete in multi_turns_to_remove.items():
                mapping_key = (session_type, channel_id)
                if mapping_key in self.mapping_caches:
                    keys_to_delete = [
                        msg_id for msg_id, t_info in self.mapping_caches[mapping_key].items()
                        if isinstance(t_info, (list, tuple)) and len(t_info) > 2 and t_info[2] in lines_to_delete
                    ]
                    for msg_id in keys_to_delete:
                        self.mapping_caches[mapping_key].pop(msg_id, None)

            cleaned_turns_count = 0
            
            # Process multi/freewill sessions
            for (channel_id, session_type), turn_ids_to_delete in multi_turns_to_remove.items():
                mapping_key = (session_type, channel_id)
                if mapping_key in self.mapping_caches:
                    self.mapping_caches[mapping_key].pop('grounding_checkpoint', None)

                # Restore missing increment
                cleaned_turns_count += len(turn_ids_to_delete)
                session = self.multi_profile_channels.get(channel_id)
                if session:
                    if not session.get("is_hydrated"):
                        session = self._ensure_session_hydrated(channel_id, session_type)
                    
                    # Decrement Individual Counters
                    for t_id in turn_ids_to_delete:
                        turn_obj = next((turn for turn in session.get("unified_log", []) if turn.get("turn_id") == t_id), None)
                        if turn_obj:
                            s_key = tuple(turn_obj.get("speaker_key", []))
                            for p in session.get('profiles', []):
                                if (p['owner_id'], p['profile_name']) == s_key:
                                    p['ltm_counter'] = max(0, p.get('ltm_counter', 0) - 1)
                                    break

                    original_log_len = len(session.get("unified_log", []))
                    session["unified_log"] = [
                        turn for turn in session.get("unified_log", [])
                        if turn.get("turn_id") not in turn_ids_to_delete
                    ]

                    # If the log was modified, we must rebuild all in-memory chat histories
                    if len(session["unified_log"]) < original_log_len:
                        for p_data in session["profiles"]:
                            p_key = (p_data['owner_id'], p_data['profile_name'])
                            participant_history = []
                            for turn in session["unified_log"]:
                                speaker_key = tuple(turn.get("speaker_key", []))
                                role = 'model' if speaker_key == p_key else 'user'
                                participant_history.append({'role': role, 'parts': [turn.get("content")]})
                            session["chat_sessions"][p_key] = GoogleGenAIChatSession(history=participant_history)

                    # Check if the session is now effectively empty (contains no public turns)
                    is_effectively_empty = not session.get("unified_log") or all(
                        turn.get("type") in ["whisper", "private_response"] for turn in session.get("unified_log", [])
                    )
                    if is_effectively_empty:
                        dummy_session_key = (channel_id, None, None)
                        self._delete_session_from_disk(dummy_session_key, session_type)
                        
                        mapping_key = (session_type, channel_id)
                        path = self._get_mapping_path(mapping_key)
                        _delete_file_shard(str(path))
                        
                        # Clear LTM cooldowns since context is gone
                        for p_key in session.get("chat_sessions", {}).keys():
                            owner_id, profile_name = p_key
                            full_session_key = (channel_id, owner_id, profile_name)
                            self.ltm_recall_history.pop(full_session_key, None)

                    self.session_last_accessed[channel_id] = time.time()

            spec = f" from {user.mention}" if user else ""
            await progress_message.edit(content=f"Deleted {len(messages_to_delete)} message(s){spec} and cleaned {cleaned_turns_count} turn(s) from memory.")

        except Exception as e:
            await interaction.followup.send(f"An error occurred during purge: {e}", ephemeral=True)
            traceback.print_exc()

    @profile_group.command(name="global_chat", description="Have a persistent, private conversation with a profile.")
    @app_commands.checks.cooldown(5, 60.0, key=lambda i: i.user.id)
    @app_commands.autocomplete(profile_name=CoreMixin.global_chat_profile_autocomplete)
    @app_commands.describe(
        profile_name="The profile to chat with. Leave blank to view history of recent chats.",
        message="The message to send. If omitted, shows history or input modal.",
        refresh="Set to True to clear your conversation history with this profile.",
        suspend="Set to True to permanently delete ALL global chat histories for every profile."
    )
    async def global_chat_slash(self, interaction: discord.Interaction, profile_name: Optional[str] = None, message: Optional[str] = None, refresh: Optional[bool] = False, suspend: Optional[bool] = False):
        user_id = interaction.user.id
        
        if suspend:
            await interaction.response.defer(ephemeral=True)
            import shutil
            
            keys_to_del = [k for k in self.global_chat_sessions.keys() if isinstance(k, tuple) and len(k) == 3 and k[0] == 'global' and k[1] == user_id]
            for k in keys_to_del:
                self.global_chat_sessions.pop(k, None)
                self.session_last_accessed.pop(k, None)
                self.ltm_recall_history.pop(k, None)
            
            mapping_key = ('global_chat', user_id)
            self.mapping_caches.pop(mapping_key, None)

            dir_path = pathlib.Path(self.SESSIONS_GLOBAL_DIR) / str(user_id)
            if dir_path.is_dir():
                try:
                    shutil.rmtree(dir_path)
                except Exception as e:
                    print(f"Error suspending all global chats for {user_id}: {e}")

            await interaction.followup.send("✅ All global conversation histories have been permanently deleted.", ephemeral=True)
            return

        if not profile_name:
            # Open the History UI which acts as the selector
            view = GlobalChatHistoryView(self, interaction, user_id)
            if not view.available_profiles:
                await interaction.response.send_message("You have no active global chat histories.", ephemeral=True)
            else:
                await interaction.response.send_message(embed=view.get_embed(), view=view, ephemeral=True)
            return

        profile_name_lower = profile_name.lower().strip()

        if refresh:
            await interaction.response.defer(ephemeral=True)
            session_key = ('global', user_id, profile_name_lower)
            
            # Clear from memory
            self.global_chat_sessions.pop(session_key, None)
            self.session_last_accessed.pop(session_key, None)
            self.ltm_recall_history.pop(session_key, None)

            # Clear from disk
            self._delete_session_from_disk(session_key, 'global_chat')
            
            await interaction.followup.send(f"Your global chat history with '{profile_name_lower}' has been cleared.", ephemeral=True)
            return

        if message:
            await interaction.response.defer(ephemeral=False, thinking=True)
            await self._execute_global_chat(interaction, profile_name_lower, message)
        else:
            user_data = self._get_user_data_entry(user_id)
            is_personal = profile_name_lower in user_data.get("profiles", {})
            is_borrowed = profile_name_lower in user_data.get("borrowed_profiles", {})
            if not is_personal and not is_borrowed:
                await interaction.response.send_message(f"Profile '{profile_name_lower}' not found.", ephemeral=True)
                return

            # Load the session to show history
            model_cache_key = ('global', user_id, profile_name_lower)
            session_data = self.global_chat_sessions.get(model_cache_key)
            
            if not session_data:
                session_data = self._load_session_from_disk(model_cache_key, 'global_chat')
                if session_data: 
                    self.global_chat_sessions[model_cache_key] = session_data
            
            if not session_data or not session_data.get('unified_log'):
                # If empty, fall back to the modal input
                async def modal_callback(modal_interaction: discord.Interaction, message_text: str):
                    await modal_interaction.response.defer(ephemeral=False, thinking=True)
                    await self._execute_global_chat(modal_interaction, profile_name_lower, message_text)

                modal = ActionTextInputModal(
                    title=f"Global Chat with '{profile_name_lower}'",
                    label="Message",
                    placeholder="Enter your message...",
                    on_submit_callback=modal_callback
                )
                await interaction.response.send_modal(modal)
            else:
                # Open the History View
                await interaction.response.defer(ephemeral=True)
                view = GlobalChatHistoryView(self, interaction, model_cache_key, session_data['unified_log'])
                await interaction.followup.send(embed=view.get_embed(), view=view, ephemeral=True)

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

    @app_commands.command(name="transparency", description="Displays info about background AI models and their API costs.")
    @app_commands.checks.cooldown(1, 30.0, key=lambda i: i.user.id)
    async def transparency_slash(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)

        pages_content = [
            ("Overview", 
             "MimicAI uses several specialized, high-speed AI models behind the scenes. This page provides transparency on what these models are, how they work, and their impact on API usage.\n\n*All prices are based on Google's pay-as-you-go tier and are calculated per 1 million tokens. A \"token\" is a piece of a word; roughly 750 words are equal to 1,000 tokens.*"),
            
            ("Long-Term Memories & Safety (The Workhorse)",
             "**Purpose:**\n"
             "- **Memory Creation (LTM):** Summarizes conversations to create long-term memories.\n"
             "- **Safety:** Moderates content for public profiles.\n\n"
             "**Pricing Impact:**\n"
             "Uses `gemini-flash-lite-latest`.\n"
             "• **Input:** $0.10 / 1M tokens\n"
             "• **Output:** $0.40 / 1M tokens"),

            ("Grounding (The Researcher)",
             "**Purpose:**\n"
             "- **Web Search (Grounding):** Decides if a search is needed, then finds and summarizes information from Google Search.\n\n"
             "**Requirements:**\n"
             "• **Google API Key:** Required (Free or Paid Tier). This feature **cannot** use an OpenRouter key as it relies on Google's integrated Search tool.\n\n"
             "**Pricing Impact:**\n"
             "Uses `gemini-2.0-flash`. If enabled, this adds **one extra API call** per message.\n"
             "• **Input:** $0.10 / 1M tokens\n"
             "• **Output:** $0.40 / 1M tokens\n"
             "Grounding has a unique pricing model based on **Requests Per Day (RPD)**.\n"
             "• **Free Tier:** Up to 500 RPD free.\n"
             "• **Paid Tier:** Up to 1,500 RPD free. $35.00 per 1,000 requests after the free limit."),

            ("Image Generation (The Artist)",
             "**Purpose:**\n"
             "- **Image Generation:** Creates all images for the `!image` and `!imagine` commands.\n\n"
             "**Requirements:**\n"
             "• **Google API Key:** Required (**Paid Tier Only**). This model is not available on free-tier API keys.\n\n"
             "**Pricing Impact:**\n"
             "Uses `gemini-2.5-flash-image`.\n"
             "Each use requires **two** API calls (Image Model + Chat Model). The image generation itself costs **~$0.039 per image**."),

            ("Embedding (The Librarian)",
             "**Purpose:**\n"
             "- **Semantic Search:** Converts text into numerical 'fingerprints' (embeddings) to power the search for LTM and Training Examples.\n"
             "- **Optimization:** Uses **Matryoshka Representation Learning** to output truncated 256-dimensional vectors, significantly reducing database size without losing accuracy.\n\n"
             "**Pricing Impact:**\n"
             "Uses `gemini-embedding-001`.\n"
             "This is the **cheapest operation** the bot performs. The cost is based on the amount of text being converted.\n"
             "• **Per 1 Million Tokens:** ~$0.15"),

            ("Profile Model (Variable)",
             "**Purpose:**\n"
             "- **Conversation:** The model you actually talk to (e.g., `GOOGLE/gemini-2.5-flash`, `OPENROUTER/anthropic/claude-3-opus`).\n\n"
             "**Pricing Impact:**\n"
             "This cost depends entirely on your selection in `/profile manage`.\n"
             "• **Google Models:** Prices vary by tier (Flash vs Pro).\n"
             "• **OpenRouter Models:** Prices are set by the provider (Anthropic, Meta, etc.).\n"
             "• **Fallback:** If your primary model fails, the bot attempts to use your configured Fallback Model (defaulting to `gemini-flash-lite`), which incurs its own standard cost."),
             
            ("Anti-Repetition Critic (The Editor)",
             "**Purpose:**\n"
             "- **Loop Detection:** If enabled, a lightweight model analyzes the conversation history *before* the main model replies to detect verbatim repetition loops.\n"
             "- **Quality Control:** Injects negative constraints into the system prompt to force the AI to break the loop.\n\n"
             "**Pricing Impact:**\n"
             "Uses `gemini-2.0-flash`. If enabled, this adds **one extra API call** per message.\n"
             "• **Input:** $0.10 / 1M tokens\n"
             "• **Output:** $0.40 / 1M tokens")
        ]

        embeds = []
        page_titles = [item[0] for item in pages_content]
        for i, (title, text) in enumerate(pages_content):
            embed = discord.Embed(title=f"Transparency: {title}", description=text, color=discord.Color.from_rgb(0, 128, 255))
            embed.set_footer(text=f"Page {i+1}/{len(pages_content)} | Info subject to change based on provider pricing.")
            embeds.append(embed)

        view = PaginatedEmbedView(embeds, page_titles)
        # Re-attach the external link button to the view
        view.add_item(ui.Button(label="Official Google Pricing Page", url="https://ai.google.dev/gemini-api/docs/pricing", row=2))
        
        await interaction.followup.send(embed=embeds[0], view=view, ephemeral=True)

    @app_commands.command(name="help", description="Displays detailed documentation about the bot's features and commands.")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    async def help_slash(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)

        admin_setup_content = (
            "Authentication Requirements",
            "This system requires API authentication to function. All generation tasks are passed to external inference endpoints.\n\n"
            "**Primary Authentication (Google Gemini):**\n"
            "1. Acquire a standard API key via Google AI Studio.\n"
            "2. Execute the `/settings` command via Direct Message to access the configuration panel.\n\n"
            "**Secondary Authentication (OpenRouter):**\n"
            "1. Acquire an API key via OpenRouter to utilise alternative models (e.g., Anthropic, Meta).\n\n"
            "**Key Pool Mechanism:**\n"
            "Administrators may set a 'Primary Key' for server-wide execution. Users may also contribute personal keys to the 'Pool'. The system will cycle through valid keys in the pool if the Primary Key is exhausted."
        )

        user_docs_content = [
            ("System Architecture Overview",
             "MimicAI operates as a multimodal orchestration layer. It manages character state, session history, and external API requests.\n\n"
             "The system is built upon a 'Profile' architecture. A Profile acts as a container for system instructions, inference parameters, and persistent memory. These profiles are executed within 'Sessions'—isolated conversational contexts tied to specific channels or direct messages.\n\n"
             "Use the dropdown menu below to navigate the technical documentation."),
            
            ("Profile Classes",
             "The system distinguishes between two classes of Profiles:\n\n"
             "**Personal Profiles:**\n"
             "These are primary entities owned and maintained by the user. The owner possesses full read/write privileges over generation parameters, memory vectors, and systemic instructions.\n\n"
             "**Borrowed Profiles:**\n"
             "These are read-only symbolic links to another user's Personal Profile, acquired via the Public Library or direct share code. Local configuration (such as temporal awareness and formatting) may be overridden, but core identity variables remain synchronised with the origin."),
            
            ("Persona Definition",
             "The core identity of a Profile is constructed using modular text fields that are compiled into XML-tagged system instructions during inference.\n\n"
             "The Persona module comprises:\n"
             "- **Backstory:** Historical and contextual data.\n"
             "- **Personality Traits:** Core psychological characteristics.\n"
             "- **Likes/Dislikes:** Positive and negative biases.\n"
             "- **Appearance:** Physical description (utilised heavily by the Visual Generation module for self-portraits)."),
            
            ("AI Instruction Protocol",
             "AI Instructions dictate operational behaviour and formatting constraints. These bypass the Persona module and instruct the underlying model on strict structural adherence.\n\n"
             "The interface provides four distinct data blocks to prevent context truncation. It is recommended to utilise standard declarative constraints (e.g., 'Do not output markdown', 'Keep responses under two sentences'). These instructions take priority over persona biases during generation."),
            
            ("Generation Parameters",
             "Users may manipulate the statistical variance of the output via standard sampling parameters:\n\n"
             "- **Temperature:** Controls the randomness of token selection. Lower values yield deterministic outputs; higher values yield diverse outputs.\n"
             "- **Top P (Nucleus Sampling):** The model considers only the tokens comprising the top probability mass. Set to 1.0 to disable.\n"
             "- **Top K:** Limits the token selection pool to the K most probable tokens.\n"
             "- **STM Length:** Short-Term Memory. Defines how many previous conversational turns are appended to the context window."),
             
            ("Advanced Heuristics (OpenRouter)",
             "For endpoints supporting OpenRouter specifications, additional penalisation parameters are available:\n\n"
             "- **Frequency Penalty (-2.0 to 2.0):** Applies an additive penalty to tokens based on their existing frequency in the text, reducing verbatim repetition.\n"
             "- **Presence Penalty (-2.0 to 2.0):** Applies an absolute penalty if a token exists at all, encouraging topic divergence.\n"
             "- **Repetition Penalty:** Multiplicative penalty against recently generated tokens.\n"
             "- **Min P / Top A:** Advanced thresholding metrics to dynamically cull low-probability tokens."),
             
            ("Reasoning Configuration",
             "Specific models (e.g., Gemini 2.0 Flash Thinking, Gemini 3.0) support overt reasoning phases prior to text generation.\n\n"
             "- **Thinking Level:** Determines the computational effort allocated to reasoning (None, Minimal, Low, Medium, High, XHigh).\n"
             "- **Token Budget:** For models that accept explicit limits (e.g., Gemini 2.5), dictates the exact token cap allocated to internal thought.\n"
             "- **Summary Visibility:** Toggles whether the raw reasoning tokens are exported and delivered as an attachment alongside the response."),
             
            ("Vocal Synthesis (Director's Desk)",
             "The Director's Desk provides semantic priming for the Text-to-Speech (TTS) module.\n\n"
             "Instead of raw audio manipulation, the TTS engine is a generative model capable of inferring tone from contextual metadata. Users may define:\n"
             "- **Archetype:** E.g., 'A grizzled detective.'\n"
             "- **Accent:** E.g., 'Standard British.'\n"
             "- **Dynamics:** E.g., 'Speaking in a whisper in a cavern.'\n"
             "- **Pacing & Style:** E.g., 'Rapid delivery with a subtle smile.'"),
             
            ("Vocal Synthesis (Hardware Settings)",
             "Beyond semantic instruction, explicit TTS hardware configurations include:\n\n"
             "- **Voice Name:** The designated prebuilt voice identity (e.g., 'Aoede', 'Kore', 'Charon').\n"
             "- **Speech Model:** The backend endpoint (defaults to `gemini-2.5-flash-preview-tts`).\n"
             "- **Prosody (Temperature):** Dictates the expressive variance in the audio generation. Values outside standard 1.0 may result in erratic audio artefacts."),
             
            ("Multimodal Audio Output",
             "Audio delivery modes modify how the bot processes standard text generation:\n\n"
             "- **Text-Only:** Default behaviour.\n"
             "- **Audio + Text:** Delivers the primary text payload alongside a synthesised `.wav` attachment.\n"
             "- **Audio-Only:** Suppresses the text output, delivering solely the audio file.\n"
             "- **Multi-Audio:** In multi-profile sessions, delays audio transmission until the conclusion of the round, stitching all segments into a continuous master file."),

            ("Short-Term Memory (STM)",
             "The STM acts as the immediate conversational buffer. The system maintains a synchronised cache (`unified_log`) representing the chronological sequence of recent interactions.\n\n"
             "This data is hydrated into specific `ChatSession` arrays prior to inference. The depth of this buffer is determined by the profile's STM Length parameter. Exceeding context limits may degrade the model's adherence to systemic instructions. The `/refresh` command flushes this buffer for the current channel."),

            ("Long-Term Memory (LTM) Architecture",
             "The LTM subsystem provides persistent data retention via an automated summarisation pipeline.\n\n"
             "**Auto-Creation Pipeline:**\n"
             "When a user communicates with a profile, an internal counter tracks the exchange volume. Once the `Creation Interval` threshold is met, the system extracts the last `Summarization Context` turns and submits them to an auxiliary model (`gemini-flash-lite`). The model condenses the excerpt into a strict, third-person factual summary."),
             
            ("LTM Retrieval Logistics",
             "Memories are encoded using Matryoshka Representation Learning (256-dimensional embeddings) and stored locally.\n\n"
             "During active chat, the user's prompt is embedded and cross-referenced against the profile's memory vault via cosine similarity. If the metric exceeds the profile's `Relevance Threshold`, the top results are injected into the system prompt as `<archive_context>`.\n\n"
             "**LTM Scopes:**\n"
             "- User: Recalled exclusively for the original author.\n"
             "- Server: Recalled for any user within the originating guild.\n"
             "- Global: Recalled universally."),

            ("Contextual Training Examples",
             "Training Examples are explicit, user-defined input/output pairs that dictate specific stylistic behaviour.\n\n"
             "Like LTMs, they are embedded and retrieved via vector search. Upon matching the `Relevance Threshold`, the pairs are injected into the system instructions. This ensures the model dynamically alters its vernacular, sentence structure, and formatting based on semantic similarities to the user's current prompt. Manage these via `/profile data manage`."),

            ("Redundancy (Model Fallback)",
             "The system features a fault-tolerant generation loop. If the primary inference request fails due to rate limits, server timeouts, or strict safety blocks, the system automatically redirects the payload to the defined Fallback Model.\n\n"
             "It is highly recommended to designate an efficient, low-cost model (e.g., `gemini-flash-lite-latest`) as the fallback to maintain uptime during primary endpoint degradation."),

            ("Web Grounding (Search Indexing)",
             "Grounding enables real-time information retrieval via Google Search.\n\n"
             "If the Grounding Mode is active (`On` or `On+`), the system routes the query through a secondary decision-model to determine if external facts are required. If verified, a Google Search tool is executed, and the resultant chunks are parsed into `<external_context>` for the primary generation phase.\n\n"
             "The `On+` mode forces the AI to append citation links beneath the final message."),

            ("Web Grounding (URL Parsing)",
             "If URL Fetching is enabled, the system intercepts HTTP/HTTPS links present in the user's prompt.\n\n"
             "A headless asynchronous request extracts the raw HTML, stripping structural noise, scripts, and styling. The decoded text is injected into the payload as `<document_context>`. This process bypasses Google Search and relies strictly on direct endpoint resolution. Media objects (images) located at the endpoint are also appended for multimodal analysis."),

            ("Visual Generation & Processing",
             "Profiles may generate `.png` visuals using the `!image` or `!imagine` prefix.\n\n"
             "This requires a paid-tier Google API key routing to the `gemini-2.5-flash-image` endpoint. The generation prompt automatically appends the profile's `Appearance` data if second-person pronouns are detected. Once the image is generated, it is fed back into the text model to produce an in-character comment attached to the resulting file.\n\n"
             "To execute style transfers or iterative edits, reply to an existing image with the `!image` command."),

            ("Output Modification: Anti-Repetition Critic",
             "The Critic module acts as an automated linguistic quality assurance layer.\n\n"
             "When enabled, the system reviews the last three model outputs prior to execution. If structural repetition is detected (e.g., reusing identical opening phrases or structural loops common in auto-regressive models), it formulates a negative constraint. This constraint is injected into the prompt, explicitly banning the repetitive pattern for the impending turn."),

            ("Output Modification: Typing Simulation",
             "Realistic Typing simulates human interface delays.\n\n"
             "The output string is parsed via regex to identify sentence boundaries (accounting for standard abbreviations). The execution loop calculates a reading/typing delay based on character volume. Sentences are sequentially transmitted to the Discord Webhook API, creating a fragmented, continuous stream of text rather than a single monolithic block."),

            ("Temporal Awareness Integration",
             "If Time Tracking is enabled, the execution loop parses the profile's designated IANA Timezone (e.g., `Europe/London`).\n\n"
             "A UTC datetime object is shifted to the local equivalent and formatted into a human-readable string. This data is injected as `<time_context>` into the system prompt, ensuring the model maintains continuity regarding the time of day, day of the week, and date."),

            ("Standard Session Management",
             "All conversational interactions occur within Sessions.\n\n"
             "**Regular Mode (`/session config`):**\n"
             "Initiates a manual session requiring an administrator to define the participating profiles. The operational mode determines execution flow:\n"
             "- Sequential: Participants execute in the exact order established during configuration.\n"
             "- Random: Execution order is shuffled per round.\n\n"
             "Standard `/session swap` operations allow dynamic manipulation of the active cast during runtime."),

            ("Session Management: Response Modes",
             "Profiles may be configured to format their network payload differently depending on use-case requirements:\n\n"
             "- **Regular:** Standard webhook message execution.\n"
             "- **Mention:** Prepends the user's Discord ID to the message payload.\n"
             "- **Reply:** Triggers the Discord API message reference parameter to visually connect the response to the user's prompt.\n"
             "- **Mention + Reply:** A hybrid execution of both methodologies."),

            ("Session Control via Reactions",
             "Users may dynamically influence session state and execution flow by applying specific emoji reactions to messages.\n\n"
             "- **Regenerate (🔁):** Regenerates a profile's message using the same context that lead to that response.\n"
             "- **Next Speaker (⏯️):** Triggers the next profile participant to respond.\n"
             "- **Continue Round (🍿):** Triggers a new round for all profile participants to respond.\n"
             "- **Mute Turn (🔇):** Hides the targeted message from the session's transcript. The message remains in the channel but becomes invisible to the profiles when they respond.\n"
             "- **Skip Participant (❌):** Suspends a specific profile from responding in the session."),

            ("Automated Event Scheduling (Freewill)",
             "The Freewill system enables autonomous bot execution without direct user invocation.\n\n"
             "Administrators classify channels as either 'Living' (fully autonomous) or 'Lurking' (reactive only). Active users must opt their profiles into the system via the configuration dashboard. The system evaluates participation based on the defined 'Personality' percentages (Introverted, Regular, Outgoing), which determine the probability of intervention."),

            ("Freewill Mode: Proactive Generation",
             "In 'Living' channels, an asynchronous loop executes at one-minute intervals.\n\n"
             "If the channel's `Event Cooldown` timer has expired, the system calculates a probability check based on the `Event Chance`. Upon success, it gathers an ad-hoc cast of opted-in profiles and forces a 'Director's Prompt' (e.g., 'You see X walk into the room. React.'). The profiles converse autonomously until the defined proactive round counter depletes."),

            ("Freewill Mode: Reactive Triggers",
             "In both 'Living' and 'Lurking' channels, the system evaluates all standard user messages against the active profiles.\n\n"
             "If a user's text string contains an exact match to a profile's defined 'Wakeword', the profile immediately bypasses standard turn logic to interject. In the absence of wakewords, a probabilistic roll is made against the profile's personality metric to determine random interjection."),

            ("Inter-Process Communication (Child Bots)",
             "Child Bots are discrete Discord Application binaries orchestrated by the primary instance (Hivemind).\n\n"
             "The system establishes a WebSocket Inter-Process Communication (IPC) layer on `ws://127.0.0.1:8765`. A background subprocess handles the asynchronous execution of `discord.py` clients. These child instances possess no internal logic; they blindly execute payloads formatted and routed by the primary engine. This allows profiles to exist as independent server members with dedicated online presence."),

            ("Child Bot Registration & Sync",
             "To instantiate a Child Bot:\n"
             "1. Register a new Application within the Discord Developer Portal.\n"
             "2. Extract the bot token and submit it via `/settings`.\n"
             "3. The primary instance encrypts the token, establishes the IPC socket, and dispatches an initialization command.\n\n"
             "The parent instance automatically synchronises the profile's display name and avatar URL to the child application via API calls upon creation or modification."),

            ("Public Distribution Index (Hub)",
             "The Hub (`/profile hub`) operates as a global metadata index for Profile distribution.\n\n"
             "Premium users may publish profiles to the index. Upon request, the system executes an automated moderation check using `gemini-flash-lite` to evaluate the profile name, display name, and avatar for graphic or explicit violations. Upon clearance, the profile is appended to the `index.json` structure, allowing global read access for borrowing operations."),

            ("Private Distribution (Share Codes)",
             "For targeted distribution, users may generate temporary cryptographic Share Codes via the Hub interface.\n\n"
             "These codes encode the original owner's ID and the profile string. The generated hex key is retained in a volatile dictionary with a 300-second (5 minute) Time-To-Live (TTL). When a recipient redeems the code, the system validates the TTL, reads the encoded source, and duplicates a symbolic link to their local directory."),

            ("Data Import and Export Validation",
             "Profiles and associated persistent memory (LTM/Training) may be exported to standard `.mimic` JSON files.\n\n"
             "The export function decrypts local storage and packages the dictionaries into plaintext formatting. During import, the system validates the JSON schema, handles namespace collisions via UUID appending, re-encrypts the text blobs, and shunts the data to the appropriate `user_data` and `storage` shards."),

            ("Command Reference: Administration",
             "- `/session config`: Opens the interface to initialise standard or Freewill sessions.\n"
             "- `/suspend`: Terminates the active session and flushes the buffer.\n"
             "- `/purge`: Executes a bulk message deletion operation and recursively scrubs associated internal memory indices.\n"
             "- `/profile speak`: Forces an anonymous execution of a designated profile payload. (Requires Webhook/Child Bot delivery)."),

            ("Command Reference: Configuration",
             "- `/settings`: Private interface for API key authentication and Child Bot provisioning.\n"
             "- `/profile manage`: Central interface for profile manipulation.\n"
             "- `/profile bulk manage`: Executes batch manipulation across multiple profiles simultaneously.\n"
             "- `/profile data manage`: Interface for manual CRUD operations on vector-embedded databases (LTM/Training)."),

            ("Command Reference: Interaction",
             "- `/profile swap`: Changes the active profile context for the current channel.\n"
             "- `/profile list`: Enumerates valid user indices.\n"
             "- `/session view`: Dumps session variables and participant statuses to chat.\n"
             "- `/refresh`: Performs a targeted wipe of the channel's Short-Term Memory buffer without destroying the session structure.\n"
             "- `/whisper`: Transmits hidden variables directly to a profile's context window, eliciting an ephemeral response."),
             
             ("System Content Moderation Framework",
             "The system strictly adheres to external provider usage guidelines regarding illicit content.\n\n"
             "Users may classify profiles under specific safety tiers (Low, Medium, High). The 'Unrestricted 18+' tier disables standard filters; however, execution of these profiles is mathematically restricted by the core engine to channels designated as NSFW via Discord API flags. Attempts to execute unrestricted profiles in standard channels will abort with a silent failure or system error log."),

            ("Contextual Metadata & Token Overhead",
             "To maintain coherent multi-participant communication, the system forcefully injects hardcoded metadata into the prompt payload.\n\n"
             "Every user and profile response within the history buffer is strictly prefixed with a designated `[Name] [Timestamp]:` string. While this incurs a nominal token overhead per conversational turn, it establishes a foundational chronological and spatial awareness for the model. This guarantees the AI can differentiate between multiple actors and comprehend the passage of time without confusing internal dialogue with external user prompts."),

            ("Extensible Markup Language (XML) Standards",
             "The orchestration layer heavily utilises XML formatting (e.g., `<document_context>`, `<archive_context>`) when appending system-generated data to the context window.\n\n"
             "Large Language Models possess a high mechanical affinity for XML boundary demarcation. By segregating raw conversational text from background technical operations using explicit tags, the system establishes a hard partition. This significantly mitigates prompt injection vectors, reduces contextual hallucination, and ensures the model accurately separates 'what the user said' from 'what the system is instructing it to know'."),

            ("Data Residency and End User Agreement",
             "All user-generated strings, including memory sequences and API keys, are subjected to symmetric Fernet encryption prior to disc residency.\n\n"
             "By configuring external inference endpoints, you acknowledge the terms of service of the respective API provider (Google LLC, OpenRouter, etc.).")
        ]

        final_docs_content = []
        is_configured = True 
        if interaction.guild:
            api_key = self._get_api_key_for_guild(interaction.guild.id)
            if not api_key:
                is_configured = False

        if not is_configured:
            final_docs_content.append(admin_setup_content)
        
        final_docs_content.extend(user_docs_content)

        embeds = []
        page_titles = [item[0] for item in final_docs_content]
        for i, (title, description) in enumerate(final_docs_content):
            embed = discord.Embed(title=title, description=description, color=discord.Color.blue())
            embed.set_footer(text=f"Page {i+1}/{len(final_docs_content)}")
            embeds.append(embed)

        view = PaginatedEmbedView(embeds, page_titles)
        await interaction.followup.send(embed=embeds[0], view=view, ephemeral=True)

    @app_commands.command(name="terms", description="View the MimicAI Terms of Service and Privacy Policy.")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    async def terms_slash(self, interaction: discord.Interaction):
        text = (
            "**MimicAI Terms & Privacy**\n\n"
            "**1. Self-Hosted Instance:** This instance of MimicAI is self-hosted. The developers of the MimicAI software are not responsible for the management, data retention, or uptime of this specific bot. Please contact the bot owner for local privacy inquiries.\n\n"
            "**2. Software License:** This software is provided under the Prosperity Public License 3.0.0. Unauthorized commercial use is prohibited.\n\n"
            "**3. Universal Disclaimers:** By using this software, you acknowledge that AI-generated content can be unpredictable and the software developers are not liable for any output produced by the model.\n\n"
            "For the full legal documentation, visit the link below:"
        )
        view = ui.View()
        view.add_item(ui.Button(label="Open Official Website", url="https://mimic-ai.org/", style=discord.ButtonStyle.link))
        await interaction.response.send_message(text, view=view, ephemeral=True)

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

    @app_commands.command(name="blacklist", description="Add or remove a user from the global blacklist.")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    @app_commands.dm_only()
    @is_owner_in_dm_check()
    @app_commands.describe(action="The action to perform.", user="The user to manage.")
    async def blacklist_manage_slash(self, interaction: discord.Interaction, action: Literal['add', 'remove'], user: discord.User):
        if action == 'add':
            if user.id in self.global_blacklist:
                await interaction.response.send_message(f"User `{user.name}` is already on the global blacklist.", ephemeral=True)
                return
            
            self.global_blacklist.add(user.id)
            self._save_blacklist()
            await interaction.response.send_message(f"✅ User `{user.name}` has been added to the global blacklist.", ephemeral=True)
        
        elif action == 'remove':
            if user.id not in self.global_blacklist:
                await interaction.response.send_message(f"User `{user.name}` is not on the global blacklist.", ephemeral=True)
                return
                
            self.global_blacklist.discard(user.id)
            self._save_blacklist()
            await interaction.response.send_message(f"✅ User `{user.name}` has been removed from the global blacklist.", ephemeral=True)

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
        session = self.multi_profile_channels.get(interaction.channel_id)
        if not session:
            return []

        choices = []
        for participant in session.get("profiles", []):
            owner_id = participant.get("owner_id")
            profile_name = participant.get("profile_name")
            
            # If it's the speak command, only show profiles owned by the command user (unless bot owner)
            if interaction.command.name == "speak":
                if owner_id != interaction.user.id and interaction.user.id != int(defaultConfig.DISCORD_OWNER_ID):
                    continue
            
            user_data = self._get_user_data_entry(owner_id)
            is_borrowed = profile_name in user_data.get("borrowed_profiles", {})
            effective_owner_id = owner_id
            effective_profile_name = profile_name
            if is_borrowed:
                borrowed_data = user_data["borrowed_profiles"][profile_name]
                effective_owner_id = int(borrowed_data["original_owner_id"])
                effective_profile_name = borrowed_data["original_profile_name"]
            
            display_name = effective_profile_name
            appearance_data = self.user_appearances.get(str(effective_owner_id), {}).get(effective_profile_name, {})
            if appearance_data and appearance_data.get("custom_display_name"):
                display_name = appearance_data.get("custom_display_name")

            owner_user = self.bot.get_user(effective_owner_id)
            owner_name = owner_user.name if owner_user else f"User {effective_owner_id}"

            label = f"{display_name} ({profile_name}) by ({owner_name})"
            value = f"{owner_id}:{profile_name}"

            if current.lower() in label.lower():
                choices.append(app_commands.Choice(name=label[:100], value=value))
        
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
            session = self._ensure_session_hydrated(interaction.channel_id, session.get("type", "multi"))

        user_id = interaction.user.id
        
        whisper_turns = {turn['turn_id']: turn for turn in session.get("unified_log", []) if turn.get("type") == "whisper" and turn.get("whisperer_id") == user_id}
        response_turns = {turn['turn_id']: turn for turn in session.get("unified_log", [])}

        # Pair up whispers with their responses
        paired_whispers = []
        log = session.get("unified_log", [])
        for i, turn in enumerate(log):
            if turn.get("turn_id") in whisper_turns:
                # Find the next turn that is a private_response from the same profile
                if i + 1 < len(log):
                    next_turn = log[i+1]
                    t_key = turn.get("target_key")
                    s_key = next_turn.get("speaker_key")
                    
                    # [FIXED] Ensure keys exist before casting to tuple to prevent silent crashes
                    if next_turn.get("type") == "private_response" and t_key and s_key and tuple(s_key) == tuple(t_key):
                        paired_whispers.append((turn, next_turn))

        if not paired_whispers:
            await interaction.followup.send("You have no whisper history in this session.", ephemeral=True)
            return

        view = WhisperHistoryView(self, interaction, paired_whispers)
        await interaction.followup.send(embed=view._get_current_embed(), view=view, ephemeral=True)

async def setup(bot: commands.Bot):
    await bot.add_cog(GeminiAgent(bot))
# --- End of GeminiCog.py ---