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

# [LEGACY]
import google.generativeai as genai
import google.generativeai.types as genai_types
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core import exceptions as api_exceptions
# [NEW]
from google import genai as google_genai
from google.genai import types as google_genai_types

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
import datetime
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
        genai.configure(api_key=api_key)

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
            model = genai.GenerativeModel(model_name, safety_settings=DEFAULT_SAFETY_SETTINGS)
            gen_config = genai.types.GenerationConfig(temperature=0.3)
            response = await model.generate_content_async(generation_prompt, generation_config=gen_config)
            
            if not response.text:
                raise ValueError("AI returned an empty response, possibly due to a safety filter.")

            response_text = response.text.strip()
            
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

    @app_commands.command(name="speak", description="Anonymously speak as one of your profiles (Admin Only).")
    @app_commands.checks.cooldown(2, 10.0, key=lambda i: i.user.id)
    @app_commands.guild_only()
    @is_admin_or_owner_check()
    @app_commands.autocomplete(profile_name=CoreMixin.profile_autocomplete, method=CoreMixin.speak_method_autocomplete)
    @app_commands.describe(
        profile_name="Your personal or borrowed profile to speak as.",
        message="The message to send. If omitted, a multi-line input box will appear.",
        method="The method to send the message with. Defaults to 'auto'."
    )
    async def speak_slash(self, interaction: discord.Interaction, profile_name: str, method: Literal['auto', 'webhook', 'child_bot'] = 'auto', message: Optional[str] = None):
        if message:
            await interaction.response.defer(ephemeral=True)
            await self._execute_speak_as(
                interaction_to_respond=interaction,
                channel=interaction.channel,
                author=interaction.user,
                profile_name=profile_name,
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
                    profile_name=profile_name,
                    message=message_text,
                    method=method
                )
            
            modal = ActionTextInputModal(
                title=f"Speak as '{profile_name}'",
                label="Message Content",
                placeholder="Enter the message to send...",
                on_submit_callback=modal_callback
            )
            await interaction.response.send_modal(modal)

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
        app_commands.Choice(name="Freewill (Premium)", value="freewill")
    ])
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    async def session_config_slash(self, interaction: discord.Interaction, mode: str):
        if not self.has_lock: return
        
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
            if not (interaction.user.guild_permissions.administrator or interaction.user.id == int(defaultConfig.DISCORD_OWNER_ID)):
                await interaction.response.send_message("You must be an administrator to configure a Regular session.", ephemeral=True)
                return

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
            
            if ch_id in self.multi_profile_channels:
                session = self.multi_profile_channels[ch_id]
                if session.get("type") != "freewill":
                    current_profiles = list(session.get("profiles", []))
                    current_prompt = session.get("session_prompt")
                    current_mode = session.get("session_mode", "sequential")

            view = MultiProfileSelectView(self, interaction.user.id, as_admin_scope=True, current_profiles=current_profiles, current_prompt=current_prompt, current_mode=current_mode)
            self.active_session_config_views[interaction.user.id] = view
            
            # Initial message state
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
                from google.generativeai.types import content_types
                dummy_model = genai.GenerativeModel('gemini-flash-latest')
                session["chat_sessions"][(interaction.user.id, profile_name)] = dummy_model.start_chat(history=[])
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
            from google.generativeai.types import content_types
            dummy_model = genai.GenerativeModel('gemini-flash-latest')
            new_participant_key = (new_participant['owner_id'], new_participant['profile_name'])
            
            # Rebuild history from unified log for the new participant
            participant_history = []
            if session.get("is_hydrated"):
                for turn in session.get("unified_log", []):
                    speaker_key = tuple(turn.get("speaker_key", []))
                    role = 'model' if speaker_key == new_participant_key else 'user'
                    content_obj = content_types.to_content({'role': role, 'parts': [turn.get("content")]})
                    participant_history.append(content_obj)
            
            session["chat_sessions"][new_participant_key] = dummy_model.start_chat(history=participant_history)

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
        if not session.get('is_running'):
            session['worker_task'] = self.bot.loop.create_task(self._multi_profile_worker(interaction.channel_id))

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

        dummy_model = genai.GenerativeModel('gemini-flash-latest')
        for p_key in session["chat_sessions"].keys():
            session["chat_sessions"][p_key] = dummy_model.start_chat(history=[])
        
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
        try:
            messages_to_delete = await interaction.channel.purge(limit=amount, check=check_fn, before=interaction.created_at, reason=f"Purge by {interaction.user}")
            for msg in messages_to_delete:
                self.purged_message_ids.add(msg.id)
            
            progress_message = await interaction.followup.send(f"Deleted {len(messages_to_delete)} message(s). Now cleaning them from my memory...", ephemeral=True)

            # Group turns by session to modify each session only once
            single_turns_to_remove = {} # { (session_key, session_type): {turn_indices} }
            multi_turns_to_remove = {}  # { (channel_id, session_type): {history_lines} }
            
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

                    if isinstance(turn_info[0], tuple): # Single/Global session
                        session_key, session_type, turn_index = turn_info
                        session_id = (session_key, session_type)
                        if session_id not in single_turns_to_remove:
                            single_turns_to_remove[session_id] = set()
                        single_turns_to_remove[session_id].add(turn_index)
                    elif isinstance(turn_info[0], int): # Multi/Freewill session
                        try:
                            channel_id, session_type, history_line = turn_info
                            session_id = (channel_id, session_type)
                            if session_id not in multi_turns_to_remove:
                                multi_turns_to_remove[session_id] = set()
                            multi_turns_to_remove[session_id].add(history_line)
                        except ValueError:
                            # Ignore malformed/old turn_info data
                            continue

            # --- New block: Clean mapping files before cleaning history ---
            for (session_key, session_type), indices in single_turns_to_remove.items():
                mapping_key = self._get_mapping_key_for_session(session_key, session_type)
                if mapping_key in self.mapping_caches:
                    keys_to_delete = [
                        msg_id for msg_id, t_info in self.mapping_caches[mapping_key].items()
                        if isinstance(t_info, (list, tuple)) and len(t_info) > 2 and t_info[2] in indices
                    ]
                    for msg_id in keys_to_delete:
                        self.mapping_caches[mapping_key].pop(msg_id, None)

            for (channel_id, session_type), lines_to_delete in multi_turns_to_remove.items():
                mapping_key = (session_type, channel_id)
                if mapping_key in self.mapping_caches:
                    keys_to_delete = [
                        msg_id for msg_id, t_info in self.mapping_caches[mapping_key].items()
                        if isinstance(t_info, (list, tuple)) and len(t_info) > 2 and t_info[2] in lines_to_delete
                    ]
                    for msg_id in keys_to_delete:
                        self.mapping_caches[mapping_key].pop(msg_id, None)
            # --- End of new block ---

            cleaned_turns_count = 0
            # Process single/global sessions
            for (session_key, session_type), turn_ids_to_delete in single_turns_to_remove.items():
                # Decrement Counters
                owner_id, profile_name = session_key[1], session_key[2]
                ltm_counter_key = (owner_id, profile_name, "guild")
                if ltm_counter_key in self.message_counters_for_ltm:
                    self.message_counters_for_ltm[ltm_counter_key] = max(0, self.message_counters_for_ltm[ltm_counter_key] - len(turn_ids_to_delete))

                mapping_key = self._get_mapping_key_for_session(session_key, session_type)
                if mapping_key in self.mapping_caches:
                    self.mapping_caches[mapping_key].pop('grounding_checkpoint', None)

                hot_cache = self.chat_sessions if session_type == 'single' else self.global_chat_sessions
                chat = hot_cache.get(session_key)
                if not chat:
                    dummy_model = genai.GenerativeModel('gemini-flash-latest')
                    chat = self._load_session_from_disk(session_key, session_type, dummy_model)
                    if chat: hot_cache[session_key] = chat
                
                if chat:
                    indices_to_delete = []
                    for i, turn in enumerate(chat.history):
                        if turn.role == 'user' and turn.parts and hasattr(turn.parts[0], 'text'):
                            for turn_id in turn_ids_to_delete:
                                if f"[TURN_ID:{turn_id}]" in turn.parts[0].text:
                                    indices_to_delete.append(i)
                                    break
                    
                    for index in sorted(indices_to_delete, reverse=True):
                        if len(chat.history) > index + 1:
                            del chat.history[index : index + 2]
                            cleaned_turns_count += 1
                    if not chat.history:
                        hot_cache.pop(session_key, None)
                        self.session_last_accessed.pop(session_key, None)
                        self._delete_session_from_disk(session_key, session_type)
                        self.ltm_recall_history.pop(session_key, None)
                    else:
                        self.session_last_accessed[session_key] = time.time()

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
                        from google.generativeai.types import content_types
                        dummy_model = genai.GenerativeModel('gemini-flash-latest')
                        for p_data in session["profiles"]:
                            p_key = (p_data['owner_id'], p_data['profile_name'])
                            participant_history = []
                            for turn in session["unified_log"]:
                                speaker_key = tuple(turn.get("speaker_key", []))
                                role = 'model' if speaker_key == p_key else 'user'
                                content_obj = content_types.to_content({'role': role, 'parts': [turn.get("content")]})
                                participant_history.append(content_obj)
                            session["chat_sessions"][p_key] = dummy_model.start_chat(history=participant_history)

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
    @app_commands.autocomplete(profile_name=CoreMixin.profile_autocomplete)
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
                # Need a dummy model to load session from disk
                dummy_model = genai.GenerativeModel('gemini-flash-latest') 
                session_data = self._load_session_from_disk(model_cache_key, 'global_chat', dummy_model)
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
            "Admin Setup Required!",
            "Welcome! To use MimicAI on this server, an API key must be configured.\n\n"
            "**Option A: Google Gemini (Recommended)**\n"
            "1. Go to Google AI Studio (`aistudio.google.com`).\n"
            "2. Create a free API key.\n\n"
            "**Option B: OpenRouter (Optional)**\n"
            "1. Go to `openrouter.ai` to get a key for models like Claude or Llama.\n\n"
            "**Step 2: Submit via Direct Message**\n"
            "For security, API keys are managed in a private DM with the bot.\n"
            "• Send the command ` ``/settings`` ` to me in a DM.\n"
            "• Select the server you want to configure.\n"
            "• Click **Edit Primary Key**.\n\n"
            "**New: Key Pools**\n"
            "Users can also contribute their own keys to a server's 'Pool'. The bot will use the Primary Key if available, and fall back to the Pool if needed.\n\n"
            "*Once a key is set, this help message will be replaced with the full user guide.*"
        )

        user_docs_content = [
            ("Table of Contents",
             "Welcome to MimicAI! This guide explains everything from creating your first character to advanced server administration. Use the dropdown menu below to jump to a section.\n\n"
             "**Core Concepts:**\n"
             "• Quickstart Guide\n"
             "• Core Concepts: Profiles\n"
             "• The Three Layers of Memory\n"
             "• Understanding Sessions\n\n"
             "**Features & Deep Dives:**\n"
             "• Deep Dive: Long-Term Memory (LTM)\n"
             "• Deep Dive: Training Examples\n"
             "• Deep Dive: Grounding (Web Search)\n"
             "• Deep Dive: Image Generation\n"
             "• Deep Dive: Advanced Parameters\n"
             "• Deep Dive: Anti-Repetition Critic\n"
             "• Appearances\n"
             "• Profile Hub (Sharing & Publishing)\n"
             "• Freewill System\n"
             "• Child Bots (Advanced)\n\n"
             "**Technical & Admin:**\n"
             "• Understanding Costs & Models\n"
             "• Command Reference (All Users)\n"
             "• Command Reference (Admins)\n"
             "• Permissions & Setup Guide\n"
             "• Disclaimers & Policy\n\n"
             "For more help, join our support server: https://discord.gg/yaM9Q53wGG"),
            ("Quickstart Guide: Your First Profile",
             "Follow these steps to create and activate your first AI character in under 5 minutes.\n\n"
             "**Step 1: Create the Profile**\n"
             "This command creates a new, blank character sheet.\n"
             "• `` `/profile create profile_name:<unique_name>` ``\n\n"
             "**Step 2: Open the Editor**\n"
             "This is your all-in-one dashboard for editing everything about your character.\n"
             "• `` `/profile manage profile_name:<your_profile_name>` ``\n\n"
             "**Step 3: Define the Character**\n"
             "In the dashboard, use the buttons to shape your AI:\n"
             "• **Edit Persona:** Give your character a backstory, personality, likes, and dislikes. This is their soul.\n"
             "• **Edit Instructions:** Give the AI specific rules on how to speak and behave. This is their rulebook.\n\n"
             "**Step 4: Activate in this Channel**\n"
             "To start talking to your new profile, you need to make it active here.\n"
             "• `` `/profile swap profile_name:<your_profile_name>` ``\n\n"
             "**Step 5: Start the Conversation!**\n"
             "Mention the bot to begin.\n"
             "• `@MimicAI Hello, it's nice to meet you!`"),
            ("Core Concepts: Profiles",
             "Profiles are the heart of the bot. Each profile is a unique 'character sheet' that defines an AI's personality, knowledge, and behavior.\n\n"
             "**Personal Profiles:**\n"
             "These are the profiles you create and own. You have full control over their persona, instructions, and all parameters.\n"
             "• **Free Tier:** Up to 5 personal profiles.\n"
             "• **Premium Tier:** Up to 50 personal profiles.\n\n"
             "**Borrowed Profiles:**\n"
             "When you borrow a profile from another user (via the Hub), you get a live link to its core identity. You can customize local settings (like realistic typing or memory scope) without affecting the original.\n"
             "• **Free Tier:** Up to 5 borrowed profiles.\n"
             "• **Premium Tier:** Up to 50 borrowed profiles.\n\n"
             "**Self-Healing:** If the original creator deletes a profile you have borrowed, it will automatically be removed from your list to keep your inventory clean."),
            ("The Three Layers of Memory",
             "The bot uses a sophisticated three-layer memory system to create believable and consistent characters.\n\n"
             "**1. Short-Term Memory (The Conversation)**\n"
             "• **What it is:** The bot's active memory of recent messages. This is adjustable for each of your personal profiles from **0 to 50** conversational exchanges.\n"
             "• **Function:** Provides immediate conversational context.\n"
             "• **Control:** Adjusted via `` `/profile manage` `` -> `Set Generation Parameters & STM`. Cleared with the `` `/refresh` `` command.\n\n"
             "**2. Long-Term Memory (The Notebook)**\n"
             "• **What it is:** AI-generated summaries of key facts and events from past conversations.\n"
             "• **Function:** Allows a profile to remember important details (like names, relationships, or past events) over weeks or months.\n"
             "• **Control:** Managed via `` `/profile data manage` ``.\n\n"
             "**3. Training Examples (The Rulebook)**\n"
             "• **What it is:** Your hand-written, explicit instructions on *how* a profile should speak and behave.\n"
             "• **Function:** The most powerful tool for defining a character's unique voice, personality, and style.\n"
             "• **Control:** Managed via `` `/profile data manage` ``."),
            ("Understanding Sessions",
             "**1. Regular Sessions (Manual)**\n"
             "• **What it is:** A standard chat mode where you pick a cast of characters (profiles) to participate.\n"
             "• **How to Start:** Use `` `/session config mode:Regular` ``. This opens a setup menu where you can select participants.\n"
             "• **Features:** Multi-profile conversation, image generation, and manual control via `/session swap`.\n\n"
             "**2. Freewill Sessions (Premium)**\n"
             "• **What it is:** A dynamic mode where characters can speak on their own (Proactive) or respond based on chance or keywords (Reactive).\n"
             "• **How to Start:** Use `` `/session config mode:Freewill` ``.\n"
             "• **Features:** Characters act autonomously based on their personality settings ('Introverted' vs 'Outgoing'). Great for creating a 'living' server environment.\n\n"
             "**Default Behavior:**\n"
             "If no session is active, the bot will simply not respond, unless you mention it directly."),
            ("Deep Dive: Long-Term Memory (LTM)",
             "**What is it?**\nLTM allows your profile to remember key facts from past conversations by having an AI create summaries of important interactions.\n\n"
             "**How does it work?**\nAfter a certain number of messages in a channel, the bot can automatically summarize the recent conversation and save it as a memory. When you talk to the profile later, it searches these memories for relevant information to provide better context.\n\n"
             "**What should I know?**\n"
             "• **Custom Prompts:** You can edit the exact instructions the AI uses to summarize conversations for each of your personal profiles via the `` `/profile manage` `` dashboard.\n"
             "• **Privacy (LTM Scope):** You can control who can access a profile's memories using the LTM Scope setting in `` `/profile manage` ``. The default is 'Server-Exclusive', meaning memories are only recalled in the server they were made in.\n"
             "• **Cost:** LTM creation involves an extra API call to a fast model (`flash-lite`)."),
            ("Deep Dive: Training Examples",
             "**What is it?**\nTraining Examples are specific `User Input` -> `Bot Response` pairs you provide to teach your profile *how* to speak.\n\n"
             "**How it works?**\nWhen you send a message, the bot finds the most similar Training Examples you've provided and includes them in its prompt.\n\n"
             "**Best Practices:**\nFocus on **style and personality**, not just facts.\n"
             "• **Bad:** `User: What color is the sky?` -> `Bot: The sky is blue.`\n"
             "• **Good:** `User: What a beautiful day!` -> `Bot: It's tolerable, I suppose. The sun is a bit loud.`\n\n"
             "**Cost:**\nThis feature uses the embedding API to find matches. While extremely cheap ($0.10 per 1M tokens), it is **not free** and does generate API usage."),
            ("Deep Dive: Grounding (Web Search)",
             "**What is it?**\nGrounding allows a profile to perform a Google Search to answer questions about recent events.\n\n"
             "**Requirements:**\n"
             "• **Google API Key:** You MUST provide a Google AI Studio key in `/settings`. OpenRouter keys cannot be used for this feature.\n\n"
             "**How it works?**\nThe bot uses a smart model to decide *if* your question requires a search. If 'yes', it performs the search and summarizes the results.\n\n"
             "**Reliability (Fail-Open):**\nIf the search API fails (e.g., Google is down), the bot will **fail open**: it will skip the search and answer from its own knowledge rather than crashing.\n\n"
             "**What should I know?**\n"
             "• **Cost:** A grounded response requires **one extra API call** to a specialized model (`gemini-2.0-flash`) before the final response is generated."),
            ("Deep Dive: Image Generation",
             "**What is it?**\nThis feature allows the bot's active profile to generate an image based on your text prompt, and then present it to you in-character.\n\n"
             "**Requirements:**\n"
             "• **Google API Key (Paid Tier):** This feature requires a Google API key on a project with billing enabled. It **cannot** be used with Free Tier keys or OpenRouter keys.\n\n"
             "**How it works?**\n"
             "1. Use the `!image` or `!imagine` prefix (e.g., `!image a majestic castle`).\n"
             "2. The bot first generates the image using the **Gemini 2.5 Flash Image** model.\n"
             "3. It then shows this image to the active text profile, asking it to comment on the image it 'created'.\n"
             "4. The final message includes both the in-character text and the generated image.\n\n"
             "**Generating Self-Portraits:**\n"
             "You can add a description to the **Appearance** section of a profile's Persona (via `` `/profile manage` ``). If your image prompt contains second-person pronouns (like 'you', 'your') or the profile's name, this appearance data will be automatically included in the prompt to guide the AI in creating a self-portrait.\n\n"
             "**Editing & Combining Images:**\nTo edit or combine images, simply **reply** to a message containing an image and use the `!image` command. You can also attach a second image to your reply. The bot will use up to two images (one from the reply, one from the new attachment) as references for the new generation.\n\n"
             "**COST WARNING:**\nEach use requires **two API calls**: one to the image model and one to the text model."),
            ("Deep Dive: Advanced Parameters",
             "**What is it?**\nFine-tuning controls for how the AI selects words. These settings are primarily for **OpenRouter** models but may affect some Google models.\n\n"
             "**Parameters:**\n"
             "• **Frequency Penalty (-2.0 to 2.0):** Penalizes words based on how many times they have appeared in the text so far. High values decrease repetition.\n"
             "• **Presence Penalty (-2.0 to 2.0):** Penalizes words based on *if* they have appeared at least once. Encourages introducing new topics.\n"
             "• **Repetition Penalty (0.0 to 2.0):** A multiplicative penalty for repeated tokens.\n"
             "• **Min P (0.0 to 1.0):** Discards possible tokens if their probability is less than a percentage of the most likely token's probability. Good for balancing logic and creativity.\n"
             "• **Top A (0.0 to 1.0):** Limits the token pool based on the probability of the top token. If the top token is very likely, the pool shrinks."),
            ("Deep Dive: Anti-Repetition Critic",
             "**What is it?**\nAn automated quality-control system that reads the conversation history before the bot speaks to detect repetitive loops.\n\n"
             "**How it works?**\n1. It analyzes the last few turns of conversation using a fast, cheap AI model.\n2. It looks for verbatim phrases or sentence structures that have been repeated by the character.\n3. If a loop is found, it injects a strict 'Negative Constraint' into the system prompt for the next turn (e.g., 'Do not start sentences with *Well...*').\n\n"
             "**Trade-offs:**\n"
             "• **Pros:** Drastically reduces 'broken record' behavior where a character gets stuck repeating the same phrase.\n"
             "• **Cons:** Adds **latency** (time to reply) and **cost** (one extra API call per message)."),
            ("Appearances",
             "Appearances allow you to give your profiles a custom name and avatar, separate from the main bot's identity.\n\n"
             "**How it Works:**\n"
             "1. Open the profile dashboard with `` `/profile manage` ``.\n"
             "2. Use the **'Edit Appearance'** button. This will let you create or edit an appearance linked to that profile.\n"
             "3. When that profile is active, the bot will automatically use the custom name and avatar you set.\n\n"
             "> **Permission Needed:** For this feature to work, the bot's role must have the **`Manage Webhooks`** permission in the channel.\n\n"
             "**Management:**\n"
             "• Appearances are managed directly from the `` `/profile manage` `` dashboard."),
            ("Profile Hub (Sharing & Publishing)",
             "The **Profile Hub** is your central destination for sharing and discovering characters.\n\n"
             "**Private Sharing (Free)**\n"
             "Share profiles directly with friends or generate single-use Share Codes via `Manage My Shares` -> `Private Mode`.\n\n"
             "**Public Library**\n"
             "Premium users can **publish** their profiles to the global library. Published profiles appear in the **Public Library** tab for anyone to browse and borrow.\n"
             "• **Safety First:** All published profiles undergo an automated AI safety check. Explicit content is strictly prohibited in the public library.\n"
             "• **Ownership:** You retain full control. Unpublishing a profile removes it from the library immediately."),
            ("Freewill System",
             "The Freewill system allows profiles to become active participants in a server, either by starting conversations on their own or reacting to keywords.\n\n"
             "**Living vs. Lurking Channels**\n"
             "• **Living:** In these channels, the bot can proactively start scenes between two or more opted-in profiles at random intervals.\n"
             "• **Lurking:** In these channels, profiles will only speak if they are directly replied to or if a user says one of their 'wakewords'.\n\n"
             "**How to Participate**\n"
             "1. An admin must first enable the system and designate channels using `` `/session config mode:Freewill` ``.\n"
             "2. Admins can then opt-in their profiles using the `Participants` tab in the Freewill dashboard.\n"
             "3. For each profile, you can set its **Personality** (which determines its chance to speak) and its **Wakewords**.\n\n"
             "**Interacting with Freewill:**\nProfiles in a reactive (Lurking) freewill session can be prompted to generate images using the `!image` or `!imagine` command. The bot will randomly select one of the opted-in profiles to fulfill the request."),
            ("Child Bots (Premium)",
             "Child Bots are separate Discord bot applications that you own and control, which act as dedicated 'puppets' for your profiles.\n\n"
             "**Why Use a Child Bot?**\n"
             "• **Unique Identity:** It gets its own name, avatar, and presence in the server member list.\n"
             "• **Multi-Server Presence:** A single child bot can be active on multiple servers at once.\n"
             "• **Autonomous Participation:** Child Bots can be opted into the Freewill system to act as autonomous characters in your server.\n\n"
             "**Setup Guide:**\n"
             "1. Go to the Discord Developer Portal, create a new Application, and add a 'Bot' to it. Copy the bot's **Token**.\n"
             "2. In a **DM with this bot**, use the `` `/settings` `` command.\n"
             "3. Navigate to `Child Bots` -> `Create New Child Bot` and follow the prompts.\n"
             "4. **Invite Your Bot:** Use the OAuth2 URL Generator in the Developer Portal to create an invite link with the following permissions: **`Send Messages`**, **`Read Message History`**, and **`Embed Links`**."),
            ("Command Reference (All Users)",
             "**Profile Management**\n"
             "• `` `/profile manage <name>` ``: The main dashboard for editing a profile.\n"
             "• `` `/profile swap <name>` ``: Swaps your active profile in the current channel.\n"
             "• `` `/session view` ``: Shows details about the current session and its participants.\n"
             "• `` `/profile list` ``: Lists all your personal and borrowed profiles.\n"
             "• `` `/profile hub` ``: Browse the public library and manage shares.\n\n"
             "**Data Management**\n"
             "• `` `/profile data manage <name>` ``: Add, edit, view, and delete LTM and Training Examples.\n\n"
             "**Chat Commands**\n"
             "• `` `/profile global_chat [profile] [message]` ``: Opens the Global Chat History UI. If a message is provided, sends it immediately.\n"
             "• `` `/whisper [profile] [message]` ``: Sends a private message to a session participant.\n"
             "• `` `/terms` ``: Displays a direct link to our Terms of Service and Privacy Policy."),
            ("Command Reference (Admins)",
             "**Server & Session**\n"
             "• `` `/session config` ``: Start or configure a Regular or Freewill session.\n"
             "• `` `/session swap` ``: Quickly swap, add, or remove a profile from the active session.\n"
             "• `` `/suspend` ``: Clear the active session in the current channel.\n"
             "• `` `/refresh` ``: Clear the participating profiles' short-term memory context for the channel.\n"
             "• `` `/purge` ``: Delete messages and scrub them from the session's memory.\n"
             "• `` `/profile speak` ``: Speak as any profile in the server."),
            ("Permissions & Setup Guide",
             "**Required Permissions:**\n"
             "• `Send Messages`, `Read Message History`\n\n"
             "**Optional Permissions:**\n"
             "• `Manage Webhooks` (Required for Appearances)\n"
             "• `Manage Messages` (Required for `/purge`)\n\n"
             "**Setup Steps:**\n"
             "1. DM the bot `` `/settings` `` to add API keys (Google or OpenRouter).\n"
             "2. Use `` `/session config` `` to configure your first conversation."),
            ("Disclaimers & Policy",
             "**API USAGE**\n"
             "By providing an API key, you agree to the terms of the respective provider (Google Gemini or OpenRouter). Data is processed to generate responses and is not used by the bot developer for training.\n\n"
             "**SESSION HISTORY**\n"
             "To support context, a rolling history of the last 50 exchanges is stored securely on disk. You can delete this history at any time via `/refresh` or the Global Chat UI.\n\n"
             "**COST RESPONSIBILITY**\n"
             "Using Pro models or high-end OpenRouter models (like Opus) can be expensive. The bot developer is not responsible for costs incurred on your API keys."),
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

    async def whisper_profile_autocomplete(self, interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
        session = self.multi_profile_channels.get(interaction.channel_id)
        if not session:
            return []

        choices = []
        for participant in session.get("profiles", []):
            owner_id = participant.get("owner_id")
            profile_name = participant.get("profile_name")
            
            # Determine the display name
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

            # The value must be the internal profile_name, not the display name
            if current.lower() in display_name.lower():
                choices.append(app_commands.Choice(name=display_name, value=profile_name))
        
        return choices[:25]

    @app_commands.command(name="whisper", description="Send a private message to a profile in an active multi-profile session.")
    @app_commands.checks.cooldown(3, 30.0, key=lambda i: i.user.id)
    @app_commands.guild_only()
    @app_commands.autocomplete(profile=whisper_profile_autocomplete)
    @app_commands.describe(
        profile="The participant to whisper to. Leave blank to view history.",
        message="The private message to send. Leave blank to view history."
    )
    async def whisper_slash(self, interaction: discord.Interaction, profile: Optional[str] = None, message: Optional[str] = None):
        session = self.multi_profile_channels.get(interaction.channel_id)
        if not session or session.get("type") not in ["multi", "freewill"]:
            await interaction.response.send_message("This command can only be used in an active multi-profile or freewill session.", ephemeral=True)
            return

        if profile and message:
            target_participant = next((p for p in session.get("profiles", []) if p.get("profile_name") == profile), None)
            if not target_participant:
                await interaction.response.send_message(f"Could not find a participant named '{profile}' in the current session.", ephemeral=True)
                return
            await interaction.response.defer(ephemeral=True, thinking=True)
            await self._execute_whisper(interaction, target_participant, message)
        elif profile and not message:
            target_participant = next((p for p in session.get("profiles", []) if p.get("profile_name") == profile), None)
            if not target_participant:
                await interaction.response.send_message(f"Could not find a participant named '{profile}' in the current session.", ephemeral=True)
                return

            async def modal_callback(modal_interaction: discord.Interaction, message_text: str):
                await modal_interaction.response.defer(ephemeral=True, thinking=True)
                await self._execute_whisper(modal_interaction, target_participant, message_text)

            modal = ActionTextInputModal(
                title=f"Whisper to {profile}",
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
                    if next_turn.get("type") == "private_response" and tuple(next_turn.get("speaker_key")) == tuple(turn.get("target_key")):
                        paired_whispers.append((turn, next_turn))

        if not paired_whispers:
            await interaction.followup.send("You have no whisper history in this session.", ephemeral=True)
            return

        view = WhisperHistoryView(self, interaction, paired_whispers)
        await interaction.followup.send(embed=view._get_current_embed(), view=view, ephemeral=True)

async def setup(bot: commands.Bot):
    await bot.add_cog(GeminiAgent(bot))
# --- End of GeminiCog.py ---