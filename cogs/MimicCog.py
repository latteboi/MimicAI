from .utils.constants import (
    ALLOWED_MODELS, CHANNEL_MODEL_CACHE_MAX_SIZE, COG_LOCK_FILE_PATH, DATA_DIR,
    DEFAULT_PROFILE_GENERATOR_PROMPT, DEFAULT_SAFETY_SETTINGS, DEFAULT_SYSTEM_INSTRUCTION,
    FALLBACK_MODEL_NAME, LOCK_REFRESH_INTERVAL_SECONDS, LOCK_STALE_THRESHOLD_SECONDS,
    MAX_MULTI_PROFILES, MOD_DATA_DIR, PRIMARY_MODEL_NAME, PUBLIC_PROFILES_DIR,
    GAME_CACHE_MAX_SIZE, PURGED_MESSAGE_ID_CACHE_MAX_SIZE,
    PURGE_BUSY_WAIT_TIMEOUT_SECONDS, SERVERS_DIR,
    SESSIONS_GLOBAL_DIR, SESSION_BUSY_FLAGS, TRAIN_ARMED_CACHE_MAX_SIZE, TRAIN_INPUT_EMOJI,
    TRAIN_OUTPUT_EMOJI, USERS_DIR, defaultConfig, is_admin_or_owner_check, is_owner_in_dm_check,
)
from .services.api_service import OpenRouterModel, GoogleGenAIModel
from .listeners.event_listeners import EventListeners
from .gui.base_components import ActionTextInputModal, DropdownContentView, InviteView
from .gui.gui_data import PrivacyDashboardView, ImportPassphraseModal, BulkExportView
from .gui.gui_hub import HubHomeView
from .gui.gui_sessions import (
    GlobalChatPlayView, GlobalChatHistoryView, WhisperHistoryView, SessionSwapListView,
    SessionView, SessionConfigView, SessionAuditView
)
from .gui.gui_settings import SettingsHomeView, ParentPresenceView, ShutdownConfirmView
from .gui.gui_profiles import ProfileManageView, BulkManageView
from .gui.gui_resolve import (
    autocorrect_profile, gather_owned_candidates, gather_participant_candidates,
    suggest_profile,
)
from .gui.gui_mod import ModStatsView
from .managers.storage_manager import StorageManager
from .managers.profile_manager import ProfileManager
from .managers.session_manager import SessionManager
from .managers.memory_manager import MemoryManager
from .managers.server_manager import ServerManager
from .managers.child_bot_manager import ChildBotManager
from .services.media_service import MediaService
from .services.tools_service import ToolsService
from .services.generation_service import GenerationService
from .services.api_service import APIService
from .services.help_service import HelpService
from .services.game_service import GameService
from .services.games.eights import RuleSet as GameRuleSet

from discord.ext import commands, tasks
import discord
from discord import app_commands, ui

from cryptography.fernet import Fernet
import asyncio
import os
import orjson as json
import datetime
import uuid
from typing import List, Dict, Tuple, Set, Literal, Any, Optional, get_args
import traceback
import time
import platform
from collections import OrderedDict
import re
import pathlib
from .utils.helpers import _resolve_safety_settings, _scrub_response_text

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

class MimicCog(commands.Cog, EventListeners):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.manager_queue = bot.manager_queue
        self.cog_id = str(uuid.uuid4()) 
        self.has_lock = False
        
        # Client placeholder for session-specific usage
        self.client = None
        # Documentation shards for Help Mode, embedded by HelpService, plus the
        # pre-normalised (N, dims) float32 matrix over them that the search uses.
        self.doc_vectors = []
        self.doc_matrix = None
        
        self._try_acquire_lock()

        if self.has_lock:
            print(f"MimicCog {self.cog_id} acquired lock and is ACTIVE.")
            self.refresh_lock_task.start()
        else:
            print(f"MimicCog {self.cog_id} DID NOT acquire lock. Will run in INACTIVE mode.")
            self.reacquire_lock_task.start()

        print(f"MimicCog Init. Models: Primary='{PRIMARY_MODEL_NAME}', Fallback='{FALLBACK_MODEL_NAME}'.")

        self.PUBLIC_PROFILES_DIR = PUBLIC_PROFILES_DIR
        self.USERS_DIR = USERS_DIR
        self.DATA_DIR = DATA_DIR
        self.MOD_DATA_DIR = MOD_DATA_DIR
        self.SESSIONS_GLOBAL_DIR = SESSIONS_GLOBAL_DIR
        self.SERVERS_DIR = SERVERS_DIR
        
        # Only create the active Phase 3 directories to prevent ghost folders on boot
        active_dirs =[self.USERS_DIR, self.DATA_DIR, self.PUBLIC_PROFILES_DIR, self.SERVERS_DIR, self.MOD_DATA_DIR]
        
        for d in active_dirs:
            os.makedirs(d, exist_ok=True)

        self.global_prompts: Dict[str, str] = {}

        try:
            self.fernet = Fernet(defaultConfig.ENCRYPTION_KEY)
        except Exception as e:
            print(f"CRITICAL: Failed to initialize encryption. Ensure ENCRYPTION_KEY is set in defaultConfig.py. Error: {e}")
            self.fernet = None

        self.storage_manager = StorageManager(self)
        self.profile_manager = ProfileManager(self)
        self.session_manager = SessionManager(self)
        self.memory_manager = MemoryManager(self)
        self.server_manager = ServerManager(self)
        self.child_bot_manager = ChildBotManager(self)
        self.media_service = MediaService(self)
        self.tools_service = ToolsService(self)
        self.generation_service = GenerationService(self)
        self.api_service = APIService(self)
        self.help_service = HelpService(self)
        self.game_service = GameService(self)

        self.server_manager._load_global_prompts()

        self.persona_modal_sections_order = ['backstory', 'personality_traits', 'likes', 'dislikes', 'appearance'] 
        
        self.user_indices: LRUCache = LRUCache(max_size=20)
        self.server_indices: LRUCache = LRUCache(max_size=50)
        
        # Memory-bounded caches to prevent RAM growth on long uptime
        self.user_appearances: LRUCache = LRUCache(max_size=50)
        self.message_counters_for_ltm: LRUCache = LRUCache(max_size=200)
        self.child_bot_edit_cooldowns: LRUCache = LRUCache(max_size=50)
        
        self.server_manager._load_channel_webhooks()

        self.share_codes: Dict[str, Dict[str, Any]] = {}

        self.multi_profile_channels: Dict[int, Dict[str, Any]] = {}
        self.sessions_loaded = False

        # Top-level command name -> application command id, for rendering a clickable
        # </name:id> mention. Filled once from the registered tree, so it is bounded by
        # the command count rather than by anything a user can grow.
        self.command_ids: Dict[str, int] = {}
        self.decrypted_key_cache: LRUCache = LRUCache(max_size=100)
        self.server_key_pointers: LRUCache = LRUCache(max_size=200)
        self.profile_shares: Dict[str, List[Dict[str, Any]]] = {}
        self.profile_manager._load_profile_shares()
        self.public_profiles: Dict[str, Dict[str, Any]] = {}
        self.profile_manager._load_public_profiles()
        # Sweep tombstones once at boot. Deletion normally cascades, so a survivor
        # here came from a crash mid-delete or an older build; left alone it renders
        # in the hub as an unborrowable "Unknown" entry forever.
        _pruned = self.profile_manager._prune_public_index()
        if _pruned:
            print(f"Public hub: pruned {len(_pruned)} orphaned entr"
                  f"{'y' if len(_pruned) == 1 else 'ies'} "
                  f"({', '.join(str(e['profile_name']) for e in _pruned[:10])}"
                  f"{', ...' if len(_pruned) > 10 else ''})")
        self.child_bots: Dict[str, Dict[str, Any]] = {}
        self.child_bots_by_owner_profile: Dict[Tuple[int, str], str] = {}
        self.child_bot_manager._load_child_bots()

        # Resolved channel access per (owner_id, profile_name), derived from the
        # profile's content rating. The gate runs on the turn path and the uncached
        # resolve reads up to two encrypted profile files; invalidated by
        # ProfileManager._invalidate_content_rating.
        self.content_rating_cache: LRUCache = LRUCache(max_size=512)
        # Profiles queued for content classification, so a burst of edits collapses
        # into one job each and a failed job can be retried with a bounded count.
        self.pending_classifications: Dict[Tuple[int, str], int] = {}
        # One Event per in-flight job, so a dashboard can wait for the verdict
        # instead of guessing at how long it will take. Created and torn down in
        # lockstep with pending_classifications above, so it is bounded by the same
        # thing: the number of jobs actually running.
        self.classification_events: Dict[Tuple[int, str], asyncio.Event] = {}

        # LRU rather than plain dicts: keyed by (channel_id, owner_id, profile_name), these
        # otherwise grow for the life of the process. The two are written together and read
        # together; if they fall out of step the worst case is a miss, which rebuilds.
        self.channel_models: LRUCache = LRUCache(max_size=CHANNEL_MODEL_CACHE_MAX_SIZE)
        self.channel_model_last_profile_key: LRUCache = LRUCache(max_size=CHANNEL_MODEL_CACHE_MAX_SIZE)

        self.max_history_items = defaultConfig.CHATBOT_MEMORY_LENGTH
        
        self.model_override_warnings_sent: Set[Tuple[int, int, str]] = set()
        self.debug_users: Set[int] = set()
        self.global_chat_sessions: LRUCache = LRUCache(max_size=10)
        self.purged_message_ids: LRUCache = LRUCache(PURGED_MESSAGE_ID_CACHE_MAX_SIZE)
        # /train arms a channel to capture a training example from 1️⃣/2️⃣ reactions.
        # Keyed by channel_id; see EventListeners._handle_train_reaction.
        self.armed_training_channels: LRUCache = LRUCache(TRAIN_ARMED_CACHE_MAX_SIZE)
        self.pending_child_confirmations: LRUCache = LRUCache(max_size=200)
        # Live table games, keyed by channel_id. Bounded like every other channel-keyed
        # cache, but /play refuses past GAME_MAX_CONCURRENT (which is lower), so the LRU
        # never actually evicts a running game out from under its task.
        self.active_games: LRUCache = LRUCache(max_size=GAME_CACHE_MAX_SIZE)
        # Tables that are forming but not dealt, keyed by channel_id. A lobby holds no
        # task and no rules state, so an evicted one is an abandoned guest list rather
        # than an orphan -- but it counts against GAME_MAX_CONCURRENT while it stands,
        # so it cannot be used to queue up more games than the instance will run.
        self.pending_lobbies: LRUCache = LRUCache(max_size=GAME_CACHE_MAX_SIZE)
        self.global_blacklist: Set[int] = set()
        self.server_manager._load_blacklist()
        self.session_last_accessed = {}
        self.eviction_heap = []
        # session_key -> session_type for logs with unpersisted appends. Bounded by the
        # number of live sessions, and drained every SESSION_FLUSH_INTERVAL_SECONDS.
        self.dirty_sessions: Dict[Any, str] = {}
        self.session_manager.evict_inactive_sessions_task.start()
        self.session_manager.flush_dirty_sessions_task.start()
        self.message_cooldown = commands.CooldownMapping.from_cooldown(5, 60.0, commands.BucketType.user)
        self.processed_child_messages: LRUCache = LRUCache(max_size=25)
        self.all_bot_ids: Set[int] = set()
        self.image_gen_semaphore = asyncio.Semaphore(3)
        self.ltm_recall_history: Dict[Any, Dict[str, Tuple[int, float]]] = {}

        # Priority Queues for Premium Fast-Lane
        self.image_request_queue = asyncio.PriorityQueue(maxsize=10)
        self.text_request_queue = asyncio.PriorityQueue()
        
        self.image_gen_workers = []
        self.image_finisher_worker_task = None
        self.active_session_config_views: Dict[int, ui.View] = {}
        self.background_tasks = set()
        self.child_bot_single_sessions = {}
        
        # API Key Health & Tier Tracking
        self.api_key_cooldowns: Dict[str, float] = {}
        
        # Model Stats Initialization
        self.MODELS_DATA_DIR = os.path.join(DATA_DIR, "models")
        os.makedirs(self.MODELS_DATA_DIR, exist_ok=True)
        
        self.trace_ctx_menu = app_commands.ContextMenu(
            name="View Generation Trace",
            callback=self.view_generation_trace
        )
        self.bot.tree.add_command(self.trace_ctx_menu)

    profile_group = app_commands.Group(name="profile", description="Manage your personal bot profiles (persona, instructions).")

    @profile_group.command(name="create", description="Creates a new, blank profile.")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    @app_commands.describe(
        profile_name="The name for your new profile. Must be unique.",
        system_profile="Create as a global System Profile (Bot Owner Only)."
    )
    async def create_profile_slash(self, interaction: discord.Interaction, profile_name: str, system_profile: bool = False):
        await interaction.response.defer(ephemeral=True)

        if system_profile and interaction.user.id != int(defaultConfig.DISCORD_OWNER_ID):
            await interaction.followup.send("❌ **Access Denied:** Creating System Profiles is restricted to the Bot Owner.", ephemeral=True)
            return

        has_access = await self.storage_manager._has_api_key_access(interaction.user.id, interaction.guild_id)
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
        
        is_valid, err_msg = self.profile_manager._is_valid_profile_name(profile_name)
        if not is_valid:
            await interaction.followup.send(f"❌ **Invalid Name:** {err_msg}", ephemeral=True)
            return

        if system_profile:
            owner_id = int(defaultConfig.DISCORD_OWNER_ID)
            owner_index = self.profile_manager._get_user_index(owner_id)
            if profile_name in owner_index.get("system", {}):
                await interaction.followup.send(f"A system profile with the name '{profile_name}' already exists.", ephemeral=True)
                return
            new_profile = self.profile_manager._get_or_create_system_profile(profile_name)
            await interaction.followup.send(f"Successfully created new System Profile '{profile_name}'.\nUse `/profile manage profile_name:{profile_name}` to start editing it.", ephemeral=True)
            return

        index = self.profile_manager._get_user_index(interaction.user.id)
        
        if profile_name in index.get("personal", []) or profile_name in index.get("borrowed", []):
            await interaction.followup.send(f"A profile with the name '{profile_name}' already exists.", ephemeral=True)
            return

        current_count = len(index.get("personal", []))
        limit = defaultConfig.LIMIT_PROFILES

        if current_count >= limit:
            await interaction.followup.send(
                f"**Limit Reached.**\n\nYou have reached the maximum of **{limit}** profiles.", ephemeral=True)
            return

        new_profile = self.profile_manager._get_or_create_user_profile(interaction.user.id, profile_name)
        if new_profile:
            config = new_profile.get('config', {})
            config['created_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            self.profile_manager._save_profile_config(interaction.user.id, profile_name, config)
        
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

        is_valid, err_msg = self.profile_manager._is_valid_profile_name(profile_name)
        if not is_valid:
            await interaction.followup.send(f"❌ **Invalid Name:** {err_msg}", ephemeral=True)
            return
            
        if not prompt.strip():
            await interaction.followup.send("Prompt cannot be empty.", ephemeral=True)
            return

        index = self.profile_manager._get_user_index(interaction.user.id)
        if profile_name in index.get("personal", []) or profile_name in index.get("borrowed", []):
            await interaction.followup.send(f"A profile with the name '{profile_name}' already exists.", ephemeral=True)
            return

        limit = defaultConfig.LIMIT_PROFILES

        if len(index.get("personal", [])) >= limit:
            await interaction.followup.send(f"You have reached the maximum of {limit} personal profiles.", ephemeral=True)
            return

        api_key = self.storage_manager._get_api_key_for_user(interaction.user.id)
        is_or = False
        
        if not api_key:
            api_key = self.storage_manager._get_api_key_for_user(interaction.user.id, "openrouter")
            is_or = True
            
        if not api_key:
            await interaction.followup.send("A personal API key is not configured, so I cannot generate a profile. Please configure one in your `/settings` DM.", ephemeral=True)
            return

        generation_prompt = self.global_prompts.get("PROFILE_GENERATOR", DEFAULT_PROFILE_GENERATOR_PROMPT).format(prompt=prompt)

        status = "api_error"
        try:
            if is_or:
                model_name = 'google/gemini-2.5-flash-lite'
                model = OpenRouterModel(api_key=api_key, model_name=model_name, system_instruction=None, thinking_params={})
            else:
                model_name = 'gemini-2.5-flash-lite'
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

            new_profile = self.profile_manager._get_or_create_user_profile(interaction.user.id, profile_name)
            if not new_profile:
                await interaction.followup.send("Failed to create the profile structure.", ephemeral=True)
                return

            # Encrypt and save the generated data
            encrypted_persona = {key: [self.storage_manager._encrypt_data(line) for line in value.splitlines()] for key, value in generated_data['persona'].items()}
            encrypted_instructions = self.storage_manager._encrypt_data(generated_data['ai_instructions'])

            prompts = new_profile.get('prompts', {})
            prompts['persona'] = encrypted_persona
            
            if not isinstance(prompts.get('ai_instructions'), list):
                prompts['ai_instructions'] = ["", "", "", ""]
            prompts['ai_instructions'][0] = encrypted_instructions
            
            self.profile_manager._save_profile_prompts(interaction.user.id, profile_name, prompts)

            await interaction.followup.send(f"✅ Successfully generated and created new profile '{profile_name}'.\nUse `/profile manage profile_name:{profile_name}` to view or edit it.", ephemeral=True)

        except json.JSONDecodeError:
            await interaction.followup.send("❌ **Generation Failed:** The AI returned an invalid data format. Please try again.", ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"❌ **Generation Failed:** An error occurred: {e}", ephemeral=True)

    @profile_group.command(name="manage", description="Manage all settings for a specific profile from a unified dashboard.")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    @app_commands.describe(profile_name="The name of the personal or borrowed profile to manage.")
    @app_commands.autocomplete(profile_name=EventListeners.master_autocomplete)
    async def manage_profile_slash(self, interaction: discord.Interaction, profile_name: str):
        await interaction.response.defer(ephemeral=True)

        if not self._owns_profile_name(interaction.user.id, profile_name):
            candidates = gather_owned_candidates(self, interaction.user.id)

            # Names are stored lowercase, so "Alice" is not a typo -- it is the same
            # profile typed with a capital. Resolve that silently rather than prompting.
            corrected = autocorrect_profile(profile_name, candidates)
            if corrected is not None:
                await self._open_profile_manage(interaction, corrected)
                return

            async def on_pick(pick_interaction: discord.Interaction, picked: str):
                await self._open_profile_manage(pick_interaction, picked, repaint=True)

            await suggest_profile(self, interaction, profile_name, candidates, on_pick)
            return

        await self._open_profile_manage(interaction, profile_name)

    def command_mention(self, name: str) -> str:
        """`</name:id>` if the id is known, otherwise a plain `/name` code span.

        Discord only renders the clickable form for an id it recognises, and prints the
        raw text otherwise -- so the fallback is not cosmetic. `command_ids` is empty
        until the tree has been fetched, and stays empty if that call fails.
        """
        cid = self.command_ids.get(name)
        return f"</{name}:{cid}>" if cid else f"`/{name}`"

    def _owns_profile_name(self, user_id: int, profile_name: str) -> bool:
        """Whether `user_id` can address `profile_name` at all (personal, borrowed, or system)."""
        index = self.profile_manager._get_user_index(user_id)
        if profile_name in index.get("personal", []) or profile_name in index.get("borrowed", []):
            return True
        return profile_name in self.profile_manager._system_index()

    async def _open_profile_manage(self, interaction: discord.Interaction, profile_name: str,
                                   *, repaint: bool = False):
        """Render the profile dashboard.

        `repaint` edits the interaction's existing message instead of sending a new one,
        which is how a "Did you mean?" prompt turns into the dashboard in place rather
        than leaving a spent suggestion card above it.
        """
        index = self.profile_manager._get_user_index(interaction.user.id)
        is_view_borrowed = profile_name in index.get("borrowed", [])

        embed = await self.profile_manager._build_profile_manage_embed(interaction, profile_name)
        view = ProfileManageView(self, interaction, profile_name, is_view_borrowed)

        if repaint:
            await interaction.edit_original_response(content=None, embed=embed, view=view)
        else:
            await interaction.followup.send(embed=embed, view=view, ephemeral=True)

    bulk_group = app_commands.Group(name="bulk", description="Perform actions on multiple profiles at once.", parent=profile_group)

    @bulk_group.command(name="manage", description="Open the dashboard to perform bulk actions.")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    async def bulk_manage_slash(self, interaction: discord.Interaction):
        view = BulkManageView(self, interaction)
        await view.start(interaction)

    @profile_group.command(name="list", description="Lists all of your saved profile names.")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    async def list_profiles_slash(self, interaction: discord.Interaction): 
        if not self.has_lock and interaction.guild: return
        await interaction.response.defer(ephemeral=True)
        
        index = self.profile_manager._get_user_index(interaction.user.id)
        
        personal_names = sorted(list(index.get("personal", {})))
        borrowed_names = sorted(list(index.get("borrowed", {})))
        
        # System profiles are addressable by everyone -- _resolve_effective_profile
        # has always granted access to them -- so everyone is shown them. This used
        # to read the CALLER's index behind an owner-only gate, and a member's index
        # has no "system" map, so the section was doubly dead for anyone but the bot
        # owner: they could use a System profile but never discover its name.
        # _is_system_name applies the same personal-and-borrowed-shadow-System
        # precedence as resolution, so a name listed here means what it will mean
        # when the user types it.
        system_names = sorted(
            n for n in self.profile_manager._system_index()
            if self.profile_manager._is_system_name(interaction.user.id, n))

        if not personal_names and not borrowed_names and not system_names:
            await interaction.followup.send("You have no saved profiles yet.", ephemeral=True)
            return
        
        embed = discord.Embed(title="Your Profiles", color=discord.Color.purple())

        def build_fields(title, name_list):
            fields = []
            current_val = ""
            chunk_num = 1
            for name in name_list:
                tag = f"`{name}`"
                    
                if len(current_val) + len(tag) + 2 > 1000:
                    fields.append((f"{title} (Cont.)" if chunk_num > 1 else title, current_val))
                    current_val = tag
                    chunk_num += 1
                else:
                    current_val += f", {tag}" if current_val else tag
            if current_val:
                fields.append((f"{title} (Cont.)" if chunk_num > 1 else title, current_val))
            return fields

        for t, v in build_fields("Personal Profiles", personal_names):
            embed.add_field(name=t, value=v, inline=False)

        for t, v in build_fields("Borrowed Profiles", borrowed_names):
            embed.add_field(name=t, value=v, inline=False)

        sys_title = ("System Profiles" if interaction.user.id == int(defaultConfig.DISCORD_OWNER_ID)
                     else "System Profiles (Read-Only)")
        for t, v in build_fields(sys_title, system_names):
            embed.add_field(name=t, value=v, inline=False)

        await interaction.followup.send(embed=embed, ephemeral=True)

    @app_commands.command(name="export", description="Export selected profiles and memories to a plaintext file (DM Only).")
    @app_commands.checks.cooldown(1, 60.0, key=lambda i: i.user.id)
    @app_commands.dm_only()
    async def export_command(self, interaction: discord.Interaction):
        view = BulkExportView(self, interaction.user.id)
        await interaction.response.send_message("### 📤 Profile Export\nSelect profiles and components to export.\n\n*The file will contain encrypted data. Tampered files cannot be imported.*\n", view=view, ephemeral=True)

    @app_commands.command(name="import", description="Import profiles and memories from a MimicAI export file (DM Only).")
    @app_commands.checks.cooldown(1, 30.0, key=lambda i: i.user.id)
    @app_commands.dm_only()
    @app_commands.describe(file="The .mimic file exported from a MimicAI instance.")
    async def import_command(self, interaction: discord.Interaction, file: discord.Attachment):
        if not file.filename.endswith('.mimic'):
            await interaction.response.send_message("❌ Invalid file type. Please upload a `.mimic` file.", ephemeral=True)
            return
        
        try:
            file_bytes = await file.read()
            container = json.loads(file_bytes)
            auth_mode = container.get("auth_mode")
            
            is_official = (self.bot.user and self.bot.user.id == 1376696185947164854)
            if is_official and auth_mode != "master":
                await interaction.response.send_message("❌ The official MimicAI instance strictly rejects third-party or self-hosted profile imports. You can only import files exported directly from this official bot.", ephemeral=True)
                return
                
            if auth_mode == "passphrase":
                await interaction.response.send_modal(ImportPassphraseModal(self, file_bytes))
                return
        except Exception as e:
            await interaction.response.send_message(f"❌ Failed to read or parse file: {e}", ephemeral=True)
            return
        
        await interaction.response.defer(ephemeral=True, thinking=True)
        await self.profile_manager._execute_import(interaction, file_bytes)

    @app_commands.command(name="privacy", description="Manage your data privacy and account deletion.")
    @app_commands.checks.cooldown(1, 60.0, key=lambda i: i.user.id)
    async def privacy_slash(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        view = PrivacyDashboardView(self, interaction.user.id)
        embed = discord.Embed(
            title="Privacy & Data Dashboard",
            description="Request a full export of your data or permanently delete your account and all associated profiles, memories, and settings.",
            color=discord.Color.red()
        )
        await interaction.followup.send(embed=embed, view=view, ephemeral=True)

    @app_commands.command(name="whoami", description="Displays information about this bot's identity.")
    @app_commands.checks.cooldown(1, 10.0, key=lambda i: i.user.id)
    async def whoami_slash(self, interaction: discord.Interaction):
        is_owner = interaction.user.id == int(defaultConfig.DISCORD_OWNER_ID)
        embed = discord.Embed(
            title=f"Bot Identity: {self.bot.user.name}",
            description="Managed by MimicAI Core.",
            color=discord.Color.blue()
        )
        embed.set_thumbnail(url=self.bot.user.display_avatar.url)
        
        embed.add_field(name="Version", value="v0.5.0 Beta", inline=True)
        embed.add_field(name="Global Scope", value=f"{len(self.bot.guilds)} Servers", inline=True)

        if is_owner:
            view = ParentPresenceView(self)
            await interaction.response.send_message(embed=embed, view=view, ephemeral=True)
        else:
            await interaction.response.send_message(embed=embed, ephemeral=True)

    @profile_group.command(name="hub", description="The unified dashboard for managing profiles, sharing, and the public library.")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    async def hub_slash(self, interaction: discord.Interaction):
        # [FIX] Defer immediately to prevent interaction timeout (Error 10062)
        await interaction.response.defer(ephemeral=True)
        
        # Lazy Deletion Check (Disk I/O) happens safely after defer
        removed = await self.profile_manager._validate_and_clean_borrowed_profiles(interaction.user.id)

        view = HubHomeView(self, interaction)
        await view.update_display()

        if removed > 0:
            # Use followup because response was already deferred/used
            await interaction.followup.send(f"ℹ️ Notice: {removed} borrowed profiles were removed because their original creators deleted them.", ephemeral=True)

    session_group = app_commands.Group(name="session", description="Manage chat sessions.", guild_only=True)

    @session_group.command(name="config", description="Configure the unified chat session in this channel.")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    async def session_config_slash(self, interaction: discord.Interaction):
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

        await interaction.response.defer(ephemeral=True)
        await self._open_session_config(interaction)

    def _ensure_session_shell(self, interaction: discord.Interaction) -> Dict[str, Any]:
        """The channel's live session, waking a suspended one from its blueprint.

        Extracted from `session_config_slash` so `/start` can open the same editor
        without a second copy of the wake logic. A blueprint restored by one of them
        and not the other is a session that loses its cast, its master prompt and its
        proactivity settings depending on which button was pressed.
        """
        ch_id = interaction.channel_id
        session = self.multi_profile_channels.get(ch_id)
        if session:
            return session

        server_index = self.server_manager._get_server_index(str(interaction.guild.id))
        session_config = server_index.get("active_sessions", {}).get("regular", {}).get(str(ch_id))

        DEFAULT_DIRECTOR_PROMPT = "You are an AI Director for a roleplay session. Introduce a sudden event, an environmental change, or a question to spark conversation among the cast. Keep it brief (1-2 sentences)."
        proactivity_defaults = {"enabled": False, "chance": 10, "cooldown": 300, "director_model": "off", "director_instructions": DEFAULT_DIRECTOR_PROMPT}

        if session_config:
            # Wake it up as a dehydrated shell
            session = {
                "type": "multi", "profiles": session_config.get("profiles", []),
                "unified_log": [], "is_hydrated": False,
                "owner_id": session_config.get("owner_id", interaction.user.id),
                "is_running": False, "task_queue": asyncio.Queue(), "worker_task": None,
                "session_prompt": session_config.get("session_prompt"),
                "session_mode": session_config.get("session_mode", "sequential"),
                "proactivity": session_config.get("proactivity", proactivity_defaults),
                "started": session_config.get("started", True),
            }
        else:
            # Blank session
            session = {
                "type": "multi", "profiles": [],
                "unified_log": [], "is_hydrated": False,
                "owner_id": interaction.user.id,
                "is_running": False, "task_queue": asyncio.Queue(), "worker_task": None,
                "session_prompt": None, "session_mode": "sequential",
                "proactivity": proactivity_defaults,
                # Opening the editor seats nobody and starts nothing.
                "started": False,
            }

        self.multi_profile_channels[ch_id] = session
        return session

    async def _open_session_config(self, interaction: discord.Interaction):
        """Renders the cast editor onto whatever message `interaction` has deferred.

        The caller owns the defer, which is what lets `/start` open this as a separate
        ephemeral message (defer with thinking=True) while the command itself keeps
        replacing its own (a plain defer on a component or command interaction).
        """
        session = self._ensure_session_shell(interaction)
        view = SessionConfigView(self, interaction, session)
        self.active_session_config_views[interaction.user.id] = view
        await view.update_display()

    @session_group.command(name="swap", description="Swaps, adds, or removes a profile from the current session.")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    @app_commands.guild_only()
    @is_admin_or_owner_check()
    @app_commands.autocomplete(profile_name=EventListeners.master_autocomplete)
    @app_commands.describe(
        profile_name="The profile to swap or add, or a seated one to remove. Blank + slot also removes.",
        use_child_bot="Whether to use the linked Child Bot (if available). Defaults to True.",
        slot="The participant number (1-200) to affect. See '/session swap' with no options for a list."
    )
    async def swap_session_slash(self, interaction: discord.Interaction, profile_name: Optional[str] = None, use_child_bot: Optional[bool] = None, slot: Optional[app_commands.Range[int, 1, 200]] = None):
        await interaction.response.defer(ephemeral=True)
        await self._swap_session_impl(interaction, profile_name, use_child_bot, slot)

    async def _remove_session_participant(self, interaction: discord.Interaction,
                                          session: Dict[str, Any], index: int) -> None:
        """Drops one seated participant and tells its child bot to stand down.

        Both removal routes land here -- `/session swap slot:<n>` with no profile, and
        `/session swap profile_name:<name>` naming someone already seated -- so the
        last-seat guard and the child-bot teardown cannot drift apart between them.
        The stop_typing that follows session_update_remove is not optional: a child bot
        dropped mid-round otherwise keeps its typing indicator alive in a session it is
        no longer part of.

        Removing the last participant is allowed. An empty cast is a valid session --
        the channel keeps its transcript, master prompt and proactivity settings, and
        nobody speaks until someone is seated again. Refusing it here only meant the
        one way out was `/suspend`, which throws all of that away.
        """
        removed_participant = session["profiles"].pop(index)

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

        self.session_manager._save_multi_profile_sessions()
        await interaction.followup.send(
            f"Removed `{removed_participant['profile_name']}` from the session.", ephemeral=True)

    async def _swap_session_impl(self, interaction: discord.Interaction, profile_name: Optional[str],
                                 use_child_bot: Optional[bool], slot: Optional[int]):
        """The body of /session swap, minus the opening defer.

        Split from the command so a "Did you mean?" correction can re-enter it: the
        component interaction that carries the correction has already been deferred, and
        deferring twice raises. Everything below responds through `followup`, which is
        valid for either interaction.
        """
        session = self.multi_profile_channels.get(interaction.channel_id)

        if not profile_name and not slot:
            server_id_str = str(interaction.guild_id) if interaction.guild_id else "dm"
            server_index = self.server_manager._get_server_index(server_id_str)
            channel_str = str(interaction.channel_id)
            
            active_sessions = server_index.get("active_sessions", {})
            session_data_idx = {}
            if isinstance(active_sessions, dict):
                session_data_idx = active_sessions.get("regular", {}).get(channel_str)

            if not session_data_idx or not session_data_idx.get("profiles"):
                await interaction.followup.send("There is no active session in this channel.", ephemeral=True)
                return
            
            view = SessionSwapListView(self, interaction, session_data_idx)
            await interaction.followup.send(embed=view.embed, view=view, ephemeral=True)
            return

        # If no session exists, we must have a profile name to start one
        if not session and not profile_name:
            await interaction.followup.send("There is no active session to modify. Provide a profile name to start one.", ephemeral=True)
            return

        # If session exists, check ownership
        if session and session.get("owner_id") != interaction.user.id:
            await interaction.followup.send("You are not the owner of this session and cannot modify its participants.", ephemeral=True)
            return

        # Logic for removing a participant by seat number (Session must exist here)
        if not profile_name and slot:
            if slot > len(session["profiles"]):
                await interaction.followup.send(f"Invalid slot. There are only {len(session['profiles'])} participants.", ephemeral=True)
                return
            await self._remove_session_participant(interaction, session, slot - 1)
            return

        if profile_name:
            index = self.profile_manager._get_user_index(interaction.user.id)
            is_personal = profile_name in index.get("personal", [])
            is_borrowed = profile_name in index.get("borrowed", [])
            
            owner_id = int(defaultConfig.DISCORD_OWNER_ID)
            # _is_system_name, not a bare System-index test: with the latter a user
            # who owned a profile sharing a System name got is_personal AND
            # is_system, and participant_owner below then joined the System profile
            # instead of theirs.
            is_system = self.profile_manager._is_system_name(interaction.user.id, profile_name)
            
            if not is_personal and not is_borrowed and not is_system:
                candidates = gather_owned_candidates(self, interaction.user.id)

                corrected = autocorrect_profile(profile_name, candidates)
                if corrected is not None:
                    await self._swap_session_impl(interaction, corrected, use_child_bot, slot)
                    return

                async def on_pick(pick_interaction: discord.Interaction, picked: str):
                    await self._swap_session_impl(pick_interaction, picked, use_child_bot, slot)

                await suggest_profile(self, interaction, profile_name, candidates, on_pick)
                return

            # Resolve Method (Child Bot vs Webhook)
            effective_owner_id, effective_profile_name = self.profile_manager._resolve_effective_profile(interaction.user.id, profile_name)

            linked_bot_id = self.child_bots_by_owner_profile.get((effective_owner_id, effective_profile_name))
            is_bot_in_guild = linked_bot_id and interaction.guild.get_member(int(linked_bot_id))

            method = "webhook"
            bot_id_to_use = None
            
            # Logic: Default to child bot if available, unless explicitly set to False
            should_use_bot = True if use_child_bot is None else use_child_bot
            
            if should_use_bot and is_bot_in_guild:
                method = "child_bot"
                bot_id_to_use = linked_bot_id
            
            participant_owner = owner_id if is_system else interaction.user.id

            # Create new participant object for comparison
            new_participant = {
                "owner_id": participant_owner, "profile_name": profile_name,
                "method": method, "ephemeral": False
            }
            if bot_id_to_use:
                new_participant["bot_id"] = bot_id_to_use

            # Naming someone already seated removes them. This used to be a flat
            # refusal, which left `slot:` as the only way out of a session -- so
            # dropping a profile meant first running the command bare to find out what
            # number it had been given. Both arguments now address the same participant.
            #
            # Deliberately guarded on the *bare* form. `slot:` and `use_child_bot:` are
            # statements about where or how someone sits, not about whether they should
            # be there at all, and deleting a participant in response to one would be
            # the opposite of what was asked. Each keeps an explicit answer instead,
            # which is also the only thing either could sensibly mean for a seat that
            # is already taken.
            existing_index = next(
                (n for n, p in enumerate(session["profiles"] if session else [])
                 if p["owner_id"] == new_participant["owner_id"]
                 and p["profile_name"] == new_participant["profile_name"]),
                None)

            if existing_index is not None:
                if slot is not None:
                    await interaction.followup.send(
                        f"`{profile_name}` is already in slot {existing_index + 1}. Run "
                        f"`/session swap profile_name:{profile_name}` on its own to remove it, "
                        f"or name a different profile to put in slot {slot}.", ephemeral=True)
                    return
                if use_child_bot is not None:
                    seated_as = ("its child bot"
                                 if session["profiles"][existing_index].get("method") == "child_bot"
                                 else "a webhook")
                    await interaction.followup.send(
                        f"`{profile_name}` is already a participant, speaking through {seated_as}. "
                        f"Remove it with `/session swap profile_name:{profile_name}` on its own, "
                        f"then add it back with the `use_child_bot` you want.", ephemeral=True)
                    return
                await self._remove_session_participant(interaction, session, existing_index)
                return

            # Create new session if none exists
            if not session:
                session = {
                    "type": "multi", "profiles": [new_participant],
                    "unified_log": [], "is_hydrated": False, "last_bot_message_id": None,
                    "owner_id": interaction.user.id, "is_running": False,
                    "task_queue": asyncio.Queue(),
                    "worker_task": None, "turns_since_last_ltm": 0, "session_prompt": None,
                    "session_mode": "sequential", "pending_image_gen_data": None, "pending_whispers": {},
                    # Seated, not started -- same rule the cast dropdown follows.
                    "started": False,
                }
                self.multi_profile_channels[interaction.channel_id] = session
                session["is_hydrated"] = True
                
                if method == "child_bot":
                    await self.manager_queue.put({
                        "action": "send_to_child", "bot_id": bot_id_to_use,
                        "payload": {"action": "session_update_add", "channel_id": interaction.channel_id}
                    })

                self.session_manager._save_multi_profile_sessions()
                await interaction.followup.send(
                    f"Seated `{profile_name}`. Start it with `/session config` -> **Start / Update Session**.",
                    ephemeral=True)
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

            self.session_manager._save_multi_profile_sessions()
            await interaction.followup.send(action_description, ephemeral=True)
            return

    @session_group.command(name="audit", description="Run a diagnostic and token telemetry audit on the active session.")
    @app_commands.checks.cooldown(2, 60.0, key=lambda i: i.user.id)
    @app_commands.guild_only()
    @is_admin_or_owner_check()
    async def session_audit_slash(self, interaction: discord.Interaction):
        if not self.has_lock: return
        
        session = self.multi_profile_channels.get(interaction.channel_id)
        if session and not session.get("is_hydrated"):
            session = await self.session_manager._ensure_session_hydrated(interaction.channel_id, session.get("type", "multi"))

        if not session:
            session = await self.session_manager._ensure_session_hydrated(interaction.channel_id, "multi")

        if not session:
            await interaction.response.send_message("There is no active session in this channel to audit.", ephemeral=True)
            return
            
        await interaction.response.defer(ephemeral=True)
        view = SessionAuditView(self, interaction, session, interaction.channel_id)
        await interaction.followup.send(embed=view._build_embed(), view=view, ephemeral=True)

    @session_group.command(name="view", description="View details of the current session and its participants.")
    @app_commands.checks.cooldown(5, 10.0, key=lambda i: i.user.id)
    async def session_view_slash(self, interaction: discord.Interaction):
        if not self.has_lock: return
        
        server_id_str = str(interaction.guild_id) if interaction.guild_id else "dm"
        server_index = self.server_manager._get_server_index(server_id_str)
        channel_str = str(interaction.channel_id)
        
        active_sessions = server_index.get("active_sessions", {})
        session_data_idx = None
        
        if isinstance(active_sessions, dict):
            if channel_str in active_sessions.get("regular", {}):
                session_data_idx = active_sessions["regular"][channel_str]

        if not session_data_idx:
            await interaction.response.send_message("No active session in this channel.", ephemeral=True)
            return

        type_display = "Chat Session"
        
        owner_id = session_data_idx.get("owner_id")
        owner = self.bot.get_user(owner_id)
        owner_name = owner.name if owner else f"ID: {owner_id}"
        
        profiles_for_display = session_data_idx.get("profiles", [])
        participant_count = len(profiles_for_display)
        
        started = session_data_idx.get("started", True)
        embed = discord.Embed(title=f"Session Info: #{interaction.channel.name}", color=discord.Color.gold())
        embed.add_field(name="Status", value="`Live`" if started else "`Draft`", inline=True)
        embed.add_field(name="Session Type", value=type_display, inline=True)
        embed.add_field(name="Session Admin", value=owner_name, inline=True)
        embed.add_field(name="Participants", value=str(participant_count), inline=True)
        
        if session_data_idx.get("session_prompt"):
            prompt_val = session_data_idx["session_prompt"]
            embed.add_field(name="Master Prompt", value=prompt_val[:200] + "..." if len(prompt_val) > 200 else prompt_val, inline=False)

        pro = session_data_idx.get("proactivity", {})
        if pro.get("enabled"):
            embed.add_field(name="Proactivity", value=f"**ON** | Chance: {pro.get('chance')}% | Cooldown: {pro.get('cooldown')}s", inline=False)

        session_view_data = {
            "type": "multi",
            "owner_id": owner_id,
            "profiles": profiles_for_display
        }

        if not started:
            embed.set_footer(text="Draft — start it from /session config.")

        view = SessionView(self, interaction, session_view_data)
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

    play_group = app_commands.Group(name="play", description="Table games with this channel's cast.", guild_only=True)

    @play_group.command(name="eights", description="Open a table of Mimic Eights for this channel's session and cast.")
    @app_commands.checks.cooldown(3, 60.0, key=lambda i: i.channel_id)
    @app_commands.describe(
        seats="How many seats at the table (2-6). Defaults to as many as fit.",
        join="Take a seat yourself, or open a table of profiles and watch. Defaults to sitting down.",
        stacking="Answer a Draw Two with another Draw Two instead of picking up. Default on.",
        stack_draw_four="Also allow Draw Fours to be stacked. Default off.",
        draw_to_match="Keep drawing until you get a playable card. Default off — draw one.",
        strict_draw_four="Only allow a Draw Four when you hold none of the active colour. Default off.",
        turn_seconds="Seconds a person gets before the table plays for them (15-120). Default 45.")
    async def play_eights_slash(self, interaction: discord.Interaction,
                             seats: Optional[app_commands.Range[int, 2, 6]] = None,
                             join: bool = True,
                             stacking: bool = True,
                             stack_draw_four: bool = False,
                             draw_to_match: bool = False,
                             strict_draw_four: bool = False,
                             turn_seconds: Optional[app_commands.Range[int, 15, 120]] = None):
        if not self.has_lock: return
        await interaction.response.defer(ephemeral=True)

        # Snapshotted into the game at deal time, so changing the rules mid-hand is
        # impossible by construction rather than by a check.
        rules = GameRuleSet(
            stack_draw_two=stacking,
            stack_draw_four=stack_draw_four,
            draw_to_match=draw_to_match,
            strict_draw_four=strict_draw_four,
            turn_seconds=turn_seconds if turn_seconds is not None else GameRuleSet().turn_seconds,
        )

        error = await self.game_service.open_lobby(interaction, seats_wanted=seats, join=join, rules=rules)
        if error:
            await interaction.followup.send(error, ephemeral=True)
            return

        await interaction.followup.send(
            "Table posted. It is **locked** — press 🔓 on it to let other people sit "
            "down, then **Start** when everyone is seated. Nothing is on a clock.",
            ephemeral=True)

    @play_group.command(name="stop", description="End the game running in this channel.")
    @app_commands.checks.cooldown(5, 60.0, key=lambda i: i.channel_id)
    async def play_stop_slash(self, interaction: discord.Interaction):
        if not self.has_lock: return

        game = self.active_games.get(interaction.channel_id)
        lobby = self.game_service.lobby_for(interaction.channel_id)
        if not game and not lobby:
            await interaction.response.send_message(
                "There is no game running in this channel.", ephemeral=True)
            return

        # The person who dealt it, or anyone who could have suspended the session. A
        # lobby answers to its host on the same terms -- it is the same claim on the
        # channel, just one that has not been dealt yet.
        owner_id = game.started_by if game else lobby.host_id
        is_admin = bool(getattr(interaction.user, "guild_permissions", None)
                        and interaction.user.guild_permissions.administrator)
        if interaction.user.id != owner_id and not is_admin and \
                interaction.user.id != int(defaultConfig.DISCORD_OWNER_ID):
            await interaction.response.send_message(
                "Only whoever dealt the game, or a server administrator, can end it.",
                ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)
        if lobby:
            self.game_service.cancel_lobby(interaction.channel_id)
            if lobby.message is not None:
                try:
                    await lobby.message.delete()
                except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                    pass
        if game:
            await self.game_service.stop_game(interaction.channel_id, reason="stopped early")
        await interaction.followup.send(
            "Game ended." if game else "Table cleared.", ephemeral=True)

    # Top level rather than under /session: forcing a round is something a user does
    # mid-conversation, alongside /refresh and /cancel, not something they go into a
    # configuration group to find.
    @app_commands.command(name="trigger", description="Manually triggers a new round of the session in this channel.")
    @app_commands.checks.cooldown(2, 10.0, key=lambda i: i.user.id)
    @app_commands.guild_only()
    async def trigger_session_slash(self, interaction: discord.Interaction):
        if not self.has_lock: return
        await interaction.response.defer(ephemeral=True)

        session = self.multi_profile_channels.get(interaction.channel_id)
        if not session:
            await interaction.followup.send("No active session found in this channel.", ephemeral=True)
            return

        if session.get("owner_id") != interaction.user.id and not interaction.user.guild_permissions.administrator:
            await interaction.followup.send("Only the session owner or a server administrator can trigger a round.", ephemeral=True)
            return

        if not self.session_manager.is_started(session):
            await interaction.followup.send(
                "This session has not been started. Run `/session config` -> **Start / Update Session**.",
                ephemeral=True)
            return

        if not session.get("profiles"):
            await interaction.followup.send("There is nobody in the cast to trigger.", ephemeral=True)
            return

        # Trigger logic: push a null trigger to simulate automated continuation
        await session['task_queue'].put(None)
        if not session.get('worker_task') or session['worker_task'].done():
            task = self.bot.loop.create_task(self.generation_service._multi_profile_worker(interaction.channel_id))
            session['worker_task'] = task
            self.background_tasks.add(task)

        await interaction.followup.send("Round triggered.", ephemeral=True)

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
        await self.session_manager._delete_session_from_disk(dummy_session_key, session_type)

        # [NEW] Reset counters for all participants
        for p in session.get('profiles', []):
            p['ltm_counter'] = 0
            # Also reset the global counter for this profile in this guild
            ltm_counter_key = (p['owner_id'], p['profile_name'], "guild")
            self.message_counters_for_ltm.pop(ltm_counter_key, None)
        
        # [NEW] Reset LTM recall history (penalty system) for this channel
        for p in session.get("profiles", []):
            full_session_key = (ch_id, p['owner_id'], p['profile_name'])
            self.ltm_recall_history.pop(full_session_key, None)

        session['is_hydrated'] = True
        session['unified_log'] = []

        if session.get('worker_task'):
            self.session_manager._safe_cancel_task(session['worker_task'])
            session['worker_task'] = None
            
        while not session['task_queue'].empty():
            try:
                session['task_queue'].get_nowait()
                session['task_queue'].task_done()
            except asyncio.QueueEmpty:
                break
        
        await interaction.followup.send("The session memory for this channel has been cleared. The conversation will start from scratch.", ephemeral=True)

    @app_commands.command(name="cancel", description="Stops the bot's current generation or typing in this channel.")
    @app_commands.checks.cooldown(2, 10.0, key=lambda i: i.user.id)
    @app_commands.guild_only()
    async def cancel_slash(self, interaction: discord.Interaction):
        """Stops what this channel is generating, if it is still safe to stop it.

        Two things are cancellable and they mean different things. A round in progress
        is *aborted*: nothing has been written, so the placeholder goes and no turn is
        recorded -- there is no earlier version of a fresh reply to go back to. A
        regeneration is *undone*: it overwrote a message that already existed, so
        cancelling puts the original text back.

        Both are refused once the operation reaches its sending phase -- the point the
        placeholder starts reading "Sending...". By then the model has returned, the
        turn is being assembled, speech may be part-way through synthesis, and a
        regeneration has already replaced the message it was going to restore. That is
        the one window where stopping leaves the channel in a worse state than letting
        it finish.

        Whisper and global chat are deliberately not covered: they are their own
        interactions and get their own controls rather than being reachable from a
        channel-wide command.
        """
        if not self.has_lock: return
        session = self.multi_profile_channels.get(interaction.channel_id)

        live_regens = {
            message_id: task
            for message_id, task in (session or {}).get('regen_tasks', {}).items()
            if task and not task.done()
        }
        busy = bool(session) and (session.get('is_running') or session.get('is_regenerating')
                                  or live_regens)

        if not busy:
            await interaction.response.send_message("Nothing is currently being generated in this channel.", ephemeral=True)
            return

        if session.get("owner_id") != interaction.user.id and not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message("Only the session owner or a server administrator can cancel generation.", ephemeral=True)
            return

        if self.session_manager.is_delivering(session):
            await interaction.response.send_message(
                "That response has already been generated and is being sent — cancelling now "
                "would leave it half-delivered. Wait for it to finish.", ephemeral=True)
            return

        cancelled = []

        if session.get('worker_task') and not session['worker_task'].done():
            self.session_manager._safe_cancel_task(session['worker_task'])
            session['worker_task'] = None
            cancelled.append("generation")
        session['is_running'] = False

        for task in live_regens.values():
            self.session_manager._safe_cancel_task(task)
        if live_regens:
            cancelled.append(f"{len(live_regens)} regeneration" + ("s" if len(live_regens) > 1 else ""))
        else:
            # Only safe to clear when nothing is actually running under it. Clearing it
            # unconditionally -- which this used to do -- released the gate every other
            # path waits on while a regeneration was still rewriting the log under it.
            session['is_regenerating'] = False

        for p in session.get('profiles', []):
            if p.get('method') == 'child_bot' and p.get('bot_id'):
                await self.manager_queue.put({
                    "action": "send_to_child", "bot_id": p['bot_id'],
                    "payload": {"action": "stop_typing", "channel_id": interaction.channel_id}
                })

        summary = " and ".join(cancelled) if cancelled else "typing"
        note = " The original messages have been restored." if live_regens else ""
        await interaction.response.send_message(
            f"Cancelled {summary} for this channel.{note}", ephemeral=True)

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

        if not await self.session_manager.suspend_channel_session(ch_id):
            await interaction.followup.send("There is no active session in this channel to suspend.", ephemeral=True)
            return

        self.session_manager._save_multi_profile_sessions()

        await interaction.followup.send(f"Session suspended for {interaction.channel.mention} and Freewill triggers disabled. The bot will be silent until mentioned or configured again.", ephemeral=True)

    @app_commands.command(name="purge", description="Purges messages and the associated session memory (Admin Only).")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    @app_commands.guild_only()
    @is_admin_or_owner_check()
    @app_commands.describe(amount="Messages to delete (1-100).")
    async def purge_slash(self, interaction: discord.Interaction, amount: app_commands.Range[int,1,100]):
        if not self.has_lock : return
        await interaction.response.defer(ephemeral=True)

        if not isinstance(interaction.channel, (discord.TextChannel, discord.Thread)):
            await interaction.followup.send("Purge command not supported in this channel type.", ephemeral=True)
            return

        app_perms = interaction.app_permissions
        if not app_perms or not app_perms.manage_messages:
            await interaction.followup.send("I lack 'Manage Messages' permission.", ephemeral=True); return

        def check_and_track(m):
            self.purged_message_ids[m.id] = True
            return True

        session_lock = self.multi_profile_channels.get(interaction.channel_id)

        # A purge deletes the table message out from under a running game, and the
        # game cache is independent of the session, so it has to be torn down here.
        self.game_service.teardown_channel(interaction.channel_id)

        try:
            if session_lock:
                # Re-hydrate immediately before executing the Discord API purge
                session_type = session_lock.get("type", "multi")
                if not session_lock.get("is_hydrated"):
                    session_lock = await self.session_manager._ensure_session_hydrated(interaction.channel_id, session_type)

                session_lock['is_purging'] = True
                # This wait is bounded, and it sits inside the try so the finally below
                # always clears is_purging. generation_service and both reaction
                # listeners spin on is_purging, so leaking it True wedges every future
                # turn in the channel for the life of the process -- and a worker that
                # died with is_running still set is exactly what produces that.
                wait_deadline = time.monotonic() + PURGE_BUSY_WAIT_TIMEOUT_SECONDS
                while (session_lock.get('is_running') or session_lock.get('is_regenerating')
                       or session_lock.get('is_whispering') or session_lock.get('is_memorising')):
                    if time.monotonic() > wait_deadline:
                        await interaction.followup.send(
                            f"The session is still generating after {int(PURGE_BUSY_WAIT_TIMEOUT_SECONDS)}s. "
                            "Nothing was deleted \u2014 try again in a moment.", ephemeral=True)
                        return
                    await asyncio.sleep(0.5)

            messages_to_delete = await interaction.channel.purge(limit=amount, check=check_and_track, before=interaction.created_at, reason=f"Purge by {interaction.user}")
            
            progress_message = await interaction.followup.send(f"Deleted {len(messages_to_delete)} message(s). Now cleaning them from my memory...", ephemeral=True)

            session = self.multi_profile_channels.get(interaction.channel_id)
            cleaned_turns_count = 0

            if session:
                session_type = session.get("type", "multi")
                
                deleted_msg_ids = {m.id for m in messages_to_delete}
                unified_log = session.get("unified_log", [])

                # One pass, keeping the turn objects rather than only their ids. The
                # counter decrement below used to re-scan the whole log once per deleted
                # turn to find each object again -- O(deleted x log) for something the
                # first pass already had in hand.
                turns_to_delete = [
                    turn for turn in unified_log
                    if any(mid in deleted_msg_ids for mid in turn.get("message_ids", []))
                ]

                if turns_to_delete:
                    cleaned_turns_count = len(turns_to_delete)

                    # speaker_pid -> participant, resolved once. This was previously a
                    # linear scan of the participant list, with a name->pid lookup per
                    # candidate, repeated for every deleted turn.
                    pid_to_profile = {}
                    for p in session.get('profiles', []):
                        pid = self.profile_manager._get_pid_from_name_any(p['owner_id'], p['profile_name'])
                        pid_to_profile.setdefault(pid, p)

                    for turn_obj in turns_to_delete:
                        if turn_obj.get("is_user") is False:
                            p = pid_to_profile.get(turn_obj.get("speaker_pid"))
                            if p:
                                p['ltm_counter'] = max(0, p.get('ltm_counter', 0) - 1)

                    # Filter by object identity, not by turn_id. A turn carrying no
                    # turn_id contributed None to the old id set, and every *other*
                    # turn without a turn_id then matched `not in` and was dropped with
                    # it -- silently deleting history the purge never touched.
                    doomed = {id(turn) for turn in turns_to_delete}
                    original_log_len = len(unified_log)
                    session["unified_log"] = [
                        turn for turn in unified_log if id(turn) not in doomed
                    ]

                    if len(session["unified_log"]) < original_log_len:
                        is_effectively_empty = not session.get("unified_log") or all(
                            turn.get("type") in ["whisper", "private_response"] for turn in session.get("unified_log", [])
                        )
                        
                        dummy_session_key = (interaction.channel_id, None, None)
                        if is_effectively_empty:
                            await self.session_manager._delete_session_from_disk(dummy_session_key, session_type)
                            for p in session.get("profiles", []):
                                full_session_key = (interaction.channel_id, p['owner_id'], p['profile_name'])
                                self.ltm_recall_history.pop(full_session_key, None)
                        else:
                            await self.session_manager._save_session_to_disk(dummy_session_key, session_type, session["unified_log"])

                        # Deleting turns can strand or reveal a pending whisper, which is
                        # the only state a rebuild derives -- so recompute that directly
                        # rather than re-reading the log we just wrote.
                        self.session_manager._recompute_pending_whispers(session)

                    self.session_last_accessed[interaction.channel_id] = time.time()

            await progress_message.edit(content=f"Deleted {len(messages_to_delete)} message(s) and cleaned {cleaned_turns_count} turn(s) from memory.")

        except Exception as e:
            await interaction.followup.send(f"An error occurred during purge: {e}", ephemeral=True)
            traceback.print_exc()
        finally:
            if session_lock:
                session_lock['is_purging'] = False

    @app_commands.command(name="memorise", description="Forces long-term memory summarisation for this session's cast, right now.")
    @app_commands.checks.cooldown(2, 60.0, key=lambda i: i.user.id)
    @app_commands.guild_only()
    @app_commands.describe(profile="Optional: summarise only this participant. Leave blank for the whole cast.")
    @app_commands.autocomplete(profile=EventListeners.master_autocomplete)
    async def memorise_slash(self, interaction: discord.Interaction, profile: Optional[str] = None):
        if not self.has_lock: return
        await interaction.response.defer(ephemeral=True)

        session = self.multi_profile_channels.get(interaction.channel_id)
        if not session:
            await interaction.followup.send("No active session found in this channel.", ephemeral=True)
            return

        target_participant = None
        if profile:
            try:
                p_owner_id_str, p_name = profile.split(":", 1)
                p_owner_id = int(p_owner_id_str)
            except ValueError:
                p_owner_id = None
                p_name = profile

            target_participant = next(
                (p for p in session.get("profiles", [])
                 if p.get("profile_name") == p_name and (p_owner_id is None or p.get("owner_id") == p_owner_id)),
                None,
            )
            if not target_participant:
                await interaction.followup.send(f"Could not find participant '{p_name}' in this session.", ephemeral=True)
                return

        is_admin = interaction.user.guild_permissions.administrator
        if target_participant:
            if not (is_admin or interaction.user.id == target_participant['owner_id']):
                await interaction.followup.send(
                    "You can only force memory summarisation for a profile you own (or be an administrator).",
                    ephemeral=True)
                return
        elif not is_admin:
            await interaction.followup.send(
                "Summarising the whole cast requires administrator permission. Name a single profile you own instead.",
                ephemeral=True)
            return

        session_type = session.get("type", "multi")
        if not session.get("is_hydrated"):
            session = await self.session_manager._ensure_session_hydrated(interaction.channel_id, session_type)
            if not session:
                await interaction.followup.send("Could not hydrate the session.", ephemeral=True)
                return

        self.session_last_accessed[interaction.channel_id] = time.time()

        if not await self.session_manager._wait_for_session_flags(session, SESSION_BUSY_FLAGS, PURGE_BUSY_WAIT_TIMEOUT_SECONDS):
            await interaction.followup.send(
                f"The session is still busy after {int(PURGE_BUSY_WAIT_TIMEOUT_SECONDS)}s. Try again in a moment.",
                ephemeral=True)
            return
        session['is_memorising'] = True

        try:
            guild_id = interaction.guild.id
            targets = [target_participant] if target_participant else list(session.get("profiles", []))

            summarised, skipped, failed = [], [], []
            for p in targets:
                owner_id = p['owner_id']
                p_name = p['profile_name']
                p_index = self.profile_manager._get_user_index(owner_id)
                p_is_borrowed = p_name in p_index.get("borrowed", [])
                p_settings = self.profile_manager._get_profile_config(owner_id, p_name, p_is_borrowed) or {}

                if not p_settings.get("ltm_creation_enabled", False):
                    skipped.append(f"{p_name} (LTM disabled)")
                    continue

                try:
                    created, detail = await self.generation_service._summarize_and_store_ltm(
                        interaction.channel_id, session, owner_id, p_name, p_settings,
                        guild_id, interaction.user.display_name, interaction.user.id,
                        warning_channel=interaction.channel,
                    )
                except Exception as e:
                    created, detail = False, str(e)

                p['ltm_counter'] = 0
                (summarised if created else failed).append(p_name if created else f"{p_name} ({detail})")

            lines = []
            if summarised:
                lines.append(f"**Summarised:** {', '.join(summarised)}")
            if skipped:
                lines.append(f"**Skipped:** {', '.join(skipped)}")
            if failed:
                lines.append(f"**No new memory:** {', '.join(failed)}")
            if not lines:
                lines.append("Nothing to summarise.")

            await interaction.followup.send("\n".join(lines), ephemeral=True)
        finally:
            if session:
                session['is_memorising'] = False

    @app_commands.command(name="train", description="Arms this channel to capture a training example from reactions (1️⃣ input, 2️⃣ output).")
    @app_commands.checks.cooldown(5, 60.0, key=lambda i: i.user.id)
    @app_commands.guild_only()
    @app_commands.describe(profile="Your own profile to train (borrowed and system profiles are not eligible).")
    @app_commands.autocomplete(profile=EventListeners.master_autocomplete)
    async def train_slash(self, interaction: discord.Interaction, profile: str):
        if not self.has_lock: return
        await interaction.response.defer(ephemeral=True)

        candidates = gather_owned_candidates(self, interaction.user.id, only=lambda name, kind: kind == "personal")
        candidate_names = {c.value for c in candidates}

        if profile not in candidate_names:
            corrected = autocorrect_profile(profile, candidates)
            if corrected is not None:
                profile = corrected
            else:
                async def on_pick(pick_interaction: discord.Interaction, picked: str):
                    await self.train_slash.callback(self, pick_interaction, profile=picked)

                await suggest_profile(self, interaction, profile, candidates, on_pick)
                return

        self.armed_training_channels[interaction.channel_id] = {
            "owner_id": interaction.user.id,
            "profile_name": profile,
            "armed_by": interaction.user.id,
            "guild_id": interaction.guild.id,
            "last_activity": time.time(),
            "slot1": None,
            "slot2": None,
            "interaction": interaction,
        }
        await interaction.followup.send(
            f"Armed to train **{profile}**. React {TRAIN_INPUT_EMOJI} on the input message and {TRAIN_OUTPUT_EMOJI} on "
            "the output message (any two messages, from anyone, in any order). Stays armed for more pairs -- only your "
            "reactions count, and arming expires after 15 minutes of inactivity.",
            ephemeral=True)

    @app_commands.checks.cooldown(2, 10.0, key=lambda i: i.user.id)
    async def view_generation_trace(self, interaction: discord.Interaction, message: discord.Message):
        if not self.has_lock: return
        await interaction.response.defer(ephemeral=True)
            
        channel_id = message.channel.id
        session = self.multi_profile_channels.get(channel_id)
        unified_log = None
        
        if session and session.get("is_hydrated"):
            unified_log = session.get("unified_log")
        else:
            # Check disk for dehydrated sessions
            for s_type in ["multi", "freewill"]:
                dummy_key = (channel_id, None, None)
                disk_log = await self.session_manager._load_session_from_disk(dummy_key, s_type)
                if disk_log:
                    unified_log = disk_log
                    break
                    
        if not unified_log:
            await interaction.followup.send("Could not find session data for this channel.", ephemeral=True)
            return
            
        target_turn = next((t for t in unified_log if message.id in t.get("message_ids", [])), None)
        
        if not target_turn or target_turn.get("is_user") is not False:
            await interaction.followup.send("This command can only be used on generated bot messages.", ephemeral=True)
            return
            
        meta = target_turn.get("meta")
        if not meta:
            await interaction.followup.send("No generation trace data is available for this older message.", ephemeral=True)
            return
            
        embed = discord.Embed(title="Generation Trace", color=discord.Color.blurple())
        
        model_str = f"`{meta.get('model', 'Unknown')}`"
        if meta.get("fallback"):
            model_str += " *(Fallback)*"
        embed.add_field(name="Model", value=model_str, inline=True)
        
        embed.add_field(name="Duration", value=f"`{meta.get('duration', 0.0)}s`", inline=True)
        
        if meta.get("ltm_created"):
            embed.add_field(name="Memory Event", value="`Memory Created`", inline=True)
            
        ltm_count = len(meta.get("ltms_recalled", []))
        embed.add_field(name="Memories Recalled", value=f"`{ltm_count}`", inline=True)
            
        train_count = meta.get("training_recalled", 0)
        embed.add_field(name="Training Examples Recalled", value=f"`{train_count}`", inline=True)
            
        urls = meta.get("grounding_sources", [])
        if urls:
            unique_urls = set([u for u in urls if u])
            embed.add_field(name="Web Grounding", value=f"`Yes ({len(unique_urls)} Sources)`", inline=True)
        else:
            embed.add_field(name="Web Grounding", value="`No`", inline=True)
            
        neuro_state = meta.get("neuro_state")
        if neuro_state:
            neuro_str = (
                f"Dopamine (D): `{neuro_state.get('dopamine', 0)}`\n"
                f"Cortisol (C): `{neuro_state.get('cortisol', 0)}`\n"
                f"Oxytocin (O): `{neuro_state.get('oxytocin', 0)}`\n"
                f"Adrenaline (A): `{neuro_state.get('adrenaline', 0)}`"
            )
            embed.add_field(name="Neuro Engine", value=neuro_str, inline=False)
            
        await interaction.followup.send(embed=embed, ephemeral=True)

    @profile_group.command(name="global_chat", description="Have a persistent, private conversation with a profile.")
    @app_commands.checks.cooldown(5, 60.0, key=lambda i: i.user.id)
    @app_commands.autocomplete(profile_name=EventListeners.master_autocomplete)
    @app_commands.describe(
        profile_name="The profile to chat with. Leave blank to view private history.",
        refresh="Set to True to clear your conversation history with this profile.",
        suspend="Set to True to permanently delete ALL global chat histories for every profile."
    )
    async def global_chat_slash(self, interaction: discord.Interaction, profile_name: Optional[str] = None, refresh: Optional[bool] = False, suspend: Optional[bool] = False):
        user_id = interaction.user.id
        
        if suspend:
            await interaction.response.defer(ephemeral=True)

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

            await self.session_manager._delete_session_from_disk(session_key, 'global_chat')
            
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

        index = self.profile_manager._get_user_index(user_id)
        is_personal = profile_name_lower in index.get("personal", [])
        is_borrowed = profile_name_lower in index.get("borrowed", [])
        if not is_personal and not is_borrowed:
            # Only profiles that may actually be used here are suggested -- proposing
            # one that would just hit the next rejection wastes the recovery. This is a
            # typo-recovery path, not a per-message one, so the per-candidate rating
            # lookup is affordable.
            candidates = gather_owned_candidates(
                self, user_id, include_system=False,
                only=lambda name, kind: self.profile_manager.content_capability(
                    user_id, name, "global_chat")[0],
            )

            async def on_pick(pick_interaction: discord.Interaction, picked: str):
                await self.global_chat_slash.callback(
                    self, pick_interaction, profile_name=picked,
                    refresh=refresh, suspend=suspend)

            await suggest_profile(
                self, interaction, profile_name_lower, candidates, on_pick,
                # /global_chat answers publicly, so its continuation has to own the
                # first response and choose ephemeral=False for itself.
                defer_on_pick=False,
            )
            return

        # Rating, not publication. Requiring a profile to be listed in the Public
        # Library to talk to it privately was a proxy for "somebody has vetted this",
        # and a poor one -- it forced users to publish a profile they only wanted for
        # themselves. The rating is the vetting, so it is what gets checked.
        allowed, deny_reason = self.profile_manager.content_capability(
            user_id, profile_name_lower, "global_chat")
        if not allowed:
            await interaction.response.send_message(
                f"**'{profile_name_lower}' cannot be used in Global Chat.**\n{deny_reason}\n\n"
                f"Open `/profile manage profile_name:{profile_name_lower}` and choose "
                f"**Content Safety** to rate it.",
                ephemeral=True)
            return

        await interaction.response.defer(ephemeral=False)

        model_cache_key = ('global', user_id, profile_name_lower)
        session_data = self.global_chat_sessions.get(model_cache_key)
        if not session_data:
            session_data = await self.session_manager._load_session_from_disk(model_cache_key, 'global_chat')
            if not session_data:
                session_data = {'unified_log': []}
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

    @app_commands.command(name="start", description="Guided setup: keys, characters, sessions, and how to talk to them.")
    @app_commands.checks.cooldown(5, 60.0, key=lambda i: i.user.id)
    async def start_slash(self, interaction: discord.Interaction):
        # Deliberately not named /setup: a new user typing "/s" is already choosing
        # between settings, session, speak and suspend, and this is the one command
        # they have been told to run.
        from .gui.gui_start import StartWizardView, gather_state
        await interaction.response.defer(ephemeral=True)
        state = await gather_state(self, interaction)
        view = StartWizardView(self, interaction, state)
        await view.update_display()

    @app_commands.command(name="guide", description="Displays detailed documentation about the bot's features and commands.")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    async def guide_slash(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        from .utils.content import HELP_CATEGORIES
        view = DropdownContentView(HELP_CATEGORIES, "MimicAI Help & Documentation")
        await interaction.followup.send(embed=view.get_embed(), view=view, ephemeral=True)

    @app_commands.command(name="help", description="Toggle MimicGuide into this session, or instantly ask a technical question.")
    @app_commands.checks.cooldown(2, 60.0, key=lambda i: i.user.id)
    @app_commands.describe(ask="A technical question to ask the bot directly (Optional).")
    async def help_slash(self, interaction: discord.Interaction, ask: Optional[str] = None):
        await interaction.response.defer(ephemeral=True)
        
        owner_id = int(defaultConfig.DISCORD_OWNER_ID)
        self.profile_manager._get_or_create_system_profile("mimicguide")
        
        if ask:
            if not hasattr(self, 'doc_vectors') or not self.doc_vectors:
                await interaction.followup.send("Documentation vectors are not loaded or missing. Ensure the Bot Owner has configured an API key.", ephemeral=True)
                return
                
            guild_id = interaction.guild_id if interaction.guild else 0
            protocol_block = await self.help_service._get_relevant_help_context(ask, guild_id, force_always_respond=True)
            
            if not protocol_block:
                await interaction.followup.send("I couldn't find anything in the documentation related to your question.", ephemeral=True)
                return
            
            # --- Fetch mimicguide configuration ---
            p_config = self.profile_manager._get_profile_config(owner_id, "mimicguide", False) or {}
            prompts = self.profile_manager._get_profile_prompts(owner_id, "mimicguide") or {}
            
            # --- Construct Base System Instruction (Bypassing Fluff) ---
            persona_data = prompts.get("persona", {})
            ai_instr_str = prompts.get("ai_instructions", "")
            
            final_instr_parts = []
            
            if persona_data and any(persona_data.values()):
                persona_blocks = []
                for key in self.persona_modal_sections_order: 
                    if lines := persona_data.get(key,[]):
                        decrypted_lines = [self.storage_manager._decrypt_data(line).strip() for line in lines if line.strip()]
                        if any(l.strip() for l in decrypted_lines):
                            block_content = "\n".join(decrypted_lines)
                            persona_blocks.append(f"<{key}>\n{block_content}\n</{key}>")
                if persona_blocks:
                    persona_str = "<persona_profile>\n" + "\n\n".join(persona_blocks) + "\n</persona_profile>"
                    final_instr_parts.append(persona_str)
                    
            decrypted_parts = []
            if isinstance(ai_instr_str, list):
                for part in ai_instr_str:
                    dec = self.storage_manager._decrypt_data(part)
                    if dec.strip():
                        cleaned_part = "\n".join([line.strip() for line in dec.split("\n")])
                        decrypted_parts.append(cleaned_part)
            elif isinstance(ai_instr_str, str):
                dec = self.storage_manager._decrypt_data(ai_instr_str)
                if dec.strip():
                    cleaned_part = "\n".join([line.strip() for line in dec.split("\n")])
                    decrypted_parts.append(cleaned_part)
            
            if decrypted_parts:
                final_instr_parts.append("<instructions>\n" + "\n\n".join(decrypted_parts).strip() + "\n</instructions>")
                
            final_instr_parts.append(protocol_block)
            
            sys_prompt = "\n\n".join(final_instr_parts).strip() if final_instr_parts else DEFAULT_SYSTEM_INSTRUCTION
            
            # --- Assemble Payload & Parameters ---
            user_prompt = f"<user_query>\n{ask}\n</user_query>"
            contents_for_api_call = [{'role': 'user', 'parts': [user_prompt]}]
            
            temp = float(p_config.get("temperature", 0.2))
            top_p = float(p_config.get("top_p", 0.9))
            top_k = int(p_config.get("top_k", 40))
            
            adv_params = {
                "frequency_penalty": p_config.get("frequency_penalty"),
                "presence_penalty": p_config.get("presence_penalty"),
                "repetition_penalty": p_config.get("repetition_penalty"),
                "min_p": p_config.get("min_p"),
                "top_a": p_config.get("top_a")
            }
            adv_params = {k: v for k, v in adv_params.items() if v is not None}
            gen_config = {"temperature": temp, "top_p": top_p, "top_k": top_k, "_advanced_params": adv_params}
            
            t_params_worker = {
                "thinking_summary_visible": p_config.get("thinking_summary_visible", "off"),
                "thinking_level": p_config.get("thinking_level", "none"),
                "thinking_budget": p_config.get("thinking_budget", -1)
            }
            
            d_safe = _resolve_safety_settings(interaction.channel, p_config)
            
            primary_model = p_config.get("primary_model", "GOOGLE/gemini-2.5-flash-lite")
            fallback_model_name = p_config.get("fallback_model", "GOOGLE/gemini-2.5-flash-lite")
            custom_error = p_config.get("error_response", "An error has occurred.")
            
            model_to_use = primary_model
            model_instance = None
            response_text = ""
            
            # --- Execute Pipeline ---
            try:
                # Intentionally passing None for tools to strictly disable grounding/fetching overrides
                model_instance = self.api_service._instantiate_model(model_to_use, guild_id, owner_id, sys_prompt, d_safe, t_params_worker, None, p_config)
                resp = await model_instance.generate_content_async(contents_for_api_call, generation_config=gen_config)
                
                if not resp or not resp.candidates:
                    raise ValueError("Empty or blocked response")
                    
                response_text = getattr(resp, 'text', "")
            except Exception as e:
                try:
                    model_to_use = fallback_model_name
                    model_instance = self.api_service._instantiate_model(model_to_use, guild_id, owner_id, sys_prompt, d_safe, t_params_worker, None, p_config)
                    resp = await model_instance.generate_content_async(contents_for_api_call, generation_config=gen_config)
                    
                    if not resp or not resp.candidates:
                        raise ValueError("Empty or blocked response")
                        
                    response_text = getattr(resp, 'text', "")
                except Exception as fb_e:
                    print(f"/help ask Fallback error: {fb_e}")
                    response_text = f"{custom_error}\n\n-# (API Failure)"
                    
            response_text = _scrub_response_text(response_text).strip()
            if not response_text:
                response_text = custom_error
                
            await interaction.followup.send(response_text, ephemeral=True)
            
        else:
            owner_id = int(defaultConfig.DISCORD_OWNER_ID)
            self.profile_manager._get_or_create_system_profile("mimicguide")
            
            session = self.multi_profile_channels.get(interaction.channel_id)
            action_taken = None
            result_msg = ""
            
            participant = {
                "owner_id": owner_id,
                "profile_name": "mimicguide",
                "method": "webhook",
                "ephemeral": False
            }
            
            if not session:
                session = {
                    "type": "multi", "profiles": [participant],
                    "unified_log": [], "is_hydrated": False, "owner_id": interaction.user.id, "is_running": False,
                    "task_queue": asyncio.Queue(), "worker_task": None, "session_mode": "sequential"
                }
                self.multi_profile_channels[interaction.channel_id] = session
                result_msg = "MimicGuide joined and created a new Chat Session."
            else:
                participant_index = -1
                for i, p in enumerate(session['profiles']):
                    if p.get('owner_id') == owner_id and p.get('profile_name') == "mimicguide":
                        participant_index = i
                        break
                
                if participant_index != -1:
                    session['profiles'].pop(participant_index)
                    result_msg = "MimicGuide departed the Chat Session."
                    if not session['profiles']:
                        self.multi_profile_channels.pop(interaction.channel_id, None)
                else:
                    if len(session['profiles']) >= 200:
                        await interaction.followup.send("The session is full (200 max).", ephemeral=True)
                        return
                    session['profiles'].append(participant)
                    result_msg = "MimicGuide joined the Chat Session."
            
            self.session_manager._save_multi_profile_sessions()
            await interaction.followup.send(result_msg, ephemeral=True)

    @app_commands.command(name="terms", description="View the MimicAI Terms of Service and Privacy Policy.")
    @app_commands.checks.cooldown(10, 60.0, key=lambda i: i.user.id)
    async def terms_slash(self, interaction: discord.Interaction):
        await interaction.response.send_message("View the Terms of Service and Privacy Policy here: https://mimic-ai.org/", ephemeral=True)

    @app_commands.command(name="invite", description="Get the invite link to add MimicAI to your server.")
    @app_commands.checks.cooldown(1, 10.0, key=lambda i: i.user.id)
    async def invite_slash(self, interaction: discord.Interaction):
        client_id = self.bot.user.id if self.bot.user else 1376696185947164854
        invite_url = f"https://discord.com/api/oauth2/authorize?client_id={client_id}&permissions=8&scope=bot%20applications.commands"
        
        embed = discord.Embed(
            title="Bring Your Personas to Life with MimicAI",
            description="Experience a revolutionary, unified orchestration engine that gives you complete control over your chatbot. MimicAI brings unparalleled customisation and model diversity directly into your Discord servers.",
            color=discord.Color.blue()
        )
        
        embed.add_field(
            name="Bring Your Own API (BYO API)",
            value="Connect your own Google Gemini or OpenRouter API keys to unlock virtually any language model on the market, from lightweight performance engines to frontier reasoning models. Pay only for what you generate directly with your provider.",
            inline=False
        )
        
        embed.add_field(
            name="Advanced Customisation",
            value="Configure Short-Term Memory limits, fine-tune sampling parameters (Temperature, Top P, Top K), or manipulate advanced OpenRouter heuristics. Integrate external tools like real-time Web Grounding (RAG), URL context fetching, custom placeholder emojis, and simulated neuro-endocrine emotional engines.",
            inline=False
        )
        
        embed.add_field(
            name="Multi-Profile Chat Sessions",
            value="Organise several unique characters into a single, unified Chat Session. Watch them interact with each other and server members with chronological awareness and deep memory coherence.",
            inline=False
        )
        
        embed.add_field(
            name="Disclaimer",
            value="-# All software features and customisation options within MimicAI are completely free to use. However, connecting to background AI models for text, speech, or image generation requires an active billing account with your selected API provider (Google or OpenRouter). Token overhead usage applies.",
            inline=False
        )
        
        view = InviteView(invite_url)
        await interaction.response.send_message(embed=embed, view=view, ephemeral=False)

    @app_commands.command(name="settings", description="Manage API keys and Child Bots (DM-Only).")
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
        if "OLLAMA/" in model_name.upper() or provider == "ollama": return
        
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

    def _log_api_call(self, user_id: int, guild_id: Optional[int], context: str, model_used: Any, status: str):
        if status == "success":
            allowed_contexts = [
                'multi_profile', 'global_chat', 'freewill',
                'multi_profile_fallback', 'global_chat_fallback', 'freewill_fallback'
            ]
            if context not in allowed_contexts:
                return

            model_name_str = "unknown"
            is_ollama = False
            is_google = False

            if isinstance(model_used, str):
                model_name_str = model_used
                if "OLLAMA/" in model_name_str.upper():
                    is_ollama = True
            elif model_used is not None:
                class_name = model_used.__class__.__name__
                if class_name == "OllamaModel":
                    is_ollama = True
                    model_name_str = f"OLLAMA/{model_used.model_name}"
                elif class_name == "OpenRouterModel":
                    model_name_str = f"OPENROUTER/{model_used.model_name}"
                elif class_name == "GoogleRESTModel":
                    model_name_str = f"GOOGLE/{model_used.model_name}"
                    is_google = True
                elif hasattr(model_used, "model_name"):
                    model_name_str = model_used.model_name

            if not is_ollama and not is_google:
                # Robust provider detection for recording popularity
                if model_name_str in get_args(ALLOWED_MODELS):
                    is_google = True
                elif model_name_str.startswith("models/"):
                    is_google = True
                elif "gemini" in model_name_str.lower():
                    if "/" in model_name_str and not model_name_str.startswith("models/"):
                        is_google = False
                    else:
                        is_google = True
            
            if not is_google and not is_ollama:
                self._record_model_usage(model_name_str, "openrouter")

    @app_commands.command(name="speak", description="Anonymously speak as one of your profiles (Admin Only).")
    @app_commands.checks.cooldown(2, 10.0, key=lambda i: i.user.id)
    @app_commands.guild_only()
    @is_admin_or_owner_check()
    @app_commands.autocomplete(profile_name=EventListeners.master_autocomplete, method=EventListeners.master_autocomplete)
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
            # Mirrors the permission filter master_autocomplete applies, so the prompt
            # never offers a participant the invoker would then be refused.
            bot_owner_id = int(defaultConfig.DISCORD_OWNER_ID)

            def speakable(p: dict) -> bool:
                owner = p.get("owner_id")
                return (owner == interaction.user.id
                        or interaction.user.id == bot_owner_id
                        or owner == bot_owner_id)

            async def on_pick(pick_interaction: discord.Interaction, picked: str):
                await self.speak_slash.callback(
                    self, pick_interaction, profile_name=picked,
                    method=method, message=message)

            await suggest_profile(
                self, interaction, p_name,
                gather_participant_candidates(session, only=speakable),
                on_pick,
                noun="participant",
                # Without a message, the continuation opens a modal, and a modal can
                # only answer an interaction that has not been responded to.
                defer_on_pick=False,
            )
            return

        if p_owner_id != interaction.user.id and interaction.user.id != int(defaultConfig.DISCORD_OWNER_ID):
            await interaction.response.send_message("You can only speak as your own personal or borrowed profiles.", ephemeral=True)
            return

        if message:
            await interaction.response.defer(ephemeral=True)
            await self.generation_service._execute_speak_as(
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
                await self.generation_service._execute_speak_as(
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
    @app_commands.autocomplete(profile=EventListeners.master_autocomplete)
    @app_commands.describe(
        profile="The participant to whisper to. Leave blank to view history.",
        message="The private message to send. Leave blank to view history."
    )
    async def whisper_slash(self, interaction: discord.Interaction, profile: Optional[str] = None, message: Optional[str] = None):
        session = self.multi_profile_channels.get(interaction.channel_id)
        if not session or session.get("type") != "multi":
            await interaction.response.send_message("This command can only be used in an active multi-profile session.", ephemeral=True)
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
                async def on_pick(pick_interaction: discord.Interaction, picked: str):
                    await self.whisper_slash.callback(
                        self, pick_interaction, profile=picked, message=message)

                await suggest_profile(
                    self, interaction, p_name,
                    gather_participant_candidates(session),
                    on_pick,
                    noun="participant",
                    # Without a message, the continuation opens a modal.
                    defer_on_pick=False,
                )
                return

        if profile and message:
            await interaction.response.defer(ephemeral=True, thinking=True)
            await self.generation_service._execute_whisper(interaction, target_participant, message)
        elif profile and not message:
            async def modal_callback(modal_interaction: discord.Interaction, message_text: str):
                await modal_interaction.response.defer(ephemeral=True, thinking=True)
                await self.generation_service._execute_whisper(modal_interaction, target_participant, message_text)

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
            session = await self.session_manager._ensure_session_hydrated(interaction.channel_id, session.get("type", "multi"))

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

    def _write_lock_file(self):
        """Stamps the lock with our own PID alongside the heartbeat time."""
        with open(COG_LOCK_FILE_PATH, "w") as f:
            f.write(f"{time.time()} {os.getpid()}")

    @staticmethod
    def _lock_holder_is_alive(pid):
        """True if the PID recorded in the lock file is still a running process.

        A crash (SEGV, OOM kill, SIGKILL) leaves the lock file on disk with a
        heartbeat only seconds old, so the staleness window alone cannot tell a
        crashed holder from a healthy one. Asking the OS can.

        Unknown PID, or Windows where a signal-0 probe is not meaningful, falls
        back to True so the caller uses the timestamp heuristic instead — that is
        the old behaviour, never worse than it.
        """
        if pid is None or pid <= 0:
            return True
        if platform.system() == "Windows":
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return True
        return True

    def _try_acquire_lock(self):
        try:
            if not os.path.exists(COG_LOCK_FILE_PATH):
                self._write_lock_file()
                self.has_lock = True
                return

            with open(COG_LOCK_FILE_PATH, "r") as f:
                lock_contents = f.read().strip()
            if not lock_contents:
                self._write_lock_file()
                self.has_lock = True
                return

            # Legacy lock files hold a bare timestamp; current ones append the PID.
            fields = lock_contents.split()
            lock_time = float(fields[0])
            lock_pid = int(fields[1]) if len(fields) > 1 else None

            is_own_pid = lock_pid is not None and lock_pid == os.getpid()
            holder_died = lock_pid is not None and not self._lock_holder_is_alive(lock_pid)
            is_stale = (time.time() - lock_time) > LOCK_STALE_THRESHOLD_SECONDS

            if is_own_pid or holder_died or is_stale:
                if holder_died and not is_stale:
                    # The common case after a crash: systemd restarts in 10 s but the
                    # heartbeat is under the 60 s threshold, so without this the new
                    # process would boot INACTIVE and stay that way.
                    print(f"Lock held by dead PID {lock_pid}. Reclaiming.")
                self._write_lock_file()
                self.has_lock = True
            else:
                self.has_lock = False
        except (IOError, ValueError, IndexError) as e:
            print(f"Error during lock acquisition: {e}. Assuming no lock.")
            self.has_lock = False
        except Exception as e:
            print(f"Unexpected error during lock acquisition: {e}. Assuming no lock.")
            self.has_lock = False

    @tasks.loop(seconds=LOCK_REFRESH_INTERVAL_SECONDS)
    async def refresh_lock_task(self):
        if self.has_lock:
            try:
                await asyncio.to_thread(self._write_lock_file)
            except IOError as e:
                print(f"IOError refreshing lock file: {e}. Potential lock loss.")
                self.has_lock = False
            except Exception as e:
                print(f"Unexpected error refreshing lock file: {e}. Potential lock loss.")
                self.has_lock = False

    @tasks.loop(seconds=LOCK_REFRESH_INTERVAL_SECONDS)
    async def reacquire_lock_task(self):
        """Self-heals an INACTIVE cog.

        Nothing used to retry after the single attempt in __init__, so a cog that
        lost the race once stayed half-dead until someone restarted it by hand —
        listeners and lock-guarded commands silently returning early while the
        unguarded ones kept working. Retrying costs one stat() every 30 s.
        """
        if self.has_lock:
            return
        await asyncio.to_thread(self._try_acquire_lock)
        if self.has_lock:
            print(f"MimicCog {self.cog_id} acquired lock on retry and is now ACTIVE.")
            if not self.refresh_lock_task.is_running():
                self.refresh_lock_task.start()
            self.reacquire_lock_task.cancel()

async def setup(bot: commands.Bot):
    await bot.add_cog(MimicCog(bot))
# --- End of MimicCog.py ---
