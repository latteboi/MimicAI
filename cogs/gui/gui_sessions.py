from ..utils.constants import *

import discord
from discord import ui
import datetime
import pathlib
import time
import asyncio
from typing import TYPE_CHECKING, List, Dict, Any, Optional
from ..utils.helpers import _estimate_text_tokens
from .base_components import build_pagination_controls

if TYPE_CHECKING:
    # This only runs during "hinting" and prevents the circular crash
    from ..MimicCog import MimicCog


class GlobalChatInputModal(ui.Modal, title="Draft Your Reply"):
    def __init__(self, cog, view, existing_text="", is_edit=False):
        super().__init__()
        self.cog = cog
        self.parent_view = view
        self.is_edit = is_edit
        self.input_field = ui.TextInput(
            label="Message Content",
            style=discord.TextStyle.paragraph,
            default=existing_text,
            max_length=2000,
            required=True
        )
        self.add_item(self.input_field)

    async def on_submit(self, interaction: discord.Interaction):
        session_data = self.cog.global_chat_sessions.get(self.parent_view.session_key)
        if not session_data:
            await interaction.response.send_message("Session expired.", ephemeral=True)
            return

        queue = session_data.setdefault("pending_queue", {})
        queue[interaction.user.id] = {
            "user_id": interaction.user.id,
            "display_name": interaction.user.display_name,
            "content": self.input_field.value.strip(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        
        active_typers = session_data.setdefault("active_typers", set())
        active_typers.discard(interaction.user.id)
        
        # If everyone finished typing, remove the lock immediately
        if not active_typers:
            session_data["lock_deadline"] = 0
        
        self.parent_view._build_view()
        try:
            await self.parent_view.original_interaction.edit_original_response(view=self.parent_view)
        except Exception:
            pass

        queue_view = GlobalChatQueueView(self.cog, self.parent_view, interaction.user.id)
        embed = queue_view.get_embed()
        
        if self.is_edit:
            await interaction.response.edit_message(embed=embed, view=queue_view)
        else:
            await interaction.response.send_message(embed=embed, view=queue_view, ephemeral=True)

class GlobalChatQueueView(ui.View):
    def __init__(self, cog, parent_view, user_id):
        super().__init__(timeout=None)
        self.cog = cog
        self.parent_view = parent_view
        self.user_id = user_id

    def get_embed(self):
        session_data = self.cog.global_chat_sessions.get(self.parent_view.session_key, {})
        queue = session_data.get("pending_queue", {})
        
        embed = discord.Embed(title="Queued Messages", color=discord.Color.gold())
        
        script_text = ""
        queued_turns = sorted(list(queue.values()), key=lambda x: x["timestamp"])
        for turn in queued_turns:
            preview = turn['content']
            if len(preview) > 800: preview = preview[:797] + "..."
            marker = "[You] " if turn['user_id'] == self.user_id else ""
            script_text += f"**{marker}{turn['display_name']}**: {preview}\n\n"
        
        if not script_text:
            script_text = "No messages currently in the queue."
            
        embed.description = script_text
        return embed

    @ui.button(label="Edit Reply", style=discord.ButtonStyle.primary)
    async def edit_btn(self, interaction: discord.Interaction, button: ui.Button):
        session_data = self.cog.global_chat_sessions.get(self.parent_view.session_key)
        if not session_data or self.user_id not in session_data.get("pending_queue", {}):
            await interaction.response.send_message("This turn has already been played or removed.", ephemeral=True)
            return
        
        # Re-add to active typers if they edit, but do NOT extend the absolute 30s deadline
        now = time.time()
        deadline = session_data.get("lock_deadline", 0)
        if deadline != 0 and now < deadline:
            session_data.setdefault("active_typers", set()).add(self.user_id)
            self.parent_view._build_view()
            try:
                await self.parent_view.original_interaction.edit_original_response(view=self.parent_view)
            except Exception: pass
        
        existing_text = session_data["pending_queue"][self.user_id]["content"]
        modal = GlobalChatInputModal(self.cog, self.parent_view, existing_text, is_edit=True)
        await interaction.response.send_modal(modal)

    @ui.button(label="Delete Reply", style=discord.ButtonStyle.danger)
    async def del_btn(self, interaction: discord.Interaction, button: ui.Button):
        session_data = self.cog.global_chat_sessions.get(self.parent_view.session_key)
        if session_data and self.user_id in session_data.get("pending_queue", {}):
            del session_data["pending_queue"][self.user_id]
            
            # Also ensure they are removed from typers if they somehow deleted while marked active
            session_data.setdefault("active_typers", set()).discard(self.user_id)
            if not session_data["active_typers"]:
                session_data["lock_deadline"] = 0
                
            self.parent_view._build_view()
            try:
                await self.parent_view.original_interaction.edit_original_response(view=self.parent_view)
            except Exception: pass
            
        await interaction.response.edit_message(embed=self.get_embed(), view=self)

class GlobalChatPlayView(ui.View):
    def __init__(self, cog, interaction, user_id, profile_name):
        super().__init__(timeout=None)
        self.cog = cog
        self.original_interaction = interaction
        self.user_id = user_id
        self.profile_name = profile_name
        self.session_key = ('global', user_id, profile_name)
        
    async def initialize(self):
        await self._load_current_session()
        self._build_view()
        
    async def _load_current_session(self):
        session_data = self.cog.global_chat_sessions.get(self.session_key)
        if not session_data:
            session_data = await self.cog.session_manager._load_session_from_disk(self.session_key, 'global_chat')
            if not session_data:
                session_data = {'unified_log': []}
            self.cog.global_chat_sessions[self.session_key] = session_data
            
    def _build_view(self):
        self.clear_items()
        session_data = self.cog.global_chat_sessions.get(self.session_key, {})
        queue = session_data.get("pending_queue", {})
        active_typers = session_data.get("active_typers", set())
        deadline = session_data.get("lock_deadline", 0)
        now = time.time()
        
        is_typing_locked = len(active_typers) > 0 and now < deadline
        is_session_locked = session_data.get("is_locked", True)
        
        reply_btn = ui.Button(label="Reply", style=discord.ButtonStyle.primary, row=0)
        reply_btn.callback = self.reply_callback
        self.add_item(reply_btn)
        
        if queue:
            if is_typing_locked:
                play_btn = ui.Button(label="Waiting for writers...", style=discord.ButtonStyle.secondary, disabled=True, row=0)
            else:
                play_btn = ui.Button(label=f"Play ({len(queue)})", style=discord.ButtonStyle.success, disabled=False, row=0)
        else:
            play_btn = ui.Button(label="Play", style=discord.ButtonStyle.secondary, disabled=True, row=0)
            
        play_btn.callback = self.play_callback
        self.add_item(play_btn)

        lock_style = discord.ButtonStyle.danger if is_session_locked else discord.ButtonStyle.success
        lock_emoji = "🔒" if is_session_locked else "🔓"
        lock_btn = ui.Button(style=lock_style, emoji=lock_emoji, row=0)
        lock_btn.callback = self.lock_callback
        self.add_item(lock_btn)
        
    def get_embed(self):
        session_data = self.cog.global_chat_sessions.get(self.session_key, {})
        log = session_data.get('unified_log', [])
        
        display_name = self.profile_name
        avatar_url = self.cog.bot.user.display_avatar.url
        
        index = self.cog.profile_manager._get_user_index(self.user_id)
        is_borrowed = self.profile_name in index.get("borrowed", [])
        
        eff_owner, eff_name = self.cog.profile_manager._resolve_effective_profile(self.user_id, self.profile_name)
            
        app = self.cog.profile_manager._get_user_appearance(eff_owner, eff_name)
        if app:
            display_name = app.get("custom_display_name") or display_name
            avatar_url = app.get("custom_avatar_url") or avatar_url
            
        embed = discord.Embed(color=discord.Color.dark_grey())
        embed.set_author(name=display_name, icon_url=avatar_url)
        
        last_user = None
        last_model = None
        
        for turn in reversed(log):
            if turn.get('role') == 'model' and not last_model:
                last_model = turn
            elif turn.get('role') == 'user' and not last_user:
                last_user = turn
                
            if last_user and last_model: break
        
        if not last_model:
            embed.description = "No conversation history found. Click 'Reply' to start."
        else:
            embed.description = last_model.get("content")
            if last_user:
                user_input = last_user.get("content", "")
                embed.set_footer(text=f"You: {user_input}", icon_url=self.original_interaction.user.display_avatar.url)
                
        return embed

    async def _wait_and_unlock(self, session_key, deadline, time_left):
        await asyncio.sleep(time_left + 0.5)
        session_data = self.cog.global_chat_sessions.get(session_key)
        if session_data and session_data.get("lock_deadline") == deadline:
            if time.time() >= deadline:
                session_data.setdefault("active_typers", set()).clear()
                session_data["lock_deadline"] = 0
                self._build_view()
                try:
                    await self.original_interaction.edit_original_response(view=self)
                except Exception:
                    pass

    async def lock_callback(self, interaction: discord.Interaction):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Only the session owner can lock or unlock this session.", ephemeral=True)
            return
        
        session_data = self.cog.global_chat_sessions.setdefault(self.session_key, {'unified_log': []})
        session_data["is_locked"] = not session_data.get("is_locked", True)
        self._build_view()
        await interaction.response.edit_message(view=self)

    async def reply_callback(self, interaction: discord.Interaction):
        session_data = self.cog.global_chat_sessions.setdefault(self.session_key, {'unified_log': []})
        
        if session_data.get("is_locked", True) and interaction.user.id != self.user_id:
            await interaction.response.send_message("This session is currently locked by the host.", ephemeral=True)
            return
            
        now = time.time()
        
        deadline = session_data.get("lock_deadline", 0)
        extensions = session_data.setdefault("timer_extensions", set())
        
        if deadline == 0 or now > deadline:
            # First interaction or timer died: Start fresh 10s timer and record this user
            session_data["lock_deadline"] = now + 10
            deadline = session_data["lock_deadline"]
            extensions.add(interaction.user.id)
        else:
            # Timer is currently active. Extend ONLY if this user hasn't extended yet this round
            if interaction.user.id not in extensions:
                session_data["lock_deadline"] = now + 10
                deadline = session_data["lock_deadline"]
                extensions.add(interaction.user.id)

        time_left = max(0, deadline - now)
        session_data.setdefault("active_typers", set()).add(interaction.user.id)
        
        queue = session_data.setdefault("pending_queue", {})
        existing_text = queue.get(interaction.user.id, {}).get("content", "")

        modal = GlobalChatInputModal(self.cog, self, existing_text)
        await interaction.response.send_modal(modal)

        self._build_view()
        try:
            await self.original_interaction.edit_original_response(view=self)
        except Exception:
            pass
            
        self.cog.bot.loop.create_task(self._wait_and_unlock(self.session_key, deadline, time_left))

    async def play_callback(self, interaction: discord.Interaction):
        session_data = self.cog.global_chat_sessions.get(self.session_key)
        if not session_data: return
        
        if session_data.get("is_locked", True) and interaction.user.id != self.user_id:
            await interaction.response.send_message("This session is currently locked by the host.", ephemeral=True)
            return
        
        queue = session_data.get("pending_queue", {})
        if not queue: return

        queued_turns = sorted(list(queue.values()), key=lambda x: x["timestamp"])
        session_data["pending_queue"] = {} 
        session_data["timer_extensions"] = set()
        session_data["active_typers"] = set()
        session_data["lock_deadline"] = 0
        
        self.clear_items()
        await interaction.response.edit_message(view=self)

        host_user_id = self.session_key[1]
        profile_name = self.session_key[2]
        await self.cog.generation_service._execute_global_chat(interaction, host_user_id, profile_name, queued_turns)
        
        await self._load_current_session()
        self._build_view()
        await interaction.edit_original_response(embed=self.get_embed(), view=self)

class CustomModelModal(ui.Modal, title="Enter Custom Model ID"):
    model_id_input = ui.TextInput(label="Model ID", placeholder="e.g. anthropic/claude-3 or google/gemini-2.5-flash", required=True)

    def __init__(self, view: Any, target_config_key: str):
        super().__init__()
        self.parent_view = view
        self.target_config_key = target_config_key

    async def on_submit(self, interaction: discord.Interaction):
        value = self.model_id_input.value.strip()
        
        # -----------------------------------------------------------------------------
        # DETAILED SYSTEM PREFIX RESOLUTION ARCHITECTURE:
        # 'GOOGLE/', 'OPENROUTER/', and 'OLLAMA/' are dedicated, case-sensitive system
        # routing prefixes used internally by MimicAI to route calls to the correct API adapter.
        #
        # Many providers (especially OpenRouter) host models under vendor namespaces with
        # lowercase names, such as 'google/gemini-2.5-flash' or 'meta-llama/llama-3.3-70b'.
        #
        # 1. If the user explicitly prefixes their input with one of the 3 system prefixes
        #    ('GOOGLE/', 'OPENROUTER/', 'OLLAMA/'), we honour that override directly.
        # 2. Otherwise, we automatically prepend the system prefix corresponding to the
        #    currently active view_mode (e.g. 'OPENROUTER/' if on the OpenRouter tab).
        # -----------------------------------------------------------------------------
        system_prefixes = ("GOOGLE/", "OPENROUTER/", "OLLAMA/")
        
        has_explicit_prefix = any(value.startswith(p) for p in system_prefixes)
        
        if not has_explicit_prefix:
            prefix = "GOOGLE/"
            if getattr(self.parent_view, 'view_mode', None) == "openrouter":
                prefix = "OPENROUTER/"
            elif getattr(self.parent_view, 'view_mode', None) == "ollama":
                prefix = "OLLAMA/"
            
            value = prefix + value

        self.parent_view._save_changes(self.target_config_key, value)
        self.parent_view._build_view()
        await interaction.response.edit_message(content=self.parent_view._get_selection_feedback_message(), view=self.parent_view)
        await interaction.response.edit_message(content=self.parent_view._get_selection_feedback_message(), view=self.parent_view)

class GlobalChatHistoryView(ui.View):
    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction, user_id: int, initial_profile: Optional[str] = None):
        super().__init__(timeout=600)
        self.cog = cog
        self.original_interaction = interaction
        self.user_id = user_id
        self.user_id_str = str(user_id)
        
        self.available_profiles = self._scan_profiles()
        self.selected_profile = initial_profile if initial_profile in self.available_profiles else (self.available_profiles[0] if self.available_profiles else None)
        
        self.rounds = [] 
        self.session_key = None
        self.current_page = 0
        
    async def initialize(self):
        if self.selected_profile:
            await self._load_current_session()
        self._build_view()

    def _scan_profiles(self) -> List[str]:
        profiles = set()
        
        for key in self.cog.global_chat_sessions.keys():
            if isinstance(key, tuple) and len(key) == 3 and key[0] == 'global' and key[1] == self.user_id:
                profiles.add(key[2])

        dir_path = pathlib.Path(self.cog.USERS_DIR) / self.user_id_str / "profiles"
        index = self.cog.profile_manager._get_user_index(self.user_id)
        
        all_profiles = list(index.get("personal", {})) + list(index.get("borrowed", {}))
        for p_name in all_profiles:
            pid = self.cog.profile_manager._get_pid_from_name_any(self.user_id, p_name)
            if (dir_path / pid / "global_chat.json.gz").exists():
                profiles.add(p_name)
        
        return sorted(list(profiles))

    async def _load_current_session(self):
        if not self.selected_profile: return
        self.session_key = ('global', self.user_id, self.selected_profile)
        
        session_data = self.cog.global_chat_sessions.get(self.session_key)
        if not session_data:
            session_data = await self.cog.session_manager._load_session_from_disk(self.session_key, 'global_chat')
            if session_data:
                self.cog.global_chat_sessions[self.session_key] = session_data
        
        self.rounds = []
        if session_data and 'unified_log' in session_data:
            log = session_data['unified_log']
            i = 0
            while i < len(log) - 1:
                curr = log[i]
                next_t = log[i+1]
                if curr.get('role') == 'user' and next_t.get('role') == 'model':
                    self.rounds.append((curr, next_t))
                    i += 2
                else:
                    i += 1
        
        self.current_page = max(0, len(self.rounds) - 1)

    def _build_view(self):
        self.clear_items()
        
        if not self.available_profiles:
            return

        # Row 0: Profile Select
        profile_options = []
        for p in self.available_profiles[:25]: 
            profile_options.append(discord.SelectOption(label=p, value=p, default=(p == self.selected_profile)))
        
        profile_select = ui.Select(placeholder="Select a conversation history...", options=profile_options, row=0)
        profile_select.callback = self.profile_callback
        self.add_item(profile_select)

        if not self.rounds:
            return

        self.current_page = max(0, min(self.current_page, len(self.rounds) - 1))

        # Row 1: Jump Select
        options = []
        start_jump = max(0, len(self.rounds) - 25)
        for i in range(start_jump, len(self.rounds)):
            user_turn, _ = self.rounds[i]
            ts_str = "Unknown"
            if user_turn.get("timestamp"):
                try: ts_str = datetime.datetime.fromisoformat(user_turn.get("timestamp")).strftime('%b %d, %I:%M %p')
                except: pass
            
            content_preview = user_turn.get("content", "")[:50]
            label = f"({ts_str}) {content_preview}..."
            options.append(discord.SelectOption(label=label, value=str(i), default=(i == self.current_page)))
        
        if options:
            jump_select = ui.Select(placeholder="Jump to a round...", options=options, row=1)
            jump_select.callback = self.jump_callback
            self.add_item(jump_select)

        # Row 2: Buttons
        build_pagination_controls(self, self.current_page, len(self.rounds), 2, self.prev_callback, self.next_callback)
        delete_btn = ui.Button(label="Delete", style=discord.ButtonStyle.danger, row=2)
        delete_btn.callback = self.delete_callback
        
        self.add_item(delete_btn)

    def get_embed(self) -> discord.Embed:
        display_name = self.selected_profile
        avatar_url = self.cog.bot.user.display_avatar.url
        
        effective_owner_id, effective_profile_name = self.cog.profile_manager._resolve_effective_profile(self.user_id, self.selected_profile)
        
        appearance_data = self.cog.profile_manager._get_user_appearance(effective_owner_id, effective_profile_name)
        if appearance_data:
            display_name = appearance_data.get("custom_display_name") or display_name
            avatar_url = appearance_data.get("custom_avatar_url") or avatar_url

        if not self.rounds:
            embed = discord.Embed(description="No conversation history found. Click 'Reply' to start.", color=discord.Color.dark_grey())
            embed.set_author(name=display_name, icon_url=avatar_url)
            return embed
            
        user_turn, model_turn = self.rounds[self.current_page]
        embed = discord.Embed(description=model_turn.get("content"), color=discord.Color.dark_grey())
        embed.set_author(name=display_name, icon_url=avatar_url)
        
        user_input = user_turn.get("content", "")
        embed.set_footer(text=f"You: {user_input}", icon_url=self.original_interaction.user.display_avatar.url)
        
        return embed

    async def _wait_and_unlock(self, session_key):
        await asyncio.sleep(10.5) # Slight buffer over 10s
        session_data = self.cog.global_chat_sessions.get(session_key)
        if session_data and time.time() >= session_data.get("lock_expiry", 0):
            # Enforce the deadline by clearing the active typers
            session_data.setdefault("active_typers", set()).clear()
            self._build_view()
            try:
                await self.original_interaction.edit_original_response(view=self)
            except Exception:
                pass

    async def reply_callback(self, interaction: discord.Interaction):
        session_data = self.cog.global_chat_sessions.setdefault(self.session_key, {})
        
        # Timer / Lock Logic
        lock_resets = session_data.get("lock_resets", 0)
        now = time.time()
        
        if now > session_data.get("lock_expiry", 0):
            session_data["lock_resets"] = 0
            lock_resets = 0

        # Allow max 2 resets (30s deadline)
        if lock_resets <= 2:
            session_data["lock_expiry"] = now + 10
            session_data["lock_resets"] = lock_resets + 1
            
        session_data.setdefault("active_typers", set()).add(interaction.user.id)
        
        self._build_view()
        try:
            await self.original_interaction.edit_original_response(view=self)
        except Exception:
            pass

        queue = session_data.setdefault("pending_queue", {})
        existing_text = queue.get(interaction.user.id, {}).get("content", "")

        modal = GlobalChatInputModal(self.cog, self, existing_text)
        await interaction.response.send_modal(modal)
        
        self.cog.bot.loop.create_task(self._wait_and_unlock(self.session_key))

    async def play_callback(self, interaction: discord.Interaction):
        session_data = self.cog.global_chat_sessions.get(self.session_key)
        if not session_data: return
        
        queue = session_data.get("pending_queue", {})
        if not queue: return

        queued_turns = sorted(list(queue.values()), key=lambda x: x["timestamp"])
        session_data["pending_queue"] = {} 
        
        self.clear_items()
        await interaction.response.edit_message(view=self)

        profile_name = self.session_key[2]
        await self.cog.generation_service._execute_global_chat(interaction, profile_name, queued_turns)
        
        await self._load_current_session()
        self._build_view()
        await interaction.edit_original_response(embed=self.get_embed(), view=self)

    async def profile_callback(self, interaction: discord.Interaction):
        self.selected_profile = interaction.data['values'][0]
        await self._load_current_session()
        self._build_view()
        await interaction.response.edit_message(embed=self.get_embed(), view=self)

    async def jump_callback(self, interaction: discord.Interaction):
        self.current_page = int(interaction.data['values'][0])
        self._build_view()
        await interaction.response.edit_message(embed=self.get_embed(), view=self)

    async def prev_callback(self, interaction: discord.Interaction):
        self.current_page -= 1
        self._build_view()
        await interaction.response.edit_message(embed=self.get_embed(), view=self)

    async def next_callback(self, interaction: discord.Interaction):
        self.current_page += 1
        self._build_view()
        await interaction.response.edit_message(embed=self.get_embed(), view=self)

    async def delete_callback(self, interaction: discord.Interaction):
        await interaction.response.defer()
        if not self.rounds: return

        user_turn, model_turn = self.rounds[self.current_page]
        ids_to_delete = {user_turn.get("turn_id"), model_turn.get("turn_id")}
        
        session_data = self.cog.global_chat_sessions.get(self.session_key)
        if not session_data:
             await interaction.followup.send("Session expired/unloaded.", ephemeral=True)
             return

        original_len = len(session_data['unified_log'])
        session_data['unified_log'] = [t for t in session_data['unified_log'] if t.get('turn_id') not in ids_to_delete]
        
        if len(session_data['unified_log']) < original_len:
            await self.cog.session_manager._save_session_to_disk(self.session_key, 'global_chat', session_data)

            await self._load_current_session()
            self._build_view()
            
            if not self.rounds:
                await self.cog.session_manager._delete_session_from_disk(self.session_key, 'global_chat')
                self.available_profiles.remove(self.selected_profile)
                
                if self.available_profiles:
                    self.selected_profile = self.available_profiles[0]
                    await self._load_current_session()
                    self._build_view()
                    await interaction.edit_original_response(content="Round deleted. Switching to next available profile.", embed=self.get_embed(), view=self)
                else:
                    await interaction.edit_original_response(content="History cleared and session deleted.", embed=None, view=None)
            else:
                await interaction.edit_original_response(embed=self.get_embed(), view=self)
        else:
            await interaction.followup.send("Failed to delete round.", ephemeral=True)

class WhisperHistoryView(ui.View):
    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction, all_whispers: List[Dict]):
        super().__init__(timeout=600)
        self.cog = cog
        self.original_interaction = interaction
        self.user_id = interaction.user.id
        self.channel_id = interaction.channel_id
        self.all_whispers = all_whispers # This is a list of whisper turns, paired with their responses

        self.filtered_whispers = self.all_whispers
        self.current_page = 0
        self.selected_profile_key: Optional[str] = None
        
        self._build_view()

    def _build_view(self):
        self.clear_items()
        
        # --- Build Profile Filter Dropdown ---
        profile_keys = set()
        for whisper, _ in self.all_whispers:
            target_pid = whisper.get("target_pid")
            # We must map target_pid back to a name for the UI. Since we are looking from the user's perspective, 
            # we check the user's index to resolve the PID to the local name.
            index = self.cog.profile_manager._get_user_index(self.user_id)
            p_name = "Unknown Profile"
            if isinstance(index.get("personal"), dict):
                for name, mapped_pid in index["personal"].items():
                    if mapped_pid == target_pid: p_name = name; break
            if p_name == "Unknown Profile" and isinstance(index.get("borrowed"), dict):
                for name, mapped_pid in index["borrowed"].items():
                    if mapped_pid == target_pid: p_name = name; break

            profile_keys.add((target_pid, p_name))
        
        profile_options = [discord.SelectOption(label="All Profiles", value="all", default=(self.selected_profile_key is None))]
        for pid, p_name in sorted(list(profile_keys), key=lambda x: x[1]):
            profile_options.append(discord.SelectOption(label=p_name, value=pid, default=(self.selected_profile_key == pid)))
        
        profile_select = ui.Select(placeholder="Filter by profile...", options=profile_options, row=0)
        profile_select.callback = self.profile_filter_callback
        self.add_item(profile_select)

        # --- Filter whispers based on selection ---
        if self.selected_profile_key:
            self.filtered_whispers = [pair for pair in self.all_whispers if pair[0].get("target_pid") == self.selected_profile_key]
        else:
            self.filtered_whispers = self.all_whispers

        # --- Build Whisper Jump Dropdown ---
        if self.filtered_whispers:
            self.current_page = max(0, min(self.current_page, len(self.filtered_whispers) - 1))
            whisper_options =[]
            for i, (whisper, _) in enumerate(self.filtered_whispers):
                ts_raw = whisper.get("timestamp")
                if ts_raw:
                    try: ts = datetime.datetime.fromisoformat(ts_raw)
                    except: ts = datetime.datetime.now(datetime.timezone.utc)
                else:
                    ts = datetime.datetime.now(datetime.timezone.utc)
                ts_str = ts.strftime('%b %d, %I:%M %p')
                
                c_split = whisper.get("content", "").split("\n")
                content_preview = c_split[1][:50] if len(c_split) > 1 else c_split[0][:50]
                
                whisper_options.append(discord.SelectOption(label=f"({ts_str}) {content_preview}...", value=str(i), default=(i == self.current_page)))
            
            whisper_select = ui.Select(placeholder="Jump to a whisper...", options=whisper_options[:DROPDOWN_MAX_OPTIONS], row=1)
            whisper_select.callback = self.whisper_jump_callback
            self.add_item(whisper_select)

        # --- Build Buttons ---
        async def _prev(i):
            self.current_page -= 1
            self._build_view()
            await i.response.edit_message(embed=self._get_current_embed(), view=self)
            
        async def _next(i):
            self.current_page += 1
            self._build_view()
            await i.response.edit_message(embed=self._get_current_embed(), view=self)

        build_pagination_controls(self, self.current_page, len(self.filtered_whispers), 2, _prev, _next)

        delete_button = ui.Button(label="Delete", style=discord.ButtonStyle.danger, disabled=(not self.filtered_whispers), row=2)
        delete_button.callback = self.delete_callback
        self.add_item(delete_button)

    def _get_current_embed(self) -> discord.Embed:
        if not self.filtered_whispers:
            return discord.Embed(title="Whisper History", description="No whispers found for the selected filter.", color=discord.Color.dark_grey())

        whisper_turn, response_turn = self.filtered_whispers[self.current_page]
        
        r_split = response_turn.get("content", "").split("\n")
        response_content = "\n".join(r_split[1:]).strip() if len(r_split) > 1 else r_split[0].strip()
        
        w_split = whisper_turn.get("content", "").split("\n")
        whisper_content = "\n".join(w_split[1:]).strip() if len(w_split) > 1 else w_split[0].strip()

        target_pid = whisper_turn.get("target_pid")
        index = self.cog.profile_manager._get_user_index(self.user_id)
        
        effective_profile_name = "Unknown Profile"
        if isinstance(index.get("personal"), dict):
            for name, mapped_pid in index["personal"].items():
                if mapped_pid == target_pid: effective_profile_name = name; break
        if effective_profile_name == "Unknown Profile" and isinstance(index.get("borrowed"), dict):
            for name, mapped_pid in index["borrowed"].items():
                if mapped_pid == target_pid: effective_profile_name = name; break

        effective_owner_id, effective_profile_name = self.cog.profile_manager._resolve_effective_profile(self.user_id, effective_profile_name)
        
        display_name = effective_profile_name
        avatar_url = self.cog.bot.user.display_avatar.url
        appearance = self.cog.profile_manager._get_user_appearance(effective_owner_id, effective_profile_name)
        if appearance:
            display_name = appearance.get("custom_display_name") or display_name
            avatar_url = appearance.get("custom_avatar_url") or avatar_url

        embed = discord.Embed(description=response_content, color=discord.Color.dark_grey())
        embed.set_author(name=display_name, icon_url=avatar_url)
        embed.set_footer(text=f"{whisper_content}", icon_url=self.original_interaction.user.display_avatar.url)
        
        return embed

    async def profile_filter_callback(self, interaction: discord.Interaction):
        selection = interaction.data['values'][0]
        self.selected_profile_key = selection if selection != "all" else None
        self.current_page = 0
        self._build_view()
        await interaction.response.edit_message(embed=self._get_current_embed(), view=self)

    async def whisper_jump_callback(self, interaction: discord.Interaction):
        self.current_page = int(interaction.data['values'][0])
        self._build_view()
        await interaction.response.edit_message(embed=self._get_current_embed(), view=self)

    async def delete_callback(self, interaction: discord.Interaction):
        if not self.filtered_whispers:
            await interaction.response.send_message("Nothing to delete.", ephemeral=True, delete_after=5)
            return

        whisper_turn, response_turn = self.filtered_whispers[self.current_page]
        whisper_turn_id = whisper_turn.get("turn_id")
        response_turn_id = response_turn.get("turn_id")

        # Re-using logic via the updated WhisperActionView (passing None for optional regen args)
        view = WhisperActionView(self.cog, self.original_interaction, whisper_turn_id, response_turn_id)
        await view.delete_button.callback(interaction)
        
        # After deletion, refresh the history view
        await self.cog._show_whisper_history(self.original_interaction)

class WhisperActionView(ui.View):
    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction, whisper_turn_id: str, response_turn_id: str, target_participant: Optional[Dict] = None, whisper_message: Optional[str] = None):
        super().__init__(timeout=300)
        self.cog = cog
        self.original_interaction = interaction
        self.channel_id = interaction.channel_id
        self.whisper_turn_id = whisper_turn_id
        self.response_turn_id = response_turn_id
        self.target_participant = target_participant
        self.whisper_message = whisper_message

    @ui.button(label="Delete", style=discord.ButtonStyle.danger, custom_id="delete_whisper")
    async def delete_button(self, interaction: discord.Interaction, button: ui.Button):
        session = self.cog.multi_profile_channels.get(self.channel_id)
        if not session:
            await interaction.response.edit_message(content="Session not found or has ended.", view=None, embed=None)
            return

        if not session.get("is_hydrated"):
            session = await self.cog.session_manager._ensure_session_hydrated(self.channel_id, session.get("type", "multi"))

        if not session:
            await interaction.response.edit_message(content="Failed to load session memory.", view=None, embed=None)
            return

        turn_ids_to_delete = {self.whisper_turn_id, self.response_turn_id}
        original_log_len = len(session.get("unified_log", []))
        
        target_pid = None
        for turn in session.get("unified_log", []):
            if turn.get("turn_id") in turn_ids_to_delete:
                target_pid = turn.get("target_pid") or turn.get("speaker_pid")
                break

        session["unified_log"] = [
            turn for turn in session.get("unified_log", [])
            if turn.get("turn_id") not in turn_ids_to_delete
        ]

        if len(session["unified_log"]) < original_log_len and target_pid:
            session_type = session.get("type", "multi")
            await self.cog.session_manager._save_session_to_disk((self.channel_id, None, None), session_type, session["unified_log"])
            session["is_hydrated"] = False

        await interaction.response.edit_message(content="Whisper has been deleted from the profile's memory.", view=None, embed=None)

    @ui.button(label="Regenerate", style=discord.ButtonStyle.secondary, custom_id="regenerate_whisper")
    async def regenerate_button(self, interaction: discord.Interaction, button: ui.Button):
        if not self.target_participant or not self.whisper_message:
            await interaction.response.send_message("Regeneration is only available for recent whispers.", ephemeral=True)
            return
        await self.cog.generation_service._execute_whisper_regeneration(interaction, self.whisper_turn_id, self.response_turn_id, self.target_participant, self.whisper_message)

class SessionPromptModal(ui.Modal, title="Set Master Prompt"):
    prompt_input = ui.TextInput(label="Master Prompt / Director's Note", style=discord.TextStyle.paragraph, placeholder="The persistent background instruction to set the scene...", required=False, max_length=1500)
    def __init__(self, view: 'SessionConfigView'):
        super().__init__()
        self.view = view
        if view.session.get("session_prompt"):
            self.prompt_input.default = view.session.get("session_prompt")
    async def on_submit(self, interaction: discord.Interaction):
        self.view.session["session_prompt"] = self.prompt_input.value or None
        self.view.cog.session_manager._save_multi_profile_sessions()
        await interaction.response.defer()
        await self.view.update_display()

class ReactivitySettingsModal(ui.Modal, title="Edit Reactivity"):
    chance_input = ui.TextInput(label="Chance to Respond (0-100)", placeholder="Leave blank to make no change", required=False, max_length=3)
    keywords_input = ui.TextInput(label="Wakewords (comma-separated)", style=discord.TextStyle.paragraph, placeholder="Leave blank to make no change", required=False, max_length=500)
    
    def __init__(self, view, participants: List[Dict]):
        super().__init__()
        self.view = view
        self.participants = participants
        
        if len(participants) == 1:
            p = participants[0]
            self.chance_input.default = str(p.get("chance", 100))
            self.keywords_input.default = ", ".join(p.get("wakewords", []))
            self.chance_input.placeholder = "Default: 100"
            self.keywords_input.placeholder = "e.g. hey bot, look at this"

    async def on_submit(self, interaction: discord.Interaction):
        chance_val_str = self.chance_input.value.strip()
        keywords_val_str = self.keywords_input.value.strip()
        
        parsed_chance = None
        if chance_val_str:
            try:
                val = int(chance_val_str)
                if not (0 <= val <= 100):
                    raise ValueError()
                parsed_chance = val
            except ValueError:
                await interaction.response.send_message("❌ Invalid chance percentage. Must be a number between 0 and 100.", ephemeral=True)
                return
        
        parsed_keywords = None
        if keywords_val_str:
            parsed_keywords = [w.strip().lower() for w in keywords_val_str.split(',') if w.strip()]

        for p in self.participants:
            if parsed_chance is not None:
                p["chance"] = parsed_chance
            if parsed_keywords is not None:
                p["wakewords"] = parsed_keywords
        
        self.view.selected_reactivity_profiles.clear()
        self.view.cog.session_manager._save_multi_profile_sessions()
        await interaction.response.defer()
        await self.view.update_display()

DEFAULT_DIRECTOR_PROMPT = "You are an AI Director for a roleplay session. Introduce a sudden event, an environmental change, or a question to spark conversation among the cast. Keep it brief (1-2 sentences)."

class ProactivitySettingsModal(ui.Modal, title="Proactivity & AI Director"):
    chance_input = ui.TextInput(label="Trigger Chance (0-100%)", placeholder="Default: 10", required=True, max_length=3)
    cooldown_input = ui.TextInput(label="Cooldown (Seconds)", placeholder="Default: 300", required=True, max_length=5)
    model_input = ui.TextInput(label="Director Model (on/off or Model ID)", placeholder="Default: off", required=False, max_length=100)
    instructions_input = ui.TextInput(label="Director Instructions (Blank = Default)", style=discord.TextStyle.paragraph, required=False, max_length=1000)
    def __init__(self, view: 'SessionConfigView'):
        super().__init__()
        self.view = view
        pro = view.session.get("proactivity", {})
        self.chance_input.default = str(pro.get("chance", 10))
        self.cooldown_input.default = str(pro.get("cooldown", 300))
        self.model_input.default = pro.get("director_model", "off")
        self.instructions_input.default = pro.get("director_instructions", DEFAULT_DIRECTOR_PROMPT)
    async def on_submit(self, interaction: discord.Interaction):
        try:
            pro = self.view.session.setdefault("proactivity", {})
            pro["chance"] = max(0, min(100, int(self.chance_input.value)))
            pro["cooldown"] = max(60, int(self.cooldown_input.value))
            
            model_val = self.model_input.value.strip().lower()
            if model_val in ["", "off"]:
                pro["director_model"] = "off"
            elif model_val == "on":
                pro["director_model"] = "GOOGLE/gemini-2.5-flash-lite"
            else:
                model_val_orig = self.model_input.value.strip()
                if not (model_val_orig.upper().startswith("GOOGLE/") or model_val_orig.upper().startswith("OPENROUTER/")):
                    model_val_orig = "GOOGLE/" + model_val_orig
                pro["director_model"] = model_val_orig
            
            ins_val = self.instructions_input.value.strip()
            pro["director_instructions"] = ins_val if ins_val else DEFAULT_DIRECTOR_PROMPT
            
            self.view.cog.session_manager._save_multi_profile_sessions()
            await interaction.response.defer()
            await self.view.update_display()
        except ValueError:
            await interaction.response.send_message("Invalid chance or cooldown.", ephemeral=True)

class ResponseLimitModal(ui.Modal, title="Set Response Limit"):
    limit_input = ui.TextInput(label="Max Responses per Round (1-10)", placeholder="Enter a number between 1 and 10 (Default: 10)", required=False, max_length=2)
    def __init__(self, view):
        super().__init__()
        self.view = view
        if view.session.get("max_responses"):
            self.limit_input.default = str(view.session.get("max_responses"))
    async def on_submit(self, interaction: discord.Interaction):
        val_str = self.limit_input.value.strip()
        if not val_str:
            self.view.session["max_responses"] = 10
        else:
            try:
                val = int(val_str)
                if not (1 <= val <= 10):
                    raise ValueError()
                self.view.session["max_responses"] = val
            except ValueError:
                await interaction.response.send_message("❌ Invalid input. Please enter a number between 1 and 10.", ephemeral=True)
                return
        self.view.cog.session_manager._save_multi_profile_sessions()
        await interaction.response.defer()
        await self.view.update_display()

class SessionSwapListView(ui.View):
    def __init__(self, cog, interaction, session_data_idx):
        super().__init__(timeout=180)
        self.cog = cog
        self.interaction = interaction
        self.session_data_idx = session_data_idx
        self.current_page = 0
        self.items_per_page = 20
        self.update_view()

    def update_view(self):
        self.clear_items()
        profiles = self.session_data_idx.get("profiles", [])
        total_items = len(profiles)
        num_pages = max(1, (total_items - 1) // self.items_per_page + 1)
        
        if self.current_page >= num_pages:
            self.current_page = num_pages - 1
            
        start = self.current_page * self.items_per_page
        end = start + self.items_per_page
        page_profiles = profiles[start:end]
        
        profile_list = []
        for i, p_data in enumerate(page_profiles, start=start + 1):
            p_name = p_data.get('profile_name')
            pid = p_data.get('pid', 'Unknown PID')
            method_str = "Child Bot" if p_data.get('method') == 'child_bot' else "Webhook"
            profile_list.append(f"**{i}.** `{p_name}` ({method_str}) [PID: {pid}]")
            
        owner_user = self.cog.bot.get_user(self.session_data_idx.get('owner_id'))
        admin_name = owner_user.name if owner_user else "Unknown Admin"
        
        self.embed = discord.Embed(
            title=f"Current Participants ({total_items})",
            description=f"**Session Admin:** {admin_name}\n\n" + ("\n".join(profile_list) if profile_list else "*No participants*"),
            color=discord.Color.purple()
        )
        self.embed.set_footer(text=f"Page {self.current_page + 1} of {num_pages}")
        
        if num_pages > 1:
            async def prev_cb(i):
                self.current_page -= 1
                self.update_view()
                await i.response.edit_message(embed=self.embed, view=self)
            async def next_cb(i):
                self.current_page += 1
                self.update_view()
                await i.response.edit_message(embed=self.embed, view=self)
            
            build_pagination_controls(self, self.current_page, num_pages, 0, prev_cb, next_cb)

class SessionView(ui.View):
    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction, session: Dict):
        super().__init__(timeout=600)
        self.cog = cog
        self.session = session
        self.channel_id = interaction.channel_id
        self.current_page = 0
        self.items_per_page = 20
        self.update_view()

    def update_view(self):
        self.clear_items()
        profiles = self.session.get("profiles", [])
        total_items = len(profiles)
        num_pages = max(1, (total_items - 1) // self.items_per_page + 1)
        
        if self.current_page >= num_pages:
            self.current_page = num_pages - 1
            
        start = self.current_page * self.items_per_page
        end = start + self.items_per_page
        page_profiles = profiles[start:end]

        options = []
        for i, p in enumerate(page_profiles, start=start):
            p_name = p.get("profile_name")
            method = p.get("method", "webhook")
            label = p_name
            description = f"Owner ID: {p.get('owner_id')}"
            
            if method == 'child_bot':
                bot_id = p.get("bot_id")
                bot_user = self.cog.bot.get_user(int(bot_id)) if bot_id else None
                if bot_user: 
                    label = f"{bot_user.name} ({p_name})"
                    description = "Child Bot"
                else:
                    label = f"Bot {bot_id} ({p_name})"
            
            options.append(discord.SelectOption(label=label[:100], value=str(i), description=description))

        if options:
            self.select = ui.Select(placeholder="Select a participant to view details...", options=options)
            self.select.callback = self.callback
            self.add_item(self.select)

        if num_pages > 1:
            prev_btn = ui.Button(label="◀", style=discord.ButtonStyle.secondary, disabled=(self.current_page == 0), row=1)
            async def prev_cb(i):
                self.current_page -= 1
                self.update_view()
                await i.response.edit_message(view=self)
            prev_btn.callback = prev_cb
            self.add_item(prev_btn)
            
            next_btn = ui.Button(label="▶", style=discord.ButtonStyle.secondary, disabled=(self.current_page == num_pages - 1), row=1)
            async def next_cb(i):
                self.current_page += 1
                self.update_view()
                await i.response.edit_message(view=self)
            next_btn.callback = next_cb
            self.add_item(next_btn)

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        idx = int(self.select.values[0])
        participant = self.session["profiles"][idx]
        
        embed = await self.cog.profile_manager._build_profile_embed(participant['owner_id'], participant['profile_name'], self.channel_id)
        await interaction.followup.send(embed=embed, ephemeral=True)

class SessionConfigView(ui.View):
    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction, session: dict):
        super().__init__(timeout=600)
        self.cog = cog
        self.original_interaction = interaction
        self.session = session
        self.current_tab = "cast"
        self.view_source = 'personal'
        self.current_page = 0
        self.selected_reactivity_profiles = set()
        self._load_lists()

    def _load_lists(self):
        user_id = self.original_interaction.user.id
        index = self.cog.profile_manager._get_user_index(user_id)
        self.lists = {
            'personal': sorted(list(index.get("personal", []))),
            'borrowed': sorted(list(index.get("borrowed", []))),
            'child_bot': sorted([b for b in self.cog.child_bots.values() if b['owner_id'] == user_id], key=lambda x: x.get('profile_name', ''))
        }

    def _add_nav_buttons(self):
        tabs = ["cast", "config", "reactivity", "proactivity"]
        for tab in tabs:
            btn = ui.Button(label=tab.title(), style=discord.ButtonStyle.primary if self.current_tab == tab else discord.ButtonStyle.secondary, row=4, disabled=(self.current_tab == tab))
            def make_cb(t):
                async def cb(i: discord.Interaction):
                    self.current_tab = t; self.current_page = 0
                    self.selected_reactivity_profiles.clear()
                    await i.response.defer(); await self.update_display()
                return cb
            btn.callback = make_cb(tab)
            self.add_item(btn)

    async def update_display(self):
        self.clear_items()
        self._add_nav_buttons()
        embed = discord.Embed(title=f"Chat Session: #{self.original_interaction.channel.name}", color=discord.Color.gold())

        if self.current_tab == "cast":
            embed.description = "Add or remove participants (200 max). This controls who is actively in the channel."
            active_list = self.lists[self.view_source]
            
            selected_items = []
            
            active_map = {}
            if self.view_source == 'child_bot':
                for item in active_list:
                    bid = next((k for k, v in self.cog.child_bots.items() if v is item), None)
                    if bid:
                        active_map[str(bid)] = item
            else:
                for item in active_list:
                    active_map[item] = item
                    
            for p in self.session.get('profiles', []):
                p_method = p.get('method')
                if self.view_source == 'child_bot' and p_method == 'child_bot':
                    bid_str = str(p.get('bot_id'))
                    if bid_str in active_map:
                        selected_items.append(active_map.pop(bid_str))
                elif self.view_source != 'child_bot' and p_method != 'child_bot':
                    p_name = p.get('profile_name')
                    if p_name in active_map:
                        selected_items.append(active_map.pop(p_name))
                        
            unselected_items = [item for item in active_list if item in active_map.values()]
            
            ordered_list = selected_items + unselected_items
            
            num_pages = max(1, (len(ordered_list) - 1) // 20 + 1)
            if self.current_page >= num_pages:
                self.current_page = num_pages - 1
                
            start = self.current_page * 20
            page_items = ordered_list[start : start + 20]

            options = []
            if page_items:
                page_selected_count = 0
                for item in page_items:
                    is_sel = False
                    if self.view_source == 'child_bot':
                        bid = next((k for k, v in self.cog.child_bots.items() if v is item), None)
                        if bid:
                            is_sel = any(p.get('method') == 'child_bot' and str(p.get('bot_id')) == str(bid) for p in self.session.get('profiles', []))
                    else:
                        is_sel = any(p.get('method') != 'child_bot' and p.get('profile_name') == item for p in self.session.get('profiles', []))
                    
                    if is_sel:
                        page_selected_count += 1

                page_toggle_label = "Unselect Page" if (page_selected_count == len(page_items)) else "Select Page"
                options.append(discord.SelectOption(label=page_toggle_label, value="toggle_page", description="Toggle selection for all profiles on this page.", emoji="📄"))
                
                total_selected_count = len(selected_items)
                all_toggle_label = "Unselect All" if (total_selected_count == len(ordered_list)) else "Select All"
                options.append(discord.SelectOption(label=all_toggle_label, value="toggle_all", description="Toggle selection for all profiles in this source.", emoji="📚"))

                for item in page_items:
                    if self.view_source == 'child_bot':
                        p_name = item.get('profile_name')
                        bid = next((k for k, v in self.cog.child_bots.items() if v is item), None)
                        val = f"child_{bid}"
                        lbl = p_name
                        
                        is_sel = any(p.get('method') == 'child_bot' and str(p.get('bot_id')) == str(bid) for p in self.session.get('profiles', []))
                    else:
                        val = item
                        lbl = item
                        
                        is_sel = any(p.get('method') != 'child_bot' and p.get('profile_name') == val for p in self.session.get('profiles', []))
                    
                    options.append(discord.SelectOption(label=lbl[:100], value=val[:100], default=is_sel))
            else:
                options.append(discord.SelectOption(label="No profiles found", value="none"))

            sel = ui.Select(placeholder=f"Select {self.view_source.replace('_', ' ')} profiles...", min_values=0, max_values=len(options) if page_items else 1, options=options, row=0, disabled=(not page_items))
            async def cast_cb(i: discord.Interaction):
                if "none" in i.data['values']: await i.response.defer(); return
                raw_vals = i.data['values']
                curr_vals = set(raw_vals)
                page_vals = set([o.value for o in options if o.value not in ["toggle_page", "toggle_all"]])
                
                if "toggle_page" in curr_vals:
                    page_val_list = [f"child_{next((k for k, v in self.cog.child_bots.items() if v is item), None)}" if self.view_source == 'child_bot' else item for item in page_items]
                    page_set_vals = set(page_val_list)
                    
                    already_selected = set()
                    for p in self.session.get('profiles', []):
                        val_p = f"child_{p.get('bot_id')}" if p.get('method') == 'child_bot' else p.get('profile_name')
                        if val_p in page_set_vals:
                            already_selected.add(val_p)
                            
                    if len(already_selected) == len(page_items):
                        self.session['profiles'] = [p for p in self.session['profiles'] if (f"child_{p.get('bot_id')}" if p.get('method') == 'child_bot' else p.get('profile_name')) not in page_set_vals]
                    else:
                        for item in page_items:
                            if self.view_source == 'child_bot':
                                bid = next((k for k, v in self.cog.child_bots.items() if v is item), None)
                                if bid and not any(p.get('bot_id') == bid for p in self.session['profiles']):
                                    if len(self.session['profiles']) >= 200: break
                                    bc = self.cog.child_bots.get(bid)
                                    if bc: self.session['profiles'].append({"owner_id": bc['owner_id'], "profile_name": bc['profile_name'], "method": "child_bot", "bot_id": bid, "chance": 100, "wakewords": []})
                            else:
                                if not any(p.get('profile_name') == item and p.get('method') != 'child_bot' for p in self.session['profiles']):
                                    if len(self.session['profiles']) >= 200: break
                                    self.session['profiles'].append({"owner_id": self.original_interaction.user.id, "profile_name": item, "method": "webhook", "chance": 100, "wakewords": []})
                
                elif "toggle_all" in curr_vals:
                    all_val_list = [f"child_{next((k for k, v in self.cog.child_bots.items() if v is item), None)}" if self.view_source == 'child_bot' else item for item in ordered_list]
                    all_set_vals = set(all_val_list)
                    
                    already_selected_all = set()
                    for p in self.session.get('profiles', []):
                        val_p = f"child_{p.get('bot_id')}" if p.get('method') == 'child_bot' else p.get('profile_name')
                        if val_p in all_set_vals:
                            already_selected_all.add(val_p)
                            
                    if len(already_selected_all) == len(ordered_list):
                        self.session['profiles'] = [p for p in self.session['profiles'] if (f"child_{p.get('bot_id')}" if p.get('method') == 'child_bot' else p.get('profile_name')) not in all_set_vals]
                    else:
                        for item in ordered_list:
                            if self.view_source == 'child_bot':
                                bid = next((k for k, v in self.cog.child_bots.items() if v is item), None)
                                if bid and not any(p.get('bot_id') == bid for p in self.session['profiles']):
                                    if len(self.session['profiles']) >= 200: break
                                    bc = self.cog.child_bots.get(bid)
                                    if bc: self.session['profiles'].append({"owner_id": bc['owner_id'], "profile_name": bc['profile_name'], "method": "child_bot", "bot_id": bid, "chance": 100, "wakewords": []})
                            else:
                                if not any(p.get('profile_name') == item and p.get('method') != 'child_bot' for p in self.session['profiles']):
                                    if len(self.session['profiles']) >= 200: break
                                    self.session['profiles'].append({"owner_id": self.original_interaction.user.id, "profile_name": item, "method": "webhook", "chance": 100, "wakewords": []})
                
                else:
                    self.session['profiles'] = [p for p in self.session['profiles'] if not (
                        (f"child_{p.get('bot_id')}" in page_vals and f"child_{p.get('bot_id')}" not in curr_vals) or
                        (p.get('method') != 'child_bot' and p['profile_name'] in page_vals and p['profile_name'] not in curr_vals)
                    )]
                    
                    for val in raw_vals:
                        if val.startswith("child_"):
                            bid = val.split("_")[1]
                            if not any(p.get('bot_id') == bid for p in self.session['profiles']):
                                if len(self.session['profiles']) >= 200: break
                                bc = self.cog.child_bots.get(bid)
                                if bc: self.session['profiles'].append({"owner_id": bc['owner_id'], "profile_name": bc['profile_name'], "method": "child_bot", "bot_id": bid, "chance": 100, "wakewords": []})
                        else:
                            if not any(p.get('profile_name') == val and p.get('method') != 'child_bot' for p in self.session['profiles']):
                                if len(self.session['profiles']) >= 200: break
                                self.session['profiles'].append({"owner_id": self.original_interaction.user.id, "profile_name": val, "method": "webhook", "chance": 100, "wakewords": []})
                
                self.cog.session_manager._save_multi_profile_sessions()
                
                # [NEW] Dispatch child bot presence updates
                for p_data in self.session.get('profiles', []):
                    if p_data.get('method') == 'child_bot':
                        await self.cog.manager_queue.put({
                            "action": "send_to_child", "bot_id": p_data['bot_id'],
                            "payload": {"action": "session_update_add", "channel_id": self.original_interaction.channel_id}
                        })

                await i.response.defer(); await self.update_display()
            sel.callback = cast_cb
            self.add_item(sel)

            source_btn = ui.Button(label=f"Source: {self.view_source.title().replace('_', ' ')}", style=discord.ButtonStyle.blurple, row=1)
            async def src_cb(i): 
                cycle = ['personal', 'borrowed', 'child_bot']
                self.view_source = cycle[(cycle.index(self.view_source) + 1) % 3]
                self.current_page = 0
                await i.response.defer(); await self.update_display()
            source_btn.callback = src_cb
            self.add_item(source_btn)

            if num_pages > 1:
                async def p_cb(i): self.current_page -= 1; await i.response.defer(); await self.update_display()
                async def n_cb(i): self.current_page += 1; await i.response.defer(); await self.update_display()
                build_pagination_controls(self, self.current_page, num_pages, 1, p_cb, n_cb)

            profiles = self.session.get('profiles', [])
            total_active = len(profiles)
            
            if total_active <= 20:
                cast_list = "\n".join(f"{idx+1}. `{p['profile_name']}` ({'Child Bot' if p.get('method') == 'child_bot' else 'Webhook'})" for idx, p in enumerate(profiles)) or "*No participants*"
            else:
                start_p = self.current_page * 20
                end_p = start_p + 20
                page_profiles_cast = profiles[start_p:end_p]
                
                cast_lines = []
                for idx, p in enumerate(page_profiles_cast, start=start_p + 1):
                    method_lbl = 'Child Bot' if p.get('method') == 'child_bot' else 'Webhook'
                    cast_lines.append(f"{idx}. `{p['profile_name']}` ({method_lbl})")
                cast_list = "\n".join(cast_lines) or "*No participants*"
                
            embed.add_field(name="Current Cast", value=cast_list, inline=False)

        elif self.current_tab == "config":
            embed.description = "Configure session-wide behavior."
            mp = self.session.get("session_prompt")
            audio_val = self.session.get("audio_mode", "off")
            tts_status = "**`ON`**" if audio_val == "on" else "`OFF`"
            response_limit = self.session.get("max_responses", 10)
            
            embed.add_field(name="Execution Mode", value=f"`{self.session.get('session_mode', 'sequential').title()}`", inline=True)
            embed.add_field(name="Master Prompt", value=f"`{'Set' if mp else 'Not Set'}`", inline=True)
            embed.add_field(name="\u200b", value="\u200b", inline=True)
            
            embed.add_field(name="Text-to-Speech", value=f"{tts_status}", inline=True)
            embed.add_field(name="Response Limit", value=f"`{response_limit}`", inline=True)
            embed.add_field(name="\u200b", value="\u200b", inline=True)
            
            embed.add_field(name="Master Prompt Content", value=f"```{mp[:500]}```" if mp else "`None`", inline=False)

            mode_btn = ui.Button(label="Toggle Execution", style=discord.ButtonStyle.secondary, row=0)
            async def mode_cb(i):
                self.session["session_mode"] = "random" if self.session.get("session_mode", "sequential") == "sequential" else "sequential"
                self.cog.session_manager._save_multi_profile_sessions()
                await i.response.defer(); await self.update_display()
            mode_btn.callback = mode_cb
            self.add_item(mode_btn)

            prompt_btn = ui.Button(label="Edit Master Prompt", style=discord.ButtonStyle.primary, row=0)
            async def pr_cb(i): await i.response.send_modal(SessionPromptModal(self))
            prompt_btn.callback = pr_cb
            self.add_item(prompt_btn)

            audio_btn = ui.Button(label="Toggle TTS", style=discord.ButtonStyle.secondary, row=1)
            async def audio_cb(i):
                self.session["audio_mode"] = "off" if self.session.get("audio_mode", "off") == "on" else "on"
                self.cog.session_manager._save_multi_profile_sessions()
                await i.response.defer(); await self.update_display()
            audio_btn.callback = audio_cb
            self.add_item(audio_btn)

            limit_btn = ui.Button(label="Set Response Limit", style=discord.ButtonStyle.primary, row=1)
            async def limit_cb(i): await i.response.send_modal(ResponseLimitModal(self))
            limit_btn.callback = limit_cb
            self.add_item(limit_btn)

        elif self.current_tab == "reactivity":
            embed.description = "Manage how likely participants are to respond to messages."
            profiles = self.session.get('profiles', [])
            total_items = len(profiles)
            num_pages = max(1, (total_items - 1) // 20 + 1)
            
            if self.current_page >= num_pages:
                self.current_page = max(0, num_pages - 1)
                
            start = self.current_page * 20
            page_profiles = profiles[start : start + 20]
            
            page_keys = {(p['owner_id'], p['profile_name']) for p in page_profiles}
            
            options = []
            if page_profiles:
                page_selected = page_keys.issubset(self.selected_reactivity_profiles)
                page_label = "Unselect Page" if page_selected else "Select Page"
                options.append(discord.SelectOption(label=page_label, value="toggle_page", description="Toggle selection for all profiles on this page.", emoji="📄"))
                
                all_set = {(p['owner_id'], p['profile_name']) for p in profiles}
                all_selected = all_set.issubset(self.selected_reactivity_profiles)
                all_label = "Unselect All" if all_selected else "Select All"
                options.append(discord.SelectOption(label=all_label, value="toggle_all", description="Toggle selection for all profiles.", emoji="📚"))

                for p in page_profiles:
                    p_key = (p['owner_id'], p['profile_name'])
                    is_checked = p_key in self.selected_reactivity_profiles
                    options.append(discord.SelectOption(
                        label=f"{p['profile_name']} (Chance: {p.get('chance', 100)}%)",
                        value=f"{p['owner_id']}:{p['profile_name']}",
                        default=is_checked
                    ))
            else:
                options.append(discord.SelectOption(label="No profiles found", value="none"))

            sel = ui.Select(
                placeholder="Select participant(s) to edit...", 
                min_values=0, 
                max_values=len(options) if page_profiles else 1, 
                options=options, 
                row=0,
                disabled=(not page_profiles)
            )
            
            async def react_select_cb(i: discord.Interaction):
                if "none" in i.data['values']: await i.response.defer(); return
                vals = i.data['values']
                curr_vals = set(vals)
                
                page_vals_set = {f"{p['owner_id']}:{p['profile_name']}" for p in page_profiles}
                
                if "toggle_page" in curr_vals:
                    if page_keys.issubset(self.selected_reactivity_profiles):
                        self.selected_reactivity_profiles.difference_update(page_keys)
                    else:
                        self.selected_reactivity_profiles.update(page_keys)
                elif "toggle_all" in curr_vals:
                    all_keys = {(p['owner_id'], p['profile_name']) for p in profiles}
                    if all_keys.issubset(self.selected_reactivity_profiles):
                        self.selected_reactivity_profiles.difference_update(all_keys)
                    else:
                        self.selected_reactivity_profiles.update(all_keys)
                else:
                    self.selected_reactivity_profiles.difference_update(page_keys)
                    for val in vals:
                        try:
                            o_id_str, p_name = val.split(":", 1)
                            self.selected_reactivity_profiles.add((int(o_id_str), p_name))
                        except ValueError:
                            pass
                
                await i.response.defer()
                await self.update_display()
                
            sel.callback = react_select_cb
            self.add_item(sel)

            react_lines = []
            for idx, p in enumerate(page_profiles, start=start + 1):
                p_key = (p['owner_id'], p['profile_name'])
                marker = "✅ " if p_key in self.selected_reactivity_profiles else ""
                react_lines.append(f"{idx}. {marker}**{p['profile_name']}**: {p.get('chance', 100)}% (Wakewords: {', '.join(p.get('wakewords', [])) or 'None'})")
            react_list = "\n".join(react_lines)
            embed.add_field(name="Reactivity Stats", value=react_list or "*No participants*", inline=False)

            if num_pages > 1:
                async def p_cb(i: discord.Interaction):
                    self.current_page -= 1
                    await i.response.defer()
                    await self.update_display()
                async def n_cb(i: discord.Interaction):
                    self.current_page += 1
                    await i.response.defer()
                    await self.update_display()
                build_pagination_controls(self, self.current_page, num_pages, 1, p_cb, n_cb)

            selected_count = len(self.selected_reactivity_profiles)
            btn_label = "Bulk Edit" if selected_count > 1 else "Edit"
            btn_style = discord.ButtonStyle.primary if selected_count > 0 else discord.ButtonStyle.secondary
            btn_disabled = (selected_count == 0)
            
            edit_btn = ui.Button(label=btn_label, style=btn_style, disabled=btn_disabled, row=2)
            
            async def edit_callback(i: discord.Interaction):
                targets = []
                for p in profiles:
                    p_key = (p['owner_id'], p['profile_name'])
                    if p_key in self.selected_reactivity_profiles:
                        targets.append(p)
                
                if not targets:
                    await i.response.send_message("❌ No profiles selected.", ephemeral=True)
                    return
                    
                modal = ReactivitySettingsModal(self, targets)
                await i.response.send_modal(modal)
                
            edit_btn.callback = edit_callback
            self.add_item(edit_btn)

        elif self.current_tab == "proactivity":
            pro = self.session.get("proactivity", {})
            enabled = pro.get("enabled", False)
            embed.description = "Allow the session to start conversations autonomously based on a timer."
            embed.add_field(name="Status", value="**`ON`**" if enabled else "`OFF`", inline=True)
            embed.add_field(name="Chance & Cooldown", value=f"`{pro.get('chance', 10)}%` every `{pro.get('cooldown', 300)}s`", inline=True)
            
            dir_mod = pro.get("director_model", "GOOGLE/gemini-2.5-flash-lite")
            dir_ins = pro.get("director_instructions", "(Default)") or "(Default)"
            embed.add_field(name="AI Director", value=f"Model: `{dir_mod}`\nInstructions: ```{dir_ins[:200]}```", inline=False)

            tgl_btn = ui.Button(label="Toggle Proactivity", style=discord.ButtonStyle.success if enabled else discord.ButtonStyle.danger, row=0)
            async def tgl_cb(i):
                self.session.setdefault("proactivity", {})["enabled"] = not enabled
                self.cog.session_manager._save_multi_profile_sessions()
                await i.response.defer(); await self.update_display()
            tgl_btn.callback = tgl_cb
            self.add_item(tgl_btn)

            edit_btn = ui.Button(label="Edit Settings & AI Director", style=discord.ButtonStyle.primary, row=0)
            async def edit_cb(i): await i.response.send_modal(ProactivitySettingsModal(self))
            edit_btn.callback = edit_cb
            self.add_item(edit_btn)

        try:
            await self.original_interaction.edit_original_response(embed=embed, view=self)
        except Exception as e:
            print(f"Error updating SessionConfigView: {e}")

class WakewordsModal(ui.Modal, title="Manage Wakewords"):
    wakewords_input = ui.TextInput(label="Wakewords (comma-separated)", style=discord.TextStyle.paragraph, required=False, max_length=1000)

    def __init__(self, current_wakewords: List[str]):
        super().__init__()
        self.wakewords_input.default = ", ".join(current_wakewords)

class SessionAuditView(ui.View):
    def __init__(self, cog, interaction: discord.Interaction, session: dict, channel_id: int):
        super().__init__(timeout=600)
        self.cog = cog
        self.original_interaction = interaction
        self.session = session
        self.channel_id = channel_id
        self.mode = "overview"
        self.selected_turn_id = None
        self.simulate_profile_key = None
        self.batch_start_id = None
        self.batch_end_id = None
        self.current_page = 0
        self.num_pages = 1
        
        self.all_turns = self.session.get("unified_log", []) or []
        if self.all_turns:
            self.selected_turn_id = self.all_turns[-1].get("turn_id")
            self.batch_start_id = self.all_turns[0].get("turn_id")
            self.batch_end_id = self.all_turns[-1].get("turn_id")
            self.current_page = max(0, (len(self.all_turns) - 1) // 20)
            
        profiles = self.session.get("profiles", [])
        if profiles:
            self.simulate_profile_key = f"{profiles[0]['owner_id']}:{profiles[0]['profile_name']}"

        self._build_view()

    def _resolve_turn_speaker_name(self, turn: dict) -> str:
        if turn.get("is_user"):
            name = turn.get("display_name")
            if not name and turn.get("speaker_pid", "").isdigit():
                u_obj = self.cog.bot.get_user(int(turn.get("speaker_pid")))
                if u_obj: name = u_obj.name
            if not name:
                import re
                m = re.search(r'<([^>]+)>', turn.get("content", ""))
                if m: name = m.group(1)
            return name or f"User ({turn.get('speaker_pid', 'Unknown')})"
        return turn.get("profile_name", "Bot")

    def _extract_turn_preview(self, turn: dict) -> str:
        content = turn.get("content", "")
        import re
        system_tags = [
            "archive_context", "external_context", "document_context", "time_context",
            "whisper_context", "private_whisper", "private_response", "internal_note",
            "scene_prompt", "neuro_endocrine_engine", "neuro_update", "persona_profile",
            "technical_manual", "training_data", "context_rules", "image_context",
            "system_note", "reply_context", "negative_constraints"
        ]
        tags_pattern = "|".join(system_tags)
        clean_text = re.sub(rf'<({tags_pattern})>.*?</\1>', '', content, flags=re.DOTALL | re.IGNORECASE)
        clean_text = re.sub(rf'</?({tags_pattern})>', '', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'<[^>]+>\s*\[ID:[^\]]+\]\s*\[[^\]]+\]:\s*', '', clean_text)
        clean_text = re.sub(r'</?[^>]+>', '', clean_text)
        clean_text = re.sub(r'\(\s*Thought Initiated:.*?\)\s*', '', clean_text)
        clean_text = " ".join(clean_text.split()).strip()
        
        if not clean_text:
            clean_text = "No text content"
        
        if len(clean_text) > 15:
            return f"{clean_text[:15]}..."
        return clean_text

    def _build_view(self):
        self.clear_items()
        
        # Row 0: Modes (Simulator before Turn Inspector)
        modes = [("Overview", "overview"), ("Simulator", "simulator"), ("Turn Inspector", "inspector"), ("Batch Calculator", "batch")]
        for label, val in modes:
            btn = ui.Button(label=label, style=discord.ButtonStyle.primary if self.mode == val else discord.ButtonStyle.secondary, row=0, disabled=(self.mode == val))
            def make_cb(target_mode):
                async def nav_cb(i: discord.Interaction):
                    self.mode = target_mode
                    self._build_view()
                    await i.response.edit_message(embed=self._build_embed(), view=self)
                return nav_cb
            btn.callback = make_cb(val)
            self.add_item(btn)

        # Contextual Dropdowns
        if self.mode == "inspector":
            self.all_turns = self.session.get("unified_log", []) or []
            self.num_pages = max(1, (len(self.all_turns) - 1) // 20 + 1)
            self.current_page = max(0, min(self.current_page, self.num_pages - 1))
            
            start = self.current_page * 20
            page_turns = self.all_turns[start : start + 20]
            
            opts = []
            if self.current_page > 0:
                opts.append(discord.SelectOption(label="◀ Previous Page", value="prev_page", description="Navigate to previous page of turns"))
            
            opts.append(discord.SelectOption(label=f"📄 Page {self.current_page + 1}/{self.num_pages} (Jump)", value="jump_page", description="Click to jump to a page number"))
            
            if self.current_page < self.num_pages - 1:
                opts.append(discord.SelectOption(label="▶ Next Page", value="next_page", description="Navigate to next page of turns"))
            
            for idx, t in enumerate(page_turns):
                abs_turn_num = start + idx + 1
                display = self._resolve_turn_speaker_name(t)
                preview = self._extract_turn_preview(t)
                label = f"Turn #{abs_turn_num} - {display} ({preview})"
                opts.append(discord.SelectOption(label=label[:100], value=t.get("turn_id"), default=(t.get("turn_id") == self.selected_turn_id)))
            
            if opts:
                sel = ui.Select(placeholder="Select a turn to inspect...", options=opts, row=1)
                async def sel_cb(i: discord.Interaction):
                    val = i.data['values'][0]
                    if val == "prev_page":
                        self.current_page -= 1
                        self._build_view()
                        await i.response.edit_message(embed=self._build_embed(), view=self)
                    elif val == "next_page":
                        self.current_page += 1
                        self._build_view()
                        await i.response.edit_message(embed=self._build_embed(), view=self)
                    elif val == "jump_page":
                        await i.response.send_modal(AuditPageJumpModal(self))
                    else:
                        self.selected_turn_id = val
                        self._build_view()
                        await i.response.edit_message(embed=self._build_embed(), view=self)
                sel.callback = sel_cb
                self.add_item(sel)

        elif self.mode == "simulator":
            profiles = self.session.get("profiles", []) or []
            self.num_pages = max(1, (len(profiles) - 1) // 20 + 1)
            self.current_page = max(0, min(self.current_page, self.num_pages - 1))
            
            start = self.current_page * 20
            page_profiles = profiles[start : start + 20]
            
            opts = []
            if self.current_page > 0:
                opts.append(discord.SelectOption(label="◀ Previous Page", value="prev_page", description="Navigate to previous page of profiles"))
            
            if self.num_pages > 1:
                opts.append(discord.SelectOption(label=f"📄 Page {self.current_page + 1}/{self.num_pages} (Jump)", value="jump_page", description="Click to jump to a page number"))
            
            if self.current_page < self.num_pages - 1:
                opts.append(discord.SelectOption(label="▶ Next Page", value="next_page", description="Navigate to next page of profiles"))
            
            for p in page_profiles:
                val = f"{p['owner_id']}:{p['profile_name']}"
                opts.append(discord.SelectOption(label=p['profile_name'][:100], value=val, default=(val == self.simulate_profile_key)))
                
            if opts:
                sel = ui.Select(placeholder="Select a profile to simulate next turn...", options=opts, row=1)
                async def sel_cb(i: discord.Interaction):
                    val = i.data['values'][0]
                    if val == "prev_page":
                        self.current_page -= 1
                        self._build_view()
                        await i.response.edit_message(embed=self._build_embed(), view=self)
                    elif val == "next_page":
                        self.current_page += 1
                        self._build_view()
                        await i.response.edit_message(embed=self._build_embed(), view=self)
                    elif val == "jump_page":
                        await i.response.send_modal(AuditPageJumpModal(self))
                    else:
                        self.simulate_profile_key = val
                        self._build_view()
                        await i.response.edit_message(embed=self._build_embed(), view=self)
                sel.callback = sel_cb
                self.add_item(sel)

        elif self.mode == "batch":
            self.all_turns = self.session.get("unified_log", []) or []
            self.num_pages = max(1, (len(self.all_turns) - 1) // 20 + 1)
            self.current_page = max(0, min(self.current_page, self.num_pages - 1))
            
            start = self.current_page * 20
            page_turns = self.all_turns[start : start + 20]
            
            opts_start = []
            opts_end = []
            
            if self.current_page > 0:
                opts_start.append(discord.SelectOption(label="◀ Previous Page", value="prev_page", description="Navigate to previous page"))
                opts_end.append(discord.SelectOption(label="◀ Previous Page", value="prev_page", description="Navigate to previous page"))
            
            if self.num_pages > 1:
                opts_start.append(discord.SelectOption(label=f"📄 Page {self.current_page + 1}/{self.num_pages} (Jump)", value="jump_page", description="Click to jump to a page number"))
                opts_end.append(discord.SelectOption(label=f"📄 Page {self.current_page + 1}/{self.num_pages} (Jump)", value="jump_page", description="Click to jump to a page number"))
            
            if self.current_page < self.num_pages - 1:
                opts_start.append(discord.SelectOption(label="▶ Next Page", value="next_page", description="Navigate to next page"))
                opts_end.append(discord.SelectOption(label="▶ Next Page", value="next_page", description="Navigate to next page"))

            for idx, t in enumerate(page_turns):
                abs_turn_num = start + idx + 1
                display = self._resolve_turn_speaker_name(t)
                preview = self._extract_turn_preview(t)
                label = f"Turn #{abs_turn_num} - {display} ({preview})"
                opts_start.append(discord.SelectOption(label=label[:100], value=t.get("turn_id"), default=(t.get("turn_id") == self.batch_start_id)))
                opts_end.append(discord.SelectOption(label=label[:100], value=t.get("turn_id"), default=(t.get("turn_id") == self.batch_end_id)))
            
            if opts_start:
                sel_start = ui.Select(placeholder="Select Start Turn...", options=opts_start, row=1)
                async def ss_cb(i: discord.Interaction):
                    val = i.data['values'][0]
                    if val == "prev_page":
                        self.current_page -= 1
                        self._build_view()
                        await i.response.edit_message(embed=self._build_embed(), view=self)
                    elif val == "next_page":
                        self.current_page += 1
                        self._build_view()
                        await i.response.edit_message(embed=self._build_embed(), view=self)
                    elif val == "jump_page":
                        await i.response.send_modal(AuditPageJumpModal(self))
                    else:
                        self.batch_start_id = val
                        self._build_view()
                        await i.response.edit_message(embed=self._build_embed(), view=self)
                sel_start.callback = ss_cb
                self.add_item(sel_start)
                
                sel_end = ui.Select(placeholder="Select End Turn...", options=opts_end, row=2)
                async def se_cb(i: discord.Interaction):
                    val = i.data['values'][0]
                    if val == "prev_page":
                        self.current_page -= 1
                        self._build_view()
                        await i.response.edit_message(embed=self._build_embed(), view=self)
                    elif val == "next_page":
                        self.current_page += 1
                        self._build_view()
                        await i.response.edit_message(embed=self._build_embed(), view=self)
                    elif val == "jump_page":
                        await i.response.send_modal(AuditPageJumpModal(self))
                    else:
                        self.batch_end_id = val
                        self._build_view()
                        await i.response.edit_message(embed=self._build_embed(), view=self)
                sel_end.callback = se_cb
                self.add_item(sel_end)

    def _build_embed(self) -> discord.Embed:
        try:
            embed = discord.Embed(title=f"Chat Session Diagnostic: #{self.original_interaction.channel.name}", color=discord.Color.brand_green())
            log = self.session.get("unified_log", [])

            if self.mode == "overview":
                total_in = 0
                total_out = 0
                total_cost = 0.0
                durations = []
                
                for t in log:
                    if not t.get("is_user"):
                        meta = t.get("meta") or {}
                        i_toks = meta.get("input_tokens", 0)
                        o_toks = meta.get("output_tokens", 0)
                        total_in += i_toks
                        total_out += o_toks
                        total_cost += self.cog.api_service._calculate_turn_cost(meta.get("model", ""), i_toks, o_toks)
                        dur = meta.get("duration")
                        if dur: durations.append(dur)
                
                embed.description = "Session Overview"
                embed.add_field(name="1. Overall Session Telemetry", value=f"├── Active Participants: `{len(self.session.get('profiles', []))}` Profiles\n├── Total Session Turns: `{len(log)}`\n├── Total Input Tokens Processed: `{total_in:,}`\n├── Total Output Tokens Generated: `{total_out:,}`\n└── Estimated Session API Cost: `~${total_cost:.4f} USD`", inline=False)
                
                avg_lat = sum(durations) / len(durations) if durations else 0.0
                embed.add_field(name="2. System Health Checks", value=f"└── Model Latency Average: `{avg_lat:.2f}s`", inline=False)

            elif self.mode == "inspector":
                target = next((t for t in log if t.get("turn_id") == self.selected_turn_id), None)
                if not target:
                    embed.description = "Select a turn to inspect."
                elif target.get("is_user"):
                    speaker_name = self._resolve_turn_speaker_name(target)
                    content = target.get("content", "")
                    if target.get("url_context"):
                        content += f"\n<document_context>\n{target.get('url_context')}\n</document_context>"
                    input_tokens = _estimate_text_tokens(content)
                    
                    embed.add_field(
                        name="Payload Data",
                        value=f"├── Speaker: `{speaker_name}`\n├── Speaker ID: `{target.get('speaker_pid')}`\n├── Timestamp: `{target.get('timestamp')}`\n└── Input Tokens: `{input_tokens:,}`",
                        inline=False
                    )
                else:
                    meta = target.get("meta") or {}
                    mod = meta.get("model", "Unknown")
                    i_tok = meta.get("input_tokens", 0)
                    o_tok = meta.get("output_tokens", 0)
                    r_tok = meta.get("reasoning_tokens", 0)
                    cost = self.cog.api_service._calculate_turn_cost(mod, i_tok, o_tok)
                    
                    embed.add_field(name="Turn Telemetry", value=f"├── Speaker: `{target.get('profile_name')}`\n├── Model Used: `{mod}`\n├── Duration: `{meta.get('duration', 0.0)}s`\n├── Input Tokens: `{i_tok:,}`\n├── Output Tokens: `{o_tok:,}` (Reasoning: `{r_tok:,}`)\n└── Turn Cost: `~${cost:.6f} USD`", inline=False)
                    
                    recalled = len(meta.get("ltms_recalled", []))
                    trained = meta.get("training_recalled", 0)
                    grounded = len(meta.get("grounding_sources", []))
                    
                    embed.add_field(name="Context Injections", value=f"├── LTM Archive: `{recalled} memories`\n├── Training Examples: `{trained} injected`\n└── Web Grounding: `{grounded} sources`", inline=False)

            elif self.mode == "simulator":
                if not self.simulate_profile_key:
                    embed.description = "Select a profile to simulate."
                else:
                    try:
                        o_id_str, p_name = self.simulate_profile_key.split(":", 1)
                        o_id = int(o_id_str)
                        
                        sys_instr, _, _, _, _, _, prim_mod, _ = self.cog.generation_service._construct_system_instructions(o_id, p_name, self.channel_id, is_multi_profile=True)
                        
                        p_idx = self.cog.profile_manager._get_user_index(o_id)
                        is_b = p_name in p_idx.get("borrowed", [])
                        p_cfg = self.cog.profile_manager._get_profile_config(o_id, p_name, is_b) or {}
                        stm_len = int(p_cfg.get("stm_length", defaultConfig.CHATBOT_MEMORY_LENGTH))
                        
                        sys_toks = _estimate_text_tokens(sys_instr)
                        
                        recent_hist = log[-stm_len:] if stm_len > 0 else []
                        hist_str = "\n".join([t.get("content", "") for t in recent_hist])
                        hist_toks = _estimate_text_tokens(hist_str)
                        
                        total_est = sys_toks + hist_toks
                        est_cost = self.cog.api_service._calculate_turn_cost(prim_mod, total_est, 300)
                        
                        embed.add_field(name="Pre-Inference Budget Estimate (Per Turn)", value=f"├── Target: `{p_name}`\n├── Expected Model: `{prim_mod}`\n├── System & Instructions: `~{sys_toks:,} tokens`\n├── STM History Buffer: `~{hist_toks:,} tokens`\n└── **ESTIMATED INPUT TOTAL**: `~{total_est:,} tokens`", inline=False)
                        embed.add_field(name="Financial Projection", value=f"Projected cost for next generation: `~${est_cost:.6f} USD`\n*(Assuming ~300 output tokens)*", inline=False)
                    except Exception as e:
                        embed.description = f"Simulation failed: {e}"

            elif self.mode == "batch":
                if not self.batch_start_id or not self.batch_end_id:
                    embed.description = "Select start and end turns to calculate."
                else:
                    s_idx = next((i for i, t in enumerate(log) if t.get("turn_id") == self.batch_start_id), -1)
                    e_idx = next((i for i, t in enumerate(log) if t.get("turn_id") == self.batch_end_id), -1)
                    
                    if s_idx == -1 or e_idx == -1 or s_idx > e_idx:
                        embed.description = "Invalid range selection. Start turn must occur before End turn."
                    else:
                        batch_turns = log[s_idx:e_idx+1]
                        
                        total_in = 0
                        total_out = 0
                        total_cost = 0.0
                        models_used = {}
                        bot_turns = 0
                        
                        for t in batch_turns:
                            if not t.get("is_user"):
                                meta = t.get("meta") or {}
                                m = meta.get("model", "Unknown")
                                i_toks = meta.get("input_tokens", 0)
                                o_toks = meta.get("output_tokens", 0)
                                
                                total_in += i_toks
                                total_out += o_toks
                                total_cost += self.cog.api_service._calculate_turn_cost(m, i_toks, o_toks)
                                
                                models_used[m] = models_used.get(m, 0) + 1
                                bot_turns += 1
                                
                        dist_str = "\n".join([f"├── {m}: `{c}/{bot_turns} turns`" for m, c in models_used.items()])
                        if not dist_str: dist_str = "├── No bot generations in range."
                        
                        embed.add_field(name="Batch Execution Totals", value=f"├── Turns Evaluated: `{len(batch_turns)}`\n├── Cumulative Input Tokens: `{total_in:,}`\n├── Cumulative Output Tokens: `{total_out:,}`\n└── Combined Range Cost: `~${total_cost:.6f} USD`", inline=False)
                        embed.add_field(name="Model Distribution", value=dist_str, inline=False)

            return embed
        except Exception as e:
            import traceback
            err_trace = traceback.format_exc()
            print(f"Error building audit embed: {err_trace}")
            err_embed = discord.Embed(title="Audit Error", description=f"An error occurred while building the telemetry report:\n```\n{e}\n```", color=discord.Color.red())
            return err_embed

class AuditPageJumpModal(ui.Modal, title="Jump to Page"):
    def __init__(self, parent_view: 'SessionAuditView'):
        super().__init__()
        self.parent_view = parent_view
        self.page_input = ui.TextInput(
            label="Page Number",
            placeholder=f"Enter a number between 1 and {parent_view.num_pages}",
            required=True,
            min_length=1,
            max_length=5
        )
        self.add_item(self.page_input)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            page_num = int(self.page_input.value.strip())
            if page_num < 1 or page_num > self.parent_view.num_pages:
                raise ValueError("Out of range")
            self.parent_view.current_page = page_num - 1
            self.parent_view._build_view()
            await interaction.response.defer()
            await interaction.edit_original_response(embed=self.parent_view._build_embed(), view=self.parent_view)
        except ValueError:
            await interaction.response.send_message(f"Please enter a valid number between 1 and {self.parent_view.num_pages}.", ephemeral=True)
