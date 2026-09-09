from ..utils.constants import *

import discord
from discord import ui
import datetime
import pathlib
import time
import asyncio
from typing import TYPE_CHECKING, List, Dict, Any, Optional
from ..utils.helpers import _estimate_text_tokens, resolve_openrouter_service_tier
from .base_components import (PageJumpModal, SELECT_ALL, SELECT_PAGE, add_button,
                              add_select, build_pagination_controls,
                              bulk_select_options, resolve_bulk_select)
from ..services.generation.compaction import resolve_compaction_settings
from ..services.generation.global_chat import build_global_chat_embed

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

        add_button(self, "Reply", self.reply_callback, style=discord.ButtonStyle.primary, row=0,
                   emoji="✍️")

        if queue:
            if is_typing_locked:
                # Counted, not just "waiting": in a channel the button was the only sign
                # anything was happening, and it did not say how many people it was on.
                writers = len(active_typers)
                play_btn = ui.Button(
                    label=f"{writers} still writing…" if writers else "Waiting…",
                    style=discord.ButtonStyle.secondary, disabled=True, row=0)
            else:
                play_btn = ui.Button(label=f"Play ({len(queue)})", style=discord.ButtonStyle.success, disabled=False, row=0)
        else:
            play_btn = ui.Button(label="Play", style=discord.ButtonStyle.secondary, disabled=True, row=0)

        play_btn.callback = self.play_callback
        self.add_item(play_btn)

        # Labelled, because the lock decides who may press these buttons and a bare
        # padlock read as "this conversation is private" -- which the card is not.
        add_button(self, "Host only" if is_session_locked else "Open to all", self.lock_callback,
                   style=discord.ButtonStyle.danger if is_session_locked else discord.ButtonStyle.success,
                   row=0, emoji="🔒" if is_session_locked else "🔓")

    def get_embed(self):
        return build_global_chat_embed(
            self.cog, self.user_id, self.profile_name,
            self.cog.global_chat_sessions.get(self.session_key, {}))

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

        # The dropdown pins these slots to Google, but rule 1 above would still honour a
        # typed 'OPENROUTER/' here -- and image, speech and grounding all construct a
        # Google client directly, so the id would reach the Google API verbatim and 404.
        # Refused rather than rewritten: 'OPENROUTER/x-ai/grok-4' has no Google meaning,
        # and silently saving 'GOOGLE/x-ai/grok-4' would only move the 404 later.
        if self.target_config_key in GOOGLE_ONLY_MODEL_KEYS:
            if has_explicit_prefix and not value.startswith("GOOGLE/"):
                await interaction.response.send_message(
                    f"`{self.target_config_key}` only accepts Google models — image, "
                    "speech and grounding have no OpenRouter or Ollama path in the "
                    "adapters. Enter the model id without a provider prefix.",
                    ephemeral=True)
                return
            if not has_explicit_prefix:
                value = "GOOGLE/" + value
            has_explicit_prefix = True

        if not has_explicit_prefix:
            prefix = "GOOGLE/"
            if getattr(self.parent_view, 'view_mode', None) == "openrouter":
                prefix = "OPENROUTER/"
            elif getattr(self.parent_view, 'view_mode', None) == "ollama":
                prefix = "OLLAMA/"
            
            value = prefix + value

        self.parent_view._save_changes(self.target_config_key, value)
        self.parent_view._build_view()
        # Once, not twice: the second call raised InteractionResponded on every custom
        # model entry and was swallowed by Modal.on_error as a logged traceback.
        await interaction.response.edit_message(**self.parent_view._picker_render())

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
        
        # A round is every user turn up to the model turn that answered them, not one
        # user turn and one model turn: an unlocked session logs a turn per speaker
        # before the single reply, and the old 1:1 pairing walked straight past all but
        # the last of them, so everybody else's messages vanished from the browser.
        self.rounds = []
        if session_data and 'unified_log' in session_data:
            pending = []
            for turn in session_data['unified_log']:
                if turn.get('role') == 'user':
                    pending.append(turn)
                elif turn.get('role') == 'model' and pending:
                    self.rounds.append((pending, turn))
                    pending = []

        self.current_page = max(0, len(self.rounds) - 1)

    def _build_view(self):
        self.clear_items()
        
        if not self.available_profiles:
            return

        # Row 0: Profile Select
        profile_options = []
        for p in self.available_profiles[:25]: 
            profile_options.append(discord.SelectOption(label=p, value=p, default=(p == self.selected_profile)))
        
        add_select(self, profile_options, self.profile_callback,
                   placeholder="Select a conversation history...", row=0)

        if not self.rounds:
            return

        self.current_page = max(0, min(self.current_page, len(self.rounds) - 1))

        # Row 1: Jump Select
        options = []
        start_jump = max(0, len(self.rounds) - 25)
        for i in range(start_jump, len(self.rounds)):
            user_turns, _ = self.rounds[i]
            first = user_turns[0]
            ts_str = "Unknown"
            if first.get("timestamp"):
                try: ts_str = datetime.datetime.fromisoformat(first.get("timestamp")).strftime('%b %d, %I:%M %p')
                except: pass

            content_preview = first.get("content", "")[:50]
            if len(user_turns) > 1:
                content_preview = f"[{len(user_turns)}] {content_preview}"
            label = f"({ts_str}) {content_preview}..."
            options.append(discord.SelectOption(label=label, value=str(i), default=(i == self.current_page)))
        
        if options:
            add_select(self, options, self.jump_callback, placeholder="Jump to a round...", row=1)

        # Row 2: Buttons
        build_pagination_controls(self, self.current_page, len(self.rounds), 2, self.prev_callback, self.next_callback)
        add_button(self, "Delete", self.delete_callback, style=discord.ButtonStyle.danger, row=2)

    def get_embed(self) -> discord.Embed:
        session_data = self.cog.global_chat_sessions.get(self.session_key, {}) if self.session_key else {}

        if not self.rounds:
            return build_global_chat_embed(
                self.cog, self.user_id, self.selected_profile, session_data,
                description="No conversation history found.", incoming=[],
                footer="Nothing recorded yet.", colour=discord.Colour.dark_grey())

        user_turns, model_turn = self.rounds[self.current_page]
        return build_global_chat_embed(
            self.cog, self.user_id, self.selected_profile, session_data,
            description=model_turn.get("content"), incoming=user_turns,
            footer=f"Round {self.current_page + 1} of {len(self.rounds)}",
            colour=discord.Colour.dark_grey())

    # Reply and Play used to live here too, on a second lock implementation
    # (`lock_expiry`/`lock_resets`) that had drifted from GlobalChatPlayView's
    # (`lock_deadline`/`timer_extensions`), and whose _execute_global_chat call was a
    # positional argument short of the signature. `_build_view` never added the buttons,
    # so none of it was reachable. This view browses history; the card is where you talk.

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

        user_turns, model_turn = self.rounds[self.current_page]
        # Every speaker's turn, not just the last: a round can hold several.
        ids_to_delete = {t.get("turn_id") for t in user_turns}
        ids_to_delete.add(model_turn.get("turn_id"))
        
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
        
        add_select(self, profile_options, self.profile_filter_callback,
                   placeholder="Filter by profile...", row=0)

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
            
            add_select(self, whisper_options[:DROPDOWN_MAX_OPTIONS], self.whisper_jump_callback,
                       placeholder="Jump to a whisper...", row=1)

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

        add_button(self, "Delete", self.delete_callback, style=discord.ButtonStyle.danger, row=2,
                   disabled=not self.filtered_whispers)

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

class SpeakRewriteEditModal(ui.Modal, title="Edit the line"):
    """Edits the *source* line, not the rewrite -- then re-runs it.

    Handing back the generated text to be corrected by hand would quietly turn the
    feature into `/speak` with extra steps: whatever the author typed over would be
    delivered verbatim under the character's face. The thing worth another pass is
    the input.
    """

    def __init__(self, view: 'SpeakPreviewView'):
        super().__init__()
        self.parent_view = view
        self.input_field = ui.TextInput(
            label="What should they get across?",
            style=discord.TextStyle.paragraph,
            default=view.source_text,
            max_length=SPEAK_REWRITE_MAX_INPUT_CHARS,
        )
        self.add_item(self.input_field)

    async def on_submit(self, interaction: discord.Interaction):
        self.parent_view.source_text = self.input_field.value
        # Every stored take answered the line that was just replaced, so switching
        # fidelity must not be able to bring one of them back.
        self.parent_view.by_fidelity.clear()
        await self.parent_view.regenerate(interaction)


def build_speak_placeholder_embed(cog: 'MimicCog', ctx: Dict[str, Any], source_text: str,
                                 fidelity: str, take: int = 1) -> discord.Embed:
    """The waiting state of the preview card, built here so it is the same card.

    `_generate_with_heartbeat` rewrites `description` in place every ten seconds, so
    the emoji is the whole description and everything else is the frame the finished
    take will keep: the author watches one embed fill in rather than a spinner that
    says nothing about whether the model is still alive.
    """
    name, avatar = cog.generation_service._resolve_appearance_data(
        ctx["effective_owner_id"], ctx["effective_profile_name"])
    embed = discord.Embed(
        description=ctx.get("placeholder_emoji") or PLACEHOLDER_EMOJI,
        color=discord.Color.dark_grey())
    embed.set_author(name=name, icon_url=avatar)
    embed.add_field(name="Your line", value=source_text[:1024], inline=False)
    embed.set_footer(text=f"{fidelity.title()} \u00b7 take {take} \u00b7 finding the words")
    return embed


class SpeakPreviewView(ui.View):
    """The approve-before-posting step for `/speak style:in_character`.

    A re-voiced line is a roll of the dice, and without a preview every miss is a
    public miss the author then has to delete. The message this view lives on is
    ephemeral, so only the author ever sees a rejected take.

    Nothing is claimed or held while it is open: the log append still happens in one
    uninterrupted step at Send, exactly as the verbatim path always has, and
    `_deliver_speak_as` re-reads the session rather than trusting the one resolved
    before the preview.

    Reroll is the only button that spends a generation. Switch fidelity keeps the last
    take of each, so the two can be compared and returned to; without that, swapping
    back cost a call and answered with a different line than the one it was swapped
    away from, which made the comparison it exists for impossible.
    """

    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction, ctx: Dict[str, Any],
                 source_text: str, rewritten: str, fidelity: str):
        super().__init__(timeout=300)
        self.cog = cog
        self.original_interaction = interaction
        self.ctx = ctx
        self.source_text = source_text
        self.rewritten = rewritten
        self.fidelity = fidelity
        # `takes` is what has been generated, `shown_take` is which of them is on the
        # card -- they part company the moment a stored take is restored, and only the
        # first is a count of what the model was actually asked for.
        self.takes = 1
        self.shown_take = 1
        # Keyed by fidelity, holding (text, take number). Emptied by the edit modal:
        # every entry is a take on a source line that no longer exists.
        self.by_fidelity = {fidelity: (rewritten, 1)}

    def build_embed(self) -> discord.Embed:
        name, avatar = self.cog.generation_service._resolve_appearance_data(
            self.ctx["effective_owner_id"], self.ctx["effective_profile_name"])
        embed = discord.Embed(description=self.rewritten, color=discord.Color.blurple())
        embed.set_author(name=name, icon_url=avatar)
        embed.add_field(name="Your line", value=self.source_text[:1024], inline=False)
        # "of N" only once a stored take is showing, so the card says which one you are
        # looking at rather than implying the newest is the only one there is.
        counter = f"take {self.shown_take}"
        if self.takes > self.shown_take:
            counter += f" of {self.takes}"
        embed.set_footer(text=f"{self.fidelity.title()} · {counter} · nobody has seen this yet")
        return embed

    def build_placeholder_embed(self) -> discord.Embed:
        return build_speak_placeholder_embed(
            self.cog, self.ctx, self.source_text, self.fidelity, self.takes + 1)

    def _disable(self):
        for child in self.children:
            child.disabled = True

    async def regenerate(self, interaction: discord.Interaction):
        # Swapped to the waiting card, not just greyed out: a reroll is the same model
        # call the first take was, and disabled buttons alone leave the previous take
        # sitting there looking like the answer.
        self._disable()
        await interaction.response.edit_message(
            content=None, embed=self.build_placeholder_embed(), view=self)
        placeholder = await interaction.original_response()

        rewritten, error = await self.cog.generation_service._rewrite_in_character(
            self.ctx, self.source_text, self.fidelity, placeholder)

        for child in self.children:
            child.disabled = False

        if not rewritten:
            await interaction.edit_original_response(
                content=f"That take failed.\n-# {error}", embed=self.build_embed(), view=self)
            return

        self.rewritten = rewritten
        self.takes += 1
        self.shown_take = self.takes
        self.by_fidelity[self.fidelity] = (rewritten, self.takes)
        await interaction.edit_original_response(content=None, embed=self.build_embed(), view=self)

    @ui.button(label="Send", style=discord.ButtonStyle.success)
    async def send_button(self, interaction: discord.Interaction, button: ui.Button):
        self._disable()
        await interaction.response.edit_message(view=self)

        # The button press, not the original /speak: _deliver_speak_as stamps the log
        # turn with `created_at`, and the command may have been sent minutes ago.
        delivered = await self.cog.generation_service._deliver_speak_as(
            interaction, self.ctx, self.rewritten, style='in_character')

        if delivered:
            await interaction.edit_original_response(
                content=f"Sent, after {self.takes} take(s).", embed=None, view=None)
        else:
            await interaction.edit_original_response(
                content="That profile is no longer seated in this channel's session, so nothing was sent.",
                embed=None, view=None)
        self.stop()

    @ui.button(label="Reroll", style=discord.ButtonStyle.primary)
    async def reroll_button(self, interaction: discord.Interaction, button: ui.Button):
        await self.regenerate(interaction)

    @ui.button(label="Edit", style=discord.ButtonStyle.secondary)
    async def edit_button(self, interaction: discord.Interaction, button: ui.Button):
        await interaction.response.send_modal(SpeakRewriteEditModal(self))

    @ui.button(label="Switch fidelity", style=discord.ButtonStyle.secondary)
    async def fidelity_button(self, interaction: discord.Interaction, button: ui.Button):
        self.fidelity = 'loose' if self.fidelity == 'strict' else 'strict'

        stored = self.by_fidelity.get(self.fidelity)
        if stored:
            # A comparison, not a re-roll: the other reading is already paid for, and
            # generating a third one here would throw away the one being compared to.
            self.rewritten, self.shown_take = stored
            await interaction.response.edit_message(
                content=None, embed=self.build_embed(), view=self)
            return

        await self.regenerate(interaction)

    @ui.button(label="Cancel", style=discord.ButtonStyle.danger)
    async def cancel_button(self, interaction: discord.Interaction, button: ui.Button):
        await interaction.response.edit_message(content="Discarded.", embed=None, view=None)
        self.stop()


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
            # Deleting the whisper turns themselves -- recompute rather than clearing
            # is_hydrated, which stranded the in-memory log past eviction.
            self.cog.session_manager._recompute_pending_whispers(session)

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

class CompactionSettingsModal(ui.Modal, title="Rolling Synopsis"):
    threshold_input = ui.TextInput(label=f"Compact after N turns ({COMPACTION_THRESHOLD_MIN}-{COMPACTION_THRESHOLD_MAX})", placeholder=f"Default: {COMPACTION_THRESHOLD_DEFAULT}", required=True, max_length=3)
    chunk_input = ui.TextInput(label="Turns to fold each time", placeholder=f"Default: {COMPACTION_CHUNK_DEFAULT}", required=True, max_length=3)
    model_input = ui.TextInput(label="Summariser Model", placeholder=f"Default: {COMPACTION_MODEL_DEFAULT}", required=False, max_length=100)
    fallback_input = ui.TextInput(label="Fallback Model", placeholder=f"Default: {COMPACTION_FALLBACK_MODEL_DEFAULT}", required=False, max_length=100)

    def __init__(self, view: 'SessionConfigView'):
        super().__init__()
        self.view = view
        cfg = resolve_compaction_settings(view.session)
        self.threshold_input.default = str(cfg["threshold"])
        self.chunk_input.default = str(cfg["chunk"])
        self.model_input.default = cfg["model"]
        self.fallback_input.default = cfg["fallback_model"]

    async def on_submit(self, interaction: discord.Interaction):
        try:
            threshold = int(self.threshold_input.value)
            chunk = int(self.chunk_input.value)
        except ValueError:
            await interaction.response.send_message("❌ Turn counts must be whole numbers.", ephemeral=True)
            return

        cfg = self.view.session.setdefault("compaction", {})
        cfg["threshold"] = threshold
        cfg["chunk"] = chunk
        # Unprefixed ids default to Google, matching the Director model field. Both
        # providers are valid here -- unlike image, speech and grounding, summarisation
        # is an ordinary text slot.
        for field, key, default in (
            (self.model_input, "model", COMPACTION_MODEL_DEFAULT),
            (self.fallback_input, "fallback_model", COMPACTION_FALLBACK_MODEL_DEFAULT),
        ):
            raw = field.value.strip()
            if not raw:
                cfg[key] = default
            elif raw.upper().startswith(("GOOGLE/", "OPENROUTER/", "OLLAMA/")):
                cfg[key] = raw
            else:
                cfg[key] = "GOOGLE/" + raw

        # Re-read through the clamp so the embed shows what will actually run rather
        # than what was typed.
        applied = resolve_compaction_settings(self.view.session)
        cfg["threshold"] = applied["threshold"]
        cfg["chunk"] = applied["chunk"]

        self.view.cog.session_manager._save_multi_profile_sessions()
        await interaction.response.defer()
        await self.view.update_display()


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
            async def prev_cb(i):
                self.current_page -= 1
                self.update_view()
                await i.response.edit_message(view=self)
            add_button(self, "◀", prev_cb, style=discord.ButtonStyle.secondary, row=1,
                       disabled=self.current_page == 0)
            
            async def next_cb(i):
                self.current_page += 1
                self.update_view()
                await i.response.edit_message(view=self)
            add_button(self, "▶", next_cb, style=discord.ButtonStyle.secondary, row=1,
                       disabled=self.current_page == num_pages - 1)

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
        self._seated_cache = None
        self._load_lists()

    #: Cast sources, in the order the Source dropdown lists them. `session` is last
    #: because it is the only one that cannot add: it lists the characters other people
    #: already have in this cast, so an admin can drop one without being able to see --
    #: or seat -- anything else they own.
    SOURCES = ('personal', 'borrowed', 'system', 'child_bot', 'session')

    SOURCE_LABELS = {
        'personal':  ("Personal", "Profiles you own."),
        'borrowed':  ("Borrowed", "Profiles shared with you."),
        'system':    ("System", "Profiles everyone can seat."),
        'child_bot': ("Child Bots", "Your profiles that have their own bot account."),
        'session':   ("In This Session", "Other members' seated characters — removal only."),
    }

    def _viewer_is_admin(self) -> bool:
        """Whether whoever opened this editor is an administrator (or the bot owner).

        Read per render rather than cached at construction: a view lives for ten
        minutes and a role can be taken away inside one. `guild_permissions` is absent
        on a plain User, which is why this is a getattr rather than an attribute read.
        """
        user = self.original_interaction.user
        if int(getattr(user, "id", 0)) == int(defaultConfig.DISCORD_OWNER_ID):
            return True
        perms = getattr(user, "guild_permissions", None)
        return bool(perms and perms.administrator)

    def _available_sources(self) -> tuple:
        """The sources this viewer may switch to.

        `session` is administrators only, and deliberately not tied to `cast_policy`:
        Open Casting says a member may edit the session, not that they may reach into
        other members' cast entries. A non-admin under Open Casting seats and unseats
        their own characters from the four sources that read their own indexes.
        """
        if self._viewer_is_admin():
            return self.SOURCES
        return tuple(s for s in self.SOURCES if s != 'session')

    def _load_lists(self):
        user_id = self.original_interaction.user.id
        index = self.cog.profile_manager._get_user_index(user_id)
        # System profiles are addressable by everyone, so everyone can seat one.
        # _is_system_name applies the same personal-and-borrowed-shadow-System
        # precedence as resolution, so a name offered here means the same profile it
        # will mean when the round runs.
        self.lists = {
            'personal': sorted(list(index.get("personal", []))),
            'borrowed': sorted(list(index.get("borrowed", []))),
            'system': sorted(n for n in self.cog.profile_manager._system_index()
                             if self.cog.profile_manager._is_system_name(user_id, n)),
            'child_bot': sorted([b for b in self.cog.child_bots.values() if b['owner_id'] == user_id], key=lambda x: x.get('profile_name', '')),
            # Derived per render by _session_source_items, not cached here: it is a view
            # of the cast, and the cast changes under this editor while it is open.
            'session': [],
        }

    def _session_source_items(self) -> List[Dict]:
        """Seated participants belonging to somebody other than the viewing admin.

        The one source that does not read an index, and the only one whose contents do
        not change when a different admin opens the editor -- it is a view of
        `session['profiles']`, so it shows the same characters to everyone. That is the
        point: every other source is scoped to `original_interaction.user.id`, which is
        why an admin cannot seat, or even see, another member's profiles. This lists
        only what is already in the cast, so it can remove and never add.

        System profiles are excluded. They sit under DISCORD_OWNER_ID and so read as
        "somebody else's" to every admin who is not the bot owner, but they have their
        own source that seats and unseats them properly.
        """
        viewer_id = int(self.original_interaction.user.id)
        system_owner = int(defaultConfig.DISCORD_OWNER_ID)
        items = []
        for p in self.session.get('profiles', []):
            owner_id = int(p.get('owner_id', 0))
            if owner_id == viewer_id or owner_id == system_owner:
                continue
            items.append(p)
        return sorted(items, key=lambda p: (str(p.get('profile_name') or '').lower(),
                                            int(p.get('owner_id', 0))))

    def _option_value(self, item) -> str:
        """The select-option value for one source entry.

        Lifted out of the two places that each had their own copy -- the option build
        and the callback's deselection test -- because the `session` source made them
        disagree: its entries are participant dicts rather than name strings, and a
        dict cannot be a select value. Identity is `(owner_id, pid)` there, matching
        `participant_identity`, so the value survives a rename like everything else now
        does.
        """
        if self.view_source == 'child_bot':
            return f"child_{self._bot_id_for(item)}"
        if self.view_source == 'session':
            owner_id, pid = self.cog.session_manager.participant_identity(item)
            return f"{owner_id}:{pid}"
        return item

    def _participant_key(self, participant) -> tuple:
        """Identity of a seated character: `(owner_id, pid)`, ignoring `method`.

        A child bot and a webhook are two voices for one profile, not two characters.
        Seating both put the same persona in the cast twice: it read its own lines as
        someone else's, answered itself every round, and wrote two LTM streams from
        one memory. The dropdown's old check qualified on method, so the pair was the
        one duplicate it could not see.

        Keyed on the PID rather than the name, because a name is not an identity --
        it is a label the owner can change. `_rename_profile` hot-swaps `profile_name`
        across live sessions to paper over that, but it does not persist the swap, so
        a restart in that window brought back a cast entry naming a profile that no
        longer exists. The stored PID survives the rename; the name is repaired from
        it on load.

        Delegates to `SessionManager.participant_identity`, which is the one place
        that answers this -- `/session swap` compares seats too, and a second copy of
        the rule is how the two came to disagree about a renamed profile.
        """
        return self.cog.session_manager.participant_identity(participant)

    async def _save_and_repaint(self, i: discord.Interaction):
        self.cog.session_manager._save_multi_profile_sessions()
        await i.response.defer()
        await self.update_display()

    def _add_toggle(self, label: str, enabled: bool, setter, row: int = 0, *,
                    on_off_style: bool = False):
        """A button that flips one session setting and repaints.

        `setter` is handed the new value. `on_off_style` colours the button by the
        state it is in rather than leaving it neutral -- the two standing switches
        (proactivity, the rolling synopsis) read that way; the momentary ones do not.
        """
        async def callback(i: discord.Interaction):
            setter(not enabled)
            await self._save_and_repaint(i)

        style = discord.ButtonStyle.secondary
        if on_off_style:
            style = discord.ButtonStyle.success if enabled else discord.ButtonStyle.danger
        add_button(self, label, callback, style=style, row=row)

    def _add_modal_button(self, label: str, modal_cls, row: int = 0,
                          style: discord.ButtonStyle = discord.ButtonStyle.primary):
        async def callback(i: discord.Interaction):
            await i.response.send_modal(modal_cls(self))
        add_button(self, label, callback, style=style, row=row)

    def _seated_index(self) -> Dict[tuple, Dict]:
        """`_participant_key` -> the seated participant, built once per render.

        `_seated` was a linear scan of the cast, and the cast tab asked it once per
        source entry: two partition passes over a 200-profile list against a 200-seat
        cast came to tens of thousands of `participant_identity` calls before a single
        option was built, on a quarter of a vCPU.

        Held only across one render. `_seat_items` keeps it in step as it appends --
        it has to, because it consults the index to spot duplicates inside its own
        batch -- and `_unseat_items` drops it. Anything else that writes
        `session['profiles']` must drop it too, which is why `update_display` clears
        it after re-reading the live session another admin may have moved.
        """
        if self._seated_cache is None:
            self._seated_cache = {self._participant_key(p): p
                                  for p in self.session.get('profiles', [])}
        return self._seated_cache

    def _seated(self, participant) -> Optional[Dict]:
        """The seated participant matching this one, whichever way it speaks."""
        if not participant:
            return None
        return self._seated_index().get(self._participant_key(participant))

    def _bot_id_for(self, item) -> Optional[str]:
        return next((k for k, v in self.cog.child_bots.items() if v is item), None)

    def _build_participant(self, item) -> Optional[Dict]:
        """The participant dict for one entry of the source currently in view.

        Stamps `pid` here, at the one point that knows which source the entry came
        from and therefore which owner's index resolves it. Everything downstream --
        the duplicate rule, removal, the blueprint -- then compares identities instead
        of spellings, and a participant seated through this editor is shaped like one
        restored from disk rather than missing the field the listing renders.
        """
        if self.view_source == 'session':
            # Already a seated participant -- this source lists the cast, not an index.
            return item

        if self.view_source == 'child_bot':
            bid = self._bot_id_for(item)
            bc = self.cog.child_bots.get(bid) if bid else None
            if not bc:
                return None
            # The child bot's own record carries the profile folder it was launched
            # from, which is the PID; resolving the name again would only agree.
            pid = bc.get('pid') or self.cog.profile_manager._get_pid_from_name(
                int(bc['owner_id']), bc['profile_name'])
            return {"owner_id": bc['owner_id'], "profile_name": bc['profile_name'],
                    "pid": pid, "method": "child_bot", "bot_id": bid,
                    "chance": 100, "wakewords": []}

        # A System profile lives in the bot owner's tree and is seated under the
        # owner's id, matching what /session swap builds for the same name.
        owner_id = (int(defaultConfig.DISCORD_OWNER_ID) if self.view_source == 'system'
                    else self.original_interaction.user.id)
        return {"owner_id": owner_id, "profile_name": item,
                "pid": self.cog.profile_manager._get_pid_from_name(owner_id, item),
                "method": "webhook", "chance": 100, "wakewords": []}

    def _is_selected(self, item) -> bool:
        """Whether this source entry is seated *the way this source seats it*.

        Method-qualified on purpose. The same profile seated the other way reads as
        unselected here, with a description saying so, rather than as a selection
        Unselect Page would then silently remove.
        """
        return self._is_seated_as(self._build_participant(item))

    def _is_seated_as(self, participant) -> bool:
        """`_is_selected` for a caller that has already built the participant."""
        if not participant:
            return False
        existing = self._seated(participant)
        return existing is not None and existing.get('method') == participant.get('method')

    def _seat_items(self, items) -> List[str]:
        """Seat each entry that is not already taken. Returns the ones refused.

        One add path for all three selection gestures -- a single name, Select Page and
        Select All -- so the cast limit and the duplicate rule cannot drift between
        them.
        """
        blocked = []
        index = self._seated_index()
        for item in items:
            if len(self.session['profiles']) >= 200:
                break
            participant = self._build_participant(item)
            if not participant:
                continue
            key = self._participant_key(participant)
            existing = index.get(key)
            if existing is not None:
                # Re-selecting something already seated the same way is a no-op, and
                # silent: Select All hits it on every pass. Only the cross-method
                # collision is worth a word.
                if existing.get('method') != participant.get('method'):
                    blocked.append(participant['profile_name'])
                continue
            self.session['profiles'].append(participant)
            index[key] = participant
        return blocked

    def _unseat_items(self, items) -> None:
        """Unseat each entry that is seated *the way this source seats it*.

        The mirror of `_seat_items`, and it has to share `_is_selected` with it. Both
        removal paths used to filter the cast on a bare `profile_name` drawn from the
        values on the page, and the page is built from the *viewing* admin's own lists
        -- so an entry that merely shared a name with a seated participant removed it.
        Two admins in one channel is the case that bites: the cast is keyed by
        `(owner_id, profile_name)`, so another admin's same-named character was a
        different participant that this filter could not tell apart, and clicking any
        unrelated name silently deleted it. The same hole dropped a webhook-seated
        profile while the child bot source was in view.

        `_is_selected` is already owner-aware and method-qualified -- it is the test
        the labels and option defaults read -- so deselection now means exactly what
        selection means.
        """
        drop = set()
        for item in items:
            if not self._is_selected(item):
                continue
            participant = self._build_participant(item)
            drop.add(self._participant_key(participant))
        if not drop:
            return
        self.session['profiles'] = [
            p for p in self.session['profiles']
            if self._participant_key(p) not in drop]
        self._seated_cache = None

    #: Tabs a non-administrator may open when Open casting let them in here. Casting is
    #: the whole of what Open casting grants: a member seats and unseats characters, and
    #: everything else on this card -- the master prompt, TTS, reactivity, proactivity,
    #: the memory and the session's own access policy -- is the channel's configuration
    #: and stays the administrators'.
    MEMBER_TABS = ("cast",)

    def _add_nav_buttons(self):
        """The tab bar, greyed rather than hidden for a member under Open casting.

        Greyed and labelled with a lock, not omitted: a member who simply cannot see
        the Reactivity tab has no way to learn that the silence they are getting is a
        setting somebody can change for them. It is the same choice `/start` makes for
        the steps it cannot run.
        """
        tabs = ["cast", "config", "reactivity", "proactivity", "memory"]
        is_admin = self._viewer_is_admin()
        for tab in tabs:
            locked = not is_admin and tab not in self.MEMBER_TABS
            btn = ui.Button(label=f"🔒 {tab.title()}" if locked else tab.title(),
                            style=discord.ButtonStyle.primary if self.current_tab == tab else discord.ButtonStyle.secondary,
                            row=4, disabled=locked or (self.current_tab == tab))
            def make_cb(t):
                async def cb(i: discord.Interaction):
                    # `disabled` is a client-side hint and this view outlives a role, so
                    # the gate is re-tested where it is acted on rather than trusted from
                    # when the bar was drawn.
                    if t not in self.MEMBER_TABS and not self._viewer_is_admin():
                        await i.response.send_message(
                            "Only server administrators can change that. Open casting lets "
                            "you seat and unseat characters on the **Cast** tab.",
                            ephemeral=True)
                        return
                    self.current_tab = t; self.current_page = 0
                    self.selected_reactivity_profiles.clear()
                    await i.response.defer(); await self.update_display()
                return cb
            btn.callback = make_cb(tab)
            self.add_item(btn)

    async def update_display(self):
        # Re-read the channel's live session on every render. Two admins can hold this
        # editor open at once -- `active_session_config_views` is keyed by user id and
        # only stops the same user's previous view -- and both mutate the one dict in
        # `multi_profile_channels`. Without this the card keeps drawing the cast as it
        # was when its own owner last clicked, and the next press acts on stale indices.
        live = self.cog.multi_profile_channels.get(self.original_interaction.channel_id)
        if live is not None:
            self.session = live
        self._seated_cache = None

        # An administrator can lose the role with this editor open on a tab a member may
        # not use, so the tab in hand is re-checked on every render the way `view_source`
        # is. Falling back to Cast rather than closing the view keeps whatever they were
        # allowed to do reachable.
        if self.current_tab not in self.MEMBER_TABS and not self._viewer_is_admin():
            self.current_tab = "cast"
            self.current_page = 0
            self.selected_reactivity_profiles.clear()

        self.clear_items()
        self._add_nav_buttons()
        embed = discord.Embed(title=f"Chat Session: #{self.original_interaction.channel.name}", color=discord.Color.gold())

        # One method per tab. This was a 573-line `update_display` with the five
        # bodies inline; the shared prologue and epilogue below are the whole reason
        # they were ever in one method.
        renderer = {
            "cast": self._render_cast,
            "config": self._render_config,
            "reactivity": self._render_reactivity,
            "proactivity": self._render_proactivity,
            "memory": self._render_memory,
        }.get(self.current_tab)
        if renderer:
            renderer(embed)

        self._add_commit_button()

        # On every tab, because the button is on every tab: the one thing the editor
        # must never leave ambiguous is whether the channel is already live.
        state_note = ("● Live in this channel."
                      if self.cog.session_manager.is_started(self.session)
                      else "○ Draft — nothing runs until you press Start / Update Session.")
        if not self._viewer_is_admin():
            state_note += "  ·  Open casting."
        embed.set_footer(text=state_note)

        try:
            await self.original_interaction.edit_original_response(embed=embed, view=self)
        except Exception as e:
            print(f"Error updating SessionConfigView: {e}")

    def _render_cast(self, embed: discord.Embed):
        """The cast dropdown: who is seated, and the three gestures that seat them."""
        # A role can be removed while this view is open, and `view_source` is held
        # across renders, so the source in hand is re-checked rather than trusted.
        if self.view_source not in self._available_sources():
            self.view_source = 'personal'
            self.current_page = 0

        if self.view_source == 'session':
            # Derived, not cached: the cast moves under an open editor.
            self.lists['session'] = self._session_source_items()
            embed.description = ("Characters **other members** already have in this cast. "
                                 "Deselecting removes one; nothing here can be added, and no "
                                 "profile of theirs that is not already seated is listed.")
        else:
            embed.description = ("Add or remove participants (200 max). Selecting seats a profile "
                                 "and saves it; **Start / Update Session** is what makes the channel live.")
        active_list = self.lists[self.view_source]

        # `(item, participant)` pairs, built once and carried through to the option
        # loop. Seated entries first, so a long source list opens on the cast.
        selected_pairs, unselected_pairs = [], []
        for item in active_list:
            participant = self._build_participant(item)
            (selected_pairs if self._is_seated_as(participant)
             else unselected_pairs).append((item, participant))

        ordered_pairs = selected_pairs + unselected_pairs
        ordered_list = [item for item, _ in ordered_pairs]

        num_pages = max(1, (len(ordered_list) - 1) // 20 + 1)
        if self.current_page >= num_pages:
            self.current_page = num_pages - 1

        start = self.current_page * 20
        page_pairs = ordered_pairs[start : start + 20]
        page_items = [item for item, _ in page_pairs]

        options = []
        if page_items:
            # Everything in the `session` source is seated by definition, so these
            # only ever read as the removing half. Saying so beats a generic
            # "toggle" on the one source where the toggle has one direction.
            is_session_src = self.view_source == 'session'

            options = bulk_select_options(
                page_pairs, ordered_pairs, lambda pair: self._is_seated_as(pair[1]),
                page_description=("Remove every character on this page from the cast."
                                  if is_session_src
                                  else "Toggle selection for all profiles on this page."),
                all_description=("Remove every other member's character from the cast."
                                 if is_session_src
                                 else "Toggle selection for all profiles in this source."))

            for item, participant in page_pairs:
                val = self._option_value(item)
                name = item.get('profile_name') if isinstance(item, dict) else item
                # `Name [PID]`. Two members can seat characters sharing a name, and
                # the same name can mean different profiles across sources, so the
                # PID is what tells the rows apart -- and it is the identity the
                # cast is now keyed on, so what is shown is what is compared.
                pid = (participant or {}).get('pid')
                lbl = f"{name} [{pid}]" if pid else name

                existing = self._seated(participant)
                is_sel = self._is_seated_as(participant)

                # Seated the other way round: shown, not hidden, so the clash is
                # visible where it is made rather than only after it is refused.
                desc = None
                if self.view_source == 'session':
                    # Whose character this is. get_user misses for anyone the
                    # gateway has not cached, so the id is the fallback -- it is
                    # still enough to tell two same-named characters apart.
                    owner = self.cog.bot.get_user(int(item.get('owner_id', 0)))
                    who = owner.display_name if owner else f"user {item.get('owner_id')}"
                    method_lbl = ("child bot" if item.get('method') == 'child_bot'
                                  else "webhook")
                    desc = f"{who} — speaks as a {method_lbl}. Deselect to remove."
                elif existing is not None and not is_sel:
                    desc = ("Already seated as a child bot." if existing.get('method') == 'child_bot'
                            else "Already seated as a webhook.")

                options.append(discord.SelectOption(label=lbl[:100], value=val[:100],
                                                    description=desc, default=is_sel))
        else:
            options.append(discord.SelectOption(
                label=("No other members have characters here"
                       if self.view_source == 'session' else "No profiles found"),
                value="none"))

        placeholder = ("Deselect to remove another member's character..."
                       if self.view_source == 'session'
                       else f"Select {self.view_source.replace('_', ' ')} profiles...")
        async def cast_cb(i: discord.Interaction):
            if "none" in i.data['values']: await i.response.defer(); return
            # Built during the render this click came from, and another admin may
            # have seated something since. The index is only ever valid for one
            # synchronous stretch.
            self._seated_cache = None
            raw_vals = i.data['values']
            curr_vals = set(raw_vals)

            _value_of = self._option_value

            def _all_seated(scope_items):
                return all(self._is_selected(item) for item in scope_items)

            blocked = []
            if SELECT_PAGE in curr_vals:
                if _all_seated(page_items):
                    self._unseat_items(page_items)
                else:
                    blocked = self._seat_items(page_items)

            elif SELECT_ALL in curr_vals:
                if _all_seated(ordered_list):
                    self._unseat_items(ordered_list)
                else:
                    blocked = self._seat_items(ordered_list)

            else:
                self._unseat_items([item for item in page_items
                                    if _value_of(item) not in curr_vals])

                by_value = {_value_of(item): item for item in page_items}
                blocked = self._seat_items([by_value[v] for v in raw_vals if v in by_value])

            self.cog.session_manager._save_multi_profile_sessions()

            # Only a live session needs telling. Seating a cast into a draft
            # announces nothing -- Start / Update Session does that for the whole
            # cast at once.
            if self.cog.session_manager.is_started(self.session):
                for p_data in self.session.get('profiles', []):
                    if p_data.get('method') == 'child_bot':
                        await self.cog.manager_queue.put({
                            "action": "send_to_child", "bot_id": p_data['bot_id'],
                            "payload": {"action": "session_update_add", "channel_id": self.original_interaction.channel_id}
                        })

            await i.response.defer(); await self.update_display()
            if blocked:
                names = ", ".join(f"`{n}`" for n in blocked[:5])
                more = f" (+{len(blocked) - 5})" if len(blocked) > 5 else ""
                await i.followup.send(
                    f"Skipped {names}{more} — already in the cast the other way "
                    f"(a profile speaks as a child bot **or** a webhook, not both).",
                    ephemeral=True)
        add_select(self, options, cast_cb, placeholder=placeholder, min_values=0,
                   max_values=len(options) if page_items else 1, row=0,
                   disabled=not page_items)

        # Row 1, directly under the profile dropdown. A select rather than the
        # button that cycled: five sources meant up to four presses to reach one,
        # with no way to see what the others were, and `session` has to be able to
        # disappear for a non-admin rather than be cycled past.
        source_opts = self._available_sources()
        src_sel = ui.Select(
            placeholder=f"Source: {self.SOURCE_LABELS[self.view_source][0]}",
            min_values=1, max_values=1, row=1,
            options=[discord.SelectOption(
                label=self.SOURCE_LABELS[src][0], value=src,
                description=self.SOURCE_LABELS[src][1],
                default=(src == self.view_source)) for src in source_opts])

        async def src_cb(i: discord.Interaction):
            chosen = i.data['values'][0]
            # Re-tested on submit, not just at render: a view outlives a role.
            if chosen not in self._available_sources():
                await i.response.defer()
                await self.update_display()
                return
            self.view_source = chosen
            self.current_page = 0
            await i.response.defer(); await self.update_display()
        src_sel.callback = src_cb
        self.add_item(src_sel)

        if num_pages > 1:
            async def p_cb(i): self.current_page -= 1; await i.response.defer(); await self.update_display()
            async def n_cb(i): self.current_page += 1; await i.response.defer(); await self.update_display()
            build_pagination_controls(self, self.current_page, num_pages, 2, p_cb, n_cb)

        # Row 2 now holds at most three pagination controls, so there is room for
        # this beside them. Only rendered when the cast is non-empty -- the dropdown
        # sentinels can only clear the source currently in view, so with a mixed
        # cast there is otherwise no single gesture that empties it.
        #
        # Omitted rather than greyed for a member under Open casting, which is the
        # opposite of the tab bar's choice and for the opposite reason: the tab bar
        # advertises something they can ask an administrator for, whereas this is
        # the one control on the Cast tab that reaches past their own characters and
        # empties everybody's. There is nothing to learn from seeing it, and the
        # dropdown still clears what is theirs.
        if self.session.get('profiles') and self._viewer_is_admin():
            async def clear_cast_cb(i: discord.Interaction):
                # Not drawn for a member, but this view lives ten minutes and a role
                # can go inside one -- so, like the tab bar, it is re-tested here.
                if not self._viewer_is_admin():
                    await i.response.send_message(
                        "Only server administrators can clear the whole cast. You can "
                        "still unseat your own characters from the dropdown.",
                        ephemeral=True)
                    return
                removed = list(self.session.get('profiles', []))
                self.session['profiles'] = []
                self.cog.session_manager._save_multi_profile_sessions()

                # Child bots hold their own per-channel session state, so dropping
                # them from the list is not enough -- without this they keep the
                # channel registered and go on showing a typing indicator for a
                # session they are no longer part of.
                for p_data in removed:
                    if p_data.get('method') != 'child_bot':
                        continue
                    bot_id = p_data.get('bot_id')
                    if not bot_id:
                        continue
                    await self.cog.manager_queue.put({
                        "action": "send_to_child", "bot_id": bot_id,
                        "payload": {"action": "session_update_remove",
                                    "channel_id": self.original_interaction.channel_id}
                    })
                    await self.cog.manager_queue.put({
                        "action": "send_to_child", "bot_id": bot_id,
                        "payload": {"action": "stop_typing",
                                    "channel_id": self.original_interaction.channel_id}
                    })

                self.current_page = 0
                await i.response.defer()
                await self.update_display()
            add_button(self, f"Clear Cast ({len(self.session['profiles'])})", clear_cast_cb,
                       style=discord.ButtonStyle.secondary, row=2)

        profiles = self.session.get('profiles', [])
        total_active = len(profiles)
        
        def _cast_line(idx, p):
            method_lbl = 'Child Bot' if p.get('method') == 'child_bot' else 'Webhook'
            pid = p.get('pid')
            pid_str = f" `[{pid}]`" if pid else ""
            return f"{idx}. `{p['profile_name']}`{pid_str} ({method_lbl})"

        if total_active <= 20:
            cast_list = "\n".join(_cast_line(idx + 1, p)
                                  for idx, p in enumerate(profiles)) or "*No participants*"
        else:
            start_p = self.current_page * 20
            end_p = start_p + 20
            page_profiles_cast = profiles[start_p:end_p]
            
            cast_list = "\n".join(_cast_line(idx, p) for idx, p in
                                  enumerate(page_profiles_cast, start=start_p + 1)) or "*No participants*"
            
        embed.add_field(name="Current Cast", value=cast_list, inline=False)

    def _render_config(self, embed: discord.Embed):
        """Session-wide behaviour: execution mode, master prompt, TTS, response limit, cast access."""
        embed.description = "Configure session-wide behavior."
        mp = self.session.get("session_prompt")
        audio_val = self.session.get("audio_mode", "off")
        tts_status = "**`ON`**" if audio_val == "on" else "`OFF`"
        response_limit = self.session.get("max_responses", 10)
        policy = self.session.get("cast_policy", DEFAULT_CAST_POLICY)

        embed.add_field(name="Execution Mode", value=f"`{self.session.get('session_mode', 'sequential').title()}`", inline=True)
        embed.add_field(name="Master Prompt", value=f"`{'Set' if mp else 'Not Set'}`", inline=True)
        embed.add_field(name="Cast Access", value=f"`{CAST_POLICY_LABELS.get(policy, policy)}`", inline=True)

        embed.add_field(name="Text-to-Speech", value=f"{tts_status}", inline=True)
        embed.add_field(name="Response Limit", value=f"`{response_limit}`", inline=True)
        embed.add_field(name="\u200b", value="\u200b", inline=True)
        
        embed.add_field(name="Master Prompt Content", value=f"```{mp[:500]}```" if mp else "`None`", inline=False)

        self._add_toggle(
            "Toggle Execution",
            self.session.get("session_mode", "sequential") == "random",
            lambda on: self.session.__setitem__("session_mode",
                                                "random" if on else "sequential"))
        self._add_modal_button("Edit Master Prompt", SessionPromptModal)
        self._add_toggle(
            "Toggle TTS", audio_val == "on",
            lambda on: self.session.__setitem__("audio_mode", "on" if on else "off"))
        self._add_modal_button("Set Response Limit", ResponseLimitModal)

        # Row 1, vacated by the two buttons above. A select needs a row to itself,
        # and the four config buttons fit one row with a slot to spare.
        #
        # Administrators only, and rendered disabled rather than hidden for everyone
        # else so a member under Open Casting can see the terms they are editing
        # under. It is the control that grants the access: a member who could set it
        # could lock the administrators' own setting back out from under them, and
        # could keep a channel open that an admin had closed.
        viewer_is_admin = self._viewer_is_admin()
        policy_sel = ui.Select(
            placeholder=f"Cast access: {CAST_POLICY_LABELS.get(policy, policy)}"
                        + ("" if viewer_is_admin else " (administrators only)"),
            min_values=1, max_values=1, row=1, disabled=not viewer_is_admin,
            options=[discord.SelectOption(label=lbl, value=val, description=desc,
                                          emoji=emoji, default=(val == policy))
                     for val, lbl, desc, emoji in CAST_POLICIES])

        async def policy_cb(i: discord.Interaction):
            # Re-tested on submit: `disabled` is a client-side hint, and this view
            # outlives the role that rendered it.
            if not self._viewer_is_admin():
                await i.response.send_message(
                    "Only a server administrator can change who may edit this session.",
                    ephemeral=True)
                return
            self.session["cast_policy"] = i.data['values'][0]
            await self._save_and_repaint(i)
        policy_sel.callback = policy_cb
        self.add_item(policy_sel)

    def _render_reactivity(self, embed: discord.Embed):
        """Per-participant response chance and wakewords."""
        embed.description = "Manage how likely participants are to respond to messages."
        profiles = self.session.get('profiles', [])
        total_items = len(profiles)
        num_pages = max(1, (total_items - 1) // 20 + 1)
        
        if self.current_page >= num_pages:
            self.current_page = max(0, num_pages - 1)
            
        start = self.current_page * 20
        page_profiles = profiles[start : start + 20]
        
        # `owner:name` is the option value; `(owner_id, profile_name)` is what the
        # selection is held as. Mapped rather than parsed back, so a name is never
        # split on a colon it happens to contain.
        def _value_of(p):
            return f"{p['owner_id']}:{p['profile_name']}"

        by_value = {_value_of(p): (p['owner_id'], p['profile_name']) for p in profiles}
        page_values = [_value_of(p) for p in page_profiles]

        options = []
        if page_profiles:
            options = bulk_select_options(
                page_profiles, profiles,
                lambda p: (p['owner_id'], p['profile_name']) in self.selected_reactivity_profiles,
                all_description="Toggle selection for all profiles.")

            for p in page_profiles:
                p_key = (p['owner_id'], p['profile_name'])
                is_checked = p_key in self.selected_reactivity_profiles
                options.append(discord.SelectOption(
                    label=f"{p['profile_name']} (Chance: {p.get('chance', 100)}%)",
                    value=_value_of(p),
                    default=is_checked
                ))
        else:
            options.append(discord.SelectOption(label="No profiles found", value="none"))

        async def react_select_cb(i: discord.Interaction):
            if "none" in i.data['values']: await i.response.defer(); return
            chosen = resolve_bulk_select(
                i.data['values'], page_values, list(by_value),
                {v for v, k in by_value.items()
                 if k in self.selected_reactivity_profiles})
            self.selected_reactivity_profiles = {by_value[v] for v in chosen
                                                 if v in by_value}

            await i.response.defer()
            await self.update_display()
        add_select(self, options, react_select_cb,
                   placeholder="Select participant(s) to edit...", min_values=0,
                   max_values=len(options) if page_profiles else 1, row=0,
                   disabled=not page_profiles)

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
        add_button(self, btn_label, edit_callback, style=btn_style, row=2,
                   disabled=btn_disabled)

    def _render_proactivity(self, embed: discord.Embed):
        """The timer that lets the cast open a conversation unprompted."""
        pro = self.session.get("proactivity", {})
        enabled = pro.get("enabled", False)
        embed.description = "Allow the session to start conversations autonomously based on a timer."
        embed.add_field(name="Status", value="**`ON`**" if enabled else "`OFF`", inline=True)
        embed.add_field(name="Chance & Cooldown", value=f"`{pro.get('chance', 10)}%` every `{pro.get('cooldown', 300)}s`", inline=True)
        
        dir_mod = pro.get("director_model", "GOOGLE/gemini-2.5-flash-lite")
        dir_ins = pro.get("director_instructions", "(Default)") or "(Default)"
        embed.add_field(name="AI Director", value=f"Model: `{dir_mod}`\nInstructions: ```{dir_ins[:200]}```", inline=False)

        self._add_toggle(
            "Toggle Proactivity", enabled,
            lambda on: self.session.setdefault("proactivity", {}).__setitem__("enabled", on),
            on_off_style=True)
        self._add_modal_button("Edit Settings & AI Director", ProactivitySettingsModal)

    def _render_memory(self, embed: discord.Embed):
        """The rolling synopsis: how a long scene is folded down."""
        cfg = resolve_compaction_settings(self.session)
        enabled = cfg["enabled"]
        embed.description = (
            "Fold the oldest part of a long conversation into a running synopsis, so the "
            "cast keeps the thread of a scene after it scrolls past their Short-Term Memory.\n\n"
            "Folded turns are hidden from prompts, not deleted \u2014 the transcript, "
            "regeneration and the audit view are unaffected, and turning this off brings "
            "them back. Private whispers are never summarised."
        )
        embed.add_field(name="Status", value="**`ON`**" if enabled else "`OFF`", inline=True)
        embed.add_field(name="Trigger", value=f"Every `{cfg['threshold']}` turns", inline=True)
        embed.add_field(name="Fold Size", value=f"`{cfg['chunk']}` turns", inline=True)
        embed.add_field(
            name="Summariser",
            value=f"`{cfg['model']}`\nFallback: `{cfg['fallback_model']}`",
            inline=False,
        )

        log = self.session.get("unified_log") or []
        visible = sum(
            1 for t in log
            if not t.get("type") and not t.get("compacted") and not t.get("is_hidden")
        )
        folded = sum(1 for t in log if t.get("compacted"))
        synopses = sum(1 for t in log if t.get("type") == "synopsis")
        embed.add_field(
            name="This Session",
            value=(
                f"`{visible}` live turn(s) \u2022 `{folded}` folded \u2022 "
                f"`{synopses}` synopsis block(s)"
            ),
            inline=False,
        )
        if synopses:
            latest = next(
                (t.get("content", "") for t in reversed(log) if t.get("type") == "synopsis"), ""
            )
            if latest:
                embed.add_field(name="Latest Synopsis", value=f"```{latest[:900]}```", inline=False)

        self._add_toggle(
            "Toggle Rolling Synopsis", enabled,
            lambda on: self.session.setdefault("compaction", {}).__setitem__("enabled", on),
            on_off_style=True)
        self._add_modal_button("Edit Settings", CompactionSettingsModal)


    def _add_commit_button(self):
        """The Start / Update Session button, on every tab.

        Seating and starting are two different acts. The cast dropdown applies and
        saves the moment a name is chosen -- it has to, because reactivity, wakewords
        and per-participant chance all read `session['profiles']` and would otherwise
        have nobody to edit -- but a seated cast is a draft. Nothing in the channel
        runs until this is pressed; `SessionManager.is_started` is the gate every
        trigger path asks.

        What it does that nothing else does:

        * Marks the session started, which is the whole point.
        * Persists the blueprint. A session woken as a shell by `_ensure_session_shell`
          is in memory only until something saves it, and an unstarted empty shell is
          deliberately never written at all.
        * Hydrates a dehydrated session, so the transcript is loaded before the first
          message rather than during it.
        * Tells **every** child bot in the cast that it is in this channel. The cast
          dropdown only announces on a live session, so a bot that arrived with a
          restored blueprint, or through `/session swap`, was never told.

        An empty cast is allowed. A started session with nobody in it is a valid
        state -- the channel keeps its transcript and its settings, and simply has no
        one to answer.
        """
        started = self.cog.session_manager.is_started(self.session)
        async def commit(i: discord.Interaction):
            await i.response.defer()

            was_started = self.cog.session_manager.is_started(self.session)
            self.session["started"] = True
            self.cog.session_manager._save_multi_profile_sessions()

            if not self.session.get("is_hydrated"):
                hydrated = await self.cog.session_manager._ensure_session_hydrated(
                    self.original_interaction.channel_id, self.session.get("type", "multi"))
                if hydrated:
                    hydrated["started"] = True
                    self.session = hydrated

            channel_id = self.original_interaction.channel_id
            for p in self.session.get("profiles", []):
                if p.get("method") != "child_bot" or not p.get("bot_id"):
                    continue
                await self.cog.manager_queue.put({
                    "action": "send_to_child", "bot_id": p["bot_id"],
                    "payload": {"action": "session_update_add", "channel_id": channel_id}})

            await self.update_display()

            count = len(self.session.get("profiles", []))
            if not count:
                note = "Session is live, but the cast is empty — nobody will answer."
            else:
                note = f"Session is live with {count} participant(s)."
            await i.followup.send(("▶️ " if not was_started else "🔄 ") + note, ephemeral=True)
        add_button(self, "▶️ Update Session" if started else "▶️ Start Session", commit,
                   style=discord.ButtonStyle.success, row=3)


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
        # Was a hand-copied duplicate of SYSTEM_XML_TAGS that had to be edited in
        # lockstep with constants.py; a tag added there but missed here leaked into
        # this preview. Use the one list.
        tags_pattern = "|".join(SYSTEM_XML_TAGS)
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
            self._add_paged_select(
                self.all_turns, "Select a turn to inspect...", "selected_turn_id",
                self._turn_option, nav_suffix=" of turns", always_show_jump=True)

        elif self.mode == "simulator":
            self._add_paged_select(
                self.session.get("profiles", []) or [],
                "Select a profile to simulate next turn...", "simulate_profile_key",
                lambda p, _idx, current: discord.SelectOption(
                    label=p['profile_name'][:100],
                    value=f"{p['owner_id']}:{p['profile_name']}",
                    default=(f"{p['owner_id']}:{p['profile_name']}" == current)),
                nav_suffix=" of profiles")

        elif self.mode == "batch":
            self.all_turns = self.session.get("unified_log", []) or []
            # Two selects over one shared page cursor: turning the page on either moves
            # both, which is why they are built from the same item list and page state.
            self._add_paged_select(self.all_turns, "Select Start Turn...", "batch_start_id",
                                   self._turn_option, row=1)
            self._add_paged_select(self.all_turns, "Select End Turn...", "batch_end_id",
                                   self._turn_option, row=2)

    def _turn_option(self, t: dict, abs_index: int, current) -> discord.SelectOption:
        display = self._resolve_turn_speaker_name(t)
        preview = self._extract_turn_preview(t)
        label = f"Turn #{abs_index + 1} - {display} ({preview})"
        return discord.SelectOption(label=label[:100], value=t.get("turn_id"),
                                    default=(t.get("turn_id") == current))

    def _add_paged_select(self, items, placeholder, attr, option_for, *, row=1,
                          per_page=20, nav_suffix="", always_show_jump=False):
        """Attach one paginated dropdown, page controls included.

        Written once and called four times. Each of the four used to carry its own copy
        of the page-option construction and a 16-line prev/next/jump/select callback,
        differing only in the item list, the attribute the choice lands in, and the
        wording of the nav descriptions.

        `attr` is the name of the view attribute the selection is stored on; page state
        is shared across every select on the view, so the two batch dropdowns turn
        together exactly as they did before.

        `always_show_jump` preserves an inconsistency in the original: the inspector
        offered the jump row even with a single page, while the other three hid it.
        """
        self.num_pages = max(1, (len(items) - 1) // per_page + 1)
        self.current_page = max(0, min(self.current_page, self.num_pages - 1))

        start = self.current_page * per_page
        page_items = items[start : start + per_page]

        opts = []
        if self.current_page > 0:
            opts.append(discord.SelectOption(label="◀ Previous Page", value="prev_page",
                                             description=f"Navigate to previous page{nav_suffix}"))
        if always_show_jump or self.num_pages > 1:
            opts.append(discord.SelectOption(label=f"📄 Page {self.current_page + 1}/{self.num_pages} (Jump)",
                                             value="jump_page", description="Click to jump to a page number"))
        if self.current_page < self.num_pages - 1:
            opts.append(discord.SelectOption(label="▶ Next Page", value="next_page",
                                             description=f"Navigate to next page{nav_suffix}"))

        current = getattr(self, attr)
        for idx, item in enumerate(page_items):
            opts.append(option_for(item, start + idx, current))

        if not opts:
            return

        async def sel_cb(i: discord.Interaction):
            val = i.data['values'][0]
            if val == "jump_page":
                await i.response.send_modal(self._page_jump_modal())
                return
            if val == "prev_page":
                self.current_page -= 1
            elif val == "next_page":
                self.current_page += 1
            else:
                setattr(self, attr, val)
            self._build_view()
            await i.response.edit_message(embed=self._build_embed(), view=self)
        add_select(self, opts, sel_cb, placeholder=placeholder, row=row)
    def _page_jump_modal(self) -> PageJumpModal:
        """Built here rather than at each of the four buttons that send it."""
        async def _jump(i: discord.Interaction, page: int):
            self.current_page = page
            self._build_view()
            await i.response.defer()
            await i.edit_original_response(embed=self._build_embed(), view=self)

        return PageJumpModal(self.num_pages, _jump, zero_indexed=True)

    #: How the critic reached its verdict, for the inspector's second line. "cache" is a
    #: constraint carried over by `critic_persistence` from an earlier round, which is
    #: worth showing plainly: it is the one source that costs nothing *and* did not
    #: re-examine this turn's history.
    _CRITIC_SOURCE_LABELS = {
        "lexical": "Local lexical scan (no API call)",
        "model": "Critic model pass",
        "cache": "Carried over from an earlier round",
    }

    def _critic_field(self, meta: dict) -> Optional[str]:
        """The critic's verdict for one turn, or None if it was not running.

        Absent key means the profile had the critic off for that turn, or the turn
        predates the audit record -- both render as nothing rather than as an empty
        verdict, which is the same contract `neuro_state` has.
        """
        critic = meta.get("critic")
        if not isinstance(critic, dict):
            return None

        mode = critic.get("mode", "unknown")
        scope = critic.get("scope", "self")
        strictness = critic.get("strictness", "normal")
        lookback = critic.get("lookback", 0)
        source = critic.get("source")
        text = (critic.get("text") or "").strip()

        lines = [
            f"├── Configuration: `{mode}` / scope `{scope}` / `{strictness}` "
            f"over `{lookback}` turns",
            f"├── Verdict: `{self._CRITIC_SOURCE_LABELS.get(source, 'Pass -- no repetition found')}`",
        ]
        if text:
            # The constraint is already capped at CRITIC_AUDIT_TEXT_MAX on write. The
            # trim here is for the field as a whole: the two lines above eat into the
            # same 1024 characters Discord allows.
            head = "\n".join(lines) + "\n└── Constraint injected:\n"
            room = 1024 - len(head) - 10
            body = text if len(text) <= room else text[:max(0, room - 3)] + "..."
            return f"{head}```\n{body}\n```"
        return "\n".join(lines[:-1] + [lines[-1].replace("├──", "└──")])

    def _build_embed(self) -> discord.Embed:
        try:
            embed = discord.Embed(title=f"Chat Session Diagnostic: #{self.original_interaction.channel.name}", color=discord.Color.brand_green())
            log = self.session.get("unified_log", [])

            if self.mode == "overview":
                total_in = 0
                total_out = 0
                total_cost = 0.0
                estimated = False
                durations = []
                
                for t in log:
                    if not t.get("is_user"):
                        meta = t.get("meta") or {}
                        i_toks = meta.get("input_tokens", 0)
                        o_toks = meta.get("output_tokens", 0)
                        total_in += i_toks
                        total_out += o_toks
                        cost, billed = self.cog.api_service.turn_cost(meta)
                        total_cost += cost
                        estimated = estimated or not billed
                        dur = meta.get("duration")
                        if dur: durations.append(dur)
                
                embed.description = "Session Overview"
                cost_line = (f"Estimated Session API Cost: `~${total_cost:.4f} USD`" if estimated
                             else f"Session API Cost (billed): `${total_cost:.4f} USD`")
                embed.add_field(name="1. Overall Session Telemetry", value=f"├── Active Participants: `{len(self.session.get('profiles', []))}` Profiles\n├── Total Session Turns: `{len(log)}`\n├── Total Input Tokens Processed: `{total_in:,}`\n├── Total Output Tokens Generated: `{total_out:,}`\n└── {cost_line}", inline=False)
                
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
                    cost, billed = self.cog.api_service.turn_cost(meta)
                    # The served tier, not the requested one: OpenRouter routes a model
                    # with no endpoint at the asked-for tier normally, so the request is
                    # not evidence of what ran.
                    tier = meta.get("service_tier")
                    tier_line = f"├── Service Tier: `{tier}`\n" if tier else ""
                    cost_line = (f"Turn Cost: `~${cost:.6f} USD` (estimated)" if not billed
                                 else f"Turn Cost: `${cost:.6f} USD` (billed)")
                    
                    embed.add_field(name="Turn Telemetry", value=f"├── Speaker: `{target.get('profile_name')}`\n├── Model Used: `{mod}`\n{tier_line}├── Duration: `{meta.get('duration', 0.0)}s`\n├── Input Tokens: `{i_tok:,}`\n├── Output Tokens: `{o_tok:,}` (Reasoning: `{r_tok:,}`)\n└── {cost_line}", inline=False)
                    
                    recalled = len(meta.get("ltms_recalled", []))
                    trained = meta.get("training_recalled", 0)
                    grounded = len(meta.get("grounding_sources", []))
                    
                    embed.add_field(name="Context Injections", value=f"├── LTM Archive: `{recalled} memories`\n├── Training Examples: `{trained} injected`\n└── Web Grounding: `{grounded} sources`", inline=False)

                    critic_field = self._critic_field(meta)
                    if critic_field:
                        embed.add_field(name="Anti-Repetition Critic", value=critic_field, inline=False)

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
                        # The projection is the rate table, which only knows standard
                        # rates for a listed model id. Say so when the profile has asked
                        # for a tier, rather than quoting a number the turn will not cost.
                        tier = resolve_openrouter_service_tier(p_cfg)
                        tier_note = (f"\n*(Priced at standard rates; this profile requests the "
                                     f"`{tier}` tier, so the billed figure will differ.)*") if tier else ""
                        embed.add_field(name="Financial Projection", value=f"Projected cost for next generation: `~${est_cost:.6f} USD`\n*(Assuming ~300 output tokens)*{tier_note}", inline=False)
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
                        estimated = False
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
                                cost, billed = self.cog.api_service.turn_cost(meta)
                                total_cost += cost
                                estimated = estimated or not billed
                                
                                models_used[m] = models_used.get(m, 0) + 1
                                bot_turns += 1
                                
                        dist_str = "\n".join([f"├── {m}: `{c}/{bot_turns} turns`" for m, c in models_used.items()])
                        if not dist_str: dist_str = "├── No bot generations in range."
                        
                        range_cost = (f"Combined Range Cost: `~${total_cost:.6f} USD`" if estimated
                                      else f"Combined Range Cost (billed): `${total_cost:.6f} USD`")
                        embed.add_field(name="Batch Execution Totals", value=f"├── Turns Evaluated: `{len(batch_turns)}`\n├── Cumulative Input Tokens: `{total_in:,}`\n├── Cumulative Output Tokens: `{total_out:,}`\n└── {range_cost}", inline=False)
                        embed.add_field(name="Model Distribution", value=dist_str, inline=False)

            return embed
        except Exception as e:
            import traceback
            err_trace = traceback.format_exc()
            print(f"Error building audit embed: {err_trace}")
            err_embed = discord.Embed(title="Audit Error", description=f"An error occurred while building the telemetry report:\n```\n{e}\n```", color=discord.Color.red())
            return err_embed

