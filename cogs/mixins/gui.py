from .constants import *
from .storage import _quantize_embedding

import discord
from discord import ui, app_commands
import datetime
import uuid
import io
import re
import pathlib
import traceback
import time
import copy
from zoneinfo import ZoneInfo
from collections import OrderedDict
import asyncio
import orjson as json
from typing import TYPE_CHECKING, List, Dict, Tuple, Set, Literal, Any, Optional, Union, get_args
from .storage import (
    _quantize_embedding, 
)
from .services import GoogleGenAIChatSession

if TYPE_CHECKING:
    # This only runs during "hinting" and prevents the circular crash
    from ..GeminiCog import GeminiAgent

class CustomModelModal(ui.Modal, title="Enter Custom Model ID"):
    model_id_input = ui.TextInput(label="Model ID", placeholder="e.g. anthropic/claude-3", required=True)

    def __init__(self, view: 'ModelApplyView', select_type: str):
        super().__init__()
        self.parent_view = view
        self.select_type = select_type

    async def on_submit(self, interaction: discord.Interaction):
        value = self.model_id_input.value.strip()
        
        # Remove potential system display prefixes if user typed them manually (Case-Sensitive)
        if value.startswith("GOOGLE/"):
            value = value[7:]
        elif value.startswith("OPENROUTER/"):
            value = value[11:]
        
        # Determine prefix based on mode
        prefix = "OPENROUTER/" if self.parent_view.view_mode == "openrouter" else "GOOGLE/"
        
        # Always prepend the prefix
        value = prefix + value
        
        if self.select_type == 'primary':
            self.parent_view.primary_model = value
        else:
            self.parent_view.fallback_model = value
        
        self.parent_view._build_view()
        await interaction.response.edit_message(content=self.parent_view._get_selection_feedback_message(), view=self.parent_view)

class GlobalChatHistoryView(ui.View):
    def __init__(self, cog: 'GeminiAgent', interaction: discord.Interaction, user_id: int, initial_profile: Optional[str] = None):
        super().__init__(timeout=600)
        self.cog = cog
        self.original_interaction = interaction
        self.user_id = user_id
        self.user_id_str = str(user_id)
        
        self.available_profiles = self._scan_profiles()
        self.selected_profile = initial_profile if initial_profile in self.available_profiles else (self.available_profiles[0] if self.available_profiles else None)
        
        self.rounds = [] # List of tuples: (user_turn_dict, model_turn_dict)
        self.session_key = None
        self.current_page = 0
        
        if self.selected_profile:
            self._load_current_session()
            
        self._build_view()

    def _scan_profiles(self) -> List[str]:
        profiles = set()
        
        # 1. Scan Memory
        for key in self.cog.global_chat_sessions.keys():
            # key structure: ('global', user_id (int), profile_name (str))
            if isinstance(key, tuple) and len(key) == 3 and key[0] == 'global' and key[1] == self.user_id:
                profiles.add(key[2])

        # 2. Scan Disk
        dir_path = pathlib.Path(self.cog.SESSIONS_GLOBAL_DIR) / self.user_id_str
        if dir_path.is_dir():
            for f in dir_path.iterdir():
                if f.name.endswith(".json.gz") and f.name != "_mappings.json.gz":
                    profiles.add(f.name[:-len(".json.gz")])
        
        return sorted(list(profiles))

    def _load_current_session(self):
        if not self.selected_profile: return
        self.session_key = ('global', self.user_id, self.selected_profile)
        
        session_data = self.cog.global_chat_sessions.get(self.session_key)
        if not session_data:
            session_data = self.cog._load_session_from_disk(self.session_key, 'global_chat')
            if session_data:
                self.cog.global_chat_sessions[self.session_key] = session_data
        
        self.rounds = []
        if session_data and 'unified_log' in session_data:
            log = session_data['unified_log']
            i = 0
            while i < len(log) - 1:
                curr = log[i]
                next_t = log[i+1]
                # Simple pairing strategy: User followed by Model
                if curr.get('role') == 'user' and next_t.get('role') == 'model':
                    self.rounds.append((curr, next_t))
                    i += 2
                else:
                    i += 1
            
            # If no pairs found but we have data, maybe show just models? 
            # For now strict pairing mirrors Whisper behavior best.
        
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
        prev_btn = ui.Button(label="◀", style=discord.ButtonStyle.blurple, disabled=(self.current_page == 0), row=2)
        next_btn = ui.Button(label="▶", style=discord.ButtonStyle.blurple, disabled=(self.current_page == len(self.rounds) - 1), row=2)
        delete_btn = ui.Button(label="Delete", style=discord.ButtonStyle.danger, row=2)
        
        prev_btn.callback = self.prev_callback
        next_btn.callback = self.next_callback
        delete_btn.callback = self.delete_callback
        
        self.add_item(prev_btn)
        self.add_item(next_btn)
        self.add_item(delete_btn)

    def get_embed(self) -> discord.Embed:
        if not self.rounds:
            return discord.Embed(description="No conversation history found.", color=discord.Color.dark_grey())
            
        user_turn, model_turn = self.rounds[self.current_page]
        
        # Resolve Appearance
        display_name = self.selected_profile
        avatar_url = self.cog.bot.user.display_avatar.url
        
        user_data = self.cog._get_user_data_entry(self.user_id)
        is_borrowed = self.selected_profile in user_data.get("borrowed_profiles", {})
        
        effective_owner_id = self.user_id
        effective_profile_name = self.selected_profile
        
        if is_borrowed:
            borrowed_data = user_data["borrowed_profiles"][self.selected_profile]
            effective_owner_id = int(borrowed_data["original_owner_id"])
            effective_profile_name = borrowed_data["original_profile_name"]
        
        appearance_data = self.cog.user_appearances.get(str(effective_owner_id), {}).get(effective_profile_name)
        if appearance_data:
            display_name = appearance_data.get("custom_display_name") or display_name
            avatar_url = appearance_data.get("custom_avatar_url") or avatar_url

        embed = discord.Embed(description=model_turn.get("content"), color=discord.Color.dark_grey())
        embed.set_author(name=display_name, icon_url=avatar_url)
        
        user_input = user_turn.get("content", "")
        embed.set_footer(text=f"You: {user_input}", icon_url=self.original_interaction.user.display_avatar.url)
        
        return embed

    async def profile_callback(self, interaction: discord.Interaction):
        self.selected_profile = interaction.data['values'][0]
        self._load_current_session()
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
            # Rebuild ChatSession
            new_history = []
            for t in session_data['unified_log']:
                role = 'model' if t.get('role') == 'model' else 'user'
                new_history.append({'role': role, 'parts': [t.get('content')]})
            session_data['chat_session'] = GoogleGenAIChatSession(history=new_history)
            
            # Cleanup mapping
            mapping_key = self.cog._get_mapping_key_for_session(self.session_key, 'global_chat')
            if mapping_key in self.cog.mapping_caches:
                mapping_data = self.cog.mapping_caches[mapping_key]
                keys_to_del = [k for k, v in mapping_data.items() if isinstance(v, (list, tuple)) and len(v) > 2 and v[2] in ids_to_delete]
                for k in keys_to_del:
                    del mapping_data[k]

            self._load_current_session() # Reload rounds logic
            self._build_view()
            
            if not self.rounds:
                self.cog._delete_session_from_disk(self.session_key, 'global_chat')
                # If we have other profiles, switch to one of them?
                # Or just show empty state.
                self.available_profiles = self._scan_profiles()
                if self.available_profiles:
                    self.selected_profile = self.available_profiles[0]
                    self._load_current_session()
                    self._build_view()
                    await interaction.edit_original_response(content="Round deleted. Switching to next available profile.", embed=self.get_embed(), view=self)
                else:
                    await interaction.edit_original_response(content="History cleared and session deleted.", embed=None, view=None)
            else:
                await interaction.edit_original_response(embed=self.get_embed(), view=self)
        else:
            await interaction.followup.send("Failed to delete round.", ephemeral=True)

class ProfileAdvancedParamsModal(ui.Modal, title="Advanced Parameters (OpenRouter Only)"):
    def __init__(self, cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, callback=None):
        super().__init__()
        self.cog: GeminiAgent = cog
        self.profile_name = profile_name
        self.is_borrowed = is_borrowed
        self.callback = callback

        def get_val(key):
            v = current_params.get(key)
            return str(v) if v is not None else ""

        self.add_item(ui.TextInput(label="Frequency Penalty (-2.0 to 2.0)", custom_id="frequency_penalty", default=get_val("frequency_penalty"), required=False, placeholder="Default: 0.0"))
        self.add_item(ui.TextInput(label="Presence Penalty (-2.0 to 2.0)", custom_id="presence_penalty", default=get_val("presence_penalty"), required=False, placeholder="Default: 0.0"))
        self.add_item(ui.TextInput(label="Repetition Penalty (0.0 to 2.0)", custom_id="repetition_penalty", default=get_val("repetition_penalty"), required=False, placeholder="Default: 1.0"))
        self.add_item(ui.TextInput(label="Min P (0.0 to 1.0)", custom_id="min_p", default=get_val("min_p"), required=False, placeholder="Default: 0.0 (Disabled)"))
        self.add_item(ui.TextInput(label="Top A (0.0 to 1.0)", custom_id="top_a", default=get_val("top_a"), required=False, placeholder="Default: 0.0 (Disabled)"))

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        new_params = {}
        try:
            def parse_float(val, min_v, max_v, name):
                if not val or not val.strip(): return None 
                f = float(val)
                if not (min_v <= f <= max_v): raise ValueError(f"{name} out of range ({min_v} to {max_v})")
                return f

            new_params["frequency_penalty"] = parse_float(next((c.value for c in self.children if c.custom_id == "frequency_penalty"), None), -2.0, 2.0, "Frequency Penalty")
            new_params["presence_penalty"] = parse_float(next((c.value for c in self.children if c.custom_id == "presence_penalty"), None), -2.0, 2.0, "Presence Penalty")
            new_params["repetition_penalty"] = parse_float(next((c.value for c in self.children if c.custom_id == "repetition_penalty"), None), 0.0, 2.0, "Repetition Penalty")
            new_params["min_p"] = parse_float(next((c.value for c in self.children if c.custom_id == "min_p"), None), 0.0, 1.0, "Min P")
            new_params["top_a"] = parse_float(next((c.value for c in self.children if c.custom_id == "top_a"), None), 0.0, 1.0, "Top A")

        except ValueError as e:
            await interaction.followup.send(f"❌ **Invalid Input:** {e}.", ephemeral=True)
            return
        except Exception as e:
            print(f"Error parsing adv params: {e}")
            await interaction.followup.send("❌ Error parsing input.", ephemeral=True)
            return

        if self.profile_name == "BULK_APPLY":
            pass
        else:
            try:
                success = await self.cog.update_profile_advanced_params(interaction.user.id, self.profile_name, new_params, interaction.channel_id, self.is_borrowed)
                if success:
                    await interaction.followup.send(f"✅ Advanced parameters updated for '{self.profile_name}'.", ephemeral=True)
                    if self.callback: 
                        try: await self.callback(interaction)
                        except Exception as e: print(f"Callback error in AdvParams: {e}")
                else:
                    await interaction.followup.send(f"❌ Failed to update parameters.", ephemeral=True)
            except Exception as e:
                print(f"Error updating adv params: {e}")
                traceback.print_exc()
                await interaction.followup.send("❌ An unexpected error occurred while saving.", ephemeral=True)

class WhisperHistoryView(ui.View):
    def __init__(self, cog: 'GeminiAgent', interaction: discord.Interaction, all_whispers: List[Dict]):
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
            profile_keys.add(tuple(whisper.get("target_key")))
        
        profile_options = [discord.SelectOption(label="All Profiles", value="all", default=(self.selected_profile_key is None))]
        for p_key in sorted(list(profile_keys)):
            owner_id, profile_name = p_key
            profile_options.append(discord.SelectOption(label=profile_name, value=f"{owner_id}:{profile_name}", default=(self.selected_profile_key == f"{owner_id}:{profile_name}")))
        
        profile_select = ui.Select(placeholder="Filter by profile...", options=profile_options, row=0)
        profile_select.callback = self.profile_filter_callback
        self.add_item(profile_select)

        # --- Filter whispers based on selection ---
        if self.selected_profile_key:
            owner_id, profile_name = self.selected_profile_key.split(":")
            p_key_tuple = (int(owner_id), profile_name)
            self.filtered_whispers = [pair for pair in self.all_whispers if tuple(pair[0].get("target_key")) == p_key_tuple]
        else:
            self.filtered_whispers = self.all_whispers

        # --- Build Whisper Jump Dropdown ---
        if self.filtered_whispers:
            whisper_options = []
            for i, (whisper, _) in enumerate(self.filtered_whispers):
                ts = datetime.datetime.fromisoformat(whisper.get("timestamp"))
                ts_str = ts.strftime('%b %d, %I:%M %p')
                content_preview = whisper.get("content", "").split("\n")[1][:50]
                whisper_options.append(discord.SelectOption(label=f"({ts_str}) {content_preview}...", value=str(i), default=(i == self.current_page)))
            
            whisper_select = ui.Select(placeholder="Jump to a whisper...", options=whisper_options[:DROPDOWN_MAX_OPTIONS], row=1)
            whisper_select.callback = self.whisper_jump_callback
            self.add_item(whisper_select)

        # --- Build Buttons ---
        prev_button = ui.Button(label="◀", style=discord.ButtonStyle.blurple, disabled=(self.current_page == 0), row=2)
        next_button = ui.Button(label="▶", style=discord.ButtonStyle.blurple, disabled=(self.current_page >= len(self.filtered_whispers) - 1), row=2)
        delete_button = ui.Button(label="Delete", style=discord.ButtonStyle.danger, disabled=(not self.filtered_whispers), row=2)

        prev_button.callback = self.pagination_callback
        next_button.callback = self.pagination_callback
        delete_button.callback = self.delete_callback
        
        self.add_item(prev_button)
        self.add_item(next_button)
        self.add_item(delete_button)

    def _get_current_embed(self) -> discord.Embed:
        if not self.filtered_whispers:
            return discord.Embed(title="Whisper History", description="No whispers found for the selected filter.", color=discord.Color.dark_grey())

        whisper_turn, response_turn = self.filtered_whispers[self.current_page]
        
        owner_id, profile_name = tuple(response_turn.get("speaker_key"))
        response_content = "\n".join(response_turn.get("content", "").split("\n")[1:]).strip() # Remove header line
        whisper_content = "\n".join(whisper_turn.get("content", "").split("\n")[1:]).strip() # Remove header line

        user_data = self.cog._get_user_data_entry(owner_id)
        is_borrowed = profile_name in user_data.get("borrowed_profiles", {})
        effective_owner_id, effective_profile_name = owner_id, profile_name
        if is_borrowed:
            borrowed_data = user_data["borrowed_profiles"][profile_name]
            effective_owner_id = int(borrowed_data["original_owner_id"])
            effective_profile_name = borrowed_data["original_profile_name"]
        
        display_name = effective_profile_name
        avatar_url = self.cog.bot.user.display_avatar.url
        appearance = self.cog.user_appearances.get(str(effective_owner_id), {}).get(effective_profile_name)
        if appearance:
            display_name = appearance.get("custom_display_name") or display_name
            avatar_url = appearance.get("custom_avatar_url") or avatar_url

        embed = discord.Embed(description=response_content, color=discord.Color.dark_grey())
        embed.set_author(name=display_name, icon_url=avatar_url)
        embed.set_footer(text=f"Private whisper: {whisper_content}", icon_url=self.original_interaction.user.display_avatar.url)
        
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

    async def pagination_callback(self, interaction: discord.Interaction):
        if interaction.data['custom_id'] == 'prev_button':
            self.current_page -= 1
        else:
            self.current_page += 1
        self._build_view()
        await interaction.response.edit_message(embed=self._get_current_embed(), view=self)

    async def delete_callback(self, interaction: discord.Interaction):
        if not self.filtered_whispers:
            await interaction.response.send_message("Nothing to delete.", ephemeral=True, delete_after=5)
            return

        whisper_turn, response_turn = self.filtered_whispers[self.current_page]
        whisper_turn_id = whisper_turn.get("turn_id")
        response_turn_id = response_turn.get("turn_id")

        # Use the same logic as the single-delete view
        view = WhisperActionView(self.cog, self.original_interaction, whisper_turn_id, response_turn_id)
        # Manually create a button and trigger its callback to perform the deletion
        button = ui.Button(label="Confirm")
        await view.delete_button.callback(interaction)
        
        # After deletion, refresh the history view
        await self.cog._show_whisper_history(self.original_interaction)

class WhisperActionView(ui.View):
    def __init__(self, cog: 'GeminiAgent', interaction: discord.Interaction, whisper_turn_id: str, response_turn_id: str):
        super().__init__(timeout=300)
        self.cog = cog
        self.original_interaction = interaction
        self.channel_id = interaction.channel_id
        self.whisper_turn_id = whisper_turn_id
        self.response_turn_id = response_turn_id

    @ui.button(label="Delete Whisper", style=discord.ButtonStyle.danger, custom_id="delete_whisper")
    async def delete_button(self, interaction: discord.Interaction, button: ui.Button):
        await interaction.response.defer()
        session = self.cog.multi_profile_channels.get(self.channel_id)
        if not session:
            await interaction.edit_original_response(content="Session not found or has ended.", view=None, embed=None)
            return

        if not session.get("is_hydrated"):
            session = self.cog._ensure_session_hydrated(self.channel_id, session.get("type", "multi"))

        turn_ids_to_delete = {self.whisper_turn_id, self.response_turn_id}
        
        original_log_len = len(session.get("unified_log", []))
        
        target_profile_key = None
        for turn in session.get("unified_log", []):
            if turn.get("turn_id") in turn_ids_to_delete:
                target_profile_key = tuple(turn.get("target_key") or turn.get("speaker_key"))
                break

        session["unified_log"] = [
            turn for turn in session.get("unified_log", [])
            if turn.get("turn_id") not in turn_ids_to_delete
        ]

        if len(session["unified_log"]) < original_log_len and target_profile_key:
            participant_history = []
            for turn in session["unified_log"]:
                turn_type = turn.get("type")
                if not turn_type:
                    speaker_key = tuple(turn.get("speaker_key", []))
                    role = 'model' if speaker_key == target_profile_key else 'user'
                    participant_history.append({'role': role, 'parts': [turn.get("content")]})
                elif turn_type == "whisper":
                    t_key = tuple(turn.get("target_key", []))
                    if target_profile_key == t_key:
                        participant_history.append({'role': 'user', 'parts': [turn.get("content")]})
                elif turn_type == "private_response":
                    s_key = tuple(turn.get("speaker_key", []))
                    if target_profile_key == s_key:
                        participant_history.append({'role': 'model', 'parts': [turn.get("content")]})

            session["chat_sessions"][target_profile_key] = GoogleGenAIChatSession(history=participant_history)
            session.get("pending_whispers", {}).pop(target_profile_key, None)

        await interaction.edit_original_response(content="Whisper has been deleted from the profile's memory.", view=None, embed=None)

class ProfileDirectorDeskModal(ui.Modal, title="Director's Desk: TTS Instructions"):
    def __init__(self, cog, profile_name: str, current_params: Dict[str, Any], callback=None):
        super().__init__()
        self.cog = cog
        self.profile_name = profile_name
        self.callback = callback

        self.archetype_input = ui.TextInput(
            label="Archetype (Who)", 
            default=str(current_params.get("speech_archetype", "")), 
            placeholder="e.g. A cynical noir detective, a bubbly influencer.",
            required=False, max_length=200
        )
        self.accent_input = ui.TextInput(
            label="Accent", 
            default=str(current_params.get("speech_accent", "")), 
            placeholder="e.g. Australian (Melbourne), British (Brixton).",
            required=False, max_length=200
        )
        self.dynamics_input = ui.TextInput(
            label="Dynamics (Where / Acoustics)", 
            default=str(current_params.get("speech_dynamics", "")), 
            placeholder="e.g. Speaking in a whisper, cavernous echoing hall.",
            required=False, max_length=200
        )
        self.style_input = ui.TextInput(
            label="Vocal Style (How)", 
            default=str(current_params.get("speech_style", "")), 
            placeholder="e.g. A vocal smile, breathy, gritty and gravelly.",
            required=False, max_length=200
        )
        self.pacing_input = ui.TextInput(
            label="Pacing (Tempo)", 
            default=str(current_params.get("speech_pacing", "")), 
            placeholder="e.g. Rapid-fire delivery, slow deliberate drawl.",
            required=False, max_length=200
        )
        
        self.add_item(self.archetype_input)
        self.add_item(self.accent_input)
        self.add_item(self.dynamics_input)
        self.add_item(self.style_input)
        self.add_item(self.pacing_input)

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        new_params = {
            "speech_archetype": self.archetype_input.value.strip(),
            "speech_accent": self.accent_input.value.strip(),
            "speech_dynamics": self.dynamics_input.value.strip(),
            "speech_style": self.style_input.value.strip(),
            "speech_pacing": self.pacing_input.value.strip()
        }

        user_id = interaction.user.id
        user_data = self.cog._get_user_data_entry(user_id)
        profile = user_data.get("profiles", {}).get(self.profile_name)
        
        if profile:
            profile.update(new_params)
            self.cog._save_user_data_entry(user_id, user_data)
            await interaction.followup.send(f"✅ TTS Instructions updated for '{self.profile_name}'.", ephemeral=True)
            if self.callback: await self.callback(interaction)

class ProfileSpeechSettingsModal(ui.Modal, title="Speech & Voice Settings"):
    def __init__(self, cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, callback=None):
        super().__init__()
        self.cog = cog
        self.profile_name = profile_name
        self.is_borrowed = is_borrowed
        self.callback = callback

        self.voice_input = ui.TextInput(
            label="Voice Name (Aoede, Charon, Puck, Kore, etc)", 
            default=str(current_params.get("speech_voice", "Aoede")), 
            placeholder="Identity for synthesis (e.g. Aoede)",
            required=False,
            max_length=40
        )
        self.model_input = ui.TextInput(
            label="Speech Model ID", 
            default=str(current_params.get("speech_model", "gemini-2.5-flash-preview-tts")), 
            placeholder="Model used for synthesis",
            required=False,
            max_length=80
        )
        self.temp_input = ui.TextInput(
            label="Speech Temp / Prosody (0.0 - 2.0)", 
            default=str(current_params.get("speech_temperature", 1.0)), 
            placeholder="1.0 = Default, 2.0 = High Expression",
            required=False,
            max_length=5
        )
        
        self.add_item(self.voice_input)
        self.add_item(self.model_input)
        self.add_item(self.temp_input)

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        
        voice = self.voice_input.value.strip() or "Aoede"
        model = self.model_input.value.strip() or "gemini-2.5-flash-preview-tts"
        
        try:
            temp_raw = self.temp_input.value.strip()
            temp = float(temp_raw) if temp_raw else 1.0
            if not (0.0 <= temp <= 2.0): raise ValueError()
        except ValueError:
            await interaction.followup.send("❌ **Invalid Temperature.** Enter a number between 0.0 and 2.0.", ephemeral=True)
            return

        new_params = {
            "speech_voice": voice,
            "speech_model": model,
            "speech_temperature": temp,
        }

        user_id = interaction.user.id
        user_data = self.cog._get_user_data_entry(user_id)
        target_dict = user_data.get("borrowed_profiles" if self.is_borrowed else "profiles", {})
        profile = target_dict.get(self.profile_name)
        
        if profile:
            profile.update(new_params)
            self.cog._save_user_data_entry(user_id, user_data)
            
            keys_to_clear = [
                k for k in self.cog.channel_models.keys() 
                if isinstance(k, tuple) and len(k) == 3 and k[1] == user_id and k[2] == self.profile_name
            ]
            for k in keys_to_clear:
                self.cog.channel_models.pop(k, None)
                self.cog.chat_sessions.pop(k, None)
                self.cog.channel_model_last_profile_key.pop(k, None)

            await interaction.followup.send(f"✅ Speech settings updated for '{self.profile_name}'.", ephemeral=True)
            if self.callback: await self.callback(interaction)

class PaginatedEmbedView(ui.View):
    def __init__(self, embeds: List[discord.Embed], page_titles: List[str]):
        super().__init__(timeout=300)
        self.embeds = embeds
        self.page_titles = page_titles
        self.current_page = 0
        self._build_view()

    def _build_view(self):
        self.clear_items()
        
        # Row 0: Dropdown (Chunked to 25 items max)
        chunk_index = self.current_page // 25
        start_idx = chunk_index * 25
        end_idx = min(start_idx + 25, len(self.embeds))

        options = []
        for i in range(start_idx, end_idx):
            title = self.page_titles[i]
            options.append(discord.SelectOption(
                label=f"{i+1}. {title[:80]}",
                value=str(i),
                default=(i == self.current_page)
            ))
        
        placeholder = f"Index Sections {start_idx + 1}-{end_idx}..." if len(self.embeds) > 25 else "Jump to a section..."
        select = ui.Select(placeholder=placeholder, options=options, row=0)
        select.callback = self.select_callback
        self.add_item(select)

        # Row 1: Pagination Navigation
        prev_btn = ui.Button(label="◀", style=discord.ButtonStyle.secondary, disabled=(self.current_page == 0), row=1)
        prev_btn.callback = self.prev_callback
        self.add_item(prev_btn)

        page_lbl = ui.Button(label=f"Page {self.current_page + 1}/{len(self.embeds)}", style=discord.ButtonStyle.grey, disabled=True, row=1)
        self.add_item(page_lbl)

        next_btn = ui.Button(label="▶", style=discord.ButtonStyle.secondary, disabled=(self.current_page >= len(self.embeds) - 1), row=1)
        next_btn.callback = self.next_callback
        self.add_item(next_btn)

    async def select_callback(self, interaction: discord.Interaction):
        self.current_page = int(interaction.data['values'][0])
        self._build_view()
        await interaction.response.edit_message(embed=self.embeds[self.current_page], view=self)

    async def prev_callback(self, interaction: discord.Interaction):
        self.current_page -= 1
        self._build_view()
        await interaction.response.edit_message(embed=self.embeds[self.current_page], view=self)

    async def next_callback(self, interaction: discord.Interaction):
        self.current_page += 1
        self._build_view()
        await interaction.response.edit_message(embed=self.embeds[self.current_page], view=self)

class ProfileManageView(ui.View):
    def __init__(self, cog: 'GeminiAgent', original_interaction: discord.Interaction, profile_name: str, is_borrowed: bool):
        super().__init__(timeout=600)
        self.cog = cog
        self.original_interaction = original_interaction
        self.user_id = original_interaction.user.id
        self.profile_name = profile_name
        self.is_borrowed = is_borrowed
        self.current_tab = "home"
        self._build_view()

    def _build_view(self):
        self.clear_items()
        
        # --- 1. Category Dropdown (Row 0) ---
        options = []
        if self.current_tab == "home":
            if self.profile_name != DEFAULT_PROFILE_NAME:
                options.append(discord.SelectOption(label="Rename Profile", value="rename", description="Change the local name of this profile."))
            
            if not self.is_borrowed:
                options.append(discord.SelectOption(label="Duplicate Profile", value="duplicate", description="Create a new profile from a copy of this one."))
                options.append(discord.SelectOption(label="Share Profile", value="share", description="Share this profile with others or publish it."))
                options.append(discord.SelectOption(label="Custom Error Message", value="error_response", description="Set the message shown when generation fails."))
            
            options.append(discord.SelectOption(label="Cycle Content Safety Level", value="safety_level", description="Cycle: Low -> Medium -> High -> Unrestricted 18+."))
            
            if self.profile_name != DEFAULT_PROFILE_NAME:
                label = "Remove Borrowed Profile" if self.is_borrowed else "Delete Profile"
                options.append(discord.SelectOption(label=label, value="delete", description="Permanently remove this profile and its data."))

        elif self.current_tab == "persona":
            # Tab hidden for borrowed, but kept for logic safety
            options.append(discord.SelectOption(label="Edit Persona", value="edit_persona", description="Edit backstory, traits, likes, dislikes, and appearance."))
            options.append(discord.SelectOption(label="Edit Instructions", value="edit_instructions", description="Edit specific AI behavioral instructions."))
            options.append(discord.SelectOption(label="TTS Instructions", value="tts_instructions", description="Configure the 'Director's Desk' for vocal performance."))
            options.append(discord.SelectOption(label="Edit Appearance", value="edit_appearance", description="Edit the custom Webhook name and avatar."))

        elif self.current_tab == "params":
            options.append(discord.SelectOption(label="Set Models", value="models", description="Choose Primary and Fallback AI models."))
            options.append(discord.SelectOption(label="Set Generation Parameters & STM", value="gen_params", description="Set Temp, Top P, Top K, and STM Length."))
            options.append(discord.SelectOption(label="Set Advanced Parameters (OPENROUTER)", value="adv_params", description="Set penalties, Min P, and Top A."))
            options.append(discord.SelectOption(label="Set Thinking Parameters", value="thinking_params", description="Set thinking persistence, level, and budget."))
            options.append(discord.SelectOption(label="Set Speech & Voice Settings", value="speech_settings", description="Set TTS voice, model, and prosody."))

        elif self.current_tab == "tools":
            options.append(discord.SelectOption(label="Toggle Image Generation", value="image_toggle", description="Allow this profile to generate images via !image/!imagine."))
            options.append(discord.SelectOption(label="Toggle Grounding (Web Search)", value="grounding", description="Cycle grounding: Off -> On -> On+."))
            options.append(discord.SelectOption(label="Toggle URL Context Fetching", value="url_toggle", description="Allow this profile to fetch content from links in messages."))
            options.append(discord.SelectOption(label="Cycle Response Mode", value="cycle_response", description="Cycle: Regular -> Mention -> Reply -> Mention Reply."))
            options.append(discord.SelectOption(label="Set Time & Timezone", value="time", description="Enable time awareness and set the profile's timezone."))
            options.append(discord.SelectOption(label="Toggle Realistic Typing", value="typing", description="Enable a human-like delay when the bot sends messages."))
            options.append(discord.SelectOption(label="Toggle Anti-Repetition Critic", value="critic", description="Enable semantic repetition analysis (Adds latency)."))

        elif self.current_tab == "memory":
            options.append(discord.SelectOption(label="Manage Data (LTM & Training)", value="data", description="Add, list, edit, or delete memories and examples."))
            options.append(discord.SelectOption(label="Set Training Parameters", value="train_params", description="Set training context size and relevance threshold."))
            if not self.is_borrowed:
                options.append(discord.SelectOption(label="Toggle LTM Auto-Creation", value="ltm_creation", description="Automatically create memories from conversations."))
                options.append(discord.SelectOption(label="Set LTM Parameters", value="ltm_params", description="Set frequency, context, and recall settings."))
            options.append(discord.SelectOption(label="Cycle LTM Scope", value="ltm_scope", description="Cycle visibility: Global -> Server -> User."))
            if not self.is_borrowed:
                options.append(discord.SelectOption(label="Set LTM Summarization Prompt", value="ltm_summarization", description="Customize how the AI creates memories."))
                options.append(discord.SelectOption(label="Set Image Generation Prompt", value="image_gen_prompt", description="Instruction for image creation style."))

        if options:
            select = ui.Select(placeholder=f"Select an action for {self.current_tab.title()}...", options=options, row=0)
            select.callback = self.dropdown_callback
            self.add_item(select)

        # --- 2. Navigation Buttons (Row 1) ---
        tabs = ["home", "persona", "params", "tools", "memory"]
        if self.is_borrowed: tabs.remove("persona")
        
        for tab in tabs:
            btn = ui.Button(
                label=tab.title(), 
                style=discord.ButtonStyle.primary if self.current_tab == tab else discord.ButtonStyle.secondary, 
                row=1, 
                disabled=(self.current_tab == tab)
            )
            # Use a wrapper to capture tab name
            btn.callback = self.create_nav_callback(tab)
            self.add_item(btn)

    def create_nav_callback(self, tab_name):
        async def callback(interaction: discord.Interaction):
            self.current_tab = tab_name
            self._build_view()
            await interaction.response.edit_message(view=self)
        return callback

    async def dropdown_callback(self, interaction: discord.Interaction):
        choice = interaction.data['values'][0]
        user_id = self.user_id
        profile_name = self.profile_name
        user_data = self.cog._get_user_data_entry(user_id)
        profile = user_data.get("borrowed_profiles" if self.is_borrowed else "profiles", {}).get(profile_name)
        
        if not profile:
            await interaction.response.send_message("Profile data not found.", ephemeral=True); return

        # --- Home Tab Logic ---
        if choice == "rename":
            await self._handle_rename(interaction)
        elif choice == "duplicate":
            await self._handle_duplicate(interaction)
        elif choice == "share":
            await self._handle_share(interaction)
        elif choice == "delete":
            await self._handle_delete(interaction)
        elif choice == "safety_level":
            await self._handle_safety_cycle(interaction, user_data, profile)
        elif choice == "error_response":
            user_data = self.cog._get_user_data_entry(interaction.user.id)
            is_b = getattr(self, "is_borrowed", False)
            target_profile = user_data.get("borrowed_profiles", {}).get(self.profile_name) if is_b else user_data.get("profiles", {}).get(self.profile_name)

            if not target_profile:
                await interaction.response.send_message("❌ Error: Profile not found.", ephemeral=True)
                return

            async def modal_callback(modal_interaction: discord.Interaction, new_val: str):
                await modal_interaction.response.defer(ephemeral=True)
                val_to_save = new_val.strip() or "An error has occurred."
                
                u_data = self.cog._get_user_data_entry(modal_interaction.user.id)
                target = u_data.get("borrowed_profiles", {}).get(self.profile_name) if is_b else u_data.get("profiles", {}).get(self.profile_name)
                
                if target:
                    target["error_response"] = val_to_save
                    self.cog._save_user_data_entry(modal_interaction.user.id, u_data)
                    await modal_interaction.followup.send(f"✅ Custom error message updated for '{self.profile_name}'.", ephemeral=True)
                else:
                    await modal_interaction.followup.send("❌ Error: Profile not found.", ephemeral=True)

            modal = ActionTextInputModal(
                title="Set Custom Error Message",
                label="Error Message",
                placeholder="Enter the message to show on API/Safety errors...",
                default=target_profile.get("error_response", "An error has occurred."),
                required=False,
                on_submit_callback=modal_callback
            )
            await interaction.response.send_modal(modal)

        # --- Persona Tab Logic ---
        elif choice == "edit_persona":
            modal = EditUserProfilePersonaModal(self.cog, profile_name, profile.get("persona", {}), user_id)
            await interaction.response.send_modal(modal)
        elif choice == "edit_instructions":
            modal = EditUserProfileAIInstructionsModal(self.cog, profile_name, profile.get("ai_instructions", ""), user_id)
            await interaction.response.send_modal(modal)
        elif choice == "tts_instructions":
            async def refresh_cb(modal_interaction: discord.Interaction):
                new_embed = await self.cog._build_profile_manage_embed(modal_interaction, profile_name)
                await self.original_interaction.edit_original_response(embed=new_embed, view=self)
            modal = ProfileDirectorDeskModal(self.cog, profile_name, profile, callback=refresh_cb)
            await interaction.response.send_modal(modal)
        elif choice == "edit_appearance":
            await self._handle_appearance(interaction)

        # --- Params Tab Logic ---
        elif choice == "models":
            view = SingleProfileModelView(self.cog, self.original_interaction, profile_name)
            await interaction.response.send_message(view._get_selection_feedback_message(), view=view, ephemeral=True)
        elif choice == "gen_params":
            # Callback logic updated to edit the view on the original message, but not try to defer again
            async def refresh_cb(modal_interaction: discord.Interaction):
                new_embed = await self.cog._build_profile_manage_embed(modal_interaction, profile_name)
                # Edit the MAIN message (the dashboard)
                await self.original_interaction.edit_original_response(embed=new_embed, view=self)
                
            modal = ProfileParamsModal(self.cog, profile_name, profile, self.is_borrowed, callback=refresh_cb)
            await interaction.response.send_modal(modal)
        elif choice == "adv_params":
            async def refresh_cb(modal_interaction: discord.Interaction):
                new_embed = await self.cog._build_profile_manage_embed(modal_interaction, profile_name)
                await self.original_interaction.edit_original_response(embed=new_embed, view=self)
                
            modal = ProfileAdvancedParamsModal(self.cog, profile_name, profile, self.is_borrowed, callback=refresh_cb)
            await interaction.response.send_modal(modal)
        elif choice == "thinking_params":
            async def refresh_cb(modal_interaction: discord.Interaction):
                new_embed = await self.cog._build_profile_manage_embed(modal_interaction, profile_name)
                await self.original_interaction.edit_original_response(embed=new_embed, view=self)
            
            # [UPDATED] Pass self.is_borrowed to the modal
            modal = ProfileThinkingParamsModal(self.cog, profile_name, profile, self.is_borrowed, callback=refresh_cb)
            await interaction.response.send_modal(modal)

        elif choice == "speech_settings":
            async def refresh_cb(modal_interaction: discord.Interaction):
                new_embed = await self.cog._build_profile_manage_embed(modal_interaction, profile_name)
                await self.original_interaction.edit_original_response(embed=new_embed, view=self)
            
            modal = ProfileSpeechSettingsModal(self.cog, profile_name, profile, self.is_borrowed, callback=refresh_cb)
            await interaction.response.send_modal(modal)

        # --- Tools Tab Logic ---
        elif choice == "cycle_response":
            modes = ["regular", "mention", "reply", "mention_reply"]
            curr = profile.get("response_mode", "regular")
            profile["response_mode"] = modes[(modes.index(curr) + 1) % len(modes)]
            await self._save_and_refresh(interaction, user_data, profile_name)
        elif choice == "typing":
            profile["realistic_typing_enabled"] = not profile.get("realistic_typing_enabled", False)
            await self._save_and_refresh(interaction, user_data, profile_name)
        elif choice == "grounding":
            current_mode = profile.get("grounding_mode", "off")
            if isinstance(current_mode, bool): current_mode = "on" if current_mode else "off"
            cycle_map = {"off": "on", "on": "on+", "on+": "off"}
            profile["grounding_mode"] = cycle_map.get(current_mode, "off")
            await self._save_and_refresh(interaction, user_data, profile_name)
        elif choice == "image_toggle":
            profile["image_generation_enabled"] = not profile.get("image_generation_enabled", False)
            await self._save_and_refresh(interaction, user_data, profile_name)
        elif choice == "url_toggle":
            profile["url_fetching_enabled"] = not profile.get("url_fetching_enabled", False)
            await self._save_and_refresh(interaction, user_data, profile_name)
        elif choice == "time":
            await self._handle_timezone(interaction, user_data, profile)
        elif choice == "critic":
            profile["critic_enabled"] = not profile.get("critic_enabled", False)
            await self._save_and_refresh(interaction, user_data, profile_name)

        # --- Memory Tab Logic ---
        elif choice == "data":
            owner_id_config = int(defaultConfig.DISCORD_OWNER_ID)
            effective_owner_id = user_id
            if self.is_borrowed:
                if profile_name == DEFAULT_PROFILE_NAME: effective_owner_id = owner_id_config
                else:
                    borrow_data = user_data.get("borrowed_profiles", {}).get(profile_name)
                    if borrow_data: effective_owner_id = int(borrow_data["original_owner_id"])
            view = DataManageView(self.cog, self.original_interaction, profile_name, self.is_borrowed, effective_owner_id)
            await view.start()
            await interaction.response.defer()
        elif choice == "ltm_creation":
            profile["ltm_creation_enabled"] = not profile.get("ltm_creation_enabled", False)
            await self._save_and_refresh(interaction, user_data, profile_name)
        elif choice == "ltm_scope":
            scope_cycle = {'global': 'server', 'server': 'user', 'user': 'global'}
            profile['ltm_scope'] = scope_cycle.get(profile.get('ltm_scope', 'server'), 'server')
            await self._save_and_refresh(interaction, user_data, profile_name)
        elif choice == "ltm_params":
            async def refresh_cb(i):
                new_embed = await self.cog._build_profile_manage_embed(i, profile_name)
                await self.original_interaction.edit_original_response(embed=new_embed, view=self)
            modal = ProfileLTMParamsModal(self.cog, profile_name, profile, callback=refresh_cb)
            await interaction.response.send_modal(modal)
        elif choice == "train_params":
            async def refresh_cb(i):
                new_embed = await self.cog._build_profile_manage_embed(i, profile_name)
                await self.original_interaction.edit_original_response(embed=new_embed, view=self)
            modal = ProfileTrainingParamsModal(self.cog, profile_name, profile, callback=refresh_cb)
            await interaction.response.send_modal(modal)
        elif choice == "ltm_summarization":
            instr = profile.get("ltm_summarization_instructions", DEFAULT_LTM_SUMMARIZATION_INSTRUCTIONS)
            modal = ProfileLTMSummarizationModal(self.cog, profile_name, instr)
            await interaction.response.send_modal(modal)
        elif choice == "image_gen_prompt":
            curr = profile.get("image_generation_prompt")
            modal = ProfileImageGenPromptModal(self.cog, profile_name, curr)
            await interaction.response.send_modal(modal)

    # --- Internal Helpers for UI Flow ---

    async def _save_and_refresh(self, interaction, user_data, profile_name):
        self.cog._save_user_data_entry(self.user_id, user_data)
        
        # [NEW] Hot-Swap: Invalidate model and session caches for this profile immediately
        # This ensures settings take effect even if a multi-profile session is active.
        keys_to_clear = [
            k for k in self.cog.channel_models.keys() 
            if isinstance(k, tuple) and len(k) == 3 and k[1] == self.user_id and k[2] == self.profile_name
        ]
        for k in keys_to_clear:
            self.cog.channel_models.pop(k, None)
            self.cog.chat_sessions.pop(k, None)
            self.cog.channel_model_last_profile_key.pop(k, None)

        new_embed = await self.cog._build_profile_manage_embed(interaction, profile_name)
        await interaction.response.edit_message(embed=new_embed, view=self)

    async def _handle_safety_cycle(self, interaction, user_data, profile):
        is_public = self.cog._is_profile_public(self.user_id, self.profile_name)
        cycle_full = {'low': 'medium', 'medium': 'high', 'high': 'unrestricted', 'unrestricted': 'low'}
        cycle_rest = {'low': 'medium', 'medium': 'high', 'high': 'low'}
        curr = profile.get('safety_level', 'low')
        profile['safety_level'] = (cycle_rest if (self.is_borrowed or is_public) else cycle_full).get(curr, 'low')
        await self._save_and_refresh(interaction, user_data, self.profile_name)

    async def _handle_appearance(self, interaction):
        user_apps = self.cog.user_appearances.get(str(self.user_id), {})
        if self.profile_name in user_apps:
            view = AppearanceEditView(self.cog, self.original_interaction, self.profile_name)
            await view.show(interaction)
        else:
            modal = AppearanceCreateModal(self.cog, self.original_interaction, self.profile_name)
            await interaction.response.send_modal(modal)

    async def _handle_rename(self, interaction):
        modal = ui.Modal(title=f"Rename '{self.profile_name}'")
        new_name_input = ui.TextInput(label="Enter new unique name", required=True)
        modal.add_item(new_name_input)
        async def rename_submit(i: discord.Interaction):
            await i.response.defer()
            new_name = new_name_input.value.lower().strip()
            old_name = self.profile_name
            if old_name == DEFAULT_PROFILE_NAME:
                await self.original_interaction.edit_original_response(content=f"Rename failed: '{DEFAULT_PROFILE_NAME}' cannot be renamed.", view=None, embed=None); return
            if not new_name or new_name.lower() == 'clyde':
                await self.original_interaction.edit_original_response(content="Rename failed: Invalid name.", view=None, embed=None); return
            user_data = self.cog._get_user_data_entry(self.user_id)
            if new_name in user_data.get("profiles", {}) or new_name in user_data.get("borrowed_profiles", {}):
                await self.original_interaction.edit_original_response(content="Rename failed: Name already exists.", view=None, embed=None); return
            
            p_dict_key = "borrowed_profiles" if self.is_borrowed else "profiles"
            if old_name in user_data[p_dict_key]:
                # Logic for public update omitted for brevity (keep existing logic)
                profile_data = user_data[p_dict_key].pop(old_name)
                user_data[p_dict_key][new_name] = profile_data
                for ch_id, act in user_data.get("channel_active_profiles", {}).items():
                    if act == old_name: user_data["channel_active_profiles"][ch_id] = new_name
                if not self.is_borrowed:
                    self.cog._rename_ltm_shards(str(self.user_id), old_name, new_name)
                    self.cog._rename_training_shards(str(self.user_id), old_name, new_name)
                self.cog._save_user_data_entry(self.user_id, user_data)
                await self.original_interaction.edit_original_response(content=f"Profile '{old_name}' renamed to '{new_name}'.", view=None, embed=None)
        modal.on_submit = rename_submit
        await interaction.response.send_modal(modal)

    async def _handle_duplicate(self, interaction):
        modal = ui.Modal(title=f"Duplicate '{self.profile_name}'")
        new_name_input = ui.TextInput(label="Enter name for copy", required=True)
        modal.add_item(new_name_input)
        async def duplicate_submit(i: discord.Interaction):
            await i.response.defer()
            new_name = new_name_input.value.lower().strip()
            if not new_name: return
            user_data = self.cog._get_user_data_entry(self.user_id)
            limit = defaultConfig.LIMIT_PROFILES_PREMIUM if self.cog.is_user_premium(self.user_id) else defaultConfig.LIMIT_PROFILES_FREE
            if len(user_data.get("profiles", {})) >= limit:
                await self.original_interaction.edit_original_response(content="Limit reached.", view=None, embed=None); return
            user_data["profiles"][new_name] = copy.deepcopy(user_data["profiles"][self.profile_name])
            user_data["profiles"][new_name]['created_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            self.cog._save_user_data_entry(self.user_id, user_data)
            self.cog._copy_ltm_shard(str(self.user_id), self.profile_name, new_name)
            self.cog._copy_training_shard(str(self.user_id), self.profile_name, new_name)
            await self.original_interaction.edit_original_response(content=f"Duplicated to '{new_name}'.", view=None, embed=None)
        modal.on_submit = duplicate_submit
        await interaction.response.send_modal(modal)

    async def _handle_share(self, interaction):
        view = HubShareManagerView(self.cog, interaction)
        view.selected_profiles = [self.profile_name]
        view.setup_items()
        desc = "Manage how you share your profiles."
        embed = discord.Embed(title="Share Manager", description=desc, color=discord.Color.teal())
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

    async def _handle_delete(self, interaction):
        if self.profile_name == DEFAULT_PROFILE_NAME:
            await interaction.response.send_message("The default profile cannot be deleted.", ephemeral=True); return
        confirm_view = ui.View(timeout=60)
        async def confirm_delete(i: discord.Interaction):
            await i.response.defer()
            user_data = self.cog._get_user_data_entry(self.user_id)
            p_dict_key = "borrowed_profiles" if self.is_borrowed else "profiles"
            if self.profile_name in user_data[p_dict_key]:
                del user_data[p_dict_key][self.profile_name]
                if not self.is_borrowed:
                    self.cog._delete_ltm_shard(str(self.user_id), self.profile_name)
                    self.cog._delete_training_shard(str(self.user_id), self.profile_name)
                self.cog._save_user_data_entry(self.user_id, user_data)
                await self.original_interaction.edit_original_response(content=f"Profile '{self.profile_name}' deleted.", view=None, embed=None)
        confirm_btn = ui.Button(label="Confirm Deletion", style=discord.ButtonStyle.danger)
        confirm_btn.callback = confirm_delete
        confirm_view.add_item(confirm_btn)
        await interaction.response.send_message(f"Delete profile '{self.profile_name}'?", view=confirm_view, ephemeral=True)

    async def _handle_timezone(self, interaction, user_data, profile):
        view = ui.View(timeout=180)
        common_tzs = ["UTC", "GMT", "US/Pacific", "US/Central", "US/Eastern", "Europe/London", "Europe/Berlin", "Asia/Tokyo", "Australia/Sydney"]
        opts = [discord.SelectOption(label=tz, value=tz) for tz in common_tzs]
        opts.append(discord.SelectOption(label="Set Custom Timezone...", value="custom"))
        select = ui.Select(placeholder="Choose a timezone...", options=opts)
        async def tz_cb(i: discord.Interaction):
            if select.values[0] == "custom":
                modal = ui.Modal(title="Set Custom Timezone")
                inp = ui.TextInput(label="Enter IANA Timezone (e.g. Asia/Tokyo)", required=True)
                modal.add_item(inp)
                async def custom_sub(mi: discord.Interaction):
                    try: 
                        ZoneInfo(inp.value); profile['timezone'] = inp.value
                        self.cog._save_user_data_entry(self.user_id, user_data)
                        new_embed = await self.cog._build_profile_manage_embed(mi, self.profile_name)
                        await self.original_interaction.edit_original_response(embed=new_embed, view=self)
                        await mi.response.send_message("Updated.", ephemeral=True, delete_after=3)
                    except: await mi.response.send_message("Invalid.", ephemeral=True, delete_after=5)
                modal.on_submit = custom_sub
                await i.response.send_modal(modal)
            else:
                profile['timezone'] = select.values[0]
                self.cog._save_user_data_entry(self.user_id, user_data)
                new_embed = await self.cog._build_profile_manage_embed(i, self.profile_name)
                await self.original_interaction.edit_original_response(embed=new_embed, view=self)
                await i.response.defer()
        select.callback = tz_cb
        view.add_item(select)
        await interaction.response.send_message("Select Timezone:", view=view, ephemeral=True)

    async def on_timeout(self):
        try: await self.original_interaction.edit_original_response(content="Manager timed out.", view=None)
        except: pass

def is_owner_in_dm_check(): 
    async def predicate(interaction: discord.Interaction) -> bool:
        if interaction.guild is not None:
            return False
        return interaction.user.id == int(defaultConfig.DISCORD_OWNER_ID)
    return app_commands.check(predicate)

class RedeemCodeModal(ui.Modal, title="Redeem a Share Code"):
    share_code_input = ui.TextInput(label="Enter the share code", required=True, min_length=12, max_length=16)

    def __init__(self, cog: 'GeminiAgent'):
        super().__init__()
        self.cog = cog

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        code = self.share_code_input.value.strip()
        
        share_data = self.cog.share_codes.get(code)
        if not share_data or time.time() > share_data["expires_at"]:
            await interaction.followup.send("This share code is invalid or has expired.", ephemeral=True)
            return
        
        owner_id_str = share_data["owner_id"]
        profiles_to_borrow = share_data["profile_name"]
        if not isinstance(profiles_to_borrow, list):
            profiles_to_borrow = [profiles_to_borrow]

        if owner_id_str == str(interaction.user.id):
            await interaction.followup.send("You cannot borrow a profile from yourself.", ephemeral=True)
            return

        owner = await self.cog.bot.fetch_user(int(owner_id_str))
        sharer_name = owner.name if owner else "Unknown"

        # [NEW] Dynamic Limit Check
        user_data = self.cog._get_user_data_entry(interaction.user.id)
        current_borrowed = len(user_data.get("borrowed_profiles", {}))
        
        is_premium = self.cog.is_user_premium(interaction.user.id)
        limit = defaultConfig.LIMIT_BORROWED_PREMIUM if is_premium else defaultConfig.LIMIT_BORROWED_FREE

        if current_borrowed + len(profiles_to_borrow) > limit:
            tier_name = "Premium" if is_premium else "Free"
            await interaction.followup.send(
                f"**Limit Reached.**\n"
                f"Redeeming this code would put you at {current_borrowed + len(profiles_to_borrow)}/{limit} borrowed profiles ({tier_name} Tier).\n"
                f"Please delete some profiles or upgrade to Premium.", 
                ephemeral=True
            )
            return

        accepted_profiles = []
        failed_profiles = {}

        for profile_name in profiles_to_borrow:
            # Lazy check: ensure source still exists
            owner_data = self.cog._get_user_data_entry(int(owner_id_str))
            owner_profile_data = owner_data.get("profiles", {}).get(profile_name)
            
            if not owner_profile_data:
                failed_profiles[profile_name] = "Original profile deleted by owner."
                continue

            desired_name = self.cog._generate_unique_local_name(interaction.user.id, profile_name, sharer_name)

            snapshot_data = {
                "original_owner_id": owner_id_str,
                "original_profile_name": profile_name,
                "borrowed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "grounding_enabled": owner_profile_data.get("grounding_enabled", False),
                "realistic_typing_enabled": owner_profile_data.get("realistic_typing_enabled", False),
                "time_tracking_enabled": owner_profile_data.get("time_tracking_enabled", False),
                "timezone": owner_profile_data.get("timezone", "UTC"),
                "ltm_creation_enabled": False, 
                "ltm_scope": "server", 
                "safety_level": "low"
            }
            user_data.setdefault("borrowed_profiles", {})[desired_name] = snapshot_data
            accepted_profiles.append(f"`{profile_name}` (as `{desired_name}`)")

        # Save and Delete Code
        if accepted_profiles:
            self.cog._save_user_data_entry(interaction.user.id, user_data)
            del self.cog.share_codes[code]
        
        message = ""
        if accepted_profiles:
            message += f"✅ **Successfully redeemed code!**\nBorrowed: {', '.join(accepted_profiles)}"
        if failed_profiles:
            message += f"\n\n⚠️ **Issues:**\n" + "\n".join([f"`{p}`: {r}" for p, r in failed_profiles.items()])
        
        await interaction.followup.send(message, ephemeral=True)

class HubBaseView(ui.View):
    """Base class for all Hub views to maintain persistent navigation."""
    def __init__(self, cog: 'GeminiAgent', interaction: discord.Interaction, current_tab: str):
        super().__init__(timeout=600)
        self.cog = cog
        self.original_interaction = interaction
        self.user_id = interaction.user.id
        self.current_tab = current_tab
        self._add_nav_buttons()

    def _add_nav_buttons(self):
        # Row 4 is reserved for Navigation
        # Home | Library | Incoming | Share
        
        btn_home = ui.Button(label="Home", style=discord.ButtonStyle.primary if self.current_tab == "home" else discord.ButtonStyle.secondary, row=4, disabled=(self.current_tab == "home"))
        btn_home.callback = self.nav_home
        
        btn_lib = ui.Button(label="Public Library", style=discord.ButtonStyle.primary if self.current_tab == "library" else discord.ButtonStyle.secondary, row=4, disabled=(self.current_tab == "library"))
        btn_lib.callback = self.nav_library
        
        btn_inc = ui.Button(label="Incoming Shares", style=discord.ButtonStyle.primary if self.current_tab == "incoming" else discord.ButtonStyle.secondary, row=4, disabled=(self.current_tab == "incoming"))
        btn_inc.callback = self.nav_incoming
        
        btn_share = ui.Button(label="Manage My Shares", style=discord.ButtonStyle.primary if self.current_tab == "manage" else discord.ButtonStyle.secondary, row=4, disabled=(self.current_tab == "manage"))
        btn_share.callback = self.nav_manage

        self.add_item(btn_home)
        self.add_item(btn_lib)
        self.add_item(btn_inc)
        self.add_item(btn_share)

    async def nav_home(self, i: discord.Interaction):
        await i.response.defer()
        view = HubHomeView(self.cog, self.original_interaction)
        await view.update_display()

    async def nav_library(self, i: discord.Interaction):
        await i.response.defer()
        view = HubPublicLibraryView(self.cog, self.original_interaction)
        await view.update_display()

    async def nav_incoming(self, i: discord.Interaction):
        await i.response.defer()
        view = HubIncomingView(self.cog, self.original_interaction)
        await view.update_display()

    async def nav_manage(self, i: discord.Interaction):
        await i.response.defer()
        view = HubShareManagerView(self.cog, self.original_interaction)
        await view.update_display()

class HubHomeView(HubBaseView):
    def __init__(self, cog: 'GeminiAgent', interaction: discord.Interaction):
        super().__init__(cog, interaction, "home")

    async def update_display(self):
        # Logic to gather stats
        total_public = len(self.cog.public_profiles)
        unique_creators = len(set(p['owner_id'] for p in self.cog.public_profiles.values()))
        
        user_data = self.cog._get_user_data_entry(self.user_id)
        user_owned = len(user_data.get("profiles", {}))
        user_borrowed = len(user_data.get("borrowed_profiles", {}))
        
        embed = discord.Embed(title="MimicAI Profile Hub", description=defaultConfig.MIMIC_NEWS, color=discord.Color.gold())
        
        # Use animated emote as the Hub icon
        embed.set_thumbnail(url="https://cdn.discordapp.com/emojis/1441750712160878643.gif")
        
        embed.add_field(name="Global Stats", value=f"`{total_public} Public Profiles`\n`{unique_creators} Creators`", inline=True)
        embed.add_field(name="Your Stats", value=f"`{user_owned} Personal Profiles`\n`{user_borrowed} Borrowed Profiles`", inline=True)
        
        embed.set_footer(text="Use the navigation buttons below to explore.")

        await self.original_interaction.edit_original_response(content=None, embed=embed, view=self)

    async def redeem_callback(self, i: discord.Interaction):
        modal = RedeemCodeModal(self.cog)
        await i.response.send_modal(modal)

class HubPublicLibraryView(HubBaseView):
    def __init__(self, cog: 'GeminiAgent', interaction: discord.Interaction, filtered_list=None):
        super().__init__(cog, interaction, "library")
        self.all_public = []
        self._load_public_data()
        self.filtered_list = filtered_list if filtered_list is not None else self.all_public
        
        self.current_page = 0 
        self.selected_index_on_page = 0
        
        self.setup_items()

    def _load_public_data(self):
        raw_list = []
        # Iterate over the in-memory index
        for p_id, p_info in self.cog.public_profiles.items():
            if p_info.get("status", "active") == "locked": continue
            if not p_info.get("owner_id") or not p_info.get("original_profile_name"): continue
            
            # Use snapshot metadata if available, otherwise fallback (for old data)
            raw_list.append({
                "id": p_id,
                "owner_id": p_info['owner_id'],
                "profile_name": p_info['original_profile_name'],
                "published_at": p_info.get("published_at", ""),
                "display_name": p_info.get("display_name", p_info['original_profile_name']),
                "avatar_url": p_info.get("avatar_url")
            })
        
        raw_list.sort(key=lambda x: x['published_at'], reverse=True)
        self.all_public = raw_list

    def setup_items(self):
        for item in self.children[:]:
            if item.row != 4: self.remove_item(item)

        if not self.filtered_list:
            # If list is empty, still show Search button so user can reset/change filters
            search_btn = ui.Button(label="Search / Sort", style=discord.ButtonStyle.secondary, row=1)
            search_btn.callback = self.search_cb
            self.add_item(search_btn)
            return

        num_pages = (len(self.filtered_list) - 1) // DROPDOWN_MAX_OPTIONS + 1
        if self.current_page >= num_pages: self.current_page = max(0, num_pages - 1)
        
        start = self.current_page * DROPDOWN_MAX_OPTIONS
        page_items = self.filtered_list[start : start + DROPDOWN_MAX_OPTIONS]
        
        # Row 0: Dropdown
        options = []
        for i, p in enumerate(page_items):
            owner = self.cog.bot.get_user(p['owner_id'])
            owner_name = owner.name if owner else "Unknown"
            label = f"{p['profile_name']} (by {owner_name})"[:100]
            # Value is index on the page (0-24)
            option = discord.SelectOption(label=label, value=str(i), default=(i == self.selected_index_on_page))
            options.append(option)

        if options:
            select = ui.Select(placeholder="Select a profile to view...", options=options, min_values=1, max_values=1, row=0)
            select.callback = self.select_callback
            self.add_item(select)

        # Row 1: Pagination + Actions
        prev_btn = ui.Button(label="◀", style=discord.ButtonStyle.secondary, row=1, disabled=(self.current_page == 0))
        prev_btn.callback = self.prev_page_cb
        
        page_lbl = ui.Button(label=f"{self.current_page + 1}/{num_pages}", style=discord.ButtonStyle.grey, row=1, disabled=True)
        
        next_btn = ui.Button(label="▶", style=discord.ButtonStyle.secondary, row=1, disabled=(self.current_page >= num_pages - 1))
        next_btn.callback = self.next_page_cb
        
        # Determine Borrow Button State
        abs_index = (self.current_page * DROPDOWN_MAX_OPTIONS) + self.selected_index_on_page
        
        borrow_label = "Borrow"
        borrow_style = discord.ButtonStyle.green
        borrow_disabled = False

        if abs_index < len(self.filtered_list):
            p_info = self.filtered_list[abs_index]
            
            if self.user_id == p_info['owner_id']:
                borrow_label = "Own Profile"
                borrow_style = discord.ButtonStyle.grey
                borrow_disabled = True
            else:
                user_data = self.cog._get_user_data_entry(self.user_id)
                for b_data in user_data.get("borrowed_profiles", {}).values():
                    if int(b_data["original_owner_id"]) == p_info['owner_id'] and \
                       b_data["original_profile_name"] == p_info['profile_name']:
                        borrow_label = "Borrowed"
                        borrow_style = discord.ButtonStyle.grey
                        borrow_disabled = True
                        break

        borrow_btn = ui.Button(label=borrow_label, style=borrow_style, row=1, disabled=borrow_disabled)
        borrow_btn.callback = self.borrow_cb
        
        search_btn = ui.Button(label="Search / Sort", style=discord.ButtonStyle.secondary, row=1)
        search_btn.callback = self.search_cb

        self.add_item(prev_btn)
        self.add_item(page_lbl)
        self.add_item(next_btn)
        self.add_item(borrow_btn)
        self.add_item(search_btn)

    async def update_display(self):
        if not self.filtered_list:
            embed = discord.Embed(title="Public Library", description="No profiles found.", color=discord.Color.red())
            await self.original_interaction.edit_original_response(content=None, embed=embed, view=self)
            return

        abs_index = (self.current_page * DROPDOWN_MAX_OPTIONS) + self.selected_index_on_page
        if abs_index >= len(self.filtered_list):
            abs_index = 0
            self.selected_index_on_page = 0
        
        p_info = self.filtered_list[abs_index]
        owner_id = p_info['owner_id']
        
        owner = self.cog.bot.get_user(owner_id) or await self.cog.bot.fetch_user(owner_id)
        owner_name = owner.name if owner else "Unknown"
        
        # Use snapshot metadata directly
        disp_name = p_info.get("display_name")
        avatar_url = p_info.get("avatar_url")

        embed = discord.Embed(title=disp_name, description=f"Created by **{owner_name}**", color=discord.Color.random())
        if avatar_url: embed.set_image(url=avatar_url)
        embed.set_footer(text=f"ID: {p_info['id']} | {abs_index + 1} of {len(self.filtered_list)}")
        
        await self.original_interaction.edit_original_response(content=None, embed=embed, view=self)

    async def select_callback(self, i: discord.Interaction):
        self.selected_index_on_page = int(i.data['values'][0])
        self.setup_items()
        await i.response.defer()
        await self.update_display()

    async def prev_page_cb(self, i: discord.Interaction):
        self.current_page -= 1
        self.selected_index_on_page = 0
        self.setup_items()
        await i.response.defer()
        await self.update_display()

    async def next_page_cb(self, i: discord.Interaction):
        self.current_page += 1
        self.selected_index_on_page = 0
        self.setup_items()
        await i.response.defer()
        await self.update_display()

    async def borrow_cb(self, i: discord.Interaction):
        abs_index = (self.current_page * DROPDOWN_MAX_OPTIONS) + self.selected_index_on_page
        if abs_index >= len(self.filtered_list): return
        
        p_info = self.filtered_list[abs_index]
        if i.user.id == p_info['owner_id']:
            await i.response.send_message("You cannot borrow your own profile.", ephemeral=True)
            return
        
        user_data = self.cog._get_user_data_entry(i.user.id)
        for b_data in user_data.get("borrowed_profiles", {}).values():
            if int(b_data["original_owner_id"]) == p_info['owner_id'] and b_data["original_profile_name"] == p_info['profile_name']:
                await i.response.send_message("You already have this profile.", ephemeral=True)
                return

        modal = BorrowNameModal(self.cog, self.original_interaction, p_info['owner_id'], p_info['profile_name'], is_public_borrow=True)
        await i.response.send_modal(modal)

    async def search_cb(self, i: discord.Interaction):
        modal = ui.Modal(title="Search Public Library")
        inp = ui.TextInput(label="Search Term", required=False)
        modal.add_item(inp)
        async def on_submit(mi: discord.Interaction):
            term = inp.value.lower()
            if term:
                self.filtered_list = [p for p in self.all_public if term in p['profile_name'].lower()]
                self.current_page = 0
                self.selected_index_on_page = 0
            else:
                self.filtered_list = self.all_public
            self.setup_items()
            await mi.response.defer()
            await self.update_display()
        modal.on_submit = on_submit
        await i.response.send_modal(modal)

class HubIncomingView(HubBaseView):
    def __init__(self, cog: 'GeminiAgent', interaction: discord.Interaction):
        super().__init__(cog, interaction, "incoming")
        self.selected_sharer_id = None
        self.current_page = 0
        self.setup_items()

    def setup_items(self):
        for item in self.children[:]:
            if item.row != 4: self.remove_item(item)

        shares = self.cog.profile_shares.get(str(self.user_id), [])
        
        # Group
        sharers = {}
        for s in shares:
            sharers.setdefault(s['sharer_id'], []).append(s)
        
        sharer_ids = list(sharers.keys())
        num_pages = (len(sharer_ids) - 1) // DROPDOWN_MAX_OPTIONS + 1
        if self.current_page >= num_pages: self.current_page = max(0, num_pages - 1)
        
        start = self.current_page * DROPDOWN_MAX_OPTIONS
        page_ids = sharer_ids[start : start + DROPDOWN_MAX_OPTIONS]

        # Row 0: Dropdown
        if page_ids:
            options = []
            for sid in page_ids:
                u = self.cog.bot.get_user(sid)
                name = u.name if u else f"ID: {sid}"
                count = len(sharers[sid])
                options.append(discord.SelectOption(label=f"{name} ({count} profiles)", value=str(sid), default=(sid == self.selected_sharer_id)))

            select = ui.Select(placeholder="Select a user to review shares...", options=options, min_values=1, max_values=1, row=0)
            select.callback = self.select_sharer
            self.add_item(select)

        # Row 1: Pagination (if needed)
        if num_pages > 1:
            prev_btn = ui.Button(label="◀", style=discord.ButtonStyle.secondary, row=1, disabled=(self.current_page == 0))
            page_lbl = ui.Button(label=f"{self.current_page + 1}/{num_pages}", style=discord.ButtonStyle.grey, row=1, disabled=True)
            next_btn = ui.Button(label="▶", style=discord.ButtonStyle.secondary, row=1, disabled=(self.current_page >= num_pages - 1))
            
            prev_btn.callback = self.prev_page
            next_btn.callback = self.next_page
            
            self.add_item(prev_btn)
            self.add_item(page_lbl)
            self.add_item(next_btn)

        # Row 2: Actions
        action_row = 2 if num_pages > 1 else 1 # Move up if no pagination
        
        if self.selected_sharer_id:
            acc_btn = ui.Button(label="Accept All", style=discord.ButtonStyle.green, row=action_row)
            rej_btn = ui.Button(label="Reject All", style=discord.ButtonStyle.danger, row=action_row)
            back_btn = ui.Button(label="Cancel Selection", style=discord.ButtonStyle.grey, row=action_row)
            
            acc_btn.callback = self.accept_all
            rej_btn.callback = self.reject_all
            back_btn.callback = self.clear_selection
            
            self.add_item(acc_btn)
            self.add_item(rej_btn)
            self.add_item(back_btn)
        else:
            redeem_btn = ui.Button(label="Redeem Share Code", style=discord.ButtonStyle.secondary, row=action_row, emoji="🔑")
            redeem_btn.callback = self.redeem_code_callback
            self.add_item(redeem_btn)

    async def update_display(self):
        shares = self.cog.profile_shares.get(str(self.user_id), [])
        
        if self.selected_sharer_id:
            u = self.cog.bot.get_user(self.selected_sharer_id)
            name = u.name if u else "Unknown"
            user_shares = [s['profile_name'] for s in shares if s['sharer_id'] == self.selected_sharer_id]
            desc = f"**Pending shares from {name}:**\n" + ", ".join([f"`{n}`" for n in user_shares])
            embed = discord.Embed(title="Reviewing Shares", description=desc, color=discord.Color.blue())
        elif not shares:
            embed = discord.Embed(title="Incoming Shares", description="You have no direct share requests pending.\n\nHave a code? Click the button below.", color=discord.Color.dark_grey())
        else:
            embed = discord.Embed(title="Incoming Shares", description="Select a user from the dropdown to accept or reject their shared profiles.", color=discord.Color.blue())
            
        await self.original_interaction.edit_original_response(content=None, embed=embed, view=self)

    async def select_sharer(self, i: discord.Interaction):
        self.selected_sharer_id = int(i.data['values'][0])
        self.setup_items()
        await i.response.defer()
        await self.update_display()

    async def prev_page(self, i: discord.Interaction):
        self.current_page -= 1
        self.setup_items()
        await i.response.defer()
        await self.update_display()

    async def next_page(self, i: discord.Interaction):
        self.current_page += 1
        self.setup_items()
        await i.response.defer()
        await self.update_display()

    async def clear_selection(self, i: discord.Interaction):
        self.selected_sharer_id = None
        self.setup_items()
        await i.response.defer()
        await self.update_display()

    async def redeem_code_callback(self, i: discord.Interaction):
        modal = RedeemCodeModal(self.cog)
        await i.response.send_modal(modal)

    async def accept_all(self, i: discord.Interaction):
        await i.response.defer(ephemeral=True)
        sharer_id = self.selected_sharer_id
        shares = [s for s in self.cog.profile_shares.get(str(self.user_id), []) if s['sharer_id'] == sharer_id]
        
        limit = defaultConfig.LIMIT_BORROWED_PREMIUM if self.cog.is_user_premium(self.user_id) else defaultConfig.LIMIT_BORROWED_FREE
        user_data = self.cog._get_user_data_entry(self.user_id)
        current_borrowed = len(user_data.get("borrowed_profiles", {}))

        if current_borrowed + len(shares) > limit:
            await i.followup.send(f"Limit Reached. Accepting these would exceed your limit of {limit} borrowed profiles.", ephemeral=True)
            return

        accepted = []
        sharer_user = self.cog.bot.get_user(sharer_id)
        sharer_name = sharer_user.name if sharer_user else "User"

        for s in shares:
            p_name = s['profile_name']
            sharer_data = self.cog._get_user_data_entry(sharer_id)
            if p_name not in sharer_data.get("profiles", {}):
                await self.cog._reject_share_request(self.original_interaction, sharer_id, p_name, notify_sharer=False)
                continue

            local_name = self.cog._generate_unique_local_name(self.user_id, p_name, sharer_name)
            await self.cog._accept_share_request(self.original_interaction, sharer_id, p_name, local_name)
            accepted.append(p_name)
        
        msg = f"Accepted: {', '.join(accepted)}" if accepted else "No valid profiles found."
        await i.followup.send(msg, ephemeral=True)
        self.selected_sharer_id = None
        self.setup_items()
        await self.update_display()

    async def reject_all(self, i: discord.Interaction):
        await i.response.defer(ephemeral=True)
        sharer_id = self.selected_sharer_id
        shares = [s for s in self.cog.profile_shares.get(str(self.user_id), []) if s['sharer_id'] == sharer_id]
        for s in shares:
            await self.cog._reject_share_request(self.original_interaction, sharer_id, s['profile_name'], notify_sharer=False)
        await i.followup.send(f"Rejected shares.", ephemeral=True)
        self.selected_sharer_id = None
        self.setup_items()
        await self.update_display()

class HubShareManagerView(HubBaseView):
    def __init__(self, cog: 'GeminiAgent', interaction: discord.Interaction):
        super().__init__(cog, interaction, "manage")
        self.mode = "private"
        self.selected_profiles = []
        self.selected_users = []
        self.processing = False
        
        user_data = self.cog._get_user_data_entry(self.user_id)
        self.personal_profiles = sorted(list(user_data.get("profiles", {}).keys()))
        self.current_page = 0
        
        self.setup_items()

    def setup_items(self):
        for item in self.children[:]:
            if item.row != 4: self.remove_item(item)

        # Row 0: Mode
        style = discord.ButtonStyle.blurple if self.mode == "private" else discord.ButtonStyle.green
        label = "Mode: Private Sharing" if self.mode == "private" else "Mode: Public Publishing"
        toggle_btn = ui.Button(label=label, style=style, row=0)
        toggle_btn.callback = self.toggle_mode
        self.add_item(toggle_btn)

        # Row 1: Paginated Profiles
        num_pages = (len(self.personal_profiles) - 1) // DROPDOWN_MAX_OPTIONS + 1
        if self.current_page >= num_pages: self.current_page = max(0, num_pages - 1)
        
        start = self.current_page * DROPDOWN_MAX_OPTIONS
        page_profiles = self.personal_profiles[start : start + DROPDOWN_MAX_OPTIONS]
        
        options = []
        for p in page_profiles:
            options.append(discord.SelectOption(label=p, value=p, default=(p in self.selected_profiles)))
        
        if options:
            prof_sel = ui.Select(placeholder="Select profiles...", options=options, min_values=0, max_values=len(options), row=1)
            prof_sel.callback = self.select_profiles
            self.add_item(prof_sel)

        # Row 2: Pagination Buttons (if needed) AND Action Buttons
        # We share Row 2 for buttons.
        
        btn_items = []
        if num_pages > 1:
            prev_btn = ui.Button(label="◀", style=discord.ButtonStyle.secondary, row=2, disabled=(self.current_page == 0))
            page_lbl = ui.Button(label=f"{self.current_page + 1}/{num_pages}", style=discord.ButtonStyle.grey, row=2, disabled=True)
            next_btn = ui.Button(label="▶", style=discord.ButtonStyle.secondary, row=2, disabled=(self.current_page >= num_pages - 1))
            prev_btn.callback = self.prev_page
            next_btn.callback = self.next_page
            btn_items.extend([prev_btn, page_lbl, next_btn])

        if self.mode == "private":
            if len(btn_items) <= 3: # Fits on Row 2
                send_btn = ui.Button(label="Send", style=discord.ButtonStyle.green, row=2)
                code_btn = ui.Button(label="Get Code", style=discord.ButtonStyle.secondary, row=2)
                send_btn.callback = self.send_private
                code_btn.callback = self.generate_code
                btn_items.append(send_btn)
                btn_items.append(code_btn)
            else: # Full row, move to Row 3? But user select needs Row 3.
                # Prioritize functionality: Pagination + Send. Code button moves?
                pass 
        else:
            apply_btn = ui.Button(label="Apply Changes", style=discord.ButtonStyle.green, row=2)
            apply_btn.callback = self.apply_public
            btn_items.append(apply_btn)

        for btn in btn_items:
            self.add_item(btn)

        # Row 3: User Select (Private)
        if self.mode == "private":
            user_sel = ui.UserSelect(placeholder="Select recipients...", min_values=1, max_values=10, row=3)
            user_sel.callback = self.select_users
            self.add_item(user_sel)

    async def update_display(self):
        desc = "Manage how you share your profiles.\n\n"
        if self.mode == "private":
            desc += "**Private Mode:** Share specifically with friends via DM or Code."
        else:
            desc += "**Public Mode:** Publish your profiles to the global library for anyone to borrow."
            
        embed = discord.Embed(title="Share Manager", description=desc, color=discord.Color.teal())
        
        full_text = ", ".join(self.selected_profiles)
        if len(full_text) > 4000: full_text = full_text[:4000] + "..." # Prevent total embed failure
        if not full_text: full_text = "None"
        embed.add_field(name="Selected Profiles", value=full_text, inline=False)

        if self.mode == "public":
            public_names = []
            user_id_str = str(self.user_id)
            for p_info in self.cog.public_profiles.values():
                if str(p_info.get("owner_id")) == user_id_str:
                    status = p_info.get("status", "active")
                    suffix = " (Locked)" if status == "locked" else ""
                    public_names.append(f"{p_info['original_profile_name']}{suffix}")
            
            val = ", ".join(public_names) if public_names else "None"
            if len(val) > 1024: val = val[:1021] + "..."
            embed.add_field(name="Your Currently Public Profiles", value=val, inline=False)

        await self.original_interaction.edit_original_response(content=None, embed=embed, view=self)

    async def toggle_mode(self, i: discord.Interaction):
        # [UPDATED] Free users CAN toggle to Public to UNPUBLISH. Validation happens on Apply.
        if self.mode == "private":
            self.mode = "public"
            user_id_str = str(self.user_id)
            published = []
            for p_info in self.cog.public_profiles.values():
                if str(p_info.get("owner_id")) == user_id_str:
                    published.append(p_info['original_profile_name'])
            self.selected_profiles = published
        else:
            self.mode = "private"
            self.selected_profiles = []
        
        self.current_page = 0
        self.setup_items()
        await i.response.defer()
        await self.update_display()

    async def select_profiles(self, i: discord.Interaction):
        new_page_selections = set(i.data['values'])
        start = self.current_page * DROPDOWN_MAX_OPTIONS
        page_profiles = set(self.personal_profiles[start : start + DROPDOWN_MAX_OPTIONS])
        
        self.selected_profiles = [p for p in self.selected_profiles if p not in page_profiles]
        self.selected_profiles.extend(list(new_page_selections))
        self.selected_profiles.sort() # Keep tidy
        
        self.setup_items()
        await i.response.defer()
        await self.update_display()

    async def prev_page(self, i: discord.Interaction):
        self.current_page -= 1
        self.setup_items()
        await i.response.defer()
        await self.update_display()

    async def next_page(self, i: discord.Interaction):
        self.current_page += 1
        self.setup_items()
        await i.response.defer()
        await self.update_display()

    async def select_users(self, i: discord.Interaction):
        self.selected_users = i.data['values'] 
        await i.response.defer()

    async def send_private(self, i: discord.Interaction):
        if self.processing: return
        self.processing = True

        for item in self.children:
            if isinstance(item, ui.Button) and item.label in ["Send", "Send Request", "Get Code"]: item.disabled = True
        await i.response.edit_message(view=self)

        if not self.selected_profiles or not self.selected_users:
            await i.followup.send("Select profiles and recipients first.", ephemeral=True)
            self.processing = False
            self.setup_items()
            await self.update_display()
            return
        
        success_count = 0
        for recipient_id_str in self.selected_users:
            recipient_id = int(recipient_id_str)
            recipient = self.cog.bot.get_user(recipient_id)
            if not recipient or recipient.bot or recipient.id == self.user_id: continue

            self.cog.profile_shares.setdefault(str(recipient_id), [])
            newly_shared = []
            for profile_name in self.selected_profiles:
                existing = next((s for s in self.cog.profile_shares[str(recipient_id)] if s['sharer_id'] == self.user_id and s['profile_name'] == profile_name), None)
                if not existing:
                    share_req = {"sharer_id": self.user_id, "profile_name": profile_name, "shared_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
                    self.cog.profile_shares[str(recipient_id)].append(share_req)
                    newly_shared.append(profile_name)
            
            if newly_shared:
                try:
                    await recipient.send(f"**{self.original_interaction.user.name}** shared profile(s) with you: {', '.join(newly_shared)}. Check `/profile hub`.")
                    success_count += 1
                except discord.Forbidden: pass
            
            self.cog._save_profile_share_shard(str(recipient_id), self.cog.profile_shares[str(recipient_id)])

        self.processing = False
        self.setup_items()
        await self.update_display()
        await i.followup.send(f"Sent to {success_count} users.", ephemeral=True)

    async def generate_code(self, i: discord.Interaction):
        if not self.selected_profiles:
            await i.response.send_message("Select at least one profile.", ephemeral=True); return
        code = f"SHR-{uuid.uuid4().hex[:8].upper()}"
        self.cog.share_codes[code] = {"owner_id": str(self.user_id), "profile_name": self.selected_profiles, "expires_at": time.time() + 300}
        await i.response.send_message(f"Share Code: `{code}`\nExpires in 5 minutes.", ephemeral=True)

    async def apply_public(self, i: discord.Interaction):
        if self.processing: return
        self.processing = True

        for item in self.children:
            if isinstance(item, ui.Button) and item.label in ["Apply Changes", "Apply Publishing Changes"]: item.disabled = True
        await i.response.edit_message(view=self)

        analysis_message = None
        
        user_id_str = str(self.user_id)
        current_public_set = set()
        for p_info in self.cog.public_profiles.values():
            if str(p_info.get("owner_id")) == user_id_str:
                current_public_set.add(p_info['original_profile_name'])
        
        target_set = set(self.selected_profiles)
        to_publish = target_set - current_public_set
        to_unpublish = current_public_set - target_set
        
        if to_publish and not self.cog.is_user_premium(self.user_id):
            self.processing = False
            self.setup_items()
            await self.update_display()
            await i.followup.send("Adding new profiles to the public library is a Premium feature.", ephemeral=True)
            return

        published_list = []
        failed_list = {}

        if to_publish:
            analysis_message = await i.followup.send("🔍 Analyzing profiles for safety...", ephemeral=True)
            for name in to_publish:
                appearance_data = self.cog.user_appearances.get(user_id_str, {}).get(name, {})
                disp = appearance_data.get("custom_display_name") or name
                ava = appearance_data.get("custom_avatar_url")
                
                p_data = self.cog._get_user_data_entry(self.user_id).get("profiles", {}).get(name, {})
                if p_data.get("safety_level") == "unrestricted":
                    failed_list[name] = "Safety Level is 'Unrestricted 18+'. Only 'Low', 'Medium', or 'High' can be published."
                    continue

                is_safe, reason = await self.cog._is_profile_content_safe(name, disp, ava, self.original_interaction.guild_id)
                if is_safe:
                    pid = f"pub_{uuid.uuid4().hex[:8]}"
                    # Snapshot metadata for index
                    data = {
                        "owner_id": self.user_id, 
                        "original_profile_name": name, 
                        "published_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), 
                        "status": "active",
                        "display_name": disp,
                        "avatar_url": ava
                    }
                    self.cog.public_profiles[pid] = data
                    published_list.append(name)
                else:
                    failed_list[name] = reason
            
            if published_list:
                self.cog._save_public_index()

        unpublished_list = []
        for name in to_unpublish:
            ids = [pid for pid, info in self.cog.public_profiles.items() if str(info.get("owner_id")) == user_id_str and info.get("original_profile_name") == name]
            for pid in ids:
                del self.cog.public_profiles[pid]
                unpublished_list.append(name)
        
        if unpublished_list:
            self.cog._save_public_index()

        self.processing = False
        self.setup_items()
        await self.update_display()

        report_embed = discord.Embed(title="Publishing Report", color=discord.Color.blue())
        if published_list: report_embed.add_field(name=f"✅ Published ({len(published_list)})", value=", ".join(published_list), inline=False)
        if unpublished_list: report_embed.add_field(name=f"⛔ Unpublished ({len(unpublished_list)})", value=", ".join(unpublished_list), inline=False)
        if failed_list:
            errs = "\n".join([f"• **{n}**: {r}" for n, r in failed_list.items()])
            if len(errs) > 1000: errs = errs[:997] + "..."
            report_embed.add_field(name=f"⚠️ Failed ({len(failed_list)})", value=errs, inline=False)
            report_embed.color = discord.Color.orange()
        if not (published_list or unpublished_list or failed_list): report_embed.description = "No changes were made."

        if analysis_message:
            await analysis_message.edit(content=None, embed=report_embed)
        else:
            await i.followup.send(embed=report_embed, ephemeral=True)

class ShutdownConfirmView(ui.View):
    def __init__(self, cog: 'GeminiAgent'):
        super().__init__(timeout=60)
        self.cog = cog

    @ui.button(label="Yes, Shutdown", style=discord.ButtonStyle.danger)
    async def confirm_shutdown(self, interaction: discord.Interaction, button: ui.Button):
        await interaction.response.edit_message(content="Shutting down child bots and main instance...", view=None)
        print(f"Shutdown confirmed by user {interaction.user.id} for cog {self.cog.cog_id}.")

        # Explicitly shut down all child bots via the manager queue
        if hasattr(self.cog.bot, 'child_bot_config'):
            for bot_id in list(self.cog.bot.child_bot_config.keys()):
                await self.cog.manager_queue.put({
                    "action": "shutdown_bot",
                    "bot_id": bot_id
                })
        
        await asyncio.sleep(2) # Give the manager a moment to process

        for session_key, chat_session in self.cog.global_chat_sessions.items():
            self.cog._save_session_to_disk(session_key, 'global_chat', chat_session)
        
        for ch_id, session_data in self.cog.multi_profile_channels.items():
            if session_data.get("is_hydrated"):
                session_type = session_data.get("type", "multi")
                unified_log = session_data.get("unified_log")
                if unified_log is not None:
                    dummy_session_key = (ch_id, None, None)
                    self.cog._save_session_to_disk(dummy_session_key, session_type, unified_log)
                
                mapping_key = (session_type, ch_id)
                if mapping_key in self.cog.mapping_caches:
                    self.cog._save_mapping_to_disk(mapping_key, self.cog.mapping_caches[mapping_key])

        if self.cog.has_lock:
            try:
                if os.path.exists(COG_LOCK_FILE_PATH):
                    os.remove(COG_LOCK_FILE_PATH)
            except Exception as e:
                print(f"Error releasing lock file during shutdown: {e}")
        await self.cog.bot.close()

    async def on_timeout(self):
        pass

def is_admin_or_owner_check(): 
    async def predicate(interaction: discord.Interaction) -> bool:
        if interaction.guild:
            return interaction.user.guild_permissions.administrator
        return interaction.user.id == int(defaultConfig.DISCORD_OWNER_ID)
    return app_commands.check(predicate)

class EditUserProfilePersonaModal(ui.Modal): 
    def __init__(self, cog_instance, profile_name: str, current_persona_data: Dict[str, List[str]], user_id: int):
        self.cog_instance: GeminiAgent = cog_instance
        self.profile_name = profile_name
        self.user_id = user_id
        self.persona_sections_order = cog_instance.persona_modal_sections_order

        title = f"Edit Persona for Profile: '{profile_name}'"[:45]; super().__init__(title=title)

        for key in self.persona_sections_order:
            decrypted_content = "\n".join(self.cog_instance._decrypt_data(line) for line in current_persona_data.get(key, []))
            trunc_content = decrypted_content[:PERSONA_TEXT_INPUT_MAX_LENGTH]
            if len(decrypted_content) > PERSONA_TEXT_INPUT_MAX_LENGTH:
                print(f"Warn: Section '{key}' truncated for modal (profile: {profile_name}, user: {user_id}).")
            
            lbl=key.replace('_',' ').title(); lbl=lbl[:42]+"..." if len(lbl)>45 else lbl
            self.add_item(ui.TextInput(label=lbl,custom_id=f"persona_{key}",style=discord.TextStyle.paragraph,default=trunc_content,required=False,max_length=PERSONA_TEXT_INPUT_MAX_LENGTH))
            
    async def on_submit(self, i: discord.Interaction):
        if i.response.is_done():
            return
        await i.response.defer(ephemeral=True,thinking=True)
        updated_persona_data:Dict[str,List[str]]={c.custom_id.replace("persona_",""): c.value.splitlines() for c in self.children if isinstance(c,ui.TextInput)and c.custom_id} 
        
        success = await self.cog_instance.update_user_profile_persona(
            self.user_id, self.profile_name, updated_persona_data, i.channel_id
        )
        scope = f"your profile '{self.profile_name}'"
        message = f"Persona sections for {scope} {'updated' if success else 'update failed (max profiles reached or other issue)'}."
        
        if success:
            active_profile_for_channel = self.cog_instance._get_active_user_profile_name_for_channel(i.user.id, i.channel_id)
            if active_profile_for_channel == self.profile_name:
                message += " This profile is active in this channel; changes will apply on next interaction."
            else:
                message += f" Use `/profile swap profile_name:{self.profile_name}` to make it active in this channel."


        await i.followup.send(message, ephemeral=True)
    async def on_error(self, i:discord.Interaction, e:Exception): print(f"EditUserProfilePersonaModal err: {e}"); traceback.print_exc(); await i.followup.send('Form error.',ephemeral=True)

class EditUserProfileAIInstructionsModal(ui.Modal): 
    def __init__(self, cog_instance, profile_name: str, current_instr:str, user_id: int):
        self.cog:GeminiAgent=cog_instance
        self.profile_name = profile_name
        self.user_id = user_id
        
        title=f"Edit AI Instructions for Profile: '{profile_name}'"[:45]; super().__init__(title=title)
        
        p1, p2, p3, p4 = "", "", "", ""
        if isinstance(current_instr, list):
            if len(current_instr) >= 1: p1 = self.cog._decrypt_data(current_instr[0])
            if len(current_instr) >= 2: p2 = self.cog._decrypt_data(current_instr[1])
            if len(current_instr) >= 3: p3 = self.cog._decrypt_data(current_instr[2])
            if len(current_instr) >= 4: p4 = self.cog._decrypt_data(current_instr[3])

        self.add_item(ui.TextInput(label="Part 1",custom_id="ai_p1",style=discord.TextStyle.paragraph,default=p1,required=False,max_length=AI_INSTRUCTIONS_PART_MAX_LENGTH))
        self.add_item(ui.TextInput(label="Part 2",custom_id="ai_p2",style=discord.TextStyle.paragraph,default=p2,required=False,max_length=AI_INSTRUCTIONS_PART_MAX_LENGTH))
        self.add_item(ui.TextInput(label="Part 3",custom_id="ai_p3",style=discord.TextStyle.paragraph,default=p3,required=False,max_length=AI_INSTRUCTIONS_PART_MAX_LENGTH))
        self.add_item(ui.TextInput(label="Style Guide (Reserved For Training)",custom_id="ai_p4",style=discord.TextStyle.paragraph,default=p4,required=False,max_length=AI_INSTRUCTIONS_PART_MAX_LENGTH))

    async def on_submit(self, i:discord.Interaction):
        await i.response.defer(ephemeral=True,thinking=True)
        p1=next(c.value for c in self.children if c.custom_id=="ai_p1"); p2=next(c.value for c in self.children if c.custom_id=="ai_p2")
        p3=next(c.value for c in self.children if c.custom_id=="ai_p3"); p4=next(c.value for c in self.children if c.custom_id=="ai_p4") 
        instr_list = [p1, p2, p3, p4]
        success = await self.cog.update_user_profile_ai_instructions(
            self.user_id, self.profile_name, instr_list, i.channel_id
        )
        scope=f"your profile '{self.profile_name}'"
        message = f"AI Instructions for {scope} {'updated' if success else 'update failed (max profiles reached or other issue)'}."

        if success:
            active_profile_for_channel = self.cog._get_active_user_profile_name_for_channel(i.user.id, i.channel_id)
            if active_profile_for_channel == self.profile_name:
                message += " This profile is active in this channel; changes will apply on next interaction."
            else:
                message += f" Use `/profile swap profile_name:{self.profile_name}` to make it active in this channel."
        await i.followup.send(message,ephemeral=True)
    async def on_error(self, i:discord.Interaction,e:Exception): print(f"EditUserProfileAIInstrModal err: {e}"); traceback.print_exc(); await i.followup.send('Form error.',ephemeral=True)

class EditLtmModal(ui.Modal, title="Edit Long-Term Memory"):
    summary_field = ui.TextInput(label="Memory Summary", style=discord.TextStyle.paragraph, required=True, max_length=2000)

    def __init__(self, cog, profile_owner_id: int, profile_name: str, ltm_id: str, current_summary: str):
        super().__init__()
        self.cog: GeminiAgent = cog
        self.profile_owner_id = profile_owner_id
        self.profile_name = profile_name
        self.ltm_id = ltm_id
        self.summary_field.default = current_summary

    async def on_submit(self, i: discord.Interaction):
        await i.response.defer(ephemeral=True, thinking=True)
        new_summary = self.summary_field.value
        
        guild_id = i.guild_id
        if not guild_id:
            user = self.cog.bot.get_user(i.user.id)
            if user:
                for guild in self.cog.bot.guilds:
                    if guild.get_member(user.id):
                        guild_id = guild.id
                        break
        if not guild_id:
            await i.followup.send("Could not determine a valid context to get an API key. Please try editing from a server.", ephemeral=True)
            return

        new_embedding = await self.cog._get_embedding(new_summary, guild_id, task_type="RETRIEVAL_DOCUMENT")
        if not new_embedding:
            await i.followup.send("Failed to generate embedding for the new summary. The memory was not updated.", ephemeral=True)
            return

        quantized_embedding = _quantize_embedding(new_embedding)
        success = self.cog.update_ltm(self.profile_owner_id, self.profile_name, self.ltm_id, new_summary, quantized_embedding)
        if success:
            await i.followup.send(f"LTM entry `{self.ltm_id}` for profile '{self.profile_name}' has been updated.", ephemeral=True)
        else:
            await i.followup.send(f"Failed to find and update LTM entry `{self.ltm_id}`.", ephemeral=True)
    
    async def on_error(self, i: discord.Interaction, e: Exception):
        print(f"EditLtmModal error: {e}"); traceback.print_exc()
        await i.followup.send("An error occurred with the LTM edit form.", ephemeral=True)

class AddLtmModal(ui.Modal, title="Add Long-Term Memory"):
    summary_field = ui.TextInput(label="Memory Summary", style=discord.TextStyle.paragraph, required=True, max_length=2000)

    def __init__(self, cog, profile_owner_id: int, profile_name: str, guild_id: int):
        super().__init__()
        self.cog: GeminiAgent = cog
        self.profile_owner_id = profile_owner_id
        self.profile_name = profile_name
        self.guild_id = guild_id

    async def on_submit(self, i: discord.Interaction):
        await i.response.defer(ephemeral=True, thinking=True)
        
        # [NEW] Manual Hard Block Check
        user_id_str = str(self.profile_owner_id)
        ltm_shard = self.cog._load_ltm_shard(user_id_str, self.profile_name)
        current_count = len(ltm_shard.get("guild", [])) if ltm_shard else 0
        
        is_premium = self.cog.is_user_premium(self.profile_owner_id)
        limit = defaultConfig.LIMIT_LTM_PREMIUM if is_premium else defaultConfig.LIMIT_LTM_FREE
        
        if current_count >= limit:
            msg = f"**Limit Reached.**\n"
            msg += f"You have **{current_count}** memories (Limit: {limit}).\n"
            msg += "You cannot manually add more memories while at or above the limit. Please delete old memories first."
            if not is_premium:
                msg += f"\nOr upgrade to Premium via `/subscription` to increase your limit to {defaultConfig.LIMIT_LTM_PREMIUM}."
            await i.followup.send(msg, ephemeral=True)
            return

        summary = self.summary_field.value
        
        embedding = await self.cog._get_embedding(summary, self.guild_id, task_type="RETRIEVAL_DOCUMENT")
        if not embedding:
            await i.followup.send("Failed to generate embedding for the summary. The memory was not added.", ephemeral=True)
            return

        quantized_embedding = _quantize_embedding(embedding)
        
        # The _add_ltm method now handles the rolling window logic automatically.
        self.cog._add_ltm(self.profile_owner_id, self.profile_name, summary, quantized_embedding, self.guild_id, i.user.id, i.user.display_name)
        
        # Fetch new count for feedback
        ltm_shard = self.cog._load_ltm_shard(str(self.profile_owner_id), self.profile_name)
        count = len(ltm_shard.get("guild", [])) if ltm_shard else 0
        limit = defaultConfig.LIMIT_LTM_PREMIUM if self.cog.is_user_premium(self.profile_owner_id) else defaultConfig.LIMIT_LTM_FREE
        
        msg = f"LTM entry added for '{self.profile_name}'."
        if count >= limit:
            msg += f"\nNote: You have reached the {limit} memory limit. The oldest memory was automatically replaced."
            
        await i.followup.send(msg, ephemeral=True)

    async def on_error(self, i: discord.Interaction, e: Exception):
        print(f"AddLtmModal error: {e}"); traceback.print_exc()
        await i.followup.send("An error occurred with the LTM add form.", ephemeral=True)

class AddTrainingExampleModal(ui.Modal, title="Add Profile Training Example"): 
    user_input_field=ui.TextInput(label="User Input Example",style=discord.TextStyle.paragraph,required=True,max_length=1000)
    chatbot_response_field=ui.TextInput(label="Desired Chatbot Response",style=discord.TextStyle.paragraph,required=True,max_length=2000)
    def __init__(self, cog, profile_owner_id: int, profile_name: str, guild_id: int):
        super().__init__()
        self.cog:GeminiAgent=cog
        self.profile_owner_id = profile_owner_id
        self.profile_name = profile_name
        self.guild_id = guild_id
    async def on_submit(self,i:discord.Interaction):
        await i.response.defer(ephemeral=True,thinking=True)
        
        # [NEW] Manual Hard Block Check
        # Although add_new_training_example has a check, we do it here to provide a better UI response
        # and prevent the embedding API call if blocked.
        user_id_str = str(self.profile_owner_id)
        training_shard = self.cog._load_training_shard(user_id_str, self.profile_name) or []
        current_count = len(training_shard)
        
        is_premium = self.cog.is_user_premium(self.profile_owner_id)
        limit = defaultConfig.LIMIT_TRAINING_PREMIUM if is_premium else defaultConfig.LIMIT_TRAINING_FREE
        
        if current_count >= limit:
            msg = f"**Limit Reached.**\n"
            msg += f"You have **{current_count}** training examples (Limit: {limit}).\n"
            msg += "You cannot add more examples. Please delete existing ones first."
            if not is_premium:
                msg += f"\nOr upgrade to Premium via `/subscription` to increase your limit to {defaultConfig.LIMIT_TRAINING_PREMIUM}."
            await i.followup.send(msg, ephemeral=True)
            return

        s,m=await self.cog.add_new_training_example(self.profile_owner_id, self.profile_name, self.user_input_field.value, self.chatbot_response_field.value, self.guild_id)
        await i.followup.send(m,ephemeral=True)
    async def on_error(self,i:discord.Interaction,e:Exception):print(f"AddTrainExModal err:{e}");traceback.print_exc();await i.followup.send('Oops!',ephemeral=True)

class EditTrainingExampleModal(ui.Modal, title="Edit Profile Training Example"):
    user_input_field = ui.TextInput(label="User Input Example", style=discord.TextStyle.paragraph, required=True, max_length=1000)
    chatbot_response_field = ui.TextInput(label="Desired Chatbot Response", style=discord.TextStyle.paragraph, required=True, max_length=2000)

    def __init__(self, cog, profile_owner_id: int, profile_name: str, example_id: str, current_user_input: str, current_bot_response: str, guild_id: int):
        super().__init__()
        self.cog: GeminiAgent = cog
        self.profile_owner_id = profile_owner_id
        self.profile_name = profile_name
        self.example_id = example_id
        self.guild_id = guild_id
        self.user_input_field.default = current_user_input
        self.chatbot_response_field.default = current_bot_response

    async def on_submit(self,i:discord.Interaction):
        await i.response.defer(ephemeral=True,thinking=True)
        s,m=await self.cog.update_training_example(self.profile_owner_id, self.profile_name, self.example_id, self.user_input_field.value, self.chatbot_response_field.value, self.guild_id)
        await i.followup.send(m,ephemeral=True)

    async def on_error(self, i: discord.Interaction, e: Exception):
        print(f"EditTrainingExampleModal error: {e}")
        traceback.print_exc()
        await i.followup.send("An error occurred with the edit form.", ephemeral=True)

class ProfileParamsModal(ui.Modal, title="Set Profile Generation Parameters"):
    def __init__(self, cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, callback=None):
        super().__init__()
        self.cog: GeminiAgent = cog
        self.profile_name = profile_name
        self.is_borrowed = is_borrowed
        self.callback = callback

        self.add_item(ui.TextInput(label="Temperature (0.0-2.0)", custom_id="temperature", default=str(current_params.get("temperature", defaultConfig.GEMINI_TEMPERATURE)), required=False))
        self.add_item(ui.TextInput(label="Top P (0.0-1.0)", custom_id="top_p", default=str(current_params.get("top_p", defaultConfig.GEMINI_TOP_P)), required=False))
        self.add_item(ui.TextInput(label="Top K (integer 0-100)", custom_id="top_k", default=str(current_params.get("top_k", defaultConfig.GEMINI_TOP_K)), required=False))
        self.add_item(ui.TextInput(label=f"STM Length (0-{STM_LIMIT_MAX})", custom_id="stm_length", default=str(current_params.get("stm_length", defaultConfig.CHATBOT_MEMORY_LENGTH)), required=False))

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        
        new_params = {}
        try:
            temp_str = next((c.value for c in self.children if c.custom_id == "temperature"), None)
            if temp_str: 
                val = float(temp_str)
                if not (0.0 <= val <= 2.0): raise ValueError("Temperature out of range")
                new_params["temperature"] = val
            
            topp_str = next((c.value for c in self.children if c.custom_id == "top_p"), None)
            if topp_str: 
                val = float(topp_str)
                if not (0.0 <= val <= 1.0): raise ValueError("Top P out of range")
                new_params["top_p"] = val

            topk_str = next((c.value for c in self.children if c.custom_id == "top_k"), None)
            if topk_str: 
                val = int(topk_str)
                if not (0 <= val <= 100): raise ValueError("Top K out of range")
                new_params["top_k"] = val

            stm_str = next((c.value for c in self.children if c.custom_id == "stm_length"), None)
            if stm_str:
                val = int(stm_str)
                if not (0 <= val <= STM_LIMIT_MAX): raise ValueError(f"STM Length out of range (0-{STM_LIMIT_MAX})")
                new_params["stm_length"] = val

        except ValueError as e:
            await interaction.followup.send(f"❌ **Invalid Input:** {e}. Please check your values.", ephemeral=True)
            return
        except Exception as e:
            print(f"Error parsing params: {e}")
            await interaction.followup.send("❌ Error parsing input.", ephemeral=True)
            return

        if self.profile_name == "BULK_APPLY":
            # Bulk logic handled by caller if needed, but usually this modal isn't used for bulk directly like this
            pass
        else:
            try:
                success = await self.cog.update_profile_generation_params(interaction.user.id, self.profile_name, new_params, interaction.channel_id, self.is_borrowed)
                if success:
                    await interaction.followup.send(f"✅ Generation parameters updated for '{self.profile_name}'.", ephemeral=True)
                    if self.callback: 
                        try: await self.callback(interaction)
                        except Exception as e: print(f"Callback error in ProfileParamsModal: {e}")
                else:
                    await interaction.followup.send(f"❌ Failed to update parameters for '{self.profile_name}'.", ephemeral=True)
            except Exception as e:
                print(f"Error updating params: {e}")
                traceback.print_exc()
                await interaction.followup.send("❌ An unexpected error occurred while saving.", ephemeral=True)

class ProfileTrainingParamsModal(ui.Modal, title="Set Profile Training Parameters"):
    def __init__(self, cog, profile_name: str, current_params: Dict[str, Any], callback=None):
        super().__init__()
        self.cog: GeminiAgent = cog
        self.profile_name = profile_name
        self.callback = callback
        self.add_item(ui.TextInput(label="Context Size (0-10)", custom_id="training_context_size", default=str(current_params.get("training_context_size", defaultConfig.TRAINING_CONTEXT_SIZE)), required=False))
        self.add_item(ui.TextInput(label="Relevance Threshold (0.0-1.0)", custom_id="training_relevance_threshold", default=str(current_params.get("training_relevance_threshold", defaultConfig.TRAINING_RELEVANCE_THRESHOLD)), required=False))

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        new_params = {}
        try:
            ctx_str = next((c.value for c in self.children if c.custom_id == "training_context_size"), None)
            if ctx_str: 
                val = int(ctx_str)
                if not (0 <= val <= 10): raise ValueError("Context Size out of range (0-10)")
                new_params["training_context_size"] = val
            
            rel_str = next((c.value for c in self.children if c.custom_id == "training_relevance_threshold"), None)
            if rel_str: 
                val = float(rel_str)
                if not (0.0 <= val <= 1.0): raise ValueError("Relevance Threshold out of range (0.0-1.0)")
                new_params["training_relevance_threshold"] = val
        except ValueError as e:
            await interaction.followup.send(f"❌ **Invalid Input:** {e}.", ephemeral=True)
            return

        if self.profile_name == "BULK_APPLY":
            pass
        else:
            success = await self.cog.update_profile_training_params(interaction.user.id, self.profile_name, new_params)
            if success:
                await interaction.followup.send(f"✅ Training parameters updated for '{self.profile_name}'.", ephemeral=True)
                if self.callback: await self.callback(interaction)
            else:
                await interaction.followup.send(f"❌ Failed to update training parameters for '{self.profile_name}'.", ephemeral=True)

class ProfileThinkingParamsModal(ui.Modal, title="Thinking & Reasoning Parameters"):
    def __init__(self, cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, callback=None):
        super().__init__()
        self.cog = cog
        self.profile_name = profile_name
        self.is_borrowed = is_borrowed
        self.callback = callback

        self.add_item(ui.TextInput(
            label="Thinking Summary (on/off)", 
            custom_id="thinking_summary_visible", 
            default=current_params.get("thinking_summary_visible", "off"), 
            placeholder="Display reasoning tokens below your message.",
            required=False
        ))
        self.add_item(ui.TextInput(
            label="Reasoning Effort / Level", 
            custom_id="thinking_level", 
            default=current_params.get("thinking_level", "high"), 
            placeholder="xhigh, high, medium, low, minimal, none",
            required=False
        ))
        self.add_item(ui.TextInput(
            label="Reasoning Token Budget (Gemini 2.5 Only)", 
            custom_id="thinking_budget", 
            default=str(current_params.get("thinking_budget", -1)), 
            placeholder="-1 = dynamic, 128+ = token limit",
            required=False
        ))

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        new_params = {}
        try:
            def get_v(cid): return next(c.value for c in self.children if c.custom_id == cid).strip().lower()
            
            s_val = get_v("thinking_summary_visible")
            if s_val not in ["on", "off"]: s_val = "off"
            new_params["thinking_summary_visible"] = s_val
            
            l_val = get_v("thinking_level")
            # [UPDATED] Validation for all 6 standardized effort levels
            if l_val not in ["xhigh", "high", "medium", "low", "minimal", "none"]:
                l_val = "high"
            new_params["thinking_level"] = l_val
            
            b_val = int(get_v("thinking_budget"))
            if b_val < -1: b_val = -1
            new_params["thinking_budget"] = min(b_val, 32768)

        except ValueError as e:
            await interaction.followup.send(f"❌ **Invalid Input:** {e}", ephemeral=True); return

        if self.profile_name == "BULK_APPLY":
            return

        user_data = self.cog._get_user_data_entry(interaction.user.id)
        
        target_dict = user_data.get("borrowed_profiles" if self.is_borrowed else "profiles", {})
        profile = target_dict.get(self.profile_name)
        
        if profile:
            profile.update(new_params)
            self.cog._save_user_data_entry(interaction.user.id, user_data)
            
            cache_key = (interaction.channel_id, interaction.user.id, self.profile_name)
            self.cog.channel_models.pop(cache_key, None)
            self.cog.chat_sessions.pop(cache_key, None)
            self.cog.channel_model_last_profile_key.pop(cache_key, None)

            await interaction.followup.send(f"✅ Thinking parameters updated for '{self.profile_name}'.", ephemeral=True)
            if self.callback: await self.callback(interaction)

class ProfileLTMParamsModal(ui.Modal, title="LTM Parameters"):
    def __init__(self, cog, profile_name: str, current_params: Dict[str, Any], callback=None):
        super().__init__()
        self.cog: GeminiAgent = cog
        self.profile_name = profile_name
        self.callback = callback
        
        self.add_item(ui.TextInput(label="Creation Interval (5-100 msgs)", custom_id="ltm_creation_interval", default=str(current_params.get("ltm_creation_interval", 10)), required=False, placeholder="Default: 10"))
        self.add_item(ui.TextInput(label="Summarization Context (5-50 msgs)", custom_id="ltm_summarization_context", default=str(current_params.get("ltm_summarization_context", 10)), required=False, placeholder="Default: 10"))
        self.add_item(ui.TextInput(label="Recall Context Size (0-10)", custom_id="ltm_context_size", default=str(current_params.get("ltm_context_size", 3)), required=False, placeholder="Default: 3"))
        self.add_item(ui.TextInput(label="Relevance Threshold (0.0-1.0)", custom_id="ltm_relevance_threshold", default=str(current_params.get("ltm_relevance_threshold", 0.75)), required=False, placeholder="Default: 0.75"))

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        new_params = {}
        try:
            def parse(cid, default, min_v, max_v, is_float=False):
                val = next(c.value for c in self.children if c.custom_id == cid).strip()
                if not val: return default
                res = float(val) if is_float else int(val)
                if not (min_v <= res <= max_v): raise ValueError(f"{cid} out of range")
                return res

            new_params["ltm_creation_interval"] = parse("ltm_creation_interval", 10, 5, 100)
            new_params["ltm_summarization_context"] = parse("ltm_summarization_context", 10, 5, 50)
            new_params["ltm_context_size"] = parse("ltm_context_size", 3, 0, 10)
            new_params["ltm_relevance_threshold"] = parse("ltm_relevance_threshold", 0.75, 0.0, 1.0, True)
        except ValueError as e:
            await interaction.followup.send(f"❌ **Invalid Input:** {e}.", ephemeral=True); return

        user_data = self.cog._get_user_data_entry(interaction.user.id)
        profile = user_data.get("profiles", {}).get(self.profile_name)
        if profile:
            profile.update(new_params)
            self.cog._save_user_data_entry(interaction.user.id, user_data)
            await interaction.followup.send(f"✅ LTM parameters updated for '{self.profile_name}'.", ephemeral=True)
            if self.callback: await self.callback(interaction)
            else:
                await interaction.followup.send(f"❌ Failed to update LTM parameters for '{self.profile_name}'.", ephemeral=True)

class ProfileLTMSummarizationModal(ui.Modal, title="Set LTM Summarization Instructions"):
    def __init__(self, cog, profile_name: str, current_instructions: str):
        super().__init__()
        self.cog: GeminiAgent = cog
        self.profile_name = profile_name
        decrypted_instructions = self.cog._decrypt_data(current_instructions)
        self.instructions_input = ui.TextInput(
            label="AI Instructions for Summarization",
            style=discord.TextStyle.paragraph,
            default=decrypted_instructions,
            required=True,
            max_length=2000,
            placeholder="The system will automatically append the conversation excerpt to these instructions."
        )
        self.add_item(self.instructions_input)

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        new_instructions = self.instructions_input.value.strip()
        if not new_instructions:
            new_instructions = DEFAULT_LTM_SUMMARIZATION_INSTRUCTIONS
        
        user_id = interaction.user.id
        user_data = self.cog.user_profiles.get(str(user_id))
        profile = user_data.get("profiles", {}).get(self.profile_name)
        
        if not profile:
            await interaction.followup.send("Profile not found.", ephemeral=True)
            return

        profile["ltm_summarization_instructions"] = new_instructions
        self.cog._save_user_data_entry(user_id, user_data)
        
        
        await interaction.followup.send(f"LTM summarization instructions updated for profile '{self.profile_name}'.", ephemeral=True)

class ProfileLTMTriggerModal(ui.Modal, title="Set LTM Frequency & Context"):
    def __init__(self, cog, profile_name: str, current_params: Dict[str, Any], callback=None):
        super().__init__()
        self.cog: GeminiAgent = cog
        self.profile_name = profile_name
        self.callback = callback
        self.add_item(ui.TextInput(label="Creation Interval (5-100 msgs)", custom_id="ltm_creation_interval", default=str(current_params.get("ltm_creation_interval", 10)), placeholder="How many messages before creating a memory?"))
        self.add_item(ui.TextInput(label="Summarization Context (5-50 msgs)", custom_id="ltm_summarization_context", default=str(current_params.get("ltm_summarization_context", 10)), placeholder="How many recent messages to summarize?"))

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        new_params = {}
        try:
            inv = int(next(c.value for c in self.children if c.custom_id == "ltm_creation_interval"))
            ctx = int(next(c.value for c in self.children if c.custom_id == "ltm_summarization_context"))
            if not (5 <= inv <= 100): raise ValueError("Interval out of range (5-100)")
            if not (5 <= ctx <= 50): raise ValueError("Context out of range (5-50)")
            new_params["ltm_creation_interval"] = inv
            new_params["ltm_summarization_context"] = ctx
        except ValueError as e:
            await interaction.followup.send(f"❌ **Invalid Input:** {e}.", ephemeral=True); return

        user_data = self.cog._get_user_data_entry(interaction.user.id)
        profile = user_data.get("profiles", {}).get(self.profile_name)
        if profile:
            profile.update(new_params)
            self.cog._save_user_data_entry(interaction.user.id, user_data)
            await interaction.followup.send(f"✅ LTM frequency settings updated for '{self.profile_name}'.", ephemeral=True)
            if self.callback: await self.callback(interaction)

class ProfileImageGenPromptModal(ui.Modal, title="Set Image Generation System Prompt"):
    def __init__(self, cog, profile_name: str, current_prompt: Optional[str]):
        super().__init__()
        self.cog: GeminiAgent = cog
        self.profile_name = profile_name
        decrypted_prompt = self.cog._decrypt_data(current_prompt) if current_prompt else ""
        self.prompt_input = ui.TextInput(
            label="System Instructions for Image AI",
            style=discord.TextStyle.paragraph,
            default=decrypted_prompt,
            required=False,
            max_length=2000,
            placeholder="e.g., All images must be in a gritty, photorealistic, noir style with dramatic shadows."
        )
        self.add_item(self.prompt_input)

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        new_prompt = self.prompt_input.value.strip() or None
        
        user_id = interaction.user.id
        user_data = self.cog.user_profiles.get(str(user_id))
        profile = user_data.get("profiles", {}).get(self.profile_name)
        
        if not profile:
            await interaction.followup.send("Profile not found.", ephemeral=True)
            return

        profile["image_generation_prompt"] = new_prompt
        self.cog._save_user_data_entry(user_id, user_data)
        
        await interaction.followup.send(f"Image generation prompt updated for profile '{self.profile_name}'.", ephemeral=True)

class SessionPromptModal(ui.Modal, title="Set Multi-Profile Session Prompt"):
    prompt_input = ui.TextInput(
        label="Scene Prompt / Director's Note",
        style=discord.TextStyle.paragraph,
        placeholder="e.g., The scene is a rainy night at a bus stop. The last bus just left...",
        required=False,
        max_length=1500
    )

    def __init__(self, parent_view: 'MultiProfileSelectView', current_prompt: Optional[str]):
        super().__init__()
        self.parent_view = parent_view
        if current_prompt:
            self.prompt_input.default = current_prompt

    async def on_submit(self, interaction: discord.Interaction):
        self.parent_view.session_prompt = self.prompt_input.value or None
        await interaction.response.send_message("Session prompt has been updated.", ephemeral=True, delete_after=5)

class MultiProfileSelectView(ui.View):
    def __init__(self, cog: 'GeminiAgent', user_id: int, as_admin_scope: bool, current_profiles: List[Dict] = [], current_prompt: Optional[str] = None, current_mode: str = 'sequential', current_audio_mode: str = 'text-only'):
        super().__init__(timeout=600)
        self.cog = cog
        self.user_id = user_id
        self.as_admin_scope = as_admin_scope
        self.session_prompt = current_prompt
        self.session_mode = current_mode
        self.session_audio_mode = current_audio_mode
        
        self.selection_order = OrderedDict()
        for p_data in current_profiles:
            if p_data.get('method') == 'child_bot':
                key = f"child_{str(p_data['bot_id'])}"
            else:
                key = p_data['profile_name']
            self.selection_order[key] = p_data

        self.view_source: Literal['personal', 'borrowed', 'child_bot'] = 'personal'
        self.current_page = 0
        
        # Pre-load lists - Ensure data is fetched via the helper method
        user_data = self.cog._get_user_data_entry(self.user_id)
        self.lists = {
            'personal': sorted(list(user_data.get("profiles", {}).keys())),
            'borrowed': sorted(list(user_data.get("borrowed_profiles", {}).keys())),
            'child_bot': sorted(
                [b for b_id, b in self.cog.child_bots.items() if b['owner_id'] == self.user_id],
                key=lambda x: x.get('profile_name', '')
            )
        }
        # Map child bot objects back to IDs for the key generation
        self.child_bot_map = {b['profile_name']: b_id for b_id, b in self.cog.child_bots.items() if b['owner_id'] == self.user_id}

        self._build_view()

    def _get_active_list(self):
        return self.lists[self.view_source]

    def _build_view(self):
        self.clear_items()
        
        active_list = self._get_active_list()
        num_pages = (len(active_list) - 1) // DROPDOWN_MAX_OPTIONS + 1
        if self.current_page >= num_pages: self.current_page = max(0, num_pages - 1)
        start = self.current_page * DROPDOWN_MAX_OPTIONS
        page_items = active_list[start : start + DROPDOWN_MAX_OPTIONS]

        # --- Dropdown ---
        options = []
        if page_items:
            for item in page_items:
                if self.view_source == 'child_bot':
                    # item is dict
                    p_name = item.get('profile_name')
                    bot_user_id = next((bid for bid, b in self.cog.child_bots.items() if b is item), None)
                    bot_user = self.cog.bot.get_user(int(bot_user_id)) if bot_user_id else None
                    label = f"{bot_user.name} ({p_name})" if bot_user else f"Bot {p_name}"
                    value = f"child_{bot_user_id}"
                else:
                    # item is string name
                    label = item
                    value = item
                
                is_selected = value in self.selection_order
                options.append(discord.SelectOption(label=label[:100], value=value, default=is_selected))
        else:
            options.append(discord.SelectOption(label="No profiles found in this source", value="none"))

        placeholder = f"Select {self.view_source.replace('_', ' ')} profiles..."
        select = ui.Select(placeholder=placeholder, min_values=0, max_values=len(options) if page_items else 1, options=options, row=0, disabled=(not page_items))
        select.callback = self.select_callback
        self.add_item(select)

        # --- Navigation Row (1) ---
        
        # Source Toggle
        source_labels = {'personal': 'Personal', 'borrowed': 'Borrowed', 'child_bot': 'Child Bots'}
        source_styles = {'personal': discord.ButtonStyle.blurple, 'borrowed': discord.ButtonStyle.green, 'child_bot': discord.ButtonStyle.primary}
        
        btn_source = ui.Button(label=f"Source: {source_labels[self.view_source]}", style=source_styles[self.view_source], row=1)
        btn_source.callback = self.toggle_source
        self.add_item(btn_source)

        # Pagination
        if num_pages > 1:
            btn_prev = ui.Button(label="◀", style=discord.ButtonStyle.secondary, disabled=(self.current_page == 0), row=1)
            btn_prev.callback = self.prev_page
            self.add_item(btn_prev)

            btn_page = ui.Button(label=f"{self.current_page + 1}/{num_pages}", style=discord.ButtonStyle.grey, disabled=True, row=1)
            self.add_item(btn_page)

            btn_next = ui.Button(label="▶", style=discord.ButtonStyle.secondary, disabled=(self.current_page >= num_pages - 1), row=1)
            btn_next.callback = self.next_page
            self.add_item(btn_next)

        # Clear Selection
        btn_clear = ui.Button(label="Clear All", style=discord.ButtonStyle.danger, row=1)
        btn_clear.callback = self.clear_selection
        self.add_item(btn_clear)

        # --- Configuration Row (2) ---
        
        btn_mode = ui.Button(label=f"Mode: {self.session_mode.title()}", style=discord.ButtonStyle.secondary, row=2)
        btn_mode.callback = self.toggle_mode
        self.add_item(btn_mode)

        audio_labels = {
            "text-only": "Audio: Text-Only",
            "audio+text": "Audio: Audio + Text",
            "audio-only": "Audio: Audio-Only",
            "multi-audio": "Audio: Multi-Audio"
        }
        btn_audio = ui.Button(label=audio_labels.get(self.session_audio_mode, "Audio: Text-Only"), style=discord.ButtonStyle.secondary, row=2)
        btn_audio.callback = self.toggle_audio_mode
        self.add_item(btn_audio)

        btn_prompt = ui.Button(label="Set Director's Note", style=discord.ButtonStyle.secondary, row=2)
        btn_prompt.callback = self.set_prompt
        self.add_item(btn_prompt)

        # --- Action Row (3) ---
        
        btn_start = ui.Button(label="Start / Update Session", style=discord.ButtonStyle.success, row=3)
        btn_start.callback = self.start_session
        self.add_item(btn_start)

    async def toggle_source(self, i: discord.Interaction):
        cycle = ['personal', 'borrowed', 'child_bot']
        curr_idx = cycle.index(self.view_source)
        self.view_source = cycle[(curr_idx + 1) % len(cycle)]
        self.current_page = 0
        self._build_view()
        await i.response.edit_message(view=self)

    async def prev_page(self, i: discord.Interaction):
        self.current_page -= 1
        self._build_view()
        await i.response.edit_message(view=self)

    async def next_page(self, i: discord.Interaction):
        self.current_page += 1
        self._build_view()
        await i.response.edit_message(view=self)

    async def clear_selection(self, i: discord.Interaction):
        self.selection_order.clear()
        self._build_view()
        await i.response.edit_message(content=self.get_ordered_list_message(), view=self)

    async def toggle_mode(self, i: discord.Interaction):
        self.session_mode = 'random' if self.session_mode == 'sequential' else 'sequential'
        self._build_view()
        await i.response.edit_message(view=self)

    async def toggle_audio_mode(self, i: discord.Interaction):
        modes = ["text-only", "audio+text", "audio-only", "multi-audio"]
        curr_idx = modes.index(self.session_audio_mode)
        self.session_audio_mode = modes[(curr_idx + 1) % len(modes)]
        self._build_view()
        await i.response.edit_message(view=self)

    async def set_prompt(self, i: discord.Interaction):
        modal = SessionPromptModal(self, self.session_prompt)
        await i.response.send_modal(modal)

    async def select_callback(self, i: discord.Interaction):
        if "none" in i.data['values']: 
            await i.response.defer()
            return

        current_values = set(i.data['values'])
        
        # Determine which items were on the current page to handle deselection
        active_list = self._get_active_list()
        start = self.current_page * DROPDOWN_MAX_OPTIONS
        page_items = active_list[start : start + DROPDOWN_MAX_OPTIONS]
        
        page_values = set()
        for item in page_items:
            if self.view_source == 'child_bot':
                # Reconstruct key for logic
                # item is dict, find ID
                b_id = next((bid for bid, b in self.cog.child_bots.items() if b is item), None)
                if b_id: page_values.add(f"child_{b_id}")
            else:
                page_values.add(item)

        # Logic: 
        # 1. Keep items in selection_order that are NOT on this page.
        # 2. Keep items in selection_order that ARE on this page AND are in current_values.
        # 3. Add items from current_values that are NOT in selection_order (append to end).
        
        new_order = OrderedDict()
        
        # Step 1 & 2: Filter existing
        for key, val in self.selection_order.items():
            if key not in page_values:
                new_order[key] = val
            elif key in current_values:
                new_order[key] = val
        
        # Step 3: Add new
        for key in i.data['values']:
            if key not in self.selection_order:
                # Build entry
                entry = None
                if key.startswith("child_"):
                    bot_id = key.split("_")[1]
                    bot_config = self.cog.child_bots.get(bot_id)
                    if bot_config:
                        entry = {
                            "owner_id": bot_config['owner_id'],
                            "profile_name": bot_config['profile_name'],
                            "method": "child_bot",
                            "bot_id": bot_id
                        }
                else:
                    entry = {
                        "owner_id": self.user_id,
                        "profile_name": key,
                        "method": "webhook"
                    }

                if entry:
                    # Identity Guard: Prevent same Owner + Name combo
                    is_duplicate = any(
                        p['owner_id'] == entry['owner_id'] and p['profile_name'] == entry['profile_name']
                        for p in self.selection_order.values()
                    )
                    if not is_duplicate:
                        new_order[key] = entry

        self.selection_order = new_order
        self._build_view()
        await i.response.edit_message(content=self.get_ordered_list_message(), view=self)

    def get_ordered_list_message(self) -> str:
        ordered_list = list(self.selection_order.values())
        msg = "**Session Configuration**\nUse the dropdowns to build your cast. The order below determines the speaking order in 'Sequential' mode.\n\n**Cast:**\n"
        if not ordered_list:
            msg += "*No participants selected.*"
        else:
            lines = []
            for i, p_data in enumerate(ordered_list):
                name = p_data['profile_name']
                
                if p_data.get('method') == 'child_bot':
                    method = "Child Bot"
                elif name in self.lists['borrowed']:
                    method = "Borrowed"
                else:
                    method = "Personal"
                
                lines.append(f"{i+1}. `{name}` ({method})")
            msg += "\n".join(lines)
        return msg

    async def start_session(self, interaction: discord.Interaction):
        await interaction.response.defer()
        ordered_participants = list(self.selection_order.values())
        
        if not (1 <= len(ordered_participants) <= MAX_MULTI_PROFILES):
            await interaction.followup.send(
                f"You must select between 1 and {MAX_MULTI_PROFILES} participants. You selected {len(ordered_participants)}.",
                ephemeral=True
            )
            return

        await self.cog.setup_multi_profile_session(
            interaction, 
            ordered_participants, 
            self.session_prompt, 
            self.session_mode, 
            as_admin_scope=self.as_admin_scope,
            audio_mode=self.session_audio_mode
        )

class SingleProfileModelView(ui.View):
    def __init__(self, cog: 'GeminiAgent', interaction: discord.Interaction, profile_name: str):
        super().__init__(timeout=300)
        self.cog = cog
        self.original_interaction = interaction
        self.user_id = interaction.user.id
        self.profile_name = profile_name
        self.view_mode = 'google'

        user_data = self.cog._get_user_data_entry(self.user_id)
        is_borrowed = profile_name in user_data.get("borrowed_profiles", {})
        self.profile_data = user_data.get("borrowed_profiles" if is_borrowed else "profiles", {}).get(profile_name, {})
        
        self._primary_model = self.profile_data.get("primary_model", PRIMARY_MODEL_NAME)
        self._fallback_model = self.profile_data.get("fallback_model", FALLBACK_MODEL_NAME)
        
        self._build_view()

    @property
    def primary_model(self):
        return self._primary_model

    @primary_model.setter
    def primary_model(self, value):
        self._primary_model = value
        self._save_changes("primary_model", value)

    @property
    def fallback_model(self):
        return self._fallback_model

    @fallback_model.setter
    def fallback_model(self, value):
        self._fallback_model = value
        self._save_changes("fallback_model", value)

    @property
    def show_fallback_indicator(self):
        return self.profile_data.get("show_fallback_indicator", True)

    @show_fallback_indicator.setter
    def show_fallback_indicator(self, value):
        self._save_changes("show_fallback_indicator", value)

    def _save_changes(self, key, value):
        user_data = self.cog._get_user_data_entry(self.user_id)
        is_borrowed = self.profile_name in user_data.get("borrowed_profiles", {})
        target_dict = user_data.get("borrowed_profiles" if is_borrowed else "profiles", {}).get(self.profile_name)
        
        if target_dict:
            target_dict[key] = value
            self.cog._save_user_data_entry(self.user_id, user_data)
            
            # Clear model cache for this user
            keys_to_delete = []
            for k in list(self.cog.channel_models.keys()):
                key_user_id = None
                if isinstance(k, tuple) and len(k) == 3:
                    key_user_id = k[1]
                elif isinstance(k, tuple) and len(k) == 2:
                    key_user_id = k[1]
                
                if key_user_id == self.user_id:
                    keys_to_delete.append(k)

            for k in keys_to_delete:
                self.cog.channel_models.pop(k, None)
                self.cog.chat_sessions.pop(k, None)
                self.cog.channel_model_last_profile_key.pop(k, None)

    def _get_selection_feedback_message(self) -> str:
        p_clean = self.primary_model.replace("GOOGLE/", "").replace("OPENROUTER/", "")
        f_clean = self.fallback_model.replace("GOOGLE/", "").replace("OPENROUTER/", "")
        fb_status = "ON" if self.show_fallback_indicator else "OFF"
        return f"**Profile:** `{self.profile_name}`\n**Primary:** `{p_clean}`\n**Fallback:** `{f_clean}`\n**Fallback Indicator:** `{fb_status}`\n"

    def _get_top_models(self, provider: str) -> List[str]:
        if provider == 'google':
            return list(get_args(ALLOWED_MODELS))

        filename = "openrouter_models.json"
        path = os.path.join(self.cog.MODELS_DATA_DIR, filename)
        
        data = {}
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = json.loads(f.read())
            except: data = {}

        sorted_models = sorted(data.items(), key=lambda x: x[1], reverse=True)
        return [m[0] for m in sorted_models]

    def _build_view(self):
        self.clear_items()
        top_models = self._get_top_models(self.view_mode)
        
        def create_model_options(current_val):
            opts = [discord.SelectOption(label="Custom Model...", value="custom_option", description="Enter manually via modal")]
            
            if current_val:
                label = current_val
                if self.view_mode == 'google' and label.upper().startswith("GOOGLE/"): label = label[7:]
                elif self.view_mode == 'openrouter' and label.upper().startswith("OPENROUTER/"): label = label[11:]
                opts.append(discord.SelectOption(label=f"Current: {label}", value=current_val, default=True))
            
            prefix = "OPENROUTER/" if self.view_mode == 'openrouter' else "GOOGLE/"
            added = len(opts)
            for m in top_models:
                if added >= 25: break
                val = f"{prefix}{m}"
                if current_val != val:
                    opts.append(discord.SelectOption(label=m[:100], value=val))
                    added += 1
            return opts

        self.add_item(self.PrimaryModelSelect(create_model_options(self.primary_model)))
        self.add_item(self.FallbackModelSelect(create_model_options(self.fallback_model)))

        style_g = discord.ButtonStyle.primary if self.view_mode == 'google' else discord.ButtonStyle.secondary
        style_o = discord.ButtonStyle.primary if self.view_mode == 'openrouter' else discord.ButtonStyle.secondary
        
        btn_google = ui.Button(label="Google Models", style=style_g, row=2, custom_id="mode_google")
        btn_open = ui.Button(label="OpenRouter Models", style=style_o, row=2, custom_id="mode_openrouter")
        
        # New Fallback MSG Toggle
        fb_label = "Fallback Indicator: ON" if self.show_fallback_indicator else "Fallback Indicator: OFF"
        fb_style = discord.ButtonStyle.success if self.show_fallback_indicator else discord.ButtonStyle.secondary
        btn_fallback = ui.Button(label=fb_label, style=fb_style, row=2, custom_id="toggle_fallback")
        
        async def mode_cb(i: discord.Interaction):
            self.view_mode = 'google' if i.data['custom_id'] == 'mode_google' else 'openrouter'
            self._build_view()
            await i.response.edit_message(view=self)
        
        async def fallback_cb(i: discord.Interaction):
            self.show_fallback_indicator = not self.show_fallback_indicator
            self._build_view()
            await i.response.edit_message(content=self._get_selection_feedback_message(), view=self)
        
        btn_google.callback = mode_cb; btn_open.callback = mode_cb; btn_fallback.callback = fallback_cb
        self.add_item(btn_google); self.add_item(btn_open); self.add_item(btn_fallback)

    class PrimaryModelSelect(ui.Select):
        def __init__(self, options): super().__init__(placeholder="Select Primary Model...", options=options, row=0)
        async def callback(self, interaction):
            view = self.view
            if self.values[0] == "custom_option": await interaction.response.send_modal(CustomModelModal(view, 'primary'))
            else: 
                view.primary_model = self.values[0]
                view._build_view()
                await interaction.response.edit_message(content=view._get_selection_feedback_message(), view=view)

    class FallbackModelSelect(ui.Select):
        def __init__(self, options): super().__init__(placeholder="Select Fallback Model...", options=options, row=1)
        async def callback(self, interaction):
            view = self.view
            if self.values[0] == "custom_option": await interaction.response.send_modal(CustomModelModal(view, 'fallback'))
            else: 
                view.fallback_model = self.values[0]
                view._build_view()
                await interaction.response.edit_message(content=view._get_selection_feedback_message(), view=view)

class ModelApplyView(ui.View):
    def __init__(self, cog: 'GeminiAgent', user_id: int, interaction: discord.Interaction):
        super().__init__(timeout=300)
        self.cog = cog
        self.user_id = user_id
        self.interaction = interaction
        self.primary_model: Optional[str] = None
        self.fallback_model: Optional[str] = None
        self.target_profiles: Set[str] = set()
        self.current_page = 0
        self.view_mode = 'google' 
        self.show_fallback_indicator: bool = True 

        # [UPDATED] Load ALL profiles (Personal + Borrowed)
        user_data = self.cog._get_user_data_entry(self.user_id)
        personal = list(user_data.get("profiles", {}).keys())
        borrowed = list(user_data.get("borrowed_profiles", {}).keys())
        self.all_profiles = sorted(personal + borrowed)
        
        self._build_view()

    def _get_selection_feedback_message(self) -> str:
        count = len(self.target_profiles)
        if count == 0:
            return "Use the dropdowns below to select models and the profiles to apply them to."
        
        profile_list = sorted(list(self.target_profiles))
        message = f"**Selected Profiles ({count}):**\n"
        message += "\n".join(f"- `{name}`" for name in profile_list[:10])
        if count > 10:
            message += f"\n...and {count - 10} more."
        return message

    def _get_top_models(self, provider: str) -> List[str]:
        if provider == 'google':
            return list(get_args(ALLOWED_MODELS))

        filename = "openrouter_models.json"
        path = os.path.join(self.cog.MODELS_DATA_DIR, filename)
        
        data = {}
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = json.loads(f.read())
            except: 
                data = {}

        sorted_models = sorted(data.items(), key=lambda x: x[1], reverse=True)
        return [m[0] for m in sorted_models]

    def _build_view(self):
        self.clear_items()
        top_models = self._get_top_models(self.view_mode)
        
        def create_model_options(current_val):
            opts = [discord.SelectOption(label="Custom Model...", value="custom_option", description="Enter manually via modal")]
            
            if current_val:
                label = current_val
                if self.view_mode == 'google' and label.upper().startswith("GOOGLE/"): label = label[7:]
                elif self.view_mode == 'openrouter' and label.upper().startswith("OPENROUTER/"): label = label[11:]
                opts.append(discord.SelectOption(label=f"Current: {label}", value=current_val, default=True))
            
            prefix = "OPENROUTER/" if self.view_mode == 'openrouter' else "GOOGLE/"
            added = len(opts)
            for m in top_models:
                if added >= 25: break
                val = f"{prefix}{m}"
                if current_val != val:
                    opts.append(discord.SelectOption(label=m[:100], value=val))
                    added += 1
            return opts

        self.add_item(self.PrimaryModelSelect(create_model_options(self.primary_model)))
        self.add_item(self.FallbackModelSelect(create_model_options(self.fallback_model)))

        # Row 2: Provider Toggles + Fallback MSG Toggle
        style_g = discord.ButtonStyle.primary if self.view_mode == 'google' else discord.ButtonStyle.secondary
        style_o = discord.ButtonStyle.primary if self.view_mode == 'openrouter' else discord.ButtonStyle.secondary
        
        btn_google = ui.Button(label="Google Models", style=style_g, row=2, custom_id="mode_google")
        btn_open = ui.Button(label="OpenRouter Models", style=style_o, row=2, custom_id="mode_openrouter")
        
        fb_label = "Fallback Indicator: ON" if self.show_fallback_indicator else "Fallback Indicator: OFF"
        fb_style = discord.ButtonStyle.success if self.show_fallback_indicator else discord.ButtonStyle.secondary
        btn_fallback = ui.Button(label=fb_label, style=fb_style, row=2, custom_id="toggle_fallback")
        
        async def mode_cb(i: discord.Interaction):
            self.view_mode = 'google' if i.data['custom_id'] == 'mode_google' else 'openrouter'
            self._build_view()
            await i.response.edit_message(view=self)
        
        async def fallback_cb(i: discord.Interaction):
            self.show_fallback_indicator = not self.show_fallback_indicator
            self._build_view()
            await i.response.edit_message(view=self)
        
        btn_google.callback = mode_cb; btn_open.callback = mode_cb; btn_fallback.callback = fallback_cb
        self.add_item(btn_google); self.add_item(btn_open); self.add_item(btn_fallback)

        # Row 3: Profile Select (Paginated)
        num_pages = (len(self.all_profiles) - 1) // DROPDOWN_MAX_OPTIONS + 1
        if self.current_page >= num_pages: self.current_page = max(0, num_pages - 1)
        
        start = self.current_page * DROPDOWN_MAX_OPTIONS
        page_profiles = self.all_profiles[start : start + DROPDOWN_MAX_OPTIONS]
        
        options = [discord.SelectOption(label=name, value=name, default=(name in self.target_profiles)) for name in page_profiles]
        
        if options:
            profile_select = ui.Select(placeholder="Select profiles...", min_values=0, max_values=len(options), options=options, row=3)
            profile_select.callback = self.profile_callback
            self.add_item(profile_select)

        # Row 4: Navigation & Apply
        if num_pages > 1:
            prev_btn = ui.Button(label="◀", style=discord.ButtonStyle.secondary, disabled=(self.current_page==0), row=4)
            async def prev_cb(i): self.current_page -= 1; self._build_view(); await i.response.edit_message(view=self)
            prev_btn.callback = prev_cb; self.add_item(prev_btn)

            page_btn = ui.Button(label=f"{self.current_page+1}/{num_pages}", style=discord.ButtonStyle.grey, disabled=True, row=4)
            self.add_item(page_btn)

            next_btn = ui.Button(label="▶", style=discord.ButtonStyle.secondary, disabled=(self.current_page>=num_pages-1), row=4)
            async def next_cb(i): self.current_page += 1; self._build_view(); await i.response.edit_message(view=self)
            next_btn.callback = next_cb; self.add_item(next_btn)

            toggle_btn = ui.Button(label="Select Page", style=discord.ButtonStyle.secondary, row=4)
            toggle_btn.callback = self.toggle_page_callback; self.add_item(toggle_btn)

        apply_btn = ui.Button(label="Apply", style=discord.ButtonStyle.success, row=4)
        apply_btn.callback = self.apply_settings; self.add_item(apply_btn)

    async def profile_callback(self, interaction: discord.Interaction):
        start = self.current_page * DROPDOWN_MAX_OPTIONS
        page_profiles = self.all_profiles[start : start + DROPDOWN_MAX_OPTIONS]
        self.target_profiles.difference_update(set(page_profiles))
        self.target_profiles.update(interaction.data['values'])
        self._build_view()
        await interaction.response.edit_message(content=self._get_selection_feedback_message(), view=self)

    async def toggle_page_callback(self, interaction: discord.Interaction):
        start = self.current_page * DROPDOWN_MAX_OPTIONS
        page_profiles = self.all_profiles[start : start + DROPDOWN_MAX_OPTIONS]
        page_set = set(page_profiles)
        if page_set.issubset(self.target_profiles): self.target_profiles.difference_update(page_set)
        else: self.target_profiles.update(page_set)
        self._build_view()
        await interaction.response.edit_message(content=self._get_selection_feedback_message(), view=self)

    class PrimaryModelSelect(ui.Select):
        def __init__(self, options): super().__init__(placeholder="Select Primary Model...", options=options, row=0)
        async def callback(self, interaction):
            view = self.view
            if self.values[0] == "custom_option": await interaction.response.send_modal(CustomModelModal(view, 'primary'))
            else: view.primary_model = self.values[0]; view._build_view(); await interaction.response.edit_message(view=view)

    class FallbackModelSelect(ui.Select):
        def __init__(self, options): super().__init__(placeholder="Select Fallback Model...", options=options, row=1)
        async def callback(self, interaction):
            view = self.view
            if self.values[0] == "custom_option": await interaction.response.send_modal(CustomModelModal(view, 'fallback'))
            else: view.fallback_model = self.values[0]; view._build_view(); await interaction.response.edit_message(view=view)

    async def apply_settings(self, interaction: discord.Interaction):
        await interaction.response.defer()
        if (not self.primary_model and not self.fallback_model) or not self.target_profiles:
            await interaction.followup.send("Please select at least one model (Primary or Fallback) and at least one profile.", ephemeral=True)
            return

        success_count = 0
        for profile_name in self.target_profiles:
            is_borrowed = profile_name in self.cog.user_profiles.get(str(self.user_id), {}).get("borrowed_profiles", {})
            if await self.cog.update_profile_models(self.user_id, profile_name, self.primary_model, self.fallback_model, is_borrowed, self.interaction.channel_id, show_fallback_indicator=self.show_fallback_indicator):
                success_count += 1
        
        msg = f"Updated {success_count} profiles." if success_count else "No profiles updated."
        await interaction.edit_original_response(content=msg, view=None)

    async def on_error(self, interaction: discord.Interaction, error: Exception, item: ui.Item):
        print(f"Error in ModelApplyView: {error}")
        traceback.print_exc()
        if not interaction.response.is_done():
            await interaction.response.send_message("An unexpected error occurred with this view.", ephemeral=True)
        else:
            await interaction.followup.send("An unexpected error occurred with this view.", ephemeral=True)

class BaseBulkProfileView(ui.View):
    def __init__(self, cog, user_id, include_borrowed=True, timeout=300):
        super().__init__(timeout=timeout)
        self.cog = cog
        self.user_id = user_id
        self.include_borrowed = include_borrowed
        self.selected_profiles = set()
        
        user_data = self.cog._get_user_data_entry(self.user_id)
        self.personal_profiles = sorted(list(user_data.get("profiles", {}).keys()))
        self.borrowed_profiles = sorted(list(user_data.get("borrowed_profiles", {}).keys())) if include_borrowed else []
        
        self.current_page = 0
        self.view_source = 'personal'

    def _get_active_list(self):
        return self.personal_profiles if self.view_source == 'personal' else self.borrowed_profiles

    def _build_profile_select_ui(self, row=1):
        active_list = self._get_active_list()
        
        num_pages = (len(active_list) - 1) // DROPDOWN_MAX_OPTIONS + 1
        if self.current_page >= num_pages: self.current_page = max(0, num_pages - 1)
        start = self.current_page * DROPDOWN_MAX_OPTIONS
        page_items = active_list[start : start + DROPDOWN_MAX_OPTIONS]
        
        options = []
        if page_items:
            options = [discord.SelectOption(label=name, value=name, default=(name in self.selected_profiles)) for name in page_items]
        else:
            options = [discord.SelectOption(label="No profiles found", value="none", default=False)]

        placeholder = f"Select {self.view_source} profiles..."
        select = ui.Select(placeholder=placeholder, min_values=0, max_values=len(options) if page_items else 1, options=options, custom_id="profile_select", row=row, disabled=(not page_items))
        select.callback = self.profile_select_callback
        self.add_item(select)

        btn_row = row + 1
        
        if self.include_borrowed:
            label = f"Source: {self.view_source.title()} ({self.current_page + 1}/{num_pages})"
            style = discord.ButtonStyle.blurple if self.view_source == 'personal' else discord.ButtonStyle.green
            mode_btn = ui.Button(label=label, style=style, custom_id="toggle_source", row=btn_row)
            mode_btn.callback = self.toggle_source_callback
            self.add_item(mode_btn)
        else:
            label = f"Page {self.current_page + 1}/{num_pages}"
            info_btn = ui.Button(label=label, style=discord.ButtonStyle.grey, disabled=True, row=btn_row)
            self.add_item(info_btn)

        prev_btn = ui.Button(label="◀", style=discord.ButtonStyle.secondary, custom_id="prev_page", disabled=(self.current_page == 0), row=btn_row)
        prev_btn.callback = self.pagination_callback
        self.add_item(prev_btn)

        next_btn = ui.Button(label="▶", style=discord.ButtonStyle.secondary, custom_id="next_page", disabled=(self.current_page >= num_pages - 1), row=btn_row)
        next_btn.callback = self.pagination_callback
        self.add_item(next_btn)
        
        page_set = set(page_items)
        all_selected = page_set.issubset(self.selected_profiles) if page_items else False
        toggle_label = "Unselect Page" if all_selected else "Select Page"
        toggle_btn = ui.Button(label=toggle_label, style=discord.ButtonStyle.secondary, custom_id="toggle_page_select", row=btn_row, disabled=(not page_items))
        toggle_btn.callback = self.toggle_page_select_callback
        self.add_item(toggle_btn)

    async def toggle_source_callback(self, interaction: discord.Interaction):
        self.view_source = 'borrowed' if self.view_source == 'personal' else 'personal'
        self.current_page = 0
        self._build_view()
        await interaction.response.edit_message(content=self._get_selection_feedback_message(), view=self)

    async def pagination_callback(self, interaction: discord.Interaction):
        if interaction.data['custom_id'] == 'prev_page': self.current_page -= 1
        else: self.current_page += 1
        self._build_view()
        await interaction.response.edit_message(content=self._get_selection_feedback_message(), view=self)

    async def profile_select_callback(self, interaction: discord.Interaction):
        vals = interaction.data.get('values', [])
        if "none" in vals: vals = []
        
        active_list = self._get_active_list()
        start = self.current_page * DROPDOWN_MAX_OPTIONS
        page_items = set(active_list[start : start + DROPDOWN_MAX_OPTIONS])
        
        self.selected_profiles.difference_update(page_items)
        self.selected_profiles.update(vals)
        
        self._build_view()
        await interaction.response.edit_message(content=self._get_selection_feedback_message(), view=self)

    async def toggle_page_select_callback(self, interaction: discord.Interaction):
        active_list = self._get_active_list()
        start = self.current_page * DROPDOWN_MAX_OPTIONS
        page_items = set(active_list[start : start + DROPDOWN_MAX_OPTIONS])
        
        if page_items.issubset(self.selected_profiles): self.selected_profiles.difference_update(page_items)
        else: self.selected_profiles.update(page_items)
        self._build_view()
        await interaction.response.edit_message(content=self._get_selection_feedback_message(), view=self)

    def _get_selection_feedback_message(self) -> str:
        count = len(self.selected_profiles)
        if count == 0: return "Select profiles to apply the action to."
        profile_list = sorted(list(self.selected_profiles))
        message = f"**Selected Profiles ({count}):**\n" + "\n".join(f"- `{name}`" for name in profile_list[:10])
        if count > 10: message += f"\n...and {count - 10} more."
        return message
    
    def _build_view(self):
        raise NotImplementedError

class BulkCriticView(BaseBulkProfileView):
    def __init__(self, cog, user_id):
        super().__init__(cog, user_id, include_borrowed=True)
        self.toggle_choice = None
        self._build_view()

    def _build_view(self):
        self.clear_items()
        
        toggle_options = [
            discord.SelectOption(label="Enable Critic", value="enable", default=(self.toggle_choice is True)),
            discord.SelectOption(label="Disable Critic", value="disable", default=(self.toggle_choice is False))
        ]
        toggle_select = ui.Select(placeholder="Choose an action...", options=toggle_options, row=0)
        toggle_select.callback = self.toggle_callback
        self.add_item(toggle_select)

        self._build_profile_select_ui(row=1)
        
        apply_btn = ui.Button(label="Apply", style=discord.ButtonStyle.green, row=3)
        apply_btn.callback = self.apply_action
        self.add_item(apply_btn)

    async def toggle_callback(self, interaction: discord.Interaction):
        self.toggle_choice = interaction.data['values'][0] == "enable"
        self._build_view()
        await interaction.response.edit_message(view=self)

    async def apply_action(self, interaction: discord.Interaction):
        await interaction.response.defer()
        if self.toggle_choice is None or not self.selected_profiles:
            await interaction.edit_original_response(content="Please select an action and at least one profile.", view=None); return

        updated_count = 0
        user_data = self.cog._get_user_data_entry(self.user_id)
        for profile_name in self.selected_profiles:
            is_borrowed = profile_name in user_data.get("borrowed_profiles", {})
            profile = user_data.get("borrowed_profiles" if is_borrowed else "profiles", {}).get(profile_name)
            if profile:
                profile["critic_enabled"] = self.toggle_choice
                updated_count += 1
        
        if updated_count > 0: self.cog._save_user_data_entry(self.user_id, user_data)
        
        status = "ENABLED" if self.toggle_choice else "DISABLED"
        await interaction.edit_original_response(content=f"Critic has been set to **{status}** for {updated_count} profile(s).", view=None)

class BulkActionView(BaseBulkProfileView):
    def __init__(self, cog: 'GeminiAgent', user_id: int, action: str, placeholder: str, params: Optional[Dict] = None, include_borrowed: bool = False):
        super().__init__(cog, user_id, include_borrowed=include_borrowed)
        self.action = action
        self.params = params
        self.placeholder_text = placeholder
        self._build_view()

    def _build_view(self):
        self.clear_items()
        self._build_profile_select_ui(row=0)
        apply_btn = ui.Button(label="Apply Action", style=discord.ButtonStyle.green, row=2)
        apply_btn.callback = self.apply_action_callback
        self.add_item(apply_btn)

    async def apply_action_callback(self, interaction: discord.Interaction):
        await interaction.response.defer()
        target_profiles_list = list(self.selected_profiles)
        if not target_profiles_list:
            await interaction.edit_original_response(content="You must select at least one profile.", view=None); return

        final_message = "An unknown action was attempted."
        if self.action == "apply_params":
            final_message = await self.cog.bulk_apply_generation_params(self.user_id, target_profiles_list, self.params)
        elif self.action == "apply_thinking_params":
            final_message = await self.cog.bulk_apply_thinking_params(self.user_id, target_profiles_list, self.params)
        elif self.action == "apply_training_params":
            final_message = await self.cog.bulk_apply_training_params(self.user_id, target_profiles_list, self.params)
        elif self.action == "apply_ltm_params":
            final_message = await self.cog.bulk_apply_ltm_params(self.user_id, target_profiles_list, self.params)
        elif self.action == "apply_ltm_summarization":
            final_message = await self.cog.bulk_apply_ltm_summarization_instructions(self.user_id, target_profiles_list, self.params)
        
        await interaction.edit_original_response(content=final_message, view=None)

class BulkGroundingView(BaseBulkProfileView):
    def __init__(self, cog: 'GeminiAgent', user_id: int):
        super().__init__(cog, user_id, include_borrowed=True)
        self.grounding_mode = None
        self._build_view()

    def _build_view(self):
        self.clear_items()
        mode_options = [
            discord.SelectOption(label="Off", value="off", default=(self.grounding_mode == "off")),
            discord.SelectOption(label="On (Summarize)", value="on", default=(self.grounding_mode == "on")),
            discord.SelectOption(label="On+ (Summarize & Cite)", value="on+", default=(self.grounding_mode == "on+"))
        ]
        mode_select = ui.Select(placeholder="Choose a grounding mode...", options=mode_options, row=0)
        mode_select.callback = self.mode_callback
        self.add_item(mode_select)

        self._build_profile_select_ui(row=1)
        
        apply_btn = ui.Button(label="Apply Action", style=discord.ButtonStyle.green, row=3)
        apply_btn.callback = self.apply_action
        self.add_item(apply_btn)

    async def mode_callback(self, interaction: discord.Interaction):
        self.grounding_mode = interaction.data['values'][0]
        self._build_view()
        await interaction.response.edit_message(content=self._get_selection_feedback_message(), view=self)

    async def apply_action(self, interaction: discord.Interaction, button: ui.Button = None):
        await interaction.response.defer()
        target_profiles = list(self.selected_profiles)
        if self.grounding_mode is None or not target_profiles:
            await interaction.edit_original_response(content="Please select a grounding mode and at least one profile.", view=None); return

        user_data = self.cog._get_user_data_entry(self.user_id)
        updated_count = 0
        for profile_name in target_profiles:
            is_borrowed = profile_name in user_data.get("borrowed_profiles", {})
            profile = user_data.get("borrowed_profiles" if is_borrowed else "profiles", {}).get(profile_name)
            if profile:
                profile["grounding_mode"] = self.grounding_mode
                updated_count += 1
        
        if updated_count > 0: self.cog._save_user_data_entry(self.user_id, user_data)
        
        display_mode = {"off": "Off", "on": "On", "on+": "On+"}.get(self.grounding_mode)
        await interaction.edit_original_response(content=f"Grounding has been set to **{display_mode}** for {updated_count} profile(s).", view=None)

class BulkResponseModeView(BaseBulkProfileView):
    def __init__(self, cog: 'GeminiAgent', user_id: int):
        super().__init__(cog, user_id, include_borrowed=True)
        self.mode_choice = None
        self._build_view()

    def _build_view(self):
        self.clear_items()
        opts = [
            discord.SelectOption(label="Regular", value="regular", default=(self.mode_choice == "regular")),
            discord.SelectOption(label="Mention User", value="mention", default=(self.mode_choice == "mention")),
            discord.SelectOption(label="Reply to Message", value="reply", default=(self.mode_choice == "reply")),
            discord.SelectOption(label="Mention + Reply", value="mention_reply", default=(self.mode_choice == "mention_reply"))
        ]
        select = ui.Select(placeholder="Choose a response mode...", options=opts, row=0)
        select.callback = self.choice_callback
        self.add_item(select)
        self._build_profile_select_ui(row=1)
        apply_btn = ui.Button(label="Apply Action", style=discord.ButtonStyle.green, row=3)
        apply_btn.callback = self.apply_action
        self.add_item(apply_btn)

    async def choice_callback(self, interaction: discord.Interaction):
        self.mode_choice = interaction.data['values'][0]
        self._build_view()
        await interaction.response.edit_message(content=self._get_selection_feedback_message(), view=self)

    async def apply_action(self, interaction: discord.Interaction):
        await interaction.response.defer()
        if not self.mode_choice or not self.selected_profiles:
            await interaction.edit_original_response(content="Select a mode and at least one profile.", view=None); return

        user_data = self.cog._get_user_data_entry(self.user_id)
        updated_count = 0
        for name in self.selected_profiles:
            p = user_data.get("profiles", {}).get(name) or user_data.get("borrowed_profiles", {}).get(name)
            if p: p["response_mode"] = self.mode_choice; updated_count += 1
        
        if updated_count > 0:
            self.cog._save_user_data_entry(self.user_id, user_data)
            # Hot-Swap Cache Invalidation
            keys = [k for k in self.cog.channel_models.keys() if isinstance(k, tuple) and k[1] == self.user_id]
            for k in keys: self.cog.channel_models.pop(k, None); self.cog.chat_sessions.pop(k, None)

        await interaction.edit_original_response(content=f"Response mode set to **{self.mode_choice.replace('_', ' ').title()}** for {updated_count} profiles.", view=None)

class BulkURLContextView(BaseBulkProfileView):
    def __init__(self, cog: 'GeminiAgent', user_id: int):
        super().__init__(cog, user_id, include_borrowed=True)
        self.toggle_choice = None
        self._build_view()

    def _build_view(self):
        self.clear_items()
        opts = [
            discord.SelectOption(label="Enable URL Context", value="enable", default=(self.toggle_choice is True)),
            discord.SelectOption(label="Disable URL Context", value="disable", default=(self.toggle_choice is False))
        ]
        select = ui.Select(placeholder="Choose an action...", options=opts, row=0)
        select.callback = self.choice_callback
        self.add_item(select)
        self._build_profile_select_ui(row=1)
        apply_btn = ui.Button(label="Apply Action", style=discord.ButtonStyle.green, row=3)
        apply_btn.callback = self.apply_action
        self.add_item(apply_btn)

    async def choice_callback(self, interaction: discord.Interaction):
        self.toggle_choice = (interaction.data['values'][0] == "enable")
        self._build_view()
        await interaction.response.edit_message(content=self._get_selection_feedback_message(), view=self)

    async def apply_action(self, interaction: discord.Interaction):
        await interaction.response.defer()
        if self.toggle_choice is None or not self.selected_profiles:
            await interaction.edit_original_response(content="Select an action and at least one profile.", view=None); return

        user_data = self.cog._get_user_data_entry(self.user_id)
        updated_count = 0
        for name in self.selected_profiles:
            p = user_data.get("profiles", {}).get(name) or user_data.get("borrowed_profiles", {}).get(name)
            if p: p["url_fetching_enabled"] = self.toggle_choice; updated_count += 1
        
        if updated_count > 0:
            self.cog._save_user_data_entry(self.user_id, user_data)
            keys = [k for k in self.cog.channel_models.keys() if isinstance(k, tuple) and k[1] == self.user_id]
            for k in keys: self.cog.channel_models.pop(k, None); self.cog.chat_sessions.pop(k, None)

        status = "ENABLED" if self.toggle_choice else "DISABLED"
        await interaction.edit_original_response(content=f"URL Context Fetching has been **{status}** for {updated_count} profiles.", view=None)

class BulkImageGenView(BaseBulkProfileView):
    def __init__(self, cog: 'GeminiAgent', user_id: int):
        super().__init__(cog, user_id, include_borrowed=True)
        self.toggle_choice = None
        self._build_view()

    def _build_view(self):
        self.clear_items()
        opts = [
            discord.SelectOption(label="Enable Image Gen", value="enable", default=(self.toggle_choice is True)),
            discord.SelectOption(label="Disable Image Gen", value="disable", default=(self.toggle_choice is False))
        ]
        select = ui.Select(placeholder="Choose an action...", options=opts, row=0)
        select.callback = self.choice_callback
        self.add_item(select)
        self._build_profile_select_ui(row=1)
        apply_btn = ui.Button(label="Apply Action", style=discord.ButtonStyle.green, row=3)
        apply_btn.callback = self.apply_action
        self.add_item(apply_btn)

    async def choice_callback(self, interaction: discord.Interaction):
        self.toggle_choice = (interaction.data['values'][0] == "enable")
        self._build_view()
        await interaction.response.edit_message(content=self._get_selection_feedback_message(), view=self)

    async def apply_action(self, interaction: discord.Interaction):
        await interaction.response.defer()
        if self.toggle_choice is None or not self.selected_profiles:
            await interaction.edit_original_response(content="Select an action and at least one profile.", view=None); return

        user_data = self.cog._get_user_data_entry(self.user_id)
        updated_count = 0
        for name in self.selected_profiles:
            p = user_data.get("profiles", {}).get(name) or user_data.get("borrowed_profiles", {}).get(name)
            if p: p["image_generation_enabled"] = self.toggle_choice; updated_count += 1
        
        if updated_count > 0:
            self.cog._save_user_data_entry(self.user_id, user_data)
            keys = [k for k in self.cog.channel_models.keys() if isinstance(k, tuple) and k[1] == self.user_id]
            for k in keys: self.cog.channel_models.pop(k, None); self.cog.chat_sessions.pop(k, None)

        status = "ENABLED" if self.toggle_choice else "DISABLED"
        await interaction.edit_original_response(content=f"Image Generation has been **{status}** for {updated_count} profiles.", view=None)

class BulkTimezoneModal(ui.Modal, title="Enter Custom Timezone"):
    tz_input = ui.TextInput(label="IANA Timezone ID", placeholder="e.g. Asia/Tokyo or America/New_York", required=True)

    def __init__(self, parent_view):
        super().__init__()
        self.parent_view = parent_view

    async def on_submit(self, interaction: discord.Interaction):
        tz_str = self.tz_input.value.strip()
        try:
            # Validate timezone string
            ZoneInfo(tz_str)
            self.parent_view.selected_tz = tz_str
            self.parent_view._build_view()
            await interaction.response.edit_message(content=self.parent_view._get_selection_feedback_message(), view=self.parent_view)
        except Exception:
            await interaction.response.send_message(f"❌ `{tz_str}` is not a valid IANA timezone. Please check your spelling.", ephemeral=True)

class BulkTimezoneView(BaseBulkProfileView):
    def __init__(self, cog: 'GeminiAgent', user_id: int):
        super().__init__(cog, user_id, include_borrowed=True)
        self.selected_tz = None
        self._build_view()

    def _build_view(self):
        self.clear_items()
        
        common_tzs = [
            ("Custom / Manual...", "custom"),
            ("UTC / GMT", "UTC"),
            ("US/Pacific (PT)", "US/Pacific"),
            ("US/Central (CT)", "US/Central"),
            ("US/Eastern (ET)", "US/Eastern"),
            ("Europe/London (GMT/BST)", "Europe/London"),
            ("Europe/Berlin (CET)", "Europe/Berlin"),
            ("Asia/Tokyo (JST)", "Asia/Tokyo"),
            ("Australia/Sydney (AEST)", "Australia/Sydney")
        ]
        
        opts = []
        for label, val in common_tzs:
            opts.append(discord.SelectOption(label=label, value=val, default=(self.selected_tz == val)))

        select = ui.Select(placeholder="Choose a timezone...", options=opts, row=0)
        select.callback = self.tz_callback
        self.add_item(select)

        self._build_profile_select_ui(row=1)
        
        apply_btn = ui.Button(label="Apply Timezone", style=discord.ButtonStyle.green, row=3)
        apply_btn.callback = self.apply_action
        self.add_item(apply_btn)

    async def tz_callback(self, interaction: discord.Interaction):
        val = interaction.data['values'][0]
        if val == "custom":
            await interaction.response.send_modal(BulkTimezoneModal(self))
        else:
            self.selected_tz = val
            self._build_view()
            await interaction.response.edit_message(content=self._get_selection_feedback_message(), view=self)

    async def apply_action(self, interaction: discord.Interaction):
        await interaction.response.defer()
        if not self.selected_tz or not self.selected_profiles:
            await interaction.edit_original_response(content="Select a timezone and at least one profile.", view=None); return

        user_data = self.cog._get_user_data_entry(self.user_id)
        updated_count = 0
        for name in self.selected_profiles:
            p = user_data.get("profiles", {}).get(name) or user_data.get("borrowed_profiles", {}).get(name)
            if p:
                p["timezone"] = self.selected_tz
                p["time_tracking_enabled"] = True # Force always-on
                updated_count += 1
        
        if updated_count > 0:
            self.cog._save_user_data_entry(self.user_id, user_data)
            # Flush caches for the user
            keys = [k for k in self.cog.channel_models.keys() if isinstance(k, tuple) and k[1] == self.user_id]
            for k in keys: 
                self.cog.channel_models.pop(k, None)
                self.cog.chat_sessions.pop(k, None)

        await interaction.edit_original_response(content=f"Timezone set to **{self.selected_tz}** for {updated_count} profiles.", view=None)

class BulkResetView(BaseBulkProfileView):
    def __init__(self, cog: 'GeminiAgent', user_id: int):
        super().__init__(cog, user_id, include_borrowed=True)
        self.reset_choice = None
        self._build_view()

    def _build_view(self):
        self.clear_items()
        reset_options = [
            discord.SelectOption(label="Reset Training Examples (Personal Only)", value="reset_examples", default=(self.reset_choice == "reset_examples")),
            discord.SelectOption(label="Reset Long-Term Memories (All Profiles)", value="reset_ltm", default=(self.reset_choice == "reset_ltm"))
        ]
        reset_select = ui.Select(placeholder="Choose what to reset...", options=reset_options, row=0)
        reset_select.callback = self.reset_type_callback
        self.add_item(reset_select)

        if self.reset_choice:
            # Dynamic include_borrowed update
            self.include_borrowed = (self.reset_choice == "reset_ltm")
            # If switching to a mode that doesn't support borrowed, reset view source to personal
            if not self.include_borrowed and self.view_source == 'borrowed':
                self.view_source = 'personal'
                self.current_page = 0
            
            self._build_profile_select_ui(row=1)

        apply_button = ui.Button(label="Confirm & Reset Data", style=discord.ButtonStyle.danger, row=3, disabled=(not self.reset_choice))
        apply_button.callback = self.apply_action
        self.add_item(apply_button)

    async def reset_type_callback(self, interaction: discord.Interaction):
        self.reset_choice = interaction.data['values'][0]
        self.selected_profiles.clear()
        self.current_page = 0
        self._build_view()
        await interaction.response.edit_message(content=self._get_selection_feedback_message(), view=self)

    async def apply_action(self, interaction: discord.Interaction, button: ui.Button = None):
        await interaction.response.defer()
        target_profiles = list(self.selected_profiles)
        if not self.reset_choice or not target_profiles:
            await interaction.edit_original_response(content="Please select a reset action and at least one profile.", view=None); return

        final_message = "An unknown action was attempted."
        if self.reset_choice == "reset_examples":
            final_message = await self.cog.bulk_reset_examples(self.user_id, target_profiles)
        elif self.reset_choice == "reset_ltm":
            final_message = await self.cog.bulk_reset_ltm(self.user_id, target_profiles)
        
        await interaction.edit_original_response(content=final_message, view=None)

class BulkSafetyLevelView(BaseBulkProfileView):
    def __init__(self, cog: 'GeminiAgent', user_id: int):
        super().__init__(cog, user_id, include_borrowed=True)
        self.selected_level = None
        self._build_view()

    def _build_view(self):
        self.clear_items()
        level_options = [
            discord.SelectOption(label="Unrestricted", value="unrestricted", description="No content filtering. Cannot be applied to public/borrowed profiles.", default=(self.selected_level == "unrestricted")),
            discord.SelectOption(label="Low", value="low", description="Blocks only highly probable harmful content. (Default)", default=(self.selected_level == "low")),
            discord.SelectOption(label="Medium", value="medium", description="Blocks medium and high probability harmful content.", default=(self.selected_level == "medium")),
            discord.SelectOption(label="High", value="high", description="Blocks low, medium, and high probability content.", default=(self.selected_level == "high"))
        ]
        level_select = ui.Select(placeholder="Choose a safety level to apply...", options=level_options, row=0)
        level_select.callback = self.level_select_callback
        self.add_item(level_select)

        self._build_profile_select_ui(row=1)
        
        apply_button = ui.Button(label="Apply Safety Level", style=discord.ButtonStyle.green, row=3)
        apply_button.callback = self.apply_callback
        self.add_item(apply_button)

    async def level_select_callback(self, interaction: discord.Interaction):
        self.selected_level = interaction.data['values'][0]
        self._build_view()
        await interaction.response.edit_message(content=self._get_selection_feedback_message(), view=self)

    async def apply_callback(self, interaction: discord.Interaction):
        await interaction.response.defer()
        target_profiles = list(self.selected_profiles)
        if not self.selected_level or not target_profiles:
            await interaction.edit_original_response(content="You must select a safety level and at least one profile.", view=None); return

        user_data = self.cog._get_user_data_entry(self.user_id)
        updated_count, adjusted_count = 0, 0
        
        safety_map = {"unrestricted": 0, "low": 1, "medium": 2, "high": 3}
        reverse_safety_map = {v: k for k, v in safety_map.items()}
        desired_level_num = safety_map[self.selected_level]

        for profile_name in target_profiles:
            is_borrowed = profile_name in self.borrowed_profiles
            profile_dict = user_data.get("borrowed_profiles" if is_borrowed else "profiles", {})
            
            if profile_name in profile_dict:
                is_public = self.cog._is_profile_public(self.user_id, profile_name)
                min_level_num = safety_map["low"] if is_public or is_borrowed else safety_map["unrestricted"]
                
                final_level_num = max(desired_level_num, min_level_num)
                final_level_str = reverse_safety_map[final_level_num]
                
                if final_level_num > desired_level_num: adjusted_count += 1
                profile_dict[profile_name]['safety_level'] = final_level_str
                updated_count += 1
        
        if updated_count > 0: self.cog._save_user_data_entry(self.user_id, user_data)
        
        message = f"Successfully updated safety level for {updated_count} profile(s)."
        if adjusted_count > 0: message += f"\nThe '{self.selected_level}' level was automatically adjusted to 'low' for {adjusted_count} public/borrowed profile(s)."
        await interaction.edit_original_response(content=message, view=None)

class BulkLtmScopeView(BaseBulkProfileView):
    def __init__(self, cog: 'GeminiAgent', user_id: int):
        super().__init__(cog, user_id, include_borrowed=True)
        self.selected_scope = None
        self._build_view()

    def _build_view(self):
        self.clear_items()
        scope_options = [
            discord.SelectOption(label="Global", value="global", description="Memories can be recalled in any server by anyone.", default=(self.selected_scope == "global")),
            discord.SelectOption(label="Server-Exclusive", value="server", description="Memories only recalled in the server they were made in.", default=(self.selected_scope == "server")),
            discord.SelectOption(label="User-Exclusive", value="user", description="Memories only recalled by the profile owner.", default=(self.selected_scope == "user"))
        ]
        scope_select = ui.Select(placeholder="Choose an LTM scope to apply...", options=scope_options, row=0)
        scope_select.callback = self.scope_select_callback
        self.add_item(scope_select)

        self._build_profile_select_ui(row=1)
        
        apply_button = ui.Button(label="Apply Scope", style=discord.ButtonStyle.green, row=3)
        apply_button.callback = self.apply_callback
        self.add_item(apply_button)

    async def scope_select_callback(self, interaction: discord.Interaction):
        self.selected_scope = interaction.data['values'][0]
        self._build_view()
        await interaction.response.edit_message(content=self._get_selection_feedback_message(), view=self)

    async def apply_callback(self, interaction: discord.Interaction):
        await interaction.response.defer()
        target_profiles = self.selected_profiles
        if not self.selected_scope or not target_profiles:
            await interaction.edit_original_response(content="You must select a scope and at least one profile.", view=None); return

        user_data = self.cog._get_user_data_entry(self.user_id)
        updated_count = 0
        for profile_name in target_profiles:
            is_borrowed = profile_name in self.borrowed_profiles
            profile_dict = user_data.get("borrowed_profiles" if is_borrowed else "profiles", {})
            if profile_name in profile_dict:
                profile_dict[profile_name]['ltm_scope'] = self.selected_scope
                updated_count += 1
        
        if updated_count > 0: self.cog._save_user_data_entry(self.user_id, user_data)
        await interaction.edit_original_response(content=f"Successfully set LTM scope to '{self.selected_scope}' for {updated_count} profile(s).", view=None)

class BulkDeleteView(BaseBulkProfileView):
    def __init__(self, cog: 'GeminiAgent', user_id: int):
        super().__init__(cog, user_id, include_borrowed=True)
        # Filter out default profile from lists in base class logic? 
        # Base class loads all. We filter default profile for display logic in callback.
        # But cleaner to filter source lists directly.
        if "mimic" in self.personal_profiles: self.personal_profiles.remove("mimic")
        if "mimic" in self.borrowed_profiles: self.borrowed_profiles.remove("mimic")
        self._build_view()

    def _build_view(self):
        self.clear_items()
        self._build_profile_select_ui(row=0)
        confirm_button = ui.Button(label="Confirm & Delete Selected Profiles", style=discord.ButtonStyle.danger, row=2)
        confirm_button.callback = self.confirm_delete_callback
        self.add_item(confirm_button)

    async def confirm_delete_callback(self, interaction: discord.Interaction):
        await interaction.response.defer()
        items_to_delete = list(self.selected_profiles)
        if not items_to_delete:
            await interaction.edit_original_response(content="You must select profiles to delete.", view=None); return

        deleted_count = 0
        user_id_str = str(self.user_id)
        user_data = self.cog._get_user_data_entry(self.user_id)
        
        for name in items_to_delete:
            if name in user_data.get("borrowed_profiles", {}):
                del user_data["borrowed_profiles"][name]
                deleted_count += 1
            elif name in user_data.get("profiles", {}):
                del user_data["profiles"][name]
                self.cog._delete_ltm_shard(user_id_str, name)
                self.cog._delete_training_shard(user_id_str, name)
                deleted_count += 1
        
        if deleted_count > 0: self.cog._save_user_data_entry(self.user_id, user_data)
        await interaction.edit_original_response(content=f"Successfully deleted {deleted_count} profiles.", view=None)

class SearchDataModal(ui.Modal, title="Search Data"):
    search_input = ui.TextInput(label="Enter search term (leave blank to clear)", required=False, max_length=100)

    def __init__(self, parent_view: 'DataManageView'):
        super().__init__()
        self.parent_view = parent_view
        if self.parent_view.search_term:
            self.search_input.default = self.parent_view.search_term

    async def on_submit(self, interaction: discord.Interaction):
        search_term = self.search_input.value.strip()
        self.parent_view.search_term = search_term if search_term else None
        self.parent_view.current_page = 1
        await self.parent_view._update_view(interaction)

class DataManageView(ui.View):
    def __init__(self, cog: 'GeminiAgent', interaction: discord.Interaction, profile_name: str, is_borrowed: bool, effective_owner_id: int):
        super().__init__(timeout=600)
        self.cog = cog
        self.original_interaction = interaction
        self.user_id = interaction.user.id
        self.effective_owner_id = effective_owner_id
        self.guild_id = interaction.guild_id
        self.profile_name = profile_name
        self.is_borrowed = is_borrowed
        
        self.mode: Literal['training', 'ltm'] = 'ltm' if self.is_borrowed else 'training'
        self.current_page = 1
        self.items_per_page = 1
        self.max_pages = 1
        self.current_item_id: Optional[str] = None
        self.full_data_list: List[Dict] = []
        self.search_term: Optional[str] = None
        self.displayed_data_list: List[Dict] = []
        self.ltm_filter: Optional[str] = "all"

    async def start(self):
        await self._update_view(self.original_interaction)

    async def _update_view(self, interaction: discord.Interaction):
        embed, page_items, ltm_filter_options = await self._build_embed()
        self._build_components(page_items, ltm_filter_options)
        if interaction.response.is_done():
            await interaction.edit_original_response(embed=embed, view=self)
        else:
            await interaction.response.edit_message(embed=embed, view=self)

    async def _build_embed(self) -> Tuple[discord.Embed, List[Dict], List[discord.SelectOption]]:
        user_id_str = str(self.effective_owner_id)
        title_prefix = ""
        ltm_filter_options = []

        if self.mode == 'training':
            self.full_data_list = self.cog._load_training_shard(user_id_str, self.profile_name) or []
            title_prefix = "Training Examples"
            self.displayed_data_list = self.full_data_list
        else: # ltm
            ltm_shard = self.cog._load_ltm_shard(user_id_str, self.profile_name)
            # All LTMs are currently guild-based, so we always load from the "guild" key.
            self.full_data_list = ltm_shard.get("guild", []) if ltm_shard else []
            title_prefix = "Long-Term Memories"

            # Build filter options and filter the list
            server_filters = {}
            has_user_exclusive = False
            has_global = False
            for item in self.full_data_list:
                scope = item.get('scope')
                if scope == 'server':
                    server_id = item.get('context_id')
                    if server_id and server_id not in server_filters:
                        try:
                            guild = self.cog.bot.get_guild(int(server_id))
                            server_filters[server_id] = guild.name if guild else f"Server ID: {server_id}"
                        except (ValueError, TypeError):
                            print(f"WARNING: Corrupted LTM entry found for user '{user_id_str}', profile '{self.profile_name}'. Skipping. Corrupted item: {item}")
                            continue
                elif scope == 'user':
                    has_user_exclusive = True
                elif scope == 'global':
                    has_global = True
            
            ltm_filter_options.append(discord.SelectOption(label="All Memories", value="all"))
            if has_user_exclusive:
                ltm_filter_options.append(discord.SelectOption(label="User-Exclusive Memories", value="user"))
            if has_global:
                ltm_filter_options.append(discord.SelectOption(label="Global Memories", value="global"))
            for server_id, server_name in sorted(server_filters.items(), key=lambda item: item[1]):
                ltm_filter_options.append(discord.SelectOption(label=f"Server: {server_name}", value=f"server_{server_id}"))

            for option in ltm_filter_options:
                if option.value == self.ltm_filter:
                    option.default = True

            if self.ltm_filter == "all":
                self.displayed_data_list = self.full_data_list
            elif self.ltm_filter == "user" or self.ltm_filter == "global":
                self.displayed_data_list = [item for item in self.full_data_list if item.get('scope') == self.ltm_filter]
            elif self.ltm_filter and self.ltm_filter.startswith("server_"):
                filter_server_id = self.ltm_filter.split("_", 1)[1]
                self.displayed_data_list = [item for item in self.full_data_list if item.get('scope') == 'server' and str(item.get('context_id')) == filter_server_id]
            else:
                self.displayed_data_list = self.full_data_list


        # After scope filtering, apply search term filtering on the result
        if self.search_term:
            search_term_lower = self.search_term.lower()
            
            # Note: We filter the already-scope-filtered 'displayed_data_list'
            search_filtered_list = []
            for item in self.displayed_data_list:
                content_to_search = ""
                if self.mode == 'training':
                    content_to_search = self.cog._decrypt_data(item.get('u_in', '')) + " " + self.cog._decrypt_data(item.get('b_out', ''))
                else: # ltm
                    content_to_search = self.cog._decrypt_data(item.get('sum', ''))
                
                if search_term_lower in content_to_search.lower():
                    search_filtered_list.append(item)
            self.displayed_data_list = search_filtered_list
        
        self.max_pages = len(self.displayed_data_list) or 1
        self.current_page = max(1, min(self.current_page, self.max_pages))
        start_index = self.current_page - 1
        
        page_items = self.displayed_data_list[start_index : start_index + 1]
        self.current_item_id = page_items[0].get('id') if page_items else None

        embed = discord.Embed(title=f"{title_prefix} for '{self.profile_name}'", color=discord.Color.dark_teal())
        embed.set_footer(text=f"Item {self.current_page}/{self.max_pages} | Total: {len(self.full_data_list)}")

        if not page_items:
            embed.description = f"No {title_prefix.lower()} found."
        else:
            item = page_items[0]
            item_id = item.get('id', 'N/A')

            created_ts_str = item.get('created_ts') or item.get('ts')
            modified_ts_str = item.get('modified_ts')
            ts_display = ""
            
            created_dt = None
            if created_ts_str:
                try:
                    created_dt = datetime.datetime.fromisoformat(created_ts_str)
                    ts_display += f" | Created: {created_dt.strftime('%d/%m/%y')} UTC"
                except ValueError:
                    pass

            if modified_ts_str:
                try:
                    modified_dt = datetime.datetime.fromisoformat(modified_ts_str)
                    if created_dt and (modified_dt - created_dt).total_seconds() > 5:
                        ts_display += f" | Modified: {modified_dt.strftime('%d/%m/%y')} UTC"
                except ValueError:
                    pass

            if self.mode == 'training':
                user_input = self.cog._decrypt_data(item.get('u_in', ''))
                bot_response = self.cog._decrypt_data(item.get('b_out', ''))
                embed.add_field(name=f"ID: `{item_id}`{ts_display}", value=f"**User Input:**\n{user_input}", inline=False)
                embed.add_field(name="Bot Response:", value=bot_response, inline=False)
            else: # ltm
                content = self.cog._decrypt_data(item.get('sum', ''))
                scope = item.get('scope', 'unknown').title()
                
                display_content = content
                if len(content) > 950:
                    display_content = content[:950] + "... (truncated)"

                embed.add_field(name=f"ID: `{item_id}` | Scope: {scope}{ts_display}", value=f"**Summary:**\n{display_content}", inline=False)
        
        return embed, page_items, ltm_filter_options

    def _build_components(self, page_items: List[Dict], ltm_filter_options: List[discord.SelectOption]):
        self.clear_items()

        # Row 0: Navigation and Mode
        if not self.is_borrowed:
            training_button = ui.Button(label="Training", style=discord.ButtonStyle.green if self.mode == 'training' else discord.ButtonStyle.grey, custom_id="mode_training", row=0)
            training_button.callback = self.mode_button_callback
            self.add_item(training_button)

            ltm_button = ui.Button(label="LTMs", style=discord.ButtonStyle.green if self.mode == 'ltm' else discord.ButtonStyle.grey, custom_id="mode_ltm", row=0)
            ltm_button.callback = self.mode_button_callback
            self.add_item(ltm_button)

        prev_button = ui.Button(label="◀", style=discord.ButtonStyle.blurple, disabled=(self.current_page <= 1), row=0)
        prev_button.callback = self.prev_page_callback
        self.add_item(prev_button)

        next_button = ui.Button(label="▶", style=discord.ButtonStyle.blurple, disabled=(self.current_page >= self.max_pages), row=0)
        next_button.callback = self.next_page_callback
        self.add_item(next_button)

        if self.mode == 'ltm' and page_items:
            current_scope = page_items[0].get('scope', 'server')
            scope_button = ui.Button(label=f"Scope: {current_scope.title()}", style=discord.ButtonStyle.secondary, custom_id="cycle_scope", row=0)
            scope_button.callback = self.cycle_scope_callback
            self.add_item(scope_button)
        
        # [NEW] Move Analyse button to Row 0 (Only visible in training mode)
        if self.mode == 'training' and not self.is_borrowed:
            analyse_button = ui.Button(label="Analyse", style=discord.ButtonStyle.blurple, row=0)
            async def analyse_cb(i): await i.response.send_modal(AnalyseExamplesModal(self))
            analyse_button.callback = analyse_cb
            self.add_item(analyse_button)

        # Row 1: LTM Filter
        if self.mode == 'ltm' and ltm_filter_options:
            ltm_filter_select = ui.Select(placeholder="Filter memories by scope...", options=ltm_filter_options, row=1)
            ltm_filter_select.callback = self.ltm_filter_callback
            self.add_item(ltm_filter_select)

            # New sliding window logic
            window_size = 25
            half_window = window_size // 2
            
            start_slice_index = max(0, self.current_page - 1 - half_window)
            end_slice_index = start_slice_index + window_size
            
            # Adjust if we're near the end of the list
            if end_slice_index > len(self.displayed_data_list):
                end_slice_index = len(self.displayed_data_list)
                start_slice_index = max(0, end_slice_index - window_size)

            items_for_dropdown = self.displayed_data_list[start_slice_index:end_slice_index]

            options = []
            for i, item in enumerate(items_for_dropdown):
                item_id = item.get('id', 'N/A')
                absolute_page_number = start_slice_index + i + 1
                
                if self.mode == 'training':
                    content = self.cog._decrypt_data(item.get('u_in', ''))[:80]
                    label = f"Ex ({item_id}): {content}..."
                else:
                    content = self.cog._decrypt_data(item.get('sum', ''))[:80]
                    label = f"LTM ({item_id}): {content}..."
                
                option = discord.SelectOption(label=label, value=str(absolute_page_number))
                if absolute_page_number == self.current_page:
                    option.default = True
                options.append(option)

            if options:
                select = ui.Select(placeholder="Quick Navigation...", options=options, row=2)
                select.callback = self.select_callback
                self.add_item(select)

        # [UPDATED] Row 3: Action Buttons
        search_button = ui.Button(label="🔍 Search", style=discord.ButtonStyle.secondary, row=3)
        search_button.callback = self.search_callback
        self.add_item(search_button)

        add_button = ui.Button(label="Add New", style=discord.ButtonStyle.success, row=3)
        add_button.callback = self.add_callback
        self.add_item(add_button)

        edit_button = ui.Button(label="Edit", style=discord.ButtonStyle.primary, row=3, disabled=(not page_items))
        edit_button.callback = self.edit_callback
        self.add_item(edit_button)

        delete_button = ui.Button(label="Delete", style=discord.ButtonStyle.danger, row=3, disabled=(not page_items))
        delete_button.callback = self.delete_callback
        self.add_item(delete_button)

        delete_all_button = ui.Button(label="Delete All (Filtered)", style=discord.ButtonStyle.danger, row=3, disabled=True)
        if self.mode == 'ltm' and self.ltm_filter and self.ltm_filter.startswith("server_") and self.displayed_data_list:
            delete_all_button.disabled = False
        delete_all_button.callback = self.delete_all_callback
        self.add_item(delete_all_button)

    # Callbacks
    async def delete_all_callback(self, interaction: discord.Interaction):
        if not (self.mode == 'ltm' and self.ltm_filter and self.ltm_filter.startswith("server_")):
            await interaction.response.send_message("This action is only available when filtering LTMs by a specific server.", ephemeral=True)
            return

        items_to_delete = self.displayed_data_list
        if not items_to_delete:
            await interaction.response.send_message("There are no items matching the current filter to delete.", ephemeral=True)
            return

        confirm_view = ui.View(timeout=60)
        async def confirm_action(i: discord.Interaction):
            owner_id_str = str(self.effective_owner_id)
            ltm_data = self.cog._load_ltm_shard(owner_id_str, self.profile_name)
            if not ltm_data:
                await i.response.edit_message(content="Could not load LTM data.", view=None)
                return

            ids_to_delete = {item['id'] for item in items_to_delete}
            context_type = "guild"
            original_list = ltm_data.get(context_type, [])
            
            new_list = [item for item in original_list if item.get("id") not in ids_to_delete]
            
            ltm_data[context_type] = new_list
            self.cog._save_ltm_shard(owner_id_str, self.profile_name, ltm_data)
            
            await i.response.edit_message(content=f"Successfully deleted {len(ids_to_delete)} LTM entries.", view=None)
            
            self.current_page = 1
            await self._update_view(self.original_interaction)

        confirm_button = ui.Button(label=f"Confirm Delete All ({len(items_to_delete)})", style=discord.ButtonStyle.danger)
        confirm_button.callback = confirm_action
        confirm_view.add_item(confirm_button)
        
        try:
            filter_server_id = self.ltm_filter.split("_", 1)[1]
            guild = self.cog.bot.get_guild(int(filter_server_id))
            server_name = guild.name if guild else f"ID: {filter_server_id}"
        except (IndexError, ValueError):
            server_name = "the selected server"

        await interaction.response.send_message(
            f"**Are you sure you want to delete all {len(items_to_delete)} LTMs for profile '{self.profile_name}' from '{server_name}'?**\nThis action is permanent.",
            view=confirm_view,
            ephemeral=True
        )

    async def cycle_scope_callback(self, interaction: discord.Interaction):
        if not self.current_item_id or self.mode != 'ltm':
            await interaction.response.defer()
            return

        owner_id_str = str(self.effective_owner_id)
        ltm_data = self.cog._load_ltm_shard(owner_id_str, self.profile_name)
        if not ltm_data:
            await interaction.response.defer()
            return
        
        context_type: Literal["guild", "dm"] = "guild" if self.guild_id else "dm"
        ltm_list = ltm_data.get(context_type, [])
        
        item_found = False
        for i, item in enumerate(ltm_list):
            if item.get("id") == self.current_item_id:
                current_scope = item.get('scope', 'server')
                
                if self.guild_id: # In a server, cycle through all three
                    scope_cycle = {'server': 'user', 'user': 'global', 'global': 'server'}
                    new_scope = scope_cycle.get(current_scope, 'server')
                else: # In a DM, skip the 'server' option
                    scope_cycle_dm = {'server': 'user', 'user': 'global', 'global': 'user'}
                    new_scope = scope_cycle_dm.get(current_scope, 'user')

                item['scope'] = new_scope
                
                if new_scope == 'global':
                    item['context_id'] = None
                elif new_scope == 'user':
                    item['context_id'] = str(self.effective_owner_id)
                elif new_scope == 'server':
                    # If the current context_id is not a valid server ID (e.g., from 'global' or 'user'),
                    # assign the current server's ID. Otherwise, preserve the original server ID.
                    current_context = item.get('context_id')
                    if not current_context or not str(current_context).isdigit():
                        item['context_id'] = str(self.guild_id)

                ltm_data[context_type][i] = item
                self.cog._save_ltm_shard(owner_id_str, self.profile_name, ltm_data)
                item_found = True
                break
        
        if item_found:
            await self._update_view(interaction)
        else:
            await interaction.response.defer()


    async def ltm_filter_callback(self, interaction: discord.Interaction):
        self.ltm_filter = interaction.data['values'][0]
        self.current_page = 1
        await self._update_view(interaction)

    async def mode_button_callback(self, interaction: discord.Interaction):
        self.mode = 'ltm' if interaction.data['custom_id'] == 'mode_ltm' else 'training'
        self.current_page = 1
        await self._update_view(interaction)

    async def prev_page_callback(self, interaction: discord.Interaction):
        if self.current_page > 1:
            self.current_page -= 1
            await self._update_view(interaction)

    async def next_page_callback(self, interaction: discord.Interaction):
        if self.current_page < self.max_pages:
            self.current_page += 1
            await self._update_view(interaction)

    async def select_callback(self, interaction: discord.Interaction):
        self.current_page = int(interaction.data['values'][0])
        await self._update_view(interaction)

    async def add_callback(self, interaction: discord.Interaction):
        if self.mode == 'training':
            modal = AddTrainingExampleModal(self.cog, self.user_id, self.profile_name, self.guild_id)
        else: # ltm
            modal = AddLtmModal(self.cog, self.user_id, self.profile_name, self.guild_id)
        
        original_on_submit = modal.on_submit
        async def on_submit_refresh(i: discord.Interaction):
            await original_on_submit(i)
            if not i.response.is_done(): await i.response.defer()
            self.current_page = self.max_pages + 1 # Go to the new item
            await self._update_view(self.original_interaction)
        
        modal.on_submit = on_submit_refresh
        await interaction.response.send_modal(modal)

    async def edit_callback(self, interaction: discord.Interaction):
        if not self.current_item_id: return
        
        item_to_edit = next((item for item in self.full_data_list if item.get("id") == self.current_item_id), None)
        modal = None
        if item_to_edit:
            if self.mode == 'training':
                modal = EditTrainingExampleModal(self.cog, self.user_id, self.profile_name, self.current_item_id, self.cog._decrypt_data(item_to_edit.get("u_in", "")), self.cog._decrypt_data(item_to_edit.get("b_out", "")), self.guild_id)
            else: # ltm
                modal = EditLtmModal(self.cog, self.user_id, self.profile_name, self.current_item_id, self.cog._decrypt_data(item_to_edit.get("sum", "")))
        
        if modal:
            original_on_submit = modal.on_submit
            async def on_submit_refresh(i: discord.Interaction):
                await original_on_submit(i)
                if not i.response.is_done(): await i.response.defer()
                await self._update_view(self.original_interaction)
            modal.on_submit = on_submit_refresh
            await interaction.response.send_modal(modal)
        else:
            await interaction.response.send_message("Could not find the selected item to edit.", ephemeral=True)

    async def delete_callback(self, interaction: discord.Interaction):
        if not self.current_item_id: return
        
        confirm_view = ui.View(timeout=60)
        async def confirm_delete(i: discord.Interaction):
            user_id_str = str(self.user_id)
            deleted = False
            item_id_to_delete = self.current_item_id
            if self.mode == 'training':
                training_shard = self.cog._load_training_shard(user_id_str, self.profile_name) or []
                new_list = [item for item in training_shard if item.get("id") != item_id_to_delete]
                if len(new_list) < len(training_shard):
                    self.cog._save_training_shard(user_id_str, self.profile_name, new_list)
                    deleted = True
            else: # ltm
                context_type: Literal["guild", "dm"] = "guild" if self.guild_id else "dm"
                ltm_shard = self.cog._load_ltm_shard(user_id_str, self.profile_name)
                if ltm_shard:
                    data_list = ltm_shard.get(context_type, [])
                    new_list = [item for item in data_list if item.get("id") != item_id_to_delete]
                    if len(new_list) < len(data_list):
                        ltm_shard[context_type] = new_list
                        self.cog._save_ltm_shard(user_id_str, self.profile_name, ltm_shard)
                        deleted = True
            
            if deleted:
                await i.response.edit_message(content=f"Item `{item_id_to_delete}` deleted.", view=None, embed=None)
                self.current_page = max(1, self.current_page - 1)
                await self._update_view(self.original_interaction)
            else:
                await i.response.edit_message(content="Could not find item to delete.", view=None, embed=None)

        confirm_button = ui.Button(label="Confirm Deletion", style=discord.ButtonStyle.danger)
        confirm_button.callback = confirm_delete
        confirm_view.add_item(confirm_button)
        await interaction.response.send_message(f"**Are you sure you want to delete item `{self.current_item_id}`?**", view=confirm_view, ephemeral=True)

    async def search_callback(self, interaction: discord.Interaction):
        modal = SearchDataModal(self)
        await interaction.response.send_modal(modal)

class AnalyseExamplesModal(ui.Modal, title="Analyse Training Examples"):
    def __init__(self, parent_view: 'DataManageView'):
        super().__init__()
        self.parent_view = parent_view
        self.count_input = ui.TextInput(label="Number of Examples to Process", placeholder="Default: 10", default="10", required=True, min_length=1, max_length=3)
        self.verbosity_input = ui.TextInput(label="Target Verbosity (50 - 3000 chars)", placeholder="Default: 800", default="800", required=True, min_length=2, max_length=4)
        self.model_input = ui.TextInput(label="Analysis Model", placeholder="Default: GOOGLE/gemini-flash-lite-latest", default="GOOGLE/gemini-flash-lite-latest", required=True)
        self.add_item(self.count_input)
        self.add_item(self.verbosity_input)
        self.add_item(self.model_input)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            count = int(self.count_input.value)
            verbosity = int(self.verbosity_input.value)
            model_name = self.model_input.value.strip()
            
            if not (50 <= verbosity <= 3000): raise ValueError("Verbosity out of range.")
            if count < 1: raise ValueError("Count too low.")
            if not (model_name.upper().startswith("GOOGLE/") or model_name.upper().startswith("OPENROUTER/")):
                raise ValueError("Model must start with GOOGLE/ or OPENROUTER/.")
        except ValueError as e:
            await interaction.response.send_message(f"❌ **Invalid Input:** {e}", ephemeral=True); return

        await interaction.response.defer(ephemeral=True, thinking=True)
        await self.parent_view.cog._execute_training_analysis(interaction, self.parent_view.profile_name, count, verbosity, model_name)

class SubmitAPIKeyModal(ui.Modal, title="Edit API Key"):
    key_input = ui.TextInput(label="API Key (Auto Detect)", placeholder="Paste key to set, or leave blank to remove.", required=False)

    def __init__(self, cog: 'GeminiAgent', target_type: str, guild_id: Optional[int] = None, view: Optional[ui.View] = None):
        super().__init__()
        self.cog = cog
        self.target_type = target_type 
        self.guild_id = guild_id
        self.view = view

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        raw_key = self.key_input.value.strip()
        
        # --- Deletion Logic ---
        if not raw_key:
            if self.target_type == 'personal':
                self.cog._save_personal_api_key_shard(str(interaction.user.id), None)
                if str(interaction.user.id) in self.cog.personal_api_keys: del self.cog.personal_api_keys[str(interaction.user.id)]
                msg = "Personal key removed."
                
            elif self.target_type == 'server_primary':
                sid = str(self.guild_id)
                if sid in self.cog.server_api_keys:
                    del self.cog.server_api_keys[sid]
                    subs = self.cog.key_submissions.get(sid, [])
                    self.cog._save_server_api_key_shard(sid, None, subs)
                msg = "Server Primary key removed."
                
            elif self.target_type == 'server_pool':
                sid = str(self.guild_id)
                subs = self.cog.key_submissions.get(sid, [])
                new_subs = [s for s in subs if s['submitter_id'] != interaction.user.id]
                self.cog.key_submissions[sid] = new_subs
                self.cog._save_key_submissions_shard(sid, new_subs)
                msg = "Your pool submission removed."

            if self.view: await self.view.update_display()
            await interaction.followup.send(msg, ephemeral=True)
            return

        # --- Validation & Setting Logic ---
        provider = None
        if raw_key.startswith("AIzaSy"): provider = "gemini"
        elif raw_key.startswith("sk-or-"): provider = "openrouter"
        
        if not provider:
            await interaction.followup.send("❌ **Invalid Format.** Keys must start with `AIzaSy` or `sk-or-`.", ephemeral=True)
            return

        is_valid, err, tier = await self.cog._validate_api_keys(
            raw_key if provider == "gemini" else None, 
            raw_key if provider == "openrouter" else None
        )
        
        if not is_valid:
            await interaction.followup.send(f"❌ **Validation Failed:** {err}", ephemeral=True)
            return

        encrypted_key = self.cog._encrypt_data(raw_key)
        
        if self.target_type == 'personal':
            user_id_str = str(interaction.user.id)
            path = os.path.join(self.cog.PERSONAL_KEYS_DIR, f"{user_id_str}.json.gz")
            existing_data = self.cog._load_json_gzip(path) or {}
            
            key_field = "key" if provider == "gemini" else "openrouter_key"
            existing_data[key_field] = encrypted_key
            existing_data["tier"] = tier
            self.cog._atomic_json_save_gzip(existing_data, path)
            self.cog.personal_api_keys[user_id_str] = existing_data.get("key") 
            
            msg = f"✅ Personal {provider.title()} key updated ({tier.title()} Tier)."

        elif self.target_type == 'server_primary':
            guild_id_str = str(self.guild_id)
            key_data = self.cog.server_api_keys.get(guild_id_str, {})
            if not isinstance(key_data, dict): key_data = {}
            
            key_field = "key" if provider == "gemini" else "openrouter_key"
            key_data[key_field] = encrypted_key
            key_data["submitter_id"] = interaction.user.id
            key_data["tier"] = tier
            
            self.cog.server_api_keys[guild_id_str] = key_data
            submissions = self.cog.key_submissions.get(guild_id_str, [])
            self.cog._save_server_api_key_shard(guild_id_str, key_data, submissions)
            
            msg = f"✅ Server Primary {provider.title()} key updated ({tier.title()} Tier)."

        elif self.target_type == 'server_pool':
            guild_id_str = str(self.guild_id)
            submission = {
                "submitter_id": interaction.user.id,
                "encrypted_key": encrypted_key,
                "provider": provider,
                "status": "active",
                "tier": tier,
                "submitted_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            
            current_subs = self.cog.key_submissions.get(guild_id_str, [])
            new_subs = [s for s in current_subs if s['submitter_id'] != interaction.user.id]
            new_subs.append(submission)
            
            self.cog.key_submissions[guild_id_str] = new_subs
            self.cog._save_key_submissions_shard(guild_id_str, new_subs)
            msg = f"✅ {provider.title()} key added to server pool ({tier.title()} Tier)."

        if self.view: await self.view.update_display()
        await interaction.followup.send(msg, ephemeral=True)

class SettingsBaseView(ui.View):
    def __init__(self, cog: 'GeminiAgent', interaction: discord.Interaction, current_tab: str):
        super().__init__(timeout=600)
        self.cog = cog
        self.original_interaction = interaction
        self.user_id = interaction.user.id
        self.current_tab = current_tab
        self._add_nav_buttons()

    def _add_nav_buttons(self):
        btn_dash = ui.Button(label="Home", style=discord.ButtonStyle.primary if self.current_tab == "home" else discord.ButtonStyle.secondary, row=4, disabled=(self.current_tab=="home"))
        btn_dash.callback = self.nav_home
        
        btn_api = ui.Button(label="API Keys", style=discord.ButtonStyle.primary if self.current_tab == "api" else discord.ButtonStyle.secondary, row=4, disabled=(self.current_tab=="api"))
        btn_api.callback = self.nav_api
        
        btn_bot = ui.Button(label="Child Bots", style=discord.ButtonStyle.primary if self.current_tab == "bots" else discord.ButtonStyle.secondary, row=4, disabled=(self.current_tab=="bots"))
        btn_bot.callback = self.nav_bots

        self.add_item(btn_dash)
        self.add_item(btn_api)
        self.add_item(btn_bot)

    async def nav_home(self, i: discord.Interaction):
        await i.response.defer()
        view = SettingsHomeView(self.cog, self.original_interaction)
        await view.update_display()

    async def nav_api(self, i: discord.Interaction):
        await i.response.defer()
        view = SettingsAPIView(self.cog, self.original_interaction)
        await view.update_display()

    async def nav_bots(self, i: discord.Interaction):
        await i.response.defer()
        view = SettingsChildBotView(self.cog, self.original_interaction)
        await view.update_display()

class SettingsHomeView(SettingsBaseView):
    def __init__(self, cog: 'GeminiAgent', interaction: discord.Interaction):
        super().__init__(cog, interaction, "home")

    async def update_display(self):
        # Gather Stats
        path = os.path.join(self.cog.PERSONAL_KEYS_DIR, f"{str(self.user_id)}.json.gz")
        u_data = self.cog._load_json_gzip(path)
        u_tier = u_data.get("tier", "free").title() if u_data else "Free"

        p_key_gemini = self.cog._get_api_key_for_user(self.user_id, "gemini")
        p_key_or = self.cog._get_api_key_for_user(self.user_id, "openrouter")
        
        stat_gemini = f"✅ **`Set ({u_tier} Tier)`**" if p_key_gemini else "❌ `Not Set`"
        stat_or = f"✅ **`Set (Paid Tier)`**" if p_key_or else "❌ `Not Set`"
        
        child_bots = [b for b in self.cog.child_bots.values() if b['owner_id'] == self.user_id]
        bot_text = f"You own **{len(child_bots)}** Child Bots." if child_bots else "You do not own any Child Bots."
        
        primary_count = 0
        pool_count = 0
        for gid_str, key_data in self.cog.server_api_keys.items():
            if isinstance(key_data, dict) and key_data.get('submitter_id') == self.user_id:
                primary_count += 1
        for submissions in self.cog.key_submissions.values():
            for s in submissions:
                if s.get('submitter_id') == self.user_id: pool_count += 1

        embed = discord.Embed(title="MimicAI Control Panel", description="Manage your API keys and personal bots from one place.", color=discord.Color.dark_teal())
        embed.set_thumbnail(url="https://cdn.discordapp.com/emojis/1441750712160878643.gif")
        
        embed.add_field(name="Personal API Keys", value=f"**Gemini:** {stat_gemini}\n**OpenRouter:** {stat_or}", inline=True)
        embed.add_field(name="Child Bots", value=bot_text, inline=True)
        embed.add_field(name="Server Contributions", value=f"Primary Key Owner: `{primary_count} servers`\nKey Pool Contributor: `{pool_count} servers`", inline=False)
        
        await self.original_interaction.edit_original_response(content=None, embed=embed, view=self)

class SettingsAPIView(SettingsBaseView):
    def __init__(self, cog: 'GeminiAgent', interaction: discord.Interaction):
        super().__init__(cog, interaction, "api")
        self.selected_server_id = None
        self.current_page = 0
        self.setup_items()

    def setup_items(self):
        for item in self.children[:]:
            if item.row != 4: self.remove_item(item)

        # Gather Servers
        mutual_guilds = []
        user = self.cog.bot.get_user(self.user_id)
        if user:
            for g in self.cog.bot.guilds:
                if g.get_member(self.user_id):
                    mutual_guilds.append(g)
        
        # We handle "Personal Keys" as a special case or the first option
        # To simplify pagination logic, let's treat "Personal" as a static option 
        # and paginate only the server list if needed.
        
        all_options = [discord.SelectOption(label="Personal Keys (Global Chat)", value="personal", default=(self.selected_server_id is None))]
        for g in mutual_guilds:
            all_options.append(discord.SelectOption(label=f"Server: {g.name}", value=str(g.id), default=(self.selected_server_id == g.id)))

        # Pagination Logic
        num_pages = (len(all_options) - 1) // DROPDOWN_MAX_OPTIONS + 1
        if self.current_page >= num_pages: self.current_page = max(0, num_pages - 1)
        
        start = self.current_page * DROPDOWN_MAX_OPTIONS
        page_options = all_options[start : start + DROPDOWN_MAX_OPTIONS]
        
        select = ui.Select(placeholder="Select Context...", options=page_options, row=0)
        select.callback = self.context_select
        self.add_item(select)

        # Pagination Buttons (Row 1 - If Needed)
        btn_start_row = 1
        if num_pages > 1:
            prev_btn = ui.Button(label="◀", style=discord.ButtonStyle.secondary, row=1, disabled=(self.current_page == 0))
            page_lbl = ui.Button(label=f"{self.current_page + 1}/{num_pages}", style=discord.ButtonStyle.grey, row=1, disabled=True)
            next_btn = ui.Button(label="▶", style=discord.ButtonStyle.secondary, row=1, disabled=(self.current_page >= num_pages - 1))
            
            prev_btn.callback = self.prev_page
            next_btn.callback = self.next_page
            
            self.add_item(prev_btn)
            self.add_item(page_lbl)
            self.add_item(next_btn)
            btn_start_row = 2

        # Dynamic Action Buttons (Row 1 or 2)
        if self.selected_server_id is None:
            # Personal Mode
            btn_edit = ui.Button(label="Edit Personal Key", style=discord.ButtonStyle.primary, row=btn_start_row)
            btn_edit.callback = self.personal_edit
            self.add_item(btn_edit)
        else:
            # Server Mode
            guild = self.cog.bot.get_guild(self.selected_server_id)
            if guild:
                member = guild.get_member(self.user_id)
                is_admin = member.guild_permissions.administrator if member else False
                
                if is_admin:
                    btn_prim = ui.Button(label="Edit Primary Key", style=discord.ButtonStyle.primary, row=btn_start_row)
                    btn_prim.callback = self.server_edit_primary
                    self.add_item(btn_prim)
                
                btn_pool = ui.Button(label="Edit Pool Key", style=discord.ButtonStyle.secondary, row=btn_start_row)
                btn_pool.callback = self.server_edit_pool
                self.add_item(btn_pool)

    async def update_display(self):
        embed = discord.Embed(title="API Key Management", color=discord.Color.blue())
        
        if self.selected_server_id is None:
            # Personal View
            path = os.path.join(self.cog.PERSONAL_KEYS_DIR, f"{str(self.user_id)}.json.gz")
            u_data = self.cog._load_json_gzip(path)
            u_tier = u_data.get("tier", "free").title() if u_data else "Free"

            has_gem = bool(self.cog._get_api_key_for_user(self.user_id, "gemini"))
            has_or = bool(self.cog._get_api_key_for_user(self.user_id, "openrouter"))
            
            embed.description = "Managing keys for **Global Chat** (DMs)."
            embed.add_field(name="Status", value=f"**Google Gemini:** {'✅ **`Set ('+u_tier+' Tier)`**' if has_gem else '❌ `Not Set`'}\n**OpenRouter:** {'✅ **`Set (Paid Tier)`**' if has_or else '❌ `Not Set`'}", inline=False)
        else:
            # Server View
            guild = self.cog.bot.get_guild(self.selected_server_id)
            name = guild.name if guild else "Unknown"
            embed.description = f"Managing keys for **{name}**."
            
            # Check Primary Key Status
            pk_data = self.cog.server_api_keys.get(str(self.selected_server_id))
            pk_tier = "Free"
            if pk_data and isinstance(pk_data, dict):
                has_gem_srv = bool(pk_data.get('key'))
                has_or_srv = bool(pk_data.get('openrouter_key'))
                pk_tier = pk_data.get("tier", "free").title()
            else:
                has_gem_srv = False
                has_or_srv = False
            
            pool = self.cog.key_submissions.get(str(self.selected_server_id), [])
            user_in_pool = any(s['submitter_id'] == self.user_id for s in pool)
            
            embed.add_field(name="Primary Key Status", value=f"**Google Gemini:** {'✅ **`Set ('+pk_tier+' Tier)`**' if has_gem_srv else '❌ `Not Set`'}\n**OpenRouter:** {'✅ **`Set (Paid Tier)`**' if has_or_srv else '❌ `Not Set`'}", inline=False)
            embed.add_field(name="Key Pool", value=f"**{len(pool)}** user-submitted keys active.", inline=False)
            
            if user_in_pool:
                embed.set_footer(text="You have a key contributed to this server's pool.")

        await self.original_interaction.edit_original_response(content=None, embed=embed, view=self)

    async def context_select(self, i: discord.Interaction):
        val = i.data['values'][0]
        self.selected_server_id = int(val) if val != "personal" else None
        self.setup_items()
        await i.response.defer()
        await self.update_display()

    async def prev_page(self, i: discord.Interaction):
        self.current_page -= 1
        self.setup_items()
        await i.response.defer()
        await self.update_display()

    async def next_page(self, i: discord.Interaction):
        self.current_page += 1
        self.setup_items()
        await i.response.defer()
        await self.update_display()

    async def personal_edit(self, i: discord.Interaction):
        modal = SubmitAPIKeyModal(self.cog, "personal", view=self)
        await i.response.send_modal(modal)

    async def server_edit_primary(self, i: discord.Interaction):
        modal = SubmitAPIKeyModal(self.cog, "server_primary", self.selected_server_id, view=self)
        await i.response.send_modal(modal)

    async def server_edit_pool(self, i: discord.Interaction):
        modal = SubmitAPIKeyModal(self.cog, "server_pool", self.selected_server_id, view=self)
        await i.response.send_modal(modal)

class SettingsChildBotView(SettingsBaseView):
    def __init__(self, cog: 'GeminiAgent', interaction: discord.Interaction):
        super().__init__(cog, interaction, "bots")
        self.selected_bot_id = None
        self.setup_items()

    def setup_items(self):
        for item in self.children[:]:
            if item.row != 4: self.remove_item(item)

        user_bots = [b for b_id, b in self.cog.child_bots.items() if b['owner_id'] == self.user_id]
        
        # Row 0: Select Bot
        options = []
        for b_data in user_bots:
            # We need bot_id to be in b_data or iterate items
            # Refetch with ID
            pass 
        
        # Re-iterate correctly
        user_bot_items = [(bid, b) for bid, b in self.cog.child_bots.items() if b['owner_id'] == self.user_id]
        
        for bid, b_data in user_bot_items:
            bot_user = self.cog.bot.get_user(int(bid))
            name = bot_user.name if bot_user else f"ID: {bid}"
            options.append(discord.SelectOption(label=f"{name} ({b_data.get('profile_name')})", value=bid, default=(bid == self.selected_bot_id)))

        if options:
            select = ui.Select(placeholder="Select a child bot...", options=options[:25], row=0)
            select.callback = self.select_bot
            self.add_item(select)

        # Row 1: Actions
        is_premium = self.cog.is_user_premium(self.user_id)
        create_style = discord.ButtonStyle.green if is_premium else discord.ButtonStyle.secondary
        create_label = "Create New Child Bot" if is_premium else "Create (Premium Only)"
        
        btn_create = ui.Button(label=create_label, style=create_style, row=1, disabled=(not is_premium))
        btn_create.callback = self.create_bot
        self.add_item(btn_create)

        if self.selected_bot_id:
            # [REMOVED] Manage Approved Servers button
            
            btn_del = ui.Button(label="Unlink & Delete", style=discord.ButtonStyle.danger, row=1)
            btn_del.callback = self.delete_bot
            self.add_item(btn_del)

    async def update_display(self):
        embed = discord.Embed(title="My Child Bots", description="Manage your linked bot applications.", color=discord.Color.dark_magenta())
        if self.selected_bot_id:
            bot_user = self.cog.bot.get_user(int(self.selected_bot_id))
            name = bot_user.name if bot_user else self.selected_bot_id
            b_data = self.cog.child_bots.get(self.selected_bot_id)
            p_name = b_data.get('profile_name')
            embed.add_field(name="Selected Bot", value=f"**Name:** `{name}`\n**Linked Profile:** `{p_name}`", inline=False)
        else:
            embed.add_field(name="Overview", value="Select a bot from the dropdown to manage it, or create a new one.", inline=False)
        
        await self.original_interaction.edit_original_response(content=None, embed=embed, view=self)

    async def select_bot(self, i: discord.Interaction):
        self.selected_bot_id = i.data['values'][0]
        self.setup_items()
        await i.response.defer()
        await self.update_display()

    async def create_bot(self, i: discord.Interaction):
        if not self.cog.is_user_premium(self.user_id):
            await i.response.send_message("This feature requires a Premium Tier.", ephemeral=True)
            return

        self._build_child_bot_list_ui = lambda x: self.update_rebuild(x) 
        modal = ChildBotCreateModal(self.cog, self)
        await i.response.send_modal(modal)

    async def update_rebuild(self, i: discord.Interaction):
        # Callback for the modal to refresh UI
        self.setup_items()
        await self.update_display()

    async def delete_bot(self, i: discord.Interaction):
        # Logic from delete_child_bot_callback
        bot_to_delete = self.cog.child_bots.get(self.selected_bot_id)
        if bot_to_delete:
            owner_id = bot_to_delete['owner_id']
            user_shard = self.cog._get_user_child_bot_shard(owner_id)
            if self.selected_bot_id in user_shard:
                del user_shard[self.selected_bot_id]
                self.cog._save_user_child_bot_shard(owner_id, user_shard)
                self.cog._load_child_bots()
                await self.cog.manager_queue.put({"action": "shutdown_bot", "bot_id": self.selected_bot_id})
        
        self.selected_bot_id = None
        self.setup_items()
        await self.update_display()
        await i.response.send_message("Bot deleted.", ephemeral=True)

class BorrowNameModal(ui.Modal, title="Name Your Borrowed Profile"):
    profile_name_input = ui.TextInput(label="Enter a unique local name", required=True, min_length=1, max_length=50)
    
    def __init__(self, cog: 'GeminiAgent', original_interaction: discord.Interaction, sharer_id: int, profile_to_borrow: str, is_public_borrow: bool = False):
        super().__init__()
        self.cog = cog
        self.original_interaction = original_interaction
        self.sharer_id = sharer_id
        self.profile_to_borrow = profile_to_borrow
        self.is_public_borrow = is_public_borrow

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        desired_name = self.profile_name_input.value.lower().strip()
        
        user_data = self.cog._get_user_data_entry(interaction.user.id)
        if desired_name in user_data.get("profiles", {}) or desired_name in user_data.get("borrowed_profiles", {}):
            await interaction.followup.send(f"You already have a profile named '{desired_name}'. Please choose a different name.", ephemeral=True)
            return

        await self.cog._accept_share_request(interaction, self.sharer_id, self.profile_to_borrow, desired_name, self.is_public_borrow)
        await interaction.followup.send(f"✅ Successfully borrowed profile **{self.profile_to_borrow}** and named it **{desired_name}**. You can now use it with `/profile swap`.", ephemeral=True)

class ActionTextInputModal(ui.Modal):
    def __init__(self, title: str, label: str, placeholder: str, on_submit_callback, default: Optional[str] = None, required: bool = True):
        super().__init__(title=title)
        self.on_submit_callback = on_submit_callback
        self.input = ui.TextInput(
            label=label,
            placeholder=placeholder,
            default=default,
            style=discord.TextStyle.paragraph,
            required=required
        )
        self.add_item(self.input)

    async def on_submit(self, interaction: discord.Interaction):
        await self.on_submit_callback(interaction, self.input.value)

class WakewordsModal(ui.Modal, title="Manage Wakewords"):
    wakewords_input = ui.TextInput(label="Wakewords (comma-separated)", style=discord.TextStyle.paragraph, required=False, max_length=1000)

    def __init__(self, current_wakewords: List[str]):
        super().__init__()
        self.wakewords_input.default = ", ".join(current_wakewords)

class FreewillChanceModal(ui.Modal, title="Set Response Chance"):
    chance_input = ui.TextInput(label="Probability (0-100%)", placeholder="e.g. 15", required=True, max_length=3)

    def __init__(self, cog, guild_id, channel_id, user_id, profile_name, view):
        super().__init__()
        self.cog = cog
        self.guild_id = guild_id
        self.channel_id = channel_id
        self.user_id = user_id
        self.profile_name = profile_name
        self.view = view

    async def on_submit(self, interaction: discord.Interaction):
        try:
            val = int(self.chance_input.value)
            val = max(0, min(100, val))
        except ValueError:
            await interaction.response.send_message("Invalid input. Please enter a whole number between 0 and 100.", ephemeral=True); return

        await interaction.response.defer()
        
        # Save logic
        g_str, c_str, u_str = str(self.guild_id), str(self.channel_id), str(self.user_id)
        if self.profile_name in self.cog.freewill_participation.get(g_str, {}).get(c_str, {}).get(u_str, {}):
            self.cog.freewill_participation[g_str][c_str][u_str][self.profile_name]["personality"] = val
            self.cog._save_freewill_for_server(self.guild_id)
        
        await self.view.refresh_state(interaction)

class FreewillChannelChanceModal(ui.Modal, title="Set Channel Event Chance"):
    chance_input = ui.TextInput(label="Probability (0-100%)", placeholder="e.g. 5", required=True, max_length=3)

    def __init__(self, cog, guild_id, channel_id, view):
        super().__init__()
        self.cog = cog
        self.guild_id = guild_id
        self.channel_id = channel_id
        self.view = view

    async def on_submit(self, interaction: discord.Interaction):
        try:
            val = int(self.chance_input.value)
            val = max(0, min(100, val))
        except ValueError:
            await interaction.response.send_message("Invalid input.", ephemeral=True); return

        msg = f"Channel Event Chance set to **{val}%**."
        if val > 50: msg += "\n⚠️ **Warning:** High frequency."

        config = self.cog.freewill_config.setdefault(str(self.guild_id), {})
        ch_settings = config.setdefault("channel_settings", {})
        ch_settings.setdefault(str(self.channel_id), {})["event_chance"] = val
        
        self.cog._save_channel_settings()
        
        await interaction.response.send_message(msg, ephemeral=True)
        await self.view.update_display()

class FreewillCooldownModal(ui.Modal, title="Set Event Cooldown"):
    cd_input = ui.TextInput(label="Cooldown (Minutes)", placeholder="e.g. 30", required=True, max_length=4)

    def __init__(self, cog, guild_id, channel_id, view):
        super().__init__()
        self.cog = cog
        self.guild_id = guild_id
        self.channel_id = channel_id
        self.view = view

    async def on_submit(self, interaction: discord.Interaction):
        try:
            val = int(self.cd_input.value)
            val = max(1, min(1440, val)) # 1 min to 24 hours
        except ValueError:
            await interaction.response.send_message("Invalid input.", ephemeral=True); return

        seconds = val * 60
        config = self.cog.freewill_config.setdefault(str(self.guild_id), {})
        ch_settings = config.setdefault("channel_settings", {})
        ch_settings.setdefault(str(self.channel_id), {})["event_cooldown"] = seconds
        
        self.cog._save_channel_settings()
        
        await interaction.response.send_message(f"Event Cooldown set to **{val} minutes**.", ephemeral=True)
        await self.view.update_display()

class FreewillBaseView(ui.View):
    def __init__(self, cog: 'GeminiAgent', interaction: discord.Interaction, current_tab: str):
        super().__init__(timeout=600)
        self.cog = cog
        self.original_interaction = interaction
        self.user_id = interaction.user.id
        self.guild_id = interaction.guild_id
        self.channel_id = interaction.channel_id
        self.current_tab = current_tab
        self._add_nav_buttons()

    def _add_nav_buttons(self):
        # Navigation Row (Row 4)
        style_home = discord.ButtonStyle.primary if self.current_tab == "home" else discord.ButtonStyle.secondary
        style_part = discord.ButtonStyle.primary if self.current_tab == "participants" else discord.ButtonStyle.secondary
        style_conf = discord.ButtonStyle.primary if self.current_tab == "config" else discord.ButtonStyle.secondary

        b1 = ui.Button(label="Home", style=style_home, row=4, disabled=(self.current_tab=="home"))
        b2 = ui.Button(label="Participants", style=style_part, row=4, disabled=(self.current_tab=="participants"))
        b3 = ui.Button(label="Config", style=style_conf, row=4, disabled=(self.current_tab=="config"))

        b1.callback = self.nav_home
        b2.callback = self.nav_part
        b3.callback = self.nav_conf
        
        self.add_item(b1); self.add_item(b2); self.add_item(b3)

    async def nav_home(self, i: discord.Interaction):
        await i.response.defer()
        view = FreewillHomeView(self.cog, self.original_interaction)
        await view.update_display()

    async def nav_part(self, i: discord.Interaction):
        await i.response.defer()
        view = FreewillParticipantsView(self.cog, self.original_interaction)
        await view.update_display()

    async def nav_conf(self, i: discord.Interaction):
        await i.response.defer()
        view = FreewillConfigView(self.cog, self.original_interaction)
        await view.update_display()

class FreewillHomeView(FreewillBaseView):
    def __init__(self, cog, interaction):
        super().__init__(cog, interaction, "home")

    async def update_display(self):
        config = self.cog.freewill_config.get(str(self.guild_id), {})
        
        living = config.get("living_channel_ids", [])
        lurking = config.get("lurking_channel_ids", [])
        
        ch_status = "Reactive (Lurking)" # Default
        ch_color = discord.Color.gold()
        if self.channel_id in living: 
            ch_status, ch_color = "Proactive (Living)", discord.Color.green()

        # Count participants
        count = 0
        srv_data = self.cog.freewill_participation.get(str(self.guild_id), {})
        ch_data = srv_data.get(str(self.channel_id), {})
        for u_dict in ch_data.values(): count += len(u_dict)

        embed = discord.Embed(title="Freewill System: Home", color=ch_color)
        embed.add_field(name="Channel Mode", value=f"**{ch_status}**", inline=True)
        embed.add_field(name="Participants", value=str(count), inline=True)
        embed.set_footer(text=f"Server Active Channels: {len(living)} Living | {len(lurking)} Lurking")

        self.clear_items()
        self._add_nav_buttons()
        
        mode_btn = ui.Button(label="Toggle Mode (Living/Lurking)", style=discord.ButtonStyle.primary, row=0)
        mode_btn.callback = self.toggle_channel
        self.add_item(mode_btn)

        await self.original_interaction.edit_original_response(embed=embed, view=self)

    async def toggle_channel(self, i: discord.Interaction):
        await i.response.defer()
        conf = self.cog.freewill_config.setdefault(str(self.guild_id), {})
        lid = conf.setdefault("living_channel_ids", [])
        lud = conf.setdefault("lurking_channel_ids", [])
        cid = self.channel_id
        
        if cid in lid:
            lid.remove(cid)
            lud.append(cid)
        else:
            if cid in lud: lud.remove(cid)
            lid.append(cid)
        
        self.cog._save_channel_settings()
        await self.update_display()

class FreewillParticipantsView(FreewillBaseView):
    def __init__(self, cog, interaction):
        super().__init__(cog, interaction, "participants")
        self.selected_user_id = None
        self.selected_profile_name = None

    async def update_display(self):
        srv_data = self.cog.freewill_participation.get(str(self.guild_id), {})
        ch_data = srv_data.get(str(self.channel_id), {})
        
        all_participants = []
        for uid, profiles in ch_data.items():
            for pname, settings in profiles.items():
                # Lazy load legacy conversion for display
                p_val = settings.get("personality", 10)
                if isinstance(p_val, str):
                    p_val = {"introverted": 10, "regular": 50, "outgoing": 90, "off": 0}.get(p_val, 10)
                
                label = f"{pname} ({p_val}%)"
                all_participants.append({"label": label, "value": f"{uid}:{pname}", "uid": uid, "name": pname, "chance": p_val, "settings": settings})
        
        all_participants.sort(key=lambda x: x['name'])

        self.clear_items()
        self._add_nav_buttons()

        embed = discord.Embed(title="Freewill: Participants", color=discord.Color.blue())

        if self.selected_profile_name:
            # Selected State
            p_data = next((p for p in all_participants if p['uid'] == self.selected_user_id and p['name'] == self.selected_profile_name), None)
            if p_data:
                settings = p_data['settings']
                embed.description = f"Managing **{p_data['name']}** (User ID: {p_data['uid']})"
                embed.add_field(name="Response Chance", value=f"`{p_data['chance']}%`", inline=True)
                embed.add_field(name="Method", value=f"`{settings.get('method', 'webhook')}`", inline=True)
                wakes = ", ".join(settings.get("wakewords", [])) or "None"
                if len(wakes) > 50: wakes = wakes[:47] + "..."
                embed.add_field(name="Wakewords", value=f"`{wakes}`", inline=False)

                btn_chance = ui.Button(label="Set Chance %", style=discord.ButtonStyle.primary, row=1)
                btn_chance.callback = self.set_chance_cb
                self.add_item(btn_chance)

                btn_wake = ui.Button(label="Edit Wakewords", style=discord.ButtonStyle.secondary, row=1)
                btn_wake.callback = self.edit_wake_cb
                self.add_item(btn_wake)

                btn_method = ui.Button(label="Toggle Method", style=discord.ButtonStyle.secondary, row=1)
                btn_method.callback = self.toggle_method_cb
                self.add_item(btn_method)

                btn_rem = ui.Button(label="Remove", style=discord.ButtonStyle.danger, row=1)
                btn_rem.callback = self.remove_cb
                self.add_item(btn_rem)
                
                btn_back = ui.Button(label="Back to List", style=discord.ButtonStyle.grey, row=2)
                btn_back.callback = self.back_cb
                self.add_item(btn_back)
            else:
                self.selected_profile_name = None
                await self.update_display(); return
        else:
            # List State
            embed.description = f"Total Participants: **{len(all_participants)}**\nSelect a profile to manage details."
            
            if all_participants:
                # Limit to 25 for dropdown
                options = [discord.SelectOption(label=p['label'], value=p['value']) for p in all_participants[:25]]
                sel = ui.Select(placeholder="Select a profile...", options=options, row=0)
                sel.callback = self.select_cb
                self.add_item(sel)

            btn_add = ui.Button(label="Add Profile", style=discord.ButtonStyle.success, row=1)
            btn_add.callback = self.add_profile_cb
            self.add_item(btn_add)

            if all_participants:
                btn_clear = ui.Button(label="Release All", style=discord.ButtonStyle.danger, row=1)
                btn_clear.callback = self.release_all_cb
                self.add_item(btn_clear)

        await self.original_interaction.edit_original_response(embed=embed, view=self)

    async def select_cb(self, i: discord.Interaction):
        await i.response.defer()
        uid, pname = i.data['values'][0].split(":", 1)
        self.selected_user_id = uid
        self.selected_profile_name = pname
        await self.update_display()

    async def back_cb(self, i: discord.Interaction):
        await i.response.defer()
        self.selected_profile_name = None
        await self.update_display()

    async def set_chance_cb(self, i: discord.Interaction):
        modal = FreewillChanceModal(self.cog, self.guild_id, self.channel_id, self.selected_user_id, self.selected_profile_name, self)
        await i.response.send_modal(modal)

    async def refresh_state(self, i: discord.Interaction):
        # Callback for modal to refresh UI without full rebuild
        await self.update_display()

    async def edit_wake_cb(self, i: discord.Interaction):
        # Reuse existing logic via modal
        srv = self.cog.freewill_participation.get(str(self.guild_id), {})
        chn = srv.get(str(self.channel_id), {})
        usr = chn.get(str(self.selected_user_id), {})
        prof = usr.get(self.selected_profile_name, {})
        
        modal = WakewordsModal(prof.get("wakewords", []))
        async def modal_callback(mi: discord.Interaction):
            await mi.response.defer()
            words = [w.strip().lower() for w in modal.wakewords_input.value.split(',') if w.strip()]
            self.cog.freewill_participation[str(self.guild_id)][str(self.channel_id)][str(self.selected_user_id)][self.selected_profile_name]["wakewords"] = words
            self.cog._save_freewill_for_server(self.guild_id)
            await self.update_display()
        modal.on_submit = modal_callback
        await i.response.send_modal(modal)

    async def toggle_method_cb(self, i: discord.Interaction):
        await i.response.defer()
        # Toggle Webhook <-> Child Bot
        # Note: Doesn't verify if child bot exists, Admin is responsible
        srv = self.cog.freewill_participation[str(self.guild_id)][str(self.channel_id)]
        curr = srv[str(self.selected_user_id)][self.selected_profile_name].get("method", "webhook")
        new_m = "child_bot" if curr == "webhook" else "webhook"
        srv[str(self.selected_user_id)][self.selected_profile_name]["method"] = new_m
        self.cog._save_freewill_for_server(self.guild_id)
        await self.update_display()

    async def remove_cb(self, i: discord.Interaction):
        await i.response.defer()
        del self.cog.freewill_participation[str(self.guild_id)][str(self.channel_id)][str(self.selected_user_id)][self.selected_profile_name]
        self.cog._save_freewill_for_server(self.guild_id)
        self.selected_profile_name = None
        await self.update_display()

    async def release_all_cb(self, i: discord.Interaction):
        await i.response.defer()
        del self.cog.freewill_participation[str(self.guild_id)][str(self.channel_id)]
        self.cog._save_freewill_for_server(self.guild_id)
        self.selected_profile_name = None
        await self.update_display()

    async def add_profile_cb(self, i: discord.Interaction):
        # We call the new paginated UI instead of the old temporary select
        view = FreewillAddProfileView(self.cog, i.user.id, self.guild_id, self.channel_id, self)
        
        if not view.personal_profiles:
            await i.response.send_message("All of your profiles are already participating in this channel.", ephemeral=True)
            return
            
        await i.response.send_message("### Add Profiles to Freewill\nSelect your personal profiles to enable them for autonomous interaction in this channel.", view=view, ephemeral=True)

class FreewillConfigView(FreewillBaseView):
    def __init__(self, cog, interaction):
        super().__init__(cog, interaction, "config")

    async def update_display(self):
        conf = self.cog.freewill_config.get(str(self.guild_id), {})
        ch_settings = conf.get("channel_settings", {}).get(str(self.channel_id), {})
        
        # Default to 0 (Manual) if not set for this channel
        chance = ch_settings.get("event_chance", 0)
        
        # Default to 300s (5m) if not set for this channel
        cd = ch_settings.get("event_cooldown", 300)
        cd_min = cd // 60

        embed = discord.Embed(title="Freewill: Configuration", color=discord.Color.dark_grey())
        embed.add_field(name="Channel Event Chance", value=f"`{chance}%`", inline=True)
        embed.add_field(name="Event Cooldown", value=f"`{cd_min} minutes`", inline=True)
        embed.description = f"These settings apply **only** to <#{self.channel_id}>."

        self.clear_items()
        self._add_nav_buttons()

        b1 = ui.Button(label="Set Event Chance", style=discord.ButtonStyle.primary, row=0)
        b1.callback = self.set_chance
        self.add_item(b1)

        b2 = ui.Button(label="Set Cooldown", style=discord.ButtonStyle.primary, row=0)
        b2.callback = self.set_cooldown
        self.add_item(b2)

        await self.original_interaction.edit_original_response(embed=embed, view=self)

    async def set_chance(self, i: discord.Interaction):
        modal = FreewillChannelChanceModal(self.cog, self.guild_id, self.channel_id, self)
        await i.response.send_modal(modal)

    async def set_cooldown(self, i: discord.Interaction):
        modal = FreewillCooldownModal(self.cog, self.guild_id, self.channel_id, self)
        await i.response.send_modal(modal)

class FreewillAddProfileView(BaseBulkProfileView):
    def __init__(self, cog, user_id, guild_id, channel_id, parent_view):
        super().__init__(cog, user_id, include_borrowed=False)
        self.guild_id = guild_id
        self.channel_id = channel_id
        self.parent_view = parent_view
        self._build_view()

    def _build_view(self):
        self.clear_items()
        # Filter: Only show profiles NOT already participating in this channel
        curr_part = self.cog.freewill_participation.get(str(self.guild_id), {}).get(str(self.channel_id), {}).get(str(self.user_id), {})
        self.personal_profiles = [p for p in self.personal_profiles if p not in curr_part]
        
        if not self.personal_profiles:
            self.add_item(ui.Button(label="No available profiles", disabled=True, row=0))
            return

        self._build_profile_select_ui(row=0)
        add_btn = ui.Button(label="Add Selected Profiles", style=discord.ButtonStyle.success, row=2)
        add_btn.callback = self.add_callback
        self.add_item(add_btn)

    async def add_callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        if not self.selected_profiles:
            await interaction.followup.send("Please select at least one profile.", ephemeral=True)
            return

        g, c, u = str(self.guild_id), str(self.channel_id), str(self.user_id)
        for pname in self.selected_profiles:
            self.cog.freewill_participation.setdefault(g, {}).setdefault(c, {}).setdefault(u, {})[pname] = {
                "personality": 10, "wakewords": [], "method": "webhook"
            }
        
        self.cog._save_freewill_for_server(self.guild_id)
        # Update the main Participants list UI
        await self.parent_view.update_display()
        await interaction.followup.send(f"Successfully added {len(self.selected_profiles)} profile(s) to Freewill.", ephemeral=True)

class TypingManageView(BaseBulkProfileView):
    def __init__(self, cog: 'GeminiAgent', user_id: int):
        super().__init__(cog, user_id, include_borrowed=True)
        self.toggle_choice = None
        self._build_view()

    def _build_view(self):
        self.clear_items()
        toggle_options = [
            discord.SelectOption(label="Enable Realistic Typing", value="enable", default=(self.toggle_choice is True)),
            discord.SelectOption(label="Disable Realistic Typing", value="disable", default=(self.toggle_choice is False))
        ]
        toggle_select = ui.Select(placeholder="Choose an action...", options=toggle_options, row=0)
        toggle_select.callback = self.toggle_callback
        self.add_item(toggle_select)

        self._build_profile_select_ui(row=1)
        
        apply_action = ui.Button(label="Apply Action", style=discord.ButtonStyle.green, row=3)
        apply_action.callback = self.apply_action_callback
        self.add_item(apply_action)

    async def toggle_callback(self, interaction: discord.Interaction):
        self.toggle_choice = interaction.data['values'][0] == "enable"
        self._build_view()
        await interaction.response.edit_message(content=self._get_selection_feedback_message(), view=self)

    async def apply_action_callback(self, interaction: discord.Interaction):
        await interaction.response.defer()
        target_profiles = list(self.selected_profiles)
        if self.toggle_choice is None or not target_profiles:
            await interaction.edit_original_response(content="Please select an action and at least one profile.", view=None); return

        updated_count = 0
        user_data = self.cog._get_user_data_entry(self.user_id)
        for profile_name in target_profiles:
            is_borrowed = profile_name in user_data.get("borrowed_profiles", {})
            profile = user_data.get("borrowed_profiles" if is_borrowed else "profiles", {}).get(profile_name)
            if profile:
                profile["realistic_typing_enabled"] = self.toggle_choice
                updated_count += 1
        
        if updated_count > 0: self.cog._save_user_data_entry(self.user_id, user_data)
        
        status = "ENABLED" if self.toggle_choice else "DISABLED"
        await interaction.edit_original_response(content=f"Realistic typing has been set to **{status}** for {updated_count} profile(s).", view=None)

class BulkCriticView(BaseBulkProfileView):
    def __init__(self, cog: 'GeminiAgent', user_id: int):
        super().__init__(cog, user_id, include_borrowed=True)
        self.toggle_choice = None
        self._build_view()

    def _build_view(self):
        self.clear_items()
        
        toggle_options = [
            discord.SelectOption(label="Enable Critic", value="enable", default=(self.toggle_choice is True)),
            discord.SelectOption(label="Disable Critic", value="disable", default=(self.toggle_choice is False))
        ]
        toggle_select = ui.Select(placeholder="Choose an action...", options=toggle_options, row=0)
        toggle_select.callback = self.toggle_callback
        self.add_item(toggle_select)

        self._build_profile_select_ui(row=1)
        
        apply_btn = ui.Button(label="Apply Action", style=discord.ButtonStyle.green, row=3)
        apply_btn.callback = self.apply_action
        self.add_item(apply_btn)

    async def toggle_callback(self, interaction: discord.Interaction):
        self.toggle_choice = interaction.data['values'][0] == "enable"
        self._build_view()
        await interaction.response.edit_message(content=self._get_selection_feedback_message(), view=self)

    async def apply_action(self, interaction: discord.Interaction, button: ui.Button = None):
        await interaction.response.defer()
        target_profiles = list(self.selected_profiles)
        if self.toggle_choice is None or not target_profiles:
            await interaction.edit_original_response(content="Please select an action and at least one profile.", view=None); return

        updated_count = 0
        user_data = self.cog._get_user_data_entry(self.user_id)
        for profile_name in target_profiles:
            is_borrowed = profile_name in user_data.get("borrowed_profiles", {})
            profile = user_data.get("borrowed_profiles" if is_borrowed else "profiles", {}).get(profile_name)
            if profile:
                profile["critic_enabled"] = self.toggle_choice
                updated_count += 1
        
        if updated_count > 0: self.cog._save_user_data_entry(self.user_id, user_data)
        
        status = "ENABLED" if self.toggle_choice else "DISABLED"
        await interaction.edit_original_response(content=f"Critic has been set to **{status}** for {updated_count} profile(s).", view=None)

class AppearanceCreateModal(ui.Modal):
    def __init__(self, cog: 'GeminiAgent', original_interaction: discord.Interaction, profile_name: str):
        super().__init__(title=f"Create Appearance for '{profile_name}'")
        self.cog = cog
        self.original_interaction = original_interaction
        self.profile_name = profile_name
        self.display_name_input = ui.TextInput(label="Custom Display Name (Optional)", required=False, max_length=80)
        self.avatar_url_input = ui.TextInput(label="Avatar URL (Optional, direct link)", required=False)
        self.add_item(self.display_name_input)
        self.add_item(self.avatar_url_input)

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer()
        display_name = self.display_name_input.value or None
        avatar_url = self.avatar_url_input.value or None
        user_id_str = str(interaction.user.id)

        if display_name and display_name.lower() == 'clyde':
            await interaction.followup.send("The name 'clyde' is reserved and cannot be used as a display name.", ephemeral=True)
            return

        user_apps = self.cog.user_appearances.setdefault(user_id_str, {})
        if len(user_apps) >= MAX_USER_APPEARANCES:
            await interaction.followup.send(f"Max appearances ({MAX_USER_APPEARANCES}) reached.", ephemeral=True)
            return

        user_apps[self.profile_name] = {"custom_avatar_url": avatar_url, "custom_display_name": display_name}
        self.cog._save_user_appearance_shard(user_id_str, user_apps)

        # Check if this profile is linked to a child bot and send an update
        linked_bot_id = next((bot_id for bot_id, data in self.cog.child_bots.items() if str(data.get("owner_id")) == user_id_str and data.get("profile_name") == self.profile_name), None)
        if linked_bot_id:
            now = time.time()
            cooldown_window = 600  # 10 minutes
            max_changes = 2
            
            timestamps = self.cog.child_bot_edit_cooldowns.get(linked_bot_id, [])
            valid_timestamps = [ts for ts in timestamps if now - ts < cooldown_window]

            if len(valid_timestamps) >= max_changes:
                remaining = int(cooldown_window - (now - valid_timestamps[0]))
                await interaction.followup.send(f"This child bot's appearance has been changed too frequently. Please wait {remaining // 60} more minute(s) before trying again.", ephemeral=True)
            else:
                # Send avatar update
                await self.cog.manager_queue.put({
                    "action": "send_to_child",
                    "bot_id": linked_bot_id,
                    "payload": {"action": "update_avatar", "avatar_url": avatar_url}
                })
                # Send username update
                await self.cog.manager_queue.put({
                    "action": "send_to_child",
                    "bot_id": linked_bot_id,
                    "payload": {"action": "update_username", "username": display_name}
                })
                
                valid_timestamps.append(now)
                self.cog.child_bot_edit_cooldowns[linked_bot_id] = valid_timestamps
        
        # Refresh the main profile manage embed
        new_embed = await self.cog._build_profile_manage_embed(self.original_interaction, self.profile_name)
        await self.original_interaction.edit_original_response(embed=new_embed)

class AppearanceEditModal(ui.Modal):
    def __init__(self, cog: 'GeminiAgent', original_interaction: discord.Interaction, profile_name: str):
        super().__init__(title=f"Edit Appearance: '{profile_name}'")
        self.cog = cog
        self.original_interaction = original_interaction
        self.profile_name = profile_name
        
        user_id_str = str(original_interaction.user.id)
        current_data = self.cog.user_appearances.get(user_id_str, {}).get(self.profile_name, {})
        
        self.display_name_input = ui.TextInput(label="Custom Display Name (Optional)", required=False, max_length=80, default=current_data.get("custom_display_name"))
        self.avatar_url_input = ui.TextInput(label="Avatar URL (Optional)", required=False, default=current_data.get("custom_avatar_url"))
        self.add_item(self.display_name_input)
        self.add_item(self.avatar_url_input)

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer()
        new_display_name = self.display_name_input.value or None
        new_avatar_url = self.avatar_url_input.value or None
        user_id_str = str(interaction.user.id)

        if new_display_name and new_display_name.lower() == 'clyde':
            await interaction.followup.send("The name 'clyde' is reserved and cannot be used as a display name.", ephemeral=True)
            return

        is_public = self.cog._is_profile_public(interaction.user.id, self.profile_name)
        if is_public:
            is_safe, reason = await self.cog._is_profile_content_safe(self.profile_name, new_display_name or self.profile_name, new_avatar_url, interaction.guild_id)
            if not is_safe:
                await interaction.followup.send(f"**Could not save appearance.** The new display name or avatar was flagged for a content policy violation: {reason}\n\nIf you wish to proceed with this restricted edit, you must first unpublish the profile using `/profile public manage`.", ephemeral=True)
                return

        user_apps = self.cog.user_appearances.setdefault(user_id_str, {})
        user_apps[self.profile_name] = {"custom_avatar_url": new_avatar_url, "custom_display_name": new_display_name}
        self.cog._save_user_appearance_shard(user_id_str, user_apps)

        # Check if this profile is linked to a child bot and send an update
        linked_bot_id = next((bot_id for bot_id, data in self.cog.child_bots.items() if str(data.get("owner_id")) == user_id_str and data.get("profile_name") == self.profile_name), None)
        if linked_bot_id:
            now = time.time()
            cooldown_window = 600  # 10 minutes
            max_changes = 2
            
            timestamps = self.cog.child_bot_edit_cooldowns.get(linked_bot_id, [])
            valid_timestamps = [ts for ts in timestamps if now - ts < cooldown_window]

            if len(valid_timestamps) >= max_changes:
                remaining = int(cooldown_window - (now - valid_timestamps[0]))
                await interaction.followup.send(f"This child bot's appearance has been changed too frequently. Please wait {remaining // 60} more minute(s) before trying again.", ephemeral=True)
            else:
                # Send avatar update
                await self.cog.manager_queue.put({
                    "action": "send_to_child",
                    "bot_id": linked_bot_id,
                    "payload": {"action": "update_avatar", "avatar_url": new_avatar_url}
                })
                # Send username update
                await self.cog.manager_queue.put({
                    "action": "send_to_child",
                    "bot_id": linked_bot_id,
                    "payload": {"action": "update_username", "username": new_display_name}
                })
                
                valid_timestamps.append(now)
                self.cog.child_bot_edit_cooldowns[linked_bot_id] = valid_timestamps

        # Refresh the main profile manage embed
        new_embed = await self.cog._build_profile_manage_embed(self.original_interaction, self.profile_name)
        await self.original_interaction.edit_original_response(embed=new_embed)

class AppearanceEditView(ui.View):
    def __init__(self, cog: 'GeminiAgent', original_interaction: discord.Interaction, profile_name: str):
        super().__init__(timeout=300)
        self.cog = cog
        self.original_interaction = original_interaction
        self.profile_name = profile_name
        self.user_id = original_interaction.user.id

        edit_button = ui.Button(label="Edit Details", style=discord.ButtonStyle.primary)
        edit_button.callback = self.edit_callback
        self.add_item(edit_button)

        delete_button = ui.Button(label="Delete Appearance", style=discord.ButtonStyle.danger)
        delete_button.callback = self.delete_callback
        self.add_item(delete_button)

    async def show(self, interaction: discord.Interaction):
        user_id_str = str(self.user_id)
        appearance_data = self.cog.user_appearances.get(user_id_str, {}).get(self.profile_name, {})
        
        embed = discord.Embed(title=f"Managing Appearance for '{self.profile_name}'", color=discord.Color.teal())
        disp_name = appearance_data.get("custom_display_name") or "(Uses Profile Name)"
        avatar_url = appearance_data.get("custom_avatar_url") or "(Uses Default Avatar)"
        embed.add_field(name="Display Name", value=disp_name, inline=False)
        embed.add_field(name="Avatar URL", value=avatar_url, inline=False)
        if appearance_data.get("custom_avatar_url"):
            embed.set_thumbnail(url=appearance_data.get("custom_avatar_url"))
        
        await interaction.response.send_message(embed=embed, view=self, ephemeral=True)

    async def edit_callback(self, interaction: discord.Interaction):
        modal = AppearanceEditModal(self.cog, self.original_interaction, self.profile_name)
        await interaction.response.send_modal(modal)
        await interaction.delete_original_response() # Close the edit view after opening modal

    async def delete_callback(self, interaction: discord.Interaction):
        confirm_view = ui.View(timeout=60)
        
        async def confirm_delete(i: discord.Interaction):
            await i.response.defer()
            user_id_str = str(self.user_id)
            if user_id_str in self.cog.user_appearances and self.profile_name in self.cog.user_appearances[user_id_str]:
                user_apps = self.cog.user_appearances[user_id_str]
                del user_apps[self.profile_name]
                self.cog._save_user_appearance_shard(user_id_str, user_apps)
            
            # Refresh the main profile manage embed
            new_embed = await self.cog._build_profile_manage_embed(self.original_interaction, self.profile_name)
            await self.original_interaction.edit_original_response(embed=new_embed)
            await interaction.delete_original_response()

        confirm_button = ui.Button(label="Confirm Deletion", style=discord.ButtonStyle.danger)
        confirm_button.callback = confirm_delete
        confirm_view.add_item(confirm_button)
        await interaction.response.send_message(f"**Are you sure you want to delete the appearance for '{self.profile_name}'?**", view=confirm_view, ephemeral=True)

class BulkManageView(ui.View):
    def __init__(self, cog: 'GeminiAgent', original_interaction: discord.Interaction):
        super().__init__(timeout=600)
        self.cog = cog
        self.original_interaction = original_interaction
        self.user_id = original_interaction.user.id

        options = [
            discord.SelectOption(label="Set Models", value="models", description="Apply model settings to multiple profiles."),
            discord.SelectOption(label="Set Generation Parameters & STM", value="gen_params", description="Apply Temp, Top P, Top K, and STM Length."),
            discord.SelectOption(label="Set Advanced Parameters (OpenRouter)", value="adv_params", description="Apply penalties, Min P, and Top A."),
            discord.SelectOption(label="Set Thinking Parameters", value="thinking_params", description="Apply thinking settings to multiple profiles."),
            discord.SelectOption(label="Set Response Mode", value="response_mode", description="Apply Mention/Reply behavior to multiple profiles."),
            discord.SelectOption(label="Toggle Grounding", value="grounding", description="Enable or disable grounding for multiple profiles."),
            discord.SelectOption(label="Toggle Image Generation", value="image_gen", description="Enable or disable image generation for multiple profiles."),
            discord.SelectOption(label="Toggle URL Context Fetching", value="url_context", description="Enable or disable link scraping for multiple profiles."),
            discord.SelectOption(label="Set Time & Timezone", value="timezone", description="Enable time-awareness and set a specific timezone."),
            discord.SelectOption(label="Toggle Critic (Anti-Repetition)", value="critic", description="Enable or disable the critic for multiple profiles."),
            discord.SelectOption(label="Toggle Realistic Typing", value="typing", description="Enable or disable realistic typing for multiple profiles."),
            discord.SelectOption(label="Set Safety Level", value="safety_level", description="Apply a content safety level to multiple profiles."),
            discord.SelectOption(label="Set LTM Scope", value="ltm_scope", description="Apply an LTM scope to multiple profiles."),
            discord.SelectOption(label="Set Training Parameters", value="train_params", description="Set training settings to multiple personal profiles."),
            discord.SelectOption(label="Set LTM Parameters", value="ltm_params", description="Apply LTM settings to multiple personal profiles."),
            discord.SelectOption(label="Set LTM Summarization Prompt", value="ltm_summarization", description="Apply a custom LTM summarization prompt."),
            discord.SelectOption(label="Reset Profile Data", value="data_reset", description="Reset LTM or Training Examples for personal profiles."),
            discord.SelectOption(label="Delete Profiles", value="delete_items", description="Permanently delete multiple profiles.")
        ]
        
        select = ui.Select(placeholder="Choose a bulk action to perform...", options=options)
        select.callback = self.select_callback
        self.add_item(select)
    
    async def select_callback(self, interaction: discord.Interaction):
        choice = interaction.data['values'][0]
        user_data = self.cog._get_user_data_entry(self.user_id)
        all_profiles = list(user_data.get("profiles", {}).keys()) + list(user_data.get("borrowed_profiles", {}).keys())
        
        if not all_profiles:
            await self.original_interaction.edit_original_response(content="You have no profiles to apply settings to.", view=None)
            return

        if choice == "gen_params":
            modal = ProfileParamsModal(self.cog, "BULK_APPLY", {}, is_borrowed=False)
            async def modal_callback(i: discord.Interaction):
                await i.response.defer(ephemeral=True)
                params = {}
                try:
                    stm_str = next((c.value for c in modal.children if c.custom_id == "stm_length"), None)
                    if stm_str: params["stm_length"] = int(stm_str)
                    
                    temp_str = next((c.value for c in modal.children if c.custom_id == "temperature"), None)
                    if temp_str: params["temperature"] = float(temp_str)
                    
                    topp_str = next((c.value for c in modal.children if c.custom_id == "top_p"), None)
                    if topp_str: params["top_p"] = float(topp_str)

                    topk_str = next((c.value for c in modal.children if c.custom_id == "top_k"), None)
                    if topk_str: params["top_k"] = int(topk_str)
                except ValueError:
                    await i.followup.send(f"❌ **Invalid Input**", ephemeral=True); return

                view = BulkActionView(self.cog, self.user_id, "apply_params", "Select profiles to apply parameters to...", params=params, include_borrowed=True)
                await self.original_interaction.edit_original_response(content="Parameters validated. Select the profiles to apply them to:", view=view)
            modal.on_submit = modal_callback
            await interaction.response.send_modal(modal)

        elif choice == "adv_params":
            modal = ProfileAdvancedParamsModal(self.cog, "BULK_APPLY", {}, is_borrowed=False)
            async def modal_callback(i: discord.Interaction):
                await i.response.defer(ephemeral=True)
                params = {}
                try:
                    def pf(val): return float(val) if val and val.strip() else None
                    params["frequency_penalty"] = pf(next((c.value for c in modal.children if c.custom_id == "frequency_penalty"), None))
                    params["presence_penalty"] = pf(next((c.value for c in modal.children if c.custom_id == "presence_penalty"), None))
                    params["repetition_penalty"] = pf(next((c.value for c in modal.children if c.custom_id == "repetition_penalty"), None))
                    params["min_p"] = pf(next((c.value for c in modal.children if c.custom_id == "min_p"), None))
                    params["top_a"] = pf(next((c.value for c in modal.children if c.custom_id == "top_a"), None))
                except ValueError:
                    await i.followup.send(f"❌ **Invalid Input**", ephemeral=True); return

                clean = {k:v for k,v in params.items() if v is not None}
                if not clean: await i.followup.send("No parameters set.", ephemeral=True); return

                view = BulkActionView(self.cog, self.user_id, "apply_params", "Select profiles to apply parameters to...", params=clean, include_borrowed=True)
                await self.original_interaction.edit_original_response(content="Advanced parameters validated. Select the profiles to apply them to:", view=view)
            modal.on_submit = modal_callback
            await interaction.response.send_modal(modal)

        elif choice == "thinking_params":
            modal = ProfileThinkingParamsModal(self.cog, "BULK_APPLY", {}, False)
            async def modal_callback(i: discord.Interaction):
                await i.response.defer(ephemeral=True)
                params = {}
                try:
                    def gv(cid): return next(c.value for c in modal.children if c.custom_id == cid).strip().lower()
                    params["thinking_summary_visible"] = gv("thinking_summary_visible") if gv("thinking_summary_visible") in ["on", "off"] else "off"
                    # [UPDATED] Validating against the 6 standardized effort levels in bulk mode
                    lvl = gv("thinking_level")
                    if lvl not in ["xhigh", "high", "medium", "low", "minimal", "none"]:
                        lvl = "high"
                    params["thinking_level"] = lvl
                    params["thinking_budget"] = int(gv("thinking_budget"))
                except ValueError:
                    await i.followup.send("❌ **Invalid Input**", ephemeral=True); return

                view = BulkActionView(self.cog, self.user_id, "apply_thinking_params", "Select profiles to apply thinking settings to...", params=params, include_borrowed=True)
                await self.original_interaction.edit_original_response(content="Thinking parameters validated. Select the profiles to apply them to:", view=view)
            modal.on_submit = modal_callback
            await interaction.response.send_modal(modal)

        elif choice == "train_params":
            modal = ProfileTrainingParamsModal(self.cog, "BULK_APPLY", {})
            async def modal_callback(i: discord.Interaction):
                await i.response.defer(ephemeral=True)
                params = {}
                try:
                    ctx = next((c.value for c in modal.children if c.custom_id == "training_context_size"), None)
                    if ctx: params["training_context_size"] = int(ctx)
                    rel = next((c.value for c in modal.children if c.custom_id == "training_relevance_threshold"), None)
                    if rel: params["training_relevance_threshold"] = float(rel)
                except ValueError: await i.followup.send("Invalid Input", ephemeral=True); return

                view = BulkActionView(self.cog, self.user_id, "apply_training_params", "Select personal profiles to apply settings to...", params=params, include_borrowed=False)
                await self.original_interaction.edit_original_response(content="Parameters validated. Select the profiles to apply them to:", view=view)
            modal.on_submit = modal_callback
            await interaction.response.send_modal(modal)

        elif choice == "ltm_params":
            modal = ProfileLTMParamsModal(self.cog, "BULK_APPLY", {})
            async def modal_callback(i: discord.Interaction):
                await i.response.defer(ephemeral=True)
                params = {}
                try:
                    ctx = next((c.value for c in modal.children if c.custom_id == "ltm_context_size"), None)
                    if ctx: params["ltm_context_size"] = int(ctx)
                    rel = next((c.value for c in modal.children if c.custom_id == "ltm_relevance_threshold"), None)
                    if rel: params["ltm_relevance_threshold"] = float(rel)
                except ValueError: await i.followup.send("Invalid Input", ephemeral=True); return

                view = BulkActionView(self.cog, self.user_id, "apply_ltm_params", "Select personal profiles to apply LTM settings to...", params=params, include_borrowed=False)
                await self.original_interaction.edit_original_response(content="Parameters validated. Select the profiles to apply them to:", view=view)
            modal.on_submit = modal_callback
            await interaction.response.send_modal(modal)

        elif choice == "ltm_summarization":
            modal = ProfileLTMSummarizationModal(self.cog, "BULK_APPLY", DEFAULT_LTM_SUMMARIZATION_INSTRUCTIONS)
            async def modal_callback(i: discord.Interaction):
                await i.response.defer()
                instructions = modal.instructions_input.value.strip() or DEFAULT_LTM_SUMMARIZATION_INSTRUCTIONS
                
                params = {"ltm_summarization_instructions": instructions}

                view = BulkActionView(self.cog, self.user_id, "apply_ltm_summarization", "Select personal profiles to apply LTM prompt to...", params=params, include_borrowed=False)
                await self.original_interaction.edit_original_response(content="Prompt received. Now select the profiles to apply it to:", view=view)
            modal.on_submit = modal_callback
            await interaction.response.send_modal(modal)

        elif choice == "models":
            await interaction.response.defer()
            view = ModelApplyView(self.cog, self.user_id, self.original_interaction)
            await self.original_interaction.edit_original_response(content="Select models and profiles:", view=view)

        elif choice == "grounding":
            await interaction.response.defer()
            view = BulkGroundingView(self.cog, self.user_id)
            await self.original_interaction.edit_original_response(content="Select grounding mode and profiles:", view=view)

        elif choice == "response_mode":
            await interaction.response.defer()
            view = BulkResponseModeView(self.cog, self.user_id)
            await self.original_interaction.edit_original_response(content="Select a response mode and profiles:", view=view)

        elif choice == "url_context":
            await interaction.response.defer()
            view = BulkURLContextView(self.cog, self.user_id)
            await self.original_interaction.edit_original_response(content="Select URL context action and profiles:", view=view)

        elif choice == "image_gen":
            await interaction.response.defer()
            view = BulkImageGenView(self.cog, self.user_id)
            await self.original_interaction.edit_original_response(content="Select image generation action and profiles:", view=view)
            
        elif choice == "timezone":
            await interaction.response.defer()
            view = BulkTimezoneView(self.cog, self.user_id)
            await self.original_interaction.edit_original_response(content="Select a timezone and the profiles to apply it to:", view=view)
        
        elif choice == "critic":
            await interaction.response.defer()
            view = BulkCriticView(self.cog, self.user_id)
            await self.original_interaction.edit_original_response(content="Select critic action and profiles:", view=view)

        elif choice == "typing":
            await interaction.response.defer()
            view = TypingManageView(self.cog, self.user_id)
            await self.original_interaction.edit_original_response(content="Select typing action and profiles:", view=view)

        elif choice == "data_reset":
            await interaction.response.defer()
            view = BulkResetView(self.cog, self.user_id)
            await self.original_interaction.edit_original_response(content="Select reset action:", view=view)

        elif choice == "delete_items":
            await interaction.response.defer()
            view = BulkDeleteView(self.cog, self.user_id)
            await self.original_interaction.edit_original_response(content="Select profiles to delete:", view=view)

        elif choice == "ltm_scope":
            await interaction.response.defer()
            view = BulkLtmScopeView(self.cog, self.user_id)
            await self.original_interaction.edit_original_response(content="Select scope and profiles:", view=view)

        elif choice == "safety_level":
            await interaction.response.defer()
            view = BulkSafetyLevelView(self.cog, self.user_id)
            await self.original_interaction.edit_original_response(content=view._get_selection_feedback_message(), view=view)

class ChildBotCreateModal(ui.Modal, title="Create a New Child Bot"):
    def __init__(self, cog: 'GeminiAgent', view: 'SettingsChildBotView'):
        super().__init__()
        self.cog = cog
        self.parent_view = view
        self.profile_name_input = ui.TextInput(label="Profile Name to Link", placeholder="The personal profile this bot will embody.", required=True)
        self.token_input = ui.TextInput(label="Bot Token", placeholder="Paste the token from the Discord Developer Portal.", style=discord.TextStyle.paragraph, required=True)
        self.add_item(self.profile_name_input)
        self.add_item(self.token_input)

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        token = self.token_input.value.strip()
        profile_name = self.profile_name_input.value.lower().strip()
        owner_id = interaction.user.id

        user_data = self.cog._get_user_data_entry(owner_id)
        if profile_name not in user_data.get("profiles", {}):
            await interaction.followup.send(f"❌ **Error:** You do not have a personal profile named '{profile_name}'.", ephemeral=True)
            return

        temp_client = discord.Client(intents=discord.Intents.none())
        try:
            await temp_client.login(token)
            bot_user_id = str(temp_client.user.id)
            await temp_client.close()
        except discord.LoginFailure:
            await interaction.followup.send("❌ **Error:** The provided token is invalid. Please double-check it.", ephemeral=True)
            return
        except Exception as e:
            await temp_client.close()
            await interaction.followup.send(f"❌ **Error:** An unexpected error occurred while validating the token: {e}", ephemeral=True)
            return

        if bot_user_id in self.cog.child_bots:
            await interaction.followup.send("❌ **Error:** This bot application is already registered as a child bot.", ephemeral=True)
            return

        encrypted_token = self.cog._encrypt_data(token)
        
        user_shard = self.cog._get_user_child_bot_shard(owner_id)
        user_shard[bot_user_id] = {
            "token_encrypted": encrypted_token,
            "profile_name": profile_name,
            "approved_servers": []
        }
        self.cog._save_user_child_bot_shard(owner_id, user_shard)
        self.cog._load_child_bots() # Reload all bots into memory
        
        new_bot_config = self.cog.child_bots.get(bot_user_id)
        if new_bot_config:
            await self.cog.manager_queue.put({
                "action": "launch_bot",
                "bot_id": bot_user_id,
                "token": token,
                "config": new_bot_config
            })
        
        await interaction.followup.send(f"✅ **Success!** Child bot '{temp_client.user.name}' has been linked to your profile '{profile_name}'.", ephemeral=True)
        # Call the update alias which points to setup_items + update_display
        await self.parent_view.update_rebuild(interaction)

class SessionView(ui.View):
    def __init__(self, cog: 'GeminiAgent', interaction: discord.Interaction, session: Dict):
        super().__init__(timeout=600)
        self.cog = cog
        self.session = session
        self.channel_id = interaction.channel_id
        
        options = []
        for i, p in enumerate(session.get("profiles", [])):
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
            
        if not options:
            self.stop()
            return

        self.select = ui.Select(placeholder="Select a participant to view details...", options=options[:25])
        self.select.callback = self.callback
        self.add_item(self.select)

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        idx = int(self.select.values[0])
        participant = self.session["profiles"][idx]
        
        embed = await self.cog._build_profile_embed(participant['owner_id'], participant['profile_name'], self.channel_id)
        await interaction.followup.send(embed=embed, ephemeral=True)

class BulkExportView(BaseBulkProfileView):
    def __init__(self, cog: 'GeminiAgent', user_id: int):
        super().__init__(cog, user_id, include_borrowed=False)
        self.export_filters = {"persona", "instructions", "ltm", "training", "appearance"}
        self._build_view()

    def _build_view(self):
        self.clear_items()
        self._build_profile_select_ui(row=0)
        
        filter_options = [
            discord.SelectOption(label="Persona", value="persona", description="Backstory, traits, likes/dislikes.", default="persona" in self.export_filters),
            discord.SelectOption(label="AI Instructions", value="instructions", description="Core behavioral guidelines.", default="instructions" in self.export_filters),
            discord.SelectOption(label="Long-Term Memories", value="ltm", description="AI-generated conversation summaries.", default="ltm" in self.export_filters),
            discord.SelectOption(label="Training Examples", value="training", description="User-written input/output pairs.", default="training" in self.export_filters),
            discord.SelectOption(label="Appearance", value="appearance", description="Custom name and avatar URL.", default="appearance" in self.export_filters)
        ]
        
        filter_select = ui.Select(
            placeholder="Select components to include...",
            min_values=1,
            max_values=len(filter_options),
            options=filter_options,
            row=2
        )
        filter_select.callback = self.filter_callback
        self.add_item(filter_select)

        export_btn = ui.Button(label="Export Selected to Plaintext", style=discord.ButtonStyle.green, row=3)
        export_btn.callback = self.export_callback
        self.add_item(export_btn)

    async def filter_callback(self, interaction: discord.Interaction):
        self.export_filters = set(interaction.data['values'])
        await interaction.response.defer()

    async def export_callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        if not self.selected_profiles:
            await interaction.followup.send("Select at least one profile to export.", ephemeral=True)
            return
        
        await self.cog._execute_export(interaction, list(self.selected_profiles), self.export_filters)