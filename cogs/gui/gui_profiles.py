from ..utils.constants import *

import discord
from discord import ui
import asyncio
import datetime
import traceback
import time
from zoneinfo import ZoneInfo
from typing import TYPE_CHECKING, List, Dict, Set, Any, Optional
from ..utils.content import OLLAMA_GUIDE_TEXT
from ..utils.helpers import _pf, _pi, _ps, _pb

if TYPE_CHECKING:
    # This only runs during "hinting" and prevents the circular crash
    from ..MimicCog import MimicCog

from .base_components import BaseBulkProfileView, ConfigModal, ActionTextInputModal, build_pagination_controls, build_confirm_view
from .gui_data import DataManageView
from .gui_hub import HubShareManagerView
from .gui_sessions import CustomModelModal
from .gui_settings import OllamaHostModal

def ProfileAdvancedParamsModal(cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, callback=None):
    def gv(k):
        v = current_params.get(k)
        return str(v) if v is not None else ""
    fields = [
        {"label": "Frequency Penalty (-2.0 to 2.0)", "custom_id": "frequency_penalty", "default": gv("frequency_penalty"), "required": False, "placeholder": "Default: 0.0 (Disabled)"},
        {"label": "Presence Penalty (-2.0 to 2.0)", "custom_id": "presence_penalty", "default": gv("presence_penalty"), "required": False, "placeholder": "Default: 0.0 (Disabled)"},
        {"label": "Repetition Penalty (0.0 to 2.0)", "custom_id": "repetition_penalty", "default": gv("repetition_penalty"), "required": False, "placeholder": "Default: 0.0 (Disabled)"},
        {"label": "Min P (0.0 to 1.0)", "custom_id": "min_p", "default": gv("min_p"), "required": False, "placeholder": "Default: 0.0 (Disabled)"},
        {"label": "Top A (0.0 to 1.0)", "custom_id": "top_a", "default": gv("top_a"), "required": False, "placeholder": "Default: 0.0 (Disabled)"}
    ]
    def parser(v):
        c = {}
        for k, (min_v, max_v) in [("frequency_penalty", (-2.0, 2.0)), ("presence_penalty", (-2.0, 2.0)), ("repetition_penalty", (0.0, 2.0)), ("min_p", (0.0, 1.0)), ("top_a", (0.0, 1.0))]:
            val = _pf(v[k])
            if val is not None:
                if not (min_v <= val <= max_v): raise ValueError(f"{k} out of range")
                c[k] = val
            else:
                c[k] = None
        return {"config": c}
    return ConfigModal(cog, profile_name, is_borrowed, "Advanced Parameters (OpenRouter)", fields, parser, callback)

def ProfileDirectorDeskModal(cog, profile_name: str, current_params: Dict[str, Any], callback=None, target_user_id: Optional[int] = None):
    fields = [
        {"label": "Archetype (Who)", "custom_id": "speech_archetype", "default": str(current_params.get("speech_archetype", "")), "required": False, "max_length": 200, "placeholder": "e.g. A cynical noir detective, a bubbly influencer."},
        {"label": "Accent", "custom_id": "speech_accent", "default": str(current_params.get("speech_accent", "")), "required": False, "max_length": 200, "placeholder": "e.g. Australian (Melbourne), British (Brixton)."},
        {"label": "Dynamics (Where / Acoustics)", "custom_id": "speech_dynamics", "default": str(current_params.get("speech_dynamics", "")), "required": False, "max_length": 200, "placeholder": "e.g. Speaking in a whisper, cavernous echoing hall."},
        {"label": "Vocal Style (How)", "custom_id": "speech_style", "default": str(current_params.get("speech_style", "")), "required": False, "max_length": 200, "placeholder": "e.g. A vocal smile, breathy, gritty and gravelly."},
        {"label": "Pacing (Tempo)", "custom_id": "speech_pacing", "default": str(current_params.get("speech_pacing", "")), "required": False, "max_length": 200, "placeholder": "e.g. Rapid-fire delivery, slow deliberate drawl."}
    ]
    def parser(v):
        return {"config": {k: _ps(v[k]) or "" for k in ["speech_archetype", "speech_accent", "speech_dynamics", "speech_style", "speech_pacing"]}}
    return ConfigModal(cog, profile_name, False, "Director's Desk: TTS Instructions", fields, parser, callback, target_user_id)

def ProfileSpeechSettingsModal(cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, callback=None):
    fields = [
        {"label": "Enable TTS (on/off)", "custom_id": "speech_tts_enabled", "default": "on" if current_params.get("speech_tts_enabled", False) else "off", "required": True, "max_length": 10},
        {"label": "Voice Name", "custom_id": "speech_voice", "default": str(current_params.get("speech_voice", "Aoede")), "required": False, "max_length": 40},
        {"label": "Temperature (0.0 - 2.0)", "custom_id": "speech_temperature", "default": str(current_params.get("speech_temperature", 1.0)), "required": False, "max_length": 5}
    ]
    def parser(v):
        c = {}
        c["speech_tts_enabled"] = _pb(v["speech_tts_enabled"])
        c["speech_voice"] = _ps(v["speech_voice"]) or "Aoede"
        t = _pf(v["speech_temperature"])
        if t is not None:
            if not (0.0 <= t <= 2.0): raise ValueError("Temperature out of range")
            c["speech_temperature"] = t
        return {"config": c}
    return ConfigModal(cog, profile_name, is_borrowed, "Speech & Voice Settings", fields, parser, callback)

class ProfileManageView(ui.View):
    def __init__(self, cog: 'MimicCog', original_interaction: discord.Interaction, profile_name: str, is_borrowed: bool, target_user_id: Optional[int] = None, is_mod_view: bool = False):
        super().__init__(timeout=600)
        self.cog = cog
        self.original_interaction = original_interaction
        
        owner_id = int(defaultConfig.DISCORD_OWNER_ID)
        owner_idx = self.cog.profile_manager._get_user_index(owner_id)
        if profile_name in owner_idx.get("system", {}):
            self.target_user_id = owner_id
            self.user_id = owner_id
            self.is_system = True
        else:
            self.target_user_id = target_user_id or original_interaction.user.id
            self.user_id = self.target_user_id
            self.is_system = False

        self.profile_name = profile_name
        self.is_borrowed = is_borrowed
        self.is_mod_view = is_mod_view
        self.current_tab = "home"
        self.is_read_only = self.is_system and original_interaction.user.id != owner_id
        
        self._build_view()

    def _build_view(self):
        # Deferred import to avoid a circular import with gui_mod (which imports ProfileManageView from this module)
        from .gui_mod import ModBaseView

        self.clear_items()
        if self.is_read_only:
            return

        is_mod = getattr(self, 'is_mod_view', False)

        def tab_has_options(tab: str) -> bool:
            if tab == "home":
                return True
            elif tab == "persona":
                return not self.is_borrowed
            elif tab in ["params", "tools", "memory"]:
                return not is_mod
            return False

        valid_tabs = [t for t in ["home", "persona", "params", "tools", "memory"] if tab_has_options(t)]
        if self.current_tab not in valid_tabs and valid_tabs:
            self.current_tab = valid_tabs[0]
            
        if not valid_tabs:
            if is_mod: ModBaseView.add_nav_to_other_view(self, self.cog, self.original_interaction, "profiles")
            return

        # --- 1. Category Dropdown (Row 0) ---
        options = []
        
        if self.current_tab == "home":
            if not is_mod:
                options.append(discord.SelectOption(label="Rename Profile", value="rename", description="Change the local name of this profile."))
                
                if not self.is_borrowed:
                    options.append(discord.SelectOption(label="Duplicate Profile", value="duplicate", description="Create a new profile from a copy of this one."))
                    options.append(discord.SelectOption(label="Share Profile", value="share", description="Share this profile with others or publish it."))
                    options.append(discord.SelectOption(label="Custom Error Message", value="error_response", description="Set the message shown when generation fails."))
                    options.append(discord.SelectOption(label="Generation Visual", value="generation_visual", description="Set custom placeholder emoji and child bot behavior."))
                    
                    owner_id = int(defaultConfig.DISCORD_OWNER_ID)
                    if self.original_interaction.user.id == owner_id:
                        if self.is_system:
                            options.append(discord.SelectOption(label="Copy to Personal Profile", value="convert_to_personal", description="Create a Personal Profile copy from this System Profile."))
                        else:
                            options.append(discord.SelectOption(label="Copy to System Profile", value="convert_to_system", description="Create a global System Profile copy from this profile."))

                options.append(discord.SelectOption(label="Cycle Content Safety Level", value="safety_level", description="Cycle: Low -> Medium -> High -> Unrestricted 18+."))
            
            label = "Remove Borrowed Profile" if self.is_borrowed else "Delete Profile"
            options.append(discord.SelectOption(label=label, value="delete", description="Permanently remove this profile and its data."))

        elif self.current_tab == "persona":
            options.append(discord.SelectOption(label="Edit Persona", value="edit_persona", description="Edit backstory, traits, likes, dislikes, and appearance."))
            options.append(discord.SelectOption(label="Edit Instructions", value="edit_instructions", description="Edit specific AI behavioral instructions."))
            options.append(discord.SelectOption(label="TTS Instructions", value="tts_instructions", description="Configure the 'Director's Desk' for vocal performance."))
            if not is_mod and not self.is_borrowed:
                options.append(discord.SelectOption(label="Edit Appearance", value="edit_appearance", description="Edit the custom Webhook name and avatar."))

        elif self.current_tab == "params" and not is_mod:
            options.append(discord.SelectOption(label="Set Models", value="models", description="Choose Primary and Fallback AI models."))
            options.append(discord.SelectOption(label="Set Generation Parameters & STM", value="gen_params", description="Set Temp, Top P, Top K, and STM Length."))
            options.append(discord.SelectOption(label="Set Advanced Parameters (OPENROUTER)", value="adv_params", description="Set penalties, Min P, and Top A."))
            options.append(discord.SelectOption(label="Set Thinking Parameters", value="thinking_params", description="Set thinking persistence, level, and budget."))
            options.append(discord.SelectOption(label="Set Speech & Voice Settings", value="speech_settings", description="Set TTS voice, model, and temperature."))

        elif self.current_tab == "tools" and not is_mod:
            options.append(discord.SelectOption(label="Toggle Image Generation", value="image_toggle", description="Allow this profile to generate images via !image/!imagine."))
            options.append(discord.SelectOption(label="Toggle Grounding (Web Search)", value="grounding", description="Cycle Grounding: OFF -> NATIVE -> RAG."))
            options.append(discord.SelectOption(label="Toggle URL Context Fetching", value="url_toggle", description="Cycle URL Context: OFF -> NATIVE -> RAG."))
            options.append(discord.SelectOption(label="Cycle Response Mode", value="cycle_response", description="Cycle: Regular -> Mention -> Reply -> Mention Reply."))
            options.append(discord.SelectOption(label="Set Time & Timezone", value="time", description="Enable time awareness and set the profile's timezone."))
            options.append(discord.SelectOption(label="Toggle Realistic Typing", value="typing", description="Enable a human-like delay when the bot sends messages."))
            options.append(discord.SelectOption(label="Toggle Anti-Repetition Critic", value="critic", description="Enable semantic repetition analysis (Adds latency)."))
            options.append(discord.SelectOption(label="Toggle Neuro-Endocrine Engine", value="neuro", description="Simulate hormonal states for dynamic emotions."))
            options.append(discord.SelectOption(label="Toggle Help Mode (Guide RAG)", value="help_mode", description="Allow profile to answer technical bot questions."))

        elif self.current_tab == "memory" and not is_mod:
            options.append(discord.SelectOption(label="Manage Long-Term Memories", value="manage_ltm", description="Add, list, edit, or delete memories."))
            if not self.is_borrowed:
                options.append(discord.SelectOption(label="Manage Training Examples", value="manage_training", description="Add, list, edit, or delete training examples."))
                options.append(discord.SelectOption(label="Set Training Parameters", value="train_params", description="Set training context size and relevance threshold."))
            options.append(discord.SelectOption(label="Toggle LTM Auto-Creation", value="ltm_creation", description="Automatically create memories from conversations."))
            options.append(discord.SelectOption(label="Set LTM Parameters", value="ltm_params", description="Set frequency, context, and recall settings."))
            if not self.is_borrowed:
                options.append(discord.SelectOption(label="Set LTM Summarization Prompt", value="ltm_summarization", description="Customize how the AI creates memories."))

        if options:
            select = ui.Select(placeholder=f"Select an action for {self.current_tab.title()}...", options=options, row=0)
            select.callback = self.dropdown_callback
            self.add_item(select)

        # --- 2. Navigation Buttons (Row 1) ---
        for tab in valid_tabs:
            btn = ui.Button(
                label=tab.title(), 
                style=discord.ButtonStyle.primary if self.current_tab == tab else discord.ButtonStyle.secondary, 
                row=1, 
                disabled=(self.current_tab == tab)
            )
            btn.callback = self.create_nav_callback(tab)
            self.add_item(btn)

        if is_mod:
            ModBaseView.add_nav_to_other_view(self, self.cog, self.original_interaction, "profiles")

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
        profile = self.cog.profile_manager._get_profile_config(user_id, profile_name, self.is_borrowed)
        
        if not profile:
            await interaction.response.send_message("Profile data not found.", ephemeral=True); return

        # --- Home Tab Logic ---
        if choice == "rename":
            await self._handle_rename(interaction)
        elif choice == "duplicate":
            await self._handle_duplicate(interaction)
        elif choice == "share":
            await self._handle_share(interaction)
        elif choice == "convert_to_system":
            await self._handle_convert_copy(interaction, to_system=True)
        elif choice == "convert_to_personal":
            await self._handle_convert_copy(interaction, to_system=False)
        elif choice == "delete":
            await self._handle_delete(interaction)
        elif choice == "safety_level":
            await self._handle_safety_cycle(interaction, profile)
        elif choice == "error_response":
            is_b = getattr(self, "is_borrowed", False)
            target_profile = self.cog.profile_manager._get_profile_config(interaction.user.id, self.profile_name, is_b)

            if not target_profile:
                await interaction.response.send_message("❌ Error: Profile not found.", ephemeral=True)
                return

            async def modal_callback(modal_interaction: discord.Interaction, new_val: str):
                await modal_interaction.response.defer(ephemeral=True)
                val_to_save = new_val.strip() or "An error has occurred."
                
                target = self.cog.profile_manager._get_profile_config(modal_interaction.user.id, self.profile_name, is_b)
                
                if target:
                    target["error_response"] = val_to_save
                    self.cog.profile_manager._save_profile_config(modal_interaction.user.id, self.profile_name, target, is_b)
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

        elif choice == "generation_visual":
            async def refresh_cb(modal_interaction: discord.Interaction):
                new_embed = await self.cog.profile_manager._build_profile_manage_embed(modal_interaction, self.profile_name)
                await self.original_interaction.edit_original_response(embed=new_embed, view=self)
            modal = ProfileGenerationVisualModal(self.cog, self.profile_name, profile, self.is_borrowed, callback=refresh_cb)
            await interaction.response.send_modal(modal)

        # --- Persona Tab Logic ---
        elif choice == "edit_persona":
            prompts = self.cog.profile_manager._get_profile_prompts(user_id, profile_name) or {}
            modal = EditUserProfilePersonaModal(self.cog, profile_name, prompts.get("persona", {}), user_id)
            await interaction.response.send_modal(modal)
        elif choice == "edit_instructions":
            prompts = self.cog.profile_manager._get_profile_prompts(user_id, profile_name) or {}
            modal = EditUserProfileAIInstructionsModal(self.cog, profile_name, prompts.get("ai_instructions", ""), user_id)
            await interaction.response.send_modal(modal)
        elif choice == "tts_instructions":
            async def refresh_cb(modal_interaction: discord.Interaction):
                new_embed = await self.cog.profile_manager._build_profile_manage_embed(modal_interaction, profile_name, target_user_id=self.user_id)
                await self.original_interaction.edit_original_response(embed=new_embed, view=self)
            modal = ProfileDirectorDeskModal(self.cog, profile_name, profile, callback=refresh_cb, target_user_id=self.user_id)
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
                new_embed = await self.cog.profile_manager._build_profile_manage_embed(modal_interaction, profile_name)
                # Edit the MAIN message (the dashboard)
                await self.original_interaction.edit_original_response(embed=new_embed, view=self)
                
            modal = ProfileParamsModal(self.cog, profile_name, profile, self.is_borrowed, callback=refresh_cb)
            await interaction.response.send_modal(modal)
        elif choice == "adv_params":
            async def refresh_cb(modal_interaction: discord.Interaction):
                new_embed = await self.cog.profile_manager._build_profile_manage_embed(modal_interaction, profile_name)
                await self.original_interaction.edit_original_response(embed=new_embed, view=self)
                
            modal = ProfileAdvancedParamsModal(self.cog, profile_name, profile, self.is_borrowed, callback=refresh_cb)
            await interaction.response.send_modal(modal)
        elif choice == "thinking_params":
            async def refresh_cb(modal_interaction: discord.Interaction):
                new_embed = await self.cog.profile_manager._build_profile_manage_embed(modal_interaction, profile_name)
                await self.original_interaction.edit_original_response(embed=new_embed, view=self)
            
            # [UPDATED] Pass self.is_borrowed to the modal
            modal = ProfileThinkingParamsModal(self.cog, profile_name, profile, self.is_borrowed, callback=refresh_cb)
            await interaction.response.send_modal(modal)

        elif choice == "speech_settings":
            async def refresh_cb(modal_interaction: discord.Interaction):
                new_embed = await self.cog.profile_manager._build_profile_manage_embed(modal_interaction, profile_name)
                await self.original_interaction.edit_original_response(embed=new_embed, view=self)
            
            modal = ProfileSpeechSettingsModal(self.cog, profile_name, profile, self.is_borrowed, callback=refresh_cb)
            await interaction.response.send_modal(modal)

        # --- Tools Tab Logic ---
        elif choice == "cycle_response":
            modes = ["regular", "mention", "reply", "mention_reply"]
            curr = profile.get("response_mode", "regular")
            profile["response_mode"] = modes[(modes.index(curr) + 1) % len(modes)]
            await self._save_and_refresh(interaction, profile, profile_name, self.is_borrowed)
        elif choice == "image_toggle":
            # Inject prompt into current_params to avoid breaking the modal signature
            if not self.is_borrowed:
                prompts = self.cog.profile_manager._get_profile_prompts(self.user_id, profile_name) or {}
                profile["image_generation_prompt"] = prompts.get("image_generation_prompt")
                
            async def refresh_cb(modal_interaction: discord.Interaction):
                new_embed = await self.cog.profile_manager._build_profile_manage_embed(modal_interaction, profile_name)
                await self.original_interaction.edit_original_response(embed=new_embed, view=self)
            modal = ProfileImageGenSettingsModal(self.cog, profile_name, profile, self.is_borrowed, callback=refresh_cb)
            await interaction.response.send_modal(modal)
        elif choice == "typing":
            async def refresh_cb(modal_interaction: discord.Interaction):
                new_embed = await self.cog.profile_manager._build_profile_manage_embed(modal_interaction, profile_name)
                await self.original_interaction.edit_original_response(embed=new_embed, view=self)
            modal = ProfileTypingSettingsModal(self.cog, profile_name, profile, self.is_borrowed, callback=refresh_cb)
            await interaction.response.send_modal(modal)
        elif choice == "grounding":
            current_mode = profile.get("grounding_mode", "off")
            if isinstance(current_mode, bool): current_mode = "rag" if current_mode else "off"
            elif current_mode == "on" or current_mode == "on+": current_mode = "rag" # Legacy migration
            cycle_map = {"off": "native", "native": "rag", "rag": "off"}
            profile["grounding_mode"] = cycle_map.get(current_mode, "off")
            await self._save_and_refresh(interaction, profile, profile_name, self.is_borrowed)
        elif choice == "url_toggle":
            current_mode = profile.get("url_mode", "off")
            if "url_mode" not in profile:
                current_mode = "rag" if profile.get("url_fetching_enabled", False) else "off"
            cycle_map = {"off": "native", "native": "rag", "rag": "off"}
            profile["url_mode"] = cycle_map.get(current_mode, "off")
            profile["url_fetching_enabled"] = (profile["url_mode"] == "rag") # Legacy support
            await self._save_and_refresh(interaction, profile, profile_name, self.is_borrowed)
        elif choice == "time":
            await self._handle_timezone(interaction, profile, self.is_borrowed)
        elif choice == "critic":
            profile["critic_enabled"] = not profile.get("critic_enabled", False)
            await self._save_and_refresh(interaction, profile, profile_name, self.is_borrowed)
        elif choice == "help_mode":
            profile["help_mode_enabled"] = not profile.get("help_mode_enabled", False)
            await self._save_and_refresh(interaction, profile, profile_name, self.is_borrowed)
        elif choice == "neuro":
            async def refresh_cb(modal_interaction: discord.Interaction):
                new_embed = await self.cog.profile_manager._build_profile_manage_embed(modal_interaction, self.profile_name)
                await self.original_interaction.edit_original_response(embed=new_embed, view=self)
            modal = ProfileNeuroModal(self.cog, self.profile_name, profile, self.is_borrowed, callback=refresh_cb)
            await interaction.response.send_modal(modal)

        # --- Memory Tab Logic ---
        elif choice == "manage_ltm":
            view = DataManageView(self.cog, interaction, profile_name, self.is_borrowed, mode='ltm', parent_manage_view=self)
            await view.start()
        elif choice == "manage_training":
            view = DataManageView(self.cog, interaction, profile_name, self.is_borrowed, mode='training', parent_manage_view=self)
            await view.start()
            await interaction.response.defer()
        elif choice == "ltm_creation":
            profile["ltm_creation_enabled"] = not profile.get("ltm_creation_enabled", False)
            await self._save_and_refresh(interaction, profile, profile_name, self.is_borrowed)
        elif choice == "ltm_params":
            async def refresh_cb(i):
                new_embed = await self.cog.profile_manager._build_profile_manage_embed(i, profile_name)
                await self.original_interaction.edit_original_response(embed=new_embed, view=self)
            modal = ProfileLTMParamsModal(self.cog, profile_name, profile, callback=refresh_cb)
            await interaction.response.send_modal(modal)
        elif choice == "train_params":
            async def refresh_cb(i):
                new_embed = await self.cog.profile_manager._build_profile_manage_embed(i, profile_name)
                await self.original_interaction.edit_original_response(embed=new_embed, view=self)
            modal = ProfileTrainingParamsModal(self.cog, profile_name, profile, callback=refresh_cb)
            await interaction.response.send_modal(modal)
        elif choice == "ltm_summarization":
            instr = profile.get("ltm_summarization_instructions", DEFAULT_LTM_SUMMARIZATION_INSTRUCTIONS)
            modal = ProfileLTMSummarizationModal(self.cog, profile_name, instr)
            await interaction.response.send_modal(modal)

    # --- Internal Helpers for UI Flow ---

    async def _save_and_refresh(self, interaction, profile, profile_name, is_borrowed):
        self.cog.profile_manager._save_profile_config(self.user_id, profile_name, profile, is_borrowed)
        
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

        new_embed = await self.cog.profile_manager._build_profile_manage_embed(interaction, profile_name)
        await interaction.response.edit_message(embed=new_embed, view=self)

    async def _handle_safety_cycle(self, interaction, profile):
        is_public = self.cog.profile_manager._is_profile_public(self.user_id, self.profile_name)
        cycle_full = {'low': 'medium', 'medium': 'high', 'high': 'unrestricted', 'unrestricted': 'low'}
        cycle_rest = {'low': 'medium', 'medium': 'high', 'high': 'low'}
        curr = profile.get('safety_level', 'low')
        profile['safety_level'] = (cycle_rest if (self.is_borrowed or is_public) else cycle_full).get(curr, 'low')
        await self._save_and_refresh(interaction, profile, self.profile_name, self.is_borrowed)

    async def _handle_appearance(self, interaction):
        modal = AppearanceModal(self.cog, self.original_interaction, self.profile_name)
        await interaction.response.send_modal(modal)

    async def _handle_rename(self, interaction):
        modal = ui.Modal(title=f"Rename '{self.profile_name}'")
        new_name_input = ui.TextInput(label="Enter new unique name", required=True)
        modal.add_item(new_name_input)
        async def rename_submit(i: discord.Interaction):
            await i.response.defer()
            new_name = new_name_input.value.lower().strip()
            old_name = self.profile_name
            
            is_valid, err_msg = self.cog.profile_manager._is_valid_profile_name(new_name)
            if not is_valid:
                await self.original_interaction.edit_original_response(content=f"Rename failed: {err_msg}", view=None, embed=None); return
                
            if not new_name or new_name.lower() == 'clyde':
                await self.original_interaction.edit_original_response(content="Rename failed: Invalid name.", view=None, embed=None); return
            user_index = self.cog.profile_manager._get_user_index(self.user_id)
            if new_name in user_index.get("personal", []) or new_name in user_index.get("borrowed", []):
                await self.original_interaction.edit_original_response(content="Rename failed: Name already exists.", view=None, embed=None); return
            
            p_dict_key = "borrowed" if self.is_borrowed else "personal"
            if old_name in user_index.get(p_dict_key, {}):
                if isinstance(user_index[p_dict_key], dict):
                    pid = user_index[p_dict_key].pop(old_name)
                    user_index[p_dict_key][new_name] = pid
                    # Update local name text file
                    with open(os.path.join(self.cog.USERS_DIR, str(self.user_id), "profiles", pid, "name.txt"), "w", encoding="utf-8") as f:
                        f.write(new_name)
                else:
                    user_index[p_dict_key].remove(old_name)
                    user_index[p_dict_key].append(new_name)
                    old_dir = os.path.join(self.cog.USERS_DIR, str(self.user_id), "profiles", old_name)
                    new_dir = os.path.join(self.cog.USERS_DIR, str(self.user_id), "profiles", new_name)
                    if os.path.exists(old_dir):
                        os.rename(old_dir, new_dir)
                
                self.cog.profile_manager._save_user_index(self.user_id, user_index)

                # Hot-swap live sessions and models to prevent corruption
                for ch_id, session in self.cog.multi_profile_channels.items():
                    for p in session.get("profiles", []):
                        if p["owner_id"] == self.user_id and p["profile_name"] == old_name:
                            p["profile_name"] = new_name
                    
                    old_key = (self.user_id, old_name)
                    new_key = (self.user_id, new_name)
                    if old_key in session.get("chat_sessions", {}):
                        session["chat_sessions"][new_key] = session["chat_sessions"].pop(old_key)
                
                keys_to_clear = [k for k in self.cog.channel_models.keys() if isinstance(k, tuple) and k[1] == self.user_id and k[2] == old_name]
                for k in keys_to_clear:
                    self.cog.channel_models.pop(k, None)
                    self.cog.channel_model_last_profile_key.pop(k, None)
                
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
            
            is_valid, err_msg = self.cog.profile_manager._is_valid_profile_name(new_name)
            if not is_valid:
                await self.original_interaction.edit_original_response(content=f"Duplicate failed: {err_msg}", view=None, embed=None); return
                
            user_index = self.cog.profile_manager._get_user_index(self.user_id)
            if new_name in user_index.get("personal", []) or new_name in user_index.get("borrowed", []):
                await self.original_interaction.edit_original_response(content="Duplicate failed: Name already exists.", view=None, embed=None); return
            
            limit = defaultConfig.LIMIT_PROFILES_PREMIUM if self.cog.profile_manager.is_user_premium(self.user_id) else defaultConfig.LIMIT_PROFILES_FREE
            if len(user_index.get("personal", {})) >= limit:
                await self.original_interaction.edit_original_response(content="Limit reached.", view=None, embed=None); return
            
            old_pid = self.cog.profile_manager._get_pid_from_name_any(self.user_id, self.profile_name)
            old_dir = os.path.join(self.cog.USERS_DIR, str(self.user_id), "profiles", old_pid)
            
            import uuid
            if not isinstance(user_index.get("personal"), dict):
                legacy_personal = user_index.get("personal", [])
                user_index["personal"] = {}
                if isinstance(legacy_personal, list):
                    for p_name in legacy_personal:
                        user_index["personal"][p_name] = p_name
            
            new_pid = f"A{uuid.uuid4().hex[:15].upper()}"
            user_index["personal"][new_name] = new_pid
                
            new_dir = os.path.join(self.cog.USERS_DIR, str(self.user_id), "profiles", new_pid)
            
            import shutil
            try:
                os.makedirs(new_dir, exist_ok=True)
                for item in os.listdir(old_dir):
                    if item in ["child_bot.json.gz", "global_chat.json.gz", "ltm.json.gz"]:
                        continue
                    s = os.path.join(old_dir, item)
                    d = os.path.join(new_dir, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d)
                    else:
                        shutil.copy2(s, d)
                
                if isinstance(user_index.get("personal"), dict):
                    with open(os.path.join(new_dir, "name.txt"), "w", encoding="utf-8") as f:
                        f.write(new_name)
            except Exception as e:
                print(f"Error duplicating profile directory: {e}")
            
            self.cog.profile_manager._save_user_index(self.user_id, user_index)
            
            config = self.cog.profile_manager._get_profile_config(self.user_id, new_name, False)
            if config:
                import uuid
                config['profile_id'] = str(uuid.uuid4().hex[:8].upper()) # Force unique PID for duplicate
                config['created_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                self.cog.profile_manager._save_profile_config(self.user_id, new_name, config, False)

            self.cog.memory_manager._copy_ltm_shard(str(self.user_id), self.profile_name, new_name)
            self.cog.memory_manager._copy_training_shard(str(self.user_id), self.profile_name, new_name)
            await self.original_interaction.edit_original_response(content=f"Duplicated to '{new_name}'.", view=None, embed=None)
        modal.on_submit = duplicate_submit
        await interaction.response.send_modal(modal)

    async def _handle_convert_copy(self, interaction: discord.Interaction, to_system: bool):
        target_type = "System Profile" if to_system else "Personal Profile"
        modal = ui.Modal(title=f"Copy to {target_type}")
        new_name_input = ui.TextInput(
            label="Destination Profile Name",
            default=self.profile_name,
            required=True,
            max_length=30
        )
        modal.add_item(new_name_input)

        async def convert_submit(i: discord.Interaction):
            await i.response.defer(ephemeral=True, thinking=True)
            target_name = new_name_input.value.lower().strip()
            
            success, msg = await self.cog.profile_manager._convert_copy_profile(
                user_id=i.user.id,
                source_name=self.profile_name,
                target_name=target_name,
                to_system=to_system
            )
            
            if success:
                await i.followup.send(f"✅ {msg}", ephemeral=True)
            else:
                await i.followup.send(f"❌ **Conversion Failed:** {msg}", ephemeral=True)

        modal.on_submit = convert_submit
        await interaction.response.send_modal(modal)

    async def _handle_share(self, interaction):
        view = HubShareManagerView(self.cog, interaction)
        view.selected_profiles = [self.profile_name]
        view.setup_items()
        desc = "Manage how you share your profiles."
        embed = discord.Embed(title="Share Manager", description=desc, color=discord.Color.teal())
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

    async def _handle_delete(self, interaction):
        async def confirm_delete(i: discord.Interaction):
            await i.response.defer()

            def _sync_delete():
                user_index = self.cog.profile_manager._get_user_index(self.user_id)
                list_key = "borrowed" if self.is_borrowed else "personal"

                if self.profile_name not in user_index.get(list_key, {}):
                    return False

                if isinstance(user_index[list_key], dict):
                    pid = user_index[list_key].pop(self.profile_name)
                else:
                    user_index[list_key].remove(self.profile_name)
                    pid = self.profile_name

                if not self.is_borrowed:
                    self.cog.profile_manager._cascade_delete_borrowed_profiles(self.user_id, pid, self.profile_name)

                self.cog.profile_manager._save_user_index(self.user_id, user_index)

                import shutil
                p_dir = os.path.join(self.cog.USERS_DIR, str(self.user_id), "profiles", pid)
                shutil.rmtree(p_dir, ignore_errors=True)
                return True

            if await asyncio.to_thread(_sync_delete):
                await self.original_interaction.edit_original_response(content=f"Profile '{self.profile_name}' deleted.", view=None, embed=None)
        confirm_view = build_confirm_view("Confirm Deletion", confirm_delete)
        await interaction.response.send_message(f"Delete profile '{self.profile_name}'?", view=confirm_view, ephemeral=True)

    async def _handle_timezone(self, interaction, profile, is_borrowed):
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
                        self.cog.profile_manager._save_profile_config(self.user_id, self.profile_name, profile, is_borrowed)
                        new_embed = await self.cog.profile_manager._build_profile_manage_embed(mi, self.profile_name)
                        await self.original_interaction.edit_original_response(embed=new_embed, view=self)
                        await mi.response.send_message("Updated.", ephemeral=True, delete_after=3)
                    except: await mi.response.send_message("Invalid.", ephemeral=True, delete_after=5)
                modal.on_submit = custom_sub
                await i.response.send_modal(modal)
            else:
                profile['timezone'] = select.values[0]
                self.cog.profile_manager._save_profile_config(self.user_id, self.profile_name, profile, is_borrowed)
                new_embed = await self.cog.profile_manager._build_profile_manage_embed(i, self.profile_name)
                await self.original_interaction.edit_original_response(embed=new_embed, view=self)
                await i.response.defer()
        select.callback = tz_cb
        view.add_item(select)
        await interaction.response.send_message("Select Timezone:", view=view, ephemeral=True)

    async def on_timeout(self):
        try: await self.original_interaction.edit_original_response(content="Manager timed out.", view=None)
        except: pass

class EditUserProfilePersonaModal(ui.Modal): 
    def __init__(self, cog_instance, profile_name: str, current_persona_data: Dict[str, List[str]], user_id: int):
        self.cog_instance: MimicCog = cog_instance
        self.profile_name = profile_name
        self.user_id = user_id
        self.persona_sections_order = cog_instance.persona_modal_sections_order

        title = f"Edit Persona for Profile: '{profile_name}'"[:45]; super().__init__(title=title)

        for key in self.persona_sections_order:
            decrypted_content = "\n".join(self.cog_instance.storage_manager._decrypt_data(line) for line in current_persona_data.get(key, []))
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
        
        success = await self.cog_instance.profile_manager.update_user_profile_persona(
            self.user_id, self.profile_name, updated_persona_data, i.channel_id
        )
        scope = f"your profile '{self.profile_name}'"
        message = f"Persona sections for {scope} {'updated' if success else 'update failed (max profiles reached or other issue)'}."

        await i.followup.send(message, ephemeral=True)
    async def on_error(self, i:discord.Interaction, e:Exception): print(f"EditUserProfilePersonaModal err: {e}"); traceback.print_exc(); await i.followup.send('Form error.',ephemeral=True)

class EditUserProfileAIInstructionsModal(ui.Modal): 
    def __init__(self, cog_instance, profile_name: str, current_instr:str, user_id: int):
        self.cog:MimicCog=cog_instance
        self.profile_name = profile_name
        self.user_id = user_id
        
        title=f"Edit AI Instructions for Profile: '{profile_name}'"[:45]; super().__init__(title=title)
        
        p1, p2, p3, p4 = "", "", "", ""
        if isinstance(current_instr, list):
            if len(current_instr) >= 1: p1 = self.cog.storage_manager._decrypt_data(current_instr[0])
            if len(current_instr) >= 2: p2 = self.cog.storage_manager._decrypt_data(current_instr[1])
            if len(current_instr) >= 3: p3 = self.cog.storage_manager._decrypt_data(current_instr[2])
            if len(current_instr) >= 4: p4 = self.cog.storage_manager._decrypt_data(current_instr[3])

        self.add_item(ui.TextInput(label="Part 1",custom_id="ai_p1",style=discord.TextStyle.paragraph,default=p1,required=False,max_length=AI_INSTRUCTIONS_PART_MAX_LENGTH))
        self.add_item(ui.TextInput(label="Part 2",custom_id="ai_p2",style=discord.TextStyle.paragraph,default=p2,required=False,max_length=AI_INSTRUCTIONS_PART_MAX_LENGTH))
        self.add_item(ui.TextInput(label="Part 3",custom_id="ai_p3",style=discord.TextStyle.paragraph,default=p3,required=False,max_length=AI_INSTRUCTIONS_PART_MAX_LENGTH))
        self.add_item(ui.TextInput(label="Style Guide (Reserved For Training)",custom_id="ai_p4",style=discord.TextStyle.paragraph,default=p4,required=False,max_length=AI_INSTRUCTIONS_PART_MAX_LENGTH))

    async def on_submit(self, i:discord.Interaction):
        await i.response.defer(ephemeral=True,thinking=True)
        p1=next(c.value for c in self.children if c.custom_id=="ai_p1"); p2=next(c.value for c in self.children if c.custom_id=="ai_p2")
        p3=next(c.value for c in self.children if c.custom_id=="ai_p3"); p4=next(c.value for c in self.children if c.custom_id=="ai_p4") 
        instr_list = [p1, p2, p3, p4]
        success = await self.cog.profile_manager.update_user_profile_ai_instructions(
            self.user_id, self.profile_name, instr_list, i.channel_id
        )
        scope=f"your profile '{self.profile_name}'"
        message = f"AI Instructions for {scope} {'updated' if success else 'update failed (max profiles reached or other issue)'}."

        await i.followup.send(message,ephemeral=True)
    async def on_error(self, i:discord.Interaction,e:Exception): print(f"EditUserProfileAIInstrModal err: {e}"); traceback.print_exc(); await i.followup.send('Form error.',ephemeral=True)

def ProfileParamsModal(cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, callback=None):
    fields = [
        {"label": "Temperature (0.0-2.0)", "custom_id": "temperature", "default": str(current_params.get("temperature", defaultConfig.GEMINI_TEMPERATURE)), "required": False},
        {"label": "Top P (0.0-1.0)", "custom_id": "top_p", "default": str(current_params.get("top_p", defaultConfig.GEMINI_TOP_P)), "required": False},
        {"label": "Top K (integer 0-100)", "custom_id": "top_k", "default": str(current_params.get("top_k", defaultConfig.GEMINI_TOP_K)), "required": False},
        {"label": f"STM Length (0-{STM_LIMIT_MAX})", "custom_id": "stm_length", "default": str(current_params.get("stm_length", defaultConfig.CHATBOT_MEMORY_LENGTH)), "required": False}
    ]
    def parser(v):
        c = {}
        t = _pf(v["temperature"]); p = _pf(v["top_p"]); k = _pi(v["top_k"]); s = _pi(v["stm_length"])
        if t is not None:
            if not (0.0 <= t <= 2.0): raise ValueError("Temperature out of range")
            c["temperature"] = t
        if p is not None:
            if not (0.0 <= p <= 1.0): raise ValueError("Top P out of range")
            c["top_p"] = p
        if k is not None:
            if not (0 <= k <= 100): raise ValueError("Top K out of range")
            c["top_k"] = k
        if s is not None:
            if not (0 <= s <= STM_LIMIT_MAX): raise ValueError(f"STM Length out of range (0-{STM_LIMIT_MAX})")
            c["stm_length"] = s
        return {"config": c}
    return ConfigModal(cog, profile_name, is_borrowed, "Set Profile Generation Parameters", fields, parser, callback)

def ProfileTrainingParamsModal(cog, profile_name: str, current_params: Dict[str, Any], callback=None):
    fields = [
        {"label": "Context Size (0-10)", "custom_id": "training_context_size", "default": str(current_params.get("training_context_size", defaultConfig.TRAINING_CONTEXT_SIZE)), "required": False},
        {"label": "Relevance Threshold (0.0-1.0)", "custom_id": "training_relevance_threshold", "default": str(current_params.get("training_relevance_threshold", defaultConfig.TRAINING_RELEVANCE_THRESHOLD)), "required": False}
    ]
    def parser(v):
        c = {}
        cs = _pi(v["training_context_size"]); rt = _pf(v["training_relevance_threshold"])
        if cs is not None:
            if not (0 <= cs <= 10): raise ValueError("Context Size out of range")
            c["training_context_size"] = cs
        if rt is not None:
            if not (0.0 <= rt <= 1.0): raise ValueError("Relevance Threshold out of range")
            c["training_relevance_threshold"] = rt
        return {"config": c}
    return ConfigModal(cog, profile_name, False, "Set Profile Training Parameters", fields, parser, callback)

def ProfileThinkingParamsModal(cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, callback=None):
    fields = [
        {"label": "Thinking Summary (on/off)", "custom_id": "thinking_summary_visible", "default": current_params.get("thinking_summary_visible", "off"), "required": False, "placeholder": "Display reasoning tokens below your message."},
        {"label": "Reasoning Effort / Level", "custom_id": "thinking_level", "default": current_params.get("thinking_level", "high"), "required": False, "placeholder": "xhigh, high, medium, low, minimal, none"},
        {"label": "Reasoning Token Budget (-1=dyn)", "custom_id": "thinking_budget", "default": str(current_params.get("thinking_budget", -1)), "required": False, "placeholder": "-1 = dynamic, 128+ = token limit"},
        {"label": "Thought Signatures (on/off)", "custom_id": "thinking_signatures_enabled", "default": current_params.get("thinking_signatures_enabled", "off"), "required": False, "placeholder": "Preserve reasoning context across turns."}
    ]
    def parser(v):
        c = {}
        sv = _ps(v["thinking_summary_visible"])
        c["thinking_summary_visible"] = "on" if sv and sv.lower() == "on" else "off"
        
        lv = _ps(v["thinking_level"])
        c["thinking_level"] = lv.lower() if lv and lv.lower() in ["xhigh", "high", "medium", "low", "minimal", "none"] else "high"
        
        bv = _pi(v["thinking_budget"])
        c["thinking_budget"] = min(bv if bv is not None and bv >= -1 else -1, 32768)
        
        ts = _ps(v["thinking_signatures_enabled"])
        c["thinking_signatures_enabled"] = "on" if ts and ts.lower() == "on" else "off"
        return {"config": c}
    return ConfigModal(cog, profile_name, is_borrowed, "Thinking & Reasoning Parameters", fields, parser, callback)

def ProfileLTMParamsModal(cog, profile_name: str, current_params: Dict[str, Any], callback=None):
    fields = [
        {"label": "Creation Interval (5-100 msgs)", "custom_id": "ltm_creation_interval", "default": str(current_params.get("ltm_creation_interval", 10)), "required": False, "placeholder": "Default: 10"},
        {"label": "Summarization Context (5-50 msgs)", "custom_id": "ltm_summarization_context", "default": str(current_params.get("ltm_summarization_context", 10)), "required": False, "placeholder": "Default: 10"},
        {"label": "Recall Context Size (0-10)", "custom_id": "ltm_context_size", "default": str(current_params.get("ltm_context_size", 3)), "required": False, "placeholder": "Default: 3"},
        {"label": "Relevance Threshold (0.0-1.0)", "custom_id": "ltm_relevance_threshold", "default": str(current_params.get("ltm_relevance_threshold", 0.75)), "required": False, "placeholder": "Default: 0.75"}
    ]
    def parser(v):
        c = {}
        inv = _pi(v["ltm_creation_interval"]); ctx = _pi(v["ltm_summarization_context"])
        rs = _pi(v["ltm_context_size"]); rt = _pf(v["ltm_relevance_threshold"])
        if inv is not None:
            if not (5 <= inv <= 100): raise ValueError("Interval out of range")
            c["ltm_creation_interval"] = inv
        if ctx is not None:
            if not (5 <= ctx <= 50): raise ValueError("Context out of range")
            c["ltm_summarization_context"] = ctx
        if rs is not None:
            if not (0 <= rs <= 10): raise ValueError("Context Size out of range")
            c["ltm_context_size"] = rs
        if rt is not None:
            if not (0.0 <= rt <= 1.0): raise ValueError("Relevance Threshold out of range")
            c["ltm_relevance_threshold"] = rt
        return {"config": c}
    return ConfigModal(cog, profile_name, False, "LTM Parameters", fields, parser, callback)

def ProfileLTMSummarizationModal(cog, profile_name: str, current_instructions: str, callback=None):
    decrypted = cog.storage_manager._decrypt_data(current_instructions)
    fields = [{
        "label": "AI Instructions for Summarization",
        "custom_id": "ltm_summarization_instructions",
        "style": discord.TextStyle.paragraph,
        "default": decrypted,
        "required": True,
        "max_length": 2000,
        "placeholder": "The system will automatically append the conversation excerpt to these instructions."
    }]
    def parser(v):
        ins = _ps(v["ltm_summarization_instructions"]) or DEFAULT_LTM_SUMMARIZATION_INSTRUCTIONS
        return {"prompts": {"ltm_summarization_instructions": cog.storage_manager._encrypt_data(ins)}}
    return ConfigModal(cog, profile_name, False, "Set LTM Summarization Instructions", fields, parser, callback)

def ProfileTypingSettingsModal(cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, callback=None):
    fields = [
        {"label": "Enable Realistic Typing (on/off)", "custom_id": "realistic_typing_enabled", "default": "on" if current_params.get("realistic_typing_enabled") else "off", "required": True},
        {"label": "Mode (sentence/line)", "custom_id": "typing_mode", "default": current_params.get("typing_mode", "sentence"), "required": False, "placeholder": "Default: sentence"},
        {"label": "Characters per Second", "custom_id": "typing_cps", "default": str(current_params.get("typing_cps", 30.0)), "required": False, "placeholder": "Default: 30.0"},
        {"label": "Max Delay per Chunk (Seconds)", "custom_id": "typing_max_delay", "default": str(current_params.get("typing_max_delay", 2.5)), "required": False, "placeholder": "Default: 2.5"}
    ]
    def parser(v):
        c = {"realistic_typing_enabled": _pb(v["realistic_typing_enabled"])}
        m = _ps(v["typing_mode"])
        if m: c["typing_mode"] = "line" if m.lower() == "line" else "sentence"
        cps = _pf(v["typing_cps"])
        if cps is not None: c["typing_cps"] = cps
        md = _pf(v["typing_max_delay"])
        if md is not None: c["typing_max_delay"] = md
        return {"config": c}
    return ConfigModal(cog, profile_name, is_borrowed, "Realistic Typing Settings", fields, parser, callback)

def ProfileImageGenSettingsModal(cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, callback=None):
    fields = [
        {"label": "Enable Image Gen (on/off)", "custom_id": "image_generation_enabled", "default": "on" if current_params.get("image_generation_enabled") else "off", "required": True}
    ]
    if not is_borrowed:
        enc = current_params.get("image_generation_prompt")
        dec = cog.storage_manager._decrypt_data(enc) if enc else ""
        fields.append({"label": "Image Generation Prompt", "custom_id": "image_generation_prompt", "style": discord.TextStyle.paragraph, "default": dec, "required": False, "max_length": 2000})
        
    def parser(v):
        c = {"image_generation_enabled": _pb(v["image_generation_enabled"])}
        p = {}
        if not is_borrowed and "image_generation_prompt" in v:
            pr = _ps(v["image_generation_prompt"])
            p["image_generation_prompt"] = cog.storage_manager._encrypt_data(pr) if pr else None
        return {"config": c, "prompts": p}
    return ConfigModal(cog, profile_name, is_borrowed, "Image Generation Settings", fields, parser, callback)

class SingleProfileModelView(ui.View):
    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction, profile_name: str):
        super().__init__(timeout=300)
        self.cog = cog
        self.original_interaction = interaction
        self.user_id = interaction.user.id
        self.profile_name = profile_name
        self.view_mode = 'google'
        self.category = 'response' # 'response', 'media', 'tools', 'ltm'

        index = self.cog.profile_manager._get_user_index(self.user_id)
        self.is_borrowed = profile_name in index.get("borrowed", [])
        
        self._build_view()

    def _get_current_profile_data(self) -> Dict[str, Any]:
        return self.cog.profile_manager._get_profile_config(self.user_id, self.profile_name, self.is_borrowed) or {}

    def _save_changes(self, key: str, value: Any):
        target_dict = self._get_current_profile_data()
        if target_dict is not None:
            target_dict[key] = value
            self.cog.profile_manager._save_profile_config(self.user_id, self.profile_name, target_dict, self.is_borrowed)
            
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
        data = self._get_current_profile_data()
        
        def clean(val):
            if not val: return "None"
            return str(val).replace("GOOGLE/", "").replace("OPENROUTER/", "").replace("OLLAMA/", "")
            
        msg = f"**Profile:** `{self.profile_name}`\n"
        if self.view_mode == 'openrouter':
            msg += "⚠️ **Note:** OpenRouter / Custom models require **RAG Mode** for Grounding and URL Context.\n\n"
        elif self.view_mode == 'ollama':
            msg += "⚠️ **Note:** Localhost models run on your machine's hardware. Processing speed depends on your GPU/CPU.\n\n"
        
        if self.category == 'response':
            p_clean = clean(data.get("primary_model", PRIMARY_MODEL_NAME))
            f_clean = clean(data.get("fallback_model", FALLBACK_MODEL_NAME))
            fb_status = "ON" if data.get("show_fallback_indicator", True) else "OFF"
            msg += f"**Primary:** `{p_clean}`\n**Fallback:** `{f_clean}`\n**Fallback Indicator:** `{fb_status}`\n"
        elif self.category == 'media':
            i_clean = clean(data.get("image_generation_model", "gemini-2.5-flash-image"))
            t_clean = clean(data.get("speech_model", "gemini-2.5-flash-preview-tts"))
            msg += f"**Image Generation:** `{i_clean}`\n**Text-to-Speech:** `{t_clean}`\n"
        elif self.category == 'tools':
            g_clean = clean(data.get("grounding_rag_model", FALLBACK_MODEL_NAME))
            c_clean = clean(data.get("critic_model", FALLBACK_MODEL_NAME))
            msg += f"**Grounding RAG:** `{g_clean}`\n**Anti-Repetition Critic:** `{c_clean}`\n"
        elif self.category == 'ltm':
            l_clean = clean(data.get("ltm_model", FALLBACK_MODEL_NAME))
            msg += f"**LTM Summariser:** `{l_clean}`\n"
            
        return msg

    def _get_top_models(self, provider: str, target_config_key: str) -> List[str]:
        return self.cog.api_service.get_top_models(provider, target_config_key)

    async def _update_ollama_status(self):
        host_url = OLLAMA_LOCAL_URL
        if hasattr(self, 'profile_name') and self.profile_name != "BULK_APPLY":
            cfg = self.cog.profile_manager._get_profile_config(self.user_id, self.profile_name, getattr(self, 'is_borrowed', False)) or {}
            host_url = cfg.get("ollama_host_url", OLLAMA_LOCAL_URL)
        
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{host_url.rstrip('/')}/api/tags", timeout=2.0)
                self.ollama_working = (resp.status_code == 200)
                if self.ollama_working:
                    data = resp.json()
                    self.cached_ollama_models = [m['name'] for m in data.get('models', [])]
        except Exception:
            self.ollama_working = False
            self.cached_ollama_models = []

    class GenericModelSelect(ui.Select):
        def __init__(self, placeholder: str, options: list, row: int, target_config_key: str):
            super().__init__(placeholder=placeholder, options=options, row=row)
            self.target_config_key = target_config_key

        async def callback(self, interaction: discord.Interaction):
            view: SingleProfileModelView = self.view
            if self.values[0] == "ollama_offline":
                await interaction.response.send_message("Ollama is offline or has no models downloaded.", ephemeral=True)
                return
            if self.values[0] == "custom_option":
                await interaction.response.send_modal(CustomModelModal(view, self.target_config_key))
            else: 
                view._save_changes(self.target_config_key, self.values[0])
                view._build_view()
                await interaction.response.edit_message(content=view._get_selection_feedback_message(), view=view)

    def _create_model_options(self, current_val: str, target_config_key: str) -> List[discord.SelectOption]:
        top_models = self._get_top_models(self.view_mode, target_config_key)
        opts = [discord.SelectOption(label="Custom Model...", value="custom_option", description="Enter manually via modal")]
        
        if current_val:
            opts.append(discord.SelectOption(label=f"Current: {current_val}", value=current_val, default=True))
        
        prefix = "GOOGLE/"
        if self.view_mode == 'openrouter': prefix = "OPENROUTER/"
        elif self.view_mode == 'ollama': prefix = "OLLAMA/"
        
        if target_config_key in ['image_generation_model', 'speech_model']:
            prefix = "GOOGLE/"
            
        added = len(opts)
        for m in top_models:
            if added >= 25: break
            val = f"{prefix}{m}"
            if current_val != val:
                opts.append(discord.SelectOption(label=m[:100], value=val))
                added += 1
                
        if self.view_mode == 'ollama' and not top_models:
            opts.append(discord.SelectOption(label="⚠️ Ollama Offline / No Models", value="ollama_offline", description=f"Check {OLLAMA_LOCAL_URL}"))
            
        return opts

    def _build_view(self):
        self.clear_items()
        data = self._get_current_profile_data()
        
        # Enforce Google Mode for Media
        if self.category == 'media':
            self.view_mode = 'google'
            
        # --- Row 0 & 1: Dropdowns ---
        if self.category == 'response':
            p_val = data.get("primary_model", PRIMARY_MODEL_NAME)
            f_val = data.get("fallback_model", FALLBACK_MODEL_NAME)
            self.add_item(self.GenericModelSelect("Select Primary Model...", self._create_model_options(p_val, "primary_model"), 0, "primary_model"))
            self.add_item(self.GenericModelSelect("Select Fallback Model...", self._create_model_options(f_val, "fallback_model"), 1, "fallback_model"))
            
        elif self.category == 'media':
            i_val = data.get("image_generation_model", "GOOGLE/gemini-2.5-flash-image")
            t_val = data.get("speech_model", "GOOGLE/gemini-2.5-flash-preview-tts")
            self.add_item(self.GenericModelSelect("Select Image Gen Model...", self._create_model_options(i_val, "image_generation_model"), 0, "image_generation_model"))
            self.add_item(self.GenericModelSelect("Select Text-to-Speech Model...", self._create_model_options(t_val, "speech_model"), 1, "speech_model"))
            
        elif self.category == 'tools':
            g_val = data.get("grounding_rag_model", FALLBACK_MODEL_NAME)
            c_val = data.get("critic_model", FALLBACK_MODEL_NAME)
            self.add_item(self.GenericModelSelect("Select Grounding RAG Model...", self._create_model_options(g_val, "grounding_rag_model"), 0, "grounding_rag_model"))
            self.add_item(self.GenericModelSelect("Select Critic Model...", self._create_model_options(c_val, "critic_model"), 1, "critic_model"))
            
        elif self.category == 'ltm':
            l_val = data.get("ltm_model", FALLBACK_MODEL_NAME)
            self.add_item(self.GenericModelSelect("Select LTM Summariser Model...", self._create_model_options(l_val, "ltm_model"), 0, "ltm_model"))

        # --- Row 2 Actions ---
        api_modes = ['google', 'openrouter', 'ollama']
        api_labels = {'google': 'API: Google', 'openrouter': 'API: OpenRouter', 'ollama': 'API: Ollama (Local)'}
        
        btn_api = ui.Button(label=api_labels[self.view_mode], style=discord.ButtonStyle.primary, row=2, disabled=(self.category == 'media'))
        async def api_cb(i: discord.Interaction):
            next_idx = (api_modes.index(self.view_mode) + 1) % len(api_modes)
            self.view_mode = api_modes[next_idx]
            if self.view_mode == 'ollama':
                await i.response.defer()
                self.ollama_working = "processing"
                await self._update_ollama_status()
                self._build_view()
                await i.edit_original_response(content=self._get_selection_feedback_message(), view=self)
            else:
                self._build_view()
                await i.response.edit_message(content=self._get_selection_feedback_message(), view=self)
        btn_api.callback = api_cb
        self.add_item(btn_api)
        
        if self.view_mode == 'ollama':
            host_style = discord.ButtonStyle.secondary
            if getattr(self, 'ollama_working', None) == "processing":
                host_style = discord.ButtonStyle.blurple
            elif getattr(self, 'ollama_working', None) is True:
                host_style = discord.ButtonStyle.success
            elif getattr(self, 'ollama_working', None) is False:
                host_style = discord.ButtonStyle.danger
                
            btn_host = ui.Button(label="Set Host URL", style=host_style, row=2)
            async def host_cb(i: discord.Interaction):
                await i.response.send_modal(OllamaHostModal(self))
            btn_host.callback = host_cb
            self.add_item(btn_host)
            
            btn_guide = ui.Button(label="Guide", style=discord.ButtonStyle.secondary, row=2)
            async def guide_cb(i: discord.Interaction):
                await i.response.send_message(OLLAMA_GUIDE_TEXT, ephemeral=True)
            btn_guide.callback = guide_cb
            self.add_item(btn_guide)

        categories = ['response', 'media', 'tools', 'ltm']
        cat_labels = {'response': 'Response', 'media': 'Media', 'tools': 'Tools', 'ltm': 'LTM'}
        btn_cat = ui.Button(label=f"Category: {cat_labels[self.category]}", style=discord.ButtonStyle.blurple, row=2)
        async def cat_cb(i: discord.Interaction):
            next_idx = (categories.index(self.category) + 1) % len(categories)
            self.category = categories[next_idx]
            self._build_view()
            await i.response.edit_message(content=self._get_selection_feedback_message(), view=self)
        btn_cat.callback = cat_cb
        self.add_item(btn_cat)

        if self.category == 'response':
            show_fb = data.get("show_fallback_indicator", True)
            fb_label = "Fallback Indicator: ON" if show_fb else "Fallback Indicator: OFF"
            fb_style = discord.ButtonStyle.success if show_fb else discord.ButtonStyle.secondary
            btn_fallback = ui.Button(label=fb_label, style=fb_style, row=2)
            
            async def fallback_cb(i: discord.Interaction):
                curr = self._get_current_profile_data().get("show_fallback_indicator", True)
                self._save_changes("show_fallback_indicator", not curr)
                self._build_view()
                await i.response.edit_message(content=self._get_selection_feedback_message(), view=self)
                
            btn_fallback.callback = fallback_cb
            self.add_item(btn_fallback)

        # --- Row 4: Category Navigation ---
        cats = [
            ("Response", "response"),
            ("Media", "media"),
            ("Tools", "tools"),
            ("LTM", "ltm")
        ]
        
        for label, val in cats:
            btn_style = discord.ButtonStyle.primary if self.category == val else discord.ButtonStyle.secondary
            btn = ui.Button(label=label, style=btn_style, row=4, disabled=(self.category == val))
            
            # Closure for callback
            def make_nav_cb(target_cat):
                async def nav_cb(i: discord.Interaction):
                    self.category = target_cat
                    self._build_view()
                    await i.response.edit_message(content=self._get_selection_feedback_message(), view=self)
                return nav_cb
                
            btn.callback = make_nav_cb(val)
            self.add_item(btn)

class ModelApplyView(ui.View):
    def __init__(self, cog: 'MimicCog', user_id: int, interaction: discord.Interaction):
        super().__init__(timeout=300)
        self.cog = cog
        self.user_id = user_id
        self.interaction = interaction
        self.target_profiles: Set[str] = set()
        self.current_page = 0
        self.view_mode = 'google' 
        self.category = 'response'
        self.show_fallback_indicator = None 
        self.view_source = 'personal'
        
        self.models_state = {
            'primary_model': None,
            'fallback_model': None,
            'image_generation_model': None,
            'speech_model': None,
            'grounding_rag_model': None,
            'critic_model': None,
            'ltm_model': None,
            'ollama_host_url': None
        }

        self._load_lists()
        self._build_view()

    def _load_lists(self):
        index = self.cog.profile_manager._get_user_index(self.user_id)
        self.personal_profiles = sorted(list(index.get("personal", {}).keys())) if isinstance(index.get("personal"), dict) else sorted(list(index.get("personal", [])))
        self.borrowed_profiles = sorted(list(index.get("borrowed", {}).keys())) if isinstance(index.get("borrowed"), dict) else sorted(list(index.get("borrowed", [])))

    def _get_active_list(self):
        return self.personal_profiles if self.view_source == 'personal' else self.borrowed_profiles

    def _save_changes(self, key: str, value: Any):
        if key == "show_fallback_indicator":
            self.show_fallback_indicator = value
        else:
            self.models_state[key] = value

    def _get_selection_feedback_message(self) -> str:
        count = len(self.target_profiles)
        
        def clean(val):
            if not val: return "Unchanged"
            return str(val).replace("GOOGLE/", "").replace("OPENROUTER/", "").replace("OLLAMA/", "")
            
        msg = f"**Category:** `{self.category.title()}`\n"
        if self.view_mode == 'openrouter':
            msg += "⚠️ **Note:** OpenRouter / Custom models require **RAG Mode** for Grounding and URL Context.\n\n"
        elif self.view_mode == 'ollama':
            msg += "⚠️ **Note:** Localhost models run on your machine's hardware. Processing speed depends on your GPU/CPU.\n\n"
        
        if self.category == 'response':
            p_clean = clean(self.models_state['primary_model'])
            f_clean = clean(self.models_state['fallback_model'])
            fb_status = "Unchanged" if self.show_fallback_indicator is None else ("ON" if self.show_fallback_indicator else "OFF")
            msg += f"**Primary:** `{p_clean}`\n**Fallback:** `{f_clean}`\n**Fallback Indicator:** `{fb_status}`\n\n"
        elif self.category == 'media':
            i_clean = clean(self.models_state['image_generation_model'])
            t_clean = clean(self.models_state['speech_model'])
            msg += f"**Image Generation:** `{i_clean}`\n**Text-to-Speech:** `{t_clean}`\n\n"
        elif self.category == 'tools':
            g_clean = clean(self.models_state['grounding_rag_model'])
            c_clean = clean(self.models_state['critic_model'])
            msg += f"**Grounding RAG:** `{g_clean}`\n**Anti-Repetition Critic:** `{c_clean}`\n\n"
        elif self.category == 'ltm':
            l_clean = clean(self.models_state['ltm_model'])
            msg += f"**LTM Summariser:** `{l_clean}`\n\n"
            
        if count == 0:
            msg += "Use the dropdowns below to select models and the profiles to apply them to."
            return msg
        
        profile_list = sorted(list(self.target_profiles))
        msg += f"**Selected Profiles ({count}):**\n"
        msg += "\n".join(f"- `{name}`" for name in profile_list[:10])
        if count > 10:
            msg += f"\n...and {count - 10} more."
        return msg

    def _get_top_models(self, provider: str, target_config_key: str) -> List[str]:
        return self.cog.api_service.get_top_models(provider, target_config_key)
    
    class GenericBulkModelSelect(ui.Select):
        def __init__(self, placeholder: str, options: list, row: int, target_config_key: str):
            super().__init__(placeholder=placeholder, options=options, row=row)
            self.target_config_key = target_config_key

        async def callback(self, interaction: discord.Interaction):
            view = self.view
            if self.values[0] == "ollama_offline":
                await interaction.response.send_message("Ollama is offline or has no models downloaded.", ephemeral=True)
                return
            if self.values[0] == "custom_option":
                await interaction.response.send_modal(CustomModelModal(view, self.target_config_key))
            else: 
                view._save_changes(self.target_config_key, self.values[0])
                view._build_view()
                await interaction.response.edit_message(content=view._get_selection_feedback_message(), view=view)

    def _create_model_options(self, current_val: str, target_config_key: str) -> List[discord.SelectOption]:
        top_models = self._get_top_models(self.view_mode, target_config_key)
        opts = [discord.SelectOption(label="Custom Model...", value="custom_option", description="Enter manually via modal")]
        
        if current_val:
            opts.append(discord.SelectOption(label=f"Current: {current_val}", value=current_val, default=True))
        
        prefix = "GOOGLE/"
        if self.view_mode == 'openrouter': prefix = "OPENROUTER/"
        elif self.view_mode == 'ollama': prefix = "OLLAMA/"
        
        if target_config_key in ['image_generation_model', 'speech_model']:
            prefix = "GOOGLE/"
            
        added = len(opts)
        for m in top_models:
            if added >= 25: break
            val = f"{prefix}{m}"
            if current_val != val:
                opts.append(discord.SelectOption(label=m[:100], value=val))
                added += 1
                
        if self.view_mode == 'ollama' and not top_models:
            opts.append(discord.SelectOption(label="⚠️ Ollama Offline / No Models", value="ollama_offline", description=f"Check {OLLAMA_LOCAL_URL}"))
            
        return opts

    def _build_view(self):
        self.clear_items()
        
        if self.category == 'media':
            self.view_mode = 'google'
            
        if self.category == 'response':
            p_val = self.models_state["primary_model"]
            f_val = self.models_state["fallback_model"]
            self.add_item(self.GenericBulkModelSelect("Select Primary Model...", self._create_model_options(p_val, "primary_model"), 0, "primary_model"))
            self.add_item(self.GenericBulkModelSelect("Select Fallback Model...", self._create_model_options(f_val, "fallback_model"), 1, "fallback_model"))
            
        elif self.category == 'media':
            i_val = self.models_state["image_generation_model"]
            t_val = self.models_state["speech_model"]
            self.add_item(self.GenericBulkModelSelect("Select Image Gen Model...", self._create_model_options(i_val, "image_generation_model"), 0, "image_generation_model"))
            self.add_item(self.GenericBulkModelSelect("Select Text-to-Speech Model...", self._create_model_options(t_val, "speech_model"), 1, "speech_model"))
            
        elif self.category == 'tools':
            g_val = self.models_state["grounding_rag_model"]
            c_val = self.models_state["critic_model"]
            self.add_item(self.GenericBulkModelSelect("Select Grounding RAG Model...", self._create_model_options(g_val, "grounding_rag_model"), 0, "grounding_rag_model"))
            self.add_item(self.GenericBulkModelSelect("Select Critic Model...", self._create_model_options(c_val, "critic_model"), 1, "critic_model"))
            
        elif self.category == 'ltm':
            l_val = self.models_state["ltm_model"]
            self.add_item(self.GenericBulkModelSelect("Select LTM Summariser Model...", self._create_model_options(l_val, "ltm_model"), 0, "ltm_model"))

        active_list = self._get_active_list()
        per_page = 23
        num_pages = max(1, (len(active_list) - 1) // per_page + 1)
        if self.current_page >= num_pages: self.current_page = max(0, num_pages - 1)
        
        start = self.current_page * per_page
        page_profiles = active_list[start : start + per_page]

        api_modes = ['google', 'openrouter', 'ollama']
        api_labels = {'google': 'API: Google', 'openrouter': 'API: OpenRouter', 'ollama': 'API: Ollama (Local)'}
        
        btn_api = ui.Button(label=api_labels[self.view_mode], style=discord.ButtonStyle.primary, row=2, disabled=(self.category == 'media'))
        async def api_cb(i: discord.Interaction):
            next_idx = (api_modes.index(self.view_mode) + 1) % len(api_modes)
            self.view_mode = api_modes[next_idx]
            if self.view_mode == 'ollama':
                await i.response.defer()
                self.ollama_working = "processing"
                await self._update_ollama_status()
                self._build_view()
                await i.edit_original_response(content=self._get_selection_feedback_message(), view=self)
            else:
                self._build_view()
                await i.response.edit_message(content=self._get_selection_feedback_message(), view=self)
        btn_api.callback = api_cb
        self.add_item(btn_api)

        if self.view_mode == 'ollama':
            host_style = discord.ButtonStyle.secondary
            if getattr(self, 'ollama_working', None) == "processing":
                host_style = discord.ButtonStyle.blurple
            elif getattr(self, 'ollama_working', None) is True:
                host_style = discord.ButtonStyle.success
            elif getattr(self, 'ollama_working', None) is False:
                host_style = discord.ButtonStyle.danger
                
            btn_host = ui.Button(label="Set Host URL", style=host_style, row=2)
            async def host_cb(i: discord.Interaction):
                await i.response.send_modal(OllamaHostModal(self))
            btn_host.callback = host_cb
            self.add_item(btn_host)
            
            btn_guide = ui.Button(label="Guide", style=discord.ButtonStyle.secondary, row=2)
            async def guide_cb(i: discord.Interaction):
                await i.response.send_message(OLLAMA_GUIDE_TEXT, ephemeral=True)
            btn_guide.callback = guide_cb
            self.add_item(btn_guide)

        categories = ['response', 'media', 'tools', 'ltm']
        cat_labels = {'response': 'Response', 'media': 'Media', 'tools': 'Tools', 'ltm': 'LTM'}
        btn_cat = ui.Button(label=f"Category: {cat_labels[self.category]}", style=discord.ButtonStyle.blurple, row=2)
        async def cat_cb(i: discord.Interaction):
            next_idx = (categories.index(self.category) + 1) % len(categories)
            self.category = categories[next_idx]
            self._build_view()
            await i.response.edit_message(content=self._get_selection_feedback_message(), view=self)
        btn_cat.callback = cat_cb
        self.add_item(btn_cat)

        if self.category == 'response':
            fb_label = "Fallback Indicator: Unchanged" if self.show_fallback_indicator is None else ("Fallback Indicator: ON" if self.show_fallback_indicator else "Fallback Indicator: OFF")
            fb_style = discord.ButtonStyle.success if self.show_fallback_indicator else (discord.ButtonStyle.secondary if self.show_fallback_indicator is None else discord.ButtonStyle.danger)
            btn_fallback = ui.Button(label=fb_label, style=fb_style, row=2, custom_id="toggle_fallback")
            async def fallback_cb(i: discord.Interaction):
                if self.show_fallback_indicator is None:
                    self.show_fallback_indicator = True
                elif self.show_fallback_indicator:
                    self.show_fallback_indicator = False
                else:
                    self.show_fallback_indicator = None
                self._build_view()
                await i.response.edit_message(content=self._get_selection_feedback_message(), view=self)
            btn_fallback.callback = fallback_cb
            self.add_item(btn_fallback)
        
        options = []
        if page_profiles:
            page_set = set(page_profiles)
            page_selected = page_set.issubset(self.target_profiles)
            page_label = "Unselect Page" if page_selected else "Select Page"
            options.append(discord.SelectOption(label=page_label, value="toggle_page", description="Toggle selection for all profiles on this page.", emoji="📄"))
            
            all_set = set(active_list)
            all_selected = all_set.issubset(self.target_profiles)
            all_label = "Unselect All" if all_selected else "Select All"
            options.append(discord.SelectOption(label=all_label, value="toggle_all", description="Toggle selection for all profiles in this source.", emoji="📚"))

        for name in page_profiles:
            options.append(discord.SelectOption(label=name, value=name, default=(name in self.target_profiles)))
        
        if options:
            profile_select = ui.Select(placeholder=f"Select {self.view_source} profiles...", min_values=0, max_values=len(options), options=options, row=3)
            profile_select.callback = self.profile_callback
            self.add_item(profile_select)

        # Row 4 Pagination & Navigation
        style_src = discord.ButtonStyle.blurple if self.view_source == 'personal' else discord.ButtonStyle.green
        source_btn = ui.Button(label=f"Source: {self.view_source.title()}", style=style_src, row=4)
        source_btn.callback = self.toggle_source_callback
        self.add_item(source_btn)

        build_pagination_controls(self, self.current_page, num_pages, 4, self.prev_page_callback, self.next_page_callback)

        apply_btn = ui.Button(label="Apply", style=discord.ButtonStyle.success, row=4)
        apply_btn.callback = self.apply_settings
        self.add_item(apply_btn)

    async def _update_ollama_status(self):
        host_url = self.models_state.get("ollama_host_url") or OLLAMA_LOCAL_URL
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{host_url.rstrip('/')}/api/tags", timeout=2.0)
                self.ollama_working = (resp.status_code == 200)
                if self.ollama_working:
                    data = resp.json()
                    self.cached_ollama_models = [m['name'] for m in data.get('models', [])]
        except Exception:
            self.ollama_working = False
            self.cached_ollama_models = []

    async def toggle_source_callback(self, interaction: discord.Interaction):
        self.view_source = 'borrowed' if self.view_source == 'personal' else 'personal'
        self.current_page = 0
        self._build_view()
        await interaction.response.edit_message(content=self._get_selection_feedback_message(), view=self)

    async def prev_page_callback(self, interaction: discord.Interaction):
        self.current_page -= 1
        self._build_view()
        await interaction.response.edit_message(view=self)

    async def next_page_callback(self, interaction: discord.Interaction):
        self.current_page += 1
        self._build_view()
        await interaction.response.edit_message(view=self)

    async def profile_callback(self, interaction: discord.Interaction):
        vals = interaction.data.get('values', [])
        
        per_page = 23
        active_list = self._get_active_list()
        start = self.current_page * per_page
        page_profiles = active_list[start : start + per_page]
        
        if "toggle_page" in vals:
            page_set = set(page_profiles)
            if page_set.issubset(self.target_profiles): self.target_profiles.difference_update(page_set)
            else: self.target_profiles.update(page_set)
        elif "toggle_all" in vals:
            all_set = set(active_list)
            if all_set.issubset(self.target_profiles): self.target_profiles.difference_update(all_set)
            else: self.target_profiles.update(all_set)
        else:
            self.target_profiles.difference_update(set(page_profiles))
            self.target_profiles.update(vals)
            
        self._build_view()
        await interaction.response.edit_message(content=self._get_selection_feedback_message(), view=self)

    async def apply_settings(self, interaction: discord.Interaction):
        await interaction.response.defer()
        
        has_any_model = any(v is not None for v in self.models_state.values())
        if not has_any_model and self.show_fallback_indicator is None:
            await interaction.followup.send("Please select at least one setting to change.", ephemeral=True)
            return
            
        if not self.target_profiles:
            await interaction.followup.send("Please select at least one profile.", ephemeral=True)
            return

        success_count = 0
        index = self.cog.profile_manager._get_user_index(self.user_id)
        for profile_name in self.target_profiles:
            is_borrowed = profile_name in index.get("borrowed", [])
            profile = self.cog.profile_manager._get_profile_config(self.user_id, profile_name, is_borrowed)
            if profile:
                for k, v in self.models_state.items():
                    if v is not None:
                        profile[k] = v
                        
                if self.show_fallback_indicator is not None:
                    profile["show_fallback_indicator"] = self.show_fallback_indicator
                    
                self.cog.profile_manager._save_profile_config(self.user_id, profile_name, profile, is_borrowed)
                success_count += 1
                
        if success_count > 0:
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

        msg = f"Updated models for {success_count} profiles." if success_count else "No profiles updated."
        await interaction.edit_original_response(content=msg, view=None)

    async def on_error(self, interaction: discord.Interaction, error: Exception, item: ui.Item):
        print(f"Error in ModelApplyView: {error}")
        traceback.print_exc()
        if not interaction.response.is_done():
            await interaction.response.send_message("An unexpected error occurred with this view.", ephemeral=True)
        else:
            await interaction.followup.send("An unexpected error occurred with this view.", ephemeral=True)

class UnifiedBulkTargetView(BaseBulkProfileView):
    def __init__(self, cog: 'MimicCog', user_id: int, action_key: str, payload: Any, include_borrowed: bool = True):
        super().__init__(cog, user_id, include_borrowed=include_borrowed)
        self.action_key = action_key
        self.payload = payload
        self._build_view()

    def _build_view(self):
        self.clear_items()
        self._build_profile_select_ui(row=1)
        apply_btn = ui.Button(label="Apply Settings", style=discord.ButtonStyle.green, row=3)
        apply_btn.callback = self.apply_action_callback
        self.add_item(apply_btn)

    async def apply_action_callback(self, interaction: discord.Interaction):
        await interaction.response.defer()
        targets = list(self.selected_profiles)
        if not targets:
            await interaction.edit_original_response(content="You must select at least one profile.", view=None)
            return

        success_count = 0
        index = self.cog.profile_manager._get_user_index(self.user_id)
        
        for name in targets:
            is_borrowed = name in index.get("borrowed", [])
            profile = self.cog.profile_manager._get_profile_config(self.user_id, name, is_borrowed)
            
            if profile:
                if self.action_key == "update_config":
                    profile.update(self.payload)
                elif self.action_key == "set_key":
                    k, v = self.payload
                    if v is not None: profile[k] = v
                elif self.action_key == "update_prompts" and not is_borrowed:
                    prompts = self.cog.profile_manager._get_profile_prompts(self.user_id, name)
                    if prompts:
                        prompts.update(self.payload)
                        self.cog.profile_manager._save_profile_prompts(self.user_id, name, prompts)
                elif self.action_key == "update_both":
                    if "config" in self.payload:
                        profile.update(self.payload["config"])
                    if "prompts" in self.payload and not is_borrowed:
                        prompts = self.cog.profile_manager._get_profile_prompts(self.user_id, name)
                        if prompts:
                            prompts.update(self.payload["prompts"])
                            self.cog.profile_manager._save_profile_prompts(self.user_id, name, prompts)
                
                self.cog.profile_manager._save_profile_config(self.user_id, name, profile, is_borrowed)
                success_count += 1
                
        if success_count > 0:
            keys = [k for k in self.cog.channel_models.keys() if isinstance(k, tuple) and len(k) >= 2 and k[1] == self.user_id]
            for k in keys:
                self.cog.channel_models.pop(k, None)
                self.cog.chat_sessions.pop(k, None)
                self.cog.channel_model_last_profile_key.pop(k, None)

        await interaction.edit_original_response(content=f"Successfully applied settings to {success_count} profile(s).", view=None)

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
    def __init__(self, cog: 'MimicCog', user_id: int):
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

        updated_count = 0
        index = self.cog.profile_manager._get_user_index(self.user_id)
        for name in self.selected_profiles:
            is_borrowed = name in index.get("borrowed", [])
            p = self.cog.profile_manager._get_profile_config(self.user_id, name, is_borrowed)
            if p:
                p["timezone"] = self.selected_tz
                p["time_tracking_enabled"] = True # Force always-on
                self.cog.profile_manager._save_profile_config(self.user_id, name, p, is_borrowed)
                updated_count += 1
        
        if updated_count > 0:
            # Flush caches for the user
            keys = [k for k in self.cog.channel_models.keys() if isinstance(k, tuple) and k[1] == self.user_id]
            for k in keys: 
                self.cog.channel_models.pop(k, None)
                self.cog.chat_sessions.pop(k, None)

        await interaction.edit_original_response(content=f"Timezone set to **{self.selected_tz}** for {updated_count} profiles.", view=None)

def ProfileGenerationVisualModal(cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, callback=None):
    raw = current_params.get("placeholder_emoji") or ""
    name_val, id_val = "", ""
    if raw.startswith("<") and raw.endswith(">"):
        parts = raw.strip("<>").split(":")
        if len(parts) >= 3:
            name_val = f"a:{parts[1]}" if parts[0] == "a" else parts[1]
            id_val = parts[2]
    else:
        name_val = raw

    fields = [
        {"label": "Emote Name (or Native Emote)", "custom_id": "name", "default": name_val, "required": False, "max_length": 100, "placeholder": "e.g. mimic_thinking or 🤔"},
        {"label": "Emote ID (Blank if native)", "custom_id": "id", "default": id_val, "required": False, "max_length": 30, "placeholder": "e.g. 1441782350752120874"},
        {"label": "Placeholder for Child Bot (on/off)", "custom_id": "child_bot_placeholder", "default": "on" if current_params.get("child_bot_placeholder") else "off", "required": True, "max_length": 10}
    ]
    def parser(v):
        c = {}
        n = _ps(v["name"]); i = _ps(v["id"])
        p_emoji = ""
        if n:
            if n.startswith("<") and n.endswith(">"): p_emoji = n
            else:
                is_a = False
                if n.startswith("a:"): is_a = True; n = n[2:]
                n = n.strip(":")
                if i: p_emoji = f"<a:{n}:{i}>" if is_a else f"<:{n}:{i}>"
                else: p_emoji = n
        c["placeholder_emoji"] = p_emoji if p_emoji else None
        c["child_bot_placeholder"] = _pb(v["child_bot_placeholder"])
        return {"config": c}
    return ConfigModal(cog, profile_name, is_borrowed, "Generation Visual", fields, parser, callback)

class BulkResetView(BaseBulkProfileView):
    def __init__(self, cog: 'MimicCog', user_id: int):
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
            final_message = await self.cog.memory_manager.bulk_reset_examples(self.user_id, target_profiles)
        elif self.reset_choice == "reset_ltm":
            final_message = await self.cog.memory_manager.bulk_reset_ltm(self.user_id, target_profiles)
        
        await interaction.edit_original_response(content=final_message, view=None)

class BulkDeleteView(BaseBulkProfileView):
    def __init__(self, cog: 'MimicCog', user_id: int):
        super().__init__(cog, user_id, include_borrowed=True)
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
        index = self.cog.profile_manager._get_user_index(self.user_id)
        
        for name in items_to_delete:
            if name in index.get("borrowed", {}):
                if isinstance(index["borrowed"], dict):
                    pid = index["borrowed"].pop(name)
                else:
                    index["borrowed"].remove(name)
                    pid = name
                deleted_count += 1
                import shutil
                p_dir = os.path.join(self.cog.USERS_DIR, user_id_str, "profiles", pid)
                shutil.rmtree(p_dir, ignore_errors=True)
            elif name in index.get("personal", {}):
                if isinstance(index["personal"], dict):
                    pid = index["personal"].pop(name)
                else:
                    index["personal"].remove(name)
                    pid = name
                
                # Pass the pid to cascade delete to match the method signature
                self.cog.profile_manager._cascade_delete_borrowed_profiles(self.user_id, pid, name)
                deleted_count += 1
                import shutil
                p_dir = os.path.join(self.cog.USERS_DIR, user_id_str, "profiles", pid)
                shutil.rmtree(p_dir, ignore_errors=True)
        
        if deleted_count > 0: self.cog.profile_manager._save_user_index(self.user_id, index)
        await interaction.edit_original_response(content=f"Successfully deleted {deleted_count} profiles.", view=None)

class AppearanceModal(ui.Modal):
    def __init__(self, cog: 'MimicCog', original_interaction: discord.Interaction, profile_name: str):
        super().__init__(title=f"Appearance: '{profile_name[:20]}'")
        self.cog = cog
        self.original_interaction = original_interaction
        self.profile_name = profile_name
        
        user_id_str = str(original_interaction.user.id)
        current_data = self.cog.user_appearances.get(user_id_str, {}).get(self.profile_name, {})
        
        self.display_name_input = ui.TextInput(label="Custom Display Name (Blank to reset)", required=False, max_length=20, default=current_data.get("custom_display_name"))
        self.avatar_url_input = ui.TextInput(label="Avatar URL (Blank to reset)", required=False, default=current_data.get("custom_avatar_url"))
        self.add_item(self.display_name_input)
        self.add_item(self.avatar_url_input)

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer()
        new_display_name = self.display_name_input.value.strip() or None
        new_avatar_url = self.avatar_url_input.value.strip() or None
        user_id_str = str(interaction.user.id)

        if new_display_name:
            if len(new_display_name) > 32:
                await interaction.followup.send("❌ **Invalid Display Name:** Must be 32 characters or fewer.", ephemeral=True)
                return
            if any(r in new_display_name.lower() for r in ["clyde", "@everyone", "@here"]):
                await interaction.followup.send("❌ **Invalid Display Name:** Contains a reserved keyword or mention.", ephemeral=True)
                return

        is_public = self.cog.profile_manager._is_profile_public(interaction.user.id, self.profile_name)
        if is_public and (new_display_name or new_avatar_url):
            is_safe, reason = await self.cog.profile_manager._is_profile_content_safe(interaction.user.id, self.profile_name, new_display_name or self.profile_name, new_avatar_url)
            if not is_safe:
                await interaction.followup.send(f"**Safety Block:** {reason}", ephemeral=True)
                return

        config = self.cog.profile_manager._get_profile_config(interaction.user.id, self.profile_name, False)
        if config:
            if new_display_name: config["custom_display_name"] = new_display_name
            else: config.pop("custom_display_name", None)
            
            if new_avatar_url: config["custom_avatar_url"] = new_avatar_url
            else: config.pop("custom_avatar_url", None)
            
            self.cog.profile_manager._save_profile_config(interaction.user.id, self.profile_name, config, False)
            
            if new_display_name or new_avatar_url:
                self.cog.user_appearances.setdefault(user_id_str, {})[self.profile_name] = {
                    "custom_display_name": new_display_name,
                    "custom_avatar_url": new_avatar_url
                }
            else:
                if user_id_str in self.cog.user_appearances:
                    self.cog.user_appearances[user_id_str].pop(self.profile_name, None)

        linked_bot_id = next((bot_id for bot_id, data in self.cog.child_bots.items() if str(data.get("owner_id")) == user_id_str and data.get("profile_name") == self.profile_name), None)
        if linked_bot_id:
            now = time.time()
            cooldown_window = 600
            max_changes = 2
            timestamps = self.cog.child_bot_edit_cooldowns.get(linked_bot_id, [])
            valid_timestamps = [ts for ts in timestamps if now - ts < cooldown_window]

            if len(valid_timestamps) >= max_changes:
                remaining = int(cooldown_window - (now - valid_timestamps[0]))
                await interaction.followup.send(f"Child bot appearance changed too frequently. Wait {remaining // 60}m.", ephemeral=True)
            else:
                await self.cog.manager_queue.put({"action": "send_to_child", "bot_id": linked_bot_id, "payload": {"action": "update_avatar", "avatar_url": new_avatar_url}})
                await self.cog.manager_queue.put({"action": "send_to_child", "bot_id": linked_bot_id, "payload": {"action": "update_username", "username": new_display_name}})
                valid_timestamps.append(now)
                self.cog.child_bot_edit_cooldowns[linked_bot_id] = valid_timestamps

        new_embed = await self.cog.profile_manager._build_profile_manage_embed(self.original_interaction, self.profile_name)
        await self.original_interaction.edit_original_response(embed=new_embed)
        await interaction.followup.send("Appearance updated.", ephemeral=True)

def ProfileNeuroModal(cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, callback=None):
    state = current_params.get("neuro_state", {"dopamine": 50, "cortisol": 20, "oxytocin": 50, "adrenaline": 20})
    fields = [
        {"label": "Engine Status (on/off)", "custom_id": "neuro_engine_enabled", "default": "on" if current_params.get("neuro_engine_enabled") else "off", "required": False, "placeholder": "Enable or disable the engine."},
        {"label": "Dopamine (0-100)", "custom_id": "dopamine", "default": str(state.get("dopamine", 50)), "required": False, "placeholder": "Motivation and joy."},
        {"label": "Cortisol (0-100)", "custom_id": "cortisol", "default": str(state.get("cortisol", 20)), "required": False, "placeholder": "Stress and anxiety."},
        {"label": "Oxytocin (0-100)", "custom_id": "oxytocin", "default": str(state.get("oxytocin", 50)), "required": False, "placeholder": "Bonding and trust."},
        {"label": "Adrenaline (0-100)", "custom_id": "adrenaline", "default": str(state.get("adrenaline", 20)), "required": False, "placeholder": "Energy and urgency."}
    ]
    def parser(v):
        c = {}
        ns = _ps(v["neuro_engine_enabled"])
        if ns: c["neuro_engine_enabled"] = (ns.lower() == "on")
        
        nstate = {}
        for k in ["dopamine", "cortisol", "oxytocin", "adrenaline"]:
            val = _pi(v[k])
            if val is not None:
                if not (0 <= val <= 100): raise ValueError(f"{k} out of range")
                nstate[k] = val
        if nstate: c["neuro_state"] = nstate
        return {"config": c}
    return ConfigModal(cog, profile_name, is_borrowed, "Neuro-Endocrine Engine Configuration", fields, parser, callback)

class BulkManageView(ui.View):
    def __init__(self, cog: 'MimicCog', original_interaction: discord.Interaction):
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
            discord.SelectOption(label="Configure Image Generation", value="image_gen", description="Setup models, prompts, and toggles for multiple profiles."),
            discord.SelectOption(label="Toggle URL Context Fetching", value="url_context", description="Enable or disable link scraping for multiple profiles."),
            discord.SelectOption(label="Set Time & Timezone", value="timezone", description="Enable time-awareness and set a specific timezone."),
            discord.SelectOption(label="Set Generation Visual", value="generation_visual", description="Apply custom placeholder emoji to multiple profiles."),
            discord.SelectOption(label="Toggle Critic (Anti-Repetition)", value="critic", description="Enable or disable the critic for multiple profiles."),
            discord.SelectOption(label="Toggle Neuro-Endocrine Engine", value="neuro", description="Enable or disable hormonal simulation for multiple profiles."),
            discord.SelectOption(label="Toggle Help Mode (Guide RAG)", value="help_mode", description="Allow profiles to answer technical bot questions."),
            discord.SelectOption(label="Toggle Realistic Typing", value="typing", description="Enable or disable realistic typing for multiple profiles."),
            discord.SelectOption(label="Set Safety Level", value="safety_level", description="Apply a content safety level to multiple profiles."),
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
        index = self.cog.profile_manager._get_user_index(self.user_id)
        all_profiles = list(index.get("personal", [])) + list(index.get("borrowed", []))
        
        if not all_profiles:
            await interaction.response.send_message("You have no profiles to apply settings to.", ephemeral=True)
            return

        if choice == "gen_params":
            async def modal_callback(i: discord.Interaction, params: Dict):
                view = UnifiedBulkTargetView(self.cog, self.user_id, "update_config", params.get("config", {}), include_borrowed=True)
                await i.followup.send(content="Parameters validated. Select the profiles to apply them to:", view=view, ephemeral=True)
            modal = ProfileParamsModal(self.cog, "BULK_APPLY", {}, False, callback=modal_callback)
            await interaction.response.send_modal(modal)

        elif choice == "adv_params":
            async def modal_callback(i: discord.Interaction, params: Dict):
                if not params: await i.followup.send("No parameters set.", ephemeral=True); return
                view = UnifiedBulkTargetView(self.cog, self.user_id, "update_config", params.get("config", {}), include_borrowed=True)
                await i.followup.send(content="Advanced parameters validated. Select the profiles to apply them to:", view=view, ephemeral=True)
            modal = ProfileAdvancedParamsModal(self.cog, "BULK_APPLY", {}, False, callback=modal_callback)
            await interaction.response.send_modal(modal)

        elif choice == "thinking_params":
            async def modal_callback(i: discord.Interaction, params: Dict):
                view = UnifiedBulkTargetView(self.cog, self.user_id, "update_config", params.get("config", {}), include_borrowed=True)
                await i.followup.send(content="Thinking parameters validated. Select the profiles to apply them to:", view=view, ephemeral=True)
            modal = ProfileThinkingParamsModal(self.cog, "BULK_APPLY", {}, False, callback=modal_callback)
            await interaction.response.send_modal(modal)

        elif choice == "train_params":
            async def modal_callback(i: discord.Interaction, params: Dict):
                view = UnifiedBulkTargetView(self.cog, self.user_id, "update_config", params.get("config", {}), include_borrowed=False)
                await i.followup.send(content="Parameters validated. Select the profiles to apply them to:", view=view, ephemeral=True)
            modal = ProfileTrainingParamsModal(self.cog, "BULK_APPLY", {}, callback=modal_callback)
            await interaction.response.send_modal(modal)

        elif choice == "ltm_params":
            async def modal_callback(i: discord.Interaction, params: Dict):
                view = UnifiedBulkTargetView(self.cog, self.user_id, "update_config", params.get("config", {}), include_borrowed=True)
                await i.followup.send(content="Parameters validated. Select the profiles to apply them to:", view=view, ephemeral=True)
            modal = ProfileLTMParamsModal(self.cog, "BULK_APPLY", {}, callback=modal_callback)
            await interaction.response.send_modal(modal)

        elif choice == "ltm_summarization":
            async def modal_callback(i: discord.Interaction, params: Dict):
                view = UnifiedBulkTargetView(self.cog, self.user_id, "update_prompts", params.get("prompts", {}), include_borrowed=False)
                await i.followup.send(content="Prompt received. Now select the profiles to apply it to:", view=view, ephemeral=True)
            modal = ProfileLTMSummarizationModal(self.cog, "BULK_APPLY", DEFAULT_LTM_SUMMARIZATION_INSTRUCTIONS, callback=modal_callback)
            await interaction.response.send_modal(modal)

        elif choice == "models":
            view = ModelApplyView(self.cog, self.user_id, self.original_interaction)
            await interaction.response.send_message(content="Select models and profiles:", view=view, ephemeral=True)

        elif choice == "image_gen":
            async def modal_callback(i: discord.Interaction, params: Dict):
                view = UnifiedBulkTargetView(self.cog, self.user_id, "update_both", params, include_borrowed=True)
                await i.followup.send(content="Image settings validated. Select the profiles to apply them to:", view=view, ephemeral=True)
            modal = ProfileImageGenSettingsModal(self.cog, "BULK_APPLY", {}, False, callback=modal_callback)
            await interaction.response.send_modal(modal)
            
        elif choice == "generation_visual":
            async def modal_callback(i: discord.Interaction, params: Dict):
                view = UnifiedBulkTargetView(self.cog, self.user_id, "update_config", params.get("config", {}), include_borrowed=True)
                await i.followup.send(content="Visual settings validated. Select the profiles to apply them to:", view=view, ephemeral=True)
            modal = ProfileGenerationVisualModal(self.cog, "BULK_APPLY", {}, False, callback=modal_callback)
            await interaction.response.send_modal(modal)

        elif choice == "neuro":
            async def modal_callback(i: discord.Interaction, params: Dict):
                view = UnifiedBulkTargetView(self.cog, self.user_id, "update_config", params.get("config", {}), include_borrowed=True)
                await i.followup.send(content="Neuro settings validated. Select the profiles to apply them to:", view=view, ephemeral=True)
            modal = ProfileNeuroModal(self.cog, "BULK_APPLY", {}, False, callback=modal_callback)
            await interaction.response.send_modal(modal)

        elif choice == "typing":
            async def modal_callback(i: discord.Interaction, params: Dict):
                view = UnifiedBulkTargetView(self.cog, self.user_id, "update_config", params.get("config", {}), include_borrowed=True)
                await i.followup.send(content="Typing settings validated. Select the profiles to apply them to:", view=view, ephemeral=True)
            modal = ProfileTypingSettingsModal(self.cog, "BULK_APPLY", {}, False, callback=modal_callback)
            await interaction.response.send_modal(modal)

        elif choice == "grounding":
            opts = [discord.SelectOption(label=l, value=v) for l,v in [("Off", "off"), ("Native", "native"), ("RAG", "rag")]]
            view = UnifiedBulkTargetView(self.cog, self.user_id, "set_key", ("grounding_mode", None), include_borrowed=True)
            sel = ui.Select(placeholder="Select Grounding Mode...", options=opts, row=0)
            async def sel_cb(inter): view.payload = ("grounding_mode", sel.values[0]); await inter.response.defer()
            sel.callback = sel_cb
            view.add_item(sel)
            await interaction.response.send_message(content="Select mode and profiles:", view=view, ephemeral=True)

        elif choice == "response_mode":
            opts = [discord.SelectOption(label=l, value=v) for l,v in [("Regular", "regular"), ("Mention", "mention"), ("Reply", "reply"), ("Mention+Reply", "mention_reply")]]
            view = UnifiedBulkTargetView(self.cog, self.user_id, "set_key", ("response_mode", None), include_borrowed=True)
            sel = ui.Select(placeholder="Select Response Mode...", options=opts, row=0)
            async def sel_cb(inter): view.payload = ("response_mode", sel.values[0]); await inter.response.defer()
            sel.callback = sel_cb
            view.add_item(sel)
            await interaction.response.send_message(content="Select mode and profiles:", view=view, ephemeral=True)

        elif choice == "url_context":
            opts = [discord.SelectOption(label=l, value=v) for l,v in [("Off", "off"), ("Native", "native"), ("RAG", "rag")]]
            view = UnifiedBulkTargetView(self.cog, self.user_id, "set_key", ("url_mode", None), include_borrowed=True)
            sel = ui.Select(placeholder="Select URL Mode...", options=opts, row=0)
            async def sel_cb(inter): 
                view.action_key = "update_config"
                view.payload = {"url_mode": sel.values[0], "url_fetching_enabled": sel.values[0] == "rag"}
                await inter.response.defer()
            sel.callback = sel_cb
            view.add_item(sel)
            await interaction.response.send_message(content="Select mode and profiles:", view=view, ephemeral=True)

        elif choice == "critic":
            opts = [discord.SelectOption(label="Enable Critic", value="true"), discord.SelectOption(label="Disable Critic", value="false")]
            view = UnifiedBulkTargetView(self.cog, self.user_id, "set_key", ("critic_enabled", False), include_borrowed=True)
            sel = ui.Select(placeholder="Select action...", options=opts, row=0)
            async def sel_cb(inter): view.payload = ("critic_enabled", sel.values[0] == "true"); await inter.response.defer()
            sel.callback = sel_cb
            view.add_item(sel)
            await interaction.response.send_message(content="Select action and profiles:", view=view, ephemeral=True)
            
        elif choice == "help_mode":
            opts = [discord.SelectOption(label="Enable Help Mode", value="true"), discord.SelectOption(label="Disable Help Mode", value="false")]
            view = UnifiedBulkTargetView(self.cog, self.user_id, "set_key", ("help_mode_enabled", False), include_borrowed=True)
            sel = ui.Select(placeholder="Select action...", options=opts, row=0)
            async def sel_cb(inter): view.payload = ("help_mode_enabled", sel.values[0] == "true"); await inter.response.defer()
            sel.callback = sel_cb
            view.add_item(sel)
            await interaction.response.send_message(content="Select action and profiles:", view=view, ephemeral=True)
            
        elif choice == "safety_level":
            opts = [discord.SelectOption(label=l, value=v) for l,v in [("Unrestricted", "unrestricted"), ("Low", "low"), ("Medium", "medium"), ("High", "high")]]
            view = UnifiedBulkTargetView(self.cog, self.user_id, "set_key", ("safety_level", None), include_borrowed=True)
            sel = ui.Select(placeholder="Select safety level...", options=opts, row=0)
            async def sel_cb(inter): view.payload = ("safety_level", sel.values[0]); await inter.response.defer()
            sel.callback = sel_cb
            view.add_item(sel)
            await interaction.response.send_message(content="Select safety level and profiles:", view=view, ephemeral=True)

        elif choice == "timezone":
            view = BulkTimezoneView(self.cog, self.user_id)
            await interaction.response.send_message(content="Select a timezone and the profiles to apply it to:", view=view, ephemeral=True)

        elif choice == "data_reset":
            view = BulkResetView(self.cog, self.user_id)
            await interaction.response.send_message(content="Select reset action:", view=view, ephemeral=True)

        elif choice == "delete_items":
            view = BulkDeleteView(self.cog, self.user_id)
            await interaction.response.send_message(content="Select profiles to delete:", view=view, ephemeral=True)
