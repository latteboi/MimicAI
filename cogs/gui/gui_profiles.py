from ..utils.constants import *

import discord
from discord import ui
import asyncio
import datetime
import traceback
import time
from zoneinfo import ZoneInfo
from typing import TYPE_CHECKING, List, Dict, Set, Any, Optional, Union
from ..utils.content import OLLAMA_GUIDE_TEXT
from ..utils.helpers import _pf, _pi, _ps, _pb
from ..utils.http_client import get_shared_client

if TYPE_CHECKING:
    # This only runs during "hinting" and prevents the circular crash
    from ..MimicCog import MimicCog

from .base_components import BaseBulkProfileView, ConfigModal, ActionTextInputModal, build_pagination_controls, build_confirm_view
from .gui_data import DataManageView
from .gui_hub import HubShareManagerView
from .gui_sessions import CustomModelModal
from .gui_settings import OllamaHostModal

def ProfileAdvancedParamsModal(cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, callback=None, target_user_id: Optional[int] = None):
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
    return ConfigModal(cog, profile_name, is_borrowed, "Advanced Parameters (OpenRouter)", fields, parser, callback, target_user_id)

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


def ProfileSpeechSettingsModal(cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, callback=None, target_user_id: Optional[int] = None):
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
    return ConfigModal(cog, profile_name, is_borrowed, "Speech & Voice Settings", fields, parser, callback, target_user_id)

#: Tab order for ProfileManageView's nav bar. "persona" is hidden for borrowed profiles.
PROFILE_TABS = ("home", "persona", "params", "tools", "memory")


class _Action:
    """One row of ProfileManageView's dropdown.

    Label, description, visibility and handler used to live ~200 lines apart: the
    SelectOption was built in a 135-line `_build_view`, and its behaviour sat in a
    51-branch `elif` chain in `dropdown_callback`. Adding an action meant editing both,
    and reordering one silently desynchronised them. Here a row is one declaration.

    `gate` is an optional predicate on the view; a row with no gate is always shown.
    `label` may be a callable for the rows whose wording depends on the profile.

    `menu_label` is the wording used by the documentation menu map, which is rendered
    from this table with no view to evaluate a callable `label` against. Required only
    for the rows whose `label` is callable; everywhere else the static label is used.
    """

    __slots__ = ("value", "tab", "label", "description", "gate", "run", "_menu_label")

    def __init__(self, value, tab, label, description, run, gate=None, menu_label=None):
        self.value = value
        self.tab = tab
        self.label = label
        self.description = description
        self.run = run
        self.gate = gate
        self._menu_label = menu_label

    @property
    def menu_label(self) -> str:
        """View-independent wording for the generated menu map."""
        if self._menu_label:
            return self._menu_label
        if callable(self.label):
            # A callable label with no menu_label is a bug in the table, but the map
            # is documentation -- degrade to the action key rather than raising.
            return self.value.replace("_", " ").title()
        return self.label

    def visible(self, view) -> bool:
        return self.gate is None or self.gate(view)

    def option(self, view) -> discord.SelectOption:
        label = self.label(view) if callable(self.label) else self.label
        return discord.SelectOption(label=label, value=self.value, description=self.description)


def _modal(factory_name: str, *, pass_borrowed: bool = True):
    """Handler for the dominant shape: open a settings modal, refresh the dashboard.

    The modal factories all take (cog, profile_name, current_params[, is_borrowed],
    callback, target_user_id); the two that predate `is_borrowed` opt out.

    Named rather than passed by reference because the table is defined above the
    factories it points at, so the lookup has to happen at click time.
    """
    async def run(view, interaction, profile):
        args = [view.cog, view.profile_name, profile]
        if pass_borrowed:
            args.append(view.is_borrowed)
        await interaction.response.send_modal(
            globals()[factory_name](*args, callback=view._refresh_dashboard,
                                    target_user_id=view.user_id))
    return run


def _toggle(key: str):
    """Handler for a plain boolean flag on the profile config."""
    async def run(view, interaction, profile):
        profile[key] = not profile.get(key, False)
        await view._save_and_refresh(interaction, profile, view.profile_name, view.is_borrowed)
    return run


def _cycle(key: str, order: tuple, default=None):
    """Handler for a setting that advances through a fixed sequence."""
    async def run(view, interaction, profile):
        current = profile.get(key, default if default is not None else order[0])
        index = order.index(current) + 1 if current in order else 0
        profile[key] = order[index % len(order)]
        await view._save_and_refresh(interaction, profile, view.profile_name, view.is_borrowed)
    return run


def _method(name: str, *args, wants_profile: bool = False, wants_borrowed: bool = False):
    """Handler that defers to one of the view's own `_handle_*` / `_act_*` methods.

    `wants_profile` / `wants_borrowed` append the view state those older handlers take
    positionally, ahead of any literal `args` declared in the table.
    """
    async def run(view, interaction, profile):
        extra = []
        if wants_profile:
            extra.append(profile)
        if wants_borrowed:
            extra.append(view.is_borrowed)
        await getattr(view, name)(interaction, *extra, *args)
    return run


# Gates. Named rather than inlined as lambdas so the table below reads as data.
def _own(view):        return not view.is_borrowed
def _own_not_mod(view): return not view.is_borrowed and not view.is_mod_view
def _to_system(view):
    return (not view.is_borrowed and not view.is_mod_view
            and view.original_interaction.user.id == int(defaultConfig.DISCORD_OWNER_ID)
            and not view.is_system)
def _to_personal(view):
    return (not view.is_borrowed and not view.is_mod_view
            and view.original_interaction.user.id == int(defaultConfig.DISCORD_OWNER_ID)
            and view.is_system)


#: The dropdown, in render order, grouped by tab. Order within a tab is the order the
#: user sees, so rows must not be resorted.
PROFILE_ACTIONS = (
    # --- Home ---
    _Action("rename", "home", "Rename Profile", "Change the local name of this profile.",
            _method("_handle_rename")),
    _Action("duplicate", "home", "Duplicate Profile", "Create a new profile from a copy of this one.",
            _method("_handle_duplicate"), _own),
    # Share and Copy-to-System act on behalf of the profile's owner -- _handle_share
    # opens the invoker's own share manager and _handle_convert_copy reads i.user.id --
    # so both stay owner-only rather than being silently wrong under /mod.
    _Action("share", "home", "Share Profile", "Share this profile with others or publish it.",
            _method("_handle_share"), _own_not_mod),
    _Action("error_response", "home", "Custom Error Message", "Set the message shown when generation fails.",
            _method("_act_error_response", wants_profile=True), _own),
    _Action("generation_visual", "home", "Generation Visual", "Set custom placeholder emoji and child bot behavior.",
            _modal("ProfileGenerationVisualModal"), _own),
    _Action("convert_to_system", "home", "Copy to System Profile", "Create a global System Profile copy from this profile.",
            _method("_handle_convert_copy", True), _to_system),
    _Action("convert_to_personal", "home", "Copy to Personal Profile", "Create a Personal Profile copy from this System Profile.",
            _method("_handle_convert_copy", False), _to_personal),
    # One row where there were three. Declaring 18+, clearing a verdict and granting
    # an exemption are all answers to "why can this profile not do X", and as bare
    # dropdown rows none of them said what X was -- a user refused a publish had to
    # guess which applied to them. They now live inside the Content Safety dashboard,
    # next to the rating that explains them.
    _Action("content_safety", "home", "Content Safety",
            "View this profile's content rating and what it allows.",
            _method("_handle_content_safety")),
    _Action("delete", "home",
            lambda v: "Remove Borrowed Profile" if v.is_borrowed else "Delete Profile",
            "Permanently remove this profile and its data.",
            _method("_handle_delete"),
            menu_label="Delete Profile / Remove Borrowed Profile"),

    # --- Persona (tab hidden entirely for borrowed profiles) ---
    _Action("edit_persona", "persona", "Edit Persona", "Edit backstory, traits, likes, dislikes, and appearance.",
            _method("_act_edit_persona", wants_profile=True)),
    _Action("edit_instructions", "persona", "Edit Instructions", "Edit specific AI behavioral instructions.",
            _method("_act_edit_instructions", wants_profile=True)),
    _Action("tts_instructions", "persona", "TTS Instructions", "Configure the 'Director's Desk' for vocal performance.",
            _modal("ProfileDirectorDeskModal", pass_borrowed=False)),
    _Action("edit_appearance", "persona", "Edit Appearance", "Edit the custom Webhook name and avatar.",
            _method("_handle_appearance"), _own),

    # --- Params ---
    _Action("models", "params", "Set Models", "Choose Primary and Fallback AI models.",
            _method("_act_models", wants_profile=True)),
    _Action("gen_params", "params", "Set Generation Parameters & STM", "Set Temp, Top P, Top K, and STM Length.",
            _modal("ProfileParamsModal")),
    _Action("adv_params", "params", "Set Advanced Parameters (OPENROUTER)", "Set penalties, Min P, and Top A.",
            _modal("ProfileAdvancedParamsModal")),
    _Action("thinking_params", "params", "Set Thinking Parameters", "Set thinking persistence, level, and budget.",
            _modal("ProfileThinkingParamsModal")),
    _Action("speech_settings", "params", "Set Speech & Voice Settings", "Set TTS voice, model, and temperature.",
            _modal("ProfileSpeechSettingsModal")),

    # --- Tools ---
    _Action("image_toggle", "tools", "Toggle Image Generation", "Allow this profile to generate images via !image/!imagine.",
            _method("_act_image_toggle", wants_profile=True)),
    _Action("grounding", "tools", "Toggle Grounding (Web Search)", "Cycle Grounding: OFF -> NATIVE -> RAG.",
            _method("_act_grounding", wants_profile=True)),
    _Action("url_toggle", "tools", "Toggle URL Context Fetching", "Cycle URL Context: OFF -> NATIVE -> RAG.",
            _method("_act_url_toggle", wants_profile=True)),
    _Action("cycle_response", "tools", "Cycle Response Mode", "Cycle: Regular -> Mention -> Reply -> Mention Reply.",
            _cycle("response_mode", ("regular", "mention", "reply", "mention_reply"))),
    _Action("time", "tools", "Set Time & Timezone", "Enable time awareness and set the profile's timezone.",
            _method("_handle_timezone", wants_profile=True, wants_borrowed=True)),
    _Action("typing", "tools", "Toggle Realistic Typing", "Enable a human-like delay when the bot sends messages.",
            _modal("ProfileTypingSettingsModal")),
    _Action("critic", "tools", "Toggle Anti-Repetition Critic", "Enable semantic repetition analysis (Adds latency).",
            _toggle("critic_enabled")),
    _Action("neuro", "tools", "Toggle Neuro-Endocrine Engine", "Simulate hormonal states for dynamic emotions.",
            _modal("ProfileNeuroModal")),
    _Action("help_mode", "tools", "Toggle Help Mode (Guide RAG)", "Allow profile to answer technical bot questions.",
            _toggle("help_mode_enabled")),

    # --- Memory ---
    _Action("manage_ltm", "memory", "Manage Long-Term Memories", "Add, list, edit, or delete memories.",
            _method("_act_manage_ltm", wants_profile=True)),
    _Action("manage_training", "memory", "Manage Training Examples", "Add, list, edit, or delete training examples.",
            _method("_act_manage_training", wants_profile=True), _own),
    _Action("train_params", "memory", "Set Training Parameters", "Set training context size and relevance threshold.",
            _modal("ProfileTrainingParamsModal", pass_borrowed=False), _own),
    _Action("ltm_creation", "memory", "Toggle LTM Auto-Creation", "Automatically create memories from conversations.",
            _toggle("ltm_creation_enabled")),
    _Action("ltm_params", "memory", "Set LTM Parameters", "Set frequency, context, and recall settings.",
            _modal("ProfileLTMParamsModal", pass_borrowed=False)),
    _Action("ltm_summarization", "memory", "Set LTM Summarization Prompt", "Customize how the AI creates memories.",
            _method("_act_ltm_summarization", wants_profile=True), _own),
)

PROFILE_ACTIONS_BY_VALUE = {a.value: a for a in PROFILE_ACTIONS}


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
        # Whose profile list the mod-mode Back button returns to. Kept separate from
        # self.user_id because a system profile rewrites that to the bot owner's id.
        self.mod_return_user_id = target_user_id
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

        valid_tabs = [t for t in PROFILE_TABS if t == "home" or t != "persona" or not self.is_borrowed]
        if self.current_tab not in valid_tabs and valid_tabs:
            self.current_tab = valid_tabs[0]

        # --- 1. Category Dropdown (Row 0) ---
        options = [a.option(self) for a in PROFILE_ACTIONS
                   if a.tab == self.current_tab and a.visible(self)]

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
            self._add_mod_back_button()
            ModBaseView.add_nav_to_other_view(
                self, self.cog, self.original_interaction, "profiles", self.mod_return_user_id)

    def _add_mod_back_button(self):
        """Back to the moderated user's profile list.

        The mod nav bar's own "Profiles" button is disabled here -- build_tab_nav_bar
        disables the button for the current tab -- so without this there is no way
        back to the list at all, and going out via another tab dropped the target
        user id.
        """
        from .gui_mod import ModProfilesView

        btn = ui.Button(label="← Back to Profile List", style=discord.ButtonStyle.secondary, row=2)

        async def back_cb(interaction: discord.Interaction):
            await interaction.response.defer()
            view = ModProfilesView(self.cog, self.original_interaction,
                                   target_user_id=self.mod_return_user_id)
            await view.update_display()

        btn.callback = back_cb
        self.add_item(btn)

    def create_nav_callback(self, tab_name):
        async def callback(interaction: discord.Interaction):
            self.current_tab = tab_name
            self._build_view()
            await interaction.response.edit_message(view=self)
        return callback

    async def _refresh_dashboard(self, interaction: discord.Interaction):
        """Repaint the dashboard embed on the message the view owns.

        Every settings modal hands this back as its completion callback. It used to be
        redefined as an identical four-line `refresh_cb` closure inside eleven separate
        dropdown branches.
        """
        new_embed = await self.cog.profile_manager._build_profile_manage_embed(
            interaction, self.profile_name, target_user_id=self.user_id)
        await self.original_interaction.edit_original_response(embed=new_embed, view=self)

    async def dropdown_callback(self, interaction: discord.Interaction):
        choice = interaction.data['values'][0]
        profile = self.cog.profile_manager._get_profile_config(
            self.user_id, self.profile_name, self.is_borrowed)

        if not profile:
            await interaction.response.send_message("Profile data not found.", ephemeral=True); return

        action = PROFILE_ACTIONS_BY_VALUE.get(choice)
        if action is None:
            return
        await action.run(self, interaction, profile)

    # --- Bespoke dropdown handlers ---
    #
    # Everything the action table cannot express as "open this modal", "flip this flag"
    # or "advance this cycle" lives here as a named method, so the table stays a table.

    async def _act_error_response(self, interaction: discord.Interaction, profile: Dict[str, Any]):
        is_b = getattr(self, "is_borrowed", False)
        target_profile = self.cog.profile_manager._get_profile_config(self.user_id, self.profile_name, is_b)

        if not target_profile:
            await interaction.response.send_message("❌ Error: Profile not found.", ephemeral=True)
            return

        async def modal_callback(modal_interaction: discord.Interaction, new_val: str):
            await modal_interaction.response.defer(ephemeral=True)
            val_to_save = new_val.strip() or "An error has occurred."

            target = self.cog.profile_manager._get_profile_config(self.user_id, self.profile_name, is_b)

            if target:
                target["error_response"] = val_to_save
                self.cog.profile_manager._save_profile_config(self.user_id, self.profile_name, target, is_b)
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

    async def _act_edit_persona(self, interaction: discord.Interaction, profile: Dict[str, Any]):
        prompts = self.cog.profile_manager._get_profile_prompts(self.user_id, self.profile_name) or {}
        modal = EditUserProfilePersonaModal(self.cog, self.profile_name, prompts.get("persona", {}), self.user_id)
        await interaction.response.send_modal(modal)

    async def _act_edit_instructions(self, interaction: discord.Interaction, profile: Dict[str, Any]):
        prompts = self.cog.profile_manager._get_profile_prompts(self.user_id, self.profile_name) or {}
        modal = EditUserProfileAIInstructionsModal(self.cog, self.profile_name, prompts.get("ai_instructions", ""), self.user_id)
        await interaction.response.send_modal(modal)

    async def _act_models(self, interaction: discord.Interaction, profile: Dict[str, Any]):
        view = SingleProfileModelView(self.cog, self.original_interaction, self.profile_name,
                                      is_borrowed=self.is_borrowed, user_id=self.user_id)
        await interaction.response.send_message(view._get_selection_feedback_message(), view=view, ephemeral=True)

    async def _act_image_toggle(self, interaction: discord.Interaction, profile: Dict[str, Any]):
        # Inject prompt into current_params to avoid breaking the modal signature
        if not self.is_borrowed:
            prompts = self.cog.profile_manager._get_profile_prompts(self.user_id, self.profile_name) or {}
            profile["image_generation_prompt"] = prompts.get("image_generation_prompt")

        modal = ProfileImageGenSettingsModal(self.cog, self.profile_name, profile, self.is_borrowed,
                                             callback=self._refresh_dashboard, target_user_id=self.user_id)
        await interaction.response.send_modal(modal)

    async def _act_grounding(self, interaction: discord.Interaction, profile: Dict[str, Any]):
        current_mode = profile.get("grounding_mode", "off")
        if isinstance(current_mode, bool): current_mode = "rag" if current_mode else "off"
        elif current_mode == "on" or current_mode == "on+": current_mode = "rag" # Legacy migration
        cycle_map = {"off": "native", "native": "rag", "rag": "off"}
        profile["grounding_mode"] = cycle_map.get(current_mode, "off")
        await self._save_and_refresh(interaction, profile, self.profile_name, self.is_borrowed)

    async def _act_url_toggle(self, interaction: discord.Interaction, profile: Dict[str, Any]):
        current_mode = profile.get("url_mode", "off")
        if "url_mode" not in profile:
            current_mode = "rag" if profile.get("url_fetching_enabled", False) else "off"
        cycle_map = {"off": "native", "native": "rag", "rag": "off"}
        profile["url_mode"] = cycle_map.get(current_mode, "off")
        profile["url_fetching_enabled"] = (profile["url_mode"] == "rag") # Legacy support
        await self._save_and_refresh(interaction, profile, self.profile_name, self.is_borrowed)

    async def _act_manage_ltm(self, interaction: discord.Interaction, profile: Dict[str, Any]):
        view = DataManageView(self.cog, interaction, self.profile_name, self.is_borrowed,
                              mode='ltm', parent_manage_view=self, target_user_id=self.user_id)
        await view.start()

    async def _act_manage_training(self, interaction: discord.Interaction, profile: Dict[str, Any]):
        view = DataManageView(self.cog, interaction, self.profile_name, self.is_borrowed,
                              mode='training', parent_manage_view=self, target_user_id=self.user_id)
        await view.start()
        # NOTE: this defer runs *after* start() has already responded, so it raises
        # InteractionResponded and is swallowed by discord.py's handler. Preserved
        # verbatim from the pre-refactor branch rather than silently fixed -- see the
        # note accompanying this refactor.
        await interaction.response.defer()

    async def _act_ltm_summarization(self, interaction: discord.Interaction, profile: Dict[str, Any]):
        instr = profile.get("ltm_summarization_instructions") or self.cog.profile_manager._default_ltm_summarization_instructions()
        modal = ProfileLTMSummarizationModal(self.cog, self.profile_name, instr, target_user_id=self.user_id)
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
            self.cog.channel_model_last_profile_key.pop(k, None)

        new_embed = await self.cog.profile_manager._build_profile_manage_embed(interaction, profile_name, target_user_id=self.user_id)
        await interaction.response.edit_message(embed=new_embed, view=self)

    async def _handle_content_safety(self, interaction):
        view = ContentSafetyView(self.cog, self.original_interaction, self.profile_name,
                                 self.is_borrowed, self.mod_return_user_id, self.is_mod_view)
        await interaction.response.defer()
        await view.refresh_state()
        view._build_view()
        await self.original_interaction.edit_original_response(
            content=None, embed=view.get_embed(), view=view)

    async def _handle_appearance(self, interaction):
        modal = AppearanceModal(self.cog, self.original_interaction, self.profile_name, target_user_id=self.user_id)
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
            
            if self.cog.profile_manager._rename_profile(self.user_id, old_name, new_name, self.is_borrowed):
                # Hot-swap live sessions and models to prevent corruption
                for ch_id, session in self.cog.multi_profile_channels.items():
                    for p in session.get("profiles", []):
                        if p["owner_id"] == self.user_id and p["profile_name"] == old_name:
                            p["profile_name"] = new_name
                    

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
            
            limit = defaultConfig.LIMIT_PROFILES
            if len(user_index.get("personal", {})) >= limit:
                await self.original_interaction.edit_original_response(content="Limit reached.", view=None, embed=None); return
            
            success, msg = self.cog.profile_manager._duplicate_profile(self.user_id, self.profile_name, new_name)
            if success:
                await self.original_interaction.edit_original_response(content=f"Duplicated to '{new_name}'.", view=None, embed=None)
            else:
                await self.original_interaction.edit_original_response(content=f"Duplicate failed: {msg}", view=None, embed=None)
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
        view = SingleProfileTimezoneView(self.cog, self, profile, is_borrowed)
        await interaction.response.send_message(content=view._get_header_content(), view=view, ephemeral=True)

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
        if success:
            await maybe_prompt_rating_after_edit(
                self.cog_instance, i, self.user_id, self.profile_name)
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
        if success:
            await maybe_prompt_rating_after_edit(
                self.cog, i, self.user_id, self.profile_name)
    async def on_error(self, i:discord.Interaction,e:Exception): print(f"EditUserProfileAIInstrModal err: {e}"); traceback.print_exc(); await i.followup.send('Form error.',ephemeral=True)

def ProfileParamsModal(cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, callback=None, target_user_id: Optional[int] = None):
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
    return ConfigModal(cog, profile_name, is_borrowed, "Set Profile Generation Parameters", fields, parser, callback, target_user_id)

def ProfileTrainingParamsModal(cog, profile_name: str, current_params: Dict[str, Any], callback=None, target_user_id: Optional[int] = None):
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
    return ConfigModal(cog, profile_name, False, "Set Profile Training Parameters", fields, parser, callback, target_user_id)

def ProfileThinkingParamsModal(cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, callback=None, target_user_id: Optional[int] = None):
    fields = [
        {"label": "Thinking Summary (on/off)", "custom_id": "thinking_summary_visible", "default": current_params.get("thinking_summary_visible", "off"), "required": False, "placeholder": "Display reasoning tokens below your message."},
        {"label": "Reasoning Effort / Level", "custom_id": "thinking_level", "default": current_params.get("thinking_level", "low"), "required": False, "placeholder": "xhigh, high, medium, low, minimal, none"},
        {"label": "Reasoning Token Budget (-1=dyn)", "custom_id": "thinking_budget", "default": str(current_params.get("thinking_budget", -1)), "required": False, "placeholder": "-1 = dynamic, 128+ = token limit"}
    ]
    def parser(v):
        c = {}
        sv = _ps(v["thinking_summary_visible"])
        c["thinking_summary_visible"] = "on" if sv and sv.lower() == "on" else "off"
        
        lv = _ps(v["thinking_level"])
        c["thinking_level"] = lv.lower() if lv and lv.lower() in ["xhigh", "high", "medium", "low", "minimal", "none"] else "high"
        
        bv = _pi(v["thinking_budget"])
        c["thinking_budget"] = min(bv if bv is not None and bv >= -1 else -1, 32768)
        
        return {"config": c}
    return ConfigModal(cog, profile_name, is_borrowed, "Thinking & Reasoning Parameters", fields, parser, callback, target_user_id)

def ProfileLTMParamsModal(cog, profile_name: str, current_params: Dict[str, Any], callback=None, target_user_id: Optional[int] = None):
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
    return ConfigModal(cog, profile_name, False, "LTM Parameters", fields, parser, callback, target_user_id)

def ProfileLTMSummarizationModal(cog, profile_name: str, current_instructions: str, callback=None, target_user_id: Optional[int] = None):
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
        ins = _ps(v["ltm_summarization_instructions"]) or cog.profile_manager._default_ltm_summarization_instructions()
        return {"prompts": {"ltm_summarization_instructions": cog.storage_manager._encrypt_data(ins)}}
    return ConfigModal(cog, profile_name, False, "Set LTM Summarization Instructions", fields, parser, callback, target_user_id)

def ProfileTypingSettingsModal(cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, callback=None, target_user_id: Optional[int] = None):
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
    return ConfigModal(cog, profile_name, is_borrowed, "Realistic Typing Settings", fields, parser, callback, target_user_id)

def ProfileImageGenSettingsModal(cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, callback=None, target_user_id: Optional[int] = None):
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
    return ConfigModal(cog, profile_name, is_borrowed, "Image Generation Settings", fields, parser, callback, target_user_id)

class ModelPickerMixin:
    """The model-selection machinery shared by the single-profile and bulk pickers.

    SingleProfileModelView and ModelApplyView present the same picker over different
    targets -- one profile versus a selected set -- and had drifted into two copies of
    the option builder, the Ollama probe, the select class and the 48-line API/category
    button row. The copies differed only in where they read the Ollama host from, and in
    one missing truncation (see `_create_model_options`).

    Adopters must provide `view_mode`, `category`, `ollama_working`, `_build_view`,
    `_get_selection_feedback_message`, and `_ollama_host_url`.
    """

    def _get_top_models(self, provider: str, target_config_key: str) -> List[str]:
        return self.cog.api_service.get_top_models(provider, target_config_key)

    def _ollama_host_url(self) -> str:
        """Where this view reads the configured Ollama host from."""
        raise NotImplementedError

    class GenericModelSelect(ui.Select):
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
            # The [:100] truncation was present only in the single-profile copy. Discord
            # rejects an option label over 100 characters, so the bulk picker would raise
            # on a long custom model id; sharing this version fixes that.
            opts.append(discord.SelectOption(label=f"Current: {current_val}"[:100], value=current_val, default=True))
        
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

    async def _update_ollama_status(self):
        host_url = self._ollama_host_url() or OLLAMA_LOCAL_URL
        try:
            resp = await get_shared_client().get(f"{host_url.rstrip('/')}/api/tags", timeout=2.0)
            self.ollama_working = (resp.status_code == 200)
            if self.ollama_working:
                data = resp.json()
                self.cached_ollama_models = [m['name'] for m in data.get('models', [])]
        except Exception:
            self.ollama_working = False
            self.cached_ollama_models = []

    def _add_api_and_category_buttons(self, *, row: int = 2):
        """The API-mode / Ollama-host / category button row, identical in both pickers."""
        api_modes = ['google', 'openrouter', 'ollama']
        api_labels = {'google': 'API: Google', 'openrouter': 'API: OpenRouter', 'ollama': 'API: Ollama (Local)'}
        
        btn_api = ui.Button(label=api_labels[self.view_mode], style=discord.ButtonStyle.primary, row=row, disabled=(self.category == 'media'))
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
                
            btn_host = ui.Button(label="Set Host URL", style=host_style, row=row)
            async def host_cb(i: discord.Interaction):
                await i.response.send_modal(OllamaHostModal(self))
            btn_host.callback = host_cb
            self.add_item(btn_host)
            
            btn_guide = ui.Button(label="Guide", style=discord.ButtonStyle.secondary, row=row)
            async def guide_cb(i: discord.Interaction):
                await i.response.send_message(OLLAMA_GUIDE_TEXT, ephemeral=True)
            btn_guide.callback = guide_cb
            self.add_item(btn_guide)

        categories = ['response', 'media', 'tools', 'ltm']
        cat_labels = {'response': 'Response', 'media': 'Media', 'tools': 'Tools', 'ltm': 'LTM'}
        btn_cat = ui.Button(label=f"Category: {cat_labels[self.category]}", style=discord.ButtonStyle.blurple, row=row)
        async def cat_cb(i: discord.Interaction):
            next_idx = (categories.index(self.category) + 1) % len(categories)
            self.category = categories[next_idx]
            self._build_view()
            await i.response.edit_message(content=self._get_selection_feedback_message(), view=self)
        btn_cat.callback = cat_cb
        self.add_item(btn_cat)

    async def on_error(self, interaction: discord.Interaction, error: Exception, item: ui.Item):
        print(f"Error in {type(self).__name__}: {error}")
        traceback.print_exc()
        if not interaction.response.is_done():
            await interaction.response.send_message("An unexpected error occurred with this view.", ephemeral=True)
        else:
            await interaction.followup.send("An unexpected error occurred with this view.", ephemeral=True)


class SingleProfileModelView(ModelPickerMixin, ui.View):
    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction, profile_name: str, is_borrowed: Optional[bool] = None, user_id: Optional[int] = None):
        super().__init__(timeout=300)
        self.cog = cog
        self.original_interaction = interaction
        self.user_id = user_id or interaction.user.id
        self.profile_name = profile_name
        self.view_mode = 'google'
        self.category = 'response' # 'response', 'media', 'tools', 'ltm'

        if is_borrowed is not None:
            self.is_borrowed = is_borrowed
        else:
            index = self.cog.profile_manager._get_user_index(self.user_id)
            self.is_borrowed = profile_name in index.get("borrowed", [])
        
        self._build_view()

    def _get_current_profile_data(self) -> Dict[str, Any]:
        return self.cog.profile_manager._get_profile_config(self.user_id, self.profile_name, self.is_borrowed) or {}

    def _ollama_host_url(self) -> str:
        return self._get_current_profile_data().get("ollama_host_url")

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
                self.cog.channel_model_last_profile_key.pop(k, None)

    def _get_selection_feedback_message(self) -> str:
        data = self._get_current_profile_data()
        
        def clean(val):
            if not val: return "None"
            s = str(val)
            for prefix in ("GOOGLE/", "OPENROUTER/", "OLLAMA/"):
                if s.startswith(prefix):
                    return s[len(prefix):]
            return s
            
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
        self._add_api_and_category_buttons()

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
            
            def make_nav_cb(target_cat):
                async def nav_cb(i: discord.Interaction):
                    self.category = target_cat
                    self._build_view()
                    await i.response.edit_message(content=self._get_selection_feedback_message(), view=self)
                return nav_cb
                
            btn.callback = make_nav_cb(val)
            self.add_item(btn)

class ModelApplyView(ModelPickerMixin, ui.View):
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

    #: The bulk picker refers to the shared select under its historical name.
    GenericBulkModelSelect = ModelPickerMixin.GenericModelSelect

    def _ollama_host_url(self) -> str:
        return self.models_state.get("ollama_host_url")

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
            s = str(val)
            for prefix in ("GOOGLE/", "OPENROUTER/", "OLLAMA/"):
                if s.startswith(prefix):
                    return s[len(prefix):]
            return s
            
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

        self._add_api_and_category_buttons()

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
                self.cog.channel_model_last_profile_key.pop(k, None)

        msg = f"Updated models for {success_count} profiles." if success_count else "No profiles updated."
        await interaction.edit_original_response(content=msg, view=None)

class UnifiedBulkTargetView(BaseBulkProfileView):
    def __init__(self, cog: 'MimicCog', user_id: int, action_key: str, payload: Any, include_borrowed: bool = True, exclude_public: bool = False):
        super().__init__(cog, user_id, include_borrowed=include_borrowed, exclude_public=exclude_public)
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

            if self.action_key == "adult_declaration":
                # Not a config key this loop can set: the declaration is a
                # content_rating record with its own writer, its own cache
                # invalidation, and its own refusals -- a classifier verdict or an
                # exemption is not the owner's to overwrite in bulk either. The
                # manager persists it, so this skips the save below.
                if self.payload is not None and self.cog.profile_manager.set_owner_adult_declaration(
                        self.user_id, name, bool(self.payload)):
                    success_count += 1
                continue

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
                self.cog.channel_model_last_profile_key.pop(k, None)

        await interaction.edit_original_response(content=f"Successfully applied settings to {success_count} profile(s).", view=None)

from ..utils.helpers import _resolve_zoneinfo

EXTENSIVE_TIMEZONES = [
    # --- Page 0: Americas (North & Central) ---
    ("US / Pacific (Los Angeles)", "America/Los_Angeles", "UTC-8 / UTC-7 (PT)"),
    ("US / Mountain (Denver)", "America/Denver", "UTC-7 / UTC-6 (MT)"),
    ("US / Central (Chicago)", "America/Chicago", "UTC-6 / UTC-5 (CT)"),
    ("US / Eastern (New York)", "America/New_York", "UTC-5 / UTC-4 (ET)"),
    ("US / Alaska (Anchorage)", "America/Anchorage", "UTC-9 / UTC-8 (AKST)"),
    ("US / Hawaii (Honolulu)", "Pacific/Honolulu", "UTC-10 (HST)"),
    ("US / Arizona (Phoenix - No DST)", "America/Phoenix", "UTC-7 (MST)"),
    ("Canada / Pacific (Vancouver)", "America/Vancouver", "UTC-8 / UTC-7 (PT)"),
    ("Canada / Mountain (Edmonton)", "America/Edmonton", "UTC-7 / UTC-6 (MT)"),
    ("Canada / Central (Winnipeg)", "America/Winnipeg", "UTC-6 / UTC-5 (CT)"),
    ("Canada / Eastern (Toronto)", "America/Toronto", "UTC-5 / UTC-4 (ET)"),
    ("Canada / Atlantic (Halifax)", "America/Halifax", "UTC-4 / UTC-3 (AT)"),
    ("Canada / Newfoundland (St. Johns)", "America/St_Johns", "UTC-3:30 / UTC-2:30 (NT)"),
    ("Mexico / Pacific (Tijuana)", "America/Tijuana", "UTC-8 / UTC-7"),
    ("Mexico / Central (Mexico City)", "America/Mexico_City", "UTC-6 (CST)"),
    ("Mexico / Mountain (Hermosillo)", "America/Hermosillo", "UTC-7 (MST)"),
    ("Guatemala (Central America)", "America/Guatemala", "UTC-6 (CST)"),
    ("Costa Rica (San Jose)", "America/Costa_Rica", "UTC-6 (CST)"),
    ("Panama (Panama City)", "America/Panama", "UTC-5 (EST)"),
    ("Jamaica (Kingston)", "America/Jamaica", "UTC-5 (EST)"),

    # --- Page 1: South America & Atlantic ---
    ("Brazil / Southeast (Sao Paulo)", "America/Sao_Paulo", "UTC-3 (BRT)"),
    ("Brazil / East (Rio de Janeiro)", "America/Bahia", "UTC-3 (BRT)"),
    ("Brazil / Amazon (Manaus)", "America/Manaus", "UTC-4 (AMT)"),
    ("Argentina (Buenos Aires)", "America/Argentina/Buenos_Aires", "UTC-3 (ART)"),
    ("Chile (Santiago)", "America/Santiago", "UTC-4 / UTC-3 (CLT)"),
    ("Colombia (Bogota)", "America/Bogota", "UTC-5 (COT)"),
    ("Peru (Lima)", "America/Lima", "UTC-5 (PET)"),
    ("Venezuela (Caracas)", "America/Caracas", "UTC-4 (VET)"),
    ("Ecuador (Quito)", "America/Guayaquil", "UTC-5 (ECT)"),
    ("Bolivia (La Paz)", "America/La_Paz", "UTC-4 (BOT)"),
    ("Paraguay (Asuncion)", "America/Asuncion", "UTC-4 / UTC-3 (PYT)"),
    ("Uruguay (Montevideo)", "America/Montevideo", "UTC-3 (UYT)"),
    ("Puerto Rico (San Juan)", "America/Puerto_Rico", "UTC-4 (AST)"),
    ("Dominican Republic (Santo Domingo)", "America/Santo_Domingo", "UTC-4 (AST)"),
    ("Greenland (Nuuk)", "America/Nuuk", "UTC-2 / UTC-1 (WGT)"),
    ("Azores (Ponta Delgada)", "Atlantic/Azores", "UTC-1 / UTC+0 (AZOT)"),
    ("Cape Verde (Praia)", "Atlantic/Cape_Verde", "UTC-1 (CVT)"),
    ("Iceland (Reykjavik)", "Atlantic/Reykjavik", "UTC+0 (GMT)"),
    ("UTC (Coordinated Universal Time)", "UTC", "UTC+0 (Universal Standard)"),
    ("GMT (Greenwich Mean Time)", "GMT", "UTC+0 (Standard)"),

    # --- Page 2: Europe & Africa ---
    ("UK (London / GMT / BST)", "Europe/London", "UTC+0 / UTC+1 (GMT/BST)"),
    ("Ireland (Dublin)", "Europe/Dublin", "UTC+0 / UTC+1 (IST)"),
    ("France (Paris)", "Europe/Paris", "UTC+1 / UTC+2 (CET/CEST)"),
    ("Germany (Berlin)", "Europe/Berlin", "UTC+1 / UTC+2 (CET/CEST)"),
    ("Italy (Rome)", "Europe/Rome", "UTC+1 / UTC+2 (CET/CEST)"),
    ("Spain (Madrid)", "Europe/Madrid", "UTC+1 / UTC+2 (CET/CEST)"),
    ("Netherlands (Amsterdam)", "Europe/Amsterdam", "UTC+1 / UTC+2 (CET/CEST)"),
    ("Belgium (Brussels)", "Europe/Brussels", "UTC+1 / UTC+2 (CET/CEST)"),
    ("Switzerland (Zurich)", "Europe/Zurich", "UTC+1 / UTC+2 (CET/CEST)"),
    ("Sweden (Stockholm)", "Europe/Stockholm", "UTC+1 / UTC+2 (CET/CEST)"),
    ("Norway (Oslo)", "Europe/Oslo", "UTC+1 / UTC+2 (CET/CEST)"),
    ("Poland (Warsaw)", "Europe/Warsaw", "UTC+1 / UTC+2 (CET/CEST)"),
    ("Austria (Vienna)", "Europe/Vienna", "UTC+1 / UTC+2 (CET/CEST)"),
    ("Greece (Athens)", "Europe/Athens", "UTC+2 / UTC+3 (EET/EEST)"),
    ("Finland (Helsinki)", "Europe/Helsinki", "UTC+2 / UTC+3 (EET/EEST)"),
    ("Ukraine (Kyiv)", "Europe/Kyiv", "UTC+2 / UTC+3 (EET/EEST)"),
    ("Romania (Bucharest)", "Europe/Bucharest", "UTC+2 / UTC+3 (EET/EEST)"),
    ("Egypt (Cairo)", "Africa/Cairo", "UTC+2 / UTC+3 (EET)"),
    ("South Africa (Johannesburg)", "Africa/Johannesburg", "UTC+2 (SAST)"),
    ("Nigeria (Lagos)", "Africa/Lagos", "UTC+1 (WAT)"),

    # --- Page 3: Middle East, Asia & Australasia ---
    ("Turkey (Istanbul)", "Europe/Istanbul", "UTC+3 (TRT)"),
    ("Russia (Moscow)", "Europe/Moscow", "UTC+3 (MSK)"),
    ("United Arab Emirates (Dubai)", "Asia/Dubai", "UTC+4 (GST)"),
    ("Saudi Arabia (Riyadh)", "Asia/Riyadh", "UTC+3 (AST)"),
    ("India (Kolkata / New Delhi)", "Asia/Kolkata", "UTC+5:30 (IST)"),
    ("Pakistan (Karachi)", "Asia/Karachi", "UTC+5 (PKT)"),
    ("Bangladesh (Dhaka)", "Asia/Dhaka", "UTC+6 (BST)"),
    ("Thailand (Bangkok)", "Asia/Bangkok", "UTC+7 (ICT)"),
    ("Vietnam (Ho Chi Minh)", "Asia/Ho_Chi_Minh", "UTC+7 (ICT)"),
    ("Indonesia (Jakarta)", "Asia/Jakarta", "UTC+7 (WIB)"),
    ("China (Beijing / Shanghai)", "Asia/Shanghai", "UTC+8 (CST)"),
    ("Hong Kong", "Asia/Hong_Kong", "UTC+8 (HKT)"),
    ("Singapore", "Asia/Singapore", "UTC+8 (SGT)"),
    ("Japan (Tokyo)", "Asia/Tokyo", "UTC+9 (JST)"),
    ("South Korea (Seoul)", "Asia/Seoul", "UTC+9 (KST)"),
    ("Australia / NSW & VIC (Sydney)", "Australia/Sydney", "UTC+10 / UTC+11 (AEST/AEDT)"),
    ("Australia / QLD (Brisbane - No DST)", "Australia/Brisbane", "UTC+10 (AEST)"),
    ("Australia / SA (Adelaide)", "Australia/Adelaide", "UTC+9:30 / UTC+10:30 (ACST/ACDT)"),
    ("Australia / WA (Perth)", "Australia/Perth", "UTC+8 (AWST)"),
    ("New Zealand (Auckland)", "Pacific/Auckland", "UTC+12 / UTC+13 (NZST/NZDT)"),
]

PARTITION_NAMES = [
    "Americas (North & Central)",
    "South America & Atlantic",
    "Europe & Africa",
    "Asia, Middle East & Australasia"
]

class SingleProfileTimezoneView(ui.View):
    def __init__(self, cog: 'MimicCog', parent_manage_view: ProfileManageView, profile_config: Dict[str, Any], is_borrowed: bool):
        super().__init__(timeout=300)
        self.cog = cog
        self.parent_manage_view = parent_manage_view
        self.profile_config = profile_config
        self.is_borrowed = is_borrowed
        self.current_page = 0
        self.total_pages = (len(EXTENSIVE_TIMEZONES) - 1) // 20 + 1
        self._build_view()

    def _get_header_content(self) -> str:
        current_tz = self.profile_config.get("timezone", "UTC")
        try:
            tz_obj, _ = _resolve_zoneinfo(current_tz)
            now_str = datetime.datetime.now(tz_obj).strftime("%I:%M %p (%Z)")
        except Exception:
            now_str = "Unknown"
        return f"**Timezone Selector for '{self.parent_manage_view.profile_name}'**\n**Active Setting:** `{current_tz}` (Local Time: `{now_str}`)\nSelect a timezone below or jump between regional partitions:"

    def _build_view(self):
        self.clear_items()
        per_page = 20
        start = self.current_page * per_page
        page_tzs = EXTENSIVE_TIMEZONES[start:start + per_page]

        options = [
            discord.SelectOption(label="⚙️ Custom / Manual Timezone ID...", value="custom", description="Enter any custom IANA timezone ID manually.", emoji="✏️")
        ]

        # Add 3 Partition Jump options (excluding current page)
        for page_idx, p_name in enumerate(PARTITION_NAMES):
            if page_idx != self.current_page:
                options.append(discord.SelectOption(
                    label=f"🌍 Jump: {p_name}",
                    value=f"jump_{page_idx}",
                    description=f"Switch to page {page_idx + 1} ({p_name})",
                    emoji="📑"
                ))

        # Add 20 Timezone options for the active page
        current_setting = self.profile_config.get("timezone", "UTC")
        for label, tz_val, desc in page_tzs:
            options.append(discord.SelectOption(
                label=label[:100],
                value=tz_val,
                description=desc[:100],
                default=(tz_val == current_setting)
            ))

        select = ui.Select(placeholder=f"Timezones: {PARTITION_NAMES[self.current_page]} ({self.current_page + 1}/{self.total_pages})...", options=options, row=0)
        select.callback = self.select_callback
        self.add_item(select)

        # Pagination controls on Row 1
        async def prev_cb(i: discord.Interaction):
            self.current_page = max(0, self.current_page - 1)
            self._build_view()
            await i.response.edit_message(content=self._get_header_content(), view=self)

        async def next_cb(i: discord.Interaction):
            self.current_page = min(self.total_pages - 1, self.current_page + 1)
            self._build_view()
            await i.response.edit_message(content=self._get_header_content(), view=self)

        build_pagination_controls(self, self.current_page, self.total_pages, 1, prev_cb, next_cb)

    async def select_callback(self, interaction: discord.Interaction):
        choice = interaction.data['values'][0]

        if choice == "custom":
            modal = CustomTimezoneModal(self)
            await interaction.response.send_modal(modal)
            return

        if choice.startswith("jump_"):
            target_page = int(choice.split("_")[1])
            self.current_page = target_page
            self._build_view()
            await interaction.response.edit_message(content=self._get_header_content(), view=self)
            return

        # Direct timezone selection
        _, canonical_tz = _resolve_zoneinfo(choice)
        self.profile_config["timezone"] = canonical_tz
        self.profile_config["time_tracking_enabled"] = True
        self.cog.profile_manager._save_profile_config(self.parent_manage_view.user_id, self.parent_manage_view.profile_name, self.profile_config, self.is_borrowed)

        # Flush model cache for this profile
        keys = [k for k in self.cog.channel_models.keys() if isinstance(k, tuple) and k[1] == self.parent_manage_view.user_id]
        for k in keys:
            self.cog.channel_models.pop(k, None)

        new_embed = await self.cog.profile_manager._build_profile_manage_embed(interaction, self.parent_manage_view.profile_name, target_user_id=self.parent_manage_view.user_id)
        await self.parent_manage_view.original_interaction.edit_original_response(embed=new_embed, view=self.parent_manage_view)
        await interaction.response.edit_message(content=f"✅ Timezone set to **{canonical_tz}**.", view=None)

class CustomTimezoneModal(ui.Modal, title="Enter Custom Timezone"):
    tz_input = ui.TextInput(label="Timezone ID / Acronym", placeholder="e.g. Australia/Sydney, AEST, America/New_York", required=True)

    def __init__(self, parent_view: Union[SingleProfileTimezoneView, 'BulkTimezoneView']):
        super().__init__()
        self.parent_view = parent_view

    async def on_submit(self, interaction: discord.Interaction):
        raw_val = self.tz_input.value.strip()
        tz_obj, canonical_tz = _resolve_zoneinfo(raw_val)

        if isinstance(self.parent_view, SingleProfileTimezoneView):
            self.parent_view.profile_config["timezone"] = canonical_tz
            self.parent_view.profile_config["time_tracking_enabled"] = True
            self.parent_view.cog.profile_manager._save_profile_config(self.parent_view.parent_manage_view.user_id, self.parent_view.parent_manage_view.profile_name, self.parent_view.profile_config, self.parent_view.is_borrowed)

            keys = [k for k in self.parent_view.cog.channel_models.keys() if isinstance(k, tuple) and k[1] == self.parent_view.parent_manage_view.user_id]
            for k in keys:
                self.parent_view.cog.channel_models.pop(k, None)

            new_embed = await self.parent_view.cog.profile_manager._build_profile_manage_embed(interaction, self.parent_view.parent_manage_view.profile_name, target_user_id=self.parent_view.parent_manage_view.user_id)
            await self.parent_view.parent_manage_view.original_interaction.edit_original_response(embed=new_embed, view=self.parent_view.parent_manage_view)
            await interaction.response.edit_message(content=f"✅ Timezone set to **{canonical_tz}**.", view=None)
        else:
            self.parent_view.selected_tz = canonical_tz
            self.parent_view._build_view()
            await interaction.response.edit_message(content=self.parent_view._get_selection_feedback_message(), view=self.parent_view)

# Alias for backward compatibility
BulkTimezoneModal = CustomTimezoneModal

class BulkTimezoneView(BaseBulkProfileView):
    def __init__(self, cog: 'MimicCog', user_id: int):
        super().__init__(cog, user_id, include_borrowed=True)
        self.selected_tz = None
        self.tz_page = 0
        self.tz_total_pages = (len(EXTENSIVE_TIMEZONES) - 1) // 20 + 1
        self._build_view()

    def _build_view(self):
        self.clear_items()
        per_page = 20
        start = self.tz_page * per_page
        page_tzs = EXTENSIVE_TIMEZONES[start:start + per_page]

        options = [
            discord.SelectOption(label="⚙️ Custom / Manual Timezone ID...", value="custom", description="Enter any custom IANA timezone ID manually.", emoji="✏️")
        ]

        # Add 3 Partition Jump options
        for page_idx, p_name in enumerate(PARTITION_NAMES):
            if page_idx != self.tz_page:
                options.append(discord.SelectOption(
                    label=f"🌍 Jump: {p_name}",
                    value=f"jump_{page_idx}",
                    description=f"Switch to page {page_idx + 1} ({p_name})",
                    emoji="📑"
                ))

        # Add 20 Timezone options for the active page
        for label, tz_val, desc in page_tzs:
            options.append(discord.SelectOption(
                label=label[:100],
                value=tz_val,
                description=desc[:100],
                default=(tz_val == self.selected_tz)
            ))

        select = ui.Select(placeholder=f"Choose a timezone ({PARTITION_NAMES[self.tz_page]})...", options=options, row=0)
        select.callback = self.tz_callback
        self.add_item(select)

        self._build_profile_select_ui(row=1)
        
        apply_btn = ui.Button(label="Apply Timezone", style=discord.ButtonStyle.green, row=3)
        apply_btn.callback = self.apply_action
        self.add_item(apply_btn)

    async def tz_callback(self, interaction: discord.Interaction):
        val = interaction.data['values'][0]
        if val == "custom":
            await interaction.response.send_modal(CustomTimezoneModal(self))
            return

        if val.startswith("jump_"):
            target_page = int(val.split("_")[1])
            self.tz_page = target_page
            self._build_view()
            await interaction.response.edit_message(content=self._get_selection_feedback_message(), view=self)
            return

        _, canonical = _resolve_zoneinfo(val)
        self.selected_tz = canonical
        self._build_view()
        await interaction.response.edit_message(content=self._get_selection_feedback_message(), view=self)

    async def apply_action(self, interaction: discord.Interaction):
        await interaction.response.defer()
        if not self.selected_tz or not self.selected_profiles:
            await interaction.edit_original_response(content="Select a timezone and at least one profile.", view=None)
            return

        updated_count = 0
        index = self.cog.profile_manager._get_user_index(self.user_id)
        for name in self.selected_profiles:
            is_borrowed = name in index.get("borrowed", [])
            p = self.cog.profile_manager._get_profile_config(self.user_id, name, is_borrowed)
            if p:
                p["timezone"] = self.selected_tz
                p["time_tracking_enabled"] = True
                self.cog.profile_manager._save_profile_config(self.user_id, name, p, is_borrowed)
                updated_count += 1
        
        if updated_count > 0:
            keys = [k for k in self.cog.channel_models.keys() if isinstance(k, tuple) and k[1] == self.user_id]
            for k in keys: 
                self.cog.channel_models.pop(k, None)

        await interaction.edit_original_response(content=f"Timezone set to **{self.selected_tz}** for {updated_count} profiles.", view=None)

def ProfileGenerationVisualModal(cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, callback=None, target_user_id: Optional[int] = None):
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
    return ConfigModal(cog, profile_name, is_borrowed, "Generation Visual", fields, parser, callback, target_user_id)

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
    def __init__(self, cog: 'MimicCog', original_interaction: discord.Interaction, profile_name: str,
                 target_user_id: Optional[int] = None):
        super().__init__(title=f"Appearance: '{profile_name[:20]}'")
        self.cog = cog
        self.original_interaction = original_interaction
        self.profile_name = profile_name
        self.owner_id = target_user_id or original_interaction.user.id

        user_id_str = str(self.owner_id)
        current_data = self.cog.user_appearances.get(user_id_str, {}).get(self.profile_name, {})
        
        self.display_name_input = ui.TextInput(label="Custom Display Name (Blank to reset)", required=False, max_length=20, default=current_data.get("custom_display_name"))
        self.avatar_url_input = ui.TextInput(label="Avatar URL (Blank to reset)", required=False, default=current_data.get("custom_avatar_url"))
        self.add_item(self.display_name_input)
        self.add_item(self.avatar_url_input)

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer()
        new_display_name = self.display_name_input.value.strip() or None
        new_avatar_url = self.avatar_url_input.value.strip() or None
        owner_id = self.owner_id
        user_id_str = str(owner_id)

        if new_display_name:
            if len(new_display_name) > 32:
                await interaction.followup.send("❌ **Invalid Display Name:** Must be 32 characters or fewer.", ephemeral=True)
                return
            if any(r in new_display_name.lower() for r in ["clyde", "@everyone", "@here"]):
                await interaction.followup.send("❌ **Invalid Display Name:** Contains a reserved keyword or mention.", ephemeral=True)
                return

        # The published-profile safety re-check that used to run here is gone with the
        # auto-moderator. It downloaded the avatar from this host and refused the edit
        # if it could not -- which is how a profile whose avatar Discord serves
        # perfectly well ended up unable to change its own display name. The avatar is
        # now judged by the classifier, which treats a failed download as "no image"
        # rather than as a violation, and a published profile whose appearance changes
        # is picked up as a stale rating like any other edit.

        config = self.cog.profile_manager._get_profile_config(owner_id, self.profile_name, False)
        if config:
            if new_display_name: config["custom_display_name"] = new_display_name
            else: config.pop("custom_display_name", None)
            
            if new_avatar_url: config["custom_avatar_url"] = new_avatar_url
            else: config.pop("custom_avatar_url", None)
            
            self.cog.profile_manager._save_profile_config(owner_id, self.profile_name, config, False)
            # Display name and avatar are part of the classified surface, and they
            # live in config rather than prompts, so the prompts hook does not see
            # them. Invalidate only -- the rating going stale is resolved the same
            # way a persona edit is, rather than spending a call here.
            self.cog.profile_manager._invalidate_content_rating(owner_id, self.profile_name)
            
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

        new_embed = await self.cog.profile_manager._build_profile_manage_embed(self.original_interaction, self.profile_name, target_user_id=owner_id)
        await self.original_interaction.edit_original_response(embed=new_embed)
        await interaction.followup.send("Appearance updated.", ephemeral=True)
        # The display name and avatar are part of the classified surface, so an
        # appearance edit invalidates a rating exactly as a persona edit does.
        await maybe_prompt_rating_after_edit(self.cog, interaction, owner_id, self.profile_name)

def ProfileNeuroModal(cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, callback=None, target_user_id: Optional[int] = None):
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
    return ConfigModal(cog, profile_name, is_borrowed, "Neuro-Endocrine Engine Configuration", fields, parser, callback, target_user_id)

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
            discord.SelectOption(label="Set Adult 18+ Declaration", value="adult_declaration", description="Declare or withdraw 18+ across multiple profiles."),
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
            modal = ProfileLTMSummarizationModal(self.cog, "BULK_APPLY", self.cog.profile_manager._default_ltm_summarization_instructions(), callback=modal_callback)
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
            
        elif choice == "adult_declaration":
            opts = [
                discord.SelectOption(label="Declare Adult 18+", value="declare",
                                     description="Confine to age-restricted channels."),
                discord.SelectOption(label="Withdraw Declaration", value="withdraw",
                                     description="Hand the profile back to the classifier."),
            ]
            # Borrowed profiles are excluded: the rating is resolved from the source
            # profile, so writing the borrower's local copy would change nothing.
            view = UnifiedBulkTargetView(self.cog, self.user_id, "adult_declaration", None,
                                         include_borrowed=False, exclude_public=True)
            sel = ui.Select(placeholder="Declare or withdraw...", options=opts, row=0)
            async def sel_cb(inter): view.payload = sel.values[0] == "declare"; await inter.response.defer()
            sel.callback = sel_cb
            view.add_item(sel)
            # Say what was withheld rather than letting profiles quietly go missing
            # from the list.
            msg = "Select an action and the profiles to apply it to:"
            if view.excluded_public:
                names = ", ".join(f"`{n}`" for n in view.excluded_public)
                msg += (f"\n-# Withheld ({len(view.excluded_public)} published to the Public Library, "
                        f"which only accepts General profiles): {names}")
            if len(msg) > 2000:
                msg = (f"Select an action and the profiles to apply it to:\n-# Withheld "
                       f"{len(view.excluded_public)} profile(s) published to the Public Library, "
                       "which only accepts General profiles.")
            await interaction.response.send_message(content=msg, view=view, ephemeral=True)

        elif choice == "timezone":
            view = BulkTimezoneView(self.cog, self.user_id)
            await interaction.response.send_message(content="Select a timezone and the profiles to apply it to:", view=view, ephemeral=True)

        elif choice == "data_reset":
            view = BulkResetView(self.cog, self.user_id)
            await interaction.response.send_message(content="Select reset action:", view=view, ephemeral=True)

        elif choice == "delete_items":
            view = BulkDeleteView(self.cog, self.user_id)
            await interaction.response.send_message(content="Select profiles to delete:", view=view, ephemeral=True)


class ContentSafetyView(ui.View):
    """The Content Safety dashboard for one profile.

    Replaces the profile dashboard rather than opening beside it, because it is not
    a settings page -- it is the explanation of why the profile can or cannot do
    things, and it needs the room to say so. The three actions it absorbed
    (declare 18+, clear verdict, exemption) previously sat as unexplained rows in
    the Home dropdown, where a user who hit "this profile cannot be published" had
    no way to find out which of them applied.

    Every gate the bot enforces is rendered here from the same
    CONTENT_RATING_CAPABILITIES table the gates read, so the page cannot describe a
    rule the code does not implement.
    """

    def __init__(self, cog: 'MimicCog', original_interaction: discord.Interaction,
                 profile_name: str, is_borrowed: bool, target_user_id: Optional[int] = None,
                 is_mod_view: bool = False):
        super().__init__(timeout=600)
        self.cog = cog
        self.original_interaction = original_interaction
        self.profile_name = profile_name
        self.is_borrowed = is_borrowed
        self.is_mod_view = is_mod_view
        self.user_id = target_user_id or original_interaction.user.id
        self.mod_return_user_id = target_user_id
        self.is_bot_owner = original_interaction.user.id == int(defaultConfig.DISCORD_OWNER_ID)
        self.stale = False
        self.distributed = False
        self._build_view()

    # --- data -----------------------------------------------------------------

    async def refresh_state(self):
        """Reads the two facts that need I/O, off the interaction path.

        Staleness needs the persona decrypted and hashed, and distribution scans
        every user's borrow index. Both are far too heavy for a render, so they are
        resolved once when the view opens and reused until it is rebuilt.
        """
        pm = self.cog.profile_manager
        try:
            self.stale = await pm.rating_is_stale(self.user_id, self.profile_name)
            self.distributed = await asyncio.to_thread(
                pm.is_profile_distributed, self.user_id, self.profile_name)
        except Exception as e:
            print(f"Content Safety state read failed for {self.user_id}/{self.profile_name}: {e}")

    def _rating(self):
        pm = self.cog.profile_manager
        eff_owner, eff_name = pm._resolve_effective_profile(self.user_id, self.profile_name)
        config = pm._get_profile_config(eff_owner, eff_name, False) or {}
        return pm._verdict_of(config), (config.get("content_rating") or {})

    # --- layout ---------------------------------------------------------------

    def _build_view(self):
        self.clear_items()
        verdict, rating = self._rating()
        pm = self.cog.profile_manager

        back = ui.Button(label="← Back to Profile", style=discord.ButtonStyle.secondary, row=2)
        back.callback = self.back_cb
        self.add_item(back)

        # A borrowed profile is rated by whoever owns it. Showing the rating is
        # useful -- it explains the borrower's own gates -- but nothing here is
        # theirs to change.
        if self.is_borrowed:
            return

        # A Pending profile whose retry-after has elapsed is a failed submission, not
        # one in flight, and offering only "check again" would strand it.
        stalled = (verdict == CONTENT_RATING_PENDING
                   and rating.get("retry_after")
                   and time.time() >= rating["retry_after"])

        if verdict == CONTENT_RATING_UNRATED or stalled or (
                verdict == CONTENT_RATING_GENERAL and self.stale):
            submit = ui.Button(label="Try Again" if stalled else "Submit for Rating",
                               style=discord.ButtonStyle.success, row=0)
            submit.callback = self.submit_cb
            self.add_item(submit)

        if verdict == CONTENT_RATING_PENDING and not stalled:
            recheck = ui.Button(label="Check for a Verdict", style=discord.ButtonStyle.primary, row=0)
            recheck.callback = self.recheck_cb
            self.add_item(recheck)

        # The declaration is withheld exactly where it has nothing to move: on a
        # published profile, while a classifier verdict stands, and on an exemption.
        declared = pm._is_owner_declared_adult(self.user_id, self.profile_name)
        can_declare = verdict != CONTENT_RATING_EXEMPT and not pm._is_profile_public(
            self.user_id, self.profile_name)
        if verdict == CONTENT_RATING_ADULT and not declared:
            can_declare = False
        if can_declare:
            label = "Withdraw 18+ Declaration" if declared else "Declare Adult 18+"
            style = discord.ButtonStyle.secondary if declared else discord.ButtonStyle.danger
            btn = ui.Button(label=label, style=style, row=0)
            btn.callback = self.declare_cb
            self.add_item(btn)

        if self.is_mod_view and self.is_bot_owner:
            exempt = verdict == CONTENT_RATING_EXEMPT
            ex_btn = ui.Button(
                label="Remove Exemption" if exempt else "Exempt From Classification",
                style=discord.ButtonStyle.secondary if exempt else discord.ButtonStyle.primary,
                row=1)
            ex_btn.callback = self.exempt_cb
            self.add_item(ex_btn)

            if verdict == CONTENT_RATING_ADULT and not declared:
                clear = ui.Button(label="Clear Adult Verdict", style=discord.ButtonStyle.danger, row=1)
                clear.callback = self.clear_cb
                self.add_item(clear)

            force = ui.Button(label="Force Re-classify", style=discord.ButtonStyle.secondary, row=1)
            force.callback = self.force_cb
            self.add_item(force)

    def get_embed(self) -> discord.Embed:
        pm = self.cog.profile_manager
        verdict, rating = self._rating()

        colour = {
            CONTENT_RATING_UNRATED: discord.Color.light_grey(),
            CONTENT_RATING_PENDING: discord.Color.gold(),
            CONTENT_RATING_GENERAL: discord.Color.green(),
            CONTENT_RATING_ADULT: discord.Color.red(),
            CONTENT_RATING_EXEMPT: discord.Color.blurple(),
        }[verdict]

        emoji = CONTENT_RATING_EMOJI[verdict]
        label = CONTENT_RATING_LABELS[verdict]

        embed = discord.Embed(
            title=f"{emoji} Content Safety — {self.profile_name}",
            description=f"**Rating: {label}**\n{CONTENT_RATING_BLURBS[verdict]}",
            color=colour)

        # Why, when the verdict came from the classifier. A category code, never the
        # model's own words about somebody's persona.
        if verdict == CONTENT_RATING_ADULT:
            if pm._is_owner_declared_adult(self.user_id, self.profile_name):
                embed.add_field(
                    name="Set by", value="You declared this profile 18+.", inline=False)
            else:
                reason = CONTENT_RATING_REASON_LABELS.get(
                    rating.get("reason"), CONTENT_RATING_REASON_FALLBACK)
                embed.add_field(
                    name="Reason", value=f"{reason}.\nOnly the bot operator can clear this. "
                                         "Editing the persona lets you submit again.", inline=False)

        # Why a Pending profile is still pending.
        if verdict == CONTENT_RATING_PENDING:
            retry_after = rating.get("retry_after")
            if retry_after and time.time() < retry_after:
                embed.add_field(
                    name="Held up",
                    value=(f"The last attempt failed and it will retry "
                           f"<t:{int(retry_after)}:R>.\nThis usually means no API key was "
                           f"available — check `/settings`."),
                    inline=False)

        # The capability list, rendered from the table the gates actually read.
        rows = []
        for cap, cap_label in CONTENT_CAPABILITY_LABELS.items():
            allowed, reason = pm.content_capability(self.user_id, self.profile_name, cap)
            mark = "✅" if allowed else "❌"
            rows.append(f"{mark} **{cap_label}**" + ("" if allowed else f"\n　　*{reason}*"))

        access = "Age-restricted channels only" if CONTENT_RATING_CAPABILITIES[verdict]["age_restricted_only"] \
            else "Any channel"
        rows.insert(0, f"💬 **Runs in:** {access}")
        embed.add_field(name="What this profile can do", value="\n".join(rows), inline=False)

        if self.stale:
            if self.distributed:
                note = ("This profile has been edited since it was rated, and other people "
                        "are using it. It will be re-checked automatically.")
            else:
                note = ("This profile has been edited since it was rated, so the rating no "
                        "longer describes it. Submit it again when you want to share it.")
            embed.add_field(name="⚠️ Edited since rating", value=note, inline=False)

        if self.is_borrowed:
            embed.add_field(
                name="Borrowed profile",
                value="This profile is rated by its owner. You cannot change its rating.",
                inline=False)

        embed.add_field(
            name="What gets checked",
            value=("The profile name, display name, avatar image, persona and AI "
                   "instructions.\nNever your long-term memories, training examples, or "
                   "any conversation."),
            inline=False)

        if self.is_mod_view and self.is_bot_owner:
            detail = (f"verdict: `{verdict}`\n"
                      f"source: `{rating.get('source') or '—'}`\n"
                      f"model: `{rating.get('model') or '—'}`\n"
                      f"hash: `{rating.get('hash') or '—'}`\n"
                      f"at: `{rating.get('at') or '—'}`\n"
                      f"distributed: `{self.distributed}`")
            embed.add_field(name="Operator detail", value=detail, inline=False)

        return embed

    async def _refresh(self, i: discord.Interaction):
        await self.refresh_state()
        self._build_view()
        await i.edit_original_response(embed=self.get_embed(), view=self)

    # --- actions --------------------------------------------------------------

    async def back_cb(self, i: discord.Interaction):
        view = ProfileManageView(self.cog, self.original_interaction, self.profile_name,
                                 self.is_borrowed, self.mod_return_user_id, self.is_mod_view)
        await i.response.defer()
        await view.update_display()

    async def submit_cb(self, i: discord.Interaction):
        await i.response.defer()
        ok, msg = await self.cog.profile_manager.submit_for_rating(
            self.user_id, self.profile_name)
        await i.followup.send(msg, ephemeral=True)
        if ok:
            # The job is fire-and-forget; give it a beat so the common case renders
            # the finished verdict rather than a Pending the user has to refresh past.
            await asyncio.sleep(2.5)
        await self._refresh(i)

    async def recheck_cb(self, i: discord.Interaction):
        """Refreshes a Pending profile, re-queueing it if nothing is actually running.

        Pending means "a job is working on this", so if no job is in flight the state
        is a lie and refreshing forever will not fix it. Re-queueing here makes the
        button self-healing for any profile stranded by an interrupted submission --
        a restart mid-classification, or a scheduling call that never created a task.
        """
        await i.response.defer()
        pm = self.cog.profile_manager
        verdict, _ = self._rating()
        key = (int(self.user_id), self.profile_name)
        if verdict == CONTENT_RATING_PENDING and key not in self.cog.pending_classifications:
            pm.schedule_content_classification(self.user_id, self.profile_name)
            await asyncio.sleep(2.5)
        await self._refresh(i)

    async def declare_cb(self, i: discord.Interaction):
        pm = self.cog.profile_manager
        declared = pm._is_owner_declared_adult(self.user_id, self.profile_name)
        await i.response.defer()
        await asyncio.to_thread(
            pm.set_owner_adult_declaration, self.user_id, self.profile_name, not declared)
        await self._refresh(i)

    async def exempt_cb(self, i: discord.Interaction):
        pm = self.cog.profile_manager
        verdict, _ = self._rating()
        await i.response.defer()
        await asyncio.to_thread(
            pm.set_classification_exempt, self.user_id, self.profile_name,
            verdict != CONTENT_RATING_EXEMPT)
        await self._refresh(i)

    async def clear_cb(self, i: discord.Interaction):
        await i.response.defer()
        await asyncio.to_thread(
            self.cog.profile_manager.clear_adult_verdict,
            self.user_id, self.profile_name, i.user.id)
        await self._refresh(i)

    async def force_cb(self, i: discord.Interaction):
        await i.response.defer()
        self.cog.profile_manager.schedule_content_classification(self.user_id, self.profile_name)
        await i.followup.send("Re-classification queued.", ephemeral=True)
        await asyncio.sleep(2.5)
        await self._refresh(i)


class PostEditRatingView(ui.View):
    """The ephemeral prompt shown after editing a profile that was already rated.

    An edit invalidates the rating, and there are exactly two sensible responses:
    stop claiming a verdict that no longer describes the profile, or get a new one.
    Choosing for the user is wrong in both directions -- silently dropping a shared
    profile to Unrated pulls it out from under its borrowers, and silently
    re-classifying spends their API quota on an edit they may still be in the middle
    of. So the choice is theirs, taken at the one moment they have the context to
    make it.

    Ephemeral and self-timing-out: ignoring it leaves the rating stale, and
    resolve_stale_rating settles it on the next dashboard open or session start.
    """

    def __init__(self, cog: 'MimicCog', user_id: int, profile_name: str, distributed: bool):
        super().__init__(timeout=180)
        self.cog = cog
        self.user_id = user_id
        self.profile_name = profile_name
        self.distributed = distributed

    def get_content(self) -> str:
        base = (f"**'{self.profile_name}' has been edited since it was rated.**\n"
                f"Its current rating no longer describes it.")
        if self.distributed:
            base += ("\n\nOther people are using this profile, so leaving it rated means "
                     "leaving them a verdict that is out of date.")
        return base

    @ui.button(label="Re-check the rating", style=discord.ButtonStyle.success)
    async def recheck(self, i: discord.Interaction, _: ui.Button):
        await i.response.defer()
        self.cog.profile_manager.schedule_content_classification(self.user_id, self.profile_name)
        await i.edit_original_response(
            content=f"Re-checking the rating for '{self.profile_name}'.", view=None)
        self.stop()

    @ui.button(label="Set to Unrated", style=discord.ButtonStyle.secondary)
    async def unrate(self, i: discord.Interaction, _: ui.Button):
        await i.response.defer()
        await asyncio.to_thread(
            self.cog.profile_manager.drop_to_unrated, self.user_id, self.profile_name)
        await i.edit_original_response(
            content=(f"'{self.profile_name}' is now **Unrated**. It works exactly as before, "
                     f"but cannot be shared or used in Global Chat until you submit it again."),
            view=None)
        self.stop()


async def maybe_prompt_rating_after_edit(cog, interaction: discord.Interaction,
                                         user_id: int, profile_name: str):
    """Offers the post-edit choice, if there is one to make.

    Silent for the overwhelming majority of edits: an Unrated profile has no rating
    to invalidate, and a declared or exempt one cannot be moved by an edit. Costs a
    persona hash, so it is only ever called from an interactive edit path.
    """
    pm = cog.profile_manager
    try:
        if not await pm.rating_is_stale(user_id, profile_name):
            return
        distributed = await asyncio.to_thread(pm.is_profile_distributed, user_id, profile_name)
        view = PostEditRatingView(cog, user_id, profile_name, distributed)
        await interaction.followup.send(view.get_content(), view=view, ephemeral=True)
    except Exception as e:
        # Never cost the user their edit. The rating stays stale and
        # resolve_stale_rating settles it the next time the profile is opened.
        print(f"Post-edit rating prompt failed for {user_id}/{profile_name}: {e}")
