"""The `/settings` -> Defaults tab: standing preferences for profiles yet to exist.

Every profile used to start from one hardcoded template, and every borrow from a copy
of whoever wrote it. Someone who preferred a different model set it again on profile
one, profile two and profile forty, and a borrow could arrive configured for a
provider they had no key for and simply refuse to speak.

This screen is deliberately thin. It does not know which settings exist: the key table
is derived from `PROFILE_ACTIONS` in `utils/user_defaults`, and the model rows come
from `ModelPickerMixin._CATEGORY_KEYS` -- the same table the profile picker and the
bulk picker build themselves from. Adopting the mixin rather than reimplementing it is
what keeps the Google-only slots enforced here too: image, TTS and grounding pin the
API switch exactly as they do everywhere else, so a default cannot be stored that the
profile dashboard would have refused.

Values are written the moment they are chosen. The neighbouring API Keys tab stages
and waits for Save Assignments, and forgetting that button is the single most common
way a key ends up doing nothing; there is no equivalent risk here, because every
setting on this screen is reversible, sparse, and affects only profiles made later.
"""

import discord
from discord import ui
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..utils.constants import (TYPING_CURSOR_MODES, TYPING_CURSOR_NOTES,
                               DEFAULT_TYPING_CURSOR)
from ..utils.user_defaults import defaultable_keys, model_slot_labels
from .gui_profiles import ModelPickerMixin
from .gui_settings import SettingsBaseView

if TYPE_CHECKING:
    from ..MimicCog import MimicCog


#: The "not set" sentinel used by every select on this screen. Distinct from any
#: stored value, because clearing a default and setting it to whatever the bot
#: currently ships are different acts -- the first tracks a future change to the
#: shipped value, the second pins the profile to today's.
UNSET = "__unset__"

#: Behaviour rows presented as dropdowns, as (config key, wording, options).
#: Options are (label, value, description).
_BEHAVIOUR_CHOICES = (
    ("typing_cursor", "Typing Cursor",
     tuple((lbl, val, TYPING_CURSOR_NOTES[val]) for lbl, val in
           (("Below the text", "below"), ("In front of the text", "prefix"), ("Off", "off"))
           if val in TYPING_CURSOR_MODES)),
    ("thinking_level", "Reasoning Effort",
     (("Extra High", "xhigh", "Most deliberation, slowest, most expensive."),
      ("High", "high", "The shipped default."),
      ("Medium", "medium", "Balanced."),
      ("Low", "low", "Fast."),
      ("Minimal", "minimal", "Barely deliberates."),
      ("None", "none", "No reasoning pass at all."))),
)

#: Behaviour rows presented as tri-state buttons: unset -> On -> Off -> unset.
_BEHAVIOUR_TOGGLES = (
    ("realistic_typing_enabled", "Realistic Typing"),
    ("ltm_creation_enabled", "LTM Auto-Creation"),
)

#: Behaviour rows collected in a modal, as (config key, wording, placeholder, parser).
#: A parser returns the value to store, or raises ValueError with a user-facing reason.
def _parse_stm(raw: str) -> int:
    value = int(raw)
    if not 1 <= value <= 200:
        raise ValueError("Short-term memory must be between 1 and 200 turns.")
    return value


def _parse_timezone(raw: str) -> str:
    from ..utils.helpers import _resolve_zoneinfo
    if _resolve_zoneinfo(raw) is None:
        raise ValueError(f"`{raw}` is not an IANA timezone. Try `Australia/Perth`.")
    return raw


_BEHAVIOUR_NUMBERS = (
    ("stm_length", "Short-Term Memory (turns)", "1-200", _parse_stm),
    ("timezone", "Timezone", "e.g. Australia/Perth", _parse_timezone),
)


class DefaultsNumbersModal(ui.Modal, title="Default Behaviour"):
    """Blank leaves a value alone; a single '-' clears it back to the platform default.

    The same convention the image sampling modal uses. An empty field cannot mean
    "clear" here, because most visits will leave most fields untouched.
    """

    def __init__(self, view: "SettingsDefaultsView"):
        super().__init__()
        self.parent_view = view
        self.inputs = {}
        for key, wording, placeholder, _parser in _BEHAVIOUR_NUMBERS:
            current = view.defaults.get(key)
            field = ui.TextInput(
                label=wording[:45],
                placeholder=f"{placeholder} · blank = unchanged · '-' = clear",
                default=str(current) if current is not None else "",
                required=False, max_length=64)
            self.inputs[key] = field
            self.add_item(field)

    async def on_submit(self, interaction: discord.Interaction):
        errors = []
        staged: Dict[str, Any] = {}
        cleared = []

        for key, _wording, _placeholder, parser in _BEHAVIOUR_NUMBERS:
            raw = (self.inputs[key].value or "").strip()
            if not raw:
                continue
            if raw == "-":
                cleared.append(key)
                continue
            try:
                staged[key] = parser(raw)
            except ValueError as e:
                errors.append(str(e) if str(e) else f"`{raw}` is not valid for {key}.")

        if errors:
            await interaction.response.send_message("❌ " + "\n❌ ".join(errors), ephemeral=True)
            return

        for key in cleared:
            self.parent_view.defaults.pop(key, None)
        self.parent_view.defaults.update(staged)
        self.parent_view._persist()
        self.parent_view._build_view()
        await interaction.response.edit_message(**self.parent_view._picker_render())


class SettingsDefaultsView(ModelPickerMixin, SettingsBaseView):
    """Third adopter of ModelPickerMixin, alongside the single-profile and bulk pickers.

    The mixin's contract is `view_mode`, `category`, `ollama_working`, `_build_view`,
    `_get_selection_feedback_message` and `_ollama_host_url`; everything else -- the
    option builder, the Ollama probe, the API switch, the Google-only pinning -- comes
    for free and cannot drift from what the other two pickers do.
    """

    #: The mixin's own categories plus one that holds no models. `_add_category_select`
    #: reads this off the instance, so extending it here adds the row without the mixin
    #: needing to know that a non-model category exists.
    _CATEGORY_LABELS = ModelPickerMixin._CATEGORY_LABELS + (
        ("behaviour", "Behaviour", "Memory, typing, reasoning effort and timezone."),
    )

    def __init__(self, cog: "MimicCog", interaction: discord.Interaction):
        super().__init__(cog, interaction, "defaults")
        self.defaults = cog.profile_manager._get_user_defaults(interaction.user.id)
        self.view_mode = "google"
        self.category = "response"
        self.ollama_working = None
        self.cached_ollama_models = []
        self._build_view()

    # --- Mixin contract ---------------------------------------------------

    @property
    def models_state(self) -> Dict[str, Any]:
        """Named for OllamaHostModal, which reaches into a picker to seed its field.

        That modal distinguishes a single-profile picker (which has `profile_name`)
        from a staging one (which has `models_state`). This view is the latter.
        """
        return self.defaults

    def _ollama_host_url(self) -> Optional[str]:
        return self.defaults.get("ollama_host_url")

    def _get_selection_feedback_message(self) -> str:
        """Unused -- this view renders an embed -- but named by the mixin."""
        return ""

    def _picker_render(self) -> Dict[str, Any]:
        return {"content": None, "embed": self.embed(), "view": self}

    def _save_changes(self, key: str, value: Any):
        self.defaults[key] = value
        self._persist()

    # --- Storage ----------------------------------------------------------

    def _persist(self):
        self.cog.profile_manager._save_user_defaults(self.user_id, self.defaults)

    def _clear(self, key: str):
        self.defaults.pop(key, None)
        self._persist()

    # --- Rendering --------------------------------------------------------

    @staticmethod
    def _show(value: Any) -> str:
        if value is None:
            return "`Platform default`"
        if isinstance(value, bool):
            return "**`ON`**" if value else "`OFF`"
        return f"`{ModelPickerMixin.display_model(value)}`"

    def embed(self) -> discord.Embed:
        total = len(defaultable_keys())
        e = discord.Embed(
            title="My Defaults",
            description=(
                "Applied to profiles you **create or borrow from now on**. Existing "
                "profiles are untouched — use `/profile bulk manage` for those.\n"
                "-# Anything left on *Platform default* follows the bot's own value, "
                "including if that value changes later."),
            color=discord.Color.dark_teal())

        if self.category == "behaviour":
            for key, wording, _options in _BEHAVIOUR_CHOICES:
                e.add_field(name=wording, value=self._show(self.defaults.get(key)), inline=True)
            for key, wording in _BEHAVIOUR_TOGGLES:
                e.add_field(name=wording, value=self._show(self.defaults.get(key)), inline=True)
            for key, wording, _ph, _parser in _BEHAVIOUR_NUMBERS:
                e.add_field(name=wording, value=self._show(self.defaults.get(key)), inline=True)
        else:
            labels = model_slot_labels()
            for key, wording, _default in self._CATEGORY_KEYS[self.category]:
                e.add_field(name=labels.get(key, wording),
                            value=self._show(self.defaults.get(key)), inline=True)
            if self.view_mode == "ollama":
                e.add_field(name="Ollama Host",
                            value=self._show(self.defaults.get("ollama_host_url")), inline=True)

        if self.defaults:
            e.set_footer(text=f"{len(self.defaults)} of {total} settings customised")
        else:
            e.set_footer(text=f"Nothing customised — all {total} settings follow the bot.")
        return e

    # --- View -------------------------------------------------------------

    def _build_view(self):
        self.clear_items()
        if self.category in self._GOOGLE_ONLY_CATEGORIES:
            self.view_mode = "google"

        self._add_category_select(0)

        if self.category == "behaviour":
            self._build_behaviour_rows()
        else:
            self._build_model_rows()

        # SettingsBaseView puts the tab bar on row 4 in its constructor, which every
        # rebuild here clears. Re-added last so the row order matches the other tabs.
        self._add_nav_buttons()

    def _build_model_rows(self):
        for offset, (key, wording, _default) in enumerate(self._CATEGORY_KEYS[self.category]):
            self.add_item(self._DefaultModelSelect(
                f"Default {wording}...",
                self._options_with_unset(self.defaults.get(key), key), offset + 1, key))
        self._add_api_buttons(row=3)

    def _options_with_unset(self, current: Any, key: str):
        """The mixin's option list, prefixed with the row's own "not set" choice.

        `_create_model_options` marks whichever option matches `current` as default, so
        passing None when nothing is stored leaves every model unselected and lets this
        one carry the tick.
        """
        options = [discord.SelectOption(
            label="Platform default", value=UNSET,
            description="Follow the bot's own choice, including if it changes.",
            default=current is None)]
        options.extend(self._create_model_options(current, key))
        return options[:25]

    class _DefaultModelSelect(ModelPickerMixin.GenericModelSelect):
        """The shared select, plus the unset option this screen adds."""

        async def callback(self, interaction: discord.Interaction):
            if self.values[0] == UNSET:
                view = self.view
                view._clear(self.target_config_key)
                view._build_view()
                await interaction.response.edit_message(**view._picker_render())
                return
            await super().callback(interaction)

    def _build_behaviour_rows(self):
        for offset, (key, wording, options) in enumerate(_BEHAVIOUR_CHOICES):
            current = self.defaults.get(key)
            select_options = [discord.SelectOption(
                label="Platform default", value=UNSET,
                description="Follow the bot's own choice.", default=current is None)]
            select_options.extend(
                discord.SelectOption(label=label, value=value, description=desc,
                                     default=(current == value))
                for label, value, desc in options)
            select = ui.Select(placeholder=f"Default {wording}...",
                               options=select_options, row=offset + 1)

            async def callback(interaction: discord.Interaction, k=key, s=select):
                if s.values[0] == UNSET:
                    self._clear(k)
                else:
                    self._save_changes(k, s.values[0])
                self._build_view()
                await interaction.response.edit_message(**self._picker_render())

            select.callback = callback
            self.add_item(select)

        for key, wording in _BEHAVIOUR_TOGGLES:
            state = self.defaults.get(key)
            label = (f"{wording}: Default" if state is None
                     else f"{wording}: {'ON' if state else 'OFF'}")
            style = (discord.ButtonStyle.secondary if state is None
                     else (discord.ButtonStyle.success if state else discord.ButtonStyle.danger))
            btn = ui.Button(label=label, style=style, row=3)

            async def toggle(interaction: discord.Interaction, k=key, s=state):
                # Tri-state, matching the bulk picker's fallback indicator: a default
                # has to be removable, not just flippable, or "off" and "I never chose"
                # become the same thing.
                if s is None:
                    self._save_changes(k, True)
                elif s:
                    self._save_changes(k, False)
                else:
                    self._clear(k)
                self._build_view()
                await interaction.response.edit_message(**self._picker_render())

            btn.callback = toggle
            self.add_item(btn)

        numbers = ui.Button(label="Memory & Timezone…", style=discord.ButtonStyle.primary, row=3)

        async def open_numbers(interaction: discord.Interaction):
            await interaction.response.send_modal(DefaultsNumbersModal(self))

        numbers.callback = open_numbers
        self.add_item(numbers)

    async def update_display(self):
        await self.original_interaction.edit_original_response(**self._picker_render())
