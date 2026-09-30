"""`/session config` -> Proactivity -> Director Model: Set Models for the AI Director.

Its own module because it adopts `ModelPickerMixin` from `gui_profiles`, which imports
`gui_sessions`: the editor opens this lazily, as `/settings` opens its defaults screen.
"""
import discord
from discord import ui
from typing import TYPE_CHECKING, Any, List, Optional

from ..managers.session_manager import director_enabled, director_model_config
from ..utils.constants import NO_FALLBACK
from .base_components import BlockedGuard, add_button
from .gui_profiles import ModelPickerMixin, _keep_wizard_alive

if TYPE_CHECKING:
    from .gui_sessions import SessionConfigView

_KEYS = (("director_model", "Primary"), ("director_fallback_model", "Fallback"))


class DirectorModelView(BlockedGuard, ModelPickerMixin, ui.View):
    """The Director's two models, as Set Models presents a profile's, on the editor's message.

    Written to `session['proactivity']` as they are chosen, and sparse: the model the slot
    would run anyway is no override, so the Director keeps following the LTM summariser's
    chain (`director_model_config`) until somebody picks otherwise. Back returns to the tab.
    """

    #: OpenRouter and Google only: the field this replaces never took an Ollama model.
    _NO_OLLAMA_CATEGORIES = ("director",)

    def __init__(self, cog, parent: 'SessionConfigView'):
        super().__init__(timeout=600)
        self.cog = cog
        self.parent = parent
        self.original_interaction = parent.original_interaction
        # Whose key the Director runs on, and so whose Ollama and training-model rules apply.
        self.user_id = parent.session.get("owner_id") or parent.original_interaction.user.id
        self.view_mode = self.preferred_api(cog, self.user_id)
        self.category = "director"
        self._build_view()

    @property
    def session(self) -> dict:
        """The channel's live session, not the one this opened on -- see `update_display`."""
        return (self.cog.multi_profile_channels.get(self.original_interaction.channel_id)
                or self.parent.session)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Keeps the editor's timer alive while this occupies its message, so Back finds it
        listening. Not refreshed for an interaction that is about to be refused."""
        if not await super().interaction_check(interaction):
            return False
        _keep_wizard_alive(self.parent)
        return True

    # --- Where a slot's value lives ------------------------------------------

    def _pro(self) -> dict:
        """The session's proactivity settings, with the switch written down first: a session
        from before this screen holds it in the model field, and a model chosen here must
        not turn a Director that was off on."""
        pro = self.session.setdefault("proactivity", {})
        pro.setdefault("director_enabled", director_enabled(pro))
        return pro

    def _default(self, key: str) -> str:
        """The model a slot with no choice runs: the LTM summariser's, under the owner's provider."""
        primary, fallbacks = self.cog.api_service.model_chain(
            {"final_fallback_enabled": False}, "ltm_model", self.user_id)
        return primary if key == "director_model" else (fallbacks[0] if fallbacks else NO_FALLBACK)

    def _chosen(self, key: str) -> Optional[str]:
        return director_model_config(self.session.get("proactivity"))[
            "ltm_model" if key == "director_model" else "ltm_fallback_model"]

    def _value(self, key: str) -> str:
        return self._chosen(key) or self._default(key)

    def _save_changes(self, key: str, value: Any):
        pro = self._pro()
        if value == self._default(key):
            pro.pop(key, None)
        else:
            pro[key] = value
        self.cog.session_manager._save_multi_profile_sessions()

    # --- Mixin contract ------------------------------------------------------

    def _ollama_host_url(self) -> str:
        return ""

    def _get_selection_feedback_message(self) -> str:
        """Unused -- this view renders an embed -- but named by the mixin."""
        return ""

    def _api_modes(self) -> List[str]:
        return ["google", "openrouter"]

    def _allows_no_fallback(self, target_config_key: str) -> bool:
        return target_config_key == "director_fallback_model"

    def _tier_applies(self) -> bool:
        """No Hosts & Tier: a tier and a pin are a profile's settings, and this has none."""
        return False

    def refuse_custom_model(self, key: str, value: str) -> Optional[str]:
        """What CustomModelModal asks before the profile rules. See `_api_modes`."""
        if value.startswith("OLLAMA/"):
            return "The AI Director runs on Google or OpenRouter models, not Ollama."
        return None

    # --- Rendering -----------------------------------------------------------

    def embed(self) -> discord.Embed:
        enabled = director_enabled(self.session.get("proactivity"))
        e = discord.Embed(
            title="AI Director Model", colour=discord.Colour.gold(),
            description=("The model that writes the Director's note for each proactive round. "
                         "Until you choose one it runs the LTM Summariser's models, which follow "
                         "the session owner's provider. Choosing a model does not turn the "
                         "Director on."))
        e.add_field(name="Director", value="**`ON`**" if enabled else "`OFF`", inline=True)
        for key, wording in _KEYS:
            mark = " ✏️" if self._chosen(key) else ""
            e.add_field(name=f"{wording}{mark}", value=f"`{self.display_model(self._value(key))}`",
                        inline=True)
        self._add_openrouter_details(e, [(wording, self._value(key)) for key, wording in _KEYS])
        e.set_footer(text="✏️ chosen for this session · changes save as you make them")
        return e

    def _build_view(self):
        self.clear_items()
        if self.view_mode not in self._api_modes():
            self.view_mode = self._api_modes()[0]

        row = 0
        if self._shows_openrouter_browse():
            self._add_openrouter_browse_select(row)
            row += 1
        for key, wording in _KEYS:
            self.add_item(self.GenericModelSelect(
                f"Select {wording} Model...", self._create_model_options(self._value(key), key), row, key))
            row += 1

        self._add_api_buttons(row=row)

        enabled = director_enabled(self.session.get("proactivity"))

        async def toggle_cb(i: discord.Interaction):
            self._pro()["director_enabled"] = not enabled
            self.cog.session_manager._save_multi_profile_sessions()
            self._build_view()
            await i.response.edit_message(**self._picker_render())

        add_button(self, f"Director: {'ON' if enabled else 'OFF'}", toggle_cb, row=row,
                   style=discord.ButtonStyle.success if enabled else discord.ButtonStyle.danger)

        async def reset_cb(i: discord.Interaction):
            for key, _wording in _KEYS:
                self._pro().pop(key, None)
            self.cog.session_manager._save_multi_profile_sessions()
            self._build_view()
            await i.response.edit_message(**self._picker_render())

        add_button(self, "Reset to Default", reset_cb, row=row,
                   disabled=not any(self._chosen(key) for key, _wording in _KEYS))

        async def back_cb(i: discord.Interaction):
            await i.response.defer()
            await self.parent.update_display()

        add_button(self, "◀ Back", back_cb, row=row)
