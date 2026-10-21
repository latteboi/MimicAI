"""Set Models for a model pair that is no profile slot: the AI Director, the session
synopsis and training analysis.

Its own module because it adopts `ModelPickerMixin` from `gui_profiles`, which imports
`gui_sessions` and `gui_data`: each opens its screen lazily, as `/settings` opens its
defaults screen.
"""
import discord
from discord import ui
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..managers.session_manager import director_enabled, director_model_config
from ..services.generation.compaction import compaction_model_config
from ..utils.constants import NO_FALLBACK
from .base_components import BlockedGuard, add_button
from .gui_profiles import ModelPickerMixin, _keep_wizard_alive

if TYPE_CHECKING:
    from .gui_data import DataManageView
    from .gui_sessions import SessionConfigView


class _SlotPairView(BlockedGuard, ModelPickerMixin, ui.View):
    """A Primary and a Fallback, as Set Models presents a profile's, on the message of the
    screen that opened it. Back returns there.

    Sparse: the model a slot would run anyway is no choice, so the pair keeps following
    `_default` until somebody picks otherwise. A subclass says where a choice lives
    (`_chosen`, `_write`) and what the pair runs without one (`_chain`).
    """

    #: (key, wording) for the Primary, then the Fallback.
    _KEYS: Tuple[Tuple[str, str], Tuple[str, str]]
    _TITLE = ""
    _DESCRIPTION = ""
    _FOOTER = "✏️ chosen for this session · changes save as you make them"
    #: Who the pair belongs to, for the Ollama refusal: "The AI Director".
    _SUBJECT = ""

    def __init__(self, cog, parent, user_id: int):
        super().__init__(timeout=600)
        self.cog = cog
        self.parent = parent
        self.original_interaction = parent.original_interaction
        # Whose key the pair runs on, and so whose Ollama and training-model rules apply.
        self.user_id = user_id
        self.view_mode = self.preferred_api(cog, self.user_id)
        self.category = self._KEYS[0][0]
        self._build_view()

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Keeps the opener's timer alive while this occupies its message, so Back finds it
        listening. Not refreshed for an interaction that is about to be refused."""
        if not await super().interaction_check(interaction):
            return False
        _keep_wizard_alive(self.parent)
        return True

    # --- Where a slot's value lives ------------------------------------------

    def _chain(self) -> Tuple[str, Tuple[str, ...]]:
        """`model_chain`'s answer for the pair with no choice made."""
        raise NotImplementedError

    def _chosen(self, key: str) -> Optional[str]:
        raise NotImplementedError

    def _write(self, key: str, value: Optional[str]):
        """Stores `value` for `key`, or forgets the choice when it is None."""
        raise NotImplementedError

    def _default(self, key: str) -> str:
        primary, fallbacks = self._chain()
        return primary if key == self._KEYS[0][0] else (fallbacks[0] if fallbacks else NO_FALLBACK)

    def _value(self, key: str) -> str:
        return self._chosen(key) or self._default(key)

    def _save_changes(self, key: str, value: Any):
        self._write(key, None if value == self._default(key) else value)

    # --- Mixin contract ------------------------------------------------------

    def _ollama_host_url(self) -> str:
        return ""

    def _get_selection_feedback_message(self) -> str:
        """Unused -- this view renders an embed -- but named by the mixin."""
        return ""

    def _api_modes(self) -> List[str]:
        return ["google", "openrouter"]

    def _allows_no_fallback(self, target_config_key: str) -> bool:
        return target_config_key == self._KEYS[1][0]

    def _tier_applies(self) -> bool:
        """No Hosts & Tier: a tier and a pin are a profile's settings, and this has none."""
        return False

    def refuse_custom_model(self, key: str, value: str) -> Optional[str]:
        """What CustomModelModal asks before the profile rules. See `_api_modes`."""
        if value.startswith("OLLAMA/"):
            return f"{self._SUBJECT} runs on Google or OpenRouter models, not Ollama."
        return None

    # --- Rendering -----------------------------------------------------------

    def _status_fields(self, e: discord.Embed):
        """Fields ahead of the two models'."""

    def _add_buttons(self, row: int):
        """Buttons ahead of Reset and Back."""

    async def _back(self, i: discord.Interaction):
        raise NotImplementedError

    def embed(self) -> discord.Embed:
        e = discord.Embed(title=self._TITLE, colour=discord.Colour.gold(), description=self._DESCRIPTION)
        self._status_fields(e)
        for key, wording in self._KEYS:
            mark = " ✏️" if self._chosen(key) else ""
            e.add_field(name=f"{wording}{mark}", value=f"`{self.display_model(self._value(key))}`",
                        inline=True)
        self._add_openrouter_details(e, [(wording, self._value(key)) for key, wording in self._KEYS])
        e.set_footer(text=self._FOOTER)
        return e

    def _build_view(self):
        self.clear_items()
        if self.view_mode not in self._api_modes():
            self.view_mode = self._api_modes()[0]

        row = 0
        if self._shows_openrouter_browse():
            self._add_openrouter_browse_select(row)
            row += 1
        for key, wording in self._KEYS:
            self.add_item(self.GenericModelSelect(
                f"Select {wording} Model...", self._create_model_options(self._value(key), key), row, key))
            row += 1

        self._add_api_buttons(row=row)
        self._add_buttons(row)

        async def reset_cb(i: discord.Interaction):
            for key, _wording in self._KEYS:
                self._write(key, None)
            self._build_view()
            await i.response.edit_message(**self._picker_render())

        add_button(self, "Reset to Default", reset_cb, row=row,
                   disabled=not any(self._chosen(key) for key, _wording in self._KEYS))
        add_button(self, "◀ Back", self._back, row=row)


class _SessionPairView(_SlotPairView):
    """A pair kept in the channel's session, billed to its owner's key."""

    def __init__(self, cog, parent: 'SessionConfigView'):
        super().__init__(cog, parent, parent.session.get("owner_id") or parent.original_interaction.user.id)

    @property
    def session(self) -> dict:
        """The channel's live session, not the one this opened on -- see `update_display`."""
        return (self.cog.multi_profile_channels.get(self.original_interaction.channel_id)
                or self.parent.session)

    def _chain(self):
        """The LTM summariser's, under the owner's provider: what a session pair runs unchosen."""
        return self.cog.api_service.model_chain({}, "ltm_model", self.user_id)

    async def _back(self, i: discord.Interaction):
        await i.response.defer()
        await self.parent.update_display()


class DirectorModelView(_SessionPairView):
    """`/session config` -> Proactivity -> Director Model. Written to `session['proactivity']`
    as chosen; until then the Director follows the LTM summariser (`director_model_config`)."""

    _KEYS = (("director_model", "Primary"), ("director_fallback_model", "Fallback"))
    _TITLE = "AI Director Model"
    _DESCRIPTION = ("The model that writes the Director's note for each proactive round. Until "
                    "you choose one it runs the LTM Summariser's models, which follow the session "
                    "owner's provider. Choosing a model does not turn the Director on.")
    _SUBJECT = "The AI Director"

    def _pro(self) -> dict:
        """The session's proactivity settings, with the switch written down first: a session
        from before this screen holds it in the model field, and a model chosen here must
        not turn a Director that was off on."""
        pro = self.session.setdefault("proactivity", {})
        pro.setdefault("director_enabled", director_enabled(pro))
        return pro

    def _chosen(self, key: str) -> Optional[str]:
        return director_model_config(self.session.get("proactivity"))[
            "ltm_model" if key == "director_model" else "ltm_fallback_model"]

    def _write(self, key: str, value: Optional[str]):
        pro = self._pro()
        if value is None:
            pro.pop(key, None)
        else:
            pro[key] = value
        self.cog.session_manager._save_multi_profile_sessions()

    def _status_fields(self, e: discord.Embed):
        enabled = director_enabled(self.session.get("proactivity"))
        e.add_field(name="Director", value="**`ON`**" if enabled else "`OFF`", inline=True)

    def _add_buttons(self, row: int):
        enabled = director_enabled(self.session.get("proactivity"))

        async def toggle_cb(i: discord.Interaction):
            self._pro()["director_enabled"] = not enabled
            self.cog.session_manager._save_multi_profile_sessions()
            self._build_view()
            await i.response.edit_message(**self._picker_render())

        add_button(self, f"Director: {'ON' if enabled else 'OFF'}", toggle_cb, row=row,
                   style=discord.ButtonStyle.success if enabled else discord.ButtonStyle.danger)


class CompactionModelView(_SessionPairView):
    """`/session config` -> Compaction -> Summariser Model. Written to `session['compaction']`
    as chosen; until then the synopsis runs its own shipped chain (`compaction_model_config`)."""

    _KEYS = (("compaction_model", "Primary"), ("compaction_fallback_model", "Fallback"))
    #: Where each key is kept in `session['compaction']`.
    _STORED = {"compaction_model": "model", "compaction_fallback_model": "fallback_model"}
    _TITLE = "Synopsis Summariser Model"
    _DESCRIPTION = ("The model that folds old turns into the session synopsis, and the one tried "
                    "when it fails. Until you choose, they are the synopsis's own defaults for "
                    "the session owner's provider.")
    _SUBJECT = "The session synopsis"

    def _chain(self):
        return self.cog.api_service.model_chain({}, "compaction_model", self.user_id)

    def refuse_custom_model(self, key: str, value: str) -> Optional[str]:
        """Typing an Ollama model stays open, as the field this replaces was: CustomModelModal
        still refuses it to anyone but the bot owner."""
        return None

    def _chosen(self, key: str) -> Optional[str]:
        return compaction_model_config(self.session)[key]

    def _write(self, key: str, value: Optional[str]):
        cfg = self.session.setdefault("compaction", {})
        if value is None:
            cfg.pop(self._STORED[key], None)
        else:
            cfg[self._STORED[key]] = value
        # A new model, so a summariser that failed before is tried at the next round end.
        self.session.pop("_compaction_retry_at", None)
        self.cog.session_manager._save_multi_profile_sessions()


class TrainingAnalysisView(_SlotPairView):
    """Training Examples -> Analyse: the profile's Training Analysis models, the same slots Set
    Models shows, then Analyse... for the run's size. Unchosen, the category's shipped chain
    under the owner's provider."""

    _KEYS = (("training_analysis_model", "Primary"), ("training_analysis_fallback_model", "Fallback"))
    _TITLE = "Analyse Training Examples"
    _DESCRIPTION = ("Writes a style guide from this profile's newest training examples, shown "
                    "under Training \u2192 Style Guide. The models are this profile's Training "
                    "Analysis models, also under Set Models.")
    _FOOTER = "✏️ chosen for this profile · changes save as you make them"
    _SUBJECT = "Training analysis"

    def __init__(self, cog, parent: 'DataManageView'):
        self.profile_name = parent.profile_name
        super().__init__(cog, parent, parent.user_id)

    def refuse_custom_model(self, key: str, value: str) -> Optional[str]:
        """As Set Models takes this slot: CustomModelModal keeps Ollama to the bot owner."""
        return None

    def _config(self) -> Dict[str, Any]:
        return self.cog.profile_manager._get_profile_config(self.user_id, self.profile_name, False) or {}

    def _chain(self):
        return self.cog.api_service.model_chain({}, "training_analysis_model", self.user_id)

    def _chosen(self, key: str) -> Optional[str]:
        """What the profile holds, unless it is the default: an OpenRouter profile is created
        holding its provider's pair, which is no choice of anyone's."""
        value = self._config().get(key)
        return value if value and value != self._default(key) else None

    def _write(self, key: str, value: Optional[str]):
        config = self._config()
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value
        self.cog.profile_manager._save_profile_config(self.user_id, self.profile_name, config, False)

    def _add_buttons(self, row: int):
        from .gui_data import AnalyseExamplesModal

        async def analyse_cb(i: discord.Interaction):
            await i.response.send_modal(AnalyseExamplesModal(self))

        add_button(self, "Analyse…", analyse_cb, row=row, style=discord.ButtonStyle.blurple)

    async def _back(self, i: discord.Interaction):
        await self.parent._show(i)
