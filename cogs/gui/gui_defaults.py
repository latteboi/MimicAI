"""The `/settings` -> Override Defaults tab: standing preferences for profiles yet to exist.

Off until the user chooses a provider, here or in `/start`: until then a new profile is
made with no models and nothing on this screen is applied (`provider_preference`).

Every profile used to start from one hardcoded template, and every borrow from a copy
of whoever wrote it. Someone who preferred a different model set it again on profile
one, profile two and profile forty, and a borrow could arrive configured for a
provider they had no key for and simply refuse to speak.

This screen knows no settings of its own. It is the bulk manager's action step, with
one user's `index.json["defaults"]` where the selected profiles would be: the tabs are
`PROFILE_TABS`, the rows are `PROFILE_ACTIONS`, and choosing one opens the same
`_Bulk` form `/profile bulk manage` opens -- see `_ActionStagingHost`. It used to carry
a hand-written "Behaviour" category holding two dropdowns, four toggles and a modal,
which is eight of the sixty-odd keys `defaultable_keys()` says may be defaulted; the
other fifty-odd were settable on one profile, settable on forty, and not settable as a
preference. Nothing here has to be extended for a new setting, because nothing here
names one.

The exception is the model rows, which keep their own screen: `ModelApplyView` stages
a slot or leaves it alone, and this screen needs a third state per slot -- *clear this
back to the platform default* -- which only makes sense where the values are stored
rather than staged.

Values are written the moment they are chosen. The neighbouring API Keys tab stages
and waits for Save Assignments, and forgetting that button is the single most common
way a key ends up doing nothing; there is no equivalent risk here, because every
setting on this screen is reversible, sparse, and affects only profiles made later.
"""

import discord
from discord import ui
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .base_components import add_button, add_select
from ..utils.helpers import resolve_openrouter_endpoint
from ..utils.constants import MODEL_PROVIDERS
from ..utils.user_defaults import defaultable_keys, model_slot_labels, other_provider, setting_label
from .gui_profiles import (
    PROFILE_ACTIONS, PROFILE_ACTIONS_BY_VALUE, PROFILE_TABS, _ActionStagingHost,
    _BulkSession, _TAB_BUTTONS_PER_ROW, _select_chunks, ModelPickerMixin,
    OpenRouterHostView,
)
from .gui_settings import PROVIDER_PREFERENCE_NOTE, SettingsBaseView, provider_options

if TYPE_CHECKING:
    from ..MimicCog import MimicCog


#: The "not set" sentinel used by every model select on this screen. Distinct from any
#: stored value, because clearing a default and setting it to whatever the bot
#: currently ships are different acts -- the first tracks a future change to the
#: shipped value, the second pins the profile to today's.
UNSET = "__unset__"

#: Lines of stored values one row's embed field shows before it summarises the rest.
#: Thinking alone declares twenty-odd keys, and an embed field caps at 1024 characters.
_ROW_VALUE_LINES = 4


class _SparseSeed(dict):
    """The values a bulk modal opens on here: what is stored, and blank for the rest.

    A bulk modal falls back to the value the bot ships when a key is absent, which is
    right against a profile -- that is what the profile is running -- and wrong against
    a preference. A box pre-filled with today's shipped number and submitted *stores*
    today's shipped number, so someone who came to set the temperature would leave with
    Top P, Top K and the short-term memory length pinned to whatever the bot happened to
    ship that day, and a later change to any of them would reach them never. Sparse is
    the whole point of this screen -- see `utils/user_defaults` -- so an unset key opens
    blank and parses back out as unset.

    Only the scalar a text box is built from is blanked. A structured default is handed
    over as the factory asked for it, because the factory reads *through* it rather than
    printing it: the neuro modal takes `neuro_state` and then asks it for four numbers.
    """

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default if isinstance(default, (dict, list, tuple)) else ""


class SettingsDefaultsView(_ActionStagingHost, ModelPickerMixin, SettingsBaseView):
    """Fourth adopter of ModelPickerMixin, and second host of the `_Bulk` forms.

    The mixin's contract is `view_mode`, `category`, `ollama_working`, `_build_view`,
    `_get_selection_feedback_message` and `_ollama_host_url`; the host's is `session`,
    `current_action`, `refresh` and `_stage_change`. Adopting both rather than
    reimplementing either is what keeps the Google-only slots enforced here too, and
    what stops a setting being reachable in bulk and not as a preference.
    """

    #: `TabbedView.current_tab` is the *settings* tab this screen is ("defaults"), so
    #: the PROFILE_TABS strip needs a name of its own.
    _DEFAULT_TAB = "params"

    #: Row 4 is the settings tab bar, so there is no row for OpenRouter's Browse dropdown
    #: here: the model rows page through Most Popular instead.
    _BROWSE_ROW_AVAILABLE = False
    #: One fewer than the pickers, for the "Platform default" row this screen adds on top.
    _OPENROUTER_MODELS_PER_PAGE = ModelPickerMixin._OPENROUTER_MODELS_PER_PAGE - 1

    _EXPIRED_NOTICE = ("This settings panel has expired, so that default was not saved. "
                       "Run `/settings` again.")

    def __init__(self, cog: "MimicCog", interaction: discord.Interaction):
        self.defaults = cog.profile_manager._get_user_defaults(interaction.user.id)
        # One object, not a copy: every `_Bulk` form seeds itself from `session.config`
        # -- the voice picker asks it which speech model it is choosing a voice for, the
        # timezone picker asks it which zone is already chosen -- and against this host
        # the answer to "what is staged" is "what is stored".
        self.session = _BulkSession()
        self.session.config = self.defaults
        self.step = "actions"
        self.tab = self._DEFAULT_TAB
        self.current_action = None
        self._choice = None
        self._clear_picks: set = set()
        #: What the last stage could not keep, named for the embed. See `_stage_change`.
        self._notice: Optional[str] = None
        self.view_mode = self.preferred_api(cog, interaction.user.id)
        self.category = "response"
        self.ollama_working = None
        super().__init__(cog, interaction, "defaults")
        self._build_view()

    # --- Which rows this screen offers ------------------------------------

    def _keys(self, action) -> tuple:
        """The row's config keys that may be defaulted, in declaration order."""
        allowed = defaultable_keys()
        return tuple(k for k in action.bulk.keys if k in allowed)

    def _offered(self, action) -> bool:
        """Whether a row can express a preference at all.

        Three kinds fall out on their own, with no list naming them. A terminal row
        deletes files. A prompt-only row writes something a borrow does not own, which
        `defaultable_keys` already refuses. And a row whose every key is denied --
        the 18+ declaration, the neuro engine's live hormone vector -- has nothing
        left to store. What remains is gated on `_Bulk.needs` exactly as in bulk, so
        the voice picker appears once a speech model is the user's default and not
        before: a voice list means nothing without the model that offers it.
        """
        return (action.bulk is not None and not action.bulk.terminal
                and bool(self._keys(action))
                and all(self.defaults.get(k) for k in action.bulk.needs))

    def _tabs(self) -> List[str]:
        return [t for t in PROFILE_TABS
                if any(a.tab == t and self._offered(a) for a in PROFILE_ACTIONS)]

    def _rows(self, tab: str) -> List:
        return [a for a in PROFILE_ACTIONS if a.tab == tab and self._offered(a)]

    def _set_rows(self, tab: Optional[str] = None) -> List:
        """Offered rows the user has actually set something on, all or one tab's."""
        return [a for a in PROFILE_ACTIONS if self._offered(a)
                and any(k in self.defaults for k in self._keys(a))
                and (tab is None or a.tab == tab)]

    # --- Host contract ----------------------------------------------------

    def staging_footer(self) -> str:
        return "Saved as you choose · applies to profiles you create or borrow from now on"

    def modal_seed(self, current):
        """Open a bulk modal on what is stored, and blank elsewhere. See `_SparseSeed`."""
        return _SparseSeed(self.defaults) if isinstance(current, dict) else current

    def _stage_change(self, action_value: str, config: Optional[Dict] = None,
                      prompts: Optional[Dict] = None, declaration: Optional[bool] = None,
                      pins: Optional[Dict] = None):
        """Writes one row's output into the stored defaults, and says what it could not.

        A `_Bulk` form hands back everything its screen collected, and against a
        selection of profiles all of it is writable. Here three kinds are not: a prompt
        (a borrow stores none, so a default prompt is a write nothing reads), a content
        rating, and a key `DEFAULT_DENY` holds back. Dropping them silently would leave
        someone who typed an image-generation prompt on this screen with no prompt and
        no message, so the row that ignored something says so on the panel it returns to.

        `None` is not stored. It is how a modal says "send no value", which is exactly
        what an absent default already means -- keeping it would count as a customised
        setting, and write a null onto every profile made afterwards.
        """
        allowed = defaultable_keys()
        ignored = []
        for key, value in (config or {}).items():
            if key not in allowed:
                ignored.append(setting_label(key))
            elif value is None:
                self.defaults.pop(key, None)
            else:
                self.defaults[key] = value
        ignored += [setting_label(k) for k in (prompts or {})]
        ignored += [setting_label(k) for k in (pins or {})]
        if declaration is not None:
            ignored.append("Adult 18+ Declaration")

        self._persist()
        action = PROFILE_ACTIONS_BY_VALUE.get(action_value)
        label = action.bulk_label() if action else action_value
        self._notice = (
            f"**{label}** — {self._join(ignored)} "
            f"{'were' if len(ignored) > 1 else 'was'} not saved here. "
            f"{'They belong' if len(ignored) > 1 else 'It belongs'} to one character, so "
            f"{'they are' if len(ignored) > 1 else 'it is'} set on that profile rather "
            f"than on every profile you go on to make."
            if ignored else None)
        self._choice = None
        self.step = "actions"

    @staticmethod
    def _join(names: List[str]) -> str:
        names = sorted(set(names))
        if len(names) == 1:
            return f"**{names[0]}**"
        return ", ".join(f"**{n}**" for n in names[:-1]) + f" and **{names[-1]}**"

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

    def _get_current_profile_data(self) -> Dict[str, Any]:
        """Named for OpenRouterHostView, which reads and writes its parent's config
        through this. Here that config is the stored defaults themselves -- a pin is an
        ordinary defaultable key, so it is kept exactly as a model choice is."""
        return self.defaults

    def _pinnable_slots(self) -> List[tuple]:
        """(config key, wording, stored default) for each slot on this tab a pin could
        reach. A slot left on the platform default reads as None, which the Hosts screen
        renders as having no model to pin rather than as an empty one."""
        return [(key, wording, self.defaults.get(key))
                for key, wording, _default in self._CATEGORY_KEYS[self.category]
                if key not in self._NON_CHAT_MODEL_KEYS]

    def _add_openrouter_buttons(self, *, row: int):
        """Hosts & Tier, onto DefaultsOpenRouterHostView.

        The tier used to sit on this row as a cycling button of its own, because there
        was no host list to put it beside: a preference held no model to ask OpenRouter
        about. It does now -- a default model is a model -- and the two answer the same
        question, so this screen asks it the same way both pickers do.
        """
        lit = self._current_service_tier() is not None or any(
            resolve_openrouter_endpoint(self.defaults, m)
            for m in self._pinnable_openrouter_models())
        self._add_hosts_button(row=row, view_cls=DefaultsOpenRouterHostView, lit=lit)

    #: This screen's fourth state is an absent default, not a pending one.
    _TIER_UNSET_WORDING = "Platform default"

    def _tier_cycle(self) -> tuple:
        return (None, "", "flex", "priority")

    def _current_service_tier(self):
        return self.defaults.get("openrouter_service_tier")

    def _set_service_tier(self, value):
        """`None` clears rather than stores, the same rule every other row here follows.

        A stored None would count towards "n of m settings customised" and would be
        written onto new profiles as a null, which is not what "I never chose" means.
        """
        if value is None:
            self._clear("openrouter_service_tier")
        else:
            self._save_changes("openrouter_service_tier", value)

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
        if value == "":
            return "`None`"
        if isinstance(value, dict):
            return f"`{len(value)} entr{'y' if len(value) == 1 else 'ies'}`"
        return f"`{ModelPickerMixin.display_model(value)}`"

    def _row_value(self, action) -> str:
        """One row's stored settings, or that it has none."""
        lines = [f"{setting_label(k)}: {self._show(self.defaults[k])}"
                 for k in self._keys(action) if k in self.defaults]
        if not lines:
            return "`Platform default`"
        if len(lines) > _ROW_VALUE_LINES:
            extra = len(lines) - _ROW_VALUE_LINES
            lines = lines[:_ROW_VALUE_LINES] + [f"-# …and {extra} more"]
        return "\n".join(lines)[:1024]

    def embed(self) -> discord.Embed:
        return getattr(self, f"_embed_{self.step}")()

    def _footer(self) -> str:
        total = len(defaultable_keys())
        if not self.defaults:
            return f"Nothing customised — all {total} settings follow the bot."
        return f"{len(self.defaults)} of {total} settings customised"

    def _base_embed(self, title: str, description: str) -> discord.Embed:
        e = discord.Embed(title=title, description=description,
                          color=discord.Color.dark_teal())
        e.set_footer(text=self._footer())
        return e

    def _preferred(self) -> Optional[str]:
        preferred = self.cog.profile_manager.provider_preference(self.user_id)
        return preferred if preferred in MODEL_PROVIDERS else None

    def _embed_actions(self) -> discord.Embed:
        if not self._preferred():
            return self._base_embed(
                "Override Defaults",
                "**Off.** Profiles you make start with no models, and nothing here is "
                "applied, until you choose a provider below (or in `/start`). An API key "
                "alone does not choose one.")
        e = self._base_embed(
            "Override Defaults",
            "Applied to profiles you **create**, and offered on those you **borrow**, from "
            "now on. Existing profiles are untouched — use `/profile bulk manage` for those.\n"
            "-# Anything left on *Platform default* follows the bot's own value for your "
            "provider, including if that value changes later. The tabs and the settings "
            "under them are the ones on a profile's own dashboard.")
        e.add_field(name="Provider", value=f"`{MODEL_PROVIDERS[self._preferred()]}` — "
                                           f"{PROVIDER_PREFERENCE_NOTE}"[:1024], inline=False)
        rows = self._rows(self.tab)
        for action in rows:
            e.add_field(name=action.bulk_label(), value=self._row_value(action), inline=True)
        if not rows:
            e.add_field(name="Nothing here", value="No setting on this tab can be "
                                                   "defaulted.", inline=False)
        if self._notice:
            e.add_field(name="Not saved", value=self._notice[:1024], inline=False)
        return e

    def _embed_models(self) -> discord.Embed:
        labels = model_slot_labels()
        slots = self._CATEGORY_KEYS[self.category]
        wording = next((lbl for value, lbl, _d in self._CATEGORY_LABELS
                        if value == self.category), self.category.title())
        e = self._base_embed(
            "Override Defaults — Set Models",
            f"**{wording}** — the model a profile you make from now on starts on. "
            f"Switch categories for the rest.\n"
            f"-# *Platform default* on a slot follows the bot's own choice, including "
            f"if it changes.")
        for key, fallback_wording, _default in slots:
            e.add_field(name=labels.get(key, fallback_wording),
                        value=self._show(self.defaults.get(key)), inline=True)
        if self.view_mode == "ollama":
            e.add_field(name="Ollama Host",
                        value=self._show(self.defaults.get("ollama_host_url")), inline=True)
        if self.view_mode == "openrouter" and self._tier_applies():
            tier = self.defaults.get("openrouter_service_tier")
            e.add_field(name="Service Tier",
                        value=("`Platform default`" if tier is None
                               else f"`{self.tier_wording(tier)[1]}`"), inline=True)
        self._add_openrouter_details(
            e, [(labels.get(key, w), self.defaults.get(key)) for key, w, _d in slots])
        return e

    def _embed_choice(self) -> discord.Embed:
        action = PROFILE_ACTIONS_BY_VALUE.get(self._choice["action"]) if self._choice else None
        lead = f"{action.bulk_description()}\n\n" if (action and action.bulk_description()) else ""
        return self._base_embed(
            f"Override Defaults — {action.bulk_label() if action else 'Choose a value'}",
            f"{lead}Pick a value; it is saved and you return to the settings list.")

    def _embed_clear(self) -> discord.Embed:
        rows = self._set_rows()
        e = self._base_embed(
            "Override Defaults — Clear",
            "Tick anything that should go back to following the bot's own value.\n"
            "-# Clearing is not the same as choosing the value the bot ships today: a "
            "cleared setting tracks that value if it ever changes.")
        if not rows:
            e.description = ("Nothing is customised, so there is nothing to clear. "
                             "Every setting already follows the bot.")
            return e
        picked = [PROFILE_ACTIONS_BY_VALUE[v] for v in self._clear_picks
                  if v in PROFILE_ACTIONS_BY_VALUE]
        if picked:
            e.add_field(
                name=f"Selected ({len(picked)})",
                value="\n".join(f"• {a.bulk_label()}"
                                for a in sorted(picked, key=lambda a: a.bulk_label()))[:1024],
                inline=False)
        return e

    # --- View -------------------------------------------------------------

    def _build_view(self):
        self.clear_items()
        getattr(self, f"_build_{self.step}_step")()
        # SettingsBaseView puts the tab bar on row 4 in its constructor, which every
        # rebuild here clears. Re-added last so the row order matches the other tabs.
        self._add_nav_buttons()

    def _build_actions_step(self):
        if not self._preferred():
            add_select(self, provider_options(None), self._set_provider,
                       placeholder="Choose a provider to turn defaults on...", row=0)
            return
        tabs = self._tabs()
        if self.tab not in tabs and tabs:
            self.tab = tabs[0]

        rows = self._rows(self.tab)
        if rows:
            options = []
            for action in rows:
                label = action.bulk_label()
                if any(k in self.defaults for k in self._keys(action)):
                    label = f"✓ {label}"
                options.append(discord.SelectOption(
                    label=label[:100], value=action.value,
                    description=(action.bulk_description() or "")[:100] or None))
            add_select(self, options, self._action_callback,
                       placeholder=f"Choose a {self.tab.title()} default…", row=0)

        nav_row = self._add_tab_row(tabs)
        add_button(self, "Clear…", self._nav("clear"), style=discord.ButtonStyle.secondary,
                   row=nav_row, disabled=not self._set_rows())
        add_button(self, f"Provider: {MODEL_PROVIDERS[self._preferred()]}", self._set_provider,
                   style=discord.ButtonStyle.secondary, row=nav_row, emoji="🔀")

    def _add_tab_row(self, tabs, row: int = 1) -> int:
        """The PROFILE_TABS strip, wrapping exactly as the dashboard's and the wizard's
        do. Returns the first row below it."""
        if not tabs:
            return row
        for position, tab in enumerate(tabs):
            add_button(self, tab.title(), self._pick_tab(tab),
                       row=row + position // _TAB_BUTTONS_PER_ROW,
                       disabled=(tab == self.tab),
                       style=(discord.ButtonStyle.primary if tab == self.tab
                              else discord.ButtonStyle.secondary))
        return row + (len(tabs) + _TAB_BUTTONS_PER_ROW - 1) // _TAB_BUTTONS_PER_ROW

    def _build_models_step(self):
        if self.category in self._GOOGLE_ONLY_CATEGORIES:
            self.view_mode = "google"
        self._add_category_select(0)
        for offset, (key, wording, _default) in enumerate(self._CATEGORY_KEYS[self.category]):
            self.add_item(self._DefaultModelSelect(
                f"Default {wording}...",
                self._options_with_unset(self.defaults.get(key), key), offset + 1, key))
        self._add_api_buttons(row=3)
        add_button(self, "◀ Back", self._nav("actions"), style=discord.ButtonStyle.secondary,
                   row=3)

    def _build_choice_step(self):
        for option in self._choice["options"]:
            option.default = (option.value == self._choice["chosen"])
        chunks = _select_chunks(self._choice["options"])
        for part, chunk in enumerate(chunks):
            placeholder = self._choice["placeholder"]
            if len(chunks) > 1:
                placeholder = f"{placeholder} ({part + 1}/{len(chunks)})"
            add_select(self, chunk, self._choice_callback, placeholder=placeholder, row=part)
        add_button(self, "◀ Back", self._nav("actions"), style=discord.ButtonStyle.secondary,
                   row=len(chunks))

    def _build_clear_step(self):
        """Tab-scoped, as the wizard's inherit step is, and for the same reason: one
        select owning one tab's rows can never overflow Discord's 25 options, and a row
        deselected on one tab cannot be read as a row deselected everywhere."""
        tabs = [t for t in PROFILE_TABS if self._set_rows(t)]
        if self.tab not in tabs and tabs:
            self.tab = tabs[0]

        rows = self._set_rows(self.tab)
        if rows:
            options = [discord.SelectOption(
                label=a.bulk_label()[:100], value=a.value,
                description=f"{sum(1 for k in self._keys(a) if k in self.defaults)} set",
                default=(a.value in self._clear_picks)) for a in rows]
            add_select(self, options, self._clear_callback,
                       placeholder=f"Choose {self.tab.title()} settings to clear…",
                       min_values=0, max_values=len(options), row=0)

        nav_row = self._add_tab_row(tabs)
        add_button(self, "◀ Back", self._nav("actions"), style=discord.ButtonStyle.secondary,
                   row=nav_row)
        add_button(self, f"Clear Selected ({len(self._clear_picks)})", self._clear_selected,
                   style=discord.ButtonStyle.danger, row=nav_row,
                   disabled=not self._clear_picks)
        add_button(self, "Clear Everything", self._clear_all, style=discord.ButtonStyle.danger,
                   row=nav_row, disabled=not self.defaults)

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

    # --- Navigation -------------------------------------------------------

    async def refresh(self, interaction: discord.Interaction):
        self._build_view()
        await interaction.response.edit_message(**self._picker_render())

    async def update_display(self):
        await self.original_interaction.edit_original_response(**self._picker_render())

    def _nav(self, step: str):
        async def callback(interaction: discord.Interaction):
            if step != "clear":
                self._clear_picks.clear()
            self._notice = None
            self._choice = None
            self.step = step
            await self.refresh(interaction)
        return callback

    def _pick_tab(self, tab: str):
        async def callback(interaction: discord.Interaction):
            self.tab = tab
            self._notice = None
            await self.refresh(interaction)
        return callback

    async def _set_provider(self, interaction: discord.Interaction):
        """The dropdown turns defaults on; the button beside Clear swaps provider."""
        values = (interaction.data or {}).get("values")
        provider = values[0] if values else other_provider(self._preferred())
        self.cog.profile_manager.set_provider_preference(self.user_id, provider)
        self.view_mode = self.preferred_api(self.cog, self.user_id)
        await self.refresh(interaction)

    async def _action_callback(self, interaction: discord.Interaction):
        action = PROFILE_ACTIONS_BY_VALUE.get(interaction.data["values"][0])
        if action is None or not self._offered(action):
            await self.refresh(interaction)
            return
        self._notice = None
        self.current_action = action
        # Set Models is the one row with a screen of its own here: only stored values
        # can offer "clear this slot", which a staging picker has no state for.
        if action.value == "models":
            self.step = "models"
            await self.refresh(interaction)
            return
        await action.bulk.run(self, interaction)

    async def _choice_callback(self, interaction: discord.Interaction):
        value = interaction.data["values"][0]
        self._choice["chosen"] = value
        # on_pick saves the value, which also returns the screen to the settings list.
        self._choice["on_pick"](self, value)
        await self.refresh(interaction)

    async def _clear_callback(self, interaction: discord.Interaction):
        # Picks are kept across tabs, and Discord reports only what is *selected* -- so
        # the dropdown clears its own tab's rows before adding back what it holds.
        self._clear_picks -= {a.value for a in self._set_rows(self.tab)}
        self._clear_picks.update(interaction.data.get("values", []))
        await self.refresh(interaction)

    async def _clear_selected(self, interaction: discord.Interaction):
        for value in sorted(self._clear_picks):
            action = PROFILE_ACTIONS_BY_VALUE.get(value)
            if action is None or action.bulk is None:
                continue
            for key in self._keys(action):
                self.defaults.pop(key, None)
        self._clear_picks.clear()
        self._persist()
        self.step = "actions"
        await self.refresh(interaction)

    async def _clear_all(self, interaction: discord.Interaction):
        self.defaults.clear()
        self._clear_picks.clear()
        self._persist()
        self.step = "actions"
        await self.refresh(interaction)


class DefaultsOpenRouterHostView(OpenRouterHostView):
    """The Hosts & Tier screen for My Defaults: a pin and a tier as standing preferences.

    A pin is an ordinary defaultable key (`openrouter_endpoints`), so it is stored and
    cleared here exactly as a model choice is -- which is why this screen needs none of
    the bulk one's Unchanged state, and why it is written straight through rather than
    staged. What it does need is the third state the rest of the screen has: *absent*.
    An empty pin map is not "no pins chosen", it is a customised setting that counts
    towards the footer and is written onto every profile made afterwards, so the last
    pin removed takes the key with it.
    """

    _TIER_REACH = ("Every OpenRouter text model a profile you make from now on leaves on "
                   "Auto, on every tab.")

    def _choose(self, model_id: str, chosen: str):
        super()._choose(model_id, chosen)
        if not self.parent.defaults.get("openrouter_endpoints"):
            self.parent._clear("openrouter_endpoints")

    def _tier(self):
        value, wording, desc = super()._tier()
        if value is None:
            # The base's None is the bulk screen's "not part of the changeset". Here it
            # is the absence every other row on this screen means by it.
            desc = "New profiles follow the bot's own tier, including if it changes."
        return value, wording, desc

    def _head_options(self, chosen: str, tier, tier_wording: str):
        options = super()._head_options(chosen, tier, tier_wording)
        if tier is None:
            options[0].description = "Any host, at the bot's own tier."
        return options

    def _no_host(self, value) -> tuple:
        if value is None:
            return ("`Platform default` — set a model on this slot to choose the host it "
                    "is sent to.", "no default model on this slot")
        return super()._no_host(value)

    def _intro(self) -> str:
        return (super()._intro() + "\n-# A host is pinned for a *model*, so it reaches "
                "every profile you go on to make that runs on that model.")

    def _footer(self) -> str:
        return self.parent.staging_footer()
