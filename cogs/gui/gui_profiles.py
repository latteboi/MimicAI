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
from ..utils.helpers import (
    _pf, _pi, _ps, _pb, is_real_model, image_model_caps, resolve_critic_settings,
)
from ..utils.http_client import get_shared_client

if TYPE_CHECKING:
    # This only runs during "hinting" and prevents the circular crash
    from ..MimicCog import MimicCog

from .base_components import (
    BaseBulkProfileView, ConfigModal, ActionTextInputModal, TimeoutCleanupMixin,
    build_pagination_controls, build_confirm_view,
)
from .gui_data import DataManageView
from .gui_hub import HubShareManagerView
from .gui_sessions import CustomModelModal
from .gui_settings import OllamaHostModal

def ProfileAdvancedParamsModal(cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, values_only: bool = False, callback=None, target_user_id: Optional[int] = None):
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

def ProfileDirectorDeskModal(cog, profile_name: str, current_params: Dict[str, Any], values_only: bool = False, callback=None, target_user_id: Optional[int] = None):
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


def ProfileSpeechSettingsModal(cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, values_only: bool = False, callback=None, target_user_id: Optional[int] = None):
    # No voice field: it moved to the Choose TTS Voice picker. A text box here accepted
    # any string, and an unknown voice name comes back as a 400 that _generate_google_tts
    # turns into silence -- the profile looked configured and simply never spoke.
    # values_only drops the fields the setting's own screen renders as controls. The
    # full form is what the bulk wizard still opens, so nothing is unreachable there.
    fields = [] if values_only else [
        {"label": "Enable TTS (on/off)", "custom_id": "speech_tts_enabled", "default": "on" if current_params.get("speech_tts_enabled", False) else "off", "required": True, "max_length": 10},
    ]
    fields.append(
        {"label": "Temperature (0.0 - 2.0)", "custom_id": "speech_temperature", "default": str(current_params.get("speech_temperature", 1.0)), "required": False, "max_length": 5}
    )
    def parser(v):
        c = {}
        if "speech_tts_enabled" in v:
            c["speech_tts_enabled"] = _pb(v["speech_tts_enabled"])
        t = _pf(v["speech_temperature"])
        if t is not None:
            if not (0.0 <= t <= 2.0): raise ValueError("Temperature out of range")
            c["speech_temperature"] = t
        return {"config": c}
    return ConfigModal(cog, profile_name, is_borrowed, "Speech Settings", fields, parser, callback, target_user_id)

#: Tab order for ProfileManageView's nav bar. "persona" is hidden for borrowed profiles.
#: Tab order for ProfileManageView's nav bar. "persona" is hidden for borrowed profiles.
PROFILE_TABS = ("home", "persona", "params", "tools", "memory")


class _Bulk:
    """How one PROFILE_ACTIONS row behaves in the bulk manager.

    BulkManageView used to carry its own hand-written list of ~20 SelectOptions and a
    parallel `elif` chain, which is exactly the shape `_Action` was introduced to kill
    for the single-profile dashboard. The two drifted: Speech & Voice, Custom Error
    Message, LTM Auto-Creation and the whole Persona tab were reachable for one profile
    and unreachable for forty. Declaring the bulk behaviour beside the single-profile
    behaviour means a new setting cannot be added to one and forgotten in the other.

    `scope` is "all" (personal + borrowed) or "personal"; prompts live only on profiles
    the user owns, so anything writing them must be personal-only or it silently no-ops
    on the borrowed half of the selection.

    `destructive` carries the row's warning into the review step, which turns red and
    names the profile count before anything is written. Reserved for the rows that
    overwrite authored content rather than flipping a setting -- restoring forty
    personas by hand is not an undo.

    `terminal` marks the two rows that are not settings at all: deleting profiles and
    wiping LTM or training data. Everything else reduces to merging into a config or
    prompts dict and can therefore be staged alongside anything else; these two remove
    files, so they run on their own and refuse to share a pass with staged edits.

    `keys` / `prompt_keys` name what the row writes, which is what makes it copyable
    from an anchor profile: the inherit step reads exactly these keys off the anchor
    rather than needing a second table of its own. They are checked against what the
    row's own handler actually stages, so a modal that grows a field and forgets to
    declare it fails the suite instead of silently dropping out of every copy. A row
    with neither is not offered for copying -- the 18+ declaration deliberately so,
    since a content rating is not a setting to be propagated.
    """

    __slots__ = ("run", "scope", "destructive", "terminal", "warning", "description",
                 "label", "keys", "prompt_keys")

    def __init__(self, run, *, scope="all", destructive=False, terminal=False,
                 warning=None, description=None, label=None, keys=(), prompt_keys=()):
        self.run = run
        self.scope = scope
        self.destructive = destructive
        self.terminal = terminal
        self.warning = warning
        self.description = description
        self.label = label
        self.keys = tuple(keys)
        self.prompt_keys = tuple(prompt_keys)

    @property
    def copyable(self) -> bool:
        return bool(self.keys or self.prompt_keys)

    @property
    def include_borrowed(self) -> bool:
        return self.scope == "all"


class _Toggle:
    """A boolean a function screen flips with one click.

    A two-option select would cost two interactions to say the same thing, so toggles
    stay buttons. `read` and `to_payload` exist for the settings whose stored shape is
    not a bare bool -- thinking_summary_visible is the string "on"/"off", and url_mode
    has to drag its legacy companion flag along with it.
    """

    __slots__ = ("key", "label", "read", "to_payload")

    def __init__(self, key, label, *, read=None, to_payload=None):
        self.key = key
        self.label = label
        self.read = read or (lambda config: bool(config.get(key, False)))
        self.to_payload = to_payload or (lambda on: {key: on})


class _Choice:
    """A fixed set of values a function screen picks with a select.

    Every one of these was a free-text field in a modal, validated only on submit:
    "lexical", "session", "strict", "native", "mention_reply" all had to be spelled
    correctly with nothing on screen saying what the alternatives were. A select cannot
    be misspelled and shows the options and the current value at once.

    `options` are (label, value) or (label, value, description).
    """

    __slots__ = ("key", "label", "options", "read", "to_payload", "placeholder")

    def __init__(self, key, label, options, *, read=None, to_payload=None, placeholder=None):
        self.key = key
        self.label = label
        self.options = tuple(options)
        self.read = read or (lambda config: config.get(key))
        self.to_payload = to_payload or (lambda value: {key: value})
        self.placeholder = placeholder or f"{label}..."


class _Screen:
    """One setting's own screen: its controls, and the modal holding its free values.

    Declared on the `_Action` beside the row that opens it, for the same reason `render`
    and `bulk` are. Discord has no numeric or paragraph control outside a modal, so a
    screen carries selects and buttons for everything enumerable and hands the rest to
    `modal` behind an Edit button -- the split is a platform constraint, not a
    preference. A row with no `_Screen` keeps whatever it did before: the model, voice,
    image-output and timezone pickers are already purpose-built screens, and the
    operations (rename, duplicate, delete, the data managers) are not settings at all.
    """

    __slots__ = ("controls", "modal", "modal_label", "sub_view", "note")

    def __init__(self, *controls, modal=None, modal_label="Edit values…",
                 sub_view=None, note=None):
        self.controls = tuple(controls)
        self.modal = modal
        self.modal_label = modal_label
        #: (button label, row handler) for a screen that also opens a purpose-built
        #: picker. The handler has the same shape as `_Action.run` and is called with
        #: the parent dashboard, so the timezone picker is reached the same way it
        #: always was rather than being reimplemented as controls.
        self.sub_view = sub_view
        self.note = note

    @property
    def choices(self):
        return tuple(c for c in self.controls if isinstance(c, _Choice))

    @property
    def toggles(self):
        return tuple(c for c in self.controls if isinstance(c, _Toggle))


def _write_profile_config(cog, user_id: str, profile_name: str, is_borrowed: bool,
                          updates: Dict[str, Any]) -> bool:
    """Merge `updates` into a profile's config and drop any live model built from it.

    The invalidation is what makes a setting take effect mid-session: a cached model
    instance carries the system instruction and sampling parameters it was built with.
    Copied out of _save_and_refresh, which is now one caller of this rather than the
    only place that knew to do it.
    """
    target = cog.profile_manager._get_profile_config(user_id, profile_name, is_borrowed)
    if target is None:
        return False
    target.update(updates)
    cog.profile_manager._save_profile_config(user_id, profile_name, target, is_borrowed)

    stale = [k for k in cog.channel_models
             if isinstance(k, tuple) and len(k) == 3 and k[1] == user_id and k[2] == profile_name]
    for k in stale:
        cog.channel_models.pop(k, None)
        cog.channel_model_last_profile_key.pop(k, None)
    return True


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

    `bulk` is an optional `_Bulk` describing the same setting applied to many profiles
    at once. Rows with no `bulk` are the ones that cannot mean anything in bulk -- a
    rename, a duplicate, or an item-by-item data editor.

    `render` is how this row shows up on the dashboard embed: a callable taking the
    render context and returning `(field_name, field_value, inline)`, or None to
    contribute no field. It lives here for the same reason `bulk` does -- a setting
    that gains an action and a bulk form but no dashboard line is exactly the drift
    this table exists to make impossible, and the embed used to be a 170-line wall
    maintained in a different file from the actions it described.
    """

    __slots__ = ("value", "tab", "label", "description", "gate", "run", "_menu_label",
                 "render", "screen", "bulk")

    def __init__(self, value, tab, label, description, run, gate=None, menu_label=None,
                 render=None, screen=None,
                 bulk=None):
        self.value = value
        self.tab = tab
        self.label = label
        self.description = description
        self.run = run
        self.gate = gate
        self._menu_label = menu_label
        self.render = render
        self.screen = screen
        self.bulk = bulk

    def bulk_label(self) -> str:
        """Wording for the bulk dropdown, which has no view to resolve a callable label."""
        if self.bulk is not None and self.bulk.label:
            return self.bulk.label
        return self.menu_label

    def bulk_description(self) -> str:
        if self.bulk is not None and self.bulk.description:
            return self.bulk.description
        return self.description

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


def _open_screen(action_value: str):
    """Row handler that swaps the dashboard onto that setting's own screen.

    Named by value rather than closing over the `_Action` because the table is defined
    below the rows it points at -- the same reason `_modal` looks its factory up at
    click time.
    """
    async def run(view, interaction, profile):
        action = PROFILE_ACTIONS_BY_VALUE[action_value]
        screen = ProfileFunctionView(view, action)
        await interaction.response.edit_message(embed=await screen.embed(), view=screen)
    return run


# _toggle and _cycle lived here. Every row that used them now declares a `_Screen`
# instead: the toggles became buttons that say which way they will go, and the two
# cyclers became selects. A cycle was the worst of the three -- reaching RAG from Off
# took two clicks through a Native setting that silently does nothing on OpenRouter,
# with no screen anywhere saying so.


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


# --- Bulk handlers -------------------------------------------------------------
#
# The bulk counterparts of _modal / _toggle / _cycle above. Each returns a coroutine
# taking (wizard, interaction), and its only job is to collect a value: which profiles
# receive it was settled two steps earlier, and whether the write needs confirming is
# settled two steps later. A handler ends by staging into the wizard's changeset, not
# by writing anything.

def _bulk_modal(factory_name: str, *, action_key: str = "update_config",
                pass_borrowed: bool = True, seed=None):
    """Open a settings modal in BULK_APPLY mode, then stage what it parsed.

    The modal factories short-circuit on the sentinel profile name "BULK_APPLY" and
    hand their parsed payload to `callback` instead of writing it, so the bulk flow
    reuses the exact fields and validation the single-profile flow uses -- the two
    cannot drift, because there is only one of them.

    `seed` supplies the third positional argument for the factories that read it as
    something other than a config dict; ProfileLTMSummarizationModal takes an
    instruction *string* there and feeds it straight to a TextInput default, so the
    empty dict every other factory tolerates is not valid for it. Callable so a
    default can be resolved against the live cog rather than at import time.
    """
    async def run(wizard, interaction):
        # Captured now rather than read back off the wizard on submit: the user can
        # page around the panel while the modal is open, and the payload belongs to
        # the row that opened it.
        action_value = wizard.current_action.value

        async def modal_callback(i: discord.Interaction, params: Dict):
            if action_key == "update_both":
                config = params.get("config") or {}
                prompts = params.get("prompts") or {}
            elif action_key == "update_prompts":
                config, prompts = {}, (params.get("prompts") or {})
            else:
                config, prompts = (params.get("config") or {}), {}
            await wizard._stage_from_modal(i, action_value, config, prompts)

        current = seed(wizard.cog) if callable(seed) else ({} if seed is None else seed)
        args = [wizard.cog, "BULK_APPLY", current]
        if pass_borrowed:
            args.append(False)
        await interaction.response.send_modal(
            globals()[factory_name](*args, callback=modal_callback))
    return run


def _bulk_choice(placeholder: str, choices, *, to_payload):
    """A fixed set of options, chosen once and staged for every selected profile.

    `choices` is a sequence of (label, value[, description]); `to_payload` maps the
    chosen value to the config keys it writes. Rendered as a step of the wizard rather
    than as a dropdown bolted onto the profile picker, which is where it used to live
    -- and where every rebuild of that picker deleted it.
    """
    async def run(wizard, interaction):
        action = wizard.current_action
        opts = []
        for c in choices:
            label, value = str(c[0]), str(c[1])
            desc = str(c[2]) if len(c) > 2 else None
            opts.append(discord.SelectOption(label=label[:100], value=value,
                                             description=desc[:100] if desc else None))
        wizard._open_choice(
            action.value, placeholder, opts,
            lambda w, v: w._stage_change(action.value, config=to_payload(v)))
        await wizard.refresh(interaction)
    return run


def _bulk_sub(view_factory: str):
    """Bulk rows that need a picker of their own, or that act rather than stage.

    Models and timezone have too many controls to sit inside the action step; delete
    and data-reset are not settings at all. Both kinds swap onto the wizard's own
    message and carry a Back button, so the flow never leaves the panel it began in --
    which is the difference between a step and a dead ephemeral message with a fresh
    one stacked under it.
    """
    async def run(wizard, interaction):
        sub = globals()[view_factory](wizard)
        await interaction.response.edit_message(embed=sub.embed(), view=sub)
    return run


def _bulk_method(name: str):
    """Bulk row that defers to one of BulkManageView's own methods."""
    async def run(wizard, interaction):
        await getattr(wizard, name)(interaction)
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


# --- Dashboard field renderers ------------------------------------------------
#
# One per setting, declared beside the action that changes it and pulled into the
# embed by tab. The embed used to be a single 170-line block in ProfileManager that
# rendered every group on every repaint regardless of which tab the view was on, so
# the Tools tab showed you Top-K and the Params tab showed you hormone levels.
#
# Each takes the render context ProfileManager._build_profile_embed assembles and
# returns (name, value, inline) or None. Returning None omits the field entirely,
# which is how a row says "nothing to report here" -- see the persona sections.

_ON = "**`ON`**"
_OFF = "`OFF`"


def _flag(value) -> str:
    return _ON if value else _OFF


def _mode_display(raw, *, legacy_true="rag") -> str:
    """off/native/rag, coercing the two legacy encodings this setting has had.

    grounding_mode was a bool before it was a string and briefly took "on"/"on+";
    url_mode reads off the older url_fetching_enabled flag when it is absent.
    """
    if isinstance(raw, bool):
        raw = legacy_true if raw else "off"
    elif raw in ("on", "on+"):
        raw = legacy_true
    return {"off": _OFF, "native": "**`NATIVE`**", "rag": "**`RAG`**"}.get(raw, _OFF)


def _grounding_mode(config) -> str:
    raw = config.get("grounding_mode", "off")
    if isinstance(raw, bool):
        return "rag" if raw else "off"
    return "rag" if raw in ("on", "on+") else raw


def _url_mode(config) -> str:
    if "url_mode" not in config:
        return "rag" if config.get("url_fetching_enabled", False) else "off"
    return config.get("url_mode", "off")








def _render_thinking(ctx):
    config = ctx["config"]
    budget = config.get("thinking_budget", -1)
    return "Thinking / Reasoning", (
        f"Summary: {_flag(str(config.get('thinking_summary_visible', 'off')).lower() == 'on')}\n"
        f"Effort: `{str(config.get('thinking_level', 'high')).title()}`\n"
        f"Budget: `{'Dynamic (-1)' if budget == -1 else budget}`"
    ), True


def _render_speech(ctx):
    config = ctx["config"]
    return "Speech TTS", (
        f"Enabled: {_flag(config.get('speech_tts_enabled', False))}\n"
        f"Temperature: `{config.get('speech_temperature', 1.0)}`"
    ), True




def _render_image_toggle(ctx):
    config = ctx["config"]
    return "Image Generation", (
        f"Enabled: {_flag(config.get('image_generation_enabled', False))}\n"
        f"Model: `{config.get('image_generation_model') or DEFAULT_IMAGE_MODEL}`"
    ), True




def _render_grounding(ctx):
    return "Grounding (Web Search)", _mode_display(ctx["config"].get("grounding_mode", "off")), True


def _render_url(ctx):
    return "URL Context", _mode_display(_url_mode(ctx["config"])), True


def _render_help_mode(ctx):
    return "Help Mode (Guide RAG)", _flag(ctx["config"].get("help_mode_enabled", False)), True


def _render_response_mode(ctx):
    raw = str(ctx["config"].get("response_mode", "regular"))
    return "Response Mode", f"`{raw.replace('_', ' ').title()}`", True


def _render_time(ctx):
    config = ctx["config"]
    return "Time & Timezone", (
        f"Tracking: {_flag(config.get('time_tracking_enabled', False))}\n"
        f"Zone: `{config.get('timezone', 'UTC')}`"
    ), True


def _render_typing(ctx):
    config = ctx["config"]
    if not config.get("realistic_typing_enabled", False):
        return "Realistic Typing", _OFF, True
    return "Realistic Typing", (
        f"{_ON}\n"
        f"Mode: `{config.get('typing_mode', 'sentence')}`\n"
        f"Rate: `{config.get('typing_cps', 30.0)}` cps\n"
        f"Max Delay: `{config.get('typing_max_delay', 2.5)}`s"
    ), True


def _render_critic(ctx):
    critic = resolve_critic_settings(ctx["config"])
    if critic["mode"] == "off":
        return "Anti-Repetition Critic", _OFF, True
    lines = [f"Mode: **`{critic['mode'].upper()}`**",
             f"Scope: `{critic['scope']}`",
             f"Strictness: `{critic['strictness']}`",
             f"Lookback: `{critic['lookback']}` turns",
             f"Persistence: `{critic['persistence']}` rounds"]
    if critic["mode"] == "lexical":
        lines.append("-# Local scan only, no API call.")
    return "Anti-Repetition Critic", "\n".join(lines), True




def _render_neuro(ctx):
    config = ctx["config"]
    if not config.get("neuro_engine_enabled", False):
        return "Neuro Engine", _OFF, True
    state = config.get("neuro_state") or {}
    return "Neuro Engine", (
        f"{_ON}\n"
        f"Dopamine: `{state.get('dopamine', 50)}`\n"
        f"Cortisol: `{state.get('cortisol', 20)}`\n"
        f"Oxytocin: `{state.get('oxytocin', 50)}`\n"
        f"Adrenaline: `{state.get('adrenaline', 20)}`"
    ), True












def _render_ltm_creation(ctx):
    return "LTM Auto-Creation", _flag(ctx["config"].get("ltm_creation_enabled", False)), True












def _render_generation_visual(ctx):
    config = ctx["config"]
    return "Generation Visual", (
        f"Placeholder: {config.get('placeholder_emoji') or '`Default`'}\n"
        f"Child Bot Placeholder: {_flag(config.get('child_bot_placeholder', False))}"
    ), True




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
            _method("_act_error_response", wants_profile=True), _own,
            bulk=_Bulk(_bulk_method("_bulk_error_response"), scope="all",
                       keys=("error_response",))),
    _Action("generation_visual", "home", "Generation Visual", "Set custom placeholder emoji and child bot behavior.",
            _open_screen("generation_visual"), _own,
            render=_render_generation_visual,
            screen=_Screen(_Toggle("child_bot_placeholder", "Child Bot Placeholder"),
                           modal="ProfileGenerationVisualModal", modal_label="Edit emoji…"),
            bulk=_Bulk(_bulk_modal("ProfileGenerationVisualModal"), scope="all",
                       keys=("placeholder_emoji", "child_bot_placeholder"))),
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
            _method("_handle_content_safety"),
            # The dashboard itself is per-profile, but the one action inside it that
            # is the owner's to set in bulk -- the 18+ declaration -- is not. A
            # classifier verdict and an exemption are deliberately excluded.
            bulk=_Bulk(_bulk_method("_bulk_adult_declaration"), scope="personal",
                       label="Set Adult 18+ Declaration",
                       description="Declare or withdraw 18+ across multiple profiles.")),
    _Action("delete", "home",
            lambda v: "Remove Borrowed Profile" if v.is_borrowed else "Delete Profile",
            "Permanently remove this profile and its data.",
            _method("_handle_delete"),
            menu_label="Delete Profile / Remove Borrowed Profile",
            bulk=_Bulk(_bulk_sub("BulkDeleteView"), scope="all", terminal=True,
                       label="Delete Profiles",
                       description="Permanently delete the selected profiles.")),

    # --- Persona (tab hidden entirely for borrowed profiles) ---
    _Action("edit_persona", "persona", "Edit Persona", "Edit backstory, traits, likes, dislikes, and appearance.",
            _method("_act_edit_persona", wants_profile=True),
            bulk=_Bulk(_bulk_method("_bulk_edit_persona"), scope="personal", destructive=True,
                       prompt_keys=("persona",),
                       description="Overwrite backstory, traits, likes, dislikes and appearance.",
                       warning="Every selected profile's **entire persona** is replaced with what you "
                               "typed. Sections you left blank are cleared, not preserved. The previous "
                               "text is not recoverable.")),
    _Action("edit_instructions", "persona", "Edit Instructions", "Edit specific AI behavioral instructions.",
            _method("_act_edit_instructions", wants_profile=True),
            bulk=_Bulk(_bulk_method("_bulk_edit_instructions"), scope="personal", destructive=True,
                       prompt_keys=("ai_instructions",),
                       description="Overwrite all four AI instruction parts.",
                       warning="Every selected profile's **AI instructions** are replaced with what you "
                               "typed, across all four parts. Part 4 is the slot the training analyser "
                               "writes to, so any generated style guide is overwritten too. The previous "
                               "text is not recoverable.")),
    _Action("tts_instructions", "persona", "TTS Instructions", "Configure the 'Director's Desk' for vocal performance.",
            _modal("ProfileDirectorDeskModal", pass_borrowed=False),
            bulk=_Bulk(_bulk_modal("ProfileDirectorDeskModal", pass_borrowed=False),
                       scope="personal",
                       keys=("speech_archetype", "speech_accent", "speech_pacing",
                             "speech_dynamics", "speech_style"))),
    _Action("edit_appearance", "persona", "Edit Appearance", "Edit the custom Webhook name and avatar.",
            _method("_handle_appearance"), _own),

    # --- Params ---
    _Action("models", "params", "Set Models", "Choose Primary and Fallback AI models.",
            _method("_act_models", wants_profile=True),
            bulk=_Bulk(_bulk_sub("ModelApplyView"), scope="all",
                       description="Stage primary, fallback and utility models.",
                       keys=("primary_model", "fallback_model", "show_fallback_indicator",
                             "image_generation_model", "image_generation_fallback_model",
                             "speech_model", "speech_fallback_model",
                             "grounding_rag_model", "grounding_rag_fallback_model",
                             "critic_model", "critic_fallback_model",
                             "ltm_model", "ltm_fallback_model",
                             "ollama_host_url"))),
    _Action("gen_params", "params", "Set Generation Parameters & STM", "Set Temp, Top P, Top K, and STM Length.",
            _modal("ProfileParamsModal"),
            bulk=_Bulk(_bulk_modal("ProfileParamsModal"), scope="all",
                       keys=("temperature", "top_p", "top_k", "stm_length"))),
    _Action("adv_params", "params", "Set Advanced Parameters (OPENROUTER)", "Set penalties, Min P, and Top A.",
            _modal("ProfileAdvancedParamsModal"),
            bulk=_Bulk(_bulk_modal("ProfileAdvancedParamsModal"), scope="all",
                       keys=("frequency_penalty", "presence_penalty", "repetition_penalty",
                             "min_p", "top_a"))),
    _Action("thinking_params", "params", "Set Thinking Parameters", "Set thinking persistence, level, and budget.",
            _open_screen("thinking_params"), render=_render_thinking,
            screen=_Screen(
                _Choice("thinking_level", "Reasoning Effort",
                        (("Extra High", "xhigh"), ("High", "high"), ("Medium", "medium"),
                         ("Low", "low"), ("Minimal", "minimal"), ("None", "none")),
                        read=lambda c: str(c.get("thinking_level", "high")).lower(),
                        placeholder="Reasoning effort..."),
                _Toggle("thinking_summary_visible", "Show Summary",
                        read=lambda c: str(c.get("thinking_summary_visible", "off")).lower() == "on",
                        to_payload=lambda on: {"thinking_summary_visible": "on" if on else "off"}),
                modal="ProfileThinkingParamsModal", modal_label="Edit budget…"),
            bulk=_Bulk(_bulk_modal("ProfileThinkingParamsModal"), scope="all",
                       keys=("thinking_level", "thinking_budget", "thinking_summary_visible"))),
    _Action("speech_settings", "params", "Set Speech Settings", "Turn TTS on or off and set its temperature.",
            _open_screen("speech_settings"), render=_render_speech,
            screen=_Screen(_Toggle("speech_tts_enabled", "TTS"),
                           modal="ProfileSpeechSettingsModal", modal_label="Edit temperature…"),
            bulk=_Bulk(_bulk_modal("ProfileSpeechSettingsModal"), scope="all",
                       keys=("speech_tts_enabled", "speech_temperature"))),
    _Action("voice", "params", "Choose TTS Voice", "Pick from the thirty prebuilt Gemini voices.",
            _method("_act_voice", wants_profile=True),
            bulk=_Bulk(_bulk_sub("VoiceApplyView"), scope="all", label="Choose TTS Voice",
                       description="Stage one of the thirty prebuilt voices.",
                       keys=("speech_voice",))),

    # --- Tools ---
    _Action("image_toggle", "tools", "Image Generation", "Allow this profile to generate images via !image/!imagine.",
            _open_screen("image_toggle"), render=_render_image_toggle,
            screen=_Screen(_Toggle("image_generation_enabled", "Image Generation"),
                           modal="ProfileImageGenSettingsModal", modal_label="Edit prompt…"),
            bulk=_Bulk(_bulk_modal("ProfileImageGenSettingsModal", action_key="update_both"),
                       scope="all", label="Configure Image Generation",
                       description="Set up models, prompts, and toggles for multiple profiles.",
                       keys=("image_generation_enabled",),
                       prompt_keys=("image_generation_prompt",))),
    _Action("image_output", "tools", "Set Image Output", "Set aspect ratio, resolution and thinking level.",
            _method("_act_image_output", wants_profile=True),
            bulk=_Bulk(_bulk_sub("ImageOutputApplyView"), scope="all", label="Set Image Output",
                       description="Stage aspect ratio, resolution and thinking level.",
                       keys=IMAGE_OUTPUT_KEYS)),
    _Action("grounding", "tools", "Grounding (Web Search)", "Choose Off, Native or RAG web search.",
            _open_screen("grounding"), render=_render_grounding,
            # A select, not the old three-way cycle: going Off -> Native -> RAG -> Off
            # meant two clicks to reach RAG and no indication that Native is
            # Google-only until the turn failed.
            screen=_Screen(_Choice(
                "grounding_mode", "Grounding",
                (("Off", "off", "No web search."),
                 ("Native", "native", "Provider-side Google Search. Google models only."),
                 ("RAG", "rag", "Search, then summarise. Works on any provider.")),
                read=_grounding_mode, placeholder="Grounding mode..."),
                note="-# OpenRouter and Ollama models must use **RAG**; Native is a Google-side tool."),
            bulk=_Bulk(_bulk_choice("Select Grounding Mode...",
                                    [("Off", "off"), ("Native", "native"), ("RAG", "rag")],
                                    to_payload=lambda v: {"grounding_mode": v}),
                       scope="all", label="Set Grounding Mode", keys=("grounding_mode",),
                       description="Choose Off, Native or RAG for every selected profile.")),
    _Action("url_toggle", "tools", "URL Context Fetching", "Choose Off, Native or RAG link reading.",
            _open_screen("url_toggle"), render=_render_url,
            # url_fetching_enabled is the legacy flag the turn path still reads, so it
            # has to move with url_mode -- on the screen as well as in bulk, or setting
            # it one way leaves the two disagreeing.
            screen=_Screen(_Choice(
                "url_mode", "URL Context",
                (("Off", "off", "Links posted in chat are ignored."),
                 ("Native", "native", "Provider-side URL fetching. Google models only."),
                 ("RAG", "rag", "Fetch and scrape the page. Works on any provider.")),
                read=_url_mode,
                to_payload=lambda v: {"url_mode": v, "url_fetching_enabled": v == "rag"},
                placeholder="URL context mode..."),
                note="-# OpenRouter and Ollama models must use **RAG**; Native is a Google-side tool."),
            bulk=_Bulk(_bulk_choice("Select URL Mode...",
                                    [("Off", "off"), ("Native", "native"), ("RAG", "rag")],
                                    to_payload=lambda v: {"url_mode": v,
                                                          "url_fetching_enabled": v == "rag"}),
                       scope="all", label="Set URL Context Mode",
                       keys=("url_mode", "url_fetching_enabled"),
                       description="Choose Off, Native or RAG for every selected profile.")),
    _Action("cycle_response", "tools", "Response Mode", "Choose how a reply attaches to your message.",
            _open_screen("cycle_response"), render=_render_response_mode,
            screen=_Screen(_Choice(
                "response_mode", "Response Mode",
                (("Regular", "regular", "Post as a normal message."),
                 ("Mention", "mention", "Mention the user who triggered the reply."),
                 ("Reply", "reply", "Reply to the triggering message."),
                 ("Mention + Reply", "mention_reply", "Both.")),
                read=lambda c: c.get("response_mode", "regular"),
                placeholder="Response mode...")),
            bulk=_Bulk(_bulk_choice("Select Response Mode...",
                                    [("Regular", "regular"), ("Mention", "mention"),
                                     ("Reply", "reply"), ("Mention+Reply", "mention_reply")],
                                    to_payload=lambda v: {"response_mode": v}),
                       scope="all", label="Set Response Mode", keys=("response_mode",),
                       description="Choose Regular, Mention, Reply or Mention+Reply.")),
    _Action("time", "tools", "Set Time & Timezone", "Enable time awareness and set the profile's timezone.",
            _open_screen("time"), render=_render_time,
            # Picking a timezone force-sets time_tracking_enabled, and nothing in the
            # single-profile GUI could set it back -- a profile that had ever chosen a
            # zone was stuck with time awareness on unless it went through the bulk
            # manager. The toggle here is the off switch that was missing.
            screen=_Screen(
                _Toggle("time_tracking_enabled", "Time Awareness"),
                sub_view=("Choose timezone…",
                          _method("_handle_timezone", wants_profile=True, wants_borrowed=True))),
            bulk=_Bulk(_bulk_sub("BulkTimezoneView"), scope="all",
                       keys=("timezone", "time_tracking_enabled"))),
    _Action("typing", "tools", "Realistic Typing", "Enable a human-like delay when the bot sends messages.",
            _open_screen("typing"), render=_render_typing,
            screen=_Screen(
                _Choice("typing_mode", "Chunking",
                        (("Sentence", "sentence", "Split the reply on sentence boundaries."),
                         ("Line", "line", "Split the reply on line breaks.")),
                        read=lambda c: c.get("typing_mode", "sentence"),
                        placeholder="Chunking mode..."),
                _Toggle("realistic_typing_enabled", "Realistic Typing"),
                modal="ProfileTypingSettingsModal", modal_label="Edit speed…"),
            bulk=_Bulk(_bulk_modal("ProfileTypingSettingsModal"), scope="all",
                       keys=("realistic_typing_enabled", "typing_mode", "typing_cps",
                             "typing_max_delay"))),
    _Action("critic", "tools", "Configure Anti-Repetition Critic",
            "Mode, scope, strictness, lookback and constraint persistence.",
            _open_screen("critic"), render=_render_critic,
            screen=_Screen(
                _Choice("critic_mode", "Mode",
                        (("Off", "off", "No repetition screening."),
                         ("Lexical", "lexical", "Local n-gram scan. No API call, no added latency."),
                         ("Full", "full", "Adds a model pass. One extra call per reply.")),
                        read=lambda c: resolve_critic_settings(c)["mode"],
                        to_payload=lambda v: {"critic_mode": v, "critic_enabled": v != "off"},
                        placeholder="Critic mode..."),
                _Choice("critic_scope", "Scope",
                        (("Self", "self", "Screen against this profile's own recent replies."),
                         ("Session", "session", "Screen against every participant's replies.")),
                        read=lambda c: resolve_critic_settings(c)["scope"],
                        placeholder="Critic scope..."),
                _Choice("critic_strictness", "Strictness",
                        (("Lenient", "lenient", "Only longer repeated phrases count."),
                         ("Normal", "normal", "The shipped default."),
                         ("Strict", "strict", "Short repeated phrases count.")),
                        read=lambda c: resolve_critic_settings(c)["strictness"],
                        placeholder="Critic strictness..."),
                modal="ProfileCriticSettingsModal", modal_label="Edit lookback & persistence…"),
            bulk=_Bulk(_bulk_modal("ProfileCriticSettingsModal"), scope="all",
                       label="Configure Anti-Repetition Critic",
                       description="Set mode, scope, strictness, lookback and persistence.",
                       keys=("critic_mode", "critic_enabled", "critic_scope",
                             "critic_strictness", "critic_lookback", "critic_persistence"))),
    _Action("critic_instructions", "tools", "Set Critic Instructions",
            "Customise the prompt the critic screens replies with.",
            _method("_act_critic_instructions", wants_profile=True), _own,
            bulk=_Bulk(_bulk_modal("ProfileCriticInstructionsModal",
                                   action_key="update_prompts", pass_borrowed=False,
                                   seed=lambda c: c.profile_manager._default_critic_instructions()),
                       scope="personal", destructive=True,
                       prompt_keys=("critic_instructions",),
                       warning="Every selected profile's **critic prompt** is replaced. "
                               "Profiles using a customised prompt lose it.")),
    _Action("neuro", "tools", "Neuro-Endocrine Engine", "Simulate hormonal states for dynamic emotions.",
            _open_screen("neuro"), render=_render_neuro,
            screen=_Screen(_Toggle("neuro_engine_enabled", "Neuro Engine"),
                           modal="ProfileNeuroModal", modal_label="Edit hormones…"),
            bulk=_Bulk(_bulk_modal("ProfileNeuroModal"), scope="all",
                       keys=("neuro_engine_enabled", "neuro_state"))),
    _Action("help_mode", "tools", "Help Mode (Guide RAG)", "Allow profile to answer technical bot questions.",
            _open_screen("help_mode"), render=_render_help_mode,
            screen=_Screen(_Toggle("help_mode_enabled", "Help Mode")),
            bulk=_Bulk(_bulk_choice("Select action...",
                                    [("Enable Help Mode", "true"), ("Disable Help Mode", "false")],
                                    to_payload=lambda v: {"help_mode_enabled": v == "true"}),
                       scope="all", label="Set Help Mode", keys=("help_mode_enabled",),
                       description="Turn the documentation RAG on or off.")),

    # --- Behaviour (how the profile conducts itself) ---

    # --- Memory ---
    _Action("manage_ltm", "memory", "Manage Long-Term Memories", "Add, list, edit, or delete memories.",
            _method("_act_manage_ltm", wants_profile=True),
            # Editing memories one at a time has no bulk form; wiping them does.
            bulk=_Bulk(_bulk_sub("BulkResetView"), scope="all", terminal=True,
                       label="Reset Profile Data",
                       description="Wipe long-term memories or training examples.")),
    _Action("manage_training", "memory", "Manage Training Examples", "Add, list, edit, or delete training examples.",
            _method("_act_manage_training", wants_profile=True), _own),
    _Action("train_params", "memory", "Set Training Parameters", "Set training context size and relevance threshold.",
            _modal("ProfileTrainingParamsModal", pass_borrowed=False), _own,
            bulk=_Bulk(_bulk_modal("ProfileTrainingParamsModal", pass_borrowed=False),
                       scope="personal",
                       keys=("training_context_size", "training_relevance_threshold"))),
    _Action("ltm_creation", "memory", "LTM Auto-Creation", "Automatically create memories from conversations.",
            _open_screen("ltm_creation"), render=_render_ltm_creation,
            screen=_Screen(_Toggle("ltm_creation_enabled", "Auto-Creation")),
            bulk=_Bulk(_bulk_choice("Select action...",
                                    [("Enable LTM Auto-Creation", "true"),
                                     ("Disable LTM Auto-Creation", "false")],
                                    to_payload=lambda v: {"ltm_creation_enabled": v == "true"}),
                       scope="all", label="Set LTM Auto-Creation", keys=("ltm_creation_enabled",),
                       description="Turn automatic memory creation on or off.")),
    _Action("ltm_params", "memory", "Set LTM Parameters", "Set frequency, context, and recall settings.",
            _modal("ProfileLTMParamsModal", pass_borrowed=False),
            bulk=_Bulk(_bulk_modal("ProfileLTMParamsModal", pass_borrowed=False), scope="all",
                       keys=("ltm_creation_interval", "ltm_summarization_context",
                             "ltm_context_size", "ltm_relevance_threshold"))),
    _Action("ltm_summarization", "memory", "Set LTM Summarization Prompt", "Customize how the AI creates memories.",
            _method("_act_ltm_summarization", wants_profile=True), _own,
            bulk=_Bulk(_bulk_modal("ProfileLTMSummarizationModal",
                                   action_key="update_prompts", pass_borrowed=False,
                                   seed=lambda c: c.profile_manager._default_ltm_summarization_instructions()),
                       scope="personal", destructive=True,
                       prompt_keys=("ltm_summarization_instructions",),
                       warning="Every selected profile's **LTM summarisation prompt** is replaced. "
                               "Profiles using a customised prompt lose it.")),
)

PROFILE_ACTIONS_BY_VALUE = {a.value: a for a in PROFILE_ACTIONS}


class ProfileManageView(ui.View):
    def __init__(self, cog: 'MimicCog', original_interaction: discord.Interaction, profile_name: str, is_borrowed: bool, target_user_id: Optional[int] = None, is_mod_view: bool = False):
        super().__init__(timeout=600)
        self.cog = cog
        self.original_interaction = original_interaction
        
        owner_id = int(defaultConfig.DISCORD_OWNER_ID)
        # _is_system_name, not a bare System-index test. The bare test ran before any
        # personal lookup, so a user who owned a profile sharing a System name had
        # this view rewrite user_id to the bot owner and open the System profile
        # read-only -- their own profile was unreachable from the dashboard.
        viewer_id = target_user_id or original_interaction.user.id
        if self.cog.profile_manager._is_system_name(viewer_id, profile_name):
            self.target_user_id = owner_id
            self.user_id = owner_id
            self.is_system = True
        else:
            self.target_user_id = viewer_id
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

        # --- 2. Navigation Buttons (Rows 1-2) ---
        # Discord fits five components to an action row, and there are six tabs. They
        # all used to be pinned to row 1, which was fine at five and raises
        # "item would not fit at row 1" at six -- inside __init__, so the whole
        # dashboard failed to open rather than degrading. Split evenly rather than
        # 5 + 1, so the second row reads as a continuation instead of an orphan.
        per_row = len(valid_tabs) if len(valid_tabs) <= 5 else (len(valid_tabs) + 1) // 2
        for position, tab in enumerate(valid_tabs):
            btn = ui.Button(
                label=tab.title(),
                style=discord.ButtonStyle.primary if self.current_tab == tab else discord.ButtonStyle.secondary,
                row=1 + (position // per_row),
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

    async def _act_image_output(self, interaction: discord.Interaction, profile: Dict[str, Any]):
        await self._open_media_options(interaction, "image")

    async def _act_voice(self, interaction: discord.Interaction, profile: Dict[str, Any]):
        await self._open_media_options(interaction, "voice")

    async def _open_media_options(self, interaction: discord.Interaction, mode: str):
        view = SingleProfileMediaOptionsView(self.cog, self.original_interaction, self.profile_name,
                                             mode, is_borrowed=self.is_borrowed, user_id=self.user_id)
        await interaction.response.send_message(view._feedback(), view=view, ephemeral=True)

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

    async def _act_critic_instructions(self, interaction: discord.Interaction, profile: Dict[str, Any]):
        # Resolved, not read raw: the box has to show what this profile actually screens
        # with today -- its own prompt, else the instance-wide one, else the default --
        # or editing it starts from a blank that means something different.
        instr = self.cog.profile_manager.resolve_critic_instructions(self.user_id, self.profile_name)
        modal = ProfileCriticInstructionsModal(self.cog, self.profile_name, instr, target_user_id=self.user_id)
        await interaction.response.send_modal(modal)

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

        new_embed = await self.cog.profile_manager._build_profile_manage_embed(
            interaction, profile_name, target_user_id=self.user_id)
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

class ProfileFunctionView(TimeoutCleanupMixin, ui.View):
    """One setting's screen, rendered from its `_Screen` declaration.

    Swaps onto the dashboard's own message rather than stacking a fresh ephemeral under
    it, so Back returns to the tab it came from instead of leaving a dead panel behind
    -- the same rule _BulkSubView follows.

    Layout is forced by Discord's five-components-per-row: a select occupies a whole
    row, so the choices take rows 0..n and everything that is a button -- the toggles,
    the Edit button, Back -- shares the row after them. No declared screen has more
    than three choices or one toggle, which is what keeps that inside the five-row cap.
    """

    def __init__(self, parent: 'ProfileManageView', action: '_Action'):
        super().__init__(timeout=300)
        self.parent = parent
        self.cog = parent.cog
        self.action = action
        self.screen = action.screen
        self.original_interaction = parent.original_interaction
        self._build_view()

    @property
    def _config(self) -> Dict[str, Any]:
        return self.cog.profile_manager._get_profile_config(
            self.parent.user_id, self.parent.profile_name, self.parent.is_borrowed) or {}

    async def embed(self) -> discord.Embed:
        e = await self.cog.profile_manager.build_function_embed(
            self.parent.user_id, self.parent.profile_name,
            self.original_interaction.channel_id, self.action.value)
        if self.screen.note:
            e.description = f"{e.description}\n\n{self.screen.note}" if e.description else self.screen.note
        return e

    async def _apply(self, interaction: discord.Interaction, updates: Dict[str, Any]):
        _write_profile_config(self.cog, self.parent.user_id, self.parent.profile_name,
                              self.parent.is_borrowed, updates)
        self._build_view()
        await interaction.response.edit_message(embed=await self.embed(), view=self)

    def _build_view(self):
        self.clear_items()
        config = self._config

        row = 0
        for choice in self.screen.choices:
            current = choice.read(config)
            options = []
            for option in choice.options:
                label, value = option[0], option[1]
                description = option[2] if len(option) > 2 else None
                options.append(discord.SelectOption(
                    label=label[:100], value=value,
                    description=description[:100] if description else None,
                    default=(value == current)))
            select = ui.Select(placeholder=choice.placeholder, options=options, row=row)
            select.callback = self._choice_callback(choice)
            self.add_item(select)
            row += 1

        for toggle in self.screen.toggles:
            on = toggle.read(config)
            btn = ui.Button(
                label=f"{toggle.label}: {'On' if on else 'Off'}",
                style=discord.ButtonStyle.success if on else discord.ButtonStyle.secondary,
                row=row)
            btn.callback = self._toggle_callback(toggle, on)
            self.add_item(btn)

        if self.screen.modal:
            btn = ui.Button(label=self.screen.modal_label, style=discord.ButtonStyle.primary, row=row)
            btn.callback = self._modal_callback
            self.add_item(btn)

        if self.screen.sub_view:
            label, handler = self.screen.sub_view
            btn = ui.Button(label=label, style=discord.ButtonStyle.primary, row=row)

            async def sub_callback(interaction: discord.Interaction, _handler=handler):
                await _handler(self.parent, interaction, self._config)

            btn.callback = sub_callback
            self.add_item(btn)

        back = ui.Button(label="◀ Back", style=discord.ButtonStyle.secondary, row=row)
        back.callback = self._back_callback
        self.add_item(back)

    def _choice_callback(self, choice: '_Choice'):
        async def callback(interaction: discord.Interaction):
            await self._apply(interaction, choice.to_payload(interaction.data['values'][0]))
        return callback

    def _toggle_callback(self, toggle: '_Toggle', currently_on: bool):
        # The current value is captured when the button is built rather than re-read on
        # click: the button's own label already told the user which way it will go, and
        # re-reading would let a double click land on a stale read and flip twice.
        async def callback(interaction: discord.Interaction):
            await self._apply(interaction, toggle.to_payload(not currently_on))
        return callback

    async def _modal_callback(self, interaction: discord.Interaction):
        factory = globals()[self.screen.modal]
        config = dict(self._config)

        # The image prompt lives in `prompts`, encrypted, and its modal reads it off the
        # config dict it is handed. Seeded here for the same reason _act_image_toggle
        # seeded it: the factory signature takes one dict.
        if self.screen.modal == "ProfileImageGenSettingsModal" and not self.parent.is_borrowed:
            prompts = self.cog.profile_manager._get_profile_prompts(
                self.parent.user_id, self.parent.profile_name) or {}
            config["image_generation_prompt"] = prompts.get("image_generation_prompt")

        args = [self.cog, self.parent.profile_name, config]
        if _MODAL_TAKES_BORROWED.get(self.screen.modal, True):
            args.append(self.parent.is_borrowed)
        await interaction.response.send_modal(factory(
            *args, values_only=True, callback=self._after_modal,
            target_user_id=self.parent.user_id))

    async def _after_modal(self, interaction: discord.Interaction):
        self._build_view()
        await self.original_interaction.edit_original_response(embed=await self.embed(), view=self)

    async def _back_callback(self, interaction: discord.Interaction):
        self.stop()
        self.parent._build_view()
        embed = await self.cog.profile_manager._build_profile_manage_embed(
            interaction, self.parent.profile_name,
            target_user_id=self.parent.user_id)
        await interaction.response.edit_message(embed=embed, view=self.parent)

    async def on_error(self, interaction: discord.Interaction, error: Exception, item: ui.Item):
        print(f"ProfileFunctionView({self.action.value}) error: {error}")
        traceback.print_exc()
        try:
            await interaction.followup.send("Something went wrong applying that.", ephemeral=True)
        except Exception:
            pass


#: Factories that predate the `is_borrowed` positional and do not take it.
_MODAL_TAKES_BORROWED = {
    "ProfileDirectorDeskModal": False,
    "ProfileTrainingParamsModal": False,
    "ProfileLTMParamsModal": False,
    "ProfileLTMSummarizationModal": False,
    "ProfileCriticInstructionsModal": False,
}




class EditUserProfilePersonaModal(ui.Modal):
    def __init__(self, cog_instance, profile_name: str, current_persona_data: Dict[str, List[str]], user_id: int, callback=None):
        self.cog_instance: MimicCog = cog_instance
        self.profile_name = profile_name
        self.user_id = user_id
        self.callback = callback
        self.persona_sections_order = cog_instance.persona_modal_sections_order

        # Same sentinel ConfigModal uses: in bulk the fields are collected here and
        # handed to `callback`, so the bulk flow can never present a different set of
        # persona sections from the single-profile flow.
        title = ("Set Persona for Multiple Profiles" if profile_name == "BULK_APPLY"
                 else f"Edit Persona for Profile: '{profile_name}'")[:45]
        super().__init__(title=title)

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

        if self.profile_name == "BULK_APPLY":
            if self.callback:
                await self.callback(i, updated_persona_data)
            return

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
    def __init__(self, cog_instance, profile_name: str, current_instr:str, user_id: int, callback=None):
        self.cog:MimicCog=cog_instance
        self.profile_name = profile_name
        self.user_id = user_id
        self.callback = callback

        title = ("Set AI Instructions for Multiple Profiles" if profile_name == "BULK_APPLY"
                 else f"Edit AI Instructions for Profile: '{profile_name}'")[:45]
        super().__init__(title=title)
        
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

        if self.profile_name == "BULK_APPLY":
            if self.callback:
                await self.callback(i, instr_list)
            return

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

def ProfileParamsModal(cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, values_only: bool = False, callback=None, target_user_id: Optional[int] = None):
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

def ProfileTrainingParamsModal(cog, profile_name: str, current_params: Dict[str, Any], values_only: bool = False, callback=None, target_user_id: Optional[int] = None):
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

def ProfileThinkingParamsModal(cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, values_only: bool = False, callback=None, target_user_id: Optional[int] = None):
    fields = [] if values_only else [
        {"label": "Thinking Summary (on/off)", "custom_id": "thinking_summary_visible", "default": current_params.get("thinking_summary_visible", "off"), "required": False, "placeholder": "Display reasoning tokens below your message."},
        {"label": "Reasoning Effort / Level", "custom_id": "thinking_level", "default": current_params.get("thinking_level", "low"), "required": False, "placeholder": "xhigh, high, medium, low, minimal, none"},
    ]
    fields.append(
        {"label": "Reasoning Token Budget (-1=dyn)", "custom_id": "thinking_budget", "default": str(current_params.get("thinking_budget", -1)), "required": False, "placeholder": "-1 = dynamic, 128+ = token limit"}
    )
    def parser(v):
        c = {}
        if "thinking_summary_visible" in v:
            sv = _ps(v["thinking_summary_visible"])
            c["thinking_summary_visible"] = "on" if sv and sv.lower() == "on" else "off"

        if "thinking_level" in v:
            lv = _ps(v["thinking_level"])
            c["thinking_level"] = lv.lower() if lv and lv.lower() in ["xhigh", "high", "medium", "low", "minimal", "none"] else "high"

        bv = _pi(v["thinking_budget"])
        c["thinking_budget"] = min(bv if bv is not None and bv >= -1 else -1, 32768)
        
        return {"config": c}
    return ConfigModal(cog, profile_name, is_borrowed, "Thinking & Reasoning Parameters", fields, parser, callback, target_user_id)

def ProfileLTMParamsModal(cog, profile_name: str, current_params: Dict[str, Any], values_only: bool = False, callback=None, target_user_id: Optional[int] = None):
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

def ProfileLTMSummarizationModal(cog, profile_name: str, current_instructions: str, values_only: bool = False, callback=None, target_user_id: Optional[int] = None):
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

def ProfileCriticSettingsModal(cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, values_only: bool = False, callback=None, target_user_id: Optional[int] = None):
    """The Anti-Repetition Critic's five knobs.

    Mode replaces what was a plain on/off toggle. The lexical scan is in-process and
    costs nothing, but the boolean only ever bought it together with a model call on
    every turn, so "lexical" is the setting that gives a profile repetition screening
    for free -- which matters most on the smallest deployments, where the critic was
    the first thing the docs told people to switch off.
    """
    current = resolve_critic_settings(current_params)
    fields = [] if values_only else [
        {"label": "Mode (off/lexical/full)", "custom_id": "critic_mode",
         "default": current["mode"], "required": True,
         "placeholder": "lexical = free local scan; full = adds a model call"},
        {"label": "Scope (self/session)", "custom_id": "critic_scope",
         "default": current["scope"], "required": False,
         "placeholder": "session = screen against every participant's lines"},
        {"label": "Strictness (lenient/normal/strict)", "custom_id": "critic_strictness",
         "default": current["strictness"], "required": False, "placeholder": "Default: normal"},
    ]
    fields.extend([
        {"label": f"Lookback ({CRITIC_LOOKBACK_MIN}-{CRITIC_LOOKBACK_MAX} turns)", "custom_id": "critic_lookback",
         "default": str(current["lookback"]), "required": False,
         "placeholder": f"Default: {DEFAULT_CRITIC_LOOKBACK}"},
        {"label": f"Persistence ({CRITIC_PERSISTENCE_MIN}-{CRITIC_PERSISTENCE_MAX} rounds)", "custom_id": "critic_persistence",
         "default": str(current["persistence"]), "required": False,
         "placeholder": f"Extra rounds a constraint holds. Default: {DEFAULT_CRITIC_PERSISTENCE}"},
    ])

    def parser(v):
        c = {}
        if "critic_mode" in v:
            mode = (_ps(v["critic_mode"]) or "").lower()
            if mode not in CRITIC_MODES:
                raise ValueError(f"Mode must be one of: {', '.join(CRITIC_MODES)}")
            # critic_enabled is written alongside critic_mode, not retired. Borrowed
            # copies and exports made on an older build read the boolean, and a profile
            # round-tripping through one of those must not come back with the critic
            # silently on.
            c["critic_mode"] = mode
            c["critic_enabled"] = mode != "off"

        if "critic_scope" in v:
            scope = (_ps(v["critic_scope"]) or DEFAULT_CRITIC_SCOPE).lower()
            if scope not in CRITIC_SCOPES:
                raise ValueError(f"Scope must be one of: {', '.join(CRITIC_SCOPES)}")
            c["critic_scope"] = scope

        if "critic_strictness" in v:
            strictness = (_ps(v["critic_strictness"]) or DEFAULT_CRITIC_STRICTNESS).lower()
            if strictness not in CRITIC_STRICTNESS_LEVELS:
                raise ValueError(f"Strictness must be one of: {', '.join(CRITIC_STRICTNESS_LEVELS)}")
            c["critic_strictness"] = strictness

        lb = _pi(v["critic_lookback"])
        if lb is not None:
            if not (CRITIC_LOOKBACK_MIN <= lb <= CRITIC_LOOKBACK_MAX):
                raise ValueError(f"Lookback out of range ({CRITIC_LOOKBACK_MIN}-{CRITIC_LOOKBACK_MAX})")
            c["critic_lookback"] = lb

        pr = _pi(v["critic_persistence"])
        if pr is not None:
            if not (CRITIC_PERSISTENCE_MIN <= pr <= CRITIC_PERSISTENCE_MAX):
                raise ValueError(f"Persistence out of range ({CRITIC_PERSISTENCE_MIN}-{CRITIC_PERSISTENCE_MAX})")
            c["critic_persistence"] = pr

        return {"config": c}

    return ConfigModal(cog, profile_name, is_borrowed, "Anti-Repetition Critic", fields, parser, callback, target_user_id)


def ProfileCriticInstructionsModal(cog, profile_name: str, current_instructions: str, values_only: bool = False, callback=None, target_user_id: Optional[int] = None):
    """Per-profile critic prompt, seeded from whatever the profile resolves to today.

    Submitting blank clears the override rather than storing an empty prompt, so the
    profile goes back to following /mod's instance-wide ANTI_REPETITION.
    """
    fields = [{
        "label": "Critic Instructions",
        "custom_id": "critic_instructions",
        "style": discord.TextStyle.paragraph,
        "default": current_instructions,
        "required": False,
        "max_length": 2000,
        "placeholder": "{char_name} is substituted. Clear the box to follow the global prompt.",
    }]

    def parser(v):
        ins = _ps(v["critic_instructions"])
        return {"prompts": {"critic_instructions": cog.storage_manager._encrypt_data(ins) if ins else ""}}

    return ConfigModal(cog, profile_name, False, "Set Critic Instructions", fields, parser, callback, target_user_id)


def ProfileTypingSettingsModal(cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, values_only: bool = False, callback=None, target_user_id: Optional[int] = None):
    fields = [] if values_only else [
        {"label": "Enable Realistic Typing (on/off)", "custom_id": "realistic_typing_enabled", "default": "on" if current_params.get("realistic_typing_enabled") else "off", "required": True},
        {"label": "Mode (sentence/line)", "custom_id": "typing_mode", "default": current_params.get("typing_mode", "sentence"), "required": False, "placeholder": "Default: sentence"},
    ]
    fields.extend([
        {"label": "Characters per Second", "custom_id": "typing_cps", "default": str(current_params.get("typing_cps", 30.0)), "required": False, "placeholder": "Default: 30.0"},
        {"label": "Max Delay per Chunk (Seconds)", "custom_id": "typing_max_delay", "default": str(current_params.get("typing_max_delay", 2.5)), "required": False, "placeholder": "Default: 2.5"}
    ])
    def parser(v):
        c = {}
        if "realistic_typing_enabled" in v:
            c["realistic_typing_enabled"] = _pb(v["realistic_typing_enabled"])
        if "typing_mode" in v:
            m = _ps(v["typing_mode"])
            if m: c["typing_mode"] = "line" if m.lower() == "line" else "sentence"
        cps = _pf(v["typing_cps"])
        if cps is not None: c["typing_cps"] = cps
        md = _pf(v["typing_max_delay"])
        if md is not None: c["typing_max_delay"] = md
        return {"config": c}
    return ConfigModal(cog, profile_name, is_borrowed, "Realistic Typing Settings", fields, parser, callback, target_user_id)

def ProfileImageGenSettingsModal(cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, values_only: bool = False, callback=None, target_user_id: Optional[int] = None):
    fields = [] if values_only else [
        {"label": "Enable Image Gen (on/off)", "custom_id": "image_generation_enabled", "default": "on" if current_params.get("image_generation_enabled") else "off", "required": True}
    ]
    if not is_borrowed:
        enc = current_params.get("image_generation_prompt")
        dec = cog.storage_manager._decrypt_data(enc) if enc else ""
        fields.append({"label": "Image Generation Prompt", "custom_id": "image_generation_prompt", "style": discord.TextStyle.paragraph, "default": dec, "required": False, "max_length": 2000})
        
    def parser(v):
        c = {}
        if "image_generation_enabled" in v:
            c["image_generation_enabled"] = _pb(v["image_generation_enabled"])
        p = {}
        if not is_borrowed and "image_generation_prompt" in v:
            pr = _ps(v["image_generation_prompt"])
            p["image_generation_prompt"] = cog.storage_manager._encrypt_data(pr) if pr else None
        return {"config": c, "prompts": p}
    return ConfigModal(cog, profile_name, is_borrowed, "Image Generation Settings", fields, parser, callback, target_user_id)

class MediaOptionsMixin:
    """The dropdown machinery shared by the image-output and voice pickers.

    Both settings are fixed enumerations the API validates strictly, and neither
    survives a text box: "16;9" and "Kore " are accepted by a modal and answered by a
    400 -- which reaches the user as a missing image, or, for TTS, as silence, because
    `_generate_google_tts` swallows the failure and returns no stream. A dropdown of the
    values the model actually carries is the only version of this that cannot be typed
    wrong.

    Adopters provide `_current_value(key)`, `_apply(key, value)`, `_render()` and
    `_build_view()`. The single-profile pickers write straight through to the profile;
    the bulk ones stage into the wizard's changeset, which is the only difference
    between them.
    """

    #: The "let the model decide" row. Applied as "" rather than as a sentinel, because
    #: an absent value is already what MediaService.resolve_image_output_params reads as
    #: "send no such field" -- storing a marker would only mean stripping it back out.
    AUTO = "__auto__"

    def _add_choice_select(self, key: str, placeholder: str, values, notes: Dict[str, str],
                           row: int, auto_label: str = "Model default"):
        current = self._current_value(key)
        options = [discord.SelectOption(
            label=auto_label, value=self.AUTO,
            description="Send no preference; the model picks.",
            default=not current)]
        for value in values:
            options.append(discord.SelectOption(
                label=value, value=value,
                description=(notes.get(value) or None),
                default=(str(current or "") == value)))

        select = ui.Select(placeholder=placeholder, options=options[:25], row=row)

        async def callback(interaction: discord.Interaction):
            chosen = select.values[0]
            self._apply(key, "" if chosen == self.AUTO else chosen)
            self._build_view()
            await interaction.response.edit_message(**self._render())

        select.callback = callback
        self.add_item(select)

    def _add_voice_select(self, key: str, row: int):
        """The voice list, one gender at a time, with the other reachable from inside.

        Paged through an option rather than a pair of buttons because the select has an
        option cap but the view has rows to spare, and it keeps the whole control in one
        component -- the timezone picker settled the same question the same way.

        The gender is Google's own attribute for the voice, published on the Cloud
        Text-to-Speech side rather than in the Gemini API docs. It is the page break
        because it is the filter someone casting a character reaches for first; the
        one-word character narrows it from there.
        """
        current = self._current_value(key) or DEFAULT_SPEECH_VOICE
        gender, chunk = TTS_VOICE_GROUPS[self.voice_page]

        options = []
        for page_idx, (other_gender, other) in enumerate(TTS_VOICE_GROUPS):
            if page_idx == self.voice_page:
                continue
            options.append(discord.SelectOption(
                label=f"Switch to {other_gender.lower()} voices",
                value=f"__page_{page_idx}", emoji="📑",
                description=f"{len(other)} voices"))

        for name, character, _gender in chunk:
            options.append(discord.SelectOption(
                label=name, value=name, description=character,
                default=(name == current)))

        select = ui.Select(
            placeholder=f"Choose a voice ({gender.lower()}, {len(chunk)})...",
            options=options, row=row)

        async def callback(interaction: discord.Interaction):
            chosen = select.values[0]
            if chosen.startswith("__page_"):
                self.voice_page = int(chosen.rsplit("_", 1)[1])
            else:
                self._apply(key, chosen)
            self._build_view()
            await interaction.response.edit_message(**self._render())

        select.callback = callback
        self.add_item(select)


class SingleProfileMediaOptionsView(MediaOptionsMixin, ui.View):
    """Image output settings and TTS voice for one profile, written as they are chosen.

    One view over both subjects rather than two near-identical ones: they differ only
    in which selects get built, which is `mode`.
    """

    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction, profile_name: str,
                 mode: str, is_borrowed: bool = False, user_id: Optional[int] = None):
        super().__init__(timeout=300)
        self.cog = cog
        self.original_interaction = interaction
        self.user_id = user_id or interaction.user.id
        self.profile_name = profile_name
        self.is_borrowed = is_borrowed
        self.mode = mode
        self.voice_page = 0
        self._build_view()

    def _profile(self) -> Dict[str, Any]:
        return self.cog.profile_manager._get_profile_config(
            self.user_id, self.profile_name, self.is_borrowed) or {}

    def _current_value(self, key: str):
        return self._profile().get(key)

    def _apply(self, key: str, value: Any):
        data = self._profile()
        data[key] = value
        self.cog.profile_manager._save_profile_config(
            self.user_id, self.profile_name, data, self.is_borrowed)

    def _render(self) -> Dict[str, Any]:
        return {"content": self._feedback(), "view": self}

    def _feedback(self) -> str:
        data = self._profile()
        lines = [f"**Profile:** `{self.profile_name}`"]

        if self.mode == "image":
            raw = data.get("image_generation_model") or DEFAULT_IMAGE_MODEL
            caps = image_model_caps(raw)
            lines.append(f"**Image model:** `{raw}`")
            lines.append(f"**Aspect ratio:** `{data.get('image_aspect_ratio') or 'model default'}`")
            if caps["sizes"]:
                lines.append(f"**Resolution:** `{data.get('image_size') or 'model default'}`")
            else:
                lines.append("**Resolution:** `fixed by the model` — this model renders at one "
                             "size and rejects a resolution request.")
            if caps["thinking"]:
                lines.append(f"**Thinking:** `{data.get('image_thinking_level') or 'model default'}`")
            lines.append(f"\nOptions are limited to what `{raw}` accepts. Anything this model "
                         "does not carry is kept on the profile but left out of the request, so "
                         "switching models back restores it.")
            if not data.get("image_generation_enabled"):
                lines.append("\n⚠️ Image generation is currently **off** for this profile.")
        else:
            voice = data.get("speech_voice") or DEFAULT_SPEECH_VOICE
            described = " · ".join(d for d in (TTS_VOICE_GENDER.get(voice),
                                               TTS_VOICE_CHARACTER.get(voice)) if d)
            lines.append(f"**Voice:** `{voice}`" + (f" ({described})" if described else ""))
            lines.append(f"**Speech model:** `{data.get('speech_model') or DEFAULT_SPEECH_MODEL}`")
            lines.append("\nVoices are grouped by gender, then described by Google's own "
                         "one-word character. Everything beyond that — accent, mood, pacing — is "
                         "the Director's Desk, not the voice.")
            if not data.get("speech_tts_enabled"):
                lines.append("\n⚠️ TTS is currently **off** for this profile.")

        return "\n".join(lines)

    def _build_view(self):
        self.clear_items()
        if self.mode == "image":
            caps = image_model_caps(self._profile().get("image_generation_model") or DEFAULT_IMAGE_MODEL)
            self._add_choice_select("image_aspect_ratio", "Aspect ratio...",
                                    caps["ratios"], IMAGE_ASPECT_RATIO_NOTES, 0)
            if caps["sizes"]:
                self._add_choice_select("image_size", "Resolution...",
                                        caps["sizes"], IMAGE_SIZE_NOTES, 1)
            if caps["thinking"]:
                self._add_choice_select("image_thinking_level", "Thinking level...",
                                        IMAGE_THINKING_LEVELS, IMAGE_THINKING_NOTES, 2)
        else:
            self._add_voice_select("speech_voice", 0)


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

    #: category -> the (config key, wording, default) triples it presents, in row order.
    #: Shared so the two pickers cannot disagree about which models a category holds --
    #: they already had, with the single-profile summary defaulting the image model to an
    #: unprefixed id that its own builder prefixed.
    #: One category per function, each holding that function's primary and its retry.
    #: Media and Tools used to pair two unrelated functions in one category, which left
    #: nowhere to put a fallback for either of them -- two slots, both already spoken
    #: for. Splitting them gives every model in the bot the same primary/fallback shape
    #: the response model has always had.
    _CATEGORY_KEYS = {
        'response':  (("primary_model", "Primary", PRIMARY_MODEL_NAME),
                      ("fallback_model", "Fallback", FALLBACK_MODEL_NAME)),
        'image':     (("image_generation_model", "Image Generation", DEFAULT_IMAGE_MODEL),
                      ("image_generation_fallback_model", "Image Fallback", NO_FALLBACK)),
        'tts':       (("speech_model", "Text-to-Speech", DEFAULT_SPEECH_MODEL),
                      ("speech_fallback_model", "TTS Fallback", NO_FALLBACK)),
        'grounding': (("grounding_rag_model", "Grounding Summariser", FALLBACK_MODEL_NAME),
                      ("grounding_rag_fallback_model", "Grounding Fallback", NO_FALLBACK)),
        'critic':    (("critic_model", "Anti-Repetition Critic", FALLBACK_MODEL_NAME),
                      ("critic_fallback_model", "Critic Fallback", NO_FALLBACK)),
        'ltm':       (("ltm_model", "LTM Summariser", FALLBACK_MODEL_NAME),
                      ("ltm_fallback_model", "LTM Fallback", NO_FALLBACK)),
    }

    _CATEGORY_LABELS = (
        ("response", "Response", "The models behind the profile's own replies."),
        ("image", "Image Generation", "The model behind !image and !imagine."),
        ("tts", "TTS Generation", "The speech model used for voice rounds."),
        ("grounding", "Grounding Summariser", "Summarises web search results in RAG mode."),
        ("critic", "Anti-Repetition Critic", "Screens replies for semantic repetition."),
        ("ltm", "LTM Summariser", "Turns conversations into long-term memories."),
    )

    #: Categories whose every slot is in GOOGLE_ONLY_MODEL_KEYS. They pin the API
    #: switch to Google rather than letting a stale mode sit behind a disabled button.
    #: Grounding joins image and TTS here: it attaches the native `google_search` tool,
    #: so an OpenRouter id in that slot was never honoured -- it resolved to the Google
    #: default at call time, which read as the picker having accepted the choice.
    _GOOGLE_ONLY_CATEGORIES = ("image", "tts", "grounding")

    @classmethod
    def display_model(cls, value) -> str:
        """How a stored model value reads in a summary."""
        if not is_real_model(value):
            return "None (no retry)"
        return cls.strip_prefix(value)

    @staticmethod
    def strip_prefix(value) -> str:
        """Drops the internal routing prefix so the user sees the model they picked."""
        text = str(value)
        for prefix in ("GOOGLE/", "OPENROUTER/", "OLLAMA/"):
            if text.startswith(prefix):
                return text[len(prefix):]
        return text

    def _add_category_select(self, row: int = 0):
        """The category picker, as a dropdown rather than a button that cycles.

        It was one button labelled with the *current* category, advancing a step per
        click: reaching LTM from Response took three clicks, and the other three
        categories were named nowhere in the interface. The single-profile view then
        carried a second, redundant row of category buttons besides. One dropdown, four
        named options, one click to any of them.
        """
        options = [discord.SelectOption(label=f"Category: {label}", value=value,
                                        description=desc, default=(self.category == value))
                   for value, label, desc in self._CATEGORY_LABELS]
        select = ui.Select(placeholder="Choose a model category...", options=options, row=row)

        async def callback(interaction: discord.Interaction):
            self.category = select.values[0]
            self._build_view()
            await interaction.response.edit_message(**self._picker_render())

        select.callback = callback
        self.add_item(select)

    def _picker_render(self) -> Dict[str, Any]:
        """Edit kwargs for this picker's own message.

        A hook because the two adopters render differently: the single-profile picker
        is a plain-text message, and the bulk picker is a step of the embed-rendered
        wizard. Every rebuild in this mixin -- and in CustomModelModal and
        OllamaHostModal, which reach back into the view -- goes through here, so
        neither one can overwrite the other's message body with the wrong shape.
        """
        return {"content": self._get_selection_feedback_message(), "view": self}

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
                await interaction.response.edit_message(**view._picker_render())

    def _create_model_options(self, current_val: str, target_config_key: str) -> List[discord.SelectOption]:
        top_models = self._get_top_models(self.view_mode, target_config_key)
        opts = [discord.SelectOption(label="Custom Model...", value="custom_option", description="Enter manually via modal")]

        # Only the five utility fallbacks can be switched off. The response fallback is
        # what _instantiate_model retries onto when the primary will not construct, so
        # it has to name a real model.
        if target_config_key in UTILITY_FALLBACK_KEYS.values():
            opts.append(discord.SelectOption(
                label="None (no retry)", value=NO_FALLBACK,
                description="Do not try a second model when this one fails.",
                default=not is_real_model(current_val)))
            if not is_real_model(current_val):
                current_val = None

        if current_val:
            # The [:100] truncation was present only in the single-profile copy. Discord
            # rejects an option label over 100 characters, so the bulk picker would raise
            # on a long custom model id; sharing this version fixes that.
            opts.append(discord.SelectOption(label=f"Current: {current_val}"[:100], value=current_val, default=True))
        
        prefix = "GOOGLE/"
        if self.view_mode == 'openrouter': prefix = "OPENROUTER/"
        elif self.view_mode == 'ollama': prefix = "OLLAMA/"
        
        if target_config_key in GOOGLE_ONLY_MODEL_KEYS:
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

    def _add_api_buttons(self, *, row: int = 3):
        """The API-mode and Ollama-host buttons, identical in both pickers.

        The category button that used to live here is now `_add_category_select`, which
        leaves room on this row for the fallback-indicator toggle that had a row of its
        own.
        """
        api_modes = ['google', 'openrouter', 'ollama']
        api_labels = {'google': 'API: Google', 'openrouter': 'API: OpenRouter', 'ollama': 'API: Ollama (Local)'}
        
        btn_api = ui.Button(label=api_labels[self.view_mode], style=discord.ButtonStyle.primary, row=row,
                            disabled=(self.category in self._GOOGLE_ONLY_CATEGORIES))
        async def api_cb(i: discord.Interaction):
            next_idx = (api_modes.index(self.view_mode) + 1) % len(api_modes)
            self.view_mode = api_modes[next_idx]
            if self.view_mode == 'ollama':
                await i.response.defer()
                self.ollama_working = "processing"
                await self._update_ollama_status()
                self._build_view()
                await i.edit_original_response(**self._picker_render())
            else:
                self._build_view()
                await i.response.edit_message(**self._picker_render())
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
        lines = [f"**Profile:** `{self.profile_name}`"]
        if self.view_mode == 'openrouter':
            lines.append("⚠️ **Note:** OpenRouter / Custom models require **RAG Mode** for "
                         "Grounding and URL Context.\n")
        elif self.view_mode == 'ollama':
            lines.append("⚠️ **Note:** Localhost models run on your machine's hardware. "
                         "Processing speed depends on your GPU/CPU.\n")

        for key, wording, default in self._CATEGORY_KEYS[self.category]:
            lines.append(f"**{wording}:** `{self.display_model(data.get(key, default))}`")
        if self.category == 'response':
            state = "ON" if data.get("show_fallback_indicator", True) else "OFF"
            lines.append(f"**Fallback Indicator:** `{state}`")
        return "\n".join(lines)

    def _build_view(self):
        self.clear_items()
        data = self._get_current_profile_data()

        if self.category in self._GOOGLE_ONLY_CATEGORIES:
            self.view_mode = 'google'

        self._add_category_select(0)

        for offset, (key, wording, default) in enumerate(self._CATEGORY_KEYS[self.category]):
            self.add_item(self.GenericModelSelect(
                f"Select {wording} Model...",
                self._create_model_options(data.get(key, default), key), offset + 1, key))

        self._add_api_buttons(row=3)

        if self.category == 'response':
            show_fb = data.get("show_fallback_indicator", True)
            btn_fallback = ui.Button(
                label=f"Fallback Indicator: {'ON' if show_fb else 'OFF'}",
                style=discord.ButtonStyle.success if show_fb else discord.ButtonStyle.secondary,
                row=3)

            async def fallback_cb(i: discord.Interaction):
                self._save_changes("show_fallback_indicator", not show_fb)
                self._build_view()
                await i.response.edit_message(**self._picker_render())

            btn_fallback.callback = fallback_cb
            self.add_item(btn_fallback)


#: Shared by the wizard and every sub-view. Deliberately under fifteen minutes: after a
#: modal submit the panel can only be refreshed through the slash command's own
#: interaction token, and that token dies at fifteen. The views have to go first.
_BULK_TIMEOUT = 840


class _BulkSubView(ui.View):
    """A wizard step that borrows the wizard's message instead of opening its own.

    Every step of the bulk flow used to be a fresh `followup.send(ephemeral=True)`,
    which is why there was never a Back button: there was nothing to go back *to*, only
    a trail of dead ephemeral messages with the live one at the bottom. A sub-view is
    edited onto the same message and carries a Back that restores the panel exactly as
    it was left.
    """

    def __init__(self, wizard, *, timeout: int = _BULK_TIMEOUT):
        super().__init__(timeout=timeout)
        self.wizard = wizard
        self.cog = wizard.cog
        self.user_id = wizard.user_id
        self.session = wizard.session

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Keeps the wizard's timer alive while the user is working inside a sub-view.

        A view's timeout only resets when its own components are used, and a sub-view
        occupies the wizard's message -- so a long visit to the timezone or model picker
        would let the panel expire underneath it and discard everything staged.
        `_refresh_timeout` is discord.py's own mechanism for this (View.dispatch calls it
        on every interaction); guarded so a rename in a future release degrades to the
        old behaviour rather than breaking the view.
        """
        refresh = getattr(self.wizard, "_refresh_timeout", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:
                pass
        return True

    def _add_back(self, row: int, label: str = "◀ Back"):
        btn = ui.Button(label=label, style=discord.ButtonStyle.secondary, row=row)

        async def cb(interaction: discord.Interaction):
            await self.wizard.refresh(interaction)

        btn.callback = cb
        self.add_item(btn)

    def embed(self) -> discord.Embed:
        raise NotImplementedError

    def _build_view(self):
        raise NotImplementedError

    async def refresh(self, interaction: discord.Interaction):
        self._build_view()
        await interaction.response.edit_message(embed=self.embed(), view=self)

    async def on_error(self, interaction: discord.Interaction, error: Exception, item: ui.Item):
        print(f"Error in {type(self).__name__}: {error}")
        traceback.print_exc()
        if not interaction.response.is_done():
            await interaction.response.send_message("An unexpected error occurred with this view.", ephemeral=True)
        else:
            await interaction.followup.send("An unexpected error occurred with this view.", ephemeral=True)


class ModelApplyView(ModelPickerMixin, _BulkSubView):
    """The bulk model picker, as a staging step.

    It used to own a profile picker and an apply loop of its own, which is precisely
    why choosing models could not be combined with anything else -- it selected its own
    targets and finished. It now writes into the wizard's changeset and hands control
    back, so "set the primary model, raise the temperature and turn realistic typing
    on" costs one read-modify-write per profile rather than three.
    """

    #: The bulk picker refers to the shared select under its historical name.
    GenericBulkModelSelect = ModelPickerMixin.GenericModelSelect

    def __init__(self, wizard):
        super().__init__(wizard)
        self.view_mode = 'google'
        self.category = 'response'
        self.ollama_working = None
        self.cached_ollama_models = []
        self.models_state = {k: None for k in (
            'primary_model', 'fallback_model',
            'image_generation_model', 'image_generation_fallback_model',
            'speech_model', 'speech_fallback_model',
            'grounding_rag_model', 'grounding_rag_fallback_model',
            'critic_model', 'critic_fallback_model',
            'ltm_model', 'ltm_fallback_model',
            'ollama_host_url')}
        self.show_fallback_indicator = None

        # Seeded from whatever an earlier visit already staged, so reopening the picker
        # shows the pending value instead of reporting "Unchanged" over the top of it.
        for key in self.models_state:
            if key in self.session.config:
                self.models_state[key] = self.session.config[key]
        if "show_fallback_indicator" in self.session.config:
            self.show_fallback_indicator = self.session.config["show_fallback_indicator"]

        self._build_view()

    def _ollama_host_url(self) -> str:
        return self.models_state.get("ollama_host_url")

    def _save_changes(self, key: str, value: Any):
        if key == "show_fallback_indicator":
            self.show_fallback_indicator = value
        else:
            self.models_state[key] = value

    def _picker_render(self) -> Dict[str, Any]:
        return {"content": None, "embed": self.embed(), "view": self}

    @classmethod
    def _clean(cls, val) -> str:
        """As the mixin's, plus the nothing-staged case only the bulk picker has.

        `None` means "this slot is not part of the changeset"; the explicit NONE
        sentinel means "stage the absence of a retry". They read differently because
        they do different things.
        """
        if val is None:
            return "Unchanged"
        if isinstance(val, bool):
            return "On" if val else "Off"
        return cls.display_model(val)

    def _pending(self) -> Dict[str, Any]:
        """The config keys this picker would stage, given what is currently chosen."""
        updates = {k: v for k, v in self.models_state.items() if v is not None}
        if self.show_fallback_indicator is not None:
            updates["show_fallback_indicator"] = self.show_fallback_indicator
        return updates

    def _get_selection_feedback_message(self) -> str:
        """Named by the mixin. This picker renders an embed, so it is only a fallback."""
        return self.embed().description or ""

    def embed(self) -> discord.Embed:
        e = discord.Embed(title="Set Models", colour=discord.Colour.blurple())
        # The declared label, not category.title(): the ids are short keys, so titling
        # them rendered "Tts" and "Ltm" at the user.
        wording = next((lbl for value, lbl, _ in self._CATEGORY_LABELS if value == self.category),
                       self.category.title())
        lines = [f"**{wording}** — switch categories to stage more than one kind of model "
                 f"in the same pass."]
        if self.view_mode == 'openrouter':
            lines.append("-# OpenRouter and custom models require **RAG mode** for Grounding "
                         "and URL Context.")
        elif self.view_mode == 'ollama':
            lines.append("-# Localhost models run on your own hardware; speed depends on "
                         "your GPU/CPU.")
        e.description = "\n".join(lines)

        for key, wording, _default in self._CATEGORY_KEYS[self.category]:
            e.add_field(name=wording, value=f"`{self._clean(self.models_state[key])}`", inline=True)
        if self.category == 'response':
            state = self.show_fallback_indicator
            e.add_field(name="Fallback Indicator",
                        value=f"`{'Unchanged' if state is None else self._clean(state)}`",
                        inline=True)

        pending = self._pending()
        if pending:
            body = "\n".join(f"• {k.replace('_', ' ').title()}: `{self._clean(v)}`"
                             for k, v in pending.items())
            e.add_field(name=f"Staged from this picker ({len(pending)})",
                        value=body[:1024], inline=False)

        e.set_footer(text=f"{len(self.wizard.selected_profiles)} profile(s) selected · "
                          f"nothing is written until Apply")
        return e

    def _build_view(self):
        self.clear_items()
        if self.category in self._GOOGLE_ONLY_CATEGORIES:
            self.view_mode = 'google'

        self._add_category_select(0)

        for offset, (key, wording, _default) in enumerate(self._CATEGORY_KEYS[self.category]):
            self.add_item(self.GenericBulkModelSelect(
                f"Select {wording} Model...",
                self._create_model_options(self.models_state[key], key), offset + 1, key))

        self._add_api_buttons(row=3)

        if self.category == 'response':
            state = self.show_fallback_indicator
            label = ("Fallback Indicator: Unchanged" if state is None
                     else f"Fallback Indicator: {'ON' if state else 'OFF'}")
            style = (discord.ButtonStyle.secondary if state is None
                     else (discord.ButtonStyle.success if state else discord.ButtonStyle.danger))
            btn = ui.Button(label=label, style=style, row=3)

            async def fallback_cb(i: discord.Interaction):
                # Tri-state: unchanged -> on -> off -> unchanged, so the indicator can be
                # taken back out of the changeset without abandoning the model choices.
                self.show_fallback_indicator = True if state is None else (False if state else None)
                self._build_view()
                await i.response.edit_message(**self._picker_render())

            btn.callback = fallback_cb
            self.add_item(btn)

        self._add_back(4)
        stage = ui.Button(label="Stage Models", style=discord.ButtonStyle.success, row=4,
                          disabled=not self._pending())
        stage.callback = self._stage_callback
        self.add_item(stage)

    async def _stage_callback(self, interaction: discord.Interaction):
        pending = self._pending()
        if not pending:
            await interaction.response.send_message(
                "Choose at least one model first.", ephemeral=True)
            return
        self.wizard._stage_change("models", config=pending)
        await self.wizard.refresh(interaction)


class _MediaOptionsApplyView(MediaOptionsMixin, _BulkSubView):
    """The bulk counterpart: same dropdowns, staged instead of written.

    Offers the *full* option set rather than one model's, because the selection can hold
    profiles on four different image models. Nothing is lost by that:
    `resolve_image_output_params` drops a setting the chosen model does not carry at
    request time, per profile, which is the same answer the single-profile picker
    reaches by filtering the list up front.
    """

    #: Subclass hooks: which keys this view stages, and under which action name.
    ACTION = ""
    TITLE = ""

    def __init__(self, wizard):
        super().__init__(wizard)
        self.voice_page = 0
        self.staged: Dict[str, Any] = {}
        self._build_view()

    def _current_value(self, key: str):
        return self.staged.get(key)

    def _apply(self, key: str, value: Any):
        self.staged[key] = value

    def _render(self) -> Dict[str, Any]:
        return {"embed": self.embed(), "view": self}

    def _stage_row(self, row: int, label: str):
        self._add_back(row)
        stage = ui.Button(label=label, style=discord.ButtonStyle.success, row=row,
                          disabled=not self.staged)
        stage.callback = self._stage_callback
        self.add_item(stage)

    async def _stage_callback(self, interaction: discord.Interaction):
        if not self.staged:
            await interaction.response.send_message("Choose something first.", ephemeral=True)
            return
        self.wizard._stage_change(self.ACTION, config=dict(self.staged))
        await self.wizard.refresh(interaction)


class ImageOutputApplyView(_MediaOptionsApplyView):
    ACTION = "image_output"
    TITLE = "Set Image Output"

    def embed(self) -> discord.Embed:
        chosen = {k: self.staged.get(k) for k in IMAGE_OUTPUT_KEYS}
        e = discord.Embed(
            title=self.TITLE, colour=discord.Colour.blurple(),
            description=("Aspect ratio, resolution and reasoning depth for generated "
                         "images.\n\nThe full list is offered here because the selected "
                         "profiles may be on different image models. Each request drops "
                         "whatever its own model does not accept, so a profile on a model "
                         "with one fixed resolution simply ignores a staged one."))
        for key, label in (("image_aspect_ratio", "Aspect ratio"),
                           ("image_size", "Resolution"),
                           ("image_thinking_level", "Thinking")):
            value = chosen.get(key)
            shown = "model default" if value == "" else (value or "unchanged")
            e.add_field(name=label, value=f"`{shown}`", inline=True)
        e.set_footer(text=f"{len(self.wizard.selected_profiles)} profile(s) selected · "
                          f"nothing is written until Apply")
        return e

    def _build_view(self):
        self.clear_items()
        self._add_choice_select("image_aspect_ratio", "Aspect ratio...",
                                IMAGE_ASPECT_RATIOS_FULL, IMAGE_ASPECT_RATIO_NOTES, 0)
        self._add_choice_select("image_size", "Resolution...",
                                IMAGE_SIZES_ALL, IMAGE_SIZE_NOTES, 1)
        self._add_choice_select("image_thinking_level", "Thinking level...",
                                IMAGE_THINKING_LEVELS, IMAGE_THINKING_NOTES, 2)
        self._stage_row(3, "Stage Image Output")


class VoiceApplyView(_MediaOptionsApplyView):
    ACTION = "voice"
    TITLE = "Choose TTS Voice"

    def embed(self) -> discord.Embed:
        chosen = self.staged.get("speech_voice")
        e = discord.Embed(
            title=self.TITLE, colour=discord.Colour.blurple(),
            description=("One of the thirty prebuilt Gemini voices, grouped by gender and "
                         "described by Google's own one-word character.\n\nThe voice is the "
                         "instrument; accent, mood and pacing come from the Director's Desk, "
                         "which is a separate row."))
        described = " · ".join(d for d in (TTS_VOICE_GENDER.get(chosen),
                                           TTS_VOICE_CHARACTER.get(chosen)) if d)
        e.add_field(name="Voice",
                    value=f"`{chosen or 'unchanged'}`" + (f" ({described})" if described else ""),
                    inline=True)
        e.set_footer(text=f"{len(self.wizard.selected_profiles)} profile(s) selected · "
                          f"nothing is written until Apply")
        return e

    def _build_view(self):
        self.clear_items()
        self._add_voice_select("speech_voice", 0)
        self._stage_row(1, "Stage Voice")


class _BulkSession:
    """Everything the wizard has collected, as one changeset.

    This is the whole point of the wizard shape. The four action keys the old bulk
    manager carried -- `update_config`, `set_key`, `update_both` and `update_prompts` --
    all reduce to merging into one of two dicts, so several actions staged together cost
    one read-modify-write per profile instead of one per action per profile. On the
    deployment target that is the difference between forty and a hundred and twenty
    Fernet+zstd round trips for three settings across forty profiles.

    `scope` is fixed at step one and is what lets the action list be filtered rather
    than the write silently skipping: a personal-only row is never offered against a
    selection that contains borrowed profiles.
    """

    __slots__ = ("scope", "targets", "config", "prompts", "declaration", "staged")

    def __init__(self):
        self.scope = "personal"
        self.targets: Set[str] = set()
        self.config: Dict[str, Any] = {}
        self.prompts: Dict[str, Any] = {}
        # Not a config key: the 18+ declaration is a content_rating record with its own
        # writer and its own refusals, applied separately below.
        self.declaration: Optional[bool] = None
        # action value -> bulk label, in stage order. Drives the review screen's warning
        # list and the ✓ marks; the *values* shown on review come from config/prompts, so
        # two actions touching one key show the resolved truth rather than both writes.
        self.staged: Dict[str, str] = {}

    @property
    def has_changes(self) -> bool:
        return bool(self.config or self.prompts or self.declaration is not None)

    def clear_changes(self):
        self.config.clear()
        self.prompts.clear()
        self.declaration = None
        self.staged.clear()


async def _apply_bulk_session(cog, user_id: int, session: _BulkSession) -> Dict[str, int]:
    """Writes the whole changeset in one pass over the targets.

    Returns counts rather than a message so the caller decides the wording:
    `changed`, plus `skipped_prompts` and `skipped_declaration` for the profiles a
    write could not legitimately touch. Reporting those explicitly is deliberate --
    the previous apply loop saved an untouched config back to disk for a borrowed
    profile and counted it as a success, so it claimed to have replaced forty personas
    while replacing none.
    """
    counts = {"changed": 0, "skipped_prompts": 0, "skipped_declaration": 0}
    index = cog.profile_manager._get_user_index(user_id)
    borrowed = set(index.get("borrowed", []) or [])

    published: Set[str] = set()
    if session.declaration is not None:
        # A published profile flipped to 18+ fails the publish gate and has to be found
        # and reverted by hand, so it is withheld here rather than half-applied. One
        # pass over the public index, not an is-published call per profile.
        published = {d["profile_name"] for d in cog.profile_manager._iter_public_entries(user_id)}

    for position, name in enumerate(sorted(session.targets)):
        is_borrowed = name in borrowed
        touched = False

        # Declaration first: it re-reads and rewrites the config itself, so doing it
        # ahead of the config merge means the merge below reads the post-declaration
        # file rather than clobbering it from a stale copy.
        if session.declaration is not None:
            if is_borrowed or name in published:
                counts["skipped_declaration"] += 1
            elif cog.profile_manager.set_owner_adult_declaration(user_id, name, session.declaration):
                touched = True
            else:
                counts["skipped_declaration"] += 1

        if session.prompts:
            # Prompts live only on profiles the user owns.
            if is_borrowed:
                counts["skipped_prompts"] += 1
            else:
                prompts = cog.profile_manager._get_profile_prompts(user_id, name)
                if prompts:
                    prompts.update(session.prompts)
                    cog.profile_manager._save_profile_prompts(user_id, name, prompts)
                    touched = True

        if session.config:
            profile = cog.profile_manager._get_profile_config(user_id, name, is_borrowed)
            if profile:
                profile.update(session.config)
                cog.profile_manager._save_profile_config(user_id, name, profile, is_borrowed)
                touched = True

        if touched:
            # One profile counts once, however many files it took.
            counts["changed"] += 1

        # Each profile is a Fernet decrypt, a zstd round trip and an atomic write, and
        # a hundred of them back to back holds the loop long enough to miss a Discord
        # heartbeat. Yielding periodically costs nothing and keeps the gateway alive.
        if position % 10 == 9:
            await asyncio.sleep(0)

    if counts["changed"]:
        # Hot-swap: a live session holding a cached model for any of this user's
        # profiles would otherwise keep the pre-edit settings until eviction.
        keys = [k for k in cog.channel_models.keys()
                if isinstance(k, tuple) and len(k) >= 2 and k[1] == user_id]
        for k in keys:
            cog.channel_models.pop(k, None)
            cog.channel_model_last_profile_key.pop(k, None)

    return counts


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

        new_embed = await self.cog.profile_manager._build_profile_manage_embed(
            interaction, self.parent_manage_view.profile_name,
            target_user_id=self.parent_manage_view.user_id)
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

            new_embed = await self.parent_view.cog.profile_manager._build_profile_manage_embed(
                interaction, self.parent_view.parent_manage_view.profile_name,
                target_user_id=self.parent_view.parent_manage_view.user_id)
            await self.parent_view.parent_manage_view.original_interaction.edit_original_response(embed=new_embed, view=self.parent_view.parent_manage_view)
            await interaction.response.edit_message(content=f"✅ Timezone set to **{canonical_tz}**.", view=None)
        else:
            self.parent_view.selected_tz = canonical_tz
            await self.parent_view.refresh(interaction)

# Alias for backward compatibility
BulkTimezoneModal = CustomTimezoneModal

class BulkTimezoneView(_BulkSubView):
    """Timezone picker, as a staging step.

    Four pages of twenty zones plus manual IANA entry is more than fits beside anything
    else, which is why this is a sub-view rather than a plain choice step. It no longer
    owns a profile picker or an apply loop: it produces two config keys and hands them
    back.
    """

    def __init__(self, wizard):
        super().__init__(wizard)
        self.selected_tz = self.session.config.get("timezone")
        self.tz_page = 0
        self.tz_total_pages = (len(EXTENSIVE_TIMEZONES) - 1) // 20 + 1
        self._build_view()

    def embed(self) -> discord.Embed:
        e = discord.Embed(
            title="Set Time & Timezone",
            colour=discord.Colour.blurple(),
            description=("Staging a timezone also switches **time awareness on** for every "
                         "selected profile — a zone set on a profile that never reads the "
                         "clock does nothing.\n\nBrowse a region, jump between regions, or "
                         "enter any IANA timezone ID by hand."))
        e.add_field(name="Region", value=PARTITION_NAMES[self.tz_page], inline=True)
        e.add_field(name="Chosen", value=f"`{self.selected_tz or 'nothing yet'}`", inline=True)
        e.set_footer(text=f"{len(self.wizard.selected_profiles)} profile(s) selected · "
                          f"nothing is written until Apply")
        return e

    def _build_view(self):
        self.clear_items()
        per_page = 20
        start = self.tz_page * per_page

        options = [discord.SelectOption(
            label="⚙️ Custom / Manual Timezone ID...", value="custom",
            description="Enter any custom IANA timezone ID manually.", emoji="✏️")]
        for page_idx, p_name in enumerate(PARTITION_NAMES):
            if page_idx != self.tz_page:
                options.append(discord.SelectOption(
                    label=f"🌍 Jump: {p_name}", value=f"jump_{page_idx}",
                    description=f"Switch to page {page_idx + 1} ({p_name})", emoji="📑"))
        for label, tz_val, desc in EXTENSIVE_TIMEZONES[start:start + per_page]:
            options.append(discord.SelectOption(
                label=label[:100], value=tz_val, description=desc[:100],
                default=(tz_val == self.selected_tz)))

        select = ui.Select(placeholder=f"Choose a timezone ({PARTITION_NAMES[self.tz_page]})...",
                           options=options, row=0)
        select.callback = self.tz_callback
        self.add_item(select)

        self._add_back(1)
        stage = ui.Button(label="Stage Timezone", style=discord.ButtonStyle.success, row=1,
                          disabled=not self.selected_tz)
        stage.callback = self._stage_callback
        self.add_item(stage)

    async def tz_callback(self, interaction: discord.Interaction):
        val = interaction.data['values'][0]
        if val == "custom":
            await interaction.response.send_modal(CustomTimezoneModal(self))
            return
        if val.startswith("jump_"):
            self.tz_page = int(val.split("_")[1])
            await self.refresh(interaction)
            return
        _, canonical = _resolve_zoneinfo(val)
        self.selected_tz = canonical
        await self.refresh(interaction)

    async def _stage_callback(self, interaction: discord.Interaction):
        if not self.selected_tz:
            await interaction.response.send_message("Choose a timezone first.", ephemeral=True)
            return
        self.wizard._stage_change(
            "time", config={"timezone": self.selected_tz, "time_tracking_enabled": True})
        await self.wizard.refresh(interaction)


def ProfileGenerationVisualModal(cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, values_only: bool = False, callback=None, target_user_id: Optional[int] = None):
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
    ]
    if not values_only:
        fields.append(
            {"label": "Placeholder for Child Bot (on/off)", "custom_id": "child_bot_placeholder", "default": "on" if current_params.get("child_bot_placeholder") else "off", "required": True, "max_length": 10}
        )
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
        if "child_bot_placeholder" in v:
            c["child_bot_placeholder"] = _pb(v["child_bot_placeholder"])
        return {"config": c}
    return ConfigModal(cog, profile_name, is_borrowed, "Generation Visual", fields, parser, callback, target_user_id)

def _unlink_profiles(cog, user_id: int, names: List[str]) -> List[str]:
    """Removes profiles from the index and returns the directories left to remove.

    Split from the removal itself deliberately. `_get_user_index` hands back the dict
    it caches on the cog, shared with every other handler, so mutating it from a worker
    thread is a live race -- but `shutil.rmtree` over a hundred profile directories is
    exactly the kind of blocking work that should not sit on the event loop. The index
    work stays here; the caller threads the filesystem half.

    Personal profiles cascade to the copies other users borrowed from them; a borrowed
    entry only unlinks the borrower's own copy.
    """
    user_id_str = str(user_id)
    index = cog.profile_manager._get_user_index(user_id)
    doomed = []

    for name in names:
        if name in index.get("borrowed", {}):
            if isinstance(index["borrowed"], dict):
                pid = index["borrowed"].pop(name)
            else:
                index["borrowed"].remove(name)
                pid = name
        elif name in index.get("personal", {}):
            if isinstance(index["personal"], dict):
                pid = index["personal"].pop(name)
            else:
                index["personal"].remove(name)
                pid = name
            cog.profile_manager._cascade_delete_borrowed_profiles(user_id, pid, name)
        else:
            continue
        doomed.append(os.path.join(cog.USERS_DIR, user_id_str, "profiles", pid))

    if doomed:
        cog.profile_manager._save_user_index(user_id, index)
    return doomed


def _remove_profile_dirs(paths: List[str]):
    """The blocking half of a bulk delete. Always called through asyncio.to_thread."""
    import shutil
    for path in paths:
        shutil.rmtree(path, ignore_errors=True)


class BulkResetView(_BulkSubView):
    """Wiping long-term memories or training examples across the selected profiles.

    Terminal rather than staged: this deletes shard files. There is nothing to merge
    into a changeset, and nothing sensible about running it in the same pass as a
    settings edit -- which is why the wizard refuses to open it while changes are
    pending.
    """

    def __init__(self, wizard):
        super().__init__(wizard)
        self.reset_choice = None
        self._build_view()

    def _targets(self) -> List[str]:
        """Training examples exist only on owned profiles, so borrowed ones drop out."""
        names = sorted(self.wizard.selected_profiles)
        if self.reset_choice != "reset_examples":
            return names
        index = self.cog.profile_manager._get_user_index(self.user_id)
        borrowed = set(index.get("borrowed", []) or [])
        return [n for n in names if n not in borrowed]

    def embed(self) -> discord.Embed:
        if not self.reset_choice:
            return discord.Embed(
                title="Reset Profile Data",
                colour=discord.Colour.orange(),
                description=("Choose what to wipe from the "
                             f"**{len(self.wizard.selected_profiles)} selected profile(s)**.\n\n"
                             "This deletes stored data outright and is not undoable. It runs on "
                             "its own rather than being staged with settings changes."))

        targets = self._targets()
        wording = ("training examples" if self.reset_choice == "reset_examples"
                   else "long-term memories")
        e = discord.Embed(
            title="Reset Profile Data",
            colour=discord.Colour.red(),
            description=f"⚠️ **All {wording} are deleted from {len(targets)} profile(s). "
                        f"This cannot be undone.**")
        e.add_field(name=f"Affected ({len(targets)})",
                    value=_cap_names(targets) or "*none*", inline=False)
        dropped = len(self.wizard.selected_profiles) - len(targets)
        if dropped:
            e.add_field(name="Withheld",
                        value=f"{dropped} borrowed profile(s) — training examples belong to "
                              f"the profile's owner.", inline=False)
        return e

    def _build_view(self):
        self.clear_items()
        options = [
            discord.SelectOption(label="Reset Training Examples", value="reset_examples",
                                 description="Personal profiles only.",
                                 default=(self.reset_choice == "reset_examples")),
            discord.SelectOption(label="Reset Long-Term Memories", value="reset_ltm",
                                 description="Applies to every selected profile.",
                                 default=(self.reset_choice == "reset_ltm")),
        ]
        select = ui.Select(placeholder="Choose what to reset...", options=options, row=0)
        select.callback = self.reset_type_callback
        self.add_item(select)

        self._add_back(1)
        confirm = ui.Button(label="Confirm & Reset Data", style=discord.ButtonStyle.danger,
                            row=1, disabled=(not self.reset_choice or not self._targets()))
        confirm.callback = self.apply_action
        self.add_item(confirm)

    async def reset_type_callback(self, interaction: discord.Interaction):
        self.reset_choice = interaction.data['values'][0]
        await self.refresh(interaction)

    async def apply_action(self, interaction: discord.Interaction):
        await interaction.response.defer()
        targets = self._targets()
        if not self.reset_choice or not targets:
            await interaction.edit_original_response(
                content="Nothing to reset.", embed=None, view=None)
            return

        if self.reset_choice == "reset_examples":
            message = await self.cog.memory_manager.bulk_reset_examples(self.user_id, targets)
        else:
            message = await self.cog.memory_manager.bulk_reset_ltm(self.user_id, targets)

        await interaction.edit_original_response(
            content=None, embed=discord.Embed(description=f"✅ {message}",
                                              colour=discord.Colour.green()), view=None)


class BulkDeleteView(_BulkSubView):
    """The confirmation for deleting the selected profiles.

    No picker of its own any more -- the wizard already knows which profiles are in
    play, and re-asking on a second screen was how a selection made under one scope
    could be silently widened under another.
    """

    def __init__(self, wizard):
        super().__init__(wizard)
        self._build_view()

    def embed(self) -> discord.Embed:
        names = sorted(self.wizard.selected_profiles)
        e = discord.Embed(
            title="Delete Profiles",
            colour=discord.Colour.red(),
            description=(f"⚠️ **{len(names)} profile(s) and all of their data are deleted "
                         f"permanently.** Personas, memories, training examples and child-bot "
                         f"links go with them, and deleting a personal profile also removes "
                         f"every copy other users borrowed from it."))
        e.add_field(name=f"Affected ({len(names)})", value=_cap_names(names) or "*none*",
                    inline=False)
        return e

    def _build_view(self):
        self.clear_items()
        self._add_back(0)
        confirm = ui.Button(label="Confirm & Delete", style=discord.ButtonStyle.danger, row=0,
                            disabled=not self.wizard.selected_profiles)
        confirm.callback = self.confirm_delete_callback
        self.add_item(confirm)

    async def confirm_delete_callback(self, interaction: discord.Interaction):
        await interaction.response.defer()
        names = sorted(self.wizard.selected_profiles)
        if not names:
            await interaction.edit_original_response(
                content="No profiles selected.", embed=None, view=None)
            return
        doomed = _unlink_profiles(self.cog, self.user_id, names)
        await asyncio.to_thread(_remove_profile_dirs, doomed)
        await interaction.edit_original_response(
            content=None,
            embed=discord.Embed(description=f"✅ Deleted {len(doomed)} profile(s).",
                                colour=discord.Colour.green()),
            view=None)


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

        new_embed = await self.cog.profile_manager._build_profile_manage_embed(
            self.original_interaction, self.profile_name, target_user_id=owner_id)
        await self.original_interaction.edit_original_response(embed=new_embed)
        await interaction.followup.send("Appearance updated.", ephemeral=True)
        # The display name and avatar are part of the classified surface, so an
        # appearance edit invalidates a rating exactly as a persona edit does.
        await maybe_prompt_rating_after_edit(self.cog, interaction, owner_id, self.profile_name)

def ProfileNeuroModal(cog, profile_name: str, current_params: Dict[str, Any], is_borrowed: bool, values_only: bool = False, callback=None, target_user_id: Optional[int] = None):
    state = current_params.get("neuro_state", {"dopamine": 50, "cortisol": 20, "oxytocin": 50, "adrenaline": 20})
    fields = [] if values_only else [
        {"label": "Engine Status (on/off)", "custom_id": "neuro_engine_enabled", "default": "on" if current_params.get("neuro_engine_enabled") else "off", "required": False, "placeholder": "Enable or disable the engine."},
    ]
    fields.extend([
        {"label": "Dopamine (0-100)", "custom_id": "dopamine", "default": str(state.get("dopamine", 50)), "required": False, "placeholder": "Motivation and joy."},
        {"label": "Cortisol (0-100)", "custom_id": "cortisol", "default": str(state.get("cortisol", 20)), "required": False, "placeholder": "Stress and anxiety."},
        {"label": "Oxytocin (0-100)", "custom_id": "oxytocin", "default": str(state.get("oxytocin", 50)), "required": False, "placeholder": "Bonding and trust."},
        {"label": "Adrenaline (0-100)", "custom_id": "adrenaline", "default": str(state.get("adrenaline", 20)), "required": False, "placeholder": "Energy and urgency."}
    ])
    def parser(v):
        c = {}
        if "neuro_engine_enabled" in v:
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

_SCOPE_LABELS = {"personal": "Personal", "borrowed": "Borrowed", "both": "Both"}

#: Steps the footer counts through. "choice" is a detour inside "actions", not a step.
_WIZARD_STEPS = ("scope", "targets", "actions", "review")


def _cap_names(names, limit: int = 900) -> str:
    """A backticked name list truncated to fit an embed field.

    An embed field caps at 1024 characters and LIMIT_PROFILES is 100, so a full
    selection overruns on the names alone. Truncating here rather than at each call
    site means the review screen cannot be the one place that forgot.
    """
    names = list(names)
    out, used = [], 0
    for index, name in enumerate(names):
        piece = f"`{name}`"
        if used + len(piece) + 2 > limit:
            out.append(f"…and {len(names) - index} more")
            break
        out.append(piece)
        used += len(piece) + 2
    return ", ".join(out)


def _describe_value(value) -> str:
    """One short, readable rendering of a staged config value."""
    if isinstance(value, bool):
        return "On" if value else "Off"
    if isinstance(value, (list, tuple)):
        return f"{len(value)} item(s)"
    if isinstance(value, dict):
        return f"{len(value)} section(s)"
    text = str(value)
    for prefix in ("GOOGLE/", "OPENROUTER/", "OLLAMA/"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text if len(text) <= 60 else text[:57] + "…"


class BulkManageView(BaseBulkProfileView):
    """The bulk dashboard: one message, four steps, one changeset.

    The old shape was action-first and one-action-at-a-time: pick a setting, pick the
    profiles, apply, and start again from the top for the next setting. Every step was
    a fresh `followup.send(ephemeral=True)`, so there was no Back button and no panel to
    go back to -- only a trail of dead messages with the live one at the bottom.

    Scope now comes first, and it is load-bearing rather than cosmetic. `_Bulk.scope`
    used to be enforced at write time, with the profile picker offering a Source toggle
    that let one selection span both sources; a personal-only row then silently dropped
    the borrowed half of it. Choosing Personal, Borrowed or Both up front turns scope
    into a session invariant that filters the *action list*, so a row that cannot apply
    to the selection is never offered in the first place.

    Targets then come before actions, which is what makes staging possible: with the
    profiles already settled, an action only has to produce a value. Those values merge
    into one `_BulkSession`, and the whole changeset is written in a single pass -- so
    "set the models, raise the temperature and turn on realistic typing" is one
    read-modify-write per profile instead of three, and one confirmation instead of
    three.

    Two rows do not stage. Delete and Reset Profile Data remove files rather than
    merging into a dict, so they run alone and the wizard refuses to open them while
    anything is pending.
    """

    def __init__(self, cog: 'MimicCog', original_interaction: discord.Interaction):
        # Set before super().__init__, which reaches _load_profile_lists below.
        self.session = _BulkSession()
        self.step = "scope"
        self.current_tab = "home"
        self.current_action = None
        self._choice = None
        self._anchor = None
        self._anchor_config: Dict[str, Any] = {}
        self._anchor_prompts: Dict[str, Any] = {}
        self._anchor_page = 0
        self._inherit_picks: Set[str] = set()
        super().__init__(cog, original_interaction.user.id, include_borrowed=True,
                         timeout=_BULK_TIMEOUT)
        self.original_interaction = original_interaction
        self._build_view()

    # --- Scope -------------------------------------------------------------

    def _load_profile_lists(self):
        """Scope-aware source lists, re-read whenever the scope changes.

        `_n_personal` / `_n_borrowed` keep the unscoped totals, because the scope step
        has to offer counts for the sources it is not currently showing.
        """
        index = self.cog.profile_manager._get_user_index(self.user_id)
        personal = sorted(list(index.get("personal", [])))
        borrowed = sorted(list(index.get("borrowed", [])))
        self._n_personal, self._n_borrowed = len(personal), len(borrowed)
        self.excluded_public = []

        scope = self.session.scope
        if scope == "personal":
            self.personal_profiles, self.borrowed_profiles = personal, []
            self.view_source, self.include_borrowed = "personal", False
        elif scope == "borrowed":
            self.personal_profiles, self.borrowed_profiles = [], borrowed
            self.view_source, self.include_borrowed = "borrowed", False
        else:
            self.personal_profiles, self.borrowed_profiles = personal, borrowed
            self.view_source, self.include_borrowed = "personal", True

        self._cached_personal_opts = [discord.SelectOption(label=n, value=n) for n in self.personal_profiles]
        self._cached_borrowed_opts = [discord.SelectOption(label=n, value=n) for n in self.borrowed_profiles]
        self.current_page = 0

    def _allowed(self, action) -> bool:
        """Whether a row can apply to every profile the current scope can reach."""
        return (action.bulk is not None
                and (action.bulk.scope == "all" or self.session.scope == "personal"))

    def _tabs(self):
        return [t for t in PROFILE_TABS
                if any(a.tab == t and self._allowed(a) for a in PROFILE_ACTIONS)]

    def _actions_for_tab(self, tab):
        return [a for a in PROFILE_ACTIONS if a.tab == tab and self._allowed(a)]

    # --- Anchor (copy one profile's setup onto the rest) --------------------

    def _load_anchor(self, name: str):
        """Reads the anchor's settings once, when it is chosen.

        Prompt blobs are carried across verbatim rather than decrypted and re-encrypted:
        the recipients are the same user's profiles under the same key, so the stored
        ciphertext is already the right ciphertext.
        """
        self._anchor = name
        index = self.cog.profile_manager._get_user_index(self.user_id)
        is_borrowed = name in (index.get("borrowed", []) or [])
        self._anchor_config = self.cog.profile_manager._get_profile_config(
            self.user_id, name, is_borrowed) or {}
        # Prompts exist only on owned profiles, and every prompt-writing row is
        # personal-scope anyway, so a borrowed anchor simply offers none.
        self._anchor_prompts = ({} if is_borrowed else
                                (self.cog.profile_manager._get_profile_prompts(self.user_id, name) or {}))
        self._inherit_picks = set()

    def _anchor_present(self, action) -> int:
        """How many of a row's declared settings the anchor actually has set.

        Only what the anchor has is copied. A setting it never touched is on the
        default, and writing a default over a recipient's deliberate override would
        make "inherit" mean "reset" for some rows and not others.
        """
        return (sum(1 for k in action.bulk.keys if k in self._anchor_config)
                + sum(1 for k in action.bulk.prompt_keys if k in self._anchor_prompts))

    def _inherit_rows(self, tab: Optional[str] = None):
        """Rows the anchor can actually contribute to, in the current scope."""
        return [a for a in PROFILE_ACTIONS
                if a.bulk is not None and a.bulk.copyable and self._allowed(a)
                and self._anchor_present(a)
                and (tab is None or a.tab == tab)]

    def _inherit_tabs(self):
        return [t for t in PROFILE_TABS if self._inherit_rows(t)]

    # --- Changeset ---------------------------------------------------------

    def _stage_change(self, action_value: str, config: Optional[Dict] = None,
                      prompts: Optional[Dict] = None, declaration: Optional[bool] = None):
        """Merges one action's output into the changeset and returns to the action list."""
        if config:
            self.session.config.update(config)
        if prompts:
            self.session.prompts.update(prompts)
        if declaration is not None:
            self.session.declaration = declaration
        action = PROFILE_ACTIONS_BY_VALUE.get(action_value)
        self.session.staged[action_value] = action.bulk_label() if action else action_value
        self._choice = None
        self.step = "actions"

    async def _stage_from_modal(self, interaction: discord.Interaction, action_value: str,
                                config: Dict, prompts: Dict):
        """Stages a modal's payload and puts the panel back where the user left it.

        ConfigModal and the persona/instruction modals defer with `thinking=True` before
        handing over, which spends the response on a placeholder message -- so the panel
        cannot be edited through that interaction at all. The placeholder is removed and
        the panel refreshed through the wizard's own token instead. Modals that have not
        responded (ActionTextInputModal) edit the message directly, one round trip fewer.
        """
        if not (config or prompts):
            await self._modal_reply(interaction, "Nothing was entered, so nothing was staged.")
            return

        self._stage_change(action_value, config=config, prompts=prompts)
        self._build_view()

        if interaction.response.is_done():
            try:
                await self.original_interaction.edit_original_response(
                    embed=self.embed(), view=self)
            except discord.HTTPException:
                # Sub-views keep this view's timer alive, so it can outlive the fifteen
                # minutes the slash command's token lasts. Say so rather than leaving the
                # change staged on a panel that will never show it again.
                self.stop()
                await interaction.followup.send(
                    "This bulk panel has expired, so that change could not be added to it. "
                    "Nothing was written — run `/profile bulk manage` again.", ephemeral=True)
                return
            try:
                await interaction.delete_original_response()
            except discord.HTTPException:
                pass
        else:
            await interaction.response.edit_message(embed=self.embed(), view=self)

    async def _modal_reply(self, interaction: discord.Interaction, text: str):
        if interaction.response.is_done():
            await interaction.followup.send(text, ephemeral=True)
        else:
            await interaction.response.send_message(text, ephemeral=True)

    def _open_choice(self, action_value: str, placeholder: str,
                     options: List[discord.SelectOption], on_pick):
        """Routes to the value step for a row whose value comes from a fixed list."""
        self._choice = {"action": action_value, "placeholder": placeholder,
                        "options": options, "on_pick": on_pick, "chosen": None}
        self.step = "choice"

    def _warnings(self) -> List[str]:
        out = []
        for value in self.session.staged:
            action = PROFILE_ACTIONS_BY_VALUE.get(value)
            if action and action.bulk and action.bulk.destructive and action.bulk.warning:
                out.append(f"**{action.bulk_label()}** — {action.bulk.warning}")
        return out

    def _resolved_field(self) -> str:
        """The changeset as resolved fields, not as a list of actions.

        Two staged rows can touch the same key -- Set Response Mode and a modal that
        also carries it -- and last write wins. Showing the actions would imply two
        writes; showing the resolved keys shows the value that will actually land.
        """
        lines = [f"• {k.replace('_', ' ').title()} → `{_describe_value(v)}`"
                 for k, v in self.session.config.items()]
        lines += [f"• {k.replace('_', ' ').title()} → *replaced*" for k in self.session.prompts]
        if self.session.declaration is not None:
            state = "declared" if self.session.declaration else "withdrawn"
            lines.append(f"• Adult 18+ Declaration → `{state}`")
        if not lines:
            return "*nothing yet*"

        kept, used = [], 0
        for index, line in enumerate(lines):
            if used + len(line) + 1 > 960:
                kept.append(f"…and {len(lines) - index} more")
                break
            kept.append(line)
            used += len(line) + 1
        return "\n".join(kept)

    # --- Rendering ---------------------------------------------------------

    def embed(self) -> discord.Embed:
        e = getattr(self, f"_embed_{self.step}")()
        e.set_footer(text=self._footer())
        return e

    def _footer(self) -> str:
        # The value pickers and the anchor detour all resolve back to the action step,
        # which is the step the user is really on while inside them.
        step = "actions" if self.step in ("choice", "anchor", "inherit") else self.step
        bits = [f"Step {_WIZARD_STEPS.index(step) + 1} of {len(_WIZARD_STEPS)}"]
        if self.step != "scope":
            bits.append(f"{_SCOPE_LABELS[self.session.scope]} · "
                        f"{len(self.selected_profiles)} selected")
        if self.session.staged:
            bits.append(f"{len(self.session.staged)} staged")
        bits.append("nothing is written until you press Apply")
        return " · ".join(bits)

    def _embed_scope(self) -> discord.Embed:
        e = discord.Embed(
            title="Bulk Manage — Choose a scope",
            colour=discord.Colour.blurple(),
            description=("Which profiles are you working on?\n\n"
                         "This fixes which actions are available for the rest of the session. "
                         "Persona, Instructions, TTS Instructions, Training Parameters, the LTM "
                         "Summarisation Prompt and the 18+ declaration all write content only a "
                         "profile's owner holds, so they are offered under **Personal** only."))
        e.add_field(name="Personal", value=f"{self._n_personal} profile(s)", inline=True)
        e.add_field(name="Borrowed", value=f"{self._n_borrowed} profile(s)", inline=True)
        return e

    def _embed_targets(self) -> discord.Embed:
        chosen = sorted(self.selected_profiles)
        e = discord.Embed(
            title="Bulk Manage — Select profiles",
            colour=discord.Colour.blurple(),
            description=(f"Choose which of your **{_SCOPE_LABELS[self.session.scope].lower()}** "
                         f"profiles receive the changes. *Select All* takes the whole source in "
                         f"one click."))
        e.add_field(name=f"Selected ({len(chosen)})",
                    value=_cap_names(chosen) or "*none yet*", inline=False)
        return e

    def _embed_actions(self) -> discord.Embed:
        e = discord.Embed(
            title="Bulk Manage — Stage changes",
            colour=discord.Colour.blurple(),
            description=("Pick an action, give it a value, and it joins the changeset below. "
                         "Stage as many as you like — they are written in a single pass over "
                         "the selected profiles."))
        if self.session.staged:
            e.add_field(name=f"Staged actions ({len(self.session.staged)})",
                        value="\n".join(f"• {label}" for label in self.session.staged.values())[:1024],
                        inline=False)
            e.add_field(name="Resolved values", value=self._resolved_field(), inline=False)
        else:
            e.add_field(name="Staged actions (0)", value="*nothing yet*", inline=False)

        if self.session.scope != "personal":
            e.add_field(
                name="Hidden for this scope",
                value=("Persona, Instructions, TTS Instructions, Training Parameters, the LTM "
                       "Summarisation Prompt and the 18+ declaration write content only a "
                       "profile's owner holds. Choose **Personal** at step one to reach them."),
                inline=False)
        return e

    def _embed_anchor(self) -> discord.Embed:
        return discord.Embed(
            title="Bulk Manage — Copy from a profile",
            colour=discord.Colour.blurple(),
            description=("Choose one of the selected profiles as the **anchor**. Its settings "
                         "are read once and staged as ordinary changes, so the rest of the "
                         "selection ends up matching it.\n\n"
                         "You choose which parts to inherit next, and you can keep staging your "
                         "own changes on top afterwards. Only settings the anchor has actually "
                         "set are copied — anything it leaves on the default is left alone on "
                         "the others."))

    def _embed_inherit(self) -> discord.Embed:
        e = discord.Embed(title=f"Bulk Manage — Inherit from `{self._anchor}`",
                          colour=discord.Colour.blurple())
        rows = self._inherit_rows()
        if not rows:
            e.description = (f"`{self._anchor}` has nothing to copy — every setting reachable "
                             f"in this scope is still on its default. Go back and choose a "
                             f"different anchor, or set the values yourself.")
            return e

        e.description = ("Tick everything the other profiles should inherit. Each entry copies "
                         "that group of settings exactly as the anchor holds them."
                         "\n-# The anchor is part of the selection too; it simply keeps its "
                         "own values.")

        picked = [PROFILE_ACTIONS_BY_VALUE[v] for v in self._inherit_picks
                  if v in PROFILE_ACTIONS_BY_VALUE]
        if picked:
            total = sum(self._anchor_present(a) for a in picked)
            body = "\n".join(f"• {a.bulk_label()}"
                             for a in sorted(picked, key=lambda a: a.bulk_label()))
            e.add_field(name=f"Selected ({len(picked)} group(s), {total} setting(s))",
                        value=body[:1024], inline=False)
            if any(a.bulk.destructive for a in picked):
                e.add_field(
                    name="⚠️ One of these overwrites authored content",
                    value=("Copying a persona, instruction set or summarisation prompt replaces "
                           "whatever the other profiles have. The review step names them again "
                           "before anything is written."), inline=False)
        return e

    def _embed_choice(self) -> discord.Embed:
        action = PROFILE_ACTIONS_BY_VALUE.get(self._choice["action"]) if self._choice else None
        lead = f"{action.bulk_description()}\n\n" if (action and action.bulk_description()) else ""
        return discord.Embed(
            title=f"Bulk Manage — {action.bulk_label() if action else 'Choose a value'}",
            colour=discord.Colour.blurple(),
            description=f"{lead}Pick a value; it is staged and you return to the action list.")

    def _embed_review(self) -> discord.Embed:
        warnings = self._warnings()
        names = sorted(self.selected_profiles)
        e = discord.Embed(
            title="Bulk Manage — Review",
            colour=discord.Colour.red() if warnings else discord.Colour.green(),
            description=(f"**{len(names)} profile(s)** will be updated with "
                         f"**{len(self.session.staged)} change(s)**. Nothing has been written "
                         f"yet; this is the last step before it is."))
        e.add_field(name=f"Profiles ({len(names)})", value=_cap_names(names) or "*none*",
                    inline=False)
        e.add_field(name="Changes", value=self._resolved_field(), inline=False)
        if warnings:
            e.add_field(name="⚠️ Overwrites authored content, and cannot be undone",
                        value="\n\n".join(warnings)[:1024], inline=False)
        return e

    def _build_view(self):
        self.clear_items()
        getattr(self, f"_build_{self.step}_step")()

    def _build_scope_step(self):
        for scope in ("personal", "borrowed", "both"):
            if scope == "both":
                # Offering "Both" with one source empty is offering the same set twice.
                if not (self._n_personal and self._n_borrowed):
                    continue
                count = self._n_personal + self._n_borrowed
            else:
                count = self._n_personal if scope == "personal" else self._n_borrowed
            btn = ui.Button(
                label=f"{_SCOPE_LABELS[scope]} ({count})", row=0, disabled=(count == 0),
                style=(discord.ButtonStyle.primary if self.session.scope == scope
                       else discord.ButtonStyle.secondary))
            btn.callback = self._pick_scope(scope)
            self.add_item(btn)
        self._add_cancel(1)

    def _build_targets_step(self):
        self._build_profile_select_ui(row=0)  # occupies rows 0 and 1
        back = ui.Button(label="◀ Scope", style=discord.ButtonStyle.secondary, row=2)
        back.callback = self._nav("scope", clear_changes=True)
        self.add_item(back)
        copy = ui.Button(label="Copy From Profile ▶", style=discord.ButtonStyle.secondary,
                         row=2, disabled=len(self.selected_profiles) < 2)
        copy.callback = self._nav("anchor")
        self.add_item(copy)
        nxt = ui.Button(label="Choose Actions ▶", style=discord.ButtonStyle.primary, row=2,
                        disabled=not self.selected_profiles)
        nxt.callback = self._nav("actions")
        self.add_item(nxt)
        self._add_cancel(2)

    def _build_anchor_step(self):
        names = sorted(self.selected_profiles)
        per_page = 25
        pages = max(1, (len(names) - 1) // per_page + 1)
        self._anchor_page = max(0, min(self._anchor_page, pages - 1))
        start = self._anchor_page * per_page

        options = [discord.SelectOption(label=n[:100], value=n, default=(n == self._anchor))
                   for n in names[start:start + per_page]]
        select = ui.Select(placeholder="Choose the profile to copy from…", options=options, row=0)
        select.callback = self._anchor_callback
        self.add_item(select)

        async def prev_cb(i: discord.Interaction):
            self._anchor_page -= 1
            await self.refresh(i)

        async def next_cb(i: discord.Interaction):
            self._anchor_page += 1
            await self.refresh(i)

        build_pagination_controls(self, self._anchor_page, pages, 1, prev_cb, next_cb)

        back = ui.Button(label="◀ Profiles", style=discord.ButtonStyle.secondary, row=2)
        back.callback = self._nav("targets")
        self.add_item(back)
        self._add_cancel(2)

    def _build_inherit_step(self):
        tabs = self._inherit_tabs()
        if self.current_tab not in tabs and tabs:
            self.current_tab = tabs[0]

        rows = self._inherit_rows(self.current_tab)
        if rows:
            options = []
            for action in rows:
                found = self._anchor_present(action)
                label = action.bulk_label()
                if action.bulk.destructive:
                    label = f"⚠️ {label}"
                options.append(discord.SelectOption(
                    label=label[:100], value=action.value,
                    description=f"{found} setting{'' if found == 1 else 's'} to copy",
                    default=(action.value in self._inherit_picks)))
            select = ui.Select(
                placeholder=f"Choose {self.current_tab.title()} settings to inherit…",
                options=options, row=0, min_values=0, max_values=len(options))
            select.callback = self._inherit_callback
            self.add_item(select)

        for tab in tabs:
            btn = ui.Button(
                label=tab.title(), row=1, disabled=(tab == self.current_tab),
                style=(discord.ButtonStyle.primary if tab == self.current_tab
                       else discord.ButtonStyle.secondary))
            btn.callback = self._pick_tab(tab)
            self.add_item(btn)

        back = ui.Button(label="◀ Anchor", style=discord.ButtonStyle.secondary, row=2)
        back.callback = self._nav("anchor")
        self.add_item(back)
        copy = ui.Button(label=f"Copy Selected ({len(self._inherit_picks)}) ▶",
                         style=discord.ButtonStyle.primary, row=2,
                         disabled=not self._inherit_picks)
        copy.callback = self._copy_callback
        self.add_item(copy)
        self._add_cancel(2)

    def _build_actions_step(self):
        tabs = self._tabs()
        if self.current_tab not in tabs and tabs:
            self.current_tab = tabs[0]

        options = []
        for action in self._actions_for_tab(self.current_tab):
            label, desc = action.bulk_label(), action.bulk_description()
            if action.value in self.session.staged:
                label = f"✓ {label}"
            elif action.bulk.destructive:
                label = f"⚠️ {label}"
            elif action.bulk.terminal:
                label = f"🗑️ {label}"
            options.append(discord.SelectOption(
                label=label[:100], value=action.value,
                description=(desc[:100] if desc else None)))
        if options:
            select = ui.Select(placeholder=f"Choose a {self.current_tab.title()} action…",
                               options=options, row=0)
            select.callback = self._action_callback
            self.add_item(select)

        for tab in tabs:
            btn = ui.Button(
                label=tab.title(), row=1, disabled=(tab == self.current_tab),
                style=(discord.ButtonStyle.primary if tab == self.current_tab
                       else discord.ButtonStyle.secondary))
            btn.callback = self._pick_tab(tab)
            self.add_item(btn)

        back = ui.Button(label="◀ Profiles", style=discord.ButtonStyle.secondary, row=2)
        back.callback = self._nav("targets")
        self.add_item(back)
        if self.session.has_changes:
            clear = ui.Button(label="Clear Staged", style=discord.ButtonStyle.secondary, row=2)
            clear.callback = self._nav("actions", clear_changes=True)
            self.add_item(clear)
        review = ui.Button(label=f"Review & Apply ({len(self.session.staged)})",
                           style=discord.ButtonStyle.success, row=2,
                           disabled=not self.session.has_changes)
        review.callback = self._nav("review")
        self.add_item(review)
        self._add_cancel(2)

    def _build_choice_step(self):
        for option in self._choice["options"]:
            option.default = (option.value == self._choice["chosen"])
        select = ui.Select(placeholder=self._choice["placeholder"],
                           options=self._choice["options"], row=0)
        select.callback = self._choice_callback
        self.add_item(select)
        back = ui.Button(label="◀ Back", style=discord.ButtonStyle.secondary, row=1)
        back.callback = self._nav("actions")
        self.add_item(back)
        self._add_cancel(1)

    def _build_review_step(self):
        destructive = bool(self._warnings())
        apply_btn = ui.Button(
            label="Overwrite & Apply" if destructive else "Apply",
            style=discord.ButtonStyle.danger if destructive else discord.ButtonStyle.success,
            row=0, disabled=not (self.session.has_changes and self.selected_profiles))
        apply_btn.callback = self._apply_callback
        self.add_item(apply_btn)
        back = ui.Button(label="◀ Back", style=discord.ButtonStyle.secondary, row=0)
        back.callback = self._nav("actions")
        self.add_item(back)
        self._add_cancel(0)

    def _add_cancel(self, row: int):
        btn = ui.Button(label="Cancel", style=discord.ButtonStyle.secondary, row=row)

        async def callback(interaction: discord.Interaction):
            self.stop()
            await interaction.response.edit_message(
                embed=discord.Embed(description="Cancelled. Nothing was changed.",
                                    colour=discord.Colour.greyple()), view=None)

        btn.callback = callback
        self.add_item(btn)

    # --- Navigation --------------------------------------------------------

    async def _edit(self, interaction: discord.Interaction):
        """The hook BaseBulkProfileView's own callbacks re-render through."""
        await interaction.response.edit_message(embed=self.embed(), view=self)

    async def refresh(self, interaction: discord.Interaction):
        self._build_view()
        await interaction.response.edit_message(embed=self.embed(), view=self)

    async def start(self, interaction: discord.Interaction):
        await interaction.response.send_message(embed=self.embed(), view=self, ephemeral=True)

    def _pick_scope(self, scope: str):
        async def callback(interaction: discord.Interaction):
            self.session.scope = scope
            # A new scope is a new set of reachable profiles and a new set of applicable
            # actions, so neither the selection nor the changeset survives it.
            self.selected_profiles.clear()
            self.session.clear_changes()
            self._load_profile_lists()
            self.step = "targets"
            await self.refresh(interaction)
        return callback

    def _pick_tab(self, tab: str):
        async def callback(interaction: discord.Interaction):
            self.current_tab = tab
            await self.refresh(interaction)
        return callback

    def _nav(self, step: str, *, clear_changes: bool = False):
        async def callback(interaction: discord.Interaction):
            if clear_changes:
                self.session.clear_changes()
            if step == "review":
                self.session.targets = set(self.selected_profiles)
            if step == "anchor":
                self._anchor_page = 0
            self._choice = None
            self.step = step
            await self.refresh(interaction)
        return callback

    async def _action_callback(self, interaction: discord.Interaction):
        action = PROFILE_ACTIONS_BY_VALUE.get(interaction.data['values'][0])
        if action is None or not self._allowed(action):
            await self.refresh(interaction)
            return
        if action.bulk.terminal and self.session.has_changes:
            await interaction.response.send_message(
                f"**{action.bulk_label()}** removes data rather than changing a setting, so it "
                f"runs on its own. Apply or clear your {len(self.session.staged)} staged "
                f"change(s) first.", ephemeral=True)
            return
        self.current_action = action
        await action.bulk.run(self, interaction)

    async def _anchor_callback(self, interaction: discord.Interaction):
        self._load_anchor(interaction.data['values'][0])
        self.step = "inherit"
        await self.refresh(interaction)

    async def _inherit_callback(self, interaction: discord.Interaction):
        # Picks are kept across tabs, so a dropdown only ever owns its own tab's rows.
        self._inherit_picks -= {a.value for a in self._inherit_rows(self.current_tab)}
        self._inherit_picks.update(interaction.data.get('values', []))
        await self.refresh(interaction)

    async def _copy_callback(self, interaction: discord.Interaction):
        for value in sorted(self._inherit_picks):
            action = PROFILE_ACTIONS_BY_VALUE.get(value)
            if action is None or action.bulk is None or not action.bulk.copyable:
                continue
            config = {k: self._anchor_config[k] for k in action.bulk.keys
                      if k in self._anchor_config}
            prompts = {k: self._anchor_prompts[k] for k in action.bulk.prompt_keys
                       if k in self._anchor_prompts}
            if config or prompts:
                # Staged exactly as if the row had been filled in by hand, so the review
                # screen, the warnings and the apply pass need to know nothing about
                # where the values came from.
                self._stage_change(value, config=config, prompts=prompts)
        self._inherit_picks.clear()
        self.step = "actions"
        await self.refresh(interaction)

    async def _choice_callback(self, interaction: discord.Interaction):
        value = interaction.data['values'][0]
        self._choice["chosen"] = value
        # on_pick stages the value, which also returns the wizard to the action list.
        self._choice["on_pick"](self, value)
        await self.refresh(interaction)

    async def _apply_callback(self, interaction: discord.Interaction):
        await interaction.response.defer()
        self.session.targets = set(self.selected_profiles)
        counts = await _apply_bulk_session(self.cog, self.user_id, self.session)
        self.stop()

        changed = counts["changed"]
        e = discord.Embed(
            title="Bulk Manage — Applied" if changed else "Bulk Manage — Nothing Changed",
            colour=discord.Colour.green() if changed else discord.Colour.greyple(),
            description=(f"Updated **{changed}** of {len(self.session.targets)} selected "
                         f"profile(s) with {len(self.session.staged)} change(s)."))
        notes = []
        if counts["skipped_prompts"]:
            notes.append(f"• {counts['skipped_prompts']} borrowed profile(s) kept their persona, "
                         f"instructions and prompts — those belong to the profile's owner.")
        if counts["skipped_declaration"]:
            notes.append(f"• {counts['skipped_declaration']} profile(s) kept their content "
                         f"rating — a published profile, a classifier verdict and an exemption "
                         f"are none of them the owner's to overwrite in bulk.")
        if notes:
            e.add_field(name="Skipped", value="\n".join(notes)[:1024], inline=False)

        await interaction.edit_original_response(embed=e, view=None)

    async def on_timeout(self):
        try:
            await self.original_interaction.edit_original_response(
                embed=discord.Embed(
                    description=("This bulk session timed out and nothing was written. Run "
                                 "`/profile bulk manage` again to start over."),
                    colour=discord.Colour.greyple()),
                view=None)
        except Exception:
            pass

    # --- Bespoke rows ------------------------------------------------------

    async def _bulk_error_response(self, interaction: discord.Interaction):
        async def modal_callback(i: discord.Interaction, new_val: str):
            await self._stage_from_modal(
                i, "error_response",
                {"error_response": new_val.strip() or "An error has occurred."}, {})

        await interaction.response.send_modal(ActionTextInputModal(
            title="Set Custom Error Message",
            label="Error Message",
            placeholder="Shown to users when generation fails...",
            default="An error has occurred.",
            required=False,
            on_submit_callback=modal_callback))

    async def _bulk_edit_persona(self, interaction: discord.Interaction):
        async def modal_callback(i: discord.Interaction, persona_data: Dict[str, List[str]]):
            # An entirely blank form stages "clear every persona section", which is a
            # legitimate thing to want and an unlikely thing to mean. Refuse it here
            # rather than carry it to a confirmation the user reads as an overwrite.
            if not any(any(line.strip() for line in lines) for lines in persona_data.values()):
                await self._modal_reply(
                    i, "Every persona section was left blank, so nothing was staged.")
                return
            await self._stage_from_modal(i, "edit_persona", {}, {"persona": persona_data})

        await interaction.response.send_modal(EditUserProfilePersonaModal(
            self.cog, "BULK_APPLY", {}, self.user_id, callback=modal_callback))

    async def _bulk_edit_instructions(self, interaction: discord.Interaction):
        async def modal_callback(i: discord.Interaction, instr_list: List[str]):
            if not any(part.strip() for part in instr_list):
                await self._modal_reply(
                    i, "Every instruction part was left blank, so nothing was staged.")
                return
            await self._stage_from_modal(i, "edit_instructions", {}, {"ai_instructions": instr_list})

        await interaction.response.send_modal(EditUserProfileAIInstructionsModal(
            self.cog, "BULK_APPLY", "", self.user_id, callback=modal_callback))

    async def _bulk_adult_declaration(self, interaction: discord.Interaction):
        options = [
            discord.SelectOption(label="Declare Adult 18+", value="declare",
                                 description="Confine these profiles to age-restricted channels."),
            discord.SelectOption(label="Withdraw Declaration", value="withdraw",
                                 description="Hand the profile back to the classifier."),
        ]
        # Profiles in the Public Library are withheld at apply time and reported, rather
        # than filtered out of the picker: the selection was made before this row was
        # chosen, so quietly shrinking it here would be the change going missing.
        self._open_choice(
            "content_safety", "Declare or withdraw…", options,
            lambda w, v: w._stage_change("content_safety", declaration=(v == "declare")))
        await self.refresh(interaction)


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
            in_flight = (int(self.user_id), self.profile_name) in self.cog.pending_classifications
            if in_flight:
                # Pending meant two different things on this page -- "a job is
                # working on it" and "a job failed and it is waiting to retry" --
                # and the embed rendered them identically, so a user watching a
                # live classification could not tell it from a stalled one.
                embed.add_field(
                    name="Analysing",
                    value="The classifier is running now. This page will update "
                          "itself when the verdict lands.",
                    inline=False)
            elif retry_after and time.time() < retry_after:
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

    async def _repaint(self, i: discord.Interaction):
        """Redraws from state already in hand.

        refresh_state hashes the persona and scans every user's borrow index, which
        is far too much to spend on the interim "it is running now" frame -- nothing
        it reads can have changed since the click that got us here.
        """
        self._build_view()
        await i.edit_original_response(embed=self.get_embed(), view=self)

    async def _await_verdict(self, i: discord.Interaction):
        """Shows that the job started, waits for it, then shows the verdict.

        Replaces a fixed `asyncio.sleep(2.5)`. The sleep was shorter than a real
        classifier call, so the repaint after it usually rendered the same Pending
        the user was already looking at, and nothing updated the page afterwards --
        which is why reopening the dashboard was the only way to see a verdict.
        """
        await self._repaint(i)
        await self.cog.profile_manager.await_classification(self.user_id, self.profile_name)
        await self._refresh(i)

    # --- actions --------------------------------------------------------------

    async def back_cb(self, i: discord.Interaction):
        view = ProfileManageView(self.cog, self.original_interaction, self.profile_name,
                                 self.is_borrowed, self.mod_return_user_id, self.is_mod_view)
        await i.response.defer()
        await view._refresh_dashboard(i)

    async def submit_cb(self, i: discord.Interaction):
        await i.response.defer()
        ok, msg = await self.cog.profile_manager.submit_for_rating(
            self.user_id, self.profile_name)
        await i.followup.send(msg, ephemeral=True)
        if ok:
            await self._await_verdict(i)
        else:
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
        if key in self.cog.pending_classifications:
            # Wait whether this call queued the job or found one already running --
            # the button's whole purpose is to answer "is it done yet", and it used
            # to return the same Pending unless it had queued the job itself.
            await self._await_verdict(i)
        else:
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
        await self._await_verdict(i)


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
        pm = self.cog.profile_manager
        pm.schedule_content_classification(self.user_id, self.profile_name)
        await i.edit_original_response(
            content=f"Re-checking the rating for '{self.profile_name}'…", view=None)
        self.stop()
        # This prompt used to end here, so the one place the user was told a
        # re-check had started was also the last thing they heard about it. Waiting
        # costs nothing -- the view is already stopped and the message is already
        # theirs -- and it turns a dead end into the answer.
        await pm.await_classification(self.user_id, self.profile_name)
        verdict, _rating = pm._content_rating_state(self.user_id, self.profile_name)
        await i.edit_original_response(
            content=(f"'{self.profile_name}' is now **{CONTENT_RATING_LABELS[verdict]}**."),
            view=None)

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
