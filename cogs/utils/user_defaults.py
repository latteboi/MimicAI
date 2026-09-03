"""Per-user default settings, applied to profiles as they are created or borrowed.

A new profile is built from one hardcoded template and a borrow from a snapshot of
whoever wrote it, so a user who prefers a different model has always had to set it
again on every profile they own. Worse, the borrow snapshot carries the *author's*
model: borrowing an OpenRouter-configured profile with only a Google key produced a
profile that could not generate at all, and said so in a message about API keys the
user had just finished setting up.

These defaults are held per user, sparsely -- only what was actually chosen -- in the
plaintext `index.json` beside the profile lists, which `_get_user_index` already keeps
in memory. Sparse matters: "unset" and "set to the value the bot happens to ship
today" have to stay distinguishable, or bumping PRIMARY_MODEL_NAME would silently
strand every user who ever opened this screen at the old value.

WHAT MAY BE DEFAULTED IS NOT DECLARED HERE. It is derived from `PROFILE_ACTIONS` --
the same table that builds the dashboard, the bulk manager and the Help Mode menu map
-- because a fourth hand-written list of setting keys is a fourth thing to forget.
Two properties of that table do the work:

* A row's `keys` are the config keys it writes; its `prompt_keys` are the prompt keys.
  Only `keys` are eligible. Prompts are not a matter of taste here: a borrow stores
  `"prompts": {}` and generation reads the *source's* prompts through
  `_resolve_effective_profile`, so a prompt written onto a borrow is not an override
  that loses to the author -- it is a write nothing ever reads. A setting that can
  only fail silently must not be offered.

* A row's `scope` already states whether it may touch a borrow at all. `scope="all"`
  means both; `scope="personal"` means the row writes something a borrower does not
  own, so those keys default onto new profiles only. Nothing new had to be decided
  for the Director's Desk or the training parameters -- they were already declared.

A row that declares neither is not offered, which is how the 18+ declaration excludes
itself: `_Bulk.copyable` is False for it deliberately, and a content rating is no more
a default than it is a thing to propagate in bulk.
"""

from typing import Any, Dict, List, Optional

#: Config keys that must never be defaulted regardless of what the table says.
#:
#: The resolver reads all but the last of these. `original_*` and `pointer` are how
#: `_resolve_effective_profile` and `_resolve_borrowed_pointer` find the profile a
#: borrow points at -- overwrite one and the borrow generates with no persona, or
#: orphans. `profile_id` backs the PID reconciliation in `_get_profile_config`.
#: `neuro_state` is the one that is not identity: it is the *live* hormone vector,
#: rewritten as a character reacts, so defaulting it would reset a mood mid-session.
#: Only `neuro_state` is reachable from the table today (the neuro row declares it);
#: the rest are here so that adding one to a `_Bulk` later cannot quietly expose it.
DEFAULT_DENY = frozenset({
    "original_owner_id", "original_pid", "original_profile_name", "original_profile_id",
    "pointer", "borrowed_at", "profile_id", "created_at", "neuro_state",
})

#: Keys forced to new-profile-only even though their row is `scope="all"`.
#:
#: The table's scope answers "may a borrower write this?", and for all of these the
#: answer is yes -- they are local config and the borrower's dashboard can set them
#: one profile at a time. This list answers a narrower question: "should *my standing
#: preference* overwrite what the author chose, on a profile I did not write?" For
#: anything that is part of how a character reads or performs, no. A persona tuned at
#: 0.4 does not survive being reset to someone's global 1.1, and it fails as bad
#: writing rather than as an error anyone can trace back to this screen.
DEFAULT_NEW_ONLY = frozenset({
    # Sampling. Authored tuning, and the most damaging to override blind.
    "temperature", "top_p", "top_k",
    "frequency_penalty", "presence_penalty", "repetition_penalty", "min_p", "top_a",
    # Voice and presentation belong to the character, not to the person borrowing it.
    "speech_voice", "placeholder_emoji",
    # Written in the character's voice, and inherited from the author by the snapshot.
    "error_response",
})

# Built once from the action table. PROFILE_ACTIONS is fixed at import, so this cannot
# change for the life of the process, and it is consulted on every profile creation.
_CACHED_KEYS: Optional[Dict[str, str]] = None


def defaultable_keys() -> Dict[str, str]:
    """Maps each defaultable config key to the scope it applies in: "both" or "new".

    Never raises. A failure to read the action table must not take profile creation
    with it -- the caller then behaves exactly as it did before defaults existed.
    """
    global _CACHED_KEYS
    if _CACHED_KEYS is not None:
        return _CACHED_KEYS

    table: Dict[str, str] = {}
    try:
        # Imported here, not at module scope: gui_profiles imports from utils, so a
        # top-level import would close a cycle. Same reason menu_map defers its own.
        from ..gui.gui_profiles import PROFILE_ACTIONS

        for action in PROFILE_ACTIONS:
            bulk = action.bulk
            if bulk is None:
                continue
            row_scope = "both" if bulk.scope == "all" else "new"
            for key in bulk.keys:
                if key in DEFAULT_DENY:
                    continue
                # A key declared by two rows keeps the narrower scope.
                if table.get(key) == "new":
                    continue
                table[key] = "new" if key in DEFAULT_NEW_ONLY else row_scope
    except Exception as e:
        print(f"Failed to derive the user-default key table: {type(e).__name__}({e})")
        table = {}

    _CACHED_KEYS = table
    return table


def sanitise_defaults(defaults: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Drops anything that is not currently defaultable.

    Stored defaults outlive the table that produced them: a setting can be removed, or
    have its row's scope tightened, long after someone chose a value for it. Filtering
    on read rather than migrating on upgrade means a stale key is inert instead of
    being written onto new profiles by a rule that no longer exists, and it comes back
    if the key does.
    """
    if not defaults:
        return {}
    allowed = defaultable_keys()
    return {k: v for k, v in defaults.items() if k in allowed}


def apply_defaults(config: Dict[str, Any], defaults: Optional[Dict[str, Any]], *,
                   borrowed: bool) -> List[str]:
    """Writes the user's defaults into `config` in place. Returns the keys applied.

    `borrowed` selects the scope: a borrow takes only the keys whose row permits it,
    a new profile takes everything. The return value exists so the borrow path can
    tell the user what it changed -- a profile that arrives already differing from the
    one they were shown in the library needs to say so.
    """
    cleaned = sanitise_defaults(defaults)
    if not cleaned:
        return []

    allowed = defaultable_keys()
    applied = []
    for key, value in cleaned.items():
        if borrowed and allowed.get(key) != "both":
            continue
        config[key] = value
        applied.append(key)
    return sorted(applied)


#: Prefix -> the provider name `_get_api_key_for_user` and the key slots use.
_PROVIDER_PREFIXES = (("OPENROUTER/", "openrouter"), ("OLLAMA/", "ollama"),
                      ("GOOGLE/", "gemini"))


def model_provider(value: Optional[str]) -> str:
    """Which provider a stored model id routes to.

    Unprefixed ids are Google: that is how `_instantiate_model` reads them, and the
    legacy bare `gemini-2.5-flash-lite` values still on older profiles depend on it.
    """
    text = str(value or "").upper()
    for prefix, provider in _PROVIDER_PREFIXES:
        if text.startswith(prefix):
            return provider
    return "gemini"


def platform_model_defaults() -> Dict[str, str]:
    """Every model slot mapped to the value a fresh profile ships with.

    Read off `ModelPickerMixin._CATEGORY_KEYS`, the table the pickers already build
    their dropdowns from, so a seventh category is covered the day it is added.
    """
    out: Dict[str, str] = {}
    try:
        from ..gui.gui_profiles import ModelPickerMixin
        for slots in ModelPickerMixin._CATEGORY_KEYS.values():
            for key, _label, default in slots:
                out[key] = default
    except Exception as e:
        print(f"Failed to read the platform model defaults: {type(e).__name__}({e})")
    return out


def model_slot_labels() -> Dict[str, str]:
    """Model slot keys mapped to "Category · Slot" wording for user-facing messages.

    Built from the same two tables the picker labels itself with, so a slot never has
    one name in the dashboard and another in the message explaining it was changed.
    """
    out: Dict[str, str] = {}
    try:
        from ..gui.gui_profiles import ModelPickerMixin
        titles = {cat: title for cat, title, _desc in ModelPickerMixin._CATEGORY_LABELS}
        for category, slots in ModelPickerMixin._CATEGORY_KEYS.items():
            for key, label, _default in slots:
                title = titles.get(category, category.title())
                # The primary slot of most categories is named after the category
                # itself ("Anti-Repetition Critic"), and "X · X" reads as a bug.
                out[key] = label if label == title else f"{title} · {label}"
    except Exception as e:
        print(f"Failed to read the model slot labels: {type(e).__name__}({e})")
    return out


#: Readable wording for the non-model keys that show up in user-facing messages.
#: Cosmetic only: `setting_label` title-cases anything missing here, so a key added
#: without an entry reads slightly awkwardly rather than breaking anything. The model
#: slots are not listed -- they come from `model_slot_labels`, which is generated.
SETTING_LABELS = {
    "stm_length": "Short-Term Memory",
    "ltm_creation_enabled": "LTM Auto-Creation",
    "ltm_creation_interval": "LTM Creation Interval",
    "ltm_context_size": "LTM Recall Depth",
    "ltm_relevance_threshold": "LTM Relevance Threshold",
    "ltm_summarization_context": "LTM Summarisation Context",
    "thinking_level": "Reasoning Effort",
    "thinking_budget": "Thinking Budget",
    "thinking_summary_visible": "Thinking Summary",
    "realistic_typing_enabled": "Realistic Typing",
    "typing_cursor": "Typing Cursor",
    "typing_mode": "Typing Chunking",
    "typing_cps": "Typing Speed",
    "typing_max_delay": "Typing Maximum Delay",
    "time_tracking_enabled": "Time Awareness",
    "timezone": "Timezone",
    "response_mode": "Response Mode",
    "grounding_mode": "Grounding",
    "url_mode": "URL Context",
    "url_fetching_enabled": "URL Fetching",
    "help_mode_enabled": "Help Mode",
    "critic_mode": "Critic Mode",
    "critic_enabled": "Anti-Repetition Critic",
    "critic_scope": "Critic Scope",
    "critic_strictness": "Critic Strictness",
    "critic_lookback": "Critic Lookback",
    "critic_persistence": "Critic Persistence",
    "neuro_engine_enabled": "Neuro Engine",
    "image_generation_enabled": "Image Generation",
    "speech_tts_enabled": "Text-to-Speech",
    "speech_temperature": "Speech Temperature",
    "show_fallback_indicator": "Fallback Indicator",
    "child_bot_placeholder": "Child Bot Placeholder",
    "ollama_host_url": "Ollama Host URL",
}


def setting_label(key: str) -> str:
    """How one config key is named to a user. Falls back to a title-cased key."""
    labels = model_slot_labels()
    if key in labels:
        return labels[key]
    if key in SETTING_LABELS:
        return SETTING_LABELS[key]
    return key.replace("_", " ").title()
