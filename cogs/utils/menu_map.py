"""The dashboard tree shown to profiles running in Help Mode.

This map used to be a hand-written string literal in `constants.py`, and it drifted:
by the time anyone noticed, it was still telling users to click "Edit Personal Key"
in a `/settings` tab that had been replaced by the four-slot key manager, and it
listed a `Set Response Mode` action under a tab that no longer owned it. A map that
lies is worse than no map, because the model states the wrong path confidently.

The `/profile manage` half is therefore *generated* from `PROFILE_ACTIONS` -- the same
declarative table that builds the real dropdown -- so a new action appears here the
moment it appears in the UI, and a removed one disappears. Nothing to remember.

The other four dashboards are not registry-driven; their tabs and buttons are built
imperatively across `gui_settings`, `gui_sessions`, `gui_hub` and `gui_mod`. Scraping
those reliably is not worth the fragility, so they stay declared here, adjacent to the
generated half and easy to diff against the views.
"""

from typing import List, Optional

# Rendered map, built once on first use. The action table is fixed at import, so this
# never changes for the life of the process -- and Help Mode is a per-turn path, which
# is not the place to re-render a 60-line tree.
_CACHED_MAP: Optional[str] = None

# Wording for the `/profile manage` tab buttons, keyed by the tab ids in PROFILE_TABS.
_TAB_TITLES = {
    "home": "Home (profile-level operations)",
    "persona": "Persona (identity and voice)",
    "params": "Params (model and sampling)",
    "tools": "Tools (external systems and per-turn behaviour)",
    "memory": "Memory (long-term memory and training examples)",
}

# Dashboards whose views are built imperatively. Kept as (heading, [(tab, [actions])]).
_STATIC_DASHBOARDS = [
    (
        "`/settings` (DM only)",
        [
            ("Home", ["Integration summary: key slots, child bots, server assignments"]),
            ("API Keys", [
                "Four key slots: Google Gemini 1-2, OpenRouter 1-2",
                "Submit Key / Edit Key / Delete Key on the selected slot",
                "Assign this key to... (Personal, and any server you administrate)",
                "Save Assignments (an assignment is not stored until this is clicked)",
            ]),
            ("Child Bots", [
                "Create New Child Bot (bot owner only; links a PID to a bot token)",
                "Unlink & Delete (disconnects the client and deletes its config)",
                "Set presence: online status and activity",
            ]),
        ],
    ),
    (
        "`/session config` (server administrators)",
        [
            ("Cast", ["Add or remove personal, borrowed and child-bot participants (max 200)"]),
            ("Config", [
                "Toggle Execution (sequential or random turn order)",
                "Edit Master Prompt (the scene prompt every participant sees)",
                "Toggle TTS (per-round audio, including stitched multi-audio)",
                "Set Response Limit (maximum replies per round)",
            ]),
            ("Reactivity", ["Edit Chance & Wakewords (probability rolls and exact-match interjection)"]),
            ("Proactivity", [
                "Toggle Proactivity (the autonomous round timer)",
                "Edit Settings & AI Director (chance, cooldown, director model)",
            ]),
        ],
    ),
    (
        "`/profile hub`",
        [
            ("Home", ["Library statistics and your own publication status"]),
            ("Public Library", ["Browse, search and borrow published profiles"]),
            ("Incoming Shares", ["Accept or reject profiles shared directly with you"]),
            ("Manage My Shares", ["Publish or unpublish, and revoke existing shares"]),
            ("Profile Cloning", ["Generate a clone code, producing an independent copy rather than a link"]),
        ],
    ),
    (
        "`/mod` (bot owner only)",
        [
            ("Stats", ["Instance-wide usage and model statistics"]),
            ("Profiles", ["Inspect any user's profiles; clear classifier verdicts",
                          "Reset All Content Ratings (instance-wide, one-off baseline reset)"]),
            ("Prompts", ["Override the global system prompts, including CONTENT_POLICY and HELP_MODE_INJECTION"]),
            ("Docs", ["Edit the documentation shards backing Help Mode; re-embeds on save"]),
            ("Blacklist", ["Block users from the instance"]),
        ],
    ),
]


def _render_profile_dashboard() -> List[str]:
    """Renders the `/profile manage` tree from the live action table."""
    # Imported here, not at module scope: gui_profiles imports from utils, so a
    # top-level import would close a cycle.
    from ..gui.gui_profiles import PROFILE_ACTIONS, PROFILE_TABS

    lines = ["DASHBOARD: `/profile manage [profile_name]`"]
    tabs = [t for t in PROFILE_TABS if any(a.tab == t for a in PROFILE_ACTIONS)]

    for tab_index, tab in enumerate(tabs):
        is_last_tab = tab_index == len(tabs) - 1
        tab_prefix = "└──" if is_last_tab else "├──"
        continuation = "    " if is_last_tab else "│   "
        lines.append(f"  {tab_prefix} Tab: [{_TAB_TITLES.get(tab, tab.title())}]")

        actions = [a for a in PROFILE_ACTIONS if a.tab == tab]
        for action_index, action in enumerate(actions):
            is_last_action = action_index == len(actions) - 1
            action_prefix = "└──" if is_last_action else "├──"
            lines.append(f"  {continuation}{action_prefix} {action.menu_label}: {action.description}")

        if not is_last_tab:
            lines.append("  │")

    return lines


def _render_static_dashboards() -> List[str]:
    lines: List[str] = []
    for heading, tabs in _STATIC_DASHBOARDS:
        lines.append("")
        lines.append(f"DASHBOARD: {heading}")
        for tab_index, (tab_name, actions) in enumerate(tabs):
            is_last_tab = tab_index == len(tabs) - 1
            tab_prefix = "└──" if is_last_tab else "├──"
            continuation = "    " if is_last_tab else "│   "
            lines.append(f"  {tab_prefix} Tab: [{tab_name}]")
            for action_index, action in enumerate(actions):
                is_last_action = action_index == len(actions) - 1
                action_prefix = "└──" if is_last_action else "├──"
                lines.append(f"  {continuation}{action_prefix} {action}")
    return lines


def build_menu_map() -> str:
    """Returns the full dashboard map, building it on first call.

    Never raises: Help Mode degrades to documentation-only rather than losing the turn
    if the action table cannot be read for any reason.
    """
    global _CACHED_MAP
    if _CACHED_MAP is not None:
        return _CACHED_MAP

    try:
        lines = _render_profile_dashboard() + _render_static_dashboards()
        _CACHED_MAP = "\n".join(lines)
    except Exception as e:
        print(f"Failed to build the documentation menu map: {type(e).__name__}({e})")
        _CACHED_MAP = "(The dashboard map is unavailable in this instance.)"

    return _CACHED_MAP
