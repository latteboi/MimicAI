"""`/start` -- the guided setup wizard.

Setup crosses more than one context: a channel's cast is configured by a server
administrator (`/session config` is guild_only, and admin unless the channel is on Open
casting), and only in a server. So this is context-aware rather than linear. Steps that
cannot run where you are stay **visible and greyed**, because a member who cannot see
the seat step has no way to learn why the bot is silent.

**No progress is stored.** Every step's completion is probed from state that already
exists -- the provider preference, assigned keys, the profile index, the channel's
session -- which means the wizard cannot desynchronise from reality, "run it again to
pick up where you left off" is literally true, and there is no new per-user dict to keep
bounded. An open wizard re-probes on a timer (`watch`) rather than being told: a step
finishes in a draft view, the library, the cast editor or the channel itself, and a
poll is one place where hooks in each would be five.

It is a router, not a second implementation. The key step opens `/settings`'
`SubmitAPIKeyModal` and its server picker is `/settings`' own (`KeyScopeMixin`),
applying each pick as it is made instead of staging it for a Save. The other steps
launch the real dashboard -- `HubPublicLibraryView`, `/profile generate`,
`SessionConfigView` -- as a *separate* ephemeral message, by deferring with
`thinking=True`. On a component interaction that sends a new message rather than
updating this one, so those views' `edit_original_response` lands on the new message
and the wizard survives underneath to be refreshed. Home, once nothing is left to set up
where it was run, is more of the same launchers: no screen of its own.

Prose lives in `content.WIZARD_COPY`, and depth is not repeated: each step names a
`HELP_CATEGORIES` page, and the Guide button opens the guide browser on it.
"""

import asyncio
import discord
from discord import ui
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..utils.constants import (BOT_PERMISSIONS, CAST_POLICY_OPEN, CHILD_BOT_PERMISSIONS,
                               DEFAULT_CAST_POLICY, MODEL_PROVIDERS, PROVIDER_CHOICES,
                               PROVIDER_SUBSCRIPTION, SUBSCRIPTION_SLOT, defaultConfig)
from ..utils import openrouter_signin
from ..utils.user_defaults import offered_providers
from ..managers.subscriptions import ACTIVE, tier_name
from ..utils.data_policy import is_paid_gemini_slot, training_opt_in
from ..utils.content import (HELP_CATEGORIES, LEGAL_EFFECTIVE_DATE, PERMISSION_EFFECTS,
                             SERVER_KEY_UNASSIGNED, WIZARD_COPY, WIZARD_HOME, WIZARD_TOUR,
                             server_key_pitch)
from .base_components import (PAGE_NAV_VALUES, BlockedGuard, DropdownContentView, PageJumpModal,
                              TimeoutCleanupMixin, add_button, add_select, add_topgg_links,
                              paged_nav_options, refuse_unregistered)
from .gui_generate import unsaved_draft
from .gui_settings import (SLOT_PROVIDER, KeyScopeMixin, OverrideConfirmView, SettingsAPIView,
                           SettingsHomeView, SubmitAPIKeyModal, SubscribeView, _fit_list)
from .base_components import text_embed

if TYPE_CHECKING:
    from ..MimicCog import MimicCog


#: Where each provider hands out keys, behind the key step's link button.
KEY_PAGES = {"openrouter": "https://openrouter.ai/settings/keys",
             "gemini": "https://aistudio.google.com/app/apikey"}

#: How often an open wizard re-probes (`StartWizardView.watch`). A probe is cached reads
#: and one thread hop; a Discord edit is sent only when what it shows changed.
WATCH_SECONDS = 5
#: `step_key` for Home, picked by its button: no step is focused.
HOME = "home"
#: Characters per page of Home's dropdown: 25 options, less the three page controls.
_HOME_PAGE = 22


class _Step:
    """One row of the setup checklist.

    `probe` reads the state dict assembled once per repaint rather than touching disk
    itself, so adding a step costs no extra reads unless it needs something new.

    `context` is "guild" or "any". `requires` names keys in the same state dict that
    must be truthy, so a gate that is not a fixed property of the user -- seating is
    open to administrators *or* to anyone in an Open casting channel -- is probed at
    repaint like every other fact here rather than baked into the table. Both describe
    where the step's *action* can run; a step that cannot run here is still drawn, with
    the reason of the first gate that is shut.

    `actions` names methods on the view. Kept as names so this table stays readable as
    data, and so a step with no action -- the last one, which is just "talk" -- simply
    declares none. A `repeatable` step keeps its actions once done, when picked from
    the dropdown: another character, another cast, the other provider.
    """

    __slots__ = ("key", "title", "help_ref", "context", "requires", "probe",
                 "actions", "done_detail", "repeatable")

    def __init__(self, key, title, help_ref, probe, *, context="any",
                 requires=None, actions=(), done_detail=None, repeatable=False):
        self.key = key
        self.title = title
        self.help_ref = help_ref
        self.probe = probe
        self.context = context
        self.requires = (requires,) if isinstance(requires, str) else tuple(requires or ())
        self.actions = actions
        self.done_detail = done_detail
        self.repeatable = repeatable

    def blurb(self, state: Dict[str, Any]) -> str:
        # Where the subscription is sold, a step may say so (`provider_sold`).
        copy = (state.get("sells") and WIZARD_COPY.get(f"{self.key}_sold")) or WIZARD_COPY.get(self.key, "")
        # The key step speaks for the provider chosen before it, and only that one --
        # and, where this instance offers it, of signing in rather than making a key.
        if isinstance(copy, dict):
            provider = state.get("provider") or "openrouter"
            if provider == PROVIDER_SUBSCRIPTION and not (state.get("sells") or state.get("subscribed")):
                provider = "openrouter"  # not sold here, or paused: it ran on OpenRouter
            if openrouter_signin.enabled() and f"{provider}_signin" in copy:
                provider += "_signin"
            return copy.get(provider, "")
        return copy

    def in_context(self, state: Dict[str, Any]) -> bool:
        return self.context != "guild" or state["in_guild"]

    def available(self, state: Dict[str, Any]) -> bool:
        """Whether this step's action can be taken from where the command was run."""
        return self.in_context(state) and all(state.get(gate) for gate in self.requires)

    def blocker(self, state: Dict[str, Any]) -> str:
        """Why this step is not actionable here. Only read when `available` is False."""
        if not self.in_context(state):
            return "only in a server channel"
        for gate in self.requires:
            if not state.get(gate):
                return _GATE_REASONS.get(gate, "not available here")
        return "not available here"


# Why each `requires` gate is shut, in the checklist's own voice. Kept beside the
# step table rather than inside `_Step` so a new gate is one line in each of two
# places that sit together, and never a message assembled in the renderer.
_GATE_REASONS = {
    "provider": "create your account first",
    "server_key": "needs an API key assigned to this server",
    "can_cast": "needs an administrator, or Open casting (anyone may add characters)",
    "seated": "add a character to this channel first",
}


WIZARD_STEPS = (
    # First: it decides which key the next step asks for, and it can be answered anywhere.
    # Answering it is registering -- the one way in (`UserSettings.is_registered`) -- so
    # every step that would store something waits on it. The first answer comes only
    # through `CreateAccountModal`, with the Terms accepted; the buttons are for switching.
    _Step("provider", "Create your account",
          ("1. Getting Started", "API Keys and Where They Apply"),
          lambda s: bool(s["provider"]), repeatable=True,
          done_detail=lambda s: PROVIDER_CHOICES.get(s["provider"], s["provider"])),
    _Step("key", "Add your API key",
          ("1. Getting Started", "API Keys and Where They Apply"),
          # In a server you run, not done until it has a key: Personal alone runs no session
          # here, and step 4 would refuse with nothing pointing back to this one.
          lambda s: s["has_key"] and (s["server_key"] or not s["is_admin"]), requires="provider",
          actions=("_act_get_key", "_act_paste_key"),
          done_detail=lambda s: s["key_detail"] or "using this server's key"),
    # Done once a character has something written: a borrow and a Generate draft arrive
    # that way, and a blank `/profile create` is not yet a character to talk to.
    _Step("profile", "Get a character",
          ("2. Writing a Character", "Persona vs Instructions"),
          lambda s: s["has_written"], requires="provider", repeatable=True,
          actions=("_act_library", "_act_generate"),
          done_detail=lambda s: f"`{s['written_name']}` is ready"),
    _Step("seat", "Add it to this channel",
          ("5. Sessions", "Starting and Shaping a Session"),
          lambda s: s["seated"], context="guild", requires=("server_key", "can_cast"),
          actions=("_act_cast",), repeatable=True,
          done_detail=lambda s: f"{s['seated_count']} in the cast"),
    _Step("speak", "Say something to it",
          ("5. Sessions", "Reactivity and Proactivity"),
          lambda s: s["has_spoken"], context="guild", requires="seated",
          done_detail=lambda s: "it has spoken here"),
)
#: The step Home's Session button stands for: its gates are the cast editor's.
_SEAT = next(step for step in WIZARD_STEPS if step.key == "seat")


def _lacking(perms, wanted: discord.Permissions) -> List[str]:
    """The flags `wanted` sets that `perms` does not grant, by discord.py's flag names."""
    return [name for name, on in wanted if on and not getattr(perms, name)]


def _missing_permissions(interaction: discord.Interaction) -> List[str]:
    """What of `BOT_PERMISSIONS` the bot lacks in this channel.

    Read off the gateway cache, so Refresh sees a fix. A channel the cache does not hold
    falls back to the snapshot sent with the command: a PartialMessageable answers every
    permission question with none.
    """
    guild, channel = interaction.guild, interaction.channel
    if guild is None:
        return []
    if guild.me is not None and channel is not None and not isinstance(
            channel, discord.PartialMessageable):
        perms = channel.permissions_for(guild.me)
    else:
        perms = interaction.app_permissions
    return _lacking(perms, BOT_PERMISSIONS)


def _child_bot_gaps(cog: "MimicCog", guild, channel_id: int,
                    profiles: List[Dict]) -> List[Tuple[str, Optional[List[str]]]]:
    """(bot name, what of `CHILD_BOT_PERMISSIONS` it lacks here) for each seated child bot
    that falls short, with None for one that is not in this server at all.

    A child bot that is not connected is passed over: nothing can be read off it, and being
    offline is not something an admin can grant.
    """
    if guild is None:
        return []
    gaps = []
    for p in profiles:
        if p.get("method") != "child_bot":
            continue
        child = cog.child_bot_manager.clients.get(str(p.get("bot_id")))
        if child is None or not child.is_ready():
            continue
        child_guild = child.get_guild(guild.id)
        if child_guild is None:
            gaps.append((child.user.name, None))
            continue
        # Every channel arrives with the guild, so None is a thread it has not cached.
        channel = child_guild.get_channel_or_thread(channel_id)
        if channel is None or child_guild.me is None:
            continue
        lacking = _lacking(channel.permissions_for(child_guild.me), CHILD_BOT_PERMISSIONS)
        if lacking:
            gaps.append((child.user.name, lacking))
    return gaps


def _permission_lines(flags: List[str]) -> str:
    lines = []
    for flag in flags:
        name, effect = PERMISSION_EFFECTS.get(flag, (flag.replace("_", " ").title(), ""))
        lines.append(f"• **{name}**" + (f": {effect}" if effect else ""))
    return "\n".join(lines)


def _permission_warning(mine: List[str], children: List[Tuple[str, Optional[List[str]]]]) -> str:
    """The banner's warning: what each bot lacks in this channel, and what that stops."""
    parts = ["⚠️ **Missing permissions in this channel.** An admin can grant them in the "
             "channel's **Permissions** settings."]
    if mine:
        parts.append("**Me:**\n" + _permission_lines(mine))
    for name, lacking in children:
        if lacking is None:
            parts.append(f"**{name}** (child bot) isn't in this server, so it can't speak here. "
                         "Its owner's `/invite` has the link.")
        else:
            parts.append(f"**{name}** (child bot):\n" + _permission_lines(lacking))
    return "\n".join(parts)


async def gather_state(cog: "MimicCog", interaction: discord.Interaction) -> Dict[str, Any]:
    """Everything every probe and the banner needs, read once.

    One `to_thread` for the whole sweep. The individual reads are small -- the profile
    index is plaintext and already cached, the session is in memory -- but `keys.json.gz`
    is an AES-GCM+zstd decrypt, and doing any of it inline on a repaint is the kind of
    thing that adds up on a shared event loop.
    """
    user_id = interaction.user.id
    guild = interaction.guild
    channel_id = interaction.channel_id

    has_key = await cog.storage_manager._has_api_key_access(user_id, interaction.guild_id)
    session = cog.multi_profile_channels.get(channel_id)
    sub = cog.subscriptions.row(user_id) if cog.subscriptions.enabled else None
    subscribed = bool(sub and sub.get("status") == ACTIVE)

    def _sync() -> Dict[str, Any]:
        index = cog.profile_manager._get_user_index(user_id) or {}
        personal = sorted(index.get("personal", {}) or {})
        borrowed = sorted(index.get("borrowed", {}) or {})

        # A borrow arrives already written, so its existence settles the "voice" step
        # without reading anything. Personal profiles are checked one at a time and the
        # walk stops at the first that has content -- a user with a hundred profiles
        # must not pay a hundred decrypts to be told step three is done.
        written_name = borrowed[0] if borrowed else None
        if not written_name:
            for name in personal:
                prompts = cog.profile_manager._get_profile_prompts(user_id, name) or {}
                persona = prompts.get("persona") or {}
                instructions = prompts.get("ai_instructions") or []
                if any(any(str(line).strip() for line in (lines or []))
                       for lines in persona.values()):
                    written_name = name
                    break
                if any(str(part).strip() for part in instructions):
                    written_name = name
                    break

        # A free-tier Gemini key carries no conversation -- in a server or in Global Chat --
        # unless the bot owner opened that server to it (cogs/utils/data_policy). The key
        # step says so rather than just ticking, and a server whose only key is one is not
        # counted as keyed.
        gemini_open_here = guild is not None and training_opt_in(
            cog.server_manager._get_server_index(str(guild.id)) or {}, "gemini")

        key_detail = ""
        personal_assigned = False
        if has_key:
            keys_data = cog.storage_manager._get_user_keys_data(user_id) or {}
            personal_assigned = bool(keys_data.get("personal_assignments"))
            slots = [s for s in (keys_data.get("slots") or {}).values()
                     if isinstance(s, dict) and s.get("key")]
            names = {"gemini": "Gemini", "openrouter": "OpenRouter"}
            labels = [f"Subscription · {tier_name(sub.get('tier'))}"] if subscribed else []
            for provider in sorted({s.get("provider", "?") for s in slots},
                                   key=lambda p: (p != "openrouter", p)):
                mine = [s for s in slots if s.get("provider", "?") == provider]
                tiers = "/".join(sorted({str(s.get("tier", "free")).title() for s in mine}))
                label = f"{names.get(provider, provider)} · {tiers} tier"
                if (provider == "gemini" and not gemini_open_here
                        and not any(is_paid_gemini_slot(s) for s in mine)):
                    label += " (not used in servers or Global Chat)"
                labels.append(label)
            key_detail = ", ".join(labels)

        server_has_key = False
        server_key_held = False
        if guild is not None:
            idx = cog.server_manager._get_server_index(str(guild.id)) or {}
            server_key_held = cog.storage_manager.gemini_blocked_for_guild(guild.id)
            held = {"gemini"} if server_key_held else set()
            server_has_key = bool(set(idx.get("assigned_keys") or {}) - held)

        # A dehydrated session has an empty in-memory log but a blueprint on disk. It
        # has been used; reading the log back to prove it would mean decrypting a whole
        # transcript to render a tick.
        seated_disk = False
        if guild is not None:
            idx = cog.server_manager._get_server_index(str(guild.id)) or {}
            blueprint = (idx.get("active_sessions", {}) or {}).get("regular", {}) or {}
            saved = blueprint.get(str(channel_id)) or {}
            # A cast that was seated but never started is a draft, and the step it
            # belongs to is not done. `started` is absent on blueprints written before
            # the flag existed, and those were live under the old rules.
            seated_disk = bool(saved.get("profiles")) and saved.get("started", True)

        # Read through the manager rather than off the blueprint above: it is the same
        # answer `/session config` gates on, live session first, and asking it here
        # keeps the wizard from greying out a step the command would have allowed.
        cast_policy = (cog.session_manager.cast_policy_for_channel(guild.id, channel_id)
                       if guild is not None else DEFAULT_CAST_POLICY)

        # `_has_api_key_access` counts any key assigned to the server, including one the
        # data policy holds back, which lets nothing here talk.
        usable_key = has_key and (guild is None or personal_assigned or server_has_key)

        return {"personal": personal, "borrowed": borrowed, "written_name": written_name,
                "provider": (index.get("about") or {}).get("provider"),
                "has_key": usable_key, "key_detail": key_detail,
                "server_has_key": server_has_key, "server_key_held": server_key_held,
                # A key of yours that would work here: Use it in <server> assigns it.
                "usable_slot": (cog.storage_manager.usable_key_slot(user_id, guild.id)
                                if guild is not None else None),
                "seated_disk": seated_disk,
                "cast_policy": cast_policy,
                "sells": cog.subscriptions.sells, "subscribed": subscribed}

    state = await asyncio.to_thread(_sync)

    is_owner = user_id == int(defaultConfig.DISCORD_OWNER_ID)
    is_admin = bool(
        guild is not None
        and (is_owner or getattr(interaction.user, "guild_permissions", None)
             and interaction.user.guild_permissions.administrator))

    profiles = (session or {}).get("profiles") or []
    log = (session or {}).get("unified_log") or []
    hydrated = bool((session or {}).get("is_hydrated"))

    state.update({
        "in_guild": guild is not None,
        "is_admin": is_admin,
        "is_owner": is_owner,
        # The `/session config` gate, mirrored: administrators always, everyone else
        # only where the channel's cast policy says Open casting. `/session swap` is
        # not reachable from here and stays admin-only regardless.
        "can_cast": bool(guild is not None
                         and (is_admin or state["cast_policy"] == CAST_POLICY_OPEN)),
        # `/session config`'s other gate, mirrored for the same reason.
        "server_key": guild is not None and cog._session_key_block(guild, user_id) is None,
        "guild": guild,
        "channel": interaction.channel,
        # Every invite asks for exactly BOT_PERMISSIONS, which a channel's overrides can
        # revoke. A bot invited with Administrator (a self-hosted one, before 0.7.8) lacks none.
        "missing_perms": _missing_permissions(interaction),
        "child_gaps": _child_bot_gaps(cog, guild, channel_id, profiles),
        "has_written": bool(state["written_name"]),
        "seated": (bool(profiles) and cog.session_manager.is_started(session)
                   ) or state["seated_disk"],
        "seated_count": len(profiles),
        # SYSTEM turns and synopses carry no profile_name, so they cannot pass this;
        # only a character actually speaking does. An unhydrated session that exists on
        # disk has been used, and is taken at its word.
        "has_spoken": (state["seated_disk"] and not hydrated) or any(
            not t.get("is_user") and t.get("profile_name") for t in log),
    })
    return state


class StartWizardView(BlockedGuard, TimeoutCleanupMixin, KeyScopeMixin, ui.View):
    timeout_message = "Setup closed. Run `/start` again — it picks up where you left off."
    # Every key added here is Personal; the picker under the key step is for servers.
    _OFFER_PERSONAL = False

    def __init__(self, cog: "MimicCog", interaction: discord.Interaction,
                 state: Dict[str, Any]):
        super().__init__(timeout=900)
        self.cog = cog
        self.original_interaction = interaction
        self.user_id = interaction.user.id
        self.state = state
        self.screen = "setup"
        # The step picked from the dropdown; None follows the first one not done.
        self.step_key: Optional[str] = None
        self.tour_page = next(iter(WIZARD_TOUR))
        self.home_page = 0
        self._init_scopes()
        if interaction.guild is not None:
            # The server it was run in first: the one its runner came here to fix.
            self.admin_guilds.sort(key=lambda g: g.id != interaction.guild.id)
        self._sync_slot()
        self._build_view()

    # --- state ------------------------------------------------------------

    @property
    def focus(self) -> Optional[_Step]:
        """The step on screen: the one picked, else the first not done. None is Home."""
        if self.step_key == HOME:
            return None
        picked = next((s for s in WIZARD_STEPS if s.key == self.step_key), None)
        return picked or self._next_incomplete()

    def _done(self, step: _Step) -> bool:
        try:
            return bool(step.probe(self.state))
        except Exception:
            return False

    def _next_incomplete(self) -> Optional[_Step]:
        """The first step not done that can be taken here. None opens Home: a DM, or a member
        who can't cast here, has nothing left to do in this place, and the checklist still
        shows the rest greyed with its reason."""
        return next((s for s in WIZARD_STEPS
                     if not self._done(s) and s.available(self.state)), None)

    def _settled(self) -> bool:
        """The steps that are yours rather than a channel's are done: Home has something to show."""
        return all(self._done(s) for s in WIZARD_STEPS if s.context == "any")

    def _sync_slot(self):
        """The key the server picker assigns: your Personal key for the provider you
        chose, else any Personal key you hold. Reloaded from disk on every repaint,
        since every pick here is saved as it is made."""
        keys = self.cog.storage_manager._get_user_keys_data(self.user_id) or {}
        personal = keys.get("personal_assignments") or {}
        chosen = self.state.get("provider")
        slot = (personal.get("openrouter" if chosen == PROVIDER_SUBSCRIPTION else chosen)
                or next(iter(personal.values()), None))
        if not slot and chosen == PROVIDER_SUBSCRIPTION and self.state.get("subscribed"):
            slot = SUBSCRIPTION_SLOT  # bought here, assigned nowhere yet: the picker places it
        self.selected_slot = slot if slot in SLOT_PROVIDER else None
        self._load_scopes()

    async def _repaint(self):
        """Re-probe, rebuild, redraw. What every action that changes state ends on --
        and what the key form and the override prompt call back into."""
        self.state = await gather_state(self.cog, self.original_interaction)
        self._sync_slot()
        self._build_view()
        await self.update_display()

    def watch(self):
        """Re-probe every WATCH_SECONDS while open, so a step finished in another message --
        a draft saved, a borrow, a cast started, the first reply, a sign-in -- ticks itself
        and the wizard moves on. 🔄 Refresh stays as the backup."""
        self._watcher = asyncio.create_task(self._watch())

    async def _watch(self):
        while not self.is_finished():
            await asyncio.sleep(WATCH_SECONDS)
            if self.screen != "setup" or self.is_finished():
                continue
            try:
                shown = self.embed().to_dict()
                self.state = await gather_state(self.cog, self.original_interaction)
                self._sync_slot()
                # Rebuilt only to be sent: a rebuild gives every component a new custom_id,
                # so one Discord never saw leaves the buttons on screen dead -- and an edit
                # closes a dropdown the user has open, so nothing unchanged is sent.
                if self.is_finished() or self.embed().to_dict() == shown:
                    continue
                self._build_view()
                await self.update_display()
            except discord.HTTPException as e:
                if e.status < 500:
                    return  # dismissed, or the token lapsed: nothing left to paint
            except Exception as e:
                print(f"StartWizardView watch: {type(e).__name__}({e})")

    async def _scopes_picked(self, interaction: discord.Interaction):
        """Applied as picked, Personal kept: nothing here waits on a Save."""
        if not self.selected_slot:
            return
        scopes = self.selected_scopes | {"personal"}
        conflicts = self._assignment_conflicts(self.selected_slot, scopes)
        if conflicts:
            # Nothing is taken from another key unasked. The ticks fall back to what is
            # saved on the repaint, and the prompt applies them if confirmed.
            await interaction.followup.send(
                embed=text_embed(OverrideConfirmView.prompt(conflicts)),
                view=OverrideConfirmView(self, self.selected_slot, scopes), ephemeral=True)
            return
        self._apply_assignments(self.selected_slot, scopes)

    # --- rendering --------------------------------------------------------

    def _banner(self) -> str:
        s = self.state
        if not s["in_guild"]:
            admins = len(self.admin_guilds)
            admin_line = f"admin of **{admins}**" if admins else "not an admin anywhere yet"
            return (f"📍 **You're in** a direct message with me\n"
                    f"👤 **You** are in {len(self.cog.bot.guilds)} server(s) I'm in, {admin_line}\n\n"
                    "Adding a character to a channel happens in a server: run `/start` there "
                    "when you reach it.")

        guild, channel = s["guild"], s["channel"]
        where = f"📍 **You're in** #{getattr(channel, 'name', 'this channel')} · **{guild.name}**\n"
        # A subscription is assigned to a server like any key, so where it is sold it is named.
        what = "subscription or API key" if s.get("sells") else "API key"
        if s["server_has_key"]:
            key_line = f"🔑 **This server** has a {what} assigned"
            if s["server_key_held"]:
                key_line += " (its free-tier Google key is not used here)"
        elif s["server_key_held"]:
            key_line = ("🔑 **This server**'s only key is a free-tier Google key, which is not "
                        "used for server messages — nothing here can generate")
        else:
            key_line = f"🔑 **This server** has no {what} assigned yet — nothing here can generate"
        if s["missing_perms"] or s["child_gaps"]:
            key_line += "\n" + _permission_warning(s["missing_perms"], s["child_gaps"])

        if s["is_admin"]:
            # In the banner, not the key step: that field is near Discord's 1024 cap.
            pitch = ("" if s["server_has_key"]
                     else "\n\n" + (SERVER_KEY_UNASSIGNED.format(server=guild.name)
                                    if s.get("usable_slot") else server_key_pitch(s.get("sells"))))
            return f"{where}🛡️ **Your role** Server administrator\n{key_line}{pitch}"
        role = "👤 **Your role** Member (not an administrator)\n"
        # Who may seat a character, said on the step that seats one, and on Home until one is.
        if (self.focus.key != "seat") if self.focus else s["seated"]:
            return f"{where}{role}{key_line}"
        if s["can_cast"]:
            note = ("This channel is on **Open casting** (anyone may add characters) — you "
                    "can add them here yourself.")
        else:
            note = ("Only admins can add characters to this channel.\n\n"
                    "💡 **Want somewhere to test freely?** Make your own server — it's free "
                    "and takes about thirty seconds (**+** in your server list → *Create My "
                    "Own*). You'll be its admin, and `/invite` adds me to it. Your profiles "
                    "come with you: they belong to you, not to a server.")
        return f"{where}{role}{key_line}\n\n{note}"

    def _mark(self, step: _Step) -> str:
        if self._done(step):
            return "✅"
        if not step.available(self.state):
            return "🔒" if step.in_context(self.state) else "↗️"
        return "⬜"

    def _checklist(self) -> str:
        lines = []
        focus = self.focus
        for number, step in enumerate(WIZARD_STEPS, start=1):
            mark = self._mark(step)
            if mark == "✅":
                detail = step.done_detail(self.state) if step.done_detail else ""
            elif mark != "⬜":
                detail = step.blocker(self.state)
            else:
                detail = "← you are here" if step is focus else ""
            padded = f"{mark} **{number}. {step.title}**"
            lines.append(f"{padded}  ·  {detail}" if detail else padded)
        return "\n".join(lines)

    def _step_text(self, step: _Step) -> str:
        """The focused step's field: what to do, then where it stands."""
        parts = [step.blurb(self.state)]
        mark = self._mark(step)
        if mark == "✅":
            detail = step.done_detail(self.state) if step.done_detail else ""
            parts.append("✅ **Done**" + (f" · {detail}" if detail else ""))
        elif mark != "⬜":
            reason = step.blocker(self.state)
            parts.append(f"{mark} **Not yet** — {reason[0].upper()}{reason[1:]}.")
        if step.key == "key" and self.selected_slot:
            parts.append("**In use:** " + _fit_list(self._scope_labels(self.selected_scopes),
                                                    ", ", 300))
        return "\n\n".join(p for p in parts if p)

    def embed(self) -> discord.Embed:
        if self.screen == "tour":
            return discord.Embed(title=self.tour_page, description=WIZARD_TOUR[self.tour_page],
                                 color=discord.Color.blurple()
                                 ).set_author(name="MimicAI · Using it")

        focus = self.focus
        if focus is None:
            e = discord.Embed(title="MimicAI Home", description=self._banner(),
                              color=discord.Color.green())
            e.add_field(name="Your setup", value=self._checklist(), inline=False)
            e.add_field(name="Where to?", value=self._home_text(), inline=False)
            return e
        done = sum(1 for s in WIZARD_STEPS if self._done(s))
        e = discord.Embed(title="Getting Started", description=self._banner(),
                          color=(discord.Color.green() if done == len(WIZARD_STEPS)
                                 else discord.Color.blurple()))
        e.add_field(name=f"Setup — {done} of {len(WIZARD_STEPS)} done",
                    value=self._checklist(), inline=False)
        e.add_field(name=f"Step {WIZARD_STEPS.index(focus) + 1}. {focus.title}",
                    value=self._step_text(focus), inline=False)
        return e

    def _offers_subscription(self) -> bool:
        return bool(self.state.get("sells") or self.state.get("subscribed"))

    def _home_text(self) -> str:
        session = WIZARD_HOME["session"]
        if not _SEAT.available(self.state):
            session += f" · *{_SEAT.blocker(self.state)}*"
        lines = [WIZARD_HOME["intro"], WIZARD_HOME["character"], session,
                 WIZARD_HOME["library"], WIZARD_HOME["settings"]]
        if self._offers_subscription():
            lines.append(WIZARD_HOME["subscription"])
        return "\n".join(lines + [WIZARD_HOME["help"], WIZARD_HOME["vote"]])

    def render(self) -> Dict[str, Any]:
        return {"content": None, "embed": self.embed(), "view": self}

    # --- view -------------------------------------------------------------

    def _build_view(self):
        self.clear_items()
        if self.screen == "tour":
            self._build_tour()
        else:
            self._build_setup()

    def _build_setup(self):
        focus = self.focus
        if focus is not None:
            done = self._done(focus)
            if focus.available(self.state) and (
                    not done or (focus.repeatable and self.step_key == focus.key)):
                for button in self._action_buttons(focus):
                    self.add_item(button)
            if done and self.step_key and self._next_incomplete() is not None:
                add_button(self, "Next step ▸", self._act_next,
                           style=discord.ButtonStyle.primary, row=0)
            if focus.key == "key" and self.selected_slot and self.admin_guilds:
                self._add_scope_select(row=1, placeholder="Use this key in servers you run...")

        options = []
        for number, step in enumerate(WIZARD_STEPS, start=1):
            options.append(discord.SelectOption(
                label=f"{self._mark(step)} {number}. {step.title}"[:100], value=step.key,
                description=step.blurb(self.state).replace("**", "").split("\n")[0][:100] or None,
                default=step is focus))
        add_select(self, options, self._act_pick_step, placeholder="Jump to a step...", row=2)

        add_button(self, "🔄 Refresh", self._act_refresh, style=discord.ButtonStyle.secondary,
                   row=3)
        # Not before: what there is to do once it talks is noise while it cannot.
        if focus is None:
            self._build_home()
            add_button(self, "❓ Help", self._act_help, row=3)
            add_button(self, "Using it ▸", self._act_tour, style=discord.ButtonStyle.primary,
                       row=3)
            add_topgg_links(self, row=4)
        else:
            add_button(self, "📖 Guide", self._act_guide, style=discord.ButtonStyle.secondary, row=3)
            if self._settled():
                add_button(self, "🏠 Home", self._act_home, row=3)

    def _home_names(self) -> List[Tuple[str, str]]:
        """(name, kind) for each character Home's dropdown offers, from the state's own read."""
        return ([(n, "Personal") for n in self.state["personal"]]
                + [(n, "Borrowed") for n in self.state["borrowed"]])

    def _build_home(self):
        """Home's controls. Each opens the screen its command opens, as a new message."""
        add_button(self, "Session", self._act_cast, emoji="🎬", row=0,
                   disabled=not _SEAT.available(self.state))
        add_button(self, "Library", self._act_library, emoji="🏛️", row=0)
        add_button(self, "Settings", self._act_settings, emoji="⚙️", row=0)
        if self._offers_subscription():
            add_button(self, "Subscription", self._act_subscribe, emoji="⭐", row=0)
        names = self._home_names()
        if not names:
            return
        pages = -(-len(names) // _HOME_PAGE)
        self.home_page = max(0, min(self.home_page, pages - 1))
        start = self.home_page * _HOME_PAGE
        # Prefixed: a character may be named like a page control.
        options = paged_nav_options(self.home_page, pages, nav_suffix=" of characters") + [
            discord.SelectOption(label=name[:100], value=f"p:{name}", description=kind)
            for name, kind in names[start:start + _HOME_PAGE]]
        add_select(self, options, self._act_manage, placeholder="🎭 Manage a character...", row=1)

    def _build_tour(self):
        options = [discord.SelectOption(label=page, value=page,
                                        default=(page == self.tour_page))
                   for page in WIZARD_TOUR]
        select = ui.Select(placeholder="Choose a topic...", options=options, row=0)

        async def pick(interaction: discord.Interaction):
            self.tour_page = select.values[0]
            self._build_view()
            await interaction.response.edit_message(**self.render())

        select.callback = pick
        self.add_item(select)

        async def go_back(interaction: discord.Interaction):
            self.screen, self.step_key = "setup", None
            self._build_view()
            await interaction.response.edit_message(**self.render())
        add_button(self, "◂ Setup", go_back, style=discord.ButtonStyle.primary, row=1)

        add_button(self, "📖 Guide", self._act_guide, style=discord.ButtonStyle.secondary, row=1)

    _ACTION_LABELS = {
        "_act_create_account": ("Create Account", discord.ButtonStyle.success),
        "_act_terms": ("📜 Read the Terms", discord.ButtonStyle.secondary),
        "_act_library": ("🏛️ Browse Library", discord.ButtonStyle.success),
        "_act_generate": ("✨ Generate one", discord.ButtonStyle.primary),
        "_act_cast": ("Open Cast Editor", discord.ButtonStyle.success),
    }

    def _key_providers(self) -> List[str]:
        """The provider chosen, or with None chosen, either: the key is theirs to pick. The
        subscription runs on OpenRouter, so where it is no longer sold, that key."""
        chosen = self.state.get("provider")
        chosen = "openrouter" if chosen == PROVIDER_SUBSCRIPTION else chosen
        return [chosen] if chosen in KEY_PAGES else list(KEY_PAGES)

    def _provider_buttons(self) -> List[ui.Button]:
        """A switch of provider, one button per choice offered, the recommended first."""
        buttons = []
        for n, provider in enumerate(offered_providers(self.cog.subscriptions.sells)):
            btn = ui.Button(label=PROVIDER_CHOICES[provider] + (" (recommended)" if n == 0 else ""),
                            emoji="⭐" if provider == PROVIDER_SUBSCRIPTION else None, row=0,
                            style=discord.ButtonStyle.success if n == 0 else discord.ButtonStyle.secondary)
            btn.callback = lambda i, p=provider: self._prefer(i, p)
            buttons.append(btn)
        return buttons

    def _action_buttons(self, step: _Step) -> List[ui.Button]:
        if step.key == "provider":
            if self._done(step):
                return self._provider_buttons()
            return [self._make_action_button(name) for name in ("_act_create_account", "_act_terms")]
        if step.key != "key":
            return [self._make_action_button(name) for name in step.actions]
        providers = self._key_providers()
        named = len(providers) > 1
        buttons = []
        s = self.state
        sells = self.cog.subscriptions.sells
        # A subscriber keeps their way to it while the owner has new sales paused.
        subscribing = (sells or s.get("subscribed")) and s.get("provider") == PROVIDER_SUBSCRIPTION
        if s.get("usable_slot") and s["is_admin"] and not s["server_key"]:
            use = ui.Button(label=f"Use it in {s['guild'].name}"[:80],
                            style=discord.ButtonStyle.primary, row=0)
            use.callback = self._act_use_here
            buttons.append(use)
        if subscribing:
            sub = ui.Button(label="Your subscription" if s.get("subscribed") else "Choose a tier",
                            emoji="⭐", style=discord.ButtonStyle.success, row=0)
            sub.callback = self._act_subscribe
            buttons.append(sub)
            return buttons
        for provider in providers:
            name = MODEL_PROVIDERS[provider]
            # A link: Discord opens it, and no interaction reaches us.
            if provider == "openrouter" and openrouter_signin.enabled():
                buttons.append(ui.Button(label="Sign in with OpenRouter ↗", row=0,
                                         url=self._sign_in_link()))
            else:
                buttons.append(ui.Button(label=f"Get {name} key ↗" if named else "Get a key ↗",
                                         row=0, url=KEY_PAGES[provider]))
            paste = ui.Button(label=f"🔑 Paste {name} key" if named else "🔑 Paste key",
                              style=discord.ButtonStyle.success, row=0)
            paste.callback = lambda i, p=provider: self._act_paste_key(i, p)
            buttons.append(paste)
        if sells:
            # Row 0 can already hold five.
            sub = ui.Button(label="⭐ MimicAI Subscription", style=discord.ButtonStyle.secondary, row=3)
            sub.callback = self._act_subscribe
            buttons.append(sub)
        return buttons

    def _make_action_button(self, name: str) -> ui.Button:
        label, style = self._ACTION_LABELS.get(name, (name, discord.ButtonStyle.secondary))
        btn = ui.Button(label=label, style=style, row=0)
        btn.callback = getattr(self, name)
        return btn

    # --- actions ----------------------------------------------------------
    #
    # Every launcher defers with thinking=True. On a component interaction that is a
    # deferred_channel_message rather than a message update, so the view it opens edits
    # a NEW ephemeral message and this one is still on screen to refresh afterwards.

    async def _act_pick_step(self, interaction: discord.Interaction):
        self.step_key = interaction.data["values"][0]
        self._build_view()
        await interaction.response.edit_message(**self.render())

    async def _act_next(self, interaction: discord.Interaction):
        self.step_key = None
        self._build_view()
        await interaction.response.edit_message(**self.render())

    async def _act_refresh(self, interaction: discord.Interaction):
        await interaction.response.defer()
        await self._repaint()

    async def _act_home(self, interaction: discord.Interaction):
        self.step_key = HOME
        self._build_view()
        await interaction.response.edit_message(**self.render())

    async def _act_settings(self, interaction: discord.Interaction):
        if await refuse_unregistered(self.cog, interaction):
            return
        await interaction.response.defer(thinking=True, ephemeral=True)
        await SettingsHomeView(self.cog, interaction).update_display()

    async def _act_help(self, interaction: discord.Interaction):
        # Sends its own new message, as `_act_terms` does.
        await self.cog.help_slash.callback(self.cog, interaction)

    async def _act_manage(self, interaction: discord.Interaction):
        value = interaction.data["values"][0]
        prev_page, jump_page, next_page = PAGE_NAV_VALUES
        if value == jump_page:
            async def jump(inner: discord.Interaction, page: int):
                self.home_page = page
                self._build_view()
                await inner.response.edit_message(**self.render())
            pages = -(-len(self._home_names()) // _HOME_PAGE)
            await interaction.response.send_modal(PageJumpModal(pages, jump, zero_indexed=True))
            return
        if value in (prev_page, next_page):
            self.home_page += -1 if value == prev_page else 1
            self._build_view()
            await interaction.response.edit_message(**self.render())
            return
        # `/profile manage`'s own path: a character deleted since this was drawn is offered
        # its closest matches, not an error.
        deferral = asyncio.create_task(interaction.response.defer(thinking=True, ephemeral=True))
        try:
            await self.cog._manage_profile(interaction, value.removeprefix("p:"), deferral)
        finally:
            await deferral

    async def _act_tour(self, interaction: discord.Interaction):
        self.screen = "tour"
        self._build_view()
        await interaction.response.edit_message(**self.render())

    async def _act_guide(self, interaction: discord.Interaction):
        focus = self.focus if self.screen == "setup" else None
        category, page = focus.help_ref if focus else (None, None)
        view = DropdownContentView(HELP_CATEGORIES, "MimicAI Help & Documentation",
                                   start_category=category, start_page=page)
        await interaction.response.send_message(embed=view.get_embed(), view=view, ephemeral=True)

    async def _act_use_here(self, interaction: discord.Interaction):
        """A key you already hold, made this server's: one added before it was automatic."""
        await interaction.response.defer()
        guild = self.state["guild"]
        told = self._claim_server(self.state["usable_slot"], guild).strip()
        await self._repaint()
        await interaction.followup.send(embed=text_embed(told or (
            f"**{guild.name}** kept the key it has: `/settings` → API Keys can replace it.")),
            ephemeral=True)

    async def _act_paste_key(self, interaction: discord.Interaction, provider: str):
        await interaction.response.send_modal(
            SubmitAPIKeyModal(self.cog, self._slot_for(provider), provider, view=self,
                              assign_personal=True))

    async def _act_subscribe(self, interaction: discord.Interaction):
        await interaction.response.defer(thinking=True, ephemeral=True)
        if not self.state.get("subscribed"):
            # Straight to the tiers; a purchase assigns itself and repaints this screen.
            await SubscribeView(self).open(interaction)
            return
        # The subscription's own screen, as `/subscribe` opens it.
        view = SettingsAPIView(self.cog, interaction)
        view.show_subscription(dedicated=True)
        await view.update_display()

    def _key_landed(self, slot: str, guild: Optional[discord.Guild]) -> str:
        if self.admin_guilds:
            # Stay on this step once it is done: the server picker appears under it.
            self.step_key = "key"
        return super()._key_landed(slot, guild)

    async def _act_create_account(self, interaction: discord.Interaction):
        await interaction.response.send_modal(CreateAccountModal(self))

    async def _act_terms(self, interaction: discord.Interaction):
        await self.cog.terms_slash.callback(self.cog, interaction)

    async def _prefer(self, interaction: discord.Interaction, provider: str):
        """A provider button, which switches an account's provider. One left on a wizard
        drawn before its account was deleted opens the Create Account form instead: an
        account is never made without the Terms accepted."""
        if not self.cog.user_settings.is_registered(self.user_id):
            await interaction.response.send_modal(CreateAccountModal(self))
            return
        await self._choose(interaction, provider)

    async def _choose(self, interaction: discord.Interaction, provider: str):
        """Asked here, stored by the setter Override Defaults uses, and changed in either.
        The first answer registers the user: see `UserSettings.is_registered`."""
        await interaction.response.defer()
        self.cog.user_settings.set_provider_preference(self.user_id, provider)
        self.step_key = None
        await self._repaint()

    async def _act_library(self, interaction: discord.Interaction):
        from .gui_hub import HubPublicLibraryView
        await interaction.response.defer(thinking=True, ephemeral=True)
        view = HubPublicLibraryView(self.cog, interaction)
        await view.update_display()

    async def _act_cast(self, interaction: discord.Interaction):
        # The step's own gate already tested this, but it is a shared surface reached
        # by a button, and a view can outlive both the permissions of the person holding
        # it and the channel's policy -- so it is re-probed at the point of use rather
        # than trusted from when the message was drawn.
        if not self.state["in_guild"] or not (
                self.state["is_admin"]
                or self.cog.session_manager.cast_policy_for_channel(
                    self.state["guild"].id, self.original_interaction.channel_id
                ) == CAST_POLICY_OPEN):
            await interaction.response.send_message(
                embed=text_embed("You must be a server administrator to configure sessions. An "
                "administrator can set this channel's session to **Open casting** to let "
                "anyone edit it."), ephemeral=True)
            return
        block = self.cog._session_key_block(self.state["guild"], interaction.user.id)
        if block:
            await interaction.response.send_message(embed=text_embed(block), ephemeral=True)
            return
        await interaction.response.defer(thinking=True, ephemeral=True)
        await self.cog._open_session_config(interaction)

    async def _act_generate(self, interaction: discord.Interaction):
        await open_generate(self.cog, interaction)

    async def update_display(self):
        """Paints onto the command interaction's own deferred response.

        Not a followup: TimeoutCleanupMixin strips this view through
        `original_interaction.edit_original_response`, so the view has to *be* the
        original response or the timeout tidies away an empty placeholder and leaves
        the wizard sitting there with dead buttons -- the exact failure that mixin
        exists to prevent.
        """
        await self.original_interaction.edit_original_response(**self.render())

    async def on_error(self, interaction: discord.Interaction, error: Exception, item: ui.Item):
        print(f"Error in StartWizardView: {type(error).__name__}({error})")
        try:
            message = "Something went wrong there. Press 🔄 Refresh and try again."
            if interaction.response.is_done():
                await interaction.followup.send(embed=text_embed(message), ephemeral=True)
            else:
                await interaction.response.send_message(embed=text_embed(message), ephemeral=True)
        except Exception:
            pass


class NewProfileModal(ui.Modal):
    """Collects a name and anything more, and hands off to `_generate_profile`, which drafts
    from all of it together (`compose_concept`).

    The one form behind `/profile generate`, the wizard's Generate and the Public Library's,
    so the three cannot drift: name validation, the limits, the draft and its error
    messages all live in that one method.
    """

    def __init__(self, cog: 'MimicCog'):
        super().__init__(title="Generate a Character")
        self.cog = cog
        self.name_input = ui.TextInput(
            label="Internal Name", max_length=32, required=True,
            placeholder="e.g. detective. Enough on its own; anything below adds to it.")
        self.add_item(self.name_input)
        self.concept_input = ui.TextInput(
            label="Concept (optional)", style=discord.TextStyle.paragraph,
            max_length=500, required=False,
            placeholder="e.g. A cynical noir detective who never removes his coat.")
        self.add_item(self.concept_input)
        self.image_input = ui.FileUpload(required=False)
        self.add_item(ui.Label(
            text="Image (optional)", component=self.image_input,
            description="Drafted from with the rest. It becomes the avatar unless you give a URL."))
        self.display_name_input = ui.TextInput(
            label="Display Name (optional)", max_length=20, required=False,
            placeholder="The name it speaks under, and the draft writes about. Blank: the draft picks.")
        self.add_item(self.display_name_input)
        self.avatar_url_input = ui.TextInput(
            label="Avatar URL (optional)", required=False, placeholder="https://...")
        self.add_item(self.avatar_url_input)  # Five components: all a modal holds.

    async def on_submit(self, interaction: discord.Interaction):
        await self.cog._generate_profile(
            interaction, (self.name_input.value or "").strip(),
            prompt=(self.concept_input.value or "").strip() or None,
            image=self.image_input.values[0] if self.image_input.values else None,
            display_name=(self.display_name_input.value or "").strip() or None,
            avatar_url=(self.avatar_url_input.value or "").strip() or None)


async def open_generate(cog: 'MimicCog', interaction: discord.Interaction):
    """Generate, from any of its three ways in: the form, or while an unsaved draft is still
    open, the offer of it first -- its message may have been dismissed by mistake."""
    draft = unsaved_draft(cog, interaction.user.id)
    if draft is None:
        await interaction.response.send_modal(NewProfileModal(cog))
        return
    title = draft.draft.get("display_name") or draft.profile_name
    await interaction.response.send_message(embed=text_embed(
        f"📝 **You have an unsaved draft:** {title}, to be saved as `{draft.profile_name}`.\n"
        "Reopen it, or start a new character."), view=ResumeDraftView(cog, interaction, draft), ephemeral=True)


class ResumeDraftView(BlockedGuard, TimeoutCleanupMixin, ui.View):
    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction, draft):
        super().__init__(timeout=120)
        self.cog = cog
        self.original_interaction = interaction
        self.draft = draft
        add_button(self, "Reopen draft", self.reopen, style=discord.ButtonStyle.green, emoji="📝")
        add_button(self, "New character", self.new)

    async def reopen(self, interaction: discord.Interaction):
        if self.draft.is_finished() or self.draft.saved:
            await interaction.response.edit_message(embed=text_embed(
                "That draft was saved, discarded or has expired."), view=None)
            return
        await self.draft.reopen(interaction)
        if self.draft.is_finished():
            self.stop()

    async def new(self, interaction: discord.Interaction):
        # Not stopped: the draft stays open, and Reopen with it, until a new one replaces it.
        await interaction.response.send_modal(NewProfileModal(self.cog))


#: The Create Account form's choices, (label, description), offered as `offered_providers`
#: orders them. A description is at most 100 characters.
_ACCOUNT_PROVIDERS = {
    PROVIDER_SUBSCRIPTION: ("MimicAI Subscription", "No key to set up. Monthly plan on curated "
                                                    "high-performance models. No images or speech."),
    "openrouter": ("OpenRouter", "Your own key. Hundreds of models (some free), plus images and speech."),
    "gemini": ("Google", "Your own key, Google's models only. Needs billing turned on."),
}

#: What Create Account must have ticked, every one.
_CONSENTS = {"terms": "I accept the Terms of Service",
             "privacy": "I have read the Privacy Policy"}


class CreateAccountModal(ui.Modal):
    """Step 1's form: the Terms and the provider, in one submit, which is registering.

    One form rather than an accept step ahead of the provider step, so there is never an
    account without a provider for `is_registered` to answer about. Nothing new is stored:
    `registered_at` is when the Terms were accepted, and which Terms is the version in
    force that day.
    """

    def __init__(self, wizard: StartWizardView):
        super().__init__(title="Create Your Account")
        self.wizard = wizard
        self.add_item(ui.TextDisplay(WIZARD_COPY["account"].format(date=LEGAL_EFFECTIVE_DATE)))
        self.consent = ui.CheckboxGroup(
            min_values=len(_CONSENTS), max_values=len(_CONSENTS),
            options=[discord.CheckboxGroupOption(label=label, value=value)
                     for value, label in _CONSENTS.items()])
        self.add_item(ui.Label(text="Tick both to continue", component=self.consent))
        self.offered = offered_providers(wizard.cog.subscriptions.sells)
        self.provider = ui.RadioGroup(options=[
            discord.RadioGroupOption(label=_ACCOUNT_PROVIDERS[value][0] + (" (recommended)" if n == 0 else ""),
                                     value=value, description=_ACCOUNT_PROVIDERS[value][1], default=n == 0)
            for n, value in enumerate(self.offered)])
        self.add_item(ui.Label(text="How your characters run", component=self.provider, description=(
            "The subscription needs no key. The others use your own."
            if PROVIDER_SUBSCRIPTION in self.offered else "Both use an API key of your own.")))

    async def on_submit(self, interaction: discord.Interaction):
        # Discord enforces both, but this is where an account is made, so it is checked here.
        if set(self.consent.values) != set(_CONSENTS) or self.provider.value not in self.offered:
            await interaction.response.send_message(
                embed=text_embed("Tick both boxes and choose a provider to create your account."), ephemeral=True)
            return
        await self.wizard._choose(interaction, self.provider.value)
