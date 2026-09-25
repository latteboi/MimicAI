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
bounded. A stale wizard is harmless for the same reason: its buttons re-probe, so it
needs none of the `active_session_config_views` machinery `/session config` carries.

It is a router, not a second implementation. The key step opens `/settings`'
`SubmitAPIKeyModal` and its server picker is `/settings`' own (`KeyScopeMixin`),
applying each pick as it is made instead of staging it for a Save. The other steps
launch the real dashboard -- `HubPublicLibraryView`, `/profile generate`,
`SessionConfigView` -- as a *separate* ephemeral message, by deferring with
`thinking=True`. On a component interaction that sends a new message rather than
updating this one, so those views' `edit_original_response` lands on the new message
and the wizard survives underneath to be refreshed.

Prose lives in `content.WIZARD_COPY`, and depth is not repeated: each step names a
`HELP_CATEGORIES` page, and the Guide button opens the guide browser on it.
"""

import asyncio
import discord
from discord import ui
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..utils.constants import CAST_POLICY_OPEN, DEFAULT_CAST_POLICY, MODEL_PROVIDERS, defaultConfig
from ..utils.data_policy import is_paid_gemini_slot, training_opt_in
from ..utils.content import HELP_CATEGORIES, WIZARD_COPY, WIZARD_TOUR
from .base_components import (BlockedGuard, DropdownContentView, TimeoutCleanupMixin, add_button,
                              add_select)
from .gui_settings import (KEY_SLOTS, SLOT_PROVIDER, KeyScopeMixin, OverrideConfirmView,
                           SubmitAPIKeyModal, _fit_list)

if TYPE_CHECKING:
    from ..MimicCog import MimicCog


#: Where each provider hands out keys, behind the key step's link button.
KEY_PAGES = {"openrouter": "https://openrouter.ai/settings/keys",
             "gemini": "https://aistudio.google.com/app/apikey"}


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
        copy = WIZARD_COPY.get(self.key, "")
        # The key step speaks for the provider chosen before it, and only that one.
        if isinstance(copy, dict):
            return copy.get(state.get("provider") or "openrouter", "")
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
    "provider": "choose a provider first",
    "server_key": "needs an API key assigned to this server",
    "can_cast": "needs administrator, or Open casting",
}


WIZARD_STEPS = (
    # First: it decides which key the next step asks for, and it can be answered anywhere.
    _Step("provider", "Choose a provider",
          ("1. Getting Started", "API Keys and Where They Apply"),
          lambda s: bool(s["provider"]), repeatable=True,
          actions=("_act_prefer_openrouter", "_act_prefer_gemini"),
          done_detail=lambda s: MODEL_PROVIDERS.get(s["provider"], s["provider"])),
    _Step("key", "Add your API key",
          ("1. Getting Started", "API Keys and Where They Apply"),
          lambda s: s["has_key"], requires="provider",
          actions=("_act_get_key", "_act_paste_key"),
          done_detail=lambda s: s["key_detail"] or "using this server's key"),
    # Done once a character has something written: a borrow and a Generate draft arrive
    # that way, and a blank `/profile create` is not yet a character to talk to.
    _Step("profile", "Get a character",
          ("2. Writing a Character", "Persona vs Instructions"),
          lambda s: s["has_written"], repeatable=True,
          actions=("_act_library", "_act_generate"),
          done_detail=lambda s: f"`{s['written_name']}` is ready"),
    _Step("seat", "Seat it in this channel",
          ("5. Sessions", "Starting and Shaping a Session"),
          lambda s: s["seated"], context="guild", requires=("server_key", "can_cast"),
          actions=("_act_cast",), repeatable=True,
          done_detail=lambda s: f"{s['seated_count']} in the cast"),
    _Step("speak", "Say something to it",
          ("5. Sessions", "Reactivity and Proactivity"),
          lambda s: s["has_spoken"], context="guild",
          done_detail=lambda s: "it has spoken here"),
)


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
            labels = []
            for provider in sorted({s.get("provider", "?") for s in slots}):
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
                "seated_disk": seated_disk,
                "cast_policy": cast_policy}

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
        self._init_scopes()
        if interaction.guild is not None:
            # The server it was run in first: the one its runner came here to fix.
            self.admin_guilds.sort(key=lambda g: g.id != interaction.guild.id)
        self._sync_slot()
        self._build_view()

    # --- state ------------------------------------------------------------

    @property
    def focus(self) -> Optional[_Step]:
        """The step on screen: the one picked, else the first not done."""
        picked = next((s for s in WIZARD_STEPS if s.key == self.step_key), None)
        return picked or self._next_incomplete()

    def _done(self, step: _Step) -> bool:
        try:
            return bool(step.probe(self.state))
        except Exception:
            return False

    def _next_incomplete(self) -> Optional[_Step]:
        return next((s for s in WIZARD_STEPS if not self._done(s)), None)

    def _sync_slot(self):
        """The key the server picker assigns: your Personal key for the provider you
        chose, else any Personal key you hold. Reloaded from disk on every repaint,
        since every pick here is saved as it is made."""
        keys = self.cog.storage_manager._get_user_keys_data(self.user_id) or {}
        personal = keys.get("personal_assignments") or {}
        slot = personal.get(self.state.get("provider")) or next(iter(personal.values()), None)
        self.selected_slot = slot if slot in SLOT_PROVIDER else None
        self._load_scopes()

    async def _repaint(self):
        """Re-probe, rebuild, redraw. What every action that changes state ends on --
        and what the key form and the override prompt call back into."""
        self.state = await gather_state(self.cog, self.original_interaction)
        self._sync_slot()
        self._build_view()
        await self.update_display()

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
                OverrideConfirmView.prompt(conflicts),
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
                    "Seating a character happens in a server channel: run `/start` there "
                    "when you reach it.")

        guild, channel = s["guild"], s["channel"]
        where = f"📍 **You're in** #{getattr(channel, 'name', 'this channel')} · **{guild.name}**\n"
        if s["server_has_key"]:
            key_line = "🔑 **This server** has an API key assigned"
            if s["server_key_held"]:
                key_line += " (its free-tier Google key is not used here)"
        elif s["server_key_held"]:
            key_line = ("🔑 **This server**'s only key is a free-tier Google key, which is not "
                        "used for server messages — nothing here can generate")
        else:
            key_line = "🔑 **This server** has no API key assigned yet — nothing here can generate"

        if s["is_admin"]:
            return f"{where}🛡️ **Your role** Server administrator\n{key_line}"
        role = "👤 **Your role** Member (not an administrator)\n"
        if s["can_cast"]:
            note = ("This channel is on **Open casting** — you can seat characters here "
                    "yourself.")
        else:
            note = ("Only admins can seat characters in this channel.\n\n"
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

        done = sum(1 for s in WIZARD_STEPS if self._done(s))
        e = discord.Embed(title="Getting Started", description=self._banner(),
                          color=(discord.Color.green() if done == len(WIZARD_STEPS)
                                 else discord.Color.blurple()))
        e.add_field(name=f"Setup — {done} of {len(WIZARD_STEPS)} done",
                    value=self._checklist(), inline=False)
        focus = self.focus
        if focus is None:
            e.add_field(name="You're set up",
                        value="Just talk in the channel. **Using it ▸** covers what else there is.",
                        inline=False)
        else:
            e.add_field(name=f"Step {WIZARD_STEPS.index(focus) + 1}. {focus.title}",
                        value=self._step_text(focus), inline=False)
        return e

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
                for name in focus.actions:
                    self.add_item(self._make_action_button(name))
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
        add_button(self, "📖 Guide", self._act_guide, style=discord.ButtonStyle.secondary, row=3)
        # Not before: what there is to do once it talks is noise while it cannot.
        if focus is None:
            add_button(self, "Using it ▸", self._act_tour, style=discord.ButtonStyle.primary,
                       row=3)

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
        "_act_prefer_openrouter": ("OpenRouter (recommended)", discord.ButtonStyle.success),
        "_act_prefer_gemini": ("Google", discord.ButtonStyle.secondary),
        "_act_paste_key": ("🔑 Paste key", discord.ButtonStyle.success),
        "_act_library": ("🏛️ Browse Library", discord.ButtonStyle.success),
        "_act_generate": ("✨ Generate one", discord.ButtonStyle.primary),
        "_act_cast": ("Open Cast Editor", discord.ButtonStyle.success),
    }

    def _make_action_button(self, name: str) -> ui.Button:
        if name == "_act_get_key":
            # A link: Discord opens it, and no interaction reaches us.
            return ui.Button(label="Get a key ↗", row=0,
                             url=KEY_PAGES[self.state.get("provider") or "openrouter"])
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

    async def _act_paste_key(self, interaction: discord.Interaction):
        provider = self.state.get("provider") or "openrouter"
        filled = (self.cog.storage_manager._get_user_keys_data(self.user_id) or {}).get("slots") or {}
        mine = [slot for slot, _label, p in KEY_SLOTS if p == provider]
        # An empty slot first; with both full, the first is replaced, as Edit Key would.
        slot = next((s for s in mine if s not in filled), mine[0])
        if self.admin_guilds:
            # Stay on this step once it is done: the server picker appears under it.
            self.step_key = "key"
        await interaction.response.send_modal(
            SubmitAPIKeyModal(self.cog, slot, provider, view=self, assign_personal=True))

    async def _act_prefer_gemini(self, interaction: discord.Interaction):
        await self._prefer(interaction, "gemini")

    async def _act_prefer_openrouter(self, interaction: discord.Interaction):
        await self._prefer(interaction, "openrouter")

    async def _prefer(self, interaction: discord.Interaction, provider: str):
        """Asked here, stored by the setter About Me uses, and changed in either."""
        await interaction.response.defer()
        self.cog.profile_manager.set_provider_preference(self.user_id, provider)
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
                "You must be a server administrator to configure sessions. An "
                "administrator can set this channel's session to **Open casting** to let "
                "anyone edit it.", ephemeral=True)
            return
        block = self.cog._session_key_block(self.state["guild"], interaction.user.id)
        if block:
            await interaction.response.send_message(block, ephemeral=True)
            return
        await interaction.response.defer(thinking=True, ephemeral=True)
        await self.cog._open_session_config(interaction)

    async def _act_generate(self, interaction: discord.Interaction):
        await interaction.response.send_modal(NewProfileModal(self.cog))

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
                await interaction.followup.send(message, ephemeral=True)
            else:
                await interaction.response.send_message(message, ephemeral=True)
        except Exception:
            pass


class NewProfileModal(ui.Modal):
    """Collects a name and a concept and hands off to `/profile generate`.

    That command carries name validation, the profile and key-access limits, and ninety
    lines of prompt assembly and parsing. Calling its callback directly means the wizard
    -- and the Public Library's Generate button -- cannot drift from what it does,
    including its error messages, which are the ones the rest of the documentation
    describes.
    """

    def __init__(self, cog: 'MimicCog'):
        super().__init__(title="Generate a Character")
        self.cog = cog
        self.name_input = ui.TextInput(
            label="Internal Name", placeholder="e.g. detective", max_length=32, required=True)
        self.add_item(self.name_input)
        self.concept_input = ui.TextInput(
            label="Concept", style=discord.TextStyle.paragraph, max_length=500,
            placeholder="e.g. A cynical noir detective who never removes his coat.",
            required=True)
        self.add_item(self.concept_input)
        self.display_name_input = ui.TextInput(
            label="Display Name (optional)", max_length=20, required=False,
            placeholder="The name it speaks under. Blank lets the draft pick one.")
        self.add_item(self.display_name_input)
        self.avatar_url_input = ui.TextInput(
            label="Avatar URL (optional)", required=False, placeholder="https://...")
        self.add_item(self.avatar_url_input)

    async def on_submit(self, interaction: discord.Interaction):
        await self.cog.profile_generate_slash.callback(
            self.cog, interaction, (self.concept_input.value or "").strip(),
            (self.name_input.value or "").strip(),
            (self.display_name_input.value or "").strip() or None,
            (self.avatar_url_input.value or "").strip() or None)
