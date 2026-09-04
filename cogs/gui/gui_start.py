"""`/start` -- the guided setup wizard.

Setup crosses two contexts and cannot complete in one message: API keys are entered in
a DM (`/settings` is dm_only), and a channel's cast is configured by a server
administrator (`/session config` is guild_only, and admin unless the channel is on Open
casting). So this is context-aware
rather than linear. Steps that cannot run where you are stay **visible and greyed**,
because a member who cannot see step 4 has no way to learn why the bot is silent.

**No progress is stored.** Every step's completion is probed from state that already
exists -- assigned keys, the profile index, the channel's session -- which means the
wizard cannot desynchronise from reality, "run it again to pick up where you left off"
is literally true, and there is no new per-user dict to keep bounded. A stale wizard is
harmless for the same reason: its buttons re-probe, so it needs none of the
`active_session_config_views` machinery `/session config` carries.

It is a router, not a second implementation. Each step launches the real dashboard --
`SettingsAPIView`, `HubPublicLibraryView`, `ProfileManageView`, `SessionConfigView` --
as a *separate* ephemeral message, by deferring with `thinking=True`. On a component
interaction that sends a new message rather than updating this one, so those views'
`edit_original_response` lands on the new message and the wizard survives underneath to
be refreshed. Nothing in them needed changing to be reachable from here.

Prose lives in `content.WIZARD_COPY`, and depth is not repeated: each step names a
`HELP_CATEGORIES` page, and Read More opens the guide browser on it.
"""

import asyncio
import discord
from discord import ui
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..utils.constants import CAST_POLICY_OPEN, DEFAULT_CAST_POLICY, defaultConfig
from ..utils.content import HELP_CATEGORIES, WIZARD_COPY, WIZARD_TOUR
from .base_components import DropdownContentView, TimeoutCleanupMixin

if TYPE_CHECKING:
    from ..MimicCog import MimicCog


class _Step:
    """One row of the setup checklist.

    `probe` reads the state dict assembled once per repaint rather than touching disk
    itself, so adding a step costs no extra reads unless it needs something new.

    `context` is "dm", "guild" or "any". `requires` names a boolean key in the same
    state dict, so a gate that is not a fixed property of the user -- seating is open
    to administrators *or* to anyone in an Open casting channel -- is probed at repaint
    like every other fact here rather than baked into the table. Both describe where the
    step's *action* can run; a step that cannot run here is still drawn, with the reason.

    `actions` names methods on the view. Kept as names so this table stays readable as
    data, and so a step with no action -- the last one, which is just "talk" -- simply
    declares none.
    """

    __slots__ = ("key", "title", "help_ref", "context", "requires", "probe",
                 "actions", "done_detail")

    def __init__(self, key, title, help_ref, probe, *, context="any",
                 requires=None, actions=(), done_detail=None):
        self.key = key
        self.title = title
        self.help_ref = help_ref
        self.probe = probe
        self.context = context
        self.requires = requires
        self.actions = actions
        self.done_detail = done_detail

    @property
    def blurb(self) -> str:
        return WIZARD_COPY.get(self.key, "")

    def available(self, state: Dict[str, Any]) -> bool:
        """Whether this step's action can be taken from where the command was run."""
        if self.context == "dm" and state["in_guild"]:
            return False
        if self.context == "guild" and not state["in_guild"]:
            return False
        if self.requires and not state.get(self.requires):
            return False
        return True

    def blocker(self, state: Dict[str, Any]) -> str:
        """Why this step is not actionable here. Only read when `available` is False."""
        if self.context == "dm" and state["in_guild"]:
            return "in a DM with me"
        if self.context == "guild" and not state["in_guild"]:
            return "in a server channel"
        if self.requires:
            return _GATE_REASONS.get(self.requires, "not available here")
        return "not available here"


# Why each `requires` gate is shut, in the checklist's own voice. Kept beside the
# step table rather than inside `_Step` so a new gate is one line in each of two
# places that sit together, and never a message assembled in the renderer.
_GATE_REASONS = {
    "can_cast": "needs administrator, or Open casting",
}


WIZARD_STEPS = (
    _Step("key", "Connect an API key",
          ("1. Getting Started", "API Keys and Where They Apply"),
          lambda s: s["has_key"], context="dm",
          actions=("_act_open_keys",),
          done_detail=lambda s: s["key_detail"]),
    _Step("profile", "Get a character",
          ("1. Getting Started", "Profile Classes (PIDs)"),
          lambda s: bool(s["personal"] or s["borrowed"]),
          actions=("_act_library", "_act_generate", "_act_create"),
          done_detail=lambda s: f"{len(s['personal']) + len(s['borrowed'])} profile(s)"),
    _Step("voice", "Give it a voice",
          ("2. Writing a Character", "Persona vs Instructions"),
          lambda s: s["has_written"],
          actions=("_act_open_dashboard",),
          done_detail=lambda s: (f"`{s['written_name']}` is written"
                                 if s["written_name"] else "ready")),
    _Step("seat", "Seat it in this channel",
          ("5. Sessions", "Starting and Shaping a Session"),
          lambda s: s["seated"], context="guild", requires="can_cast",
          actions=("_act_cast",),
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
    is a Fernet+zstd decrypt, and doing any of it inline on a repaint is the kind of
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

        key_detail = ""
        if has_key:
            keys_data = cog.storage_manager._get_user_keys_data(user_id) or {}
            slots = [s for s in (keys_data.get("slots") or {}).values()
                     if isinstance(s, dict) and s.get("key")]
            providers = sorted({s.get("provider", "?") for s in slots})
            tiers = sorted({str(s.get("tier", "free")).title() for s in slots})
            names = {"gemini": "Gemini", "openrouter": "OpenRouter"}
            key_detail = (", ".join(names.get(p, p) for p in providers)
                          + (f" · {'/'.join(tiers)} tier" if tiers else ""))

        server_has_key = False
        admin_guilds = 0
        if guild is not None:
            idx = cog.server_manager._get_server_index(str(guild.id)) or {}
            server_has_key = bool(idx.get("assigned_keys"))
        else:
            for g in cog.bot.guilds:
                member = g.get_member(user_id)
                if member and member.guild_permissions.administrator:
                    admin_guilds += 1

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

        return {"personal": personal, "borrowed": borrowed, "written_name": written_name,
                "key_detail": key_detail, "server_has_key": server_has_key,
                "admin_guilds": admin_guilds, "seated_disk": seated_disk,
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
        "has_key": has_key,
        "in_guild": guild is not None,
        "is_admin": is_admin,
        "is_owner": is_owner,
        # The `/session config` gate, mirrored: administrators always, everyone else
        # only where the channel's cast policy says Open casting. `/session swap` is
        # not reachable from here and stays admin-only regardless.
        "can_cast": bool(guild is not None
                         and (is_admin or state["cast_policy"] == CAST_POLICY_OPEN)),
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


class StartWizardView(TimeoutCleanupMixin, ui.View):
    timeout_message = "Setup closed. Run `/start` again — it picks up where you left off."

    def __init__(self, cog: "MimicCog", interaction: discord.Interaction,
                 state: Dict[str, Any]):
        super().__init__(timeout=900)
        self.cog = cog
        self.original_interaction = interaction
        self.user_id = interaction.user.id
        self.state = state
        self.screen = "overview"
        self.step_key: Optional[str] = None
        self.tour_page = next(iter(WIZARD_TOUR))
        self._build_view()

    # --- state ------------------------------------------------------------

    @property
    def step(self) -> Optional[_Step]:
        return next((s for s in WIZARD_STEPS if s.key == self.step_key), None)

    def _done(self, step: _Step) -> bool:
        try:
            return bool(step.probe(self.state))
        except Exception:
            return False

    def _next_incomplete(self) -> Optional[_Step]:
        return next((s for s in WIZARD_STEPS if not self._done(s)), None)

    async def _refresh_state(self, interaction: discord.Interaction):
        self.state = await gather_state(self.cog, self.original_interaction)

    # --- rendering --------------------------------------------------------

    def _banner(self) -> str:
        s = self.state
        if not s["in_guild"]:
            admin_line = (f"admin of **{s['admin_guilds']}**" if s["admin_guilds"]
                          else "not an admin anywhere yet")
            return (f"📍 **You're in** a direct message with me\n"
                    f"👤 **You** are in {len(self.cog.bot.guilds)} server(s) I'm in, {admin_line}\n\n"
                    "This is the only place API keys can be entered. Steps 1–3 work here; "
                    "for 4 and 5, run `/start` again in a channel.")

        guild, channel = s["guild"], s["channel"]
        where = f"📍 **You're in** #{getattr(channel, 'name', 'this channel')} · **{guild.name}**\n"
        key_line = ("🔑 **This server** has an API key assigned"
                    if s["server_has_key"] else
                    "🔑 **This server** has no API key assigned yet — nothing here can generate")

        if s["is_admin"]:
            role = "🛡️ **Your role** Server administrator\n"
            note = ("You can do everything here. Adding an API key is the one thing "
                    "Discord makes you do in a DM.")
        else:
            role = "👤 **Your role** Member (not an administrator)\n"
            note = (("You can build characters here, and this channel is on **Open "
                     "casting** — you can seat them yourself."
                     if s["can_cast"] else
                     "You can build characters here, but only admins can seat them in a "
                     "channel.")
                    + "\n\n💡 **Want somewhere to test freely?** Make your own server — "
                    "it's free and takes about thirty seconds (**+** in your server list → "
                    "*Create My Own*). You'll be its admin, and `/invite` adds me to it. "
                    "Your profiles come with you: they belong to you, not to a server.")
        return f"{where}{role}{key_line}\n\n{note}"

    def _checklist(self) -> str:
        lines = []
        for number, step in enumerate(WIZARD_STEPS, start=1):
            done = self._done(step)
            available = step.available(self.state)
            if done:
                mark, detail = "✅", (step.done_detail(self.state) if step.done_detail else "")
            elif not available:
                mark = "🔒" if step.requires and self.state["in_guild"] else "↗️"
                detail = step.blocker(self.state)
            else:
                mark = "⬜"
                detail = "← you are here" if step is self._next_incomplete() else ""
            padded = f"{mark} **{number}. {step.title}**"
            lines.append(f"{padded}  ·  {detail}" if detail else padded)
        return "\n".join(lines)

    def embed(self) -> discord.Embed:
        if self.screen == "tour":
            return discord.Embed(title=self.tour_page, description=WIZARD_TOUR[self.tour_page],
                                 color=discord.Color.blurple()
                                 ).set_author(name="MimicAI · Using it")

        if self.screen == "step" and self.step:
            step = self.step
            number = WIZARD_STEPS.index(step) + 1
            e = discord.Embed(title=f"Step {number}. {step.title}", description=step.blurb,
                              color=(discord.Color.green() if self._done(step)
                                     else discord.Color.blurple()))
            e.set_author(name="MimicAI · Getting Started")
            if self._done(step):
                detail = step.done_detail(self.state) if step.done_detail else ""
                e.add_field(name="Done", value=detail or "Already set up.", inline=False)
            elif not step.available(self.state):
                e.add_field(name="Not from here",
                            value=f"This step has to be done {step.blocker(self.state)}.",
                            inline=False)
            return e

        done = sum(1 for s in WIZARD_STEPS if self._done(s))
        e = discord.Embed(title="Getting Started", description=self._banner(),
                          color=(discord.Color.green() if done == len(WIZARD_STEPS)
                                 else discord.Color.blurple()))
        e.add_field(name=f"Setup — {done} of {len(WIZARD_STEPS)} done",
                    value=self._checklist(), inline=False)
        if done == len(WIZARD_STEPS):
            e.add_field(name="You're set up",
                        value="Just talk in the channel. **Using it ▸** covers what else there is.",
                        inline=False)
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
        options = []
        if self.screen == "step":
            options.append(discord.SelectOption(label="◂ Overview", value="__overview__",
                                                description="The whole checklist."))
        for number, step in enumerate(WIZARD_STEPS, start=1):
            mark = "✅" if self._done(step) else ("🔒" if not step.available(self.state) else "⬜")
            options.append(discord.SelectOption(
                label=f"{mark} {number}. {step.title}"[:100], value=step.key,
                description=step.blurb.replace("**", "").split("\n")[0][:100],
                default=(step.key == self.step_key and self.screen == "step")))

        select = ui.Select(placeholder="Open a step...", options=options[:25], row=0)

        async def pick(interaction: discord.Interaction):
            value = select.values[0]
            if value == "__overview__":
                self.screen, self.step_key = "overview", None
            else:
                self.screen, self.step_key = "step", value
            self._build_view()
            await interaction.response.edit_message(**self.render())

        select.callback = pick
        self.add_item(select)

        target = self.step if self.screen == "step" else self._next_incomplete()
        if target is not None and not self._done(target):
            if target.available(self.state):
                for name in target.actions:
                    self.add_item(self._make_action_button(target, name))
            elif target.context == "dm" and self.state["in_guild"]:
                # Otherwise a guild user with no key sees the step they are blocked on
                # and no way forward from it -- the one place this wizard could dead-end.
                # A step blocked the other way (guild-only, read in a DM) needs no button:
                # "go to a channel" is not something a button can do for them.
                self.add_item(self._make_action_button(target, "_act_dm_me"))

        if self.screen == "step" and self.step is not None:
            more = ui.Button(label="📖 Read more", style=discord.ButtonStyle.secondary, row=2)
            more.callback = self._act_read_more
            self.add_item(more)

        refresh = ui.Button(label="🔄 Refresh", style=discord.ButtonStyle.secondary, row=2)
        refresh.callback = self._act_refresh
        self.add_item(refresh)

        guide = ui.Button(label="Full guide", style=discord.ButtonStyle.secondary, row=2)
        guide.callback = self._act_guide
        self.add_item(guide)

        tour = ui.Button(label="Using it ▸", style=discord.ButtonStyle.primary, row=3)
        tour.callback = self._act_tour
        self.add_item(tour)

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

        back = ui.Button(label="◂ Setup", style=discord.ButtonStyle.primary, row=1)

        async def go_back(interaction: discord.Interaction):
            self.screen, self.step_key = "overview", None
            self._build_view()
            await interaction.response.edit_message(**self.render())

        back.callback = go_back
        self.add_item(back)

        guide = ui.Button(label="Full guide", style=discord.ButtonStyle.secondary, row=1)
        guide.callback = self._act_guide
        self.add_item(guide)

    _ACTION_LABELS = {
        "_act_open_keys": ("Open API Keys", discord.ButtonStyle.success),
        "_act_library": ("🏛️ Browse Library", discord.ButtonStyle.success),
        "_act_generate": ("✨ Generate one", discord.ButtonStyle.primary),
        "_act_create": ("📝 Blank", discord.ButtonStyle.secondary),
        "_act_open_dashboard": ("Open Dashboard", discord.ButtonStyle.success),
        "_act_cast": ("Open Cast Editor", discord.ButtonStyle.success),
        "_act_dm_me": ("📩 Send me the DM version", discord.ButtonStyle.success),
    }

    def _make_action_button(self, step: _Step, name: str) -> ui.Button:
        label, style = self._ACTION_LABELS.get(name, (name, discord.ButtonStyle.secondary))
        btn = ui.Button(label=label, style=style, row=1)
        btn.callback = getattr(self, name)
        return btn

    # --- actions ----------------------------------------------------------
    #
    # Every launcher defers with thinking=True. On a component interaction that is a
    # deferred_channel_message rather than a message update, so the view it opens edits
    # a NEW ephemeral message and this one is still on screen to refresh afterwards.

    async def _act_refresh(self, interaction: discord.Interaction):
        await interaction.response.defer()
        await self._refresh_state(interaction)
        self._build_view()
        await interaction.edit_original_response(**self.render())

    async def _act_tour(self, interaction: discord.Interaction):
        self.screen = "tour"
        self._build_view()
        await interaction.response.edit_message(**self.render())

    async def _act_guide(self, interaction: discord.Interaction):
        view = DropdownContentView(HELP_CATEGORIES, "MimicAI Help & Documentation")
        await interaction.response.send_message(embed=view.get_embed(), view=view, ephemeral=True)

    async def _act_read_more(self, interaction: discord.Interaction):
        step = self.step
        category, page = step.help_ref if step else (None, None)
        view = DropdownContentView(HELP_CATEGORIES, "MimicAI Help & Documentation",
                                   start_category=category, start_page=page)
        await interaction.response.send_message(embed=view.get_embed(), view=view, ephemeral=True)

    async def _act_dm_me(self, interaction: discord.Interaction):
        """Carries the key step into a DM, where it is the only place it can be done.

        Sends the step's own copy rather than another wizard: a view posted to a DM has
        no interaction behind it, so its Refresh and its timeout cleanup would both be
        dead controls. Running `/start` there builds a real one, with the DM's context.
        """
        await interaction.response.defer(thinking=True, ephemeral=True)
        step = next(s for s in WIZARD_STEPS if s.key == "key")
        try:
            channel = await interaction.user.create_dm()
            await channel.send(
                embed=discord.Embed(title="Step 1. Connect an API key",
                                    description=step.blurb,
                                    color=discord.Color.blurple()
                                    ).set_footer(text="Run /start here to continue setup."))
        except discord.Forbidden:
            await interaction.followup.send(
                "I can't DM you — your privacy settings block direct messages from this "
                "server. Turn them on for this server, or open a DM with me yourself and "
                "run `/start` there.", ephemeral=True)
            return
        except Exception:
            await interaction.followup.send(
                "I couldn't send that DM. Open a DM with me and run `/start` there.",
                ephemeral=True)
            return
        await interaction.followup.send(
            "📨 Sent. Check your DMs and run `/start` there to add a key — then come back "
            "here and press 🔄 Refresh.", ephemeral=True)

    async def _act_open_keys(self, interaction: discord.Interaction):
        from .gui_settings import SettingsAPIView
        await interaction.response.defer(thinking=True, ephemeral=True)
        view = SettingsAPIView(self.cog, interaction)
        await view.update_display()

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
        await interaction.response.defer(thinking=True, ephemeral=True)
        await self.cog._open_session_config(interaction)

    async def _act_open_dashboard(self, interaction: discord.Interaction):
        name = (self.state["personal"] or self.state["borrowed"] or [None])[0]
        if not name:
            await interaction.response.send_message(
                "Make a character first — step 2.", ephemeral=True)
            return
        await interaction.response.defer(thinking=True, ephemeral=True)
        await self.cog._open_profile_manage(interaction, name, repaint=True)

    async def _act_create(self, interaction: discord.Interaction):
        await interaction.response.send_modal(_NewProfileModal(self, generate=False))

    async def _act_generate(self, interaction: discord.Interaction):
        await interaction.response.send_modal(_NewProfileModal(self, generate=True))

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


class _NewProfileModal(ui.Modal):
    """Collects a name (and a concept) and hands off to the real slash command.

    `/profile create` and `/profile generate` carry name validation, the profile and
    key-access limits, and in the generate case ninety lines of prompt assembly and
    parsing. Calling their callbacks directly means the wizard cannot drift from what
    those commands do -- including their error messages, which are the ones the rest of
    the documentation describes.
    """

    def __init__(self, view: StartWizardView, *, generate: bool):
        super().__init__(title="Generate a Character" if generate else "New Character")
        self.parent_view = view
        self.generate = generate
        self.name_input = ui.TextInput(
            label="Name", placeholder="e.g. detective", max_length=32, required=True)
        self.add_item(self.name_input)
        if generate:
            self.concept_input = ui.TextInput(
                label="Concept", style=discord.TextStyle.paragraph, max_length=500,
                placeholder="e.g. A cynical noir detective who never removes his coat.",
                required=True)
            self.add_item(self.concept_input)

    async def on_submit(self, interaction: discord.Interaction):
        cog = self.parent_view.cog
        name = (self.name_input.value or "").strip()
        if self.generate:
            await cog.profile_generate_slash.callback(
                cog, interaction, (self.concept_input.value or "").strip(), name)
        else:
            await cog.create_profile_slash.callback(cog, interaction, name)
