"""Recovery UI for a profile name that didn't resolve.

Every command taking a profile name used to dead-end on a bare "Profile 'x' not found."
-- accurate, and useless: the user is left to guess at spelling with no way to see what
they actually own. This module turns that dead end into a one-click correction.

The matching itself lives in `utils.fuzzy`, which also ranks `master_autocomplete`, so
the suggestions offered here are the same ones the picker would have shown had the user
paused before submitting.

Everything sent from here is ephemeral, and rendered with a native Discord embed.
"""

import discord
from discord import ui
from typing import TYPE_CHECKING, Awaitable, Callable, List, Optional, Sequence

from ..utils.fuzzy import MAX_CHOICES, best_match, rank_keyed

if TYPE_CHECKING:
    # This only runs during "hinting" and prevents the circular crash
    from ..MimicCog import MimicCog

#: How many suggestions get their own one-click button. Discord allows five components
#: per action row, and a sixth would push the dropdown down a row for no real gain.
BUTTON_SUGGESTIONS = 5

#: Badges by profile kind, so a borrowed profile is distinguishable from a personal one
#: of the same name without opening anything.
KIND_EMOJI = {
    "personal": "👤",
    "borrowed": "🤝",
    "system": "⭐",
    "participant": "🎭",
}


class ProfileCandidate:
    """One selectable profile in a suggestion prompt.

    `value` is what gets handed back to the caller's `on_pick`, and is deliberately
    separate from `name`: commands like /speak identify a participant as "owner_id:name"
    while matching on the bare name.
    """

    __slots__ = ("value", "name", "kind", "description")

    def __init__(self, value: str, name: str, kind: str = "personal",
                 description: Optional[str] = None):
        self.value = value
        self.name = name
        self.kind = kind
        self.description = description

    @property
    def emoji(self) -> str:
        return KIND_EMOJI.get(self.kind, "👤")

    @property
    def label(self) -> str:
        # Profile names are capped at 20 characters on creation, so this only ever
        # trims pathological legacy data rather than normal names.
        return self.name[:80]


class ProfileSuggestionView(ui.View):
    """"Did you mean?" prompt: top matches as buttons, the rest in a dropdown.

    Does not resolve anything itself. On a selection it invokes `on_pick`, which is the
    calling command's own continuation -- so /profile manage opens its dashboard and
    /session swap performs its swap, from the same prompt.
    """

    def __init__(self, cog: 'MimicCog', user_id: int, typed: str,
                 candidates: Sequence[ProfileCandidate],
                 on_pick: Callable[[discord.Interaction, str], Awaitable[None]],
                 *, timeout: float = 180.0, defer_on_pick: bool = True):
        super().__init__(timeout=timeout)
        self.cog = cog
        self.user_id = user_id
        self.typed = typed
        # Capped here rather than at each use, so the button row, the dropdown and the
        # embed footer all agree on how many candidates actually exist.
        self.candidates = list(candidates)[:BUTTON_SUGGESTIONS + MAX_CHOICES]
        self.on_pick = on_pick
        self.defer_on_pick = defer_on_pick
        # Set by `send`, so on_timeout can strip the controls off the right message.
        self.message: Optional[discord.Message] = None
        # Guards against a second click landing while a slow continuation is still
        # running -- see _dispatch.
        self._consumed = False
        self._build_view()

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Only the user who typed the name may act on the correction.

        These prompts are ephemeral, so this is belt-and-braces rather than a live hole
        -- but it means the view stays safe if it is ever reused somewhere public.
        """
        if interaction.user.id != self.user_id:
            await interaction.response.send_message(
                "This prompt isn't yours to answer.", ephemeral=True)
            return False
        return True

    def _build_view(self):
        self.clear_items()

        top = self.candidates[:BUTTON_SUGGESTIONS]
        rest = self.candidates[BUTTON_SUGGESTIONS:]

        for position, candidate in enumerate(top):
            btn = ui.Button(
                label=candidate.label,
                emoji=candidate.emoji,
                # The strongest match is highlighted, so the common case -- a single
                # transposed letter -- is one obvious click rather than a read.
                style=discord.ButtonStyle.success if position == 0 else discord.ButtonStyle.secondary,
                row=0,
            )
            btn.callback = self._make_pick_callback(candidate.value)
            self.add_item(btn)

        if rest:
            options = [
                discord.SelectOption(
                    label=candidate.label[:100],
                    value=candidate.value[:100],
                    description=(candidate.description or candidate.kind.title())[:100],
                    emoji=candidate.emoji,
                )
                for candidate in rest
            ]
            select = ui.Select(placeholder="More profiles...", options=options, row=1)
            select.callback = self._select_callback
            self.add_item(select)

        cancel = ui.Button(label="Cancel", style=discord.ButtonStyle.secondary, row=2)
        cancel.callback = self._cancel_callback
        self.add_item(cancel)

    def _make_pick_callback(self, value: str):
        async def callback(interaction: discord.Interaction):
            await self._dispatch(interaction, value)
        return callback

    async def _select_callback(self, interaction: discord.Interaction):
        values = (interaction.data or {}).get("values") or []
        if not values:
            return
        await self._dispatch(interaction, values[0])

    async def _dispatch(self, interaction: discord.Interaction, value: str):
        """Hand the chosen value to the caller's continuation.

        Two hand-off modes, because commands need different things from the interaction:

        `defer_on_pick=True` (the default) responds with a deferred *update*, giving the
        continuation an interaction whose `edit_original_response` addresses the prompt
        message -- so the prompt becomes the result in place, with no spent "Did you
        mean?" card left above it.

        `defer_on_pick=False` hands over an untouched interaction, for continuations that
        cannot work with a deferred one: a modal can only be sent in response to a fresh
        interaction, and a command whose real output is public needs to choose
        `ephemeral=False` for itself. The prompt is retired separately once the
        continuation has responded.
        """
        if self._consumed:
            # A second click that raced the first. discord.py stops dispatching only
            # after the view is evicted from its store, so a slow continuation leaves a
            # real window here.
            await interaction.response.send_message(
                "That selection is already being handled.", ephemeral=True)
            return
        self._consumed = True
        self.stop()

        if self.defer_on_pick:
            await interaction.response.defer()
            await self.on_pick(interaction, value)
            return

        try:
            await self.on_pick(interaction, value)
        finally:
            # In `finally` because the prompt is already spent -- `_consumed` is set, so
            # its buttons are dead. Leaving the card on screen after a continuation that
            # raised would give the user controls that silently do nothing.
            #
            # The continuation responds on its own interaction, so this message is ours
            # alone to clean up: it is addressed through the *original* command's
            # webhook, not this component's.
            await self._retire(f"Continuing with `{value}`...")

    async def _retire(self, content: str):
        """Strip the prompt down to a one-line note. Silent if it is already gone."""
        if self.message is None:
            return
        try:
            await self.message.edit(content=content, embed=None, view=None)
        except discord.HTTPException:
            pass

    async def _cancel_callback(self, interaction: discord.Interaction):
        self._consumed = True
        self.stop()
        await interaction.response.edit_message(
            content="Cancelled.", embed=None, view=None)

    async def on_timeout(self):
        # Ephemeral messages expire on their own, and a prompt we can no longer edit is
        # exactly the state this is trying to reach -- _retire swallows that.
        await self._retire("This suggestion prompt has expired. Run the command again.")

    def build_embed(self, *, noun: str = "profile") -> discord.Embed:
        if self.candidates:
            title = f"❓ No {noun} named '{self.typed[:60]}'"
            description = "Did you mean one of these?"
        else:
            title = f"❌ No {noun} named '{self.typed[:60]}'"
            description = (
                f"Nothing close enough to suggest. Use `/profile list` to see everything "
                f"you own, or `/profile create` to make a new one."
            )

        embed = discord.Embed(
            title=title,
            description=description,
            color=discord.Color.orange() if self.candidates else discord.Color.red(),
        )

        if self.candidates:
            # Names are capped at 20 characters on creation, so the per-line trim below
            # only ever touches legacy data -- but an embed field over 1024 characters
            # is rejected outright, which would fail the whole recovery prompt for the
            # user it is meant to rescue.
            lines = []
            budget = 1024
            for candidate in self.candidates[:BUTTON_SUGGESTIONS]:
                description = (candidate.description or candidate.kind.title())[:60]
                line = f"{candidate.emoji} `{candidate.name[:60]}` — {description}"
                if len(line) + 1 > budget:
                    break
                lines.append(line)
                budget -= len(line) + 1

            if lines:
                embed.add_field(name="Closest matches", value="\n".join(lines), inline=False)

            overflow = len(self.candidates) - BUTTON_SUGGESTIONS
            if overflow > 0:
                embed.set_footer(text=f"{overflow} more in the dropdown below.")

        return embed

    async def send(self, interaction: discord.Interaction, *, noun: str = "profile"):
        """Deliver the prompt, coping with either interaction state.

        Commands differ on whether they have deferred by the time they discover the name
        is bad, and sending the wrong one of these raises.
        """
        embed = self.build_embed(noun=noun)
        if interaction.response.is_done():
            self.message = await interaction.followup.send(
                embed=embed, view=self, ephemeral=True, wait=True)
        else:
            await interaction.response.send_message(embed=embed, view=self, ephemeral=True)
            try:
                self.message = await interaction.original_response()
            except discord.HTTPException:
                self.message = None


def gather_owned_candidates(cog: 'MimicCog', user_id: int, *,
                            include_system: bool = True,
                            only: Optional[Callable[[str, str], bool]] = None
                            ) -> List[ProfileCandidate]:
    """Every profile `user_id` can name, as candidates.

    Reads only the cached user index -- no profile bodies are loaded, because this runs
    on a failure path that must stay cheap even for an account at the 100/100 limit.
    `only` filters on (name, kind) for commands that accept a subset.
    """
    index = cog.profile_manager._get_user_index(user_id)
    candidates: List[ProfileCandidate] = []

    for name in sorted(index.get("personal", {})):
        if only is None or only(name, "personal"):
            candidates.append(ProfileCandidate(name, name, "personal", "Personal profile"))

    for name in sorted(index.get("borrowed", {})):
        if only is None or only(name, "borrowed"):
            candidates.append(ProfileCandidate(name, name, "borrowed", "Borrowed profile"))

    if include_system:
        # _is_system_name skips names the user has already claimed personally or by
        # borrow -- those were emitted above, and without the filter a clashing name
        # appeared twice in the same "Did you mean?" card under two different kinds.
        for name in sorted(cog.profile_manager._system_index()):
            if not cog.profile_manager._is_system_name(user_id, name):
                continue
            if only is None or only(name, "system"):
                candidates.append(ProfileCandidate(name, name, "system", "System profile"))

    return candidates


def gather_participant_candidates(session: dict, *,
                                  only: Optional[Callable[[dict], bool]] = None
                                  ) -> List[ProfileCandidate]:
    """Active participants in `session`, valued as "owner_id:name".

    /speak and /whisper both address a participant rather than a profile the user owns,
    and both already parse that composite value, so it is what the prompt hands back.
    """
    candidates: List[ProfileCandidate] = []
    for participant in session.get("profiles", []):
        if only is not None and not only(participant):
            continue
        name = participant.get("profile_name")
        owner_id = participant.get("owner_id")
        if not name or owner_id is None:
            continue
        candidates.append(ProfileCandidate(
            f"{owner_id}:{name}", name, "participant", "Active in this session"))
    return candidates


def autocorrect_profile(typed: str, candidates: Sequence[ProfileCandidate]) -> Optional[str]:
    """The value of the one candidate `typed` unambiguously means, or None.

    Only resolves differences in case, spacing and punctuation -- never a genuine typo,
    which stays the user's call to confirm. Profile names are lowercased at creation but
    several commands never lowercase their input, so "/profile manage Alice" was failing
    on a correctly-spelled name; this is the case that fixes.

    Returns None on a tie, so two profiles differing only in case still prompt.
    """
    by_name = {c.name: c for c in candidates}
    match = best_match(typed, by_name.keys())
    return by_name[match].value if match is not None else None


async def suggest_profile(cog: 'MimicCog', interaction: discord.Interaction, typed: str,
                          candidates: Sequence[ProfileCandidate],
                          on_pick: Callable[[discord.Interaction, str], Awaitable[None]],
                          *, noun: str = "profile", defer_on_pick: bool = True) -> None:
    """Send a "Did you mean?" prompt for `typed` against `candidates`.

    Ranks on the profile name, keeps the ordering the scorer produced, and falls back to
    showing everything the user owns when nothing scored above the cutoff -- an
    unhelpful list still beats a dead end.

    `on_pick` is awaited as on_pick(component_interaction, value). By default that
    interaction arrives already deferred as an update, so the continuation repaints the
    prompt into its result with `interaction.edit_original_response(...)` or adds a
    message with `interaction.followup.send(...)`.

    Pass `defer_on_pick=False` where the continuation needs an untouched interaction --
    to send a modal, or to respond publicly. It then owns the first response, and must
    make it within Discord's three-second window.
    """
    # Keyed by value, not name: two session participants can carry the same profile
    # name under different owners, and collapsing them would silently drop one.
    by_value = {c.value: c for c in candidates}
    ranked = rank_keyed(
        typed,
        [(c.value, c.name) for c in candidates],
        limit=MAX_CHOICES + BUTTON_SUGGESTIONS,
    )

    ordered = [by_value[value] for value, _ in ranked]
    if not ordered:
        ordered = list(candidates)[:MAX_CHOICES + BUTTON_SUGGESTIONS]

    view = ProfileSuggestionView(cog, interaction.user.id, typed, ordered, on_pick,
                                 defer_on_pick=defer_on_pick)
    await view.send(interaction, noun=noun)
