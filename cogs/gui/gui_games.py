"""Discord surface for table games: the table's button, and a seat's private hand.

Two constraints shape everything here.

**A hand is private, and the only place it may be rendered is an ephemeral response
to the seat's own interaction.** Nothing in this module writes a card into a channel
message, and nothing hands a hand to anything but the player holding it.

**Discord allows 25 components per message and 25 options per select.** A late-game
hand after a couple of Draw Fours exceeds both, so the select never lists the *hand* --
it lists the distinct *legal plays*, and the full hand is shown as embed text, which has
no component limit at all.

That turns out to settle the limit question completely: the legal-play set is bounded at
**18** no matter how large the hand grows (thirteen distinct cards of the active colour,
three matching the top value in other colours, two wilds), which
`tests/test_game_ui.py` pins by brute force over the whole deck. The colour-grouping
step below is therefore a readability threshold, not a limit workaround -- an
eighteen-option dropdown is simply worse to read than a colour and then a card.

**One panel per seat, edited in place.** The fifteen-minute interaction token expiry is
real but narrower than it looks: it bounds what the *bot* may push unprompted, not what
a click may do. A component interaction mints its own token targeting the message the
component sits on, so a panel driven by its own buttons edits forever, however long the
game runs. Only the unprompted refresh -- `GameService.refresh_panel`, for when someone
else's move changed your hand -- lives inside the window, and it is re-bound on every
click, so anyone actually playing always has a live handle.

That is why nothing here sends a second message. `GameService.bind_panel` records the
handle, `close_panel` guarantees one panel per seat, and a move `defer`s so the run loop
can complete the same message once the move has actually been applied.

**The views do not time out.** discord.py retires a view after 180 idle seconds and it
then stops answering its own components, which on a table where four other people are
playing is most of a game -- the panel looked alive and did nothing. The game's own
lifecycle retires them instead: `bind_panel` stops the one it replaces and `close_panel`
stops the last, so a panel is live for exactly as long as the game behind it.

**Last Card is called by saying it.** There is no button. A seated player says `one` or
`last card` in the channel, `GameService.arm_last_call` takes it and the listener stops
there -- the call never queues a round of its own. The next play carries it. The two-click
arm-then-play the button asked for was the same contract with worse manners, and the
call is the one part of this game that has always belonged out loud.
"""

from typing import TYPE_CHECKING, Dict, List, Optional

import discord
from discord import ui

from ..services.games import eights
from ..services.games.eights import COLOURS, Card, Move

if TYPE_CHECKING:
    from ..MimicCog import MimicCog
    from ..services.game_service import Game

#: Discord's hard ceilings. Named rather than inlined because both are load-bearing.
MAX_SELECT_OPTIONS = 25
MAX_HAND_FIELD_CHARS = 1000

#: Above this many *distinct* legal plays, pick a colour first. This is a readability
#: threshold, not a limit workaround -- see `HandView._add_play_control` for why the
#: 25-option ceiling cannot actually be reached by a legal-play list.
GROUP_PLAYS_ABOVE = 12

_SWATCH = {
    eights.RED: "\U0001F7E5", eights.YELLOW: "\U0001F7E8",
    eights.GREEN: "\U0001F7E9", eights.BLUE: "\U0001F7E6", eights.WILD: "⬛",
}
_LABEL: Dict[str, str] = {
    eights.SKIP: "Skip", eights.REVERSE: "Reverse", eights.DRAW_TWO: "Draw Two",
    eights.WILD_PLAIN: "Wild", eights.DRAW_FOUR: "Draw Four",
}
_COLOUR_NAME = {
    eights.RED: "Red", eights.YELLOW: "Yellow", eights.GREEN: "Green", eights.BLUE: "Blue",
    eights.WILD: "Wild",
}


def card_text(card: Card) -> str:
    return f"{_SWATCH.get(card[0], '')} {_LABEL.get(card[1], card[1])}".strip()


def encode(card: Card) -> str:
    return f"{card[0]}|{card[1]}"


def decode(value: str) -> Card:
    colour, _, val = value.partition("|")
    return (colour, val)


# --------------------------------------------------------------------------- hand

def build_hand_embed(game: "Game", seat_id: str) -> discord.Embed:
    """The seat's own cards, grouped by colour.

    Rendered as text rather than components so hand size is never a limit -- a
    forty-card hand shows in full here, while the select below is bounded by the legal
    plays rather than by the hand.
    """
    state = game.state
    seat = state.seat(seat_id)
    hand = seat.hand

    embed = discord.Embed(
        title=f"Your hand — {len(hand)} card{'' if len(hand) == 1 else 's'}",
        colour=0x2B2F36,
    )
    for colour in (*COLOURS, eights.WILD):
        cards = [c for c in hand if c[0] == colour]
        if not cards:
            continue
        cards.sort(key=lambda c: (len(c[1]), c[1]))
        line = " · ".join(_LABEL.get(c[1], c[1]) for c in cards)
        if len(line) > MAX_HAND_FIELD_CHARS:
            line = line[:MAX_HAND_FIELD_CHARS].rsplit(" · ", 1)[0] + f" · …(+more)"
        embed.add_field(
            name=f"{_SWATCH.get(colour, '')} {_COLOUR_NAME.get(colour, colour)}"
                 f" ({len(cards)})",
            value=line, inline=False,
        )

    top = state.top
    board = [f"**Top** {card_text(top)}",
             f"**Colour** {_SWATCH.get(state.active_colour, '')} "
             f"{_COLOUR_NAME.get(state.active_colour, state.active_colour)}"]
    if state.pending_draw:
        board.append(f"**Pending** ⚠️ {state.pending_draw} to draw")
    embed.description = " · ".join(board)

    armed = bool(getattr(game, "last_call_armed", {}).get(seat_id))
    if armed:
        embed.set_footer(text="Last Card called — it rides on your next play.")
    elif len(hand) == 2:
        embed.set_footer(text='One card after this — say "one" in the channel, or take '
                              "the penalty.")
    elif state.current.seat_id != seat_id:
        embed.set_footer(text="Not your turn yet.")
    return embed


class ColourChoiceView(ui.View):
    """Second step of playing a wild. The colour arrives with the move, so the engine
    never needs a `choosing_colour` phase.

    Swapped onto the existing panel rather than sent as a second message -- the wild is
    one decision in two clicks, not two conversations.
    """

    def __init__(self, hand_view: "HandView", card: Card):
        super().__init__(timeout=None)
        self.hand_view = hand_view
        self.card = card
        for colour in COLOURS:
            button = ui.Button(
                label=_COLOUR_NAME[colour],
                emoji=_SWATCH[colour],
                style=discord.ButtonStyle.secondary,
            )
            button.callback = self._make_callback(colour)
            self.add_item(button)
        back = ui.Button(label="Back", style=discord.ButtonStyle.secondary, row=1)
        back.callback = self._on_back
        self.add_item(back)

    def _make_callback(self, colour: str):
        async def callback(interaction: discord.Interaction):
            await self.hand_view.submit(
                interaction, Move("play", self.card, declared_colour=colour))
        return callback

    async def _on_back(self, interaction: discord.Interaction) -> None:
        await self.hand_view.redraw(interaction)


class HandView(ui.View):
    """A seat's private controls. Rebuilt per interaction, but always onto the *same*
    message -- see the module docstring for why that is safe past fifteen minutes."""

    def __init__(self, cog: "MimicCog", game: "Game", seat_id: str,
                 colour_filter: Optional[str] = None):
        super().__init__(timeout=None)
        self.cog = cog
        self.game = game
        self.seat_id = seat_id
        self.colour_filter = colour_filter
        self._build()

    # -- construction --------------------------------------------------------------

    def _legal_plays(self) -> List[Move]:
        return [m for m in eights.legal_moves(self.game.state, self.seat_id)
                if m.kind == "play" and m.card]

    def _build(self) -> None:
        state = self.game.state
        my_turn = (state.phase == "playing"
                   and state.current.seat_id == self.seat_id)
        plays = self._legal_plays() if my_turn else []
        kinds = {m.kind for m in eights.legal_moves(state, self.seat_id)} if my_turn else set()

        if my_turn:
            self._add_play_control(plays)

        draw = ui.Button(
            label=("Take the stack" if state.pending_draw else "Draw"),
            emoji="🫳", style=discord.ButtonStyle.primary, row=1,
            disabled=not my_turn or "draw" not in kinds,
        )
        draw.callback = self._on_draw
        self.add_item(draw)

        skip = ui.Button(label="Pass", style=discord.ButtonStyle.secondary, row=1,
                         disabled=not my_turn or "pass" not in kinds)
        skip.callback = self._on_pass
        self.add_item(skip)

        refresh = ui.Button(label="Refresh", emoji="🔄",
                            style=discord.ButtonStyle.secondary, row=1)
        refresh.callback = self._on_refresh
        self.add_item(refresh)

    def _add_play_control(self, plays: List[Move]) -> None:
        """One select of legal plays, or a colour step first when there are many.

        Worth recording what the numbers actually are, because the obvious reading is
        wrong. A select tops out at 25 options, and a hand can hold sixty cards -- but
        the select lists *distinct legal plays*, and that set is bounded much lower than
        the hand: at most thirteen distinct cards of the active colour, three more
        matching the top card's value in other colours, and two wilds. Holding all 108
        cards at once yields **18**, which `tests/test_game_ui.py` pins by brute force.

        So Discord's ceiling is unreachable here and the colour step is not protecting
        against it. It exists because an eighteen-option dropdown is genuinely worse to
        read than picking a colour and then a card, and it kicks in at
        `GROUP_PLAYS_ABOVE` for that reason alone.

        The threshold counts *distinct* cards. Counting duplicates instead would send a
        hand of thirty reds -- thirteen distinct plays -- down the grouping path, to a
        pointless dropdown holding one colour.
        """
        if not plays:
            return

        if self.colour_filter:
            plays = [m for m in plays if m.card and m.card[0] == self.colour_filter]

        distinct: List[Card] = []
        for move in plays:
            if move.card and move.card not in distinct:
                distinct.append(move.card)

        if len(distinct) > GROUP_PLAYS_ABOVE and not self.colour_filter:
            groups: Dict[str, int] = {}
            for card in distinct:
                groups[card[0]] = groups.get(card[0], 0) + 1
            select = ui.Select(
                placeholder=f"{len(plays)} playable — pick a colour first",
                row=0,
                options=[
                    discord.SelectOption(
                        label=f"{_COLOUR_NAME.get(c, c)} ({n} playable)",
                        value=c, emoji=_SWATCH.get(c))
                    for c, n in sorted(groups.items(), key=lambda kv: -kv[1])
                ][:MAX_SELECT_OPTIONS],
            )
            select.callback = self._on_pick_colour_group
            self.add_item(select)
            return

        options = []
        for card in distinct:
            options.append(discord.SelectOption(
                label=_LABEL.get(card[1], f"{_COLOUR_NAME.get(card[0], card[0])} {card[1]}"),
                description=_COLOUR_NAME.get(card[0], card[0]),
                value=encode(card), emoji=_SWATCH.get(card[0]),
            ))
            if len(options) >= MAX_SELECT_OPTIONS:
                break

        if not options:
            return
        placeholder = ("Play a card" if not self.colour_filter
                       else f"Play a {_COLOUR_NAME.get(self.colour_filter, '')} card")
        select = ui.Select(placeholder=placeholder, row=0, options=options)
        select.callback = self._on_pick_card
        self.add_item(select)

        if self.colour_filter:
            back = ui.Button(label="Other colours", style=discord.ButtonStyle.secondary,
                             row=2)
            back.callback = self._on_clear_filter
            self.add_item(back)

    # -- helpers -------------------------------------------------------------------

    def _rebind(self, interaction: discord.Interaction, view=None) -> None:
        """Hand the service the freshest token for this panel, and the view now on it.

        The view goes across so the service can stop the one it replaces. Nothing else
        holds a reference to a panel's view, and a view that is never stopped sits in
        discord.py's store for the life of the process.
        """
        self.cog.game_service.bind_panel(self.game, self.seat_id, interaction,
                                         view=view if view is not None else self)

    async def redraw(self, interaction: discord.Interaction,
                     note: Optional[str] = None) -> None:
        """Update this panel in place from the click that asked for it.

        `edit_message` responds to *this* interaction and targets the message its
        component sits on, so it needs nothing from the fifteen-minute-old token that
        opened the panel and works however long the game has been running.
        """
        if self.game.state.phase != "playing":
            await interaction.response.edit_message(
                content="That game has finished.", embed=None, view=None)
            return
        view = HandView(self.cog, self.game, self.seat_id,
                        colour_filter=self.colour_filter)
        embed = build_hand_embed(self.game, self.seat_id)
        await interaction.response.edit_message(content=note, embed=embed, view=view)
        self._rebind(interaction, view)

    def _not_your_turn(self) -> bool:
        state = self.game.state
        return (state.phase != "playing"
                or self.game.stopping
                or state.current.seat_id != self.seat_id
                or self.game.pending_move is not None)

    async def submit(self, interaction: discord.Interaction, move: Move) -> None:
        """Hand a chosen move to the game loop, which is waiting on `turn_event`.

        Deferred rather than answered. The move has not been applied yet -- the run loop
        does that -- so there is no post-move hand to draw, and drawing the pre-move one
        would show cards that are about to leave. Binding the panel to this interaction
        hands the run loop the token to finish the edit with once `eights.apply` returns.

        A stale panel cannot get past this in a way that matters: the guard rejects a
        turn that has moved on, and `eights.apply` raises `IllegalMove` on anything that no
        longer fits the board. The worst a stale click achieves is a polite refusal.
        """
        if self._not_your_turn():
            await self.redraw(interaction,
                              note="That move is no longer available — "
                                   "the turn has moved on.")
            return
        self.game.pending_move = move
        await interaction.response.defer()
        self._rebind(interaction)
        if self.game.turn_event:
            self.game.turn_event.set()

    # -- callbacks -----------------------------------------------------------------

    async def _on_pick_card(self, interaction: discord.Interaction) -> None:
        card = decode(interaction.data["values"][0])
        if eights.is_wild(card):
            colours = ColourChoiceView(self, card)
            await interaction.response.edit_message(
                content=f"Playing {card_text(card)} — choose a colour:", view=colours)
            self._rebind(interaction, colours)
            return
        await self.submit(interaction, Move("play", card))

    async def _on_pick_colour_group(self, interaction: discord.Interaction) -> None:
        self.colour_filter = interaction.data["values"][0]
        await self.redraw(interaction)

    async def _on_clear_filter(self, interaction: discord.Interaction) -> None:
        self.colour_filter = None
        await self.redraw(interaction)

    async def _on_draw(self, interaction: discord.Interaction) -> None:
        await self.submit(interaction, Move("draw"))

    async def _on_pass(self, interaction: discord.Interaction) -> None:
        await self.submit(interaction, Move("pass"))

    async def _on_refresh(self, interaction: discord.Interaction) -> None:
        await self.redraw(interaction)


# --------------------------------------------------------------------------- table

class TableView(ui.View):
    """The one public control, attached to the table message.

    Anyone may click it; what comes back depends on whether they hold a seat. That is
    the whole access model -- a spectator gets a polite refusal, a player gets their
    own cards, and neither can see anyone else's.
    """

    def __init__(self, cog: "MimicCog", channel_id: int):
        super().__init__(timeout=None)
        self.cog = cog
        self.channel_id = channel_id

    @ui.button(label="Your hand", emoji="🃏", style=discord.ButtonStyle.primary)
    async def open_hand(self, interaction: discord.Interaction, _button: ui.Button):
        game = self.cog.active_games.get(self.channel_id)
        if not game or game.state.phase != "playing":
            await interaction.response.send_message(
                "There is no game running here any more.", ephemeral=True)
            return

        seat_id = f"user:{interaction.user.id}"
        if game.seat(seat_id) is None:
            await interaction.response.send_message(
                "You are not seated at this table — you are welcome to watch.",
                ephemeral=True)
            return

        # A table click cannot edit an ephemeral it does not own, so this is the one
        # path that still sends -- and it closes the previous panel first, which is what
        # keeps the invariant at one open panel per seat however often it is pressed.
        await self.cog.game_service.close_panel(game, seat_id)
        view = HandView(self.cog, game, seat_id)
        await interaction.response.send_message(
            embed=build_hand_embed(game, seat_id), view=view, ephemeral=True)
        self.cog.game_service.bind_panel(game, seat_id, interaction, view=view)
