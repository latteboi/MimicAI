"""The table's data model: one seat, one forming lobby, one live game.

Split out of `game_service.py` so the narration and embed modules can name these
types without importing the service that drives them. Nothing here knows what a
channel is beyond holding the handles Discord hands back -- the rules live in
`eights.py`, the orchestration in `game_service.py`.
"""

import asyncio
import random
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import discord

from ...utils.constants import GAME_CONTEXT_EVENTS_KEEP, GAME_PANEL_PUSH_WINDOW_SECONDS
from . import ledger as ledger_mod
from ._shared import Event
from .eights import GameState, Move, RuleSet
from .neuro import Temperament


@dataclass
class Panel:
    """A seat's live private hand message, and the handle that can still edit it.

    The handle is the *most recent* interaction on the panel, not the one that created
    it. That distinction is the whole trick: a component click mints its own fifteen
    minute token targeting the message the component sits on, so re-binding on every
    click means anyone actually playing always has a live handle, while someone who
    walked away quietly stops being pushable. Their next click revives it.
    """

    interaction: Any
    stamped: float = field(default_factory=time.monotonic)

    #: Everything the panel is *showing*, at the last render. `_refresh_open_panels`
    #: compares it against the live one to tell a stale panel from a correct one.
    #:
    #: This used to be the hand size alone, and that was wrong in a way that looked
    #: exactly like the panel silently dying: your cards do not change while other
    #: people play, but the top card, the active colour, the pending-draw pile and
    #: whose turn it is all do -- and so do the enabled/disabled states of every
    #: control derived from them. A panel opened early would sit there showing a board
    #: six moves out of date, refusing to refresh because the one thing it was watching
    #: had not moved. None means "unknown, redraw me".
    signature: Optional[tuple] = None

    #: The live view object, so it can be stopped when the panel is replaced or the
    #: game ends. Views here have no timeout -- a game outlives discord.py's default
    #: 180 seconds many times over, and a timed-out view stops answering its own
    #: buttons -- so the only thing that ever retires one is this.
    view: Any = None

    @property
    def pushable(self) -> bool:
        """Whether the bot can still edit this panel *unprompted*. A click can always
        edit it regardless -- that path uses the click's own token, not this one."""
        return (time.monotonic() - self.stamped) < GAME_PANEL_PUSH_WINDOW_SECONDS


@dataclass
class Seat:
    """One place at the table, and how to speak as whoever is in it."""

    seat_id: str
    display: str
    kind: str = "profile"                 # "profile" | "human"
    owner_id: Optional[int] = None
    profile_name: Optional[str] = None
    method: str = "webhook"               # "webhook" | "child_bot"
    bot_id: Optional[str] = None


@dataclass
class Lobby:
    """A table that is forming but has not been dealt.

    Separate from `Game` rather than a phase on it, because everything that reads
    `cog.active_games` -- `has_live_game`, the channel listener's Last Card hook,
    `context_block`, the finale -- would otherwise have to learn to ignore a game with
    no `state`. A lobby has no rules engine behind it at all; it is a guest list.

    `humans` is a dict for the ordering as much as the lookup: seats are dealt in the
    order people sat down, and insertion order is what preserves that.
    """

    channel_id: int
    guild_id: Optional[int]
    host_id: int
    rules: RuleSet
    #: Total seats the host asked for, profiles included. None means "as many as fit".
    seats_wanted: Optional[int]
    humans: "OrderedDict[int, str]" = field(default_factory=OrderedDict)
    #: Locked by default, mirroring a global chat session: until the host unlocks it,
    #: nobody else can take a seat. The emoji on the button is the whole UI for this.
    open: bool = False
    message: Optional[discord.Message] = None

    @property
    def limit(self) -> int:
        """Seats at this table, profiles included."""
        return min(self.seats_wanted or self.rules.seats_max, self.rules.seats_max)


@dataclass
class Game:
    kind: str
    state: GameState
    seats: List[Seat]
    neuro: Dict[str, Dict[str, int]]
    temperaments: Dict[str, Temperament]
    channel_id: int
    guild_id: Optional[int]
    started_by: int
    rng: random.Random
    message_id: Optional[int] = None
    task: Optional[asyncio.Task] = None
    started_at: float = field(default_factory=time.monotonic)
    lap: int = 0
    turns: int = 0
    log: List[Event] = field(default_factory=list)
    stopping: bool = False
    _last_render: float = 0.0

    #: Set by a seat's controls when it submits a move; the run loop waits on it.
    #: The turn timer is this wait's timeout rather than a separate task -- the game
    #: already owns a task, and a `wait_for` deadline costs nothing extra.
    turn_event: Optional[asyncio.Event] = None
    pending_move: Optional["Move"] = None
    deadline: Optional[float] = None

    #: Plain-language descriptions of the last few things that happened, for the
    #: `<game_context>` block. Bounded, and never written to `unified_log` -- the log
    #: gets dialogue, this gets the bookkeeping.
    recent: Deque[str] = field(default_factory=lambda: deque(maxlen=GAME_CONTEXT_EVENTS_KEEP))

    #: One live hand panel per seat, keyed by seat_id. See `Panel`.
    panels: Dict[str, Panel] = field(default_factory=dict)

    #: Seats that have called Last Card and not yet spent it, keyed by seat_id. It lives on
    #: the game rather than on the panel or the view because it is now armed from the
    #: channel -- a player types "last card" and the next play carries it -- and because a
    #: pushed refresh builds a fresh view, which would otherwise silently disarm
    #: someone who armed it and was then made to pick up four cards.
    last_call_armed: Dict[str, bool] = field(default_factory=dict)

    #: Sticky-table bookkeeping. `resink_pending` is what stops a burst of dialogue
    #: becoming a burst of reposts, and what stops the repost's own gateway echo
    #: triggering another one.
    message: Optional[discord.Message] = None
    resink_pending: bool = False
    _last_repost: float = 0.0

    #: Serialises `_render`. Two tasks can now reach it -- the run loop and a resink
    #: scheduled by `nudge_table` -- and a repost is a delete followed by a send with an
    #: await in between. Without this, both could observe the table as buried, both
    #: delete (the second harmlessly failing), and both post: two tables, one of them
    #: orphaned with a live view attached.
    render_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    #: Running record of the sitting. Deliberately not LTM -- see `games/ledger.py`.
    ledger: Optional["ledger_mod.Ledger"] = None

    #: Model calls this game has spent, capped at GAME_REACTION_MAX_CALLS. Counted
    #: rather than assumed -- `tests/test_game_dialogue.py` asserts on it.
    generations: int = 0

    def seat(self, seat_id: str) -> Optional[Seat]:
        return next((s for s in self.seats if s.seat_id == seat_id), None)

