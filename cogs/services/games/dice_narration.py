"""Mimic Dice's ledger and narration: `ledger.py` and `narration.py` for a dice table.

Same surface as those two, so `game_service` reads either game the same way, and the
same constraint: everything here is built from public state and counted by the engine.

`Ev.BLUFFED` has no branch in either half, and must not get one. It is the one private
event a dice table emits, and the neuro table is the only thing allowed to carry it --
a line saying who bluffed would read a cup out loud.
"""

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from ...utils.constants import GAME_REACTION_MAX_CALLS
from . import dice, neuro
from ._shared import Event, Ev
from .ledger import RECENT_KEEP
from .narration import describe_ledger, reactor_for  # noqa: F401 -- the same for any game
from .table import Game, Seat

_CALLS = (Ev.LIAR_RIGHT, Ev.LIAR_WRONG)


# --------------------------------------------------------------------------- ledger

@dataclass
class SeatLedger:
    seat_id: str
    bids: int = 0
    calls: int = 0
    calls_right: int = 0
    #: This seat's bids that were called, by how they came out.
    short: int = 0
    stood: int = 0
    dice_lost: int = 0
    openings: Counter = field(default_factory=Counter)

    def summary(self) -> str:
        bits = [f"{self.bids} bids"]
        if self.calls:
            bits.append(f"called Liar {self.calls}x, right {self.calls_right}")
        if self.dice_lost:
            bits.append(f"lost {self.dice_lost} dice")
        return ", ".join(bits)


def _tells(led: SeatLedger) -> List[str]:
    """Patterns worth saying out loud, as `ledger._tells` -- thresholds, all counted."""
    out: List[str] = []
    if led.openings:
        face, count = led.openings.most_common(1)[0]
        total = sum(led.openings.values())
        if total >= 3 and count == total:
            out.append(f"has opened on {dice.PLURAL[face]} every time ({total})")
    wrong = led.calls - led.calls_right
    if wrong >= 3 and wrong > led.calls_right:
        out.append(f"has called Liar wrongly {wrong} times")
    if led.calls_right >= 3:
        out.append(f"has called {led.calls_right} short bids")
    if led.short >= 2:
        out.append(f"has had {led.short} bids called and found short")
    if led.stood >= 2:
        out.append(f"has had {led.stood} bids called and seen them stand")
    return out


class DiceLedger:
    """The table's record. Lives and dies with the game, like `ledger.Ledger`."""

    def __init__(self, seat_ids: List[str]):
        self.seats: Dict[str, SeatLedger] = {s: SeatLedger(s) for s in seat_ids}
        self.recent: Deque[str] = deque(maxlen=RECENT_KEEP)
        self.turns = 0
        self.winner: Optional[str] = None
        self._opening = True

    def observe(self, events: List[Event], hand_sizes: Dict[str, int],
                names: Dict[str, str]) -> None:
        """`Ledger.observe`'s signature. Dice counts arrive as LOST_DIE instead, so
        `hand_sizes` goes unread."""
        for event in events:
            actor = self.seats.get(event.seat_id or "")
            target = self.seats.get(event.target_id or "")
            who = names.get(event.seat_id or "", "someone")
            whom = names.get(event.target_id or "", "someone")

            if event.kind == Ev.BID and actor:
                actor.bids += 1
                if self._opening:
                    actor.openings[event.face] += 1
                self._opening = False
            elif event.kind in _CALLS and actor:
                right = event.kind == Ev.LIAR_RIGHT
                actor.calls += 1
                actor.calls_right += right
                if target:
                    target.short += right
                    target.stood += not right
                self._opening = True
                self.recent.append(f"{who} called Liar on {whom} and was "
                                   f"{'right' if right else 'wrong'}")
            elif event.kind == Ev.LOST_DIE and actor:
                actor.dice_lost += 1
            elif event.kind == Ev.ELIMINATED:
                self.recent.append(f"{who} was knocked out")
            elif event.kind == Ev.WON:
                self.winner = event.seat_id
        self.turns += 1

    def render(self, names: Dict[str, str]) -> str:
        """The block for `<game_context>`, shaped as `Ledger.render`'s."""
        lines = [f"So far this game ({self.turns} turns played):"]
        for seat_id, led in self.seats.items():
            row = f"- {names.get(seat_id, seat_id)}: {led.summary()}"
            tells = _tells(led)
            if tells:
                row += ". " + "; ".join(tells).capitalize()
            lines.append(row)
        if self.recent:
            lines.append("Recently: " + "; ".join(self.recent) + ".")
        return "\n".join(lines)


# --------------------------------------------------------------------------- narration

REACT_ON = frozenset({*_CALLS, Ev.ELIMINATED})


def _name(game: Game, seat_id: Optional[str]) -> str:
    seat = game.seat(seat_id) if seat_id else None
    return seat.display if seat else "someone"


def describe_event(game: Game, event: Event) -> Optional[str]:
    """One event in plain language, or None for anything nobody would narrate --
    including `BLUFFED`, which nobody may."""
    who, whom = _name(game, event.seat_id), _name(game, event.target_id)
    if event.kind == Ev.BID:
        return f"{who} bid {dice.bid_text(event.amount, event.face)}."
    if event.kind in _CALLS:
        bid = (game.state.reveal or {}).get("bid") or (event.amount, event.face)
        verdict = "short" if event.kind == Ev.LIAR_RIGHT else "good"
        return (f"{who} called Liar on {whom}'s {dice.bid_text(*bid)}. The cups came up "
                f"with {event.amount} -- the bid was {verdict}.")
    if event.kind == Ev.LOST_DIE and event.amount:
        return f"{who} lost a die and has {event.amount} left."
    if event.kind == Ev.ELIMINATED:
        return f"{who} lost their last die and is out."
    if event.kind == Ev.TIMED_OUT:
        return f"{who} ran out of time and was played for."
    return None


def describe_events(game: Game, events: List[Event]) -> List[str]:
    return [line for line in (describe_event(game, e) for e in events) if line]


def beat_for(game: Game, events: List[Event]):
    """The `(seat, description)` this move earned a reaction for, or None.

    Every call settles a round, and a six-hander has twenty-nine of them -- twice the
    reaction budget. So a call earns a beat when it is the game's first, or when it
    leaves the loser on two dice or fewer: the stakes the table actually feels. The
    speaker is whoever lost the die, which `speaker_for` already picks.
    """
    if game.generations >= GAME_REACTION_MAX_CALLS:
        return None
    call = next((e for e in events if e.kind in _CALLS), None)
    lost = next((e for e in events if e.kind == Ev.LOST_DIE), None)
    if call is None or lost is None:
        return None
    if game.state.round_no != 2 and lost.amount > 2:
        return None
    seat: Optional[Seat] = reactor_for(game, call)
    if seat is None:
        return None
    tail = describe_event(game, Event(Ev.ELIMINATED, seat_id=lost.seat_id)
                          if not lost.amount else lost)
    return seat, f"{describe_event(game, call)} {tail}"


def describe_cast(game: Game) -> str:
    counts = {s.seat_id: len(s.hand) for s in game.state.seats}
    rows = []
    for seat in game.seats:
        held = counts.get(seat.seat_id, 0)
        if not held:
            rows.append(f"- {seat.display}: out")
            continue
        mood = neuro.describe(game.neuro.get(seat.seat_id, neuro.BASELINE))
        rows.append(f"- {seat.display}: {held} {'die' if held == 1 else 'dice'}, "
                    f"looks {mood}")
    return "\n".join(rows)


def describe_table(game: Game) -> str:
    view = dice.public_view(game.state)
    bits = [f"Round {view['round_no']}, {view['dice_in_play']} dice in play."]
    if view["bid"]:
        bits.append(f"Standing bid: {dice.bid_text(*view['bid'])}, by "
                    f"{_name(game, view['bidder'])}.")
    elif view["current_seat"]:
        bits.append(f"No bid yet this round; {_name(game, view['current_seat'])} opens.")
    reveal = view["reveal"]
    if reveal:
        cups = "; ".join(f"{_name(game, sid)}: {', '.join(map(str, held))}"
                         for sid, held in reveal["dice"].items())
        bits.append(f"Last cups lifted, for {dice.bid_text(*reveal['bid'])}: {cups}.")
    return " ".join(bits)


def describe_rules(game: Game) -> str:
    """How this table runs, read off the snapshotted rules like `narration.describe_rules`."""
    rules = game.state.rules
    lines = [
        f"- Everyone started with {rules.dice_per_seat} dice under a cup, and every "
        "round everyone rolls in secret.",
        "- On your turn you raise the bid -- more dice, or the same number of a higher "
        "face -- or call Liar on the last bid. A bid counts every cup at the table.",
        "- Ones are wild: they count as every face, so nobody bids on ones."
        if rules.ones_wild else "- Ones are an ordinary face.",
        "- When Liar is called every cup is lifted. If the bid was there the caller "
        "loses a die, otherwise the bidder does. Whoever lost opens the next round.",
        f"- A person at this table has {rules.turn_seconds} seconds to move, and is "
        "played for automatically if the clock beats them. You are not; your turns "
        "are taken for you as soon as they come around.",
        "- Lose your last die and you are out. The last player holding dice wins.",
    ]
    return "\n".join(lines)


def result_line(game: Game) -> str:
    winner = game.seat(game.state.winner) if game.state.winner else None
    return (f"The game is over -- {winner.display} is the last one holding dice."
            if winner else "The game has ended with no winner.")


def finale_beat(game: Game) -> Optional[str]:
    """How the game ended, in one line. None with no winner, as `narration.finale_beat`."""
    winner = game.seat(game.state.winner) if game.state.winner else None
    if winner is None:
        return None
    line = (f"{winner.display} is the last one holding dice and won it, after "
            f"{game.state.round_no} rounds.")
    out = [_name(game, e.seat_id) for e in game.log if e.kind == Ev.ELIMINATED]
    return f"{line} Knocked out, first to last: {', '.join(out)}." if out else line


def is_dramatic(events: List[Event]) -> bool:
    """A call redraws at once: the cups are the moment everyone is waiting for."""
    return any(e.kind in REACT_ON or e.kind == Ev.GAME_OVER for e in events)
