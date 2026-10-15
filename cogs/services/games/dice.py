"""Mimic Dice rules -- Liar's Dice. Pure: no discord, no async, no I/O, no clock.

The same surface as `eights.py`, so `game_service` drives either without asking which:

    new_game(seat_ids, rules, seed, kinds) -> GameState
    legal_moves(state, seat_id)            -> List[Move]
    apply(state, move)                     -> (GameState, List[Event])
    public_view(state) / private_view(state, seat_id)

Everyone rolls under a cup in secret. On your turn you raise the bid -- how many of one
face lie under *every* cup put together -- or call Liar on the last one. A call lifts
the cups, the wrong side loses a die, and the last seat holding dice wins.

Two simplifications over Perudo, each removing a rule the panel would have to explain:

  * With ones wild, ones cannot be bid. Perudo allows it at half the quantity, which is
    the rule every new player gets wrong.
  * No "spot on" (calling a bid exactly right): a third button for a call that is
    almost never correct.

**`public_view` is the leak barrier**, as in Eights, with one hole made on purpose: a
profile seat's bid on a face it holds none of emits `Ev.BLUFFED`, which `neuro` turns
into stress, so the mood word on the table is that character's poker tell. The ledger
and narration never mention the event, so the mood is the whole tell. A human seat never
emits it: the bot does not get to give a person away.
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ._shared import Event, Ev, IllegalMove

FACES: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
#: ⚀ to ⚅, indexed by face - 1. Text only: Discord rejects these as component emoji.
GLYPHS = "⚀⚁⚂⚃⚄⚅"
_SINGULAR = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
PLURAL = {1: "ones", 2: "twos", 3: "threes", 4: "fours", 5: "fives", 6: "sixes"}

Bid = Tuple[int, int]   # (quantity, face)


def bid_text(quantity: int, face: int) -> str:
    """'4 fives', '1 six' -- how a bid is written everywhere, table and prompt alike."""
    return f"{quantity} {_SINGULAR[face] if quantity == 1 else PLURAL[face]}"


@dataclass(frozen=True)
class RuleSet:
    """House rules, snapshotted at `new_game` so a mid-game change cannot reach it."""

    dice_per_seat: int = 5
    #: Ones count toward every face, and so cannot be bid themselves.
    ones_wild: bool = True
    turn_seconds: int = 45
    seats_max: int = 6


@dataclass
class Seat:
    seat_id: str
    kind: str = "profile"           # "human" | "profile"
    #: The dice under this seat's cup. Named as Eights names a seat's cards, so the
    #: service counts a holding the same way for either game.
    hand: List[int] = field(default_factory=list)

    @property
    def out(self) -> bool:
        return not self.hand


@dataclass
class Move:
    kind: str                       # "bid" | "liar"
    quantity: int = 0
    face: int = 0


@dataclass
class GameState:
    seats: List[Seat]
    rules: RuleSet
    rng: random.Random
    turn_idx: int = 0
    bid: Optional[Bid] = None
    bidder: Optional[str] = None
    phase: str = "playing"          # "playing" | "over"
    winner: Optional[str] = None
    turn_no: int = 0
    round_no: int = 1
    #: The cups the last call lifted: every seat's dice, the bid, the count and who
    #: lost. Public from the moment it is set -- those dice were shown, then re-rolled.
    reveal: Optional[Dict[str, Any]] = None

    @property
    def current(self) -> Seat:
        return self.seats[self.turn_idx]

    def seat(self, seat_id: str) -> Seat:
        for s in self.seats:
            if s.seat_id == seat_id:
                return s
        raise KeyError(seat_id)

    @property
    def dice_in_play(self) -> int:
        return sum(len(s.hand) for s in self.seats)


# --------------------------------------------------------------------------- setup

def new_game(
    seat_ids: List[str],
    rules: Optional[RuleSet] = None,
    seed: Optional[int] = None,
    kinds: Optional[Dict[str, str]] = None,
) -> GameState:
    rules = rules or RuleSet()
    if not 2 <= len(seat_ids) <= rules.seats_max:
        raise ValueError(f"Mimic Dice needs 2-{rules.seats_max} seats, got {len(seat_ids)}")
    if len(set(seat_ids)) != len(seat_ids):
        raise ValueError("duplicate seat_id")
    kinds = kinds or {}
    seats = [Seat(sid, kinds.get(sid, "profile"), [0] * rules.dice_per_seat)
             for sid in seat_ids]
    state = GameState(seats=seats, rules=rules, rng=random.Random(seed))
    _roll(state)
    return state


def _roll(state: GameState) -> None:
    for s in state.seats:
        s.hand = sorted(state.rng.randint(1, 6) for _ in s.hand)


# --------------------------------------------------------------------------- legality

def matches(die: int, face: int, ones_wild: bool) -> bool:
    """Whether `die` counts toward a bid on `face`."""
    return die == face or (ones_wild and die == 1)


def biddable_faces(rules: RuleSet) -> Tuple[int, ...]:
    return FACES[1:] if rules.ones_wild else FACES


def min_quantity(bid: Optional[Bid], face: int) -> int:
    """The smallest legal bid on `face`: the same quantity on a higher face, else one more."""
    if bid is None:
        return 1
    quantity, current = bid
    return quantity if face > current else quantity + 1


def legal_moves(state: GameState, seat_id: str) -> List[Move]:
    """Every move `seat_id` may make now, bids in face-then-quantity order. Empty when it
    is not their turn.

    Quantities stop at the dice in play -- a bid above that can only be false -- so the
    list is bounded at 30 x 6 and is never empty: at the top bid only Liar is left.
    """
    if state.phase != "playing" or state.current.seat_id != seat_id:
        return []
    total = state.dice_in_play
    moves = [Move("bid", q, f)
             for f in biddable_faces(state.rules)
             for q in range(min_quantity(state.bid, f), total + 1)]
    if state.bid is not None:
        moves.append(Move("liar"))
    return moves


def _legal(state: GameState, move: Move) -> bool:
    if move.kind == "liar":
        return state.bid is not None
    if move.kind != "bid" or move.face not in biddable_faces(state.rules):
        return False
    return min_quantity(state.bid, move.face) <= move.quantity <= state.dice_in_play


def _next_in(state: GameState, idx: int) -> int:
    """The next seat after `idx` still holding dice."""
    n = len(state.seats)
    for step in range(1, n + 1):
        j = (idx + step) % n
        if not state.seats[j].out:
            return j
    return idx


# --------------------------------------------------------------------------- apply

def apply(state: GameState, move: Move) -> Tuple[GameState, List[Event]]:
    """Resolve one move. Mutates and returns `state`, plus everything that happened.

    Raises `IllegalMove` for anything `legal_moves` would not have offered.
    """
    if state.phase != "playing":
        raise IllegalMove("game is over")
    if not _legal(state, move):
        raise IllegalMove(f"{move!r} is not legal here")

    seat = state.current
    events: List[Event] = []
    state.turn_no += 1

    if move.kind == "liar":
        _call(state, seat, events)
        return state, events

    state.bid = (move.quantity, move.face)
    state.bidder = seat.seat_id
    events.append(Event(Ev.BID, seat_id=seat.seat_id, amount=move.quantity, face=move.face))
    if seat.kind == "profile" and not any(
            matches(d, move.face, state.rules.ones_wild) for d in seat.hand):
        events.append(Event(Ev.BLUFFED, seat_id=seat.seat_id))
    state.turn_idx = _next_in(state, state.turn_idx)
    return state, events


def _call(state: GameState, caller: Seat, events: List[Event]) -> None:
    """Lift the cups on the standing bid; the wrong side loses a die and opens next."""
    assert state.bid is not None and state.bidder is not None   # `_legal` saw to it
    quantity, face = state.bid
    bidder = state.seat(state.bidder)
    found = sum(matches(d, face, state.rules.ones_wild)
                for s in state.seats for d in s.hand)
    stood = found >= quantity
    loser = caller if stood else bidder

    events.append(Event(Ev.LIAR_WRONG if stood else Ev.LIAR_RIGHT,
                        seat_id=caller.seat_id, target_id=bidder.seat_id,
                        amount=found, face=face))
    state.reveal = {
        "dice": {s.seat_id: list(s.hand) for s in state.seats if s.hand},
        "bid": state.bid, "bidder": bidder.seat_id, "caller": caller.seat_id,
        "found": found, "loser": loser.seat_id,
    }

    loser.hand.pop()
    events.append(Event(Ev.LOST_DIE, seat_id=loser.seat_id, amount=len(loser.hand)))
    if loser.out:
        events.append(Event(Ev.ELIMINATED, seat_id=loser.seat_id))

    standing = [s for s in state.seats if not s.out]
    if len(standing) == 1:
        state.phase = "over"
        state.winner = standing[0].seat_id
        events.append(Event(Ev.WON, seat_id=state.winner))
        events.append(Event(Ev.GAME_OVER, seat_id=state.winner))
        return

    state.bid = state.bidder = None
    state.round_no += 1
    idx = state.seats.index(loser)
    state.turn_idx = idx if not loser.out else _next_in(state, idx)
    _roll(state)


# --------------------------------------------------------------------------- views

def public_view(state: GameState) -> Dict[str, Any]:
    """Everything a spectator may know: dice *counts*, never a cup this round.

    `reveal` holds dice, but only the last call's, which were shown to the table and
    then re-rolled. If you add a field here, ask whether it narrows what is under a cup.
    """
    return {
        "phase": state.phase,
        "winner": state.winner,
        "turn_no": state.turn_no,
        "round_no": state.round_no,
        "bid": state.bid,
        "bidder": state.bidder,
        "current_seat": None if state.phase != "playing" else state.current.seat_id,
        "dice_in_play": state.dice_in_play,
        "ones_wild": state.rules.ones_wild,
        "reveal": state.reveal,
        "seats": [
            {"seat_id": s.seat_id, "kind": s.kind, "dice": len(s.hand)}
            for s in state.seats
        ],
    }


def private_view(state: GameState, seat_id: str) -> Dict[str, Any]:
    """The public view plus one seat's own cup and legal moves. Same rule as Eights':
    only `game_service` calls it, for the policy or that seat's own panel."""
    view = public_view(state)
    view["you"] = {
        "seat_id": seat_id,
        "hand": list(state.seat(seat_id).hand),
        "legal": legal_moves(state, seat_id),
    }
    return view
