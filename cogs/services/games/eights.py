"""Mimic Eights rules. Pure: no discord, no async, no I/O, no clock.

The public surface is four functions:

    new_game(seat_ids, rules, seed)  -> GameState
    legal_moves(state, seat_id)      -> List[Move]
    apply(state, move)               -> (GameState, List[Event])
    public_view(state)               -> Dict

`public_view` is the leak barrier. It returns card *counts*, never hands, and it is
what feeds both the table embed and `<game_context>` -- so if this one
function is correct, no downstream consumer can leak a hand even by accident. It is
the highest-value test in `tests/test_eights_rules.py` for that reason.

`apply` mutates `state` in place and returns it. Deep-copying a 108-card game per move
would allocate roughly sixty times per game for no benefit: determinism here comes from
the seeded `state.rng`, not from immutability, and the caller owns exactly one game.

Two deliberate simplifications over the printed rules, both to remove a state the UI
would otherwise have to render:

  * The opening upcard is drawn until it is a plain number card. Flipping a Wild first
    means asking a player to choose a colour before anyone has played, which needs a
    whole extra phase for a case that arises 8 times in 108.
  * A wild's colour arrives *with* the move (`Move.declared_colour`), because the UI
    picks a colour before it submits a card. That removes the `choosing_colour` phase
    entirely rather than leaving a phase in the state machine that is never entered.
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ._shared import Card, Event, Ev, IllegalMove

# --------------------------------------------------------------------------- deck

RED, YELLOW, GREEN, BLUE = "red", "yellow", "green", "blue"
COLOURS: Tuple[str, ...] = (RED, YELLOW, GREEN, BLUE)
WILD = "wild"

SKIP, REVERSE, DRAW_TWO = "skip", "reverse", "draw2"
WILD_PLAIN, DRAW_FOUR = "wild", "draw4"

ACTIONS = (SKIP, REVERSE, DRAW_TWO)
NUMBERS = tuple(str(n) for n in range(10))

#: Point values for the optional across-hands scoring the service may layer on top.
CARD_POINTS = {WILD_PLAIN: 50, DRAW_FOUR: 50, SKIP: 20, REVERSE: 20, DRAW_TWO: 20}

DECK_SIZE = 108


def build_deck() -> List[Card]:
    """A fresh 108-card deck in canonical order.

    25 per colour (one 0, two each of 1-9, two each of Skip / Reverse / Draw Two),
    plus four Wild and four Wild Draw Four.
    """
    deck: List[Card] = []
    for colour in COLOURS:
        deck.append((colour, "0"))
        for n in range(1, 10):
            deck.extend([(colour, str(n))] * 2)
        for action in ACTIONS:
            deck.extend([(colour, action)] * 2)
    deck.extend([(WILD, WILD_PLAIN)] * 4)
    deck.extend([(WILD, DRAW_FOUR)] * 4)
    return deck


def is_wild(card: Card) -> bool:
    return card[0] == WILD


def card_points(card: Card) -> int:
    return CARD_POINTS.get(card[1], int(card[1]) if card[1] in NUMBERS else 0)


# --------------------------------------------------------------------------- config

@dataclass(frozen=True)
class RuleSet:
    """House rules, snapshotted into the game at `new_game` so a mid-game settings
    change cannot alter a hand in progress.

    Stacking is same-type only: a Draw Two answers a Draw Two, a Draw Four answers a
    Draw Four. Cross-stacking is a house rule common enough to be argued about and rare
    enough not to be worth a third flag until somebody asks.
    """

    stack_draw_two: bool = True
    stack_draw_four: bool = False
    draw_to_match: bool = False
    play_after_draw: bool = True
    #: On, a Draw Four is only legal with no card of the active colour in hand.
    strict_draw_four: bool = False
    #: On, going to one card without calling is penalised immediately. Off, the seat
    #: stays catchable via `catch_last_call` until it plays again.
    auto_call_penalty: bool = True
    miss_penalty: int = 2
    turn_seconds: int = 45
    seats_max: int = 6
    initial_hand: int = 7


# --------------------------------------------------------------------------- state

@dataclass
class Seat:
    seat_id: str
    kind: str = "profile"          # "human" | "profile"
    hand: List[Card] = field(default_factory=list)
    called_last: bool = False
    drew_this_turn: bool = False
    #: The card just drawn, when only that card may be played this turn.
    drawn_card: Optional[Card] = None
    connected: bool = True


@dataclass
class Move:
    """A single action. Sourced from `legal_moves`; `declared_colour` is filled in by
    the caller for wilds, which is the one field the UI supplies rather than the engine.
    """

    kind: str                                  # "play" | "draw" | "pass"
    card: Optional[Card] = None
    declared_colour: Optional[str] = None
    call_last: bool = False


@dataclass
class GameState:
    seats: List[Seat]
    rules: RuleSet
    rng: random.Random
    turn_idx: int = 0
    direction: int = 1                         # +1 / -1; Reverse flips
    draw_pile: List[Card] = field(default_factory=list)
    discard: List[Card] = field(default_factory=list)
    active_colour: str = RED
    pending_draw: int = 0
    #: What the accumulated draw is made of, so stacking stays same-type.
    pending_kind: Optional[str] = None
    phase: str = "playing"                     # "playing" | "over"
    winner: Optional[str] = None
    turn_no: int = 0
    #: Consecutive seats that could neither play nor draw. A full lap ends the game;
    #: without this a table holding every card between them never terminates.
    passes_in_a_row: int = 0

    # -- small accessors, kept here so callers never index seats by hand ------------
    @property
    def current(self) -> Seat:
        return self.seats[self.turn_idx]

    def seat(self, seat_id: str) -> Seat:
        for s in self.seats:
            if s.seat_id == seat_id:
                return s
        raise KeyError(seat_id)

    @property
    def top(self) -> Card:
        return self.discard[-1]


# --------------------------------------------------------------------------- setup

def new_game(
    seat_ids: List[str],
    rules: Optional[RuleSet] = None,
    seed: Optional[int] = None,
    kinds: Optional[Dict[str, str]] = None,
) -> GameState:
    """Deal a game. `seed` makes the whole thing reproducible, which is what lets the
    tests replay a full game and lets a policy regression be diffed move for move.
    """
    rules = rules or RuleSet()
    if not 2 <= len(seat_ids) <= rules.seats_max:
        raise ValueError(f"Mimic Eights needs 2-{rules.seats_max} seats, got {len(seat_ids)}")
    if len(set(seat_ids)) != len(seat_ids):
        raise ValueError("duplicate seat_id")

    rng = random.Random(seed)
    deck = build_deck()
    rng.shuffle(deck)

    kinds = kinds or {}
    seats = [Seat(seat_id=sid, kind=kinds.get(sid, "profile")) for sid in seat_ids]
    for _ in range(rules.initial_hand):
        for s in seats:
            s.hand.append(deck.pop())

    # Opening upcard: skip past anything with a special effect, so the first seat
    # faces a plain colour-and-number board.
    upcard_at = next(i for i in range(len(deck) - 1, -1, -1) if deck[i][1] in NUMBERS)
    upcard = deck.pop(upcard_at)

    return GameState(
        seats=seats,
        rules=rules,
        rng=rng,
        draw_pile=deck,
        discard=[upcard],
        active_colour=upcard[0],
    )


# --------------------------------------------------------------------------- drawing

def _refill(state: GameState, events: List[Event]) -> bool:
    """Fold the discard back under the top card. Returns False when there is nothing
    left to fold, which is what makes an exhausted table terminate rather than spin.

    A wild keeps its `("wild", …)` identity in the discard -- the declared colour lives
    in `state.active_colour`, not on the card -- so nothing needs resetting here.
    """
    if len(state.discard) <= 1:
        return False
    top = state.discard.pop()
    recycled = state.discard
    state.rng.shuffle(recycled)
    state.draw_pile = recycled
    state.discard = [top]
    events.append(Event(Ev.RESHUFFLED, amount=len(state.draw_pile)))
    return True


def _draw_cards(state: GameState, seat: Seat, n: int, events: List[Event]) -> int:
    """Move up to `n` cards into `seat`. Returns how many were actually dealt, which
    can be short of `n` when the table is holding everything between them.
    """
    dealt = 0
    for _ in range(n):
        if not state.draw_pile and not _refill(state, events):
            break
        seat.hand.append(state.draw_pile.pop())
        dealt += 1
    if dealt:
        # Drawing always breaks a one-card claim.
        seat.called_last = False
    return dealt


# --------------------------------------------------------------------------- legality

def _matches(card: Card, state: GameState) -> bool:
    """Whether `card` may be played onto the current board, ignoring pending draws."""
    if is_wild(card):
        return True
    return card[0] == state.active_colour or card[1] == state.top[1]


def _draw_four_ok(state: GameState, seat: Seat) -> bool:
    """Strict rule: a Draw Four is only legal holding no card of the active colour."""
    if not state.rules.strict_draw_four:
        return True
    return not any(c[0] == state.active_colour for c in seat.hand)


def legal_moves(state: GameState, seat_id: str) -> List[Move]:
    """Every move `seat_id` may make right now. Empty when it is not their turn.

    Wilds come back with `declared_colour=None` -- one move per playable card rather
    than one per card-and-colour pair, so a hand of four wilds offers four moves and
    not sixteen. The caller fills the colour in before passing the move to `apply`.
    """
    if state.phase != "playing" or state.current.seat_id != seat_id:
        return []

    seat = state.current
    moves: List[Move] = []

    # A pending draw suspends normal play: stack the same kind, or absorb it.
    if state.pending_draw:
        stackable = (
            state.rules.stack_draw_two if state.pending_kind == DRAW_TWO
            else state.rules.stack_draw_four
        )
        if stackable:
            for card in _unique(seat.hand):
                if card[1] == state.pending_kind and (
                    card[1] != DRAW_FOUR or _draw_four_ok(state, seat)
                ):
                    moves.append(Move("play", card))
        moves.append(Move("draw"))
        return moves

    # After a draw only the drawn card is playable, and only if the rules allow it.
    if seat.drew_this_turn:
        if (
            state.rules.play_after_draw
            and seat.drawn_card is not None
            and _matches(seat.drawn_card, state)
            and (seat.drawn_card[1] != DRAW_FOUR or _draw_four_ok(state, seat))
        ):
            moves.append(Move("play", seat.drawn_card))
        moves.append(Move("pass"))
        return moves

    for card in _unique(seat.hand):
        if not _matches(card, state):
            continue
        if card[1] == DRAW_FOUR and not _draw_four_ok(state, seat):
            continue
        moves.append(Move("play", card))
    moves.append(Move("draw"))
    return moves


def _unique(hand: List[Card]) -> List[Card]:
    """Distinct cards, in hand order. Two identical cards offer one move, not two."""
    seen, out = set(), []
    for c in hand:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# --------------------------------------------------------------------------- turns

def _advance(state: GameState, steps: int = 1) -> None:
    state.turn_idx = (state.turn_idx + state.direction * steps) % len(state.seats)


def _peek_next(state: GameState) -> Seat:
    return state.seats[(state.turn_idx + state.direction) % len(state.seats)]


def _begin_turn(state: GameState) -> None:
    seat = state.current
    seat.drew_this_turn = False
    seat.drawn_card = None
    state.turn_no += 1


# --------------------------------------------------------------------------- apply

def apply(state: GameState, move: Move) -> Tuple[GameState, List[Event]]:
    """Resolve one move. Mutates and returns `state`, plus everything that happened.

    Raises `IllegalMove` for anything `legal_moves` would not have offered: the UI
    disables what cannot be played, so reaching here with a bad move is a bug rather
    than a user action.
    """
    if state.phase != "playing":
        raise IllegalMove("game is over")

    seat = state.current
    events: List[Event] = []

    if move.kind == "play":
        _apply_play(state, seat, move, events)
        turn_continues = False
    elif move.kind == "draw":
        turn_continues = _apply_draw(state, seat, events)
    elif move.kind == "pass":
        _apply_pass(state, seat, events)
        turn_continues = False
    else:
        raise IllegalMove(f"unknown move kind {move.kind!r}")

    # A seat that drew a card it may still play keeps the turn, and must keep its
    # `drawn_card` with it. Every other outcome hands over, and the seat now sitting
    # at `turn_idx` starts clean -- including when that is the same seat again, which
    # is what a Reverse or a Skip at two seats does.
    if state.phase == "playing" and not turn_continues:
        _begin_turn(state)
    return state, events


def _legal_here(state: GameState, move: Move) -> bool:
    for m in legal_moves(state, state.current.seat_id):
        if m.kind == move.kind and m.card == move.card:
            return True
    return False


def _apply_play(state: GameState, seat: Seat, move: Move, events: List[Event]) -> None:
    card = move.card
    if card is None or not _legal_here(state, move):
        raise IllegalMove(f"{card!r} is not playable")
    if is_wild(card) and move.declared_colour not in COLOURS:
        raise IllegalMove("a wild needs a declared colour")

    wild = is_wild(card)
    colour: str = move.declared_colour if (wild and move.declared_colour) else card[0]

    seat.hand.remove(card)
    state.discard.append(card)
    state.active_colour = colour
    events.append(Event(Ev.PLAYED, seat_id=seat.seat_id, card=card))
    if wild:
        events.append(Event(Ev.COLOUR_CHOSEN, seat_id=seat.seat_id, colour=colour))

    _resolve_call_claim(state, seat, move, events)

    if not seat.hand:
        state.phase = "over"
        state.winner = seat.seat_id
        events.append(Event(Ev.WENT_OUT, seat_id=seat.seat_id))
        events.append(Event(Ev.GAME_OVER, seat_id=seat.seat_id))
        return

    state.passes_in_a_row = 0
    _apply_card_effect(state, seat, card, events)


def _resolve_call_claim(
    state: GameState, seat: Seat, move: Move, events: List[Event]
) -> None:
    """Settle the one-card call for the seat that just played.

    Going out entirely does not require a call -- there is no card left to be caught
    holding -- so this only bites at exactly one card.
    """
    if len(seat.hand) != 1:
        seat.called_last = False
        return
    if move.call_last:
        seat.called_last = True
        events.append(Event(Ev.CALL_MADE, seat_id=seat.seat_id))
    elif state.rules.auto_call_penalty:
        seat.called_last = False
        events.append(Event(Ev.CALL_MISSED, seat_id=seat.seat_id))
        dealt = _draw_cards(state, seat, state.rules.miss_penalty, events)
        events.append(Event(Ev.PENALTY, seat_id=seat.seat_id, amount=dealt))
    else:
        # Catchable until this seat plays again; see `catch_last_call`.
        seat.called_last = False


def _apply_card_effect(
    state: GameState, seat: Seat, card: Card, events: List[Event]
) -> None:
    """Turn advance and the action-card effects, for a play that did not end the game."""
    value = card[1]

    if value == REVERSE:
        # At two seats a Reverse hands the turn straight back, which is a Skip.
        if len(state.seats) == 2:
            events.append(Event(Ev.SKIPPED, seat_id=seat.seat_id,
                                target_id=_peek_next(state).seat_id))
            return
        state.direction *= -1
        events.append(Event(Ev.REVERSED, seat_id=seat.seat_id))
        _advance(state)
        return

    if value == SKIP:
        events.append(Event(Ev.SKIPPED, seat_id=seat.seat_id,
                            target_id=_peek_next(state).seat_id))
        _advance(state, 2)
        return

    if value in (DRAW_TWO, DRAW_FOUR):
        amount = 2 if value == DRAW_TWO else 4
        state.pending_draw += amount
        state.pending_kind = value
        events.append(Event(Ev.STACKED, seat_id=seat.seat_id, card=card,
                            target_id=_peek_next(state).seat_id,
                            amount=state.pending_draw))
        _advance(state)
        return

    _advance(state)


def _apply_draw(state: GameState, seat: Seat, events: List[Event]) -> bool:
    """Voluntary draw, or absorbing an accumulated Draw Two / Draw Four stack.

    Returns True when the seat keeps the turn, which happens only when it drew a card
    it is allowed to play immediately.
    """
    if state.pending_draw:
        amount = state.pending_draw
        dealt = _draw_cards(state, seat, amount, events)
        events.append(Event(Ev.HIT_BY_DRAW, seat_id=seat.seat_id,
                            target_id=seat.seat_id, amount=dealt,
                            colour=state.pending_kind))
        state.pending_draw = 0
        state.pending_kind = None
        state.passes_in_a_row = 0
        _advance(state)              # absorbing costs the turn
        return False

    dealt = _draw_cards(state, seat, 1, events)
    if state.rules.draw_to_match and dealt:
        while not _matches(seat.hand[-1], state):
            extra = _draw_cards(state, seat, 1, events)
            if not extra:
                break
            dealt += extra

    if not dealt:
        # Nothing to draw and nothing playable: the table is holding every card.
        _apply_pass(state, seat, events)
        return False

    events.append(Event(Ev.DREW, seat_id=seat.seat_id, amount=dealt))
    seat.drew_this_turn = True
    seat.drawn_card = seat.hand[-1]
    state.passes_in_a_row = 0

    playable = (
        state.rules.play_after_draw
        and _matches(seat.drawn_card, state)
        and (seat.drawn_card[1] != DRAW_FOUR or _draw_four_ok(state, seat))
    )
    if not playable:
        # A drawn card that cannot be played ends the turn immediately rather than
        # leaving the seat to click a Pass button that has only one outcome.
        _advance(state)
        return False
    return True


def _apply_pass(state: GameState, seat: Seat, events: List[Event]) -> None:
    events.append(Event(Ev.PASSED, seat_id=seat.seat_id))
    state.passes_in_a_row += 1
    if state.passes_in_a_row >= len(state.seats):
        state.phase = "over"
        state.winner = None
        events.append(Event(Ev.GAME_OVER, seat_id=None))
        return
    _advance(state)


# -------------------------------------------------------------------- last-call catch

def catch_last_call(state: GameState, catcher_id: str, target_id: str) -> List[Event]:
    """Penalise a seat sitting on one card without having called it.

    Only live when `auto_call_penalty` is off; with it on the penalty already landed at
    the moment the seat played down to one card.
    """
    if state.phase != "playing" or state.rules.auto_call_penalty:
        return []
    target = state.seat(target_id)
    if len(target.hand) != 1 or target.called_last:
        return []

    events = [Event(Ev.CALL_MISSED, seat_id=target_id, target_id=catcher_id)]
    dealt = _draw_cards(state, target, state.rules.miss_penalty, events)
    events.append(Event(Ev.PENALTY, seat_id=target_id, amount=dealt))
    return events


# --------------------------------------------------------------------------- views

def public_view(state: GameState) -> Dict[str, Any]:
    """Everything a spectator may know. Never hands.

    This is the leak barrier named in Law II of the design spec: the table embed and
    the `<game_context>` block are both built from this and nothing else, so a hand
    cannot reach a prompt even by a caller's mistake. If you add a field here, ask
    first whether it could be used to deduce a specific card.
    """
    return {
        "phase": state.phase,
        "winner": state.winner,
        "turn_no": state.turn_no,
        "top": state.top,
        "active_colour": state.active_colour,
        "direction": state.direction,
        "pending_draw": state.pending_draw,
        "pending_kind": state.pending_kind,
        "current_seat": None if state.phase != "playing" else state.current.seat_id,
        "draw_pile_size": len(state.draw_pile),
        "seats": [
            {
                "seat_id": s.seat_id,
                "kind": s.kind,
                "cards": len(s.hand),
                "called_last": s.called_last,
                "connected": s.connected,
            }
            for s in state.seats
        ],
    }


def private_view(state: GameState, seat_id: str) -> Dict[str, Any]:
    """The public view plus one seat's own hand and legal moves.

    Only `game_service` may call this, and only to build that seat's own decision or
    to render its own ephemeral hand message. It must never reach a session prompt.
    """
    seat = state.seat(seat_id)
    view = public_view(state)
    view["you"] = {
        "seat_id": seat_id,
        "hand": list(seat.hand),
        "legal": legal_moves(state, seat_id),
        "must_call_last": len(seat.hand) == 2,
    }
    return view


def hand_points(state: GameState, seat_id: str) -> int:
    """Standard scoring value of a seat's remaining cards, for across-hands play."""
    return sum(card_points(c) for c in state.seat(seat_id).hand)


def card_count(state: GameState) -> int:
    """Every card in play. Invariant: always `DECK_SIZE`.

    Cheap enough to assert in tests after every move, and it catches the whole class of
    bug where a card is duplicated or dropped by a mishandled reshuffle.
    """
    return (
        len(state.draw_pile)
        + len(state.discard)
        + sum(len(s.hand) for s in state.seats)
    )
