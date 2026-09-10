"""Turning what just happened at the table into something a character can react to.

Pure: every function here takes a `Game` and returns text. It reads public state only
-- the same constraint the status embed is under -- so nothing it produces can leak a
hand into `<game_context>` or into a prompt.

Nothing here generates dialogue or queues a trigger. It says what happened; deciding
who speaks and paying for the model call is `game_service`'s job.
"""

from typing import List, Optional

from ...utils.constants import GAME_REACTION_MAX_CALLS
from . import eights, neuro
from ._shared import Event, Ev, speaker_for
from .table import Game, Seat
from .table_view import _LABEL, card_label


REACT_ON = frozenset({
    Ev.HIT_BY_DRAW, Ev.CALL_MADE, Ev.CALL_MISSED, Ev.WENT_OUT, Ev.PENALTY,
})


def describe_event(game: Game, event: Event) -> Optional[str]:
    """One event in plain language, from public information only.

    Returns None for the mechanical ones nobody would narrate -- a card being drawn
    off the pile, a colour being declared. `<game_context>` is a short window and
    should not spend it on bookkeeping.
    """
    actor = game.seat(event.seat_id) if event.seat_id else None
    target = game.seat(event.target_id) if event.target_id else None
    who = actor.display if actor else "someone"
    whom = target.display if target else "the next player"
    if event.kind == Ev.HIT_BY_DRAW:
        return f"{whom} had to pick up {event.amount} cards."
    if event.kind == Ev.CALL_MADE:
        return f"{who} is down to one card and called Last Card."
    if event.kind == Ev.CALL_MISSED:
        return f"{who} reached one card but forgot to call Last Card."
    if event.kind == Ev.PENALTY and event.amount:
        return f"{who} was penalised {event.amount} cards."
    if event.kind == Ev.SKIPPED:
        return f"{who} skipped {whom}."
    if event.kind == Ev.REVERSED:
        return f"{who} reversed the order of play."
    if event.kind == Ev.STACKED and event.card:
        return (f"{who} played a {_LABEL.get(event.card[1], event.card[1])} on "
                f"{whom}; {event.amount} now pending.")
    if event.kind == Ev.WENT_OUT:
        return f"{who} played their last card and won."
    if event.kind == Ev.TIMED_OUT:
        return f"{who} ran out of time and was played for."
    if event.kind == Ev.RESHUFFLED:
        return "The draw pile ran out and was reshuffled."
    return None


def describe_events(game: Game, events: List[Event]) -> List[str]:
    return [line for line in (describe_event(game, e) for e in events) if line]


def reactor_for(game: Game, event: Event) -> Optional[Seat]:
    """Whose line this is.

    `speaker_for` gives the seat the event happened *to*, which is the interesting
    one, and the actor is the fallback. A human seat is skipped at every step: the
    bot does not get to put words in a player's mouth. It falls through to whoever
    else was involved rather than dropping the beat, so landing a Draw Four on a
    person still gets a reaction -- from the character who threw it, which is the
    funnier half anyway. A beat with no profile on either end passes in silence.
    """
    for candidate in (speaker_for(event), event.seat_id, event.target_id):
        if not candidate:
            continue
        seat = game.seat(candidate)
        if seat is not None and seat.kind == "profile" and seat.owner_id:
            return seat
    return None


def beat_for(game: Game, events: List[Event]):
    """The `(seat, description)` this move earned a reaction for, or None.

    Three gates, cheapest first: the per-game ceiling, then whether anything loud
    actually happened, then whether there is a character available to say it.
    """
    if game.generations >= GAME_REACTION_MAX_CALLS:
        return None
    event = next((e for e in events if e.kind in REACT_ON), None)
    if event is None:
        return None
    seat = reactor_for(game, event)
    if seat is None:
        return None
    beat = describe_event(game, event)
    return (seat, beat) if beat else None


def describe_cast(game: Game) -> str:
    counts = {s["seat_id"]: s["cards"] for s in eights.public_view(game.state)["seats"]}
    rows = []
    for seat in game.seats:
        state = game.neuro.get(seat.seat_id, neuro.BASELINE)
        rows.append(f"- {seat.display}: {counts.get(seat.seat_id, 0)} cards, "
                    f"looks {neuro.describe(state)}")
    return "\n".join(rows)


def describe_ledger(game: Game) -> str:
    """The sitting's record, for a character to draw a barbed line out of.

    Every figure here was counted by the engine, so a character bringing one up is
    citing a fact rather than inventing a plausible-sounding detail. It is also the
    cheapest thing in the payload: counters, not a retrieval.
    """
    if not game.ledger:
        return ""
    return game.ledger.render({s.seat_id: s.display for s in game.seats})


def describe_rules(game: Game) -> str:
    """How this particular table runs, in plain language.

    Read off the snapshotted `RuleSet` rather than written out once, so it is
    always the rules actually in force. `house_rule_summary` answers a different
    question -- what deviates from the default, for a footer that has to fit -- and
    the deviations are the wrong half here: a character needs to know that drawing
    gives it one card even when that is the default, because the alternative is
    guessing at it out loud.
    """
    rules = game.state.rules
    lines = [
        f"- Everyone was dealt {rules.initial_hand} cards. Play passes around the "
        "table; Skip, Reverse and the draw cards do what they say.",
    ]
    if rules.stack_draw_two and rules.stack_draw_four:
        lines.append("- Draw Twos stack onto Draw Twos and Draw Fours onto Draw "
                     "Fours, so a pile can build before someone picks it all up.")
    elif rules.stack_draw_two:
        lines.append("- Draw Twos stack onto Draw Twos, so a pile can build before "
                     "someone picks it all up. Draw Fours do not stack.")
    else:
        lines.append("- Nothing stacks: a draw card is picked up by the next "
                     "player straight away.")
    lines.append(
        "- A Wild Draw Four may only be played by someone with no card of the "
        "active colour." if rules.strict_draw_four else
        "- A Wild Draw Four may be played at any time; nobody is challenged on it.")
    if rules.draw_to_match:
        lines.append("- A player who cannot go keeps drawing until they can.")
    elif rules.play_after_draw:
        lines.append("- A player who cannot go draws one card, and may play that "
                     "card immediately if it fits.")
    else:
        lines.append("- A player who cannot go draws one card and the turn passes.")
    lines.append(
        f"- Going down to one card without calling Last Card costs {rules.miss_penalty} "
        "cards, applied the moment it happens." if rules.auto_call_penalty else
        f"- Going down to one card without calling Last Card can be caught by anyone "
        f"else, for {rules.miss_penalty} cards.")
    lines.append(
        f"- A person at this table has {rules.turn_seconds} seconds to move, and is "
        "played for automatically if the clock beats them. You are not; your turns "
        "are taken for you as soon as they come around.")
    lines.append("- The game ends the moment somebody plays their last card.")
    return "\n".join(lines)


def describe_table(game: Game) -> str:
    view = eights.public_view(game.state)
    return (f"Top card: {card_label(view['top'])}. "
            f"Active colour: {view['active_colour']}. "
            f"Cards left in the pile: {view['draw_pile_size']}.")


def finale_beat(game: Game) -> Optional[str]:
    """How the game ended, in one line, from public information only.

    Returns None for a game with no winner -- an abandoned table has nothing to
    toast, and the cast saying "well, that was that" about nothing is worse than
    the silence it replaces.
    """
    winner = game.seat(game.state.winner) if game.state.winner else None
    if winner is None:
        return None
    counts = {s["seat_id"]: s["cards"] for s in eights.public_view(game.state)["seats"]}
    left = ", ".join(f"{s.display} on {counts.get(s.seat_id, 0)}"
                     for s in game.seats if s.seat_id != winner.seat_id)
    line = (f"{winner.display} played their last card and won it, "
            f"after {game.turns} turns.")
    return f"{line} Everyone else was still holding cards: {left}." if left else line


def is_dramatic(events: List[Event]) -> bool:
    """Whether this move deserves an immediate redraw rather than waiting for the
    end of the lap."""
    loud = {Ev.HIT_BY_DRAW, Ev.CALL_MADE, Ev.CALL_MISSED, Ev.WENT_OUT, Ev.GAME_OVER}
    return any(e.kind in loud for e in events)
