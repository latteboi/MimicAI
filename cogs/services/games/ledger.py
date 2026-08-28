"""What happened at the table so far. Pure: no discord, no async, no I/O.

Deliberately not long-term memory. LTM is embed-summarise-retrieve, and for a bounded
sitting that is wrong three times over: retrieval exists because a corpus outgrows the
context window, and one game does not; retrieving individual memories destroys patterns
that live in a *sequence*, which is exactly where the comedy is ("busted three hands
running"); and this content is a fact table, not prose.

So instead: running aggregates plus a bounded ring of recent moments, rendered straight
into `<game_context>` as standing context. No embedding call, no summarisation call, no
retrieval. Read, format, inject -- and because the engine wrote every number, a
character citing its own record cannot be wrong about it. That is Law I extended from
the rules to the memory.
"""

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from ._shared import Event, Ev

#: Recent moments kept per game. Small on purpose: this is rendered into a prompt on
#: every generation in the channel, and the aggregates below carry the long view.
RECENT_KEEP = 6


@dataclass
class SeatLedger:
    """One seat's record for this sitting."""

    seat_id: str
    plays: int = 0
    cards_drawn: int = 0
    draw_turns: int = 0
    times_hit: int = 0
    cards_taken: int = 0
    attacks_landed: int = 0
    skips_given: int = 0
    skips_taken: int = 0
    reversals: int = 0
    wilds_played: int = 0
    calls_made: int = 0
    calls_missed: int = 0
    penalty_cards: int = 0
    timeouts: int = 0
    biggest_hand: int = 0
    colours_declared: Counter = field(default_factory=Counter)

    def summary(self) -> str:
        bits = [f"{self.plays} cards played"]
        if self.cards_drawn:
            bits.append(f"{self.cards_drawn} drawn")
        if self.times_hit:
            bits.append(f"hit {self.times_hit}x")
        if self.attacks_landed:
            bits.append(f"landed {self.attacks_landed}")
        if self.calls_made:
            bits.append(f"{self.calls_made}x at one card")
        return ", ".join(bits)


def _tells(led: SeatLedger, went_out: bool) -> List[str]:
    """The handful of facts actually worth saying out loud.

    Thresholds rather than everything: a character noting it drew two cards is noise,
    one noting it has drawn twelve is a joke. Each of these is a counter, and each is
    true by construction.
    """
    out: List[str] = []

    if led.colours_declared:
        colour, count = led.colours_declared.most_common(1)[0]
        total = sum(led.colours_declared.values())
        if total >= 3 and count == total:
            out.append(f"has declared {colour} on every single wild ({total})")
        elif total >= 4 and count >= total * 0.75:
            out.append(f"keeps declaring {colour} ({count} of {total} wilds)")

    if led.cards_drawn >= 8:
        out.append(f"has drawn {led.cards_drawn} cards this game")
    if led.times_hit >= 3:
        out.append(f"has been made to pick up {led.times_hit} separate times "
                   f"({led.cards_taken} cards)")
    if led.calls_made >= 2 and not went_out:
        out.append(f"has been down to one card {led.calls_made} times and still not won")
    if led.calls_missed:
        out.append(f"forgot to call Last Card {led.calls_missed}x")
    if led.skips_taken >= 2:
        out.append(f"has been skipped {led.skips_taken} times")
    if led.attacks_landed >= 3:
        out.append(f"has landed {led.attacks_landed} draw cards on other people")
    if led.timeouts >= 2:
        out.append(f"has run down the clock {led.timeouts} times")
    if led.biggest_hand >= 12:
        out.append(f"was holding {led.biggest_hand} cards at the worst of it")
    return out


class Ledger:
    """The whole table's record. Lives and dies with the game."""

    def __init__(self, seat_ids: List[str]):
        self.seats: Dict[str, SeatLedger] = {s: SeatLedger(s) for s in seat_ids}
        self.recent: Deque[str] = deque(maxlen=RECENT_KEEP)
        self.turns = 0
        self.reshuffles = 0
        self.winner: Optional[str] = None

    def get(self, seat_id: str) -> Optional[SeatLedger]:
        return self.seats.get(seat_id)

    def observe(self, events: List[Event], hand_sizes: Dict[str, int],
                names: Dict[str, str]) -> None:
        """Fold one move's events in. `names` maps seat_id to display name, used only
        for the recent-moments strings."""
        for seat_id, size in hand_sizes.items():
            led = self.seats.get(seat_id)
            if led and size > led.biggest_hand:
                led.biggest_hand = size

        for event in events:
            actor = self.seats.get(event.seat_id or "")
            target = self.seats.get(event.target_id or "")
            who = names.get(event.seat_id or "", "someone")
            whom = names.get(event.target_id or "", "someone")

            if event.kind == Ev.PLAYED and actor:
                actor.plays += 1
                if event.card and event.card[0] == "wild":
                    actor.wilds_played += 1
            elif event.kind == Ev.COLOUR_CHOSEN and actor and event.colour:
                actor.colours_declared[event.colour] += 1
            elif event.kind == Ev.DREW and actor:
                actor.draw_turns += 1
                actor.cards_drawn += event.amount
            elif event.kind == Ev.HIT_BY_DRAW and actor:
                actor.times_hit += 1
                actor.cards_taken += event.amount
                actor.cards_drawn += event.amount
                self.recent.append(f"{who} picked up {event.amount}")
            elif event.kind == Ev.STACKED and actor:
                actor.attacks_landed += 1
            elif event.kind == Ev.SKIPPED:
                if actor:
                    actor.skips_given += 1
                if target:
                    target.skips_taken += 1
                    self.recent.append(f"{who} skipped {whom}")
            elif event.kind == Ev.REVERSED and actor:
                actor.reversals += 1
            elif event.kind == Ev.CALL_MADE and actor:
                actor.calls_made += 1
                self.recent.append(f"{who} called Last Card")
            elif event.kind == Ev.CALL_MISSED and actor:
                actor.calls_missed += 1
                self.recent.append(f"{who} forgot to call Last Card")
            elif event.kind == Ev.PENALTY and actor:
                actor.penalty_cards += event.amount
                actor.cards_drawn += event.amount
            elif event.kind == Ev.TIMED_OUT and actor:
                actor.timeouts += 1
            elif event.kind == Ev.RESHUFFLED:
                self.reshuffles += 1
                self.recent.append("the deck ran out and was reshuffled")
            elif event.kind == Ev.WENT_OUT:
                self.winner = event.seat_id
                self.recent.append(f"{who} went out")

        self.turns += 1

    def render(self, names: Dict[str, str]) -> str:
        """The block that goes into `<game_context>`.

        Roughly 150 tokens for a four-hander, and every figure in it was counted by the
        engine -- so a character that brings one up is citing a fact, not inventing a
        detail that happens to sound plausible.
        """
        lines = [f"So far this game ({self.turns} turns played):"]
        for seat_id, led in self.seats.items():
            name = names.get(seat_id, seat_id)
            tells = _tells(led, went_out=(self.winner == seat_id))
            row = f"- {name}: {led.summary()}"
            if tells:
                row += ". " + "; ".join(tells).capitalize()
            lines.append(row)
        if self.recent:
            lines.append("Recently: " + "; ".join(self.recent) + ".")
        return "\n".join(lines)

    def facts(self) -> Dict[str, Any]:
        """Flat view for tests and for a future end-of-game summary."""
        return {
            "turns": self.turns,
            "reshuffles": self.reshuffles,
            "winner": self.winner,
            "seats": {sid: vars(led) for sid, led in self.seats.items()},
        }
