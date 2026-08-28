"""Types shared by every game in this package.

`Event` is the load-bearing one. A game's `apply()` returns a list of these, and three
unrelated subsystems consume them -- the neuro delta table, the ledger, and the beat
that sends a character to react in the channel. Deriving all three from one stream is
what stops them ever disagreeing about what happened at the table.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


class IllegalMove(Exception):
    """Raised when `apply()` is handed a move `legal_moves()` would not have offered.

    Callers are expected to source moves from `legal_moves()`, so this is a programming
    error rather than a user-facing condition -- the UI disables what cannot be played
    rather than letting a click through and catching this.
    """


# (colour, value). Colour is one of COLOURS or "wild"; see eights.COLOURS / eights.WILD.
Card = Tuple[str, str]


class Ev:
    """Event kinds. String constants rather than an enum so they survive a round trip
    through orjson into the ledger without a custom encoder."""

    PLAYED         = "PLAYED"           # a legal card left a hand
    DREW           = "DREW"             # a seat took cards from the pile voluntarily
    RESHUFFLED     = "RESHUFFLED"       # discard folded back into an empty draw pile
    COLOUR_CHOSEN  = "COLOUR_CHOSEN"    # a wild set the active colour
    SKIPPED        = "SKIPPED"          # a seat lost its turn
    REVERSED       = "REVERSED"         # direction flipped
    STACKED        = "STACKED"          # pending_draw grew instead of resolving
    HIT_BY_DRAW    = "HIT_BY_DRAW"      # accumulated draw resolved onto a seat
    CALL_MADE     = "CALL_MADE"       # reached one card and called it
    CALL_MISSED     = "CALL_MISSED"       # reached one card silently, and was caught
    PENALTY        = "PENALTY"          # penalty cards dealt
    PASSED         = "PASSED"           # nothing legal, nothing left to draw
    WENT_OUT       = "WENT_OUT"         # a hand emptied
    GAME_OVER      = "GAME_OVER"        # terminal; carries the winner, or None if stalled

    #: Emitted by the service's turn timer, not by `apply` -- the rules module has no
    #: clock. The move itself still goes through `apply` as a normal one; this rides
    #: alongside so the neuro table can tell it apart from a real choice.
    TIMED_OUT      = "TIMED_OUT"


@dataclass(frozen=True)
class Event:
    """One thing that happened, from the acting seat's point of view.

    `seat_id` is who acted; `target_id` is who it happened to. For HIT_BY_DRAW those
    differ, which is exactly the distinction the neuro table needs -- landing a Draw
    Four and being hit by one move the same axes in opposite directions.
    """

    kind: str
    seat_id: Optional[str] = None
    target_id: Optional[str] = None
    card: Optional[Card] = None
    amount: int = 0
    colour: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """Plain-dict form for the ledger and for prompt payloads."""
        d: Dict[str, Any] = {"kind": self.kind}
        if self.seat_id:   d["seat"] = self.seat_id
        if self.target_id: d["target"] = self.target_id
        if self.card:      d["card"] = f"{self.card[0]}:{self.card[1]}"
        if self.amount:    d["amount"] = self.amount
        if self.colour:    d["colour"] = self.colour
        return d


def speaker_for(event: Event) -> Optional[str]:
    """Which seat this event is *about* -- the one worth hearing from.

    A Skip and a Draw land *on* somebody, and their reaction is the interesting one;
    everything else belongs to whoever acted. This is the whole of what survived the
    canned-bark tier: picking the right character was the only part of it that was
    doing real work.
    """
    if event.kind in (Ev.SKIPPED, Ev.HIT_BY_DRAW):
        return event.target_id or event.seat_id
    return event.seat_id
