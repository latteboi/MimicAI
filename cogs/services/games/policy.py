"""Persona-weighted move selection. Pure: no discord, no async, no I/O, no model call.

This is what makes near-continuous play affordable. A four-hander at thirty turns each
is ninety decisions; asking a model for each would throttle a free-tier key before the
first game finished. It is also what puts personality into *decisions* rather than only
into commentary -- blackjack has a correct play in almost every spot, so character could
only ever be deviation, but Mimic Eights has no solved strategy and a genuine question at every
turn about who to hurt and what to keep.

`choose` takes a `private_view`, never a `GameState`. That is not a convenience: the
view carries this seat's own hand, everyone else's card *counts*, and nothing more, so
the policy is structurally incapable of playing against cards it should not know about.
Law II holds here by construction rather than by discipline.

The loop the design is after closes across three modules: game events move the chemistry
(`neuro.py`), the chemistry reweights the scoring here, the scoring changes what is
actually played, and the play generates the next events.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

from .neuro import BASELINE, PRESETS, Temperament
from .eights import COLOURS, DRAW_FOUR, DRAW_TWO, REVERSE, SKIP, Move, card_points, is_wild

#: Attack cards, and how much raw pressure each applies to the next seat.
_PRESSURE = {DRAW_FOUR: 4.0, DRAW_TWO: 2.5, SKIP: 1.5, REVERSE: 0.8}


def _next_seat(view: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The seat a Skip or Draw would land on, given the current direction."""
    seats = view["seats"]
    ids = [s["seat_id"] for s in seats]
    try:
        here = ids.index(view["current_seat"])
    except (ValueError, TypeError):
        return None
    return seats[(here + view["direction"]) % len(seats)]


def _leader(view: Dict[str, Any], exclude: str) -> Optional[Dict[str, Any]]:
    """Whoever is closest to going out, ignoring this seat."""
    others = [s for s in view["seats"] if s["seat_id"] != exclude]
    return min(others, key=lambda s: s["cards"]) if others else None


def _colour_strength(hand: List[Tuple[str, str]]) -> Dict[str, int]:
    counts = {c: 0 for c in COLOURS}
    for colour, _ in hand:
        if colour in counts:
            counts[colour] += 1
    return counts


def score_move(
    move: Move,
    view: Dict[str, Any],
    neuro: Dict[str, int],
    temperament: Temperament,
) -> float:
    """How much this seat wants to make `move`, right now, feeling how it feels.

    Deterministic. `choose` adds the only randomness, as a small jitter, so that two
    equally-rated plays do not always resolve the same way.
    """
    hand = view["you"]["hand"]
    adrenaline = neuro["adrenaline"] / 100.0
    cortisol = neuro["cortisol"] / 100.0
    dopamine = neuro["dopamine"] / 100.0

    if move.kind == "pass":
        return -10.0
    if move.kind == "draw":
        # Drawing is the fallback, but a stack big enough to hurt makes it worse than
        # merely dull -- which is what pushes a wired profile to stack back instead.
        return -5.0 - view["pending_draw"] * 0.4

    card = move.card
    if card is None:
        return -10.0
    colour, value = card

    # -- shed weight ---------------------------------------------------------------
    # High-value cards are dead weight if the game ends while they are still in hand,
    # and action cards are worth more played than held.
    #
    # Wilds are exempt. They score 50 apiece, so counting them here would swamp the
    # hoarding term below and every profile would dump its wilds on sight -- but a
    # wild's real worth is that it is always playable, and that outlives its point
    # cost until the endgame. Flexibility is priced by `hoarding`, not here.
    score = 1.0 + (0.0 if is_wild(card) else card_points(card) * 0.05)

    # -- aggression ----------------------------------------------------------------
    # The constant makes low `aggression` read as distaste rather than indifference.
    # Without it the shed-weight term above -- an attack card is worth 20 points, so
    # everyone wants rid of it -- sets a floor that swamps the weights, and a meek
    # profile plays exactly as much Draw Two as a reckless one. A character who would
    # rather not hurt anybody has to be able to score the option below zero.
    pressure = _PRESSURE.get(value, 0.0)
    if pressure:
        appetite = temperament.aggression * (0.35 + adrenaline) - 0.5
        score += pressure * appetite

        victim = _next_seat(view)
        if victim:
            # Hitting someone who is nearly out is worth more than hitting someone
            # sitting on ten cards, to everyone -- and much more to the spiteful.
            closeness = max(0.0, 4.0 - victim["cards"]) / 4.0
            score += pressure * closeness * 0.6
            leader = _leader(view, view["current_seat"])
            if leader and victim["seat_id"] == leader["seat_id"]:
                score += pressure * temperament.spite * 0.5
            if victim["called_last"]:
                score += pressure * (1.0 + temperament.spite)

    # -- safety --------------------------------------------------------------------
    # Prefer a play that leaves a follow-up in the same colour. Cortisol sharpens this
    # into outright defensiveness: a rattled profile dumps whatever keeps it safe.
    after = [c for c in hand if c != card]
    new_colour = colour if not is_wild(card) else _best_colour(after)
    follow_ups = sum(1 for c in after if c[0] == new_colour or is_wild(c))
    score += min(follow_ups, 3) * 0.5 * temperament.caution * (0.3 + cortisol)

    # -- hoarding ------------------------------------------------------------------
    # Wilds are the only cards that are always playable, so spending one early costs
    # future flexibility. Going badly (low dopamine) makes a profile cling harder.
    if is_wild(card):
        score -= 2.0 * temperament.hoarding * (1.4 - dopamine)
        if len(hand) <= 2:
            score += 6.0          # no future left to save it for

    # -- closing -------------------------------------------------------------------
    if len(hand) == 1:
        score += 25.0             # this play wins

    return score


def _best_colour(hand: List[Tuple[str, str]]) -> str:
    strength = _colour_strength(hand)
    return max(COLOURS, key=lambda c: strength[c])


def choose_colour(
    view: Dict[str, Any],
    neuro: Dict[str, int],
    temperament: Temperament,
    rng: random.Random,
) -> str:
    """Which colour to declare on a wild.

    Own-majority, because the alternative -- guessing what an opponent is short of --
    would need information this seat does not have and must not be given. A rattled
    profile is likelier to just repeat the colour already showing, which is the small
    tell that makes a ledger line like "declared blue on every wild" worth writing.
    """
    hand = [c for c in view["you"]["hand"] if not is_wild(c)]
    strength = _colour_strength(hand)
    best = max(strength.values()) if strength else 0

    # Under stress, take the path of least thought and name the colour already showing.
    # Same excess-over-resting shape as `will_call_last`: it is being rattled that makes
    # someone default, not ordinary alertness.
    rest = BASELINE["cortisol"]
    excess = max(0, neuro["cortisol"] - rest) / float(100 - rest)
    stick = min(0.5, excess * 0.5 * temperament.fluster)

    if not best:
        if rng.random() < stick:
            return view["active_colour"]
        return rng.choice(COLOURS)

    tied = [c for c in COLOURS if strength[c] == best]
    if view["active_colour"] in tied and rng.random() < stick:
        return view["active_colour"]
    return rng.choice(tied)


def will_call_last(
    neuro: Dict[str, int], temperament: Temperament, rng: random.Random
) -> bool:
    """Whether the seat remembers to call at one card.

    Only cortisol *above* resting level causes a slip. Ordinary alertness is not what
    makes people forget -- being rattled is -- so a calm profile calls every time and
    the lapse stays legible as a symptom rather than as background noise. Capped at 40%
    so it remains a hazard rather than a handicap.

    It is also the source of some of the better moments at the table: a character too
    flustered to call, then indignant about the penalty, is doing something no scripted
    bot does.
    """
    rest = BASELINE["cortisol"]
    excess = max(0, neuro["cortisol"] - rest) / float(100 - rest)
    slip = min(0.4, excess * 0.45 * temperament.fluster)
    return rng.random() >= slip


def choose(
    view: Dict[str, Any],
    neuro: Optional[Dict[str, int]] = None,
    temperament: Optional[Temperament] = None,
    rng: Optional[random.Random] = None,
    jitter: float = 0.35,
) -> Move:
    """Pick this seat's move from its own private view of the table.

    `view` must come from `eights.private_view`. Raises ValueError if handed a public view,
    because a policy that silently played at random when it lost sight of its own hand
    would be a very quiet bug.
    """
    if "you" not in view:
        raise ValueError("choose() needs a private_view, not a public one")

    legal: List[Move] = view["you"]["legal"]
    if not legal:
        raise ValueError(f"no legal moves for {view['you']['seat_id']}")

    rng = rng or random.Random()
    temperament = temperament or PRESETS["steady"]
    neuro = neuro or {"dopamine": 50, "cortisol": 20, "oxytocin": 50, "adrenaline": 20}

    best, best_score = legal[0], float("-inf")
    for move in legal:
        score = score_move(move, view, neuro, temperament)
        score += rng.uniform(-jitter, jitter)
        if score > best_score:
            best, best_score = move, score

    hand = view["you"]["hand"]
    declared = None
    if best.kind == "play" and best.card is not None and is_wild(best.card):
        declared = choose_colour(view, neuro, temperament, rng)

    # Calling is only meaningful on the play that takes this seat to exactly one card.
    calling = False
    if best.kind == "play" and len(hand) == 2:
        calling = will_call_last(neuro, temperament, rng)

    return Move(best.kind, best.card, declared, call_last=calling)
