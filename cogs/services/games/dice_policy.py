"""Mimic Dice move selection. Pure: no discord, no async, no I/O, no model call.

Takes a `private_view`, like `policy.choose`, so it can only reason from its own cup
and the bid on the table. Every option is priced by the one number that matters -- the
chance it costs a die -- and temperament and chemistry tilt the prices:

  * aggression, amplified by adrenaline: appetite for calling Liar, and for bluffing.
  * caution, amplified by cortisol: how much a shaky raise is feared, and how safe an
    opening bid has to be.
  * spite: extra appetite for calling the seat holding the most dice.
  * fluster: cortisol above resting becomes noise, so a rattled seat misjudges.

Liar's Dice has no solved line either: whether to push a bid or call it is a judgement
about odds, which is exactly where a temperament can show without playing badly.
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple

from .dice import Move, matches
from .neuro import BASELINE, PRESETS, Temperament


def _tail(n: int, p: float) -> List[float]:
    """`tail[k]` = P(at least k of `n` unseen dice match), for k in 0..n+1.

    Built once per decision, so pricing every face is a lookup rather than a binomial
    sum per option -- this runs on the event loop.
    """
    tail = [0.0] * (n + 2)
    for k in range(n, -1, -1):
        tail[k] = tail[k + 1] + math.comb(n, k) * p ** k * (1 - p) ** (n - k)
    return tail


def _leader(view: Dict[str, Any], exclude: str) -> Optional[str]:
    """Whoever else holds the most dice."""
    others = [s for s in view["seats"] if s["seat_id"] != exclude and s["dice"]]
    return max(others, key=lambda s: s["dice"])["seat_id"] if others else None


def choose(
    view: Dict[str, Any],
    neuro: Optional[Dict[str, int]] = None,
    temperament: Optional[Temperament] = None,
    rng: Optional[random.Random] = None,
    jitter: float = 0.05,
) -> Move:
    """Pick this seat's move from its own private view. Raises ValueError on a public
    view, for the same reason `policy.choose` does."""
    if "you" not in view:
        raise ValueError("choose() needs a private_view, not a public one")
    legal: List[Move] = view["you"]["legal"]
    if not legal:
        raise ValueError(f"no legal moves for {view['you']['seat_id']}")

    rng = rng or random.Random()
    temperament = temperament or PRESETS["steady"]
    neuro = neuro or dict(BASELINE)

    hand = view["you"]["hand"]
    wild = view["ones_wild"]
    tail = _tail(view["dice_in_play"] - len(hand), 1 / 3 if wild else 1 / 6)

    def held(face: int) -> int:
        return sum(1 for d in hand if matches(d, face, wild))

    def p_true(quantity: int, face: int) -> float:
        need = quantity - held(face)
        if need <= 0:
            return 1.0
        return tail[need] if need < len(tail) else 0.0

    adrenaline = neuro["adrenaline"] / 100.0
    cortisol = neuro["cortisol"] / 100.0
    rest = BASELINE["cortisol"]
    excess = max(0, neuro["cortisol"] - rest) / float(100 - rest)
    # Same shape as `policy.score_move`'s: below zero reads as distaste, not indifference.
    appetite = temperament.aggression * (0.35 + adrenaline) - 0.5
    fear = temperament.caution * (0.3 + cortisol)
    noise = jitter + 0.15 * excess * temperament.fluster
    opening = view["bid"] is None
    #: How sure an opening bid must be. Without it every round opens on "1 two" and
    #: crawls up a face at a time; with it a round is a handful of bids, then a call.
    confidence = min(0.95, max(0.55, 0.75 + 0.1 * fear - 0.15 * appetite))
    # Caution prices raises and appetite prices calls, and never the other way: a bold
    # seat that also found raises cheap would cancel its own appetite for calling.
    exposure = 0.7 + 0.2 * fear

    # One candidate per face: the cheapest raise, or when opening, the boldest bid this
    # seat believes. `legal` is in face-then-quantity order, which is what makes the
    # last overwrite the highest quantity still above `confidence`.
    bids: Dict[int, Move] = {}
    for m in legal:
        if m.kind == "bid" and (m.face not in bids or (
                opening and p_true(m.quantity, m.face) >= confidence)):
            bids[m.face] = m

    priced: List[Tuple[float, Move]] = []
    for m in bids.values():
        risk = (1 - p_true(m.quantity, m.face)) * exposure
        if opening:
            risk -= 0.02 * m.quantity
        priced.append((risk + rng.uniform(-noise, noise), m))
    if view["bid"] is not None:
        risk = p_true(*view["bid"]) - 0.5 * appetite
        if view["bidder"] == _leader(view, view["you"]["seat_id"]):
            risk -= 0.05 * temperament.spite
        priced.append((risk + rng.uniform(-noise, noise), Move("liar")))

    priced.sort(key=lambda rm: rm[0])
    best_risk, best = priced[0]

    # A bluff: a bid on a face this seat holds none of, taken when it is nearly as safe
    # as the honest choice. This is what feeds the tell -- see `dice.py`.
    if best.kind == "bid" and rng.random() < min(0.5, max(0.0, 0.15 + 0.35 * appetite)):
        blind = [m for r, m in priced
                 if m.kind == "bid" and not held(m.face) and r <= best_risk + 0.2]
        if blind:
            return blind[0]
    return best
