"""Game events to neuro-endocrine movement. Pure: no discord, no async, no I/O.

The existing engine runs the other way round. `DEFAULT_NEURO_INSTRUCTION` hands the
model its current D/C/O/A and asks it to emit a `<neuro_update>` block, which
`prompt_builder` parses back out -- so the *model* decides how it feels. That is fine
for conversation, where there is nothing objective to react to, but it breaks twice in
a game: a profile losing badly can simply decline to be stressed, and most turns make
no model call at all, so the state would sit stale for the whole hand.

So during a game the direction inverts. The engine writes the deltas from what actually
happened at the table, and the model only ever reads them. This is the ground truth the
neuro engine has never had.

State is the same plain dict shape `profile_manager` already persists
(`{"dopamine": 50, "cortisol": 20, "oxytocin": 50, "adrenaline": 20}`), so nothing needs
converting at the boundary.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from ._shared import Event, Ev

AXES = ("dopamine", "cortisol", "oxytocin", "adrenaline")

#: Matches the defaults written by `profile_manager` for a new profile.
BASELINE: Dict[str, int] = {
    "dopamine": 50, "cortisol": 20, "oxytocin": 50, "adrenaline": 20,
}

#: Fraction of the distance back to baseline surrendered at each lap of the table.
#: Without this a rough game leaves a profile pinned at cortisol 100 permanently, and
#: it then carries that into every ordinary conversation it has afterwards.
DECAY_RATE = 0.12


@dataclass(frozen=True)
class Delta:
    d: int = 0
    c: int = 0
    o: int = 0
    a: int = 0

    def scaled(self, t: "Temperament") -> "Delta":
        return Delta(
            round(self.d * t.dopamine), round(self.c * t.cortisol),
            round(self.o * t.oxytocin), round(self.a * t.adrenaline),
        )


@dataclass(frozen=True)
class Temperament:
    """Per-profile tuning for both halves of the loop.

    The first four fields scale how hard events land; the rest weight how the resulting
    chemistry pushes the play policy. Keeping them in one object rather than two
    parallel dicts is deliberate -- they are one authored personality, and a profile
    that feels things keenly but plays placidly is a bug in the authoring, not a
    combination worth making easy.
    """

    name: str = "steady"

    # -- how hard events land ------------------------------------------------------
    dopamine: float = 1.0
    cortisol: float = 1.0
    oxytocin: float = 1.0
    adrenaline: float = 1.0

    # -- how the chemistry steers play ---------------------------------------------
    #: Pull toward attack cards, amplified by adrenaline.
    aggression: float = 1.0
    #: Pull toward plays that leave a follow-up, amplified by cortisol.
    caution: float = 1.0
    #: Reluctance to spend a wild while things are going well.
    hoarding: float = 1.0
    #: Appetite for aiming attacks at whoever is closest to going out.
    spite: float = 0.0
    #: How readily stress makes them forget to call at one card.
    fluster: float = 1.0


PRESETS: Dict[str, Temperament] = {
    "steady": Temperament("steady"),
    "reckless": Temperament(
        "reckless", dopamine=1.3, cortisol=0.6, adrenaline=1.4,
        aggression=1.8, caution=0.4, hoarding=0.3, spite=0.4, fluster=0.7),
    "anxious": Temperament(
        "anxious", cortisol=1.6, adrenaline=1.2, oxytocin=1.2,
        aggression=0.5, caution=1.9, hoarding=1.6, spite=0.0, fluster=1.8),
    "vindictive": Temperament(
        "vindictive", dopamine=1.1, cortisol=0.9, oxytocin=0.6,
        aggression=1.5, caution=0.9, hoarding=0.8, spite=2.2, fluster=0.9),
    "meek": Temperament(
        "meek", dopamine=0.8, cortisol=1.3, oxytocin=1.4,
        aggression=0.2, caution=1.5, hoarding=1.3, spite=0.0, fluster=1.4),
}


# --------------------------------------------------------------------------- table

#: kind -> scope -> delta. Scopes are "actor" (who did it), "target" (who it happened
#: to) and "others" (everyone at the table except the actor).
#:
#: Two events are deliberately absent. PENALTY always accompanies CALL_MISSED and
#: GAME_OVER always accompanies WENT_OUT, so mapping both halves of either pair would
#: count one moment twice.
_TABLE: Dict[str, Dict[str, Delta]] = {
    Ev.PLAYED:     {"actor": Delta(d=1, c=-1)},
    Ev.DREW:       {"actor": Delta(d=-3, c=4, a=2)},
    Ev.PASSED:     {"actor": Delta(d=-2, c=3)},
    Ev.SKIPPED:    {"actor": Delta(d=4, a=3), "target": Delta(d=-5, c=6, o=-3, a=3)},
    Ev.REVERSED:   {"actor": Delta(d=3, a=2)},
    Ev.CALL_MADE: {"actor": Delta(d=15, c=8, a=25),
                    "others": Delta(d=-4, c=12, a=12)},
    Ev.CALL_MISSED: {"actor": Delta(d=-12, c=15, o=-5, a=5)},
    Ev.WENT_OUT:   {"actor": Delta(d=35, c=-25, o=5, a=-10),
                    "others": Delta(d=-10, c=8, a=-8)},
    Ev.TIMED_OUT:  {"actor": Delta(d=-4, c=5)},
    Ev.RESHUFFLED: {},
    Ev.COLOUR_CHOSEN: {},
    Ev.PENALTY:    {},
    Ev.GAME_OVER:  {},
}

#: HIT_BY_DRAW and STACKED both scale with what was played, so they resolve from the
#: event payload rather than sitting flat in the table above.
_HIT = {
    "draw2": Delta(d=-8, c=10, o=-5, a=8),
    "draw4": Delta(d=-15, c=18, o=-10, a=14),
}
_LANDED = {
    "draw2": Delta(d=7, c=-2, o=-3, a=6),
    "draw4": Delta(d=12, c=-4, o=-6, a=10),
}
#: What the seat about to absorb a stack feels while it is still pending.
_INCOMING = Delta(c=3, a=4)


def deltas_for(event: Event) -> Dict[str, Delta]:
    """Unscaled movement this event causes, by scope."""
    if event.kind == Ev.HIT_BY_DRAW:
        return {"actor": _HIT.get(event.colour or "draw2", _HIT["draw2"])}
    if event.kind == Ev.STACKED:
        value = event.card[1] if event.card else "draw2"
        return {"actor": _LANDED.get(value, _LANDED["draw2"]), "target": _INCOMING}
    return _TABLE.get(event.kind, {})


# --------------------------------------------------------------------------- apply

def default_state() -> Dict[str, int]:
    return dict(BASELINE)


def _clamp(value: float) -> int:
    return max(0, min(100, int(round(value))))


def apply_events(
    states: Dict[str, Dict[str, int]],
    events: Iterable[Event],
    seat_ids: List[str],
    temperaments: Optional[Dict[str, Temperament]] = None,
) -> Dict[str, Dict[str, int]]:
    """Fold a batch of events into per-seat neuro state.

    Mutates `states` in place and returns the net movement per seat -- the change only,
    not the resulting values -- so a reaction can say a character's stress spiked
    without having to diff the state itself.
    """
    temperaments = temperaments or {}
    moved: Dict[str, Dict[str, int]] = {}

    for event in events:
        scopes = deltas_for(event)
        if not scopes:
            continue
        for scope, delta in scopes.items():
            if scope == "actor":
                targets = [event.seat_id] if event.seat_id else []
            elif scope == "target":
                targets = [event.target_id] if event.target_id else []
            else:
                targets = [s for s in seat_ids if s != event.seat_id]

            for seat_id in targets:
                if seat_id not in states:
                    continue
                temperament = temperaments.get(seat_id) or PRESETS["steady"]
                scaled = delta.scaled(temperament)
                state = states[seat_id]
                bucket = moved.setdefault(seat_id, {a: 0 for a in AXES})
                for axis, amount in (
                    ("dopamine", scaled.d), ("cortisol", scaled.c),
                    ("oxytocin", scaled.o), ("adrenaline", scaled.a),
                ):
                    if not amount:
                        continue
                    before = state[axis]
                    state[axis] = _clamp(before + amount)
                    bucket[axis] += state[axis] - before

    return moved


def decay(
    states: Dict[str, Dict[str, int]],
    baselines: Optional[Dict[str, Dict[str, int]]] = None,
    rate: float = DECAY_RATE,
) -> None:
    """Surrender part of the distance back to each profile's resting state.

    Called once per lap of the table rather than per event, so a fast exchange keeps
    its intensity and a long grind settles.
    """
    baselines = baselines or {}
    for seat_id, state in states.items():
        rest = baselines.get(seat_id) or BASELINE
        for axis in AXES:
            gap = rest[axis] - state[axis]
            if not gap:
                continue
            # Move at least one point while a gap remains. A proportional step alone
            # rounds to nothing once the distance drops below about four, which would
            # strand a profile permanently short of its own resting state.
            step = gap * rate
            if abs(step) < 1:
                step = 1 if gap > 0 else -1
            state[axis] = _clamp(state[axis] + step)


def describe(state: Dict[str, int]) -> str:
    """A one-word mood, for the table embed and the `<game_context>` cast list.

    Ordered by which reading is most worth saying out loud: distress and urgency read
    louder than contentment, so they win a tie.
    """
    if state["cortisol"] >= 70:
        return "rattled"
    if state["adrenaline"] >= 70:
        return "wired"
    if state["dopamine"] >= 70:
        return "buoyant"
    if state["dopamine"] <= 25:
        return "flat"
    if state["cortisol"] <= 10 and state["adrenaline"] <= 15:
        return "becalmed"
    return "steady"


def as_prompt_line(state: Dict[str, int]) -> str:
    """The exact format `DEFAULT_NEURO_INSTRUCTION` already uses for CURRENT STATE."""
    return (
        f"D:{state['dopamine']} | C:{state['cortisol']} | "
        f"O:{state['oxytocin']} | A:{state['adrenaline']}"
    )
