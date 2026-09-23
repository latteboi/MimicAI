"""How long each model usually takes to answer, and when a slower call counts as stalled.

A reply's primary used to have the heartbeat's whole four minutes before its fallback
was tried, so a host that accepted the request and then sat on it cost the channel four
minutes of a ticking placeholder and an answer from the fallback at the end of it. Past
`race_after` seconds the fallback is started beside it instead, and whichever answers
first is the reply -- see `ReplyMixin._attempt_reply`.

Nothing streams, so time is the only sign of a stall there is. A fixed line would race
a reasoning model on every turn it thinks for a minute, which it may do every turn; so
the line is a multiple of what that model has recently taken, measured here from the
moment its call took a generation slot to the moment its answer landed.

In memory only, and bounded. A restart starts every model from the default.
"""
import collections
import statistics
from typing import Deque, Dict

from ...utils.constants import (
    FALLBACK_RACE_CEILING_SECONDS, FALLBACK_RACE_DEFAULT_SECONDS, FALLBACK_RACE_FLOOR_SECONDS,
    FALLBACK_RACE_MIN_SAMPLES, FALLBACK_RACE_MULTIPLE,
)

#: Answers remembered per model. The median of twenty is steady against one slow turn
#: and still moves within the hour when a host gets slower.
_SAMPLES = 20
#: Models remembered. A server uses a few dozen at most; past this the least recently
#: heard from is forgotten and goes back to the default.
_MODELS = 64

_recent: "collections.OrderedDict[str, Deque[float]]" = collections.OrderedDict()


def record(model_name: str, seconds: float) -> None:
    """One answered call. A call that timed out or was cancelled is not an answer."""
    if not model_name or seconds < 0:
        return
    samples = _recent.get(model_name)
    if samples is None:
        samples = _recent[model_name] = collections.deque(maxlen=_SAMPLES)
        if len(_recent) > _MODELS:
            _recent.popitem(last=False)
    else:
        _recent.move_to_end(model_name)
    samples.append(seconds)


def race_after(model_name: str) -> float:
    """Seconds a call on `model_name` may run before its fallback is started beside it."""
    samples = _recent.get(model_name)
    if not samples or len(samples) < FALLBACK_RACE_MIN_SAMPLES:
        return FALLBACK_RACE_DEFAULT_SECONDS
    line = FALLBACK_RACE_MULTIPLE * statistics.median(samples)
    return min(max(line, FALLBACK_RACE_FLOOR_SECONDS), FALLBACK_RACE_CEILING_SECONDS)


def snapshot() -> Dict[str, float]:
    """Each remembered model's current line, for a probe or a test."""
    return {name: race_after(name) for name in _recent}
