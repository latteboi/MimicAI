"""The output-token cap every text request carries, and the models that refused it.

One cap for every provider, `defaultConfig.LIMIT_OUTPUT_TOKENS`, and it counts thinking:
each provider bills reasoning as output and stops both at the same limit. A model whose
own output ceiling sits below the cap may answer it with a 400 rather than clamping.
That refusal is the probe, as it is for Ollama's `think`: the call is redone once
without the field and the model is not sent it again. Uncapped, such a model is still
bounded by its own ceiling, which is the lower of the two.
"""

import re
from typing import Optional

from ...utils.constants import defaultConfig

#: Models that refused the cap. Bounded because CLAUDE.md forbids an unbounded set keyed
#: by model; a server uses a few dozen at most, and past the bound a refusal is raised.
_REFUSED: set = set()
_REFUSED_MAX = 64
#: The field as each provider names it in a 400 (`max_tokens`, `maxOutputTokens`), and
#: OpenRouter's context-length error, which counts the cap as output the model's window
#: has to hold beside the prompt.
_CAP_ERROR = re.compile(r"max_?(?:output_?|completion_?)?tokens|context length", re.IGNORECASE)


class RetryUncapped(Exception):
    """Internal signal: redo this call without the cap. An exception where the decision
    is made inside an `async with` the retry has to happen outside of."""


def output_cap(model: str) -> Optional[int]:
    """The cap to send `model`, or None once it has refused one."""
    return None if model in _REFUSED else defaultConfig.LIMIT_OUTPUT_TOKENS


def refused_output_cap(model: str, status: int, detail: str) -> bool:
    """Whether a capped request's error is `model` refusing the cap. Records it when so;
    the caller then redoes the call once, and `output_cap` sends it nothing.

    400 only: OpenRouter's out-of-credit error is a 402 that also names `max_tokens`,
    and retrying that uncapped would ask for more credit, not less.
    """
    if (status != 400 or model in _REFUSED or len(_REFUSED) >= _REFUSED_MAX
            or not _CAP_ERROR.search(detail or "")):
        return False
    _REFUSED.add(model)
    print(f"{model} refused a {defaultConfig.LIMIT_OUTPUT_TOKENS}-token output cap; "
          f"retrying without it and remembering.")
    return True
