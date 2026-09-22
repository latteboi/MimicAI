"""Running the functions a character asked for, and getting the answers back to it.

Two kinds of declared function, and the difference decides everything here:

  * **Answering** (`recall`, `search_web`): the model needs the result before it can
    finish its reply, so the turn has to go back to the provider with the call and its
    answer appended. That is another whole request carrying the whole conversation, per
    seated character, which is why `LIMIT_FUNCTION_CALL_ROUNDS` exists.
  * **Recording** (`set_mood`): nothing comes back. The call *is* the effect, and it
    is applied from the response it arrived on. A recording call must never drive
    another request -- a character that writes down how it feels has not asked a
    question, and looping on it would double the cost of every turn with the neuro
    engine on.

`LOOPING_FUNCTIONS` is that distinction, and it is the thing to update when a
declaration is added: a new answering function left out of it is called once and then
silently ignored, because the loop stops before it hands anything back.
"""

from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from ...utils.constants import NEURO_TOOL_NAME, RECALL_TOOL_NAME, SEARCH_TOOL_NAME
from ..api.function_calls import FunctionCall, call_part, result_part

#: The functions whose result the model has to see before it can finish.
LOOPING_FUNCTIONS = frozenset({RECALL_TOOL_NAME, SEARCH_TOOL_NAME})

#: The functions that are their own effect. Nothing meaningful goes back -- but see
#: `needs_continuation`: an OpenAI-compatible endpoint stops generating the moment it
#: emits any tool call, so "nothing goes back" and "nothing has to go back" are not the
#: same statement, and a model that recorded its mood and then said nothing is waiting.
RECORDING_FUNCTIONS = frozenset({NEURO_TOOL_NAME})

#: What a recording call is answered with when one has to be answered at all. Content,
#: not emptiness: a tool message with an empty body reads as a failed call to some
#: hosts, and the model then apologises for it in character.
RECORDING_ACK = {"ok": True}


class FunctionContext(NamedTuple):
    """Who is asking, which is everything a handler needs that the arguments do not say.

    A function runs on behalf of one seated character, and `recall` reaches that
    character's own memory shard -- so the owner and profile here are the answer to
    "whose memories", not merely logging. Passing them rather than letting a handler
    resolve them keeps the borrow rules in one place.

    `safety_settings` are the destination channel's, resolved once by the caller that
    has the channel. `search_web` sends a query to a second model, and that model has
    to answer under the same rules the reply does -- a search refused on a setting the
    channel is not held to fails the turn's research for no reason. Per CLAUDE.md these
    key off the channel and never the profile, which is precisely why a handler cannot
    work them out for itself.
    """
    owner_id: int
    profile_name: str
    author_dn: str
    guild_id: Optional[int]
    triggering_user_id: int
    safety_settings: Optional[Dict[str, Any]] = None


async def execute(cog, calls: List[FunctionCall],
                  ctx: FunctionContext) -> List[Tuple[FunctionCall, Any]]:
    """Runs each answering call and pairs it with its result.

    A handler that raises is answered with an error object rather than being allowed
    to fail the turn. The model asked a question; "that lookup failed" is a usable
    answer and a dropped turn is not, and a character whose memory search errors
    should still be able to say something.
    """
    out: List[Tuple[FunctionCall, Any]] = []
    for call in calls:
        try:
            if call.name == RECALL_TOOL_NAME:
                query = call.args.get("query")
                if not isinstance(query, str) or not query.strip():
                    result: Any = {"error": "recall needs a `query` string."}
                else:
                    result = await cog.memory_manager.recall_for_tool(
                        ctx.owner_id, ctx.profile_name, query.strip(),
                        ctx.author_dn, ctx.guild_id, ctx.triggering_user_id)
            elif call.name == SEARCH_TOOL_NAME:
                query = call.args.get("query")
                if not isinstance(query, str) or not query.strip():
                    result = {"error": "search_web needs a `query` string."}
                else:
                    result = await cog.tools_service.search_for_tool(
                        query.strip(), guild_id=ctx.guild_id, owner_id=ctx.owner_id,
                        profile_name=ctx.profile_name,
                        safety_settings=ctx.safety_settings)
            elif call.name in RECORDING_FUNCTIONS:
                # Already applied from the response it arrived on. This only exists so
                # that a model which withheld its reply has something to continue from.
                result = RECORDING_ACK
            else:
                result = {"error": f"No function named {call.name!r}."}
        except Exception as e:  # noqa: BLE001 -- see docstring
            print(f"Function {call.name!r} failed: {e}")
            result = {"error": "That lookup failed."}
        out.append((call, result))
    return out


def exchange_turns(results: List[Tuple[FunctionCall, Any]],
                   spoken: str) -> List[Dict[str, Any]]:
    """The two conversation turns that record one round of calls and their answers.

    A `model` turn holding the calls, then a `user` turn holding the results -- the
    neutral spelling both adapters translate. `spoken` is whatever the model said
    alongside its calls, kept because dropping it would leave the next request looking
    as though the character had gone silent mid-sentence.
    """
    if not results:
        return []
    model_parts: List[Any] = [spoken] if spoken else []
    model_parts.extend(call_part(call) for call, _ in results)
    return [
        {"role": "model", "parts": model_parts},
        {"role": "user", "parts": [result_part(call, value) for call, value in results]},
    ]


def split_calls(response) -> Tuple[List[FunctionCall], List[FunctionCall]]:
    """`(answering, recording)` -- the calls that need another request, and the rest."""
    calls = list(getattr(response, "function_calls", None) or ())
    answering = [c for c in calls if c.name in LOOPING_FUNCTIONS]
    recording = [c for c in calls if c.name not in LOOPING_FUNCTIONS]
    return answering, recording


def needs_continuation(response, answering: List[FunctionCall],
                       recording: List[FunctionCall]) -> List[FunctionCall]:
    """The calls that must be answered before this turn has a reply, which is not the
    same set as the calls that wanted answering.

    An answering call always qualifies: `recall` exists to be answered.

    A *recording* call qualifies only when the model said nothing alongside it, and
    that case turned out to be the common one rather than the exception. An
    OpenAI-compatible endpoint stops generating at its first tool call, so
    `google/gemini-2.5-flash` returning `set_mood` returns `"content": null` with it --
    measured at 5 runs out of 5. Treating that as fire-and-forget leaves the turn with
    no text at all, which fails it. So the mood is applied from the response it came on,
    as designed, and the model is then handed a bare acknowledgement purely so it can go
    on and speak.

    A model that says its piece *and* records its mood in one response -- Google's own
    endpoint does -- costs nothing extra, which is why this is measured per response
    rather than switched on for the profile.
    """
    if answering:
        return answering + recording
    spoken = (getattr(response, "text", "") or "").strip()
    return list(recording) if (recording and not spoken) else []


def carry_forward(response, recording: List[FunctionCall]) -> None:
    """Moves recording calls made on an intermediate response onto the final one.

    A model that calls `set_mood` and `recall` in one breath has its mood applied from
    whichever response finally speaks -- and that is a later one. Without this the
    intermediate call is thrown away with the response it arrived on, so turning on
    `recall` would quietly stop the neuro engine updating on exactly the turns where
    something interesting enough to look up had happened.
    """
    if not recording or response is None:
        return
    existing = list(getattr(response, "function_calls", None) or ())
    # Earlier first: a later call is the model's more considered answer, and dict-style
    # merging downstream lets the last one win.
    response.function_calls = recording + existing


#: How much of a query the audit line keeps. Long enough to recognise what was asked,
#: short enough that a turn's metadata stays metadata -- it is persisted in the session
#: log, once per call, forever.
_LABEL_QUERY_CHARS = 60


def function_call_label(call: FunctionCall, result: Any) -> str:
    """One line for `/session audit`: what was asked, and what came back.

    The name alone answers "did the tool fire", which is the smaller half of the
    question. Whether a character asks *good* questions, and whether its own archive
    can answer them, is only visible with the query and the count beside it -- and that
    is the thing to look at before deciding a recall threshold is wrong.
    """
    query = call.args.get("query")
    asked = call.name
    if isinstance(query, str) and query.strip():
        text = query.strip()
        if len(text) > _LABEL_QUERY_CHARS:
            text = text[:_LABEL_QUERY_CHARS - 1] + "…"
        asked = f"{call.name}: {text}"
    if isinstance(result, dict):
        if "error" in result:
            return f"{asked} -> failed"
        # Whichever kind of result this call returns: memories for `recall`, cited
        # sources for `search_web`. A count is the whole of "did asking help".
        for key in ("memories", "sources"):
            found = result.get(key)
            if isinstance(found, list):
                return f"{asked} -> {len(found)}"
    return asked
