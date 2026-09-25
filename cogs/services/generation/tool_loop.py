"""The functions a character may call, and the one loop that hands their answers back.

Each function is one `CharacterFunction` in `FUNCTIONS`, and that entry is everything
about it: the declaration a provider is sent, the block telling the character it has
it, the profile setting that turns it on, whether it needs a server, what runs it and
what its result holds. Nothing else names a function. That is what used to go wrong:
the declaration, the prompt and the loop were decided in three places, and a character
could be told to call something it was never sent -- so it narrated a search it could
not run and nothing was logged.

Whoever builds a character's model asks `functions_for` once and hands the same tuple
to the prompt builder and to the model factory; the factory stamps it on the model as
`model.functions`, and `run` answers exactly that set. Every path that speaks as a
character -- a session reply, a regeneration, a whisper, Global Chat, a speak rewrite,
an image's presentation -- goes through `run`, so a function is offered everywhere or
nowhere, never on one path and silently absent on the next.

Every function answers: the model needs the result to finish, so a call costs another
whole request carrying the conversation. `LIMIT_FUNCTION_CALL_ROUNDS` bounds that, and
the request after the last round is sent with calls forbidden, so a character that
would keep asking has to answer instead of ending the turn with no text.
"""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from ...utils.constants import (
    DEFAULT_RECALL_INSTRUCTION, DEFAULT_SEARCH_INSTRUCTION, LIMIT_FUNCTION_CALL_ROUNDS,
    LTM_AUTO_THRESHOLD_WITH_TOOL, RECALL_TOOL_DECLARATION, RECALL_TOOL_NAME,
    SEARCH_TOOL_DECLARATION, SEARCH_TOOL_NAME,
)
from ...utils.helpers import provider_takes_functions, resolve_grounding_mode
from ..api.function_calls import (FunctionCall, call_part, forbid_calls, result_part,
                                  verbatim_part)


class FunctionContext(NamedTuple):
    """Who is asking, which is everything a handler needs that the arguments do not say.

    A function runs on behalf of one character, and `recall` reaches that character's
    own memory shard -- so the owner and profile here are the answer to "whose
    memories", not merely logging. Passing them rather than letting a handler resolve
    them keeps the borrow rules in one place.

    `safety_settings` are the destination channel's, resolved once by the caller that
    has the channel: `search_web` asks a second model, which has to answer under the
    rules the reply does, and those key off the channel, never the profile.

    `policy` is Global Chat's `{"policy_guild_id", "conversation"}`, handed to the
    researcher's model exactly as the reply's own models get it: the host's keys, judged
    against the policy of the server the card is open in. None means the server's keys,
    as everywhere else.
    """
    owner_id: int
    profile_name: str
    author_dn: str
    guild_id: Optional[int]
    triggering_user_id: int
    safety_settings: Optional[Dict[str, Any]] = None
    policy: Optional[Dict[str, Any]] = None


async def _recall(cog, query: str, ctx: FunctionContext) -> Dict[str, Any]:
    return await cog.memory_manager.recall_for_tool(
        ctx.owner_id, ctx.profile_name, query, ctx.author_dn, ctx.guild_id,
        ctx.triggering_user_id)


async def _search_web(cog, query: str, ctx: FunctionContext) -> Dict[str, Any]:
    return await cog.tools_service.search_for_tool(
        query, guild_id=ctx.guild_id, owner_id=ctx.owner_id,
        profile_name=ctx.profile_name, safety_settings=ctx.safety_settings,
        policy=ctx.policy)


@dataclass(frozen=True)
class CharacterFunction:
    """One function a character may call, whole."""
    name: str
    #: What the provider is sent, in the neutral spelling `api.function_calls` converts.
    #: Every declaration takes one required `query` string, which `execute` checks.
    declaration: Dict[str, Any]
    #: The /mod key of the block telling the character it has this, and the shipped text.
    instruction_key: str
    default_instruction: str
    #: The profile's own switch. Whether the route can carry a declaration at all is
    #: asked once, for every function, in `functions_for`.
    enabled: Callable[[Dict[str, Any]], bool]
    #: (cog, query, ctx) -> the object handed back to the model. Never raises for a
    #: miss: "nothing found" is an answer, where silence invites an invented one.
    handler: Callable[[Any, str, FunctionContext], Awaitable[Dict[str, Any]]]
    #: The list a result holds that the turn's trace keeps: memories, or sources.
    found_key: str
    #: Memories are filed per server, so off one -- Global Chat -- `recall` could only
    #: ever come back empty, and each empty answer would cost a request.
    needs_server: bool = False

    def instruction(self, global_prompts: Dict[str, str]) -> str:
        return global_prompts.get(self.instruction_key, self.default_instruction)


RECALL = CharacterFunction(
    name=RECALL_TOOL_NAME, declaration=RECALL_TOOL_DECLARATION,
    instruction_key="RECALL_INSTRUCTION", default_instruction=DEFAULT_RECALL_INSTRUCTION,
    enabled=lambda config: bool(config.get("ltm_recall_tool_enabled")),
    handler=_recall, found_key="memories", needs_server=True)

SEARCH_WEB = CharacterFunction(
    name=SEARCH_TOOL_NAME, declaration=SEARCH_TOOL_DECLARATION,
    instruction_key="SEARCH_INSTRUCTION", default_instruction=DEFAULT_SEARCH_INSTRUCTION,
    # Grounding's "tool" mode is this declaration and nothing else: shown as RAG.
    enabled=lambda config: resolve_grounding_mode(config) == "tool",
    handler=_search_web, found_key="sources")

#: Every function a character can call, in the order they are declared and described.
FUNCTIONS: Tuple[CharacterFunction, ...] = (RECALL, SEARCH_WEB)
_BY_NAME = {f.name: f for f in FUNCTIONS}


def functions_for(config: Optional[Dict[str, Any]], *, has_server: bool = True,
                  can_search: bool = True) -> Tuple[CharacterFunction, ...]:
    """The functions this profile's model declares and its prompt describes -- one
    answer for both, so the two cannot disagree.

    Nothing on a pair with an Ollama slot: that adapter streams, and a call arrives
    split across chunks with no accumulator written for it. Both slots and not either,
    because the prompt is written once for a turn the fallback may answer.

    `has_server` False drops what needs one (`recall`). `can_search` False drops
    `search_web`: Global Chat has no server key, and without one of its host's own a
    search could only come back "unavailable", a request spent to learn nothing.
    """
    config = config or {}
    if not provider_takes_functions(config.get("primary_model"), config.get("fallback_model")):
        return ()
    return tuple(f for f in FUNCTIONS
                 if f.enabled(config) and (has_server or not f.needs_server)
                 and (can_search or f is not SEARCH_WEB))


def declarations(functions: Sequence[CharacterFunction]) -> Optional[List[Dict[str, Any]]]:
    """What a model is sent for `functions`; None rather than an empty `tools` array."""
    return [f.declaration for f in functions] or None


def declared_on(model) -> Tuple[CharacterFunction, ...]:
    """The functions `model` was built declaring, as `_instantiate_model` stamped them."""
    return tuple(getattr(model, "functions", None) or ())


def ltm_auto_threshold(config: Optional[Dict[str, Any]]) -> Optional[float]:
    """The relevance threshold the automatic LTM pass uses for this profile, or None for
    the profile's own.

    Narrowed only where `recall` is really offered, which is `functions_for`'s answer:
    narrowing the unasked pass with nothing to ask for the rest is just less memory.
    """
    return LTM_AUTO_THRESHOLD_WITH_TOOL if RECALL in functions_for(config) else None


async def execute(cog, calls: List[FunctionCall], ctx: FunctionContext,
                  offered: Sequence[CharacterFunction] = FUNCTIONS) -> List[Tuple[FunctionCall, Any]]:
    """Runs each call and pairs it with its result.

    A call to anything not `offered` is answered with a refusal rather than run: a
    function this model was not declared may still be named in its text, and running
    it anyway would bill a search to a profile that never turned one on.

    A handler that raises is answered with an error object rather than failing the
    turn. The model asked a question; "that lookup failed" is a usable answer and a
    dropped turn is not.
    """
    allowed = {f.name: f for f in offered}
    out: List[Tuple[FunctionCall, Any]] = []
    for call in calls:
        spec = allowed.get(call.name)
        try:
            if spec is None:
                result: Any = {"error": f"No function named {call.name!r}."}
            else:
                query = call.args.get("query")
                if not isinstance(query, str) or not query.strip():
                    result = {"error": f"{call.name} needs a `query` string."}
                else:
                    result = await spec.handler(cog, query.strip(), ctx)
        except Exception as e:  # noqa: BLE001 -- see docstring
            print(f"Function {call.name!r} failed: {e}")
            result = {"error": "That lookup failed."}
        out.append((call, result))
    return out


def exchange_turns(results: List[Tuple[FunctionCall, Any]], spoken: str,
                   echo: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """The two conversation turns that record one round of calls and their answers.

    A `model` turn holding the calls, then a `user` turn holding the results -- the
    neutral spelling both adapters translate. `spoken` is whatever the model said
    alongside its calls, kept because dropping it would leave the next request looking
    as though the character had gone silent mid-sentence.

    `echo` is the response's own `model_parts` where it has them (Google), and then the
    model turn is those, untouched: they carry `spoken` already, and the signatures and
    built-in tool parts a rebuilt turn would lose.
    """
    if not results:
        return []
    if echo:
        model_parts: List[Any] = [verbatim_part(p) for p in echo]
    else:
        model_parts = [spoken] if spoken else []
        model_parts.extend(call_part(call) for call, _ in results)
    return [
        {"role": "model", "parts": model_parts},
        {"role": "user", "parts": [result_part(call, value) for call, value in results]},
    ]


#: How much of a query the audit line keeps. Long enough to recognise what was asked,
#: short enough that a turn's metadata stays metadata -- it is persisted in the session
#: log, once per call, forever.
_LABEL_QUERY_CHARS = 60


def function_call_label(call: FunctionCall, result: Any) -> str:
    """One line for `/session audit`: what was asked, and what came back.

    The name alone answers "did the tool fire", which is the smaller half of the
    question. Whether a character asks *good* questions, and whether what it asks can
    be answered, is only visible with the query and the count beside it.
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
        spec = _BY_NAME.get(call.name)
        found = result.get(spec.found_key) if spec else None
        if isinstance(found, list):
            return f"{asked} -> {len(found)}"
    return asked


@dataclass
class LoopResult:
    """What one character's reply cost in calls, and what they brought back."""
    #: The last response received -- the one that answered in words, if any did.
    response: Any = None
    #: One `function_call_label` per call, in the order they were made.
    calls: List[str] = field(default_factory=list)
    memories: List[str] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    #: The budget ran out: the last request forbade calls, so the character answered
    #: with what it had rather than asking again.
    capped: bool = False

    def record(self, meta: Dict[str, Any]) -> None:
        """Writes the calls onto a turn's trace -- sparse, like the rest of it.

        Replaces rather than adds: a primary that asked and then failed leaves its
        calls behind, and the trace describes the reply that was actually posted.
        """
        meta.pop("function_calls", None)
        meta.pop("function_calls_capped", None)
        if self.calls:
            meta["function_calls"] = list(self.calls)
        if self.capped:
            meta["function_calls_capped"] = True


async def run(cog, model, contents: List[Any], gen_config: Optional[Dict[str, Any]],
              ctx: FunctionContext,
              send: Callable[[List[Any], Dict[str, Any]], Awaitable[Any]]) -> LoopResult:
    """One character reply: generate, answer what it asks for, and generate again.

    `send(contents, gen_config)` makes one request on `model` and returns the response;
    the caller owns everything a request means on its path -- the placeholder it ticks,
    the gate slot, what counts as blocked -- and raises as it always did. This owns only
    the conversation between requests, and answers exactly what `model` declared.

    `contents` is copied, not extended: a caller that falls back to another model
    starts from the conversation it had, not from the lookups the first model made.
    A model that declared nothing makes one request, as it always did.
    """
    offered = declared_on(model)
    turn = list(contents)
    gen_config = dict(gen_config or {})
    result = LoopResult()
    for round_no in range(LIMIT_FUNCTION_CALL_ROUNDS + 1):
        last = bool(offered) and round_no == LIMIT_FUNCTION_CALL_ROUNDS
        result.capped = last
        result.response = await send(turn, forbid_calls(gen_config) if last else gen_config)
        calls = list(getattr(result.response, "function_calls", None) or ())
        if not offered or not calls or last:
            break
        answered = await execute(cog, calls, ctx, offered)
        for call, value in answered:
            result.calls.append(function_call_label(call, value))
            if isinstance(value, dict):
                result.memories.extend(m for m in (value.get(RECALL.found_key) or ())
                                       if isinstance(m, str))
                result.sources.extend(src for src in (value.get(SEARCH_WEB.found_key) or ())
                                      if isinstance(src, dict) and src.get("uri"))
        turn.extend(exchange_turns(answered, getattr(result.response, "text", "") or "",
                                   getattr(result.response, "model_parts", None)))
    return result
