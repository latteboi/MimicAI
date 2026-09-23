"""One shape for a declared function, whichever provider answers.

The two providers disagree at both ends. Google takes `functionDeclarations` and
answers with a `functionCall` part sitting alongside the text parts; OpenRouter takes
OpenAI's `tools` array and answers with `tool_calls`, whose arguments are a *string*
of JSON rather than an object. Normalising both here means a call site reads
`response.function_calls` and never learns which one it was talking to -- the same
service `_RestView` performs for camelCase.

Declarations are written once, in the neutral JSON-Schema spelling OpenAI uses
(lowercase `type`), and converted per provider on the way out. Google's schema enum
is documented in upper case, so `as_google_tool` raises it; nothing else differs.

Why this is worth a module rather than two inline branches: a malformed argument set
is now *detectable*. The hand-rolled parsers this replaces -- the `<neuro_update>`
regex, the yes/no first line -- could only fail silently, applying half a result or
none and reporting neither.
"""

import json
from typing import Any, Dict, List, NamedTuple, Optional


#: The generation-config key that forbids calls on one request while keeping the
#: declarations -- OpenRouter's `tool_choice: "none"`, Google's function-calling mode
#: NONE. Declarations stay because the conversation being sent already holds calls and
#: their results, which some hosts refuse to read without them.
_FUNCTION_CALLING = "function_calling"


def forbid_calls(generation_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """A copy of `generation_config` for a request that must answer in words."""
    return {**(generation_config or {}), _FUNCTION_CALLING: "none"}


def calls_forbidden(generation_config: Any) -> bool:
    """Whether this request was made by `forbid_calls`."""
    return isinstance(generation_config, dict) and generation_config.get(_FUNCTION_CALLING) == "none"


class FunctionCall(NamedTuple):
    """One call the model asked for.

    `call_id` is OpenRouter's correlation id, which a function *result* must quote
    when it goes back. Google correlates by name instead and leaves this empty, so
    nothing may treat it as present.
    """
    name: str
    args: Dict[str, Any]
    call_id: str = ""


def declaration(name: str, description: str,
                properties: Optional[Dict[str, Any]] = None,
                required: Optional[List[str]] = None) -> Dict[str, Any]:
    """One function declaration in the neutral spelling.

    A declaration with no properties still carries an empty object schema: both
    providers reject a parameterless function declared with `parameters` absent.
    """
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties or {},
            "required": list(required or ()),
        },
    }


def _google_schema(schema: Any) -> Any:
    """`schema` with every `type` raised to Google's documented enum spelling.

    Walks `properties` and `items` rather than every dict, so a property whose *name*
    happens to be "type" is left alone -- it is a key in `properties`, not a schema
    node, and rewriting its value would corrupt the declaration.
    """
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    if isinstance(out.get("type"), str):
        out["type"] = out["type"].upper()
    props = out.get("properties")
    if isinstance(props, dict):
        out["properties"] = {k: _google_schema(v) for k, v in props.items()}
    items = out.get("items")
    if isinstance(items, dict):
        out["items"] = _google_schema(items)
    return out


def as_google_tool(declarations: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """The one `tools` entry Google takes for a whole set of declarations.

    Snake_case on the outer key because `_build_tools` camelCases exactly that and
    passes the value through untouched -- which is also why the declaration's own
    keys (`name`, `description`, `parameters`) have to be right here.
    """
    if not declarations:
        return None
    return {"function_declarations": [
        {**d, "parameters": _google_schema(d.get("parameters"))} for d in declarations
    ]}


def as_openai_tools(declarations: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """The `tools` array OpenRouter takes. The neutral spelling is already OpenAI's."""
    if not declarations:
        return None
    return [{"type": "function", "function": d} for d in declarations]


def from_google_parts(parts) -> List[FunctionCall]:
    """Every `functionCall` part in one candidate's content.

    `part.args` arrives as a plain dict -- see `_RAW_DICT_ATTRS` in rest_view -- so
    there is nothing to decode. A call with no name is dropped rather than passed on
    as an empty string that would miss every dispatch entry and look like a refusal.
    """
    calls = []
    for part in parts or ():
        fc = getattr(part, "function_call", None)
        if not fc:
            continue
        name = getattr(fc, "name", None)
        if not name:
            continue
        args = getattr(fc, "args", None)
        calls.append(FunctionCall(name=name, args=args if isinstance(args, dict) else {}))
    return calls


def from_openrouter_message(msg_obj: Dict[str, Any]) -> List[FunctionCall]:
    """Every `tool_calls` entry on one OpenRouter message.

    `arguments` is a JSON *string*, and a model that produces a malformed one is
    common enough to be ordinary rather than exceptional -- so the call is dropped
    and the others kept, instead of the whole response failing on one bad set.
    """
    calls = []
    for entry in (msg_obj.get("tool_calls") or ()):
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not name:
            continue
        raw = fn.get("arguments")
        if isinstance(raw, dict):
            args = raw
        else:
            try:
                args = json.loads(raw) if raw else {}
            except (ValueError, TypeError):
                print(f"Function call {name!r} arrived with unparseable arguments; dropped.")
                continue
            if not isinstance(args, dict):
                continue
        calls.append(FunctionCall(name=name, args=args, call_id=entry.get("id") or ""))
    return calls


# --------------------------------------------------------------------------- #
# The way back: a call and its result, carried in `contents` between turns.
#
# A tool loop re-sends the whole conversation with the call the model just made and
# the answer to it appended, so both have to survive the neutral `{'role', 'parts'}`
# shape the adapters take. These two part spellings are that, and each adapter
# translates them on the way out -- Google into `functionCall`/`functionResponse`
# parts, OpenRouter into an assistant message carrying `tool_calls` followed by one
# `role: "tool"` message per result. The shapes are far enough apart that leaving the
# translation to the call sites would mean writing the loop twice.
# --------------------------------------------------------------------------- #


def call_part(call: FunctionCall) -> Dict[str, Any]:
    """The part that records, in a `model` turn, that this call was asked for."""
    return {"function_call": {"name": call.name, "args": call.args, "id": call.call_id}}


def result_part(call: FunctionCall, result: Any) -> Dict[str, Any]:
    """The part that answers `call`, for the `user` turn that follows it.

    `result` is wrapped in an object rather than sent bare because Google's
    `functionResponse.response` must be one, and a handler that returns a list or a
    string is the common case -- `recall` returns memories.
    """
    payload = result if isinstance(result, dict) else {"result": result}
    return {"function_response": {"name": call.name, "response": payload,
                                  "id": call.call_id}}


def google_part(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One neutral function part in Google's spelling, or None if it is not one."""
    call = part.get("function_call")
    if isinstance(call, dict):
        return {"functionCall": {"name": call.get("name") or "",
                                 "args": call.get("args") or {}}}
    result = part.get("function_response")
    if isinstance(result, dict):
        return {"functionResponse": {"name": result.get("name") or "",
                                     "response": result.get("response") or {}}}
    return None


def openai_messages(parts: List[Any], text: str) -> List[Dict[str, Any]]:
    """The OpenAI-shaped messages for one turn's function parts.

    Returns an assistant message carrying `tool_calls` when the turn made calls, and
    one `role: "tool"` message per result. `tool_call_id` is required on the result
    and has to match the id that came back, which is why `FunctionCall` keeps one at
    all -- Google correlates by name and needs none.

    An assistant turn that made a call may also have said something; `content` carries
    it, and is None rather than "" when it did not, because some hosts reject an empty
    string alongside `tool_calls`.
    """
    calls, results = [], []
    for part in parts:
        if not isinstance(part, dict):
            continue
        call = part.get("function_call")
        if isinstance(call, dict):
            calls.append({
                "id": call.get("id") or call.get("name") or "call",
                "type": "function",
                "function": {"name": call.get("name") or "",
                             "arguments": json.dumps(call.get("args") or {})},
            })
            continue
        result = part.get("function_response")
        if isinstance(result, dict):
            results.append({
                "role": "tool",
                "tool_call_id": result.get("id") or result.get("name") or "call",
                "content": json.dumps(result.get("response") or {}),
            })
    out = []
    if calls:
        out.append({"role": "assistant", "content": text or None, "tool_calls": calls})
    out.extend(results)
    return out


def has_function_parts(parts) -> bool:
    """Whether a turn carries any function part, and so needs the messages above
    instead of the ordinary content-part path."""
    return any(isinstance(p, dict) and ("function_call" in p or "function_response" in p)
               for p in (parts or ()))
