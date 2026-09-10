"""Attribute views over parsed REST JSON.

Split out of `api_service.py`: the adapters and every response class read through
these, and nothing here knows what a provider is. `_RestView` maps snake_case
attribute reads onto camelCase JSON keys and returns None for anything absent, which
is what the `hasattr(...)` guards at the call sites already expect of SDK objects.
"""

import base64

from ...utils.blob_stream import is_blob_sentinel, sentinel_path



def _to_camel(snake: str) -> str:
    """snake_case attribute name -> camelCase JSON key."""
    head, _, tail = snake.partition("_")
    if not tail:
        return snake
    return head + "".join(w[:1].upper() + w[1:] for w in tail.split("_"))


class _EnumStr(str):
    """A REST enum value.

    The SDK delivers these as enum objects and the call sites read `.name`
    (`finish_reason.name`, `block_reason.name`). REST delivers a bare string.
    Subclassing str keeps both spellings working, so `x.name == 'STOP'` and
    `x == 'STOP'` are both true.
    """
    __slots__ = ()

    @property
    def name(self) -> str:
        return str(self)


# Attribute names whose values are enums in the SDK and plain strings over REST.
_ENUM_ATTRS = frozenset({"finish_reason", "block_reason"})


class _RestView:
    """Attribute view over one parsed REST JSON object.

    The SDK hands call sites objects with snake_case attributes; the wire format is
    camelCase JSON. Rather than hand-translate each response shape, this maps
    attribute reads onto the JSON keys and returns None for anything absent — which
    is exactly what the `hasattr(...)` / `... is not None` guards at the call sites
    already expect of the SDK objects.

    The consumed surface, verified by grep across cogs/ — a wrapper that quietly
    misses one of these looks like a model that "doesn't support images":

        candidates[0].content.parts[].text / .thought
        candidates[0].content.parts[].inline_data.data / .mime_type
        candidates[0].finish_reason.name
        candidates[0].grounding_metadata.grounding_chunks[].web.uri / .title
        candidates[0].grounding_metadata.grounding_supports[].segment.end_index
        candidates[0].grounding_metadata.grounding_supports[].grounding_chunk_indices
        candidates[0].url_context_metadata.url_metadata[].retrieved_url
        prompt_feedback.block_reason.name
        usage_metadata.prompt_token_count / .candidates_token_count

    Wrapped values are memoised. Call sites read the same attribute more than once
    (media_service tests `part.inline_data.data` for truthiness before binding it),
    and inline_data.data base64-decodes to a multi-megabyte blob — decoding it twice
    is a transient the e2-micro cannot spare.
    """

    __slots__ = ("_data", "_memo")

    def __init__(self, data: dict):
        self._data = data
        self._memo = {}

    def __getattr__(self, name):
        # Guard dunder lookups so copy/pickle protocols do not resolve to None.
        if name.startswith("__"):
            raise AttributeError(name)

        data = object.__getattribute__(self, "_data")
        memo = object.__getattribute__(self, "_memo")
        if name in memo:
            return memo[name]

        if name in data:
            key = name
        else:
            key = _to_camel(name)
            if key not in data:
                memo[name] = None
                return None
        value = data[key]

        wrapped = _wrap_rest(name, value)
        memo[name] = wrapped

        if name == "data" and isinstance(wrapped, bytes):
            # `value` is the base64 text of the blob just decoded -- about 1.33x
            # its size -- and the memo means nothing will read it again. An image
            # response otherwise carries the payload twice over, in two forms, for
            # as long as the caller holds the response. Drop the encoded original.
            data.pop(key, None)

        return wrapped

    def __bool__(self):
        return bool(object.__getattribute__(self, "_data"))

    def __repr__(self):
        return f"_RestView({object.__getattribute__(self, '_data')!r})"


class _BlobRef:
    """An `inline_data.data` value that was streamed to disk instead of decoded
    into RAM.

    Returned in place of `bytes` for any blob over `blob_stream`'s threshold, which
    in practice means every generated image and most synthesised audio. Truthy, so
    the `if part.inline_data.data` guards at the call sites read the same as they
    always did; `materialise_inline_data` is what turns it into a path.

    Ownership sits with the `GoogleRESTResponse` that produced it until a caller
    takes it, so an abandoned response -- a safety block, a fallback retry, an
    exception between here and the write -- does not strand the file.
    """

    __slots__ = ("path",)

    def __init__(self, path: str):
        self.path = path

    def __bool__(self):
        return True

    def read_bytes(self) -> bytes:
        with open(self.path, 'rb') as f:
            return f.read()

    def __repr__(self):
        return f"_BlobRef({self.path!r})"


def _wrap_rest(name: str, value):
    if isinstance(value, dict):
        return _RestView(value)
    if isinstance(value, list):
        return [_wrap_rest(name, v) for v in value]
    if isinstance(value, str):
        if name in _ENUM_ATTRS:
            return _EnumStr(value)
        if name == "data":
            # inline_data.data is base64 on the wire; the SDK hands call sites bytes.
            # Anything large enough to matter never got as far as this string --
            # blob_stream diverted it to a file on the way off the socket and left
            # a sentinel here in its place.
            if is_blob_sentinel(value):
                return _BlobRef(sentinel_path(value))
            try:
                return base64.b64decode(value)
            except Exception:
                return value
    return value
