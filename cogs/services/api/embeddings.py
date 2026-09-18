"""Embeddings, the routes they may take, and the small cache in front of them.

One vector per (text, task type, dimensionality). The cache exists because the same
query is embedded more than once per turn -- LTM recall and help retrieval both ask.

Every vector comes from Google's gemini-embedding-001, asked for directly on a Gemini key
or through OpenRouter on an OpenRouter key. Both give the same vector for the same text,
so one archive holds either; a different model would not, which is why OpenRouter is not
offered any other.
"""

import asyncio
import orjson as json
from collections import OrderedDict
from typing import Any, List, NamedTuple, Optional, Sequence

from ...utils.http_client import get_openrouter_client
from .google_rest import get_google_rest_client


class EmbeddingRoute(NamedTuple):
    """One key an embedding may be requested on. Resolved by
    `StorageManager._embedding_routes`, which also decides `data_collection`."""
    provider: str  # "gemini" or "openrouter"
    api_key: str
    #: OpenRouter's `provider.data_collection`; None sends nothing.
    data_collection: Optional[str] = None


_OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
_OPENROUTER_EMBEDDING_MODEL = "google/gemini-embedding-001"

#: `input_type` and the hosts allowed, per task type. Measured against Google's API
#: directly (September 2026): both of OpenRouter's Google hosts return Google's own
#: vectors and honour `dimensions`, but only Vertex passes the task type on. AI Studio
#: embeds everything as a query, so a memory saved through it would sit at ~0.93 of the
#: vector search expects, for good. A query may go to either; a document only to Vertex.
_OPENROUTER_TASKS = {
    "RETRIEVAL_QUERY": ("search_query", ("google-vertex", "google-ai-studio")),
    "RETRIEVAL_DOCUMENT": ("search_document", ("google-vertex",)),
}


# --- Query embedding cache -----------------------------------------------------
#
# A single turn asks for the *same* embedding three times: LTM recall, training-example
# recall and help-mode RAG all embed `dynamic_context_for_turn` at the same task type
# and the same 256 dimensions, and generation_service gathers them concurrently -- so a
# plain read-through cache misses on all three. Hence single-flight: the first caller
# issues the request, the other two await its future.
#
# Only RETRIEVAL_QUERY is cached. RETRIEVAL_DOCUMENT is by construction unique per call
# (a newly written memory summary, a new training example), so caching those could only
# ever burn memory and evict live query entries during a bulk import.
#
# Embeddings are deterministic for a given (text, task, dims), so entries need no TTL --
# the LRU bound is the whole eviction story. Values are held as float32 ndarrays rather
# than Python lists: 1 KB against roughly 8 KB for 256 boxed floats, and every caller
# converts straight back to numpy anyway. `.tolist()` costs ~4 us and hands out a fresh
# list each time, so the stored array can never be mutated by a caller.

_EMBED_CACHE_MAX = 256
_embed_cache: "OrderedDict[Any, Any]" = OrderedDict()
_embed_inflight: dict = {}


def _embed_cache_key(text: str, task_type: str, output_dimensionality: int):
    """Hash the text rather than keying on it: a round context runs to several KB, and
    holding those strings alive as dict keys is the bulk of what the cache would cost."""
    import hashlib

    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
    return (digest, task_type, output_dimensionality)


def _embed_cache_get(key) -> Optional[List[float]]:
    hit = _embed_cache.get(key)
    if hit is None:
        return None
    _embed_cache.move_to_end(key)
    return hit.tolist()


def _embed_cache_put(key, values: List[float]):
    import numpy as np

    _embed_cache[key] = np.asarray(values, dtype=np.float32)
    _embed_cache.move_to_end(key)
    while len(_embed_cache) > _EMBED_CACHE_MAX:
        _embed_cache.popitem(last=False)


async def get_embedding_vector(
    routes: Sequence[EmbeddingRoute],
    text: str,
    task_type: str = "RETRIEVAL_QUERY",
    output_dimensionality: int = 256,
    timeout: float = 5.0,
) -> Optional[List[float]]:
    """Returns the embedding for `text` from the first route that answers, or None.

    `timeout` applies to each route in turn, not to the whole call.
    """
    if not routes or not text or not text.strip():
        return None

    cache_key = None
    if task_type == "RETRIEVAL_QUERY":
        cache_key = _embed_cache_key(text, task_type, output_dimensionality)

        cached = _embed_cache_get(cache_key)
        if cached is not None:
            return cached

        inflight = _embed_inflight.get(cache_key)
        if inflight is not None:
            # A sibling retrieval in this same turn is already fetching this exact
            # vector. Shielded so a cancelled waiter does not kill the request the
            # others are waiting on.
            try:
                result = await asyncio.shield(inflight)
            except Exception:
                return None
            return list(result) if result is not None else None

    future = None
    if cache_key is not None:
        future = asyncio.get_running_loop().create_future()
        _embed_inflight[cache_key] = future

    values: Optional[List[float]] = None
    try:
        for route in routes:
            values = await _fetch_embedding_vector(
                route, text, task_type, output_dimensionality, timeout
            )
            if values is not None:
                break
        if cache_key is not None and values is not None:
            _embed_cache_put(cache_key, values)
        return values
    finally:
        if cache_key is not None:
            _embed_inflight.pop(cache_key, None)
            if future is not None and not future.done():
                # Resolved rather than raised: waiters take the same None-means-skip
                # path the uncached call always had, and nothing is left as an
                # unretrieved exception. `values` stays None if the fetch raised.
                future.set_result(values)


async def _fetch_embedding_vector(
    route: EmbeddingRoute,
    text: str,
    task_type: str,
    output_dimensionality: int,
    timeout: float,
) -> Optional[List[float]]:
    fetch = _fetch_openrouter if route.provider == "openrouter" else _fetch_google
    try:
        values = await asyncio.wait_for(
            fetch(route, text, task_type, output_dimensionality), timeout=timeout
        )
    except Exception:
        return None
    # A row of any other length breaks `reshape(len(rows), -1)` for every row stored
    # beside it. A longer one is cut to size: Gemini's leading dimensions *are* the
    # smaller embedding (Matryoshka), which is all `outputDimensionality` does.
    if not isinstance(values, list) or len(values) < output_dimensionality:
        return None
    return values[:output_dimensionality]


async def _fetch_google(route: EmbeddingRoute, text: str, task_type: str, dims: int):
    payload = {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": text}]},
        "taskType": task_type,
        "outputDimensionality": dims,
    }
    response = await get_google_rest_client().post(
        "/v1beta/models/gemini-embedding-001:embedContent",
        content=json.dumps(payload),
        headers={"x-goog-api-key": route.api_key, "Content-Type": "application/json"},
    )
    # None is the whole failure contract: every caller treats it as "skip this
    # recall", and a transient 5xx is the common case. Nothing is printed -- the
    # text is conversation content and does not belong in the host's journal.
    if response.status_code != 200:
        return None
    return json.loads(response.content).get("embedding", {}).get("values")


async def _fetch_openrouter(route: EmbeddingRoute, text: str, task_type: str, dims: int):
    task = _OPENROUTER_TASKS.get(task_type)
    if task is None:
        return None
    input_type, hosts = task
    provider: dict = {"only": hosts}
    if route.data_collection:
        provider["data_collection"] = route.data_collection
    payload = {
        "model": _OPENROUTER_EMBEDDING_MODEL,
        "input": text,
        "dimensions": dims,
        "input_type": input_type,
        "provider": provider,
    }
    response = await get_openrouter_client().post(
        _OPENROUTER_EMBEDDINGS_URL,
        content=json.dumps(payload),
        headers={
            "Authorization": f"Bearer {route.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://discord.com",
            "X-Title": "MimicAI Discord Bot",
        },
    )
    if response.status_code != 200:
        return None
    data = json.loads(response.content).get("data") or []
    return data[0].get("embedding") if data else None
