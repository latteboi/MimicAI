"""Query embeddings, and the small cache in front of them.

One vector per (text, task type, dimensionality). The cache exists because the same
query is embedded more than once per turn -- LTM recall and help retrieval both ask.
"""

import asyncio
import orjson as json
import time
from collections import OrderedDict
from typing import Any, List, Optional

from .google_rest import get_google_rest_client



# Migration 2 step 2. One function, three call sites (memory_manager LTM/training
# recall, help_service's two RAG builders) — all three already share this exact
# payload shape, so the flag is applied here rather than at each site, same as
# GoogleGenAIModel above.
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
    api_key: str,
    text: str,
    task_type: str = "RETRIEVAL_QUERY",
    output_dimensionality: int = 256,
    timeout: float = 5.0,
) -> Optional[List[float]]:
    """Returns the embedding for `text`, or None on any failure."""
    if not text or not text.strip():
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
        values = await _fetch_embedding_vector(
            api_key, text, task_type, output_dimensionality, timeout
        )
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
    api_key: str,
    text: str,
    task_type: str,
    output_dimensionality: int,
    timeout: float,
) -> Optional[List[float]]:
    payload = {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": text}]},
        "taskType": task_type,
        "outputDimensionality": output_dimensionality,
    }
    try:
        client = get_google_rest_client()
        response = await asyncio.wait_for(
            client.post(
                "/v1beta/models/gemini-embedding-001:embedContent",
                content=json.dumps(payload),
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            ),
            timeout=timeout,
        )
        if response.status_code != 200:
            print(f"Embedding err for '{text[:30]}...': Google API Error {response.status_code}: {response.text}")
            return None
        body = json.loads(response.content)
        return body.get("embedding", {}).get("values")
    except Exception as e:
        print(f"Embedding err for '{text[:30]}...': {e}")
        return None
