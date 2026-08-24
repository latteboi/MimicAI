import os
import time
import asyncio
import base64
import datetime
import httpx
import orjson as json
from collections import OrderedDict
from typing import get_args, Any, List, Optional, Tuple
from discord.ext import tasks

from ..utils.constants import (
    OLLAMA_LOCAL_URL, MODELS_DATA_DIR, PRICING_CACHE_FILE, IMAGE_MODELS, AUDIO_MODELS,
    ALLOWED_MODELS, defaultConfig, PRIMARY_MODEL_NAME, FALLBACK_MODEL_NAME,
    IMAGE_MODEL_KEYS, AUDIO_MODEL_KEYS, DEFAULT_SPEECH_VOICE,
)
from ..utils.helpers import _resolve_safety_settings, is_real_model
from ..utils.http_client import get_shared_client
from ..utils.memory_tuning import maybe_trim_malloc

# How long a key that just got rate-limited is skipped for. storage_manager's
# _get_api_key_for_guild/_get_api_key_for_user consult cog.api_key_cooldowns before
# handing a key back out, so this is what stops a 429'd BYO key from being retried
# on the very next turn instead of backing off.
_KEY_COOLDOWN_SECONDS = 60.0





def _cooldown_key_on_rate_limit(cog, api_key: Optional[str], error: BaseException) -> None:
    if not api_key:
        return
    err_str = str(error)
    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
        cog.api_key_cooldowns[api_key] = time.time() + _KEY_COOLDOWN_SECONDS


def _with_key_cooldown_tracking(cog, model, api_key: str):
    """Wraps model.generate_content_async so a rate-limit response cools the BYO key down."""
    original = model.generate_content_async

    async def _tracked(*args, **kwargs):
        try:
            return await original(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _cooldown_key_on_rate_limit(cog, api_key, e)
            raise

    model.generate_content_async = _tracked
    return model


class OpenRouterModel:
    def __init__(self, model_name, api_key, system_instruction=None, thinking_params=None, **kwargs):
        self.model_name = model_name.replace("OPENROUTER/", "").replace("GOOGLE/", "")
        self.api_key = api_key
        self.system_instruction = system_instruction
        self.thinking_params = thinking_params or {} # [NEW]

    async def generate_content_async(self, contents, generation_config=None, safety_settings=None, stream_state=None):
        messages = []
        if self.system_instruction:
            messages.append({"role": "system", "content": self.system_instruction})

        for content in contents:
            if isinstance(content, str):
                content = {'role': 'user', 'parts': [content]}

            role = "assistant" if content.get('role', 'user') == "model" else "user"
            message_parts = []

            for p in content.get('parts', []):
                if isinstance(p, str) and p.strip():
                    message_parts.append({"type": "text", "text": p})
                elif isinstance(p, dict) and 'mime_type' in p and 'data' in p:
                    mime_type = p['mime_type']
                    if mime_type.startswith("image/"):
                        try:
                            b64_data = base64.b64encode(p['data']).decode('utf-8')
                            data_uri = f"data:{mime_type};base64,{b64_data}"
                            message_parts.append({"type": "image_url", "image_url": {"url": data_uri}})
                        except Exception as e:
                            print(f"Error encoding image for OpenRouter: {e}")
                elif hasattr(p, 'inline_data') and p.inline_data:
                    mime_type = p.inline_data.mime_type
                    if mime_type.startswith("image/"):
                        try:
                            b64_data = base64.b64encode(p.inline_data.data).decode('utf-8')
                            data_uri = f"data:{mime_type};base64,{b64_data}"
                            message_parts.append({"type": "image_url", "image_url": {"url": data_uri}})
                        except Exception as e:
                            print(f"Error encoding legacy image for OpenRouter: {e}")
                elif isinstance(p, dict) and 'url' in p:
                    mime_type = p.get('mime_type', 'image/png')
                    url = p['url']
                    if mime_type.startswith("image/"):
                        if url.startswith(('http://', 'https://')) or url.startswith('data:'):
                            message_parts.append({"type": "image_url", "image_url": {"url": url}})
                        elif os.path.exists(url):
                            try:
                                with open(url, 'rb') as img_f:
                                    b64_data = base64.b64encode(img_f.read()).decode('utf-8')
                                data_uri = f"data:{mime_type};base64,{b64_data}"
                                message_parts.append({"type": "image_url", "image_url": {"url": data_uri}})
                            except Exception as e:
                                print(f"Error encoding local image for OpenRouter: {e}")

            if message_parts:
                if len(message_parts) == 1 and message_parts[0]["type"] == "text":
                    messages.append({"role": role, "content": message_parts[0]["text"]})
                else:
                    messages.append({"role": role, "content": message_parts})

        temp = 1.0
        top_p = 1.0
        advanced = {}
        include_thoughts = self.thinking_params.get("thinking_summary_visible") == "on"

        if isinstance(generation_config, dict):
            temp = generation_config.get("temperature", 1.0)
            top_p = generation_config.get("top_p", 1.0)
            advanced = generation_config.get("_advanced_params", {})
            if generation_config.get("thinking_config"):
                include_thoughts = generation_config["thinking_config"].get("include_thoughts", include_thoughts)
        elif generation_config:
            temp = getattr(generation_config, 'temperature', 1.0)
            top_p = getattr(generation_config, 'top_p', 1.0)
            if hasattr(generation_config, '_advanced_params') and generation_config._advanced_params:
                advanced = generation_config._advanced_params
            if hasattr(generation_config, 'thinking_config') and generation_config.thinking_config:
                include_thoughts = generation_config.thinking_config.include_thoughts

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temp,
            "top_p": top_p,
        }

        budget = int(self.thinking_params.get("thinking_budget", -1))
        level = self.thinking_params.get("thinking_level", "high").lower()

        if include_thoughts or budget > 0 or level != "none":
            payload["reasoning"] = {"exclude": not include_thoughts}
            if budget > 0:
                payload["reasoning"]["max_tokens"] = budget
            elif level != "none":
                payload["reasoning"]["effort"] = level

        if advanced:
            payload.update(advanced)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://discord.com",
            "X-Title": "MimicAI Discord Bot"
        }

        try:
            client = get_shared_client()
            response = await client.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=120.0)
            if response.status_code != 200:
                raise Exception(f"OpenRouter API Error {response.status_code}: {response.text}")

            data = response.json()
            if 'error' in data:
                 raise Exception(f"OpenRouter API Error: {data['error']}")

            choice = data['choices'][0]
            msg_obj = choice['message']
            usage_obj = data.get('usage', {})

            class OpenRouterThoughtResponse:
                def __init__(self, content, reasoning, finish_reason, input_toks, output_toks):
                    self.text = content
                    self.thought = reasoning or ""
                    self.input_tokens = input_toks
                    self.output_tokens = output_toks
                    self.reasoning_tokens = int(len(self.thought) / 3.8) if self.thought else 0

                    mock_part = type('obj', (object,), {'text': content})
                    mock_content = type('obj', (object,), {'parts': [mock_part]})

                    self.candidates = [type('obj', (object,), {
                        'content': mock_content,
                        'finish_reason': type('obj', (object,), {'name': finish_reason})
                    })]

                def __bool__(self): return True

            return OpenRouterThoughtResponse(
                msg_obj.get('content', ''),
                msg_obj.get('reasoning', ''),
                (choice.get('finish_reason') or 'STOP').upper(),
                usage_obj.get('prompt_tokens', 0),
                usage_obj.get('completion_tokens', 0)
            )
        except httpx.RequestError as e:
            raise Exception(f"OpenRouter Network Error: {str(e)}")
        except asyncio.CancelledError:
            raise

_ollama_global_lock = asyncio.Lock()

class OllamaResponse:
    def __init__(self, message_dict, finish_reason):
        self.text = message_dict.get('content', '') or ''
        self.thought = message_dict.get('reasoning', '') or message_dict.get('reasoning_content', '') or ''

        if not self.thought and "<think>" in self.text.lower():
            text_lower = self.text.lower()
            think_start = text_lower.find("<think>")
            think_end = text_lower.find("</think>")

            if think_start != -1:
                if think_end != -1:
                    self.thought = self.text[think_start+7:think_end].strip()
                    self.text = (self.text[:think_start] + self.text[think_end+8:]).strip()
                else:
                    self.thought = self.text[think_start+7:].strip()
                    self.text = self.text[:think_start].strip()

        mock_part = type('obj', (object,), {'text': self.text})
        mock_content = type('obj', (object,), {'parts': [mock_part]})
        self.candidates = [type('obj', (object,), {
            'content': mock_content,
            'finish_reason': type('obj', (object,), {'name': finish_reason})
        })]

    def __bool__(self): return True

class OllamaModel:
    def __init__(self, model_name, api_url=OLLAMA_LOCAL_URL, system_instruction=None, thinking_params=None, **kwargs):
        self.model_name = model_name.replace("OLLAMA/", "").replace("GOOGLE/", "").replace("OPENROUTER/", "")
        self.api_url = api_url.rstrip("/")
        self.system_instruction = system_instruction
        self.thinking_params = thinking_params or {}

    async def generate_content_async(self, contents, generation_config=None, safety_settings=None, stream_state=None):
        import re
        messages = []
        if self.system_instruction:
            messages.append({"role": "system", "content": self.system_instruction})

        for content in contents:
            if isinstance(content, str):
                content = {'role': 'user', 'parts': [content]}

            role = "assistant" if content.get('role', 'user') == "model" else "user"
            text_parts = []
            images = []

            for p in content.get('parts', []):
                if isinstance(p, str) and p.strip():
                    text_parts.append(p)
                elif isinstance(p, dict) and 'mime_type' in p and 'data' in p:
                    mime_type = p['mime_type']
                    if mime_type.startswith("image/"):
                        try:
                            b64_data = base64.b64encode(p['data']).decode('utf-8')
                            images.append(b64_data)
                        except Exception as e:
                            print(f"Error encoding image for Ollama: {e}")
                elif hasattr(p, 'inline_data') and p.inline_data:
                    mime_type = p.inline_data.mime_type
                    if mime_type.startswith("image/"):
                        try:
                            b64_data = base64.b64encode(p.inline_data.data).decode('utf-8')
                            images.append(b64_data)
                        except Exception as e:
                            print(f"Error encoding legacy image for Ollama: {e}")
                elif isinstance(p, dict) and 'url' in p:
                    mime_type = p.get('mime_type', 'image/png')
                    url = p['url']
                    if mime_type.startswith("image/"):
                        if url.startswith(('http://', 'https://')):
                            try:
                                resp = await get_shared_client().get(url, follow_redirects=True, timeout=15.0)
                                resp.raise_for_status()
                                b64_data = base64.b64encode(resp.content).decode('utf-8')
                                images.append(b64_data)
                            except Exception as e:
                                print(f"Ollama failed to fetch and encode remote image {url}: {e}")
                        elif os.path.exists(url):
                            try:
                                with open(url, 'rb') as img_f:
                                    b64_data = base64.b64encode(img_f.read()).decode('utf-8')
                                images.append(b64_data)
                            except Exception as e:
                                print(f"Ollama failed to read local image {url}: {e}")
                        else:
                            match = re.match(r'data:image/[^;]+;base64,(.+)', url)
                            if match:
                                images.append(match.group(1))

            msg_obj = {"role": role, "content": "\n".join(text_parts)}
            if images:
                msg_obj["images"] = images
            messages.append(msg_obj)

        temp = 1.0
        top_p = 1.0
        advanced = {}

        if isinstance(generation_config, dict):
            temp = generation_config.get("temperature", 1.0)
            top_p = generation_config.get("top_p", 1.0)
            advanced = generation_config.get("_advanced_params", {})
        elif generation_config:
            temp = getattr(generation_config, 'temperature', 1.0)
            top_p = getattr(generation_config, 'top_p', 1.0)
            if hasattr(generation_config, '_advanced_params') and generation_config._advanced_params:
                advanced = generation_config._advanced_params

        payload = {
            "model": self.model_name,
            "messages": messages,
            "options": {
                "temperature": temp,
                "top_p": top_p,
            },
            "stream": True
        }

        if advanced:
            if "frequency_penalty" in advanced: payload["options"]["frequency_penalty"] = advanced["frequency_penalty"]
            if "presence_penalty" in advanced: payload["options"]["presence_penalty"] = advanced["presence_penalty"]

        headers = {
            "Content-Type": "application/json",
            "Connection": "keep-alive"
        }

        global _ollama_global_lock
        async with _ollama_global_lock:
            try:
                client = get_shared_client()
                full_content = ""
                reasoning_content = ""
                finish_reason = "STOP"

                async with client.stream("POST", f"{self.api_url}/api/chat", json=payload, headers=headers, timeout=120.0) as response:
                    if response.status_code != 200:
                        err_text = await response.aread()
                        raise Exception(f"Ollama API Error {response.status_code}: {err_text.decode('utf-8', errors='ignore')}")

                    async for line in response.aiter_lines():
                        if not line.strip(): continue
                        try:
                            chunk = json.loads(line)
                            msg = chunk.get("message", {})

                            if "content" in msg and msg["content"]:
                                full_content += msg["content"]
                            if "thinking" in msg and msg["thinking"]:
                                reasoning_content += msg["thinking"]

                            if chunk.get("done"):
                                done_reason = chunk.get("done_reason")
                                if done_reason:
                                    finish_reason = done_reason
                        except Exception:
                            pass

                msg_obj = {"content": full_content, "reasoning": reasoning_content}

                class OllamaResponseWrapper:
                    def __init__(self, m_obj, f_reason, p_eval, e_count):
                        self.text = m_obj.get('content', '') or ''
                        self.thought = m_obj.get('reasoning', '') or ''
                        self.input_tokens = p_eval
                        self.output_tokens = e_count
                        self.reasoning_tokens = int(len(self.thought) / 3.8) if self.thought else 0

                        if not self.thought and "<think>" in self.text.lower():
                            text_lower = self.text.lower()
                            think_start = text_lower.find("<think>")
                            think_end = text_lower.find("</think>")

                            if think_start != -1:
                                if think_end != -1:
                                    self.thought = self.text[think_start+7:think_end].strip()
                                    self.text = (self.text[:think_start] + self.text[think_end+8:]).strip()
                                    self.reasoning_tokens = int(len(self.thought) / 3.8)
                                else:
                                    self.thought = self.text[think_start+7:].strip()
                                    self.text = self.text[:think_start].strip()
                                    self.reasoning_tokens = int(len(self.thought) / 3.8)

                        mock_part = type('obj', (object,), {'text': self.text})
                        mock_content = type('obj', (object,), {'parts': [mock_part]})
                        self.candidates = [type('obj', (object,), {
                            'content': mock_content,
                            'finish_reason': type('obj', (object,), {'name': f_reason})
                        })]

                    def __bool__(self): return True

                # Ollama streaming returns eval counts on the final chunk
                p_eval_count = chunk.get("prompt_eval_count", 0) if 'chunk' in locals() else 0
                eval_count = chunk.get("eval_count", 0) if 'chunk' in locals() else 0

                return OllamaResponseWrapper(msg_obj, (finish_reason or 'STOP').upper(), p_eval_count, eval_count)
            except httpx.RequestError as e:
                raise Exception(f"Ollama Network Error: {str(e)}")
            except asyncio.CancelledError:
                raise


# --- Google REST adapter -------------------------------------------------------
#
# Migration 2, complete. google-genai was already REST over httpx — there was no gRPC
# and this changed no wire format. What it bought is 70 MB of import baseline never
# returned to the OS, and dropping aiohttp/websockets/requests/pydantic from the
# dependency tree; on a 1 GB e2-micro that is 7% of RAM before a message is handled.
#
# This is now the only Google adapter. GoogleSDKModel and the genai.Client cache are
# gone; the migration's GOOGLE_REST_ADAPTER flag is gone with them.

_GOOGLE_API_BASE = "https://generativelanguage.googleapis.com"

# One client for every Google REST call rather than one per request: building an
# httpx.AsyncClient costs ~14 ms and ~0.8 MB, and nothing here varies per call — the
# API key travels as a per-request header, not in the client. Created lazily so it
# binds to the running event loop, and closed from MimicCog.cog_unload.
_google_rest_client: Optional[httpx.AsyncClient] = None


def get_google_rest_client() -> httpx.AsyncClient:
    global _google_rest_client
    if _google_rest_client is None or _google_rest_client.is_closed:
        _google_rest_client = httpx.AsyncClient(
            base_url=_GOOGLE_API_BASE,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
    return _google_rest_client


async def close_google_rest_client():
    global _google_rest_client
    if _google_rest_client is not None and not _google_rest_client.is_closed:
        await _google_rest_client.aclose()
    _google_rest_client = None


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


# 256 KB per read. Large enough that a 6 MB attachment costs ~24 reads rather
# than ~750, small enough that each one stays well under the ~10 ms event-loop
# budget — a read of this size from the page cache (and the file was written
# moments earlier, so it is hot) costs tens of microseconds.
_UPLOAD_CHUNK_BYTES = 256 * 1024

# Download chunk for the streaming fetch. 8 KB meant ~750 allocation/append
# cycles for a single phone photo; 64 KB cuts that by a factor of eight without
# meaningfully raising the peak.
_DOWNLOAD_CHUNK_BYTES = 64 * 1024


async def _aiter_file_bytes(path: str, chunk_size: int = _UPLOAD_CHUNK_BYTES):
    """Yields `path` in chunks for use as an httpx request body.

    Must be an *async* generator: httpx refuses a sync iterable body on an
    AsyncClient. httpx would normally pair an async body with
    `Transfer-Encoding: chunked`, but `Request._prepare` skips that default when
    Content-Length is already set explicitly — which the caller does, from
    `os.path.getsize`. So the request goes out byte-identical to the buffered
    version it replaces, and the resumable protocol sees no difference.
    """
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


# --- Gemini File API URI cache -------------------------------------------------
#
# _build_parts resolves every {'url': ...} part by downloading the bytes and
# uploading them to the File API, and it runs once per participant per round. A
# four-profile round therefore downloaded and re-uploaded the same attachment four
# times, and each upload buffers the whole file — which is exactly the "memory grows
# with participant count" behaviour on the e2-micro. The bytes are identical every
# time, so resolve once and share the URI for the rest of the round.
#
# Bounded and TTL'd. File API entries expire server-side after 48 h, so a cached URI
# must never outlive that; 30 minutes is far inside it and comfortably longer than
# any single round.

_FILE_URI_CACHE_MAX = 32
_FILE_URI_CACHE_TTL = 1800.0
_file_uri_cache: "OrderedDict[Any, Tuple[float, str]]" = OrderedDict()

# Single-flight: participants in a round are sequential, but two channels can be
# mid-round on the same image at once. Waiters share one upload instead of racing.
_file_uri_inflight: dict = {}


def _file_cache_key(source: str, mime_type: str):
    """Cache key for one media source.

    Local paths key on a content stamp as well as the name: these are mkstemp temp
    files that are deleted at end of round, and the name can be handed out again
    afterwards. A stale hit on a reused name would attach the wrong image to a turn.
    """
    if source.startswith(("http://", "https://")):
        return (source, mime_type)
    try:
        st = os.stat(source)
        return (source, st.st_size, st.st_mtime_ns, mime_type)
    except OSError:
        return (source, mime_type)


def _file_uri_cache_get(key) -> Optional[str]:
    entry = _file_uri_cache.get(key)
    if entry is None:
        return None
    stamped_at, uri = entry
    if time.monotonic() - stamped_at > _FILE_URI_CACHE_TTL:
        _file_uri_cache.pop(key, None)
        return None
    _file_uri_cache.move_to_end(key)
    return uri


def _file_uri_cache_put(key, uri: str):
    _file_uri_cache[key] = (time.monotonic(), uri)
    _file_uri_cache.move_to_end(key)
    while len(_file_uri_cache) > _FILE_URI_CACHE_MAX:
        _file_uri_cache.popitem(last=False)


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
            try:
                return base64.b64decode(value)
            except Exception:
                return value
    return value


class GoogleRESTModel:
    """Google Gemini over raw REST, satisfying the same adapter interface as
    OpenRouterModel and OllamaModel: generate_content_async(contents,
    generation_config, ...) returning an object with .text, .thought, .candidates,
    .prompt_feedback and token counts.
    """

    def __init__(self, api_key, model_name, system_instruction=None, safety_settings=None, thinking_params=None, tools=None, image_params=None):
        self.api_key = api_key
        self.model_name = model_name.replace("OPENROUTER/", "").replace("GOOGLE/", "")
        self.system_instruction = system_instruction
        self.safety_settings = safety_settings
        self.thinking_params = thinking_params or {}
        self.tools = tools
        #: Output controls for an image request: aspect_ratio, image_size,
        #: thinking_level. Carried on the model rather than passed per call because
        #: MediaService.build_image_model is the one constructor every image path goes
        #: through, and none of those three call sites builds a generation_config.
        #: Already validated against IMAGE_MODEL_CAPS by the time it arrives -- this
        #: class only maps it onto the wire shape.
        self.image_params = image_params or {}

    # -- media ------------------------------------------------------------------

    async def _upload_file(self, path: str, mime_type: str) -> Optional[str]:
        """Uploads a file already on disk, retrying once on a transport failure.

        The retry exists because the body streams from a generator (see
        `_upload_file_once`). httpcore will transparently re-issue a request whose
        body is a plain `bytes` when the pooled connection turns out to be dead,
        but it cannot replay a consumed async generator — so a connection that
        went away between the start request and the finalize surfaces as a
        `TransportError` here instead of being retried underneath us. Redoing the
        whole two-request sequence gets a fresh upload session, so there is no
        ambiguity about how much of the previous body the server accepted.
        """
        last_exc = None
        for attempt in range(2):
            try:
                return await self._upload_file_once(path, mime_type)
            except (httpx.TransportError, httpx.RemoteProtocolError) as e:
                last_exc = e
                if attempt == 0:
                    print(
                        f"File API upload of {os.path.basename(path)} hit "
                        f"{type(e).__name__}({e or 'no detail'}); retrying once."
                    )
                    continue
                raise
        raise last_exc  # unreachable; keeps the type checker honest

    async def _upload_file_once(self, path: str, mime_type: str) -> Optional[str]:
        """Uploads a file already on disk via the resumable upload protocol and
        returns its file URI.

        Two requests: a "start" that returns an upload URL in the
        X-Goog-Upload-URL response header, then an "upload, finalize" carrying
        the bytes. The body streams off disk in `_UPLOAD_CHUNK_BYTES` pieces
        rather than being read into one buffer: the caller staged the file
        precisely so the *download* never sat in RAM in full, and buffering it
        back for the upload gave that right back. This is not the hand-rolled
        chunked protocol the previous comment here warned against — the
        Content-Length header below keeps httpx off `Transfer-Encoding: chunked`,
        so the request on the wire is identical to the buffered version. Polls
        the file resource's `state` only if the API reports PROCESSING; small
        image/audio files finalize as ACTIVE immediately and skip the poll
        entirely.
        """
        file_size = os.path.getsize(path)
        client = get_google_rest_client()

        start_resp = await client.post(
            "/upload/v1beta/files",
            headers={
                "x-goog-api-key": self.api_key,
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(file_size),
                "X-Goog-Upload-Header-Content-Type": mime_type,
                "Content-Type": "application/json",
            },
            content=json.dumps({"file": {"display_name": os.path.basename(path)}}),
        )
        if start_resp.status_code != 200:
            raise Exception(f"Google API Error {start_resp.status_code}: {start_resp.text}")

        upload_url = start_resp.headers.get("x-goog-upload-url")
        if not upload_url:
            raise Exception("Google API Error: resumable upload start returned no upload URL")

        upload_resp = await client.post(
            upload_url,
            headers={
                # Explicit, and load-bearing: it suppresses httpx's chunked
                # transfer-encoding default for a streamed body.
                "Content-Length": str(file_size),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            content=_aiter_file_bytes(path),
        )
        if upload_resp.status_code != 200:
            raise Exception(f"Google API Error {upload_resp.status_code}: {upload_resp.text}")

        file_resource = json.loads(upload_resp.content).get("file", {}) or {}

        poll_attempts = 0
        while file_resource.get("state") == "PROCESSING" and file_resource.get("name") and poll_attempts < 10:
            await asyncio.sleep(1.0)
            poll_resp = await client.get(f"/v1beta/{file_resource['name']}", headers={"x-goog-api-key": self.api_key})
            if poll_resp.status_code != 200:
                break
            file_resource = json.loads(poll_resp.content)
            poll_attempts += 1

        return file_resource.get("uri")

    async def _build_parts(self, raw_parts) -> List[dict]:
        parts = []
        for p in raw_parts:
            if isinstance(p, str):
                parts.append({"text": p})
            elif isinstance(p, dict) and 'mime_type' in p and 'data' in p:
                parts.append({
                    "inlineData": {
                        "mimeType": p['mime_type'],
                        "data": base64.b64encode(p['data']).decode('ascii'),
                    }
                })
            elif isinstance(p, dict) and 'url' in p:
                url = p['url']
                mime_type = p.get('mime_type', '')
                is_remote = url.startswith(('http://', 'https://'))
                if is_remote or os.path.exists(url):
                    resolved_mime = mime_type or ('image/jpeg' if is_remote else 'image/png')
                    file_uri = await self._resolve_media_uri(url, resolved_mime, is_remote)
                    if file_uri:
                        parts.append({"fileData": {"fileUri": file_uri, "mimeType": resolved_mime}})
                else:
                    parts.append({"fileData": {"fileUri": url, "mimeType": mime_type}})
        return parts

    async def _resolve_media_uri(self, url: str, mime_type: str, is_remote: bool) -> Optional[str]:
        """Returns a File API URI for `url`, reusing a cached one when the same
        bytes were already uploaded. Returns None on failure rather than raising —
        a media part that cannot be resolved is dropped, as it was before.
        """
        key = _file_cache_key(url, mime_type)

        cached = _file_uri_cache_get(key)
        if cached:
            return cached

        inflight = _file_uri_inflight.get(key)
        if inflight is not None:
            # Another round is already uploading these exact bytes. Shielded so a
            # cancelled waiter does not kill the upload the others are waiting on.
            try:
                return await asyncio.shield(inflight)
            except Exception:
                return None

        future = asyncio.get_running_loop().create_future()
        _file_uri_inflight[key] = future
        uri = None
        try:
            if is_remote:
                uri = await self._download_and_upload(url, mime_type)
            else:
                uri = await self._upload_file(url, mime_type)
            if uri:
                _file_uri_cache_put(key, uri)
        except Exception as e:
            # An unresolvable media part is dropped, never fatal to the turn — the
            # per-URL try/except this replaced behaved the same way. CancelledError
            # is a BaseException and still propagates, as it must.
            print(
                f"Failed to resolve media {url} for the Gemini File API: "
                f"{type(e).__name__}({e or 'no detail'})"
            )
            uri = None
        finally:
            # Media transfers churn the allocator harder than anything else the
            # bot does. Hand the freed pages back rather than letting them sit at
            # the top of an arena until the process exits. Placed here rather than
            # in _download_and_upload so the local-file branch -- which is how a
            # *generated* image reaches the model -- is covered too. Rate-limited,
            # so a multi-participant round pays for this once.
            maybe_trim_malloc()
            _file_uri_inflight.pop(key, None)
            if not future.done():
                # Resolved rather than raised: waiters take the same None-means-drop
                # path, and nothing is left as an unretrieved exception.
                future.set_result(uri)
        return uri

    async def _download_and_upload(self, url: str, mime_type: str) -> Optional[str]:
        temp_path = None
        # Which half failed matters: a bad Discord CDN URL and a failed File API
        # upload need completely different investigation, and this used to report
        # both as "Failed to fetch media from URL".
        stage = "download"
        try:
            import tempfile

            # Streamed to disk rather than buffered, so a large attachment
            # never sits in RAM in full.
            fd, temp_path = tempfile.mkstemp(suffix=".tmp")
            client_http = get_google_rest_client()
            async with client_http.stream("GET", url, follow_redirects=True, timeout=15.0) as resp:
                resp.raise_for_status()
                with os.fdopen(fd, 'wb') as f:
                    async for chunk in resp.aiter_bytes(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                        f.write(chunk)

            stage = "upload"
            return await self._upload_file(temp_path, mime_type)
        except Exception as e:
            # Transport errors out of httpx/anyio frequently carry an empty
            # message -- a bare `{e}` then prints nothing at all and the log line
            # is useless. The type is always worth having.
            detail = str(e) or "no detail"
            size = ""
            if stage == "upload" and temp_path:
                try:
                    size = f", {os.path.getsize(temp_path)} bytes staged"
                except OSError:
                    pass
            print(f"Media {stage} failed for {url}{size}: {type(e).__name__}({detail})")
            return None
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    # -- request ----------------------------------------------------------------

    async def _build_contents(self, contents) -> List[dict]:
        formatted = []
        for item in contents:
            if isinstance(item, str):
                formatted.append({"role": "user", "parts": [{"text": item}]})
            elif isinstance(item, dict):
                parts = await self._build_parts(item.get('parts', []))
                formatted.append({"role": item.get('role', 'user'), "parts": parts})
            elif hasattr(item, 'role') and hasattr(item, 'parts'):
                # Fallback for legacy SDK objects still reaching the adapter.
                new_parts = []
                for p in item.parts:
                    if getattr(p, 'text', None):
                        new_parts.append({"text": p.text})
                    elif getattr(p, 'inline_data', None):
                        new_parts.append({
                            "inlineData": {
                                "mimeType": p.inline_data.mime_type,
                                "data": base64.b64encode(p.inline_data.data).decode('ascii'),
                            }
                        })
                formatted.append({"role": item.role, "parts": new_parts})
        return formatted

    def _build_generation_config(self, generation_config) -> dict:
        model_lower = self.model_name.lower()

        # Utility models do not accept a thinking config at all.
        is_utility_model = any(suffix in model_lower for suffix in ["-image", "-tts", "-embedding"])
        include_thoughts = self.thinking_params.get("thinking_summary_visible") == "on"

        if isinstance(generation_config, dict):
            temp = generation_config.get("temperature")
            top_p = generation_config.get("top_p")
            top_k = generation_config.get("top_k")
            if generation_config.get("thinking_config"):
                include_thoughts = generation_config["thinking_config"].get("include_thoughts", include_thoughts)
        else:
            temp = generation_config.temperature if generation_config else None
            top_p = generation_config.top_p if generation_config else None
            top_k = generation_config.top_k if generation_config else None
            if generation_config and getattr(generation_config, 'thinking_config', None):
                include_thoughts = generation_config.thinking_config.include_thoughts

        cfg = {}
        if temp is not None:
            cfg["temperature"] = temp
        if top_p is not None:
            cfg["topP"] = top_p
        if top_k is not None:
            cfg["topK"] = top_k

        if not is_utility_model:
            if "gemini-3" in model_lower:
                lvl = self.thinking_params.get("thinking_level", "high").lower()
                if "pro" in model_lower:
                    mapped_lvl = "LOW" if lvl in ["low", "minimal", "none"] else "HIGH"
                else:
                    mapped_lvl = {
                        "xhigh": "HIGH", "high": "HIGH", "medium": "MEDIUM",
                        "low": "LOW", "minimal": "MINIMAL", "none": "MINIMAL"
                    }.get(lvl, "HIGH")
                cfg["thinkingConfig"] = {"includeThoughts": include_thoughts, "thinkingLevel": mapped_lvl}
            elif "gemini-2.5" in model_lower:
                budget = int(self.thinking_params.get("thinking_budget", -1))
                if "lite" not in model_lower:
                    if "pro" in model_lower and 0 <= budget < 128:
                        budget = 128
                    cfg["thinkingConfig"] = {"includeThoughts": include_thoughts, "thinkingBudget": budget}

        # Image output controls. Sits outside the is_utility_model branch above on
        # purpose: an image model rejects the *text* thinking config, but the 3.x image
        # models do take a thinkingLevel of their own, and only when one was explicitly
        # chosen -- an absent key means "let the model use its own default", which is
        # not the same as sending MINIMAL.
        if self.image_params:
            # Pinned so an image model cannot answer with text alone. Which combination
            # is legal is per model -- see IMAGE_MODEL_CAPS -- and an unknown model gets
            # none of this rather than a guess.
            if self.image_params.get("modalities"):
                cfg["responseModalities"] = list(self.image_params["modalities"])

            # `imageConfig`, not `responseFormat.image`. v1beta carries both, and they
            # are not the same field: ResponseFormatConfig.image is a strict protobuf
            # enum wanting ASPECT_RATIO_NINE_BY_SIXTEEN and IMAGE_SIZE_FIVE_TWELVE,
            # while ImageConfig takes the "9:16" / "512" strings the docs show. The
            # published curl examples send the plain strings to responseFormat because
            # they post to /v1; this client posts to /v1beta, where that combination is
            # a 400 naming both fields. Checked against the v1beta discovery document.
            image_cfg = {}
            if self.image_params.get("aspect_ratio"):
                image_cfg["aspectRatio"] = self.image_params["aspect_ratio"]
            if self.image_params.get("image_size"):
                image_cfg["imageSize"] = self.image_params["image_size"]
            if image_cfg:
                cfg["imageConfig"] = image_cfg
            level = self.image_params.get("thinking_level")
            if level:
                # includeThoughts stays off: the thought parts on an image request are
                # interstitial draft images, and _write_img takes the first inline_data
                # part it finds. Billed either way, per the API docs.
                cfg["thinkingConfig"] = {"includeThoughts": False, "thinkingLevel": level}

        return cfg

    def _build_safety_settings(self) -> List[dict]:
        out = []
        if self.safety_settings:
            for cat, thresh in self.safety_settings.items():
                out.append({
                    "category": cat.name if hasattr(cat, 'name') else str(cat),
                    "threshold": thresh.name if hasattr(thresh, 'name') else str(thresh),
                })
        return out

    def _build_tools(self) -> Optional[List[dict]]:
        """Tool declarations arrive as snake_case dicts ({"google_search": {}}),
        which the SDK converted for us. REST wants camelCase keys.
        """
        if not self.tools:
            return None
        out = []
        for tool in self.tools:
            if isinstance(tool, dict):
                out.append({_to_camel(k): v for k, v in tool.items()})
            else:
                out.append(tool)
        return out or None

    async def generate_content_async(self, contents, generation_config=None, stream_state=None):
        payload = {"contents": await self._build_contents(contents)}

        if self.system_instruction:
            # role is what the SDK sends here; the API ignores it, but matching keeps
            # the two adapters' wire payloads diffable while both are live.
            payload["systemInstruction"] = {"role": "user", "parts": [{"text": self.system_instruction}]}

        safety = self._build_safety_settings()
        if safety:
            payload["safetySettings"] = safety

        tools = self._build_tools()
        if tools:
            payload["tools"] = tools

        gen_cfg = self._build_generation_config(generation_config)
        if gen_cfg:
            payload["generationConfig"] = gen_cfg

        model_path = self.model_name if self.model_name.startswith("models/") else f"models/{self.model_name}"

        try:
            client = get_google_rest_client()
            response = await client.post(
                f"/v1beta/{model_path}:generateContent",
                content=json.dumps(payload),
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            )
        except httpx.RequestError as e:
            raise Exception(f"Google API Network Error: {str(e)}")
        except asyncio.CancelledError:
            raise
        finally:
            payload.clear()

        if response.status_code != 200:
            # The body carries the status name ("RESOURCE_EXHAUSTED") that
            # helpers._get_friendly_api_error matches on, so pass it through intact.
            raise Exception(f"Google API Error {response.status_code}: {response.text}")

        return GoogleRESTResponse(json.loads(response.content))


class GoogleRESTResponse:
    """Normalises a parsed generateContent body into the shared adapter interface.

    Presents the same attribute surface the SDK adapter's ThoughtResponse did, so
    the call sites that reach past the shared interface into .raw keep working.
    """

    def __init__(self, body: dict):
        self.raw = _RestView(body)
        self.text = ""
        self.thought = ""
        self.candidates = self.raw.candidates or []
        self.prompt_feedback = self.raw.prompt_feedback
        self.usage_metadata = self.raw.usage_metadata

        self.input_tokens = (self.usage_metadata.prompt_token_count or 0) if self.usage_metadata else 0
        self.output_tokens = (self.usage_metadata.candidates_token_count or 0) if self.usage_metadata else 0

        if self.candidates and self.candidates[0].content and self.candidates[0].content.parts:
            for part in self.candidates[0].content.parts:
                if part.thought:
                    self.thought += part.text or ""
                elif part.text:
                    self.text += part.text

        self.reasoning_tokens = int(len(self.thought) / 3.8) if self.thought else 0

    def __bool__(self):
        return bool(self.candidates)



# Migration 2 step 3. TTS calls generateContent directly with a response_modalities /
# speechConfig shape that no other caller needs, so it stays outside the shared
# generate_content_async interface rather than widening it for one consumer. Same
# endpoint, same routing switch as GoogleGenAIModel and get_embedding_vector above.
async def generate_google_tts_audio(
    api_key: str,
    model_id: str,
    text: str,
    voice_name: str = DEFAULT_SPEECH_VOICE,
    temperature: float = 1.0,
) -> Optional[bytes]:
    """Returns raw PCM audio bytes for `text`.

    Raises on network/API failure — same contract as generate_content_async — so
    media_service's existing try/except keeps handling errors uniformly. A response
    that parsed cleanly but carried no audio raises too: see the retry loop below.
    """
    if model_id.upper().startswith("GOOGLE/"):
        model_id = model_id[7:]

    payload = {
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "temperature": temperature,
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": voice_name}
                }
            },
        },
    }
    model_path = model_id if model_id.startswith("models/") else f"models/{model_id}"
    client = get_google_rest_client()

    # Two attempts on the *same* model, which is not the retry run_with_fallback
    # performs: that one moves to a different model, and skips entirely when no second
    # model is configured. Google documents a failure mode this cannot cover -- the
    # 3.1 TTS model occasionally emits text tokens instead of audio, which the server
    # answers with a 500, "randomly in a very small percentage of requests", with an
    # explicit recommendation to retry. Retrying the same model is the only thing that
    # helps there; a fallback model would be answering a fault the primary does not
    # actually have.
    last_error = None
    for attempt in range(2):
        try:
            response = await client.post(
                f"/v1beta/{model_path}:generateContent",
                content=json.dumps(payload),
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            )
        except httpx.RequestError as e:
            raise Exception(f"Google API Network Error: {str(e)}")

        if response.status_code != 200:
            error = Exception(f"Google API Error {response.status_code}: {response.text}")
            # 4xx is a bad request -- a voice that does not exist, a model that is not a
            # TTS model, an exhausted quota. Sending it again changes nothing.
            if response.status_code < 500 or attempt == 1:
                raise error
            last_error = error
            print(f"Google TTS: {model_id} returned {response.status_code}; retrying once.")
            continue

        parsed = GoogleRESTResponse(json.loads(response.content))
        if parsed.candidates and parsed.candidates[0].content and parsed.candidates[0].content.parts:
            for part in parsed.candidates[0].content.parts:
                if getattr(part, 'inline_data', None) and part.inline_data.data:
                    return part.inline_data.data

        # A 200 carrying no audio part is the same fault surfacing without the 500.
        # Raising rather than returning None is deliberate: None reads as "this text
        # produced no speech" and stops there, so a configured fallback model was never
        # tried for what is a transient fault on the primary.
        last_error = Exception(f"Google TTS Error: {model_id} returned no audio data.")
        if attempt == 0:
            print(f"Google TTS: {model_id} returned no audio; retrying once.")

    raise last_error


# --- Google adapter routing ----------------------------------------------------
#
# Retained as an alias because ten construction sites across six files import this
# name. Migration 2 is complete and GoogleRESTModel is the only Google adapter, so
# there is nothing left to route — but renaming those ten sites is a separate diff
# from deleting the SDK, and this file is the one that had to change.
GoogleGenAIModel = GoogleRESTModel


class APIService:
    """Owns model-instantiation routing: resolves a raw model name (GOOGLE/, OPENROUTER/,
    OLLAMA/, or bare) and the caller's API key into the correct adapter instance.

    Holds a back-reference to the parent cog for state/logic not yet migrated
    (API key resolution), per the transitional Dependency Injection pattern in
    CLAUDE.md.
    """

    def __init__(self, cog):
        self.cog = cog

    def _instantiate_model(self, raw_model_name: str, guild_id, user_id, system_instruction=None, safety_settings=None, thinking_params=None, tools=None, profile_settings=None, openrouter_key_error: str = None, google_key_error: str = None, use_broad_openrouter_heuristic: bool = True):
        # System prefixes 'GOOGLE/', 'OPENROUTER/', and 'OLLAMA/' are strictly case-sensitive.
        # OpenRouter hosts models under lowercase creator namespaces like 'google/gemini-2.5-flash'.
        actual_name = raw_model_name
        is_openrouter = False
        is_ollama = False

        if raw_model_name.startswith("OPENROUTER/"):
            actual_name = raw_model_name[11:]
            is_openrouter = True
        elif raw_model_name.startswith("OLLAMA/"):
            actual_name = raw_model_name[7:]
            is_ollama = True
        elif raw_model_name.startswith("GOOGLE/"):
            actual_name = raw_model_name[7:]
        elif "/" in raw_model_name or (use_broad_openrouter_heuristic and ("grok" in raw_model_name.lower() or "anthropic" in raw_model_name.lower())):
            is_openrouter = True

        t_params = thinking_params or {}
        p_settings = profile_settings or {}

        if is_openrouter:
            api_key = self.cog.storage_manager._get_api_key_for_guild(guild_id, "openrouter") if guild_id else self.cog.storage_manager._get_api_key_for_user(user_id, "openrouter")
            if not api_key: raise ValueError(openrouter_key_error or "OpenRouter API Key not found. Use `/settings` to add one.")
            model = OpenRouterModel(actual_name, api_key=api_key, system_instruction=system_instruction, thinking_params=t_params)
            return _with_key_cooldown_tracking(self.cog, model, api_key)
        elif is_ollama:
            ollama_host = p_settings.get("ollama_host_url", OLLAMA_LOCAL_URL)
            return OllamaModel(actual_name, api_url=ollama_host, system_instruction=system_instruction, thinking_params=t_params)
        else:
            api_key = self.cog.storage_manager._get_api_key_for_guild(guild_id) if guild_id else self.cog.storage_manager._get_api_key_for_user(user_id)
            if not api_key: raise ValueError(google_key_error or "Google API Key not found. Use `/settings` to add one.")
            model = GoogleGenAIModel(api_key=api_key, model_name=actual_name, system_instruction=system_instruction, safety_settings=safety_settings, thinking_params=t_params, tools=tools)
            return _with_key_cooldown_tracking(self.cog, model, api_key)

    async def run_with_fallback(self, primary: str, fallback: Optional[str], attempt,
                                *, label: str = "utility"):
        """Runs one utility generation on `primary`, retrying once on `fallback`.

        `attempt(model_name, is_fallback)` owns the construction and the response
        handling; this owns only which name to try and when to stop. That puts the five
        utility paths -- critic, grounding, LTM, image and speech -- on one retry policy
        without forcing them into one call shape, which they genuinely do not share: one
        returns audio bytes, one drives a heartbeat, three return a candidate list.

        Only exceptions retry. An empty or safety-blocked response is a decision about
        the content, not a statement about the model being unavailable, and re-rolling
        it on a second model spends another call to be refused again.

        A fallback equal to the primary is skipped rather than tried twice, which is
        what makes the shipped defaults cost nothing: every utility fallback defaults to
        the same model as its primary, so the retry only becomes live once someone
        actually changes one of them.

        Returns (result, model_used, used_fallback). If both fail the second error is
        raised -- callers report the error they are handed, and the fallback's is the
        one that actually ended the attempt.
        """
        attempts = [(primary, False)]
        if is_real_model(fallback) and fallback != primary:
            attempts.append((fallback, True))

        last_error = None
        for name, is_fallback in attempts:
            try:
                return await attempt(name, is_fallback), name, is_fallback
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_error = e
                if not is_fallback and len(attempts) > 1:
                    print(f"{label}: primary '{name}' failed "
                          f"({type(e).__name__}: {e}); retrying on '{attempts[1][0]}'.")
        raise last_error

    def get_top_models(self, provider: str, target_config_key: str) -> List[str]:
        if target_config_key in IMAGE_MODEL_KEYS: return list(get_args(IMAGE_MODELS))
        if target_config_key in AUDIO_MODEL_KEYS: return list(get_args(AUDIO_MODELS))
        if provider == 'google': return list(get_args(ALLOWED_MODELS))
        elif provider == 'ollama': return getattr(self, 'cached_ollama_models', [])

        import json as std_json
        path = os.path.join(self.cog.MODELS_DATA_DIR, "openrouter_models.json")
        data = {}
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f: data = std_json.load(f)
            except: pass

        sorted_models = sorted(data.items(), key=lambda x: x[1], reverse=True)
        return [m[0] for m in sorted_models]

    async def _get_or_create_model_for_channel(self, channel_id: int, actual_message_author_id: int, guild_id: int, profile_owner_override: Optional[int] = None, profile_name_override: Optional[str] = None, prompt_content: Optional[str] = None) -> Tuple[Optional[Any], bool, float, float, int, Optional[str], Optional[str]]:
        
        api_key = self.cog.storage_manager._get_api_key_for_guild(guild_id)
        
        if profile_owner_override is not None and profile_name_override is not None:
            profile_owner_id_for_instructions = profile_owner_override
            profile_name_for_instructions = profile_name_override
        else:
            profile_owner_id_for_instructions: Optional[int] = actual_message_author_id
            profile_name_for_instructions: str = self.cog.session_manager._get_active_user_profile_name_for_channel(profile_owner_id_for_instructions, channel_id)
        
        channel = self.cog.bot.get_channel(channel_id)
        if not channel:
            return None, True, 0.0, 0.0, 0, "Could not find the channel for this interaction.", None

        if not self.cog.profile_manager._check_unrestricted_safety_policy(profile_owner_id_for_instructions, profile_name_for_instructions, channel):
            return None, True, 0.0, 0.0, 0, "Profiles with 'Unrestricted 18+' safety can only be used in age-restricted channels.", None

        model_cache_key = (channel_id, profile_owner_id_for_instructions, profile_name_for_instructions)

        user_index = self.cog.profile_manager._get_user_index(profile_owner_id_for_instructions)
        is_borrowed = profile_name_for_instructions in user_index.get("borrowed", [])

        original_owner_id, original_profile_name = self.cog.profile_manager._resolve_effective_profile(profile_owner_id_for_instructions, profile_name_for_instructions)

        current_profile_key_for_model = (original_owner_id, original_profile_name)

        training_examples_list = []
        if prompt_content:
            training_examples_list = await self.cog.memory_manager._get_relevant_training_examples(
                profile_owner_id_for_instructions,
                profile_name_for_instructions,
                prompt_content,
                guild_id
            )

        current_instructions, error_in_instr_constr, _, temperature, top_p, top_k, primary_model, fallback_model = self.cog.generation_service._construct_system_instructions(
            profile_owner_id_for_instructions,
            profile_name_for_instructions,
            channel_id,
            training_examples_list=training_examples_list
        )
        
        if not api_key and not primary_model.upper().startswith("OLLAMA/"):
             return None, True, 0.0, 0.0, 0, "Server API key is not configured.", None
        
        warning_message = None

        recreate_model = True
        if model_cache_key in self.cog.channel_models and not training_examples_list:
            last_profile_key = self.cog.channel_model_last_profile_key.get(model_cache_key)
            if last_profile_key == current_profile_key_for_model:
                 recreate_model = False 
            
        if recreate_model and model_cache_key in self.cog.channel_models:
            del self.cog.channel_models[model_cache_key]
            self.cog.channel_model_last_profile_key.pop(model_cache_key, None)
        
        if model_cache_key in self.cog.channel_models and not recreate_model: 
            model_instance, model_init_error_state, cached_model_name = self.cog.channel_models[model_cache_key]
            return model_instance, model_init_error_state, temperature, top_p, top_k, warning_message, fallback_model

        model_instance, model_init_error = None, True
        
        profile_data_for_safety = self.cog.profile_manager._get_profile_config(profile_owner_id_for_instructions, profile_name_for_instructions, is_borrowed) or {}
        dynamic_safety_settings = _resolve_safety_settings(channel, profile_data_for_safety)

        model_to_create = primary_model
        
        # Extract parameters once for either provider
        p_sett_thinking = self.cog.profile_manager._get_profile_config(profile_owner_id_for_instructions, profile_name_for_instructions, is_borrowed) or {}
        t_params = {
            "thinking_summary_visible": p_sett_thinking.get("thinking_summary_visible", "off"),
            "thinking_level": p_sett_thinking.get("thinking_level", "high"),
            "thinking_budget": p_sett_thinking.get("thinking_budget", -1)
        }

        # [NEW] Native Tool Construction
        grounding_mode = p_sett_thinking.get("grounding_mode", "off")
        if isinstance(grounding_mode, bool): grounding_mode = "rag" if grounding_mode else "off"
        elif grounding_mode in ["on", "on+"]: grounding_mode = "rag"
        
        url_mode = p_sett_thinking.get("url_mode", "off")
        if "url_mode" not in p_sett_thinking:
            url_mode = "rag" if p_sett_thinking.get("url_fetching_enabled", False) else "off"

        model_tools_list = []
        if grounding_mode == "native":
            model_tools_list.append({"google_search": {}})
        if url_mode == "native":
            model_tools_list.append({"url_context": {}})
            
        model_tools = model_tools_list if model_tools_list else None

        try:
            model_instance = self._instantiate_model(model_to_create, guild_id, profile_owner_id_for_instructions, current_instructions, dynamic_safety_settings, t_params, model_tools, p_sett_thinking)
            model_init_error = False
        except Exception as e1:
            print(f"Err '{model_to_create}' key {model_cache_key}: {e1}. Fallback.")
            model_to_create = fallback_model
            try:
                model_instance = self._instantiate_model(model_to_create, guild_id, profile_owner_id_for_instructions, current_instructions, dynamic_safety_settings, t_params, model_tools, p_sett_thinking)
                model_init_error = False
            except Exception as e2:
                return None, True, temperature, top_p, top_k, f"Model Initialization Error: Failed to load Primary ('{primary_model}') and Fallback ('{fallback_model}') models. Check your API key.", fallback_model
        
        final_error_state = error_in_instr_constr or model_init_error
        self.cog.channel_models[model_cache_key] = (model_instance, final_error_state, model_to_create)
        self.cog.channel_model_last_profile_key[model_cache_key] = current_profile_key_for_model
        return model_instance, final_error_state, temperature, top_p, top_k, warning_message, fallback_model

    async def _get_or_create_model_for_global_chat(self, user_id: int, profile_name: str) -> Tuple[Optional[Any], float, float, int, Optional[str], Optional[str]]:
        source_owner_id, source_profile_name = self.cog.profile_manager._resolve_effective_profile(user_id, profile_name)
        
        profile_data = self.cog.profile_manager._get_profile_config(source_owner_id, source_profile_name, False)
        if not profile_data:
            return None, 0.0, 0.0, 0, f"The source for your active global profile ('{profile_name}') could not be found.", None

        temp = profile_data.get("temperature", defaultConfig.GEMINI_TEMPERATURE)
        top_p = profile_data.get("top_p", defaultConfig.GEMINI_TOP_P)
        top_k = profile_data.get("top_k", defaultConfig.GEMINI_TOP_K)
        primary_model = profile_data.get("primary_model", PRIMARY_MODEL_NAME)
        fallback_model = profile_data.get("fallback_model", FALLBACK_MODEL_NAME)
        
        warning_message = None
        system_instructions, _, _, _, _, _, _, _ = self.cog.generation_service._construct_system_instructions(user_id, profile_name, 0)
        
        user_api_key = self.cog.storage_manager._get_api_key_for_user(user_id, "gemini")
        or_key = self.cog.storage_manager._get_api_key_for_user(user_id, "openrouter")
        
        if not user_api_key and not or_key and not primary_model.upper().startswith("OLLAMA/"):
            return None, 0.0, 0.0, 0, "This feature requires a personal API key. Use `/settings` to add one.", None
        
        # Global chat runs in a DM, which can never be age-restricted -- and an
        # adult-rated profile is refused from the feature outright. Passing the
        # absent channel resolves to BLOCK_ONLY_HIGH, which is what this path
        # already sent.
        safety_settings = _resolve_safety_settings(None, profile_data)

        try:
            t_params = {
                "thinking_summary_visible": profile_data.get("thinking_summary_visible", "off"),
                "thinking_level": profile_data.get("thinking_level", "high"),
                "thinking_budget": profile_data.get("thinking_budget", -1)
            }
            
            model_tools = None
            if not primary_model.upper().startswith(("OPENROUTER/", "OLLAMA/")) and "/" not in primary_model:
                grounding_mode = profile_data.get("grounding_mode", "off")
                if isinstance(grounding_mode, bool): grounding_mode = "rag" if grounding_mode else "off"
                elif grounding_mode in ["on", "on+"]: grounding_mode = "rag"
                
                url_mode = profile_data.get("url_mode", "off")
                if "url_mode" not in profile_data:
                    url_mode = "rag" if profile_data.get("url_fetching_enabled", False) else "off"

                model_tools_list = []
                if grounding_mode == "native":
                    model_tools_list.append({"google_search": {}})
                if url_mode == "native":
                    model_tools_list.append({"url_context": {}})
                    
                model_tools = model_tools_list if model_tools_list else None
                    
            model = self._instantiate_model(primary_model, None, user_id, system_instructions, safety_settings, t_params, model_tools, profile_data)
            
            return model, temp, top_p, top_k, warning_message, fallback_model
        except Exception as e:
            print(f"Error creating model for global chat (user: {user_id}, profile: {profile_name}): {e}")
            return None, 0.0, 0.0, 0, "A critical error occurred while creating the AI model.", None
        

    async def _validate_api_keys(self, gemini_key: str, openrouter_key: str) -> Tuple[bool, str, str]:
        """Validates API keys against the REST API. Returns (is_valid, error_message, tier).

        The tier describes whether the **user's Google key has billing enabled** —
        image models are rejected on unbilled keys. It is not a user tier.

        Kept on v1alpha, as it was under the SDK.
        """
        detected_tier = "free"

        if gemini_key:
            async def _ping(model_name: str):
                client = get_google_rest_client()
                return await client.post(
                    f"/v1alpha/models/{model_name}:generateContent",
                    content=json.dumps({
                        "contents": [{"role": "user", "parts": [{"text": "ping"}]}],
                        "generationConfig": {"maxOutputTokens": 1},
                    }),
                    headers={"x-goog-api-key": gemini_key, "Content-Type": "application/json"},
                )

            try:
                # Step 1: Authentication Check (Is the key valid?)
                auth_resp = await _ping('gemini-flash-lite-latest')
                if auth_resp.status_code != 200:
                    return False, f"Google Gemini API validation failed: {auth_resp.status_code}: {auth_resp.text}", "none"

                # Step 2: Billing Detection (Does it have access to image models?)
                billing_resp = await _ping('gemini-3.1-flash-image')
                detected_tier = "paid" if billing_resp.status_code == 200 else "free"

            except Exception as e:
                return False, f"Google Gemini API validation failed: {str(e)}", "none"

        if openrouter_key:
            try:
                client = get_shared_client()
                headers = {"Authorization": f"Bearer {openrouter_key}"}
                # 5s explicitly: this used to ride on httpx.AsyncClient's own
                # default, which the shared client raises to 30s.
                response = await client.get("https://openrouter.ai/api/v1/auth/key", headers=headers, timeout=5.0)
                
                if response.status_code == 401:
                    return False, "The OpenRouter API key provided is invalid or has been revoked.", "none"
                elif response.status_code != 200:
                    return False, f"OpenRouter validation failed with status code: {response.status_code}", "none"
                
                detected_tier = "paid" 

            except httpx.RequestError as e:
                return False, f"Could not validate the OpenRouter key due to a network error: {e}", "none"
            except Exception as e:
                return False, f"An unexpected error occurred while validating the OpenRouter key: {e}", "none"
        
        return True, "", detected_tier
    

    @tasks.loop(hours=24)
    async def pricing_sync_task(self):
        try:
            os.makedirs(MODELS_DATA_DIR, exist_ok=True)
            rates = {
                # Official Google Gemini Standard Tier Pricing (per 1M tokens in USD)
                "GOOGLE/gemini-3.7-flash": {"input_1m": 0.75, "output_1m": 3.75},
                "GOOGLE/gemini-3.6-flash": {"input_1m": 1.50, "output_1m": 7.50},
                "GOOGLE/gemini-3.5-flash": {"input_1m": 1.50, "output_1m": 9.00},
                "GOOGLE/gemini-3.5-flash-lite": {"input_1m": 0.30, "output_1m": 2.50},
                "GOOGLE/gemini-3.1-pro-preview": {"input_1m": 2.00, "output_1m": 12.00},
                "GOOGLE/gemini-3.1-flash-lite": {"input_1m": 0.25, "output_1m": 1.50},
                "GOOGLE/gemini-3-flash-preview": {"input_1m": 0.50, "output_1m": 3.00},
                "GOOGLE/gemini-2.5-pro": {"input_1m": 1.25, "output_1m": 10.00},
                "GOOGLE/gemini-2.5-flash": {"input_1m": 0.30, "output_1m": 2.50},
                "GOOGLE/gemini-2.5-flash-lite": {"input_1m": 0.10, "output_1m": 0.40},
                "GOOGLE/gemini-flash-latest": {"input_1m": 1.50, "output_1m": 7.50},
                "GOOGLE/gemini-pro-latest": {"input_1m": 2.00, "output_1m": 12.00},
                "GOOGLE/gemini-flash-lite-latest": {"input_1m": 0.30, "output_1m": 2.50}
            }

            try:
                resp = await get_shared_client().get("https://openrouter.ai/api/v1/models", timeout=15.0)
                if resp.status_code == 200:
                    data = resp.json()
                    for model in data.get("data", []):
                        m_id = model.get("id")
                        pricing = model.get("pricing", {})
                        try:
                            prompt_rate = float(pricing.get("prompt", 0.0)) * 1000000
                            completion_rate = float(pricing.get("completion", 0.0)) * 1000000
                            rates[f"OPENROUTER/{m_id}"] = {
                                "input_1m": prompt_rate,
                                "output_1m": completion_rate
                            }
                        except (ValueError, TypeError):
                            pass
            except Exception as e:
                print(f"Warning: Failed to fetch OpenRouter pricing: {e}")

            cache_data = {
                "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "rates": rates
            }
            with open(PRICING_CACHE_FILE, "w", encoding="utf-8") as f:
                f.write(json.dumps(cache_data, option=json.OPT_INDENT_2).decode('utf-8'))
                
        except Exception as e:
            print(f"Error in pricing_sync_task: {e}")

    def _get_model_pricing(self, model_name: str) -> Tuple[float, float]:
        try:
            if os.path.exists(PRICING_CACHE_FILE):
                with open(PRICING_CACHE_FILE, "rb") as f:
                    cache_data = json.loads(f.read())
                    rates = cache_data.get("rates", {})
                    # Ensure lookup maps cleanly based on stored structure
                    mapped_name = model_name
                    if not mapped_name.startswith(("GOOGLE/", "OPENROUTER/", "OLLAMA/")):
                        if "/" in mapped_name:
                            mapped_name = f"OPENROUTER/{mapped_name}"
                        else:
                            mapped_name = f"GOOGLE/{mapped_name}"
                    
                    pricing = rates.get(mapped_name)
                    if pricing:
                        return float(pricing.get("input_1m", 0.0)), float(pricing.get("output_1m", 0.0))
        except Exception as e:
            pass
        return 0.0, 0.0

    def _calculate_turn_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        input_rate, output_rate = self._get_model_pricing(model_name)
        cost_input = (input_tokens / 1000000) * input_rate
        cost_output = (output_tokens / 1000000) * output_rate
        return cost_input + cost_output
