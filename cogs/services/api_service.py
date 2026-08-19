import os
import asyncio
import base64
import datetime
import threading
import httpx
import orjson as json
from collections import OrderedDict
from typing import get_args, Any, List, Optional, Tuple
from discord.ext import tasks

from google import genai
from google.genai import types

from ..utils.constants import (
    OLLAMA_LOCAL_URL, MODELS_DATA_DIR, PRICING_CACHE_FILE, IMAGE_MODELS, AUDIO_MODELS,
    ALLOWED_MODELS, defaultConfig, PRIMARY_MODEL_NAME, FALLBACK_MODEL_NAME,
    HarmBlockThreshold, HarmCategory,
    USE_GOOGLE_REST_ADAPTER,
)


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
            async with httpx.AsyncClient() as client:
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
                                async with httpx.AsyncClient() as client:
                                    resp = await client.get(url, follow_redirects=True, timeout=15.0)
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
                async with httpx.AsyncClient() as client:
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

# --- Shared google-genai client cache -----------------------------------------
#
# Constructing a genai.Client builds three separate SSL contexts (httpx, aiohttp,
# websockets), each loading the full CA bundle: ~44 ms of CPU and ~2.4 MB of RSS
# that is never returned to the OS. On the e2-micro that cost is unaffordable on a
# per-turn path, and it was previously paid per model, per embedding and per TTS
# segment. Clients are stateless with respect to requests, so one per
# (api_key, api_version) is sufficient.
#
# Bounded to _GENAI_CLIENT_CACHE_MAX entries, evicted least-recently-used, because
# the key space is per-guild API keys and must not grow without limit. The lock
# guards the move_to_end/popitem sequence: the sync client is also touched from
# worker threads via asyncio.to_thread (see files.upload below).

_GENAI_CLIENT_CACHE_MAX = 8
_genai_client_cache: "OrderedDict[Tuple[str, str], genai.Client]" = OrderedDict()
_genai_client_lock = threading.Lock()


def get_genai_client(api_key: str, api_version: str = 'v1beta') -> genai.Client:
    """Returns a shared genai.Client for this (api_key, api_version) pair.

    Always prefer this over constructing genai.Client directly.
    """
    key = (api_key, api_version)
    with _genai_client_lock:
        client = _genai_client_cache.get(key)
        if client is not None:
            _genai_client_cache.move_to_end(key)
            return client

    # Built outside the lock: construction is slow and two concurrent misses
    # racing is harmless — the loser's client is simply discarded.
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version=api_version))

    with _genai_client_lock:
        existing = _genai_client_cache.get(key)
        if existing is not None:
            _genai_client_cache.move_to_end(key)
            return existing
        _genai_client_cache[key] = client
        while len(_genai_client_cache) > _GENAI_CLIENT_CACHE_MAX:
            _genai_client_cache.popitem(last=False)
    return client


# --- Google REST adapter -------------------------------------------------------
#
# Migration 2. google-genai is already REST over httpx — there is no gRPC and no
# protocol change here. What it buys is 70 MB of import baseline never returned to
# the OS, and dropping aiohttp/websockets/requests/pydantic from the dependency
# tree; on a 1 GB e2-micro that is 7% of RAM before a message is handled.
#
# Selected by USE_GOOGLE_REST_ADAPTER (see constants.py). GoogleSDKModel below stays
# live alongside it until the flag is flipped for good.

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
            value = data[name]
        else:
            camel = _to_camel(name)
            if camel not in data:
                memo[name] = None
                return None
            value = data[camel]

        wrapped = _wrap_rest(name, value)
        memo[name] = wrapped
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

    def __init__(self, api_key, model_name, system_instruction=None, safety_settings=None, thinking_params=None, tools=None):
        self.api_key = api_key
        self.model_name = model_name.replace("OPENROUTER/", "").replace("GOOGLE/", "")
        self.system_instruction = system_instruction
        self.safety_settings = safety_settings
        self.thinking_params = thinking_params or {}
        self.tools = tools

    # -- media ------------------------------------------------------------------

    async def _upload_file(self, path: str, mime_type: str) -> Optional[str]:
        """Uploads a file already on disk and returns its file URI.

        Still routed through the SDK's resumable upload while google-genai is
        present. This is the single seam that Migration 2 step 4 replaces with the
        hand-rolled resumable protocol; keeping it isolated means the media path —
        the least-exercised one in the bot — changes in exactly one place.
        """
        client = get_genai_client(self.api_key, 'v1beta')
        uploaded = await asyncio.to_thread(client.files.upload, file=path, config={'mime_type': mime_type})
        return getattr(uploaded, 'uri', None)

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
                if url.startswith(('http://', 'https://')):
                    temp_path = None
                    try:
                        import tempfile

                        # Streamed to disk rather than buffered, so a large attachment
                        # never sits in RAM in full.
                        fd, temp_path = tempfile.mkstemp(suffix=".tmp")
                        client_http = get_google_rest_client()
                        async with client_http.stream("GET", url, follow_redirects=True, timeout=15.0) as resp:
                            resp.raise_for_status()
                            with os.fdopen(fd, 'wb') as f:
                                async for chunk in resp.aiter_bytes(chunk_size=8192):
                                    f.write(chunk)

                        file_uri = await self._upload_file(temp_path, mime_type or 'image/jpeg')
                        if file_uri:
                            parts.append({"fileData": {"fileUri": file_uri, "mimeType": mime_type or 'image/jpeg'}})
                    except Exception as e:
                        print(f"Failed to fetch media from URL {url}: {e}")
                    finally:
                        if temp_path and os.path.exists(temp_path):
                            os.remove(temp_path)
                elif os.path.exists(url):
                    try:
                        file_uri = await self._upload_file(url, mime_type or 'image/png')
                        if file_uri:
                            parts.append({"fileData": {"fileUri": file_uri, "mimeType": mime_type or 'image/png'}})
                    except Exception as e:
                        print(f"Failed to upload local image {url} to Gemini File API: {e}")
                else:
                    parts.append({"fileData": {"fileUri": url, "mimeType": mime_type}})
        return parts

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

    The same shape GoogleSDKModel's ThoughtResponse produces — only the input
    changes, from an SDK object to a _RestView over the JSON.
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


class GoogleSDKModel:
    def __init__(self, api_key, model_name, system_instruction=None, safety_settings=None, thinking_params=None, tools=None):
        # Force v1beta for stable "Thinking" part delivery
        self.client = get_genai_client(api_key, 'v1beta')
        self.model_name = model_name.replace("OPENROUTER/", "").replace("GOOGLE/", "")
        self.system_instruction = system_instruction
        self.safety_settings = safety_settings
        self.thinking_params = thinking_params or {}
        self.tools = tools

    async def generate_content_async(self, contents, generation_config=None, stream_state=None):
        formatted_contents = []
        for item in contents:
            if isinstance(item, str):
                formatted_contents.append(item)
            elif isinstance(item, dict):
                role = item.get('role', 'user')
                parts = []
                for p in item.get('parts', []):
                    if isinstance(p, str):
                        parts.append(types.Part.from_text(text=p))
                    elif isinstance(p, dict) and 'mime_type' in p and 'data' in p:
                        parts.append(types.Part.from_bytes(data=p['data'], mime_type=p['mime_type']))
                    elif isinstance(p, dict) and 'url' in p:
                        url = p['url']
                        mime_type = p.get('mime_type', '')
                        if url.startswith(('http://', 'https://')):
                            try:
                                import tempfile

                                # 1. Stream directly to disk to prevent RAM spikes
                                fd, temp_path = tempfile.mkstemp(suffix=".tmp")
                                async with httpx.AsyncClient() as client_http:
                                    async with client_http.stream("GET", url, follow_redirects=True, timeout=15.0) as resp:
                                        resp.raise_for_status()
                                        with os.fdopen(fd, 'wb') as f:
                                            async for chunk in resp.aiter_bytes(chunk_size=8192):
                                                f.write(chunk)

                                # 2. Upload via File API (Disk Overloading)
                                # Using asyncio.to_thread to prevent blocking the event loop with the sync upload
                                upload_config = {'mime_type': mime_type or 'image/jpeg'}
                                uploaded_file = await asyncio.to_thread(self.client.files.upload, file=temp_path, config=upload_config)

                                # 3. Pass the URI reference instead of raw bytes
                                parts.append(types.Part.from_uri(file_uri=uploaded_file.uri, mime_type=mime_type or 'image/jpeg'))

                                # 4. Cleanup temp file
                                os.remove(temp_path)

                            except Exception as e:
                                print(f"Failed to fetch media from URL {url}: {e}")
                                if 'temp_path' in locals() and os.path.exists(temp_path):
                                    os.remove(temp_path)
                        elif os.path.exists(url):
                            try:
                                # Disk Overloading: Upload local temp file directly to Gemini File API
                                upload_config = {'mime_type': mime_type or 'image/png'}
                                uploaded_file = await asyncio.to_thread(self.client.files.upload, file=url, config=upload_config)
                                parts.append(types.Part.from_uri(file_uri=uploaded_file.uri, mime_type=mime_type or 'image/png'))
                            except Exception as e:
                                print(f"Failed to upload local image {url} to Gemini File API: {e}")
                        else:
                            parts.append(types.Part.from_uri(file_uri=url, mime_type=mime_type))

                formatted_contents.append(types.Content(role=role, parts=parts))
            elif hasattr(item, 'role') and hasattr(item, 'parts'):
                # Fallback for legacy objects
                new_parts = []
                for p in item.parts:
                    if hasattr(p, 'text') and p.text:
                        new_parts.append(types.Part.from_text(text=p.text))
                    elif hasattr(p, 'inline_data') and p.inline_data:
                        new_parts.append(types.Part.from_bytes(data=p.inline_data.data, mime_type=p.inline_data.mime_type))
                formatted_contents.append(types.Content(role=item.role, parts=new_parts))
            else:
                formatted_contents.append(item)

        v2_safety = []
        if self.safety_settings:
            for cat, thresh in self.safety_settings.items():
                v2_safety.append(types.SafetySetting(
                    category=cat.name if hasattr(cat, 'name') else str(cat),
                    threshold=thresh.name if hasattr(thresh, 'name') else str(thresh)
                ))

        thinking_cfg = None
        model_lower = self.model_name.lower()

        # [NEW] Exclude utility models that do not support reasoning tokens/thinking config
        is_utility_model = any(suffix in model_lower for suffix in ["-image", "-tts", "-embedding"])

        include_thoughts = self.thinking_params.get("thinking_summary_visible") == "on"

        temp = None
        top_p = None
        top_k = None

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
            if generation_config and hasattr(generation_config, 'thinking_config') and generation_config.thinking_config:
                include_thoughts = generation_config.thinking_config.include_thoughts

        if not is_utility_model:
            if "gemini-3" in model_lower:
                lvl = self.thinking_params.get("thinking_level", "high").lower()
                is_pro = "pro" in model_lower
                if is_pro:
                    mapped_lvl = "LOW" if lvl in ["low", "minimal", "none"] else "HIGH"
                else:
                    mapped_lvl = {
                        "xhigh": "HIGH", "high": "HIGH", "medium": "MEDIUM",
                        "low": "LOW", "minimal": "MINIMAL", "none": "MINIMAL"
                    }.get(lvl, "HIGH")

                thinking_cfg = types.ThinkingConfig(
                    include_thoughts=include_thoughts,
                    thinking_level=mapped_lvl
                )
            elif "gemini-2.5" in model_lower:
                budget = int(self.thinking_params.get("thinking_budget", -1))
                if "lite" not in model_lower:
                    if "pro" in model_lower and 0 <= budget < 128:
                        budget = 128
                    thinking_cfg = types.ThinkingConfig(
                        include_thoughts=include_thoughts,
                        thinking_budget=budget
                    )

        config = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            safety_settings=v2_safety if v2_safety else None,
            thinking_config=thinking_cfg,
            tools=self.tools
        )

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=formatted_contents,
            config=config
        )

        formatted_contents.clear()
        del formatted_contents

        class ThoughtResponse:
            def __init__(self, raw_resp):
                self.raw = raw_resp
                self.text = ""
                self.thought = ""
                self.candidates = raw_resp.candidates
                self.prompt_feedback = getattr(raw_resp, 'prompt_feedback', None)
                self.usage_metadata = getattr(raw_resp, 'usage_metadata', None)

                self.input_tokens = getattr(self.usage_metadata, 'prompt_token_count', 0) if self.usage_metadata else 0
                self.output_tokens = getattr(self.usage_metadata, 'candidates_token_count', 0) if self.usage_metadata else 0

                if raw_resp.candidates and raw_resp.candidates[0].content and raw_resp.candidates[0].content.parts:
                    for part in raw_resp.candidates[0].content.parts:
                        is_thought = getattr(part, 'thought', False)
                        if is_thought:
                            self.thought += part.text or ""
                        elif part.text:
                            self.text += part.text

                self.reasoning_tokens = int(len(self.thought) / 3.8) if self.thought else 0

            def __bool__(self): return bool(self.candidates)

        return ThoughtResponse(response)


# --- Google adapter routing ----------------------------------------------------
#
# Every Google construction site in the bot imports this name, so the flag is applied
# here rather than at ten call sites. A function, not a class, because the two
# adapters share no implementation — only an interface.
#
# MimicCog's cost-recording branch matches on __class__.__name__, so it tests for
# both concrete names; nothing else in the codebase inspects the adapter's type.
def GoogleGenAIModel(*args, **kwargs):
    """Constructs the Google adapter selected by USE_GOOGLE_REST_ADAPTER."""
    cls = GoogleRESTModel if USE_GOOGLE_REST_ADAPTER else GoogleSDKModel
    return cls(*args, **kwargs)


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
            return OpenRouterModel(actual_name, api_key=api_key, system_instruction=system_instruction, thinking_params=t_params)
        elif is_ollama:
            ollama_host = p_settings.get("ollama_host_url", OLLAMA_LOCAL_URL)
            return OllamaModel(actual_name, api_url=ollama_host, system_instruction=system_instruction, thinking_params=t_params)
        else:
            api_key = self.cog.storage_manager._get_api_key_for_guild(guild_id) if guild_id else self.cog.storage_manager._get_api_key_for_user(user_id)
            if not api_key: raise ValueError(google_key_error or "Google API Key not found. Use `/settings` to add one.")
            return GoogleGenAIModel(api_key=api_key, model_name=actual_name, system_instruction=system_instruction, safety_settings=safety_settings, thinking_params=t_params, tools=tools)

    def get_top_models(self, provider: str, target_config_key: str) -> List[str]:
        if target_config_key == 'image_generation_model': return list(get_args(IMAGE_MODELS))
        if target_config_key == 'speech_model': return list(get_args(AUDIO_MODELS))
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
        
        safety_level_str = "low" 
        profile_data_for_safety = self.cog.profile_manager._get_profile_config(profile_owner_id_for_instructions, profile_name_for_instructions, is_borrowed) or {}
        safety_level_str = profile_data_for_safety.get("safety_level", "low")

        safety_map = {
            "unrestricted": HarmBlockThreshold.BLOCK_NONE,
            "low": HarmBlockThreshold.BLOCK_ONLY_HIGH,
            "medium": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            "high": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        }
        threshold = safety_map.get(safety_level_str, HarmBlockThreshold.BLOCK_ONLY_HIGH)
        
        dynamic_safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: threshold,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: threshold,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: threshold,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: threshold,
        }

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
        
        safety_level_str = profile_data.get("safety_level", "low")
        safety_map = { "unrestricted": HarmBlockThreshold.BLOCK_NONE, "low": HarmBlockThreshold.BLOCK_ONLY_HIGH, "medium": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, "high": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE }
        threshold = safety_map.get(safety_level_str, HarmBlockThreshold.BLOCK_ONLY_HIGH)
        safety_settings = { cat: threshold for cat in get_args(HarmCategory) }

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
        """Validates API keys using the new Google Gen AI SDK. Returns (is_valid, error_message, tier)."""
        detected_tier = "free"
        
        if gemini_key:
            try:
                # Initialize new SDK client
                test_client = get_genai_client(gemini_key, 'v1alpha')
                
                # Step 1: Authentication Check (Is the key valid?)
                await test_client.aio.models.generate_content(
                    model='gemini-flash-lite-latest', 
                    contents="ping",
                    config=types.GenerateContentConfig(max_output_tokens=1)
                )

                # Step 2: Tier Detection (Does it have access to premium-only models?)
                try:
                    await test_client.aio.models.generate_content(
                        model='gemini-3.1-flash-image', 
                        contents="ping",
                        config=types.GenerateContentConfig(max_output_tokens=1)
                    )
                    detected_tier = "paid"
                except Exception:
                    # Key is valid, but rejected by a restricted model
                    detected_tier = "free"

            except Exception as e:
                return False, f"Google Gemini API validation failed: {str(e)}", "none"

        if openrouter_key:
            try:
                async with httpx.AsyncClient() as client:
                    headers = {"Authorization": f"Bearer {openrouter_key}"}
                    response = await client.get("https://openrouter.ai/api/v1/auth/key", headers=headers)
                    
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
                async with httpx.AsyncClient() as client:
                    resp = await client.get("https://openrouter.ai/api/v1/models", timeout=15.0)
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
