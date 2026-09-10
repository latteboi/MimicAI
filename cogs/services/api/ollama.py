"""The Ollama adapter.

Ollama has no way to say "this model does not think": `"max"` is never sent, and a
refusal is the probe -- see `_OLLAMA_NO_THINK`.
"""

import asyncio
import base64
import os
import re
import orjson as json
from typing import List

import httpx

from ...utils.constants import OLLAMA_LOCAL_URL, THINKING_LEVELS_TO_OLLAMA
from ...utils.http_client import get_shared_client
from .rest_view import _BlobRef, _RestView
from .streaming import (
    _FILE_BLOB_TOKEN, _aiter_streamed_body, _plan_streamed_body, _stream_to_tempfile,
)



#: Ollama model names that answered a `think` field with an error. Ollama accepts
#: `think` as a bool or one of "low"/"medium"/"high"/"max", but only on models built
#: for it -- everything else 400s with a message naming thinking. The adapter retries
#: once without the field and records the name here so the next round does not pay for
#: the same discovery. Bounded because CLAUDE.md forbids an unbounded dict keyed by
#: model; a local Ollama install holds a handful of models, and the cap is a backstop
#: rather than a working limit.
_OLLAMA_NO_THINK: set = set()
_OLLAMA_NO_THINK_CAP = 64


class _OllamaRetryWithoutThink(Exception):
    """Internal signal: this call must be redone with no `think` field.

    An exception rather than a return because the decision is made inside an
    `async with client.stream(...)` inside the global Ollama lock, and the retry has to
    happen outside both.
    """


def _ollama_think_value(thinking_params: dict):
    """The `think` field for one set of resolved thinking parameters, or None.

    None means "send nothing", which is what this adapter did for every request before
    it learned the field existed -- so an unrecognised level still gets the model's own
    default. `THINKING_LEVELS_TO_OLLAMA` carries the coarsening, including why neither
    Max nor Extra High sends Ollama's own "max": only newer builds accept it, and an
    older server answers an unknown value with an error.
    """
    level = str((thinking_params or {}).get("thinking_level") or "").lower()
    return THINKING_LEVELS_TO_OLLAMA.get(level)

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

        self.candidates = [_RestView({
            'content': {'parts': [{'text': self.text}]},
            'finish_reason': finish_reason,
        })]

    def __bool__(self): return True

class OllamaModel:
    def __init__(self, model_name, api_url=OLLAMA_LOCAL_URL, system_instruction=None, thinking_params=None, **kwargs):
        self.model_name = model_name.replace("OLLAMA/", "").replace("GOOGLE/", "").replace("OPENROUTER/", "")
        self.api_url = api_url.rstrip("/")
        self.system_instruction = system_instruction
        self.thinking_params = thinking_params or {}

    async def generate_content_async(self, contents, generation_config=None, safety_settings=None, stream_state=None):
        messages = []
        # As in the OpenRouter adapter: files whose base64 is spliced in as the
        # body streams. `staged_files` are the ones this call downloaded and so
        # has to delete again.
        blob_files: List[str] = []
        staged_files: List[str] = []
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
                            value = p.inline_data.data
                            if isinstance(value, _BlobRef):
                                images.append(_FILE_BLOB_TOKEN.format(len(blob_files)))
                                blob_files.append(value.path)
                            else:
                                images.append(base64.b64encode(value).decode('utf-8'))
                        except Exception as e:
                            print(f"Error encoding legacy image for Ollama: {e}")
                elif isinstance(p, dict) and 'url' in p:
                    mime_type = p.get('mime_type', 'image/png')
                    url = p['url']
                    if mime_type.startswith("image/"):
                        if url.startswith(('http://', 'https://')):
                            try:
                                # Staged to disk rather than buffered: this used to
                                # be resp.content plus its base64, both full size,
                                # for every participant that referenced the image.
                                path = await _stream_to_tempfile(url, get_shared_client())
                                staged_files.append(path)
                                images.append(_FILE_BLOB_TOKEN.format(len(blob_files)))
                                blob_files.append(path)
                            except Exception as e:
                                print(f"Ollama failed to fetch remote image {url}: {e}")
                        elif os.path.exists(url):
                            try:
                                images.append(_FILE_BLOB_TOKEN.format(len(blob_files)))
                                blob_files.append(url)
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

        # The adapter has always *read* `message.thinking` off the stream and never
        # asked for it, so every Ollama slot ran at whatever the model defaulted to.
        think = _ollama_think_value(self.thinking_params)
        sent_think = think is not None and self.model_name not in _OLLAMA_NO_THINK
        if sent_think:
            payload["think"] = think

        if advanced:
            if "frequency_penalty" in advanced: payload["options"]["frequency_penalty"] = advanced["frequency_penalty"]
            if "presence_penalty" in advanced: payload["options"]["presence_penalty"] = advanced["presence_penalty"]

        headers = {
            "Content-Type": "application/json",
            "Connection": "keep-alive"
        }

        needs_retry = False
        global _ollama_global_lock
        async with _ollama_global_lock:
            try:
                client = get_shared_client()
                full_content = ""
                reasoning_content = ""
                finish_reason = "STOP"
                retry_without_think = False

                if blob_files:
                    segments, content_length = _plan_streamed_body(payload, blob_files)
                    payload.clear()
                    request_kwargs = {
                        "content": _aiter_streamed_body(segments),
                        "headers": {**headers, "Content-Length": str(content_length)},
                    }
                else:
                    request_kwargs = {"json": payload, "headers": headers}

                async with client.stream("POST", f"{self.api_url}/api/chat", timeout=120.0, **request_kwargs) as response:
                    if response.status_code != 200:
                        err_text = await response.aread()
                        detail = err_text.decode('utf-8', errors='ignore')
                        # A model not built for thinking rejects the field outright.
                        # There is no capability table to consult -- Ollama serves
                        # whatever the user pulled -- so the refusal *is* the probe.
                        # Recorded, then the whole call is redone without it; the set
                        # membership makes this reachable at most once per model, so
                        # there is no recursion to bound beyond that.
                        if (sent_think and "think" in detail.lower()
                                and len(_OLLAMA_NO_THINK) < _OLLAMA_NO_THINK_CAP):
                            _OLLAMA_NO_THINK.add(self.model_name)
                            print(f"Ollama model {self.model_name} refused a `think` "
                                  f"field; retrying without it and remembering.")
                            retry_without_think = True
                        else:
                            raise Exception(f"Ollama API Error {response.status_code}: {detail}")

                    if retry_without_think:
                        raise _OllamaRetryWithoutThink()

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

                        self.candidates = [_RestView({
                            'content': {'parts': [{'text': self.text}]},
                            'finish_reason': f_reason,
                        })]

                    def __bool__(self): return True

                # Ollama streaming returns eval counts on the final chunk
                p_eval_count = chunk.get("prompt_eval_count", 0) if 'chunk' in locals() else 0
                eval_count = chunk.get("eval_count", 0) if 'chunk' in locals() else 0

                return OllamaResponseWrapper(msg_obj, (finish_reason or 'STOP').upper(), p_eval_count, eval_count)
            except _OllamaRetryWithoutThink:
                needs_retry = True
            except httpx.RequestError as e:
                raise Exception(f"Ollama Network Error: {str(e)}")
            except asyncio.CancelledError:
                raise
            finally:
                # Only the ones this call downloaded. A local path handed in by the
                # caller -- a generated image the round still needs -- is not ours
                # to delete.
                for path in staged_files:
                    try:
                        os.remove(path)
                    except OSError:
                        pass

        # Outside the global lock, because the retry takes it again. Rebuilt from
        # `contents` rather than resent, since the streamed-body planner consumes the
        # payload it was given -- and a message list carrying blob tokens cannot be
        # replayed without it.
        if needs_retry:
            return await self.generate_content_async(
                contents, generation_config=generation_config,
                safety_settings=safety_settings, stream_state=stream_state)
