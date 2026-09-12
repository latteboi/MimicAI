"""The OpenRouter adapter.

Images are referenced, never interpolated: a local file becomes a `_FILE_BLOB_TOKEN`
that `_plan_streamed_body` splices in as the request body streams.
"""

import asyncio
import base64
import os
from typing import List

import httpx

from ...utils.constants import OPENROUTER_DATA_POLICY_BLOCKED
from ...utils.helpers import (
    resolve_openrouter_image_detail, resolve_openrouter_service_tier,
)
from ...utils.http_client import get_shared_client
from .rest_view import _BlobRef, _RestView
from .streaming import _FILE_BLOB_TOKEN, _aiter_streamed_body, _plan_streamed_body



class OpenRouterModel:
    def __init__(self, model_name, api_key, system_instruction=None, thinking_params=None,
                 image_detail=None, service_tier=None, data_collection=None, **kwargs):
        self.model_name = model_name.replace("OPENROUTER/", "").replace("GOOGLE/", "")
        self.api_key = api_key
        self.system_instruction = system_instruction
        self.thinking_params = thinking_params or {} # [NEW]
        #: OpenRouter's answer to Google's mediaResolution, and a coarser one: the
        #: OpenAI-compatible per-part `detail` hint, "low" or "high". None means send
        #: nothing and let the route decide, which is what this adapter always did.
        self.image_detail = image_detail
        #: "flex", "priority" or None. Resolved once in `_instantiate_model` from the
        #: profile, so every OpenRouter slot the profile uses asks for the same tier.
        self.service_tier = service_tier
        #: "deny" keeps the request off hosts that may train on prompts; None sends no
        #: preference. Decided by the model factory from the data policy the request
        #: answers to -- a server's, or a Global Chat's -- see cogs/utils/data_policy.
        self.data_collection = data_collection

    async def generate_content_async(self, contents, generation_config=None, safety_settings=None, stream_state=None):
        messages = []
        # Local files referenced by the payload. Their base64 is spliced in while
        # the body streams (see _plan_streamed_body) rather than interpolated into
        # it here: OpenRouter needs the bytes inline, but nothing needs five copies
        # of them in the heap at once.
        blob_files: List[str] = []
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
                            value = p.inline_data.data
                            if isinstance(value, _BlobRef):
                                token = _FILE_BLOB_TOKEN.format(len(blob_files))
                                blob_files.append(value.path)
                                message_parts.append({"type": "image_url", "image_url": {
                                    "url": f"data:{mime_type};base64,{token}"}})
                            else:
                                b64_data = base64.b64encode(value).decode('utf-8')
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
                                token = _FILE_BLOB_TOKEN.format(len(blob_files))
                                blob_files.append(url)
                                message_parts.append({"type": "image_url", "image_url": {
                                    "url": f"data:{mime_type};base64,{token}"}})
                            except Exception as e:
                                print(f"Error referencing local image for OpenRouter: {e}")

            if message_parts:
                if len(message_parts) == 1 and message_parts[0]["type"] == "text":
                    messages.append({"role": role, "content": message_parts[0]["text"]})
                else:
                    messages.append({"role": role, "content": message_parts})

        if self.image_detail:
            for message in messages:
                body = message.get("content")
                if not isinstance(body, list):
                    continue
                for part in body:
                    if part.get("type") == "image_url":
                        part["image_url"]["detail"] = self.image_detail

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

        # Set before the advanced splice, so an operator who puts `service_tier` in
        # the advanced params still wins over the picker rather than being overwritten
        # by it -- the same precedence every other key in that dict already has.
        if self.service_tier:
            payload["service_tier"] = self.service_tier

        if advanced:
            payload.update(advanced)

        # After the advanced splice, and merged into whatever `provider` object that
        # carried, so no profile setting can reopen hosts the server's policy closed.
        if self.data_collection:
            provider = payload.get("provider")
            provider = dict(provider) if isinstance(provider, dict) else {}
            provider["data_collection"] = self.data_collection
            payload["provider"] = provider

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://discord.com",
            "X-Title": "MimicAI Discord Bot"
        }

        try:
            client = get_shared_client()
            if blob_files:
                # Explicit Content-Length keeps httpx off chunked encoding, so the
                # request on the wire matches the buffered one this replaces.
                segments, content_length = _plan_streamed_body(payload, blob_files)
                payload.clear()
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    content=_aiter_streamed_body(segments),
                    headers={**headers, "Content-Type": "application/json",
                             "Content-Length": str(content_length)},
                    timeout=120.0)
            else:
                response = await client.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=120.0)
            if response.status_code != 200:
                err = Exception(f"OpenRouter API Error {response.status_code}: {response.text}")
                # OpenRouter answers a request whose data policy no host can meet with
                # "No endpoints found matching your data policy". Said plainly instead,
                # and carried whole: _format_api_error would cut it at 80 characters.
                if self.data_collection == "deny" and "data policy" in response.text.lower():
                    err.formatted_reason = OPENROUTER_DATA_POLICY_BLOCKED
                raise err

            data = response.json()
            if 'error' in data:
                 raise Exception(f"OpenRouter API Error: {data['error']}")

            choice = data['choices'][0]
            msg_obj = choice['message']
            usage_obj = data.get('usage', {})

            class OpenRouterThoughtResponse:
                def __init__(self, content, reasoning, finish_reason, input_toks, output_toks,
                             billed_cost=None, served_tier=None):
                    self.text = content
                    self.thought = reasoning or ""
                    self.input_tokens = input_toks
                    self.output_tokens = output_toks
                    self.reasoning_tokens = int(len(self.thought) / 3.8) if self.thought else 0
                    #: What OpenRouter says it charged, and which tier served it. The
                    #: gateway returns both unasked. Kept as None when absent rather
                    #: than defaulted to 0.0, because a turn that reported no cost and
                    #: a turn that genuinely cost nothing are different facts and
                    #: `/session audit` labels them differently.
                    self.billed_cost = billed_cost
                    self.service_tier = served_tier

                    # One _RestView in place of four throwaway classes per response:
                    # it presents the same candidates[0].content.parts[].text surface
                    # and gives finish_reason its .name through _EnumStr.
                    self.candidates = [_RestView({
                        'content': {'parts': [{'text': content}]},
                        'finish_reason': finish_reason,
                    })]

                def __bool__(self): return True

            return OpenRouterThoughtResponse(
                msg_obj.get('content', ''),
                msg_obj.get('reasoning', ''),
                (choice.get('finish_reason') or 'STOP').upper(),
                usage_obj.get('prompt_tokens', 0),
                usage_obj.get('completion_tokens', 0),
                usage_obj.get('cost'),
                data.get('service_tier'),
            )
        except httpx.RequestError as e:
            raise Exception(f"OpenRouter Network Error: {str(e)}")
        except asyncio.CancelledError:
            raise
