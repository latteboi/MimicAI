"""The OpenRouter Image API adapter.

`POST /api/v1/images` is an endpoint of its own rather than a chat completion, so this is an
adapter of its own rather than a mode of `OpenRouterModel`. It answers with a
`GoogleRESTResponse` on purpose: every image path reads
`candidates[0].content.parts[].inline_data` and hands the value to
`materialise_inline_data`, and owning the blob files the same way keeps `close()` meaning
the same thing on both providers.

The image arrives as base64 in `data[].b64_json` and goes through the same
`InlineBlobExtractor` Gemini's `inlineData` does, so it lands in a file on the way off the
socket rather than in the heap. Reference images go the way the text adapter's do: a URL is
sent as a URL, and a local file is spliced into the body as it streams.

Nothing here sends `provider.data_collection` -- the endpoint takes no such field. Whether a
model may serve a server at all is decided before a request exists, in
`APIService._instantiate_model`.
"""

import os
from typing import List, Optional, Tuple

import httpx
import orjson as json

from ...utils.blob_stream import InlineBlobExtractor
from ...utils.constants import IMAGE_PROMPT_EMPTY, OPENROUTER_DATA_POLICY_BLOCKED
from ...utils.http_client import get_openrouter_client
from .google_rest import GoogleRESTResponse
from .streaming import (
    _DOWNLOAD_CHUNK_BYTES, _FILE_BLOB_TOKEN, _aiter_streamed_body, _close_body_segments,
    _plan_streamed_body,
)

_IMAGES_URL = "https://openrouter.ai/api/v1/images"

#: Generation takes tens of seconds. The heartbeat's watchdog is the real limit; this only
#: has to outlast it.
_TIMEOUT = httpx.Timeout(300.0, connect=10.0)

#: The image's type when OpenRouter leaves `media_type` out: whatever format was asked for.
_FORMAT_MIMES = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}

#: What a host refusing to draw something says. OpenRouter passes the host's own words
#: and the host's own status code through -- 400 from one, 403 from another -- so the
#: words are the only thing left to key off.
_REFUSAL_MARKERS = ("flagged", "content policy", "content_policy", "moderation",
                    "safety system", "blocked this request", "prohibited")


def _content_refusal(status: int, detail: str) -> Optional[str]:
    """The host's own sentence when it refused to draw this, else None.

    A 429 is traffic and a 5xx is the host being unwell; neither is a decision about the
    request, so only the rest of the 4xx range is read for one.
    """
    if status == 429 or not 400 <= status < 500:
        return None
    message = ""
    try:
        parsed = json.loads(detail)
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
            message = parsed["error"].get("message") or ""
    except Exception:
        pass
    text = message if isinstance(message, str) and message else detail
    if not any(marker in text.lower() for marker in _REFUSAL_MARKERS):
        return None
    return " ".join(text.split())[:300]


class OpenRouterImageModel:
    def __init__(self, model_name, api_key, system_instruction=None, image_params=None):
        self.model_name = model_name.replace("OPENROUTER/", "")
        self.api_key = api_key
        #: The profile's image style prompt. The Image API has no system field, so it leads
        #: the prompt instead.
        self.system_instruction = system_instruction
        #: resolve_image_output_params for this model: aspect_ratio, image_size, quality,
        #: output_format and max_refs, each present only when the model takes it.
        self.image_params = image_params or {}

    def _build_payload(self, contents) -> Tuple[dict, List[str]]:
        """The request body, and the local files its placeholders stand for."""
        texts: List[str] = []
        references: List[str] = []
        blob_files: List[str] = []
        max_refs = int(self.image_params.get("max_refs") or 0)

        for content in contents:
            if isinstance(content, str):
                content = {'role': 'user', 'parts': [content]}
            for part in content.get('parts', []):
                if isinstance(part, str):
                    if part.strip():
                        texts.append(part.strip())
                    continue
                if not (isinstance(part, dict) and 'url' in part):
                    continue
                mime_type = part.get('mime_type') or 'image/png'
                if not mime_type.startswith('image/') or len(references) >= max_refs:
                    continue
                url = part['url']
                if url.startswith(('http://', 'https://')):
                    references.append(url)
                elif os.path.exists(url):
                    token = _FILE_BLOB_TOKEN.format(len(blob_files))
                    blob_files.append(url)
                    references.append(f"data:{mime_type};base64,{token}")

        if self.system_instruction and self.system_instruction.strip():
            texts.insert(0, self.system_instruction.strip())
        prompt = "\n\n".join(texts)
        if not prompt:
            # Refused before it is sent: the endpoint answers an empty prompt with a
            # schema error, and `run_with_fallback` would then hand the fallback model
            # the same empty prompt to be refused by. Nothing upstream should reach here
            # -- `helpers.image_command_prompt` is where a request with nothing to draw
            # stops -- so this is the guard that says so rather than a 400.
            error = ValueError(f"OpenRouter image request carried no prompt ({self.model_name})")
            error.formatted_reason = IMAGE_PROMPT_EMPTY
            error.retryable = False
            raise error
        payload = {"model": self.model_name, "prompt": prompt, "n": 1}
        for stored, wire in (("aspect_ratio", "aspect_ratio"), ("image_size", "resolution"),
                             ("quality", "quality"), ("output_format", "output_format")):
            if self.image_params.get(stored):
                payload[wire] = self.image_params[stored]
        if references:
            payload["input_references"] = [
                {"type": "image_url", "image_url": {"url": url}} for url in references]
        return payload, blob_files

    async def generate_content_async(self, contents, generation_config=None, stream_state=None):
        payload, blob_files = self._build_payload(contents)
        requested_format = payload.get("output_format")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        segments = None
        if blob_files:
            # Explicit Content-Length keeps httpx off chunked encoding, as in OpenRouterModel.
            segments, content_length = _plan_streamed_body(payload, blob_files)
            body = _aiter_streamed_body(segments)
            headers["Content-Length"] = str(content_length)
        else:
            body = json.dumps(payload)
        payload.clear()

        # Streamed for the reason GoogleRESTModel's response is: the image is base64 inside
        # the JSON, and a buffered read holds it several times over before anything could
        # divert it (cogs/utils/blob_stream).
        extractor = InlineBlobExtractor(key=b"b64_json", suffix=".img")
        try:
            async with get_openrouter_client().stream(
                    "POST", _IMAGES_URL, content=body, headers=headers, timeout=_TIMEOUT) as response:
                if response.status_code != 200:
                    # Error bodies are small; read in one go.
                    detail = (await response.aread()).decode('utf-8', 'replace')
                    err = Exception(f"OpenRouter API Error {response.status_code}: {detail}")
                    # Said plainly, and carried whole: _format_api_error cuts at 80 characters.
                    if "data policy" in detail.lower():
                        err.formatted_reason = OPENROUTER_DATA_POLICY_BLOCKED
                    elif refusal := _content_refusal(response.status_code, detail):
                        # The host judged the request rather than failed at it, so the
                        # fallback model is a second opinion on a decision -- and usually
                        # the same one, more strictly. Google's image block already ends
                        # the attempt this way by returning no candidates instead of
                        # raising; `retryable = False` is how a raised one says the same.
                        err.formatted_reason = refusal
                        err.retryable = False
                    raise err
                async for chunk in response.aiter_bytes(_DOWNLOAD_CHUNK_BYTES):
                    extractor.feed(chunk)
                skeleton, blob_paths = extractor.finish()
        except httpx.RequestError as e:
            extractor.cleanup()
            raise Exception(f"OpenRouter Network Error: {str(e)}")
        except BaseException:
            # CancelledError included, which is exactly when a half-written blob would
            # otherwise be left behind.
            extractor.cleanup()
            raise
        finally:
            if segments is not None:
                # A request refused before its body was read never exhausts the generator
                # that would otherwise close these.
                _close_body_segments(segments)

        try:
            parsed = json.loads(skeleton)
        except json.JSONDecodeError:
            _unlink(blob_paths)
            raise Exception("OpenRouter API Error: the image response could not be read")
        skeleton = None
        if not isinstance(parsed, dict) or parsed.get("error"):
            _unlink(blob_paths)
            detail = parsed.get("error") if isinstance(parsed, dict) else parsed
            raise Exception(f"OpenRouter API Error: {detail}")
        return _as_gemini_response(parsed, blob_paths, requested_format)


def _unlink(paths: List[str]) -> None:
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass


def _as_gemini_response(body: dict, blob_paths: List[str],
                        requested_format: Optional[str]) -> GoogleRESTResponse:
    """The Image API body, in the shape every image path already reads.

    `inlineData.data` carries whatever the extractor left -- a blob sentinel, or a small
    image's base64 -- and `_RestView` turns either into what `materialise_inline_data`
    takes. An answer with no image is still a finished candidate, so it reads as "no image
    data returned" rather than as a safety block it was not.
    """
    fallback_mime = _FORMAT_MIMES.get(requested_format or "", "image/png")
    parts = []
    for item in body.get("data") or []:
        if isinstance(item, dict) and item.get("b64_json"):
            parts.append({"inlineData": {"mimeType": item.get("media_type") or fallback_mime,
                                         "data": item["b64_json"]}})
    usage = body.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    response = GoogleRESTResponse({
        "candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": usage.get("prompt_tokens") or 0,
                          "candidatesTokenCount": usage.get("completion_tokens") or 0},
    }, blob_paths=blob_paths)
    # What OpenRouter charged, as OpenRouterModel's responses carry it. Absent stays None:
    # no reported cost and no cost are different facts.
    cost = usage.get("cost")
    response.billed_cost = (float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool)
                            else None)
    return response
