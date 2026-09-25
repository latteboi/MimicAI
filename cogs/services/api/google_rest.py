"""The Google REST adapter: the model, its response, and the two things only it does.

`materialise_inline_data` and `generate_google_tts_audio` live here rather than beside
the other adapters because both are Google-shaped -- inline blobs come off this wire
and TTS calls generateContent with a speechConfig no other provider takes.
"""

import asyncio
import base64
import os
import random
import re
import tempfile
import time
import wave
import orjson as json
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ...utils.blob_stream import InlineBlobExtractor, TruncatedJSONError
from ...utils.constants import (
    DEFAULT_SPEECH_VOICE, ERR_REASON_NO_AUDIO, ERR_REASON_SPEECH_PROHIBITED,
    ERR_REASON_SPEECH_REFUSED, ERR_REASON_SPEECH_TIMED_OUT, THINKING_BUDGET_MAX,
    THINKING_LEVELS_TO_GOOGLE, THINKING_LEVELS_TO_GOOGLE_BINARY, TTS_VOICE_LOOKUP,
    defaultConfig,
)
from ...utils.helpers import google_thinking_caps, resolve_media_resolution
from ...utils.http_client import get_shared_client
from ...utils.memory_tuning import maybe_trim_malloc
from ...utils.net_guard import safe_stream
from ...utils import mem_probe
from .function_calls import calls_forbidden, from_google_parts, google_part
from .output_cap import RetryUncapped, output_cap, refused_output_cap
from .rest_view import _BlobRef, _RestView, _to_camel, _wrap_rest
from .streaming import (
    _DOWNLOAD_CHUNK_BYTES, _aiter_file_bytes, _stream_to_tempfile,
)



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


#: Models that refused a built-in tool beside declared functions -- Gemini 2.5 answers
#: "Tool call context circulation is not enabled", and the combination is Gemini 3's
#: alone. The refusal is the probe, as for `output_cap`: learnt once, then those models
#: are sent the functions without the built-in tools. Bounded for the same reason.
_NO_CIRCULATION: set = set()
_NO_CIRCULATION_MAX = 64


class _RetryWithoutBuiltins(Exception):
    """Internal signal, as RetryUncapped: redo this call now `_NO_CIRCULATION` knows."""


class GoogleRESTModel:
    """Google Gemini over raw REST, satisfying the same adapter interface as
    OpenRouterModel and OllamaModel: generate_content_async(contents,
    generation_config, ...) returning an object with .text, .thought, .candidates,
    .prompt_feedback and token counts.
    """

    def __init__(self, api_key, model_name, system_instruction=None, safety_settings=None, thinking_params=None, tools=None, image_params=None, media_resolution=None):
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
        #: Google's `mediaResolution`: how many tokens an *input* image or PDF is worth.
        #: Empty means send nothing, which is not the same as sending UNSPECIFIED --
        #: the absent field lets the model apply its own default, and on Gemini 3 that
        #: default is not the cheapest option.
        self.media_resolution = media_resolution or ""

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
            elif isinstance(p, dict) and ('function_call' in p or 'function_response' in p
                                          or 'verbatim' in p):
                # A tool loop's own bookkeeping, echoed back so the model sees the call
                # it made and the answer it got. Translated rather than passed through:
                # the neutral spelling is shared with OpenRouter, which needs whole
                # messages for the same two things.
                translated = google_part(p)
                if translated:
                    parts.append(translated)
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
            with mem_probe.probe("  media -> File API"):
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
            # Streamed to disk rather than buffered, so a large attachment
            # never sits in RAM in full.
            temp_path = await _stream_to_tempfile(url, get_google_rest_client())

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
                        value = p.inline_data.data
                        # A part echoed back out of a streamed response carries a
                        # _BlobRef, not bytes. Reading it back is the wrong shape
                        # for a large blob, but this branch only ever sees the
                        # small legacy parts described above -- and raising a
                        # TypeError here instead would drop the part silently.
                        if isinstance(value, _BlobRef):
                            value = value.read_bytes()
                        new_parts.append({
                            "inlineData": {
                                "mimeType": p.inline_data.mime_type,
                                "data": base64.b64encode(value).decode('ascii'),
                            }
                        })
                formatted.append({"role": item.role, "parts": new_parts})
        return formatted

    def _build_generation_config(self, generation_config) -> dict:
        # Which fields this model takes at all. `google_thinking_caps` answers the
        # thinking half; this flag is what gates mediaResolution, which an image, TTS
        # or embedding model is handed no media to resolve and rejects.
        is_utility_model = any(suffix in self.model_name.lower()
                               for suffix in ["-image", "-tts", "-embedding"])
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

        # Which of the two thinking fields this model takes -- or neither -- is
        # `google_thinking_caps`' single answer rather than three families of substring
        # test inlined here. The picker asks the same function so it can grey out a
        # control the chosen model will not honour, which is the whole reason it moved.
        caps = google_thinking_caps(self.model_name)
        if caps["mode"] == "level":
            lvl = self.thinking_params.get("thinking_level", "high").lower()
            table = (THINKING_LEVELS_TO_GOOGLE_BINARY if caps["levels"] == "binary"
                     else THINKING_LEVELS_TO_GOOGLE)
            mapped_lvl = table.get(lvl, "HIGH")
            cfg["thinkingConfig"] = {"includeThoughts": include_thoughts, "thinkingLevel": mapped_lvl}
        elif caps["mode"] == "budget":
            budget = int(self.thinking_params.get("thinking_budget", -1))
            # -1 is dynamic and always legal; a floor only bites a real token count.
            if 0 <= budget < caps["budget_floor"]:
                budget = caps["budget_floor"]
            budget = min(budget, THINKING_BUDGET_MAX)
            cfg["thinkingConfig"] = {"includeThoughts": include_thoughts, "thinkingBudget": budget}

        # How many tokens an input image or PDF is worth. Gated on the same utility
        # check as the thinking config: a TTS or embedding model is handed no media to
        # resolve and rejects the field. Sent at request level rather than per Part --
        # Gemini 3 allows the per-Part override, but a profile setting has no
        # per-attachment interface to hang off.
        if self.media_resolution and not is_utility_model:
            cfg["mediaResolution"] = self.media_resolution

        # Text models only -- see output_cap. An image model's output is the picture
        # imageConfig sizes, and a speech line carries its own, tighter cap.
        cap = None if is_utility_model or self.image_params else output_cap(self.model_name)
        if cap:
            cfg["maxOutputTokens"] = cap

        # Image output controls. Reached by an image model on purpose: such a model
        # rejects the *text* thinking config -- google_thinking_caps returns no mode for
        # it -- but the 3.x image models do take a thinkingLevel of their own, and only
        # when one was explicitly chosen. An absent key means "let the model use its own
        # default", which is not the same as sending MINIMAL.
        if self.image_params:
            # Sampling for the image slot, from the profile's own image_* keys rather
            # than the text profile's. Every image path calls generate_content_async
            # with generation_config=None, so `temp`/`top_p`/`top_k` above are None on
            # all of them -- but an explicit caller-supplied value still wins, which is
            # what the `not in cfg` guard preserves.
            for stored, wire in (("temperature", "temperature"),
                                 ("top_p", "topP"), ("top_k", "topK")):
                if wire not in cfg and self.image_params.get(stored) is not None:
                    cfg[wire] = self.image_params[stored]

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
                # interstitial draft images, and the call sites take the first
                # inline_data part they find. Billed either way, per the API docs.
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
        with mem_probe.probe("    api: build payload"):
            payload = {"contents": await self._build_contents(contents)}

            if self.system_instruction:
                # role is what the SDK sends here; the API ignores it, but matching keeps
                # the two adapters' wire payloads diffable while both are live.
                payload["systemInstruction"] = {"role": "user", "parts": [{"text": self.system_instruction}]}

            safety = self._build_safety_settings()
            if safety:
                payload["safetySettings"] = safety

            tools = self._build_tools()
            circulating = False
            if tools:
                declares = any(isinstance(t, dict) and "functionDeclarations" in t for t in tools)
                if declares and self.model_name in _NO_CIRCULATION:
                    # The functions win: the prompt describes them, and a character told
                    # of a function it was not sent narrates a lookup it cannot run.
                    tools = [t for t in tools if isinstance(t, dict) and "functionDeclarations" in t]
                payload["tools"] = tools
                tool_config = {}
                # A built-in tool (google_search, url_context) beside declared functions
                # is refused outright on Gemini 3 -- a 400 on every turn of a Native
                # profile with `recall` on -- unless the request opts in. The parts it
                # then answers in must go back verbatim: see `model_parts`.
                if declares and len(tools) > 1:
                    tool_config["includeServerSideToolInvocations"] = circulating = True
                # Only where functions are declared: the mode governs them alone, and
                # a request carrying nothing but google_search has none to forbid.
                if declares and calls_forbidden(generation_config):
                    tool_config["functionCallingConfig"] = {"mode": "NONE"}
                if tool_config:
                    payload["toolConfig"] = tool_config

            gen_cfg = self._build_generation_config(generation_config)
            if gen_cfg:
                payload["generationConfig"] = gen_cfg
            capped = "maxOutputTokens" in gen_cfg

            model_path = self.model_name if self.model_name.startswith("models/") else f"models/{self.model_name}"

            body = json.dumps(payload)
            payload.clear()

        # Streamed, not buffered, and the reason is memory rather than latency: an
        # image response carries the PNG as base64, and reading it with
        # `json.loads(response.content)` costs ~3.7x the image in simultaneous
        # copies (see cogs/utils/blob_stream). The extractor below diverts the blob
        # to a file as it comes off the socket, so the peak is one chunk whatever
        # the image weighs.
        extractor = InlineBlobExtractor()
        wire_bytes = 0
        try:
            client = get_google_rest_client()
            with mem_probe.probe("    api: stream+extract"):
                async with client.stream(
                    "POST",
                    f"/v1beta/{model_path}:generateContent",
                    content=body,
                    headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                ) as response:
                    if response.status_code != 200:
                        # The body carries the status name ("RESOURCE_EXHAUSTED") that
                        # helpers._get_friendly_api_error matches on, so pass it through
                        # intact. Error bodies are small; read it in one go.
                        detail = (await response.aread()).decode('utf-8', 'replace')
                        if capped and refused_output_cap(self.model_name, response.status_code, detail):
                            raise RetryUncapped()
                        if (circulating and response.status_code == 400
                                and "context circulation" in detail.lower()
                                and len(_NO_CIRCULATION) < _NO_CIRCULATION_MAX):
                            _NO_CIRCULATION.add(self.model_name)
                            print(f"{self.model_name} cannot take a built-in tool beside declared "
                                  f"functions; retrying without Native search/URL context and "
                                  f"remembering.")
                            raise _RetryWithoutBuiltins()
                        raise Exception(f"Google API Error {response.status_code}: {detail}")

                    async for chunk in response.aiter_bytes(_DOWNLOAD_CHUNK_BYTES):
                        wire_bytes += len(chunk)
                        extractor.feed(chunk)
                    skeleton, blob_paths = extractor.finish()
        except httpx.RequestError as e:
            extractor.cleanup()
            raise Exception(f"Google API Network Error: {str(e)}")
        except (RetryUncapped, _RetryWithoutBuiltins):
            extractor.cleanup()
            return await self.generate_content_async(contents, generation_config, stream_state)
        except BaseException:
            # Covers CancelledError as well, which is exactly when a half-written
            # blob would otherwise be left behind.
            extractor.cleanup()
            raise

        if mem_probe.ENABLED:
            # The skeleton is the part that still has to be parsed into the heap, so
            # `wire` vs `skeleton` is the whole question: a skeleton that tracks the
            # wire size means a blob was *not* diverted and the old buffered cost is
            # back, whatever the extractor's unit tests say.
            print(f"[mem]     wire {wire_bytes / 1e6:7.2f} MB   "
                  f"skeleton {len(skeleton) / 1e6:7.2f} MB   blobs {len(blob_paths)}")

        with mem_probe.probe("    api: parse skeleton"):
            parsed = json.loads(skeleton)
        skeleton = None
        mem_probe.describe_bulk(parsed)
        return GoogleRESTResponse(parsed, blob_paths=blob_paths)


class GoogleRESTResponse:
    """Normalises a parsed generateContent body into the shared adapter interface.

    Presents the same attribute surface the SDK adapter's ThoughtResponse did, so
    the call sites that reach past the shared interface into .raw keep working.
    """

    def __init__(self, body: dict, blob_paths: Optional[List[str]] = None):
        #: Files blob_stream wrote while this response streamed in, still owned by
        #: this object. `materialise_inline_data` moves one out; `close()` unlinks
        #: whatever is left.
        self._blob_paths = list(blob_paths or [])
        self.raw = _RestView(body)
        self.text = ""
        self.thought = ""
        self.candidates = self.raw.candidates or []
        self.prompt_feedback = self.raw.prompt_feedback
        self.usage_metadata = self.raw.usage_metadata

        self.input_tokens = (self.usage_metadata.prompt_token_count or 0) if self.usage_metadata else 0
        self.output_tokens = (self.usage_metadata.candidates_token_count or 0) if self.usage_metadata else 0
        #: Reported apart from the reply and billed at the output rate. The candidate
        #: count above leaves it out, so without this a thinking model's turns were
        #: estimated at the price of their visible text alone.
        self.thinking_tokens = (self.usage_metadata.thoughts_token_count or 0) if self.usage_metadata else 0

        if self.candidates and self.candidates[0].content and self.candidates[0].content.parts:
            for part in self.candidates[0].content.parts:
                if part.thought:
                    self.thought += part.text or ""
                elif part.text:
                    self.text += part.text

        #: Every function the model asked for, normalised. Empty on all but the
        #: handful of calls that declared one, and read without knowing the provider
        #: -- OpenRouterModel presents the same attribute.
        self.function_calls = from_google_parts(
            self.candidates[0].content.parts
            if self.candidates and self.candidates[0].content else None)
        #: The candidate's parts exactly as they came, for a tool loop to send back.
        #: Gemini 3 signs its parts (`thoughtSignature`), and with a built-in tool
        #: beside declared functions it also answers in `toolCall`/`toolResponse` parts
        #: of its own. A turn rebuilt from each call's name and args drops all of that,
        #: and the next request is refused.
        first = (body.get("candidates") or [{}])[0]
        self.model_parts = ((first.get("content") or {}).get("parts") or []) \
            if self.function_calls else []

        self.reasoning_tokens = int(len(self.thought) / 3.8) if self.thought else 0

    def __bool__(self):
        return bool(self.candidates)

    def release(self, path: str) -> None:
        """Hands ownership of one streamed blob to the caller, so `close()` leaves
        it alone."""
        try:
            self._blob_paths.remove(path)
        except ValueError:
            pass

    def close(self) -> None:
        """Unlinks any streamed blob nobody took.

        Worth calling explicitly on every path that finishes with a response --
        a safety-blocked image response still carries whatever partial part the
        model returned, and `run_with_fallback` can produce one of these per
        attempt.
        """
        paths, self._blob_paths = self._blob_paths, []
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                pass

    def __del__(self):
        # Backstop, not the mechanism: the explicit close() calls are. This only
        # covers the paths that drop a response without one -- an exception between
        # the parse and the write, mostly -- where the alternative is a multi-megabyte
        # file left in /tmp on a box with no room for it.
        try:
            if self._blob_paths:
                self.close()
        except Exception:
            pass


async def materialise_inline_data(response, value, suffix: str = ".png") -> Optional[str]:
    """Returns a filesystem path for an `inline_data.data` value, whichever form it
    arrived in, transferring ownership to the caller.

    Large values are already on disk (blob_stream streamed them there) and cost
    nothing here. Small ones -- under the divert threshold, so still `bytes` -- are
    written the way they always were. Call sites get one path either way and no
    longer care which happened.
    """
    if isinstance(value, _BlobRef):
        path = value.path
        if suffix and not path.endswith(suffix):
            # The extractor cannot know the mime type: `data` sometimes precedes
            # `mimeType` on the wire. Renaming is a metadata operation within the
            # same directory, so it costs nothing and keeps /tmp legible.
            renamed = path[:path.rfind('.')] + suffix if '.' in os.path.basename(path) else path + suffix
            try:
                os.rename(path, renamed)
                path = renamed
            except OSError:
                pass
        if response is not None:
            response.release(value.path)
        return path

    if isinstance(value, (bytes, bytearray)):
        data = bytes(value)

        def _write():
            import tempfile
            fd, path = tempfile.mkstemp(suffix=suffix)
            with os.fdopen(fd, 'wb') as f:
                f.write(data)
            return path

        return await asyncio.to_thread(_write)

    return None


# Migration 2 step 3. TTS calls generateContent directly with a response_modalities /
# speechConfig shape that no other caller needs, so it stays outside the shared
# generate_content_async interface rather than widening it for one consumer. Same
# endpoint, same routing switch as GoogleGenAIModel and get_embedding_vector above.

#: Seconds to wait before the one same-model retry, drawn fresh each time: long enough not
#: to land inside the blip that caused the failure, short because a turn is waiting on it.
_TTS_RETRY_DELAY = (0.5, 1.5)

#: Gemini's speech output rate, for an audio part whose mimeType does not name one.
_TTS_DEFAULT_RATE = 24000

#: finishReason values meaning the model declined the text rather than failed to voice it.
#: The same text is declined the same way, by this model or any other.
_TTS_REFUSAL_FINISH_REASONS = frozenset({
    "SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION",
})

#: Gemini bills speech at 25 output tokens for each second of audio.
_TTS_TOKENS_PER_SECOND = 25

#: The longest a line may take to say, in seconds per character: 0.2 for most scripts,
#: about three times an ordinary speaking pace, and 0.5 for a CJK character, which carries
#: more of a word. Generous on purpose -- the cap is for audio that runs on past its words,
#: and a slow, directed delivery must never be what it cuts short.
_TTS_SECONDS_PER_CHAR = 0.2
_TTS_SECONDS_PER_WIDE_CHAR = 0.5
#: The first code point of the CJK blocks, counted at the wide rate.
_TTS_WIDE_FROM = 0x2E80
#: Seconds any line is allowed however short, for a sigh or a pause around one word.
_TTS_MIN_SECONDS = 10
#: A WAV file's header, ahead of its PCM frames.
_WAV_HEADER_BYTES = 44

#: Models that refused `maxOutputTokens`, recorded the way `_OLLAMA_NO_THINK` records a
#: refused `think`, and bounded as it is: a server holds a handful of speech models.
_TTS_NO_OUTPUT_CAP: set = set()
_TTS_NO_OUTPUT_CAP_MAX = 16
_OUTPUT_CAP_FIELD = re.compile(r"max_?output_?tokens", re.IGNORECASE)
#: Models that refused `speechConfig.languageCode`, recorded and bounded the same way: the
#: field is documented for speech, but not per model.
_TTS_NO_LANGUAGE: set = set()
_LANGUAGE_FIELD = re.compile(r"language_?code", re.IGNORECASE)


def _refused_speech_field(detail: str, capped: bool, languaged: bool):
    """(the record of models refusing it, how to say it) for the field a 400 names, or None.

    A field the request did not carry is never blamed, and a 400 naming neither -- an
    unknown voice -- is not taken for a refusal of either.
    """
    if capped and _OUTPUT_CAP_FIELD.search(detail):
        return _TTS_NO_OUTPUT_CAP, "an output cap"
    if languaged and _LANGUAGE_FIELD.search(detail):
        return _TTS_NO_LANGUAGE, "a language code"
    return None


def _speech_token_cap(transcript: str, max_bytes: Optional[int] = None) -> int:
    """`maxOutputTokens` for one spoken line: the audio its words can need, and no more.

    A Gemini speech request has no length limit short of the model's output ceiling --
    minutes of audio -- and every second is billed. A line that runs on past its words, in
    trailing silence or with its direction read aloud, is billed to that ceiling, and a
    timeout retry bills it again. So the cap follows the transcript alone, never the
    Director's Desk wrapped around it, and stops at the largest WAV `max_bytes` allows:
    audio the server cannot upload is audio paid for and never heard.
    """
    wide = sum(1 for ch in transcript if ord(ch) >= _TTS_WIDE_FROM)
    seconds = (_TTS_MIN_SECONDS + (len(transcript) - wide) * _TTS_SECONDS_PER_CHAR
               + wide * _TTS_SECONDS_PER_WIDE_CHAR)
    if max_bytes:
        seconds = min(seconds, (max_bytes - _WAV_HEADER_BYTES) / (_TTS_DEFAULT_RATE * 2))
    return max(1, int(seconds * _TTS_TOKENS_PER_SECOND))


def _log_speech_usage(model_id: str, tokens: int, finish: str, cap: Optional[int]) -> None:
    """One terminal line per line spoken: the audio billed, and whether the cap ended it.

    Nothing else in the bot records a speech call, so this is what traces a day's bill back
    to the lines that ran it up. A count past the cap the request carried means the model
    does not honour `maxOutputTokens`, and the line says so.
    """
    if not tokens:
        return
    if finish == "MAX_TOKENS":
        note = f", stopped by its {cap}-token cap" if cap else ", stopped at the model's output limit"
    elif cap and tokens > cap:
        note = f", past its {cap}-token cap: this model does not honour the cap"
    else:
        note = ""
    print(f"Google TTS: {model_id} generated {tokens / _TTS_TOKENS_PER_SECOND:.0f} s of audio "
          f"({tokens} output tokens{note}).")


def _speech_refusal(code: str) -> Exception:
    """A content refusal phrased for the channel, and marked `retryable = False` -- which
    both the retry below and `run_with_fallback` honour."""
    reason = (ERR_REASON_SPEECH_PROHIBITED if code == "PROHIBITED_CONTENT"
              else code.replace('_', ' ').title())
    error = Exception(f"Google TTS refused the text: {code}")
    error.formatted_reason = ERR_REASON_SPEECH_REFUSED.format(reason=reason)
    error.retryable = False
    return error


def _pcm_rate(mime_type: Optional[str]) -> int:
    """The sample rate named in an `audio/L16;codec=pcm;rate=24000` mimeType."""
    match = re.search(r"rate=(\d+)", mime_type or "")
    return int(match.group(1)) if match else _TTS_DEFAULT_RATE


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _pcm_file_to_wav(pcm_path: str, rate: int) -> str:
    """Wraps a 16-bit mono PCM file in a WAV header, file to file, and removes the PCM.

    Blocking: run it in a thread.
    """
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        with open(pcm_path, 'rb') as src, wave.open(wav_path, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            while True:
                chunk = src.read(_DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                wav.writeframesraw(chunk)
    except BaseException:
        _remove_quietly(wav_path)
        raise
    finally:
        _remove_quietly(pcm_path)
    return wav_path


async def generate_google_tts_audio(
    api_key: str,
    model_id: str,
    text: str,
    voice_name: str = DEFAULT_SPEECH_VOICE,
    temperature: float = 1.0,
    max_output_tokens: Optional[int] = None,
    language_code: Optional[str] = None,
) -> str:
    """Returns the path of a WAV file of `text` spoken, and hands the file to the caller.

    Raises on every failure, with an error `_format_api_error` can word. This used to
    return bytes or None, and None reached the channel as "API Error or Unknown" whatever
    had gone wrong. The audio streams to disk as it comes off the socket and is wrapped
    into a WAV there, so a long line is never held in the heap.

    `max_output_tokens` caps the audio billed -- see `_speech_token_cap`. `language_code` is
    a profile's chosen `speechConfig.languageCode`. A model that refuses either field is
    asked once more without it, and is not sent it again.
    """
    if model_id.upper().startswith("GOOGLE/"):
        model_id = model_id[7:]

    capped = bool(max_output_tokens) and model_id not in _TTS_NO_OUTPUT_CAP
    languaged = bool(language_code) and model_id not in _TTS_NO_LANGUAGE
    refused_field = False
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
    if capped:
        payload["generationConfig"]["maxOutputTokens"] = max_output_tokens
    if languaged:
        payload["generationConfig"]["speechConfig"]["languageCode"] = language_code
    model_path = model_id if model_id.startswith("models/") else f"models/{model_id}"
    client = get_google_rest_client()

    # Two attempts on the *same* model, which is not the retry run_with_fallback
    # performs: that one moves to a different model, and skips entirely when no second
    # model is configured. Google documents a failure mode this cannot cover -- the
    # 3.1 TTS model occasionally emits text tokens instead of audio, which the server
    # answers with a 500, "randomly in a very small percentage of requests", with an
    # explicit recommendation to retry. Retrying the same model is the only thing that
    # helps there; a fallback model would be answering a fault the primary does not
    # actually have. A dropped connection gets the same second chance. A refusal gets
    # none, here or on the fallback: it is a decision about the text.
    body = json.dumps(payload)

    last_error = None
    for attempt in range(2):
        if attempt:
            await asyncio.sleep(random.uniform(*_TTS_RETRY_DELAY))
        # Streamed for the same reason generate_content_async is: PCM does not
        # compress on the wire the way a PNG does, so a long line costs as much
        # in transient copies as a small image.
        extractor = InlineBlobExtractor(suffix=".pcm")
        try:
            async with client.stream(
                "POST",
                f"/v1beta/{model_path}:generateContent",
                content=body,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            ) as response:
                if response.status_code != 200:
                    detail = (await response.aread()).decode('utf-8', 'replace')
                    # The speech classifier can refuse the request outright, as well as
                    # answer it with the block reasons read below.
                    if response.status_code < 500 and "PROHIBITED_CONTENT" in detail:
                        raise _speech_refusal("PROHIBITED_CONTENT")
                    # A model that takes no output cap, or no language code, names the field
                    # in its 400. The refusal is the probe: remembered, and the line asked
                    # for once more without the field, below.
                    refused = (_refused_speech_field(detail, capped, languaged)
                               if response.status_code == 400 else None)
                    if refused and len(refused[0]) < _TTS_NO_OUTPUT_CAP_MAX:
                        refused[0].add(model_id)
                        print(f"Google TTS: {model_id} refused {refused[1]}; asking again without it, and remembering.")
                        extractor.cleanup()
                        refused_field = True
                        break
                    error = Exception(f"Google API Error {response.status_code}: {detail}")
                    # 4xx is a bad request -- a voice that does not exist, a model that
                    # is not a TTS model, an exhausted quota. Sending it again changes
                    # nothing.
                    if response.status_code < 500 or attempt == 1:
                        raise error
                    last_error = error
                    print(f"Google TTS: {model_id} returned {response.status_code}; retrying once.")
                    continue
                async for chunk in response.aiter_bytes(_DOWNLOAD_CHUNK_BYTES):
                    extractor.feed(chunk)
                skeleton, blob_paths = extractor.finish()
        except (httpx.ReadTimeout, httpx.WriteTimeout) as e:
            # Not retried here: the audio may already have been generated and billed.
            # synthesise_speech allows a turn one retry after a timeout, across both models.
            extractor.cleanup()
            error = Exception(f"Google TTS: {model_id} timed out")
            error.formatted_reason = ERR_REASON_SPEECH_TIMED_OUT
            error.timed_out = True
            raise error from e
        except (httpx.RequestError, TruncatedJSONError) as e:
            extractor.cleanup()
            last_error = Exception(f"Google API Network Error: {str(e) or type(e).__name__}")
            if attempt == 1:
                raise last_error from e
            print(f"Google TTS: {model_id} connection failed ({type(e).__name__}); retrying once.")
            continue
        except BaseException:
            extractor.cleanup()
            raise

        try:
            body_json = json.loads(skeleton)
        except json.JSONDecodeError:
            for path in blob_paths:
                _remove_quietly(path)
            raise Exception(f"Google API Error: {model_id} sent a speech response that could not be read.")
        skeleton = None

        parsed = GoogleRESTResponse(body_json, blob_paths=blob_paths)
        body_json = None
        try:
            block = parsed.prompt_feedback.block_reason if parsed.prompt_feedback else None
            if block:
                raise _speech_refusal(str(block))
            candidate = parsed.candidates[0] if parsed.candidates else None
            finish = str(candidate.finish_reason or "") if candidate else ""
            if finish in _TTS_REFUSAL_FINISH_REASONS:
                raise _speech_refusal(finish)
            parts = (candidate.content.parts or []) if candidate and candidate.content else []
            for part in parts:
                inline = part.inline_data
                if not (inline and inline.data):
                    continue
                # A path either way: a long line was diverted to disk as it streamed, and a
                # short one is still bytes and is written out here.
                pcm_path = await materialise_inline_data(parsed, inline.data, ".pcm")
                if pcm_path:
                    _log_speech_usage(model_id, parsed.output_tokens, finish,
                                      max_output_tokens if capped else None)
                    return await asyncio.to_thread(_pcm_file_to_wav, pcm_path, _pcm_rate(inline.mime_type))
        finally:
            parsed.close()

        # A 200 carrying no audio part is the same fault surfacing without the 500.
        # Raising rather than returning None is deliberate: None reads as "this text
        # produced no speech" and stops there, so a configured fallback model was never
        # tried for what is a transient fault on the primary.
        last_error = Exception(f"Google TTS Error: {model_id} returned no audio data.")
        last_error.formatted_reason = ERR_REASON_NO_AUDIO
        if attempt == 0:
            print(f"Google TTS: {model_id} returned no audio; retrying once.")

    if refused_field:
        # The refused field is remembered now and not sent again, so the same refusal
        # cannot bring the line back here.
        return await generate_google_tts_audio(api_key, model_id, text, voice_name=voice_name,
                                               temperature=temperature,
                                               max_output_tokens=max_output_tokens,
                                               language_code=language_code)
    raise last_error


class GoogleSpeechModel:
    """A Gemini TTS model, as `APIService._instantiate_model(speech=True)` hands it out.

    It holds the key the factory resolved -- after the data policy's gate -- so speech never
    looks one up for itself, and `synthesise` is what the factory's cooldown tracking wraps,
    so a 429 rests the key as it does for every other slot.
    """

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

    async def synthesise(self, transcript: str, directed_prompt: Optional[str] = None,
                         voice_name: Optional[str] = None, temperature: float = 1.0,
                         voice_sample=None, max_bytes: Optional[int] = None,
                         speed: Optional[float] = None, language_code: Optional[str] = None) -> str:
        """The path of an audio file of the reply spoken; the caller owns it.

        Gemini is sent the Director's Desk prompt when there is one. A voice it does not
        carry -- one a profile kept from an OpenRouter model -- is swapped for the default
        rather than sent to a 400. Gemini clones no voice, so `voice_sample` is never loaded,
        and takes no speed. The audio billed is capped by the transcript and `max_bytes`,
        never by the prompt.
        """
        voice = TTS_VOICE_LOOKUP.get((voice_name or "").lower(), DEFAULT_SPEECH_VOICE)
        return await generate_google_tts_audio(
            self.api_key, self.model_name, directed_prompt or transcript,
            voice_name=voice, temperature=temperature,
            max_output_tokens=min(_speech_token_cap(transcript, max_bytes),
                                  defaultConfig.LIMIT_OUTPUT_TOKENS),
            language_code=language_code)
