"""The OpenRouter speech adapter.

`POST /api/v1/audio/speech` answers with the audio itself rather than JSON, so the body is
written to a file as it arrives and the caller gets a path, as on every other media path. MP3
is asked for -- OpenRouter does not document the PCM sample rate its hosts send, and MP3 is a
fraction of a WAV's size against Discord's upload limit -- but the file is named for the type
the host labels its answer with, since a host may not honour the request.

What is sent differs from the Google adapter in three ways. The reply goes alone, without the
Director's Desk, and without audio tags (bar `_AUDIO_TAG_MODELS`) or stand-alone actions: most
of these models read `input` aloud word for word, headings and all. The
voice has to be one the model lists, which differs per model -- see `_voice_for`. And a model
whose host clones voices is sent the profile's voice sample in place of a preset voice,
spliced into the body as it streams like every other file this bot sends a provider.

Nothing here sends `provider.data_collection` -- the endpoint takes no such field. Whether a
model may serve a server at all is decided before a request exists, in
`APIService._instantiate_model`.
"""

import os
import re
import tempfile
from typing import Awaitable, Callable, Optional, Sequence

import httpx
import orjson as json

from ...utils.constants import (
    ERR_REASON_NO_AUDIO, ERR_REASON_NOTHING_TO_SPEAK, ERR_REASON_SPEECH_TIMED_OUT,
    OPENROUTER_DATA_POLICY_BLOCKED,
)
from ...utils.http_client import get_openrouter_client
from .streaming import (
    _DOWNLOAD_CHUNK_BYTES, _FILE_BLOB_TOKEN, _aiter_streamed_body, _close_body_segments,
    _plan_streamed_body,
)

_SPEECH_URL = "https://openrouter.ai/api/v1/audio/speech"

#: Two minutes without a byte, as the Google client allows. synthesise_speech retries one
#: timeout per turn, and two of these have to fit inside the delivery watchdog.
_TIMEOUT = httpx.Timeout(120.0, connect=10.0)

#: The file suffix for each audio type a host may label its answer with. An answer labelled
#: anything else is taken to be the MP3 that was asked for.
_AUDIO_SUFFIXES = {
    "audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/wav": ".wav", "audio/x-wav": ".wav",
    "audio/wave": ".wav", "audio/ogg": ".ogg", "audio/opus": ".ogg", "audio/flac": ".flac",
    "audio/aac": ".aac", "audio/mp4": ".m4a", "audio/webm": ".webm",
}

#: A voice sample's type as the `format` its reference part names, where there is one.
_SAMPLE_FORMATS = {
    "audio/wav": "wav", "audio/x-wav": "wav", "audio/wave": "wav", "audio/mpeg": "mp3",
    "audio/mp3": "mp3", "audio/ogg": "ogg", "audio/flac": "flac", "audio/mp4": "m4a",
    "audio/webm": "webm",
}

#: The longest transcript OpenRouter takes beside a voice sample.
_TRANSCRIPT_MAX = 10_000

#: Discord markup a speech model would read out: custom emoji (the name is kept), links (a
#: masked link keeps its text), subtext markers, and the emphasis, strike and spoiler characters.
_CUSTOM_EMOJI = re.compile(r"<a?:(\w+):\d+>")
_MASKED_LINK = re.compile(r"\[([^\[\]\n]+)\]\(<?https?://[^\s)>]+>?\)")
_LINK = re.compile(r"<?https?://\S+>?")
_SUBTEXT = re.compile(r"^-#\s*", re.MULTILINE)
_MARKERS = re.compile(r"[*_~`|]+")
_SPACES = re.compile(r"[ \t]{2,}")

#: What a reply says about its own delivery, which a model that does not understand it reads
#: out as words: `[whispers]` as "whispers", `*sighs*` as "sighs".
_AUDIO_TAG = re.compile(r"\[[^\[\]\n]*\]")
_ITALIC = re.compile(r"(?<!\*)\*(?!\*)([^*\n]+)\*(?!\*)")

#: The models that take audio tags in the text itself, so theirs are left in: Fish Audio S2
#: reads any description, Qwen Audio 3.0 a list of its own. Grok and MiniMax take tags too,
#: but spelt differently, and a tag in the wrong spelling is read out like any other.
_AUDIO_TAG_MODELS = ("fish-audio/s2", "qwen/qwen-audio-3")


def _is_action(text: str, match: "re.Match") -> bool:
    """Whether an italic span is a roleplay action standing as a sentence of its own, rather
    than a stressed word: it starts a line or follows the end of a sentence (or another
    action), and it ends the line or a new sentence follows it. "*sighs* Fine." is an action;
    "I *never* said that" and "*Never* again" are not."""
    before = text[text.rfind("\n", 0, match.start()) + 1:match.start()].rstrip().rstrip("\"'”’)")
    if before and before[-1] not in ".!?…*":
        return False
    end = text.find("\n", match.end())
    after = text[match.end():end if end != -1 else len(text)]
    following = after.lstrip()
    return not following or (after[0].isspace() and not following[0].islower())


def speakable_text(text: str, keep_tags: bool = False) -> str:
    """The reply as a speech model should hear it: the words, without Discord's markup, a
    roleplay action, or -- unless `keep_tags` -- an audio tag."""
    text = _CUSTOM_EMOJI.sub(r"\1", text or "")
    text = _MASKED_LINK.sub(r"\1", text)
    text = _LINK.sub("", text)
    source = _SUBTEXT.sub("", text)
    text = _ITALIC.sub(lambda m: "" if _is_action(source, m) else m.group(0), source)
    if not keep_tags:
        text = _AUDIO_TAG.sub("", text)
    text = _MARKERS.sub("", text)
    return _SPACES.sub(" ", text).strip()


def _voice_for(requested: Optional[str], offered: Optional[Sequence[str]]) -> Optional[str]:
    """The voice to send: the one asked for if the model lists it, else the model's first.

    A profile keeps one voice across models, so the one it stores is often another model's
    -- a Gemini voice on a Voxtral fallback. `offered` is None for a model the catalogue does
    not know, which is sent the voice as asked, there being nothing to check it against. A
    model that lists no voices is sent none.
    """
    if offered is None:
        return requested or None
    if not offered:
        return None
    if requested in offered:
        return requested
    by_lower = {voice.lower(): voice for voice in offered}
    return by_lower.get((requested or "").lower(), offered[0])


def _remove(path: Optional[str]) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except OSError:
        pass


class OpenRouterSpeechModel:
    def __init__(self, model_name: str, api_key: str, voices: Optional[Sequence[str]] = None,
                 clones: bool = False):
        self.model_name = model_name.replace("OPENROUTER/", "")
        self.api_key = api_key
        #: The voices the speech catalogue lists for this model, or None when it does not
        #: list the model.
        self.voices = tuple(voices) if voices is not None else None
        #: Whether the catalogue lists a host of this model that takes a voice sample.
        self.clones = clones

    async def synthesise(self, transcript: str, directed_prompt: Optional[str] = None,
                         voice_name: Optional[str] = None, temperature: float = 1.0,
                         voice_sample: Optional[Callable[[], Awaitable[Optional[dict]]]] = None,
                         max_bytes: Optional[int] = None, speed: Optional[float] = None,
                         language_code: Optional[str] = None) -> str:
        """The path of an audio file of `transcript` spoken; the caller owns the file.

        `voice_sample` decrypts the profile's sample to a temp file. It is called only for a
        model that clones, and the file is removed here once the request is done. `speed` is
        sent as given, and a host that does not support it ignores it.
        `directed_prompt`, `temperature`, `max_bytes` and `language_code` are the Google
        adapter's and go unused: the speech endpoint takes no such fields.
        """
        text = speakable_text(transcript, keep_tags=self.model_name.lower().startswith(_AUDIO_TAG_MODELS))
        if not text:
            # Raised before a request exists, and not retried: a fallback would be handed
            # the same empty reply.
            error = Exception("OpenRouter speech: the reply has nothing to say aloud")
            error.formatted_reason = ERR_REASON_NOTHING_TO_SPEAK
            error.retryable = False
            raise error

        sample = await voice_sample() if (voice_sample and self.clones) else None
        try:
            return await self._request(text, voice_name, sample, speed)
        finally:
            if sample:
                _remove(sample.get("path"))

    async def _request(self, text: str, voice_name: Optional[str], sample: Optional[dict],
                       speed: Optional[float] = None) -> str:
        payload = {"model": self.model_name, "input": text, "response_format": "mp3"}
        if speed:
            payload["speed"] = speed
        if sample:
            # A cloned voice takes the place of a preset one rather than competing with it.
            audio = {"data": f"data:{sample['mime_type']};base64,{_FILE_BLOB_TOKEN.format(0)}"}
            if _SAMPLE_FORMATS.get(sample["mime_type"]):
                audio["format"] = _SAMPLE_FORMATS[sample["mime_type"]]
            references = [{"type": "input_audio", "input_audio": audio}]
            if sample.get("transcript"):
                references.append({"type": "text", "text": sample["transcript"][:_TRANSCRIPT_MAX]})
            payload["input_references"] = references
        else:
            voice = _voice_for(voice_name, self.voices)
            if voice:
                payload["voice"] = voice
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://discord.com",
            "X-Title": "MimicAI Discord Bot",
        }

        segments = None
        if sample:
            # Explicit Content-Length keeps httpx off chunked encoding, as in OpenRouterModel.
            segments, content_length = _plan_streamed_body(payload, [sample["path"]])
            body = _aiter_streamed_body(segments)
            headers["Content-Length"] = str(content_length)
        else:
            body = json.dumps(payload)
        payload.clear()

        path = None
        written = 0
        try:
            async with get_openrouter_client().stream(
                    "POST", _SPEECH_URL, content=body, headers=headers, timeout=_TIMEOUT) as response:
                content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                # A refusal is a small JSON body, sometimes under a 200. Anything else under
                # a 200 is the audio, whatever type its host labels it.
                if response.status_code != 200 or content_type == "application/json":
                    detail = (await response.aread()).decode("utf-8", "replace")
                    error = Exception(f"OpenRouter API Error {response.status_code}: {detail}")
                    # Said plainly, and carried whole: _format_api_error cuts at 80 characters.
                    if "data policy" in detail.lower():
                        error.formatted_reason = OPENROUTER_DATA_POLICY_BLOCKED
                    raise error
                fd, path = tempfile.mkstemp(suffix=_AUDIO_SUFFIXES.get(content_type, ".mp3"))
                with os.fdopen(fd, "wb") as out:
                    async for chunk in response.aiter_bytes(_DOWNLOAD_CHUNK_BYTES):
                        out.write(chunk)
                        written += len(chunk)
        except (httpx.ReadTimeout, httpx.WriteTimeout) as e:
            # Not retried here: the audio may already have been generated and billed.
            # synthesise_speech allows a turn one retry after a timeout, across both models.
            _remove(path)
            error = Exception(f"OpenRouter speech: {self.model_name} timed out")
            error.formatted_reason = ERR_REASON_SPEECH_TIMED_OUT
            error.timed_out = True
            raise error from e
        except httpx.RequestError as e:
            _remove(path)
            raise Exception(f"OpenRouter Network Error: {str(e) or type(e).__name__}")
        except BaseException:
            # CancelledError included, which is exactly when a half-written file would
            # otherwise be left behind.
            _remove(path)
            raise
        finally:
            if segments is not None:
                # A request refused before its body was read never exhausts the generator
                # that would otherwise close the sample's handle.
                _close_body_segments(segments)

        if not written:
            _remove(path)
            error = Exception(f"OpenRouter speech: {self.model_name} returned no audio.")
            error.formatted_reason = ERR_REASON_NO_AUDIO
            raise error
        return path
