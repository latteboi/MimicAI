"""Simulated vision: a cheap model reads an attachment for a profile that cannot.

Reached only from the degrade path in `reply.py`, and only once both of a profile's own
models have refused the attachment. The description is written into that one request and
nowhere else -- it is never logged, never shown to a profile that did not ask for it, and
never seen by a profile whose model read the file itself. Same rule as grounding and URL
context: what one participant paid to fetch is that participant's context, not the room's.

Built like the grounding summariser, for the same reasons: one utility model, one pass,
`run_with_fallback`'s retry policy, and a failure that costs the caller nothing but the
description -- a profile whose describe pass fails is a profile in `off` mode, which is a
complete behaviour rather than an error state.
"""
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

from ...utils.constants import (
    DEFAULT_MEDIA_DESCRIPTION, MEDIA_DESCRIBER_FALLBACK, MEDIA_DESCRIBER_MODEL,
    MEDIA_DESCRIBER_PAID, MEDIA_DESCRIBER_RESOLUTION, GREEDY_SAMPLING,
    MEDIA_DESCRIPTION_MAX_CHARS, MEDIA_DESCRIPTION_NONE,
)
from ...utils.helpers import _resolve_safety_settings, resolve_thinking_params


def _media_key(part: Any) -> Optional[str]:
    """What identifies an attachment between requests, or None when nothing does.

    A Discord CDN url carries a signature that is reissued on every fetch, so the query
    string is cut off: the same file arriving twice has to look like the same file, or a
    second character in the same round buys a second description of one picture. What is
    left is the channel and attachment ids, which is exactly the identity wanted.

    An inline part -- bytes with no url -- has nothing stable and cheap to hash, so it is
    described every time rather than risking two different files sharing a key.
    """
    if isinstance(part, dict):
        url = part.get("url")
        if url:
            return str(url).split("?", 1)[0]
    return None


def _file_name(part: Any, n: int) -> str:
    """What the character's turn calls this attachment, or `File n` when nothing does.

    Discord's CDN keeps the uploaded name as the url's last segment -- the same name the
    `[Attached Image: cat.png]` tag carries -- so a description headed with it is one the
    character can match to the tag without counting.
    """
    url = part.get("url") if isinstance(part, dict) else None
    name = unquote(str(url).split("?", 1)[0].rsplit("/", 1)[-1]) if url else ""
    return name or f"File {n}"


def _batch_key(parts: List[Any]) -> Optional[str]:
    """One key for the whole batch, which is the unit a round describes in.

    A round hands every participant the same attachments, so the second blind character
    to speak finds the first one's description already waiting. Describing per file would
    cache more tightly and cost N calls where this costs one; the batch is the unit
    because the round is.
    """
    keys = []
    for part in parts:
        key = _media_key(part)
        if key is None:
            return None
        keys.append(key)
    return "|".join(keys) if keys else None


class MediaDescriptionMixin:
    """The describe pass for `unreadable_media_mode = simulated`."""

    async def _describe_media(self, media_parts: List[Any], *, channel, p_settings: Dict,
                              owner_id: int, user_id: int,
                              meta: Optional[Dict] = None) -> Tuple[Optional[str], Optional[str]]:
        """Describe `media_parts`, as (description, model name), or (None, None).

        Returns None for every failure there is -- no key for either provider, a key the
        data policy will not spend here, a refusal by both, an empty answer, a timeout.
        The caller degrades to `off` on None, so a server that cannot run this pass loses
        the description and keeps the reply, rather than the other way round.

        `meta`, when given, receives the tokens this pass cost, so the turn's trace can
        show the description as a line of its own instead of hiding it in the reply's
        numbers or leaving it out of the bill entirely.
        """
        if not media_parts:
            return None, None

        cache = self.cog.media_descriptions
        key = _batch_key(media_parts)
        if key and key in cache:
            # The model that answered is cached with its description: the fallback reads
            # the file when the primary cannot, and the warning under the reply has to
            # name whichever of them the reader is actually being told about.
            return cache[key]

        instruction = self.cog.global_prompts.get("MEDIA_DESCRIPTION", DEFAULT_MEDIA_DESCRIPTION)
        # One turn carrying the files, never the files alongside it. Every adapter reads
        # `contents` as a list of turns and asks each item for its `parts`; a bare media
        # dict passed as an item has no `parts`, so it arrives as an empty turn and the
        # file is dropped without a word. The model is then asked to describe
        # attachments it was never sent, and obligingly invents some. No other text: the
        # system instruction is the whole request, and several files each follow their
        # name, which is how the prompt tells the model to head each description.
        parts = list(media_parts) if len(media_parts) == 1 else [
            item for n, part in enumerate(media_parts, 1) for item in (_file_name(part, n), part)]
        contents = [{"role": "user", "parts": parts}]

        async def run(name: str, _is_fallback: bool):
            model = self.cog.api_service._instantiate_model(
                name, channel.guild.id if channel.guild else None, user_id,
                system_instruction=instruction,
                # The destination channel's settings, exactly as the reply itself
                # resolves them. Built with the defaults instead, every attachment in an
                # age-restricted session comes back refused, which reads as the feature
                # being broken rather than as a filter doing its job.
                safety_settings=_resolve_safety_settings(channel, p_settings),
                # `utility`: a one-shot internal pass with no character in it to
                # configure, like the Director's Note and the profile generator.
                thinking_params=resolve_thinking_params({}, "utility", "primary"),
                # The one setting this pass overrides. Its own look at the file is the
                # profile's only look at it, so it is bought at full detail whatever the
                # profile spends on media it can read itself. One setting, both wire
                # shapes: Google's enum, and OpenRouter's coarser `detail` hint.
                profile_settings={"media_input_resolution": MEDIA_DESCRIBER_RESOLUTION},
                config_owner_id=owner_id)
            response = await model.generate_content_async(
                contents, generation_config=dict(GREEDY_SAMPLING))
            answer = (getattr(response, "text", "") or "").strip()
            if not answer or answer.lower().startswith(MEDIA_DESCRIPTION_NONE):
                # Retried on the next model, unlike the other utility passes, which
                # treat an empty answer as a decision about the content and stop. The
                # last sits on another provider behind another filter: the Google one
                # whose safety settings this channel has already stood down -- a file
                # the first will not describe is what the last is for.
                raise ValueError("The describer returned no description")
            return response, answer

        try:
            (response, text), used, _was_fallback = await self.cog.api_service.run_with_fallback(
                MEDIA_DESCRIBER_MODEL, (MEDIA_DESCRIBER_PAID, MEDIA_DESCRIBER_FALLBACK), run,
                label="Attachment describer")
        except Exception as e:
            print(f"Attachment describer: {type(e).__name__}: {e}")
            self._log_description_call(channel, user_id, MEDIA_DESCRIBER_MODEL, "api_error")
            return None, None

        text = text[:MEDIA_DESCRIPTION_MAX_CHARS * len(media_parts)]
        if isinstance(meta, dict):
            tokens = ((getattr(response, "input_tokens", 0) or 0)
                      + (getattr(response, "output_tokens", 0) or 0))
            if tokens:
                meta["media_description_tokens"] = tokens
        if key:
            cache[key] = (text, used)
        self._log_description_call(channel, user_id, used, "success")
        return text, used

    def _log_description_call(self, channel, user_id: int, model: str, status: str) -> None:
        """One row per describe pass, under its own context.

        Its own, rather than folded into the turn that triggered it: this is a second
        model on a second key, and a bill that arrives without a row naming it is the
        kind nobody can account for afterwards.
        """
        self.cog._log_api_call(
            user_id=user_id, guild_id=channel.guild.id if channel.guild else None,
            context="media_description", model_used=model, status=status)
