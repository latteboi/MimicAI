"""One reply: the primary model, then the fallback, and what the turn records about it.

The round worker and regeneration both run this -- build the model, generate, fall back,
turn the result into text and warnings, and write the trace. Each used to carry its own
copy, and the copies drifted: regeneration built its thinking parameters by hand, so a
profile's fallback thinking level never reached a regenerated reply, and it routed its
fallback with a narrower provider guess than the worker's. Both recorded the primary as
the model that answered when the fallback had.
"""
import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...utils import mem_probe
from ...utils.constants import (
    ERR_GENERAL_ERROR, ERR_REASON_EMPTY_RESPONSE, ERR_REASON_TIMEOUT_BOTH, ERR_SAFETY_BLOCK,
    LIMIT_FUNCTION_CALL_ROUNDS,
    MEDIA_DESCRIBED_NOTE, MEDIA_UNREADABLE_NOTE, WARN_BOTH_MODELS_FAILED, WARN_FALLBACK_USED,
    WARN_MAIN_MODEL_FAILED, WARN_MEDIA_DESCRIBED, WARN_MEDIA_UNREADABLE,
)
from ...utils.helpers import (
    _add_inline_citations, _format_api_error, _resolve_safety_settings, _scrub_response_text,
    clean_model_name, is_real_model, media_kinds, record_billed_usage,
    resolve_function_tools, resolve_native_tools,
    resolve_thinking_params, resolve_unreadable_media_mode, split_media_parts,
    unreadable_media_modality,
)
from ._shared import _strip_neuro_update_and_scrub
from .tool_loop import (FunctionContext, carry_forward, execute, exchange_turns,
                        function_call_label, needs_continuation, split_calls)

#: Sampling keys a profile may set beyond temperature, top_p and top_k. Sent only when set.
_ADVANCED_SAMPLING_KEYS = ("frequency_penalty", "presence_penalty", "repetition_penalty", "min_p", "top_a")


@dataclass
class ReplyAttempt:
    """What one run of primary-then-fallback produced.

    `response` is the last one received, valid or not: a blocked reply carries the block
    reason the warnings report. `error` is why there is no reply, `main_error` why the
    primary failed when the fallback answered instead.
    """
    primary_name: Optional[str] = None
    fallback_name: Optional[str] = None
    response: Any = None
    primary: Any = None
    answered_by: Any = None
    fallback_used: bool = False
    main_error: Optional[str] = None
    error: Optional[str] = None
    status: str = "api_error"
    #: Set when the attachments were taken out and the turn retried without them: what
    #: kind they were ("images"), and who described them first, if anyone did.
    media_dropped: Optional[str] = None
    media_described_by: Optional[str] = None
    #: Merged into the turn's trace. Carries what the describe pass cost, which belongs
    #: on the turn that paid for it rather than inside the reply's own token counts.
    extra_meta: Dict[str, Any] = field(default_factory=dict)
    #: Memories the character fetched by calling `recall`, as text. They reach the model
    #: through a function response rather than the injected archive block, so nothing
    #: downstream can recover them from the prompt -- and the trace is judged on them.
    recalled_memories: List[str] = field(default_factory=list)
    #: Pages the character's own `search_web` calls cited. The legacy grounding mode
    #: collects its sources before the turn starts and the native one leaves them on the
    #: final response; this mode's arrive mid-turn on a function result, so without this
    #: a grounded reply would post with no citations under it and an empty audit line.
    search_sources: List[Dict[str, str]] = field(default_factory=list)

    @property
    def has_fallback(self) -> bool:
        # is_real_model, not truthiness: an explicit "no fallback" reads back as the
        # string NONE, which is truthy and unequal to the primary.
        return is_real_model(self.fallback_name) and self.primary_name != self.fallback_name

    @property
    def model_label(self) -> str:
        """The model that answered, as the trace shows it."""
        name = (getattr(self.answered_by or self.primary, 'model_name', None)
                or (self.fallback_name if self.fallback_used else self.primary_name) or "Unknown")
        return clean_model_name(name)


@dataclass
class ReplyText:
    """The text a reply posts and records, and the warnings shown under it.

    `sources` is None when there was no text to cite.
    """
    text: str
    warnings: List[str] = field(default_factory=list)
    blocked: bool = False
    neuro_state: Optional[Dict[str, int]] = None
    sources: Optional[List[Dict[str, str]]] = None


def reply_gen_config(p_settings: Dict, temperature, top_p, top_k) -> Dict:
    advanced = {k: p_settings[k] for k in _ADVANCED_SAMPLING_KEYS if p_settings.get(k) is not None}
    return {"temperature": temperature, "top_p": top_p, "top_k": top_k, "_advanced_params": advanced}


def response_sources(candidate) -> List[Dict[str, str]]:
    """Web and URL-context sources a response cites, as {'uri', 'title'}."""
    sources = []
    chunks = getattr(getattr(candidate, 'grounding_metadata', None), 'grounding_chunks', None)
    for chunk in chunks or []:
        if hasattr(chunk, 'web'):
            sources.append({'uri': chunk.web.uri, 'title': chunk.web.title})
    urls = getattr(getattr(candidate, 'url_context_metadata', None), 'url_metadata', None)
    for u in urls or []:
        if getattr(u, 'retrieved_url', None):
            sources.append({'uri': u.retrieved_url, 'title': 'URL Context'})
    return sources


def _merge_sources(*groups: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Every source once, in the order first seen, keyed on the URI."""
    seen, out = set(), []
    for group in groups:
        for source in group or ():
            uri = source.get('uri')
            if uri and uri not in seen:
                seen.add(uri)
                out.append(source)
    return out


def reply_meta(attempt: ReplyAttempt, *, duration: float, training_examples, ltm_recall_text,
               sources, neuro_state, critic: Optional[Dict] = None) -> Dict:
    """The trace a generated turn carries. Small: it rides in every turn of the log."""
    response = attempt.response
    meta = {
        "duration": round(duration, 2),
        "model": attempt.model_label,
        "fallback": attempt.fallback_used,
        "input_tokens": getattr(response, 'input_tokens', 0) if response else 0,
        "output_tokens": getattr(response, 'output_tokens', 0) if response else 0,
        "reasoning_tokens": getattr(response, 'reasoning_tokens', 0) if response else 0,
        "training_recalled": len(training_examples) if training_examples else 0,
        "grounding_sources": [s.get('uri') for s in sources or [] if isinstance(s, dict) and s.get('uri')],
        "ltms_recalled": [],
    }
    record_billed_usage(meta, response)
    # Both halves of retrieval, in the order the character received them: what the
    # automatic pass injected, then whatever it fetched by calling `recall`. Only the
    # first used to be recorded, and a tool result never touches the prompt text -- so
    # switching Memory Search on, which raises the automatic threshold, made the audit
    # read `0 memories` on precisely the turns that had recalled the most.
    lines = ([l.strip() for l in ltm_recall_text.split('\n')
              if l.strip() and not l.startswith("<")] if ltm_recall_text else [])
    lines.extend(m.strip() for m in attempt.recalled_memories if m and m.strip())
    seen = set()
    for line in lines:
        clipped = line[:100] + "..." if len(line) > 100 else line
        if clipped not in seen:
            seen.add(clipped)
            meta["ltms_recalled"].append(clipped)
    if neuro_state:
        meta["neuro_state"] = neuro_state
    if critic:
        meta["critic"] = critic
    # Sparse, like everything else here: a turn that read its attachments itself says
    # nothing at all about them.
    if attempt.media_dropped:
        meta["media_dropped"] = attempt.media_dropped
    if attempt.media_described_by:
        meta["media_described_by"] = attempt.media_described_by
    meta.update(attempt.extra_meta)
    return meta


class ReplyMixin:
    """The generate-and-fall-back step and the text it yields, for GenerationService."""

    def _participant_names(self, session: Dict) -> List[str]:
        """Every seated profile's display name, which a reply must not prefix itself with."""
        return [self._resolve_appearance_data(p['owner_id'], p['profile_name'])[0]
                for p in session.get("profiles", [])]

    async def _attempt_reply(self, *, channel, participant, p_settings: Dict, owner_id: int,
                             user_id: int, system_instruction, primary_model: str,
                             fallback_model_name: Optional[str], history: List, gen_config: Dict,
                             msg_a_id, app_name: str, app_avatar, state_container: Dict,
                             participant_names: List[str], log_context: Optional[str] = None,
                             probe_label: Optional[str] = None) -> ReplyAttempt:
        """Generates with the primary, then the fallback if the primary fails.

        `state_container` is the caller's open container and is updated in place. A
        cancellation propagates: the caller owns the teardown, which differs -- the worker
        deletes its placeholder, regeneration puts the original text back.
        """
        attempt = ReplyAttempt(primary_name=primary_model, fallback_name=fallback_model_name)
        safety_settings = _resolve_safety_settings(channel, p_settings)
        tools = resolve_native_tools(p_settings)
        # The one path with a function loop, so the one path that may declare a
        # function whose answer the model has to be handed back.
        fn_tools = resolve_function_tools(p_settings, with_loop=True)

        def build(name: str, role: str, **key_errors):
            thinking = resolve_thinking_params(p_settings, "response", role)
            # Carried across because this dict has always held it. Nothing in any adapter
            # reads it.
            thinking["thinking_persistence"] = p_settings.get("thinking_persistence", 10)
            return self.cog.api_service._instantiate_model(
                name, channel.guild.id, user_id, system_instruction, safety_settings,
                thinking, tools, p_settings, config_owner_id=owner_id,
                function_tools=fn_tools, **key_errors)

        fn_ctx = FunctionContext(
            owner_id=owner_id,
            profile_name=(participant or {}).get('profile_name', ''),
            author_dn=app_name,
            guild_id=channel.guild.id if getattr(channel, 'guild', None) else None,
            triggering_user_id=user_id,
            # Already resolved for this reply, off the destination channel. `search_web`
            # asks a second model on behalf of this turn and must be held to the same
            # rules the turn is, not to Google's unset defaults.
            safety_settings=safety_settings)

        async def generate(model, is_fallback: bool, contents: Optional[List] = None):
            # A copy, because a function round appends to it: `history` is the caller's
            # and is reused by the fallback attempt, which must start from the same
            # conversation rather than inheriting the primary's abandoned lookups.
            turn = list(history if contents is None else contents)
            carried, ran, found, cited = [], [], [], []
            for _round in range(LIMIT_FUNCTION_CALL_ROUNDS + 1):
                attempt.response, _ = await self._generate_with_heartbeat(
                    model, turn, gen_config, channel,
                    participant, msg_a_id, is_fallback=is_fallback,
                    app_name=app_name, app_avatar=app_avatar, existing_state=state_container)
                if not attempt.response or not attempt.response.candidates:
                    raise ValueError("Response blocked or empty")
                answering, recording = split_calls(attempt.response)
                # Not `if not answering`: a model that recorded its mood and said
                # nothing has to be let go on, or the turn has no reply at all.
                pending = needs_continuation(attempt.response, answering, recording)
                if not pending:
                    break
                # Kept for the response that finally speaks -- see carry_forward.
                carried.extend(recording)
                results = await execute(self.cog, pending, fn_ctx)
                for call, value in results:
                    if call not in answering:
                        continue
                    ran.append(function_call_label(call, value))
                    if isinstance(value, dict):
                        found.extend(m for m in (value.get("memories") or ()) if isinstance(m, str))
                        cited.extend(src for src in (value.get("sources") or ())
                                     if isinstance(src, dict) and src.get("uri"))
                turn.extend(exchange_turns(results, getattr(attempt.response, 'text', "") or ""))
            else:
                # Out of budget with a call still pending. The response below may
                # therefore have no text, which raises like any other empty reply --
                # but the audit should say why rather than leaving it as a mystery.
                attempt.extra_meta["function_calls_truncated"] = True
            carry_forward(attempt.response, carried)
            if ran:
                attempt.extra_meta["function_calls"] = ran
            # Assigned, not extended: the fallback ran its own searches from the same
            # history, and the trace describes the reply that was actually posted.
            attempt.recalled_memories = found
            attempt.search_sources = cited
            if not attempt.response or not attempt.response.candidates:
                raise ValueError("Response blocked or empty")
            text = getattr(attempt.response, 'text', "").strip()
            if not _strip_neuro_update_and_scrub(text, participant_names):
                raise ValueError("Empty Response (AI produced no text content)")
            attempt.answered_by = model

        init_error = None
        try:
            attempt.primary = build(
                primary_model, "primary",
                openrouter_key_error=f"API Configuration Error: OpenRouter API Key missing for this server. Cannot load model '{primary_model}'.",
                google_key_error=f"API Configuration Error: Google API Key missing for this server. Cannot load model '{primary_model}'.")
        except ValueError as e:
            init_error = str(e)
        except Exception as e:
            init_error = f"Model Initialization Error: Failed to instantiate model '{primary_model}'. {e}"

        try:
            # A primary that could not be constructed is an error like any other, so a
            # configured fallback -- quite possibly a provider whose key is present -- still
            # gets its chance. The text rides on the exception already phrased for the
            # user; _format_api_error would cut it to 80 characters.
            if attempt.primary is None:
                error = RuntimeError(init_error or "Internal API Initialization Error")
                error.formatted_reason = init_error or "Internal API Initialization Error"
                raise error
            probe = mem_probe.probe(probe_label, peak=False) if probe_label else contextlib.nullcontext()
            with probe:
                await generate(attempt.primary, False)
            attempt.status = "success"
        except Exception as e:
            timed_out = isinstance(e, TimeoutError)
            attempt.main_error = getattr(e, 'formatted_reason', None) or _format_api_error(e)
            # Which model refused to read the attachment, if that is what happened. The
            # primary is preferred: it is the character's own voice, and the fallback is
            # only standing in because of a file neither of them was asked about.
            blind_model = attempt.primary if unreadable_media_modality(e) else None
            fallback_model = None
            if not attempt.has_fallback:
                attempt.error = attempt.main_error
            else:
                try:
                    fallback_model = build(fallback_model_name, "fallback")
                    await generate(fallback_model, True)
                    attempt.fallback_used = True
                    if log_context:
                        self.cog._log_api_call(user_id=user_id, guild_id=channel.guild.id,
                                               context=f"{log_context}_fallback",
                                               model_used=fallback_model_name, status="success")
                except Exception as retry_e:
                    attempt.error = (ERR_REASON_TIMEOUT_BOTH if timed_out and isinstance(retry_e, TimeoutError)
                                     else _format_api_error(retry_e))
                    if blind_model is None and unreadable_media_modality(retry_e):
                        blind_model = fallback_model

            # Both models have now refused the attachment -- if the fallback had read it,
            # its answer would be sitting in `attempt` and this would not run. Only now is
            # the profile's unreadable-media setting reached: a profile whose fallback has
            # vision never sees it, and keeps answering with the fallback as it always did.
            if blind_model is not None and attempt.answered_by is None:
                await self._retry_without_media(
                    attempt, generate, blind_model, history, p_settings=p_settings,
                    channel=channel, owner_id=owner_id, user_id=user_id,
                    is_fallback=blind_model is fallback_model)
        finally:
            if log_context:
                # `or primary_model`: a row naming the model that could not be built says
                # more than one naming none.
                self.cog._log_api_call(user_id=user_id, guild_id=channel.guild.id, context=log_context,
                                       model_used=attempt.primary or primary_model, status=attempt.status)
        return attempt

    async def _retry_without_media(self, attempt: ReplyAttempt, generate, model, history: List,
                                   *, p_settings: Dict, channel, owner_id: int, user_id: int,
                                   is_fallback: bool) -> None:
        """Run the turn again with the attachments taken out, and say so in their place.

        The reply used to be abandoned here: the profile posted its `error_response` and
        dropped out of the round because somebody attached a picture, while every other
        character in the session carried on. A conversation is not the attachment, and a
        character that cannot see one can still answer what was said -- so the file comes
        out, something goes in to say it was there, and the turn is asked again.

        What goes in depends on the profile's `unreadable_media_mode`. `off` names the
        kind and forbids guessing; `simulated` buys a description first and hands that
        over instead, falling back to `off` when there is no description to be had -- no
        Google key, a refusal, a timeout. The filename is in the turn's own text either
        way, written there when the message was taken in, for every model.

        The history is rebuilt rather than edited: the request that failed is what the
        trace and any later attempt refer to, and it has to keep saying what was sent.
        """
        blind_history, media = split_media_parts(history)
        if not media:
            return

        kinds = media_kinds(media)
        parts: List[Any] = [MEDIA_UNREADABLE_NOTE.format(kinds=kinds)]
        described_by = None
        if resolve_unreadable_media_mode(p_settings) == "simulated":
            description, describer = await self._describe_media(
                media, channel=channel, p_settings=p_settings, owner_id=owner_id,
                user_id=user_id, meta=attempt.extra_meta)
            if description:
                described_by = describer
                parts = [MEDIA_DESCRIBED_NOTE.format(kinds=kinds),
                         f"<attachment_description>\n{description}\n</attachment_description>"]

        # Onto the last user turn, where the attachments themselves were, so the note sits
        # against the message it is about rather than at the top of a scene.
        if blind_history and blind_history[-1].get("role") == "user":
            blind_history[-1] = {**blind_history[-1],
                                 "parts": list(blind_history[-1].get("parts") or []) + parts}
        else:
            blind_history.append({"role": "user", "parts": parts})

        try:
            await generate(model, is_fallback, contents=blind_history)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # The refusal already in `attempt.error` is kept: it is the one that explains
            # the turn, and this second failure is usually the same model saying so twice.
            # Phrased rather than raw, for the reason `run_with_fallback`'s line is.
            print(f"Retry without attachments failed: {type(e).__name__}: {_format_api_error(e)}")
            return

        attempt.media_dropped = kinds
        attempt.media_described_by = described_by
        attempt.error = None
        attempt.fallback_used = is_fallback
        if not is_fallback:
            attempt.status = "success"

    def _reply_text(self, attempt: ReplyAttempt, p_settings: Dict, owner_id: int,
                    profile_name: str, participant_names: List[str]) -> ReplyText:
        """The text to post and record, the warnings to show under it, and what it cites.

        Applies the neuro state the reply carries, as a side effect of reading it.
        """
        error_text = p_settings.get("error_response", ERR_GENERAL_ERROR)
        reply = ReplyText(text=error_text)
        response = attempt.response

        if not response or not response.candidates:
            reason = attempt.error or "Unknown Error"
            feedback = getattr(response, 'prompt_feedback', None) if response else None
            if feedback and feedback.block_reason:
                reply.warnings.append(ERR_SAFETY_BLOCK.format(
                    reason=feedback.block_reason.name.replace('_', ' ').title()))
            elif "Rate Limit" in reason:
                reply.warnings.append(reason)
            else:
                template = WARN_BOTH_MODELS_FAILED if attempt.has_fallback else WARN_MAIN_MODEL_FAILED
                reply.warnings.append(template.format(reason=reason))
            reply.blocked = True
        else:
            try:
                candidates = getattr(getattr(response, 'raw', None), 'candidates', None) or []
                candidate = candidates[0] if candidates else None
                # The filtered text attribute, which leaves the thoughts out.
                raw_text = getattr(response, 'text', "")
                if hasattr(candidate, 'grounding_metadata'):
                    raw_text = _add_inline_citations(raw_text, candidate.grounding_metadata)
                raw_text, reply.neuro_state = self._extract_and_apply_neuro_state(
                    raw_text.strip(), owner_id, profile_name, response=response)
                text = _scrub_response_text(raw_text, participant_names=participant_names)
                if text:
                    # Both halves: what the answering response cited natively, and what
                    # the character's own `search_web` calls brought back. Deduplicated
                    # on the URI, because a Google slot in tool mode can cite the same
                    # page through both routes in one turn.
                    reply.text = text
                    reply.sources = _merge_sources(response_sources(candidate),
                                                   attempt.search_sources)
                else:
                    template = WARN_BOTH_MODELS_FAILED if attempt.fallback_used else WARN_MAIN_MODEL_FAILED
                    reply.warnings.append(template.format(reason=ERR_REASON_EMPTY_RESPONSE))
                    reply.blocked = True
            except ValueError:
                reason = response.candidates[0].finish_reason.name.replace('_', ' ').title()
                reply.text, reply.sources = error_text, None
                reply.warnings.append(ERR_SAFETY_BLOCK.format(reason=reason))
                reply.blocked = True

        if attempt.fallback_used and p_settings.get("show_fallback_indicator", True):
            reply.warnings.append(WARN_FALLBACK_USED)
            reply.warnings.append(WARN_MAIN_MODEL_FAILED.format(reason=attempt.main_error))
        # Always shown, unlike the fallback indicator: this one explains a gap the reader
        # can see without being told -- a character who said nothing about the picture
        # they were just sent. Described wins over dropped, because it is the truer of
        # the two and saying both in one breath reads as a contradiction.
        if attempt.media_described_by:
            reply.warnings.append(WARN_MEDIA_DESCRIBED.format(
                model=clean_model_name(attempt.media_described_by)))
        elif attempt.media_dropped:
            reply.warnings.append(WARN_MEDIA_UNREADABLE.format(kinds=attempt.media_dropped))
        return reply
