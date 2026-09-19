"""One reply: the primary model, then the fallback, and what the turn records about it.

The round worker and regeneration both run this -- build the model, generate, fall back,
turn the result into text and warnings, and write the trace. Each used to carry its own
copy, and the copies drifted: regeneration built its thinking parameters by hand, so a
profile's fallback thinking level never reached a regenerated reply, and it routed its
fallback with a narrower provider guess than the worker's. Both recorded the primary as
the model that answered when the fallback had.
"""
import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...utils import mem_probe
from ...utils.constants import (
    ERR_GENERAL_ERROR, ERR_REASON_EMPTY_RESPONSE, ERR_REASON_TIMEOUT_BOTH, ERR_SAFETY_BLOCK,
    WARN_BOTH_MODELS_FAILED, WARN_FALLBACK_USED, WARN_MAIN_MODEL_FAILED,
)
from ...utils.helpers import (
    _add_inline_citations, _format_api_error, _resolve_safety_settings, _scrub_response_text,
    is_real_model, record_billed_usage, resolve_native_tools, resolve_thinking_params,
)
from ._shared import _strip_neuro_update_and_scrub

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
        return name.replace("models/", "").replace("OPENROUTER/", "").replace("GOOGLE/", "")


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
    if ltm_recall_text:
        lines = [l.strip() for l in ltm_recall_text.split('\n') if l.strip() and not l.startswith("<")]
        meta["ltms_recalled"] = [l[:100] + "..." if len(l) > 100 else l for l in lines]
    if neuro_state:
        meta["neuro_state"] = neuro_state
    if critic:
        meta["critic"] = critic
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

        def build(name: str, role: str, **key_errors):
            thinking = resolve_thinking_params(p_settings, "response", role)
            # Carried across because this dict has always held it. Nothing in any adapter
            # reads it.
            thinking["thinking_persistence"] = p_settings.get("thinking_persistence", 10)
            return self.cog.api_service._instantiate_model(
                name, channel.guild.id, user_id, system_instruction, safety_settings,
                thinking, tools, p_settings, config_owner_id=owner_id, **key_errors)

        async def generate(model, is_fallback: bool):
            attempt.response, _ = await self._generate_with_heartbeat(
                model, history, gen_config, channel, participant, msg_a_id, is_fallback=is_fallback,
                app_name=app_name, app_avatar=app_avatar, existing_state=state_container)
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
            if not attempt.has_fallback:
                attempt.error = attempt.main_error
            else:
                try:
                    await generate(build(fallback_model_name, "fallback"), True)
                    attempt.fallback_used = True
                    if log_context:
                        self.cog._log_api_call(user_id=user_id, guild_id=channel.guild.id,
                                               context=f"{log_context}_fallback",
                                               model_used=fallback_model_name, status="success")
                except Exception as retry_e:
                    attempt.error = (ERR_REASON_TIMEOUT_BOTH if timed_out and isinstance(retry_e, TimeoutError)
                                     else _format_api_error(retry_e))
        finally:
            if log_context:
                # `or primary_model`: a row naming the model that could not be built says
                # more than one naming none.
                self.cog._log_api_call(user_id=user_id, guild_id=channel.guild.id, context=log_context,
                                       model_used=attempt.primary or primary_model, status=attempt.status)
        return attempt

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
                    raw_text.strip(), owner_id, profile_name)
                text = _scrub_response_text(raw_text, participant_names=participant_names)
                if text:
                    reply.text, reply.sources = text, response_sources(candidate)
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
        return reply
