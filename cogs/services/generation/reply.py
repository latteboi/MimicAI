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
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...utils import mem_probe
from ...utils.constants import (
    ERR_GENERAL_ERROR, ERR_REASON_EMPTY_RESPONSE, ERR_REASON_EMPTY_STOPPED, ERR_REASON_STALLED,
    ERR_REASON_TIMEOUT_BOTH, ERR_SAFETY_BLOCK, MEDIA_DESCRIBED_NOTE, MEDIA_UNREADABLE_NOTE,
    WARN_ALL_MODELS_FAILED, WARN_BOTH_MODELS_FAILED, WARN_FALLBACK_MODEL_FAILED,
    WARN_FALLBACK_USED, WARN_MAIN_MODEL_FAILED, WARN_MEDIA_DESCRIBED, WARN_MEDIA_UNREADABLE,
    WARN_FINAL_FALLBACK_USED, WARN_FINAL_MODEL_FAILED,
)
from ...utils.helpers import (
    _add_inline_citations, _format_api_error, _resolve_safety_settings, _scrub_response_text,
    clean_model_name, is_real_model, media_kinds, record_billed_usage,
    resolve_native_tools,
    resolve_thinking_params, resolve_unreadable_media_mode, split_media_parts,
    unreadable_media_modality,
)
from ..api_service import MissingKeyError
from ._shared import _strip_neuro_update_and_scrub
from . import latency, tool_loop
from .tool_loop import FunctionContext

#: Sampling keys a profile may set beyond temperature, top_p and top_k. Sent only when set.
_ADVANCED_SAMPLING_KEYS = ("frequency_penalty", "presence_penalty", "repetition_penalty", "min_p", "top_a")

#: The tags a whispered reply comes back wrapped in, echoing its history. Stripped before
#: the scrub, which drops these blocks contents and all -- the whole reply, in a whisper.
_PRIVATE_WRAPPERS = re.compile(r'</?(?:private_response|whisper_context|private_context)>', re.IGNORECASE)


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
    #: A whisper's reply, whose wrapper tags are the character's own words -- `_PRIVATE_WRAPPERS`.
    private: bool = False
    #: Each standby that was tried and failed, as (rank, reason): rank 0 is the Fallback,
    #: 1 the Final Fallback. One skipped for want of a key after a real failure is left
    #: out, as `run_with_fallback` leaves it out of the error it raises.
    fallback_errors: List[Tuple[int, str]] = field(default_factory=list)
    #: Which standby answered, when one did.
    fallback_rank: int = 0

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
class _Run:
    """One model's go at a reply.

    Kept apart from the ReplyAttempt because a stalled primary and the fallback started
    beside it are in flight at once, and each would otherwise write its responses over
    the other's. The run that is reported is copied across when the race is settled.
    """
    is_fallback: bool
    model: Any = None
    response: Any = None
    looped: Any = None
    error: Optional[BaseException] = None
    task: Optional[asyncio.Future] = None
    #: The fallback run's: see ReplyAttempt.fallback_errors and fallback_rank.
    errors: List[Tuple[int, str]] = field(default_factory=list)
    rank: int = 0


async def _first_answer(runs: List[_Run], failed: List[_Run]) -> Optional[_Run]:
    """The first of `runs` to answer, or None once all of them have failed.

    Each run that fails meanwhile is appended to `failed`, in the order it failed. Runs
    still going when one answers are left for the caller to cancel.
    """
    pending = {run.task: run for run in runs}
    while pending:
        done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        # The primary first when both land in the same tick: it is the character's own
        # voice, and the fallback only stands in for it.
        for task in sorted(done, key=lambda t: pending[t].is_fallback):
            run = pending.pop(task)
            if task.result():
                return run
            failed.append(run)
    return None


def _reason(error: Optional[BaseException]) -> str:
    """A failure as the warning line says it."""
    if error is None:
        return "Unknown Error"
    return getattr(error, 'formatted_reason', None) or _format_api_error(error)


def _empty_response_error(response) -> ValueError:
    """The failure for a response with nothing to post, saying why the model stopped.

    Rare, and otherwise undiagnosable from the channel: a reasoning model whose thinking
    spent the whole output cap stops on a length limit with no text, and reads exactly
    like a model that chose to say nothing.
    """
    candidate = (getattr(response, 'candidates', None) or [None])[0]
    finish = str(getattr(candidate, 'finish_reason', None) or "").upper()
    error = ValueError("Empty Response (AI produced no text content)")
    error.formatted_reason = (
        ERR_REASON_EMPTY_STOPPED.format(finish=finish.replace('_', ' ').title())
        if finish and finish not in ("STOP", "FINISH_REASON_UNSPECIFIED") else ERR_REASON_EMPTY_RESPONSE)
    return error


#: What failed, by standby rank: the Fallback, then the Final Fallback.
_STANDBY_FAILED = (WARN_FALLBACK_MODEL_FAILED, WARN_FINAL_MODEL_FAILED)


def _failure_warnings(attempt: "ReplyAttempt", fallback_reason: Optional[str] = None) -> List[str]:
    """The lines under a reply that has no text: each model that was tried, and why.

    One line when there was no fallback, or when every model failed alike -- Main &
    Fallback, or Main, Fallback & Final Fallback. One per model when they failed
    differently. It used to be one line always, and it named the primary alone whenever
    the fallback's failure left the primary's response behind -- so a fallback that had
    been tried and had failed was never mentioned at all.

    `fallback_reason` is for a standby that answered with nothing to post.
    """
    def line(template: str, reason: str) -> str:
        # A rate limit arrives as a whole bolded sentence of its own.
        return reason if "Rate Limit" in reason else template.format(reason=reason)

    main = attempt.main_error or attempt.error or "Unknown Error"
    standby = list(attempt.fallback_errors)
    if fallback_reason is not None:
        standby.append((attempt.fallback_rank, fallback_reason))
    if not standby:
        if not attempt.has_fallback:
            return [line(WARN_MAIN_MODEL_FAILED, attempt.error or main)]
        standby = [(0, attempt.error or main)]
    if attempt.error == ERR_REASON_TIMEOUT_BOTH or all(r == main for _, r in standby):
        reason = attempt.error if attempt.error == ERR_REASON_TIMEOUT_BOTH else main
        final = any(rank for rank, _ in standby)
        return [line(WARN_ALL_MODELS_FAILED if final else WARN_BOTH_MODELS_FAILED, reason)]
    return [line(WARN_MAIN_MODEL_FAILED, main)] + [
        line(_STANDBY_FAILED[min(rank, 1)], reason) for rank, reason in standby]


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
                             probe_label: Optional[str] = None,
                             functions: Sequence[Any] = (), private: bool = False,
                             conversation: bool = False,
                             function_context: Optional[FunctionContext] = None) -> ReplyAttempt:
        """Generates with the primary, then the fallback if the primary fails.

        A primary that has not answered in `latency.race_after` seconds has the fallback
        started beside it rather than after it. Whichever answers first is the reply and
        the other is cancelled; one failing leaves the other to finish.

        `state_container` is the caller's open container and is updated in place. A
        cancellation propagates: the caller owns the teardown, which differs -- the worker
        deletes its placeholder, regeneration puts the original text back.

        `functions` is the tuple the caller built `system_instruction` with; both models
        declare it, and `tool_loop.run` answers it.

        The fallback run walks the profile's Fallback, then the Final Fallback on the
        provider the primary is not on -- `model_chain`, which every utility pass uses too.
        `attempt.fallback_name` ends up naming whichever of them answered.

        Session replies, their regeneration, whispers (`private`) and Global Chat all run
        this. `conversation` is Global Chat's: every model on `user_id`'s own key, policed
        as a conversation in the server the card is open in (`_instantiate_model`), under
        the safety settings of no channel, since the card can be opened in any. It brings
        its own `function_context`, which bills a search to that same key.
        """
        _, chain = self.cog.api_service.model_chain(
            {**p_settings, "primary_model": primary_model, "fallback_model": fallback_model_name},
            "primary_model", owner_id)
        fallback_names = [name for name in dict.fromkeys(chain) if name != primary_model]
        attempt = ReplyAttempt(primary_name=primary_model,
                               fallback_name=fallback_names[0] if fallback_names else None,
                               private=private)
        server_id = getattr(getattr(channel, 'guild', None), 'id', None)
        guild_id = None if conversation else server_id
        policy = {"policy_guild_id": server_id, "conversation": True} if conversation else {}
        safety_settings = _resolve_safety_settings(None if conversation else channel, p_settings)
        tools = resolve_native_tools(p_settings)

        def build(name: str, role: str, **key_errors):
            thinking = resolve_thinking_params(p_settings, "response", role)
            # Carried across because this dict has always held it. Nothing in any adapter
            # reads it.
            thinking["thinking_persistence"] = p_settings.get("thinking_persistence", 10)
            return self.cog.api_service._instantiate_model(
                name, guild_id, user_id, system_instruction, safety_settings,
                thinking, tools, p_settings, config_owner_id=owner_id,
                functions=functions, **policy, **key_errors)

        fn_ctx = function_context or FunctionContext(
            owner_id=owner_id,
            profile_name=(participant or {}).get('profile_name', ''),
            author_dn=app_name,
            guild_id=server_id,
            triggering_user_id=user_id,
            # Already resolved for this reply, off the destination channel. `search_web`
            # asks a second model on behalf of this turn and must be held to the same
            # rules the turn is, not to Google's unset defaults.
            safety_settings=safety_settings)

        async def generate(run: _Run, contents: Optional[List] = None,
                           slow_after: Optional[float] = None, on_slow=None):
            # `history` is the caller's and is reused by the fallback attempt, which must
            # start from the same conversation rather than inheriting the primary's
            # abandoned lookups -- `tool_loop.run` copies it rather than extending it.
            async def send(turn, cfg):
                run.response, _ = await self._generate_with_heartbeat(
                    run.model, turn, cfg, channel,
                    participant, msg_a_id, is_fallback=run.is_fallback,
                    app_name=app_name, app_avatar=app_avatar, existing_state=state_container,
                    slow_after=slow_after, on_slow=on_slow)
                if not run.response or not run.response.candidates:
                    raise ValueError("Response blocked or empty")
                return run.response

            run.looped = await tool_loop.run(self.cog, run.model, history if contents is None else contents,
                                              gen_config, fn_ctx, send)
            text = getattr(run.response, 'text', "").strip()
            if private:
                text = _PRIVATE_WRAPPERS.sub('', text)
            if not _strip_neuro_update_and_scrub(text, participant_names):
                raise _empty_response_error(run.response)

        def settle(run: _Run, answered: bool) -> None:
            """Copies one run onto the attempt: the reply's, or a failure's explanation."""
            # The last response received explains a failure -- a block reason rides on
            # it -- so a run that received none leaves the one already there.
            if run.response is not None:
                attempt.response = run.response
            if run.looped is not None:
                # Assigned, not extended: the fallback ran its own searches from the same
                # history, and the trace describes the reply that was actually posted.
                run.looped.record(attempt.extra_meta)
                attempt.recalled_memories = run.looped.memories
                attempt.search_sources = run.looped.sources
            if answered:
                attempt.answered_by = run.model

        async def generate_once(model, is_fallback: bool, contents: Optional[List] = None):
            """One run, settled either way. What the attachment retry goes through."""
            run = _Run(is_fallback, model)
            try:
                await generate(run, contents)
            except BaseException:
                settle(run, False)
                raise
            settle(run, True)

        async def go(run: _Run, make=None, **slow) -> bool:
            """True when the run answered; its error is kept on it when it did not."""
            try:
                if make is not None:
                    run.model = make()
                await generate(run, **slow)
                return True
            except asyncio.CancelledError:
                raise
            except Exception as e:
                run.error = e
                return False

        async def go_chain(run: _Run) -> bool:
            """The fallback run: each fallback name in turn, until one answers."""
            for n, name in enumerate(fallback_names):
                earlier = run.error
                if await go(run, make=lambda: build(name, "fallback")):
                    attempt.fallback_name = name
                    run.rank = n
                    return True
                # A provider nobody here holds a key for says nothing about the turn; the
                # failure before it does.
                if earlier is not None and isinstance(run.error, MissingKeyError):
                    run.error = earlier
                    continue
                run.errors.append((n, _reason(run.error)))
                if n + 1 < len(fallback_names):
                    print(f"Reply fallback '{name}' failed ({_reason(run.error)}); "
                          f"trying '{fallback_names[n + 1]}'.")
            return False

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

        primary, fallback = _Run(False, attempt.primary), None
        failed: List[_Run] = []
        raced_after = None
        try:
            probe = mem_probe.probe(probe_label, peak=False) if probe_label else contextlib.nullcontext()
            with probe:
                winner = None
                if attempt.primary is None:
                    # A primary that could not be constructed is an error like any other,
                    # so a configured fallback -- quite possibly a provider whose key is
                    # present -- still gets its chance. The text rides on the exception
                    # already phrased for the user; _format_api_error would cut it to 80
                    # characters.
                    primary.error = RuntimeError(init_error or "Internal API Initialization Error")
                    primary.error.formatted_reason = init_error or "Internal API Initialization Error"
                    failed.append(primary)
                else:
                    # A primary that sits on the request past its usual time has the
                    # fallback started beside it, rather than waiting out the heartbeat's
                    # four minutes first -- see latency. Only when there is a fallback.
                    slow_after = (latency.race_after(getattr(attempt.primary, 'model_name', None) or primary_model)
                                  if attempt.has_fallback else None)
                    stalled = asyncio.Event()
                    primary.task = asyncio.ensure_future(
                        go(primary, slow_after=slow_after, on_slow=stalled.set if slow_after else None))
                    stall = asyncio.ensure_future(stalled.wait())
                    try:
                        await asyncio.wait((primary.task, stall), return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        stall.cancel()
                    if not primary.task.done():
                        raced_after = slow_after
                        if state_container is not None:
                            state_container['fallback_racing'] = True
                    elif primary.task.result():
                        winner = primary
                    else:
                        failed.append(primary)

                if winner is None and attempt.has_fallback:
                    fallback = _Run(True)
                    fallback.task = asyncio.ensure_future(go_chain(fallback))

                if winner is None:
                    winner = await _first_answer([r for r in (primary, fallback)
                                                  if r is not None and r.task is not None
                                                  and not r.task.done()], failed)

            if raced_after is not None:
                who = "the primary" if winner is primary else "the fallback" if winner is fallback else "neither"
                print(f"Fallback race: '{primary_model}' gave no reply in {raced_after:.0f}s, "
                      f"so '{fallback_names[0]}' was started beside it; {who} answered first.")

            if winner is primary:
                attempt.status = "success"
                settle(primary, True)
            elif winner is not None:
                attempt.fallback_used = True
                # Still running when the fallback answered: it had not failed, it had
                # been too slow, and that is what the indicator says.
                attempt.main_error = (ERR_REASON_STALLED.format(seconds=f"{raced_after:.0f}")
                                      if primary.error is None else _reason(primary.error))
                attempt.fallback_rank, attempt.fallback_errors = winner.rank, list(winner.errors)
                settle(winner, True)
                if log_context:
                    self.cog._log_api_call(user_id=user_id, guild_id=guild_id,
                                           context=f"{log_context}_fallback",
                                           model_used=attempt.fallback_name, status="success")
            else:
                for run in failed:
                    settle(run, False)
                attempt.main_error = _reason(primary.error)
                if fallback is not None:
                    attempt.fallback_errors = list(fallback.errors)
                if fallback is None:
                    attempt.error = attempt.main_error
                elif isinstance(primary.error, TimeoutError) and isinstance(fallback.error, TimeoutError):
                    attempt.error = ERR_REASON_TIMEOUT_BOTH
                else:
                    attempt.error = _reason(fallback.error)

                # Which model refused to read the attachment, if that is what happened.
                # The primary is preferred: it is the character's own voice, and the
                # fallback is only standing in because of a file neither of them was
                # asked about. Both have refused it by now -- if the fallback had read
                # it, its answer would be the reply and this would not run. Only now is
                # the profile's unreadable-media setting reached: a profile whose
                # fallback has vision never sees it.
                blind_model = None
                if attempt.primary is not None and unreadable_media_modality(primary.error):
                    blind_model = attempt.primary
                elif fallback is not None and fallback.model is not None and unreadable_media_modality(fallback.error):
                    blind_model = fallback.model
                if blind_model is not None:
                    await self._retry_without_media(
                        attempt, generate_once, blind_model, history, p_settings=p_settings,
                        channel=channel, owner_id=owner_id, user_id=user_id,
                        is_fallback=blind_model is not attempt.primary)
        finally:
            # A run still going is the loser of a race, or this turn is being cancelled.
            losers = [r.task for r in (primary, fallback)
                      if r is not None and r.task is not None and not r.task.done()]
            for task in losers:
                task.cancel()
            if losers:
                await asyncio.gather(*losers, return_exceptions=True)
            if state_container is not None:
                state_container.pop('fallback_racing', None)
            if log_context:
                # `or primary_model`: a row naming the model that could not be built says
                # more than one naming none.
                self.cog._log_api_call(user_id=user_id, guild_id=guild_id, context=log_context,
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
        # The refusals were of the attachment, which is gone; the reply stands as the
        # indicator's alone, as it did before the chain had a third model.
        attempt.fallback_errors, attempt.fallback_rank = [], 0
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
        #: Set once the failure lines have named the fallback, so the indicator below
        #: does not say it a second time.
        fallback_named = False

        # `attempt.error` first: a failed attempt still carries the last response it
        # received, and one with candidates but no text passed the test below as a reply
        # -- which is how a fallback that had been tried and had failed was reported as
        # the primary failing alone.
        if attempt.error is not None or not response or not response.candidates:
            feedback = getattr(response, 'prompt_feedback', None) if response else None
            if feedback and feedback.block_reason:
                reply.warnings.append(ERR_SAFETY_BLOCK.format(
                    reason=feedback.block_reason.name.replace('_', ' ').title()))
            else:
                reply.warnings.extend(_failure_warnings(attempt))
            reply.blocked = True
        else:
            try:
                candidates = getattr(getattr(response, 'raw', None), 'candidates', None) or []
                candidate = candidates[0] if candidates else None
                # The filtered text attribute, which leaves the thoughts out.
                raw_text = getattr(response, 'text', "")
                if attempt.private:
                    raw_text = _PRIVATE_WRAPPERS.sub('', raw_text)
                if hasattr(candidate, 'grounding_metadata'):
                    raw_text = _add_inline_citations(raw_text, candidate.grounding_metadata)
                raw_text, reply.neuro_state = self._extract_and_apply_neuro_state(
                    raw_text.strip(), owner_id, profile_name)
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
                    if attempt.fallback_used:
                        reply.warnings.extend(_failure_warnings(attempt, ERR_REASON_EMPTY_RESPONSE))
                        fallback_named = True
                    else:
                        reply.warnings.append(WARN_MAIN_MODEL_FAILED.format(reason=ERR_REASON_EMPTY_RESPONSE))
                    reply.blocked = True
            except ValueError:
                reason = response.candidates[0].finish_reason.name.replace('_', ' ').title()
                reply.text, reply.sources = error_text, None
                reply.warnings.append(ERR_SAFETY_BLOCK.format(reason=reason))
                reply.blocked = True

        if attempt.fallback_used and not fallback_named and p_settings.get("show_fallback_indicator", True):
            reply.warnings.append(WARN_FINAL_FALLBACK_USED if attempt.fallback_rank else WARN_FALLBACK_USED)
            reply.warnings.append(WARN_MAIN_MODEL_FAILED.format(reason=attempt.main_error))
            # The Final answering means the Fallback failed first; that says why.
            reply.warnings.extend(_STANDBY_FAILED[min(rank, 1)].format(reason=reason)
                                  for rank, reason in attempt.fallback_errors)
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
