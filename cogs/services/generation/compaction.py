import uuid
import datetime
from typing import Any, Dict, List, Optional, Tuple

from ...utils.constants import (
    COMPACTION_CHUNK_DEFAULT, COMPACTION_CHUNK_MIN, COMPACTION_FALLBACK_MODEL_DEFAULT,
    COMPACTION_MAX_CHUNK_RATIO, COMPACTION_MODEL_DEFAULT, COMPACTION_SYNOPSIS_MAX_WORDS,
    COMPACTION_THRESHOLD_DEFAULT, COMPACTION_THRESHOLD_MAX, COMPACTION_THRESHOLD_MIN,
    DEFAULT_SESSION_SYNOPSIS_PROMPT, DEFAULT_SESSION_SYNOPSIS_USER_PROMPT,
    SESSION_BUSY_FLAGS,
)
from ...managers.session_manager import intern_turn


def resolve_compaction_settings(session: Dict[str, Any]) -> Dict[str, Any]:
    """The session's compaction config, clamped to something that can actually run.

    Session config is user-entered and persisted, so a value that was valid when it was
    written may not be now -- and a chunk that swallows its own threshold would compact
    the whole transcript on the first pass and then have nothing left to summarise
    against. Clamping here rather than at the modal means old sessions and hand-edited
    blueprints get the same guarantees.
    """
    raw = session.get("compaction") or {}

    threshold = raw.get("threshold", COMPACTION_THRESHOLD_DEFAULT)
    try:
        threshold = int(threshold)
    except (TypeError, ValueError):
        threshold = COMPACTION_THRESHOLD_DEFAULT
    threshold = max(COMPACTION_THRESHOLD_MIN, min(COMPACTION_THRESHOLD_MAX, threshold))

    chunk = raw.get("chunk", COMPACTION_CHUNK_DEFAULT)
    try:
        chunk = int(chunk)
    except (TypeError, ValueError):
        chunk = COMPACTION_CHUNK_DEFAULT
    # Never fold away the whole window: something has to survive for the next round to
    # read, and for the next synopsis to be written against.
    chunk = max(COMPACTION_CHUNK_MIN, min(chunk, int(threshold * COMPACTION_MAX_CHUNK_RATIO)))

    model = (raw.get("model") or COMPACTION_MODEL_DEFAULT).strip() or COMPACTION_MODEL_DEFAULT
    fallback = (raw.get("fallback_model") or COMPACTION_FALLBACK_MODEL_DEFAULT).strip()

    return {
        "enabled": bool(raw.get("enabled", False)),
        "threshold": threshold,
        "chunk": chunk,
        "model": model,
        "fallback_model": fallback or COMPACTION_FALLBACK_MODEL_DEFAULT,
    }


class SessionCompactionMixin:
    """Rolling synopsis: folds the oldest public turns of a long session into one
    `<session_synopsis>` block so the scene stays coherent past the STM window.

    Three properties this deliberately holds to:

    *Only public turns are eligible.* Whispers and private responses are per-participant
    by construction -- `_build_history_for_participant` shows each profile only its own.
    One shared synopsis cannot represent them without telling every profile what another
    was told privately, so they are skipped as sources and left visible as turns.

    *Compaction hides, it does not delete.* Folded turns stay in `unified_log` under a
    `compacted` flag. The audit view, `/regenerate`, message-delete cleanup and LTM all
    keep working on the full transcript, and turning compaction off restores it.

    *`compacted` is not `is_hidden`.* Mute (`is_hidden`) retracts a turn, which is why
    the pending-whisper derivation skips hidden turns -- a retracted answer should make
    its whisper pending again. A compacted turn still happened, so it must keep stopping
    that backwards walk, or every session would re-inject whispers it answered hours ago.
    """

    def _compaction_settings(self, session: Dict[str, Any]) -> Dict[str, Any]:
        return resolve_compaction_settings(session)

    @staticmethod
    def _compactable_indices(unified_log: List[Dict]) -> List[int]:
        """Positions of the public turns a synopsis may be built from, oldest first."""
        return [
            i for i, turn in enumerate(unified_log)
            if not turn.get("type")
            and not turn.get("compacted")
            and not turn.get("is_hidden")
        ]

    @staticmethod
    def _latest_synopsis(unified_log: List[Dict], before_index: Optional[int] = None) -> Optional[str]:
        limit = len(unified_log) if before_index is None else before_index
        for i in range(limit - 1, -1, -1):
            if unified_log[i].get("type") == "synopsis":
                return (unified_log[i].get("content") or "").strip() or None
        return None

    def _plan_compaction(self, session: Dict[str, Any]) -> Optional[Tuple[List[int], Dict[str, Any]]]:
        """The indices to fold this pass, or None if the session is not due.

        Returns positions rather than turn objects because the synopsis has to be
        inserted at the end of the folded range to keep the transcript chronological.
        """
        settings = self._compaction_settings(session)
        if not settings["enabled"]:
            return None

        unified_log = session.get("unified_log")
        if not unified_log:
            return None

        candidates = self._compactable_indices(unified_log)
        if len(candidates) < settings["threshold"]:
            return None

        return candidates[:settings["chunk"]], settings

    async def _run_session_compaction(self, channel_id: int) -> bool:
        """Fold this session's oldest public turns into a synopsis. True if it ran.

        Called at round end, once the round's own flush has landed. Never during a
        round: inserting a turn shifts every index after it, and `batch_start_index`
        is captured at round start and read when each participant's prompt is built.
        """
        session = self.cog.multi_profile_channels.get(channel_id)
        if not session or not session.get("is_hydrated"):
            return False

        # is_running is set by the worker that called us, so it is excluded here; the
        # rest are the claims that mean another operation is mid-flight on this channel.
        if any(session.get(flag) for flag in SESSION_BUSY_FLAGS if flag != 'is_running'):
            return False

        plan = self._plan_compaction(session)
        if plan is None:
            return False
        indices, settings = plan

        unified_log = session["unified_log"]
        turns = [unified_log[i] for i in indices]
        transcript = "\n".join((t.get("content") or "").strip() for t in turns if t.get("content"))
        if not transcript.strip():
            return False

        previous = self._latest_synopsis(unified_log, before_index=indices[0])

        synopsis = await self._generate_synopsis(channel_id, session, transcript, previous, settings)
        if not synopsis:
            return False

        # Re-read the log: generating awaited, and a whisper or a delete could have
        # landed on it. Positions are only meaningful against the list we planned from.
        if session.get("unified_log") is not unified_log:
            return False

        marked = 0
        for turn in turns:
            if not turn.get("compacted"):
                turn["compacted"] = True
                marked += 1
        if not marked:
            return False

        synopsis_turn = intern_turn({
            "turn_id": str(uuid.uuid4()),
            "type": "synopsis",
            "is_user": False,
            "speaker_pid": "SYSTEM",
            "message_ids": [],
            "content": synopsis,
            "covers": marked,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
        # After the last folded turn, so the synopsis sits where the range it replaces
        # used to be rather than after conversation that came later.
        unified_log.insert(indices[-1] + 1, synopsis_turn)

        # Structural: the flags land on turns anywhere in the log, including ones already
        # sealed into the cold segment, and the insert shifts the tail boundary.
        session["_log_cold_len"] = 0
        await self.cog.session_manager.flush_session(
            (channel_id, None, None), session.get("type", "multi"), structural=True)

        print(f"[Compaction] Channel {channel_id}: folded {marked} turn(s) into a synopsis.")
        return True

    async def _generate_synopsis(self, channel_id: int, session: Dict[str, Any], transcript: str,
                                 previous: Optional[str], settings: Dict[str, Any]) -> Optional[str]:
        channel = self.cog.bot.get_channel(channel_id)
        guild_id = channel.guild.id if channel and getattr(channel, 'guild', None) else 0
        if not guild_id:
            return None

        system_instruction = self.cog.global_prompts.get(
            "SESSION_SYNOPSIS", DEFAULT_SESSION_SYNOPSIS_PROMPT
        ).format(max_words=COMPACTION_SYNOPSIS_MAX_WORDS)

        previous_block = f"Synopsis of everything before this excerpt:\n{previous}\n\n" if previous else ""
        user_prompt = self.cog.global_prompts.get(
            "SESSION_SYNOPSIS_USER", DEFAULT_SESSION_SYNOPSIS_USER_PROMPT
        ).format(previous_synopsis=previous_block, transcript=transcript)

        async def _attempt(model_name, _is_fallback):
            model = self.cog.api_service._instantiate_model(
                model_name, guild_id, session.get("owner_id"),
                system_instruction=system_instruction, thinking_params={})
            return await model.generate_content_async(
                [user_prompt], generation_config={"temperature": 0.2, "top_p": 0.95})

        try:
            resp, _used, _was_fallback = await self.cog.api_service.run_with_fallback(
                settings["model"], settings["fallback_model"], _attempt,
                label="Session synopsis")
        except Exception as e:
            print(f"[Compaction] Synopsis generation failed for channel {channel_id}: {e}")
            return None

        text = getattr(resp, "text", None) if resp else None
        if not text or not text.strip():
            return None

        from ...utils.helpers import _scrub_response_text
        # The model was told not to emit tags, but its output is wrapped in one and
        # replayed as context on every later turn -- a stray tag would compound.
        return _scrub_response_text(text).strip() or None
