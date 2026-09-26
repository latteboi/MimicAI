import uuid
import datetime
from typing import Any, Dict, List, Optional, Tuple

from ...utils.constants import (
    COMPACTION_CHUNK_DEFAULT, COMPACTION_CHUNK_MIN, COMPACTION_MAX_CHUNK_RATIO,
    COMPACTION_SYNOPSIS_WORDS_DEFAULT, COMPACTION_SYNOPSIS_WORDS_MAX,
    COMPACTION_SYNOPSIS_WORDS_MIN, COMPACTION_THRESHOLD_DEFAULT, COMPACTION_THRESHOLD_MAX,
    COMPACTION_THRESHOLD_MIN, DEFAULT_SESSION_SYNOPSIS_PROMPT,
    DEFAULT_SESSION_SYNOPSIS_USER_PROMPT, GREEDY_SAMPLING, SESSION_BUSY_FLAGS,
    COMPACT_KEEP_DEFAULT, COMPACT_PASS_MAX_CHARS, PURGE_BUSY_WAIT_TIMEOUT_SECONDS,
)
from ...utils.helpers import _resolve_zoneinfo, restamp_turn, resolve_thinking_params, turn_posted_at
from ...managers.session_manager import SessionManager, intern_turn, reply_tags, turn_lookup, with_reply

#: What the settings modal wrote into every session it saved, typed or not: the shipped
#: pair of the day, seeded rather than resolved. Read as unset, so those sessions follow
#: the shipped chain like the rest instead of staying on nova-micro for good.
_SEEDED_MODELS = {"model": "OPENROUTER/amazon/nova-micro-v1",
                  "fallback_model": "GOOGLE/gemini-2.5-flash-lite"}


def compaction_model_config(session: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """The session's own summariser choice as LTM slots, None where it made none.

    Compaction is the LTM summariser's job on a whole scene, so it runs that category's
    chain -- `model_chain(this, "ltm_model", ...)` -- under the session owner's preference.
    With its Final Fallback always: it is billed to the server's key, which may hold
    either provider, and there is no profile to hang the switch on.
    """
    raw = session.get("compaction") or {}
    chosen = {key: (str(raw.get(key) or "").strip() or None) for key in _SEEDED_MODELS}
    return {**{slot: None if chosen[key] == _SEEDED_MODELS[key] else chosen[key]
               for key, slot in (("model", "ltm_model"), ("fallback_model", "ltm_fallback_model"))},
            "final_fallback_enabled": True}


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

    words = raw.get("max_words", COMPACTION_SYNOPSIS_WORDS_DEFAULT)
    try:
        words = int(words)
    except (TypeError, ValueError):
        words = COMPACTION_SYNOPSIS_WORDS_DEFAULT
    words = max(COMPACTION_SYNOPSIS_WORDS_MIN, min(COMPACTION_SYNOPSIS_WORDS_MAX, words))

    return {
        "enabled": SessionManager.compaction_enabled(session),
        "threshold": threshold,
        "chunk": chunk,
        "max_words": words,
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
        # Backing off after a failure. On a server where neither summariser can run,
        # every round end used to spend both attempts again, for the life of the session.
        if len(candidates) < session.get("_compaction_retry_at", 0):
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

        folded = await self._fold(channel_id, session, indices, settings)
        if folded is None:
            # Tried again once another fold's worth of public turns has built up. Held in
            # memory only: a restart is a fair moment to try again.
            session["_compaction_retry_at"] = (
                len(self._compactable_indices(session.get("unified_log") or [])) + settings["chunk"])
            return False
        if not folded:
            return False
        session.pop("_compaction_retry_at", None)
        await self._flush_folds(channel_id, session)
        return True

    @staticmethod
    def _fold_text(turn: Dict[str, Any], clock, tags, find) -> str:
        """A turn as the summariser reads it: what was said, what it replied to, and the
        page it linked.

        The page travels beside the turn rather than in it (see `keep_url_context`), and
        a folded turn is never shown again -- so a synopsis written from `content` alone
        remembered that a link was posted and nothing of what it said.
        """
        text = with_reply(restamp_turn(turn.get("content") or "", turn_posted_at(turn), clock),
                          turn, clock, tags, find).strip()
        if turn.get("url_context"):
            text += f"\n<document_context>\n{turn['url_context']}\n</document_context>"
        return text

    async def _fold(self, channel_id: int, session: Dict[str, Any], indices: List[int],
                    settings: Dict[str, Any]) -> Optional[int]:
        """Fold the turns at `indices` into one synopsis placed after them.

        Returns how many were folded; 0 when there was nothing to summarise or the log
        was replaced while the summariser ran; None when the summariser failed. Saving
        is the caller's: a manual compact folds several times and writes the log once.
        """
        unified_log = session["unified_log"]
        turns = [unified_log[i] for i in indices]
        # One clock for a synopsis the whole cast shares: the session owner's. Stored
        # turns carry each speaker's own, and a date read across two zones is wrong.
        clock, _ = _resolve_zoneinfo(self.cog.profile_manager.user_timezone(session.get("owner_id")))
        tags, find = reply_tags(turns), turn_lookup(unified_log)
        transcript = "\n".join(text for text in (self._fold_text(t, clock, tags, find) for t in turns) if text)
        if not transcript.strip():
            return 0

        previous = self._latest_synopsis(unified_log, before_index=indices[0])

        synopsis = await self._generate_synopsis(channel_id, session, transcript, previous, settings)
        if not synopsis:
            return None

        # Re-read the log: generating awaited, and a whisper or a delete could have
        # landed on it. Positions are only meaningful against the list we planned from.
        if session.get("unified_log") is not unified_log:
            return 0

        marked_ids = []
        for turn in turns:
            if not turn.get("compacted"):
                turn["compacted"] = True
                marked_ids.append(turn.get("turn_id"))
        marked = len(marked_ids)
        if not marked:
            return 0

        synopsis_turn = intern_turn({
            "turn_id": str(uuid.uuid4()),
            "type": "synopsis",
            "is_user": False,
            "speaker_pid": "SYSTEM",
            "message_ids": [],
            "content": synopsis,
            "covers": marked,
            # Which turns it folded, so deleting one of them can find this synopsis --
            # see turn_deletion. Left off when a folded turn has no id to name, and that
            # synopsis is then found by position like the ones written before this.
            **({"covers_turn_ids": marked_ids} if all(marked_ids) else {}),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
        # After the last folded turn, so the synopsis sits where the range it replaces
        # used to be rather than after conversation that came later.
        unified_log.insert(indices[-1] + 1, synopsis_turn)
        return marked

    async def _flush_folds(self, channel_id: int, session: Dict[str, Any]):
        # Structural: the flags land on turns anywhere in the log, including ones already
        # sealed into the cold segment, and an insert or removal shifts the tail boundary.
        session["_log_cold_len"] = 0
        await self.cog.session_manager.flush_session(
            (channel_id, None, None), session.get("type", "multi"), structural=True)

    async def manual_compaction(self, channel_id: int, *, undo: bool = False,
                                keep: int = COMPACT_KEEP_DEFAULT) -> str:
        """`/compact` and the Compaction tab's buttons: fold now, or unfold everything.

        Returns what to tell whoever asked. Claims the channel for the whole run, as
        /memorise does: a round captures `batch_start_index` when it starts, and a
        synopsis inserted under it would shift every turn that round is about to read.
        """
        session = self.cog.multi_profile_channels.get(channel_id)
        if not session:
            return "There is no session in this channel."
        if not session.get("is_hydrated"):
            session = await self.cog.session_manager._ensure_session_hydrated(
                channel_id, session.get("type", "multi"))
            if not session:
                return "Could not load this session's transcript."
        if not await self.cog.session_manager._wait_for_session_flags(
                session, SESSION_BUSY_FLAGS, PURGE_BUSY_WAIT_TIMEOUT_SECONDS):
            return (f"The session is still busy after {int(PURGE_BUSY_WAIT_TIMEOUT_SECONDS)}s. "
                    "Nothing was changed \u2014 try again in a moment.")
        session["is_compacting"] = True
        try:
            if undo:
                return await self._unfold_all(channel_id, session)
            return await self._fold_all(channel_id, session, keep)
        finally:
            session["is_compacting"] = False

    async def _fold_all(self, channel_id: int, session: Dict[str, Any], keep: int) -> str:
        """Fold every public turn but the newest `keep`, in passes of bounded size."""
        settings = self._compaction_settings(session)
        folded, failed = 0, False
        while True:
            log = session.get("unified_log") or []
            candidates = self._compactable_indices(log)
            foldable = candidates[:max(0, len(candidates) - keep)]
            if not foldable:
                break
            # Oldest first, up to the size cap, and always at least one turn: a single
            # turn over the cap -- a long pasted file -- is still a turn to fold.
            indices, size = [], 0
            for i in foldable:
                turn = log[i]
                size += len(turn.get("content") or "") + len(turn.get("url_context") or "")
                if indices and size > COMPACT_PASS_MAX_CHARS:
                    break
                indices.append(i)
            count = await self._fold(channel_id, session, indices, settings)
            if count is None:
                failed = True
            if not count:
                break
            folded += count

        if not folded:
            if failed:
                return "The summariser could not run, so nothing was folded."
            live = len(self._compactable_indices(session.get("unified_log") or []))
            return f"Nothing to fold: the session has {live} live turn(s), and the newest {keep} are kept."

        session.pop("_compaction_retry_at", None)
        notes = []
        if not settings["enabled"]:
            # Folded turns are hidden, and the synopsis sent, only while it is on. Left
            # off, this would have paid for a synopsis nobody reads and hidden nothing.
            session.setdefault("compaction", {})["enabled"] = True
            self.cog.session_manager._save_multi_profile_sessions()
            notes.append("The rolling synopsis was off and is now on, or the synopsis would never be sent.")
        await self._flush_folds(channel_id, session)

        remaining = len(self._compactable_indices(session["unified_log"]))
        message = f"Folded {folded} turn(s) into the session synopsis; {remaining} live turn(s) remain."
        if failed:
            message += " The summariser failed partway, so some of those were meant to be folded too."
        return " ".join([message, *notes])

    async def _unfold_all(self, channel_id: int, session: Dict[str, Any]) -> str:
        """Unfold every folded turn and drop every synopsis block, back to the transcript."""
        log = session.get("unified_log") or []
        folded = [turn for turn in log if turn.get("compacted")]
        synopses = sum(1 for turn in log if turn.get("type") == "synopsis")
        if not folded and not synopses:
            return "Nothing in this session is folded."
        for turn in folded:
            del turn["compacted"]
        # In place: whatever holds this list keeps seeing the session's log.
        log[:] = [turn for turn in log if turn.get("type") != "synopsis"]
        session.pop("_compaction_retry_at", None)
        await self._flush_folds(channel_id, session)

        message = f"Unfolded {len(folded)} turn(s) and removed {synopses} synopsis block(s)."
        settings = self._compaction_settings(session)
        if settings["enabled"]:
            message += (f" The rolling synopsis is still on, so it folds {settings['chunk']} again at a "
                        f"round's end whenever {settings['threshold']} or more turns are live. Turn it off "
                        "on the Compaction tab of `/session config` to keep them.")
        return message

    async def _generate_synopsis(self, channel_id: int, session: Dict[str, Any], transcript: str,
                                 previous: Optional[str], settings: Dict[str, Any]) -> Optional[str]:
        channel = self.cog.bot.get_channel(channel_id)
        guild_id = channel.guild.id if channel and getattr(channel, 'guild', None) else 0
        if not guild_id:
            return None

        system_instruction = self.cog.global_prompts.get(
            "SESSION_SYNOPSIS", DEFAULT_SESSION_SYNOPSIS_PROMPT
        ).format(max_words=settings["max_words"])

        previous_block = f"Synopsis of everything before this excerpt:\n{previous}\n\n" if previous else ""
        user_prompt = self.cog.global_prompts.get(
            "SESSION_SYNOPSIS_USER", DEFAULT_SESSION_SYNOPSIS_USER_PROMPT
        ).format(previous_synopsis=previous_block, transcript=transcript)

        async def _attempt(model_name, is_fallback):
            # `{}` here was the same defect the LTM summariser had: not "no thinking"
            # but "whatever the adapter defaults to", which is high. Compaction is a
            # session setting rather than a profile one, so it has no configurable slot
            # and takes the shared `utility` default.
            model = self.cog.api_service._instantiate_model(
                model_name, guild_id, session.get("owner_id"),
                system_instruction=system_instruction,
                config_owner_id=session.get("owner_id"),
                thinking_params=resolve_thinking_params(None, "utility"))
            return await model.generate_content_async(
                [user_prompt], generation_config=dict(GREEDY_SAMPLING))

        try:
            resp, _used, _was_fallback = await self.cog.api_service.run_with_fallback(
                *self.cog.api_service.model_chain(
                    compaction_model_config(session), "ltm_model", session.get("owner_id")),
                _attempt, label="Session synopsis")
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
