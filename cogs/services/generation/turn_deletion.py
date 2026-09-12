"""Deleting a turn deletes all of it.

A turn is often several messages -- a reply split past 2000 characters, citation lines,
warnings, a thinking-summary file, an audio follow-up -- and every one of them is on the
same log entry's `message_ids`. Removing some of them used to drop the whole turn from
memory while the rest stayed in the channel with nothing pointing at them, so 🔁, 🔇 and
deletion could no longer reach them: a /purge window that ended mid-turn did it, and so
did someone deleting a single citation line. The unit here is the turn. Whatever removes
any part of one removes all of it, from the channel and from the log.

A deleted turn can also live on in a rolling synopsis. Each synopsis is written from the
previous one plus the next chunk of turns, and only the latest is injected into prompts,
so a synopsis that folded a deleted turn *and every synopsis after it* still carry it.
Those are dropped, and the turns they covered are un-compacted for compaction to fold
again without it.
"""
import asyncio
import time
from typing import Dict, Iterable, List, Optional, Set

import discord

from ...utils.constants import SESSION_BUSY_FLAGS, TURN_DELETE_DEFER_TIMEOUT_SECONDS

#: Log entries that never appear in the channel, whatever their message_ids hold.
_INVISIBLE_TURN_TYPES = frozenset({"whisper", "private_response", "synopsis"})

#: Discord refuses to bulk-delete a message older than fourteen days. Kept a little
#: inside that, so nothing ages past it between the check and the request.
_BULK_DELETE_MAX_AGE_SECONDS = 14 * 24 * 3600 - 300


def is_visible_turn(turn: Dict) -> bool:
    """A turn the channel can see: it posted messages, and it is not a private exchange."""
    return bool(turn.get("message_ids")) and turn.get("type") not in _INVISIBLE_TURN_TYPES


def _covering_synopsis_index(log: List[Dict], index: int) -> Optional[int]:
    """Position of the synopsis that folded log[index], or None.

    A synopsis written since `covers_turn_ids` existed names its turns. An older one does
    not, and is found by position instead: it sits right after the last turn it folded,
    so it is the first unnamed synopsis after the turn.
    """
    turn_id = log[index].get("turn_id")
    unnamed_after = None
    any_after = None
    for i in range(index + 1, len(log)):
        entry = log[i]
        if entry.get("type") != "synopsis":
            continue
        if any_after is None:
            any_after = i
        named = entry.get("covers_turn_ids")
        if named is None:
            if unnamed_after is None:
                unnamed_after = i
        elif turn_id and turn_id in named:
            return i
    # Nothing names it and nothing is unnamed: fall back to the first synopsis after it,
    # which over-drops at worst -- the safe direction for content that must not survive.
    return unnamed_after if unnamed_after is not None else any_after


def _uncompact_uncovered(log: List[Dict]) -> None:
    """Clears `compacted` on every turn that no remaining synopsis covers.

    A compacted turn is hidden from history on the strength of a synopsis standing in for
    it. With that synopsis gone the turn would be neither shown nor summarised.
    """
    named: Set[str] = set()
    positional: Set[int] = set()
    previous = -1
    for i, entry in enumerate(log):
        if entry.get("type") != "synopsis":
            continue
        ids = entry.get("covers_turn_ids")
        if ids is not None:
            named.update(ids)
        else:
            positional.update(id(log[j]) for j in range(previous + 1, i) if log[j].get("compacted"))
        previous = i
    for entry in log:
        if entry.get("compacted") and id(entry) not in positional and entry.get("turn_id") not in named:
            entry.pop("compacted", None)


def remove_turns_from_log(session: Dict, doomed: Iterable[Dict]) -> int:
    """Removes `doomed` from the log, by identity, with every synopsis that could repeat them.

    Rebinds `unified_log`, so the caller persists with `_save_session_to_disk`, never a
    tail write. Returns how many synopses were dropped.
    """
    log = session.get("unified_log") or []
    doomed_ids = {id(turn) for turn in doomed}

    first_tainted = None
    for i, entry in enumerate(log):
        if id(entry) in doomed_ids and entry.get("compacted"):
            covering = _covering_synopsis_index(log, i)
            if covering is not None and (first_tainted is None or covering < first_tainted):
                first_tainted = covering

    dropped = set()
    if first_tainted is not None:
        dropped = {id(entry) for entry in log[first_tainted:] if entry.get("type") == "synopsis"}

    kept = [entry for entry in log if id(entry) not in doomed_ids and id(entry) not in dropped]
    if dropped:
        _uncompact_uncovered(kept)
    session["unified_log"] = kept
    return len(dropped)


class TurnDeletionMixin:
    """/delete, /purge's whole-turn expansion, and the message-delete listeners' cascade."""

    async def delete_turns(self, channel, session: Dict, turns: List[Dict], *,
                           already_gone: Iterable[int] = ()) -> Dict[str, int]:
        """Deletes whole turns: every message they own, their log entries, and what repeats them.

        The caller holds the channel (`is_purging`). The log changes first and without an
        await, so nothing can find these turns mid-way and act on them twice.
        """
        turns = [turn for turn in turns if turn is not None]
        if not turns:
            return {"turns": 0, "messages": 0, "synopses": 0}

        gone = set(already_gone)
        remaining = [mid for turn in turns for mid in turn.get("message_ids", []) if mid not in gone]

        self._decrement_ltm_counters(session, turns)
        synopses = remove_turns_from_log(session, turns)
        messages = await self._delete_channel_messages(channel, remaining)
        await self._persist_after_turn_removal(channel.id, session)
        return {"turns": len(turns), "messages": messages, "synopses": synopses}

    def _decrement_ltm_counters(self, session: Dict, turns: List[Dict]) -> None:
        # speaker_pid -> participant, resolved once rather than per turn.
        pid_to_profile = {}
        for p in session.get('profiles', []):
            pid = self.cog.profile_manager._get_pid_from_name_any(p['owner_id'], p['profile_name'])
            pid_to_profile.setdefault(pid, p)
        for turn in turns:
            if turn.get("is_user") is False:
                p = pid_to_profile.get(turn.get("speaker_pid"))
                if p:
                    p['ltm_counter'] = max(0, p.get('ltm_counter', 0) - 1)

    async def _persist_after_turn_removal(self, channel_id: int, session: Dict) -> None:
        session_type = session.get("type", "multi")
        key = (channel_id, None, None)
        effectively_empty = not session.get("unified_log") or all(
            turn.get("type") in ("whisper", "private_response") for turn in session["unified_log"])
        if effectively_empty:
            await self.cog.session_manager._delete_session_from_disk(key, session_type)
            for p in session.get("profiles", []):
                self.cog.ltm_recall_history.pop((channel_id, p['owner_id'], p['profile_name']), None)
        else:
            await self.cog.session_manager._save_session_to_disk(key, session_type, session["unified_log"])
        # Deleting turns can strand or reveal a pending whisper -- the one thing a rebuild
        # derived -- so that is recomputed rather than re-reading the log just written.
        self.cog.session_manager._recompute_pending_whispers(session)
        self.cog.session_last_accessed[channel_id] = time.time()

    async def _delete_channel_messages(self, channel, message_ids: Iterable[int]) -> int:
        """Deletes these messages, whoever posted them. Returns how many are now gone.

        Every id is marked in `purged_message_ids` first, so the delete listeners do not
        mistake the bot's own deletions for somebody else's.
        """
        ids = [mid for mid in dict.fromkeys(message_ids) if mid]
        if not ids or channel is None:
            return 0
        for mid in ids:
            self.cog.purged_message_ids[mid] = True

        guild = getattr(channel, "guild", None)
        me = guild.me if guild else None
        if me is None or not channel.permissions_for(me).manage_messages:
            gone = 0
            for mid in ids:
                if await self._delete_own_message(channel, mid):
                    gone += 1
            return gone

        now = time.time()
        recent = [mid for mid in ids
                  if now - discord.utils.snowflake_time(mid).timestamp() < _BULK_DELETE_MAX_AGE_SECONDS]
        singles = [mid for mid in ids if mid not in set(recent)]
        gone = 0
        for start in range(0, len(recent), 100):
            chunk = recent[start:start + 100]
            try:
                await channel.delete_messages([discord.Object(id=mid) for mid in chunk])
                gone += len(chunk)
            except discord.HTTPException:
                singles.extend(chunk)
        for mid in singles:
            try:
                await channel.get_partial_message(mid).delete()
                gone += 1
            except discord.NotFound:
                gone += 1
            except discord.HTTPException:
                pass
        return gone

    async def _delete_own_message(self, channel, message_id: int) -> bool:
        """Without Manage Messages the bot can still remove what it posted itself."""
        try:
            message = await channel.fetch_message(message_id)
        except discord.NotFound:
            return True
        except discord.HTTPException:
            return False
        try:
            if message.webhook_id:
                await self.cog.server_manager.run_webhook(channel, "delete_message", message_id)
            elif message.author.id == self.cog.bot.user.id:
                await message.delete()
            elif str(message.author.id) in self.cog.child_bots:
                await self.cog.manager_queue.put({
                    "action": "send_to_child", "bot_id": str(message.author.id),
                    "payload": {"action": "delete_message", "channel_id": channel.id,
                                "message_id": message_id}})
            else:
                return False
            return True
        except discord.NotFound:
            return True
        except discord.HTTPException:
            return False

    async def on_turn_messages_deleted(self, channel_id: int, deleted_ids: Set[int]) -> None:
        """Somebody deleted part of a turn: delete the rest of it, now or once the channel is idle.

        While the channel is busy the turn may still be gaining messages -- citations,
        warnings and files go out after the reply -- so it waits, and the drain picks it up
        with everything it ended up posting.
        """
        session = self.cog.multi_profile_channels.get(channel_id)
        if not session:
            return
        if not session.get("is_hydrated"):
            session = await self.cog.session_manager._ensure_session_hydrated(
                channel_id, session.get("type", "multi"))
            if not session:
                return

        turns = [turn for turn in session.get("unified_log", [])
                 if any(mid in deleted_ids for mid in turn.get("message_ids", []))]
        if not turns:
            return

        if any(session.get(flag) for flag in SESSION_BUSY_FLAGS):
            pending = session.setdefault("pending_turn_deletions", {})
            undeferrable = []
            for turn in turns:
                if turn.get("turn_id"):
                    pending.setdefault(turn["turn_id"], set()).update(deleted_ids)
                else:
                    undeferrable.append(turn)
            if pending:
                task = session.get("pending_turn_deletion_task")
                if task is None or task.done():
                    task = asyncio.create_task(self._drain_pending_turn_deletions(channel_id, session))
                    session["pending_turn_deletion_task"] = task
                    self.cog.background_tasks.add(task)
                    task.add_done_callback(self.cog.background_tasks.discard)
            turns = undeferrable
            if not turns:
                return

        channel = self.cog.bot.get_channel(channel_id)
        if channel is None:
            return
        already_busy = session.get("is_purging")
        session["is_purging"] = True
        try:
            await self.delete_turns(channel, session, turns, already_gone=deleted_ids)
        finally:
            if not already_busy:
                session["is_purging"] = False

    async def _drain_pending_turn_deletions(self, channel_id: int, session: Dict) -> None:
        while session.get("pending_turn_deletions"):
            # Bounded, and proceeds on timeout: a leaked flag must not keep a deleted
            # message's siblings in the channel for the life of the process.
            await self.cog.session_manager._wait_for_session_flags(
                session, SESSION_BUSY_FLAGS, TURN_DELETE_DEFER_TIMEOUT_SECONDS)
            pending = session.pop("pending_turn_deletions", None) or {}
            turns = [turn for turn in session.get("unified_log", []) if turn.get("turn_id") in pending]
            channel = self.cog.bot.get_channel(channel_id)
            if not turns or channel is None:
                continue
            gone = set().union(*pending.values())
            session["is_purging"] = True
            try:
                await self.delete_turns(channel, session, turns, already_gone=gone)
            except Exception as e:
                print(f"Deferred turn deletion failed in channel {channel_id}: {type(e).__name__}: {e}")
            finally:
                session["is_purging"] = False
