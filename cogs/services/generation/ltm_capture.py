import asyncio
import datetime
from typing import Any, Dict, List, Optional, Tuple

import discord

from ...utils.constants import (LTM_DUPLICATE_SIMILARITY, LTM_EXCERPT_TURNS,
                                LTM_FLAG_MIN_REPLIES, MIN_HISTORY_FOR_LTM_CREATION)
from ...utils.helpers import _resolve_zoneinfo, ltm_flag_enabled, restamp_turn, turn_posted_at
from ...managers.memory_manager import encode_embedding_b64
from ...managers.session_manager import reply_tags, turn_lookup, with_reply


def ltm_backlog(log: List[Dict[str, Any]], pid: str, read_through: Optional[str] = None,
                read_through_id: Optional[str] = None) -> Tuple[List[Dict[str, Any]], int]:
    """The public turns a seat has made no memory from yet, oldest first, and how many
    of them are its own replies.

    "Yet" is everything logged after `read_through_id`, the newest turn its last memory
    read. Log order, not time: a message sent while a reply was still generating is
    logged after that reply but stamped before it, and a time bookmark dropped it from
    both memories. When that turn is gone -- deleted, or trimmed off the log -- its
    moment, `read_through`, stands in. Whispers and private replies are left out: a
    memory is recalled in every channel, and a secret must not surface in one it was
    never told in. Folded turns are read like any other: compaction summarised them for
    the scene, which is not this character remembering them.

    Walks back from the end and stops at the bookmark, so it costs the backlog, not the
    log -- except while the bookmarked turn is missing, and the log is capped at
    LOG_TRIM_HIGH_WATER regardless.
    """
    since = None
    if read_through:
        try:
            since = datetime.datetime.fromisoformat(read_through)
        except (TypeError, ValueError):
            pass
    turns, found = [], False
    for turn in reversed(log):
        if read_through_id and turn.get("turn_id") == read_through_id:
            found = True
            break
        if turn.get("type") or turn.get("is_hidden"):
            continue
        if since and not read_through_id:
            moment = turn_posted_at(turn)
            if moment and moment <= since:
                break
        turns.append(turn)
    if since and read_through_id and not found:
        turns = [t for t in turns if not ((m := turn_posted_at(t)) and m <= since)]
    turns.reverse()
    return turns, sum(t.get("speaker_pid") == pid for t in turns)


def memory_written_ago(log: List[Dict[str, Any]], pid: str) -> Optional[int]:
    """How many of a seat's own replies back the memory it last wrote was made after
    (1 = its latest reply), or None once that is more than `LTM_FLAG_MIN_REPLIES` ago.

    Read off the turn `_summarize_and_store_ltm` stamps `ltm_created` on, so it is derived
    from the log like everything else here. Past the floor a flag is honoured at once, and
    there is nothing left to tell the character.
    """
    back = 0
    for turn in reversed(log):
        if turn.get("speaker_pid") != pid:
            continue
        back += 1
        if (turn.get("meta") or {}).get("ltm_created"):
            return back
        if back >= LTM_FLAG_MIN_REPLIES:
            return None
    return None


class LtmCaptureMixin:
    """Owns the summarise -> embed -> store chain for one seat's long-term memory.

    A seat's bookmark -- `ltm_read_through_id` and its moment, `ltm_read_through` -- is
    the only state: when a memory is due and what it reads are both derived from the
    log against it. Round-end wraps this in
    asyncio.create_task so a slow summary never blocks the queue. /memorise awaits it
    directly instead, since it has to report per-profile success back to the admin who
    ran it -- a fire-and-forget task can't do that.
    """

    def _ltm_due(self, session: Dict[str, Any], seat: Dict[str, Any],
                 p_settings: Dict[str, Any]) -> bool:
        """Whether this seat has replied `ltm_creation_interval` times since its last
        memory, or since a failed one's retry point. A deleted turn uncounts itself.

        In flag mode (`ltm_flag_enabled`) a seat whose character flagged a reply is due
        sooner -- `LTM_FLAG_MIN_REPLIES` replies in -- and the interval is the ceiling,
        so the mode only ever brings a memory earlier. The retry point holds either way.

        Either mode is also due once the backlog fills the excerpt window: the interval
        counts replies and the window counts turns, so a big cast reaches the window first
        and the oldest turns would fall off the front of every capture, unread for good.
        """
        if seat.get("_ltm_running"):
            return False
        pid = self.cog.profile_manager._get_pid_from_name_any(seat["owner_id"], seat["profile_name"])
        turns, own = ltm_backlog(session.get("unified_log", []), pid, seat.get("ltm_read_through"),
                                 seat.get("ltm_read_through_id"))
        need = int(p_settings.get("ltm_creation_interval", 10))
        if seat.get("_ltm_flag") and ltm_flag_enabled(p_settings):
            need = min(need, LTM_FLAG_MIN_REPLIES)
        if own < seat.get("_ltm_retry_at", 0):
            return False
        return own >= need or (own > 0 and len(turns) >= LTM_EXCERPT_TURNS)

    async def memorise_seats(
        self, session: Dict[str, Any], seats: List[Dict[str, Any]], guild_id: Optional[int],
        author: str, user_id: int, channel: Optional[discord.abc.Messageable] = None,
        source: str = "auto",
    ) -> Tuple[List[str], List[str], List[str]]:
        """Writes a memory now for each of `seats`: (made, skipped, none) as display names.

        The body of `/memorise`, and what `/suspend` runs once the log is deleted, from the
        session it still holds: the turns since a seat's last memory are held nowhere else.
        """
        made, skipped, none = [], [], []
        for seat in seats:
            name = seat["profile_name"]
            index = self.cog.profile_manager._get_user_index(seat["owner_id"])
            settings = self.cog.profile_manager._get_profile_config(
                seat["owner_id"], name, name in index.get("borrowed", [])) or {}
            if not settings.get("ltm_creation_enabled", False):
                skipped.append(f"{name} (LTM disabled)")
                continue
            if seat.get("_ltm_running"):
                skipped.append(f"{name} (a memory is already being written)")
                continue
            seat["_ltm_running"] = True
            try:
                created, detail = await self._summarize_and_store_ltm(
                    session, seat, settings, guild_id, author, user_id,
                    warning_channel=channel, source=source)
            except Exception as e:
                created, detail = False, str(e)
            finally:
                seat.pop("_ltm_running", None)
            (made if created else none).append(name if created else f"{name} ({detail})")
        return made, skipped, none

    async def _summarize_and_store_ltm(
        self, session: Dict[str, Any], seat: Dict[str, Any], p_settings: Dict[str, Any],
        guild_id: Optional[int], r_author: str, triggering_user_id: int,
        warning_channel: Optional[discord.abc.Messageable] = None, source: str = "auto",
    ) -> Tuple[bool, str]:
        owner_id, profile_name = seat["owner_id"], seat["profile_name"]
        pid = self.cog.profile_manager._get_pid_from_name_any(owner_id, profile_name)
        log = session.get("unified_log", [])
        # Spent whatever the verdict, like the bookmark. A flag raised while this runs
        # is the next memory's, so it is cleared here, before the awaits.
        seat.pop("_ltm_flag", None)
        turns, own = ltm_backlog(log, pid, seat.get("ltm_read_through"), seat.get("ltm_read_through_id"))
        turns = turns[-LTM_EXCERPT_TURNS:]
        if len(turns) < MIN_HISTORY_FOR_LTM_CREATION:
            return False, "nothing new since the last memory"

        # Taken before the awaits: turns that land while this generates are the next
        # memory's, not lost to this one.
        read_through = turn_posted_at(turns[-1]) or datetime.datetime.now(datetime.timezone.utc)
        read_through_id = turns[-1].get("turn_id")
        clock, _ = _resolve_zoneinfo(p_settings.get("timezone"))
        tags, find = reply_tags(turns), turn_lookup(log)
        excerpt = [with_reply(restamp_turn(t.get("content") or "", turn_posted_at(t), clock), t, clock, tags, find)
                   for t in turns]

        # The name this seat's turns carry, `<Name> [ID: ...]`, so the summariser is told
        # which speaker in the transcript it is writing for.
        own_turn = next((t for t in reversed(log) if t.get("speaker_pid") == pid), None)
        header = (own_turn or {}).get("content") or ""
        end = header.find("> [ID: ")
        character_name = header[1:end] if header.startswith("<") and end > 1 else profile_name

        mm = self.cog.memory_manager
        memories = await mm._generate_ltm_data_from_history(
            excerpt, r_author, guild_id, character_name,
            profile_owner_id=owner_id, profile_name=profile_name, warning_channel=warning_channel,
            background=self.cog.session_manager.get_latest_synopsis(session),
        )
        stored = 0
        if memories:
            embeddings = await asyncio.gather(*(
                mm._get_embedding(m, guild_id, task_type="RETRIEVAL_DOCUMENT") for m in memories))
            # All or none: a partial set would move the bookmark past the rest for good.
            if not all(embeddings):
                memories = None
            else:
                stored = await mm._add_ltms(
                    owner_id, profile_name,
                    [(m, encode_embedding_b64(e)) for m, e in zip(memories, embeddings)],
                    guild_id, r_author, duplicate_similarity=LTM_DUPLICATE_SIMILARITY, source=source)
        if memories is None:
            # The bookmark stays, so the same turns are read again -- once another
            # interval's worth of replies has built up, not on every round while the
            # API is down. In memory only: a restart is a fair moment to try again.
            seat["_ltm_retry_at"] = own + int(p_settings.get("ltm_creation_interval", 10))
            return False, "summarisation failed"

        # Judged, whatever the verdict: "nothing worth keeping" and "already known" are
        # answers about these turns, and reading them again would only repeat them.
        seat["ltm_read_through"] = read_through.isoformat()
        if read_through_id:
            seat["ltm_read_through_id"] = read_through_id
        else:
            seat.pop("ltm_read_through_id", None)
        seat.pop("_ltm_retry_at", None)
        self.cog.session_manager._save_multi_profile_sessions()

        if not memories:
            return False, "nothing worth remembering"
        if not stored:
            return False, "already remembered"
        # Link LTM creation to the turn metadata for trace transparency
        if own_turn and "meta" in own_turn:
            own_turn["meta"]["ltm_created"] = stored
        return True, "created" if stored == 1 else f"created {stored}"
