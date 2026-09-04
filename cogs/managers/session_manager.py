import os
import sys
import gzip
import zstandard as zstd
import uuid
import heapq
import shutil
import asyncio
import random
import time
from discord.ext import tasks
import pathlib
import datetime
import collections
import discord
from typing import Dict, List, Any, Optional, Union, Tuple
from cryptography.fernet import InvalidToken
import orjson as json

from ..utils.constants import (
    USERS_DIR, SERVERS_DIR, SESSIONS_GLOBAL_DIR, defaultConfig,
    PRIMARY_MODEL_NAME, FALLBACK_MODEL_NAME,
    COMPACTION_THRESHOLD_DEFAULT, COMPACTION_CHUNK_DEFAULT,
    COMPACTION_MODEL_DEFAULT, COMPACTION_FALLBACK_MODEL_DEFAULT,
    DEFAULT_CAST_POLICY,
)
from .storage_manager import IOManager, _delete_file_shard, _get_compressor, _get_decompressor

# Turns allowed to accumulate in the tail sidecar before the next flush re-seals the
# whole log into the cold segment. Sized so the common case -- a round appending a
# handful of turns -- never rewrites the 264 KiB cold file, while the tail itself stays
# small enough that writing it is cheap on the e2-micro.
SESSION_HOT_TAIL_MAX = 250

# How often the coalescing flusher drains the dirty set. Every operation that ends a
# turn, deletes history, or evicts a session still flushes immediately; this interval
# only bounds how long a mid-round append can sit in memory.
SESSION_FLUSH_INTERVAL_SECONDS = 5.0

# Compaction is opt-in: a session that never enables it behaves exactly as before.
DEFAULT_COMPACTION_CONFIG = {
    "enabled": False,
    "threshold": COMPACTION_THRESHOLD_DEFAULT,
    "chunk": COMPACTION_CHUNK_DEFAULT,
    "model": COMPACTION_MODEL_DEFAULT,
    "fallback_model": COMPACTION_FALLBACK_MODEL_DEFAULT,
}

# Repeated verbatim on every turn of a session's log: interning them means one string
# object per session instead of one per turn.
_INTERNED_TURN_FIELDS = ("speaker_pid", "target_pid", "profile_name", "role", "type", "speaker_name")


def intern_turn(turn: Dict[str, Any]) -> Dict[str, Any]:
    """Collapse a turn's repeated field values onto shared string objects, in place.

    A 1000-turn log carries 1000 copies of the same handful of pids, profile names and
    role tags -- roughly 10% of the log's ~1 MB resident footprint for data with a
    handful of distinct values. Returns the same dict so it can wrap an append.
    """
    for field in _INTERNED_TURN_FIELDS:
        value = turn.get(field)
        if type(value) is str:
            turn[field] = sys.intern(value)
    return turn


class SessionManager:
    """Owns chat session state: history hydration/dehydration to disk, multi-profile
    channel memory tracking, and the inactive-session eviction sweep.

    Holds a back-reference to the parent cog for state/logic not yet migrated
    (bot, manager_queue, in-memory caches, and profile/server-index lookups),
    per the transitional Dependency Injection pattern in CLAUDE.md.
    """

    def __init__(self, cog):
        self.cog = cog

    def _get_session_dir_path(self, session_key: Any, session_type: str) -> pathlib.Path:
        if session_type == 'global_chat':
            _, user_id, _ = session_key
            return pathlib.Path(SESSIONS_GLOBAL_DIR) / str(user_id)

        channel_id, _, _ = session_key
        channel = self.cog.bot.get_channel(channel_id)
        server_id = channel.guild.id if channel and getattr(channel, 'guild', None) else None
        if not server_id:
            raise ValueError("Sessions are not supported in Direct Messages.")
        return pathlib.Path(SERVERS_DIR) / str(server_id) / "sessions" / str(channel_id) / session_type

    def _get_session_path(self, session_key: Any, session_type: str) -> pathlib.Path:
        if session_type == 'global_chat':
            _, user_id, profile_name = session_key
            pid = self.cog.profile_manager._get_pid_from_name_any(user_id, profile_name)
            return pathlib.Path(USERS_DIR) / str(user_id) / "profiles" / pid / "global_chat.json.gz"

        dir_path = self._get_session_dir_path(session_key, session_type)
        # All other session types use a unified log
        return dir_path / "session_log.json.gz"

    def _get_session_hot_path(self, session_key: Any, session_type: str) -> Optional[pathlib.Path]:
        """Sidecar holding the unsealed tail of a channel session's unified_log.

        None for global_chat, which is already capped at STM_LIMIT_MAX * 2 turns and
        is not worth segmenting.
        """
        if session_type == 'global_chat':
            return None
        return self._get_session_dir_path(session_key, session_type) / "session_log.hot.json.gz"

    def _encode_session_bytes(self, payload: Any) -> bytes:
        """orjson -> zstd -> Fernet, the on-disk format for every session file.

        Only ever called from inside asyncio.to_thread, so it goes through
        _get_compressor() rather than building a ZstdCompressor per call -- see the
        thread-safety note at the top of storage_manager.
        """
        serialized_bytes = json.dumps(payload, option=json.OPT_SERIALIZE_NUMPY | json.OPT_NON_STR_KEYS)
        return self.cog.fernet.encrypt(_get_compressor().compress(serialized_bytes))

    @staticmethod
    def _atomic_write(path: pathlib.Path, blob: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + '.tmp')
        with open(temp_path, 'wb') as f:
            f.write(blob)
        try:
            os.replace(temp_path, path)
        except FileNotFoundError:
            path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temp_path, path)

    def _get_live_session(self, session_key: Any, session_type: str) -> Optional[Dict]:
        """The in-memory session dict a save/flush refers to, or None if it is gone."""
        if session_type == 'global_chat':
            return self.cog.global_chat_sessions.get(session_key)
        channel_id, _, _ = session_key
        return self.cog.multi_profile_channels.get(channel_id)

    def _note_cold_length(self, session_key: Any, session_type: str, saved_log: Any, length: int) -> None:
        """Record how much of unified_log the cold segment now holds.

        The tail flush is only correct while `cold file == unified_log[:_log_cold_len]`.
        Every caller that saves a channel session passes session['unified_log'] itself,
        so identity is a reliable test that the invariant holds; anything else resets the
        counter to 0, which costs one full flush rather than writing a tail that does not
        line up with its cold segment.
        """
        if session_type == 'global_chat':
            return
        session = self._get_live_session(session_key, session_type)
        if session is None:
            return
        session['_log_cold_len'] = length if session.get('unified_log') is saved_log else 0

    async def _save_session_tail_to_disk(self, session_key: Any, session_type: str, session: Dict) -> bool:
        """Persist only the turns appended since the last full flush.

        A channel session's log is append-mostly, but every writer used to rewrite the
        whole file: a 1000-turn log is 264 KiB on disk and ~1.6 ms of orjson/zstd/Fernet
        per save, and a four-participant round did that six or more times. This writes a
        second file holding just unified_log[_log_cold_len:], so a round's flush is
        proportional to the turns it actually added.

        Returns False when the tail has grown past SESSION_HOT_TAIL_MAX (or the invariant
        cannot be trusted), meaning the caller should fall back to a full flush -- which
        re-seals everything into the cold segment and starts a fresh tail.
        """
        unified_log = session.get('unified_log')
        if unified_log is None:
            return True

        cold_len = session.get('_log_cold_len', 0)
        if not isinstance(cold_len, int) or cold_len <= 0 or cold_len > len(unified_log):
            return False

        tail = unified_log[cold_len:]
        if len(tail) > SESSION_HOT_TAIL_MAX:
            return False
        if not tail:
            return True

        hot_path = self._get_session_hot_path(session_key, session_type)
        if hot_path is None:
            return False

        channel_id, _, _ = session_key
        channel = self.cog.bot.get_channel(channel_id)
        if not channel or not getattr(channel, 'guild', None):
            return True

        payload = {"cold_len": cold_len, "turns": tail}
        try:
            await asyncio.to_thread(
                lambda: self._atomic_write(hot_path, self._encode_session_bytes(payload))
            )
        except Exception as e:
            print(f"Error saving session tail for key {session_key}: {e}")
            return False
        return True

    async def _save_session_to_disk(self, session_key: Any, session_type: str, session_data: Union[List[Dict], Dict]):
        if not session_data:
            await self._delete_session_from_disk(session_key, session_type)
            return

        if session_type != 'global_chat':
            channel_id, _, _ = session_key
            channel = self.cog.bot.get_channel(channel_id)
            if not channel or not getattr(channel, 'guild', None):
                return

        data_to_save = session_data

        if session_type == 'global_chat':
            if isinstance(session_data, dict):
                if 'unified_log' in session_data:
                    if not session_data['unified_log']:
                        await self._delete_session_from_disk(session_key, session_type)
                        return
                    data_to_save = session_data['unified_log']
                else:
                    return
            elif hasattr(session_data, 'history'):
                 log = []
                 for content in session_data.history:
                     parts_text = "".join(p if isinstance(p, str) else p.get('text', '') for p in content.get('parts', []))
                     log.append({
                         "turn_id": str(uuid.uuid4()), "role": content.get('role', 'user'), "content": parts_text,
                         "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                     })
                 data_to_save = log

        if hasattr(data_to_save, 'history') and not data_to_save.history:
             await self._delete_session_from_disk(session_key, session_type)
             return

        try:
            path = self._get_session_path(session_key, session_type)
            hot_path = self._get_session_hot_path(session_key, session_type)
            data_copy = list(data_to_save) if isinstance(data_to_save, list) else data_to_save.copy()

            def _thread_save():
                self._atomic_write(path, self._encode_session_bytes(data_copy))
                if hot_path is not None:
                    # A full flush seals the whole log into the cold segment, so the tail
                    # sidecar is now redundant. Written first, deleted second: a crash in
                    # between leaves a stale tail that _load_session_from_disk reconciles
                    # back to the pre-flush state rather than to a torn one.
                    try:
                        os.unlink(hot_path)
                    except FileNotFoundError:
                        pass

            await asyncio.to_thread(_thread_save)
            self._note_cold_length(session_key, session_type, data_to_save, len(data_copy))
            # The log on disk is now current, so any queued flush is satisfied.
            self.cog.dirty_sessions.pop(session_key, None)
        except Exception as e:
            print(f"Error saving session for key {session_key}: {e}")

    def _decode_session_file(self, path: pathlib.Path) -> Any:
        """Read one session file. Runs inside asyncio.to_thread."""
        with open(path, 'rb') as f:
            encrypted_compressed_bytes = f.read()
        decrypted_compressed_bytes = self.cog.fernet.decrypt(encrypted_compressed_bytes)
        try:
            json_bytes = _get_decompressor().decompress(decrypted_compressed_bytes)
        except zstd.ZstdError:
            json_bytes = gzip.decompress(decrypted_compressed_bytes)
        if not json_bytes:
            return None
        return json.loads(json_bytes)

    @staticmethod
    def _reconcile_log_segments(cold: Any, hot: Any) -> Tuple[Any, int]:
        """Splice a cold segment and its tail sidecar back into one log.

        Returns (log, cold_len). A session written before segmentation existed has no
        sidecar and loads as-is, which is what keeps the format backward compatible.

        The tail records the cold length it was written against. Anything else means a
        crash landed between the two writes of a full flush: the tail is either stale
        (its cold_len is behind, so the prefix it names still reconstructs the log the
        session had before that flush) or orphaned ahead of a cold segment that never
        landed, in which case the only self-consistent state is the cold segment alone.
        """
        if not isinstance(cold, list):
            return cold, 0
        if not isinstance(hot, dict):
            return cold, len(cold)

        cold_len = hot.get("cold_len")
        turns = hot.get("turns")
        if not isinstance(cold_len, int) or not isinstance(turns, list) or not (0 <= cold_len <= len(cold)):
            return cold, len(cold)
        return cold[:cold_len] + turns, cold_len

    async def _load_session_from_disk(self, session_key: Any, session_type: str) -> Optional[Union[List[Dict], Dict]]:
        data, _ = await self._load_session_segments(session_key, session_type)
        return data

    async def _load_session_segments(self, session_key: Any, session_type: str) -> Tuple[Optional[Union[List[Dict], Dict]], int]:
        """_load_session_from_disk, plus how much of the returned log is already sealed.

        Only _ensure_session_hydrated needs the second value, and only when it actually
        adopts the disk log -- if it keeps a longer in-memory log instead, the disk
        segment boundary does not describe it and the counter must stay 0.
        """
        try:
            path = self._get_session_path(session_key, session_type)
            if not path.exists():
                return None, 0
            hot_path = self._get_session_hot_path(session_key, session_type)

            def _thread_load():
                cold = self._decode_session_file(path)
                hot = None
                if hot_path is not None and hot_path.exists():
                    try:
                        hot = self._decode_session_file(hot_path)
                    except Exception as e:
                        # A tail we cannot read is recoverable -- the cold segment is a
                        # complete log as of the last full flush. Only the unsealed turns
                        # are lost, so fall through rather than failing the whole load.
                        print(f"Warning: unreadable session tail at {hot_path}: {e}. Using cold segment only.")
                return self._reconcile_log_segments(cold, hot)

            data, cold_len = await asyncio.to_thread(_thread_load)

            if not data:
                await self._delete_session_from_disk(session_key, session_type)
                return None, 0

            if session_type == 'global_chat':
                if data and isinstance(data, list) and 'parts' in data[0]:
                    unified_log = []
                    for item in data:
                        role = item.get('role')
                        parts = item.get('parts', [])
                        content = "".join(p.get('text', '') for p in parts)
                        log_item = {
                            "turn_id": str(uuid.uuid4()),
                            "role": role,
                            "content": content,
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                        }
                        unified_log.append(intern_turn(log_item))

                    return {'unified_log': unified_log}, 0

                elif data and isinstance(data, list) and 'turn_id' in data[0]:
                    for item in data:
                        intern_turn(item)
                    return {'unified_log': data}, 0

                return None, 0

            else:
                if isinstance(data, list):
                    # One pass at hydration collapses the log's repeated pid/name/role
                    # strings onto shared objects for as long as the session stays
                    # resident; appends afterwards intern as they arrive.
                    for item in data:
                        intern_turn(item)
                return data, cold_len
        except (gzip.BadGzipFile, zstd.ZstdError, json.JSONDecodeError, InvalidToken):
             print(f"Warning: Corrupted or old-format session file for key {session_key}. Deleting file.")
             await self._delete_session_from_disk(session_key, session_type)
        except Exception as e:
            print(f"Error loading session for key {session_key}: {e}")
        return None, 0

    async def _delete_session_from_disk(self, session_key: Any, session_type: str):
        try:
            path = self._get_session_path(session_key, session_type)
            await asyncio.to_thread(_delete_file_shard, str(path))
            hot_path = self._get_session_hot_path(session_key, session_type)
            if hot_path is not None:
                await asyncio.to_thread(_delete_file_shard, str(hot_path))
            self._note_cold_length(session_key, session_type, None, 0)
            # A queued flush predates the delete, so honouring it would write the file
            # back -- with whatever the delete was meant to remove still in it.
            self.cog.dirty_sessions.pop(session_key, None)
        except Exception as e:
            print(f"Error deleting session file for key {session_key}: {e}")

    async def suspend_channel_session(self, channel_id: int) -> bool:
        """Tear one channel's session down -- the body of `/suspend`.

        Pops the session, clears its per-profile LTM counters and recall penalties,
        tells child bot participants to drop the channel and stop typing, deletes the
        log from disk and cancels the worker. Returns False if there was no session.

        Does *not* persist the blueprint index: a caller suspending many channels
        calls `_save_multi_profile_sessions` once at the end instead of per channel,
        since that rewrites every server index each time.
        """
        # A live game outlives the session unless it is explicitly stopped -- the game
        # cache is keyed by channel and holds no reference to the session at all, so a
        # suspended channel would otherwise keep a table running against a cast that
        # no longer exists. Done before the early return so it also covers a game left
        # behind by a session that was already gone.
        game_service = getattr(self.cog, "game_service", None)
        if game_service:
            game_service.teardown_channel(channel_id)

        session = self.cog.multi_profile_channels.pop(channel_id, None)
        if not session:
            return False

        for participant in session.get("profiles", []):
            p_oid = participant.get("owner_id")
            p_name = participant.get("profile_name")
            if p_oid and p_name:
                # Clear the round counter and the LTM recall penalty history for this
                # profile, or a fresh session inherits a half-elapsed cadence.
                self.cog.message_counters_for_ltm.pop((p_oid, p_name, "guild"), None)
                self.cog.ltm_recall_history.pop((channel_id, p_oid, p_name), None)

            if participant.get("method") == "child_bot":
                bot_id = participant.get("bot_id")
                if bot_id:
                    await self.cog.manager_queue.put({
                        "action": "send_to_child", "bot_id": bot_id,
                        "payload": {"action": "session_update_remove", "channel_id": channel_id}
                    })
                    await self.cog.manager_queue.put({
                        "action": "send_to_child", "bot_id": bot_id,
                        "payload": {"action": "stop_typing", "channel_id": channel_id}
                    })

        session_type = session.get("type", "multi")
        await self._delete_session_from_disk((channel_id, None, None), session_type)

        if session.get('worker_task'):
            self._safe_cancel_task(session['worker_task'])

        self.cog.session_last_accessed.pop(channel_id, None)
        return True

    async def suspend_all_sessions(self) -> Tuple[int, int, int]:
        """Suspend every session on this instance.

        Returns (channels suspended, guilds touched, leftover session dirs swept).
        """
        guild_ids = set()
        suspended = 0

        for channel_id in list(self.cog.multi_profile_channels.keys()):
            channel = self.cog.bot.get_channel(channel_id)
            guild = getattr(channel, "guild", None)
            if await self.suspend_channel_session(channel_id):
                suspended += 1
                if guild:
                    guild_ids.add(guild.id)
            # 0.25 vCPU baseline: each channel deletes two files and may queue child
            # bot traffic, so yield between them rather than holding the loop for the
            # whole instance.
            await asyncio.sleep(0)

        # Clearing the dict is what empties every server index, so this has to land
        # even when no channel was live.
        self._save_multi_profile_sessions()

        swept = await asyncio.to_thread(self._sweep_orphaned_session_dirs)
        return suspended, len(guild_ids), swept

    def _sweep_orphaned_session_dirs(self) -> int:
        """Delete every `<server>/sessions/` tree left on disk. Returns the count.

        `_delete_session_from_disk` resolves a channel's directory through
        `bot.get_channel`, so a session whose channel is uncached -- deleted, or in a
        guild the bot has lost access to -- keeps its log no matter how many times it
        is suspended. Those are precisely the logs a global wipe exists to reach, so
        the sweep goes by path rather than by live channel. Nothing but session logs
        lives under `sessions/`; profiles, memories and the server index sit elsewhere.
        """
        removed = 0
        servers_path = pathlib.Path(SERVERS_DIR)
        if not servers_path.is_dir():
            return 0

        for server_dir in list(servers_path.iterdir()):
            sessions_dir = server_dir / "sessions"
            if not sessions_dir.is_dir():
                continue
            for channel_dir in list(sessions_dir.iterdir()):
                if not channel_dir.is_dir():
                    continue
                try:
                    shutil.rmtree(channel_dir)
                    removed += 1
                except OSError as e:
                    print(f"Error removing session directory {channel_dir}: {e}")
        return removed

    def mark_session_dirty(self, session_key: Any, session_type: str, structural: bool = False) -> None:
        """Queue a session's log for the next coalesced flush.

        The turn path used to persist synchronously after every append, so a four-
        participant round rewrote the whole log six or more times to reach a state only
        the last of those writes described. Marking instead collapses a round into a
        single write, and the tail segmentation keeps that write proportional to the
        turns the round added rather than to the length of the log.

        Only for changes that a later flush is guaranteed to supersede. Anything that
        finishes an operation -- a delivered turn's message_ids, a whisper landing, a
        regeneration, an edit, any deletion, eviction -- flushes immediately instead, so
        the coalescing window can never swallow a completed turn or resurrect deleted
        history.

        Pass structural=True when the change touched turns that may already be sealed
        into the cold segment -- an in-place edit, a filtered log, anything that is not
        an append. A tail write only rewrites unified_log[_log_cold_len:], so without
        this the change would never reach disk.
        """
        existing = self.cog.dirty_sessions.get(session_key)
        if existing is not None:
            structural = structural or existing[1]
        self.cog.dirty_sessions[session_key] = (session_type, structural)

    async def flush_session(self, session_key: Any, session_type: str, structural: bool = False) -> None:
        """Write one session now, preferring the tail sidecar over a full rewrite."""
        pending = self.cog.dirty_sessions.pop(session_key, None)
        if pending is not None:
            structural = structural or pending[1]

        session = self._get_live_session(session_key, session_type)
        if session is None:
            return

        if session_type == 'global_chat':
            await self._save_session_to_disk(session_key, session_type, session)
            return

        # Dehydrated between the mark and now: eviction already wrote the log out and
        # dropped it, so there is nothing left to persist.
        if not session.get("is_hydrated") or session.get("unified_log") is None:
            return

        if not structural and await self._save_session_tail_to_disk(session_key, session_type, session):
            return
        await self._save_session_to_disk(session_key, session_type, session["unified_log"])

    async def flush_all_dirty(self) -> None:
        """Drain the dirty set. Used by the periodic task and by shutdown."""
        for session_key, (session_type, structural) in list(self.cog.dirty_sessions.items()):
            try:
                await self.flush_session(session_key, session_type, structural=structural)
            except Exception as e:
                print(f"Error flushing session {session_key}: {e}")
                self.cog.dirty_sessions.pop(session_key, None)

    @tasks.loop(seconds=SESSION_FLUSH_INTERVAL_SECONDS)
    async def flush_dirty_sessions_task(self):
        if not self.cog.has_lock:
            return
        await self.flush_all_dirty()

    def _get_mapping_path(self, mapping_key: Any) -> pathlib.Path:
        session_type, key_id = mapping_key
        if session_type == 'global_chat':
            user_id = key_id
            dummy_session_key = (None, user_id, None)
            dir_path = self._get_session_dir_path(dummy_session_key, session_type)
        elif session_type in ['multi', 'freewill']:
            channel_id = key_id
            dummy_session_key = (channel_id, None, None)
            dir_path = self._get_session_dir_path(dummy_session_key, session_type)
        else:
            raise ValueError(f"Unknown mapping key type: {session_type}")

        return dir_path / "_mappings.json.gz"

    def _load_mapping_from_disk(self, mapping_key: Any) -> Dict[str, Any]:
        try:
            path = self._get_mapping_path(mapping_key)
            if path.exists():
                data = IOManager.read_json_gzip(str(path), self.cog.fernet) or {}
                return data
        except Exception as e:
            print(f"Error loading mapping for key {mapping_key}: {e}")
        return {}

    def _save_mapping_to_disk(self, mapping_key: Any, data: Dict[str, Any]):
        try:
            path = self._get_mapping_path(mapping_key)
            if not data:
                _delete_file_shard(str(path))
            else:
                IOManager.write_json_gzip(data, str(path), self.cog.fernet)
        except Exception as e:
            print(f"Error saving mapping for key {mapping_key}: {e}")

    def _get_mapping_key_for_session(self, session_key: Any, session_type: str) -> Any:
        if session_type == 'global_chat':
            _, user_id, _ = session_key
            return ('global_chat', user_id)

        # All other session types are keyed by channel_id
        elif session_type in ['multi']:
            channel_id, _, _ = session_key
            return (session_type, channel_id)
        return None

    def cast_policy_for_channel(self, guild_id: int, channel_id: int) -> str:
        """This channel's cast policy, without waking or creating a session.

        The `/session config` gate has to answer this *before* `_ensure_session_shell`
        runs, or the act of checking whether a member may open the editor would create
        the session that decides it -- and a channel that has never had one would then
        answer with a freshly minted default instead of "no".

        Live session first, then the saved blueprint, then CLOSED. Absent means closed
        at every step: a blueprint written before the field existed keeps the
        admin-only access it was configured under.
        """
        session = self.cog.multi_profile_channels.get(channel_id)
        if session:
            return session.get("cast_policy") or DEFAULT_CAST_POLICY

        try:
            server_index = self.cog.server_manager._get_server_index(str(guild_id))
            active = server_index.get("active_sessions", {})
            if not isinstance(active, dict):
                return DEFAULT_CAST_POLICY
            blueprint = (active.get("regular", {}) or {}).get(str(channel_id))
            if blueprint:
                return blueprint.get("cast_policy") or DEFAULT_CAST_POLICY
        except Exception:
            # A gate that cannot read its own setting refuses rather than opens.
            return DEFAULT_CAST_POLICY

        return DEFAULT_CAST_POLICY

    def participant_identity(self, participant: Dict) -> Tuple[int, str]:
        """Canonical identity of a seated participant: `(owner_id, pid)`.

        One function because there were three comparisons -- the cast editor's
        `_participant_key`, `/session swap`'s inline `owner_id`/`profile_name` test,
        and the removal filters -- and a rename made them disagree about whether a
        character was already seated. `method` is deliberately not part of it: a child
        bot and a webhook are two voices for one profile, not two characters.

        Falls back to `_get_pid_from_name_any` (the *soft* resolver) for an entry
        seated before PIDs were stamped, so a legacy participant whose name no longer
        resolves still gets a token unique to that name and stays individually
        comparable, where a strict None would collide every unresolvable entry into
        one identity.
        """
        owner_id = int(participant["owner_id"])
        pid = participant.get("pid") or self.cog.profile_manager._get_pid_from_name_any(
            owner_id, participant.get("profile_name") or "")
        return (owner_id, pid)

    def _repair_participant_identity(self, participants: List[Dict]) -> bool:
        """Stamp a missing PID and re-spell a stale name, in place. True if changed.

        A blueprint has always recorded both the PID and the name for every seated
        participant, but only the name was ever read back. That made the name the de
        facto identity of a cast entry, and a name is a label the owner can change:
        `_rename_profile` hot-swaps `profile_name` across live sessions but never
        persists the swap, so a restart in that window restored a cast naming a
        profile that no longer exists -- and `_get_pid_from_name_any` then answered
        with the dead name itself, which stopped matching the `speaker_pid` on the
        profile's own logged turns.

        The PID is the half that survives a rename, so it wins: the name is
        re-derived from it here, on every load, which makes the repair idempotent and
        self-healing rather than dependent on a save landing. Entries written before
        PIDs were stamped carry only a name, so they are resolved forward once and
        stamped -- strictly, because a name-shaped PID is exactly what this exists to
        stop being written.

        A PID that resolves to nothing is left alone: the profile was deleted, and
        removing the participant is the hydration sweep's job, not this one's.
        """
        pm = self.cog.profile_manager
        changed = False

        for p in participants:
            owner_id = p.get("owner_id")
            if owner_id is None:
                continue
            owner_id = int(owner_id)

            pid = p.get("pid")
            if not pid:
                resolved = pm._get_pid_from_name(owner_id, p.get("profile_name") or "")
                if resolved:
                    p["pid"] = resolved
                    changed = True
                continue

            # include_borrowed: a seated borrow's local name is the borrower's own, and
            # re-spelling it here is not re-sharing anything.
            current = pm._get_name_from_pid(owner_id, pid, include_borrowed=True)
            if current and current != p.get("profile_name"):
                p["profile_name"] = current
                changed = True

        return changed

    async def _load_multi_profile_sessions(self):
        await self.cog.bot.wait_until_ready()

        # Set by the identity repair below. Persisted once, after every server has been
        # walked, rather than per session -- _save_multi_profile_sessions rewrites all
        # of them anyway, so calling it inside the loop would be O(servers) full saves.
        repaired_any = False

        servers_dir = SERVERS_DIR
        if not os.path.isdir(servers_dir):
            return

        for server_id_str in os.listdir(servers_dir):
            if not server_id_str.isdigit():
                continue

            server_index = self.cog.server_manager._get_server_index(server_id_str)
            active_sessions = server_index.get("active_sessions", {})
            if isinstance(active_sessions, list):
                continue

            # Combine regular and legacy freewill for memory loading
            all_sessions = {**active_sessions.get("regular", {}), **active_sessions.get("freewill", {})}

            for ch_id_str, session_data in all_sessions.items():
                try:
                    channel_id = int(ch_id_str)
                    channel = self.cog.bot.get_channel(channel_id)
                    if not channel or not channel.guild: continue

                    owner_id = session_data.get("owner_id")
                    profiles_data = session_data.get("profiles",[])
                    if self._repair_participant_identity(profiles_data):
                        repaired_any = True

                    # An empty cast is a valid session -- the master prompt,
                    # proactivity and compaction settings are still worth restoring,
                    # and the user may simply not have seated anyone yet. Only a
                    # session that is both empty AND unstarted is discarded, and
                    # _save_multi_profile_sessions never writes one of those.
                    if not owner_id or not (profiles_data or session_data.get("started")):
                        continue

                    self.cog.multi_profile_channels[channel_id] = {
                        "profiles": profiles_data,
                        "is_hydrated": False,
                        "last_bot_message_id": None,
                        "owner_id": owner_id,
                        "is_running": False,
                        "task_queue": asyncio.Queue(),
                        "worker_task": None,
                        "turns_since_last_ltm": 0,
                        "session_prompt": session_data.get("session_prompt"),
                        "session_mode": session_data.get("session_mode", "sequential"),
                        "type": "multi",
                        "proactivity": session_data.get("proactivity", {"enabled": False, "chance": 20, "cooldown": 300, "director_model": "off", "director_instructions": "You are an AI Director for a roleplay session. Introduce a sudden event, an environmental change, or a question to spark conversation among the cast. Keep it brief (1-2 sentences)."}),
                        "compaction": session_data.get("compaction", DEFAULT_COMPACTION_CONFIG.copy()),
                        # Absent means CLOSED: a blueprint written before this field
                        # existed keeps the admin-only access it was configured under.
                        "cast_policy": session_data.get("cast_policy", DEFAULT_CAST_POLICY),
                        "started": session_data.get("started", True),
                    }
                except Exception as e:
                    print(f"Unexpected error reloading multi-profile sessions for server {server_id_str}, channel {ch_id_str}: {e}")

        if repaired_any:
            self._save_multi_profile_sessions()

    def _save_multi_profile_sessions(self):
        try:
            current_server_sessions = collections.defaultdict(lambda: {"regular": {}})

            for channel_id, session_data in self.cog.multi_profile_channels.items():
                # A shell opened by `/session config` and abandoned before anyone was
                # seated is not a session, and writing it would put a blueprint in
                # every server index for every channel the editor was ever opened in.
                if not session_data.get("profiles") and not session_data.get("started"):
                    continue

                channel = self.cog.bot.get_channel(channel_id)
                server_id_str = str(channel.guild.id) if channel and getattr(channel, 'guild', None) else "dm"

                category = "regular"

                profiles_to_save =[]
                for p in session_data.get("profiles",[]):
                    # The stamped PID wins. Re-resolving from the name would overwrite a
                    # good PID with a name-shaped one during the window where a rename
                    # has landed in the index but not yet in this cast entry.
                    pid = p.get("pid") or self.cog.profile_manager._get_pid_from_name_any(
                        p["owner_id"], p["profile_name"])
                    profiles_to_save.append({
                        "pid": pid,
                        "profile_name": p["profile_name"],
                        "owner_id": p["owner_id"],
                        "method": p.get("method", "webhook"),
                        "bot_id": p.get("bot_id"),
                        "ephemeral": p.get("ephemeral", False),
                        "chance": p.get("chance", 100),
                        "wakewords": p.get("wakewords",[])
                    })

                blueprint = {
                    "owner_id": session_data.get("owner_id"),
                    "profiles": profiles_to_save,
                    "session_prompt": session_data.get("session_prompt"),
                    "session_mode": session_data.get("session_mode", "sequential"),
                    "type": "multi",
                    "proactivity": session_data.get("proactivity", {"enabled": False, "chance": 10, "cooldown": 300, "director_model": "off", "director_instructions": "You are an AI Director for a roleplay session. Introduce a sudden event, an environmental change, or a question to spark conversation among the cast. Keep it brief (1-2 sentences)."}),
                    "compaction": session_data.get("compaction", DEFAULT_COMPACTION_CONFIG.copy()),
                    "cast_policy": session_data.get("cast_policy", DEFAULT_CAST_POLICY),
                    "started": bool(session_data.get("started", True)),
                }

                current_server_sessions[server_id_str][category][str(channel_id)] = blueprint

            servers_dir = SERVERS_DIR
            if os.path.exists(servers_dir):
                for s_dir in os.listdir(servers_dir):
                    if os.path.isdir(os.path.join(servers_dir, s_dir)) and s_dir not in current_server_sessions:
                        current_server_sessions[s_dir] = {"regular": {}}

            for server_id_str, sessions_for_server in current_server_sessions.items():
                server_index = self.cog.server_manager._get_server_index(server_id_str)
                server_index["active_sessions"] = sessions_for_server
                self.cog.server_manager._save_server_index(server_id_str, server_index)

                # Housekeeping: Delete legacy file if it exists
                old_file = os.path.join(servers_dir, server_id_str, "sessions.json.gz")
                if os.path.exists(old_file):
                    _delete_file_shard(old_file)

        except Exception as e:
            print(f"Error saving sharded multi-profile sessions to index: {e}")

    def _get_user_profile_for_model(self, user_id: int, channel_id: int, profile_name_override: Optional[str] = None) -> Tuple[Dict[str, List[str]], str, bool, float, float, int, int, float, str, str]:
        active_profile_name = profile_name_override if profile_name_override else self._get_active_user_profile_name_for_channel(user_id, channel_id)

        if not active_profile_name:
            return {}, "", False, defaultConfig.GEMINI_TEMPERATURE, defaultConfig.GEMINI_TOP_P, defaultConfig.GEMINI_TOP_K, defaultConfig.TRAINING_CONTEXT_SIZE, defaultConfig.TRAINING_RELEVANCE_THRESHOLD, PRIMARY_MODEL_NAME, FALLBACK_MODEL_NAME

        index = self.cog.profile_manager._get_user_index(user_id)
        is_borrowed = active_profile_name in index.get("borrowed", [])

        config = self.cog.profile_manager._get_profile_config(user_id, active_profile_name, is_borrowed)
        if not config:
            return {}, "", False, defaultConfig.GEMINI_TEMPERATURE, defaultConfig.GEMINI_TOP_P, defaultConfig.GEMINI_TOP_K, defaultConfig.TRAINING_CONTEXT_SIZE, defaultConfig.TRAINING_RELEVANCE_THRESHOLD, PRIMARY_MODEL_NAME, FALLBACK_MODEL_NAME

        source_owner_id, source_profile_name = self.cog.profile_manager._resolve_effective_profile(user_id, active_profile_name)

        prompts = self.cog.profile_manager._get_profile_prompts(source_owner_id, source_profile_name) or {}

        persona = prompts.get("persona", {})
        ai_instructions = prompts.get("ai_instructions", "")

        training_context_size = config.get("training_context_size", defaultConfig.TRAINING_CONTEXT_SIZE)
        training_relevance_threshold = config.get("training_relevance_threshold", defaultConfig.TRAINING_RELEVANCE_THRESHOLD)
        temperature = config.get("temperature", defaultConfig.GEMINI_TEMPERATURE)
        top_p = config.get("top_p", defaultConfig.GEMINI_TOP_P)
        top_k = config.get("top_k", defaultConfig.GEMINI_TOP_K)
        primary_model = config.get("primary_model", PRIMARY_MODEL_NAME)
        fallback_model = config.get("fallback_model", FALLBACK_MODEL_NAME)
        grounding_enabled = config.get("grounding_enabled", False)

        return (persona, ai_instructions, grounding_enabled, float(temperature), float(top_p), int(top_k), int(training_context_size), float(training_relevance_threshold), primary_model, fallback_model)

    def _get_active_user_profile_name_for_channel(self, user_id: int, channel_id: int) -> Optional[str]:
        channel = self.cog.bot.get_channel(channel_id)
        server_id_str = str(channel.guild.id) if channel and getattr(channel, 'guild', None) else "dm"
        server_index = self.cog.server_manager._get_server_index(server_id_str)
        active = server_index.get("user_active_profiles", {}).get(str(user_id), {}).get(str(channel_id))
        if active: return active

        index = self.cog.profile_manager._get_user_index(user_id)
        personal = index.get("personal", [])
        if personal:
            return next(iter(personal))
        borrowed = index.get("borrowed", [])
        if borrowed:
            return next(iter(borrowed))
        return None

    def _get_active_user_profile_data(self, user_id: int, channel_id: int) -> Optional[Dict[str, Any]]:
        active_profile_name = self._get_active_user_profile_name_for_channel(user_id, channel_id)
        if not active_profile_name: return None
        index = self.cog.profile_manager._get_user_index(user_id)

        is_borrowed = active_profile_name in index.get("borrowed", [])
        config = self.cog.profile_manager._get_profile_config(user_id, active_profile_name, is_borrowed)

        if config:
            config.setdefault("grounding_enabled", False)
            config.setdefault("temperature", defaultConfig.GEMINI_TEMPERATURE)
            config.setdefault("top_p", defaultConfig.GEMINI_TOP_P)
            config.setdefault("top_k", defaultConfig.GEMINI_TOP_K)
            config.setdefault("training_context_size", defaultConfig.TRAINING_CONTEXT_SIZE)
            config.setdefault("training_relevance_threshold", defaultConfig.TRAINING_RELEVANCE_THRESHOLD)
            config.setdefault("primary_model", PRIMARY_MODEL_NAME)
            config.setdefault("fallback_model", FALLBACK_MODEL_NAME)
        return config

    async def _set_active_user_profile_for_channel(self, user_id: int, channel_id: int, profile_name: str, interaction_for_feedback=None) -> bool:
        profile_name = profile_name.lower().strip()
        channel = self.cog.bot.get_channel(channel_id)
        server_id_str = str(channel.guild.id) if channel and getattr(channel, 'guild', None) else "dm"

        def _sync_switch():
            index = self.cog.profile_manager._get_user_index(user_id)
            is_borrowed = profile_name in index.get("borrowed", [])

            if not is_borrowed and profile_name not in index.get("personal", []):
                return None

            server_index = self.cog.server_manager._get_server_index(server_id_str)
            old_profile_name = server_index.get("user_active_profiles", {}).get(str(user_id), {}).get(str(channel_id))
            server_index.setdefault("user_active_profiles", {}).setdefault(str(user_id), {})[str(channel_id)] = profile_name
            self.cog.server_manager._save_server_index(server_id_str, server_index)

            effective_owner_id = user_id
            effective_profile_name = profile_name
            active_appearance = None
            if interaction_for_feedback:
                if is_borrowed:
                    b_config = self.cog.profile_manager._get_profile_config(user_id, profile_name, True)
                    if b_config:
                        effective_owner_id = int(b_config.get("original_owner_id", user_id))
                        effective_profile_name = b_config.get("original_profile_name", profile_name)

                owner_config = self.cog.profile_manager._get_profile_config(effective_owner_id, effective_profile_name)
                if owner_config and (owner_config.get("custom_display_name") or owner_config.get("custom_avatar_url")):
                    active_appearance = owner_config

            return old_profile_name, active_appearance

        result = await asyncio.to_thread(_sync_switch)
        if result is None:
            if interaction_for_feedback:
                await interaction_for_feedback.followup.send(f"Your profile '{profile_name}' not found. Cannot activate.", ephemeral=True)
            return False

        old_profile_name, active_appearance = result

        if interaction_for_feedback and interaction_for_feedback.guild:
            warning_key_to_clear = (user_id, interaction_for_feedback.guild.id, old_profile_name)
            self.cog.model_override_warnings_sent.discard(warning_key_to_clear)

        # channel_models is keyed (channel_id, profile_owner_id, profile_name) — see
        # APIService.get_or_create_model. This used to build a 2-tuple, so all three
        # evictions silently matched nothing and a profile switch left the previous
        # profile's model cached. Drop every entry for this (channel, user) pair, which
        # covers both the old and the new profile name.
        stale_keys = [
            k for k in self.cog.channel_models
            if isinstance(k, tuple) and len(k) == 3 and k[0] == channel_id and k[1] == user_id
        ]
        for k in stale_keys:
            self.cog.channel_models.pop(k, None)
            self.cog.channel_model_last_profile_key.pop(k, None)

        if interaction_for_feedback:
            channel_mention = f"<#{channel_id}>" if interaction_for_feedback.guild else "this DM"

            embed_title = f"Your Profile Preference Swapped to: '{profile_name}'"
            embed_desc = f"Your individual preferred profile in {channel_mention} is now '{profile_name}'."

            embed = discord.Embed(title=embed_title, description=embed_desc, color=discord.Color.green())

            app_name = self.cog.bot.user.name if self.cog.bot.user else "Bot"
            app_avatar_url = self.cog.bot.user.display_avatar.url if self.cog.bot.user else None

            if active_appearance:
                app_name = active_appearance.get("custom_display_name") or app_name
                app_avatar_url = active_appearance.get("custom_avatar_url") or app_avatar_url

            embed.add_field(name="Linked Appearance", value=f"Name: {app_name}", inline=False)
            if app_avatar_url:
                embed.set_thumbnail(url=app_avatar_url)

            await interaction_for_feedback.followup.send(embed=embed, ephemeral=True)
        return True

    async def _ensure_session_hydrated(self, channel_id: int, session_type: str) -> Optional[Dict]:
        """Checks if a session is in memory and hydrated. If not, loads it from disk."""
        session = self.cog.multi_profile_channels.get(channel_id)

        # Lazily validate and clean up all session participants if the session is engaged
        if session and session.get("profiles"):
            cleaned_any = False
            valid_profiles = []

            # Once per distinct owner, not once per participant. This scan reads and
            # decrypts every borrowed profile the owner has, so a five-participant
            # session sharing one owner ran that whole sweep five times per
            # hydration, on the event loop.
            for owner_id in {p["owner_id"] for p in session["profiles"]}:
                await self.cog.profile_manager._validate_and_clean_borrowed_profiles(owner_id)

            for p in list(session["profiles"]):
                p_owner_id = p["owner_id"]
                p_name = p["profile_name"]

                # Verify if the profile continues to exist for the owner
                p_index = self.cog.profile_manager._get_user_index(p_owner_id)
                exists = (p_name in p_index.get("personal", {})) or (p_name in p_index.get("borrowed", {})) or (p_name in p_index.get("system", {}))

                if exists:
                    valid_profiles.append(p)
                else:
                    cleaned_any = True
                    if p.get("method") == "child_bot" and p.get("bot_id"):
                        await self.cog.manager_queue.put({
                            "action": "send_to_child", "bot_id": p["bot_id"],
                            "payload": {"action": "session_update_remove", "channel_id": channel_id}
                        })

            if cleaned_any:
                session["profiles"] = valid_profiles
                self._save_multi_profile_sessions()
                if not session["profiles"]:
                    self.cog.multi_profile_channels.pop(channel_id, None)
                    dummy_session_key = (channel_id, None, None)
                    await self._delete_session_from_disk(dummy_session_key, session_type)
                    return None

        if session and session.get("is_hydrated"):
            return session

        # Resolve Guild ID for pathing
        guild_id_str = None
        channel = self.cog.bot.get_channel(channel_id)
        if channel and hasattr(channel, 'guild'):
            guild_id_str = str(channel.guild.id)
        else:
            for guild in self.cog.bot.guilds:
                if guild.get_channel(channel_id):
                    guild_id_str = str(guild.id); break

        if not guild_id_str: return None

        if not session or not session.get("profiles"):
            server_index = self.cog.server_manager._get_server_index(guild_id_str)
            active_sessions = server_index.get("active_sessions", {})
            if isinstance(active_sessions, dict):
                session_config = active_sessions.get("regular", {}).get(str(channel_id))
                if not session_config:
                    session_config = active_sessions.get("freewill", {}).get(str(channel_id))

                if session_config:
                    profiles = session_config.get("profiles", [])
                    for p in profiles: p.setdefault('ltm_counter', 0)
                    # Same repair as the boot path: this branch restores a blueprint
                    # that has been on disk, so its names are exactly as stale.
                    self._repair_participant_identity(profiles)

                    if not session:
                        session = {
                            "profiles": profiles,
                            "owner_id": session_config.get("owner_id"),
                            "session_prompt": session_config.get("session_prompt"),
                            "session_mode": session_config.get("session_mode", "sequential"),
                            "type": "multi",
                            "proactivity": session_config.get("proactivity", {"enabled": False, "chance": 10, "cooldown": 300, "director_model": "off", "director_instructions": "You are an AI Director for a roleplay session. Introduce a sudden event, an environmental change, or a question to spark conversation among the cast. Keep it brief (1-2 sentences)."}),
                            "compaction": session_config.get("compaction", DEFAULT_COMPACTION_CONFIG.copy()),
                            "cast_policy": session_config.get("cast_policy", DEFAULT_CAST_POLICY),
                            "task_queue": asyncio.Queue(),
                            "is_running": False,
                            "started": session_config.get("started", True),
                        }
                        self.cog.multi_profile_channels[channel_id] = session
                    else:
                        # Update the existing dehydrated shell with index data
                        session["profiles"] = profiles
                        session["owner_id"] = session_config.get("owner_id")
                        session["session_prompt"] = session_config.get("session_prompt")
                        session["session_mode"] = session_config.get("session_mode", "sequential")
                        session["type"] = "multi"
                        session["proactivity"] = session_config.get("proactivity", {"enabled": False, "chance": 10, "cooldown": 300, "director_model": "off", "director_instructions": "You are an AI Director for a roleplay session. Introduce a sudden event, an environmental change, or a question to spark conversation among the cast. Keep it brief (1-2 sentences)."})
                        session["compaction"] = session_config.get("compaction", DEFAULT_COMPACTION_CONFIG.copy())
                        session["cast_policy"] = session_config.get("cast_policy", DEFAULT_CAST_POLICY)

        if not session: return None

        # 2. Load History Log
        dummy_session_key = (channel_id, None, None)
        disk_log, disk_cold_len = await self._load_session_segments(dummy_session_key, session_type)
        disk_log = disk_log or []

        current_mem_log = session.get("unified_log", [])
        if not disk_log and current_mem_log:
            session["unified_log"] = current_mem_log
            await self._save_session_to_disk(dummy_session_key, session_type, current_mem_log)
            unified_log = current_mem_log
        elif current_mem_log and len(current_mem_log) >= len(disk_log):
            unified_log = current_mem_log
            # The in-memory log won, so the segment boundary just read off disk does not
            # describe it. Zero forces the next flush to be a full one, which re-seals.
            session["_log_cold_len"] = 0
        else:
            unified_log = disk_log
            session["_log_cold_len"] = disk_cold_len

        session["unified_log"] = unified_log

        # 3. Synchronise Memory for all current participants
        self._recompute_pending_whispers(session, force=True)

        session["is_hydrated"] = True
        return session

    def _recompute_pending_whispers(self, session: Dict, force: bool = False) -> None:
        """Re-derives session['pending_whispers'] from unified_log, in place.

        This is the *only* state a session rebuild derives. Everything else in
        _ensure_session_hydrated is either unconditional (the participant
        validation above the is_hydrated check) or dead for an engaged session
        (the server-index branch, gated on an empty profile list). So the paths
        that mutate unified_log and then flush it call this directly, instead of
        forcing an is_hydrated=False + rehydrate round-trip. That round-trip
        re-read, decrypted, decompressed and re-parsed the whole session log to
        arrive back at the log it already held correctly in memory -- measured at
        2.4 ms and ~15 MB of immediately-discarded allocation for a 1000-turn log,
        GIL-held inside orjson -- and its "no disk log, but a memory log" branch
        re-wrote session files the caller had just deleted.

        No-op while a round is in flight, on purpose. _multi_profile_worker pops a
        participant's whispers when it builds that participant's prompt, but the
        bot's own turn -- which this derivation reads as the signal that a whisper
        has been answered -- does not reach unified_log until the response lands.
        Recomputing inside that window marks consumed whispers pending again and
        re-injects them on the participant's next turn. Deferring is safe, and is
        also what recovers them: a round cancelled mid-flight never writes its
        turn, so the whispers stay derivable from the log and come back on the
        next rebuild. Hydration passes force=True -- it is establishing initial
        state, and at that point nothing has been popped.
        """
        if not force and (session.get('is_running') or session.get('is_regenerating')):
            return

        unified_log = session.get("unified_log", [])
        pending_map = {}

        for p_data in session.get("profiles", []):
            p_key = (p_data['owner_id'], p_data['profile_name'])
            bot_pid = self.cog.profile_manager._get_pid_from_name_any(p_data['owner_id'], p_data['profile_name'])

            pending = self._get_pending_whispers_for_participant(unified_log, bot_pid)
            if pending:
                pending_map[p_key] = pending

        session["pending_whispers"] = pending_map

    async def _wait_for_session_flags(self, session: Dict, flags: Tuple[str, ...], timeout: float) -> bool:
        """Spin until none of `flags` is set on the session. True if it cleared, False on timeout.

        Bounded on purpose. Every one of these flags is cleared in a finally, but a worker
        that dies in the wrong place still leaks one, and an unbounded spin then blocks the
        channel for the life of the process -- which is exactly what /purge's wait used to
        do (see PURGE_BUSY_WAIT_TIMEOUT_SECONDS). A caller that times out must decide for
        itself whether to proceed or abandon; this only reports which happened.

        A caller claiming the channel must assign its own flag *synchronously* on the True
        return, with no await in between. The event loop is single-threaded and this returns
        without yielding, so check-then-set is atomic only for as long as that holds.
        """
        deadline = time.monotonic() + timeout
        while any(session.get(flag) for flag in flags):
            if time.monotonic() > deadline:
                return False
            await asyncio.sleep(0.5)
        return True

    def _mark_session_accessed(self, key):
        ts = time.time()
        self.cog.session_last_accessed[key] = ts
        heapq.heappush(self.cog.eviction_heap, (ts, key))

    async def _evict_inactive_sessions(self):
        now = time.time()
        inactive_threshold = 600  # 10 minute strict dehydration policy

        keys_to_evict = set()

        # O(1) Heap Peek: Only evaluate sessions that have officially expired
        while self.cog.eviction_heap and now - self.cog.eviction_heap[0][0] > inactive_threshold:
            ts, key = heapq.heappop(self.cog.eviction_heap)

            # Lazy Deletion: Ensure this wasn't updated recently
            if self.cog.session_last_accessed.get(key) == ts:
                if isinstance(key, int):
                    session = self.cog.multi_profile_channels.get(key)
                    # whisper_waiting counts as busy as well: a queued whisper holds a
                    # reference to this session dict and reads its unified_log the moment it
                    # claims the channel, and dehydrating underneath it would have the
                    # profile answer with no history at all.
                    if session and (session.get('is_running') or session.get('is_regenerating')
                                    or session.get('is_purging') or session.get('is_whispering')
                                    or session.get('is_memorising') or session.get('whisper_waiting')):
                        continue
                keys_to_evict.add(key)

        for key in keys_to_evict:
            # Handle multi-profile sessions (key is channel_id int)
            if isinstance(key, int):
                session_to_evict = self.cog.multi_profile_channels.get(key)
                if session_to_evict and session_to_evict.get("is_hydrated"):
                    session_type = session_to_evict.get("type", "multi")
                    unified_log = session_to_evict.get("unified_log")

                    if unified_log is not None:
                        dummy_session_key = (key, None, None)
                        # Full flush, not a tail one: the log is about to leave memory,
                        # so the cold segment has to be complete on its own.
                        self.cog.dirty_sessions.pop(dummy_session_key, None)
                        await self._save_session_to_disk(dummy_session_key, session_type, unified_log)

                    # Use safe cancel to prevent "Task destroyed but pending"
                    if session_to_evict.get('worker_task'):
                        self._safe_cancel_task(session_to_evict['worker_task'])
                        session_to_evict['worker_task'] = None

                    # Aggressively dehydrate the session to release memory.
                    session_to_evict['is_hydrated'] = False

                    # Clear the large data structures from the in-memory dictionary.
                    if 'unified_log' in session_to_evict:
                        del session_to_evict['unified_log']
                    # Meaningless without the log it indexes, and stale if the session
                    # rehydrates from a file another process has since rewritten.
                    session_to_evict.pop('_log_cold_len', None)
                    # Both are re-derived on the next hydration (pending_whispers from
                    # the log, cancellations only matter for a queued trigger), and
                    # dropping the cancellation set is what bounds it for a session
                    # that is abandoned before its worker next drains the queue.
                    session_to_evict.pop('cancelled_reaction_triggers', None)
                    session_to_evict.pop('pending_whispers', None)
                    # Clean up LTM recall history for participants in this channel to prevent RAM leak
                    for p in session_to_evict.get("profiles", []):
                        full_key = (key, p.get("owner_id"), p.get("profile_name"))
                        self.cog.ltm_recall_history.pop(full_key, None)

                    # If it's a completely empty dormant session, pop it entirely to save memory.
                    if not session_to_evict.get("profiles"):
                        self.cog.multi_profile_channels.pop(key, None)

            # Handle global chat sessions (key is a tuple)
            elif isinstance(key, tuple):
                session_to_save = self.cog.global_chat_sessions.get(key)
                if session_to_save:
                    self.cog.dirty_sessions.pop(key, None)
                    await self._save_session_to_disk(key, 'global_chat', session_to_save)

                self.cog.global_chat_sessions.pop(key, None)
                self.cog.ltm_recall_history.pop(key, None)

            # Remove from tracking dict after processing
            self.cog.session_last_accessed.pop(key, None)

        # Prune stale child bot edit cooldown timestamps
        for bot_id in list(self.cog.child_bot_edit_cooldowns.keys()):
            timestamps = [ts for ts in self.cog.child_bot_edit_cooldowns[bot_id] if now - ts < 600]
            if timestamps:
                self.cog.child_bot_edit_cooldowns[bot_id] = timestamps
            else:
                self.cog.child_bot_edit_cooldowns.pop(bot_id, None)

    @tasks.loop(seconds=60.0)
    async def proactive_session_task(self):
        if not self.cog.has_lock: return
        now = time.time()
        for channel_id, session in list(self.cog.multi_profile_channels.items()):
            pro = session.get("proactivity", {})
            if not pro.get("enabled"): continue
            if not self.is_started(session): continue

            # The Director exists to break a silence. A live game means the channel is
            # anything but silent, and with the cast seated the round it queues would
            # find nobody to speak -- so it would spend a director model call to
            # produce nothing. Skip the channel until the table clears.
            game_service = getattr(self.cog, "game_service", None)
            if game_service and game_service.has_live_game(channel_id): continue
            
            last_event = session.setdefault("last_proactive_event", 0)
            cooldown = pro.get("cooldown", 300)
            if now - last_event < cooldown: continue
            
            chance = pro.get("chance", 10) / 100.0
            if random.random() > chance: continue
            
            session["last_proactive_event"] = now
            
            await session['task_queue'].put(('proactive_trigger', None))
            if not session.get('is_running') and (not session.get('worker_task') or session['worker_task'].done()):
                task = self.cog.bot.loop.create_task(self.cog.generation_service._multi_profile_worker(channel_id))
                session['worker_task'] = task
                self.cog.background_tasks.add(task)

    @tasks.loop(seconds=60.0)
    async def evict_inactive_sessions_task(self):
        await self._evict_inactive_sessions()

    def _get_pending_whispers_for_participant(self, log_list: List[Dict], bot_pid: str) -> List[str]:
        """Whispers aimed at bot_pid that it has not spoken since.

        Scanned backwards and stopped at the participant's own last ordinary turn: that
        turn is what clears the list, so nothing before it can contribute. The forward
        version walked the entire log to build a list it then discarded most of, once
        per participant -- 0.47 ms per hydration for a six-cast, 1000-turn session,
        against a handful of turns here.
        """
        pending = []
        for turn in reversed(log_list):
            if turn.get("is_hidden", False): continue
            turn_type = turn.get("type")
            if not turn_type:
                if turn.get("speaker_pid") == bot_pid:
                    break
            elif turn_type == "whisper":
                if turn.get("target_pid") == bot_pid:
                    pending.append(turn.get("content"))
        pending.reverse()
        return pending

    @staticmethod
    def get_latest_synopsis(session: Optional[Dict]) -> Optional[str]:
        """The most recent rolling synopsis for a session, or None.

        Read by _construct_system_instructions on every turn, so it walks backwards and
        stops at the first hit rather than scanning the log.
        """
        if not session:
            return None
        for turn in reversed(session.get("unified_log") or []):
            if turn.get("type") == "synopsis":
                return (turn.get("content") or "").strip() or None
        return None

    @staticmethod
    def _select_history_window(full_log: List[Dict], window: int) -> List[Dict]:
        """The trailing slice of `full_log` holding `window` turns that will actually
        be shown.

        Counting raw turns instead would let muted and compacted turns eat the window:
        `full_log[-40:]` on a log whose oldest 25 turns were compacted yields 40 turns
        of which only 35 survive the emit loop -- so compacting a session, which exists
        to give a profile *more* usable context, silently gave it less.

        Walks backwards and stops as soon as the quota is met, so it costs O(window)
        plus whatever hidden turns it steps over, not O(log).
        """
        if window <= 0:
            return []
        kept = 0
        for index in range(len(full_log) - 1, -1, -1):
            turn = full_log[index]
            if turn.get("is_hidden") or turn.get("compacted") or turn.get("type") == "synopsis":
                continue
            kept += 1
            if kept >= window:
                return full_log[index:]
        return list(full_log)

    def _build_history_for_participant(self, full_log: List[Dict], bot_pid: str, p_settings: Dict[str, Any], num_participants: int = 1, reserved_tail: int = 0) -> List[Dict]:
        """This participant's view of the log: the last `stm_length` turns, with other
        participants' private exchanges hidden and its own rewritten into their XML tags.

        `reserved_tail` exempts that many trailing turns from the window -- the turns a
        round has just added, which the participant is being asked to respond to. STM
        governs how far back it remembers, not how much of the present it is allowed to
        see, so a busy round must not push the conversation out of its own prompt.

        Callers used to express that by pre-slicing `past[-stm:] + batch` and passing the
        result here, which then re-sliced it to `stm` and dropped the oldest `len(batch)`
        turns of history to make room -- so a profile's memory silently shortened in
        proportion to how many people spoke at once. Windowing lives here now; pass the
        whole log and say how much of its tail is the current round.
        """
        stm_length = int(p_settings.get("stm_length", defaultConfig.CHATBOT_MEMORY_LENGTH))
        effective_stm = max(stm_length, num_participants) if stm_length > 0 else 0
        # stm_length 0 means "no memory", but the current round is still in front of it.
        window = effective_stm + max(0, reserved_tail)
        log_slice = self._select_history_window(full_log, window)

        participant_history = []
        for turn in log_slice:
            if turn.get("is_hidden") or turn.get("compacted"): continue

            turn_type = turn.get("type")
            # Not conversation. A synopsis is standing context injected into the system
            # instruction by _construct_system_instructions, for the same reason
            # <archive_context> is: it summarises turns that have left the window, so
            # placing it *in* the window means a session with more live turns than
            # stm_length never sees it -- which is every session compaction is for.
            if turn_type == "synopsis": continue
            if not turn_type:
                role = 'model' if turn.get("speaker_pid") == bot_pid else 'user'
                parts = [turn.get("content")]
                
                if role == 'user':
                    if turn.get("url_context") and p_settings.get("url_fetching_enabled", False):
                        parts.append(f"\n<document_context>\n{turn.get('url_context')}\n</document_context>")
                    if turn.get("grounding_context") and p_settings.get("grounding_mode", "off") != "off":
                        parts.append(f"\n{turn.get('grounding_context')}")
                
                if participant_history and participant_history[-1]['role'] == role:
                    participant_history[-1]['parts'].extend(parts)
                else:
                    content_obj = {'role': role, 'parts': parts}
                    participant_history.append(content_obj)
                    
            elif turn_type == "whisper":
                if turn.get("target_pid") == bot_pid:
                    clean_content = turn.get("content")
                    header, body = clean_content.split('\n', 1) if '\n' in clean_content else ("", clean_content)
                    wrapped = f"{header}\n<private_whisper>\n{body.strip()}\n</private_whisper>\n"
                    if participant_history and participant_history[-1]['role'] == 'user':
                        participant_history[-1]['parts'].append(wrapped)
                    else:
                        participant_history.append({'role': 'user', 'parts': [wrapped]})
                        
            elif turn_type == "private_response":
                if turn.get("speaker_pid") == bot_pid:
                    clean_content = turn.get("content")
                    header, body = clean_content.split('\n', 1) if '\n' in clean_content else ("", clean_content)
                    wrapped = f"{header}\n<private_response>\n{body.strip()}\n</private_response>\n"
                    if participant_history and participant_history[-1]['role'] == 'model':
                        participant_history[-1]['parts'].append(wrapped)
                    else:
                        obj = {'role': 'model', 'parts': [wrapped]}
                        participant_history.append(obj)
                        
        return participant_history

    async def setup_multi_profile_session(self, interaction: discord.Interaction, participants: List[Dict], session_prompt: Optional[str], session_mode: str, as_admin_scope: bool = False, audio_mode: str = "off"):
        user_id = interaction.user.id
        is_update = interaction.channel_id in self.cog.multi_profile_channels

        # Verify each participant's content rating is still current before the
        # session runs. Session setup is user-initiated and infrequent, which is why
        # the staleness check lives here and not in the per-turn gate -- it decrypts
        # the whole persona to hash it. Fire-and-forget: a stale verdict only ever
        # errs toward the previous, stricter answer while the recheck lands.
        for _p in {(p["owner_id"], p["profile_name"]) for p in participants}:
            asyncio.create_task(
                self.cog.profile_manager.resolve_stale_rating(_p[0], _p[1]))

        if is_update:
            session = self.cog.multi_profile_channels[interaction.channel_id]
            if not session.get("is_hydrated"):
                session = await self._ensure_session_hydrated(interaction.channel_id, session.get("type", "multi"))

        else:
            for p in participants:
                p['ltm_counter'] = 0

            session = {
                "type": "multi",
                "unified_log": [],
                "is_hydrated": False,
                "last_bot_message_id": None,
                "owner_id": interaction.user.id,
                "is_running": False,
                "task_queue": asyncio.Queue(),
                "worker_task": None,
                "turns_since_last_ltm": 0,
                "session_prompt": None,
                "session_mode": "sequential",
                "pending_image_gen_data": None,
                "pending_whispers": {},
                "audio_mode": "off",
                "compaction": DEFAULT_COMPACTION_CONFIG.copy(),
                "cast_policy": DEFAULT_CAST_POLICY,
                "started": True,
            }
            self.cog.multi_profile_channels[interaction.channel_id] = session

        session["type"] = "multi"
        session["session_prompt"] = session_prompt
        session["profiles"] = participants
        session["session_mode"] = session_mode
        session["audio_mode"] = audio_mode
        session["started"] = True
        
        for p_data in participants:
            if p_data.get('method') == 'child_bot':
                await self.cog.manager_queue.put({
                    "action": "send_to_child", "bot_id": p_data['bot_id'],
                    "payload": {"action": "session_update_add", "channel_id": interaction.channel_id}
                })
        
        self._save_multi_profile_sessions()

        profile_list_str = []
        for p_data in participants:
            if p_data.get('method') == 'child_bot':
                bot_user = self.cog.bot.get_user(int(p_data['bot_id']))
                profile_list_str.append(f"`{bot_user.name if bot_user else 'Unknown Bot'}`")
            else:
                profile_list_str.append(f"`{p_data['profile_name']}`")

        action_str = "updated" if is_update else "activated"
        msg = f"Regular session {action_str} with participants: {', '.join(profile_list_str)}."
        if as_admin_scope:
            msg = f"Regular session is now active for all users with profiles: {', '.join(profile_list_str)}."
        
        await interaction.edit_original_response(content=msg, view=None)

    @staticmethod
    def is_started(session: Optional[Dict]) -> bool:
        """Whether this channel's session has actually been started.

        Seating a profile is not the same as starting the session it sits in. The cast
        dropdown applies and saves the moment a name is chosen -- it has to, or the
        Reactivity tab would have nobody to edit -- so `Start / Update Session` is what
        makes the channel live. Every path that would run a round asks this first.

        Absent means started, and that is load-bearing: only the two paths that
        deliberately seat without starting (`_ensure_session_shell` and a `/session
        swap` that creates the session) write False. A blueprint saved before this flag
        existed, and every session an older code path built, keeps running untouched.
        """
        return bool(session) and bool(session.get("started", True))

    @staticmethod
    def register_in_flight(session: Dict, state_container: Dict) -> None:
        """Publish a live operation's heartbeat state so /cancel can see its phase.

        The container is what knows whether generation has returned: `sending_task` is
        set by _update_sending_placeholder and cleared by _stop_sending_heartbeat, so
        its presence is exactly the window the placeholder reads "Sending...".
        """
        session.setdefault("in_flight", []).append(state_container)

    @staticmethod
    def release_in_flight(session: Dict, state_container: Dict) -> None:
        containers = session.get("in_flight")
        if not containers:
            return
        # Identity, not equality: two participants in a round can hold containers that
        # compare equal while only one of them is finishing.
        session["in_flight"] = [c for c in containers if c is not state_container]

    @staticmethod
    def is_delivering(session: Dict) -> bool:
        """True once a model has returned and the turn is being built and sent.

        Cancelling here is the one case that cannot be made safe: the response exists,
        the placeholder is mid-edit, TTS may be part-way through a synthesis, and for a
        regeneration the original message has already been overwritten. Everything
        before this point either has nothing to undo or can be put back.
        """
        return any(c.get("sending_task") for c in session.get("in_flight", ()))

    def _safe_cancel_task(self, task: asyncio.Task):
        if task and not task.done():
            task.cancel()
            self.cog.background_tasks.add(task)
            task.add_done_callback(self.cog.background_tasks.discard)

    async def cancel_channel_operations(self, channel_id: int):
        """Forcefully cancels all active generation, workers, typing, and queued tasks for a channel."""
        session = self.cog.multi_profile_channels.get(channel_id)
        if session:
            session['is_running'] = False
            session['is_regenerating'] = False
            session['is_purging'] = False
            session['is_whispering'] = False
            session['is_memorising'] = False

            # Drain task queue completely
            q = session.get('task_queue')
            if q:
                while not q.empty():
                    try:
                        q.get_nowait()
                        q.task_done()
                    except (asyncio.QueueEmpty, ValueError):
                        break

            # Cancel running session worker task
            worker_task = session.get('worker_task')
            if worker_task and not worker_task.done():
                self._safe_cancel_task(worker_task)
                session['worker_task'] = None

            # Cancel any in-flight regeneration tasks
            for msg_id, r_task in list(session.get('regen_tasks', {}).items()):
                if r_task and not r_task.done():
                    r_task.cancel()
            session.get('regen_tasks', {}).clear()

        # Stop active child bot typing tasks in this channel
        for bot_id in list(self.cog.child_bots.keys()):
            await self.cog.child_bot_manager.stop_typing(bot_id, channel_id)

        # Purge pending image generation requests
        self.cog.media_service._purge_channel_image_requests(channel_id)
