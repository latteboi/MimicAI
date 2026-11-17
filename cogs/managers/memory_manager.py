# Deferred annotation evaluation, so the two `-> np.ndarray` return annotations
# below never touch the lazy proxy at definition time. See the TYPE_CHECKING branch
# under _LazyNumpy for the static half of the same problem.
from __future__ import annotations

import os
import re
import uuid
import glob
import base64
import shutil
import asyncio
import datetime
import threading
import traceback
from collections import Counter, OrderedDict
from typing import TYPE_CHECKING, Callable, Dict, List, Any, NamedTuple, Optional, Sequence, Tuple
import discord


class _LazyNumpy:
    """Imports numpy on first use and gets out of the way.

    numpy costs ~12 MB of resident memory at import, and nothing needs it until
    the bot does memory work -- an LTM/training retrieval, or encoding an
    embedding to store. An instance that boots and sits idle should not be
    carrying it. On the first attribute access this rebinds the module-level
    `np`, so every call site below is unchanged and only the very first one pays
    the indirection.
    """

    __slots__ = ()

    def __getattr__(self, name):
        import numpy

        globals()["np"] = numpy
        return getattr(numpy, name)


if TYPE_CHECKING:
    # Type checkers need the *module* behind `np`. `_LazyNumpy()` binds a variable, and
    # an attribute of a variable ("np.ndarray") is not a valid type expression -- so the
    # two annotations below were reported as errors. Under TYPE_CHECKING the real module
    # is what the checker sees; at runtime only the proxy is ever bound, so an idle
    # instance still never pays numpy's ~12 MB import.
    import numpy as np
else:
    np = _LazyNumpy()


from ..utils.constants import (
    defaultConfig, DEFAULT_SAFETY_SETTINGS,
    MAX_LTM_SUMMARY_CHARACTERS, MIN_HISTORY_FOR_LTM_CREATION, LTM_MAX_PER_CAPTURE,
    DEFAULT_TRAINING_ANALYST_PROMPT, GREEDY_SAMPLING,
    RECALL_TOOL_MAX, RECALL_TOOL_THRESHOLD, LTM_HOLD_REPLIES, LTM_RECALL_SPAN,
    DEFAULT_ARCHIVE_CONTEXT, EMBEDDING_BASELINE_FILE, LTM_IMPORT_MAX_SERVERS,
)
from ..utils.helpers import (TURN_TIME_FORMAT, Timeout, _format_api_error, _resolve_zoneinfo,
                            clip_to_sentence, clean_imported_text, ltm_auto_recall_enabled,
                            resolve_thinking_params, suppress_link_previews, system_model)
from .storage_manager import IOManager, _shard_stamp
from ..services.api.embeddings import embedding_model_key
from ..utils import fixed_examples

# Held by every read-modify-write of an LTM or training shard. Each reads the shard,
# awaits, and saves what it read plus its change, so two at once -- one character forming
# memories in two channels, a delete during an add -- saved over each other: a change
# lost, or a deleted entry back. Anything slow, an embedding call, is done before taking it.
# ponytail: one lock for every profile; per-profile locks if shard writes ever queue.
_SHARD_WRITE_LOCK = asyncio.Lock()
from ..services.api_service import get_embedding_vector
from ..gui.base_components import text_embed


def encode_embedding_b64(embedding: List[float]) -> str:
    if not embedding: return ""
    return base64.b64encode(np.array(embedding, dtype=np.float16).tobytes()).decode('ascii')


def _training_entry(said: str, reply: str, embedding: List[float], model: str) -> Dict[str, str]:
    """One stored training example: `said` is what retrieval matches, embedded by the caller
    on `model` -- see `_build_vectors` for why every vector names its model."""
    now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return {"id": str(uuid.uuid4())[:8], "created_ts": now_ts, "modified_ts": now_ts,
            "u_in": said.strip(), "b_out": reply.strip(), "u_emb_b64": encode_embedding_b64(embedding),
            "emb_model": model}


def _ltm_entry(summary: str, embedding_b64: str, guild_id: int, model: str,
               user_dn: Optional[str] = None, source: Optional[str] = None) -> Dict[str, str]:
    """One stored memory. `context_id` repeats the server its file is named for, so a file
    read on its own -- an export, a dump -- still says where each memory belongs."""
    entry = {
        # 64 random bits: delete_ltms removes every entry sharing an id, and 32 bits
        # collided in ~0.3% of profiles at LIMIT_LTM. Older 8-char ids stay valid.
        "id": os.urandom(8).hex(),
        "created_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sum": summary.strip(),
        "s_emb_b64": embedding_b64,
        "emb_model": model,
        "context_id": str(guild_id),
    }
    if user_dn:
        entry["usr"] = user_dn
    if source:
        entry["src"] = source
    return entry


#: How a summariser declines, in the forms it actually declines in. The prompt asks for
#: NO_SUMMARY and nothing else; models add a full stop, bold it, write it with a space,
#: or answer in plain words. Anchored, and the literal is the first thing required.
PATTERN_LTM_ABSTAIN = re.compile(r'^\W*no[\s_-]*summary\b', re.IGNORECASE)


def _is_abstention(text: str) -> bool:
    """Whether the summariser declined to write a memory.

    Matched against the start of the reply rather than the whole of it: "NO_SUMMARY --
    nothing here was worth keeping" is a decline, and the exact-equality test this
    replaces stored that sentence as a memory, embedded it, and injected it into every
    later turn. No real memory can open this way, so a prefix test costs nothing.
    """
    return bool(PATTERN_LTM_ABSTAIN.match((text or "").strip()))


def entry_date(entry: Dict[str, Any], tz, fmt: str = "%d %b %Y") -> str:
    """When a memory or training example was made, on `tz`'s clock, or "" when it carries no date.

    The day by default: recall injects that into every prompt, where a time costs tokens
    and invites a character to quote it.
    """
    stamp = entry.get("created_ts") or entry.get("ts")
    try:
        moment = datetime.datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return ""
    if not moment.tzinfo:
        moment = moment.replace(tzinfo=datetime.timezone.utc)
    return moment.astimezone(tz).strftime(fmt)


#: A list marker a model puts on a line despite being asked not to.
_LIST_MARKER = re.compile(r'^\s*(?:[-*\u2022]|\d+[.)])\s+')


def split_memories(text: str) -> List[str]:
    """The memories in a summariser reply: one per line, list markers and declines
    dropped, each clipped, at most LTM_MAX_PER_CAPTURE.

    Clipped one by one because the prompt's "two sentences" is a request, and a memory
    that runs on embeds as the average of everything in it.
    """
    memories = []
    for line in (text or "").splitlines():
        line = _LIST_MARKER.sub("", line).strip()
        if line and not _is_abstention(line):
            memories.append(clip_to_sentence(line, MAX_LTM_SUMMARY_CHARACTERS))
    return memories[:LTM_MAX_PER_CAPTURE]

def decode_embedding_b64(b64_str: str) -> np.ndarray:
    if not b64_str: return np.array([], dtype=np.float32)
    return np.frombuffer(base64.b64decode(b64_str), dtype=np.float16).astype(np.float32)


#: The two ways an embedding can fail, kept apart because they need different
#: actions from the user. They used to share one "embedding failed" line, which is
#: why a missing key read as a broken feature: the commonest cause by far is running
#: /profile in a DM, where there is no guild whose key could be billed.
NO_EMBEDDING_KEY_MSG = (
    "No usable API key for this action. Add a personal Gemini or OpenRouter key in "
    "`/settings` -> API Keys, or run this from a server that has one assigned."
)
EMBEDDING_FAILED_MSG = (
    "The embedding request did not come back (the API may be busy or the key "
    "rejected). Nothing was changed -- try again in a moment."
)


# --- Vector cache --------------------------------------------------------------
#
# What retrieval needs from a shard -- its rows, pre-normalised so a query is one BLAS
# `unit @ prompt`, and the few fields of each entry a recall shows -- built once per file
# and kept until the file changes. Long-term memories (one file per server) and training
# examples (one file per profile) share it; help mode keeps its own matrix.
#
# A recall used to read, decrypt and parse the whole shard every turn even when its rows
# were cached, because the winners' text lived only in the file: past the shard cache's
# per-entry ceiling, a disk read and a GIL-holding parse per participant per turn.
# Holding the entries beside the rows makes a hit one stat.
#
# Entries and rows come from one read and are stored as one object, under the stamp
# taken *before* that read. A write landing in between leaves newer content under the
# older stamp, which the next stat misses: one wasted build, never rows paired with a
# list they were not built from. The cache this replaces took its stamp after an awaited
# embedding, so a delete in that window stored the old rows under the new file, and the
# next recall indexed past the end of the list.
#
# Read from worker threads, several at once (one per channel mid-turn), hence the lock
# around every dict operation; building happens outside it. Bounded by total rows:
# 20000 at 256 float32 dims is ~20 MB plus the entries' text, the ceiling the two caches
# this replaces had between them.

_VEC_CACHE_MAX_ROWS = 20000
_vec_cache: "OrderedDict[str, Tuple[Any, Optional[_Vectors]]]" = OrderedDict()
_vec_cache_rows = 0
_vec_cache_lock = threading.Lock()

#: The fields a recall shows, by kind. Everything else in an entry stays on disk.
_LTM_FIELDS = ("id", "sum", "created_ts", "ts")
_TRAINING_FIELDS = ("id", "u_in", "b_out")


class _Vectors:
    """One shard as retrieval sees it: `unit[i]` is the pre-normalised row of `entries[i]`.

    Only entries with a usable vector are here, slimmed to the fields a recall shows.
    Shared between threads and callers, so nothing may mutate it.
    """

    __slots__ = ("entries", "unit", "id_to_row", "n")

    def __init__(self, entries, unit):
        self.entries = entries
        self.unit = unit
        self.id_to_row = {e["id"]: i for i, e in enumerate(entries) if e.get("id")}
        self.n = len(entries)


def _build_vectors(entries, emb_key: str, fields: Sequence[str], model: str,
                   baseline: str) -> Optional[_Vectors]:
    """The `_Vectors` of `entries`, or None when none has a usable vector.

    Usable is: made by `model` -- its `emb_model`, or `baseline` for an entry from before
    the field -- and as wide as most rows. Another model's vector is not a weak match, it
    is a different space; and one row of the wrong width made `reshape` raise for the
    whole profile, on every turn.
    """
    kept = []
    for e in entries or ():
        if not isinstance(e, dict) or e.get("emb_model", baseline) != model:
            continue
        try:
            raw = base64.b64decode(e.get(emb_key) or "")
        except (TypeError, ValueError):
            continue
        if raw and len(raw) % 2 == 0:
            kept.append((e, raw))
    if not kept:
        return None
    width = Counter(len(raw) for _, raw in kept).most_common(1)[0][0]
    kept = [(e, raw) for e, raw in kept if len(raw) == width]
    matrix = np.frombuffer(b"".join(raw for _, raw in kept), dtype=np.float16)
    matrix = matrix.reshape(len(kept), -1).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    # A zero row divided by 1.0 stays zero, so it scores 0 against anything.
    unit = matrix / np.where(norms == 0.0, 1.0, norms)[:, None]
    return _Vectors([{k: e[k] for k in fields if k in e} for e, _ in kept], unit)


def _cached_vectors(path: Optional[str], load: Callable[[], Any], emb_key: str,
                    fields: Sequence[str], model: str, baseline: str) -> Optional[_Vectors]:
    """`path`'s vectors from the cache, else built from `load()` and kept. Blocking.

    No `path`, or no file there, builds without keeping: a stamp that cannot be taken
    cannot invalidate anything.
    """
    global _vec_cache_rows
    stamp = _shard_stamp(path) if path else None
    key = (stamp, model, baseline)
    if stamp is not None:
        with _vec_cache_lock:
            hit = _vec_cache.get(path)
            if hit is not None and hit[0] == key:
                _vec_cache.move_to_end(path)
                return hit[1]
    vectors = _build_vectors(load(), emb_key, fields, model, baseline)
    if stamp is None:
        return vectors
    with _vec_cache_lock:
        old = _vec_cache.pop(path, None)
        if old is not None:
            _vec_cache_rows -= old[1].n if old[1] else 1
        # A file with nothing usable is kept too, as one row: after a model change every
        # old file is one, and would otherwise be read and parsed on every turn.
        _vec_cache[path] = (key, vectors)
        _vec_cache_rows += vectors.n if vectors else 1
        while _vec_cache_rows > _VEC_CACHE_MAX_ROWS and len(_vec_cache) > 1:
            _, (_, dropped) = _vec_cache.popitem(last=False)
            _vec_cache_rows -= dropped.n if dropped else 1
    return vectors


def _unit_query(unit, query) -> Optional[np.ndarray]:
    """`query` at unit length, or None when it is zero or not as wide as `unit`'s rows."""
    q = np.asarray(query, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    return q / norm if norm and q.shape == (unit.shape[1],) else None


def _closest_rows(unit, query, top: int) -> List[Tuple[float, int]]:
    """(similarity, row) for the `top` rows of `unit` nearest `query`, best first."""
    q = _unit_query(unit, query)
    if q is None:
        return []
    sims = unit @ q
    best = np.argsort(-sims, kind="stable")[:top]
    return [(float(sims[b]), int(b)) for b in best]


# --- Long-term memory files ----------------------------------------------------
#
# One file per server, `profiles/<pid>/ltm/<guild_id>.json.gz`, holding a list. A memory
# is only ever recalled in the server it formed in, so that is the only file a turn
# reads, and LIMIT_LTM applies to each server on its own: a busy server no longer
# trims a quiet one's oldest memories. A single `ltm.json.gz` of every server is the
# layout before v0.7.12, which `migrate_ltm_files` splits at boot.

#: Length of the ".json.gz" suffix.
_GZ = len(".json.gz")


def ltm_files(p_dir: str) -> List[Tuple[str, str]]:
    """(guild id, path) for each memory file in a profile folder."""
    ltm_dir = os.path.join(p_dir, "ltm")
    try:
        names = os.listdir(ltm_dir)
    except OSError:
        return []
    return [(name[:-_GZ], os.path.join(ltm_dir, name)) for name in sorted(names)
            if name.endswith(".json.gz") and name[:-_GZ].isdigit()]


def read_ltm_dir(p_dir: str, fernet) -> List[Dict[str, Any]]:
    """Every memory in a profile folder, every server, oldest first: what an export holds."""
    entries = []
    for _gid, path in ltm_files(p_dir):
        data = IOManager.read_json_gzip(path, fernet)
        entries += [e for e in data if isinstance(e, dict)] if isinstance(data, list) else []
    entries.sort(key=lambda e: e.get("created_ts") or e.get("ts") or "")
    return entries


#: The keys an imported memory may carry; anything else in the file is dropped.
_LTM_KEYS = ("id", "created_ts", "modified_ts", "ts", "sum", "s_emb_b64", "emb_model",
             "context_id", "usr", "src")


def write_imported_ltm(p_dir: str, entries: Any, fernet) -> None:
    """An imported profile's memories, filed one server a file. Blocking.

    A file someone else wrote: text is cleaned at the editor's cap, unknown keys go, a
    server id must be digits (it names a file), each server keeps its newest LIMIT_LTM,
    and only the LTM_IMPORT_MAX_SERVERS servers with the newest memories are kept.
    """
    by_guild: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries if isinstance(entries, list) else ():
        if not isinstance(e, dict):
            continue
        gid = str(e.get("context_id") or "")
        text = clean_imported_text(e.get("sum"), 2000)
        if not gid.isdigit() or not text.strip():
            continue
        # An id is how one is held, edited or deleted, so a memory without one is given one.
        by_guild.setdefault(gid, []).append({"id": os.urandom(8).hex(), **{k: e[k] for k in _LTM_KEYS if k in e},
                                             "sum": text})

    def newest(group):
        return max(e.get("created_ts") or e.get("ts") or "" for e in group)
    kept = sorted(by_guild.items(), key=lambda item: newest(item[1]), reverse=True)
    for gid, group in kept[:LTM_IMPORT_MAX_SERVERS]:
        group.sort(key=lambda e: e.get("created_ts") or e.get("ts") or "")
        IOManager.write_json_gzip(group[-defaultConfig.LIMIT_LTM:],
                                  os.path.join(p_dir, "ltm", f"{gid}.json.gz"), fernet)


def _recent_recalls(log: list, speaker_pid: str) -> Tuple[List[str], Dict[str, Tuple[int, float]]]:
    """What a speaker's recent replies recalled, read off their traces (`meta["ltm_ids"]`).

    The ids still held -- recalled within its last `LTM_HOLD_REPLIES` replies, newest
    first -- and {id: (turns back, similarity)} for the cooldown. Derived from the log
    rather than kept beside it, so it survives a restart, and a deleted or regenerated
    turn takes its recalls with it.
    """
    held: List[str] = []
    recalls: Dict[str, Tuple[int, float]] = {}
    replies = 0
    for back, turn in enumerate(reversed(log[-LTM_RECALL_SPAN:]), 1):
        if turn.get("speaker_pid") != speaker_pid:
            continue
        replies += 1
        for ltm_id, sim in (turn.get("meta") or {}).get("ltm_ids") or ():
            if ltm_id not in recalls:
                recalls[ltm_id] = (back, sim)
                if replies <= LTM_HOLD_REPLIES:
                    held.append(ltm_id)
    return held, recalls


def _trace_ids(found) -> List[List[Any]]:
    """`_search_ltm`'s fresh recalls as a reply's trace stores them: [[id, similarity]]."""
    return [[ltm_id, round(sim, 3)] for ltm_id, sim, _ in found if ltm_id]


class LtmRecall(NamedTuple):
    """One automatic pass: the block injected, the memories in it, and the ids newly
    recalled. The trace reads `summaries` rather than the block, whose wording /mod owns."""
    text: Optional[str] = None
    summaries: Sequence[str] = ()
    ids: Sequence[Sequence[Any]] = ()


class MemoryManager:
    """Owns LTM (Long-Term Memory) generation, Matryoshka vector embeddings,
    Cosine Similarity retrieval, and training-example CRUD.

    Holds a back-reference to the parent cog for state/logic not yet migrated
    (the generic shard system, profile/session lookups, model instantiation,
    and shared instance caches), per the transitional Dependency Injection
    pattern in CLAUDE.md.
    """

    def __init__(self, cog):
        self.cog = cog

    # --- Which model, which file ----------------------------------------------------

    def _embedding_models(self) -> Tuple[str, str]:
        """(the model a vector is made by now, the one an entry without `emb_model` was)."""
        current = embedding_model_key(system_model(self.cog, "embedding_model"))
        return current, getattr(self.cog, "embedding_baseline", None) or current

    def load_embedding_baseline(self) -> None:
        """Sets what an entry with no `emb_model` was embedded by. Blocking; run at boot.

        No vector named its model before v0.7.12, and the model configured on the first
        boot of a build that does is the best answer there is. It is written once, to
        EMBEDDING_BASELINE_FILE, rather than into every entry: no training shard is
        rewritten for it, and what an older build writes after a rollback reads right.
        """
        data = IOManager.read_json(EMBEDDING_BASELINE_FILE)
        model = data.get("model") if isinstance(data, dict) else None
        if not model:
            model = self._embedding_models()[0]
            IOManager.write_json({"model": model}, EMBEDDING_BASELINE_FILE)
        self.cog.embedding_baseline = model

    def _ltm_dir(self, owner_id, profile_name: str, strict: bool = False) -> Optional[str]:
        p_dir = self.cog.storage_manager._profile_dir(str(owner_id), profile_name, strict)
        return os.path.join(p_dir, "ltm") if p_dir else None

    def _ltm_path(self, owner_id, profile_name: str, guild_id, strict: bool = False) -> Optional[str]:
        """One server's memory file, or None: no such profile, or no server -- nothing is
        filed outside one. The id names a file, so it must be digits. `strict` as in
        `StorageManager._get_shard_path`: a write must not mint a folder for a stale name."""
        gid = str(guild_id or "")
        ltm_dir = self._ltm_dir(owner_id, profile_name, strict) if gid.isdigit() else None
        return os.path.join(ltm_dir, f"{gid}.json.gz") if ltm_dir else None

    def _load_ltm_shard(self, user_id: str, profile_name: str, guild_id) -> List[Dict]:
        """One server's memories, oldest first: this call's own copy. Blocking."""
        path = self._ltm_path(user_id, profile_name, guild_id)
        data = IOManager.read_json_gzip_cached(path, self.cog.storage_manager.fernet) if path else None
        return data if isinstance(data, list) else []

    def _save_ltm_shard(self, user_id: str, profile_name: str, guild_id, entries: List[Dict]):
        """Blocking. An empty list removes the server's file."""
        if not entries:
            self._delete_ltm_shard(user_id, profile_name, guild_id)
            return
        path = self._ltm_path(user_id, profile_name, guild_id, strict=True)
        if not path:
            print(f"LTM save skipped: '{profile_name}' resolves to no profile for {user_id}.")
            return
        for entry in entries:
            self._compact_ltm_entry(entry)
        IOManager.write_json_gzip(entries, path, self.cog.storage_manager.fernet)

    def _delete_ltm_shard(self, user_id: str, profile_name: str, guild_id):
        path = self._ltm_path(user_id, profile_name, guild_id)
        try:
            if path:
                os.remove(path)
        except FileNotFoundError:
            pass

    def _ltm_vectors(self, user_id: str, profile_name: str, guild_id,
                     entries: Optional[List[Dict]] = None) -> Optional[_Vectors]:
        """One server's memories as retrieval sees them. Blocking.

        `entries` is a read the caller made under `_SHARD_WRITE_LOCK`, which no writer
        can move past, so the stamp taken here still describes it.
        """
        model, baseline = self._embedding_models()
        load = (lambda: entries) if entries is not None else (
            lambda: self._load_ltm_shard(user_id, profile_name, guild_id))
        return _cached_vectors(self._ltm_path(user_id, profile_name, guild_id), load,
                               "s_emb_b64", _LTM_FIELDS, model, baseline)

    def _training_vectors(self, user_id: str, profile_name: str) -> Optional[_Vectors]:
        """A profile's training examples as retrieval sees them. Blocking."""
        model, baseline = self._embedding_models()
        return _cached_vectors(
            self.cog.storage_manager._get_shard_path("training", user_id, profile_name),
            lambda: self._load_training_shard(user_id, profile_name),
            "u_emb_b64", _TRAINING_FIELDS, model, baseline)

    def _ltm_count(self, user_id, profile_name: str, guild_id=None) -> int:
        """Memories in one server, or across every server without `guild_id`. Blocking.

        A count is a whole read: a full server file is past the shard cache's per-entry
        ceiling. Anything repainted on every edit asks for one server, never all of them.
        """
        if guild_id is not None:
            return len(self._load_ltm_shard(str(user_id), profile_name, guild_id))
        p_dir = self.cog.storage_manager._profile_dir(str(user_id), profile_name)
        return sum(len(self._load_ltm_shard(str(user_id), profile_name, gid))
                   for gid, _ in ltm_files(p_dir)) if p_dir else 0

    def _compact_ltm_entry(self, entry: Dict[str, Any]) -> None:
        """Brings an entry written by an older version to today's shape, in place.

        The one that matters is `sum`: it used to be Fernet-encrypted inside a shard
        that is itself sealed, so every search and every list paid a decrypt per entry.
        Stored plain, the next read costs nothing. The rest is dead weight -- an
        unedited entry's `modified_ts` repeats `created_ts`, and `kw`, `ts` and a list
        `s_emb` beside its base64 twin are from formats nothing reads -- multiplied by
        up to LIMIT_LTM entries in a file parsed on every recall.
        """
        text = entry.get("sum")
        if isinstance(text, str) and text.startswith("gAAAAA"):
            entry["sum"] = self.cog.storage_manager._decrypt_data(text)
        if "ts" in entry:
            entry.setdefault("created_ts", entry.pop("ts"))
        if "s_emb_b64" in entry:
            entry.pop("s_emb", None)
        entry.pop("kw", None)
        if entry.get("modified_ts") == entry.get("created_ts"):
            entry.pop("modified_ts", None)
        if entry.get("usr") is None:
            entry.pop("usr", None)

    def _copy_ltm_shard(self, user_id: str, source_profile_name: str, new_profile_name: str):
        """Every server's memories of one profile, onto another of the same owner."""
        src_dir = self._ltm_dir(user_id, source_profile_name)
        new_dir = self._ltm_dir(user_id, new_profile_name, strict=True)
        # Either name may resolve to no profile -- nothing to copy, and nowhere to put it.
        if src_dir and new_dir and os.path.isdir(src_dir):
            shutil.copytree(src_dir, new_dir, dirs_exist_ok=True)

    def _resolve_ltm_storage_target(self, profile_owner_id: int, profile_name: str, session_owner_id: Optional[int]) -> Tuple[int, str]:
        """Resolves which user/profile folder LTM data should be read from or written to,
        redirecting to the borrower's local copy when running inside a borrowed multi-profile session."""
        if session_owner_id and session_owner_id != profile_owner_id:
            pm = self.cog.profile_manager
            # The borrow index, not a read of every borrow's config.
            held = {b_pid for borrower, b_pid in self.cog.borrows._borrowers_of(
                profile_owner_id, pm._get_pid_from_name(profile_owner_id, profile_name))
                    if borrower == session_owner_id}
            borrowed = pm._get_user_index(session_owner_id).get("borrowed")
            for b_name, b_pid in (borrowed.items() if isinstance(borrowed, dict) else ()):
                if b_pid in held:
                    return session_owner_id, b_name
        return profile_owner_id, profile_name

    def _session_owner(self, channel_id) -> Optional[int]:
        """The owner of the session in `channel_id` -- whose borrow a seat's memories are
        (`_resolve_ltm_storage_target`) -- or None outside one."""
        session = self.cog.multi_profile_channels.get(channel_id) if channel_id else None
        return session.get("owner_id") if session else None

    async def _add_ltm(self, profile_owner_id: int, profile_name: str, summary: str, summary_embedding_b64: str, guild_id: Optional[int], triggering_user_id: int, user_dn: Optional[str] = None, duplicate_similarity: Optional[float] = None, source: Optional[str] = None) -> bool:
        """Appends one memory to a profile's own file for `guild_id`. False if not stored."""
        return await self._add_ltms(profile_owner_id, profile_name,
                                    [(summary, summary_embedding_b64)], guild_id, user_dn,
                                    duplicate_similarity=duplicate_similarity, source=source) == 1

    async def _add_ltms(self, profile_owner_id: int, profile_name: str,
                        memories: List[Tuple[str, str]], guild_id: Optional[int],
                        user_dn: Optional[str] = None, *,
                        duplicate_similarity: Optional[float] = None,
                        source: Optional[str] = None,
                        session_owner_id: Optional[int] = None) -> int:
        """Appends (summary, embedding_b64) memories to one server's file, in one read and
        one write. Returns how many were stored.

        `session_owner_id` is the owner of the session they formed in, which decides whose
        copy of a borrowed character keeps them (`_resolve_ltm_storage_target`). This used
        to find a session itself, the first live one seating this profile: a character in
        two people's sessions filed each one's memories under whichever came first, and a
        `/suspend` flush, its session already gone, under the owner. A memory typed on the
        memory screen passes None and lands on the profile the screen is showing.

        With `duplicate_similarity`, a memory that close to one already in this server's
        file -- or to one stored earlier in the same call -- is dropped: a scene that keeps
        returning to a joke would otherwise store it every interval. A memory someone typed
        by hand passes None and is always kept.

        `source` is how it came to be -- "auto", "memorise" or "manual" -- for the memory
        screen. `user_dn` is who that was.
        """
        if not guild_id or not memories:
            return 0
        user_id, name = self._resolve_ltm_storage_target(profile_owner_id, profile_name, session_owner_id)
        user_id = str(user_id)
        model = self._embedding_models()[0]
        async with _SHARD_WRITE_LOCK:
            entries = await asyncio.to_thread(self._load_ltm_shard, user_id, name, guild_id)

            if duplicate_similarity is not None:
                new = [decode_embedding_b64(b64) for _, b64 in memories]

                def _novel() -> List[bool]:
                    vectors = self._ltm_vectors(user_id, name, guild_id, entries) if entries else None
                    known = [vectors.unit] if vectors is not None else []
                    keep = []
                    for vec in new:
                        norm = float(np.linalg.norm(vec))
                        if not norm:
                            keep.append(True)
                            continue
                        unit = vec / norm
                        closest = max((float((k @ unit).max()) for k in known
                                       if k.shape[1] == unit.shape[0]), default=0.0)
                        keep.append(closest < duplicate_similarity)
                        if keep[-1]:
                            known.append(unit[None, :])
                    return keep

                keep = await asyncio.to_thread(_novel)
                memories = [m for m, k in zip(memories, keep) if k]
                if not memories:
                    return 0

            # Newest last and trimmed only from the front, so already in time order. One
            # slice trims a file that is already over the limit -- an import, a migration --
            # to fit, rather than one entry per write forever.
            overflow = len(entries) - defaultConfig.LIMIT_LTM + len(memories)
            if overflow > 0:
                del entries[:overflow]
            entries += [_ltm_entry(summary, b64, guild_id, model, user_dn, source)
                        for summary, b64 in memories]
            await asyncio.to_thread(self._save_ltm_shard, user_id, name, guild_id, entries)
            return len(memories)

    async def update_ltm(self, profile_owner_id: int, profile_name: str, guild_id, ltm_id: str,
                         new_summary: str, new_embedding_b64: str) -> bool:
        """Rewrites one memory in its server's file."""
        user_id = str(profile_owner_id)
        model = self._embedding_models()[0]
        async with _SHARD_WRITE_LOCK:
            entries = await asyncio.to_thread(self._load_ltm_shard, user_id, profile_name, guild_id)
            for entry in entries:
                if entry.get("id") == ltm_id:
                    entry.update(sum=new_summary.strip(), s_emb_b64=new_embedding_b64, emb_model=model,
                                 modified_ts=datetime.datetime.now(datetime.timezone.utc).isoformat())
                    await asyncio.to_thread(self._save_ltm_shard, user_id, profile_name, guild_id, entries)
                    return True
            return False

    async def load_ltms(self, owner_id: int, profile_name: str, guild_id=None) -> List[Dict]:
        """One server's memories, or every server's, newest first, `sum` in plain text --
        read and decrypted off the event loop. The entries are this call's own copies."""
        decrypt = self.cog.storage_manager._decrypt_data
        owner = str(owner_id)

        def _load():
            if guild_id:
                guilds = [guild_id]
            else:
                p_dir = self.cog.storage_manager._profile_dir(owner, profile_name)
                guilds = [gid for gid, _ in ltm_files(p_dir)] if p_dir else []
            entries = [e for gid in guilds for e in self._load_ltm_shard(owner, profile_name, gid)]
            # Reversed before a stable sort, so a batch sharing one moment lists newest first too.
            entries = sorted(reversed(entries), key=lambda e: e.get("created_ts") or e.get("ts") or "",
                             reverse=True)
            for e in entries:
                e["sum"] = decrypt(e.get("sum") or "")
            return entries
        return await asyncio.to_thread(_load)

    async def delete_ltms(self, owner_id: int, profile_name: str, guild_id, ids) -> int:
        """Removes the memories with these ids from one server's file. Returns how many went."""
        ids = set(ids)
        owner = str(owner_id)

        def _delete() -> int:
            entries = self._load_ltm_shard(owner, profile_name, guild_id)
            kept = [e for e in entries if e.get("id") not in ids]
            if len(kept) < len(entries):
                self._save_ltm_shard(owner, profile_name, guild_id, kept)
            return len(entries) - len(kept)
        async with _SHARD_WRITE_LOCK:
            return await asyncio.to_thread(_delete)

    async def rank_ltms(self, owner_id: int, profile_name: str, guild_id: str, query: str,
                        billing_guild_id: Optional[int], top: int = 5) -> Optional[List[Tuple[float, Dict]]]:
        """One server's memories nearest `query`, best first, as (similarity, entry).

        What the memory screen's Test recall shows: raw similarity, before the
        threshold, the repeat cooldown and the diversity pass pick what a turn injects.
        None when the query could not be embedded.
        """
        vectors = await asyncio.to_thread(self._ltm_vectors, str(owner_id), profile_name, guild_id)
        if vectors is None:
            return []
        emb = await self._get_embedding(query, billing_guild_id, task_type="RETRIEVAL_QUERY",
                                        owner_id=owner_id)
        if not emb:
            return None
        ranked = await asyncio.to_thread(_closest_rows, vectors.unit, emb, top)
        return [(sim, vectors.entries[r]) for sim, r in ranked]

    async def rank_training_examples(self, owner_id: int, profile_name: str, query: str,
                                     billing_guild_id: Optional[int],
                                     top: int = 5) -> Optional[List[Tuple[float, Dict]]]:
        """The examples nearest `query`, best first, as (similarity, example): the
        training screen's Test match. None when the query could not be embedded."""
        vectors = await asyncio.to_thread(self._training_vectors, str(owner_id), profile_name)
        if vectors is None:
            return []
        emb = await self._get_embedding(query, billing_guild_id, task_type="RETRIEVAL_QUERY",
                                        owner_id=owner_id)
        if not emb:
            return None
        ranked = await asyncio.to_thread(_closest_rows, vectors.unit, emb, top)
        return [(sim, vectors.entries[r]) for sim, r in ranked]

    async def load_training_examples(self, owner_id: int, profile_name: str) -> List[Dict]:
        """Every training example of a profile, newest first, in plain text -- read and
        decrypted off the event loop. The entries are this call's own copies."""
        decrypt = self.cog.storage_manager._decrypt_data

        def _load():
            examples = list(reversed(self._load_training_shard(str(owner_id), profile_name) or []))
            for e in examples:
                e["u_in"], e["b_out"] = decrypt(e.get("u_in") or ""), decrypt(e.get("b_out") or "")
            return examples
        return await asyncio.to_thread(_load)

    async def delete_training_examples(self, owner_id: int, profile_name: str, ids) -> int:
        """Removes the examples with these ids. Returns how many went."""
        ids = set(ids)

        def _delete() -> int:
            examples = self._load_training_shard(str(owner_id), profile_name) or []
            kept = [e for e in examples if e.get("id") not in ids]
            if len(kept) < len(examples):
                self._save_training_shard(str(owner_id), profile_name, kept)
            return len(examples) - len(kept)
        async with _SHARD_WRITE_LOCK:
            return await asyncio.to_thread(_delete)

    def _load_training_shard(self, user_id: str, profile_name: str) -> Optional[List[Dict]]:
        return self.cog.storage_manager._load_shard("training", user_id, profile_name)

    def _save_training_shard(self, user_id: str, profile_name: str, data: Optional[List[Dict]]):
        if not data:
            self._delete_training_shard(user_id, profile_name)
        else:
            self.cog.storage_manager._save_shard("training", user_id, data, profile_name)

    def _delete_training_shard(self, user_id: str, profile_name: str):
        self.cog.storage_manager._delete_shard("training", user_id, profile_name)

    def _copy_training_shard(self, user_id: str, source_profile_name: str, new_profile_name: str):
        src_path = self.cog.storage_manager._get_shard_path("training", user_id, source_profile_name)
        new_path = self.cog.storage_manager._get_shard_path("training", user_id, new_profile_name)
        # Either name may resolve to no profile -- nothing to copy, and nowhere to
        # put it. See _get_shard_path for why a miss is None rather than a path.
        if not src_path or not new_path:
            return
        if os.path.exists(src_path):
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            import shutil
            shutil.copy2(src_path, new_path)

    async def _search_ltm(self, session_owner_id: Optional[int], history: list, profile_owner_id: int,
                          profile_name: str, msg_content: str, author_dn: str,
                          guild_id: Optional[int], triggering_user_id: int, *,
                          threshold: Optional[float] = None, size: Optional[int] = None,
                          use_cooldown: bool = True, search: bool = True,
                          ) -> Tuple[List[str], List[Tuple[str, float, str]]]:
        """(held, found): the summaries this speaker's recent replies recalled and still
        hold, and (id, similarity, summary) for what `msg_content` newly retrieves --
        each summary prefixed with the date it was made.

        Shared by the automatic pass and the `recall` function, which want the same
        search on different terms. The keyword arguments are what differs: an explicit
        recall judges against `RECALL_TOOL_THRESHOLD` rather than the profile's, asks
        for more of them, and does not consult the repeat cooldown.

        That last one matters. The cooldown exists to stop the same memory being
        *re-injected unasked* every turn it happens to match. A character that has just
        asked for something is entitled to be told, however recently it was last shown.

        Both read off `history`, which must be the session's log: a held memory stays
        in the archive block for `LTM_HOLD_REPLIES` replies so the character can refer
        back to it, and is kept out of the search meanwhile so it never shows twice.
        `search` False returns the held ones alone, without embedding anything.

        `session_owner_id` decides whose copy of a borrowed character is searched, as it
        decides where its memories are written (`_add_ltms`): both passes must ask it, or
        the character's own `recall` searches a different archive from the one it is shown.
        """
        held_ids, recalls = (_recent_recalls(
            history, self.cog.profile_manager._get_pid_from_name_any(profile_owner_id, profile_name))
            if use_cooldown else ([], {}))
        if not search and not held_ids:
            return [], []

        pm = self.cog.profile_manager
        # A borrow's source through its PID pointer, as everywhere -- see _borrow_source.
        params_source = pm.author_config(profile_owner_id, profile_name)
        if profile_name in pm._get_user_index(profile_owner_id).get("borrowed", []):
            params_to_use = pm._get_profile_config(profile_owner_id, profile_name, True) or {}
        else:
            params_to_use = params_source

        if not params_source:
            return [], []

        ltm_context_size = int(size if size is not None else params_to_use.get(
            "ltm_context_size", params_source.get("ltm_context_size", 3)))
        ltm_relevance_threshold = float(threshold if threshold is not None else params_to_use.get(
            "ltm_relevance_threshold", params_source.get("ltm_relevance_threshold", 0.75)))

        if ltm_context_size == 0:
            return [], []
        # Bounded like the search, so the archive block is at most twice its setting.
        held_ids = held_ids[:ltm_context_size]

        ltm_user_id, ltm_profile_name = self._resolve_ltm_storage_target(
            profile_owner_id, profile_name, session_owner_id)
        # This server's file alone: a memory is recalled nowhere else. Rows and entries come
        # from one read, so the embedding awaited below cannot pair them with a newer file.
        # An empty file still skips that round trip.
        vectors = await asyncio.to_thread(self._ltm_vectors, str(ltm_user_id), ltm_profile_name, guild_id)
        if vectors is None:
            return [], []

        prompt_embedding = None
        if search:
            prompt_embedding = await self._get_embedding(msg_content, guild_id, task_type="RETRIEVAL_QUERY")
            if not prompt_embedding and not held_ids:
                return [], []

        def _thread_search_ltm():
            # A memory deleted since it was recalled has no row, and is simply not held.
            held = [vectors.entries[row]
                    for row in (vectors.id_to_row.get(i) for i in held_ids) if row is not None]
            q = _unit_query(vectors.unit, prompt_embedding) if prompt_embedding else None
            if q is None:
                return held, []
            # Rows are already unit length, so cosine is a bare dot product.
            similarities = vectors.unit @ q

            # Held, or recalled recently enough to be cooling: parked below any threshold.
            # Read off the session log rather than the file, so it cannot be cached -- but
            # it is only as large as what this session recently recalled.
            held_set = set(held_ids)
            for ltm_id, (back, last_sim) in recalls.items():
                row = vectors.id_to_row.get(ltm_id)
                if row is not None and (ltm_id in held_set or back < 5 + (1 - last_sim) * 25):
                    similarities[row] = -np.inf

            eligible = np.flatnonzero(similarities >= ltm_relevance_threshold)
            if eligible.size == 0:
                return held, []
            cand_mat = vectors.unit[eligible]
            original_sims = similarities[eligible]

            # MMR diversification. Scores are penalised in place and the winner taken by
            # argmax, so nothing is rebuilt or re-sorted per round. A picked row is
            # parked at -inf, which survives the penalty because the factor
            # (1 - cosine * 0.75) is bounded to [0.25, 1.75] and so never flips sign.
            scores = original_sims.astype(np.float32, copy=True)
            picked = []
            for _ in range(min(ltm_context_size, eligible.size)):
                best = int(np.argmax(scores))
                picked.append(best)
                scores *= 1.0 - (cand_mat @ cand_mat[best]) * 0.75
                scores[best] = -np.inf

            return held, [(vectors.entries[int(eligible[p])], float(original_sims[p])) for p in picked]

        held, found = await asyncio.to_thread(_thread_search_ltm)

        # Dated on the character's clock. Undated, "promised him cake" reads the same a
        # week on as three years on, and a character has no other way to tell.
        clock, _ = _resolve_zoneinfo(params_to_use.get("timezone"))

        def dated(ltm) -> str:
            summary = self.cog.storage_manager._decrypt_data(ltm.get('sum', ''))
            date = entry_date(ltm, clock)
            return f"[{date}] {summary}" if date else summary

        # Nothing is recorded here: the reply's trace keeps what it recalled
        # (`_trace_ids`), so a turn that is never written puts nothing on cooldown.
        return [dated(ltm) for ltm in held], [(ltm.get('id'), sim, dated(ltm)) for ltm, sim in found]

    async def recall_for_tool(self, profile_owner_id: int, profile_name: str, query: str,
                              author_dn: str, guild_id: Optional[int],
                              triggering_user_id: int, channel_id: Optional[int] = None) -> Dict[str, Any]:
        """The `recall` function's answer, as the object handed back to the model.

        No history: an explicit recall does not consult the repeat cooldown, which is the
        only thing history serves. `channel_id` finds the session, whose owner decides
        whose copy of a borrowed character is searched, as it does for the automatic pass.

        A miss is answered rather than left empty. "I searched and there is nothing"
        and "the search never ran" are the same silence to a model, and the second one
        is what invites it to invent a memory instead.

        `ltm_ids` is for the reply's trace, so what was fetched is held afterwards like
        any recall; `tool_loop.run` takes it off before the model is answered.
        """
        _, found = await self._search_ltm(
            self._session_owner(channel_id), [], profile_owner_id, profile_name, query, author_dn,
            guild_id, triggering_user_id, threshold=RECALL_TOOL_THRESHOLD, size=RECALL_TOOL_MAX,
            use_cooldown=False)
        if not found:
            return {"memories": [],
                    "note": "Nothing in your long-term memories matches that."}
        return {"memories": [summary for _, _, summary in found], "ltm_ids": _trace_ids(found)}

    async def _get_relevant_ltm_for_prompt(self, session_key: Any, history: list, profile_owner_id: int, profile_name: str, msg_content: str, author_dn: str, guild_id: Optional[int], triggering_user_id: int, *, threshold: Optional[float] = None) -> LtmRecall:
        """The automatic pass: what this round retrieves, wrapped for injection in /mod's
        `ARCHIVE_CONTEXT`, and the ids it newly recalled -- for the reply's trace
        (`meta["ltm_ids"]`), which the hold and the cooldown are read back from. Held
        memories lead the block.

        `threshold` is raised to `LTM_AUTO_THRESHOLD_WITH_TOOL` by callers whose profile
        also declares `recall`, so the unasked injection narrows to what retrieval is
        sure of while everything below it stays reachable by asking. Left alone, the
        profile's own setting applies and the behaviour is what it always was.

        The Auto-Recall toggle is read here rather than at the three call sites, so a
        fourth caller cannot be written that ignores it, and so the exit lands before
        the shard read *and* the query embedding -- a profile with this off pays nothing
        per turn rather than paying for a search whose result is discarded. `recall`
        does not come through here and is deliberately unaffected: pull-only retrieval
        is its own mode -- and what it fetched is still held with the toggle off, which
        reads the shard only while something is.
        """
        index = self.cog.profile_manager._get_user_index(profile_owner_id)
        is_borrowed = profile_name in index.get("borrowed", [])
        config = self.cog.profile_manager._get_profile_config(
            profile_owner_id, profile_name, is_borrowed) or {}
        channel_id = session_key[0] if isinstance(session_key, tuple) and session_key else None

        held, found = await self._search_ltm(
            self._session_owner(channel_id), history, profile_owner_id, profile_name, msg_content,
            author_dn, guild_id, triggering_user_id, threshold=threshold,
            search=ltm_auto_recall_enabled(config))
        summaries = held + [summary for _, _, summary in found]
        if not summaries:
            return LtmRecall()
        template = self.cog.global_prompts.get("ARCHIVE_CONTEXT", DEFAULT_ARCHIVE_CONTEXT)
        return LtmRecall(template.format(memories="\n".join(summaries)), summaries, _trace_ids(found))

    async def _get_relevant_training_examples(self, profile_owner_id: int, profile_name: str, msg_content:str, guild_id: int)->List[str]:
        """The examples nearest the round, as fixed-example blocks, for the source profile.

        Read straight off the profile config: both values wanted here come from the config
        the first lookup returns, and every accessor involved reads a cog-owned dict with
        disk only on a cold miss.
        """
        index = self.cog.profile_manager._get_user_index(profile_owner_id)
        is_borrowed = profile_name in index.get("borrowed", [])
        config = self.cog.profile_manager._get_profile_config(profile_owner_id, profile_name, is_borrowed) or {}

        training_context_size = int(config.get("training_context_size", defaultConfig.TRAINING_CONTEXT_SIZE))
        training_relevance_threshold = float(config.get("training_relevance_threshold", defaultConfig.TRAINING_RELEVANCE_THRESHOLD))
        if training_context_size == 0:
            return []

        effective_owner_id, effective_profile_name = self.cog.profile_manager._resolve_effective_profile(profile_owner_id, profile_name)

        # Rows first, embedding second: a profile with training on and no examples yet is
        # never billed a query embedding for nothing to rank.
        vectors = await asyncio.to_thread(self._training_vectors, str(effective_owner_id), effective_profile_name)
        if vectors is None:
            return []

        msg_emb = await self._get_embedding(msg_content, guild_id, task_type="RETRIEVAL_QUERY")
        if not msg_emb:
            return []

        def _thread_search_training():
            q = _unit_query(vectors.unit, msg_emb)
            if q is None:
                return []
            # Rows are already unit length, so cosine is a bare dot product.
            similarities = vectors.unit @ q
            eligible = np.flatnonzero(similarities >= training_relevance_threshold)
            if eligible.size == 0:
                return []
            top = eligible[np.argsort(-similarities[eligible], kind="stable")][:training_context_size]

            decrypt = self.cog.storage_manager._decrypt_data
            return [fixed_examples.block(decrypt(vectors.entries[int(row)].get('u_in', '')),
                                         decrypt(vectors.entries[int(row)].get('b_out', '')))
                    for row in top]

        return await asyncio.to_thread(_thread_search_training)

    async def _get_embedding(self, text: str, guild_id: Optional[int], task_type: str = "RETRIEVAL_QUERY",
                             owner_id: Optional[int] = None) -> Optional[List[float]]:
        """The vector for `text`, or None if there is no key or the request failed.

        Pass `owner_id` from anything a user drives directly against a profile's own
        data -- it is what lets the profile owner's personal key stand in when there
        is no guild key, which is every DM. See `_embedding_routes`.
        """
        if not text or not text.strip():
            return None

        routes = self.cog.storage_manager._embedding_routes(guild_id, owner_id)
        if not routes:
            return None

        return await get_embedding_vector(routes, text, task_type=task_type, output_dimensionality=256, timeout=5.0)

    async def _generate_ltm_data_from_history(self, hist: List[str], user_dn: str, guild_id: Optional[int], character_name: Optional[str] = None, profile_owner_id: int = None, profile_name: str = None, warning_channel: Optional[discord.abc.Messageable] = None, background: Optional[str] = None) -> Optional[List[str]]:
        """Summarises a slice of history into up to LTM_MAX_PER_CAPTURE long-term memories.

        Returns the memories; [] when the excerpt was judged and holds nothing worth
        keeping (declined, blocked or empty); None when no verdict was reached -- an
        API error, which the caller retries later on the same turns.

        `background` is the scene's synopsis: who is who, from turns this excerpt no
        longer shows. Marked as known, so it informs the memory without becoming one.

        The summariser model comes from the profile's own `ltm_model`, retried on
        `ltm_fallback_model`. It used to take a `model_name_to_use` argument that both
        callers filled in -- one with the response primary, one with the response
        fallback -- and that nothing in the body has ever read. Removed rather than
        wired: honouring it would have silently moved every LTM summary onto the
        response model.
        """
        if not hist or len(hist) < MIN_HISTORY_FOR_LTM_CREATION: return []

        # [UPDATED] Standardize history for the LTM Summarizer
        # Provides Name [Timestamp] and Content only, stripping metadata and summaries.
        convo_parts = []
        for raw_text in hist:
            if not raw_text: continue

            try:
                with Timeout(seconds=2, error_message="LTM regex parsing timed out"):
                    # 1. Strip technical metadata
                    text = re.sub(r'\(\s*Thought Initiated:.*?\)\s*\n?', '', raw_text).strip()

                    # 2. Strip previous contexts to avoid "recursive" memory creation
                    lines = text.split('\n')
                    filtered_lines = []
                    skip_block = False
                    for line in lines:
                        l_strip = line.strip()
                        if any(l_strip.startswith(prefix) for prefix in [
                            "<external_context>",
                            "<document_context>",
                            "<archive_context>",
                            "<internal_note>",
                            "<image_context>"
                        ]):
                            skip_block = True
                            continue
                        if skip_block:
                            if l_strip.startswith(("</external_context>", "</document_context>", "</archive_context>", "</internal_note>", "</image_context>")):
                                skip_block = False
                            continue
                        filtered_lines.append(line)

                    final_content = "\n".join(filtered_lines).strip()
                    if final_content:
                        if not re.match(r'^<.+> \[[^\]]+\]:', final_content) and not re.match(r'^.+ \[[^\]]+\]:', final_content):
                            ts_str = f"[{datetime.datetime.now(datetime.timezone.utc).strftime(TURN_TIME_FORMAT)}]"
                            convo_parts.append(f"<Unknown> {ts_str}:\n{final_content}\n</Unknown>")
                        else:
                            convo_parts.append(final_content)
            except TimeoutError:
                continue

        if not convo_parts:
            return []
        # Whole turns. A 3,000-character tail used to cut here: it opened the excerpt
        # mid-message with no speaker, and left about one round of a busy scene --
        # rarely the line that said who anyone was.
        convo = "\n\n".join(convo_parts)
        prompt = f"<target_transcript>\n{convo}\n</target_transcript>"
        if background:
            prompt = ("Already known from earlier in the scene. Background only: write no "
                      f"memory from it.\n<session_synopsis>\n{background}\n</session_synopsis>\n\n"
                      + prompt)
        if character_name:
            # Told only "a character", the summariser guessed which speaker it wrote for.
            prompt += (f"\n\nThe character whose memory this is: {character_name}. "
                       f"Times are on {character_name}'s clock.")

        # One resolver, shared with the screen that edits this prompt: a profile that
        # never wrote one of its own follows the live default rather than a copy frozen
        # into it the day it was created.
        instructions = self.cog.profile_manager.resolve_ltm_summarization_instructions(
            profile_owner_id, profile_name)
        # Bound unconditionally: it is only populated for a known profile, and the reads
        # below used to hedge on `'params_source' in locals()` -- which is correct and
        # invisible, so the next edit is one NameError away.
        params_source: Dict[str, Any] = {}
        if profile_owner_id and profile_name:
            user_index = self.cog.profile_manager._get_user_index(profile_owner_id)
            is_borrowed = profile_name in user_index.get("borrowed", [])
            params_source = self.cog.profile_manager._get_profile_config(profile_owner_id, profile_name, is_borrowed) or {}

        cfg = dict(GREEDY_SAMPLING)

        ltm_model_raw, ltm_fallbacks = self.cog.api_service.model_chain(
            params_source, "ltm_model", profile_owner_id)

        effective_guild_id = guild_id or 0

        try:
            async def _attempt(model_name, is_fallback):
                # `{}` used to sit in the thinking slot, which is not "no thinking":
                # the adapters default an absent thinking_level to "high", so every
                # long-term memory ever written was billed a full reasoning pass to
                # compress a transcript. The resolver's ltm default is low/512, and a
                # profile can now raise it deliberately.
                m = self.cog.api_service._instantiate_model(
                    model_name, effective_guild_id if effective_guild_id else None,
                    profile_owner_id, instructions, DEFAULT_SAFETY_SETTINGS,
                    resolve_thinking_params(
                        params_source, "ltm", "fallback" if is_fallback else "primary",
                        cog=self.cog),
                    None, params_source, config_owner_id=profile_owner_id)
                return await m.generate_content_async(
                    [prompt], generation_config=cfg)

            r, _used, _was_fallback = await self.cog.api_service.run_with_fallback(
                ltm_model_raw, ltm_fallbacks, _attempt, label="LTM summariser")

            response_text = ""
            if r.candidates:
                candidate = r.candidates[0]
                if candidate.content and candidate.content.parts:
                    response_text = "".join(p.text for p in candidate.content.parts if hasattr(p, 'text')).strip()

            # Normalised, not compared raw. "NO_SUMMARY." and "No summary needed." both
            # used to pass this test and be stored, embedded and injected as a memory --
            # a permanent entry whose whole content is the model declining to write one.
            # Declined, blocked or empty is [], a verdict on these turns, not a failure.
            return [] if _is_abstention(response_text) else split_memories(response_text)
        except Exception as e:
            err_str = str(e)
            # A missing/misconfigured key is a configuration state, not a transient
            # failure -- same treatment as the rate-limit/unavailable cases below, since
            # both primary and fallback routinely miss a key on a server that has only
            # configured one provider, and that should not read as a crash.
            if (not self.cog.shutting_down
                    and "429" not in err_str and "RESOURCE_EXHAUSTED" not in err_str and "503" not in err_str
                    and "UNAVAILABLE" not in err_str and "API Key not found" not in err_str
                    and not getattr(e, "rate_limited", False)):
                print(f"LTM Gen err {user_dn}: {e}")
                traceback.print_exc()
                if warning_channel:
                    await self.cog.generation_service._send_session_warning(warning_channel, f"Long-Term Memory creation failed ({_format_api_error(e)})")
        return None

    async def add_new_training_example(self, profile_owner_id: int, profile_name: str, usr_in:str, bot_out:str, guild_id: Optional[int])->Tuple[bool,str]:
        if not usr_in.strip() or not bot_out.strip(): return False,"Inputs empty."

        # [NEW] Training Limit Check
        owner_id_str = str(profile_owner_id)
        training_shard = await asyncio.to_thread(self._load_training_shard, owner_id_str, profile_name) or []

        limit = defaultConfig.LIMIT_TRAINING
        full = False, f"**Limit Reached.**\n\nYou have reached the maximum of **{limit}** training examples."

        if len(training_shard) >= limit:
            return full

        if not self.cog.storage_manager._embedding_routes(guild_id, profile_owner_id):
            return False, NO_EMBEDDING_KEY_MSG
        emb = await self._get_embedding(usr_in, guild_id, task_type="RETRIEVAL_DOCUMENT",
                                        owner_id=profile_owner_id)
        if not emb:
            return False, EMBEDDING_FAILED_MSG

        async with _SHARD_WRITE_LOCK:
            # Read again: the embedding took seconds, and what was saved meanwhile stays.
            training_shard = await asyncio.to_thread(self._load_training_shard, owner_id_str, profile_name) or []
            if len(training_shard) >= limit:
                return full
            training_shard.append(_training_entry(usr_in, bot_out, emb, self._embedding_models()[0]))
            await asyncio.to_thread(self._save_training_shard, owner_id_str, profile_name, training_shard)
        return True,f"Example added for profile '{profile_name}'. Total: {len(training_shard)}/{limit}"

    async def add_training_examples(self, owner_id: int, profile_name: str, pairs: List[Dict[str, str]],
                                    guild_id: Optional[int]) -> str:
        """An imported file's examples, after the profile's own, for `/import`. Never
        `add_new_training_example` per row: that seals the whole shard once an example.

        What is a duplicate, has nothing said to match on, or does not fit is dropped before
        anything is embedded, so no call is paid for a row that is not kept. The embeddings
        run one at a time (a burst trips a free key's rate limit) with the shard unread, and
        it is read again to append, so an edit made meanwhile is kept, not overwritten."""
        owner = str(owner_id)
        decrypt = self.cog.storage_manager._decrypt_data
        limit = defaultConfig.LIMIT_TRAINING

        def _held():
            if profile_name not in self.cog.profile_manager._get_user_index(owner_id).get("personal", []):
                return None
            return self._load_training_shard(owner, profile_name) or []

        held = await asyncio.to_thread(_held)
        # A borrow's training is its source's (`_resolve_effective_profile`): one written
        # onto it is never read.
        if held is None:
            return "❌ You can only add training examples to a personal profile you own."
        if not self.cog.storage_manager._embedding_routes(guild_id, owner_id):
            return f"❌ {NO_EMBEDDING_KEY_MSG}"
        have = {(decrypt(e.get("u_in") or ""), decrypt(e.get("b_out") or "")) for e in held}
        said = [p for p in pairs if p["said"]]
        fresh = [p for p in said if (p["said"], p["reply"]) not in have]
        todo = fresh[:max(0, limit - len(held))]

        made, failed, in_a_row = [], 0, 0
        model = self._embedding_models()[0]
        for i, p in enumerate(todo):
            emb = await self._get_embedding(p["said"], guild_id, task_type="RETRIEVAL_DOCUMENT",
                                            owner_id=owner_id)
            if emb:
                made.append(_training_entry(p["said"], p["reply"], emb, model))
                in_a_row = 0
                continue
            failed, in_a_row = failed + 1, in_a_row + 1
            # ponytail: three straight failures read as a dead key or a spent quota; past it,
            # 100 rows x 5 s x each route outlives the interaction's 15 minutes.
            if in_a_row >= 3:
                failed += len(todo) - i - 1
                break

        def _append() -> int:
            shard = self._load_training_shard(owner, profile_name) or []
            kept = made[:max(0, limit - len(shard))]
            if kept:
                self._save_training_shard(owner, profile_name, shard + kept)
            return len(kept)

        added = 0
        if made:
            async with _SHARD_WRITE_LOCK:
                added = await asyncio.to_thread(_append)
        late = len(made) - added   # filled by another write while these embedded
        message = f"Added {added} training example{'' if added == 1 else 's'} to **{profile_name}**."
        if len(said) > len(fresh):
            message += f" {len(said) - len(fresh)} it already had were skipped."
        if len(pairs) > len(said):
            message += (f" {len(pairs) - len(said)} had nothing said, which a training example is matched "
                        "on: import those as fixed examples instead.")
        if len(fresh) - len(todo) + late:
            message += f" {len(fresh) - len(todo) + late} did not fit: a profile holds {limit} training examples."
        if failed:
            message += f" {failed} could not be embedded (the API may be busy or the key rejected); import the file again to add them."
        return message

    async def update_training_example(self, profile_owner_id: int, profile_name: str, example_id: str, new_user_input: str, new_bot_response: str, guild_id: Optional[int]) -> Tuple[bool, str]:
        if not new_user_input.strip() or not new_bot_response.strip():
            return False, "Inputs cannot be empty."

        # Checked before the shard is even loaded: a missing key is the likeliest
        # failure here and has nothing to do with whether the example exists.
        if not self.cog.storage_manager._embedding_routes(guild_id, profile_owner_id):
            return False, NO_EMBEDDING_KEY_MSG

        owner_id_str = str(profile_owner_id)
        example_list = await asyncio.to_thread(self._load_training_shard, owner_id_str, profile_name)
        if example_list is None:
            return False, f"No training examples found for profile '{profile_name}'."

        example_found = False
        if any(example.get("id") == example_id for example in example_list):
            new_embedding = await self._get_embedding(new_user_input, guild_id,
                                                      task_type="RETRIEVAL_DOCUMENT",
                                                      owner_id=profile_owner_id)
            if not new_embedding:
                return False, EMBEDDING_FAILED_MSG
            b64_emb = encode_embedding_b64(new_embedding)
            model = self._embedding_models()[0]

            # Read again for the write: the embedding took seconds, and an edit or delete
            # saved meanwhile is kept rather than overwritten by the list read above.
            def _update() -> bool:
                examples = self._load_training_shard(owner_id_str, profile_name) or []
                for example in examples:
                    if example.get("id") == example_id:
                        example["u_in"] = new_user_input.strip()
                        example["b_out"] = new_bot_response.strip()
                        example["u_emb_b64"] = b64_emb
                        example["emb_model"] = model
                        example.pop("u_emb", None)
                        example["modified_ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                        self._save_training_shard(owner_id_str, profile_name, examples)
                        return True
                return False

            async with _SHARD_WRITE_LOCK:
                example_found = await asyncio.to_thread(_update)

        if example_found:
            return True, f"Successfully updated training example `{example_id}` for profile '{profile_name}'."
        else:
            return False, f"Could not find a training example with ID `{example_id}` for profile '{profile_name}'."

    async def _execute_training_analysis(self, interaction: discord.Interaction, owner_id: int,
                                         profile_name: str, count: int, verbosity: int):
        """Writes a style guide from the newest `count` examples, and turns it on. True once
        written, for the caller to offer the rating prompt: the guide is moderated text.

        On the profile's own Training Analysis chain. `owner_id` is the profile's owner,
        who is not the invoker under /mod.
        """
        examples = await asyncio.to_thread(self._load_training_shard, str(owner_id), profile_name) or []
        if not examples:
            await interaction.followup.send(embed=text_embed("\u274c No training examples found to analyse."), ephemeral=True); return

        # Take the N most recent examples
        subset = examples[-count:]
        formatted_examples = []
        for ex in subset:
            u = self.cog.storage_manager._decrypt_data(ex['u_in'])
            b = self.cog.storage_manager._decrypt_data(ex['b_out'])
            formatted_examples.append(f"User: {u}\nAssistant: {b}")

        examples_block = "\n---\n".join(formatted_examples)

        # [UPDATED] Standardized XML tagging for the Analysis Prompt
        prompt = self.cog.global_prompts.get("TRAINING_ANALYST", DEFAULT_TRAINING_ANALYST_PROMPT).format(verbosity=verbosity, examples_block=examples_block)

        try:
            index = self.cog.profile_manager._get_user_index(owner_id)
            p_is_b = profile_name in index.get("borrowed", [])
            p_cfg = self.cog.profile_manager._get_profile_config(owner_id, profile_name, p_is_b) or {}

            async def _attempt(name, is_fallback):
                # `{}` reads as "high" in every adapter, so unset a style guide thinks at its
                # slot's default, medium: it runs once per Analyse, and the guide rides every reply.
                model = self.cog.api_service._instantiate_model(
                    name, interaction.guild_id, owner_id, profile_settings=p_cfg,
                    config_owner_id=owner_id, thinking_params=resolve_thinking_params(
                        p_cfg, "training", "fallback" if is_fallback else "primary", cog=self.cog))
                return await model.generate_content_async([prompt])

            resp, _used, _was_fallback = await self.cog.api_service.run_with_fallback(
                *self.cog.api_service.model_chain(p_cfg, "training_analysis_model", owner_id),
                _attempt, label="Training analysis")
            response_text = getattr(resp, "text", None)

            if not response_text: raise ValueError("Model returned an empty response.")

            if profile_name in index.get("personal", []):
                prompts = self.cog.profile_manager._get_profile_prompts(owner_id, profile_name) or {}
                # Its own key, never an AI Instructions part: those are hand-written, and this
                # is only ever what the examples said. On, since someone just asked for it.
                prompts["style_guide"] = self.cog.storage_manager._encrypt_data(response_text[:4000])
                prompts["style_guide_enabled"] = True
                self.cog.profile_manager._save_profile_prompts(owner_id, profile_name, prompts)

                await interaction.followup.send(embed=text_embed(f"\u2705 **Analysis Complete.** Style guide saved and turned on for '{profile_name}'. **Training \u2192 Style Guide** shows it."), ephemeral=True)
                return True
            else:
                await interaction.followup.send(embed=text_embed("\u274c Profile not found."), ephemeral=True)

        except Exception as e:
            await interaction.followup.send(embed=text_embed(f"\u274c **Analysis Failed:** {suppress_link_previews(str(e))}"), ephemeral=True)

    async def bulk_reset_examples(self, user_id: int, profile_names: List[str]) -> str:
        user_id_str = str(user_id)
        reset_count = 0
        async with _SHARD_WRITE_LOCK:
            for name in profile_names:
                pid = self.cog.profile_manager._get_pid_from_name_any(user_id, name)
                shard_path = os.path.join(self.cog.USERS_DIR, user_id_str, "profiles", pid, "training.json.gz")
                if os.path.exists(shard_path):
                    self._delete_training_shard(user_id_str, name)
                    reset_count += 1

        return f"Reset all training examples for {reset_count} profile(s)."

    async def bulk_reset_ltm(self, user_id: int, profile_names: List[str]) -> str:
        dirs = [d for d in (self._ltm_dir(user_id, name) for name in profile_names) if d]

        def _reset() -> int:
            gone = [d for d in dirs if ltm_files(os.path.dirname(d))]
            for d in gone:
                shutil.rmtree(d, ignore_errors=True)
            return len(gone)
        async with _SHARD_WRITE_LOCK:
            reset_count = await asyncio.to_thread(_reset)
        return f"Reset all Long-Term Memories for {reset_count} profile(s)."

    def migrate_ltm_files(self) -> None:
        """Splits every `ltm.json.gz` -- all of a profile's servers in one file, the layout
        before v0.7.12 -- into one file per server. Blocking: run at boot, before the gateway,
        so no turn reads or writes a memory meanwhile.

        Proactive, so no reader carries the old layout. Each server's memories are merged by
        id into whatever its file already holds, and the old file goes only once all are
        written: a crash part-way is redone on the next boot, and an `ltm.json.gz` an older
        build writes after a rollback is folded in on the way back. Nothing is trimmed -- a
        server over LIMIT_LTM is cut to it by its next new memory. A memory with no server
        was never recalled or listed anywhere, and is dropped.
        """
        fernet = self.cog.storage_manager.fernet
        legacy_files = glob.glob(os.path.join(self.cog.USERS_DIR, "*", "profiles", "*", "ltm.json.gz"))
        moved, unfiled, over = 0, 0, []
        for legacy in legacy_files:
            try:
                # One that will not decode is moved to .corrupt by the read, and skipped.
                data = IOManager.read_json_gzip(legacy, fernet)
                if data is None:
                    continue
                by_guild: Dict[str, List[Dict[str, Any]]] = {}
                for entry in (data.get("guild") if isinstance(data, dict) else None) or ():
                    gid = str(entry.get("context_id") or "") if isinstance(entry, dict) else ""
                    if not gid.isdigit():
                        unfiled += 1
                        continue
                    self._compact_ltm_entry(entry)
                    by_guild.setdefault(gid, []).append(entry)
                p_dir = os.path.dirname(legacy)
                for gid, entries in by_guild.items():
                    path = os.path.join(p_dir, "ltm", f"{gid}.json.gz")
                    have = IOManager.read_json_gzip(path, fernet)
                    have = have if isinstance(have, list) else []
                    known = {e.get("id") for e in have if isinstance(e, dict)}
                    merged = have + [e for e in entries if e.get("id") not in known]
                    merged.sort(key=lambda e: e.get("created_ts") or "")
                    IOManager.write_json_gzip(merged, path, fernet)
                    moved += len(merged) - len(have)
                    if len(merged) > defaultConfig.LIMIT_LTM:
                        over.append(f"{os.path.relpath(path, self.cog.USERS_DIR)} ({len(merged)})")
                os.remove(legacy)
            except Exception as e:
                print(f"LTM migration: {legacy} left in place ({e!r}); it is tried again next boot.")
        if legacy_files:
            print(f"LTM migration: {len(legacy_files)} profile file(s) split into one file per "
                  f"server, {moved} memories moved, {unfiled} with no server dropped.")
        if over:
            print(f"LTM migration: {len(over)} server file(s) over LIMIT_LTM "
                  f"({defaultConfig.LIMIT_LTM}), trimmed by their next memory: {', '.join(over[:10])}")
