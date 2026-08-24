# Deferred annotation evaluation, so the two `-> np.ndarray` return annotations
# below never touch the lazy proxy at definition time. See the TYPE_CHECKING branch
# under _LazyNumpy for the static half of the same problem.
from __future__ import annotations

import os
import re
import uuid
import base64
import asyncio
import datetime
import traceback
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Any, Optional, Union, Tuple
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
    defaultConfig, FALLBACK_MODEL_NAME, DEFAULT_SAFETY_SETTINGS,
    MIN_HISTORY_FOR_LTM_CREATION,
    DEFAULT_TRAINING_ANALYST_PROMPT,
)
from ..utils.helpers import Timeout, _format_api_error, _get_sanitized_history_and_author
from .storage_manager import IOManager
from ..services.api_service import get_embedding_vector


def encode_embedding_b64(embedding: List[float]) -> str:
    if not embedding: return ""
    return base64.b64encode(np.array(embedding, dtype=np.float16).tobytes()).decode('ascii')

def decode_embedding_b64(b64_str: str) -> np.ndarray:
    if not b64_str: return np.array([], dtype=np.float32)
    return np.frombuffer(base64.b64decode(b64_str), dtype=np.float16).astype(np.float32)

# Try importing native Rust/C extension if compiled into the environment
try:
    import mimic_core  # type: ignore
    _HAS_NATIVE_CORE = True
except ImportError:
    _HAS_NATIVE_CORE = False

def calculate_similarities(prompt_emb: List[float], b64_embs: List[str]) -> np.ndarray:
    if not b64_embs or not prompt_emb: return np.array([], dtype=np.float32)
    
    if _HAS_NATIVE_CORE and hasattr(mimic_core, "calculate_similarities_b64"):
        return np.array(mimic_core.calculate_similarities_b64(prompt_emb, b64_embs), dtype=np.float32)

    raw_bytes = b"".join(base64.b64decode(s) for s in b64_embs)
    matrix = np.frombuffer(raw_bytes, dtype=np.float16).reshape(len(b64_embs), -1).astype(np.float32)
    prompt_vec = np.array(prompt_emb, dtype=np.float32)

    emb_norms = np.linalg.norm(matrix, axis=1)
    prompt_norm = np.linalg.norm(prompt_vec)

    emb_norms[emb_norms == 0] = 1e-10
    prompt_norm = prompt_norm if prompt_norm != 0 else 1e-10

    return np.dot(matrix, prompt_vec) / (emb_norms * prompt_norm)


# --- LTM shard vector cache ----------------------------------------------------
#
# Retrieval used to rebuild everything numeric about a shard on every single call:
# base64-decode each stored embedding, stack, upcast to float32, and compute row norms.
# At the LIMIT_LTM cap of 5000 that measured ~10.8 ms per retrieval, most of it
# Python-level looping that holds the GIL -- so running it under asyncio.to_thread
# bought nothing and it stalled heartbeats anyway, per the hot-path rule in CLAUDE.md.
# It ran once per participant per turn over a shard that had not changed.
#
# Everything cached here is a pure function of the file on disk, so the stat stamp is
# a complete invalidation signal: _save_shard writes atomically via os.replace, which
# moves both mtime_ns and size.
#
# Rows are stored *pre-normalised*, which is what makes a query a single BLAS
# `matrix @ unit_prompt` with no norms at query time. This is an in-memory
# representation only -- the on-disk float16 base64 format is untouched.
#
# Bounded by total rows rather than entry count, because entries differ in size by
# three orders of magnitude: 20000 rows at 256 float32 dims is ~20 MB, which is the
# ceiling this is allowed to occupy on the 1 GB target.

_LTM_VEC_CACHE_MAX_ROWS = 20000
_ltm_vec_cache: "OrderedDict[Any, Any]" = OrderedDict()


class _LTMVectors:
    """Numeric view of one LTM shard, valid for as long as the file is unchanged.

    `rows` are the indices into the caller's `all_profile_ltms` list that carry an
    embedding, so a row index computed here maps straight back onto the original
    records.

    `guild_rows` maps a guild id to the row indices formed in that guild. Retrieval
    only ever looks at one guild, and the membership is a pure function of the file,
    so the grouping is built once here instead of being recomputed per turn. It used
    to be `context_ids == guild_id_str` over an object-dtype array -- an element-wise
    Python string compare across the whole shard, ~0.08 ms at the LIMIT_LTM cap for
    something a dict lookup answers. Doing it here also means the dot product and
    everything after it run over one guild's rows rather than all of them.
    """

    __slots__ = ("rows", "ids", "id_to_row", "guild_rows", "unit", "n")

    def __init__(self, rows, ids, id_to_row, guild_rows, unit):
        self.rows = rows
        self.ids = ids
        self.id_to_row = id_to_row
        self.guild_rows = guild_rows
        self.unit = unit
        self.n = len(rows)


def _build_ltm_vectors(all_profile_ltms) -> Optional["_LTMVectors"]:
    rows, ids, b64_embs = [], [], []
    guild_rows: Dict[str, List[int]] = {}
    for i, ltm in enumerate(all_profile_ltms):
        if "s_emb_b64" not in ltm:
            continue
        guild_rows.setdefault(str(ltm.get("context_id")), []).append(len(rows))
        rows.append(i)
        ids.append(ltm.get("id"))
        b64_embs.append(ltm["s_emb_b64"])

    if not rows:
        return None

    raw_bytes = b"".join(base64.b64decode(s) for s in b64_embs)
    matrix = np.frombuffer(raw_bytes, dtype=np.float16).reshape(len(b64_embs), -1).astype(np.float32)

    norms = np.linalg.norm(matrix, axis=1)
    # A zero row divided by 1.0 stays zero, so its similarity is 0 -- the same answer
    # the old `norms[norms == 0] = 1e-10` produced, without manufacturing a huge vector.
    safe = np.where(norms == 0.0, 1.0, norms)
    unit = matrix / safe[:, None]

    return _LTMVectors(
        rows=rows,
        ids=ids,
        id_to_row={ltm_id: idx for idx, ltm_id in enumerate(ids) if ltm_id is not None},
        guild_rows={g: np.array(r, dtype=np.intp) for g, r in guild_rows.items()},
        unit=unit,
    )


def _cached_vectors(cache, key, max_rows, build, *build_args):
    """Fetch or build one shard's numeric view, keeping `cache` LRU and row-bounded.

    `key` is None when the shard has no file to stat -- an in-memory or just-deleted
    shard. The vectors are still built, but not stored: a stamp that cannot be
    invalidated is worse than no cache entry at all.
    """
    if key is not None:
        vectors = cache.get(key)
        if vectors is not None:
            cache.move_to_end(key)
            return vectors

    vectors = build(*build_args)
    if vectors is None or key is None:
        return vectors

    cache[key] = vectors  # a fresh key inserts at the end already
    total = sum(v.n for v in cache.values())
    # Never evict down to empty: dropping the entry just built would mean nothing is
    # ever cached. Unreachable as configured -- one shard is capped at LIMIT_LTM /
    # LIMIT_TRAINING rows, both far under the per-cache row budgets.
    while total > max_rows and len(cache) > 1:
        _, dropped = cache.popitem(last=False)
        total -= dropped.n
    return vectors


# --- Training shard vector cache -----------------------------------------------
#
# Same stat-stamped, pre-normalised scheme as the LTM cache above, for the same reason:
# the example embeddings are re-decoded from base64, re-stacked and re-normed on every
# turn over a file that has not changed. The shard itself still has to be read each turn
# -- the winning examples' text lives in it -- so this only removes the numeric rebuild,
# not the decrypt; at the LIMIT_TRAINING cap of 100 that is tens of microseconds per
# participant per turn rather than the milliseconds the LTM cache saves.
#
# Budgeted separately and much smaller than LTM's: 100 rows is the per-profile ceiling,
# so 5000 rows is ~50 profiles resident at ~5 MB.

_TRAINING_VEC_CACHE_MAX_ROWS = 5000
_train_vec_cache: "OrderedDict[Any, Any]" = OrderedDict()


class _TrainingVectors:
    """Numeric view of one training shard, valid for as long as the file is unchanged.

    `rows` are indices into the caller's example list that carry an embedding, so a
    row index maps straight back onto the original record.
    """

    __slots__ = ("rows", "unit", "n")

    def __init__(self, rows, unit):
        self.rows = rows
        self.unit = unit
        self.n = len(rows)


def _build_training_vectors(profile_examples) -> Optional["_TrainingVectors"]:
    rows, b64_embs = [], []
    for i, ex in enumerate(profile_examples):
        if "u_emb_b64" not in ex:
            continue
        rows.append(i)
        b64_embs.append(ex["u_emb_b64"])

    if not rows:
        return None

    raw_bytes = b"".join(base64.b64decode(s) for s in b64_embs)
    matrix = np.frombuffer(raw_bytes, dtype=np.float16).reshape(len(b64_embs), -1).astype(np.float32)

    norms = np.linalg.norm(matrix, axis=1)
    safe = np.where(norms == 0.0, 1.0, norms)
    return _TrainingVectors(rows=rows, unit=matrix / safe[:, None])


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

    def _load_ltm_shard(self, user_id: str, profile_name: str) -> Optional[Dict[str, List[Dict]]]:
        """Reads a profile's LTM archive: one "guild" bucket, every entry keyed by
        the guild it formed in.

        There is exactly one scope -- server -- so nothing is filtered here. This
        used to rebuild the list on every load to drop memories under retired
        scopes, which cost a pass over up to LIMIT_LTM entries on the retrieval
        path to enforce a distinction that no longer has a way to be created.
        """
        return self.cog.storage_manager._load_shard("ltm", user_id, profile_name)

    def _save_ltm_shard(self, user_id: str, profile_name: str, data: Optional[Dict[str, List[Dict]]]):
        if not data or not data.get("guild"):
            self._delete_ltm_shard(user_id, profile_name)
        else:
            self.cog.storage_manager._save_shard("ltm", user_id, data, profile_name)

    def _delete_ltm_shard(self, user_id: str, profile_name: str):
        self.cog.storage_manager._delete_shard("ltm", user_id, profile_name)

    def _copy_ltm_shard(self, user_id: str, source_profile_name: str, new_profile_name: str):
        src_path = self.cog.storage_manager._get_shard_path("ltm", user_id, source_profile_name)
        new_path = self.cog.storage_manager._get_shard_path("ltm", user_id, new_profile_name)
        if os.path.exists(src_path):
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            import shutil
            shutil.copy2(src_path, new_path)

    def _resolve_ltm_storage_target(self, profile_owner_id: int, profile_name: str, session_owner_id: Optional[int]) -> Tuple[int, str]:
        """Resolves which user/profile folder LTM data should be read from or written to,
        redirecting to the borrower's local copy when running inside a borrowed multi-profile session."""
        if session_owner_id and session_owner_id != profile_owner_id:
            recip_index = self.cog.profile_manager._get_user_index(session_owner_id)
            current_pid = self.cog.profile_manager._get_pid_from_name_any(profile_owner_id, profile_name)
            for b_name in recip_index.get("borrowed", []):
                b_config = self.cog.profile_manager._get_profile_config(session_owner_id, b_name, True) or {}
                if int(b_config.get("original_owner_id", 0)) == profile_owner_id and b_config.get("original_profile_id") == current_pid:
                    return session_owner_id, b_name
        return profile_owner_id, profile_name

    async def _add_ltm(self, profile_owner_id: int, profile_name: str, summary: str, summary_embedding_b64: str, guild_id: Optional[int], triggering_user_id: int, user_dn: Optional[str] = None):
        """Appends one memory to a profile's LTM shard.

        Async because the two shard calls below are a Fernet decrypt plus zstd
        decompress plus orjson parse of up to LIMIT_LTM entries, and the same again in
        reverse to write -- tens of milliseconds of blocking work that ran directly on
        the event loop from every caller. The session and profile lookups stay on the
        loop: they read cog-owned dicts the loop itself mutates, so iterating them from
        a worker thread risks a "dictionary changed size during iteration".
        """
        if not guild_id: return

        # Redirect LTM saves to the borrower's folder if running in a borrowed session
        session_owner_id = None
        for channel_id, session in self.cog.multi_profile_channels.items():
            for p in session.get("profiles", []):
                if p.get("owner_id") == profile_owner_id and p.get("profile_name") == profile_name:
                    session_owner_id = session.get("owner_id")
                    break
            if session_owner_id:
                break

        ltm_user_id, ltm_profile_name = self._resolve_ltm_storage_target(profile_owner_id, profile_name, session_owner_id)

        owner_id_str = str(ltm_user_id)
        ltm_data = await asyncio.to_thread(self._load_ltm_shard, owner_id_str, ltm_profile_name)
        if ltm_data is None:
            ltm_data = {"guild": []}

        context_type = "guild"
        ltm_list = ltm_data.get(context_type, [])

        limit = defaultConfig.LIMIT_LTM

        # Entries are appended newest-last and only ever trimmed from the front, so the
        # list is already in timestamp order. The full re-sort this replaces ran a Python
        # key function over up to LIMIT_LTM entries on every write, and that key --
        # `x.get('created_ts', x.get('ts'))` -- evaluated its fallback unconditionally.
        #
        # Trimming to fit happens here too. The clamp that used to follow the append was
        # `max(len(ltm_list), LIMIT_LTM)`, which is LIMIT_LTM for any list at or under
        # the cap: it never dropped anything and only copied the list. That left the
        # single `pop(0)` as the whole cap, so a shard already over it -- profile import
        # writes the bundle's entries without checking -- shed one entry per write
        # forever. One slice deletion trims to fit in a single step.
        overflow = len(ltm_list) - limit + 1
        if overflow > 0:
            del ltm_list[:overflow]

        now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        entry = {
            "id": str(uuid.uuid4())[:8],
            "created_ts": now_ts,
            "modified_ts": now_ts,
            "sum": summary.strip(),
            "s_emb_b64": summary_embedding_b64,
            "usr": user_dn,
            # The guild the memory formed in, and the only place it is ever
            # recalled: _build_ltm_vectors groups rows by this, and retrieval only
            # ever looks at the current guild's group.
            "context_id": str(guild_id)
        }
        ltm_list.append(entry)

        ltm_data[context_type] = ltm_list
        await asyncio.to_thread(self._save_ltm_shard, owner_id_str, ltm_profile_name, ltm_data)

    async def update_ltm(self, profile_owner_id: int, profile_name: str, ltm_id: str, new_summary: str, new_embedding_b64: str) -> bool:
        """Rewrites one memory in place. Async for the same shard-I/O reason as _add_ltm."""
        owner_id_str = str(profile_owner_id)
        ltm_data = await asyncio.to_thread(self._load_ltm_shard, owner_id_str, profile_name)
        if not ltm_data:
            return False

        ltm_list = ltm_data.get("guild",[])
        for i, ltm_entry in enumerate(ltm_list):
            if ltm_entry.get("id") == ltm_id:
                ltm_data["guild"][i]["sum"] = new_summary.strip()
                ltm_data["guild"][i]["s_emb_b64"] = new_embedding_b64
                if "s_emb" in ltm_data["guild"][i]: del ltm_data["guild"][i]["s_emb"]
                ltm_data["guild"][i]["modified_ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                if "kw" in ltm_data["guild"][i]:
                    del ltm_data["guild"][i]["kw"]
                await asyncio.to_thread(self._save_ltm_shard, owner_id_str, profile_name, ltm_data)
                return True
        return False

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
        if os.path.exists(src_path):
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            import shutil
            shutil.copy2(src_path, new_path)

    async def _get_relevant_ltm_for_prompt(self, session_key: Any, history: list, profile_owner_id: int, profile_name: str, msg_content: str, author_dn: str, guild_id: Optional[int], triggering_user_id: int) -> Optional[str]:
        index = self.cog.profile_manager._get_user_index(profile_owner_id)
        is_borrowed = profile_name in index.get("borrowed", [])

        if is_borrowed:
            borrowed_data = self.cog.profile_manager._get_profile_config(profile_owner_id, profile_name, True) or {}
            owner_id = int(borrowed_data.get("original_owner_id", profile_owner_id))
            owner_profile_id = borrowed_data.get("original_profile_id")

            if owner_profile_id:
                # Same dead `config.json.gz` path as the hub carried: the file is
                # profile.json.gz with the config nested inside. Reading the missing
                # path yielded {}, and the `if not params_source: return None` below
                # then returned before retrieval -- so every borrowed profile (each
                # borrow writes original_profile_id) silently recalled no long-term
                # memories at all.
                owner_profile_data = self.cog.profile_manager._get_profile_by_pid(owner_id, owner_profile_id) or {}
                params_source = owner_profile_data.get("config") or {}
            else:
                owner_profile_name = borrowed_data.get("original_profile_name", profile_name)
                params_source = self.cog.profile_manager._get_profile_config(owner_id, owner_profile_name, False) or {}

            params_to_use = borrowed_data
        else:
            params_source = self.cog.profile_manager._get_profile_config(profile_owner_id, profile_name, False) or {}
            params_to_use = params_source

        if not params_source:
            return None

        ltm_context_size = int(params_to_use.get("ltm_context_size", params_source.get("ltm_context_size", 3)))
        ltm_relevance_threshold = float(params_to_use.get("ltm_relevance_threshold", params_source.get("ltm_relevance_threshold", 0.75)))

        if ltm_context_size == 0:
            return None

        # Redirect LTM loads to the borrower's local path if running in a borrowed session
        session_owner_id = None
        if isinstance(session_key, tuple) and len(session_key) > 0:
            channel_id = session_key[0]
            session = self.cog.multi_profile_channels.get(channel_id)
            if session:
                session_owner_id = session.get("owner_id")

        ltm_user_id, ltm_profile_name = self._resolve_ltm_storage_target(profile_owner_id, profile_name, session_owner_id)

        owner_id_str = str(ltm_user_id)
        context_type = "guild"

        # Loaded once, in a thread. This shard read is a file open plus Fernet decrypt plus
        # zstd decompress plus orjson parse; it previously ran here on the event loop and
        # then a second time inside _thread_search_ltm, with the first result discarded.
        # The early exit is kept so an empty shard still skips the embedding round trip.
        ltm_data = await asyncio.to_thread(self._load_ltm_shard, owner_id_str, ltm_profile_name)
        if not ltm_data:
            return None
        all_profile_ltms = ltm_data.get(context_type, [])
        if not all_profile_ltms:
            return None

        prompt_embedding = await self._get_embedding(msg_content, guild_id, task_type="RETRIEVAL_QUERY")
        if not prompt_embedding:
            return None

        current_turn = len(history)
        session_cooldown_history = self.cog.ltm_recall_history.get(session_key, {})

        ltm_shard_path = self.cog.storage_manager._get_shard_path("ltm", owner_id_str, ltm_profile_name)
        try:
            st = os.stat(ltm_shard_path)
            ltm_cache_key = (owner_id_str, ltm_profile_name, st.st_mtime_ns, st.st_size)
        except OSError:
            # No file to stamp (an in-memory or just-deleted shard). Skip the cache
            # rather than risk serving vectors for content that is no longer on disk.
            ltm_cache_key = None

        guild_id_str = str(guild_id)

        def _thread_search_ltm():
            vectors = _cached_vectors(
                _ltm_vec_cache, ltm_cache_key, _LTM_VEC_CACHE_MAX_ROWS,
                _build_ltm_vectors, all_profile_ltms,
            )
            if vectors is None:
                return None

            # Guild membership is grouped at build time, so scoping the shard to the
            # one guild that can be recalled here is a dict lookup, and every array
            # below is already only as wide as that guild.
            guild_idx = vectors.guild_rows.get(guild_id_str)
            if guild_idx is None:
                return None

            # Cooldown is keyed off session state rather than the file, so it cannot be
            # cached -- but it is only ever as large as the set of memories this session
            # has already recalled, so it is walked directly instead of scanned for.
            blocked = []
            for ltm_id, (last_turn, last_sim) in session_cooldown_history.items():
                row = vectors.id_to_row.get(ltm_id)
                if row is not None and current_turn - last_turn < (5 + (1 - last_sim) * 25):
                    blocked.append(row)
            if blocked:
                keep = np.ones(vectors.n, dtype=bool)
                keep[np.array(blocked, dtype=np.intp)] = False
                guild_idx = guild_idx[keep[guild_idx]]
                if guild_idx.size == 0:
                    return None

            prompt_vec = np.array(prompt_embedding, dtype=np.float32)
            prompt_norm = np.linalg.norm(prompt_vec)
            prompt_unit = prompt_vec / (prompt_norm if prompt_norm != 0 else 1e-10)

            # Rows are already unit length, so cosine is a bare dot product.
            sub = vectors.unit[guild_idx]
            similarities = sub @ prompt_unit

            eligible = np.flatnonzero(similarities >= ltm_relevance_threshold)
            if eligible.size == 0:
                return None

            cand_rows = guild_idx[eligible]
            # The one gather the MMR loop needs. The version this replaces re-gathered a
            # shrinking submatrix, then re-penalised and re-sorted a list of tuples in
            # Python on every round: ~1.0 ms against ~0.12 ms here at 800 candidates,
            # and ~85% of the entire retrieval path.
            cand_mat = sub[eligible]
            original_sims = similarities[eligible]
            sub = None  # up to LIMIT_LTM x 256 float32; drop it before the loop runs

            # MMR diversification. Scores are penalised in place and the winner taken by
            # argmax, so nothing is rebuilt or re-sorted per round. A picked row is
            # parked at -inf, which survives the penalty because the factor
            # (1 - cosine * 0.75) is bounded to [0.25, 1.75] and so never flips sign.
            scores = original_sims.astype(np.float32, copy=True)
            picked = []
            for _ in range(min(ltm_context_size, cand_rows.size)):
                best = int(np.argmax(scores))
                picked.append(best)
                scores *= 1.0 - (cand_mat @ cand_mat[best]) * 0.75
                scores[best] = -np.inf

            return [
                {
                    "ltm": all_profile_ltms[vectors.rows[int(cand_rows[p])]],
                    "original_sim": float(original_sims[p]),
                }
                for p in picked
            ]

        final_memories = await asyncio.to_thread(_thread_search_ltm)

        if not final_memories:
            return None

        if session_key not in self.cog.ltm_recall_history:
            self.cog.ltm_recall_history[session_key] = {}

        recalled_summaries = []
        for mem_data in final_memories:
            ltm = mem_data["ltm"]
            self.cog.ltm_recall_history[session_key][ltm['id']] = (current_turn, mem_data["original_sim"])

            decrypted_sum = self.cog.storage_manager._decrypt_data(ltm.get('sum', ''))
            recalled_summaries.append(decrypted_sum)

        if not recalled_summaries:
            return None

        return "<archive_context>\n" + "\n".join(recalled_summaries) + "\n</archive_context>"

    async def _get_relevant_training_examples(self, profile_owner_id: int, profile_name: str, msg_content:str, guild_id: int)->List[str]:
        # [UPDATED] Check context size before disk or API activity
        #
        # Read straight off the profile config. This used to go through
        # _get_user_profile_for_model under asyncio.to_thread, which resolves the
        # effective profile and loads its prompts to build a ten-tuple, of which eight
        # entries were discarded into `_`. Both values wanted here come from the config
        # the first lookup already returns, and every accessor involved reads a
        # cog-owned dict with disk only on a cold miss -- the same trade the rest of
        # this module makes on the loop -- so the thread hop cost more than the work.
        index = self.cog.profile_manager._get_user_index(profile_owner_id)
        is_borrowed = profile_name in index.get("borrowed", [])
        config = self.cog.profile_manager._get_profile_config(profile_owner_id, profile_name, is_borrowed) or {}

        training_context_size = int(config.get("training_context_size", defaultConfig.TRAINING_CONTEXT_SIZE))
        training_relevance_threshold = float(config.get("training_relevance_threshold", defaultConfig.TRAINING_RELEVANCE_THRESHOLD))
        if training_context_size == 0:
            return []

        effective_owner_id_for_training, effective_profile_name_for_training = self.cog.profile_manager._resolve_effective_profile(profile_owner_id, profile_name)

        # Shard first, embedding second. _get_relevant_ltm_for_prompt already orders it
        # this way on purpose -- an empty shard has nothing to rank, so paying for the
        # query embedding before finding that out is a round trip bought for nothing.
        # This path had kept the opposite order, so every profile with training enabled
        # but no examples yet was billed an embedding on every single turn.
        profile_examples = await asyncio.to_thread(
            self._load_training_shard, str(effective_owner_id_for_training), effective_profile_name_for_training
        )
        if not profile_examples:
            return []

        msg_emb = await self._get_embedding(msg_content, guild_id, task_type="RETRIEVAL_QUERY")
        if not msg_emb: return []

        training_shard_path = self.cog.storage_manager._get_shard_path(
            "training", str(effective_owner_id_for_training), effective_profile_name_for_training
        )
        try:
            st = os.stat(training_shard_path)
            train_cache_key = (str(effective_owner_id_for_training), effective_profile_name_for_training,
                               st.st_mtime_ns, st.st_size)
        except OSError:
            train_cache_key = None

        def _thread_search_training():
            vectors = _cached_vectors(
                _train_vec_cache, train_cache_key, _TRAINING_VEC_CACHE_MAX_ROWS,
                _build_training_vectors, profile_examples,
            )
            if vectors is None:
                return []

            prompt_vec = np.array(msg_emb, dtype=np.float32)
            prompt_norm = np.linalg.norm(prompt_vec)
            prompt_unit = prompt_vec / (prompt_norm if prompt_norm != 0 else 1e-10)

            # Rows are already unit length, so cosine is a bare dot product. The list
            # comprehension this replaces called float(similarities[i]) twice per
            # example, then sorted dicts of Python floats.
            similarities = vectors.unit @ prompt_unit

            eligible = np.flatnonzero(similarities >= training_relevance_threshold)
            if eligible.size == 0:
                return []

            # argsort descending, then keep only as many as the caller asked for.
            top = eligible[np.argsort(-similarities[eligible], kind="stable")][:training_context_size]

            decrypt = self.cog.storage_manager._decrypt_data
            out = []
            for row in top:
                ex = profile_examples[vectors.rows[int(row)]]
                out.append(f"<example>\nUser: {decrypt(ex['u_in'])}\nYou: {decrypt(ex['b_out'])}\n</example>")
            return out

        return await asyncio.to_thread(_thread_search_training)

    async def _get_embedding(self, text: str, guild_id: int, task_type: str = "RETRIEVAL_QUERY") -> Optional[List[float]]:
        if not text or not text.strip():
            return None

        api_key = self.cog.storage_manager._get_api_key_for_guild(guild_id)
        if not api_key:
            return None

        return await get_embedding_vector(api_key, text, task_type=task_type, output_dimensionality=256, timeout=5.0)

    async def _generate_ltm_data_from_history(self, hist:list, user_dn:str, gen_config_params: Dict[str, Any], guild_id: Optional[int], bot_dn: str = "Bot", profile_owner_id: int = None, profile_name: str = None, warning_channel: Optional[discord.abc.Messageable] = None) -> Optional[str]:
        """Summarises a slice of history into a long-term memory.

        The summariser model comes from the profile's own `ltm_model`, retried on
        `ltm_fallback_model`. It used to take a `model_name_to_use` argument that both
        callers filled in -- one with the response primary, one with the response
        fallback -- and that nothing in the body has ever read. Removed rather than
        wired: honouring it would have silently moved every LTM summary onto the
        response model.
        """
        if not hist or len(hist) < MIN_HISTORY_FOR_LTM_CREATION: return None

        # [UPDATED] Standardize history for the LTM Summarizer
        # Provides Name [Timestamp] and Content only, stripping metadata and summaries.
        convo_parts = []
        for turn in hist:
            if isinstance(turn, dict) and 'role' in turn:
                display_name = user_dn if turn['role'] == 'user' else bot_dn
                parts = turn.get('parts', [])
                if not parts: continue
                raw_text = "".join(p if isinstance(p, str) else p.get('text', '') for p in parts)
            else:
                raw_text = str(turn)
                display_name = "Unknown" # String-only fallback

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
                            ts_str = datetime.datetime.now(datetime.timezone.utc).strftime("[%a, %d %b %Y, %I:%M %p UTC]")
                            convo_parts.append(f"<{display_name}> {ts_str}:\n{final_content}\n</{display_name}>")
                        else:
                            convo_parts.append(final_content)
            except TimeoutError:
                continue

        convo = "\n\n".join(convo_parts)

        if len(convo) > 3000: # Slightly higher limit for formatted text
            convo = convo[-3000:]

        instructions = self.cog.profile_manager._default_ltm_summarization_instructions()
        # Bound unconditionally: it is only populated for a known profile, and the reads
        # below used to hedge on `'params_source' in locals()` -- which is correct and
        # invisible, so the next edit is one NameError away.
        params_source: Dict[str, Any] = {}
        if profile_owner_id and profile_name:
            user_index = self.cog.profile_manager._get_user_index(profile_owner_id)
            is_borrowed = profile_name in user_index.get("borrowed", [])
            params_source = self.cog.profile_manager._get_profile_config(profile_owner_id, profile_name, is_borrowed) or {}

            source_owner_id, source_profile_name = self.cog.profile_manager._resolve_effective_profile(profile_owner_id, profile_name)
            prompts = self.cog.profile_manager._get_profile_prompts(source_owner_id, source_profile_name) or {}

            encrypted_instructions = prompts.get("ltm_summarization_instructions")
            if encrypted_instructions:
                # A stored-but-blank value used to yield blank instructions.
                instructions = (self.cog.storage_manager._decrypt_data(encrypted_instructions).strip()
                                or self.cog.profile_manager._default_ltm_summarization_instructions())

        cfg = {"temperature": 0.2}

        ltm_model_raw = params_source.get("ltm_model", FALLBACK_MODEL_NAME)
        ltm_fallback_raw = params_source.get("ltm_fallback_model")

        effective_guild_id = guild_id or 0

        status = "api_error"
        try:
            async def _attempt(model_name, _is_fallback):
                m = self.cog.api_service._instantiate_model(
                    model_name, effective_guild_id if effective_guild_id else None,
                    profile_owner_id, instructions, DEFAULT_SAFETY_SETTINGS, {}, None, params_source)
                return await m.generate_content_async(
                    [f"<target_transcript>\n{convo}\n</target_transcript>"], generation_config=cfg)

            r, _used, _was_fallback = await self.cog.api_service.run_with_fallback(
                ltm_model_raw, ltm_fallback_raw, _attempt, label="LTM summariser")
            status = "blocked_by_safety" if not r.candidates else "success"

            response_text = ""
            if r.candidates:
                candidate = r.candidates[0]
                if candidate.content and candidate.content.parts:
                    response_text = "".join(p.text for p in candidate.content.parts if hasattr(p, 'text')).strip()

            if response_text and response_text.upper() != "NO_SUMMARY":
                return response_text
        except Exception as e:
            err_str = str(e)
            if "429" not in err_str and "RESOURCE_EXHAUSTED" not in err_str and "503" not in err_str and "UNAVAILABLE" not in err_str:
                print(f"LTM Gen err {user_dn}: {e}")
                traceback.print_exc()
                if warning_channel:
                    await self.cog.generation_service._send_session_warning(warning_channel, f"Long-Term Memory creation failed ({_format_api_error(e)})")
        return None

    async def _maybe_create_ltm(self, context_obj: Union[discord.Message, discord.abc.Messageable], author_dn: str, hist: list, profile_owner_id: int, profile_name: str, gen_config_params: Dict[str, Any], triggering_user_id_override: Optional[int] = None):
        guild = getattr(context_obj, 'guild', None)
        if not guild: return

        index = self.cog.profile_manager._get_user_index(profile_owner_id)
        is_borrowed = profile_name in index.get("borrowed",[])
        profile_settings = self.cog.profile_manager._get_profile_config(profile_owner_id, profile_name, is_borrowed) or {}

        if not profile_settings.get("ltm_creation_enabled", False):
            return

        ltm_counter_key = (profile_owner_id, profile_name, "guild")
        # 1 exchange = 1 increment
        self.cog.message_counters_for_ltm[ltm_counter_key] = self.cog.message_counters_for_ltm.get(ltm_counter_key, 0) + 1

        interval = profile_settings.get("ltm_creation_interval", 10)
        context_size = profile_settings.get("ltm_summarization_context", 10)

        if self.cog.message_counters_for_ltm[ltm_counter_key] >= interval:
            self.cog.message_counters_for_ltm[ltm_counter_key] = 0
            # Turn history is now consolidated, so context size represents exact chunks
            h_sum = hist[-context_size:]
            if len(h_sum) < 2: # Minimal safety check
                return

            print(f"[DEBUG: LTM] Triggering summary for {profile_name} using {len(h_sum)} context turns.")

            guild_id = None
            channel_id = None
            author = None
            triggering_user_id = None

            if isinstance(context_obj, discord.Message):
                guild_id = context_obj.guild.id if context_obj.guild else None
                channel_id = context_obj.channel.id
                author = context_obj.author
                triggering_user_id = author.id if author else self.cog.bot.user.id
            else: # Is a TextChannel from a child bot
                guild_id = context_obj.guild.id if context_obj.guild else None
                channel_id = context_obj.id
                triggering_user_id = triggering_user_id_override or self.cog.bot.user.id

            _, _, _, temp, top_p, top_k, _, _, _, fallback_model = await asyncio.to_thread(
                self.cog.session_manager._get_user_profile_for_model, profile_owner_id, channel_id, profile_name
            )
            effective_gen_config = {"temperature": temp, "top_p": top_p, "top_k": top_k}

            user_id_map = {triggering_user_id: author_dn}
            sanitized_history, sanitized_author = _get_sanitized_history_and_author(h_sum, user_id_map, triggering_user_id)

            effective_owner_id, effective_profile_name = self.cog.profile_manager._resolve_effective_profile(profile_owner_id, profile_name)

            bot_display_name = effective_profile_name
            appearance = self.cog.user_appearances.get(str(effective_owner_id), {}).get(effective_profile_name, {})
            if appearance.get("custom_display_name"):
                bot_display_name = appearance["custom_display_name"]

            warning_target = context_obj.channel if isinstance(context_obj, discord.Message) else context_obj

            summary = await self._generate_ltm_data_from_history(sanitized_history, sanitized_author, effective_gen_config, guild_id, bot_dn=bot_display_name, profile_owner_id=profile_owner_id, profile_name=profile_name, warning_channel=warning_target)
            if summary:
                summary_embedding = await self._get_embedding(summary, guild_id, task_type="RETRIEVAL_DOCUMENT")
                if summary_embedding:
                    b64_emb = encode_embedding_b64(summary_embedding)
                    await self._add_ltm(profile_owner_id, profile_name, summary, b64_emb, guild.id if guild else None, triggering_user_id, sanitized_author)

    async def add_new_training_example(self, profile_owner_id: int, profile_name: str, usr_in:str, bot_out:str, guild_id: int)->Tuple[bool,str]:
        if not usr_in.strip() or not bot_out.strip(): return False,"Inputs empty."

        # [NEW] Training Limit Check
        owner_id_str = str(profile_owner_id)
        training_shard = await asyncio.to_thread(self._load_training_shard, owner_id_str, profile_name) or []

        limit = defaultConfig.LIMIT_TRAINING

        if len(training_shard) >= limit:
            return False, f"**Limit Reached.**\n\nYou have reached the maximum of **{limit}** training examples."

        emb=await self._get_embedding(usr_in, guild_id, task_type="RETRIEVAL_DOCUMENT")
        if not emb: return False,"Embedding failed. Ensure the server API key is valid."

        b64_emb = encode_embedding_b64(emb)
        now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        entry={"id":str(uuid.uuid4())[:8],"created_ts":now_ts, "modified_ts": now_ts, "u_in":usr_in.strip(),"b_out":bot_out.strip(),"u_emb_b64":b64_emb}
        training_shard.append(entry)

        await asyncio.to_thread(self._save_training_shard, owner_id_str, profile_name, training_shard)
        return True,f"Example added for profile '{profile_name}'. Total: {len(training_shard)}/{limit}"

    async def update_training_example(self, profile_owner_id: int, profile_name: str, example_id: str, new_user_input: str, new_bot_response: str, guild_id: int) -> Tuple[bool, str]:
        if not new_user_input.strip() or not new_bot_response.strip():
            return False, "Inputs cannot be empty."

        owner_id_str = str(profile_owner_id)
        example_list = await asyncio.to_thread(self._load_training_shard, owner_id_str, profile_name)
        if example_list is None:
            return False, f"No training examples found for profile '{profile_name}'."

        example_found = False
        for i, example in enumerate(example_list):
            if example.get("id") == example_id:
                new_embedding = await self._get_embedding(new_user_input, guild_id, task_type="RETRIEVAL_DOCUMENT")
                if not new_embedding:
                    return False, "Failed to generate embedding for the new input. The example was not updated."

                b64_emb = encode_embedding_b64(new_embedding)

                example_list[i]["u_in"] = new_user_input.strip()
                example_list[i]["b_out"] = new_bot_response.strip()
                example_list[i]["u_emb_b64"] = b64_emb
                if "u_emb" in example_list[i]: del example_list[i]["u_emb"]
                example_list[i]["modified_ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

                await asyncio.to_thread(self._save_training_shard, owner_id_str, profile_name, example_list)
                example_found = True
                break

        if example_found:
            return True, f"Successfully updated training example `{example_id}` for profile '{profile_name}'."
        else:
            return False, f"Could not find a training example with ID `{example_id}` for profile '{profile_name}'."

    async def _execute_training_analysis(self, interaction: discord.Interaction, profile_name: str, count: int, verbosity: int, model_name: str):
        user_id = interaction.user.id
        user_id_str = str(user_id)
        
        examples = self._load_training_shard(user_id_str, profile_name) or []
        if not examples:
            await interaction.followup.send("❌ No training examples found to analyse.", ephemeral=True); return
        
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
            index = self.cog.profile_manager._get_user_index(user_id)
            p_is_b = profile_name in index.get("borrowed", [])
            p_cfg = self.cog.profile_manager._get_profile_config(user_id, profile_name, p_is_b) or {}
            
            model = self.cog.api_service._instantiate_model(model_name, interaction.guild_id, user_id, None, None, {}, None, p_cfg)
            resp = await model.generate_content_async([prompt])
            response_text = resp.text

            if not response_text: raise ValueError("Model returned an empty response.")
            
            # Save to Slot 4 (Index 3)
            index = self.cog.profile_manager._get_user_index(user_id)
            if profile_name in index.get("personal", []):
                prompts = self.cog.profile_manager._get_profile_prompts(user_id, profile_name) or {}
                
                # Ensure ai_instructions is a list of 4
                if not isinstance(prompts.get("ai_instructions"), list):
                    prompts["ai_instructions"] = [prompts.get("ai_instructions", ""), "", "", ""]
                while len(prompts["ai_instructions"]) < 4:
                    prompts["ai_instructions"].append("")
                
                prompts["ai_instructions"][3] = self.cog.storage_manager._encrypt_data(response_text[:4000])
                self.cog.profile_manager._save_profile_prompts(user_id, profile_name, prompts)
                
                await interaction.followup.send(f"✅ **Analysis Complete.** Style guide saved to AI Instructions for '{profile_name}'.", ephemeral=True)
            else:
                await interaction.followup.send("❌ Profile not found.", ephemeral=True)

        except Exception as e:
            await interaction.followup.send(f"❌ **Analysis Failed:** {e}", ephemeral=True)

    async def bulk_reset_examples(self, user_id: int, profile_names: List[str]) -> str:
        user_id_str = str(user_id)
        reset_count = 0
        for name in profile_names:
            pid = self.cog.profile_manager._get_pid_from_name_any(user_id, name)
            shard_path = os.path.join(self.cog.USERS_DIR, user_id_str, "profiles", pid, "training.json.gz")
            if os.path.exists(shard_path):
                self._delete_training_shard(user_id_str, name)
                reset_count += 1
        
        return f"Reset all training examples for {reset_count} profile(s)."

    async def bulk_reset_ltm(self, user_id: int, profile_names: List[str]) -> str:
        user_id_str = str(user_id)
        reset_count = 0
        for name in profile_names:
            pid = self.cog.profile_manager._get_pid_from_name_any(user_id, name)
            shard_path = os.path.join(self.cog.USERS_DIR, user_id_str, "profiles", pid, "ltm.json.gz")
            if os.path.exists(shard_path):
                self._delete_ltm_shard(user_id_str, name)
                reset_count += 1
        
        return f"Reset all Long-Term Memories for {reset_count} profile(s)."
    
