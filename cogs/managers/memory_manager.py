import os
import re
import uuid
import base64
import asyncio
import datetime
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import discord

from google.genai import types

from ..utils.constants import (
    defaultConfig, FALLBACK_MODEL_NAME, DEFAULT_SAFETY_SETTINGS,
    DEFAULT_LTM_SUMMARIZATION_INSTRUCTIONS, MIN_HISTORY_FOR_LTM_CREATION,
    DEFAULT_TRAINING_ANALYST_PROMPT,
)
from ..utils.helpers import Timeout, _format_api_error, _get_sanitized_history_and_author
from .storage_manager import IOManager
from ..services.api_service import get_genai_client


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    np_vec1=np.array(vec1); np_vec2=np.array(vec2); dot=np.dot(np_vec1,np_vec2); n1=np.linalg.norm(np_vec1); n2=np.linalg.norm(np_vec2)
    return 0.0 if n1==0 or n2==0 else float(dot/(n1*n2))

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
        data = self.cog.storage_manager._load_shard("ltm", user_id, profile_name)
        if data:
            # Purge legacy non-server memories upon load
            guild_ltms =[item for item in data.get("guild", []) if item.get("scope") == "server"]
            data["guild"] = guild_ltms
            if "dm" in data:
                del data["dm"]
        return data

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

    def _add_ltm(self, profile_owner_id: int, profile_name: str, summary: str, summary_embedding_b64: str, guild_id: Optional[int], triggering_user_id: int, user_dn: Optional[str] = None):
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
        ltm_data = self._load_ltm_shard(owner_id_str, ltm_profile_name)
        if ltm_data is None:
            ltm_data = {"guild": []}

        context_type = "guild"
        ltm_list = ltm_data.get(context_type, [])

        limit = defaultConfig.LIMIT_LTM

        ltm_list.sort(key=lambda x: x.get('created_ts', x.get('ts')))

        if len(ltm_list) >= limit:
            ltm_list.pop(0)

        now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        entry = {
            "id": str(uuid.uuid4())[:8],
            "created_ts": now_ts,
            "modified_ts": now_ts,
            "sum": summary.strip(),
            "s_emb_b64": summary_embedding_b64,
            "usr": user_dn,
            "scope": "server",
            "context_id": str(guild_id)
        }
        ltm_list.append(entry)

        # ltm_list is already sorted (line above) and the new entry's timestamp is the newest,
        # so it stays sorted after appending — no need to re-sort before clamping.
        max_safe_clamp = max(len(ltm_list), defaultConfig.LIMIT_LTM)
        ltm_data[context_type] = ltm_list[-max_safe_clamp:]
        self._save_ltm_shard(owner_id_str, ltm_profile_name, ltm_data)

    def update_ltm(self, profile_owner_id: int, profile_name: str, ltm_id: str, new_summary: str, new_embedding_b64: str) -> bool:
        owner_id_str = str(profile_owner_id)
        ltm_data = self._load_ltm_shard(owner_id_str, profile_name)
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
                self._save_ltm_shard(owner_id_str, profile_name, ltm_data)
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
                path = os.path.join(self.cog.USERS_DIR, str(owner_id), "profiles", owner_profile_id, "config.json.gz")
                params_source = IOManager.read_json_gzip(path, self.cog.fernet) or {}
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
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        session_cooldown_history = self.cog.ltm_recall_history.get(session_key, {})

        def _thread_search_ltm():
            valid_ltms = []
            b64_embs = []

            for ltm in all_profile_ltms:
                ltm_id = ltm.get('id')
                if ltm_id in session_cooldown_history:
                    last_turn, last_sim = session_cooldown_history[ltm_id]
                    if current_turn - last_turn < (5 + (1 - last_sim) * 25): continue

                if str(ltm.get('context_id')) != str(guild_id): continue

                if "s_emb_b64" in ltm:
                    valid_ltms.append(ltm)
                    b64_embs.append(ltm["s_emb_b64"])

            if not valid_ltms: return None

            raw_bytes = b"".join(base64.b64decode(s) for s in b64_embs)
            emb_matrix = np.frombuffer(raw_bytes, dtype=np.float16).reshape(len(b64_embs), -1).astype(np.float32)
            prompt_vec = np.array(prompt_embedding, dtype=np.float32)

            matrix_norms = np.linalg.norm(emb_matrix, axis=1)
            prompt_norm = np.linalg.norm(prompt_vec)
            matrix_norms[matrix_norms == 0] = 1e-10
            prompt_norm = prompt_norm if prompt_norm != 0 else 1e-10

            raw_similarities = np.dot(emb_matrix, prompt_vec) / (matrix_norms * prompt_norm)
            candidate_indices = []

            for i, ltm in enumerate(valid_ltms):
                sim = float(raw_similarities[i])
                ts_str = ltm.get('created_ts') or ltm.get('ts')
                if ts_str:
                    try:
                        days_old = (now_utc - datetime.datetime.fromisoformat(ts_str)).total_seconds() / 86400.0
                        sim *= (0.995 ** days_old)
                    except Exception:
                        pass

                if sim >= ltm_relevance_threshold:
                    candidate_indices.append((i, sim, float(raw_similarities[i])))

            if not candidate_indices: return None
            candidate_indices.sort(key=lambda x: x[1], reverse=True)

            # High-throughput vectorised MMR diversification (Zero Base64 re-decoding)
            final_memories = []
            current_candidates = list(candidate_indices)

            while len(final_memories) < ltm_context_size and current_candidates:
                best_idx, best_sim, orig_sim = current_candidates.pop(0)
                final_memories.append({"ltm": valid_ltms[best_idx], "sim": best_sim, "original_sim": orig_sim})
                if not current_candidates: break

                best_vec = emb_matrix[best_idx]
                best_norm_val = matrix_norms[best_idx]

                rem_idxs = [c[0] for c in current_candidates]
                rem_matrix = emb_matrix[rem_idxs]
                rem_norms = matrix_norms[rem_idxs]

                # Instantaneous BLAS batch projection across remaining candidates
                inter_sims = np.dot(rem_matrix, best_vec) / (rem_norms * best_norm_val)

                updated = []
                for j, (c_idx, c_sim, o_sim) in enumerate(current_candidates):
                    penalised_sim = c_sim * (1.0 - (float(inter_sims[j]) * 0.75))
                    updated.append((c_idx, penalised_sim, o_sim))

                updated.sort(key=lambda x: x[1], reverse=True)
                current_candidates = updated

            return final_memories

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
        _, _, _, _, _, _, training_context_size, training_relevance_threshold, _, _ = await asyncio.to_thread(
            self.cog.session_manager._get_user_profile_for_model, profile_owner_id, 0, profile_name
        )
        if training_context_size == 0:
            return []

        owner_id_str = str(profile_owner_id)
        effective_owner_id_for_training, effective_profile_name_for_training = self.cog.profile_manager._resolve_effective_profile(profile_owner_id, profile_name)

        msg_emb = await self._get_embedding(msg_content, guild_id, task_type="RETRIEVAL_QUERY")
        if not msg_emb: return []

        def _thread_search_training():
            profile_examples = self._load_training_shard(str(effective_owner_id_for_training), effective_profile_name_for_training)
            if not profile_examples: return []

            valid_ex = []
            b64_embs = []
            for ex in profile_examples:
                if "u_emb_b64" in ex:
                    valid_ex.append(ex)
                    b64_embs.append(ex["u_emb_b64"])

            if not valid_ex: return []

            similarities = calculate_similarities(msg_emb, b64_embs)

            sc = [{"ex": ex, "sim": float(similarities[i])} for i, ex in enumerate(valid_ex) if float(similarities[i]) >= training_relevance_threshold]
            sc.sort(key=lambda x: x["sim"], reverse=True)

            return [f"<example>\nUser: {self.cog.storage_manager._decrypt_data(i['ex']['u_in'])}\nYou: {self.cog.storage_manager._decrypt_data(i['ex']['b_out'])}\n</example>" for i in sc[:training_context_size]]

        return await asyncio.to_thread(_thread_search_training)

    async def _get_embedding(self, text: str, guild_id: int, task_type: str = "RETRIEVAL_QUERY") -> Optional[List[float]]:
        if not text or not text.strip():
            return None

        api_key = self.cog.storage_manager._get_api_key_for_guild(guild_id)
        if not api_key:
            return None

        client = get_genai_client(api_key)

        try:
            result = await asyncio.wait_for(
                client.aio.models.embed_content(
                    model='gemini-embedding-001',
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=256
                    )
                ),
                timeout=5.0
            )
            return result.embeddings[0].values
        except Exception as e:
            print(f"Embedding err for '{text[:30]}...': {e}")
            return None

    async def _generate_ltm_data_from_history(self, hist:list, user_dn:str, gen_config_params: Dict[str, Any], model_name_to_use: str, guild_id: Optional[int], bot_dn: str = "Bot", profile_owner_id: int = None, profile_name: str = None, warning_channel: Optional[discord.abc.Messageable] = None) -> Optional[str]:
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

        instructions = DEFAULT_LTM_SUMMARIZATION_INSTRUCTIONS
        if profile_owner_id and profile_name:
            user_index = self.cog.profile_manager._get_user_index(profile_owner_id)
            is_borrowed = profile_name in user_index.get("borrowed", [])
            params_source = self.cog.profile_manager._get_profile_config(profile_owner_id, profile_name, is_borrowed) or {}

            source_owner_id, source_profile_name = self.cog.profile_manager._resolve_effective_profile(profile_owner_id, profile_name)
            prompts = self.cog.profile_manager._get_profile_prompts(source_owner_id, source_profile_name) or {}

            encrypted_instructions = prompts.get("ltm_summarization_instructions", self.cog.storage_manager._encrypt_data(DEFAULT_LTM_SUMMARIZATION_INSTRUCTIONS))
            instructions = self.cog.storage_manager._decrypt_data(encrypted_instructions)

        cfg = {"temperature": 0.2}

        # [NEW] Utility Routing Logic for LTM
        ltm_model_raw = FALLBACK_MODEL_NAME
        if profile_owner_id and profile_name:
            ltm_model_raw = params_source.get("ltm_model", FALLBACK_MODEL_NAME) if 'params_source' in locals() else FALLBACK_MODEL_NAME

        effective_guild_id = guild_id or 0

        status = "api_error"
        try:
            m = self.cog.api_service._instantiate_model(ltm_model_raw, effective_guild_id if effective_guild_id else None, profile_owner_id, instructions, DEFAULT_SAFETY_SETTINGS, {}, None, params_source if 'params_source' in locals() else {})

            r = await m.generate_content_async([f"<target_transcript>\n{convo}\n</target_transcript>"], generation_config=cfg)
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

            summary = await self._generate_ltm_data_from_history(sanitized_history, sanitized_author, effective_gen_config, fallback_model, guild_id, bot_dn=bot_display_name, profile_owner_id=profile_owner_id, profile_name=profile_name, warning_channel=warning_target)
            if summary:
                summary_embedding = await self._get_embedding(summary, guild_id, task_type="RETRIEVAL_DOCUMENT")
                if summary_embedding:
                    b64_emb = encode_embedding_b64(summary_embedding)
                    self._add_ltm(profile_owner_id, profile_name, summary, b64_emb, guild.id if guild else None, triggering_user_id, sanitized_author)

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
    
