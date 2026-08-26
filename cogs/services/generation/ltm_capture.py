import asyncio
from typing import Any, Dict, List, Optional, Tuple

import discord

from ...utils.constants import defaultConfig
from ...managers.memory_manager import encode_embedding_b64


class LtmCaptureMixin:
    """Owns the summarise -> embed -> store chain for one participant's long-term memory.

    Round-end wraps this in asyncio.create_task so a slow summary never blocks the queue.
    /memorise awaits it directly instead, since it has to report per-profile success back
    to the admin who ran it -- a fire-and-forget task can't do that.
    """

    async def _summarize_and_store_ltm(
        self, channel_id: int, session: Dict[str, Any], owner_id: int, profile_name: str,
        p_settings: Dict[str, Any], guild_id: Optional[int], r_author: str,
        triggering_user_id: int, warning_channel: Optional[discord.abc.Messageable] = None,
    ) -> Tuple[bool, str]:
        context_size = p_settings.get("ltm_summarization_context", 10)

        # The STM floor keeps the summarisation window governed by ltm_summarization_context
        # rather than by this profile's STM length, which an unbounded shadow history would
        # otherwise do.
        ltm_p_settings = dict(p_settings)
        ltm_p_settings["stm_length"] = max(
            int(p_settings.get("stm_length", defaultConfig.CHATBOT_MEMORY_LENGTH)),
            context_size * 2,
        )

        ltm_bot_pid = self.cog.profile_manager._get_pid_from_name_any(owner_id, profile_name)
        profile_order_len = len(session.get("profiles", [])) or 1
        ltm_history = self.cog.session_manager._build_history_for_participant(
            session.get("unified_log", []), ltm_bot_pid, ltm_p_settings, profile_order_len
        )
        if len(ltm_history) < 2:
            return False, "not enough history"

        # Turn history is consolidated
        events_for_summary: List[str] = []
        for turn in ltm_history[-context_size:]:
            parts = turn.get('parts', [])
            if parts:
                text_val = "\n".join(p if isinstance(p, str) else p.get('text', '') for p in parts)
                events_for_summary.append(text_val)

        _, _, _, temp, top_p, top_k, primary_model, _ = await asyncio.to_thread(
            self._construct_system_instructions, owner_id, profile_name, channel_id, is_multi_profile=True
        )
        ltm_d = await self.cog.memory_manager._generate_ltm_data_from_history(
            events_for_summary, r_author, {"temperature": temp, "top_p": top_p, "top_k": top_k},
            guild_id, profile_owner_id=owner_id, profile_name=profile_name, warning_channel=warning_channel,
        )
        if not ltm_d:
            return False, "summarisation produced nothing"

        summary_embedding = await self.cog.memory_manager._get_embedding(ltm_d, guild_id, task_type="RETRIEVAL_DOCUMENT")
        if not summary_embedding:
            return False, "embedding failed"

        b64_emb = encode_embedding_b64(summary_embedding)
        await self.cog.memory_manager._add_ltm(owner_id, profile_name, ltm_d, b64_emb, guild_id, triggering_user_id, r_author)

        # Link LTM creation to the turn metadata for trace transparency
        last_turn = next((t for t in reversed(session.get("unified_log", [])) if t.get("speaker_pid") == ltm_bot_pid), None)
        if last_turn and "meta" in last_turn:
            last_turn["meta"]["ltm_created"] = True

        return True, "created"
