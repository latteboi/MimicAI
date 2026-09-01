import re
import datetime
import discord
from zoneinfo import ZoneInfo
from typing import Any, Optional, Dict, List, Tuple

from ...utils.constants import (
    defaultConfig, PRIMARY_MODEL_NAME, FALLBACK_MODEL_NAME,
    DEFAULT_SYSTEM_INSTRUCTION, DEFAULT_CONTEXT_RULES, DEFAULT_NEURO_INSTRUCTION,
    DEFAULT_TRAINING_DATA_INJECTION, DEFAULT_TIME_CONTEXT, DEFAULT_NEGATIVE_CONSTRAINTS,
    DEFAULT_CONTENT_POLICY,
)
from ...utils.helpers import Timeout


class PromptBuilderMixin:
    """Persona/system-instruction assembly and the neuro-state extraction that
    reads the model's <neuro_update> block back out of its response text.
    """

    def _resolve_appearance_data(self, owner_id: int, profile_name: str) -> Tuple[str, str]:
        app = self.cog.profile_manager._get_user_appearance(owner_id, profile_name)
        display_name = app.get("custom_display_name") or profile_name
        avatar_url = app.get("custom_avatar_url") or (self.cog.bot.user.display_avatar.url if self.cog.bot.user else "")
        return display_name, avatar_url

    @staticmethod
    def _resolve_native_tools(p_settings: Dict[str, Any]) -> Optional[List[Dict]]:
        """The `tools` list an instantiated model needs for native grounding/URL context.

        Three copies of this lived inline -- the worker's primary, the worker's fallback
        and regeneration's fallback -- each re-deriving the same legacy coercions
        (`grounding_mode` was once a bool and once "on"/"on+"; `url_mode` reads off the
        older `url_fetching_enabled` flag when absent). Regeneration's primary had no
        copy at all, so a profile on native grounding regenerated without the tool.
        """
        grounding_mode = p_settings.get("grounding_mode", "off")
        if isinstance(grounding_mode, bool):
            grounding_mode = "rag" if grounding_mode else "off"
        elif grounding_mode in ("on", "on+"):
            grounding_mode = "rag"

        url_mode = p_settings.get("url_mode", "off")
        if "url_mode" not in p_settings:
            url_mode = "rag" if p_settings.get("url_fetching_enabled", False) else "off"

        tools = []
        if grounding_mode == "native":
            tools.append({"google_search": {}})
        if url_mode == "native":
            tools.append({"url_context": {}})
        return tools or None

    def _channel_allows_adult_content(self, channel_id: int) -> bool:
        """True only when the destination channel is flagged age-restricted.

        DMs, group channels and anything the gateway cache cannot resolve count as
        not age-restricted -- the same direction _check_unrestricted_safety_policy
        already fails in, so the two agree on every channel type.

        get_channel is an in-memory cache hit, so this is safe on the turn path.
        """
        channel = self.cog.bot.get_channel(channel_id)
        if not isinstance(channel, (discord.TextChannel, discord.Thread, discord.VoiceChannel)):
            return False
        try:
            return channel.is_nsfw()
        except Exception:
            return False

    async def _send_session_warning(self, channel: discord.abc.Messageable, message: str):
        if not channel: return
        try:
            await channel.send(f"⚠️ **Session Notice:** {message}", delete_after=10)
        except Exception:
            pass

    def _construct_system_instructions(self, profile_owner_id: Optional[int], profile_name_to_use: str, channel_id: int, is_multi_profile: bool = False, training_examples_list: Optional[List[str]] = None, recalled_ltm: Optional[str] = None, critic_constraints: Optional[str] = None) -> Tuple[str, bool, bool, float, float, int, str, str]:
        persona_data: Dict[str, List[str]] = {}
        # profile_owner_id is Optional, but profile_data is read unconditionally below.
        profile_data: Dict[str, Any] = {}
        ai_instr_str: str = ""
        grounding_enabled = False
        temperature = defaultConfig.GEMINI_TEMPERATURE
        top_p = defaultConfig.GEMINI_TOP_P
        top_k = defaultConfig.GEMINI_TOP_K
        primary_model = PRIMARY_MODEL_NAME
        fallback_model = FALLBACK_MODEL_NAME
        time_tracking_enabled = False
        timezone_str = "UTC"
        neuro_enabled = False
        neuro_state = {"dopamine": 50, "cortisol": 20, "oxytocin": 50, "adrenaline": 20}

        if profile_owner_id is not None:
            user_index = self.cog.profile_manager._get_user_index(profile_owner_id)
            is_borrowed = profile_name_to_use in user_index.get("borrowed",[])
            profile_data = self.cog.profile_manager._get_profile_config(profile_owner_id, profile_name_to_use, is_borrowed) or {}

            persona_data, ai_instr_str, grounding_enabled, temperature, top_p, top_k, _, _, primary_model, fallback_model = self.cog.session_manager._get_user_profile_for_model(profile_owner_id, channel_id, profile_name_to_use)

        if profile_data:
            time_tracking_enabled = profile_data.get("time_tracking_enabled", False)
            timezone_str = profile_data.get("timezone", "UTC")
            neuro_enabled = profile_data.get("neuro_engine_enabled", False)
            neuro_state = profile_data.get("neuro_state", {"dopamine": 50, "cortisol": 20, "oxytocin": 50, "adrenaline": 20})

        final_instr_parts =[]

        if is_multi_profile:
            session = self.cog.multi_profile_channels.get(channel_id)
            if session and session.get("session_prompt"):
                final_instr_parts.append(f"<scene_prompt>\n{session['session_prompt']}\n</scene_prompt>")

            # Standing context, not a history turn: the synopsis summarises turns that
            # have already left the STM window, so competing for a slot inside that
            # window would hide it from exactly the long sessions it exists for. Shared
            # by the whole cast -- only public turns are ever compacted, so it can carry
            # nothing a participant was not already entitled to see.
            synopsis = self.cog.session_manager.get_latest_synopsis(session)
            if synopsis:
                final_instr_parts.append(f"<session_synopsis>\n{synopsis}\n</session_synopsis>")

            # Standing context for the same reason, and injected here rather than into
            # the game's own call so that *every* generation in the channel sees it --
            # a seated character answering ordinary chatter mid-hand knows what it just
            # played, which is what removed the need to bench the cast during a game.
            # Returns None on the overwhelmingly common no-game path, for one dict get.
            game_block = self.cog.game_service.context_block(channel_id)
            if game_block:
                final_instr_parts.append(f"<game_context>\n{game_block}\n</game_context>")

        if neuro_enabled:
            neuro_block = self.cog.global_prompts.get("NEURO_ENGINE", DEFAULT_NEURO_INSTRUCTION).format(
                d=neuro_state.get('dopamine', 50),
                c=neuro_state.get('cortisol', 20),
                o=neuro_state.get('oxytocin', 50),
                a=neuro_state.get('adrenaline', 20)
            )
            final_instr_parts.append(neuro_block)

        if time_tracking_enabled:
            time_template = self.cog.global_prompts.get("TIME_CONTEXT", DEFAULT_TIME_CONTEXT)
            try:
                from ...utils.helpers import _resolve_zoneinfo
                tz, _ = _resolve_zoneinfo(timezone_str)
                now = datetime.datetime.now(tz)
                time_str = now.strftime("%A, %d %B %Y, %I:%M %p (%Z)")
                final_instr_parts.append(time_template.format(time_str=time_str))
            except Exception as e:
                print(f"Error processing timezone '{timezone_str}': {e}. Defaulting to UTC.")
                now_utc = datetime.datetime.now(datetime.timezone.utc)
                time_str_utc = now_utc.strftime("%A, %d %B %Y, %I:%M %p (UTC)")
                final_instr_parts.append(time_template.format(time_str=time_str_utc))

        if persona_data and any(persona_data.values()):
            persona_blocks = []
            for key in self.cog.persona_modal_sections_order:
                if lines := persona_data.get(key,[]):
                    decrypted_lines = [self.cog.storage_manager._decrypt_data(line).strip() for line in lines if line.strip()]
                    if any(l.strip() for l in decrypted_lines):
                        block_content = "\n".join(decrypted_lines)
                        persona_blocks.append(f"<{key}>\n{block_content}\n</{key}>")

            if persona_blocks:
                persona_str = "<persona_profile>\n" + "\n\n".join(persona_blocks) + "\n</persona_profile>"
                final_instr_parts.append(persona_str)

        current_instructions_str = "\n\n".join(final_instr_parts).strip()

        decrypted_parts = []
        if isinstance(ai_instr_str, list):
            for part in ai_instr_str:
                dec = self.cog.storage_manager._decrypt_data(part)
                if dec.strip():
                    cleaned_part = "\n".join([line.strip() for line in dec.split("\n")])
                    decrypted_parts.append(cleaned_part)
        elif isinstance(ai_instr_str, str):
            dec = self.cog.storage_manager._decrypt_data(ai_instr_str)
            if dec.strip():
                cleaned_part = "\n".join([line.strip() for line in dec.split("\n")])
                decrypted_parts.append(cleaned_part)

        if decrypted_parts:
            if current_instructions_str: current_instructions_str += "\n\n"
            current_instructions_str += "<instructions>\n"
            current_instructions_str += "\n\n".join(decrypted_parts).strip()
            current_instructions_str += "\n</instructions>"

        if training_examples_list:
            examples_block = "\n---\n".join(training_examples_list)
            training_prompt = self.cog.global_prompts.get("TRAINING_DATA_INJECTION", DEFAULT_TRAINING_DATA_INJECTION).format(examples_block=examples_block)
            current_instructions_str += "\n\n" + training_prompt

        if recalled_ltm:
            current_instructions_str += f"\n\n{recalled_ltm}"

        if critic_constraints:
            constraints_block = self.cog.global_prompts.get("NEGATIVE_CONSTRAINTS", DEFAULT_NEGATIVE_CONSTRAINTS)
            current_instructions_str += "\n\n" + constraints_block.format(constraints=critic_constraints)

        rule_block = self.cog.global_prompts.get("CONTEXT_RULES", DEFAULT_CONTEXT_RULES)

        # [NEW] Dynamically inject the profile's ID into the context rules
        profile_id_val = self.cog.profile_manager._get_profile_id(profile_owner_id, profile_name_to_use)
        rule_block = rule_block.format(profile_id_placeholder=profile_id_val)

        current_instructions_str += "\n\n" + rule_block.strip()

        # Channel-level content shaping, gated on the destination rather than the
        # profile: an Adult-rated profile is already confined to age-restricted
        # channels by _check_unrestricted_safety_policy, so anything that reaches a
        # general channel should be written for one. _resolve_safety_settings keys
        # the provider thresholds off the same channel, so the two content controls
        # now move together. Appended last for recency, and it is the only one of
        # them with any effect on OpenRouter and Ollama, which ignore
        # safety_settings entirely.
        if not self._channel_allows_adult_content(channel_id):
            policy_block = self.cog.global_prompts.get("CONTENT_POLICY", DEFAULT_CONTENT_POLICY).strip()
            if policy_block:
                current_instructions_str += "\n\n" + policy_block

        final_system_instruction = current_instructions_str if current_instructions_str.strip() else DEFAULT_SYSTEM_INSTRUCTION
        return final_system_instruction, False, grounding_enabled, temperature, top_p, top_k, primary_model, fallback_model

    def _extract_and_apply_neuro_state(self, raw_text: str, owner_id: int, profile_name: str) -> Tuple[str, Optional[Dict[str, int]]]:
        xml_pattern = r'<neuro_update>\s*(.*?)\s*</neuro_update>'
        data_str = None
        clean_text = raw_text

        try:
            with Timeout(seconds=1, error_message="Neuro extraction timed out"):
                match = re.search(xml_pattern, raw_text, flags=re.IGNORECASE | re.DOTALL)
                if match:
                    data_str = match.group(1)
                    clean_text = re.sub(xml_pattern, '', raw_text, flags=re.IGNORECASE | re.DOTALL)
                else:
                    relaxed_pattern = r'(?:D:\d{1,3}\s*\|\s*C:\d{1,3}\s*\|\s*O:\d{1,3}\s*\|\s*A:\d{1,3})'
                    match = re.search(relaxed_pattern, raw_text, flags=re.IGNORECASE)
                    if match:
                        data_str = match.group(0)
                        clean_text = re.sub(relaxed_pattern, '', raw_text, flags=re.IGNORECASE)
        except TimeoutError:
            return raw_text.strip(), None

        if not data_str:
            return raw_text.strip(), None

        new_state = {}
        # Normalise separators for splitting
        normalised_data = data_str.replace('|', ':').replace(' ', '')
        kv_pairs = normalised_data.split(':')

        # Iterating pairs (K, V)
        for i in range(0, len(kv_pairs) - 1, 2):
            k = kv_pairs[i].strip().upper()
            try:
                v = int(kv_pairs[i+1].strip())
                v = max(0, min(100, v))
                if k == 'D': new_state['dopamine'] = v
                elif k == 'C': new_state['cortisol'] = v
                elif k == 'O': new_state['oxytocin'] = v
                elif k == 'A': new_state['adrenaline'] = v
            except (ValueError, IndexError):
                continue

        clean_text = clean_text.strip()
        final_state = None
        if new_state:
            index = self.cog.profile_manager._get_user_index(owner_id)
            is_borrowed = profile_name in index.get("borrowed", [])
            p_config = self.cog.profile_manager._get_profile_config(owner_id, profile_name, is_borrowed)

            if p_config and p_config.get("neuro_engine_enabled"):
                current_state = p_config.get("neuro_state", {"dopamine": 50, "cortisol": 20, "oxytocin": 50, "adrenaline": 20}).copy()
                current_state.update(new_state)
                p_config["neuro_state"] = current_state
                self.cog.profile_manager._save_profile_config(owner_id, profile_name, p_config, is_borrowed)
                final_state = current_state

        return clean_text, final_state
