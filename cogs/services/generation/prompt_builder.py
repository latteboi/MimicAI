import re
import datetime
import itertools
import discord
from typing import Any, Optional, Dict, List, Sequence, Tuple

from ...utils.constants import (
    defaultConfig, PRIMARY_MODEL_NAME, FALLBACK_MODEL_NAME,
    DEFAULT_SYSTEM_INSTRUCTION, DEFAULT_SESSION_RULES, DEFAULT_NEURO_INSTRUCTION,
    NEURO_AXES,
    DEFAULT_TRAINING_DATA_INJECTION, DEFAULT_CURRENT_TIME, DEFAULT_NEGATIVE_CONSTRAINTS,
    DEFAULT_CONTENT_POLICY, DEFAULT_BIRTHDAY_CONTEXT,
)
from ...utils.birthdays import birthday_offset, describe_birthday
from ...utils.helpers import (TURN_TIME_FORMAT, Timeout, _get_user_hash,
                              default_profile_avatar_url)
from ._shared import NEURO_BARE_PATTERN, NEURO_TAG_PATTERN

#: One axis as a model writes it inside the tag: a letter or the whole name, `:` or `=`,
#: then the number. Found one at a time, so the separators between them are free.
_NEURO_AXIS = re.compile(r'\b(dopamine|cortisol|oxytocin|adrenaline|[DCOA])\s*[:=]\s*(\d{1,3})\b',
                         re.IGNORECASE)
_AXIS_BY_KEY = {**{axis: axis for axis in NEURO_AXES}, **{axis[0]: axis for axis in NEURO_AXES}}

#: The most users whose birthdays one prompt carries. A history window rarely holds more
#: people than this, and the bound keeps a crowded channel from growing every prompt.
BIRTHDAY_USERS_MAX = 10


class PromptBuilderMixin:
    """Persona/system-instruction assembly and the neuro-state extraction that
    reads the model's state back out of the <neuro_update> tag at the end of its reply.
    """

    def _resolve_appearance_data(self, owner_id: int, profile_name: str) -> Tuple[str, str]:
        """The name and avatar a profile speaks under, set or not.

        The fallback is one of Discord's default avatars rather than the bot's own: in a
        channel of unconfigured characters, the bot's face made every one of them look
        like the bot. `default_profile_avatar_url` is stable across restarts.
        """
        app = self.cog.profile_manager._get_user_appearance(owner_id, profile_name)
        display_name = app.get("custom_display_name") or profile_name
        avatar_url = app.get("custom_avatar_url") or default_profile_avatar_url(profile_name)
        return display_name, avatar_url

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

    def _users_in_history_window(self, session: Optional[Dict], profile_data: Dict[str, Any]) -> List[Tuple[int, str]]:
        """(user id, display name) for each user with a public turn inside this profile's
        history window, most recent speaker first.

        The window is the profile's `stm_length`, the span `_build_history_for_participant`
        shows it. A user who spoke earlier than that, or only in a whisper, is not part of
        the conversation this prompt answers, and their details stay out of it.
        """
        log = (session or {}).get("unified_log") or []
        stm_length = int(profile_data.get("stm_length", defaultConfig.CHATBOT_MEMORY_LENGTH))
        users: Dict[int, str] = {}
        for turn in itertools.islice(reversed(log), max(stm_length, 0)):
            if turn.get("is_user") is not True or turn.get("type") or turn.get("is_hidden"):
                continue
            try:
                user_id = int(turn.get("speaker_pid"))
            except (TypeError, ValueError):
                continue
            if user_id in users:
                continue
            # The name the model reads in that turn's header, `<Name> [ID: hash] ...`.
            content = turn.get("content") or ""
            end = content.find("> [ID: ")
            users[user_id] = content[1:end] if content.startswith("<") and end > 1 else "a user"
            if len(users) >= BIRTHDAY_USERS_MAX:
                break
        return list(users.items())

    def _birthday_lines(self, profile_owner_id: Optional[int], profile_name: str,
                        profile_today: datetime.date,
                        users: Sequence[Tuple[int, str]],
                        cast: Sequence[Tuple[int, str]] = ()) -> List[str]:
        """What the model is told about birthdays within a day of today, if anything.

        The character's own is its source profile's, so a borrow keeps the original's, and
        is judged on the character's clock. Each user's comes from their About Me and is
        judged on theirs: a birthday is the date where its owner is.

        `cast` is (owner id, profile name) for every seat in the session, this one included.
        The rest of the cast is told a character's birthday as well: told only to the
        character, its birthday lines reach everyone else with nothing to explain them.
        """
        from ...utils.helpers import _resolve_zoneinfo
        pm = self.cog.profile_manager
        lines = []
        if profile_owner_id is not None:
            own = describe_birthday(pm.character_birthday(profile_owner_id, profile_name), profile_today)
            if own:
                lines.append(own)

        utc_today = datetime.datetime.now(datetime.timezone.utc).date()
        for owner_id, name in cast:
            if (owner_id, name) == (profile_owner_id, profile_name):
                continue
            birthday = pm.character_birthday(owner_id, name)
            # Settled on the UTC date first, so a character with no birthday this week
            # costs no config read.
            if birthday_offset(birthday, utc_today, reach=2) is None:
                continue
            is_borrowed = name in pm._get_user_index(owner_id).get("borrowed", [])
            config = pm._get_profile_config(owner_id, name, is_borrowed) or {}
            try:
                tz, _ = _resolve_zoneinfo(config.get("timezone") or "UTC")
            except Exception:
                tz = datetime.timezone.utc
            # Named as its turns are headed: the name it speaks under, and its PID.
            shown = self._resolve_appearance_data(owner_id, name)[0]
            line = describe_birthday(birthday, datetime.datetime.now(tz).date(),
                                     f"{shown} [ID: {pm._get_profile_id(owner_id, name)}]")
            if line:
                lines.append(line)

        seen = set()
        for user_id, display_name in users:
            if user_id in seen:
                continue
            seen.add(user_id)
            about = self.cog.profile_manager.get_user_about(user_id)
            if not about.get("birthday"):
                continue
            try:
                tz, _ = _resolve_zoneinfo(about.get("timezone") or "UTC")
            except Exception:
                tz = datetime.timezone.utc
            line = describe_birthday(about["birthday"], datetime.datetime.now(tz).date(),
                                     f"{display_name} [ID: {_get_user_hash(user_id)}]")
            if line:
                lines.append(line)
        return lines

    def _construct_system_instructions(self, profile_owner_id: Optional[int], profile_name_to_use: str, channel_id: int, is_multi_profile: bool = False, training_examples_list: Optional[List[str]] = None, recalled_ltm: Optional[str] = None, critic_constraints: Optional[str] = None, present_users: Optional[Sequence[Tuple[int, str]]] = None, functions: Sequence[Any] = ()) -> Tuple[str, bool, bool, float, float, int, str, str]:
        """The system instruction for one profile's generation, plus its sampling values.

        `present_users` is (user id, display name) for the people in the conversation, whose
        birthdays the character may know. A session derives them from its own log, so only
        a caller with no session -- Global Chat -- passes them.

        `functions` are the ones the caller's model declares -- the tuple
        `tool_loop.functions_for` gave it, handed to the model factory as well -- and
        each is described here. Empty for a caller that sends this to no model.
        """
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
        timezone_str = "UTC"
        neuro_enabled = False
        neuro_state = {"dopamine": 50, "cortisol": 20, "oxytocin": 50, "adrenaline": 20}

        if profile_owner_id is not None:
            user_index = self.cog.profile_manager._get_user_index(profile_owner_id)
            is_borrowed = profile_name_to_use in user_index.get("borrowed",[])
            profile_data = self.cog.profile_manager._get_profile_config(profile_owner_id, profile_name_to_use, is_borrowed) or {}

            persona_data, ai_instr_str, grounding_enabled, temperature, top_p, top_k, _, _, primary_model, fallback_model = self.cog.session_manager._get_user_profile_for_model(profile_owner_id, channel_id, profile_name_to_use)

        if profile_data:
            timezone_str = profile_data.get("timezone", "UTC")
            neuro_enabled = profile_data.get("neuro_engine_enabled", False)
            neuro_state = profile_data.get("neuro_state", {"dopamine": 50, "cortisol": 20, "oxytocin": 50, "adrenaline": 20})

        # Assembled most-stable-first, and that is a cost decision rather than a
        # stylistic one. Providers cache on a shared prefix, so the first block that
        # changes invalidates every token after it -- and <current_time> is formatted to
        # the minute. With the persona and the character instructions sitting *behind*
        # it, as they used to, the largest and most stable part of every prompt was
        # re-billed uncached on every turn that crossed a minute boundary.
        #
        # <session_rules> and <content_policy> stay at the very end despite being stable
        # themselves: they are the output-format and hard-content rules and they want
        # recency, and by that point a volatile block already sits in front of them --
        # so nothing past `stable_parts` was ever going to cache anyway.
        stable_parts = []
        volatile_parts = []

        session = self.cog.multi_profile_channels.get(channel_id) if is_multi_profile else None
        if is_multi_profile:
            if session and session.get("session_prompt"):
                stable_parts.append(f"<scene_prompt>\n{session['session_prompt']}\n</scene_prompt>")

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
                stable_parts.append(persona_str)

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
            # Renamed from the bare <instructions>: it was the one generically-named tag
            # in the set and it carries the most important user-authored content, which
            # made it both the least informative name for the model and the one tag that
            # could not safely be added to the orphan scrub pattern. Both spellings are
            # in SYSTEM_XML_TAGS, so a model echoing either is still caught.
            stable_parts.append("<character_instructions>\n"
                                + "\n\n".join(decrypted_parts).strip()
                                + "\n</character_instructions>")

        # Stable, so they sit with the persona rather than among the per-turn blocks:
        # they never change, and anything volatile placed ahead of them would
        # invalidate the cached prefix they belong to. Exactly the functions the
        # caller's model declares, from the one tuple both were handed -- a character
        # told about a function it was never sent narrates a search it cannot run.
        for function in functions:
            stable_parts.append(function.instruction(self.cog.global_prompts))

        if is_multi_profile:
            # Standing context, not a history turn: the synopsis summarises turns that
            # have already left the STM window, so competing for a slot inside that
            # window would hide it from exactly the long sessions it exists for. Shared
            # by the whole cast -- only public turns are ever compacted, so it can carry
            # nothing a participant was not already entitled to see.
            synopsis = self.cog.session_manager.get_latest_synopsis(session)
            if synopsis:
                volatile_parts.append(f"<session_synopsis>\n{synopsis}\n</session_synopsis>")

            # Standing context for the same reason, and injected here rather than into
            # the game's own call so that *every* generation in the channel sees it --
            # a seated character answering ordinary chatter mid-hand knows what it just
            # played, which is what removed the need to bench the cast during a game.
            # Returns None on the overwhelmingly common no-game path, for one dict get.
            game_block = self.cog.game_service.context_block(channel_id)
            if game_block:
                volatile_parts.append(f"<game_context>\n{game_block}\n</game_context>")

        if neuro_enabled:
            neuro_block = self.cog.global_prompts.get("NEURO_ENGINE", DEFAULT_NEURO_INSTRUCTION).format(
                d=neuro_state.get('dopamine', 50),
                c=neuro_state.get('cortisol', 20),
                o=neuro_state.get('oxytocin', 50),
                a=neuro_state.get('adrenaline', 20)
            )
            volatile_parts.append(neuro_block)

        # Always sent. `time_tracking_enabled` used to switch this block off; that mode is
        # retired and the key is no longer read, so a profile still carrying False gets
        # its clock like every other.
        time_template = self.cog.global_prompts.get("TIME_CONTEXT", DEFAULT_CURRENT_TIME)
        try:
            from ...utils.helpers import _resolve_zoneinfo
            tz, _ = _resolve_zoneinfo(timezone_str)
            now = datetime.datetime.now(tz)
        except Exception as e:
            print(f"Error processing timezone '{timezone_str}': {e}. Defaulting to UTC.")
            now = datetime.datetime.now(datetime.timezone.utc)
        volatile_parts.append(time_template.format(time_str=now.strftime(TURN_TIME_FORMAT)))

        # Beside <current_time>, which already changes every minute, so it costs no
        # prompt caching the clock was not already costing.
        if present_users is None:
            present_users = self._users_in_history_window(session, profile_data) if session else []
        cast = [(seat.get("owner_id"), seat.get("profile_name"))
                for seat in (session or {}).get("profiles") or []]
        birthday_lines = self._birthday_lines(profile_owner_id, profile_name_to_use, now.date(),
                                              present_users, cast)
        if birthday_lines:
            birthday_template = self.cog.global_prompts.get("BIRTHDAY_CONTEXT", DEFAULT_BIRTHDAY_CONTEXT)
            volatile_parts.append(birthday_template.format(birthdays="\n".join(birthday_lines)))

        if training_examples_list:
            examples_block = "\n---\n".join(training_examples_list)
            volatile_parts.append(
                self.cog.global_prompts.get("TRAINING_DATA_INJECTION", DEFAULT_TRAINING_DATA_INJECTION)
                .format(examples_block=examples_block))

        if recalled_ltm:
            volatile_parts.append(recalled_ltm)

        # The critic's constraints, placed here rather than appended by the caller after
        # <content_policy>. This parameter existed and no caller passed it: the worker
        # built the same block itself once the critic had run, which put a style rule
        # after the hard content rule and left two placements for one block, one of them
        # fiction. The worker now runs its critic before this call and passes the result.
        if critic_constraints:
            constraints_block = self.cog.global_prompts.get("NEGATIVE_CONSTRAINTS", DEFAULT_NEGATIVE_CONSTRAINTS)
            volatile_parts.append(constraints_block.format(constraints=critic_constraints))

        rule_block = self.cog.global_prompts.get("CONTEXT_RULES", DEFAULT_SESSION_RULES)

        # [NEW] Dynamically inject the profile's ID into the context rules
        profile_id_val = self.cog.profile_manager._get_profile_id(profile_owner_id, profile_name_to_use)
        trailing_parts = [rule_block.format(profile_id_placeholder=profile_id_val).strip()]

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
                trailing_parts.append(policy_block)

        current_instructions_str = "\n\n".join(
            p for p in (stable_parts + volatile_parts + trailing_parts) if p and p.strip()
        ).strip()


        final_system_instruction = current_instructions_str if current_instructions_str.strip() else DEFAULT_SYSTEM_INSTRUCTION
        return final_system_instruction, False, grounding_enabled, temperature, top_p, top_k, primary_model, fallback_model

    def _neuro_state_from_text(self, raw_text: str) -> Tuple[str, Dict[str, int]]:
        """The state the reply's `<neuro_update>` tag reports, and the reply without it.

        The one way the engine reports, whichever provider answered. Read axis by axis,
        so whatever separator a model chooses survives and a whole name reads as its
        letter -- splitting on `|` and `:` lost the lot to `D:70, C:15`. A tag that was
        there and yielded no axis is logged: otherwise it is a character whose mood
        silently never moves.

        Falls back to a bare `D:70|C:15|O:60|A:25` for a model that drops the tag. The
        marker is scrubbed either way; one left in is said out loud in Discord.
        """
        data_str = None
        clean_text = raw_text

        try:
            with Timeout(seconds=1, error_message="Neuro extraction timed out"):
                match = NEURO_TAG_PATTERN.search(raw_text)
                if match:
                    data_str = match.group(1)
                    clean_text = NEURO_TAG_PATTERN.sub('', raw_text)
                else:
                    match = NEURO_BARE_PATTERN.search(raw_text)
                    if match:
                        data_str = match.group(0)
                        clean_text = NEURO_BARE_PATTERN.sub('', raw_text)
        except TimeoutError:
            return raw_text.strip(), {}

        if not data_str:
            return raw_text.strip(), {}

        new_state: Dict[str, int] = {}
        for key, value in _NEURO_AXIS.findall(data_str):
            axis = _AXIS_BY_KEY.get(key.lower())
            if axis:
                new_state[axis] = max(0, min(100, int(value)))
        if not new_state:
            print(f"Neuro engine: a <neuro_update> tag carried no readable axis: {data_str[:80]!r}")

        return clean_text.strip(), new_state

    def _extract_and_apply_neuro_state(self, raw_text: str, owner_id: int,
                                       profile_name: str) -> Tuple[str, Optional[Dict[str, int]]]:
        """The reply with its state marker removed, and the state it left behind.

        An axis the tag leaves out keeps its stored value: a partial update means "the
        rest did not move". Nothing is written when nothing was reported, which is most
        turns.
        """
        clean_text, new_state = self._neuro_state_from_text(raw_text)

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
