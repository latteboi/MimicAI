import io
import os
import re
import time
import uuid
import random
import asyncio
import discord
import traceback
import collections
import datetime
from zoneinfo import ZoneInfo
from ..utils.constants import (
    defaultConfig, PLACEHOLDER_EMOJI, GAME_BEAT_STALE_SECONDS,
    ERR_GENERAL_ERROR, ERR_REASON_EMPTY_RESPONSE, ERR_REASON_TIMEOUT_BOTH, ERR_SAFETY_BLOCK,
    WHISPER_BUSY_WAIT_TIMEOUT_SECONDS,
    WARN_BOTH_MODELS_FAILED, WARN_FALLBACK_USED, WARN_MAIN_MODEL_FAILED, WARN_VOICE_SYNTHESIS_FAILED,
    DEFAULT_KICKSTART_START, DEFAULT_KICKSTART_CONTINUE, DEFAULT_KICKSTART_IDLE,
    DEFAULT_WHISPER_RECAP, DEFAULT_DIRECTOR_USER_PROMPT,
    DEFAULT_IMAGE_GROUNDING, DEFAULT_NEGATIVE_CONSTRAINTS, DEFAULT_IMAGE_PRESENT,
    DEFAULT_IMAGE_PRESENT_OTHER, DEFAULT_IMAGE_FAILED,
    DEFAULT_SPEECH_VOICE, TTS_SYNTHESIS_PREAMBLE,
)
from ..utils.helpers import (
    _add_inline_citations, _format_api_error, _format_citation_subtext, _format_debug_prompt,
    _format_history_entry, _get_user_hash, _resolve_safety_settings, _scrub_response_text,
    _split_into_sentences_with_abbreviations, is_real_model, resolve_critic_settings,
)
from ..utils import mem_probe
from ..managers.session_manager import intern_turn

from .generation._shared import _strip_neuro_update_and_scrub
from .generation.heartbeat import HeartbeatMixin
from .generation.prompt_builder import PromptBuilderMixin
from .generation.delivery import DeliveryMixin
from .generation.regeneration import RegenerationMixin
from .generation.speak import SpeakAsMixin
from .generation.global_chat import GlobalChatMixin
from .generation.whisper import WhisperMixin
from .generation.image_round import ImageRoundMixin
from .generation.triggers import TriggerIntakeMixin
from .generation.compaction import SessionCompactionMixin
from .generation.ltm_capture import LtmCaptureMixin


class GenerationService(HeartbeatMixin, PromptBuilderMixin, DeliveryMixin, RegenerationMixin,
                        SpeakAsMixin, GlobalChatMixin, WhisperMixin, ImageRoundMixin,
                        TriggerIntakeMixin, SessionCompactionMixin, LtmCaptureMixin):
    """Owns the core generation engine: the multi-participant turn-rotation worker
    (_multi_profile_worker, defined here) plus the heartbeat/prompt-building/delivery/
    regeneration/speak/global-chat/whisper mixins (each in cogs/services/generation/) that
    together implement the rest of the generation surface.

    Holds a back-reference to the parent cog for state/logic not yet migrated
    (bot, queues, caches, and the many profile/session/memory/tools/media
    lookups this engine orchestrates), per the transitional Dependency
    Injection pattern in CLAUDE.md.
    """

    def __init__(self, cog):
        self.cog = cog

    @staticmethod
    def _promote_game_beats(session, triggers):
        """Drop the game beats that are no longer worth answering, and put the survivor
        at the front of the round.

        Returns `(triggers, seated_participant_or_None, cast)`, where `cast` is empty
        for an ordinary beat and holds the whole seated table for a finale.

        Two beats are dropped rather than answered. One that has waited out a long round
        is describing a table that has since moved on -- reacting to it late is worse
        than not reacting, because it reads as a character talking about the wrong hand.
        And one whose character has been muted in this session goes too: the alternative
        is the starting override falling through to somebody else, who would then react
        to a Draw Four that landed on a different player.

        A finale is the same trigger with a cast attached. It supersedes any ordinary
        beat batched with it -- the game has ended, which is the most permanent version
        of "the table has moved on" the stale rule exists for -- and it survives one
        thing an ordinary beat does not: a muted lead. The end of a game belongs to everyone who
        played it, so a muted winner hands the round to the next seat rather than taking
        the aftermath down with it. That rebinds the trigger, because index 2 is what
        `_collect_round_triggers` reads the starting override from.

        The reorder is the queue jump. Everything else batched into this round still
        gets answered -- it just answers after the character the table was actually
        looking at. `_collect_round_triggers` reads the starting override off index 0
        only, so putting the beat there is what carries it.
        """
        beat_participant = None
        beat_cast = []
        surviving = []
        queue = session.get('task_queue')

        def drop():
            if queue is not None:
                try:
                    queue.task_done()
                except ValueError:
                    pass

        # A finale in the batch supersedes every ordinary beat in it. The game has
        # ended, which is the most permanent version of "the table has moved on" the
        # stale rule exists for -- and it also settles the ordering, since the round's
        # starting override is read off index 0 and there can only be one of those.
        has_finale = any(isinstance(t, tuple) and t[0] == 'game_beat'
                         and t[1].get('finale') for t in triggers)

        for trigger in triggers:
            if isinstance(trigger, tuple) and trigger[0] == 'game_beat':
                payload, seated = trigger[1], trigger[2]
                if has_finale and not payload.get('finale'):
                    drop()
                    continue
                stale = (time.monotonic() - payload.get('queued_at', 0)
                         > payload.get('stale_after', GAME_BEAT_STALE_SECONDS))
                cast = [p for p in (payload.get('cast') or [])
                        if not p.get('is_skipped')]
                if seated.get('is_skipped'):
                    seated = cast[0] if cast else None
                    if seated is not None:
                        trigger = (trigger[0], payload, seated)
                if stale or seated is None:
                    drop()
                    continue
                if beat_participant is None:
                    beat_participant, beat_cast = seated, cast
            surviving.append(trigger)

        if beat_participant is not None:
            surviving.sort(key=lambda t: 0 if (isinstance(t, tuple)
                                               and t[0] == 'game_beat') else 1)
        return surviving, beat_participant, beat_cast

    async def _multi_profile_worker(self, channel_id: int):
        session = self.cog.multi_profile_channels.get(channel_id)
        if not session: return

        session_type = session.get("type", "multi")
        session = await self.cog.session_manager._ensure_session_hydrated(channel_id, session_type)
        if not session:
            print(f"Worker for channel {channel_id} could not hydrate session. Aborting.")
            return

        # Ensure task queue is firmly initialised
        if 'task_queue' not in session or session['task_queue'] is None:
            session['task_queue'] = asyncio.Queue()

        session['is_running'] = False
        recent_processed_ids = collections.deque(maxlen=20)
        
        while True:
            try:
                initial_trigger = None
                is_proactive_auto_round = False
                
                try:
                    initial_trigger = await session['task_queue'].get()
                    
                    while session.get('is_purging') or session.get('is_regenerating') or session.get('is_memorising'):
                        await asyncio.sleep(0.5)

                    # Yield to a whisper that is already queued for this channel. Without
                    # this the whisper polls for an idle instant that a busy channel never
                    # offers -- the worker re-arms is_running within the same event-loop
                    # tick that the previous round cleared it, and the whisper's 0.5 s poll
                    # loses every time. Waiting on the counter rather than on is_whispering
                    # is what closes that gap: it is set before the whisper starts waiting.
                    # This cannot deadlock -- a whisper never waits on this counter, and it
                    # drops the counter at the instant it claims is_whispering.
                    if session.get('whisper_waiting'):
                        await self.cog.session_manager._wait_for_session_flags(
                            session, ('whisper_waiting',), WHISPER_BUSY_WAIT_TIMEOUT_SECONDS)

                    session['is_running'] = True

                except asyncio.CancelledError:
                    raise

                # [NEW] Gather all batched triggers immediately
                all_triggers_for_round = [initial_trigger]
                while not session['task_queue'].empty():
                    try: all_triggers_for_round.append(session['task_queue'].get_nowait())
                    except asyncio.QueueEmpty: break
                
                # [NEW] Filter out cancelled reaction triggers
                valid_triggers = []
                for t in all_triggers_for_round:
                    if isinstance(t, tuple) and t[0] in ['reaction', 'reaction_single']:
                        cancellation_key = (t[1].message_id, str(t[1].emoji))
                        if cancellation_key in session.get('cancelled_reaction_triggers', set()):
                            session['cancelled_reaction_triggers'].remove(cancellation_key)
                            continue
                    valid_triggers.append(t)
                
                all_triggers_for_round = valid_triggers

                (all_triggers_for_round, game_beat_participant,
                 game_beat_cast) = self._promote_game_beats(
                    session, all_triggers_for_round)
                if game_beat_participant is not None:
                    initial_trigger = all_triggers_for_round[0]

                # This drain is the only consumer of cancelled_reaction_triggers, so a
                # key that survives it is stale: it cancelled a trigger that was never
                # queued, or one an earlier round already ran. Leaving them made this an
                # unbounded set keyed by message id -- and worse, a stale
                # (message_id, emoji) silently swallowed the *next* identical reaction on
                # that message. It is reachable whenever msg.clear_reaction fails, which
                # needs Manage Messages -- a permission the bot often lacks, as /purge
                # checking for it explicitly implies -- because the user then removes the
                # emoji by hand and strands a key every single time. There is no await
                # between the filter above and this clear, so nothing can be recorded in
                # the gap.
                cancelled_keys = session.get('cancelled_reaction_triggers')
                if cancelled_keys:
                    cancelled_keys.clear()

                # [NEW] Record the start of this batch for Hybrid STM
                batch_start_index = len(session.get("unified_log", []))

                is_proactive_auto_round = False
                for i, t in enumerate(all_triggers_for_round):
                    if isinstance(t, tuple) and t[0] == 'proactive_trigger':
                        is_proactive_auto_round = True
                        pro = session.get("proactivity", {})
                        director_prompt = None
                        model_raw = pro.get("director_model", "off")
                        
                        if model_raw.lower() != "off":
                            sys_instr = pro.get("director_instructions")
                            if sys_instr:
                                try:
                                    channel = self.cog.bot.get_channel(channel_id)
                                    guild_id = channel.guild.id if channel and getattr(channel, 'guild', None) else 0
                                    # Via the factory rather than a local copy of the prefix parsing.
                                    # The copy matched prefixes case-insensitively, which the factory
                                    # deliberately does not: OpenRouter namespaces its models as
                                    # 'google/gemini-2.5-flash', so an upper()-ed match read that as a
                                    # GOOGLE/ prefix and sent an OpenRouter model id to Google.
                                    m = self.cog.api_service._instantiate_model(
                                        model_raw, guild_id, session.get("owner_id"),
                                        system_instruction=sys_instr, thinking_params={})
                                    hist_text = ""
                                    for ht in session.get("unified_log", [])[-10:]:
                                        hist_text += f"{ht.get('content', '')}\n"
                                    director_template = self.cog.global_prompts.get("DIRECTOR_USER_PROMPT", DEFAULT_DIRECTOR_USER_PROMPT)
                                    resp = await m.generate_content_async([director_template.format(history=hist_text)])
                                    if resp and resp.text: director_prompt = f"<internal_note>Director's Note: {resp.text.strip()}</internal_note>"
                                except Exception as e: print(f"AI Director failed: {e}")
                        all_triggers_for_round[i] = director_prompt

                if not all_triggers_for_round and not is_proactive_auto_round:
                    session['is_running'] = False
                    continue
                
                # We re-verify hydration here to catch sessions that were dehydrated during the await queue.get()
                if not session.get("is_hydrated"):
                    session = await self.cog.session_manager._ensure_session_hydrated(channel_id, session_type)

                channel = self.cog.bot.get_channel(channel_id)
                has_gemini = self.cog.storage_manager._get_api_key_for_guild(channel.guild.id, "gemini")
                has_openrouter = self.cog.storage_manager._get_api_key_for_guild(channel.guild.id, "openrouter")
                
                has_ollama = False
                for p in session['profiles']:
                    p_index = self.cog.profile_manager._get_user_index(p['owner_id'])
                    p_is_b = p['profile_name'] in p_index.get("borrowed", [])
                    p_cfg = self.cog.profile_manager._get_profile_config(p['owner_id'], p['profile_name'], p_is_b) or {}
                    if p_cfg.get("primary_model", "").upper().startswith("OLLAMA/"):
                        has_ollama = True
                        break
                
                if not has_gemini and not has_openrouter and not has_ollama:
                    try:
                        await channel.send("An API key has not been configured for this server. You can use the `/settings` command in my DM to set one.")
                    except discord.Forbidden: pass
                    
                    # Mark triggers as done to prevent queue stalling
                    for trigger in all_triggers_for_round:
                        if trigger is not None: session['task_queue'].task_done()
                    continue

                # Now process triggers into new_round_turn_data and unified_log...
                primary_eager_placeholder = None
                is_image_gen_round = False
                image_gen_prompt = ""
                image_gen_anchor_message = None
                generated_image_path_for_round = None
                # [UPDATED] Store tuples of (base_text, url_context_text, media_parts) 
                new_round_turn_data = [] 
                round_author_name = "A user"
                starting_profile_override = None
                triggering_user_id = session.get("owner_id")
                
                # Bind channel for the entire round early to prevent UnboundLocalErrors
                channel = self.cog.bot.get_channel(channel_id)
                
                # [NEW] Standardized initialization to prevent UnboundLocalErrors
                url_media_parts = []
                pre_generation_warnings = []
                pending_url_fetches = []

                if is_proactive_auto_round and session.get("proactive_initial_rounds") == 1:
                    cast = session['profiles']
                    if len(cast) > 1:
                        target_participant = cast[1]
                        target_id = target_participant['owner_id']
                        target_profile = target_participant['profile_name']
                        
                        target_index = self.cog.profile_manager._get_user_index(target_id)
                        target_appearance_name = target_profile
                        if target_profile in target_index.get("borrowed", []):
                            borrowed_data = self.cog.profile_manager._get_profile_config(target_id, target_profile, True) or {}
                            target_appearance_name = borrowed_data.get("original_profile_name", target_profile)
                        
                        target_display_name = target_appearance_name
                        if str(target_id) in self.cog.user_appearances and target_appearance_name in self.cog.user_appearances[str(target_id)]:
                            appearance = self.cog.user_appearances[str(target_id)][target_appearance_name]
                            if appearance.get("custom_display_name"):
                                target_display_name = appearance["custom_display_name"]

                        scene_starters = [
                            "You see {target} walk into the room. What do you say or do?",
                            "The topic of {topic} comes to mind. You decide to bring it up with {target}.",
                            "You notice {target} seems lost in thought. You approach them.",
                            "You and {target} are the only two left in the channel. The silence is getting awkward. You decide to break it."
                        ]
                        topics = ["the weather", "a recent rumor", "a strange noise", "an old memory", "a new idea"]
                        prompt_template = random.choice(scene_starters)
                        director_prompt = prompt_template.format(target=target_display_name, topic=random.choice(topics))
                        
                        # [UPDATED] Use new_round_turn_data with tuple format
                        new_round_turn_data.append((director_prompt, None, []))

                (is_image_gen_round, image_gen_prompt, starting_profile_override,
                 round_author_name, triggering_user_id) = await self._collect_round_triggers(
                    session, session_type, channel_id, all_triggers_for_round,
                    new_round_turn_data, pending_url_fetches, recent_processed_ids,
                    is_image_gen_round, image_gen_prompt, starting_profile_override,
                    round_author_name, triggering_user_id,
                )

                for content_obj in new_round_turn_data:
                    pass

                profile_order = []
                session_mode = session.get("session_mode", "sequential")
                channel = self.cog.bot.get_channel(channel_id)

                is_single_turn_only = False
                if isinstance(initial_trigger, tuple) and initial_trigger[0] == 'reaction_single':
                    is_single_turn_only = True
                elif game_beat_participant is not None and not game_beat_cast and all(
                        isinstance(t, tuple) and t[0] == 'game_beat'
                        for t in all_triggers_for_round if t is not None):
                    # A beat on its own is one character's moment, not a cue for the
                    # whole cast -- five profiles all reacting to one Draw Four is the
                    # noise the batched narrator was removed for. A beat batched with
                    # real conversation is different: the others are answering the
                    # people, not the card, so they keep their turn.
                    #
                    # A finale carries a cast, and is the one beat that is meant for all
                    # of them: the game has ended, and a table that empties in silence
                    # is what this whole path exists to prevent.
                    is_single_turn_only = True

                trigger_content_lower = ""
                for t in all_triggers_for_round:
                    if isinstance(t, discord.Message): trigger_content_lower += t.clean_content.lower() + " "
                    elif isinstance(t, tuple) and len(t) > 1:
                        if isinstance(t[1], discord.Message): trigger_content_lower += t[1].clean_content.lower() + " "
                        elif isinstance(t[1], dict) and 'content' in t[1]: trigger_content_lower += t[1]['content'].lower() + " "
                    elif isinstance(t, str): trigger_content_lower += t.lower() + " "

                active_participants = []
                for p in session['profiles']:
                    if p.get('is_skipped', False): continue
                    chance = p.get('chance', 100)
                    wakewords = p.get('wakewords', [])
                    will_respond = False
                    if wakewords and any(w.lower() in trigger_content_lower for w in wakewords if w.strip()):
                        will_respond = True
                    else:
                        will_respond = (random.randint(1, 100) <= chance)

                    if will_respond: active_participants.append(p)

                # The seat the beat landed on speaks regardless of its response chance.
                # A reaction that rolls a die against `chance` is a reaction that
                # silently vanishes for the profile the table is looking at, and the
                # beat has already been paid for by the time it reaches here. For a
                # finale that is every seat that played, not just the lead -- the whole
                # point of the aftermath is that the table answers it.
                if game_beat_participant is not None:
                    for seated in (game_beat_cast or [game_beat_participant]):
                        if not any(p is seated for p in active_participants):
                            active_participants.insert(0, seated)

                if not active_participants:
                    for trigger in all_triggers_for_round:
                        if trigger is not None: session['task_queue'].task_done()
                    session['is_running'] = False
                    continue

                if starting_profile_override:
                    start_p = starting_profile_override
                    if start_p.get('is_skipped') or start_p not in active_participants:
                        start_p = active_participants[0]

                    if session_mode == 'sequential':
                        try:
                            start_idx = session['profiles'].index(start_p)
                            new_order = session['profiles'][start_idx:] + session['profiles'][:start_idx]
                            session['profiles'] = new_order
                            self.cog.session_manager._save_multi_profile_sessions()
                        except ValueError: pass

                    if is_single_turn_only:
                        profile_order = [start_p]
                    else:
                        profile_order = [p for p in session['profiles'] if p in active_participants]
                        if session_mode == 'random':
                            if start_p in profile_order: profile_order.remove(start_p)
                            random.shuffle(profile_order)
                            profile_order.insert(0, start_p)
                else:
                    profile_order = [p for p in session['profiles'] if p in active_participants]
                    if session_mode == 'random':
                        random.shuffle(profile_order)
                    elif session.get('last_speaker_key'):
                        try:
                            last_speaker_index = next(i for i, p in enumerate(session['profiles']) if (p['owner_id'], p['profile_name']) == session['last_speaker_key'])
                            start_index = (last_speaker_index + 1) % len(session['profiles'])
                            rotated = session['profiles'][start_index:] + session['profiles'][:start_index]
                            profile_order = [p for p in rotated if p in active_participants]
                        except (ValueError, StopIteration): pass

                # Apply response limit if set
                max_responses = session.get("max_responses", 10)
                if len(profile_order) > max_responses:
                    profile_order = profile_order[:max_responses]

                # --- Ephemeral Participant Injection ---
                ephemeral_participant = None
                if isinstance(initial_trigger, tuple):
                    if initial_trigger[0] in ['child_mention']:
                        _, _, ephemeral_participant = initial_trigger
                    elif initial_trigger[0] == 'reply' and starting_profile_override and starting_profile_override.get('ephemeral'):
                        ephemeral_participant = starting_profile_override

                if ephemeral_participant:
                    # For child bots, bot_id is the key. For parent bot (webhook), profile_name/owner_id is the key.
                    existing_permanent = None
                    if ephemeral_participant.get('method') == 'child_bot':
                        existing_permanent = next((p for p in profile_order if p.get('bot_id') == ephemeral_participant.get('bot_id')), None)
                    else:
                        existing_permanent = next((p for p in profile_order if p['owner_id'] == ephemeral_participant['owner_id'] and p['profile_name'] == ephemeral_participant['profile_name']), None)

                    if existing_permanent:
                        profile_order.remove(existing_permanent)
                        profile_order.insert(0, existing_permanent)
                    else:
                        profile_order.insert(0, ephemeral_participant)

                channel = self.cog.bot.get_channel(channel_id)
                has_gemini = self.cog.storage_manager._get_api_key_for_guild(channel.guild.id, "gemini")
                has_openrouter = self.cog.storage_manager._get_api_key_for_guild(channel.guild.id, "openrouter")
                
                if not has_gemini and not has_openrouter:
                    try:
                        await channel.send("An API key has not been configured for this server. You can use the `/settings` command in my DM to set one.")
                    except discord.Forbidden: pass
                    
                    # Mark triggers as done to prevent queue stalling
                    for trigger in all_triggers_for_round:
                        if trigger is not None: session['task_queue'].task_done()
                    continue

                # --- Age-restriction gate, applied before any feedback is dispatched ---
                # The per-participant loop below re-checks this and skips the turn, but the
                # placeholder and typing indicator are dispatched *here*, so a blocked
                # profile used to flash its emoji or a "typing..." bubble and then fall
                # silent. Worse, profile_order[0] could itself be blocked, so the placeholder
                # wore the name and avatar of a profile that was never going to speak.
                # Filtering first also collapses the refusal notice to one per profile per
                # round instead of one per turn.
                blocked_participants = [
                    p for p in profile_order
                    if not self.cog.profile_manager._check_unrestricted_safety_policy(
                        p['owner_id'], p['profile_name'], channel)
                ]
                if blocked_participants:
                    profile_order = [p for p in profile_order if p not in blocked_participants]
                    seen_blocked = set()
                    for p in blocked_participants:
                        key = (p['owner_id'], p['profile_name'])
                        if key in seen_blocked:
                            continue
                        seen_blocked.add(key)
                        await self._send_channel_message(
                            channel,
                            f"[System Notice: '{p['profile_name']}' cannot respond. Profiles with "
                            "'Unrestricted 18+' safety are only permitted in age-restricted channels.]")

                if not profile_order:
                    for trigger in all_triggers_for_round:
                        if trigger is not None: session['task_queue'].task_done()
                    continue

                # --- Synchronised Feedback Step ---
                # Hoisted to here, directly after the channel and API-key guards. The three
                # things a placeholder needs — a resolved channel, a valid key, and
                # profile_order[0] for the webhook name and avatar — are all settled by this
                # point, and the age-restriction filter above has already removed every
                # participant that would decline to respond.
                # It previously sat below the anchor-message resolution, so the user waited on
                # a channel.fetch_message round trip before seeing any feedback at all.
                first_participant = profile_order[0] if profile_order else None
                first_placeholder_message = None
                feedback_task = None

                if first_participant:
                    if first_participant.get('method') == 'child_bot':
                        p_index = self.cog.profile_manager._get_user_index(first_participant['owner_id'])
                        p_is_b = first_participant['profile_name'] in p_index.get("borrowed", [])
                        fp_settings = self.cog.profile_manager._get_profile_config(first_participant['owner_id'], first_participant['profile_name'], p_is_b) or {}

                        if fp_settings.get("child_bot_placeholder", False):
                            custom_emoji = fp_settings.get("placeholder_emoji") or PLACEHOLDER_EMOJI
                            feedback_task = asyncio.create_task(self._send_child_bot_placeholder(first_participant['bot_id'], channel_id, custom_emoji))
                        else:
                            await self.cog.manager_queue.put({
                                "action": "send_to_child", "bot_id": first_participant['bot_id'],
                                "payload": {"action": "start_typing", "channel_id": channel_id}
                            })
                    else: # Webhook
                        feedback_task = asyncio.create_task(self._send_channel_message(
                            channel, f"{PLACEHOLDER_EMOJI}",
                            profile_owner_id_for_appearance=first_participant['owner_id'],
                            profile_name_for_appearance=first_participant['profile_name'],
                            bypass_typing=True
                        ))

                # --- Whisper barrier ---
                # A whisper claims the channel and everything else queues behind it. The
                # wait sits *here*, deliberately: the placeholder and typing indicator above
                # have already been dispatched, so the profile that is up next shows its
                # feedback immediately and does the waiting silently, exactly as if it were
                # composing a long reply. Putting it earlier -- with the other flag waits at
                # the top of the loop -- would leave the channel showing nothing at all for
                # the length of the whisper.
                #
                # It also has to be before the prompt is built rather than only before the
                # model call: the whisper appends its turns to unified_log when it lands, and
                # a round that had already derived its history would answer a conversation
                # missing them.
                #
                # Bounded, and proceeds anyway on timeout -- a leaked is_whispering must
                # degrade to "the round runs late", never to a channel that stops answering.
                if session.get('is_whispering'):
                    if not await self.cog.session_manager._wait_for_session_flags(
                        session, ('is_whispering',), WHISPER_BUSY_WAIT_TIMEOUT_SECONDS
                    ):
                        print(f"Whisper barrier timed out in channel {channel_id}; proceeding with the round.")

                was_blocked = False
                generator_profile_key = None
                generator_display_name = "A participant"
                image_gen_error_msg = None
                image_gen_placeholder_id = None

                if is_image_gen_round:
                    if starting_profile_override:
                        generator_profile_key = (starting_profile_override['owner_id'], starting_profile_override['profile_name'])
                    elif profile_order:
                        first_participant = profile_order[0]
                        generator_profile_key = (first_participant['owner_id'], first_participant['profile_name'])
                    
                    if generator_profile_key:
                        gen_owner_id, gen_profile_name = generator_profile_key
                        gen_effective_owner_id, gen_effective_profile_name = self.cog.profile_manager._resolve_effective_profile(gen_owner_id, gen_profile_name)
                        
                        gen_appearance_data = self.cog.user_appearances.get(str(gen_effective_owner_id), {}).get(gen_effective_profile_name, {})
                        if gen_appearance_data.get("custom_display_name"):
                            generator_display_name = gen_appearance_data["custom_display_name"]
                        else:
                            generator_display_name = gen_effective_profile_name

                responses_this_round = []
                # [NEW] Track round-specific audio segments for stitching
                round_audio_segments = []
                # [FIXED] Populate initial context immediately from batched triggers
                initial_round_context = "\n".join([t[0] for t in new_round_turn_data])

                # [NEW] Determine Anchor Message for Response Modes
                anchor_message = None
                if isinstance(initial_trigger, discord.Message):
                    anchor_message = initial_trigger
                elif isinstance(initial_trigger, tuple) and len(initial_trigger) > 1 and isinstance(initial_trigger[1], discord.Message):
                    anchor_message = initial_trigger[1]
                else:
                    # Auto-continue or reactor: use the last bot message in the session
                    try:
                        last_mid = session.get('last_bot_message_id')
                        if last_mid: anchor_message = await channel.fetch_message(last_mid)
                    except: pass

                grounding_context, grounding_sources = None, []
                grounding_profile_key = None
                grounding_mode_for_citator = "off"

                grounding_target_participant = starting_profile_override or (profile_order[0] if profile_order else None)

                if grounding_target_participant:
                    g_owner_id = grounding_target_participant['owner_id']
                    g_profile_name = grounding_target_participant['profile_name']
                    g_index = self.cog.profile_manager._get_user_index(g_owner_id)
                    g_is_borrowed = g_profile_name in g_index.get("borrowed", [])
                    g_profile_settings = self.cog.profile_manager._get_profile_config(g_owner_id, g_profile_name, g_is_borrowed) or {}
                    
                    grounding_mode = g_profile_settings.get("grounding_mode", "off")
                    if isinstance(grounding_mode, bool): grounding_mode = "rag" if grounding_mode else "off"
                    elif grounding_mode in ["on", "on+"]: grounding_mode = "rag"
                    
                    grounding_mode_for_citator = grounding_mode

                    # Parallelise Grounding RAG and URL Research Context Fetching
                    grounding_task = None
                    if grounding_mode == "rag":
                        g_participant_key = (g_owner_id, g_profile_name)
                        # Derived from unified_log rather than the shadow chat_sessions copy.
                        g_stm_length = int(g_profile_settings.get("stm_length", defaultConfig.CHATBOT_MEMORY_LENGTH))
                        g_stm_capped = min(10, g_stm_length)
                        history_for_grounding = []
                        if g_stm_capped > 0:
                            g_bot_pid = self.cog.profile_manager._get_pid_from_name_any(g_owner_id, g_profile_name)
                            history_for_grounding = self.cog.session_manager._build_history_for_participant(
                                session.get("unified_log", []), g_bot_pid, g_profile_settings, len(profile_order) or 1
                            )[-(g_stm_capped * 2):]

                        # Safety Logic for Grounding
                        g_dynamic_safety_settings = _resolve_safety_settings(channel, g_profile_settings)

                        is_for_image_flag = is_image_gen_round
                        grounding_query = image_gen_prompt if is_image_gen_round else initial_round_context

                        mapping_key = (session.get("type", "multi"), channel.id)
                        grounding_task = self.cog.tools_service._get_hybrid_grounding_context(grounding_query, channel.guild.id, history_for_grounding, mapping_key, safety_settings=g_dynamic_safety_settings, is_for_image=is_for_image_flag, warning_channel=channel)

                ## [NEW] Phase: Research Once (URL Context)
                round_url_text_contexts = []
                
                url_tasks = []
                if pending_url_fetches:
                    for fetch in pending_url_fetches:
                        url_tasks.append(self.cog.tools_service._process_urls_in_content(fetch["content"], fetch["guild_id"], {"url_fetching_enabled": True}))

                # Gather Grounding and URL tasks to run concurrently
                tasks_to_gather = []
                if grounding_task:
                    tasks_to_gather.append(grounding_task)
                for ut in url_tasks:
                    tasks_to_gather.append(ut)

                gathered_results = []
                if tasks_to_gather:
                    gathered_results = await asyncio.gather(*tasks_to_gather)

                grounding_result = None
                if grounding_task:
                    grounding_result = gathered_results[0]
                    url_results = gathered_results[1:]
                else:
                    url_results = gathered_results

                # Unpack and apply Grounding results
                if grounding_result:
                    g_context, g_sources, _, g_warning = grounding_result
                    if g_warning:
                        pre_generation_warnings.append(g_warning)
                    
                    if g_context:
                        if is_image_gen_round:
                            grounding_template = self.cog.global_prompts.get("IMAGE_GROUNDING", DEFAULT_IMAGE_GROUNDING)
                            image_gen_prompt = grounding_template.format(prompt=image_gen_prompt, grounding=g_context)
                        else:
                            grounding_context = g_context
                            # [NEW] Sticky Grounding: Purge previous search results from history
                            for turn in session.get("unified_log", []):
                                if "grounding_context" in turn:
                                    del turn["grounding_context"]
                            
                            # Attach new summary to the latest turn (the trigger)
                            if session.get("unified_log"):
                                session["unified_log"][-1]["grounding_context"] = g_context

                            # The purge above rewrites turns anywhere in the log, including
                            # ones already sealed into the cold segment, so this cannot ride
                            # out on a tail write -- a stale grounding blob left on disk is
                            # re-injected as context on the next hydration.
                            self.cog.session_manager.mark_session_dirty(
                                (channel_id, None, None), session_type, structural=True)

                        grounding_sources = g_sources
                    grounding_profile_key = (g_owner_id, g_profile_name)

                # Unpack and apply URL results
                url_updates_made = False
                url_purged_cold_turns = False
                for i, (u_t, u_m, u_w) in enumerate(url_results):
                    fetch_info = pending_url_fetches[i]
                    pre_generation_warnings.extend(u_w)
                    
                    if u_t:
                        url_text_content = "\n".join(u_t)
                        round_url_text_contexts.append(url_text_content)
                        
                        # Update turn_object
                        if "turn_object" in fetch_info:
                            fetch_info["turn_object"]["url_context"] = url_text_content
                            url_updates_made = True
                            # Clear previous URL contexts from log
                            for turn in session.get("unified_log", []):
                                if turn is not fetch_info["turn_object"] and "url_context" in turn:
                                    del turn["url_context"]
                                    url_purged_cold_turns = True
                                    
                    if u_m:
                        url_media_parts.extend(u_m)
                        # Update new_round_turn_data
                        idx = fetch_info["turn_data_index"]
                        user_line, old_url_text, old_media = new_round_turn_data[idx]
                        old_media.extend(u_m)
                        new_round_turn_data[idx] = (user_line, url_text_content if u_t else old_url_text, old_media)
                        
                if url_updates_made:
                    # Structural for the same reason as the grounding purge above: the
                    # loop clears url_context from older turns as well as setting it on
                    # the current one.
                    self.cog.session_manager.mark_session_dirty(
                        (channel_id, None, None), session_type, structural=url_purged_cold_turns)

                # --- NEW IMAGE GENERATION LOGIC ---
                if is_image_gen_round and generator_profile_key:
                    with mem_probe.probe("image round (total)", peak=False):
                        (generated_image_path_for_round,
                         image_gen_placeholder_id,
                         image_gen_error_msg) = await self._run_image_generation_round(
                            session, channel, generator_profile_key, image_gen_prompt,
                            generator_display_name, first_participant, feedback_task,
                            new_round_turn_data, generated_image_path_for_round,
                            image_gen_placeholder_id, image_gen_error_msg,
                        )

                for i, participant in enumerate(profile_order):
                    turn_warnings = []
                    if i == 0:
                        turn_warnings.extend(pre_generation_warnings)
                    
                    channel = self.cog.bot.get_channel(channel_id)
                    api_key = self.cog.storage_manager._get_api_key_for_guild(channel.guild.id, "gemini")
                    or_key = self.cog.storage_manager._get_api_key_for_guild(channel.guild.id, "openrouter")
                    
                    # Initialize turn-specific variables at the very start of the loop
                    is_generator = False
                    p_settings = {}
                    participant_key = (participant['owner_id'], participant['profile_name'])
                    turn_grounding_sources = []
                    if participant_key == grounding_profile_key:
                        turn_grounding_sources.extend(grounding_sources)
                    sources_text_list = []
                    contents_for_api_call = [] 
                    fallback_used = False
                    response_text = ""
                    was_blocked = False
                    placeholder_message = None
                    
                    # Resolve Real-time settings
                    p_owner_id = participant['owner_id']
                    p_name = participant['profile_name']
                    p_index = self.cog.profile_manager._get_user_index(p_owner_id)
                    p_is_b = p_name in p_index.get("borrowed", [])
                    p_settings = self.cog.profile_manager._get_profile_config(p_owner_id, p_name, p_is_b) or {}

                    is_ollama = p_settings.get("primary_model", "").upper().startswith("OLLAMA/")
                    
                    if not api_key and not or_key and not is_ollama:
                        if i == 0:
                            try:
                                await channel.send("An API key must be configured on this server for sessions.")
                            except discord.Forbidden:
                                pass
                        break

                    # Ensure custom_main is safely bound early for error fallback
                    custom_main = p_settings.get("error_response", "An error has occurred.")

                    # Check Image Gen intent vs Profile Toggle
                    if is_image_gen_round and participant_key == generator_profile_key:
                        if p_settings.get("image_generation_enabled", True):
                            is_generator = True
                        else:
                            # Re-inject prefix if toggle is OFF for the target generator
                            is_generator = False
                            initial_round_context = f"!image {image_gen_prompt}\n{initial_round_context}"

                    placeholder_message = None
                    response_text = ""

                    profile_settings = {} # Initialize to prevent UnboundLocalError
                    t1_start_mono = time.monotonic()
                    t1_start_utc = datetime.datetime.now(datetime.timezone.utc)
                    self.cog.session_last_accessed[channel_id] = time.time()
                    participant_key = (participant['owner_id'], participant['profile_name'])
                    
                    if participant.get('method') == 'child_bot':
                        p_owner_id_typing = participant['owner_id']
                        p_name_typing = participant['profile_name']
                        p_index_typing = self.cog.profile_manager._get_user_index(p_owner_id_typing)
                        p_is_b_typing = p_name_typing in p_index_typing.get("borrowed", [])
                        p_settings_typing = self.cog.profile_manager._get_profile_config(p_owner_id_typing, p_name_typing, p_is_b_typing) or {}
                        
                        if not p_settings_typing.get("child_bot_placeholder", False):
                            await self.cog.manager_queue.put({
                                "action": "send_to_child", "bot_id": participant['bot_id'],
                                "payload": {"action": "start_typing", "channel_id": channel_id}
                            })

                    owner_id = participant['owner_id']
                    profile_name = participant['profile_name']
                    channel = self.cog.bot.get_channel(channel_id)
                    session_key = (channel_id, owner_id, profile_name)
                    model = None

                    # [FIX] Initialize these before the try block to prevent UnboundLocalError
                    response = None
                    fallback_used = False
                    response_text = ""
                    was_blocked = False
                    # Assigned from _construct_system_instructions well inside the try below,
                    # but read by the meta block after the except handler. Anything that
                    # raises before that point — a missing guild API key is the common one —
                    # left it unbound, so the recovery path itself died with an
                    # UnboundLocalError and masked the real error.
                    fallback_model_name = None

                    # Backstop only. profile_order was filtered before the feedback step, so
                    # this fires only if the channel's age-restricted flag is flipped mid-round.
                    if not self.cog.profile_manager._check_unrestricted_safety_policy(owner_id, profile_name, channel):
                        error_message = f"[System Notice: '{profile_name}' cannot respond. Profiles with 'Unrestricted 18+' safety are only permitted in age-restricted channels.]"
                        await self._send_channel_message(channel, error_message)
                        continue

                    user_index = self.cog.profile_manager._get_user_index(owner_id)
                    is_borrowed = profile_name in user_index.get("borrowed", [])
                    effective_owner_id, effective_profile_name = self.cog.profile_manager._resolve_effective_profile(owner_id, profile_name)

                    speaker_display_name = profile_name
                    appearance_data = self.cog.user_appearances.get(str(effective_owner_id), {}).get(effective_profile_name, {})
                    if appearance_data.get("custom_display_name"):
                        speaker_display_name = appearance_data["custom_display_name"]

                    placeholder_message = None
                    response_text = ""
                    
                    if session.get('pending_image_gen_data'):
                        is_image_gen_round = True
                        image_gen_prompt = session['pending_image_gen_data']['prompt']
                        image_gen_anchor_message = session['pending_image_gen_data']['anchor_message']
                        generator_profile_key = participant_key
                        session['pending_image_gen_data'] = None

                        # This block is crucial to define the generator's variables when triggered mid-round
                        gen_owner_id, gen_profile_name = generator_profile_key
                        gen_effective_owner_id, gen_effective_profile_name = self.cog.profile_manager._resolve_effective_profile(gen_owner_id, gen_profile_name)
                        
                        gen_appearance_data = self.cog.profile_manager._get_user_appearance(gen_effective_owner_id, gen_effective_profile_name)
                        if gen_appearance_data.get("custom_display_name"):
                            generator_display_name = gen_appearance_data["custom_display_name"]
                        else:
                            generator_display_name = gen_effective_profile_name
                    
                    # Resolve Real-time settings
                        p_owner_id = participant['owner_id']
                        p_name = participant['profile_name']
                        p_index = self.cog.profile_manager._get_user_index(p_owner_id)
                        p_is_b = p_name in p_index.get("borrowed", [])
                        p_settings = self.cog.profile_manager._get_profile_config(p_owner_id, p_name, p_is_b) or {}

                        if p_settings.get("url_fetching_enabled", False) and round_url_text_contexts:
                            url_instr = "<url_research>\n[Context from links in current messages]:\n" + "\n".join(round_url_text_contexts) + "\n</url_research>"
                            contents_for_api_call.append({'role': 'user', 'parts': [url_instr]})

                        if not contents_for_api_call:
                            contents_for_api_call.append({'role': 'user', 'parts':[self.cog.global_prompts.get("KICKSTART_START", DEFAULT_KICKSTART_START)]})

                        dynamic_context_for_turn = image_gen_prompt

                        # _get_relevant_ltm_for_prompt uses this only as len(history) for the recall
                        # cooldown, so unified_log is the direct and more accurate equivalent.
                        ltm_recall_text = await self.cog.memory_manager._get_relevant_ltm_for_prompt(session_key, session.get("unified_log", []), owner_id, profile_name, dynamic_context_for_turn, round_author_name, channel.guild.id, triggering_user_id)

                        # [NEW] Check Image Gen intent vs Profile Toggle
                        turn_is_image_gen = False
                        if is_image_gen_round:
                            if p_settings.get("image_generation_enabled", False):
                                turn_is_image_gen = True
                            else:
                                # Re-inject prefix if toggle is OFF
                                initial_round_context = f"!image {image_gen_prompt}\n{initial_round_context}"

                        is_generator = turn_is_image_gen and participant_key == generator_profile_key

                    try:
                        api_key = self.cog.storage_manager._get_api_key_for_guild(channel.guild.id)
                        if not api_key: raise ValueError("Server API key is not configured.")
                        
                        msg_a_id = None
                        app_name, app_avatar = self._resolve_appearance_data(owner_id, profile_name)
                        
                        feedback_task_i = None
                        if i == 0:
                            feedback_task_i = feedback_task
                            # Adopt any placeholder the image-gen heartbeat created. It was sent
                            # by profile_order[0]'s bot, which is this participant, so the
                            # appearance already matches and the normal end-of-turn deletion
                            # path picks it up once it is in state_container.
                            if image_gen_placeholder_id:
                                msg_a_id = image_gen_placeholder_id
                        elif i > 0:
                            if participant.get('method') == 'child_bot':
                                if p_settings.get("child_bot_placeholder", False):
                                    custom_emoji = p_settings.get("placeholder_emoji") or PLACEHOLDER_EMOJI
                                    feedback_task_i = asyncio.create_task(self._send_child_bot_placeholder(participant['bot_id'], channel_id, custom_emoji))
                                else:
                                    await self.cog.manager_queue.put({
                                        "action": "send_to_child", "bot_id": participant['bot_id'],
                                        "payload": {"action": "start_typing", "channel_id": channel_id}
                                    })
                            else:
                                feedback_task_i = asyncio.create_task(self._send_channel_message(
                                    channel, f"{PLACEHOLDER_EMOJI}",
                                    profile_owner_id_for_appearance=owner_id, profile_name_for_appearance=profile_name
                                ))

                        # Initialise the persistent state container before we generate any media
                        custom_emoji = p_settings.get("placeholder_emoji") or PLACEHOLDER_EMOJI
                        state_container = {
                            'msg_a_id': msg_a_id,
                            'msg_b_id': None,
                            'app_name': app_name,
                            'app_avatar': app_avatar,
                            'message_type': "text",
                            'custom_emoji': custom_emoji
                        }

                        image_gen_error_msg = None

                        # [NEW] Hybrid STM: Rebuild history dynamically from unified_log.
                        # Whatever this round appended is reserved, so a batch of user
                        # messages does not push history out of the window. Anything built
                        # into contents_for_api_call before this point would be discarded
                        # by the assignment -- the URL context and the kickstart fallback
                        # are both re-applied below, from supplementary_parts.
                        bot_pid = self.cog.profile_manager._get_pid_from_name_any(owner_id, profile_name)

                        unified_log = session.get("unified_log", [])
                        contents_for_api_call = self.cog.session_manager._build_history_for_participant(
                            unified_log, bot_pid, p_settings,
                            len(profile_order) or 1,
                            reserved_tail=len(unified_log) - batch_start_index,
                        )

                        round_context_text = "\n".join([t[0] for t in new_round_turn_data])
                        dynamic_context_for_turn = (round_context_text + "\n" + "\n".join(responses_this_round)).strip()
                        
                        if not dynamic_context_for_turn and session.get("unified_log"):
                            # Fallback to the last available turn for vector search context
                            dynamic_context_for_turn = session["unified_log"][-1].get("content", "")

                        # Parallelise Training Examples, LTM retrieval, and Help Context
                        ltm_task = self.cog.memory_manager._get_relevant_ltm_for_prompt(session_key, contents_for_api_call, owner_id, profile_name, dynamic_context_for_turn, round_author_name, channel.guild.id, triggering_user_id)
                        training_task = self.cog.memory_manager._get_relevant_training_examples(owner_id, profile_name, dynamic_context_for_turn, channel.guild.id)
                        
                        help_task = None
                        if p_settings.get("help_mode_enabled", False):
                            help_task = self.cog.help_service._get_relevant_help_context(dynamic_context_for_turn, channel.guild.id)
                        
                        tasks_to_gather = [ltm_task, training_task]
                        if help_task: tasks_to_gather.append(help_task)
                            
                        gathered_results = await asyncio.gather(*tasks_to_gather)
                        ltm_recall_text = gathered_results[0]
                        training_examples_list = gathered_results[1]
                        help_context_text = gathered_results[2] if help_task else None

                        full_system_instruction, _, grounding_enabled, temp, top_p, top_k, primary_model, fallback_model_name = await asyncio.to_thread(
                            self._construct_system_instructions,
                            owner_id, profile_name, channel.id, is_multi_profile=True, training_examples_list=training_examples_list, recalled_ltm=ltm_recall_text
                        )
                        
                        # Critic persistence: how many further rounds a generated
                        # constraint stays in force is the profile's to set now, where it
                        # was fixed at one.
                        critic_constraints = None
                        critic_settings = resolve_critic_settings(p_settings)
                        if critic_settings["enabled"]:
                            # Check for cached constraints in the session
                            cache = session.setdefault("critic_cache", {}).get(participant_key)

                            if cache and cache.get("rounds", 0) > 0:
                                critic_constraints = cache["text"]
                                cache["rounds"] -= 1
                            else:
                                # "session" scope screens this profile against every
                                # participant's recent lines instead of only its own.
                                # Walked backwards and stopped at the quota, not filtered
                                # over the whole log: this is tail-local state.
                                session_transcript = None
                                if critic_settings["scope"] == "session":
                                    session_transcript = []
                                    for _t in reversed(unified_log):
                                        if (_t.get("is_user") or _t.get("type")
                                                or _t.get("is_hidden") or _t.get("compacted")):
                                            continue
                                        session_transcript.append(_t.get("content", ""))
                                        if len(session_transcript) >= critic_settings["lookback"]:
                                            break
                                    session_transcript.reverse()

                                # Generate fresh constraints
                                # _run_critic scans recent role=='model' turns; contents_for_api_call is
                                # already the unified_log-derived history for this participant.
                                critic_constraints = await self.cog.tools_service._run_critic(
                                    contents_for_api_call, speaker_display_name, channel.guild.id,
                                    p_config=p_settings,
                                    session_transcript=session_transcript,
                                    instructions=self.cog.profile_manager.resolve_critic_instructions(
                                        owner_id, profile_name),
                                )
                                if critic_constraints and critic_settings["persistence"] > 0:
                                    session["critic_cache"][participant_key] = {
                                        "text": critic_constraints,
                                        "rounds": critic_settings["persistence"],
                                    }

                        if critic_constraints:
                            constraints_block = self.cog.global_prompts.get("NEGATIVE_CONSTRAINTS", DEFAULT_NEGATIVE_CONSTRAINTS)
                            full_system_instruction += "\n\n" + constraints_block.format(constraints=critic_constraints)

                        dynamic_safety_settings = _resolve_safety_settings(channel, p_settings)

                        model = None
                        warning_message = None

                        # Pass thinking parameters to the model instance in the worker
                        t_params_worker = {
                            "thinking_persistence": p_settings.get("thinking_persistence", 10),
                            "thinking_summary_visible": p_settings.get("thinking_summary_visible", "off"),
                            "thinking_level": p_settings.get("thinking_level", "high"),
                            "thinking_budget": p_settings.get("thinking_budget", -1)
                        }
                        
                        # [NEW] Re-evaluate Tools for internal model reconstruction
                        model_tools = self._resolve_native_tools(p_settings)

                        # Provider resolution goes through APIService._instantiate_model, the one
                        # factory. The worker used to inline its own copy of the prefix parsing,
                        # and the two had already drifted: the factory reads a slash-free name
                        # containing "anthropic" as OpenRouter, this copy did not. The fallback
                        # path below already calls the factory, so the same configured model
                        # resolved to a different provider on the primary and fallback attempts.
                        try:
                            model = self.cog.api_service._instantiate_model(
                                primary_model, channel.guild.id, triggering_user_id,
                                full_system_instruction, dynamic_safety_settings,
                                t_params_worker, model_tools, p_settings,
                                openrouter_key_error=f"API Configuration Error: OpenRouter API Key missing for this server. Cannot load model '{primary_model}'.",
                                google_key_error=f"API Configuration Error: Google API Key missing for this server. Cannot load model '{primary_model}'.",
                            )
                        except ValueError as e:
                            warning_message = str(e)
                        except Exception as e:
                            warning_message = f"Model Initialization Error: Failed to instantiate model '{primary_model}'. {e}"
                        
                        session_key = (channel.id, owner_id, profile_name)

                        # Check if the last turn was from this model itself
                        if contents_for_api_call and contents_for_api_call[-1].get('role', contents_for_api_call[-1].get('role', 'user')) == 'model':
                            last_model_text = "".join(p if isinstance(p, str) else p.get('text', '') for p in contents_for_api_call[-1].get('parts', []))
                            if "<private_response>" in last_model_text:
                                pseudo_user_turn = {'role': 'user', 'parts': [self.cog.global_prompts.get("KICKSTART_CONTINUE", DEFAULT_KICKSTART_CONTINUE)]}
                            else:
                                pseudo_user_turn = {'role': 'user', 'parts': [self.cog.global_prompts.get("KICKSTART_IDLE", DEFAULT_KICKSTART_IDLE)]}
                            contents_for_api_call.append(pseudo_user_turn)

                        # Collect all supplementary context to inject into the final user turn
                        supplementary_parts = []

                        # [UPDATED] Standardised XML injection for pending whispers
                        pending_whispers = session.get("pending_whispers", {}).pop(participant_key, None)
                        if pending_whispers:
                            recap_template = self.cog.global_prompts.get("WHISPER_RECAP", DEFAULT_WHISPER_RECAP)
                            whisper_context = recap_template.format(whispers="\n---\n".join(pending_whispers))
                            supplementary_parts.append(whisper_context)

                        if grounding_context and p_settings.get("grounding_mode", "off") != "off":
                            # Already wrapped by _get_hybrid_grounding_context, which returns
                            # the summary inside <external_context> along with the footnote
                            # instruction. Re-wrapping here nested the tag inside itself for
                            # every grounded participant.
                            supplementary_parts.append(grounding_context)

                        if p_settings.get("url_fetching_enabled", False) and round_url_text_contexts:
                            url_instr = "<document_context>\n" + "\n".join(round_url_text_contexts) + "\n</document_context>"
                            supplementary_parts.append(url_instr)
                            
                        if help_context_text:
                            supplementary_parts.append(help_context_text)

                        # [FIXED] Ephemeral Media Injection: Manually add all current round media to the API call
                        # This allows participants to see images this round without them persisting in RAM history.
                        all_current_media = []
                        for _, _, turn_media in new_round_turn_data:
                            all_current_media.extend(turn_media)
                        
                        if all_current_media:
                            supplementary_parts.extend(all_current_media)
                        
                        if is_image_gen_round:
                            if generated_image_path_for_round:
                                if is_generator:
                                    present_template = self.cog.global_prompts.get("IMAGE_PRESENT", DEFAULT_IMAGE_PRESENT)
                                    system_note = present_template.format(prompt=image_gen_prompt)
                                else:
                                    other_template = self.cog.global_prompts.get("IMAGE_PRESENT_OTHER", DEFAULT_IMAGE_PRESENT_OTHER)
                                    system_note = other_template.format(name=generator_display_name, prompt=image_gen_prompt)
                                
                                text_gen_parts = [
                                    system_note, 
                                    {"mime_type": "image/png", "url": generated_image_path_for_round}
                                ]
                                supplementary_parts.extend(text_gen_parts)
                            else:
                                if is_generator:
                                    fail_reason = image_gen_error_msg or "Safety Filter / Unknown"
                                    failed_template = self.cog.global_prompts.get("IMAGE_FAILED", DEFAULT_IMAGE_FAILED)
                                    system_note = failed_template.format(prompt=image_gen_prompt, reason=fail_reason)
                                    supplementary_parts.append(system_note)

                        if not contents_for_api_call:
                            contents_for_api_call.append({'role': 'user', 'parts': [self.cog.global_prompts.get("KICKSTART_START", DEFAULT_KICKSTART_START)]})

                        # Inject supplementary parts into the final user turn to ensure alternating roles
                        if supplementary_parts:
                            if contents_for_api_call[-1].get('role') == 'user':
                                contents_for_api_call[-1]['parts'].extend(supplementary_parts)
                            else:
                                contents_for_api_call.append({'role': 'user', 'parts': supplementary_parts})

                        # [NEW] Advanced Params Injection
                        p_index = self.cog.profile_manager._get_user_index(owner_id)
                        p_is_borrowed = profile_name in p_index.get("borrowed", [])
                        profile_settings = self.cog.profile_manager._get_profile_config(owner_id, profile_name, p_is_borrowed) or {}
                        
                        adv_params = {
                            "frequency_penalty": profile_settings.get("frequency_penalty"),
                            "presence_penalty": profile_settings.get("presence_penalty"),
                            "repetition_penalty": profile_settings.get("repetition_penalty"),
                            "min_p": profile_settings.get("min_p"),
                            "top_a": profile_settings.get("top_a")
                        }
                        adv_params = {k: v for k, v in adv_params.items() if v is not None}

                        gen_config = {"temperature": temp, "top_p": top_p, "top_k": top_k, "_advanced_params": adv_params}

                        status = "api_error"
                        response = None
                        fallback_used = False
                        api_error_reason = None
                        main_api_error = None
                        # Persist the existing state container populated during image generation
                        if state_container is None:
                            custom_emoji = p_settings.get("placeholder_emoji") or PLACEHOLDER_EMOJI
                            state_container = {
                                'msg_a_id': msg_a_id,
                                'msg_b_id': None,
                                'app_name': app_name,
                                'app_avatar': app_avatar,
                                'message_type': "text",
                                'custom_emoji': custom_emoji
                            }

                        # Published so /cancel can tell generation from delivery. Released
                        # below, after the heartbeat is stopped and the placeholders are
                        # gone -- releasing earlier would let a cancel land in the gap.
                        self.cog.session_manager.register_in_flight(session, state_container)
                        
                        all_participant_names = []
                        for p_data_temp in session.get("profiles", []):
                            p_owner_id_temp = p_data_temp['owner_id']
                            p_name_temp = p_data_temp['profile_name']
                            p_index_temp = self.cog.profile_manager._get_user_index(p_owner_id_temp)
                            p_is_borrowed_temp = p_name_temp in p_index_temp.get("borrowed", [])
                            p_effective_owner_id = p_owner_id_temp
                            p_effective_profile_name = p_name_temp
                            if p_is_borrowed_temp:
                                borrowed_data = self.cog.profile_manager._get_profile_config(p_owner_id_temp, p_name_temp, True) or {}
                                p_effective_owner_id = int(borrowed_data.get("original_owner_id", p_owner_id_temp))
                                p_effective_profile_name = borrowed_data.get("original_profile_name", p_name_temp)
                            display_name_temp = p_effective_profile_name
                            appearance_data_temp = self.cog.profile_manager._get_user_appearance(p_effective_owner_id, p_effective_profile_name)
                            if appearance_data_temp.get("custom_display_name"):
                                display_name_temp = appearance_data_temp["custom_display_name"]
                            all_participant_names.append(display_name_temp)
                        
                        try:
                            if feedback_task_i:
                                try:
                                    feedback_result_i = await feedback_task_i
                                    if participant.get('method') == 'child_bot' and p_settings.get("child_bot_placeholder", False):
                                        if feedback_result_i:
                                            try: first_placeholder_message = await channel.fetch_message(feedback_result_i)
                                            except: pass
                                            msg_a_id = feedback_result_i
                                    else:
                                        if feedback_result_i:
                                            first_placeholder_message = feedback_result_i[0]
                                            msg_a_id = first_placeholder_message.id
                                            
                                    if state_container:
                                        state_container['msg_a_id'] = msg_a_id
                                except Exception as e:
                                    print(f"Feedback task error: {e}")

                            # A primary that could not even be constructed is an error like any other,
                            # and has to reach the same handler. Reporting it here instead meant a
                            # configured fallback -- quite possibly a different provider whose key is
                            # present -- never got the chance it gets whenever the primary constructs
                            # and then fails, so a missing OpenRouter key was fatal rather than a
                            # reason to fall back. The pre-formatted text rides on the exception so
                            # the handler reports the configuration error, not a truncation of it.
                            if not model:
                                init_error = RuntimeError(warning_message or 'Internal API Initialization Error')
                                init_error.formatted_reason = warning_message or 'Internal API Initialization Error'
                                raise init_error

                            gen_task = asyncio.create_task(self._generate_with_heartbeat(
                                model, contents_for_api_call, gen_config, channel, participant, msg_a_id, is_fallback=False, app_name=app_name, app_avatar=app_avatar, existing_state=state_container
                            ))

                            with mem_probe.probe(f"  participant turn {i}", peak=False):
                                response, state_container = await gen_task

                            if not response or not response.candidates:
                                raise ValueError("Response blocked or empty")
                            
                            raw_text_check = getattr(response, 'text', "").strip()
                            temp_scrubbed = _strip_neuro_update_and_scrub(raw_text_check, all_participant_names)

                            if not temp_scrubbed:
                                raise ValueError("Empty Response (AI produced no text content)")

                            status = "success"
                        except asyncio.CancelledError:
                            if state_container and state_container.get('sending_task'):
                                state_container['sending_task'].cancel()
                            await self._safe_delete_placeholder(channel, state_container.get('msg_a_id') if state_container else msg_a_id, bot_id=participant.get('bot_id'))
                            await self._safe_delete_placeholder(channel, state_container.get('msg_b_id') if state_container else None, bot_id=participant.get('bot_id'))
                            if 'contents_for_api_call' in locals():
                                contents_for_api_call.clear()
                                del contents_for_api_call
                            raise
                        except Exception as e:
                            is_timeout_main = isinstance(e, TimeoutError)
                            # An instantiation failure arrives already phrased for the
                            # user; _format_api_error would truncate that to 80 chars and
                            # lose the half naming the key that is missing.
                            main_api_error = getattr(e, 'formatted_reason', None) or _format_api_error(e)
                            if hasattr(e, 'state_container'): state_container = e.state_container

                            # is_real_model, not truthiness: an explicit "no fallback"
                            # reads back as the string NONE, which is truthy and unequal
                            # to the primary, so the retry was spent instantiating a model
                            # by that name. Same three-way answer run_with_fallback uses.
                            if not is_real_model(fallback_model_name) or primary_model == fallback_model_name:
                                api_error_reason = main_api_error
                            else:
                                try:
                                    fb_name = fallback_model_name
                                    fallback_instance = self.cog.api_service._instantiate_model(fb_name, channel.guild.id, triggering_user_id, full_system_instruction, dynamic_safety_settings, t_params_worker, model_tools, p_settings)
                                    
                                    response, state_container = await self._generate_with_heartbeat(
                                        fallback_instance, contents_for_api_call, gen_config, channel, participant, msg_a_id, is_fallback=True, app_name=app_name, app_avatar=app_avatar, existing_state=state_container
                                    )
                                        
                                    if not response or not response.candidates:
                                        raise ValueError("Response blocked or empty")
                                    
                                    fb_raw_check = getattr(response, 'text', "").strip()
                                    temp_scrubbed = _strip_neuro_update_and_scrub(fb_raw_check, all_participant_names)

                                    if not temp_scrubbed:
                                        raise ValueError("Empty Response (AI produced no text content)")

                                    fallback_used = True
                                    self.cog._log_api_call(user_id=triggering_user_id, guild_id=channel.guild.id, context="multi_profile_fallback", model_used=fb_name, status="success")
                                except asyncio.CancelledError:
                                    if state_container and state_container.get('sending_task'):
                                        state_container['sending_task'].cancel()
                                    await self._safe_delete_placeholder(channel, state_container.get('msg_a_id') if state_container else msg_a_id, bot_id=participant.get('bot_id'))
                                    await self._safe_delete_placeholder(channel, state_container.get('msg_b_id') if state_container else None, bot_id=participant.get('bot_id'))
                                    if 'contents_for_api_call' in locals():
                                        contents_for_api_call.clear()
                                        del contents_for_api_call
                                    raise
                                except Exception as retry_e:
                                    is_timeout_fallback = isinstance(retry_e, TimeoutError)
                                    if hasattr(retry_e, 'state_container'): state_container = retry_e.state_container
                                    
                                    if is_timeout_main and is_timeout_fallback:
                                        api_error_reason = ERR_REASON_TIMEOUT_BOTH
                                    else:
                                        api_error_reason = _format_api_error(retry_e)
                                    status = "api_error"
                        finally:
                            # `or primary_model`: this block now also runs when the primary
                            # never constructed, and a log row naming no model at all says
                            # less than one naming the model that could not be built.
                            self.cog._log_api_call(user_id=triggering_user_id, guild_id=channel.guild.id, context="multi_profile", model_used=model or primary_model, status=status)

                        was_blocked = False
                        if not response or not response.candidates:
                            reason = api_error_reason or "Unknown Error"
                            is_safety = False
                            if response and response.prompt_feedback and response.prompt_feedback.block_reason: 
                                reason = response.prompt_feedback.block_reason.name.replace('_', ' ').title()
                                is_safety = True
                            
                            custom_main = p_settings.get("error_response", ERR_GENERAL_ERROR)
                            response_text = custom_main
                            
                            if is_safety:
                                turn_warnings.append(ERR_SAFETY_BLOCK.format(reason=reason))
                            elif "Rate Limit" in reason:
                                turn_warnings.append(reason)
                            else:
                                if is_real_model(fallback_model_name) and primary_model != fallback_model_name:
                                    turn_warnings.append(WARN_BOTH_MODELS_FAILED.format(reason=reason))
                                else:
                                    turn_warnings.append(WARN_MAIN_MODEL_FAILED.format(reason=reason))
                            
                            was_blocked = True
                        else:
                            try:
                                # Use the filtered text attribute from the model wrapper to exclude thoughts
                                raw_text = getattr(response, 'text', "")
                                if hasattr(response, 'raw') and response.raw.candidates and hasattr(response.raw.candidates[0], 'grounding_metadata'):
                                    raw_text = _add_inline_citations(raw_text, response.raw.candidates[0].grounding_metadata)
                                raw_text = raw_text.strip()
                                
                                raw_text, parsed_neuro_state = self._extract_and_apply_neuro_state(raw_text, p_owner_id, p_name)

                                response_text = _scrub_response_text(raw_text, participant_names=all_participant_names)
                                
                                # Extract Native grounding sources & URL context
                                if response_text:
                                    if hasattr(response, 'raw') and response.raw.candidates:
                                        if hasattr(response.raw.candidates[0], 'grounding_metadata'):
                                            metadata = response.raw.candidates[0].grounding_metadata
                                            if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks is not None:
                                                for chunk in metadata.grounding_chunks:
                                                    if hasattr(chunk, 'web'):
                                                        turn_grounding_sources.append({'uri': chunk.web.uri, 'title': chunk.web.title})

                                        if hasattr(response.raw.candidates[0], 'url_context_metadata'):
                                            url_metadata = response.raw.candidates[0].url_context_metadata
                                            if hasattr(url_metadata, 'url_metadata') and url_metadata.url_metadata is not None:
                                                for u in url_metadata.url_metadata:
                                                    if hasattr(u, 'retrieved_url') and u.retrieved_url:
                                                        turn_grounding_sources.append({'uri': u.retrieved_url, 'title': 'URL Context'})

                                    sources_text_list = _format_citation_subtext(turn_grounding_sources)
                                
                                # [UPDATED] Differentiated error messaging for spammed vs empty content
                                if not response_text:
                                    custom_main = p_settings.get("error_response", ERR_GENERAL_ERROR)
                                    response_text = custom_main
                                    warn_tmp = WARN_BOTH_MODELS_FAILED if fallback_used else WARN_MAIN_MODEL_FAILED
                                    turn_warnings.append(warn_tmp.format(reason=ERR_REASON_EMPTY_RESPONSE))
                                    was_blocked = True
                                
                            except ValueError:
                                reason = response.candidates[0].finish_reason.name.replace('_', ' ').title()
                                response_text = p_settings.get("error_response", ERR_GENERAL_ERROR)
                                turn_warnings.append(ERR_SAFETY_BLOCK.format(reason=reason))
                                was_blocked = True

                        if fallback_used and p_settings.get("show_fallback_indicator", True):
                            turn_warnings.append(WARN_FALLBACK_USED)
                            turn_warnings.append(WARN_MAIN_MODEL_FAILED.format(reason=main_api_error))
                            
                        # --- Native Tool Extraction (Citations) ---
                        # Native citations are now handled inline directly in the text response above.
                            
                        # --- Placeholder Update & Sending ---
                        if not was_blocked:
                            # spawn_after: everything below this point -- speech synthesis
                            # above all -- happens before the turn is delivered, and a child
                            # bot with no placeholder of its own has nothing on screen once
                            # its typing indicator expires nine seconds in.
                            await self._update_sending_placeholder(channel, participant.get('method', 'webhook'), participant.get('bot_id'), state_container, t1_start_mono, spawn_after=10.0)

                        t2_end_mono = time.monotonic()
                        duration = t2_end_mono - t1_start_mono
                        sent_timestamp = datetime.datetime.now(datetime.timezone.utc) # Approximation

                        timezone_str = profile_settings.get("timezone", "UTC")
                        main_history_line = _format_history_entry(
                            speaker_display_name, sent_timestamp, response_text, timezone_str,
                            entity_id=self.cog.profile_manager._get_profile_id(owner_id, profile_name))
                        try:
                            t1_formatted = t1_start_utc.astimezone(ZoneInfo(timezone_str)).strftime('%I:%M:%S %p %Z')
                        except Exception:
                            t1_formatted = t1_start_utc.strftime('%I:%M:%S %p UTC')
                        metadata_line = f"(Thought Initiated: {t1_formatted} | Duration: {duration:.2f}s)"
                        history_line = f"{main_history_line.strip()}\n{metadata_line}\n"

                        model_content_obj = {'role': 'model', 'parts': [history_line]}
                        user_content_obj = {'role': 'user', 'parts': [history_line]}

                        if owner_id in self.cog.debug_users:
                            try:
                                user_to_dm = self.cog.bot.get_user(owner_id)
                                if user_to_dm:
                                    turns_for_debug = []
                                    if grounding_context and participant_key == grounding_profile_key:
                                        turns_for_debug.append({'role': 'user', 'parts': [grounding_context, "\n"]})
                                    if ltm_recall_text:
                                        turns_for_debug.append({'role': 'user', 'parts': [ltm_recall_text, "\n"]})
                                    
                                    turns_for_debug.append(model_content_obj)

                                    debug_message = _format_debug_prompt(turns_for_debug)
                                    await user_to_dm.send(debug_message)
                            except Exception as e:
                                print(f"Failed to send debug DM to user {owner_id}: {e}")

                    except Exception as e:
                        print(f"Multi-profile generation error for '{profile_name}': {e}")
                        traceback.print_exc()
                        response_text = f"{custom_main}\n\n-# Blocked due to: **Unknown**."

                    participant_key = (participant['owner_id'], participant['profile_name'])

                    # Inside the participant loop, after response extraction:
                    
                    thought_text = ""
                    if hasattr(response, 'thought') and response.thought:
                        thought_text = response.thought.strip()
                    
                    # Deduplication logic (handled in previous step)
                    if thought_text and response_text:
                        if thought_text in response_text:
                            response_text = response_text.replace(thought_text, "").strip()
                        
                        response_text = re.sub(r'^\**Thoughts:?\**\n?', '', response_text, flags=re.IGNORECASE).strip()
                        response_text = re.sub(r'^\**Reasoning:?\**\n?', '', response_text, flags=re.IGNORECASE).strip()

                        if len(thought_text) > 50:
                            snippet = thought_text[:50]
                            if snippet in response_text:
                                parts = response_text.split(snippet, 1)
                                if len(parts) > 1:
                                    response_text = parts[1].strip()

                    # [NEW] Reformat summary text: one sentence per line
                    if thought_text:
                        sentences = _split_into_sentences_with_abbreviations(thought_text)
                        thought_text = "\n".join(sentences)

                    # Update Display Text and Prepare Thought File
                    display_text = response_text
                    thought_file_to_send = None
                    if thought_text and p_settings.get("thinking_summary_visible") == "on":
                        thought_file_to_send = discord.File(io.BytesIO(thought_text.encode('utf-8')), filename="thinking_summary.txt")

                    t2_end_mono = time.monotonic()
                    duration = t2_end_mono - t1_start_mono
                    sent_timestamp = datetime.datetime.now(datetime.timezone.utc) 

                    timezone_str = profile_settings.get("timezone", "UTC")
                    profile_id = self.cog.profile_manager._get_profile_id(owner_id, profile_name)
                    main_history_line = _format_history_entry(speaker_display_name, sent_timestamp, response_text, timezone_str, entity_id=profile_id)
                    
                    history_line = main_history_line

                    turn_id = str(uuid.uuid4())
                    bot_pid = self.cog.profile_manager._get_pid_from_name_any(owner_id, profile_name)
                    
                    # [NEW] Meta collection for App Context Menu tracing
                    meta = {
                        "duration": round(duration, 2),
                        "model": model.model_name.replace("models/", "").replace("OPENROUTER/", "").replace("GOOGLE/", "") if hasattr(model, 'model_name') else (fallback_model_name or "Unknown"),
                        "fallback": fallback_used,
                        "input_tokens": getattr(response, 'input_tokens', 0) if response else 0,
                        "output_tokens": getattr(response, 'output_tokens', 0) if response else 0,
                        "reasoning_tokens": getattr(response, 'reasoning_tokens', 0) if response else 0,
                        "training_recalled": len(training_examples_list) if 'training_examples_list' in locals() and training_examples_list else 0,
                        "grounding_sources":[s.get('uri') for s in turn_grounding_sources if isinstance(s, dict) and s.get('uri')] if 'turn_grounding_sources' in locals() and turn_grounding_sources else [],
                        "ltms_recalled":[]
                    }
                    if 'ltm_recall_text' in locals() and ltm_recall_text:
                        lines = ltm_recall_text.split('\n')
                        clean_lines = [l.strip() for l in lines if l.strip() and not l.startswith("<")]
                        meta["ltms_recalled"] = [l[:100] + "..." if len(l) > 100 else l for l in clean_lines]
                        
                    if 'parsed_neuro_state' in locals() and parsed_neuro_state:
                        meta["neuro_state"] = parsed_neuro_state

                    turn_object = {
                        "turn_id": turn_id,
                        "is_user": False,
                        "speaker_pid": bot_pid,
                        "owner_id": owner_id,
                        "profile_name": profile_name,
                        "message_ids": [],
                        "content": history_line,
                        "meta": meta
                    }
                    
                    # Clean up any legacy signature if it exists
                    turn_object.pop('thought_signature', None)
                    
                    session.setdefault("unified_log", []).append(intern_turn(turn_object))
                    session['last_speaker_key'] = participant_key

                    # Append only. The delivery branches below flush as soon as the
                    # message_ids are known, and the round-end flush backs that up.
                    session_type = session.get("type", "multi")
                    self.cog.session_manager.mark_session_dirty((channel_id, None, None), session_type)

                    # [NEW] Unified Synthesis Logic
                    audio_file_for_send = None
                    
                    if profile_settings.get("speech_tts_enabled", False) and session.get("audio_mode", "off") == "on":
                        s_voice = profile_settings.get("speech_voice", DEFAULT_SPEECH_VOICE)
                        s_model = profile_settings.get("speech_model")
                        if s_model and "none" not in s_model.lower():
                            # 1. Build Contextual Round Transcript
                            round_transcript = ""
                            for idx, prev_resp in enumerate(responses_this_round[:-1]):
                                prev_p = profile_order[idx]
                                prev_app = self.cog.user_appearances.get(str(prev_p['owner_id']), {}).get(prev_p['profile_name'], {})
                                prev_name = prev_app.get("custom_display_name") or prev_p['profile_name']
                                round_transcript += f"{prev_name}: {prev_resp}\n\n"

                            # 2. Construct Conditional Markdown Prompt
                            s_temp = float(profile_settings.get("speech_temperature", 1.0))
                            s_arch = profile_settings.get("speech_archetype", "")
                            s_acc = profile_settings.get("speech_accent", "")
                            s_dyn = profile_settings.get("speech_dynamics", "")
                            s_styl = profile_settings.get("speech_style", "")
                            s_pace = profile_settings.get("speech_pacing", "")

                            prompt_parts = []
                            if s_arch or s_acc:
                                part = f"# AUDIO PROFILE: {speaker_display_name}\n"
                                if s_arch: part += f"Archetype: {s_arch}\n"
                                if s_acc: part += f"Accent: {s_acc}\n"
                                prompt_parts.append(part.strip())
                            if s_dyn:
                                part = f"## THE SCENE\nDynamics: {s_dyn}\nAction: Fluid conversation."
                                prompt_parts.append(part.strip())
                            if s_styl or s_pace:
                                part = "### DIRECTOR'S NOTES\n"
                                if s_styl: part += f"Style: {s_styl}\n"
                                if s_pace: part += f"Pacing: {s_pace}\n"
                                prompt_parts.append(part.strip())
                            if round_transcript:
                                prompt_parts.append(f"#### SAMPLE CONTEXT\nPrevious turn flow:\n{round_transcript.strip()}")
                            prompt_parts.append(f"#### TRANSCRIPT\n{speaker_display_name}: {response_text}")

                            # Only once there is direction above it -- see
                            # TTS_SYNTHESIS_PREAMBLE. prompt_parts always ends with the
                            # transcript, so anything beyond it is a director section.
                            if len(prompt_parts) > 1:
                                prompt_parts.insert(0, TTS_SYNTHESIS_PREAMBLE)

                            tts_priming_prompt = "\n\n".join(prompt_parts)
                            
                            # 3. Synthesise Audio
                            #
                            # The heartbeat started before this is still the one running, so
                            # the phase is renamed rather than restarted. Without it a TTS
                            # round trip reads as "Sending..." for half a minute, which is
                            # both wrong and the least informative thing it could say.
                            if state_container:
                                state_container['phase_label'] = "Synthesising speech"
                            try:
                                turn_audio_stream = await self.cog.media_service._generate_google_tts(
                                    tts_priming_prompt,
                                    channel.guild.id,
                                    model_id=s_model,
                                    voice_name=s_voice,
                                    temperature=s_temp,
                                    fallback_model_id=profile_settings.get("speech_fallback_model"),
                                )
                            finally:
                                if state_container:
                                    state_container['phase_label'] = "Sending"
                            
                            if turn_audio_stream:
                                audio_file_for_send = discord.File(turn_audio_stream, filename=f"voice_{turn_id[:4]}.wav")
                            else:
                                turn_warnings.append(WARN_VOICE_SYNTHESIS_FAILED.format(reason="API Error or Unknown"))

                    file_to_send = None
                    extra_audio_file = None
                    if is_generator and generated_image_path_for_round:
                        file_to_send = discord.File(generated_image_path_for_round, filename="generated_image.png")
                        if audio_file_for_send:
                            extra_audio_file = audio_file_for_send
                    elif audio_file_for_send:
                        file_to_send = audio_file_for_send

                    is_realistic_typing = profile_settings.get("realistic_typing_enabled", False)

                    # Realistic typing streams its own first chunk instead of editing the
                    # placeholder, so the placeholder has to come down before the send --
                    # but not a moment earlier. This teardown used to sit above the speech
                    # synthesis block, which meant that with realistic typing *and* audio
                    # enabled the placeholder vanished and the channel then showed nothing
                    # at all for the whole duration of the TTS round-trip. Deferring it to
                    # here keeps the placeholder (and its "Sending..." heartbeat) up until
                    # the turn is genuinely ready to be delivered.
                    if is_realistic_typing:
                        await self._stop_sending_heartbeat(state_container)
                        msg_a_to_delete = state_container.get('msg_a_id') if state_container else msg_a_id
                        msg_b_to_delete = state_container.get('msg_b_id') if state_container else None
                        await self._safe_delete_placeholder(channel, msg_a_to_delete)
                        await self._safe_delete_placeholder(channel, msg_b_to_delete)
                        if state_container:
                            state_container['msg_a_id'] = None
                            state_container['msg_b_id'] = None

                    if participant.get('method') == 'child_bot':
                        # Stop any pending sending heartbeat task immediately
                        await self._stop_sending_heartbeat(state_container)

                        # Delete placeholder before child bot delivers
                        msg_to_delete = state_container.get('msg_a_id') if state_container else msg_a_id
                        if msg_to_delete:
                            await self._safe_delete_placeholder(channel, msg_to_delete, bot_id=participant.get('bot_id'))
                            if state_container:
                                state_container['msg_a_id'] = None

                        rmode = profile_settings.get("response_mode", "regular")
                        reply_id = None
                        should_ping = False

                        if i == 0:
                            if anchor_message and rmode == "mention":
                                display_text = f"{anchor_message.author.mention} {display_text}"
                            reply_id = anchor_message.id if (anchor_message and rmode in ["reply", "mention_reply"]) else None
                            should_ping = (rmode == "mention_reply")

                        payload = {
                            "channel_id": channel.id, "content": display_text,
                            "realistic_typing": is_realistic_typing,
                            "typing_cps": profile_settings.get("typing_cps", 30.0),
                            "typing_max_delay": profile_settings.get("typing_max_delay", 2.5),
                            "typing_mode": profile_settings.get("typing_mode", "sentence"),
                            "reply_to_id": reply_id, "ping": should_ping
                        }

                        if file_to_send:
                            # The child bot lives in this process, so it can open the
                            # file itself: no read, no base64, nothing resident here.
                            attachment_data = None
                            if is_generator and generated_image_path_for_round:
                                attachment_data = {
                                    "filename": "generated_image.png",
                                    "path": generated_image_path_for_round
                                }
                            elif audio_file_for_send:
                                turn_audio_stream.seek(0)
                                attachment_data = {
                                    "filename": f"voice_{turn_id[:4]}.wav",
                                    "data": turn_audio_stream.read()
                                }
                            if attachment_data:
                                payload["attachment"] = attachment_data
                            if attachment_data and attachment_data.get("path"):
                                # file_to_send was opened from this same path for the
                                # webhook branch and is never sent here -- execute_send
                                # opens its own handle. Release the descriptor rather
                                # than leaving it to the round's teardown.
                                file_to_send.close()
                                file_to_send = None

                        # Direct in-process execution (replaces 45-second queue wait)
                        sent_child_messages = await self.cog.child_bot_manager.execute_send(participant['bot_id'], payload)
                        
                        if sent_child_messages:
                            session['last_bot_message_id'] = sent_child_messages[-1].id
                            for sm in sent_child_messages:
                                turn_object.setdefault("message_ids", []).append(sm.id)

                        # Dispatch citation sources directly
                        if sources_text_list:
                            for source_msg in sources_text_list:
                                s_msgs = await self.cog.child_bot_manager.execute_send(participant['bot_id'], {
                                    "channel_id": channel.id, "content": source_msg,
                                    "realistic_typing": False, "reply_to_id": None, "ping": False
                                })
                                if s_msgs:
                                    for sm in s_msgs:
                                        turn_object.setdefault("message_ids", []).append(sm.id)

                        # Dispatch extra audio attachment directly
                        if extra_audio_file and 'turn_audio_stream' in locals() and turn_audio_stream:
                            turn_audio_stream.seek(0)
                            a_msgs = await self.cog.child_bot_manager.execute_send(participant['bot_id'], {
                                "channel_id": channel.id, "content": "", "realistic_typing": False,
                                "attachment": {"filename": f"voice_{turn_id[:4]}.wav", "data": turn_audio_stream.read()}
                            })
                            if a_msgs:
                                for sm in a_msgs:
                                    turn_object.setdefault("message_ids", []).append(sm.id)

                        # Dispatch thinking summary directly
                        if thought_file_to_send and thought_text:
                            t_msgs = await self.cog.child_bot_manager.execute_send(participant['bot_id'], {
                                "channel_id": channel.id, "content": "", "realistic_typing": False,
                                "attachment": {"filename": "thinking_summary.txt", "data": thought_text.encode('utf-8')}
                            })
                            if t_msgs:
                                for sm in t_msgs:
                                    turn_object.setdefault("message_ids", []).append(sm.id)

                        # The webhook branch persists its message_ids the moment they are
                        # known; this branch used to leave them to whichever save came next
                        # -- the following participant's, or the round-level one. A crash in
                        # that window left the turn on disk with an empty message_ids list,
                        # which is what regenerate, delete and the audit view key off, so the
                        # messages stayed in the channel with nothing able to address them.
                        if turn_object.get("message_ids"):
                            session_type = session.get("type", "multi")
                            await self.cog.session_manager.flush_session((channel_id, None, None), session_type)

                    else: # Webhook logic
                        sent_messages = await self._send_channel_message(
                            channel, display_text, target_message_to_edit=None,
                            profile_owner_id_for_appearance=owner_id, profile_name_for_appearance=profile_name,
                            file=file_to_send, reply_to=(anchor_message if i == 0 else None)
                        )
                        
                        if sources_text_list:
                            for source_msg in sources_text_list:
                                s_msgs = await self._send_channel_message(
                                    channel, source_msg, bypass_typing=True,
                                    profile_owner_id_for_appearance=owner_id, profile_name_for_appearance=profile_name
                                )
                                if s_msgs: sent_messages.extend(s_msgs)
                        
                        # [NEW] Dispatch extra audio follow-up for Webhook
                        if extra_audio_file:
                            await self._send_channel_message(
                                channel, "", file=extra_audio_file,
                                profile_owner_id_for_appearance=owner_id, profile_name_for_appearance=profile_name,
                                bypass_typing=True
                            )

                        if thought_file_to_send:
                            t_msgs = await self._send_channel_message(
                                channel, "", file=thought_file_to_send,
                                profile_owner_id_for_appearance=owner_id, profile_name_for_appearance=profile_name,
                                bypass_typing=True
                            )
                            if t_msgs: sent_messages.extend(t_msgs)

                        if sent_messages:
                            session['last_bot_message_id'] = sent_messages[-1].id
                            session_type = session.get("type", "multi")
                            
                            for msg in sent_messages:
                                turn_object.setdefault("message_ids", []).append(msg.id)
                            
                            await self.cog.session_manager.flush_session((channel_id, None, None), session_type)
                    
                    # --- Dispatch Warnings and Clean Up Placeholders ---
                    await self._stop_sending_heartbeat(state_container)
                    self.cog.session_manager.release_in_flight(session, state_container)

                    msg_a_to_delete = state_container.get('msg_a_id') if state_container else msg_a_id
                    msg_b_to_delete = state_container.get('msg_b_id') if state_container else None

                    await self._safe_delete_placeholder(channel, msg_a_to_delete, bot_id=participant.get('bot_id'))
                    await self._safe_delete_placeholder(channel, msg_b_to_delete, bot_id=participant.get('bot_id'))

                    if state_container:
                        state_container['msg_a_id'] = None
                        state_container['msg_b_id'] = None
                        
                    await self._dispatch_warnings(channel, participant.get('method', 'webhook'), participant.get('bot_id'), turn_warnings, owner_id, profile_name, session, turn_object)
                    
                    # [FIXED] Turn cleanup: release turn-specific buffers without purging the round's generated image
                    if 'contents_for_api_call' in locals():
                        contents_for_api_call.clear()
                        del contents_for_api_call
                    if 'response' in locals():
                        del response

                    # Check if session was cancelled mid-turn and stop the round
                    if not session.get('is_running') and not session.get('is_regenerating'):
                        break

                    is_last_participant = (participant == profile_order[-1])
                    if not is_last_participant:
                        # Pipelined visual feedback for next participant
                        next_p = profile_order[i + 1]
                        if next_p.get('method') == 'child_bot':
                            next_p_index = self.cog.profile_manager._get_user_index(next_p['owner_id'])
                            next_p_is_b = next_p['profile_name'] in next_p_index.get("borrowed", [])
                            next_p_settings = self.cog.profile_manager._get_profile_config(next_p['owner_id'], next_p['profile_name'], next_p_is_b) or {}
                            if not next_p_settings.get("child_bot_placeholder", False):
                                await self.cog.manager_queue.put({
                                    "action": "send_to_child", "bot_id": next_p['bot_id'],
                                    "payload": {"action": "start_typing", "channel_id": channel_id}
                                })

                        # Yield briefly to ensure Discord orders messages correctly
                        await asyncio.sleep(0.2)
                        
                        batched_triggers = []
                        while not session['task_queue'].empty():
                            try: batched_triggers.append(session['task_queue'].get_nowait())
                            except asyncio.QueueEmpty: break
                        
                        if batched_triggers:
                            for trigger in batched_triggers:
                                # [UPDATED] Unpack structured tuples in mid-round batches to ensure all messages are read
                                if isinstance(trigger, tuple) and len(trigger) > 1 and isinstance(trigger[1], discord.Message):
                                    trigger = trigger[1]

                                if isinstance(trigger, discord.Message):
                                    content_lower = trigger.clean_content.lower()
                                    image_prefixes = ("!image", "!imagine")
                                    if content_lower.startswith(image_prefixes):
                                        used_prefix = next((p for p in image_prefixes if content_lower.startswith(p)), "!image")
                                        session['pending_image_gen_data'] = {
                                            'prompt': trigger.clean_content[len(used_prefix):].strip(),
                                            'anchor_message': trigger
                                        }

                                    author_name = trigger.author.display_name
                                    reply_context = await self._resolve_reply_context(trigger)
                                    # [UPDATED] Apply newline separator to mid-round batch content
                                    content = f"{reply_context}\n{trigger.clean_content}" if reply_context else trigger.clean_content
                                    
                                    # [NEW] Batch URL Context Logic
                                    any_url_enabled_batch = False
                                    for p in session['profiles']:
                                        p_index_batch = self.cog.profile_manager._get_user_index(p['owner_id'])
                                        p_is_b_batch = p['profile_name'] in p_index_batch.get("borrowed", [])
                                        p_settings_batch = self.cog.profile_manager._get_profile_config(p['owner_id'], p['profile_name'], p_is_b_batch) or {}
                                        
                                        u_mode = p_settings_batch.get("url_mode", "off")
                                        if "url_mode" not in p_settings_batch:
                                            u_mode = "rag" if p_settings_batch.get("url_fetching_enabled", False) else "off"
                                            
                                        if u_mode == "rag":
                                            any_url_enabled_batch = True; break
                                    
                                    url_text_batch = None
                                    url_media_batch = []
                                    
                                    if any_url_enabled_batch:
                                        url_text_list, url_media, _ = await self.cog.tools_service._process_urls_in_content(content, trigger.guild.id, {"url_fetching_enabled": True}, warning_channel=channel)
                                        if url_text_list: url_text_batch = "\n".join(url_text_list)
                                        url_media_batch = url_media

                                    # [NEW] Localized User Timestamp Logic (Batch)
                                    u_index_batch2 = self.cog.profile_manager._get_user_index(trigger.author.id)
                                    u_prof_batch = self.cog.session_manager._get_active_user_profile_name_for_channel(trigger.author.id, channel_id)
                                    u_is_b_batch = u_prof_batch in u_index_batch2.get("borrowed", [])
                                    u_sett_batch = self.cog.profile_manager._get_profile_config(trigger.author.id, u_prof_batch, u_is_b_batch) or {}
                                    batch_tz = u_sett_batch.get("timezone", "UTC")
                                    batch_hash = _get_user_hash(trigger.author.id)

                                    user_line = _format_history_entry(author_name, trigger.created_at, content, batch_tz, entity_id=batch_hash)
                                    
                                    batch_msg_media = []
                                    message_attachments = [a for a in trigger.attachments if a.content_type and (a.content_type.startswith("image/") or a.content_type.startswith("audio/") or a.content_type.startswith("video/"))]
                                    for attachment in message_attachments:
                                        try:
                                            batch_msg_media.append({"url": attachment.url, "mime_type": attachment.content_type})
                                        except Exception as e:
                                            print(f"Failed to process batched media attachment {attachment.filename}: {e}")

                                    if trigger.author.id in self.cog.debug_users:
                                        try:
                                            user_to_dm = self.cog.bot.get_user(trigger.author.id)
                                            if user_to_dm:
                                                # Create temporary debug object with all context
                                                debug_parts = [user_line]
                                                if url_text_batch: debug_parts.append(url_text_batch)
                                                debug_parts.extend(url_media_batch)
                                                debug_parts.extend(batch_msg_media)
                                                debug_obj = {'role': 'user', 'parts': debug_parts}
                                                
                                                debug_message = _format_debug_prompt([debug_obj])
                                                await user_to_dm.send(debug_message)
                                        except Exception as e:
                                            print(f"Failed to send batched user turn debug DM to user {trigger.author.id}: {e}")
                                    
                                    new_turn_id = str(uuid.uuid4())
                                    new_turn_object = {
                                        "turn_id": new_turn_id,
                                        "is_user": True,
                                        "speaker_pid": str(trigger.author.id),
                                        "message_ids": [trigger.id],
                                        "content": user_line
                                    }
                                    if url_text_batch:
                                        new_turn_object["url_context"] = url_text_batch
                                    session.setdefault("unified_log", []).append(intern_turn(new_turn_object))

                            all_triggers_for_round.extend(batched_triggers)
                    
                    # No forced gc.collect() here. It ran once per participant per round and
                    # cost ~11 ms of pure GIL-held CPU on a warm heap (worse on the e2-micro),
                    # stalling Discord heartbeats; asyncio.to_thread does not avoid that, since
                    # a collection holds the GIL regardless of which thread calls it. Image
                    # buffers are freed by refcounting the moment the last reference drops, so
                    # the pass only ever reclaimed reference cycles the automatic collector
                    # would have caught anyway.

                self.cog.session_manager.mark_session_dirty((channel_id, None, None), session_type)

                for trigger in all_triggers_for_round:
                    if trigger is not None:
                        try:
                            session['task_queue'].task_done()
                        except (ValueError, RuntimeError):
                            pass

                # [FIXED] Round-End Memory Purge: Purge all generated and context data after all participants finish
                if 'new_round_turn_data' in locals():
                    for i in range(len(new_round_turn_data)):
                        base, url, media = new_round_turn_data[i]
                        media.clear()
                    del new_round_turn_data
                
                if 'url_media_parts' in locals():
                    url_media_parts.clear()
                    del url_media_parts

                if 'shared_media_content_obj' in locals():
                    del shared_media_content_obj

                if generated_image_path_for_round:
                    if os.path.exists(generated_image_path_for_round):
                        try: os.remove(generated_image_path_for_round)
                        except OSError: pass
                    generated_image_path_for_round = None

                guild_id = self.cog.bot.get_channel(channel_id).guild.id
                
                if not was_blocked:
                    # Check each participant that ACTUALLY SPOKE this round
                    for participant in profile_order:
                        owner_id = participant['owner_id']
                        profile_name = participant['profile_name']
                        p_index = self.cog.profile_manager._get_user_index(owner_id)
                        p_is_borrowed = profile_name in p_index.get("borrowed", [])
                        p_settings = self.cog.profile_manager._get_profile_config(owner_id, profile_name, p_is_borrowed) or {}
                        
                        if not p_settings.get("ltm_creation_enabled", False): continue
                        
                        # Increment individual counter
                        participant['ltm_counter'] = participant.get('ltm_counter', 0) + 1
                        
                        interval = p_settings.get("ltm_creation_interval", 10)

                        if participant['ltm_counter'] >= interval:
                            # Offloaded to a background task so a slow summary doesn't block the
                            # queue. The actual summarise -> embed -> store chain lives in
                            # LtmCaptureMixin._summarize_and_store_ltm, shared with /memorise, which
                            # awaits it directly instead since it has to report success back to the
                            # admin who ran it.
                            async def background_ltm_gen(o_id, p_name, p_stgs, r_author, g_id, t_user_id):
                                try:
                                    await self._summarize_and_store_ltm(
                                        channel_id, session, o_id, p_name, p_stgs, g_id, r_author, t_user_id
                                    )
                                except Exception as e:
                                    print(f"Background LTM generation failed for {p_name}: {e}")

                            asyncio.create_task(background_ltm_gen(owner_id, profile_name, p_settings, round_author_name, guild_id, triggering_user_id))

                            participant['ltm_counter'] = 0

                # AGGRESSIVE GC: Clear references
                if 'new_round_content_objects' in locals(): del new_round_content_objects
                if 'all_triggers_for_round' in locals(): del all_triggers_for_round

                # Trim the unified log to be the single source of truth for the session's history window.
                if len(session.get("unified_log", [])) > 1000:
                    session["unified_log"] = session["unified_log"][-1000:]
                    # A new list object whose head no longer matches the cold segment.
                    # Zeroing the boundary makes the round-end flush a full rewrite,
                    # which re-seals the trimmed log.
                    session["_log_cold_len"] = 0

                # [NEW] Mandatory Round-End Persistence
                # Ensures the transcript is saved immediately after the last participant speaks.
                dummy_session_key = (channel_id, None, None)
                with mem_probe.probe("  end-of-round flush"):
                    await self.cog.session_manager.flush_session(dummy_session_key, session_type)

                # Rolling synopsis, after the round's own flush has landed. Not earlier:
                # inserting a synopsis turn shifts every index after it, and
                # batch_start_index -- captured at round start -- is read while each
                # participant's prompt is built. No-ops unless the session enables it.
                try:
                    await self._run_session_compaction(channel_id)
                except Exception as e:
                    print(f"Session compaction failed for channel {channel_id}: {e}")

            except asyncio.CancelledError:
                break
            except RuntimeError as e:
                if "Session is closed" in str(e):
                    break
                print(f"Error in multi-profile worker for channel {channel_id}: {e}")
                traceback.print_exc()
            except Exception as e:
                print(f"Error in multi-profile worker for channel {channel_id}: {e}")
                traceback.print_exc()
            finally:
                # Round has concluded, AI is no longer active
                session['is_running'] = False
                # Nothing this round published is in flight any more. The per-participant
                # release below the delivery step handles the normal path; this catches
                # the cancelled and errored ones, which leave the loop without reaching
                # it and would otherwise grow this list by one per abandoned round.
                session['in_flight'] = [c for c in session.get('in_flight', ())
                                        if c.get('sending_task')]
        
        # [NEW] Lifecycle protection: Remove from background set and clear reference
        ctask = asyncio.current_task()
        self.cog.background_tasks.discard(ctask)
        if session.get('worker_task') == ctask:
            session['worker_task'] = None

