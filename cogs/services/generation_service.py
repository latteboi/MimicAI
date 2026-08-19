import io
import os
import re
import time
import uuid
import random
import base64
import asyncio
import discord
import traceback
import collections
import datetime
import httpx
from zoneinfo import ZoneInfo
from ..utils.constants import (
    defaultConfig, OLLAMA_LOCAL_URL, PLACEHOLDER_EMOJI,
    ERR_GENERAL_ERROR, ERR_REASON_EMPTY_RESPONSE, ERR_REASON_TIMEOUT_BOTH, ERR_SAFETY_BLOCK,
    WARN_BOTH_MODELS_FAILED, WARN_FALLBACK_USED, WARN_MAIN_MODEL_FAILED, WARN_VOICE_SYNTHESIS_FAILED,
)
from ..utils.helpers import _add_inline_citations, _format_api_error, _format_citation_subtext, _format_debug_prompt, _format_history_entry, _get_user_hash, _scrub_response_text, _split_into_sentences_with_abbreviations
from ..managers.memory_manager import encode_embedding_b64
from .api_service import GoogleGenAIModel, OllamaModel, OpenRouterModel

from .generation._shared import _resolve_safety_settings, _strip_neuro_update_and_scrub
from .generation.heartbeat import HeartbeatMixin
from .generation.prompt_builder import PromptBuilderMixin
from .generation.delivery import DeliveryMixin
from .generation.regeneration import RegenerationMixin
from .generation.speak import SpeakAsMixin
from .generation.global_chat import GlobalChatMixin
from .generation.whisper import WhisperMixin


class GenerationService(HeartbeatMixin, PromptBuilderMixin, DeliveryMixin, RegenerationMixin, SpeakAsMixin, GlobalChatMixin, WhisperMixin):
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
                    
                    while session.get('is_purging') or session.get('is_regenerating'):
                        await asyncio.sleep(0.5)
                        
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
                                    is_or = model_raw.upper().startswith("OPENROUTER/")
                                    model_name = model_raw[11:] if is_or else (model_raw[7:] if model_raw.upper().startswith("GOOGLE/") else model_raw)
                                    channel = self.cog.bot.get_channel(channel_id)
                                    guild_id = channel.guild.id if channel and getattr(channel, 'guild', None) else 0
                                    api_key = self.cog.storage_manager._get_api_key_for_guild(guild_id, "openrouter" if is_or else "gemini")
                                    if api_key:
                                        if is_or: m = OpenRouterModel(model_name, api_key=api_key, system_instruction=sys_instr, thinking_params={})
                                        else: m = GoogleGenAIModel(api_key=api_key, model_name=model_name, system_instruction=sys_instr)
                                        hist_text = ""
                                        for ht in session.get("unified_log", [])[-10:]:
                                            hist_text += f"{ht.get('content', '')}\n"
                                        resp = await m.generate_content_async([f"Recent History:\n{hist_text}\n\nGenerate your Director's prompt."])
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
                generated_image_bytes_for_round = None
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

                for i, trigger in enumerate(all_triggers_for_round):
                    if not trigger:
                        continue

                    message_trigger, reaction_trigger, message_payload = None, None, None
                    
                    if i == 0:
                        if isinstance(trigger, tuple):
                            if trigger[0] == 'reply':
                                _, message_trigger, starting_profile_override = trigger
                            elif trigger[0] == 'reaction': 
                                _, reaction_trigger, starting_profile_override = trigger
                                try:
                                    ch = self.cog.bot.get_channel(reaction_trigger.channel_id)
                                    msg_obj = await ch.fetch_message(reaction_trigger.message_id)
                                    await msg_obj.clear_reaction(reaction_trigger.emoji)
                                except: pass
                            elif trigger[0] == 'reaction_single': 
                                _, reaction_trigger, starting_profile_override = trigger
                                try:
                                    ch = self.cog.bot.get_channel(reaction_trigger.channel_id)
                                    msg_obj = await ch.fetch_message(reaction_trigger.message_id)
                                    await msg_obj.clear_reaction(reaction_trigger.emoji)
                                except: pass
                            elif trigger[0] == 'child_mention': _, message_payload, starting_profile_override = trigger
                        elif isinstance(trigger, discord.RawReactionActionEvent): reaction_trigger = trigger
                        elif isinstance(trigger, str):
                            # Handle string prompts
                            content = trigger
                            turn_id = str(uuid.uuid4())
                            
                            # XML-standardised system turn without Name/Timestamp header
                            system_content = f"<system_note>\n{content}\n</system_note>"
                            
                            turn_object = {
                                "turn_id": turn_id,
                                "is_user": False,
                                "speaker_pid": "SYSTEM",
                                "message_ids": [],
                                "content": system_content
                            }
                            session.setdefault("unified_log", []).append(turn_object)
                            
                            new_round_turn_data.append((system_content, None, []))
                            
                            message_trigger = None
                        else: message_trigger = trigger
                    else:
                        # [UPDATED] Unpack structured tuples in batches to prevent lost replies/mentions/child bots
                        if isinstance(trigger, discord.Message):
                            message_trigger = trigger
                        elif isinstance(trigger, tuple) and len(trigger) > 1:
                            if isinstance(trigger[1], discord.Message):
                                message_trigger = trigger[1]
                            elif isinstance(trigger[1], dict):
                                message_payload = trigger[1]

                    # --- Deduplication Check ---
                    check_id = None
                    if message_trigger: check_id = message_trigger.id
                    elif message_payload: check_id = message_payload.get('id')
                    
                    if check_id:
                        if check_id in recent_processed_ids:
                            continue # Skip duplicate trigger
                        recent_processed_ids.append(check_id)
                    # ---------------------------

                    if (message_trigger or message_payload) and not is_image_gen_round:
                        trigger_content = message_payload['content'] if message_payload else message_trigger.clean_content
                        content_lower = trigger_content.lower()
                        
                        image_prefixes = ("!image", "!imagine")
                        if any(content_lower.startswith(p) for p in image_prefixes):
                            # Detection is now prefix-based only
                            is_image_gen_round = True
                            used_prefix = next((p for p in image_prefixes if content_lower.startswith(p)), "!image")
                            image_gen_prompt = trigger_content[len(used_prefix):].strip()
                            image_gen_anchor_message = message_trigger or message_payload

                    if message_trigger or message_payload:
                        is_child_mention = message_payload is not None
                        trigger_obj = message_payload if is_child_mention else message_trigger
                        
                        triggering_user_id = trigger_obj['author_id'] if is_child_mention else trigger_obj.author.id
                        author_name = trigger_obj['author_name'] if is_child_mention else trigger_obj.author.display_name
                        if round_author_name == "A user": round_author_name = author_name
                        
                        reply_context = ""
                        if is_child_mention and trigger_obj.get('replied_to'):
                            reply_context = "[Replying to a previous message]"
                        elif message_trigger:
                            reply_context = await self._resolve_reply_context(message_trigger)

                        content = trigger_obj['content'] if is_child_mention else trigger_obj.clean_content
                        
                        raw_att_list = trigger_obj['attachments'] if is_child_mention else trigger_obj.attachments
                        async with httpx.AsyncClient(timeout=10.0) as text_client:
                            text_att_content = await self.cog.media_service._process_text_attachments(raw_att_list, text_client)
                        
                        if text_att_content:
                            content = f"{content}\n\n{text_att_content}"

                        content = f"{reply_context}\n{content}" if reply_context else content
                        
                        # [NEW] URL Context Logic: Enforce Profile Setting & Separation
                        any_url_enabled = False
                        any_url_rag = False
                        for p in session['profiles']:
                            p_index = self.cog.profile_manager._get_user_index(p['owner_id'])
                            p_is_b = p['profile_name'] in p_index.get("borrowed", [])
                            p_settings = self.cog.profile_manager._get_profile_config(p['owner_id'], p['profile_name'], p_is_b) or {}
                            
                            u_mode = p_settings.get("url_mode", "off")
                            if "url_mode" not in p_settings:
                                u_mode = "rag" if p_settings.get("url_fetching_enabled", False) else "off"
                                
                            if u_mode != "off":
                                any_url_enabled = True
                            if u_mode == "rag":
                                any_url_rag = True
                        
                        url_text_content = None
                        trigger_media_parts = []
                        
                        if any_url_enabled and any_url_rag:
                            # Defer URL fetching until after placeholder is sent
                            pending_url_fetches.append({
                                "content": content,
                                "guild_id": trigger_obj['guild_id'] if is_child_mention else trigger_obj.guild.id,
                                "turn_data_index": len(new_round_turn_data)
                            })

                        # [NEW] Localized User Timestamp Logic
                        u_index_author = self.cog.profile_manager._get_user_index(triggering_user_id)
                        u_prof_author = self.cog.session_manager._get_active_user_profile_name_for_channel(triggering_user_id, channel_id)
                        u_is_b_author = u_prof_author in u_index_author.get("borrowed", [])
                        u_sett_author = self.cog.profile_manager._get_profile_config(triggering_user_id, u_prof_author, u_is_b_author) or {}
                        author_tz = u_sett_author.get("timezone", "UTC")
                        user_hash = _get_user_hash(triggering_user_id)

                        created_at = datetime.datetime.now(datetime.timezone.utc) if is_child_mention else trigger_obj.created_at
                        user_line = _format_history_entry(author_name, created_at, content, author_tz, entity_id=user_hash)
                        
                        turn_id = str(uuid.uuid4())
                        trigger_id = trigger_obj['id'] if is_child_mention else trigger_obj.id
                        
                        turn_object = {
                            "turn_id": turn_id, 
                            "is_user": True,
                            "speaker_pid": str(triggering_user_id),
                            "message_ids": [trigger_id],
                            "content": user_line
                        }
                        if url_text_content:
                            # Clear any previous URL context from the log to make the new one exclusive
                            for turn in session.get("unified_log", []):
                                if "url_context" in turn:
                                    del turn["url_context"]
                            turn_object["url_context"] = url_text_content
                            
                        session.setdefault("unified_log", []).append(turn_object)
                        
                        if pending_url_fetches and pending_url_fetches[-1]["turn_data_index"] == len(new_round_turn_data):
                            pending_url_fetches[-1]["turn_object"] = turn_object

                        # [NEW] Immediate persistence for user turns
                        await self.cog.session_manager._save_session_to_disk((channel_id, None, None), session_type, session.get("unified_log", []))
                        
                        # Initialize list for standard message attachments/reply images
                        new_message_parts = []
                        
                        # --- Logic to fetch image from replied-to message ---
                        msg_for_ref = message_trigger
                        if not msg_for_ref and message_payload:
                            try:
                                # For child bots, we only have payload, so fetch the discord.Message
                                r_ch = self.cog.bot.get_channel(message_payload['channel_id'])
                                if r_ch:
                                    msg_for_ref = await r_ch.fetch_message(message_payload['id'])
                            except Exception: pass

                        if msg_for_ref and msg_for_ref.reference:
                            ref_img = None 
                            try:
                                ref_msg = msg_for_ref.reference.resolved
                                if not ref_msg:
                                    r_ch = self.cog.bot.get_channel(msg_for_ref.reference.channel_id)
                                    if r_ch:
                                        ref_msg = await r_ch.fetch_message(msg_for_ref.reference.message_id)
                                
                                if ref_msg and ref_msg.attachments:
                                    # Find the first image/audio/video attachment in the referenced message
                                    ref_media = next((a for a in ref_msg.attachments if a.content_type and (a.content_type.startswith("image/") or a.content_type.startswith("audio/") or a.content_type.startswith("video/"))), None)
                                    if ref_media:
                                        new_message_parts.append({"url": ref_media.url, "mime_type": ref_media.content_type})
                            except Exception as e:
                                print(f"Error fetching replied media: {e}")

                        attachments = trigger_obj['attachments'] if is_child_mention else [
                            a for a in trigger_obj.attachments 
                            if a.content_type and (
                                a.content_type.startswith("image/") or 
                                a.content_type.startswith("audio/") or 
                                a.content_type.startswith("video/")
                            )
                        ]

                        if attachments:
                            att_tags = []
                            for attachment in attachments:
                                try:
                                    attachment_url = attachment['url'] if is_child_mention else attachment.url
                                    fname = attachment.get('filename', 'attachment.png') if is_child_mention and isinstance(attachment, dict) else getattr(attachment, 'filename', 'attachment.png')
                                    
                                    ctype = "image/png"
                                    if not is_child_mention and attachment.content_type:
                                        ctype = attachment.content_type
                                    elif is_child_mention and isinstance(attachment, dict):
                                        ctype = attachment.get('content_type', "image/png")
                                    
                                    new_message_parts.append({"url": attachment_url, "mime_type": ctype})
                                    att_tags.append(f"[Attached Image: {fname}]")
                                except Exception as e:
                                    print(f"Failed to process media attachment in multi-profile trigger: {e}")
                            
                            if att_tags:
                                content = f"{' '.join(att_tags)}\n{content}".strip()
                                user_line = _format_history_entry(author_name, created_at, content, author_tz, entity_id=user_hash)
                                turn_object["content"] = user_line

                        # Combine standard attachments with URL-extracted media
                        trigger_media_parts.extend(new_message_parts)
                        
                        # Store raw components for gating logic
                        new_round_turn_data.append((user_line, url_text_content, trigger_media_parts))

                        if triggering_user_id in self.cog.debug_users:
                            try:
                                user_to_dm = self.cog.bot.get_user(triggering_user_id)
                                if user_to_dm:
                                    # Create a temporary debug obj
                                    debug_parts = [user_line]
                                    if url_text_content: debug_parts.append(url_text_content)
                                    debug_parts.extend(trigger_media_parts)
                                    debug_obj = {'role': 'user', 'parts': debug_parts}
                                    
                                    debug_message = _format_debug_prompt([debug_obj])
                                    await user_to_dm.send(debug_message)
                            except Exception as e:
                                print(f"Failed to send user turn debug DM to user {triggering_user_id}: {e}")

                    elif reaction_trigger and i == 0:
                        triggering_user_id = reaction_trigger.user_id
                        user_obj = self.cog.bot.get_user(triggering_user_id)
                        if user_obj:
                            round_author_name = user_obj.display_name

                for content_obj in new_round_turn_data:
                    pass

                profile_order = []
                session_mode = session.get("session_mode", "sequential")
                channel = self.cog.bot.get_channel(channel_id)

                is_single_turn_only = False
                if isinstance(initial_trigger, tuple) and initial_trigger[0] == 'reaction_single':
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

                # --- Synchronised Feedback Step ---
                # Hoisted to here, directly after the channel and API-key guards. The three
                # things a placeholder needs — a resolved channel, a valid key, and
                # profile_order[0] for the webhook name and avatar — are all settled by this
                # point, and nothing between here and generation can decide not to respond.
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

                was_blocked = False
                generated_image_bytes_for_round = None
                generator_profile_key = None
                generator_display_name = "A participant"
                image_gen_error_msg = None

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
                        g_safety_level = g_profile_settings.get('safety_level', 'low')
                        g_dynamic_safety_settings = _resolve_safety_settings(g_safety_level)

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
                            image_gen_prompt = f"{image_gen_prompt}\n\nUse this information to help generate the image:\n{g_context}"
                        else:
                            grounding_context = g_context
                            # [NEW] Sticky Grounding: Purge previous search results from history
                            for turn in session.get("unified_log", []):
                                if "grounding_context" in turn:
                                    del turn["grounding_context"]
                            
                            # Attach new summary to the latest turn (the trigger)
                            if session.get("unified_log"):
                                session["unified_log"][-1]["grounding_context"] = g_context

                        grounding_sources = g_sources
                    grounding_profile_key = (g_owner_id, g_profile_name)

                # Unpack and apply URL results
                url_updates_made = False
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
                                    
                    if u_m:
                        url_media_parts.extend(u_m)
                        # Update new_round_turn_data
                        idx = fetch_info["turn_data_index"]
                        user_line, old_url_text, old_media = new_round_turn_data[idx]
                        old_media.extend(u_m)
                        new_round_turn_data[idx] = (user_line, url_text_content if u_t else old_url_text, old_media)
                        
                if url_updates_made:
                    await self.cog.session_manager._save_session_to_disk((channel_id, None, None), session_type, session.get("unified_log", []))

                # --- NEW IMAGE GENERATION LOGIC ---
                if is_image_gen_round and generator_profile_key:
                    gen_owner_id, gen_profile_name = generator_profile_key
                    gen_idx = self.cog.profile_manager._get_user_index(gen_owner_id)
                    gen_is_b = gen_profile_name in gen_idx.get("borrowed", [])
                    gen_cfg = self.cog.profile_manager._get_profile_config(gen_owner_id, gen_profile_name, gen_is_b) or {}
                    
                    if gen_cfg.get("image_generation_enabled", False):
                        try:
                            api_key = self.cog.storage_manager._get_api_key_for_guild(channel.guild.id)
                            if not api_key: raise ValueError("Server API key not configured.")
                            
                            img_model_name = gen_cfg.get("image_generation_model", "GOOGLE/gemini-2.5-flash-image")
                            if img_model_name.upper().startswith("GOOGLE/"): img_model_name = img_model_name[7:]
                            
                            system_instruction = self.cog.media_service._get_image_gen_system_instruction(gen_owner_id, gen_profile_name)
                            
                            # Combine prompt with appearance if needed
                            appearance_text = ""
                            source_prompts = self.cog.profile_manager._get_profile_prompts(gen_owner_id, gen_profile_name) or {}
                            if source_prompts:
                                appearance_lines = source_prompts.get("persona", {}).get("appearance", [])
                                appearance_text = "\n".join([self.cog.storage_manager._decrypt_data(line) for line in appearance_lines])
                            
                            final_prompt_text = image_gen_prompt
                            if appearance_text.strip():
                                prompt_lower = image_gen_prompt.lower()
                                second_person_pronouns = ["you", "your", "yourself", "u", "ur"]
                                if any(pronoun in prompt_lower.split() for pronoun in second_person_pronouns) or \
                                   generator_display_name.lower() in prompt_lower or \
                                   gen_profile_name.lower() in prompt_lower:
                                    final_prompt_text = f"Your appearance:\n{appearance_text.strip()}\n\nUser's prompt:\n{image_gen_prompt}"
                            
                            ref_images = []
                            for _, _, turn_media in new_round_turn_data:
                                for media in turn_media:
                                    if media.get("mime_type", "").startswith("image/"):
                                        ref_images.append(media)
                            
                            parts = [final_prompt_text]
                            for ref in ref_images[:10]:
                                parts.append({"url": ref["url"], "mime_type": ref.get("mime_type", "image/png")})

                            # Determine safety
                            safety_level_str = gen_cfg.get("safety_level", "low")
                            dynamic_safety_settings = _resolve_safety_settings(safety_level_str)

                            image_model = GoogleGenAIModel(
                                api_key=api_key,
                                model_name=img_model_name,
                                system_instruction=system_instruction,
                                safety_settings=dynamic_safety_settings
                            )
                            
                            status = "api_error"

                            # Image generation is the slowest call in the system (tens of
                            # seconds) and was the one path with no heartbeat: the placeholder
                            # created above just sat as a static emoji until the image landed.
                            # Resolve the placeholder id first so _generate_with_heartbeat has
                            # something to edit. Awaiting feedback_task here is safe — it is an
                            # asyncio.Task, so the later await in the participant loop returns
                            # the same cached result rather than re-running it.
                            img_msg_a_id = None
                            if feedback_task is not None:
                                try:
                                    fb_result = await feedback_task
                                    if fb_result:
                                        if first_participant and first_participant.get('method') == 'child_bot':
                                            img_msg_a_id = fb_result
                                        else:
                                            img_msg_a_id = fb_result[0].id
                                except Exception as e:
                                    print(f"Image-gen feedback task error: {e}")

                            gen_app_name, gen_app_avatar = self._resolve_appearance_data(gen_owner_id, gen_profile_name)
                            image_state_container = {
                                'msg_a_id': img_msg_a_id,
                                'msg_b_id': None,
                                'app_name': gen_app_name,
                                'app_avatar': gen_app_avatar,
                                'message_type': "text",
                                'custom_emoji': gen_cfg.get("placeholder_emoji") or PLACEHOLDER_EMOJI,
                            }

                            response, image_state_container = await self._generate_with_heartbeat(
                                image_model,
                                [{'role': 'user', 'parts': parts}],
                                None,
                                channel,
                                first_participant,
                                img_msg_a_id,
                                app_name=gen_app_name,
                                app_avatar=gen_app_avatar,
                                existing_state=image_state_container,
                            )
                            status = "blocked_by_safety" if not response.candidates else "success"
                            
                            if not response.candidates:
                                reason = "Safety Filter"
                                if response.prompt_feedback and response.prompt_feedback.block_reason: 
                                    reason = response.prompt_feedback.block_reason.name.replace('_', ' ').title()
                                image_gen_error_msg = f"the safety filter ({reason})"
                            else:
                                candidate = response.candidates[0]
                                if candidate.finish_reason.name != 'STOP':
                                    image_gen_error_msg = f"process stopped: {candidate.finish_reason.name.replace('_', ' ').title()}"
                                else:
                                    img_bytes = next((part.inline_data.data for part in candidate.content.parts if getattr(part, 'inline_data', None) and part.inline_data.mime_type.startswith('image/')), None)
                                    if img_bytes:
                                        generated_image_bytes_for_round = img_bytes
                                        def _write_img():
                                            import tempfile
                                            fd, path = tempfile.mkstemp(suffix=".png")
                                            with os.fdopen(fd, 'wb') as f:
                                                f.write(img_bytes)
                                            return path
                                        generated_image_path_for_round = await asyncio.to_thread(_write_img)
                                    else:
                                        image_gen_error_msg = "no image data returned"
                                        
                            self.cog._log_api_call(user_id=session.get('owner_id', 0), guild_id=channel.guild.id, context="image_generation_multi", model_used=image_model, status=status)
                                
                        except Exception as e:
                            image_gen_error_msg = _format_api_error(e)
                            print(f"Error generating image in multi-profile round: {e}")

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

                    if not self.cog.profile_manager._check_unrestricted_safety_policy(owner_id, profile_name, channel):
                        error_message = f"[System Notice: '{profile_name}' cannot respond. Profiles with 'Unrestricted 18+' safety are only permitted in age-restricted channels.]"
                        
                        # Send the message immediately, bypassing placeholders/typing for this turn
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
                            contents_for_api_call.append({'role': 'user', 'parts':["<internal_note>Start the conversation.</internal_note>"]})

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
                        if p_settings.get("url_fetching_enabled", False) and round_url_text_contexts:
                            url_instr = "<url_research>\n[Context from links in current messages]:\n" + "\n".join(round_url_text_contexts) + "\n</url_research>"
                            contents_for_api_call.append({'role': 'user', 'parts': [url_instr]})

                        if not contents_for_api_call:
                            contents_for_api_call.append({'role': 'user', 'parts':["<internal_note>Start the conversation.</internal_note>"]})

                        # [NEW] Hybrid STM: Rebuild history dynamically from unified_log
                        bot_pid = self.cog.profile_manager._get_pid_from_name_any(owner_id, profile_name)
                        
                        unified_log = session.get("unified_log", [])
                        past_log = unified_log[:batch_start_index]
                        current_batch_log = unified_log[batch_start_index:]
                        
                        stm_length = int(p_settings.get("stm_length", defaultConfig.CHATBOT_MEMORY_LENGTH))
                        if stm_length > 0:
                            past_log = past_log[-stm_length:]
                        else:
                            past_log = []
                            
                        combined_log = past_log + current_batch_log
                        contents_for_api_call = self.cog.session_manager._build_history_for_participant(combined_log, bot_pid, p_settings)

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
                        
                        # [UPDATED] Critic Persistence Logic (2 Rounds)
                        critic_constraints = None
                        if p_settings.get("critic_enabled", False):
                            # Check for cached constraints in the session
                            cache = session.setdefault("critic_cache", {}).get(participant_key)
                            
                            if cache and cache.get("rounds", 0) > 0:
                                critic_constraints = cache["text"]
                                cache["rounds"] -= 1
                            else:
                                # Generate fresh constraints
                                # _run_critic scans recent role=='model' turns; contents_for_api_call is
                                # already the unified_log-derived history for this participant.
                                critic_constraints = await self.cog.tools_service._run_critic(contents_for_api_call, speaker_display_name, channel.guild.id)
                                if critic_constraints:
                                    # Store constraints and set to 1 round (current + 1 future)
                                    session["critic_cache"][participant_key] = {"text": critic_constraints, "rounds": 1}

                        if critic_constraints:
                            full_system_instruction += f"\n\nNEGATIVE CONSTRAINTS (STRICT ADHERENCE REQUIRED):\n{critic_constraints}"

                        safety_level_str = p_settings.get('safety_level', 'low')

                        dynamic_safety_settings = _resolve_safety_settings(safety_level_str)

                        # Factory Logic
                        actual_name = primary_model
                        is_openrouter = False
                        is_ollama = False
                        
                        if primary_model.startswith("OPENROUTER/"):
                            actual_name = primary_model[11:]
                            is_openrouter = True
                        elif primary_model.startswith("OLLAMA/"):
                            actual_name = primary_model[7:]
                            is_ollama = True
                        elif primary_model.startswith("GOOGLE/"):
                            actual_name = primary_model[7:]
                        elif "/" in primary_model or "grok" in primary_model.lower():
                            # Heuristic for OpenRouter models without explicit prefix
                            is_openrouter = True

                        model = None
                        warning_message = None

                        # [FIXED] Pass thinking parameters to the model instance in the worker
                        t_params_worker = {
                            "thinking_persistence": p_settings.get("thinking_persistence", 10),
                            "thinking_summary_visible": p_settings.get("thinking_summary_visible", "off"),
                            "thinking_level": p_settings.get("thinking_level", "high"),
                            "thinking_budget": p_settings.get("thinking_budget", -1)
                        }
                        
                        # [NEW] Re-evaluate Tools for internal model reconstruction
                        grounding_mode_native = p_settings.get("grounding_mode", "off")
                        if isinstance(grounding_mode_native, bool): grounding_mode_native = "rag" if grounding_mode_native else "off"
                        elif grounding_mode_native in ["on", "on+"]: grounding_mode_native = "rag"
                        
                        url_mode_native = p_settings.get("url_mode", "off")
                        if "url_mode" not in p_settings:
                            url_mode_native = "rag" if p_settings.get("url_fetching_enabled", False) else "off"

                        model_tools_list = []
                        if grounding_mode_native == "native":
                            model_tools_list.append({"google_search": {}})
                        if url_mode_native == "native":
                            model_tools_list.append({"url_context": {}})
                            
                        model_tools = model_tools_list if model_tools_list else None

                        if is_openrouter:
                            or_key = self.cog.storage_manager._get_api_key_for_guild(channel.guild.id, provider="openrouter")
                            if or_key:
                                # [FIXED] Passing thinking_params to OpenRouter constructor
                                model = OpenRouterModel(actual_name, api_key=or_key, system_instruction=full_system_instruction, thinking_params=t_params_worker)
                            else:
                                warning_message = f"API Configuration Error: OpenRouter API Key missing for this server. Cannot load model '{primary_model}'."
                        elif is_ollama:
                            ollama_host = p_settings.get("ollama_host_url", OLLAMA_LOCAL_URL)
                            model = OllamaModel(actual_name, api_url=ollama_host, system_instruction=full_system_instruction, thinking_params=t_params_worker)
                        else:
                            try:
                                # [NEW] Pass thinking_params and tools here
                                model = GoogleGenAIModel(
                                    api_key=api_key, 
                                    model_name=actual_name, 
                                    system_instruction=full_system_instruction, 
                                    safety_settings=dynamic_safety_settings,
                                    thinking_params=t_params_worker,
                                    tools=model_tools
                                )
                            except Exception as e:
                                warning_message = f"Model Initialization Error: Failed to instantiate Google model '{actual_name}'. {e}"
                        
                        session_key = (channel.id, owner_id, profile_name)

                        # Check if the last turn was from this model itself
                        if contents_for_api_call and contents_for_api_call[-1].get('role', contents_for_api_call[-1].get('role', 'user')) == 'model':
                            last_model_text = "".join(p if isinstance(p, str) else p.get('text', '') for p in contents_for_api_call[-1].get('parts', []))
                            if "<private_response>" in last_model_text:
                                pseudo_user_turn = {'role': 'user', 'parts': ["<internal_note>Continue the public conversation.</internal_note>"]}
                            else:
                                pseudo_user_turn = {'role': 'user', 'parts': ["<internal_note>No response from anyone OR no user is present.</internal_note>"]}
                            contents_for_api_call.append(pseudo_user_turn)

                        # Collect all supplementary context to inject into the final user turn
                        supplementary_parts = []

                        # [UPDATED] Standardised XML injection for pending whispers
                        pending_whispers = session.get("pending_whispers", {}).pop(participant_key, None)
                        if pending_whispers:
                            whisper_context = "<whisper_context>\n"
                            whisper_context += "SYSTEM NOTE: You previously received and replied to these private whispers. Keep them in mind for context, but behave how you would treat whispers.\n"
                            whisper_context += "\n---\n" + "\n---\n".join(pending_whispers) + "\n</whisper_context>"
                            supplementary_parts.append(whisper_context)

                        if grounding_context and p_settings.get("grounding_mode", "off") != "off":
                            g_instr = f"<external_context>\n{grounding_context}\n</external_context>"
                            supplementary_parts.append(g_instr)

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
                                system_note = f"<image_context>You have just generated the following image based on the prompt: '{image_gen_prompt}'. Present it with a comment.</image_context>" if is_generator else f"<image_context>'{generator_display_name}' just generated the following image based on the prompt: '{image_gen_prompt}'. Comment on it.</image_context>"
                                
                                text_gen_parts = [
                                    system_note, 
                                    {"mime_type": "image/png", "url": generated_image_path_for_round}
                                ]
                                supplementary_parts.extend(text_gen_parts)
                            else:
                                if is_generator:
                                    fail_reason = image_gen_error_msg or "Safety Filter / Unknown"
                                    system_note = f"<image_context>Your attempt to generate an image based on the prompt '{image_gen_prompt}' failed due to: {fail_reason}. Comment on this failure in character.</image_context>"
                                    supplementary_parts.append(system_note)

                        if not contents_for_api_call:
                            contents_for_api_call.append({'role': 'user', 'parts': ["<internal_note>Begin conversation.</internal_note>"]})

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
                        
                        if model:
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

                                gen_task = asyncio.create_task(self._generate_with_heartbeat(
                                    model, contents_for_api_call, gen_config, channel, participant, msg_a_id, is_fallback=False, app_name=app_name, app_avatar=app_avatar, existing_state=state_container
                                ))

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
                                main_api_error = _format_api_error(e)
                                if hasattr(e, 'state_container'): state_container = e.state_container
                                
                                if not fallback_model_name or primary_model == fallback_model_name:
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
                                self.cog._log_api_call(user_id=triggering_user_id, guild_id=channel.guild.id, context="multi_profile", model_used=model, status=status)
                        else:
                            api_error_reason = warning_message or "Internal API Initialization Error"

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
                                if fallback_model_name and primary_model != fallback_model_name:
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
                            await self._update_sending_placeholder(channel, participant.get('method', 'webhook'), participant.get('bot_id'), state_container, t1_start_mono)

                        t2_end_mono = time.monotonic()
                        duration = t2_end_mono - t1_start_mono
                        sent_timestamp = datetime.datetime.now(datetime.timezone.utc) # Approximation

                        timezone_str = profile_settings.get("timezone", "UTC")
                        main_history_line = _format_history_entry(speaker_display_name, sent_timestamp, response_text, timezone_str)
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
                        "model": model.model_name.replace("models/", "").replace("OPENROUTER/", "").replace("GOOGLE/", "") if hasattr(model, 'model_name') else fallback_model_name,
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
                    
                    session.setdefault("unified_log", []).append(turn_object)
                    session['last_speaker_key'] = participant_key

                    # [UPDATED] Persist log immediately
                    session_type = session.get("type", "multi")
                    await self.cog.session_manager._save_session_to_disk((channel_id, None, None), session_type, session.get("unified_log", []))

                    is_realistic_typing = profile_settings.get("realistic_typing_enabled", False)

                    if is_realistic_typing:
                        if state_container and state_container.get('sending_task'):
                            state_container['sending_task'].cancel()
                        msg_a_to_delete = state_container.get('msg_a_id') if state_container else msg_a_id
                        msg_b_to_delete = state_container.get('msg_b_id') if state_container else None
                        await self._safe_delete_placeholder(channel, msg_a_to_delete)
                        await self._safe_delete_placeholder(channel, msg_b_to_delete)
                        if state_container:
                            state_container['msg_a_id'] = None
                            state_container['msg_b_id'] = None

                    # [NEW] Unified Synthesis Logic
                    audio_file_for_send = None
                    
                    if profile_settings.get("speech_tts_enabled", False) and session.get("audio_mode", "off") == "on":
                        s_voice = profile_settings.get("speech_voice", "Aoede")
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

                            tts_priming_prompt = "\n\n".join(prompt_parts)
                            
                            # 3. Synthesise Audio
                            turn_audio_stream = await self.cog.media_service._generate_google_tts(
                                tts_priming_prompt, 
                                channel.guild.id, 
                                model_id=s_model, 
                                voice_name=s_voice, 
                                temperature=s_temp
                            )
                            
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

                    if participant.get('method') == 'child_bot':
                        # Stop any pending sending heartbeat task immediately
                        if state_container and state_container.get('sending_task'):
                            state_container['sending_task'].cancel()

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
                            attachment_data = None
                            if is_generator and generated_image_path_for_round:
                                def _read_b64():
                                    with open(generated_image_path_for_round, 'rb') as f:
                                        return base64.b64encode(f.read()).decode('utf-8')
                                b64_data = await asyncio.to_thread(_read_b64)
                                attachment_data = {
                                    "filename": "generated_image.png",
                                    "data_base64": b64_data
                                }
                            elif audio_file_for_send:
                                turn_audio_stream.seek(0)
                                attachment_data = {
                                    "filename": f"voice_{turn_id[:4]}.wav",
                                    "data_base64": base64.b64encode(turn_audio_stream.read()).decode('utf-8')
                                }
                            if attachment_data:
                                payload["attachment"] = attachment_data

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
                            audio_b64 = base64.b64encode(turn_audio_stream.read()).decode('utf-8')
                            a_msgs = await self.cog.child_bot_manager.execute_send(participant['bot_id'], {
                                "channel_id": channel.id, "content": "", "realistic_typing": False,
                                "attachment": {"filename": f"voice_{turn_id[:4]}.wav", "data_base64": audio_b64}
                            })
                            if a_msgs:
                                for sm in a_msgs:
                                    turn_object.setdefault("message_ids", []).append(sm.id)

                        # Dispatch thinking summary directly
                        if thought_file_to_send and thought_text:
                            thought_b64 = base64.b64encode(thought_text.encode('utf-8')).decode('utf-8')
                            t_msgs = await self.cog.child_bot_manager.execute_send(participant['bot_id'], {
                                "channel_id": channel.id, "content": "", "realistic_typing": False,
                                "attachment": {"filename": "thinking_summary.txt", "data_base64": thought_b64}
                            })
                            if t_msgs:
                                for sm in t_msgs:
                                    turn_object.setdefault("message_ids", []).append(sm.id)

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
                            
                            await self.cog.session_manager._save_session_to_disk((channel_id, None, None), session_type, session["unified_log"])
                    
                    # --- Dispatch Warnings and Clean Up Placeholders ---
                    if state_container and state_container.get('sending_task'):
                        state_container['sending_task'].cancel()
                        
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
                                    session.setdefault("unified_log", []).append(new_turn_object)

                            all_triggers_for_round.extend(batched_triggers)
                    
                    # No forced gc.collect() here. It ran once per participant per round and
                    # cost ~11 ms of pure GIL-held CPU on a warm heap (worse on the e2-micro),
                    # stalling Discord heartbeats; asyncio.to_thread does not avoid that, since
                    # a collection holds the GIL regardless of which thread calls it. Image
                    # buffers are freed by refcounting the moment the last reference drops, so
                    # the pass only ever reclaimed reference cycles the automatic collector
                    # would have caught anyway.

                await self.cog.session_manager._save_session_to_disk((channel_id, None, None), session_type, session.get("unified_log", []))

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
                generated_image_bytes_for_round = None

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
                        context_size = p_settings.get("ltm_summarization_context", 10)
              
                        if participant['ltm_counter'] >= interval:
                            # Derived per participant from unified_log. This previously read
                            # next(iter(session['chat_sessions'].values())) — whichever participant
                            # happened to be first in dict order — so in a multi-profile session
                            # every profile's long-term memory was written from another profile's
                            # view of the conversation, private turns included.
                            #
                            # The STM floor keeps the summarisation window governed by
                            # ltm_summarization_context rather than by this profile's STM length,
                            # which is what the old unbounded shadow history effectively did.
                            ltm_p_settings = dict(p_settings)
                            ltm_p_settings["stm_length"] = max(
                                int(p_settings.get("stm_length", defaultConfig.CHATBOT_MEMORY_LENGTH)),
                                context_size * 2,
                            )
                            ltm_bot_pid = self.cog.profile_manager._get_pid_from_name_any(owner_id, profile_name)
                            ltm_history = self.cog.session_manager._build_history_for_participant(
                                session.get("unified_log", []), ltm_bot_pid, ltm_p_settings, len(profile_order) or 1
                            )
                            if len(ltm_history) >= 2:
                                # Turn history is consolidated
                                events_for_summary = []
                                for turn in ltm_history[-context_size:]:
                                    parts = turn.get('parts', [])
                                    if parts:
                                        text_val = "\n".join(p if isinstance(p, str) else p.get('text', '') for p in parts)
                                        events_for_summary.append(text_val)
                                
                                # [FIX] Offload LTM generation to a background task so it doesn't block the queue
                                async def background_ltm_gen(o_id, p_name, evts, r_author, g_id, t_user_id):
                                    try:
                                        _, _, _, temp, top_p, top_k, primary_model, _ = await asyncio.to_thread(
                                            self._construct_system_instructions, o_id, p_name, channel_id, is_multi_profile=True
                                        )
                                        ltm_d = await self.cog.memory_manager._generate_ltm_data_from_history(evts, r_author, {"temperature": temp, "top_p": top_p, "top_k": top_k}, primary_model, g_id, profile_owner_id=o_id, profile_name=p_name)
                                        if ltm_d:
                                            summary_embedding = await self.cog.memory_manager._get_embedding(ltm_d, g_id, task_type="RETRIEVAL_DOCUMENT")
                                            if summary_embedding:
                                                b64_emb = encode_embedding_b64(summary_embedding)
                                                self.cog.memory_manager._add_ltm(o_id, p_name, ltm_d, b64_emb, g_id, t_user_id, r_author)
                                                
                                                # Link LTM creation to the turn metadata for trace transparency
                                                bot_pid = self.cog.profile_manager._get_pid_from_name_any(o_id, p_name)
                                                last_turn = next((t for t in reversed(session.get("unified_log", [])) if t.get("speaker_pid") == bot_pid), None)
                                                if last_turn and "meta" in last_turn:
                                                    last_turn["meta"]["ltm_created"] = True
                                    except Exception as e:
                                        print(f"Background LTM generation failed for {p_name}: {e}")

                                asyncio.create_task(background_ltm_gen(owner_id, profile_name, events_for_summary, round_author_name, guild_id, triggering_user_id))
                            
                            participant['ltm_counter'] = 0

                # AGGRESSIVE GC: Clear references
                if 'new_round_content_objects' in locals(): del new_round_content_objects
                if 'all_triggers_for_round' in locals(): del all_triggers_for_round

                # Trim the unified log to be the single source of truth for the session's history window.
                if len(session.get("unified_log", [])) > 1000:
                    session["unified_log"] = session["unified_log"][-1000:]

                # [NEW] Mandatory Round-End Persistence
                # Ensures the transcript is saved immediately after the last participant speaks.
                dummy_session_key = (channel_id, None, None)
                await self.cog.session_manager._save_session_to_disk(dummy_session_key, session_type, session.get("unified_log", []))

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
        
        # [NEW] Lifecycle protection: Remove from background set and clear reference
        ctask = asyncio.current_task()
        self.cog.background_tasks.discard(ctask)
        if session.get('worker_task') == ctask:
            session['worker_task'] = None

