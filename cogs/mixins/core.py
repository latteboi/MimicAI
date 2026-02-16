from .gui import *
from .services import *

import discord
from discord.ext import commands, tasks
import asyncio
import datetime
import uuid
import time
import random
import pathlib
import os
import io
import re
import traceback
import orjson as json
from zoneinfo import ZoneInfo, available_timezones
import google.generativeai as genai
from google.genai import types as google_genai_types
from google.api_core import exceptions as api_exceptions
from typing import List, Dict, Tuple, Set, Literal, Any, Optional, Union, get_args
from .constants import *
from .storage import _delete_file_shard, _atomic_json_save, _quantize_embedding, _dequantize_embedding

class CoreMixin:

    @commands.Cog.listener()
    async def on_ready(self):
        if not self.sessions_loaded:
            await self._load_multi_profile_sessions()
            self.sessions_loaded = True
        
        self.all_bot_ids = {self.bot.user.id} | {int(bot_id) for bot_id in self.child_bots.keys()}

        if self.has_lock:
            if not self.freewill_task.is_running():
                self.freewill_task.start()
            
            if not self.weekly_cleanup_task.is_running():
                print("Performing initial data cleanup on boot...")
                await self._perform_data_cleanup()
                print("Initial cleanup finished. Starting weekly cleanup task.")
                self.weekly_cleanup_task.start()

        if self.has_lock and not self.image_finisher_worker_task:
            self.image_finisher_worker_task = self.bot.loop.create_task(self._image_finisher_worker())
            # Start 5 pre-fetching workers
            for i in range(5):
                worker = self.bot.loop.create_task(self._image_gen_worker(i))
                self.image_gen_workers.append(worker)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.id in self.processed_child_messages:
            return

        if message.author.id in self.global_blacklist:
            return
        
        if not message.guild or not self.has_lock or message.author.bot:
            return

        # --- 1. Session Reply Logic (Priority) ---
        if message.reference and message.reference.message_id:
            if message.reference.resolved and isinstance(message.reference.resolved, discord.Message) and message.reference.resolved.author.id in self.all_bot_ids:
                if message.reference.resolved.author.id != self.bot.user.id and message.channel.id not in self.multi_profile_channels:
                    return

            turn_info = self.message_to_history_turn.get(message.reference.message_id)
            if not turn_info:
                session_type = self.multi_profile_channels.get(message.channel.id, {}).get("type", "multi")
                mapping_key = (session_type, message.channel.id)
                if mapping_key not in self.mapping_caches:
                    self.mapping_caches[mapping_key] = self._load_mapping_from_disk(mapping_key)
                turn_info = self.mapping_caches[mapping_key].get(str(message.reference.message_id))

            if turn_info and isinstance(turn_info, (list, tuple)) and len(turn_info) == 3:
                channel_id, session_type, turn_id_to_find = turn_info
                session = self._ensure_session_hydrated(channel_id, session_type)
                if session:
                    turn_object = next((turn for turn in session.get("unified_log", []) if turn.get("turn_id") == turn_id_to_find), None)
                    if turn_object:
                        speaker_key = tuple(turn_object.get("speaker_key", []))
                        replied_to_participant = next((p for p in session['profiles'] if (p['owner_id'], p['profile_name']) == speaker_key), None)
                        if replied_to_participant:
                            eager_placeholder = None
                            if replied_to_participant.get('method') == 'child_bot':
                                asyncio.create_task(self.manager_queue.put({
                                    "action": "send_to_child", "bot_id": replied_to_participant['bot_id'],
                                    "payload": {"action": "start_typing", "channel_id": message.channel.id}
                                }))
                            elif session.get("is_hydrated"):
                                placeholders = await self._send_channel_message(
                                    message.channel, f"{PLACEHOLDER_EMOJI}",
                                    profile_owner_id_for_appearance=replied_to_participant['owner_id'],
                                    profile_name_for_appearance=replied_to_participant['profile_name'],
                                    reply_to=message
                                )
                                if placeholders: eager_placeholder = placeholders[0]
                            else:
                                await message.channel.typing()

                            reply_trigger = ('reply', message, replied_to_participant, eager_placeholder)
                            await session['task_queue'].put(reply_trigger)
                            
                            if not session.get('is_running'):
                                # [NEW] Safeguard: If a stale task object exists, clear it before creating new
                                if session.get('worker_task'):
                                    self._safe_cancel_task(session['worker_task'])
                                session['worker_task'] = self.bot.loop.create_task(self._multi_profile_worker(channel_id))
                            return

        # --- 2. Standalone Child Bot Detection ---
        mentioned_child_ids = []
        for user in message.mentions:
            if str(user.id) in self.child_bots:
                mentioned_child_ids.append(str(user.id))
        
        ref_msg = None
        if message.reference:
            if message.reference.resolved and isinstance(message.reference.resolved, discord.Message):
                ref_msg = message.reference.resolved
            else:
                try: ref_msg = await message.channel.fetch_message(message.reference.message_id)
                except: pass

        if ref_msg:
            ref_author_id = str(ref_msg.author.id)
            if ref_author_id in self.child_bots and ref_author_id not in mentioned_child_ids:
                mentioned_child_ids.append(ref_author_id)

        if mentioned_child_ids:
            if not self._get_api_key_for_guild(message.guild.id):
                for bot_id in mentioned_child_ids:
                    asyncio.create_task(self.manager_queue.put({
                        "action": "send_to_child", "bot_id": bot_id,
                        "payload": {
                            "action": "send_message", "channel_id": message.channel.id,
                            "content": "An API key has not been configured for this server. You can use the `/settings` command in the parent bot's DM to set one."
                        }
                    }))
                return

            for bot_id in mentioned_child_ids:
                asyncio.create_task(self.manager_queue.put({
                    "action": "send_to_child", "bot_id": bot_id,
                    "payload": {"action": "start_typing", "channel_id": message.channel.id}
                }))

            content_lower = message.content.lower()
            image_prefixes = ("!image", "!imagine")
            is_image_request = content_lower.startswith(image_prefixes)

            attachments_data = [{"url": a.url, "content_type": a.content_type} for a in message.attachments if a.content_type and (a.content_type.startswith("image/") or a.content_type.startswith("audio/") or a.content_type.startswith("video/"))]
            
            reply_data = None
            if ref_msg:
                ref_attach_url = ref_msg.attachments[0].url if ref_msg.attachments and ref_msg.attachments[0].content_type.startswith("image/") else None
                reply_data = {
                    "id": ref_msg.id, "channel_id": ref_msg.channel.id,
                    "attachment_url": ref_attach_url, "author_name": ref_msg.author.display_name
                }

            payload = {
                "id": message.id,
                "content": message.content.replace(f"<@{self.bot.user.id}>", "").strip(),
                "channel_id": message.channel.id, "guild_id": message.guild.id,
                "author_id": message.author.id, "author_name": message.author.display_name,
                "timestamp": message.created_at.isoformat(), "attachments": attachments_data,
                "replied_to": reply_data
            }

            for bot_id in mentioned_child_ids:
                event_data = {"bot_id": bot_id, "message": payload}
                if message.channel.id in self.multi_profile_channels:
                    event_data["event_type"] = "message_received"
                    await self.handle_child_bot_event(event_data)
                else:
                    if is_image_request: await self.handle_child_bot_image_request(event_data)
                    else:
                        event_data["event_type"] = "message_received"
                        await self.handle_child_bot_event(event_data)
            return

        # --- 3. Normal Session Triggering (Main Bot Mention / React / Freewill) ---
        is_main_bot_mention = self.bot.user and self.bot.user.mentioned_in(message)
        
        # Resolve Active Profile Settings for Gating
        eff_uid = message.author.id
        eff_pname = self._get_active_user_profile_name_for_channel(eff_uid, message.channel.id)
        eff_udata = self._get_user_data_entry(eff_uid)
        eff_is_b = eff_pname in eff_udata.get("borrowed_profiles", {})
        eff_profile = eff_udata.get("borrowed_profiles" if eff_is_b else "profiles", {}).get(eff_pname, {})

        if is_main_bot_mention and not session:
            content_to_process = message.clean_content.replace(f"<@{self.bot.user.id}>", "").strip()
            
            # Stateless Image Gen Gate
            content_lower = content_to_process.lower()
            image_prefixes = ("!image", "!imagine")
            if any(content_lower.startswith(p) for p in image_prefixes):
                if not eff_profile.get("image_generation_enabled", True):
                    # Proceed to session creation and treat it as text later
                    pass
                else:
                    await self._handle_image_generation_request(message, content_to_process)
                    return

            # Note: URL context is researched by the worker in the "Research Once" phase.
            # No prepending needed here.
            
            owner_id = int(defaultConfig.DISCORD_OWNER_ID)
            
            # Re-order/Logic for new session...
        if message.reference and message.reference.message_id:
            if message.reference.resolved and message.reference.resolved.author.id in self.all_bot_ids:
                if message.reference.resolved.author.id != self.bot.user.id and message.channel.id not in self.multi_profile_channels:
                    return

            turn_info = self.message_to_history_turn.get(message.reference.message_id)
            if not turn_info:
                session_type = self.multi_profile_channels.get(message.channel.id, {}).get("type", "multi")
                mapping_key = (session_type, message.channel.id)
                if mapping_key not in self.mapping_caches:
                    self.mapping_caches[mapping_key] = self._load_mapping_from_disk(mapping_key)
                turn_info = self.mapping_caches[mapping_key].get(str(message.reference.message_id))

            if turn_info and isinstance(turn_info, (list, tuple)) and len(turn_info) == 3:
                channel_id, session_type, turn_id_to_find = turn_info
                session = self._ensure_session_hydrated(channel_id, session_type)
                if session:
                    turn_object = next((turn for turn in session.get("unified_log", []) if turn.get("turn_id") == turn_id_to_find), None)
                    if turn_object:
                        speaker_key = tuple(turn_object.get("speaker_key", []))
                        replied_to_participant = next((p for p in session['profiles'] if (p['owner_id'], p['profile_name']) == speaker_key), None)
                        if replied_to_participant:
                            eager_placeholder = None
                            # [NEW] Eager Feedback
                            if replied_to_participant.get('method') == 'child_bot':
                                asyncio.create_task(self.manager_queue.put({
                                    "action": "send_to_child", "bot_id": replied_to_participant['bot_id'],
                                    "payload": {"action": "start_typing", "channel_id": message.channel.id}
                                }))
                            elif session.get("is_hydrated"):
                                # If session is hot, we can fetch appearance and send placeholder immediately
                                placeholders = await self._send_channel_message(
                                    message.channel, f"{PLACEHOLDER_EMOJI}",
                                    profile_owner_id_for_appearance=replied_to_participant['owner_id'],
                                    profile_name_for_appearance=replied_to_participant['profile_name'],
                                    reply_to=message
                                )
                                if placeholders: eager_placeholder = placeholders[0]
                            else:
                                # If cold, just show generic typing to acknowledge receipt
                                await message.channel.typing()

                            # Pass placeholder in the tuple (4 elements)
                            reply_trigger = ('reply', message, replied_to_participant, eager_placeholder)
                            await session['task_queue'].put(reply_trigger)
                            
                            if not session.get('is_running'):
                                # [NEW] Safeguard: If a stale task object exists, clear it before creating new
                                if session.get('worker_task'):
                                    self._safe_cancel_task(session['worker_task'])
                                session['worker_task'] = self.bot.loop.create_task(self._multi_profile_worker(channel_id))
                            return

        # --- Freewill Trigger Logic ---
        guild_id_str = str(message.guild.id)
        fw_config = self.freewill_config.get(guild_id_str, {})
        triggered_profile = None

        # Ensure we don't interrupt a manual multi-profile session with freewill logic
        session = self.multi_profile_channels.get(message.channel.id)
        is_freewill_compatible = not session or session.get("type") == "freewill"

        if fw_config.get("enabled", False) and is_freewill_compatible:
            is_living = message.channel.id in fw_config.get("living_channel_ids", [])
            is_lurking = message.channel.id in fw_config.get("lurking_channel_ids", [])
            
            if is_living or is_lurking:
                content_lower = message.content.lower()
                image_prefixes = ("!image", "!imagine")
                is_image_gen_trigger = content_lower.startswith(image_prefixes)
                
                server_participation = self.freewill_participation.get(str(message.guild.id), {})
                channel_participants = server_participation.get(str(message.channel.id), {})
                
                valid_participants_wakeword = []
                valid_participants_image = []
                valid_participants_rng = []

                for user_id_str, profiles in channel_participants.items():
                    member = message.guild.get_member(int(user_id_str))
                    if not member: continue

                    for profile_name, settings in profiles.items():
                        # Wakeword check
                        if any(w.lower() in content_lower for w in settings.get("wakewords", [])):
                            p_dict = self._build_freewill_participant_dict(int(user_id_str), profile_name, message.channel)
                            if p_dict: valid_participants_wakeword.append(p_dict)
                        
                        # Image Gen capability check
                        if settings.get("personality", "off") != "off":
                            if is_image_gen_trigger:
                                p_dict = self._build_freewill_participant_dict(int(user_id_str), profile_name, message.channel)
                                if p_dict: valid_participants_image.append(p_dict)
                            
                            # RNG check preparation
                            personality = settings.get("personality", "off")
                            valid_participants_rng.append((personality, int(user_id_str), profile_name))

                # Priority: Wakeword > Image Gen > RNG
                if valid_participants_wakeword:
                    triggered_profile = random.choice(valid_participants_wakeword)
                elif is_image_gen_trigger and valid_participants_image:
                    triggered_profile = random.choice(valid_participants_image)
                elif not is_image_gen_trigger and valid_participants_rng:
                    personality_chances = {"introverted": 0.03, "regular": 0.10, "outgoing": 0.30}
                    possible_rolls = []
                    for personality, uid, pname in valid_participants_rng:
                        if personality in personality_chances and random.random() <= personality_chances[personality]:
                            p_dict = self._build_freewill_participant_dict(uid, pname, message.channel)
                            if p_dict: possible_rolls.append(p_dict)
                    if possible_rolls:
                        triggered_profile = random.choice(possible_rolls)

        if triggered_profile:
            # Ensure session exists or is created
            if not session:
                all_opted_in = [] # Populate with all potential participants for context
                server_participation = self.freewill_participation.get(str(message.guild.id), {})
                channel_participants = server_participation.get(str(message.channel.id), {})
                
                for user_id_str, profiles in channel_participants.items():
                    member = message.guild.get_member(int(user_id_str))
                    if member:
                        for profile_name, settings in profiles.items():
                            if settings.get("personality", "off") != "off":
                                p_dict = self._build_freewill_participant_dict(int(user_id_str), profile_name, message.channel)
                                if p_dict: all_opted_in.append(p_dict)
                
                chat_sessions = { (p['owner_id'], p['profile_name']): None for p in all_opted_in }
                session = {
                    "type": "freewill", "freewill_mode": "reactive", "chat_sessions": chat_sessions,
                    "initial_channel_history": await self._build_freewill_history(message.channel, message),
                    "initial_turn_taken": set(), "last_bot_message_id": None, "owner_id": message.author.id,
                    "is_running": False, "task_queue": asyncio.Queue(), "worker_task": None,
                    "turns_since_last_ltm": 0, "session_prompt": None, "profiles": all_opted_in,
                    "is_hydrated": False 
                }
                self.multi_profile_channels[message.channel.id] = session
                self._save_multi_profile_sessions()

            # Important: Ensure the triggered profile is actually in the session profiles list (handling drift)
            p_key_check = (triggered_profile['owner_id'], triggered_profile['profile_name'])
            if not any((p['owner_id'], p['profile_name']) == p_key_check for p in session['profiles']):
                session['profiles'].append(triggered_profile)
                session['chat_sessions'][p_key_check] = None

            trigger_tuple = ('initial_reactive_turn', message, triggered_profile)
            await session['task_queue'].put(trigger_tuple)
            
            if not session.get('is_running'):
                session['worker_task'] = self.bot.loop.create_task(self._multi_profile_worker(message.channel.id))
            return

        # --- Generic Session Interaction Logic ---
        is_main_bot_mention = self.bot.user and self.bot.user.mentioned_in(message)

        if session:
            # If Freewill session is active, block mentions of the parent bot
            if session.get("type") == "freewill":
                if is_main_bot_mention:
                    return
            
            # If Regular Multi session is active and parent bot is mentioned
            if is_main_bot_mention and session.get("type") == "multi":
                owner_id = int(defaultConfig.DISCORD_OWNER_ID)
                profile_name = DEFAULT_PROFILE_NAME
                
                # Check if parent bot is already a participant
                is_participant = False
                for p in session['profiles']:
                    if p['owner_id'] == owner_id and p['profile_name'] == profile_name:
                        is_participant = True
                        break
                
                if not is_participant:
                    # Ad-hoc injection
                    participant = {
                        "owner_id": owner_id,
                        "profile_name": profile_name,
                        "method": "webhook",
                        "ephemeral": True
                    }
                    trigger = ('ad_hoc_mention', message, participant)
                    await session['task_queue'].put(trigger)
                    if not session.get('is_running'):
                        session['worker_task'] = self.bot.loop.create_task(self._multi_profile_worker(message.channel.id))
                    return

            await session['task_queue'].put(message)
            if not session.get('is_running'):
                session['worker_task'] = self.bot.loop.create_task(self._multi_profile_worker(message.channel.id))
            return
        
        # Check for Freewill configuration to prevent overriding with a new manual session
        fw_config = self.freewill_config.get(guild_id_str, {})
        if message.channel.id in fw_config.get("living_channel_ids", []) or message.channel.id in fw_config.get("lurking_channel_ids", []):
            return

        # --- New Session Creation Logic (Main Bot Mention) ---
        if is_main_bot_mention:
            owner_id = int(defaultConfig.DISCORD_OWNER_ID)
            profile_name = DEFAULT_PROFILE_NAME
            
            participant = {
                "owner_id": owner_id,
                "profile_name": profile_name,
                "method": "webhook",
                "ephemeral": True # Changed to True for ad-hoc behavior
            }
            
            chat_sessions = {(owner_id, profile_name): None}
            new_session = {
                "type": "multi", "profiles": [participant], "chat_sessions": chat_sessions,
                "unified_log": [], "is_hydrated": False, "last_bot_message_id": None,
                "owner_id": message.author.id, "is_running": False, "auto_continue": False,
                "auto_delay": None, "timer_handle": None, "task_queue": asyncio.Queue(),
                "worker_task": None, "turns_since_last_ltm": 0, "session_prompt": None,
                "session_mode": "sequential"
            }
            self.multi_profile_channels[message.channel.id] = new_session
            self._save_multi_profile_sessions()
            
            await new_session['task_queue'].put(message)
            if not new_session.get('is_running'):
                new_session['worker_task'] = self.bot.loop.create_task(self._multi_profile_worker(message.channel.id))
            return
        
    @commands.Cog.listener()
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        if payload.user_id in self.global_blacklist:
            return
        if not self.has_lock or payload.user_id == self.bot.user.id:
            return

        emoji_str = str(payload.emoji)
        is_regen = (emoji_str == REGENERATE_EMOJI)
        is_next = (emoji_str == NEXT_SPEAKER_EMOJI)
        is_continue = (emoji_str == CONTINUE_ROUND_EMOJI)
        is_mute = (emoji_str in MUTE_TURN_EMOJI)
        is_skip = (emoji_str in SKIP_PARTICIPANT_EMOJI)
        
        if not any([is_regen, is_next, is_continue, is_mute, is_skip]):
            return

        turn_info = self.message_to_history_turn.get(payload.message_id)
        
        # Cold path: if not in hot cache, check on-disk mapping
        if not turn_info:
            # We must guess the session type to find the right mapping file.
            # We check 'multi' first, then 'freewill' as a fallback.
            potential_mapping_keys = [
                ('multi', payload.channel_id),
                ('freewill', payload.channel_id)
            ]
            for key in potential_mapping_keys:
                if key not in self.mapping_caches:
                    self.mapping_caches[key] = self._load_mapping_from_disk(key)
                turn_info = self.mapping_caches[key].get(str(payload.message_id))
                if turn_info:
                    break
        
        if not turn_info or not isinstance(turn_info, (list, tuple)) or len(turn_info) != 3:
            return # Not a message we are tracking or malformed data

        channel_id, session_type, turn_id_to_find = turn_info
        if session_type not in ['multi', 'freewill']:
            return

        # [NEW] Re-hydrate/Reset Timer
        session = self.multi_profile_channels.get(channel_id)
        if not session or not session.get("is_hydrated"):
            session = self._ensure_session_hydrated(channel_id, session_type)
        
        if not session:
            return
            
        # [NEW] Concurrency Safeguard
        if session.get('is_running') or session.get('is_regenerating'):
            return

        self.session_last_accessed[channel_id] = time.time()

        # Find the turn in the unified log
        turn_index = -1
        turn_object = None
        for i, turn in enumerate(session.get("unified_log", [])):
            if turn.get("turn_id") == turn_id_to_find:
                turn_index = i
                turn_object = turn
                break

        if turn_object:
            speaker_key = tuple(turn_object.get("speaker_key", []))
            
            reacted_to_participant = None
            for participant in session['profiles']:
                p_key = (participant['owner_id'], participant['profile_name'])
                if p_key == speaker_key:
                    reacted_to_participant = participant
                    break
            
            if reacted_to_participant:
                if is_regen:
                    # [NEW] Immediate reaction cleanup
                    try:
                        channel = self.bot.get_channel(payload.channel_id)
                        if channel:
                            msg = await channel.fetch_message(payload.message_id)
                            # Attempt to remove the user's reaction
                            await msg.remove_reaction(payload.emoji, discord.Object(id=payload.user_id))
                    except: pass

                    asyncio.create_task(self._execute_regeneration(payload, session, turn_object, turn_index, reacted_to_participant))
                    return
                
                if is_mute:
                    turn_object["is_hidden"] = True
                    self._save_session_to_disk((channel_id, None, None), session_type, session["unified_log"])
                    session["is_hydrated"] = False
                    self._ensure_session_hydrated(channel_id, session_type)
                    return
                
                if is_skip:
                    reacted_to_participant["is_skipped"] = True
                    self._save_multi_profile_sessions()
                    return

                try:
                    next_participant = None
                    last_speaker_key = session.get('last_speaker_key')
                    reacted_to_key = (reacted_to_participant['owner_id'], reacted_to_participant['profile_name'])

                    if is_continue:
                        # Original logic: popcorn continues the round
                        if reacted_to_key == last_speaker_key:
                            session_mode = session.get("session_mode", "sequential")
                            if session_mode == 'sequential':
                                try:
                                    last_speaker_index = next(i for i, p in enumerate(session['profiles']) if (p['owner_id'], p['profile_name']) == last_speaker_key)
                                    next_speaker_index = (last_speaker_index + 1) % len(session['profiles'])
                                    next_participant = session['profiles'][next_speaker_index]
                                except (ValueError, StopIteration):
                                    if session['profiles']:
                                        next_participant = session['profiles'][0]
                            else: # Random mode
                                potential_responders = [p for p in session['profiles'] if (p['owner_id'], p['profile_name']) != last_speaker_key]
                                if potential_responders:
                                    next_participant = random.choice(potential_responders)
                        else:
                            reacted_to_index = session['profiles'].index(reacted_to_participant)
                            next_speaker_index = (reacted_to_index + 1) % len(session['profiles'])
                            next_participant = session['profiles'][next_speaker_index]
                    
                    elif is_next:
                        if reacted_to_key == last_speaker_key:
                            try:
                                last_speaker_index = next(i for i, p in enumerate(session['profiles']) if (p['owner_id'], p['profile_name']) == last_speaker_key)
                                next_speaker_index = (last_speaker_index + 1) % len(session['profiles'])
                                next_participant = session['profiles'][next_speaker_index]
                            except: pass
                        else:
                            reacted_to_index = session['profiles'].index(reacted_to_participant)
                            next_speaker_index = (reacted_to_index + 1) % len(session['profiles'])
                            next_participant = session['profiles'][next_speaker_index]

                    if next_participant:
                        session_mode = session.get("session_mode", "sequential")
                        if session_mode == 'sequential':
                            try:
                                start_idx = session['profiles'].index(next_participant)
                                new_order = session['profiles'][start_idx:] + session['profiles'][:start_idx]
                                session['profiles'] = new_order
                                self._save_multi_profile_sessions()
                            except ValueError:
                                pass

                        trigger_type = 'reaction_single' if is_next else 'reaction'
                        reaction_trigger = (trigger_type, payload, next_participant)
                        await session['task_queue'].put(reaction_trigger)
                        if not session.get('is_running'):
                            session['worker_task'] = self.bot.loop.create_task(self._multi_profile_worker(payload.channel_id))
                except (ValueError, IndexError):
                    pass

    @commands.Cog.listener()
    async def on_raw_reaction_remove(self, payload: discord.RawReactionActionEvent):
        if not self.has_lock: return
        
        emoji_str = str(payload.emoji)
        is_mute = (emoji_str in MUTE_TURN_EMOJI)
        is_skip = (emoji_str in SKIP_PARTICIPANT_EMOJI)
        
        if not is_mute and not is_skip: return

        turn_info = self.message_to_history_turn.get(payload.message_id)
        if not turn_info: return
        
        channel_id, session_type, turn_id_to_find = turn_info
        session = self.multi_profile_channels.get(channel_id)
        if not session or not session.get("is_hydrated"):
            session = self._ensure_session_hydrated(channel_id, session_type)
        
        if not session or session.get('is_running') or session.get('is_regenerating'): return

        turn_object = next((turn for turn in session.get("unified_log", []) if turn.get("turn_id") == turn_id_to_find), None)
        if turn_object:
            if is_mute:
                turn_object["is_hidden"] = False
                self._save_session_to_disk((channel_id, None, None), session_type, session["unified_log"])
                session["is_hydrated"] = False
                self._ensure_session_hydrated(channel_id, session_type)
            
            elif is_skip:
                speaker_key = tuple(turn_object.get("speaker_key", []))
                participant = next((p for p in session['profiles'] if (p['owner_id'], p['profile_name']) == speaker_key), None)
                if participant:
                    participant["is_skipped"] = False
                    self._save_multi_profile_sessions()
            
    @commands.Cog.listener()
    async def on_raw_message_delete(self, payload: discord.RawMessageDeleteEvent):
        if not self.has_lock:
            return

        deleted_message_id = payload.message_id
        if deleted_message_id in self.purged_message_ids:
            self.purged_message_ids.discard(deleted_message_id)
            return
        
        turn_info = self.message_to_history_turn.pop(deleted_message_id, None)
        
        if not turn_info:
            # Cold path: find the mapping on disk
            session_type = self.multi_profile_channels.get(payload.channel_id, {}).get("type", "multi")
            mapping_key = (session_type, payload.channel_id)
            if mapping_key not in self.mapping_caches:
                self.mapping_caches[mapping_key] = self._load_mapping_from_disk(mapping_key)
            turn_info = self.mapping_caches[mapping_key].get(str(deleted_message_id))

        if not turn_info:
            return

        # If turn_info came from JSON, its inner tuple (the session key) will be a list. Convert it back.
        if isinstance(turn_info[0], list):
            turn_info[0] = tuple(turn_info[0])

        session_key_or_channel_id, session_type, turn_data = turn_info
        
        if session_type == 'global_chat':
            session_key = session_key_or_channel_id
            turn_id_to_delete = turn_data
            
            # [NEW] Decrement Counter for Single/Global
            owner_id, profile_name = session_key[1], session_key[2]
            ltm_counter_key = (owner_id, profile_name, "guild")
            if ltm_counter_key in self.message_counters_for_ltm:
                self.message_counters_for_ltm[ltm_counter_key] = max(0, self.message_counters_for_ltm[ltm_counter_key] - 1)

            mapping_key = self._get_mapping_key_for_session(session_key, session_type)
            if mapping_key in self.mapping_caches:
                self.mapping_caches[mapping_key].pop('grounding_checkpoint', None)
                mapping_data = self.mapping_caches[mapping_key]
                keys_to_delete = [
                    msg_id for msg_id, t_info in mapping_data.items()
                    if isinstance(t_info, (list, tuple)) and len(t_info) > 2 and t_info[2] == turn_id_to_delete
                ]
                for msg_id in keys_to_delete:
                    mapping_data.pop(msg_id, None)

            session_data = self.global_chat_sessions.get(session_key)
            if not session_data:
                dummy_model = genai.GenerativeModel('gemini-flash-latest')
                session_data = self._load_session_from_disk(session_key, session_type, dummy_model)
                if session_data: self.global_chat_sessions[session_key] = session_data
            
            if session_data:
                original_len = len(session_data['unified_log'])
                session_data['unified_log'] = [t for t in session_data['unified_log'] if t.get('turn_id') != turn_id_to_delete]

                if len(session_data['unified_log']) < original_len:
                    # Rebuild history
                    from google.generativeai.types import content_types
                    dummy_model = genai.GenerativeModel('gemini-flash-latest')
                    new_history = []
                    for t in session_data['unified_log']:
                        role = 'model' if t.get('role') == 'model' else 'user'
                        new_history.append(content_types.to_content({'role': role, 'parts': [t.get('content')]}))
                    session_data['chat_session'] = dummy_model.start_chat(history=new_history)

                if not session_data['unified_log']:
                    self.global_chat_sessions.pop(session_key, None)
                    self.session_last_accessed.pop(session_key, None)
                    self._delete_session_from_disk(session_key, session_type)
                    if mapping_key in self.mapping_caches:
                        self.mapping_caches.pop(mapping_key, None)
                        self._save_mapping_to_disk(mapping_key, {})
                    self.ltm_recall_history.pop(session_key, None)
                else:
                    self.session_last_accessed[session_key] = time.time()
        
        elif session_type in ['multi', 'freewill']:
            channel_id = session_key_or_channel_id
            turn_id_to_delete = turn_data
            
            mapping_key = (session_type, channel_id)
            if mapping_key in self.mapping_caches:
                self.mapping_caches[mapping_key].pop('grounding_checkpoint', None)
                mapping_data = self.mapping_caches[mapping_key]
                
                keys_to_delete = [
                    msg_id for msg_id, t_info in mapping_data.items()
                    if isinstance(t_info, (list, tuple)) and len(t_info) > 2 and t_info[2] == turn_id_to_delete
                ]
                for msg_id in keys_to_delete:
                    mapping_data.pop(msg_id, None)

            session = self.multi_profile_channels.get(channel_id)
            if session:
                if not session.get("is_hydrated"):
                    session = self._ensure_session_hydrated(channel_id, session_type)
                
                # [NEW] Decrement Counter for Multi/Freewill
                turn_object = next((turn for turn in session.get("unified_log", []) if turn.get("turn_id") == turn_id_to_delete), None)
                if turn_object:
                    speaker_key = tuple(turn_object.get("speaker_key", []))
                    for p in session.get('profiles', []):
                        if (p['owner_id'], p['profile_name']) == speaker_key:
                            p['ltm_counter'] = max(0, p.get('ltm_counter', 0) - 1)
                            break

                original_log_len = len(session.get("unified_log", []))
                session["unified_log"] = [
                    turn for turn in session.get("unified_log", [])
                    if turn.get("turn_id") != turn_id_to_delete
                ]
                
                if len(session["unified_log"]) < original_log_len:
                    from google.generativeai.types import content_types
                    dummy_model = genai.GenerativeModel('gemini-flash-latest')
                    for p_data in session["profiles"]:
                        p_key = (p_data['owner_id'], p_data['profile_name'])
                        participant_history = []
                        for turn in session["unified_log"]:
                            speaker_key = tuple(turn.get("speaker_key", []))
                            role = 'model' if speaker_key == p_key else 'user'
                            content_obj = content_types.to_content({'role': role, 'parts': [turn.get("content")]})
                            participant_history.append(content_obj)
                        session["chat_sessions"][p_key] = dummy_model.start_chat(history=participant_history)

                is_effectively_empty = not session.get("unified_log") or all(
                    turn.get("type") in ["whisper", "private_response"] for turn in session.get("unified_log", [])
                )
                if is_effectively_empty:
                    # Clean up disk artifacts but KEEP session in memory to preserve participants
                    dummy_session_key = (channel_id, None, None)
                    self._delete_session_from_disk(dummy_session_key, session_type)
                    
                    mapping_key = (session_type, channel_id)
                    path = self._get_mapping_path(mapping_key)
                    _delete_file_shard(str(path))
                    
                    # Clear LTM cooldowns since context is gone
                    for p_key in session.get("chat_sessions", {}).keys():
                        owner_id, profile_name = p_key
                        full_session_key = (channel_id, owner_id, profile_name)
                        self.ltm_recall_history.pop(full_session_key, None)

                self.session_last_accessed[channel_id] = time.time()

    async def _execute_speak_as(self, interaction_to_respond: discord.Interaction, channel: discord.abc.Messageable, author: discord.User, profile_name: str, message: str, method: str):
        user_id = author.id
        user_data = self._get_user_data_entry(user_id)
        
        is_borrowed = profile_name in user_data.get("borrowed_profiles", {})
        is_personal = profile_name in user_data.get("profiles", {})

        if not is_borrowed and not is_personal:
            await interaction_to_respond.followup.send(f"You do not have a profile named '{profile_name}'.", ephemeral=True)
            return

        if isinstance(channel, (discord.TextChannel, discord.Thread)):
            if not channel.permissions_for(channel.guild.me).send_messages:
                await interaction_to_respond.followup.send("I do not have permission to send messages in that channel.", ephemeral=True)
                return

        effective_owner_id = user_id
        effective_profile_name = profile_name
        profile_data_source = {}

        if is_borrowed:
            borrowed_data = user_data["borrowed_profiles"][profile_name]
            effective_owner_id = int(borrowed_data["original_owner_id"])
            effective_profile_name = borrowed_data["original_profile_name"]
            owner_user_data = self._get_user_data_entry(effective_owner_id)
            profile_data_source = owner_user_data.get("profiles", {}).get(effective_profile_name, {})
        else:
            profile_data_source = user_data["profiles"][profile_name]

        if not self._check_unrestricted_safety_policy(effective_owner_id, effective_profile_name, channel):
            await interaction_to_respond.followup.send("This profile has an 'Unrestricted 18+' safety level and cannot speak in this channel because it is not marked as Age-Restricted.", ephemeral=True)
            return

        delivery_method = 'webhook'
        child_bot_id = None
        
        session = self.multi_profile_channels.get(channel.id)
        if session:
            # Fix: Use the standard hydration method to ensure ChatSession objects are created,
            # not raw lists from _load_session_from_disk.
            if not session.get("is_hydrated"):
                session = self._ensure_session_hydrated(channel.id, session.get("type", "multi"))
                if not session:
                    await interaction_to_respond.followup.send("Failed to load session data.", ephemeral=True)
                    return

            participant_data = next((p for p in session.get("profiles", []) if p.get("owner_id") == user_id and p.get("profile_name") == profile_name), None)
            if not participant_data:
                await interaction_to_respond.followup.send(f"The profile '{profile_name}' is not an active participant in this channel's multi-profile session.", ephemeral=True)
                return
            
            session_method = participant_data.get("method", "webhook")
            child_bot_id = participant_data.get("bot_id")

            if method == 'auto':
                delivery_method = session_method
            elif method == 'child_bot':
                if session_method == 'child_bot' and child_bot_id:
                    delivery_method = 'child_bot'
                else:
                    await interaction_to_respond.followup.send(f"This profile is not configured to use a Child Bot in this session. Cannot use 'child_bot' method.", ephemeral=True)
                    return
        else:
            linked_bot_id = next((bot_id for bot_id, data in self.child_bots.items() if data.get("owner_id") == effective_owner_id and data.get("profile_name") == effective_profile_name), None)
            if linked_bot_id and channel.guild.get_member(int(linked_bot_id)):
                child_bot_id = linked_bot_id

            if method == 'auto':
                if child_bot_id:
                    delivery_method = 'child_bot'
            elif method == 'child_bot':
                if child_bot_id:
                    delivery_method = 'child_bot'
                else:
                    await interaction_to_respond.followup.send(f"The profile '{effective_profile_name}' is not linked to a Child Bot, or that bot is not present in this server. Cannot use 'child_bot' method.", ephemeral=True)
                    return

        speaker_display_name = effective_profile_name
        appearance_data = self.user_appearances.get(str(effective_owner_id), {}).get(effective_profile_name, {})
        if appearance_data.get("custom_display_name"):
            speaker_display_name = appearance_data["custom_display_name"]

        from google.generativeai.types import content_types
        history_line = self._format_history_entry(speaker_display_name, interaction_to_respond.created_at, message)
        turn_info = None
        mapping_key = None
        
        if session:
            participant_key = (user_id, profile_name)
            model_content_obj = content_types.to_content({'role': 'model', 'parts': [history_line]})
            user_content_obj = content_types.to_content({'role': 'user', 'parts': [history_line]})

            # Add the turn to the unified log
            turn_id = str(uuid.uuid4())
            turn_object = {
                "turn_id": turn_id,
                "speaker_key": [user_id, profile_name],
                "content": history_line,
                "timestamp": interaction_to_respond.created_at.isoformat()
            }
            session.get("unified_log", []).append(turn_object)

            # Update in-memory histories for all participants
            for key, chat_session in session["chat_sessions"].items():
                if key == participant_key:
                    chat_session.history.append(model_content_obj)
                else:
                    chat_session.history.append(user_content_obj)
            
            session_type = session.get("type", "multi")
            turn_info = (channel.id, session_type, turn_id)
            mapping_key = (session_type, channel.id)
            self.session_last_accessed[channel.id] = time.time()
        else:
            model_cache_key = (channel.id, user_id, profile_name)
            chat_session = self.chat_sessions.get(model_cache_key)
            dummy_model = genai.GenerativeModel('gemini-flash-latest')
            if not chat_session:
                chat_session = self._load_session_from_disk(model_cache_key, 'single', dummy_model)
            if not chat_session:
                chat_session = dummy_model.start_chat(history=[])
            self.chat_sessions[model_cache_key] = chat_session

            fake_user_turn = content_types.to_content({'role': 'user', 'parts': ["<internal_note>Admin initiated message via /speak</internal_note>"]})
            model_turn = content_types.to_content({'role': 'model', 'parts': [history_line]})
            chat_session.history.extend([fake_user_turn, model_turn])
            
            turn_index = len(chat_session.history) - 2
            turn_info = (model_cache_key, 'single', turn_index)
            mapping_key = self._get_mapping_key_for_session(model_cache_key, 'single')
            self.session_last_accessed[model_cache_key] = time.time()

        sent_messages = []
        if delivery_method == 'child_bot' and child_bot_id:
            correlation_id = str(uuid.uuid4())
            
            if session:
                participant_data = next((p for p in session.get("profiles", []) if p.get("owner_id") == user_id and p.get("profile_name") == profile_name), None)
                _, _, turn_id = turn_info # Unpack the turn_id created earlier
                self.pending_child_confirmations[correlation_id] = {
                    "type": "multi_profile", "participant": participant_data,
                    "history_line": history_line, "channel_id": channel.id, "turn_id": turn_id,
                    "is_speak_as": True # Add the flag here
                }
            else:
                _, _, turn_index = turn_info
                self.pending_child_confirmations[correlation_id] = {
                    "type": "single_profile", "user_turn": fake_user_turn, "model_turn": model_turn,
                    "bot_id": child_bot_id, "channel_id": channel.id, "turn_index": turn_index
                }

            await self.manager_queue.put({
                "action": "send_to_child", "bot_id": child_bot_id,
                "payload": {
                    "action": "send_message", "channel_id": channel.id, "content": message,
                    "realistic_typing": profile_data_source.get("realistic_typing_enabled", False),
                    "correlation_id": correlation_id
                }
            })
        else:
            sent_messages = await self._send_channel_message(
                channel, message,
                profile_owner_id_for_appearance=effective_owner_id,
                profile_name_for_appearance=effective_profile_name
            )

        if sent_messages and turn_info and mapping_key:
            if mapping_key not in self.mapping_caches:
                self.mapping_caches[mapping_key] = self._load_mapping_from_disk(mapping_key)
            for msg in sent_messages:
                self.message_to_history_turn[msg.id] = turn_info
                self.mapping_caches[mapping_key][str(msg.id)] = turn_info

        await interaction_to_respond.followup.send("Message sent.", ephemeral=True)

    async def _execute_global_chat(self, interaction: discord.Interaction, profile_name: str, message: str):
        t1_start_mono = time.monotonic()
        t1_start_utc = datetime.datetime.now(datetime.timezone.utc)
        user_id = interaction.user.id
        
        if not self._is_profile_public(user_id, profile_name):
            await interaction.followup.send(f"This command is reserved for interacting with publicly published profiles. Your selected profile, **'{profile_name}'**, is not public.\n\nPlease use `/profile public manage` to publish it.", ephemeral=True)
            return

        user_data = self._get_user_data_entry(user_id)
        is_borrowed = profile_name in user_data.get("borrowed_profiles", {})
        
        source_owner_id = user_id
        source_profile_name = profile_name
        if is_borrowed:
            borrowed_data = user_data["borrowed_profiles"][profile_name]
            source_owner_id = int(borrowed_data["original_owner_id"])
            source_profile_name = borrowed_data["original_profile_name"]
        
        source_owner_data = self._get_user_data_entry(source_owner_id)
        profile_data = source_owner_data.get("profiles", {}).get(source_profile_name)

        if not profile_data:
            await interaction.followup.send(f"The source for your selected profile ('{profile_name}') could not be found.", ephemeral=True)
            return

        safety_level = profile_data.get("safety_level", "low")
        if safety_level == "unrestricted":
            await interaction.followup.send("For safety reasons, profiles with an 'Unrestricted 18+' safety level cannot be used with `/profile global_chat`. Please set the safety level to 'Low', 'Medium', or 'High'.", ephemeral=True)
            return

        user_api_key = self._get_api_key_for_user(user_id)
        if not user_api_key:
            await interaction.followup.send("This feature requires you to have your own personal API key. Please use the `/settings` command in a DM with me to submit one.", ephemeral=True)
            return
        genai.configure(api_key=user_api_key)

        model_cache_key = ('global', user_id, profile_name)
        
        try:
            model, temp, top_p, top_k, warning_message, fallback_model_name = await self._get_or_create_model_for_global_chat(user_id, profile_name)
            
            if warning_message:
                try: await interaction.user.send(warning_message)
                except discord.Forbidden: pass

            if not model:
                error_msg = warning_message or "Could not initialize the AI model for your profile."
                await interaction.followup.send(error_msg, ephemeral=True)
                return

            session_data = self.global_chat_sessions.get(model_cache_key)
            if not session_data:
                session_data = self._load_session_from_disk(model_cache_key, 'global_chat', model)
            
            if not session_data:
                chat = model.start_chat(history=[])
                session_data = {'chat_session': chat, 'unified_log': []}
            
            self.global_chat_sessions[model_cache_key] = session_data
            chat = session_data['chat_session']
            self.session_last_accessed[model_cache_key] = time.time()

            # [UPDATED] Standardized XML Headers
            from google.generativeai.types import content_types
            rebuilt_history = []
            for t in session_data['unified_log']:
                t_role = t.get('role')
                parts = [t.get('content')]
                
                if t_role == 'user':
                    if t.get('url_context') and profile_data.get('url_fetching_enabled', True):
                        parts.append(f"\n<document_context>\n{t.get('url_context')}\n</document_context>")
                    if t.get('grounding_context') and profile_data.get('grounding_mode', 'off') != 'off':
                        parts.append(f"\n<external_context>\n{t.get('grounding_context')}\n</external_context>")
                
                rebuilt_history.append(content_types.to_content({'role': t_role, 'parts': parts}))
            
            chat.history = rebuilt_history

            if len(chat.history) > STM_LIMIT_MAX * 2:
                chat.history = chat.history[-(STM_LIMIT_MAX * 2):]
                session_data['unified_log'] = session_data['unified_log'][-(STM_LIMIT_MAX * 2):]
            
            ltm_recall_text = await self._get_relevant_ltm_for_prompt(model_cache_key, chat.history, user_id, profile_name, message, interaction.user.display_name, guild_id=None, triggering_user_id=user_id)
            
            user_data = self._get_user_data_entry(user_id)
            is_borrowed = profile_name in user_data.get("borrowed_profiles", {})
            source_owner_id = user_id
            source_profile_name = profile_name
            if is_borrowed:
                borrowed_data = user_data["borrowed_profiles"][profile_name]
                source_owner_id = int(borrowed_data["original_owner_id"])
                source_profile_name = borrowed_data["original_profile_name"]
            
            bot_display_name = source_profile_name
            appearance = self.user_appearances.get(str(source_owner_id), {}).get(source_profile_name)
            if appearance and appearance.get("custom_display_name"):
                bot_display_name = appearance.get("custom_display_name")

            from google.generativeai.types import content_types
            contents_for_api_call = []
            
            # [NEW] Localized User Timestamp Logic
            user_tz = profile_data.get("timezone", "UTC")
            user_line = self._format_history_entry(interaction.user.display_name, interaction.created_at, message, user_tz)
            
            final_user_parts = []
            if ltm_recall_text:
                final_user_parts.append(ltm_recall_text)
            
            if profile_data.get('url_fetching_enabled', True):
                u_text, _ = await self._process_urls_in_content(message, 0, {"url_fetching_enabled": True})
                if u_text:
                    final_user_parts.append(f"<document_context>\n" + "\n".join(u_text) + "\n</document_context>")

            final_user_parts.append(user_line)

            user_content_obj_for_turn = content_types.to_content({'role': 'user', 'parts': final_user_parts})
            
            stm_length = int(profile_data.get("stm_length", defaultConfig.CHATBOT_MEMORY_LENGTH))
            if stm_length > 0:
                history_slice = chat.history[-(stm_length * 2):]
                contents_for_api_call.extend(history_slice)

            contents_for_api_call.append(user_content_obj_for_turn)
            
            # Enable internal thoughts but keep summary display off (UI ignores it)
            gen_config = google_genai_types.GenerateContentConfig(
                temperature=temp, top_p=top_p, top_k=top_k,
                thinking_config=google_genai_types.ThinkingConfig(include_thoughts=True)
            )
            status = "api_error"
            response = None
            fallback_used = False
            blocked_reason_override = None
            try:
                response = await model.generate_content_async(contents_for_api_call, generation_config=gen_config)
                if not response or not response.candidates:
                    raise ValueError("Response blocked or empty")
                status = "success"
            except Exception as e:
                blocked_reason_override = self._format_api_error(e)

                if fallback_model_name:
                    try:
                        sys_instr, _, _, _, _, _, _, _ = self._construct_system_instructions(user_id, profile_name, 0)
                        
                        user_data_f = self._get_user_data_entry(user_id)
                        is_borrowed_f = profile_name in user_data_f.get("borrowed_profiles", {})
                        source_id_f = user_id
                        source_name_f = profile_name
                        if is_borrowed_f:
                            bd = user_data_f["borrowed_profiles"][profile_name]
                            source_id_f = int(bd["original_owner_id"])
                            source_name_f = bd["original_profile_name"]
                        
                        p_data_f = self._get_user_data_entry(source_id_f).get("profiles", {}).get(source_name_f, {})
                        safe_lvl = p_data_f.get("safety_level", "low")
                        
                        s_map = { "unrestricted": HarmBlockThreshold.BLOCK_NONE, "low": HarmBlockThreshold.BLOCK_ONLY_HIGH, "medium": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, "high": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE }
                        thresh = s_map.get(safe_lvl, HarmBlockThreshold.BLOCK_ONLY_HIGH)
                        d_safe = { cat: thresh for cat in get_args(HarmCategory) }

                        fb_name = fallback_model_name
                        fb_is_or = False
                        
                        if fb_name.upper().startswith("OPENROUTER/"):
                            fb_name = fb_name[11:]
                            fb_is_or = True
                        elif fb_name.upper().startswith("GOOGLE/"):
                            fb_name = fb_name[7:]
                            fb_is_or = False
                        elif "/" in fb_name:
                            fb_is_or = True
                        
                        if fb_is_or:
                            or_key = self._get_api_key_for_user(user_id, provider="openrouter")
                            if or_key:
                                fallback_instance = OpenRouterModel(fb_name, api_key=or_key, system_instruction=sys_instr, thinking_params={})
                            else:
                                raise ValueError("No OR key for fallback")
                        else:
                            user_key = self._get_api_key_for_user(user_id)
                            if user_key:
                                t_params_f = {
                                    "thinking_summary_visible": p_data_f.get("thinking_summary_visible", "off"),
                                    "thinking_level": p_data_f.get("thinking_level", "high"),
                                    "thinking_budget": p_data_f.get("thinking_budget", -1)
                                }
                                fallback_instance = GoogleGenAIModel(api_key=user_key, model_name=fb_name, system_instruction=sys_instr, safety_settings=d_safe, thinking_params=t_params_f)
                            else:
                                raise ValueError("No Google key for fallback")

                        response = await fallback_instance.generate_content_async(contents_for_api_call, generation_config=gen_config)
                        status = "blocked_by_safety" if not response.candidates else "success"
                        if status == "success":
                            fallback_used = True
                            self._log_api_call(user_id=user_id, guild_id=None, context="global_chat_fallback", model_used=fb_name, status="success")
                    except Exception as retry_e:
                        print(f"Global Chat fallback retry failed: {retry_e}")
                        status = "api_error"
                else:
                    status = "api_error"

            except api_exceptions.PermissionDenied as e:
                await interaction.followup.send("An error has occurred. Your personal API key appears to be invalid or disabled.", ephemeral=True)
                return
            finally:
                # Always log the primary model's final status
                self._log_api_call(user_id=user_id, guild_id=None, context="global_chat", model_used=model.model_name if hasattr(model, 'model_name') else "unknown", status=status)

            if not response or not response.candidates:
                reason = blocked_reason_override or "Unknown"
                if response and response.prompt_feedback and response.prompt_feedback.block_reason:
                    reason = response.prompt_feedback.block_reason.name.replace('_', ' ').title()
                
                custom_main = profile_data.get("error_response", "An error has occurred.")
                await interaction.followup.send(f"{custom_main}\n\n-# Blocked due to: **{reason}**.", ephemeral=False)
                return

            model_content_obj_for_turn = content_types.to_content({'role': 'model', 'parts': [response.text]})
            chat.history.extend([user_content_obj_for_turn, model_content_obj_for_turn])

            if len(chat.history) > STM_LIMIT_MAX * 2:
                chat.history = chat.history[-(STM_LIMIT_MAX * 2):]
                session_data['unified_log'] = session_data['unified_log'][-(STM_LIMIT_MAX * 2):]
            
            raw_text = response.text.strip() if hasattr(response, 'text') else "I could not generate a response."
            
            # Apply filters
            display_name = source_profile_name
            avatar_url = self.bot.user.display_avatar.url
            if appearance:
                display_name = appearance.get("custom_display_name") or display_name
                avatar_url = appearance.get("custom_avatar_url") or avatar_url

            scrubbed_text = self._scrub_response_text(raw_text, participant_names=[display_name])
            response_text = self._deduplicate_response(scrubbed_text)

            # --- Turn Logging ---
            user_turn_id = str(uuid.uuid4())
            model_turn_id = str(uuid.uuid4())
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
            session_data['unified_log'].append({
                "turn_id": user_turn_id, "role": "user", "content": message, "timestamp": timestamp
            })
            session_data['unified_log'].append({
                "turn_id": model_turn_id, "role": "model", "content": response_text, "timestamp": timestamp
            })

            text_for_embed = response_text
            if fallback_used and profile_data.get("show_fallback_indicator", True):
                text_for_embed += f"\n\n-# Fallback Model Used ({blocked_reason_override})"

            embed = discord.Embed(description=text_for_embed, color=discord.Color.blue())
            embed.set_author(name=display_name, icon_url=avatar_url)
            embed.set_footer(text=message, icon_url=interaction.user.display_avatar.url)
            
            response_message = await interaction.followup.send(embed=embed, ephemeral=False)

            t2_end_mono = time.monotonic()
            duration = t2_end_mono - t1_start_mono
            
            # Update the history object with the metadata (Bot Only)
            timezone_str = profile_data.get("timezone", "UTC")
            main_history_line = self._format_history_entry(display_name, response_message.created_at, response_text, timezone_str)
            try:
                t1_formatted = t1_start_utc.astimezone(ZoneInfo(timezone_str)).strftime('%I:%M:%S %p %Z')
            except Exception:
                t1_formatted = t1_start_utc.strftime('%I:%M:%S %p UTC')
            metadata_line = f"(Thought Initiated: {t1_formatted} | Duration: {duration:.2f}s)"
            bot_response_formatted = f"{main_history_line.strip()}\n{metadata_line}\n"
            
            # Replace the last model turn in history with the one containing metadata for context
            if chat.history and chat.history[-1].role == 'model':
                from google.generativeai.types import content_types
                chat.history[-1] = content_types.to_content({'role': 'model', 'parts': [bot_response_formatted]})

            # Mapping Logic using model_turn_id
            turn_data = (model_cache_key, 'global_chat', model_turn_id)
            mapping_key = self._get_mapping_key_for_session(model_cache_key, 'global_chat')
            if mapping_key not in self.mapping_caches:
                self.mapping_caches[mapping_key] = self._load_mapping_from_disk(mapping_key)
            
            self.message_to_history_turn[response_message.id] = turn_data
            self.mapping_caches[mapping_key][str(response_message.id)] = turn_data

            # Persist immediately to disk for safety and UI consistency
            self._save_session_to_disk(model_cache_key, 'global_chat', session_data)
            
            # Also save mapping
            self._save_mapping_to_disk(mapping_key, self.mapping_caches[mapping_key])

            await self._maybe_create_ltm(
                interaction.channel, interaction.user.display_name, chat.history, user_id, profile_name,
                {"temperature": temp, "top_p": top_p, "top_k": top_k},
                force_user_scope=True
            )

        except Exception as e:
            await interaction.followup.send(f"An error occurred during the global chat: {e}", ephemeral=True)
            traceback.print_exc()

    async def _execute_whisper(self, interaction: discord.Interaction, target_participant: Dict, whisper_message: str):
        from google.generativeai.types import content_types
        session = self.multi_profile_channels.get(interaction.channel_id)
        if not session: return

        owner_id = target_participant['owner_id']
        profile_name = target_participant['profile_name']
        participant_key = (owner_id, profile_name)

        # Ensure session is hydrated to get history
        if not session.get("is_hydrated"):
            session = self._ensure_session_hydrated(interaction.channel_id, session.get("type", "multi"))

        chat_session = session.get("chat_sessions", {}).get(participant_key)
        if not chat_session:
            await interaction.followup.send("An error occurred: Could not find the chat session for that participant.", ephemeral=True)
            return

        # Get model and settings for the target profile
        model, _, temp, top_p, top_k, _, fallback_model_name = await self._get_or_create_model_for_channel(
            interaction.channel_id, owner_id, interaction.guild.id,
            profile_owner_override=owner_id, profile_name_override=profile_name
        )
        if not model:
            await interaction.followup.send("Could not initialize the AI model for that profile.", ephemeral=True)
            return

        # Construct the prompt for the private response
        whisper_prompt = f"<private_context author='{interaction.user.name}'>\n{whisper_message}\nYour response will be sent privately. Respond directly to the user.\n</private_context>"
        
        # Use shallow copy of history list
        contents_for_api_call = list(chat_session.history)
        contents_for_api_call.append(content_types.to_content({'role': 'user', 'parts': [whisper_prompt]}))
        
        # Enable internal thoughts but keep summary display off (UI ignores it)
        gen_config = google_genai_types.GenerateContentConfig(
            temperature=temp, top_p=top_p, top_k=top_k,
            thinking_config=google_genai_types.ThinkingConfig(include_thoughts=True)
        )
        
        status = "api_error"
        response = None
        try:
            response = await model.generate_content_async(contents_for_api_call, generation_config=gen_config)
            status = "blocked_by_safety" if not response.candidates else "success"
        except Exception as e:
            print(f"Whisper generation error: {e}")
            status = "api_error"
        finally:
            self._log_api_call(user_id=interaction.user.id, guild_id=interaction.guild.id, context="whisper", model_used=model.model_name if hasattr(model, 'model_name') else "unknown", status=status)
        
        response_text = "..."
        if not response or not response.candidates:
            reason = "Safety Filter"
            if response and response.prompt_feedback and response.prompt_feedback.block_reason:
                reason = response.prompt_feedback.block_reason.name.replace('_', ' ').title()
            
            p_settings = self._get_user_data_entry(owner_id).get("profiles", {}).get(profile_name, {})
            if not p_settings: p_settings = self._get_user_data_entry(owner_id).get("borrowed_profiles", {}).get(profile_name, {})
            
            custom_main = p_settings.get("error_response", "An error has occurred.")
            response_text = f"{custom_main}\n\n-# Blocked due to: **{reason}**."
        elif response.candidates:
            response_text = getattr(response, 'text', "...").strip()

        user_data = self._get_user_data_entry(owner_id)
        is_borrowed = profile_name in user_data.get("borrowed_profiles", {})
        effective_owner_id = owner_id
        effective_profile_name = profile_name
        if is_borrowed:
            borrowed_data = user_data["borrowed_profiles"][profile_name]
            effective_owner_id = int(borrowed_data["original_owner_id"])
            effective_profile_name = borrowed_data["original_profile_name"]
        
        display_name = effective_profile_name
        appearance = self.user_appearances.get(str(effective_owner_id), {}).get(effective_profile_name)
        if appearance:
            display_name = appearance.get("custom_display_name") or display_name

        scrubbed_text = self._scrub_response_text(response_text, participant_names=[display_name])
        response_text = self._deduplicate_response(scrubbed_text)

        # Log the whisper and the private response to the unified log
        whisper_turn_id = str(uuid.uuid4())
        whisper_content = self._format_history_entry(f"Whisper from {interaction.user.name}", interaction.created_at, whisper_message)
        session["unified_log"].append({
            "turn_id": whisper_turn_id, "type": "whisper",
            "whisperer_id": interaction.user.id, "target_key": list(participant_key),
            "content": whisper_content, "timestamp": interaction.created_at.isoformat()
        })

        response_turn_id = str(uuid.uuid4())
        response_content = self._format_history_entry(profile_name, datetime.datetime.now(datetime.timezone.utc), response_text)
        session["unified_log"].append({
            "turn_id": response_turn_id, "type": "private_response",
            "speaker_key": list(participant_key), "target_id": interaction.user.id,
            "content": response_content, "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        })

        # Add to the target's in-memory history
        chat_session.history.append(content_types.to_content({'role': 'user', 'parts': [whisper_content]}))
        chat_session.history.append(content_types.to_content({'role': 'model', 'parts': [response_content]}))

        # Add to pending whispers to be injected into the next public turn
        session.setdefault("pending_whispers", {}).setdefault(participant_key, []).append(whisper_content)

        # Send the private response to the user
        user_data = self._get_user_data_entry(owner_id)
        is_borrowed = profile_name in user_data.get("borrowed_profiles", {})
        effective_owner_id = owner_id
        effective_profile_name = profile_name
        if is_borrowed:
            borrowed_data = user_data["borrowed_profiles"][profile_name]
            effective_owner_id = int(borrowed_data["original_owner_id"])
            effective_profile_name = borrowed_data["original_profile_name"]
        
        display_name = effective_profile_name
        avatar_url = self.bot.user.display_avatar.url
        appearance = self.user_appearances.get(str(effective_owner_id), {}).get(effective_profile_name)
        if appearance:
            display_name = appearance.get("custom_display_name") or display_name
            avatar_url = appearance.get("custom_avatar_url") or avatar_url

        embed = discord.Embed(description=response_text, color=discord.Color.dark_grey())
        embed.set_author(name=display_name, icon_url=avatar_url)
        embed.set_footer(text=f"Private whisper: {whisper_message}", icon_url=interaction.user.display_avatar.url)
        
        view = WhisperActionView(self, interaction, whisper_turn_id, response_turn_id)
        await interaction.followup.send(embed=embed, view=view, ephemeral=True)

    async def _execute_export(self, interaction: discord.Interaction, profile_names: List[str], filters: Set[str]):
        user_id = interaction.user.id
        user_id_str = str(user_id)
        user_data = self._get_user_data_entry(user_id)
        
        export_data = {
            "version": "1.0",
            "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "profiles": {}
        }

        for name in profile_names:
            p_config = user_data.get("profiles", {}).get(name)
            if not p_config: continue

            p_entry = {
                "config": {k: v for k, v in p_config.items() if k not in ["persona", "ai_instructions"]},
                "persona": {},
                "ai_instructions": [] if isinstance(p_config.get("ai_instructions"), list) else "",
                "appearance": None,
                "ltm": [],
                "training": []
            }

            if "persona" in filters:
                p_entry["persona"] = {k: [self._decrypt_data(line) for line in v] for k, v in p_config.get("persona", {}).items()}
            
            if "instructions" in filters:
                raw_instr = p_config.get("ai_instructions")
                if isinstance(raw_instr, list):
                    p_entry["ai_instructions"] = [self._decrypt_data(p) for p in raw_instr]
                else:
                    p_entry["ai_instructions"] = self._decrypt_data(raw_instr or "")

            if "appearance" in filters:
                p_entry["appearance"] = self.user_appearances.get(user_id_str, {}).get(name)

            if "ltm" in filters:
                ltm_shard = self._load_ltm_shard(user_id_str, name)
                if ltm_shard:
                    for entry in ltm_shard.get("guild", []):
                        p_entry["ltm"].append({
                            "sum": self._decrypt_data(entry.get("sum", "")),
                            "s_emb": entry.get("s_emb"),
                            "scope": entry.get("scope", "server"),
                            "ts": entry.get("created_ts", entry.get("ts"))
                        })

            if "training" in filters:
                train_shard = self._load_training_shard(user_id_str, name)
                if train_shard:
                    for entry in train_shard:
                        p_entry["training"].append({
                            "u_in": self._decrypt_data(entry.get("u_in", "")),
                            "b_out": self._decrypt_data(entry.get("b_out", "")),
                            "u_emb": entry.get("u_emb"),
                            "ts": entry.get("created_ts", entry.get("ts"))
                        })
            
            export_data["profiles"][name] = p_entry

        file_data = json.dumps(export_data, option=json.OPT_INDENT_2)
        filename = f"mimic_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.mimic"
        
        buffer = io.BytesIO(file_data)
        discord_file = discord.File(buffer, filename=filename)
        
        await interaction.followup.send("✅ Export complete. Keep this file safe; it contains your data in plaintext.", file=discord_file, ephemeral=True)

    async def _execute_import(self, interaction: discord.Interaction, attachment: discord.Attachment):
        try:
            file_bytes = await attachment.read()
            data = json.loads(file_bytes)
            
            if "profiles" not in data:
                raise ValueError("Invalid MimicAI data structure.")

            user_id = interaction.user.id
            user_id_str = str(user_id)
            user_data = self._get_user_data_entry(user_id)
            
            import_log = []
            for name, p_data in data["profiles"].items():
                local_name = name
                # Handle Collisions
                if local_name in user_data["profiles"]:
                    local_name = f"{name}_imported_{uuid.uuid4().hex[:4]}"
                
                # 1. Encrypt and save Profile
                clean_persona = {k: [self._encrypt_data(line) for line in v] for k, v in p_data.get("persona", {}).items()}
                
                instr_raw = p_data.get("ai_instructions", "")
                clean_instr = [self._encrypt_data(p) for p in instr_raw] if isinstance(instr_raw, list) else self._encrypt_data(instr_raw)
                
                new_profile = p_data["config"]
                new_profile["persona"] = clean_persona
                new_profile["ai_instructions"] = clean_instr
                user_data["profiles"][local_name] = new_profile

                # 2. Encrypt and save LTM
                if p_data.get("ltm"):
                    ltm_list = []
                    for entry in p_data["ltm"]:
                        now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
                        ltm_list.append({
                            "id": uuid.uuid4().hex[:8], "sum": self._encrypt_data(entry["sum"]),
                            "s_emb": entry["s_emb"], "scope": entry.get("scope", "server"),
                            "created_ts": entry.get("ts", now_ts), "modified_ts": now_ts
                        })
                    self._save_ltm_shard(user_id_str, local_name, {"guild": ltm_list, "dm": []})

                # 3. Encrypt and save Training
                if p_data.get("training"):
                    train_list = []
                    for entry in p_data["training"]:
                        now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
                        train_list.append({
                            "id": uuid.uuid4().hex[:8], "u_in": self._encrypt_data(entry["u_in"]),
                            "b_out": self._encrypt_data(entry["b_out"]), "u_emb": entry["u_emb"],
                            "created_ts": entry.get("ts", now_ts), "modified_ts": now_ts
                        })
                    self._save_training_shard(user_id_str, local_name, train_list)

                # 4. Handle Appearance
                if p_data.get("appearance"):
                    self.user_appearances.setdefault(user_id_str, {})[local_name] = p_data["appearance"]
                
                import_log.append(f"- `{local_name}`")

            self._save_user_data_entry(user_id, user_data)
            self._save_user_appearance_shard(user_id_str, self.user_appearances.get(user_id_str))
            
            await interaction.followup.send(f"### 📥 Import Successful\nThe following profiles have been added to your vault:\n" + "\n".join(import_log), ephemeral=True)

        except Exception as e:
            await interaction.followup.send(f"❌ **Import Failed:** {e}", ephemeral=True)

    def _try_acquire_lock(self):
        try:
            if not os.path.exists(COG_LOCK_FILE_PATH):
                with open(COG_LOCK_FILE_PATH, "w") as f:
                    f.write(str(time.time()))
                self.has_lock = True
            else:
                with open(COG_LOCK_FILE_PATH, "r") as f:
                    lock_time_str = f.read().strip()
                if not lock_time_str: 
                    with open(COG_LOCK_FILE_PATH, "w") as f:
                        f.write(str(time.time()))
                    self.has_lock = True
                    return

                lock_time = float(lock_time_str)
                if (time.time() - lock_time) > LOCK_STALE_THRESHOLD_SECONDS:
                    with open(COG_LOCK_FILE_PATH, "w") as f:
                        f.write(str(time.time()))
                    self.has_lock = True
                else:
                    self.has_lock = False 
        except (IOError, ValueError) as e: 
            print(f"Error during lock acquisition: {e}. Assuming no lock.")
            self.has_lock = False 
        except Exception as e:
            print(f"Unexpected error during lock acquisition: {e}. Assuming no lock.")
            self.has_lock = False

    def is_user_premium(self, user_id: int) -> bool:
        """
        Determines if a user has premium privileges.
        In self-hosted mode, the instance owner is always premium.
        The ALL_USERS_PREMIUM flag can be used to unlock features for everyone.
        """
        is_owner = user_id == int(defaultConfig.DISCORD_OWNER_ID)
        allow_all = getattr(defaultConfig, "ALL_USERS_PREMIUM", True)
        return is_owner or allow_all
    
    def _generate_unique_local_name(self, user_id: int, original_name: str, sharer_name: str) -> str:
        user_data = self._get_user_data_entry(user_id)
        all_profile_names = set(user_data.get("profiles", {}).keys()) | set(user_data.get("borrowed_profiles", {}).keys())
        
        base_name = f"{original_name}-{sharer_name}".lower().strip()
        if base_name not in all_profile_names:
            return base_name
        
        counter = 2
        while True:
            new_name = f"{base_name}-{counter}"
            if new_name not in all_profile_names:
                return new_name
            counter += 1

    def _check_unrestricted_safety_policy(self, profile_owner_id: int, profile_name: str, channel: discord.abc.Messageable) -> bool:
        user_data = self._get_user_data_entry(profile_owner_id)
        is_borrowed = profile_name in user_data.get("borrowed_profiles", {})
        profile_data = user_data.get("borrowed_profiles" if is_borrowed else "profiles", {}).get(profile_name, {})
        
        safety_level = profile_data.get("safety_level", "low")

        if safety_level == "unrestricted":
            if not isinstance(channel, (discord.TextChannel, discord.Thread, discord.VoiceChannel)):
                return False # Cannot be NSFW in this channel type
            return channel.is_nsfw()
        
        return True # Not unrestricted, so it's allowed.
    
    async def setup_multi_profile_session(self, interaction: discord.Interaction, participants: List[Dict], session_prompt: Optional[str], session_mode: str, as_admin_scope: bool = False, audio_mode: str = "text-only"):
        user_id = interaction.user.id
        is_update = interaction.channel_id in self.multi_profile_channels
        from google.generativeai.types import content_types

        if is_update:
            session = self.multi_profile_channels[interaction.channel_id]
            if not session.get("is_hydrated"):
                session = self._ensure_session_hydrated(interaction.channel_id, session.get("type", "multi"))

            existing_keys = set(session.get("chat_sessions", {}).keys())
            new_keys = { (p['owner_id'], p['profile_name']) for p in participants }
            
            for key_to_add in new_keys - existing_keys:
                dummy_model = genai.GenerativeModel('gemini-flash-latest')
                participant_history = []
                for turn in session.get("unified_log", []):
                    speaker_key = tuple(turn.get("speaker_key", []))
                    role = 'model' if speaker_key == key_to_add else 'user'
                    content_obj = content_types.to_content({'role': role, 'parts': [turn.get("content")]})
                    participant_history.append(content_obj)
                session["chat_sessions"][key_to_add] = dummy_model.start_chat(history=participant_history)

            for key_to_remove in existing_keys - new_keys:
                session["chat_sessions"].pop(key_to_remove, None)
        else:
            chat_sessions = { (p['owner_id'], p['profile_name']): None for p in participants }
            for p in participants: p['ltm_counter'] = 0

            session = {
                "type": "multi",
                "chat_sessions": chat_sessions,
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
                "audio_mode": "text-only"
            }
            self.multi_profile_channels[interaction.channel_id] = session

        session["type"] = "multi"
        session["session_prompt"] = session_prompt
        session["profiles"] = participants
        session["session_mode"] = session_mode
        session["audio_mode"] = audio_mode
        
        for p_data in participants:
            if p_data.get('method') == 'child_bot':
                await self.manager_queue.put({
                    "action": "send_to_child", "bot_id": p_data['bot_id'],
                    "payload": {"action": "session_update_add", "channel_id": interaction.channel_id}
                })
        
        self._save_multi_profile_sessions()

        profile_list_str = []
        for p_data in participants:
            if p_data.get('method') == 'child_bot':
                bot_user = self.bot.get_user(int(p_data['bot_id']))
                profile_list_str.append(f"`{bot_user.name if bot_user else 'Unknown Bot'}`")
            else:
                profile_list_str.append(f"`{p_data['profile_name']}`")

        action_str = "updated" if is_update else "activated"
        msg = f"Regular session {action_str} with participants: {', '.join(profile_list_str)}."
        if as_admin_scope:
            msg = f"Regular session is now active for all users with profiles: {', '.join(profile_list_str)}."
        
        await interaction.edit_original_response(content=msg, view=None)

    def _ensure_session_hydrated(self, channel_id: int, session_type: str) -> Optional[Dict]:
        """Checks if a session is in memory and hydrated. If not, loads it from disk."""
        session = self.multi_profile_channels.get(channel_id)
        if session and session.get("is_hydrated"):
            return session

        # If the session object exists but isn't hydrated, we can use its participant list.
        # Otherwise, we need to load the blueprint from disk.
        if not session:
            channel = self.bot.get_channel(channel_id)
            if not channel or not hasattr(channel, 'guild'): return None
            server_id_str = str(channel.guild.id)
            sessions_file = os.path.join(self.FREEWILL_SERVERS_DIR, server_id_str, "sessions.json.gz")
            if not os.path.exists(sessions_file): return None
            all_sessions_config = self._load_json_gzip(sessions_file)
            if not all_sessions_config: return None
            session_config = all_sessions_config.get(str(channel_id))
            if not session_config: return None
            
            # This is a partial session object, the worker will fill in the rest
            profiles = session_config.get("profiles", [])
            for p in profiles:
                p.setdefault('ltm_counter', 0)
                
            session = {"profiles": profiles}
            self.multi_profile_channels[channel_id] = session
        
        from google.generativeai.types import content_types
        dummy_model = genai.GenerativeModel('gemini-flash-latest')
        dummy_session_key = (channel_id, None, None)
        unified_log = self._load_session_from_disk(dummy_session_key, session_type, dummy_model) or []
        
        session["unified_log"] = unified_log
        session["chat_sessions"] = {}

        num_participants = len(session.get("profiles", []))

        for p_data in session["profiles"]:
            p_key = (p_data['owner_id'], p_data['profile_name'])
            
            p_user_data = self._get_user_data_entry(p_data['owner_id'])
            p_is_borrowed = p_data['profile_name'] in p_user_data.get("borrowed_profiles", {})
            p_profile_settings = p_user_data.get("borrowed_profiles" if p_is_borrowed else "profiles", {}).get(p_data['profile_name'], {})
            stm_length = int(p_profile_settings.get("stm_length", defaultConfig.CHATBOT_MEMORY_LENGTH))

            history_slice = []
            if stm_length > 0:
                effective_stm = max(stm_length, num_participants)
                history_slice = unified_log[-(effective_stm * 2):]

            participant_history = []
            for turn in history_slice:
                # [NEW] Ignore turns hidden via reaction
                if turn.get("is_hidden", False):
                    continue
                    
                turn_type = turn.get("type")
                if not turn_type: # Handle old logs
                    speaker_key = tuple(turn.get("speaker_key", []))
                    role = 'model' if speaker_key == p_key else 'user'
                    
                    parts = [turn.get("content")]
                    if role == 'user' and turn.get("url_context") and p_profile_settings.get("url_fetching_enabled", True):
                        parts.append(f"\n<document_context>\n{turn.get('url_context')}\n</document_context>")
                    
                    if role == 'user' and turn.get("grounding_context") and p_profile_settings.get("grounding_mode", "off") != "off":
                        parts.append(f"\n<external_context>\n{turn.get('grounding_context')}\n</external_context>")

                    content_obj = content_types.to_content({'role': role, 'parts': parts})
                    participant_history.append(content_obj)
                elif turn_type == "whisper":
                    target_key = tuple(turn.get("target_key", []))
                    if p_key == target_key:
                        content_obj = content_types.to_content({'role': 'user', 'parts': [turn.get("content")]})
                        participant_history.append(content_obj)
                elif turn_type == "private_response":
                    speaker_key = tuple(turn.get("speaker_key", []))
                    if p_key == speaker_key:
                        content_obj = content_types.to_content({'role': 'model', 'parts': [turn.get("content")]})
                        participant_history.append(content_obj)
            
            session["chat_sessions"][p_key] = dummy_model.start_chat(history=participant_history)

        session["is_hydrated"] = True
        return session
    
    def _cleanup_freewill_session(self, channel_id: int):
        import shutil
        # 1. Cancel in-memory tasks if session is loaded
        session = self.multi_profile_channels.get(channel_id)
        if session:
            if session.get('worker_task'):
                self._safe_cancel_task(session['worker_task'])

        # 2. Delete all on-disk history files for this session by removing the directory
        session_type = "freewill"
        dummy_session_key = (channel_id, None, None)
        try:
            session_dir_path = self._get_session_dir_path(dummy_session_key, session_type)
            if session_dir_path.exists():
                shutil.rmtree(session_dir_path)
        except Exception as e:
            print(f"Error cleaning up freewill session directory for channel {channel_id}: {e}")

        # 3. Remove from in-memory caches
        self.multi_profile_channels.pop(channel_id, None)
        self.session_last_accessed.pop(channel_id, None)
        mapping_key = (session_type, channel_id)
        self.mapping_caches.pop(mapping_key, None)

        # 4. Persist the removal of the session from the main config blueprint file
        channel = self.bot.get_channel(channel_id)
        if channel and channel.guild:
            server_id_str = str(channel.guild.id)
            sessions_file = os.path.join(self.FREEWILL_SERVERS_DIR, server_id_str, "sessions.json.gz")
            if os.path.exists(sessions_file):
                all_sessions_config = self._load_json_gzip(sessions_file)
                if all_sessions_config and str(channel_id) in all_sessions_config:
                    del all_sessions_config[str(channel_id)]
                    if not all_sessions_config: # If the file is now empty
                        _delete_file_shard(sessions_file)
                    else:
                        self._atomic_json_save_gzip(all_sessions_config, sessions_file)

    async def _build_freewill_history(self, channel: discord.TextChannel, anchor_message: Optional[discord.Message] = None) -> List[Tuple[int, str, datetime.datetime, str]]:
        history_data = []
        try:
            limit = 20
            
            last_bot_message = None
            # Look back from the anchor message (if provided) to find the last bot message
            async for msg in channel.history(limit=100, before=anchor_message):
                if msg.author.id in self.all_bot_ids:
                    last_bot_message = msg
                    break
            
            # Fetch messages after the last bot message, up to the anchor message
            messages = [msg async for msg in channel.history(limit=limit, after=last_bot_message, before=anchor_message)]
            messages.reverse() # Process from oldest to newest
            
            # Add the anchor message itself to the end of the context if it exists
            if anchor_message:
                messages.append(anchor_message)

            for msg in messages:
                history_data.append((msg.author.id, msg.author.display_name, msg.created_at, msg.clean_content))

        except (discord.Forbidden, discord.HTTPException) as e:
            print(f"Could not fetch history for freewill: {e}")
            # Fallback to just the anchor message if history fetching fails
            if not history_data and anchor_message:
                 history_data.append((anchor_message.author.id, anchor_message.author.display_name, anchor_message.created_at, anchor_message.clean_content))
        
        return history_data
    
    async def _build_participant_embed(self, participant: Dict, channel_id: int) -> discord.Embed:
        owner_id = participant['owner_id']
        profile_name = participant['profile_name']
        
        # Fetch effective settings via the helper which handles overrides/defaults logic
        _, _, _, temp, topp, topk, training_ctx, training_rel, prim_model, fall_model = self._get_user_profile_for_model(
            owner_id, channel_id, profile_name_override=profile_name
        )

        user_data = self._get_user_data_entry(owner_id)
        is_borrowed = profile_name in user_data.get("borrowed_profiles", {})
        
        # Determine source data
        if is_borrowed:
            borrowed_data = user_data["borrowed_profiles"][profile_name]
            effective_owner_id = int(borrowed_data["original_owner_id"])
            effective_profile_name = borrowed_data["original_profile_name"]
            # Local overrides are in borrowed_data
            profile_data = borrowed_data
            
            # We need to fetch the source profile for some parameters that might fall back
            source_owner_data = self._get_user_data_entry(effective_owner_id)
            source_profile_data = source_owner_data.get("profiles", {}).get(effective_profile_name, {})
        else:
            effective_owner_id = owner_id
            effective_profile_name = profile_name
            profile_data = user_data.get("profiles", {}).get(profile_name, {})
            source_profile_data = profile_data

        # --- Data Gathering ---
        
        # Appearance
        appearance = self.user_appearances.get(str(effective_owner_id), {}).get(effective_profile_name, {})
        display_name = appearance.get("custom_display_name") or effective_profile_name
        appearance_text = f"`{effective_profile_name}`" if appearance else "None"
        
        # Stats
        ltm_shard = self._load_ltm_shard(str(effective_owner_id), effective_profile_name)
        ltm_count = len(ltm_shard.get("guild", [])) if ltm_shard else 0
        training_shard = self._load_training_shard(str(effective_owner_id), effective_profile_name)
        training_count = len(training_shard) if training_shard else 0

        # Settings from local profile_data (overrides) or source
        realistic_typing = profile_data.get("realistic_typing_enabled", False)
        time_tracking = profile_data.get("time_tracking_enabled", False)
        timezone_str = profile_data.get("timezone", "UTC")
        
        grounding_mode = profile_data.get("grounding_mode", "off")
        if isinstance(grounding_mode, bool): grounding_mode = "on" if grounding_mode else "off"
        grounding_display = {"off": "`OFF`", "on": "**`ON`**", "on+": "**`ON+`**"}.get(grounding_mode, "OFF")

        stm_length = source_profile_data.get("stm_length", defaultConfig.CHATBOT_MEMORY_LENGTH)
        # Note: borrowed profiles might not have these keys locally, so we default or check source
        ltm_ctx = source_profile_data.get("ltm_context_size", 3)
        ltm_rel = source_profile_data.get("ltm_relevance_threshold", 0.75)
        ltm_scope = profile_data.get("ltm_scope", "server").title()
        safety_level = profile_data.get("safety_level", "low").title()
        ltm_creation_status = "**`ON`**" if profile_data.get("ltm_creation_enabled", False) else "`OFF`"

        created_str = source_profile_data.get('created_at')
        created_display = "Unknown"
        if created_str:
            try:
                dt = datetime.datetime.fromisoformat(created_str)
                ts = int(dt.timestamp())
                created_display = f"<t:{ts}:D>\n(<t:{ts}:R>)"
            except: pass

        # Profile Type Logic
        owner_id_config = int(defaultConfig.DISCORD_OWNER_ID)
        profile_type = "Personal"
        if profile_name == DEFAULT_PROFILE_NAME:
            if owner_id == owner_id_config:
                profile_type = "Personal (Global Default)"
            else:
                profile_type = "Global Default (Borrowed)"
        elif is_borrowed:
            owner_user = self.bot.get_user(effective_owner_id)
            owner_name = owner_user.name if owner_user else "Unknown User"
            profile_type = f"Borrowed (from {owner_name})"

        # --- Embed Construction ---
        embed = discord.Embed(title=f"Participant: {display_name}", color=discord.Color.blue())
        
        embed.add_field(name="Profile Type", value=f"`{profile_type}`", inline=True)
        embed.add_field(name="Created", value=created_display, inline=True)
        embed.add_field(name="Display Name", value=f"`{display_name}`", inline=True)
        embed.add_field(name="Safety Level", value=f"`{safety_level}`", inline=True)

        embed.add_field(name="\u200b", value="**Core Settings**", inline=False)
        embed.add_field(name="Primary Model", value=f"`{prim_model}`", inline=True)
        embed.add_field(name="Fallback Model", value=f"`{fall_model}`", inline=True)
        embed.add_field(name="Appearance", value=appearance_text, inline=True)
        
        embed.add_field(name="Grounding", value=grounding_display, inline=True)
        embed.add_field(name="Realistic Typing", value="**`ON`**" if realistic_typing else "`OFF`", inline=True)
        embed.add_field(name="Timezone", value=f"`{timezone_str}`", inline=True)

        embed.add_field(name="\u200b", value="**Generation Parameters**", inline=False)
        embed.add_field(name="Temperature", value=f"`{temp}`", inline=True)
        embed.add_field(name="Top P", value=f"`{topp}`", inline=True)
        embed.add_field(name="Top K", value=f"`{topk}`", inline=True)
        embed.add_field(name="STM Length", value=f"`{stm_length}`", inline=True)

        embed.add_field(name="\u200b", value="**Training & Memory**", inline=False)
        embed.add_field(name="Train Ctx", value=f"`{training_ctx}`", inline=True)
        embed.add_field(name="Train Rel", value=f"`{training_rel}`", inline=True)
        embed.add_field(name="Train Count", value=f"`{training_count}`", inline=True)
        embed.add_field(name="LTM Ctx", value=f"`{ltm_ctx}`", inline=True)
        embed.add_field(name="LTM Rel", value=f"`{ltm_rel}`", inline=True)
        embed.add_field(name="LTM Info", value=f"Count: `{ltm_count}`\nScope: `{ltm_scope}`\nAuto-Creation: {ltm_creation_status}", inline=True)
        
        if appearance.get("custom_avatar_url"):
            embed.set_thumbnail(url=appearance["custom_avatar_url"])
            
        return embed
    
    async def _build_profile_embed(self, user_id: int, profile_name: str, channel_id: int) -> discord.Embed:
        user_data = self._get_user_data_entry(user_id)
        is_borrowed = profile_name in user_data.get("borrowed_profiles", {})
        
        embed = discord.Embed(title=f"Profile Dashboard: '{profile_name}'", color=discord.Color.blue())
        
        effective_owner_id = user_id
        effective_profile_name = profile_name
        profile_type = "Personal"
        owner_id_config = int(defaultConfig.DISCORD_OWNER_ID)

        if profile_name == DEFAULT_PROFILE_NAME:
            if user_id == owner_id_config:
                profile_type = "Personal (Global Default)"
            else:
                profile_type = "Global Default (Borrowed)"
                effective_owner_id = owner_id_config
                effective_profile_name = DEFAULT_PROFILE_NAME
        elif is_borrowed:
            borrowed_data = user_data["borrowed_profiles"][profile_name]
            effective_owner_id = int(borrowed_data["original_owner_id"])
            effective_profile_name = borrowed_data["original_profile_name"]
            owner_user = self.bot.get_user(effective_owner_id)
            profile_type = f"Borrowed (from {owner_user.name if owner_user else 'Unknown User'})"

        _, _, _, temp, top_p, top_k, train_ctx, train_rel, prim_model, fall_model = self._get_user_profile_for_model(user_id, channel_id, profile_name)

        source_profile_data = user_data.get("borrowed_profiles" if is_borrowed else "profiles", {}).get(profile_name, {})
        
        # --- Data Gathering ---
        ltm_shard = self._load_ltm_shard(str(effective_owner_id), effective_profile_name)
        ltm_count = len(ltm_shard.get("guild", [])) if ltm_shard else 0
        training_shard = self._load_training_shard(str(effective_owner_id), effective_profile_name)
        train_count = len(training_shard) if training_shard else 0

        created_str = source_profile_data.get('created_at')
        created_display = "Unknown"
        if created_str:
            try:
                dt = datetime.datetime.fromisoformat(created_str)
                ts = int(dt.timestamp())
                created_display = f"<t:{ts}:D>"
            except: pass
        
        appearance_data = self.user_appearances.get(str(effective_owner_id), {}).get(effective_profile_name, {})
        display_name = appearance_data.get("custom_display_name") or effective_profile_name
        safety_level = source_profile_data.get("safety_level", "low").title()

        # 1. Top Section
        embed.add_field(name="Profile Type", value=f"`{profile_type}`", inline=True)
        embed.add_field(name="Created", value=created_display, inline=True)
        embed.add_field(name="Display Name", value=f"`{display_name}`", inline=True)
        embed.add_field(name="Safety Level", value=f"`{safety_level}`", inline=False)

        # Invisible Separator
        embed.add_field(name="\u200b", value="\u200b", inline=False)

        embed.add_field(name="Primary Model", value=f"`{prim_model}`", inline=True)
        embed.add_field(name="Fallback Model", value=f"`{fall_model}`", inline=True)

        # Invisible Separator
        embed.add_field(name="\u200b", value="\u200b", inline=False)

        # 2. Generation Parameters Section
        stm_length = source_profile_data.get("stm_length", defaultConfig.CHATBOT_MEMORY_LENGTH)
        gen_val = (
            f"Temp: `{temp}`\n"
            f"Top P: `{top_p}`\n"
            f"Top K: `{top_k}`\n"
            f"STM Length: `{stm_length}`"
        )
        embed.add_field(name="\u200bGeneration Parameters", value=gen_val, inline=True)

        # 3. Advanced Parameters Section
        freq_p = source_profile_data.get("frequency_penalty", 0.0)
        pres_p = source_profile_data.get("presence_penalty", 0.0)
        rep_p = source_profile_data.get("repetition_penalty", 1.0)
        min_p = source_profile_data.get("min_p", 0.0)
        top_a = source_profile_data.get("top_a", 0.0)

        adv_val = (
            f"Freq P: `{freq_p}`\n"
            f"Pres P: `{pres_p}`\n"
            f"Rep P: `{rep_p}`\n"
            f"Min P: `{min_p}`\n"
            f"Top A: `{top_a}`"
        )
        embed.add_field(name="Advanced (OpenRouter Only)", value=adv_val, inline=True)

        # 4. Tools Section
        img_gen = "**`ON`**" if source_profile_data.get("image_generation_enabled", True) else "`OFF`"
        
        raw_ground_mode = source_profile_data.get("grounding_mode", "off")
        if isinstance(raw_ground_mode, bool): raw_ground_mode = "on" if raw_ground_mode else "off"
        grounding_display = {"off": "`OFF`", "on": "**`ON`**", "on+": "**`ON+`**"}.get(raw_ground_mode, "`OFF`")
        
        url_ctx = "**`ON`**" if source_profile_data.get("url_fetching_enabled", True) else "`OFF`"
        timezone = source_profile_data.get("timezone", "UTC")
        typing = "**`ON`**" if source_profile_data.get("realistic_typing_enabled", False) else "`OFF`"
        critic = "**`ON`**" if source_profile_data.get("critic_enabled", False) else "`OFF`"
        resp_mode = source_profile_data.get("response_mode", "regular").replace('_', ' ').title()

        # Invisible Separator to push the next row down
        embed.add_field(name="\u200b", value="\u200b", inline=False)

        # 4. Tools Section (Change to inline)
        tools_val = (
            f"Image Gen: {img_gen}\n"
            f"Grounding: {grounding_display}\n"
            f"URL Context: {url_ctx}\n"
            f"Response Mode: `{resp_mode}`\n"
            f"Timezone: `{timezone}`\n"
            f"Realistic Typing: {typing}\n"
            f"Critic: {critic}"
        )
        embed.add_field(name="Tools", value=tools_val, inline=True)

        # 5. [NEW] Thinking Section (Placed to the right of Tools)
        t_summary = source_profile_data.get("thinking_summary_visible", "off").upper()
        t_level = source_profile_data.get("thinking_level", "high").title()
        t_budget = source_profile_data.get("thinking_budget", -1)
        budget_display = "Dynamic (-1)" if t_budget == -1 else f"{t_budget}"

        thinking_val = (
            f"Summary: **`{t_summary}`**\n"
            f"Effort: `{t_level}`\n"
            f"Budget: `{budget_display}`"
        )
        embed.add_field(name="Thinking/Reasoning", value=thinking_val, inline=True)

        s_voice = source_profile_data.get("speech_voice", "Aoede")
        s_model = source_profile_data.get("speech_model", "gemini-2.5-flash-preview-tts")
        s_temp = source_profile_data.get("speech_temperature", 1.0)
        s_model_disp = s_model.replace("gemini-", "").replace("-preview-tts", "").title()
        
        speech_val = (
            f"Voice: `{s_voice}`\n"
            f"Model: `{s_model_disp}`\n"
            f"Prosody: `{s_temp}`"
        )
        embed.add_field(name="Speech TTS", value=speech_val, inline=True)

        # 5. Training & Memory Section (Same Row)
        train_val = (
            f"Count: `{train_count}`\n"
            f"Context Size: `{train_ctx}`\n"
            f"Relevance Threshold: `{train_rel}`"
        )
        embed.add_field(name="Training Examples", value=train_val, inline=True)

        ltm_ctx = source_profile_data.get("ltm_context_size", 3)
        ltm_rel = source_profile_data.get("ltm_relevance_threshold", 0.75)
        ltm_scope = source_profile_data.get("ltm_scope", "server").title()
        ltm_status = "**`ON`**" if source_profile_data.get("ltm_creation_enabled", False) else "`OFF`"
        ltm_inv = source_profile_data.get("ltm_creation_interval", 10)
        ltm_s_ctx = source_profile_data.get("ltm_summarization_context", 10)

        ltm_val = (
            f"Auto-Creation: {ltm_status}\n"
            f"Count: `{ltm_count}`\n"
            f"Scope: `{ltm_scope}`\n"
            f"Creation Interval: `{ltm_inv}`\n"
            f"Summ Context: `{ltm_s_ctx}`\n"
            f"Context Size: `{ltm_ctx}`\n"
            f"Relevance Threshold: `{ltm_rel}`"
        )
        embed.add_field(name="Long-Term Memories", value=ltm_val, inline=True)

        if appearance_data.get("custom_avatar_url"):
            embed.set_thumbnail(url=appearance_data["custom_avatar_url"])

        return embed
    
    async def _build_profile_manage_embed(self, interaction: discord.Interaction, profile_name: str) -> discord.Embed:
        return await self._build_profile_embed(interaction.user.id, profile_name, interaction.channel_id)
    
    async def _build_server_manage_embed(self, interaction: discord.Interaction) -> discord.Embed:
        guild = interaction.guild
        embed = discord.Embed(title=f"Server Management Dashboard for {guild.name}", color=discord.Color.dark_red())

        # API Key Status
        primary_key_status = "SET" if str(guild.id) in self.server_api_keys else "NOT SET"
        user_keys = self.key_submissions.get(str(guild.id), [])
        active_keys = sum(1 for k in user_keys if k.get("status") == "active")
        pending_keys = sum(1 for k in user_keys if k.get("status") == "pending")
        api_key_value = f"**Primary Key:** {primary_key_status}\n**User Key Pool:** {active_keys} Active / {pending_keys} Pending"
        embed.add_field(name="🔑 API Key Status", value=api_key_value, inline=False)

        # Global Feature Toggles
        grounding_status = "**ON**" if self.server_grounding_settings.get(guild.id, False) else "OFF"
        ltm_status = "**ON**" if self.server_ltm_autocreation_enabled.get(guild.id, False) else "OFF"
        url_status = "**ON**" if self.server_url_fetching_enabled.get(guild.id, False) else "OFF"
        sharing_status = "**ON**" if self.server_sharing_enabled.get(guild.id, False) else "OFF"
        image_gen_status = "**ON**" if self.server_image_generation_enabled.get(guild.id, False) else "OFF"
        toggles_value = (
            f"**Grounding (Web Search):** {grounding_status}\n"
            f"**LTM Auto-creation:** {ltm_status}\n"
            f"**URL Content Fetching:** {url_status}\n"
            f"**Profile Sharing:** {sharing_status}\n"
            f"**Image Generation (!image):** {image_gen_status}"
        )
        embed.add_field(name="⚙️ Global Feature Toggles", value=toggles_value, inline=False)

        # Model & Content Policy
        allowed_models = self.server_allowed_models.get(guild.id, [])
        if not allowed_models:
            model_list_status = "Default Safe Models Active"
        else:
            model_list_status = f"Custom List Set ({len(allowed_models)} models)\n`{'`, `'.join(allowed_models[:4])}`"
            if len(allowed_models) > 4:
                model_list_status += "..."
        embed.add_field(name="🤖 Model & Content Policy", value=model_list_status, inline=False)

        # Freewill System
        fw_config = self.freewill_config.get(str(guild.id), {})
        fw_living = len(fw_config.get("living_channel_ids", []))
        fw_lurking = len(fw_config.get("lurking_channel_ids", []))
        fw_value = f"**Active Channels:** {fw_living} Living / {fw_lurking} Lurking"
        embed.add_field(name="🎭 Roleplay & Automation", value=fw_value, inline=False)
        
        # Current Channel Defaults
        channel = interaction.channel
        activation_status = "**Active**" if channel.id in self.active_channels else "Inactive (Mention-Only)"
        response_mode = self.channel_response_modes.get(channel.id, 'regular').replace('_', ' ').title()
        scope_setting = self.channel_scoped_profiles.get(channel.id)
        scope_map = {"user": "Unlocked", "channel": "Locked", "multi": "Locked (Multi)"}
        if not scope_setting:
            scope_status = "Unlocked"
        else:
            scope_status = scope_map.get(scope_setting.get("type"), "Unlocked")
        safety_floor = self.channel_safety_floor.get(channel.id, 'low').title()
        channel_value = (
            f"**Bot Activation:** {activation_status}\n"
            f"**Response Mode:** {response_mode}\n"
            f"**Profile Scope:** {scope_status}\n"
            f"**Safety Floor:** {safety_floor}"
        )
        embed.add_field(name=f"📜 Current Channel Settings (#{channel.name})", value=channel_value, inline=False)
        
        embed.set_footer(text="Select an action from the dropdowns below to manage the server.")
        return embed
    
    async def profile_autocomplete(self, interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
        user_id_str = str(interaction.user.id)
        user_data = self._get_user_data_entry(interaction.user.id)
        
        choices = []
        # Personal Profiles
        personal_profiles = user_data.get("profiles", {}).keys()
        for p in personal_profiles:
            if current.lower() in p.lower():
                choices.append(app_commands.Choice(name=p, value=p))

        # Borrowed Profiles
        borrowed_profiles = user_data.get("borrowed_profiles", {})
        for name, data in borrowed_profiles.items():
            if current.lower() in name.lower():
                owner_id = int(data["original_owner_id"])
                owner = self.bot.get_user(owner_id) or await self.bot.fetch_user(owner_id)
                owner_name = owner.display_name if owner else "Unknown User"
                choices.append(app_commands.Choice(name=f"{name} (from {owner_name})", value=name))

        return choices[:25]
    
    async def global_chat_profile_autocomplete(self, interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
        user_id = interaction.user.id
        user_data = self._get_user_data_entry(user_id)
        
        # Build set of public pointers for O(1) lookup
        public_pointers = set()
        for p_info in self.public_profiles.values():
            oid = str(p_info.get("owner_id"))
            oname = p_info.get("original_profile_name")
            if oid and oname:
                public_pointers.add((oid, oname))
        
        choices = []
        current_lower = current.lower()

        # Personal Public Profiles
        for name in user_data.get("profiles", {}):
            if (str(user_id), name) in public_pointers:
                if current_lower in name.lower():
                    choices.append(app_commands.Choice(name=name, value=name))
        
        # Borrowed Public Profiles
        for local_name, data in user_data.get("borrowed_profiles", {}).items():
            orig_oid = str(data.get("original_owner_id"))
            orig_name = data.get("original_profile_name")
            
            if (orig_oid, orig_name) in public_pointers:
                if current_lower in local_name.lower():
                    owner = self.bot.get_user(int(orig_oid))
                    owner_name = owner.name if owner else "Unknown"
                    display = f"{local_name} (from {owner_name})"
                    choices.append(app_commands.Choice(name=display, value=local_name))
        
        return choices[:25]

    async def personal_profile_autocomplete(self, interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
        user_id_str = str(interaction.user.id)
        user_data = self._get_user_data_entry(interaction.user.id)
        
        choices = []
        personal_profiles = user_data.get("profiles", {}).keys()
        for p in personal_profiles:
            if current.lower() in p.lower():
                choices.append(app_commands.Choice(name=p, value=p))
        return choices[:25]

    async def appearance_autocomplete(self, interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
        user_id_str = str(interaction.user.id)
        if user_id_str not in self.user_appearances:
            return []
        
        appearances = self.user_appearances[user_id_str].keys()
        return [
            app_commands.Choice(name=app, value=app)
            for app in appearances if current.lower() in app.lower()
        ][:25]
    
    async def timezone_autocomplete(self, interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
        all_tzs = available_timezones()
        if not current:
            suggestions = ["UTC", "US/Eastern", "US/Central", "US/Mountain", "US/Pacific", "Europe/London", "Europe/Berlin"]
            return [app_commands.Choice(name=tz, value=tz) for tz in suggestions]

        current_lower = current.lower()
        return [
            app_commands.Choice(name=tz, value=tz)
            for tz in all_tzs if current_lower in tz.lower()
        ][:25]

    async def speak_method_autocomplete(self, interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
        choices = [
            app_commands.Choice(name="Auto (Recommended)", value="auto"),
            app_commands.Choice(name="Webhook", value="webhook")
        ]
        
        profile_name = interaction.namespace.profile_name
        if profile_name and interaction.guild:
            user_data = self._get_user_data_entry(interaction.user.id)
            is_borrowed = profile_name in user_data.get("borrowed_profiles", {})
            is_personal = profile_name in user_data.get("profiles", {})

            effective_owner_id = interaction.user.id
            effective_profile_name = profile_name

            if is_borrowed:
                borrowed_data = user_data["borrowed_profiles"][profile_name]
                effective_owner_id = int(borrowed_data["original_owner_id"])
                effective_profile_name = borrowed_data["original_profile_name"]
            
            if is_personal or is_borrowed:
                linked_bot_id = next((bot_id for bot_id, data in self.child_bots.items() if data.get("owner_id") == effective_owner_id and data.get("profile_name") == effective_profile_name), None)
                if linked_bot_id and interaction.guild.get_member(int(linked_bot_id)):
                    choices.append(app_commands.Choice(name="Child Bot", value="child_bot"))

        return [choice for choice in choices if current.lower() in choice.name.lower()]
    
    async def bulk_apply_generation_params(self, user_id: int, profile_names: List[str], params: Dict) -> str:
        user_data = self._get_user_data_entry(user_id)
        updated_count = 0
        for name in profile_names:
            # This action is only valid for personal profiles.
            if name in user_data.get("profiles", {}):
                profile = user_data["profiles"][name]
                profile.update(params)
                updated_count += 1
        
        if updated_count > 0:
            self._save_user_data_entry(user_id, user_data)
        
        return f"Updated generation parameters for {updated_count} personal profile(s)."

    async def bulk_apply_thinking_params(self, user_id: int, profile_names: List[str], params: Dict) -> str:
        user_data = self._get_user_data_entry(user_id)
        updated_count = 0
        for name in profile_names:
            # Check both personal and borrowed profiles
            profile = user_data.get("profiles", {}).get(name) or user_data.get("borrowed_profiles", {}).get(name)
            if profile:
                profile.update(params)
                updated_count += 1
        
        if updated_count > 0:
            self._save_user_data_entry(user_id, user_data)
            
            # Invalidate caches for this user
            keys_to_clear = [k for k in self.channel_models.keys() if isinstance(k, tuple) and len(k) == 3 and k[1] == user_id]
            for k in keys_to_clear:
                self.channel_models.pop(k, None)
                self.chat_sessions.pop(k, None)
                self.channel_model_last_profile_key.pop(k, None)
        
        return f"Updated thinking parameters for {updated_count} profile(s)."

    async def bulk_apply_training_params(self, user_id: int, profile_names: List[str], params: Dict) -> str:
        user_data = self._get_user_data_entry(user_id)
        updated_count = 0
        for name in profile_names:
            profile = user_data.get("profiles", {}).get(name)
            if profile:
                profile.update(params)
                updated_count += 1
        
        if updated_count > 0:
            self._save_user_data_entry(user_id, user_data)
        
        return f"Updated training parameters for {updated_count} personal profile(s)."

    async def bulk_apply_ltm_params(self, user_id: int, profile_names: List[str], params: Dict) -> str:
        user_data = self._get_user_data_entry(user_id)
        updated_count = 0
        for name in profile_names:
            profile = user_data.get("profiles", {}).get(name)
            if profile:
                profile.update(params)
                updated_count += 1
        
        if updated_count > 0:
            self._save_user_data_entry(user_id, user_data)
        
        return f"Updated LTM parameters for {updated_count} personal profile(s)."

    async def bulk_apply_ltm_summarization_instructions(self, user_id: int, profile_names: List[str], params: Dict) -> str:
        user_data = self._get_user_data_entry(user_id)
        updated_count = 0
        instructions = params.get("ltm_summarization_instructions")
        if not instructions:
            return "Error: No instructions provided."

        encrypted_instructions = self._encrypt_data(instructions)
        for name in profile_names:
            profile = user_data.get("profiles", {}).get(name)
            if profile:
                profile["ltm_summarization_instructions"] = encrypted_instructions
                updated_count += 1
        
        if updated_count > 0:
            self._save_user_data_entry(user_id, user_data)
        
        return f"Updated LTM summarization instructions for {updated_count} personal profile(s)."

    async def bulk_toggle_grounding(self, user_id: int, profile_names: List[str], enable: bool) -> str:
        user_data = self._get_user_data_entry(user_id)
        updated_count = 0
        for name in profile_names:
            profile = user_data.get("profiles", {}).get(name)
            if profile:
                profile["grounding_enabled"] = enable
                updated_count += 1
        
        if updated_count > 0:
            self._save_user_data_entry(user_id, user_data)
        
        status = "ENABLED" if enable else "DISABLED"
        return f"Grounding has been set to **{status}** for {updated_count} personal profile(s)."

    async def bulk_reset_examples(self, user_id: int, profile_names: List[str]) -> str:
        user_id_str = str(user_id)
        reset_count = 0
        for name in profile_names:
            file_path = os.path.join(self.TRAINING_DIR, user_id_str, f"{name}.json.gz")
            if os.path.exists(file_path):
                self._delete_training_shard(user_id_str, name)
                reset_count += 1
        
        return f"Reset all training examples for {reset_count} profile(s)."

    async def bulk_reset_ltm(self, user_id: int, profile_names: List[str]) -> str:
        user_id_str = str(user_id)
        reset_count = 0
        for name in profile_names:
            file_path = os.path.join(self.LTM_DIR, user_id_str, f"{name}.json.gz")
            if os.path.exists(file_path):
                self._delete_ltm_shard(user_id_str, name)
                reset_count += 1
        
        return f"Reset all Long-Term Memories for {reset_count} profile(s)."
    
    async def update_profile_advanced_params(self, user_id: int, profile_name: str, params: Dict[str, Any], channel_id_context: int, is_borrowed: bool) -> bool:
        if not self.has_lock: return False
        user_data = self._get_user_data_entry(user_id)
        
        target = user_data.get("borrowed_profiles", {}).get(profile_name) if is_borrowed else user_data.get("profiles", {}).get(profile_name)
        if not target: return False

        for k, v in params.items():
            if v is None:
                if k in target: del target[k] # Reset to default (remove override)
            else:
                target[k] = v
        
        self._save_user_data_entry(user_id, user_data)
        
        key = (channel_id_context, user_id)
        
        if self._get_active_user_profile_name_for_channel(user_id, channel_id_context) == profile_name:
            if key in self.channel_models: del self.channel_models[key]
            if key in self.chat_sessions: self.chat_sessions.pop(key, None)
            self.channel_model_last_profile_key.pop(key, None)
        
        return True
    
    async def update_user_profile_persona(self, user_id: int, profile_name: str, persona_data: Dict[str, List[str]], channel_id_context: int) -> bool:
        if not self.has_lock: return False
        profile = self._get_or_create_user_profile(user_id, profile_name)
        if not profile: return False 
        
        profile["persona"] = persona_data
        self._save_user_data_entry(user_id, self._get_user_data_entry(user_id))
        
        active_profile_for_channel = self._get_active_user_profile_name_for_channel(user_id, channel_id_context)
        if active_profile_for_channel == profile_name:
            # The cache key for a user's interaction is always (channel_id, user_id)
            # regardless of channel scope.
            model_cache_key = (channel_id_context, user_id)
            if model_cache_key in self.channel_models: del self.channel_models[model_cache_key]
            if model_cache_key in self.chat_sessions: self.chat_sessions.pop(model_cache_key, None)
            self.channel_model_last_profile_key.pop(model_cache_key, None)
        return True

    async def update_user_profile_ai_instructions(self, user_id: int, profile_name: str, instructions: str, channel_id_context: int) -> bool:
        if not self.has_lock: return False
        profile = self._get_or_create_user_profile(user_id, profile_name)
        if not profile: return False
        
        profile["ai_instructions"] = instructions
        self._save_user_data_entry(user_id, self._get_user_data_entry(user_id))

        active_profile_for_channel = self._get_active_user_profile_name_for_channel(user_id, channel_id_context)
        if active_profile_for_channel == profile_name:
            # The cache key for a user's interaction is always (channel_id, user_id)
            # regardless of channel scope.
            model_cache_key = (channel_id_context, user_id)
            if model_cache_key in self.channel_models: del self.channel_models[model_cache_key]
            if model_cache_key in self.chat_sessions: self.chat_sessions.pop(model_cache_key, None)
            self.channel_model_last_profile_key.pop(model_cache_key, None)
        return True
        
    async def update_profile_generation_params(self, user_id: int, profile_name: str, params: Dict[str, Any], channel_id_context: int, is_borrowed: bool) -> bool:
        if not self.has_lock: return False
        
        user_data = self._get_user_data_entry(user_id)
        
        if is_borrowed:
            profile = user_data.get("borrowed_profiles", {}).get(profile_name)
        else:
            profile = user_data.get("profiles", {}).get(profile_name)

        if not profile: return False

        if "temperature" in params: profile["temperature"] = params["temperature"]
        if "top_p" in params: profile["top_p"] = params["top_p"]
        if "top_k" in params: profile["top_k"] = params["top_k"]
        if "stm_length" in params: profile["stm_length"] = params["stm_length"]
        
        self._save_user_data_entry(user_id, user_data)

        model_cache_key = (channel_id_context, user_id)
        
        active_profile_for_channel = self._get_active_user_profile_name_for_channel(user_id, channel_id_context)
        if active_profile_for_channel == profile_name:
            if model_cache_key in self.channel_models: del self.channel_models[model_cache_key]
            if model_cache_key in self.chat_sessions: self.chat_sessions.pop(model_cache_key, None)
            self.channel_model_last_profile_key.pop(model_cache_key, None)
        return True

    async def update_profile_training_params(self, user_id: int, profile_name: str, params: Dict[str, Any]) -> bool:
        if not self.has_lock: return False
        profile = self._get_or_create_user_profile(user_id, profile_name)
        if not profile: return False

        if "training_context_size" in params: profile["training_context_size"] = params["training_context_size"]
        if "training_relevance_threshold" in params: profile["training_relevance_threshold"] = params["training_relevance_threshold"]
        self._save_user_data_entry(user_id, self._get_user_data_entry(user_id))
        return True
    
    async def update_profile_ltm_params(self, user_id: int, profile_name: str, params: Dict[str, Any]) -> bool:
        if not self.has_lock: return False
        profile = self._get_or_create_user_profile(user_id, profile_name)
        if not profile: return False

        if "ltm_context_size" in params: profile["ltm_context_size"] = params["ltm_context_size"]
        if "ltm_relevance_threshold" in params: profile["ltm_relevance_threshold"] = params["ltm_relevance_threshold"]
        self._save_user_data_entry(user_id, self._get_user_data_entry(user_id))
        return True
    
    async def update_profile_models(self, user_id: int, profile_name: str, primary_model: Optional[str], fallback_model: Optional[str], is_borrowed: bool, channel_id_context: int, show_fallback_indicator: Optional[bool] = None) -> bool:
        if not self.has_lock: return False
        
        user_data = self._get_user_data_entry(user_id)
        
        if is_borrowed:
            profile = user_data.get("borrowed_profiles", {}).get(profile_name)
        else:
            profile = user_data.get("profiles", {}).get(profile_name)

        if not profile: return False

        if primary_model: profile["primary_model"] = primary_model
        if fallback_model: profile["fallback_model"] = fallback_model
        if show_fallback_indicator is not None: profile["show_fallback_indicator"] = show_fallback_indicator
        
        self._save_user_data_entry(user_id, user_data)

        keys_to_delete = []
        for key in list(self.channel_models.keys()):
            key_user_id = None
            if isinstance(key, tuple) and len(key) == 2:
                key_user_id = key[1]
            
            if key_user_id == user_id:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            if key in self.channel_models: del self.channel_models[key]
            if key in self.chat_sessions: self.chat_sessions.pop(key, None)
            self.channel_model_last_profile_key.pop(key, None)

        return True
    
    async def _has_api_key_access(self, user_id: int) -> bool:
        # 1. Check for personal key
        if self._get_api_key_for_user(user_id, "gemini") or self._get_api_key_for_user(user_id, "openrouter"):
            return True
        
        # 2. Check mutual guilds for a server key
        user = self.bot.get_user(user_id)
        if not user: return False

        for guild in user.mutual_guilds:
            if self._get_api_key_for_guild(guild.id, "gemini") or self._get_api_key_for_guild(guild.id, "openrouter"):
                return True
        
        return False
    
    async def _accept_share_request(self, interaction: discord.Interaction, sharer_id: int, profile_name: str, desired_name: str, is_public_borrow: bool = False):
        owner_user_data = self._get_user_data_entry(sharer_id)
        owner_profile_data = owner_user_data.get("profiles", {}).get(profile_name)
        if not owner_profile_data:
            await interaction.followup.send("The shared profile seems to no longer exist.", ephemeral=True)
            return

        # [NEW] Dynamic Limit Check
        user_data = self._get_user_data_entry(interaction.user.id)
        current_borrowed = len(user_data.get("borrowed_profiles", {}))
        
        limit = defaultConfig.LIMIT_BORROWED_PREMIUM if self.is_user_premium(interaction.user.id) else defaultConfig.LIMIT_BORROWED_FREE

        if current_borrowed >= limit:
            tier_name = "Premium" if self.is_user_premium(interaction.user.id) else "Free"
            await interaction.followup.send(f"Limit Reached. You have {current_borrowed}/{limit} borrowed profiles ({tier_name} Tier).", ephemeral=True)
            return

        snapshot_data = {
            "original_owner_id": str(sharer_id),
            "original_profile_name": profile_name,
            "borrowed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "grounding_enabled": owner_profile_data.get("grounding_enabled", False),
            "realistic_typing_enabled": owner_profile_data.get("realistic_typing_enabled", False),
            "time_tracking_enabled": owner_profile_data.get("time_tracking_enabled", False),
            "timezone": owner_profile_data.get("timezone", "UTC"),
            "ltm_creation_enabled": False, # Borrowers start with LTM creation off
            "ltm_scope": "server", # Borrowers start with the safest default
            "safety_level": "low",
            "thinking_summary_visible": "off",
            "thinking_level": "high",
            "thinking_budget": -1
        }
        
        user_data = self._get_user_data_entry(interaction.user.id)
        user_data.setdefault("borrowed_profiles", {})[desired_name] = snapshot_data
        self._save_user_data_entry(interaction.user.id, user_data)
        
        if not is_public_borrow:
            await self._reject_share_request(interaction, sharer_id, profile_name, notify_sharer=True, accepted=True)

    async def _reject_share_request(self, interaction: discord.Interaction, sharer_id: int, profile_name: str, notify_sharer: bool = True, accepted: bool = False):
        recipient_id_str = str(interaction.user.id)
        if recipient_id_str in self.profile_shares:
            updated_shares = [s for s in self.profile_shares[recipient_id_str] if not (s['sharer_id'] == sharer_id and s['profile_name'] == profile_name)]
            if not updated_shares:
                del self.profile_shares[recipient_id_str]
            else:
                self.profile_shares[recipient_id_str] = updated_shares
            self._save_profile_share_shard(recipient_id_str, updated_shares)

        if notify_sharer:
            sharer = self.bot.get_user(sharer_id)
            if sharer:
                status = "accepted" if accepted else "rejected"
                try:
                    await sharer.send(f"Your share request for '{profile_name}' to **{interaction.user.name}** was **{status}**.")
                except discord.Forbidden:
                    pass

    async def _validate_active_profile(self, user_id: int, channel: discord.abc.Messageable) -> bool:
        user_data = self._get_user_data_entry(user_id)
        active_profile_name = self._get_active_user_profile_name_for_channel(user_id, channel.id)

        if active_profile_name in user_data.get("borrowed_profiles", {}):
            borrowed_data = user_data["borrowed_profiles"][active_profile_name]
            owner_id = int(borrowed_data["original_owner_id"])
            owner_profile_name = borrowed_data["original_profile_name"]
            
            owner_user_data = self._get_user_data_entry(owner_id)
            if not owner_user_data.get("profiles", {}).get(owner_profile_name):
                del user_data["borrowed_profiles"][active_profile_name]
                user_data["channel_active_profiles"][str(channel.id)] = DEFAULT_PROFILE_NAME
                self._save_user_data_entry(user_id, user_data)
                
                try:
                    await channel.send(f"<@{user_id}>, the borrowed profile '{active_profile_name}' is broken because the original was deleted or renamed. It has been removed from your list and your active profile in this channel has been reset to '{DEFAULT_PROFILE_NAME}'.")
                except discord.Forbidden:
                    pass
                return False
        return True
        

    async def _resolve_reply_context(self, message: discord.Message) -> Optional[str]:
        if not message.reference or not message.reference.message_id:
            return None
        
        try:
            referenced_message = await message.channel.fetch_message(message.reference.message_id)
            author_name = referenced_message.author.display_name
            content = referenced_message.clean_content
            if len(content) > 150:
                content = content[:150] + "..."
            return f"[Replying to {author_name}: '{content}']"
        except (discord.NotFound, discord.Forbidden):
            return "[Replying to a message that could not be loaded]"
        except Exception as e:
            print(f"Error resolving reply context: {e}")
            return None
        
    def _safe_cancel_task(self, task: asyncio.Task):
        if task and not task.done():
            task.cancel()
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    def _is_history_effectively_empty(self, history: list) -> bool:
        # A session is effectively empty if it contains NO model turns.
        # System notes and director prompts are injected as 'user' turns.
        # Real conversation requires a 'model' response. If none exist, no real conversation is left.
        for turn in history:
            if turn.role == 'model':
                return False
        return True

    def _truncate_text_by_char(self, text: str, max_chars: int) -> str:
        if len(text) > max_chars:
            return text[:max_chars]
        return text
    
    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return interaction.user.id not in self.global_blacklist

    async def cog_app_command_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError):
        if isinstance(error, app_commands.CommandOnCooldown):
            seconds_total = int(error.retry_after)
            if seconds_total >= 60:
                minutes = seconds_total // 60
                seconds = seconds_total % 60
                await interaction.response.send_message(f"This command is on cooldown. Please try again in {minutes} minute(s) and {seconds} second(s).", ephemeral=True)
            else:
                await interaction.response.send_message(f"This command is on cooldown. Please try again in {seconds_total} second(s).", ephemeral=True)
        elif isinstance(error, app_commands.CheckFailure):
            # The interaction_check for a blacklisted user will fail silently.
            # This part handles other permission checks (like is_admin_or_owner_check) by sending a message.
            if interaction.user.id not in self.global_blacklist:
                await interaction.response.send_message("You do not have the required permissions (e.g., Server Administrator) to use this command.", ephemeral=True)
            pass
        else:
            error_id = str(uuid.uuid4())[:8]
            print(f"Unhandled command error (ID: {error_id}): {error}")
            traceback.print_exc()
            
            # [FIX] Robust response logic to handle expired/dead interactions
            try:
                msg = f"An unexpected error occurred. Please report this to the bot owner with the following ID: `{error_id}`"
                if not interaction.response.is_done():
                    await interaction.response.send_message(msg, ephemeral=True)
                else:
                    await interaction.followup.send(msg, ephemeral=True)
            except (discord.NotFound, discord.HTTPException):
                # Interaction is completely dead, nothing more we can do
                pass

    @tasks.loop(seconds=LOCK_REFRESH_INTERVAL_SECONDS)
    async def refresh_lock_task(self):
        if self.has_lock:
            try:
                with open(COG_LOCK_FILE_PATH, "w") as f:
                    f.write(str(time.time()))
            except IOError as e:
                print(f"IOError refreshing lock file: {e}. Potential lock loss.")
                self.has_lock = False 
            except Exception as e:
                print(f"Unexpected error refreshing lock file: {e}. Potential lock loss.")
                self.has_lock = False

    def cog_unload(self):
        if self.has_lock:
            try:
                if os.path.exists(COG_LOCK_FILE_PATH):
                    os.remove(COG_LOCK_FILE_PATH)
            except OSError as e:
                print(f"OSError releasing lock file: {e}")
            except Exception as e:
                print(f"Unexpected error releasing lock file: {e}")
        self.refresh_lock_task.cancel()
        self.evict_inactive_sessions_task.cancel()
        self.weekly_cleanup_task.cancel()
        self.child_bot_integrity_task.cancel()

        if self.image_finisher_worker_task:
            self.image_finisher_worker_task.cancel()
        for worker in self.image_gen_workers:
            worker.cancel()
        
        # [FIX] Explicitly cancel all active session worker tasks to prevent "Task pending" warnings
        for session_data in self.multi_profile_channels.values():
            if session_data.get('worker_task'):
                self._safe_cancel_task(session_data['worker_task'])
                session_data['worker_task'] = None
        
        # Final flush of any remaining data
        asyncio.run_coroutine_threadsafe(self._flush_api_stats_to_db(), self.bot.loop)
        for session_key, chat_session in self.chat_sessions.items():
            self._save_session_to_disk(session_key, 'single', chat_session)
        for session_key, chat_session in self.global_chat_sessions.items():
            self._save_session_to_disk(session_key, 'global_chat', chat_session)
        for ch_id, session_data in self.multi_profile_channels.items():
            if session_data.get("is_hydrated"): # Only save sessions that are loaded in memory
                session_type = session_data.get("type", "multi")
                unified_log = session_data.get("unified_log")
                if unified_log is not None:
                    # For multi-profile, the session_key is just the channel_id for path generation
                    dummy_session_key = (ch_id, None, None)
                    self._save_session_to_disk(dummy_session_key, session_type, unified_log)
                
                # Also save the corresponding mapping file atomically with the session log
                mapping_key = (session_type, ch_id)
                if mapping_key in self.mapping_caches:
                    self._save_mapping_to_disk(mapping_key, self.mapping_caches[mapping_key])

        # Save any remaining mappings that are not for multi-profile sessions
        for mapping_key, mapping_data in self.mapping_caches.items():
            if mapping_key[0] not in ['multi', 'freewill']:
                self._save_mapping_to_disk(mapping_key, mapping_data)