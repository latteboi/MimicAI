import os
import re
import time
import uuid
import random
import asyncio
import traceback
from typing import List

import discord
from discord import app_commands
from discord.ext import commands

from ..utils.constants import *
from ..utils.helpers import _format_history_entry


class EventListeners:
    """Owns Discord gateway event handlers and Cog-framework lifecycle hooks (on_message, reaction
    events, interaction_check, cog_app_command_error, cog_unload) plus the shared autocomplete callback.

    Inherited directly by MimicCog (rather than composed) because discord.py's Cog listener
    registration and command-binding machinery walks the Cog's own MRO to find these methods.
    """

    @commands.Cog.listener()
    async def on_ready(self):
        if not self.sessions_loaded:
            await self.session_manager._load_multi_profile_sessions()
            self.sessions_loaded = True
        
        self.all_bot_ids = {self.bot.user.id} | {int(bot_id) for bot_id in self.child_bots.keys()}

        if self.has_lock:
            self.profile_manager._get_or_create_system_profile("mimicguide")
            self.bot.loop.create_task(self.help_service._load_and_embed_docs())
            
            presence = self.server_manager._load_parent_presence()
            status_val = presence.get("status", "online")
            status_map = {"online": discord.Status.online, "idle": discord.Status.idle, "dnd": discord.Status.dnd, "invisible": discord.Status.invisible}
            activity = self.server_manager._build_activity_from_dict(presence)
            await self.bot.change_presence(status=status_map.get(status_val, discord.Status.online), activity=activity)
            
            self.storage_manager._purge_legacy_default_profile()
            
            print("Running initial index.json self-repair on boot...")
            await asyncio.to_thread(self.profile_manager._repair_all_user_indices)
            print("Initial index.json self-repair complete.")
            
            await asyncio.to_thread(self.storage_manager._migrate_embeddings_to_b64)
            
            if not self.profile_manager.hourly_self_repair_task.is_running():
                self.profile_manager.hourly_self_repair_task.start()
            
            if not self.session_manager.proactive_session_task.is_running():
                self.session_manager.proactive_session_task.start()
            
            if not self.api_service.pricing_sync_task.is_running():
                self.api_service.pricing_sync_task.start()
            
            if not self.storage_manager.daily_cleanup_task.is_running():
                print("Performing initial data cleanup on boot...")
                await self.storage_manager._perform_data_cleanup()
                print("Initial cleanup finished. Starting daily cleanup task.")
                self.storage_manager.daily_cleanup_task.start()

        if self.has_lock and not self.image_finisher_worker_task:
            self.image_finisher_worker_task = self.bot.loop.create_task(self.media_service._image_finisher_worker())
            for i in range(2):
                worker = self.bot.loop.create_task(self.media_service._image_gen_worker(i))
                self.image_gen_workers.append(worker)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.id in self.processed_child_messages:
            return

        if message.author.id in self.global_blacklist:
            return
        
        if not message.guild or not self.has_lock or message.author.bot:
            return

        # Validate the user's active profile in this channel first to prune dangling references
        await self.profile_manager._validate_active_profile(message.author.id, message.channel)

        # --- 1. Session Reply Logic (Priority) ---
        if message.reference and message.reference.message_id:
            # Check if this message is a reply to one of our bots
            if message.reference.resolved and isinstance(message.reference.resolved, discord.Message) and message.reference.resolved.author.id in self.all_bot_ids:
                if message.reference.resolved.author.id != self.bot.user.id and message.channel.id not in self.multi_profile_channels:
                    return

            session = self.multi_profile_channels.get(message.channel.id)
            if session:
                session_type = session.get("type", "multi")
                if not session.get("is_hydrated"):
                    session = await self.session_manager._ensure_session_hydrated(message.channel.id, session_type)
                
                if session:
                    turn_object = next((turn for turn in session.get("unified_log", []) if message.reference.message_id in turn.get("message_ids", [])), None)
                    
                    if turn_object and turn_object.get("is_user") is False:
                        bot_pid = turn_object.get("speaker_pid")
                        o_id = turn_object.get("owner_id")
                        p_name = turn_object.get("profile_name")
                        
                        replied_to_participant = next((p for p in session.get('profiles', []) if self.profile_manager._get_pid_from_name_any(p['owner_id'], p['profile_name']) == bot_pid), None)
                        
                        if not replied_to_participant and o_id and p_name:
                            # Reconstruct ephemeral participant
                            method = 'webhook'
                            bot_id = None
                            linked_bot_id = self.child_bots_by_owner_profile.get((o_id, p_name))
                            if linked_bot_id and message.guild and message.guild.get_member(int(linked_bot_id)):
                                method = 'child_bot'
                                bot_id = linked_bot_id
                            
                            replied_to_participant = {
                                "owner_id": o_id, "profile_name": p_name,
                                "method": method, "bot_id": bot_id, "ephemeral": True
                            }

                        if replied_to_participant:
                            reply_trigger = ('reply', message, replied_to_participant)
                            await session['task_queue'].put(reply_trigger)
                            
                            if not session.get('worker_task') or session['worker_task'].done():
                                task = self.bot.loop.create_task(self.generation_service._multi_profile_worker(message.channel.id))
                                session['worker_task'] = task
                                self.background_tasks.add(task)
                            return

        # --- 2. Standalone Child Bot Detection ---
        mentioned_child_ids = []
        
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
            if not self.storage_manager._get_api_key_for_guild(message.guild.id):
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
                bot_config = self.child_bots.get(bot_id)
                if bot_config:
                    p_index = self.profile_manager._get_user_index(bot_config['owner_id'])
                    p_is_b = bot_config['profile_name'] in p_index.get("borrowed", [])
                    p_settings = self.profile_manager._get_profile_config(bot_config['owner_id'], bot_config['profile_name'], p_is_b) or {}
                    if not p_settings.get("child_bot_placeholder", False):
                        asyncio.create_task(self.manager_queue.put({
                            "action": "send_to_child", "bot_id": bot_id,
                            "payload": {"action": "start_typing", "channel_id": message.channel.id}
                        }))

            content_lower = message.content.lower()
            image_prefixes = ("!image", "!imagine")
            is_image_request = content_lower.startswith(image_prefixes)

            attachments_data = [{"url": a.url, "filename": a.filename, "content_type": a.content_type} for a in message.attachments if a.content_type and (a.content_type.startswith("image/") or a.content_type.startswith("audio/") or a.content_type.startswith("video/") or a.content_type.startswith("text/") or a.filename.lower().endswith(('.txt', '.log', '.md', '.csv', '.json', '.py', '.js', '.html', '.css', '.xml')))]
            
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
                    await self.child_bot_manager.handle_child_bot_event(event_data)
                else:
                    if is_image_request: await self.child_bot_manager.handle_child_bot_image_request(event_data)
                    else:
                        event_data["event_type"] = "message_received"
                        await self.child_bot_manager.handle_child_bot_event(event_data)
            return

        # --- 3. Normal Session Triggering ---
        session = self.multi_profile_channels.get(message.channel.id)

        if session:
            await session['task_queue'].put(message)
            if not session.get('worker_task') or session['worker_task'].done():
                task = self.bot.loop.create_task(self.generation_service._multi_profile_worker(message.channel.id))
                session['worker_task'] = task
                self.background_tasks.add(task)
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

        channel_id = payload.channel_id
        session = self.multi_profile_channels.get(channel_id)
        if not session: return
        
        session_type = session.get("type", "multi")
        if not session.get("is_hydrated"):
            session = await self.session_manager._ensure_session_hydrated(channel_id, session_type)
        if not session: return

        self.session_last_accessed[channel_id] = time.time()

        turn_index = -1
        turn_object = None
        for i, turn in enumerate(session.get("unified_log", [])):
            if payload.message_id in turn.get("message_ids", []):
                turn_index = i
                turn_object = turn
                break

        if turn_object:
            turn_id_to_find = turn_object["turn_id"]
            if is_mute:
                turn_object["is_hidden"] = True
                await self.session_manager._save_session_to_disk((channel_id, None, None), session_type, session["unified_log"])
                session["is_hydrated"] = False
                await self.session_manager._ensure_session_hydrated(channel_id, session_type)
                try:
                    channel = self.bot.get_channel(payload.channel_id)
                    msg = await channel.fetch_message(payload.message_id)
                    await msg.add_reaction(payload.emoji)
                except: pass
                return

            if turn_object.get("is_user") is True:
                return # Can't skip/regen a user turn

            speaker_pid = turn_object.get("speaker_pid")
            
            reacted_to_participant = None
            for participant in session['profiles']:
                if self.profile_manager._get_pid_from_name_any(participant['owner_id'], participant['profile_name']) == speaker_pid:
                    reacted_to_participant = participant
                    break
            
            if reacted_to_participant:
                if is_regen:
                    is_busy = session.get('is_running') or session.get('is_regenerating') or session.get('is_purging')
                    msg_ref = None
                    try:
                        channel = self.bot.get_channel(payload.channel_id)
                        if channel:
                            msg_ref = await channel.fetch_message(payload.message_id)
                            if is_busy:
                                await msg_ref.add_reaction(payload.emoji)
                            else:
                                await msg_ref.clear_reaction(payload.emoji)
                    except: pass

                    async def queue_regeneration():
                        was_busy = is_busy
                        while session.get('is_running') or session.get('is_regenerating') or session.get('is_purging'):
                            await asyncio.sleep(1)
                        
                        session.get('regen_tasks', {}).pop(payload.message_id, None)
                        
                        if was_busy and msg_ref:
                            try:
                                await msg_ref.clear_reaction(payload.emoji)
                            except: pass
                        
                        still_exists = any(t.get("turn_id") == turn_id_to_find for t in session.get("unified_log",[]))
                        if still_exists:
                            await self.generation_service._execute_regeneration(payload, session, turn_id_to_find, reacted_to_participant)

                    task = asyncio.create_task(queue_regeneration())
                    session.setdefault('regen_tasks', {})[payload.message_id] = task
                    return
                
                if is_skip:
                    reacted_to_participant["is_skipped"] = True
                    self.session_manager._save_multi_profile_sessions()
                    try:
                        channel = self.bot.get_channel(payload.channel_id)
                        msg = await channel.fetch_message(payload.message_id)
                        await msg.add_reaction(payload.emoji)
                    except: pass
                    return

                try:
                    next_participant = None
                    last_speaker_key = session.get('last_speaker_key')
                    reacted_to_key = (reacted_to_participant['owner_id'], reacted_to_participant['profile_name'])

                    if is_continue or is_next:
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

                    if next_participant:
                        session_mode = session.get("session_mode", "sequential")
                        if session_mode == 'sequential':
                            try:
                                start_idx = session['profiles'].index(next_participant)
                                new_order = session['profiles'][start_idx:] + session['profiles'][:start_idx]
                                session['profiles'] = new_order
                                self.session_manager._save_multi_profile_sessions()
                            except ValueError:
                                pass

                        is_busy = session.get('is_running') or session.get('is_regenerating') or session.get('is_purging')
                        try:
                            channel = self.bot.get_channel(payload.channel_id)
                            msg_ref = await channel.fetch_message(payload.message_id)
                            if is_busy:
                                await msg_ref.add_reaction(payload.emoji)
                            else:
                                await msg_ref.clear_reaction(payload.emoji)
                        except: pass

                        trigger_type = 'reaction_single' if is_next else 'reaction'
                        reaction_trigger = (trigger_type, payload, next_participant)
                        await session['task_queue'].put(reaction_trigger)
                        if not session.get('worker_task') or session['worker_task'].done():
                            task = self.bot.loop.create_task(self.generation_service._multi_profile_worker(payload.channel_id))
                            session['worker_task'] = task
                            self.background_tasks.add(task)
                except (ValueError, IndexError):
                    pass

    @commands.Cog.listener()
    async def on_raw_reaction_remove(self, payload: discord.RawReactionActionEvent):
        if not self.has_lock: return
        
        emoji_str = str(payload.emoji)
        is_mute = (emoji_str in MUTE_TURN_EMOJI)
        is_skip = (emoji_str in SKIP_PARTICIPANT_EMOJI)
        is_regen = (emoji_str == REGENERATE_EMOJI)
        is_next = (emoji_str == NEXT_SPEAKER_EMOJI)
        is_continue = (emoji_str == CONTINUE_ROUND_EMOJI)
        
        if not any([is_mute, is_skip, is_regen, is_next, is_continue]): return

        channel_id = payload.channel_id
        session = self.multi_profile_channels.get(channel_id)
        if not session: return
        
        session_type = session.get("type", "multi")
        if not session.get("is_hydrated"):
            session = await self.session_manager._ensure_session_hydrated(channel_id, session_type)
        if not session: return

        turn_object = None
        for turn in session.get("unified_log",[]):
            if payload.message_id in turn.get("message_ids",[]):
                turn_object = turn
                break

        if turn_object:
            if is_regen:
                regen_task = session.get('regen_tasks', {}).pop(payload.message_id, None)
                if regen_task and not regen_task.done():
                    regen_task.cancel()
                    try:
                        channel = self.bot.get_channel(payload.channel_id)
                        msg = await channel.fetch_message(payload.message_id)
                        await msg.clear_reaction(payload.emoji)
                    except: pass
                return
            
            elif is_continue or is_next:
                session.setdefault('cancelled_reaction_triggers', set()).add((payload.message_id, emoji_str))
                try:
                    channel = self.bot.get_channel(payload.channel_id)
                    msg = await channel.fetch_message(payload.message_id)
                    await msg.clear_reaction(payload.emoji)
                except: pass
                return
                
            elif is_mute:
                turn_object["is_hidden"] = False
                await self.session_manager._save_session_to_disk((channel_id, None, None), session_type, session["unified_log"])
                session["is_hydrated"] = False
                await self.session_manager._ensure_session_hydrated(channel_id, session_type)
                try:
                    channel = self.bot.get_channel(payload.channel_id)
                    msg = await channel.fetch_message(payload.message_id)
                    await msg.remove_reaction(payload.emoji, self.bot.user)
                except: pass
                return
            
            elif is_skip and turn_object.get("is_user") is False:
                speaker_pid = turn_object.get("speaker_pid")
                participant = next((p for p in session['profiles'] if self.profile_manager._get_pid_from_name_any(p['owner_id'], p['profile_name']) == speaker_pid), None)
                if participant:
                    participant["is_skipped"] = False
                    self.session_manager._save_multi_profile_sessions()
                    try:
                        channel = self.bot.get_channel(payload.channel_id)
                        msg = await channel.fetch_message(payload.message_id)
                        await msg.remove_reaction(payload.emoji, self.bot.user)
                    except: pass
            

    @commands.Cog.listener()
    async def on_raw_message_delete(self, payload: discord.RawMessageDeleteEvent):
        if not self.has_lock:
            return

        deleted_message_id = payload.message_id
        if deleted_message_id in self.purged_message_ids:
            self.purged_message_ids.discard(deleted_message_id)
            return
        
        # 1. Check Global Chat Sessions
        if not payload.guild_id:
            channel = self.bot.get_channel(payload.channel_id)
            user_id = None
            if channel and isinstance(channel, discord.DMChannel):
                user_id = channel.recipient.id
            
            if user_id:
                index = self.profile_manager._get_user_index(user_id)
                all_pnames = list(index.get("personal", [])) + list(index.get("borrowed", []))
                
                for p_name in all_pnames:
                    session_key = ('global', user_id, p_name)
                    session_data = self.global_chat_sessions.get(session_key)
                    
                    if not session_data:
                        session_data = await self.session_manager._load_session_from_disk(session_key, 'global_chat')
                        if session_data:
                            self.global_chat_sessions[session_key] = session_data

                    if session_data:
                        turn_id_to_delete = None
                        for turn in session_data.get("unified_log", []):
                            if deleted_message_id in turn.get("message_ids", []):
                                turn_id_to_delete = turn.get("turn_id")
                                break
                        
                        if turn_id_to_delete:
                            original_len = len(session_data['unified_log'])
                            session_data['unified_log'] = [t for t in session_data['unified_log'] if t.get('turn_id') != turn_id_to_delete]

                            if len(session_data['unified_log']) < original_len:
                                new_history = []
                                for t in session_data['unified_log']:
                                    role = 'model' if t.get('is_user') is False else 'user'
                                    new_history.append({'role': role, 'parts': [t.get('content')]})
                                session_data['chat_session'] = GoogleGenAIChatSession(history=new_history)

                            if not session_data['unified_log']:
                                self.global_chat_sessions.pop(session_key, None)
                                self.session_last_accessed.pop(session_key, None)
                                await self.session_manager._delete_session_from_disk(session_key, 'global_chat')
                                self.ltm_recall_history.pop(session_key, None)
                            else:
                                await self.session_manager._save_session_to_disk(session_key, 'global_chat', session_data)
                                self.session_last_accessed[session_key] = time.time()
                            return

        # 2. Check Multi/Freewill Sessions (Servers)
        session = self.multi_profile_channels.get(payload.channel_id)
        if not session: return
        
        session_type = session.get("type", "multi")
        if not session.get("is_hydrated"):
            session = await self.session_manager._ensure_session_hydrated(payload.channel_id, session_type)
        if not session: return

        turn_id_to_delete = None
        turn_object = None
        for turn in session.get("unified_log", []):
            if deleted_message_id in turn.get("message_ids", []):
                turn_id_to_delete = turn.get("turn_id")
                turn_object = turn
                break

        if turn_id_to_delete and turn_object:
            if turn_object.get("is_user") is False:
                bot_pid = turn_object.get("speaker_pid")
                for p in session.get('profiles', []):
                    if self.profile_manager._get_pid_from_name_any(p['owner_id'], p['profile_name']) == bot_pid:
                        p['ltm_counter'] = max(0, p.get('ltm_counter', 0) - 1)
                        break

            original_log_len = len(session.get("unified_log", []))
            session["unified_log"] = [t for t in session.get("unified_log", []) if t.get("turn_id") != turn_id_to_delete]
            
            if len(session["unified_log"]) < original_log_len:
                is_effectively_empty = not session.get("unified_log") or all(
                    turn.get("type") in ["whisper", "private_response"] for turn in session.get("unified_log", [])
                )
                
                dummy_session_key = (payload.channel_id, None, None)
                if is_effectively_empty:
                    await self.session_manager._delete_session_from_disk(dummy_session_key, session_type)
                    for p_key in session.get("chat_sessions", {}).keys():
                        owner_id, profile_name = p_key
                        full_session_key = (payload.channel_id, owner_id, profile_name)
                        self.ltm_recall_history.pop(full_session_key, None)
                else:
                    await self.session_manager._save_session_to_disk(dummy_session_key, session_type, session["unified_log"])

                # Now that the correct state is on disk, force a re-read and rebuild
                session["is_hydrated"] = False
                await self.session_manager._ensure_session_hydrated(payload.channel_id, session_type)

            self.session_last_accessed[payload.channel_id] = time.time()

    @commands.Cog.listener()
    async def on_raw_message_edit(self, payload: discord.RawMessageUpdateEvent):
        if not self.has_lock: return
        
        message_id = payload.message_id
        channel_id = payload.channel_id
        
        session = self.multi_profile_channels.get(channel_id)
        if not session: return
        
        session_type = session.get("type", "multi")
        if not session.get("is_hydrated"):
            session = await self.session_manager._ensure_session_hydrated(channel_id, session_type)
        if not session: return

        turn_object = None
        for turn in session.get("unified_log", []):
            if message_id in turn.get("message_ids", []):
                turn_object = turn
                break

        if turn_object:
            # We only edit user messages. If a bot message is edited natively, we ignore it to prevent looping.
            if turn_object.get("is_user") is False:
                return
            
            channel = self.bot.get_channel(channel_id)
            if not channel: return
            
            try:
                msg = await channel.fetch_message(message_id)
            except discord.NotFound:
                return
            
            author_id = msg.author.id
            u_index = self.profile_manager._get_user_index(author_id)
            u_prof = self.session_manager._get_active_user_profile_name_for_channel(author_id, channel_id)
            u_is_b = u_prof in u_index.get("borrowed", [])
            u_sett = self.profile_manager._get_profile_config(author_id, u_prof, u_is_b) or {}
            user_tz = u_sett.get("timezone", "UTC")
            
            # Format the new content
            new_content = msg.clean_content
            reply_context = await self.generation_service._resolve_reply_context(msg)
            if reply_context:
                new_content = f"{reply_context}\n{new_content}"
                
            new_content += "\n(edited)"
            
            # Format and inject, keeping the original timestamp
            original_ts = msg.created_at
            new_history_line = _format_history_entry(msg.author.display_name, original_ts, new_content, user_tz)
            
            turn_object["content"] = new_history_line
            
            # Flush changes to disk
            dummy_session_key = (channel_id, None, None)
            await self.session_manager._save_session_to_disk(dummy_session_key, session_type, session["unified_log"])
            
            # Force Re-hydration so all participant histories get the updated log context instantly
            session["is_hydrated"] = False
            await self.session_manager._ensure_session_hydrated(channel_id, session_type)
            self.session_last_accessed[channel_id] = time.time()

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

    async def cog_unload(self):
        if self.has_lock:
            try:
                if os.path.exists(COG_LOCK_FILE_PATH):
                    os.remove(COG_LOCK_FILE_PATH)
            except OSError as e:
                print(f"OSError releasing lock file: {e}")
            except Exception as e:
                print(f"Unexpected error releasing lock file: {e}")
        
        self.bot.tree.remove_command(self.trace_ctx_menu.name, type=self.trace_ctx_menu.type)
        
        self.refresh_lock_task.cancel()
        self.session_manager.evict_inactive_sessions_task.cancel()
        self.profile_manager.hourly_self_repair_task.cancel()
        self.storage_manager.daily_cleanup_task.cancel()

        if self.image_finisher_worker_task:
            self.image_finisher_worker_task.cancel()
        for worker in self.image_gen_workers:
            worker.cancel()
        
        # [FIX] Explicitly cancel all active session worker tasks to prevent "Task pending" warnings
        for session_data in self.multi_profile_channels.values():
            if session_data.get('worker_task'):
                self.session_manager._safe_cancel_task(session_data['worker_task'])
                session_data['worker_task'] = None
        
        for session_key, chat_session in self.global_chat_sessions.items():
            await self.session_manager._save_session_to_disk(session_key, 'global_chat', chat_session)
        for ch_id, session_data in self.multi_profile_channels.items():
            if session_data.get("is_hydrated"): # Only save sessions that are loaded in memory
                session_type = session_data.get("type", "multi")
                unified_log = session_data.get("unified_log")
                if unified_log is not None:
                    # For multi-profile, the session_key is just the channel_id for path generation
                    dummy_session_key = (ch_id, None, None)
                    await self.session_manager._save_session_to_disk(dummy_session_key, session_type, unified_log)

    async def master_autocomplete(self, interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
        def get_focused(options):
            for opt in options:
                if opt.get('type') in (1, 2):
                    res = get_focused(opt.get('options', []))
                    if res: return res
                elif opt.get('focused'):
                    return opt.get('name')
            return None
        
        focused = get_focused(interaction.data.get('options', []))
        cmd_name = interaction.command.name

        if focused == 'method':
            choices = [
                app_commands.Choice(name="Auto (Recommended)", value="auto"),
                app_commands.Choice(name="Webhook", value="webhook")
            ]
            profile_val = interaction.namespace.profile_name
            if profile_val and interaction.guild:
                try:
                    p_owner_id_str, p_name = profile_val.split(":", 1)
                    p_owner_id = int(p_owner_id_str)
                except ValueError:
                    p_owner_id = interaction.user.id
                    p_name = profile_val
                eff_owner_id, eff_p_name = self.profile_manager._resolve_effective_profile(p_owner_id, p_name)
                linked_bot_id = self.child_bots_by_owner_profile.get((eff_owner_id, eff_p_name))
                if linked_bot_id and interaction.guild.get_member(int(linked_bot_id)):
                    choices.append(app_commands.Choice(name="Child Bot", value="child_bot"))
            return [c for c in choices if current.lower() in c.name.lower()]

        choices = []
        current_lower = current.lower()
        
        def format_choice_name(display_name, internal_name, pid, creator_name, is_system=False):
            suffix = " ┃ [System Profile]" if is_system else f" ┃ By {creator_name}"
            base_str = f" ┃ [{pid}]{suffix}"
            rem_len = 100 - len(base_str)
            name_str = f"{display_name} ({internal_name})"
            if len(name_str) > rem_len:
                half = max(1, (rem_len - 5) // 2)
                disp_trunc = display_name[:half] + "..." if len(display_name) > half else display_name
                int_trunc = internal_name[:half] + "..." if len(internal_name) > half else internal_name
                name_str = f"{disp_trunc} ({int_trunc})"
                if len(name_str) > rem_len:
                    name_str = name_str[:max(1, rem_len-3)] + "..."
            return f"{name_str}{base_str}"

        if cmd_name in ["speak", "whisper"]:
            server_id_str = str(interaction.guild_id) if interaction.guild_id else "dm"
            server_index = self.server_manager._get_server_index(server_id_str)
            channel_str = str(interaction.channel_id)
            session_data = server_index.get("active_sessions", {}).get("regular", {}).get(channel_str)
            if not session_data: return []
            
            for p in session_data.get("profiles", []):
                o_id = p.get("owner_id")
                p_name = p.get("profile_name")
                
                if cmd_name == "speak" and o_id != interaction.user.id and interaction.user.id != int(defaultConfig.DISCORD_OWNER_ID):
                    if o_id != int(defaultConfig.DISCORD_OWNER_ID):
                        continue
                
                eff_owner, eff_name = self.profile_manager._resolve_effective_profile(o_id, p_name)
                app = self.profile_manager._get_user_appearance(eff_owner, eff_name)
                disp_name = app.get("custom_display_name") or eff_name
                pid = self.profile_manager._get_pid_from_name_any(o_id, p_name)
                
                creator_name = "Unknown"
                is_sys = pid.startswith("X")
                if not is_sys:
                    creator = self.bot.get_user(eff_owner)
                    creator_name = creator.name if creator else str(eff_owner)
                    
                formatted_name = format_choice_name(disp_name, p_name, pid, creator_name, is_sys)
                val = f"{o_id}:{p_name}"
                if current_lower in p_name.lower() or current_lower in disp_name.lower():
                    choices.append(app_commands.Choice(name=formatted_name, value=val))
        
        elif cmd_name == "global_chat":
            user_id = interaction.user.id
            index = self.profile_manager._get_user_index(user_id)
            public_pointers = set()
            for p_info in self.public_profiles.values():
                if isinstance(p_info, str) and ":" in p_info:
                    public_pointers.add(p_info)
                elif isinstance(p_info, dict):
                    oid = str(p_info.get("owner_id"))
                    opid = p_info.get("original_pid")
                    if oid and opid: public_pointers.add(f"{oid}:{opid}")
            
            for p_name in index.get("personal", []):
                pid = self.profile_manager._get_pid_from_name_any(user_id, p_name)
                if f"{user_id}:{pid}" in public_pointers and current_lower in p_name.lower():
                    app = self.profile_manager._get_user_appearance(user_id, p_name)
                    disp_name = app.get("custom_display_name") or p_name
                    formatted = format_choice_name(disp_name, p_name, pid, interaction.user.name, False)
                    choices.append(app_commands.Choice(name=formatted, value=p_name))
                    
            for b_name in index.get("borrowed", []):
                b_cfg = self.profile_manager._get_profile_config(user_id, b_name, True)
                if not b_cfg: continue
                orig_oid = str(b_cfg.get("original_owner_id"))
                orig_pid = b_cfg.get("original_pid") or b_cfg.get("original_profile_id")
                if f"{orig_oid}:{orig_pid}" in public_pointers and current_lower in b_name.lower():
                    eff_owner, eff_name = self.profile_manager._resolve_effective_profile(user_id, b_name)
                    app = self.profile_manager._get_user_appearance(eff_owner, eff_name)
                    disp_name = app.get("custom_display_name") or eff_name
                    creator = self.bot.get_user(int(orig_oid))
                    c_name = creator.name if creator else orig_oid
                    formatted = format_choice_name(disp_name, b_name, orig_pid, c_name, str(orig_pid).startswith("X"))
                    choices.append(app_commands.Choice(name=formatted, value=b_name))
        
        else:
            user_id = interaction.user.id
            index = self.profile_manager._get_user_index(user_id)
            for p_name in index.get("personal", []):
                if current_lower in p_name.lower():
                    pid = self.profile_manager._get_pid_from_name_any(user_id, p_name)
                    app = self.profile_manager._get_user_appearance(user_id, p_name)
                    disp_name = app.get("custom_display_name") or p_name
                    formatted = format_choice_name(disp_name, p_name, pid, interaction.user.name, False)
                    choices.append(app_commands.Choice(name=formatted, value=p_name))
                    
            for b_name in index.get("borrowed", []):
                if current_lower in b_name.lower():
                    b_cfg = self.profile_manager._get_profile_config(user_id, b_name, True) or {}
                    orig_oid = b_cfg.get("original_owner_id", user_id)
                    orig_pid = b_cfg.get("original_pid") or b_cfg.get("original_profile_id", "Unknown")
                    eff_owner, eff_name = self.profile_manager._resolve_effective_profile(user_id, b_name)
                    app = self.profile_manager._get_user_appearance(eff_owner, eff_name)
                    disp_name = app.get("custom_display_name") or eff_name
                    creator = self.bot.get_user(int(orig_oid))
                    c_name = creator.name if creator else str(orig_oid)
                    formatted = format_choice_name(disp_name, b_name, orig_pid, c_name, str(orig_pid).startswith("X"))
                    choices.append(app_commands.Choice(name=formatted, value=b_name))
                    
            if user_id == int(defaultConfig.DISCORD_OWNER_ID):
                owner_idx = self.profile_manager._get_user_index(user_id)
                for s_name in owner_idx.get("system", {}):
                    if current_lower in s_name.lower():
                        pid = owner_idx["system"][s_name]
                        app = self.profile_manager._get_user_appearance(user_id, s_name)
                        disp_name = app.get("custom_display_name") or s_name
                        formatted = format_choice_name(disp_name, s_name, pid, "System", True)
                        choices.append(app_commands.Choice(name=formatted, value=s_name))
                        
        return choices[:25]
