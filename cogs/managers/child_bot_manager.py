import os
import time
import asyncio
import traceback
from typing import Dict, Any, Optional, get_args

from ..utils.constants import PLACEHOLDER_EMOJI, HarmBlockThreshold, HarmCategory
from .storage_manager import IOManager


class ChildBotManager:
    """Owns child bot configuration discovery and per-owner child bot shard access.

    Holds a back-reference to the parent cog for shared instance state,
    per the transitional Dependency Injection pattern in CLAUDE.md.
    """

    def __init__(self, cog):
        self.cog = cog

    def _find_borrowed_name_for_owner(self, author_id: int, original_owner_id: int, original_profile_name: str) -> Optional[str]:
        """Finds the name under which author_id has borrowed original_owner_id's original_profile_name profile, if any."""
        author_index = self.cog.profile_manager._get_user_index(author_id)
        for b_name in author_index.get("borrowed", []):
            b_data = self.cog.profile_manager._get_profile_config(author_id, b_name, True) or {}
            if int(b_data.get("original_owner_id", 0)) == original_owner_id and b_data.get("original_profile_name") == original_profile_name:
                return b_name
        return None

    def _load_child_bots(self):
        self.cog.child_bots = {}
        self.cog.child_bots_by_owner_profile = {}
        if not os.path.isdir(self.cog.USERS_DIR):
            return

        for user_id_str in os.listdir(self.cog.USERS_DIR):
            if not user_id_str.isdigit(): continue
            profiles_dir = os.path.join(self.cog.USERS_DIR, user_id_str, "profiles")
            if not os.path.isdir(profiles_dir): continue

            for pid_folder in os.listdir(profiles_dir):
                bot_file = os.path.join(profiles_dir, pid_folder, "child_bot.json.gz")
                if os.path.exists(bot_file):
                    bot_data = IOManager.read_json_gzip(bot_file, encrypted=False)
                    if bot_data and "bot_id" in bot_data:
                        # Dynamically resolve the profile name from the directory's name.txt
                        name_file = os.path.join(profiles_dir, pid_folder, "name.txt")
                        p_name = None
                        if os.path.exists(name_file):
                            with open(name_file, 'r', encoding='utf-8') as nf:
                                p_name = nf.read().strip()

                        if p_name:
                            bot_data["owner_id"] = int(user_id_str)
                            bot_data["profile_name"] = p_name
                            bot_data["pid"] = pid_folder
                            self.cog.child_bots[bot_data["bot_id"]] = bot_data
                            self.cog.child_bots_by_owner_profile[(bot_data["owner_id"], bot_data["profile_name"])] = bot_data["bot_id"]

    async def handle_child_bot_event(self, event_data: Dict):
        if event_data.get("message", {}).get("author_id") in self.cog.global_blacklist: return
        
        event_type = event_data.get("event_type")
        bot_id = event_data.get("bot_id")
        
        if event_type == "message_received":
            message_payload = event_data.get("message", {})
            channel_id = message_payload.get("channel_id")
            guild_id = message_payload.get("guild_id")
            
            message_id = message_payload.get("id")

            if message_id:
                self.cog.processed_child_messages[message_id] = True

            bot_config = self.cog.child_bots.get(bot_id)
            if not bot_config: return
            
            # [NEW] Premium Gate for Runtime Execution
            # If owner is not premium, ignore the event silently to save resources.
            if not self.cog.profile_manager.is_user_premium(bot_config['owner_id']):
                return

            original_owner_id = bot_config['owner_id']
            original_profile_name = bot_config['profile_name']
            
            effective_owner_id = original_owner_id
            effective_profile_name = original_profile_name
            
            guild = self.cog.bot.get_guild(guild_id) if guild_id else None
            if guild and not guild.get_member(original_owner_id):
                author_id = message_payload.get("author_id")
                borrowed_name = self._find_borrowed_name_for_owner(author_id, original_owner_id, original_profile_name)

                if borrowed_name:
                    effective_owner_id = author_id
                    effective_profile_name = borrowed_name
                else:
                    session = self.cog.multi_profile_channels.get(channel_id)
                    found_in_session = False
                    if session:
                        for p in session.get("profiles", []):
                            p_index = self.cog.profile_manager._get_user_index(p['owner_id'])
                            if p['profile_name'] in p_index.get("borrowed", []):
                                b_data = self.cog.profile_manager._get_profile_config(p['owner_id'], p['profile_name'], True) or {}
                                if int(b_data.get("original_owner_id", 0)) == original_owner_id and b_data.get("original_profile_name") == original_profile_name:
                                    effective_owner_id = p['owner_id']
                                    effective_profile_name = p['profile_name']
                                    found_in_session = True
                                    break
                    
                    if not found_in_session:
                        await self.cog.manager_queue.put({
                            "action": "send_to_child", "bot_id": bot_id,
                            "payload": {
                                "action": "send_message", "channel_id": channel_id,
                                "content": "My original owner is not in this server, and you have not borrowed my profile. Use `/profile hub` to find and borrow me first!"
                            }
                        })
                        return

            session = self.cog.multi_profile_channels.get(channel_id)
            
            ephemeral_participant = {
                "owner_id": effective_owner_id,
                "profile_name": effective_profile_name,
                "method": "child_bot",
                "bot_id": bot_id,
                "ephemeral": True # Mark as a guest star
            }

            if not session:
                # Create a new session on the fly
                session = {
                    "type": "multi", "chat_sessions": {}, "unified_log": [], "is_hydrated": False,
                    "last_bot_message_id": None, "owner_id": message_payload.get("author_id"), "is_running": False,
                    "task_queue": asyncio.Queue(), "worker_task": None, "turns_since_last_ltm": 0,
                    "session_prompt": None, "session_mode": "sequential", "profiles": []
                }
                self.cog.multi_profile_channels[channel_id] = session

            # Put the trigger into the queue
            trigger = ('child_mention', message_payload, ephemeral_participant)
            await session['task_queue'].put(trigger)
            
            # Start the worker if it's not running
            if not session.get('is_running'):
                session['worker_task'] = self.cog.bot.loop.create_task(self.cog.generation_service._multi_profile_worker(channel_id))

    async def handle_child_bot_presence(self, event_data: Dict):
        bot_id = str(event_data.get("bot_id"))
        presence_update = event_data.get("presence")
        if not bot_id or not presence_update: return
        
        bot_config = self.cog.child_bots.get(bot_id)
        if bot_config:
            owner_id = bot_config['owner_id']
            pid = bot_config['pid']
            bot_file = os.path.join(self.cog.USERS_DIR, str(owner_id), "profiles", pid, "child_bot.json.gz")

            def _sync_update_presence():
                saved_config = IOManager.read_json_gzip(bot_file, encrypted=False) or {}
                current_presence = saved_config.get("presence", {})
                current_presence.update(presence_update)
                saved_config["presence"] = current_presence
                IOManager.write_json_gzip(saved_config, bot_file, encrypted=False)
                return current_presence

            bot_config["presence"] = await asyncio.to_thread(_sync_update_presence)

    async def handle_child_bot_image_request(self, event_data: Dict):
        if event_data.get("message", {}).get("author_id") in self.cog.global_blacklist: return
        
        bot_id = event_data.get("bot_id")
        message_data = event_data.get("message", {})
        channel_id = message_data.get("channel_id")

        if channel_id in self.cog.multi_profile_channels:
            return

        async def send_notification_to_child(content: str):
            await self.cog.manager_queue.put({
                "action": "send_to_child", "bot_id": bot_id,
                "payload": {"action": "send_message", "channel_id": channel_id, "content": f"(Notice for {message_data['author_name']}): {content}"}
            })

        try:
            if self.cog.image_request_queue.full():
                await send_notification_to_child("The image generation backlog is currently full. Please try again in a moment.")
                return

            if self.cog.image_gen_semaphore.locked():
                qsize = self.cog.image_request_queue.qsize()
                await send_notification_to_child(f"Your image generation request has been queued. You are #{qsize + 1} in line.")
            else:
                await self.cog.manager_queue.put({"action": "send_to_child", "bot_id": bot_id, "payload": {"action": "start_typing", "channel_id": channel_id}})

            bot_config = self.cog.child_bots.get(bot_id)
            if not bot_config: return

            # [NEW] Premium Gate for Child Bot Images
            if not self.cog.profile_manager.is_user_premium(bot_config['owner_id']):
                # If we already sent a "Queue" DM notification above, we might want to tell them why it failed.
                # However, for simplicity and hard-gating, we just return.
                return

            guild_id = message_data.get("guild_id")
            original_owner_id = bot_config['owner_id']
            original_profile_name = bot_config['profile_name']
            
            owner_id = original_owner_id
            profile_name = original_profile_name
            
            guild = self.cog.bot.get_guild(guild_id) if guild_id else None
            if guild and not guild.get_member(original_owner_id):
                author_id = message_data.get("author_id")
                borrowed_name = self._find_borrowed_name_for_owner(author_id, original_owner_id, original_profile_name)

                if borrowed_name:
                    owner_id = author_id
                    profile_name = borrowed_name
                else:
                    await send_notification_to_child("My original owner is not in this server, and you have not borrowed my profile. Use `/profile hub` to find and borrow me first!")
                    return

            index = self.cog.profile_manager._get_user_index(owner_id)
            is_borrowed = profile_name in index.get("borrowed", [])
            profile_data = self.cog.profile_manager._get_profile_config(owner_id, profile_name, is_borrowed) or {}

            placeholder_message_obj = None
            if self.cog.image_request_queue.full():
                await send_notification_to_child("The image generation backlog is currently full. Please try again in a moment.")
                return

            if self.cog.image_gen_semaphore.locked():
                qsize = self.cog.image_request_queue.qsize()
                await send_notification_to_child(f"Your image generation request has been queued. You are #{qsize + 1} in line.")
            else:
                if profile_data.get("child_bot_placeholder", False):
                    custom_emoji = profile_data.get("placeholder_emoji") or PLACEHOLDER_EMOJI
                    msg_id = await self.cog.generation_service._send_child_bot_placeholder(bot_id, channel_id, custom_emoji)
                    if msg_id:
                        try:
                            ch = self.cog.bot.get_channel(channel_id)
                            placeholder_message_obj = await ch.fetch_message(msg_id)
                        except: pass
                else:
                    await self.cog.manager_queue.put({"action": "send_to_child", "bot_id": bot_id, "payload": {"action": "start_typing", "channel_id": channel_id}})

            image_prefixes = ("!image", "!imagine")
            used_prefix = next((p for p in image_prefixes if message_data.get("content", "").lower().startswith(p)), "!image")
            prompt_text = message_data.get("content", "")[len(used_prefix):].strip()
            if not prompt_text: return
            
            index = self.cog.profile_manager._get_user_index(owner_id)
            is_borrowed = profile_name in index.get("borrowed", [])
            profile_data = self.cog.profile_manager._get_profile_config(owner_id, profile_name, is_borrowed) or {}

            if not profile_data.get("image_generation_enabled", False):
                return

            safety_level_str = profile_data.get("safety_level", "low")
            
            safety_map = { "unrestricted": HarmBlockThreshold.BLOCK_NONE, "low": HarmBlockThreshold.BLOCK_ONLY_HIGH, "medium": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, "high": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE }
            threshold = safety_map.get(safety_level_str, HarmBlockThreshold.BLOCK_ONLY_HIGH)
            dynamic_safety_settings = { cat: threshold for cat in get_args(HarmCategory) }
            
            # Get appearance text
            source_owner_id = owner_id
            source_profile_name = profile_name
            if is_borrowed:
                borrowed_data = self.cog.profile_manager._get_profile_config(owner_id, profile_name, True) or {}
                source_owner_id = int(borrowed_data.get("original_owner_id", owner_id))
                source_profile_name = borrowed_data.get("original_profile_name", profile_name)
            
            source_prompts = self.cog.profile_manager._get_profile_prompts(source_owner_id, source_profile_name) or {}
            persona = source_prompts.get("persona", {})
            appearance_lines_encrypted = persona.get("appearance", [])
            appearance_text = "\n".join([self.cog.storage_manager._decrypt_data(line) for line in appearance_lines_encrypted])

            bot_user = self.cog.bot.get_user(int(bot_id)); bot_display_name = bot_user.name if bot_user else profile_name

            final_prompt_text = prompt_text
            if appearance_text.strip():
                prompt_lower = prompt_text.lower()
                second_person_pronouns = ["you", "your", "yourself", "u", "ur"]
                # Check for pronouns or the profile's names
                if any(pronoun in prompt_lower.split() for pronoun in second_person_pronouns) or \
                   bot_display_name.lower() in prompt_lower or \
                   profile_name.lower() in prompt_lower:
                    final_prompt_text = f"Your appearance:\n{appearance_text.strip()}\n\nUser's prompt:\n{prompt_text}"

            system_instruction = self.cog.media_service._get_image_gen_system_instruction(owner_id, profile_name)

            reference_image_urls = []
            replied_to_data = message_data.get("replied_to")
            if replied_to_data and replied_to_data.get("attachment_url"):
                reference_image_urls.append({"url": replied_to_data["attachment_url"], "mime_type": "image/png"})

            attachments_data = message_data.get("attachments", [])
            if len(reference_image_urls) < 10 and attachments_data:
                for attachment in attachments_data:
                    if attachment.get("url"):
                        reference_image_urls.append({"url": attachment.get("url"), "mime_type": attachment.get("content_type", "image/png")})
                        if len(reference_image_urls) >= 10: break

            grounding_sources = []
            grounding_mode = profile_data.get("grounding_mode", "off")
            if isinstance(grounding_mode, bool): grounding_mode = "on" if grounding_mode else "off"

            if grounding_mode in ["on", "on+"]:
                session_key = (channel_id, owner_id, profile_name)
                chat = self.cog.chat_sessions.get(session_key)
                history_for_grounding = chat.history if chat else []
                
                mapping_key = self.cog.session_manager._get_mapping_key_for_session(session_key, 'single')
                ch_obj = self.cog.bot.get_channel(channel_id)
                grounding_result = await self.cog.tools_service._get_hybrid_grounding_context(prompt_text, guild_id, history_for_grounding, mapping_key, is_for_image=True, warning_channel=ch_obj)
                if grounding_result:
                    grounding_context, sources, *_ = grounding_result
                    if grounding_context:
                        final_prompt_text = f"{prompt_text}\n\nUse this information to help generate the image:\n{grounding_context}"
                        grounding_sources = sources

            request_data = {
                "is_child_bot": True, "bot_id": bot_id, "author_id": message_data['author_id'],
                "channel_id": channel_id, "guild_id": guild_id, "original_message_id": message_data['id'], 
                "original_content": message_data['content'], "prompt_text": final_prompt_text, 
                "effective_profile_owner_id": owner_id, "effective_profile_name": profile_name,
                "bot_display_name": bot_display_name, "safety_settings": dynamic_safety_settings,
                "system_instruction": system_instruction, "reference_image_urls": reference_image_urls, "placeholder_message": placeholder_message_obj, 
                "grounding_sources": grounding_sources, "grounding_mode": grounding_mode,
                "image_generation_model": profile_data.get("image_generation_model", "gemini-2.5-flash-image")
            }
            
            # [NEW] Priority Logic
            is_premium = self.cog.profile_manager.is_user_premium(owner_id)
            priority = 10 if is_premium else 20
            
            await self.cog.image_request_queue.put((priority, time.time(), request_data))
        except Exception as e:
            print(f"Error dispatching child bot image request for bot {bot_id}: {e}"); traceback.print_exc()

    async def handle_child_bot_toggle(self, event_data: Dict):
        bot_id = str(event_data.get("bot_id")) # Ensure string
        channel_id = event_data.get("channel_id")
        correlation_id = event_data.get("correlation_id")
        
        bot_config = self.cog.child_bots.get(bot_id)
        if not bot_config: return

        session = self.cog.multi_profile_channels.get(channel_id)
        
        action_taken = None
        result_msg = ""

        if not session:
            # If no session, create one and add the bot.
            participant = {
                "owner_id": bot_config['owner_id'], "profile_name": bot_config['profile_name'],
                "method": "child_bot", "bot_id": bot_id, "ephemeral": False
            }
            chat_sessions = {(participant['owner_id'], participant['profile_name']): None}
            session = {
                "type": "multi", "profiles": [participant], "chat_sessions": chat_sessions,
                "unified_log": [], "is_hydrated": False, "last_bot_message_id": None,
                "owner_id": event_data.get("user_id"), "is_running": False,
                "task_queue": asyncio.Queue(),
                "worker_task": None, "turns_since_last_ltm": 0, "session_prompt": None,
                "session_mode": "sequential", "audio_mode": "off"
            }
            self.cog.multi_profile_channels[channel_id] = session
            action_taken = "add"
            result_msg = "Created a new Chat Session."
        else:
            # Session exists, check if bot is already a participant
            participant_index = -1
            for i, p in enumerate(session['profiles']):
                if str(p.get('bot_id')) == bot_id:
                    participant_index = i
                    break
            
            if participant_index != -1:
                # Remove it
                removed_p = session['profiles'].pop(participant_index)
                session['chat_sessions'].pop((removed_p['owner_id'], removed_p['profile_name']), None)
                action_taken = "remove"
                result_msg = "Removed from the current Chat Session."
                
                if not session['profiles']:
                    self.cog.multi_profile_channels.pop(channel_id, None)
            else:
                if len(session['profiles']) >= 200:
                    if correlation_id:
                        await self.cog.manager_queue.put({
                            "action": "send_to_child", "bot_id": bot_id,
                            "payload": {"action": "toggle_result", "correlation_id": correlation_id, "result": "The current Chat Session contains the maximum of 200 participating profiles. Please remove a profile and try again."}
                        })
                    return

                # Add it
                participant = {
                    "owner_id": bot_config['owner_id'], "profile_name": bot_config['profile_name'],
                    "method": "child_bot", "bot_id": bot_id, "ephemeral": False
                }
                session['profiles'].append(participant)
                # Also create the placeholder for the chat session
                session['chat_sessions'][(participant['owner_id'], participant['profile_name'])] = None
                action_taken = "add"
                result_msg = "Added to the current Chat Session."
        
        self.cog.session_manager._save_multi_profile_sessions()

        if action_taken:
            ipc_action = "session_update_add" if action_taken == "add" else "session_update_remove"
            await self.cog.manager_queue.put({
                "action": "send_to_child", "bot_id": bot_id,
                "payload": {"action": ipc_action, "channel_id": channel_id}
            })
            if action_taken == "remove":
                await self.cog.manager_queue.put({
                    "action": "send_to_child", "bot_id": bot_id,
                    "payload": {"action": "stop_typing", "channel_id": channel_id}
                })
            
            if correlation_id:
                await self.cog.manager_queue.put({
                    "action": "send_to_child", "bot_id": bot_id,
                    "payload": {"action": "toggle_result", "correlation_id": correlation_id, "result": result_msg}
                })

    async def handle_child_bot_refresh(self, command_data: Dict):
        bot_id = command_data.get("bot_id")
        channel_id = command_data.get("channel_id")
        if not bot_id or not channel_id:
            return

        bot_config = self.cog.child_bots.get(bot_id)
        if not bot_config:
            return

        owner_id = bot_config['owner_id']
        profile_name = bot_config['profile_name']
        
        # 1. Cancel the running worker task for this specific child bot session
        worker_key = (channel_id, bot_id)
        if worker_key in self.cog.child_bot_single_sessions:
            worker_data = self.cog.child_bot_single_sessions.pop(worker_key)
            if worker_data and worker_data.get('task'):
                self.cog.session_manager._safe_cancel_task(worker_data['task'])

        # 2. Clear all caches and delete the on-disk session file
        session_key = (channel_id, owner_id, profile_name)
        
        self.cog.chat_sessions.pop(session_key, None)
        self.cog.channel_models.pop(session_key, None)
        self.cog.channel_model_last_profile_key.pop(session_key, None)
        self.cog.session_last_accessed.pop(session_key, None)
        await self.cog.session_manager._delete_session_from_disk(session_key, 'single')
        self.cog.ltm_recall_history.pop(session_key, None)

        # 3. Reset the LTM creation counter
        ltm_counter_key = (owner_id, profile_name, "guild")
        self.cog.message_counters_for_ltm.pop(ltm_counter_key, None)

    async def handle_child_bot_confirmation(self, event_data: Dict):
        correlation_id = event_data.get("correlation_id")
        if not correlation_id or correlation_id not in self.cog.pending_child_confirmations:
            return

        confirmation_data = self.cog.pending_child_confirmations.pop(correlation_id)
        message_ids = event_data.get("message_ids", [])
        if not message_ids: return
        
        try:
            if confirmation_data.get("type") == "heartbeat_placeholder":
                confirmation_data["message_ids"] = message_ids

            elif confirmation_data.get("type") == "multi_profile":
                channel_id = confirmation_data.get("channel_id")
                turn_id = confirmation_data.get("turn_id")

                if not all([channel_id, turn_id]):
                    return

                session = self.cog.multi_profile_channels.get(channel_id)
                if session:
                    session['last_bot_message_id'] = message_ids[-1]
                    for turn in session.get("unified_log", []):
                        if turn.get("turn_id") == turn_id:
                            turn.setdefault("message_ids", []).extend(message_ids)
                            break
                    
                    session_type = session.get("type", "multi")
                    await self.cog.session_manager._save_session_to_disk((channel_id, None, None), session_type, session.get("unified_log", []))
        
        except Exception as e:
            print(f"Error during child bot confirmation ({correlation_id}): {e}")
            traceback.print_exc()
        
        finally:
            if "event" in confirmation_data and confirmation_data["event"]:
                confirmation_data["event"].set()
