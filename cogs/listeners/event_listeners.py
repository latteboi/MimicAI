import os
import re
import time
import uuid
import random
import asyncio
import traceback
from typing import Any, Dict, List, Tuple

import discord
from discord import app_commands
from discord.ext import commands

from ..utils.constants import *
from ..utils import mem_probe
from ..utils.content import WELCOME_MESSAGE, WELCOME_CHANNEL_HINTS
from ..utils.helpers import _format_history_entry, _get_user_hash
from ..utils.fuzzy import rank_keyed


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

        self.child_bot_manager._load_child_bots()
        self.all_bot_ids = {self.bot.user.id} | {int(bot_id) for bot_id in self.child_bots.keys()}
        await self.child_bot_manager.start_all_child_bots()

        if self.has_lock:
            await self._cache_command_ids()
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

        if mem_probe.ENABLED and not getattr(self, '_mem_reporter_task', None):
            self._mem_reporter_task = self.bot.loop.create_task(mem_probe.reporter())

        if self.has_lock and not self.image_finisher_worker_task:
            self.image_finisher_worker_task = self.bot.loop.create_task(self.media_service._image_finisher_worker())
            for i in range(2):
                worker = self.bot.loop.create_task(self.media_service._image_gen_worker(i))
                self.image_gen_workers.append(worker)

    async def _cache_command_ids(self):
        """Resolve top-level command names to ids so `</start:id>` renders clickable.

        One request, once per boot, and only the top level -- a subcommand is mentioned
        through its parent's id, which is already here. Failure is not worth retrying
        or reporting to anyone: `command_mention` falls back to a `/start` code span,
        which is what every other line of copy in the bot already uses.
        """
        if self.command_ids:
            return
        try:
            for cmd in await self.bot.tree.fetch_commands():
                self.command_ids[cmd.name] = cmd.id
        except Exception as e:
            print(f"Could not fetch command ids for mentions: {type(e).__name__}({e})")

    def _pick_welcome_channel(self, guild: discord.Guild):
        """The best channel to say hello in, or None if there is nowhere we may speak.

        `system_channel` first because that is literally the server's welcome channel --
        it is where Discord itself posts join notices, so a greeting there is in the one
        place the server has already nominated for greetings. Everything after it is a
        fallback for a server that has none set.

        Permissions are checked rather than assumed. A bot invited with a narrow role
        can see channels it cannot post in, and an unhandled Forbidden on join is how a
        bot looks broken in the first thirty seconds anyone has known it.
        """
        me = guild.me
        if me is None:
            return None

        def usable(channel) -> bool:
            # Both sources below already yield text channels; None is the real case.
            if channel is None:
                return False
            perms = channel.permissions_for(me)
            return perms.view_channel and perms.send_messages

        if usable(guild.system_channel):
            return guild.system_channel

        # Named like a welcome or general channel, in hint order, so a server with both
        # #welcome and #general gets the more deliberate of the two.
        candidates = [c for c in guild.text_channels if usable(c)]
        for hint in WELCOME_CHANNEL_HINTS:
            match = next((c for c in candidates if hint in c.name.lower()), None)
            if match is not None:
                return match

        # Otherwise the topmost channel we can actually speak in.
        return candidates[0] if candidates else None

    @commands.Cog.listener()
    async def on_guild_join(self, guild: discord.Guild):
        """The one message the bot sends unasked: a greeting, once, on being invited.

        No state backs the "once" -- `on_guild_join` fires on a genuine join and not on
        a reconnect, a restart or a resume, so the event *is* the guarantee, and a
        re-invite genuinely greets again. A stored "already greeted" flag would only
        add a way for the two to disagree, and something else to migrate.

        `has_lock` is the guard that matters: two instances of the bot are both in the
        guild and would both post.
        """
        if not self.has_lock:
            return

        channel = self._pick_welcome_channel(guild)
        if channel is None:
            return

        await self._cache_command_ids()
        try:
            await channel.send(WELCOME_MESSAGE.format(
                start=self.command_mention("start"), help=self.command_mention("help")))
        except discord.Forbidden:
            pass
        except Exception as e:
            print(f"Welcome message failed in guild {guild.id}: {type(e).__name__}({e})")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        # The game table is a sticky message, so anything landing in the channel buries
        # it. This sits above every guard below deliberately: during a game most of the
        # channel's traffic is webhook and child-bot posts, and none of that gets past
        # the `author.bot` return a few lines down. `nudge_table` is synchronous and
        # costs one dict lookup, which is what the emptiness check in front of it keeps
        # it to on the overwhelmingly common no-game path.
        if self.active_games:
            self.game_service.nudge_table(message.channel.id, message.id)

        if message.id in self.processed_child_messages:
            return

        if message.author.id in self.global_blacklist:
            return
        
        if not message.guild or not self.has_lock or message.author.bot:
            return

        # Calling Last Card is a thing you say, not a button you press, so this sits
        # ahead of the trigger paths below rather than inside them -- and it *returns*.
        # The call is a game control input, not a line of dialogue: letting it fall
        # through queued a whole round off the word "one". The cast still hears it,
        # because `Ev.CALL_MADE` is in `GameService.REACT_ON` and draws a beat when the
        # call rides the next play, which is the moment it actually lands. Ordinary chat
        # is untouched -- `arm_last_call` answers in one dict lookup with no game here,
        # and returns False for anyone who is not seated at this table.
        if self.active_games and self.game_service.arm_last_call(
                message.channel.id, message.author.id, message.content):
            try:
                await message.add_reaction("🖐")
            except Exception:
                pass
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
            if session and self.session_manager.is_started(session):
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
                # A seated-but-unstarted session is not a session as far as an
                # ordinary message is concerned, so a mention routes the standalone
                # way -- exactly as it would in a channel with no session at all.
                if self.session_manager.is_started(self.multi_profile_channels.get(message.channel.id)):
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

        # Seating a cast does not make the channel live. Until `Start / Update Session`
        # is pressed the session exists only as a draft, and ordinary chat passes
        # through it untouched.
        if session and self.session_manager.is_started(session):
            if 'task_queue' not in session or session['task_queue'] is None:
                session['task_queue'] = asyncio.Queue()

            await session['task_queue'].put(message)
            
            if not session.get('worker_task') or session['worker_task'].done():
                task = self.bot.loop.create_task(self.generation_service._multi_profile_worker(message.channel.id))
                session['worker_task'] = task
                self.background_tasks.add(task)
            return

    def _reaction_user_is_admin(self, payload: discord.RawReactionActionEvent) -> bool:
        """Administrator (or bot owner) test for a raw reaction.

        `payload.member` is populated on adds in a guild, which is the only place these
        controls exist, so this normally costs nothing; the guild walk is the fallback
        for a member the cache does not hold. No member and no cache means no
        permission, because a control that fails open is not a control.
        """
        try:
            if payload.user_id == int(defaultConfig.DISCORD_OWNER_ID):
                return True
        except (TypeError, ValueError):
            pass
        member = payload.member
        if member is None and payload.guild_id:
            guild = self.bot.get_guild(payload.guild_id)
            member = guild.get_member(payload.user_id) if guild else None
        perms = getattr(member, "guild_permissions", None)
        return bool(perms and perms.administrator)

    async def _pull_back_reaction(self, payload: discord.RawReactionActionEvent):
        """Take a denied reaction back off, so the refusal is visible.

        There is no interaction behind a reaction and therefore no ephemeral message to
        answer with, and a reaction that sits there while nothing happens reads as the
        bot being broken -- the same complaint an unanswered slash command draws. This
        needs Manage Messages and is best-effort: without it the deny is simply silent,
        which is no worse than doing nothing at all.
        """
        try:
            channel = self.bot.get_channel(payload.channel_id)
            if channel is None:
                return
            message = await channel.fetch_message(payload.message_id)
            await message.remove_reaction(payload.emoji, discord.Object(id=payload.user_id))
        except Exception:
            pass

    @commands.Cog.listener()
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        if payload.user_id in self.global_blacklist:
            return
        if not self.has_lock or payload.user_id == self.bot.user.id:
            return

        emoji_str = str(payload.emoji)

        # /train reactions work whether or not this channel has any active/logged chat
        # session at all, so this is handled and short-circuited before the
        # session-required bail-out below.
        if emoji_str in (TRAIN_INPUT_EMOJI, TRAIN_OUTPUT_EMOJI):
            await self._handle_train_reaction(payload, emoji_str == TRAIN_INPUT_EMOJI)
            return

        is_regen = (emoji_str == REGENERATE_EMOJI)
        is_next = (emoji_str == NEXT_SPEAKER_EMOJI)
        is_continue = (emoji_str == CONTINUE_ROUND_EMOJI)
        is_mute = (emoji_str in MUTE_TURN_EMOJI)
        is_skip = (emoji_str in SKIP_PARTICIPANT_EMOJI)

        if not any([is_regen, is_next, is_continue, is_mute, is_skip]):
            return

        # Muting is deliberately open: it hides one turn from the transcript, which is
        # the ordinary repair anyone in a scene needs when a reply lands badly, and it
        # is reversible from the same place it was done.
        if is_skip and not self._reaction_user_is_admin(payload):
            # Skipping is not turn-local. It suspends a participant from every round
            # until somebody unskips it, so one press can silence another member's
            # character indefinitely, and nothing in the channel says who did it or
            # that it happened. Checked here rather than where the skip is applied so a
            # denied press never wakes a dehydrated session.
            asyncio.create_task(self._pull_back_reaction(payload))
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
                
                # Optimistically acknowledge reaction immediately
                async def _ack_mute():
                    try:
                        ch = self.bot.get_channel(payload.channel_id)
                        if ch:
                            msg = await ch.fetch_message(payload.message_id)
                            await msg.add_reaction(payload.emoji)
                    except Exception: pass
                asyncio.create_task(_ack_mute())

                await self.session_manager._save_session_to_disk((channel_id, None, None), session_type, session["unified_log"])
                # Hidden turns are skipped by the whisper derivation, so muting a turn
                # can resurrect or drop a pending whisper.
                self.session_manager._recompute_pending_whispers(session)
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
                    is_busy = any(session.get(flag) for flag in SESSION_BUSY_FLAGS)
                    
                    async def _ack_regen_reaction(busy: bool):
                        try:
                            ch = self.bot.get_channel(payload.channel_id)
                            if ch:
                                msg = await ch.fetch_message(payload.message_id)
                                if busy:
                                    await msg.add_reaction(payload.emoji)
                                else:
                                    await msg.clear_reaction(payload.emoji)
                        except Exception: pass
                    
                    asyncio.create_task(_ack_regen_reaction(is_busy))

                    async def queue_regeneration():
                        was_busy = is_busy
                        try:
                            # Now bounded. This spun unbounded at 2 Hz on flags it does not own,
                            # and it waits on is_whispering as well now, so a leaked flag would
                            # have stranded the task for the life of the process. On timeout the
                            # regeneration is dropped rather than run against a live round --
                            # the same outcome as the user pulling the reaction back off.
                            if not await self.session_manager._wait_for_session_flags(
                                session, SESSION_BUSY_FLAGS, WHISPER_BUSY_WAIT_TIMEOUT_SECONDS
                            ):
                                print(f"Regeneration for message {payload.message_id} timed out waiting for the channel; dropped.")
                                return

                            if was_busy:
                                asyncio.create_task(_ack_regen_reaction(False))

                            still_exists = any(t.get("turn_id") == turn_id_to_find for t in session.get("unified_log", []))
                            if still_exists:
                                await self.generation_service._execute_regeneration(payload, session, turn_id_to_find, reacted_to_participant)
                        finally:
                            # Deregistered when the work is actually over, not before it
                            # starts. This used to pop the handle immediately after the
                            # wait and then run the regeneration, so for its entire
                            # lifetime nothing held a reference to it -- /cancel had
                            # nothing to cancel, which is why regeneration ignored it.
                            session.get('regen_tasks', {}).pop(payload.message_id, None)

                    task = asyncio.create_task(queue_regeneration())
                    session.setdefault('regen_tasks', {})[payload.message_id] = task
                    return
                
                if is_skip:
                    reacted_to_participant["is_skipped"] = True
                    
                    # Optimistically acknowledge reaction immediately
                    async def _ack_skip():
                        try:
                            ch = self.bot.get_channel(payload.channel_id)
                            if ch:
                                msg = await ch.fetch_message(payload.message_id)
                                await msg.add_reaction(payload.emoji)
                        except Exception: pass
                    asyncio.create_task(_ack_skip())

                    self.session_manager._save_multi_profile_sessions()
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

                        is_busy = any(session.get(flag) for flag in SESSION_BUSY_FLAGS)
                        
                        async def _ack_nav_reaction(busy: bool):
                            try:
                                ch = self.bot.get_channel(payload.channel_id)
                                if ch:
                                    msg = await ch.fetch_message(payload.message_id)
                                    if busy:
                                        await msg.add_reaction(payload.emoji)
                                    else:
                                        await msg.clear_reaction(payload.emoji)
                            except Exception: pass

                        asyncio.create_task(_ack_nav_reaction(is_busy))

                        trigger_type = 'reaction_single' if is_next else 'reaction'
                        reaction_trigger = (trigger_type, payload, next_participant)
                        
                        if 'task_queue' not in session or session['task_queue'] is None:
                            session['task_queue'] = asyncio.Queue()

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

        if emoji_str in (TRAIN_INPUT_EMOJI, TRAIN_OUTPUT_EMOJI):
            entry = self.armed_training_channels.get((payload.channel_id, payload.user_id))
            if entry:
                slot_key = "slot1" if emoji_str == TRAIN_INPUT_EMOJI else "slot2"
                slot = entry.get(slot_key)
                if slot and slot["message_id"] == payload.message_id:
                    entry[slot_key] = None
            return

        is_mute = (emoji_str in MUTE_TURN_EMOJI)
        is_skip = (emoji_str in SKIP_PARTICIPANT_EMOJI)
        is_regen = (emoji_str == REGENERATE_EMOJI)
        is_next = (emoji_str == NEXT_SPEAKER_EMOJI)
        is_continue = (emoji_str == CONTINUE_ROUND_EMOJI)

        # Unskipping lives on the removal, so gating only the add would have left the
        # half of the control that *undoes* an administrator's skip open to everyone --
        # and pulling a denied ❌ back off raises this very event for the member who was
        # just refused, handing them the unskip on the way out. `payload.member` is not
        # populated on removals, so this leans on the member cache the members intent
        # fills; an uncached member is refused, which leaves the participant skipped
        # rather than silenced-then-freed by a stranger.
        if is_skip and not self._reaction_user_is_admin(payload):
            return

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
                # Same derivation as the mute path above, in reverse.
                self.session_manager._recompute_pending_whispers(session)
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


    async def _handle_train_reaction(self, payload: discord.RawReactionActionEvent, is_input_slot: bool) -> bool:
        """React 1️⃣/2️⃣ into a training example while /train has armed this channel.

        Works whether or not the reacted message belongs to any active/logged session --
        by design, per /train, neither the reacted messages' session membership nor their
        authorship matters.
        """
        # (channel, reactor). Only the user who ran /train counts -- authorship of the
        # reacted messages is irrelevant, so anyone else's reaction would otherwise let
        # them write training data onto someone else's profile. That used to be a
        # separate `armed_by` comparison against the channel's single entry; the key
        # carries it now, which is also what lets two people arm the same channel.
        arm_key = (payload.channel_id, payload.user_id)
        entry = self.armed_training_channels.get(arm_key)
        if not entry:
            return False

        if time.time() - entry["last_activity"] > TRAIN_ARM_TIMEOUT_SECONDS:
            self.armed_training_channels.pop(arm_key, None)
            return False

        text, is_user = await self._resolve_reacted_message_text(payload)
        if text is None:
            return False

        slot_key = "slot1" if is_input_slot else "slot2"
        other_key = "slot2" if is_input_slot else "slot1"
        other = entry.get(other_key)
        if other and other["message_id"] == payload.message_id:
            return False  # same message reacted with both emoji -- not a valid pair

        entry[slot_key] = {
            "message_id": payload.message_id,
            "text": text[:1000 if is_input_slot else 2000],
            "is_user": is_user,
        }
        entry["last_activity"] = time.time()

        # Same indicator pattern as mute/skip: react back with the same emoji immediately
        # so the armer sees the reaction was registered. Cleared once the pair commits.
        async def _ack():
            try:
                ch = self.bot.get_channel(payload.channel_id)
                msg = await ch.fetch_message(payload.message_id)
                await msg.add_reaction(payload.emoji)
            except Exception:
                pass
        asyncio.create_task(_ack())

        if entry.get("slot1") and entry.get("slot2"):
            await self._commit_training_pair(arm_key, entry)
        return True

    async def _resolve_reacted_message_text(self, payload: discord.RawReactionActionEvent):
        """The text and (if known) speaker of the reacted message, for /train capture.

        Prefers the already-cleaned unified_log turn when this channel has a hydrated
        session and the message is in it -- no I/O. Otherwise falls back to fetching the
        raw Discord message; a session is deliberately never force-hydrated just to check,
        since that would defeat the point of /train working independent of any session.
        """
        session = self.multi_profile_channels.get(payload.channel_id)
        if session and session.get("is_hydrated"):
            for turn in session.get("unified_log", []):
                if payload.message_id in turn.get("message_ids", []):
                    return turn.get("content", ""), turn.get("is_user")

        try:
            ch = self.bot.get_channel(payload.channel_id)
            if ch is None:
                return None, None
            msg = await ch.fetch_message(payload.message_id)
            return msg.content, None
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            return None, None

    async def _commit_training_pair(self, arm_key: Tuple[int, int], entry: Dict[str, Any]):
        channel_id = arm_key[0]
        s1, s2 = entry["slot1"], entry["slot2"]
        warn = ""
        if s1["is_user"] is not None and s2["is_user"] is not None and s1["is_user"] == s2["is_user"]:
            warn = "\n⚠️ Both messages look like they're from the same side of the conversation."

        success, msg = await self.memory_manager.add_new_training_example(
            entry["owner_id"], entry["profile_name"], s1["text"], s2["text"], entry["guild_id"])

        entry["slot1"] = None
        entry["slot2"] = None

        disarmed = False
        if not success and "Limit Reached" in msg:
            self.armed_training_channels.pop(arm_key, None)
            msg += " Disarming your `/train` in this channel."
            disarmed = True

        interaction = entry.get("interaction")
        if interaction:
            content = f"{'✅' if success else '❌'} {msg}{warn}"
            if success and not disarmed:
                content += "\n\nStill armed -- react to capture another pair."
            try:
                await interaction.edit_original_response(content=content, embed=None, view=None)
            except discord.HTTPException:
                pass

        ch = self.bot.get_channel(channel_id)
        if ch:
            # Each message only ever carried its own slot's emoji (the armer's reaction
            # plus this bot's own ack), so clearing that one emoji per message is enough --
            # same clear_reaction(emoji) call a queued reaction trigger makes once it's
            # actually picked up, in cogs/services/generation/triggers.py.
            async def _cleanup():
                try:
                    m1 = await ch.fetch_message(s1["message_id"])
                    await m1.clear_reaction(TRAIN_INPUT_EMOJI)
                except Exception:
                    pass
                try:
                    m2 = await ch.fetch_message(s2["message_id"])
                    await m2.clear_reaction(TRAIN_OUTPUT_EMOJI)
                except Exception:
                    pass
            asyncio.create_task(_cleanup())

    @commands.Cog.listener()
    async def on_raw_message_delete(self, payload: discord.RawMessageDeleteEvent):
        if not self.has_lock:
            return

        deleted_message_id = payload.message_id
        if deleted_message_id in self.purged_message_ids:
            self.purged_message_ids.pop(deleted_message_id, None)
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
                            session_data['unified_log'] = [t for t in session_data['unified_log'] if t.get('turn_id') != turn_id_to_delete]

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
                    for p in session.get("profiles", []):
                        full_session_key = (payload.channel_id, p['owner_id'], p['profile_name'])
                        self.ltm_recall_history.pop(full_session_key, None)
                else:
                    await self.session_manager._save_session_to_disk(dummy_session_key, session_type, session["unified_log"])

                # See the note on _recompute_pending_whispers: this is what the
                # rebuild used to derive, and all it derived.
                self.session_manager._recompute_pending_whispers(session)

            self.session_last_accessed[payload.channel_id] = time.time()

    @commands.Cog.listener()
    async def on_raw_bulk_message_delete(self, payload: discord.RawBulkMessageDeleteEvent):
        """Handles MESSAGE_DELETE_BULK, which on_raw_message_delete never receives.

        Discord fires MESSAGE_DELETE for one deletion and MESSAGE_DELETE_BULK for two or
        more, and discord.py's purge() switches endpoints at exactly that boundary
        (delete_messages: "if the number of messages is 1 then single message delete is
        done. If it's more than two, then bulk delete is used."). With no listener here,
        /purge 1 and /purge 2+ behaved differently: single deletes were reconciled, bulk
        deletes were not seen at all. Two consequences, both fixed by this method --
        purged ids were never cleared from purged_message_ids, and a bulk delete
        performed by anything *else* (a moderation bot, a manual sweep) left its turns
        in unified_log permanently.

        Batched on purpose: one pass over the log and at most one disk write for the
        whole payload, rather than running the single-delete path once per message.

        No global-chat branch, unlike on_raw_message_delete -- Discord's bulk delete is
        a guild-channel endpoint and never fires for DMs.
        """
        if not self.has_lock:
            return

        message_ids = set(payload.message_ids)

        # Ids we deleted ourselves were already reconciled by /purge itself. Drop them
        # and act only on the remainder.
        ours = {mid for mid in message_ids if mid in self.purged_message_ids}
        for mid in ours:
            self.purged_message_ids.pop(mid, None)
        message_ids -= ours
        if not message_ids:
            return

        session = self.multi_profile_channels.get(payload.channel_id)
        if not session:
            return

        session_type = session.get("type", "multi")
        if not session.get("is_hydrated"):
            session = await self.session_manager._ensure_session_hydrated(payload.channel_id, session_type)
        if not session:
            return

        unified_log = session.get("unified_log", [])
        turns_to_delete = [
            turn for turn in unified_log
            if any(mid in message_ids for mid in turn.get("message_ids", []))
        ]
        if not turns_to_delete:
            return

        pid_to_profile = {}
        for p in session.get('profiles', []):
            pid = self.profile_manager._get_pid_from_name_any(p['owner_id'], p['profile_name'])
            pid_to_profile.setdefault(pid, p)

        for turn_obj in turns_to_delete:
            if turn_obj.get("is_user") is False:
                p = pid_to_profile.get(turn_obj.get("speaker_pid"))
                if p:
                    p['ltm_counter'] = max(0, p.get('ltm_counter', 0) - 1)

        # Identity, not turn_id -- see the note on /purge in CLAUDE.md.
        doomed = {id(turn) for turn in turns_to_delete}
        session["unified_log"] = [t for t in unified_log if id(t) not in doomed]

        is_effectively_empty = not session.get("unified_log") or all(
            turn.get("type") in ["whisper", "private_response"] for turn in session.get("unified_log", [])
        )

        dummy_session_key = (payload.channel_id, None, None)
        if is_effectively_empty:
            await self.session_manager._delete_session_from_disk(dummy_session_key, session_type)
            for p in session.get("profiles", []):
                full_session_key = (payload.channel_id, p['owner_id'], p['profile_name'])
                self.ltm_recall_history.pop(full_session_key, None)
        else:
            await self.session_manager._save_session_to_disk(dummy_session_key, session_type, session["unified_log"])

        self.session_manager._recompute_pending_whispers(session)
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
            # We only rewrite plain user messages.
            #
            # `is not True`, not `is False`. A bot turn written before `is_user` existed
            # reads back as absent, and `None is False` is False -- so the guard let it
            # through and this handler rewrote the profile's own turn as if a user had
            # typed it: the webhook's display name, the rendered text with "(edited)"
            # appended, and no entity id. Regeneration edits the bot's message through
            # the webhook, which fires this event, so regenerating a turn from an older
            # session was enough to trigger it.
            #
            # Whispers and private responses carry is_user but are not channel messages
            # either; rewriting one here would flatten its type and leak it.
            if turn_object.get("is_user") is not True or turn_object.get("type"):
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
            # The same hash triggers.py stamped on the turn when it was first written.
            # Omitting it rewrote a real user id as "00000000" on every message edit,
            # so an edited message stopped matching the author it came from.
            new_history_line = _format_history_entry(
                msg.author.display_name, original_ts, new_content, user_tz,
                entity_id=_get_user_hash(author_id))
            
            turn_object["content"] = new_history_line
            
            # Flush changes to disk
            dummy_session_key = (channel_id, None, None)
            await self.session_manager._save_session_to_disk(dummy_session_key, session_type, session["unified_log"])
            
            # No rebuild here. "Re-hydrate so all participant histories get the updated
            # log context" predates Migration 1 -- there are no per-participant history
            # objects left to refresh, every reader derives from unified_log, and the
            # edit above is already in it. The whisper derivation reads type/target_pid/
            # speaker_pid/is_hidden and never content, so it cannot change either.
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

        await self.child_bot_manager.shutdown_all()

        from ..services.tools_service import close_url_fetch_client
        await close_url_fetch_client()

        from ..services.api_service import close_google_rest_client
        await close_google_rest_client()

        from ..utils.http_client import close_shared_client
        await close_shared_client()

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
        
        self.dirty_sessions.clear()
        for session_key, session_data in self.global_chat_sessions.items():
            await self.session_manager._save_session_to_disk(session_key, 'global_chat', session_data)
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

        # Two phases, deliberately. Phase one collects (value, matchable text) pairs and
        # ranks them; phase two formats only the survivors. Matching used to be a plain
        # substring test -- which returned nothing at all for a typo, and returned an
        # arbitrary 25 in insertion order when a user had many profiles, so an exact
        # match could be truncated away. Formatting also ran for every match rather than
        # the 25 that survive, and each one costs an appearance lookup that can reach
        # disk for a borrowed profile.
        pending: List[Tuple[str, str]] = []
        meta: Dict[str, Any] = {}

        if cmd_name in ["speak", "whisper", "memorise"]:
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

                val = f"{o_id}:{p_name}"
                # Participants are capped at MAX_MULTI_PROFILES, so resolving appearance
                # up front here is bounded and lets the display name be matched on too --
                # which the old code did for these commands and only these.
                pending.append((val, p_name))
                pending.append((val, disp_name))
                meta[val] = {"owner": o_id, "name": p_name, "eff_owner": eff_owner, "disp": disp_name}

            ranked = rank_keyed(current, pending, limit=25)
            for val, _ in ranked:
                m = meta[val]
                pid = self.profile_manager._get_pid_from_name_any(m["owner"], m["name"])
                is_sys = pid.startswith("X")
                creator_name = "Unknown"
                if not is_sys:
                    creator = self.bot.get_user(m["eff_owner"])
                    creator_name = creator.name if creator else str(m["eff_owner"])
                choices.append(app_commands.Choice(
                    name=format_choice_name(m["disp"], m["name"], pid, creator_name, is_sys),
                    value=val))
            return choices

        elif cmd_name == "global_chat":
            # Gated on the content rating, which is what the command itself checks.
            # This used to offer only profiles listed in the Public Library -- the
            # proxy the command dropped when content_capability became the single
            # gate -- so a user whose profiles were rated but unpublished got an
            # empty menu for profiles the command accepts, which is every user who
            # has never published. Publication is not required to talk to your own
            # profile privately.
            #
            # The rating check reads the profile body (and the source's, for a
            # borrow), so it runs on the <=25 ranked survivors in the shared loop
            # below rather than over the whole index on every keystroke.
            user_id = interaction.user.id
            index = self.profile_manager._get_user_index(user_id)

            for p_name in index.get("personal", []):
                pending.append((p_name, p_name))
                meta[p_name] = {"kind": "personal", "pid": None, "needs_global_chat": True}

            for b_name in index.get("borrowed", []):
                pending.append((b_name, b_name))
                meta[b_name] = {"kind": "borrowed", "pid": None, "needs_global_chat": True}

        else:
            user_id = interaction.user.id
            index = self.profile_manager._get_user_index(user_id)

            for p_name in index.get("personal", []):
                pending.append((p_name, p_name))
                meta[p_name] = {"kind": "personal", "pid": None}

            for b_name in index.get("borrowed", []):
                pending.append((b_name, b_name))
                meta[b_name] = {"kind": "borrowed", "pid": None}

            # Offered to everyone, not just the bot owner. This read the CALLER's
            # index behind an owner-only gate, and a member's index has no "system"
            # map, so System profiles never appeared for anyone but the owner --
            # while _resolve_effective_profile accepted them from everyone. The
            # names were usable and undiscoverable.
            system_index = self.profile_manager._system_index()
            for s_name, s_pid in system_index.items():
                # Personal and borrowed shadow System, so a name the user has
                # already claimed is offered once, as theirs.
                if not self.profile_manager._is_system_name(user_id, s_name):
                    continue
                pending.append((s_name, s_name))
                meta[s_name] = {"kind": "system", "pid": s_pid}

        # Shared formatting for the two owned-profile branches. Only the ranked
        # survivors reach here, so the per-candidate config reads below are bounded at
        # 25 no matter how many profiles the account holds.
        user_id = interaction.user.id
        for name, _ in rank_keyed(current, pending, limit=25):
            m = meta[name]
            kind = m["kind"]

            if m.get("needs_global_chat") and not self.profile_manager.content_capability(
                    user_id, name, "global_chat")[0]:
                continue

            if kind == "system":
                app = self.profile_manager._get_user_appearance(
                    int(defaultConfig.DISCORD_OWNER_ID), name)
                disp_name = app.get("custom_display_name") or name
                choices.append(app_commands.Choice(
                    name=format_choice_name(disp_name, name, m["pid"], "System", True),
                    value=name))
                continue

            if kind == "personal":
                pid = m["pid"] or self.profile_manager._get_pid_from_name_any(user_id, name)
                app = self.profile_manager._get_user_appearance(user_id, name)
                disp_name = app.get("custom_display_name") or name
                choices.append(app_commands.Choice(
                    name=format_choice_name(disp_name, name, pid, interaction.user.name, False),
                    value=name))
                continue

            b_cfg = self.profile_manager._get_profile_config(user_id, name, True) or {}
            orig_oid = m.get("orig_oid") or b_cfg.get("original_owner_id", user_id)
            orig_pid = m["pid"] or b_cfg.get("original_pid") or b_cfg.get("original_profile_id", "Unknown")
            eff_owner, eff_name = self.profile_manager._resolve_effective_profile(user_id, name)
            app = self.profile_manager._get_user_appearance(eff_owner, eff_name)
            disp_name = app.get("custom_display_name") or eff_name
            creator = self.bot.get_user(int(orig_oid))
            c_name = creator.name if creator else str(orig_oid)
            choices.append(app_commands.Choice(
                name=format_choice_name(disp_name, name, orig_pid, c_name, str(orig_pid).startswith("X")),
                value=name))

        return choices
