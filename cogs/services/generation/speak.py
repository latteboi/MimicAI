import time
import uuid
import discord

from ...utils.helpers import _format_history_entry


class SpeakAsMixin:
    """The `/speak` one-off message injection."""

    async def _execute_speak_as(self, interaction_to_respond: discord.Interaction, channel: discord.abc.Messageable, author: discord.User, profile_name: str, message: str, method: str):
        user_id = author.id
        index = self.cog.profile_manager._get_user_index(user_id)

        is_borrowed = profile_name in index.get("borrowed", [])
        is_personal = profile_name in index.get("personal", [])

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
            borrowed_data = self.cog.profile_manager._get_profile_config(user_id, profile_name, True) or {}
            effective_owner_id = int(borrowed_data.get("original_owner_id", user_id))
            effective_profile_name = borrowed_data.get("original_profile_name", profile_name)
            profile_data_source = self.cog.profile_manager._get_profile_config(effective_owner_id, effective_profile_name, False) or {}
        else:
            profile_data_source = self.cog.profile_manager._get_profile_config(user_id, profile_name, False) or {}

        if not self.cog.profile_manager._check_unrestricted_safety_policy(effective_owner_id, effective_profile_name, channel):
            await interaction_to_respond.followup.send("This profile has an 'Unrestricted 18+' safety level and cannot speak in this channel because it is not marked as Age-Restricted.", ephemeral=True)
            return

        delivery_method = 'webhook'
        child_bot_id = None

        session = self.cog.multi_profile_channels.get(channel.id)
        
        # [NEW] Force hydration if session exists or might exist on disk
        if not session or not session.get("is_hydrated"):
            session = await self.cog.session_manager._ensure_session_hydrated(channel.id, "multi")

        if session:
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
            linked_bot_id = next((bot_id for bot_id, data in self.cog.child_bots.items() if data.get("owner_id") == effective_owner_id and data.get("profile_name") == effective_profile_name), None)
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
        appearance_data = self.cog.profile_manager._get_user_appearance(effective_owner_id, effective_profile_name)
        if appearance_data.get("custom_display_name"):
            speaker_display_name = appearance_data["custom_display_name"]

        profile_id = self.cog.profile_manager._get_profile_id(effective_owner_id, effective_profile_name)
        history_line = _format_history_entry(speaker_display_name, interaction_to_respond.created_at, message, entity_id=profile_id)

        display_message = f"{message}\n\n||-# Authored by {author.mention} ({author.id}).||"

        turn_object = None
        if session:
            participant_key = (user_id, profile_name)
            model_content_obj = {'role': 'model', 'parts': [history_line]}
            user_content_obj = {'role': 'user', 'parts': [history_line]}

            turn_id = str(uuid.uuid4())
            turn_object = {
                "turn_id": turn_id,
                "is_user": False,
                "speaker_pid": self.cog.profile_manager._get_pid_from_name_any(user_id, profile_name),
                "owner_id": effective_owner_id,
                "profile_name": effective_profile_name,
                "message_ids": [],
                "content": history_line
            }
            session.setdefault("unified_log", []).append(turn_object)

            for key, chat_session in session["chat_sessions"].items():
                if key == participant_key:
                    chat_session.history.append(model_content_obj)
                else:
                    chat_session.history.append(user_content_obj)

            session_type = session.get("type", "multi")
            self.cog.session_last_accessed[channel.id] = time.time()
            await self.cog.session_manager._save_session_to_disk((channel.id, None, None), session_type, session["unified_log"])

        sent_messages = []
        if delivery_method == 'child_bot' and child_bot_id:
            correlation_id = str(uuid.uuid4())

            if session:
                participant_data = next((p for p in session.get("profiles", []) if p.get("owner_id") == user_id and p.get("profile_name") == profile_name), None)
                self.cog.pending_child_confirmations[correlation_id] = {
                    "type": "multi_profile", "participant": participant_data,
                    "history_line": history_line, "channel_id": channel.id, "turn_id": turn_id,
                    "is_speak_as": True
                }

            await self.cog.manager_queue.put({
                "action": "send_to_child", "bot_id": child_bot_id,
                "payload": {
                    "action": "send_message", "channel_id": channel.id, "content": display_message,
                    "realistic_typing": profile_data_source.get("realistic_typing_enabled", False),
                    "typing_cps": profile_data_source.get("typing_cps", 30.0),
                    "typing_max_delay": profile_data_source.get("typing_max_delay", 2.5),
                    "typing_mode": profile_data_source.get("typing_mode", "sentence"),
                    "correlation_id": correlation_id
                }
            })
        else:
            sent_messages = await self._send_channel_message(
                channel, display_message,
                profile_owner_id_for_appearance=effective_owner_id,
                profile_name_for_appearance=effective_profile_name
            )

        if sent_messages and turn_object:
            for msg in sent_messages:
                turn_object.setdefault("message_ids", []).append(msg.id)
            await self.cog.session_manager._save_session_to_disk((channel.id, None, None), session.get("type", "multi"), session["unified_log"])

        await interaction_to_respond.followup.send("Message sent.", ephemeral=True)