import uuid
import discord
import datetime

from ...utils.helpers import _format_debug_prompt, _format_history_entry, _get_user_hash
from ...utils.http_client import get_shared_client


class TriggerIntakeMixin:
    """Turns the raw triggers batched for one multi-profile round -- messages,
    reactions, replies, proactive kicks -- into the round's user-side history.
    """

    async def _collect_round_triggers(
        self, session, session_type, channel_id, all_triggers_for_round,
        new_round_turn_data, pending_url_fetches, recent_processed_ids,
        is_image_gen_round, image_gen_prompt, starting_profile_override,
        round_author_name, triggering_user_id,
    ):
        """Normalises the round's batched triggers into history turns.

        Appends to new_round_turn_data, pending_url_fetches and recent_processed_ids in
        place, and returns the five round-scoped values a trigger can change:
        (is_image_gen_round, image_gen_prompt, starting_profile_override,
        round_author_name, triggering_user_id). They are passed in as well as returned
        so a round whose triggers set none of them keeps the caller's values.

        The long parameter list is the real coupling this step has to the round, not a
        shape worth hiding: it was previously all ambient locals in the worker frame.
        """
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
                # Shared client: _process_text_attachments sets its own
                # per-request timeout, so nothing is lost by not owning one.
                text_att_content = await self.cog.media_service._process_text_attachments(
                    raw_att_list, get_shared_client()
                )

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

        return (is_image_gen_round, image_gen_prompt, starting_profile_override,
                round_author_name, triggering_user_id)
