import time
import uuid
import asyncio
import discord
import traceback
from typing import Optional, List

from ...utils.constants import PLACEHOLDER_EMOJI
from ...utils.helpers import _split_into_sentences_with_abbreviations, _yield_message_chunks


class DeliveryMixin:
    """Discord message delivery: webhook/appearance resolution, realistic-typing
    simulation, chunked sending, and turn-warning dispatch (to a channel or a child bot).
    """

    async def _send_channel_message(self,
                                   channel: discord.abc.Messageable,
                                   content: str,
                                   embeds: Optional[List[discord.Embed]] = None,
                                   reply_to: Optional[discord.Message] = None,
                                   mention_user: bool = False,
                                   store_prompt_for_id: Optional[str] = None,
                                   target_message_to_edit: Optional[discord.Message] = None,
                                   profile_owner_id_for_appearance: Optional[int] = None,
                                   profile_name_for_appearance: Optional[str] = None,
                                   file: Optional[discord.File] = None,
                                   bypass_typing: bool = False # [NEW] Flag to skip delays
                                   ) -> List[discord.Message]:

        if not content.strip() and target_message_to_edit:
            try:
                await target_message_to_edit.delete()
            except (discord.NotFound, discord.Forbidden):
                pass
            return []

        is_placeholder = (content == f"{PLACEHOLDER_EMOJI}")

        # Seeded before the appearance block, which is the only place they were bound.
        # A caller that passes no profile -- the unrestricted-in-general-channel refusal
        # notice in _multi_profile_worker is the one such call site -- skipped that block
        # entirely and then hit `if is_realistic_typing` and `if use_webhook`, raising
        # UnboundLocalError while trying to deliver the very message explaining the
        # refusal. Defaults describe a plain bot-authored message with no typing
        # simulation, which is what a system notice should be.
        use_webhook = False
        is_realistic_typing = False
        typing_cps = 30.0
        typing_max_delay = 2.5
        custom_display_name_to_use = None
        custom_avatar_url_to_use = None

        if profile_owner_id_for_appearance is not None and profile_name_for_appearance:
            index = self.cog.profile_manager._get_user_index(profile_owner_id_for_appearance)
            is_borrowed = profile_name_for_appearance in index.get("borrowed", [])
            profile_data_to_use = self.cog.profile_manager._get_profile_config(profile_owner_id_for_appearance, profile_name_for_appearance, is_borrowed) or {}
            
            custom_emoji = profile_data_to_use.get("placeholder_emoji")
            if is_placeholder and custom_emoji:
                content = custom_emoji

            is_realistic_typing = profile_data_to_use.get("realistic_typing_enabled", False)
            typing_cps = profile_data_to_use.get("typing_cps", 30.0)
            typing_max_delay = profile_data_to_use.get("typing_max_delay", 2.5)

            effective_owner_id, effective_profile_name = self.cog.profile_manager._resolve_effective_profile(profile_owner_id_for_appearance, profile_name_for_appearance)

            owner_id_str = str(effective_owner_id)
            appearance_data = self.cog.profile_manager._get_user_appearance(effective_owner_id, effective_profile_name)

            # Seeded before the branch below reads it. The `or custom_display_name_to_use`
            # fallback on the custom_display_name line is only reached when the profile has
            # a custom avatar but no custom display name, and the name was unbound there —
            # an UnboundLocalError on that one combination. The profile name is the same
            # fallback the elif branch uses.
            custom_display_name_to_use = profile_name_for_appearance

            if appearance_data and (appearance_data.get("custom_display_name") or appearance_data.get("custom_avatar_url")):
                use_webhook = True
                custom_display_name_to_use = appearance_data.get("custom_display_name") or custom_display_name_to_use

                if appearance_data.get("custom_avatar_url"):
                    custom_avatar_url_to_use = appearance_data["custom_avatar_url"]
                else:
                    avatar_index = hash(effective_profile_name) % 6
                    custom_avatar_url_to_use = f"https://cdn.discordapp.com/embed/avatars/{avatar_index}.png"

            elif profile_name_for_appearance:
                use_webhook = True
                custom_display_name_to_use = profile_name_for_appearance
                avatar_index = hash(profile_name_for_appearance) % 6
                custom_avatar_url_to_use = f"https://cdn.discordapp.com/embed/avatars/{avatar_index}.png"

        # [FIX] Never execute realistic typing on placeholder dispatches or when bypassed
        if is_placeholder or bypass_typing:
            is_realistic_typing = False

        if target_message_to_edit and use_webhook and content != f"{PLACEHOLDER_EMOJI}":
            try:
                webhook_for_delete = await self.cog.server_manager._get_or_create_webhook(channel) if isinstance(channel, (discord.TextChannel, discord.Thread)) else None
                if webhook_for_delete:
                    try:
                        await webhook_for_delete.delete_message(target_message_to_edit.id)
                    except discord.NotFound:
                        pass
                    except discord.HTTPException:
                        await target_message_to_edit.delete()
                else:
                    await target_message_to_edit.delete()
            except discord.NotFound:
                pass
            except Exception as e:
                print(f"Failed to delete placeholder message {target_message_to_edit.id}: {e}")
            target_message_to_edit = None

        sent_messages_list: List[discord.Message] = []

        if file and not is_realistic_typing:
            if not content.endswith("\n​"):
                content += "\n​"

        if is_realistic_typing and isinstance(channel, (discord.TextChannel, discord.Thread)) and content != f"{PLACEHOLDER_EMOJI}":
            try:
                webhook_to_use = await self.cog.server_manager._get_or_create_webhook(channel)
                if not webhook_to_use:
                    raise ValueError("Could not get webhook for realistic typing.")

                typing_mode = "sentence"
                if profile_owner_id_for_appearance is not None and profile_name_for_appearance:
                    index = self.cog.profile_manager._get_user_index(profile_owner_id_for_appearance)
                    is_borrowed = profile_name_for_appearance in index.get("borrowed", [])
                    p_config = self.cog.profile_manager._get_profile_config(profile_owner_id_for_appearance, profile_name_for_appearance, is_borrowed) or {}
                    typing_mode = p_config.get("typing_mode", "sentence")

                chunks = _split_into_sentences_with_abbreviations(content)

                displayed_text = ""
                last_edit_time = 0
                sent_message = None

                try:
                    typing_cps_float = float(typing_cps)
                    if typing_cps_float <= 0: typing_cps_float = 30.0
                except: typing_cps_float = 30.0

                try:
                    typing_max_delay_float = float(typing_max_delay)
                except: typing_max_delay_float = 2.5

                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue

                    delay = max(0.5, min(len(chunk) / typing_cps_float, typing_max_delay_float))
                    await asyncio.sleep(delay)

                    separator = "\n" if typing_mode == "line" and displayed_text else (" " if displayed_text else "")

                    if len(displayed_text) + len(separator) + len(chunk) > 2000:
                        displayed_text = chunk
                        sent_message = None
                    else:
                        displayed_text += separator + chunk

                    if not sent_message:
                        send_kwargs = {
                            "content": displayed_text,
                            "username": custom_display_name_to_use,
                            "avatar_url": custom_avatar_url_to_use,
                            "embeds": embeds if (i == 0 and embeds) else [],
                            "wait": True
                        }
                        if i == 0 and file: send_kwargs["file"] = file
                        if isinstance(channel, discord.Thread): send_kwargs["thread"] = channel

                        if i == 0 and target_message_to_edit:
                            try: await target_message_to_edit.delete()
                            except: pass

                        try:
                            sent_message = await webhook_to_use.send(**send_kwargs)
                        except RuntimeError as e:
                            if "Session is closed" in str(e): return sent_messages_list
                            raise
                        sent_messages_list.append(sent_message)
                        if i == 0 and store_prompt_for_id:
                            self.cog.message_id_to_original_prompt[sent_message.id] = store_prompt_for_id
                        last_edit_time = time.monotonic()
                    else:
                        now = time.monotonic()
                        if now - last_edit_time < 1.5:
                            await asyncio.sleep(1.5 - (now - last_edit_time))

                        try:
                            await webhook_to_use.edit_message(sent_message.id, content=displayed_text)
                        except RuntimeError as e:
                            if "Session is closed" in str(e): return sent_messages_list
                            print(f"Typing edit failed: {e}")
                        except Exception as e:
                            print(f"Typing edit failed: {e}")
                        last_edit_time = time.monotonic()

                return sent_messages_list
            except asyncio.CancelledError:
                # Flush the remaining un-sent content immediately on cancel
                try:
                    send_kwargs = {
                        "content": content,
                        "username": custom_display_name_to_use,
                        "avatar_url": custom_avatar_url_to_use,
                        "embeds": embeds,
                        "wait": True
                    }
                    if file: send_kwargs["file"] = file
                    if isinstance(channel, discord.Thread): send_kwargs["thread"] = channel

                    if target_message_to_edit:
                        try: await target_message_to_edit.delete()
                        except: pass

                    # If we already sent a partial message, edit it to show the full text instead of double sending
                    if sent_messages_list:
                        await webhook_to_use.edit_message(sent_messages_list[0].id, content=content)
                    else:
                        sent_msg = await webhook_to_use.send(**send_kwargs)
                        sent_messages_list.append(sent_msg)
                except RuntimeError as e:
                    if "Session is closed" not in str(e):
                        print(f"Failed to flush realistic typing on cancel: {e}")
                except Exception as flush_err:
                    print(f"Failed to flush realistic typing on cancel: {flush_err}")
                return sent_messages_list
            except asyncio.CancelledError:
                # Discard cleanly if cancelled during placeholder setup without forcing un-sent text
                if not is_placeholder and sent_messages_list and webhook_to_use:
                    try:
                        await webhook_to_use.edit_message(sent_messages_list[0].id, content=content)
                    except Exception:
                        pass
                return sent_messages_list
            except Exception as e:
                print(f"Realistic typing failed, falling back to standard send. Error: {e}")
                traceback.print_exc()

        if reply_to and not is_placeholder and profile_owner_id_for_appearance is not None and profile_name_for_appearance:
            index = self.cog.profile_manager._get_user_index(profile_owner_id_for_appearance)
            is_borrowed = profile_name_for_appearance in index.get("borrowed", [])
            target_profile_settings = self.cog.profile_manager._get_profile_config(profile_owner_id_for_appearance, profile_name_for_appearance, is_borrowed) or {}

            rmode = target_profile_settings.get("response_mode", "regular")
            if rmode in ["mention", "mention_reply"]:
                content = f"{reply_to.author.mention} {content}"

        is_first_chunk = True

        for chunk in _yield_message_chunks(content):
            current_target_to_edit = target_message_to_edit if is_first_chunk else None
            current_reply_to = reply_to if is_first_chunk else None
            current_store_prompt = store_prompt_for_id if is_first_chunk else None
            current_embeds_for_api = embeds if is_first_chunk and embeds else []
            current_file_for_api = file if is_first_chunk and file else None

            if current_file_for_api and current_target_to_edit:
                try:
                    await current_target_to_edit.delete()
                except (discord.NotFound, discord.Forbidden):
                    pass
                current_target_to_edit = None

            final_content_for_send = chunk

            sent_message_part: Optional[discord.Message] = None

            webhook_to_use = None
            if use_webhook and isinstance(channel, (discord.TextChannel, discord.Thread)):
                webhook_to_use = await self.cog.server_manager._get_or_create_webhook(channel)

            if webhook_to_use:
                try:
                    send_kwargs = {
                        "content": final_content_for_send,
                        "username": custom_display_name_to_use,
                        "avatar_url": custom_avatar_url_to_use,
                        "embeds": current_embeds_for_api,
                        "wait": True
                    }
                    if current_file_for_api: send_kwargs["file"] = current_file_for_api
                    if isinstance(channel, discord.Thread): send_kwargs["thread"] = channel

                    sent_message_part = await webhook_to_use.send(**send_kwargs)
                except RuntimeError as e:
                    if "Session is closed" in str(e): return sent_messages_list
                    print(f"Webhook send failed, falling back to regular message. Error: {e}")
                    sent_message_part = None
                except Exception as e:
                    print(f"Webhook send failed, falling back to regular message. Error: {e}")
                    sent_message_part = None

            if not sent_message_part:
                try:
                    if current_target_to_edit:
                        try:
                            sent_message_part = await current_target_to_edit.edit(content=final_content_for_send, embeds=current_embeds_for_api)
                        except discord.HTTPException:
                            sent_message_part = await channel.send(final_content_for_send, embeds=current_embeds_for_api, file=current_file_for_api)
                    else:
                        sent_message_part = await channel.send(final_content_for_send, embeds=current_embeds_for_api, file=current_file_for_api)
                except RuntimeError as e:
                    if "Session is closed" in str(e): return sent_messages_list
                    raise

            if sent_message_part:
                sent_messages_list.append(sent_message_part)
                if current_store_prompt:
                    self.cog.message_id_to_original_prompt[sent_message_part.id] = store_prompt_for_id

            is_first_chunk = False
            await asyncio.sleep(0.5)

        return sent_messages_list

    async def _dispatch_warnings(self, channel, participant_method, bot_id, warnings, owner_id, profile_name, session=None, turn_object=None):
        if not warnings: return
        warning_text = "\n".join([f"-# {i+1}. {w}" for i, w in enumerate(warnings)])
        try:
            if participant_method == 'child_bot' and bot_id:
                corr_id = str(uuid.uuid4())
                if turn_object and session:
                    self.cog.pending_child_confirmations[corr_id] = {
                        "event": asyncio.Event(), "type": "multi_profile", "participant": {"bot_id": bot_id},
                        "channel_id": channel.id, "turn_id": turn_object["turn_id"]
                    }
                await self.cog.manager_queue.put({
                    "action": "send_to_child", "bot_id": bot_id,
                    "payload": {
                        "action": "send_message", "channel_id": channel.id,
                        "content": warning_text, "realistic_typing": False,
                        "reply_to_id": None, "ping": False,
                        "correlation_id": corr_id if (turn_object and session) else None
                    }
                })
            else:
                sent_msgs = await self._send_channel_message(
                    channel, warning_text,
                    profile_owner_id_for_appearance=owner_id,
                    profile_name_for_appearance=profile_name,
                    bypass_typing=True
                )
                if sent_msgs and turn_object and session:
                    turn_object.setdefault("message_ids", []).extend([m.id for m in sent_msgs])
                    await self.cog.session_manager._save_session_to_disk((channel.id, None, None), session.get("type", "multi"), session.get("unified_log", []))
        except Exception as e:
            print(f"Failed to dispatch warnings: {e}")
