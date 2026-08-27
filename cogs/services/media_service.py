import os
import io
import time
import uuid
import asyncio
import traceback
import discord
import httpx
from typing import List, Any, Optional

from ..utils.constants import (
    PLACEHOLDER_EMOJI, WARN_IMAGE_GEN_FAILED, ERR_GENERAL_ERROR, ERR_SAFETY_BLOCK,
    WARN_MAIN_MODEL_FAILED, ERR_REASON_EMPTY_RESPONSE,
    defaultConfig, IMAGE_QUEUE_PRIORITY,
    DEFAULT_IMAGE_PRESENT, DEFAULT_IMAGE_FAILED, DEFAULT_IMAGE_APPEARANCE, DEFAULT_IMAGE_GROUNDING,
    DEFAULT_IMAGE_MODEL, DEFAULT_SPEECH_MODEL, DEFAULT_SPEECH_VOICE,
    IMAGE_OUTPUT_KEYS,
)
from .api_service import GoogleGenAIModel, generate_google_tts_audio
from ..utils.helpers import _add_inline_citations, _format_api_error, _format_citation_subtext, _resolve_safety_settings, _scrub_response_text, resolve_image_output_params
from ..utils.memory_tuning import maybe_trim_malloc


class MediaService:
    """Owns TTS audio generation/stitching and the image generation worker pipeline.

    Holds a back-reference to the parent cog for state/logic not yet migrated
    (queues, caches, the generation engine, and Discord message dispatch),
    per the transitional Dependency Injection pattern in CLAUDE.md.
    """

    def __init__(self, cog):
        self.cog = cog

    @staticmethod
    def resolve_image_output_params(image_config, raw_name: str) -> dict:
        """See helpers.resolve_image_output_params -- kept here as the name the image
        paths and their tests already reach for."""
        return resolve_image_output_params(image_config, raw_name)

    @staticmethod
    def build_image_model(raw_name: str, api_key: str, system_instruction, safety_settings,
                          image_config=None):
        """One image-model constructor for the three places that build one.

        All three stripped the GOOGLE/ prefix by hand and two of them defaulted to an
        unprefixed id that the third prefixed, so the "same" default was two different
        strings depending on which path reached it.

        `image_config` is the profile's raw IMAGE_OUTPUT_KEYS, not a validated payload:
        validation depends on `raw_name`, and this is the one place that knows both.
        """
        name = raw_name[7:] if raw_name.upper().startswith("GOOGLE/") else raw_name
        return GoogleGenAIModel(api_key=api_key, model_name=name,
                                system_instruction=system_instruction,
                                safety_settings=safety_settings,
                                image_params=MediaService.resolve_image_output_params(
                                    image_config, raw_name))

    async def _generate_google_tts(self, text: str, guild_id: int, model_id: str = DEFAULT_SPEECH_MODEL, voice_name: str = "Aoede", temperature: float = 1.0, fallback_model_id: Optional[str] = None):
        """Generates a playable WAV audio stream utilising Google Gemini Speech Generation models.

        One choke point for every TTS call in the bot, which is why the retry lives here
        rather than at each caller: a preview speech model going 404 or 503 mid-session
        otherwise silences the whole round with nothing tried in its place.
        """
        import wave
        api_key = self.cog.storage_manager._get_api_key_for_guild(guild_id)
        if not api_key:
            return None

        try:
            async def _attempt(name, _is_fallback):
                return await generate_google_tts_audio(
                    api_key, name, text, voice_name=voice_name, temperature=temperature)

            raw_audio_bytes, _used, _was_fallback = await self.cog.api_service.run_with_fallback(
                model_id, fallback_model_id, _attempt, label="Text-to-speech")

            if raw_audio_bytes:
                wav_io = io.BytesIO()
                with wave.open(wav_io, 'wb') as wav_file:
                    wav_file.setnchannels(1)      # Mono
                    wav_file.setsampwidth(2)      # 16-bit
                    wav_file.setframerate(24000)  # 24kHz
                    wav_file.writeframes(raw_audio_bytes)
                wav_io.seek(0)
                return wav_io
            return None
        except Exception as e:
            err_msg = _format_api_error(e)
            if "404" not in err_msg:
                print(f"Google TTS Error: {err_msg}")
            return None

    def _stitch_wav_segments(self, segments):
        """Concatenates multiple WAV Byte streams into a single Master stream without re-encoding."""
        import wave
        output = io.BytesIO()
        if not segments: return output

        with wave.open(output, 'wb') as master:
            # Initialise master parameters from the first segment
            segments[0].seek(0)
            with wave.open(segments[0], 'rb') as first:
                master.setparams(first.getparams())

            for seg in segments:
                seg.seek(0)
                try:
                    with wave.open(seg, 'rb') as reader:
                        master.writeframes(reader.readframes(reader.getnframes()))
                except Exception as e:
                    print(f"Skipping corrupted audio segment: {e}")

        output.seek(0)
        return output

    async def _image_finisher_worker(self):
        """Consumes generated images, generates text, and sends the final message."""
        while True:
            try:
                # [FIXED] Unpack Priority Tuple
                item = await self.cog.text_request_queue.get()
                if item is None: break

                priority, _, package = item

                async with self.cog.image_gen_semaphore:
                    placeholder_message = package.get("placeholder_message")
                    final_response_text = "An error occurred."
                    image_file_to_send = None
                    sources_text_list = []

                    is_child_bot = package.get("is_child_bot", False)
                    channel = self.cog.bot.get_channel(package['channel_id'])
                    if not channel: self.cog.text_request_queue.task_done(); continue

                    # If this was a queued request, it won't have a placeholder yet. Create one now.
                    if not placeholder_message and not is_child_bot:
                        placeholders = await self.cog.generation_service._send_channel_message(
                            channel, f"{PLACEHOLDER_EMOJI}",
                            profile_owner_id_for_appearance=package['effective_profile_owner_id'],
                            profile_name_for_appearance=package['effective_profile_name']
                        )
                        placeholder_message = placeholders[0] if placeholders else None
                    elif is_child_bot and not package.get("reference_image_urls"):
                         p_index = self.cog.profile_manager._get_user_index(package['effective_profile_owner_id'])
                         p_is_b = package['effective_profile_name'] in p_index.get("borrowed", [])
                         p_settings = self.cog.profile_manager._get_profile_config(package['effective_profile_owner_id'], package['effective_profile_name'], p_is_b) or {}

                         if p_settings.get("child_bot_placeholder", False):
                             custom_emoji = p_settings.get("placeholder_emoji") or PLACEHOLDER_EMOJI
                             msg_id = await self.cog.generation_service._send_child_bot_placeholder(package['bot_id'], channel.id, custom_emoji)
                             if msg_id:
                                 try: placeholder_message = await channel.fetch_message(msg_id)
                                 except: pass
                         else:
                             # Typing was already started for non-ref images. For ref-images, start it now.
                             await self.cog.manager_queue.put({"action": "send_to_child", "bot_id": package['bot_id'], "payload": {"action": "start_typing", "channel_id": channel.id}})

                    try:
                        # --- Just-in-Time Generation for Reference Images ---
                        if package.get("reference_image_urls"):
                            image_data, failure_reason, response = None, None, None
                            # response is reset per request, not merely on the error path:
                            # these workers loop inside one frame, so a generation that
                            # raises before rebinding it would otherwise leave the
                            # *previous* request's response to be closed here.
                            try:
                                api_key = self.cog.storage_manager._get_api_key_for_guild(package['guild_id'])
                                if not api_key: raise ValueError("Server API key not configured.")

                                img_model_raw = package.get("image_generation_model", DEFAULT_IMAGE_MODEL)
                                img_fallback_raw = package.get("image_generation_fallback_model")
                                image_model = self.build_image_model(
                                    img_model_raw, api_key, package['system_instruction'],
                                    package['safety_settings'], package.get('image_output'))
                                parts = [package['prompt_text']]
                                for ref in package.get("reference_image_urls", []):
                                    parts.append({"url": ref["url"], "mime_type": ref.get("mime_type", "image/png")})

                                status = "api_error"
                                response = None
                                try:
                                    msg_a_id = placeholder_message.id if placeholder_message else None
                                    app_name = package.get("bot_display_name", "Bot")
                                    app_avatar = package.get("avatar_url")

                                    state_container = {
                                        'msg_a_id': msg_a_id,
                                        'msg_b_id': None,
                                        'app_name': app_name,
                                        'app_avatar': app_avatar,
                                        'message_type': "embed" if is_child_bot else "text",
                                        'custom_emoji': PLACEHOLDER_EMOJI
                                    }

                                    participant = {"method": "child_bot", "bot_id": package.get("bot_id")} if is_child_bot else None

                                    async def _attempt(raw_name, _is_fallback):
                                        nonlocal image_model
                                        image_model = self.build_image_model(
                                            raw_name, api_key, package['system_instruction'],
                                            package['safety_settings'], package.get('image_output'))
                                        # The state container is mutated in place, so a retry
                                        # re-uses the placeholder the first attempt created
                                        # rather than stacking a second one beside it.
                                        return await self.cog.generation_service._generate_with_heartbeat(
                                            image_model, [{'role': 'user', 'parts': parts}], None, channel, participant, msg_a_id, is_fallback=False, app_name=app_name, app_avatar=app_avatar, existing_state=state_container, message_type=state_container['message_type']
                                        )

                                    result, _used, _was_fallback = await self.cog.api_service.run_with_fallback(
                                        img_model_raw, img_fallback_raw, _attempt, label="Image generation")
                                    response, state_container = result
                                    status = "blocked_by_safety" if not response.candidates else "success"
                                except Exception as e:
                                    failure_reason = _format_api_error(e)
                                    status = "api_error"
                                finally:
                                    self.cog._log_api_call(user_id=package['author_id'], guild_id=package['guild_id'], context="image_generation_jit", model_used=image_model, status=status)

                                del parts

                                if response:
                                    if not response.candidates:
                                        reason = "Safety Filter";
                                        if response.prompt_feedback and response.prompt_feedback.block_reason: reason = response.prompt_feedback.block_reason.name.replace('_', ' ').title()
                                        failure_reason = f"the safety filter ({reason})"
                                    else:
                                        candidate = response.candidates[0]
                                        if candidate.finish_reason.name != 'STOP': failure_reason = f"the process being stopped for reason: **{candidate.finish_reason.name.replace('_', ' ').title()}**"
                                        else:
                                            image_data = next((part.inline_data.data for part in candidate.content.parts if getattr(part, 'inline_data', None) and part.inline_data.mime_type.startswith('image/')), None)
                                            if not image_data: failure_reason = "an unknown issue (the model returned no image data)"
                            except Exception as e:
                                if not failure_reason: failure_reason = f"an unexpected error: `{e}`"

                            if image_data:
                                # The image streamed straight to a file on the way
                                # off the socket (cogs/utils/blob_stream), so this
                                # takes ownership of that file rather than writing
                                # one. A small enough image is still bytes and is
                                # written here, as it always was.
                                package['generated_image_path'] = await self.cog.api_service.materialise_inline_data(
                                    response, image_data, ".png")
                                image_data = None
                            if response is not None:
                                # Unlinks any blob the package did not take -- what a
                                # safety block or a fallback attempt leaves behind.
                                try:
                                    response.close()
                                except Exception:
                                    pass
                                response = None
                            if package.get('generated_image_path'):
                                # Returns the freed pages to the OS rather than
                                # leaving them at the top of a glibc arena for the
                                # text generation that follows to sit on top of.
                                maybe_trim_malloc()

                            package['failure_reason'] = failure_reason

                        # --- Text Generation ---
                        owner_id = package['effective_profile_owner_id']
                        profile_name = package['effective_profile_name']

                        p_index = self.cog.profile_manager._get_user_index(owner_id)
                        is_borrowed = profile_name in p_index.get("borrowed", [])
                        profile_settings = self.cog.profile_manager._get_profile_config(owner_id, profile_name, is_borrowed) or {}

                        turn_warnings = []
                        if package.get('failure_reason'):
                            turn_warnings.append(WARN_IMAGE_GEN_FAILED.format(reason=package['failure_reason']))

                        text_model, _, temp, top_p, top_k, _, _ = await self.cog.api_service._get_or_create_model_for_channel(package['channel_id'], package['author_id'], package['guild_id'], profile_owner_override=package['effective_profile_owner_id'], profile_name_override=package['effective_profile_name'])

                        # Derived from the channel's unified_log. This previously read
                        # cog.chat_sessions, a cache that only this worker ever wrote to, so
                        # the image-presentation turn saw prior image-generation turns and
                        # nothing else of the conversation it was replying to.
                        img_session = self.cog.multi_profile_channels.get(package['channel_id']) or {}
                        img_bot_pid = self.cog.profile_manager._get_pid_from_name_any(
                            package['effective_profile_owner_id'], package['effective_profile_name']
                        )
                        contents_for_api_call = self.cog.session_manager._build_history_for_participant(
                            img_session.get("unified_log", []), img_bot_pid, profile_settings
                        )

                        turn_id = str(uuid.uuid4())

                        if package.get('generated_image_path') and not package.get('failure_reason'):
                            present_template = self.cog.global_prompts.get("IMAGE_PRESENT", DEFAULT_IMAGE_PRESENT)
                            system_note = present_template.format(prompt=package['prompt_text'])

                            final_user_parts = [
                                system_note,
                                {"mime_type": "image/png", "url": package['generated_image_path']}
                            ]

                            user_turn = {'role': 'user', 'parts': final_user_parts}
                        else:
                            if package.get('failure_reason'):
                                failed_template = self.cog.global_prompts.get("IMAGE_FAILED", DEFAULT_IMAGE_FAILED)
                                system_note = failed_template.format(prompt=package['prompt_text'], reason=package['failure_reason'])
                                user_turn = {'role': 'user', 'parts': [system_note]}
                            else:
                                user_turn = {'role': 'user', 'parts': [package['prompt_text']]}

                        contents_for_api_call.append(user_turn)
                        gen_config = {"temperature": temp, "top_p": top_p, "top_k": top_k}

                        msg_a_id = placeholder_message.id if placeholder_message else None
                        app_name = package.get("bot_display_name", "Bot")
                        app_avatar = package.get("avatar_url")

                        participant = {"method": "child_bot", "bot_id": package.get("bot_id")} if is_child_bot else None

                        state_container = {
                            'msg_a_id': msg_a_id,
                            'msg_b_id': state_container.get('msg_b_id') if 'state_container' in locals() and state_container else None,
                            'app_name': app_name,
                            'app_avatar': app_avatar,
                            'message_type': "embed" if is_child_bot else "text",
                            'custom_emoji': PLACEHOLDER_EMOJI
                        }

                        text_failure_reason = None
                        try:
                            text_response, state_container = await self.cog.generation_service._generate_with_heartbeat(
                                text_model, contents_for_api_call, gen_config, channel, participant, msg_a_id, is_fallback=False, app_name=app_name, app_avatar=app_avatar, existing_state=state_container, message_type=state_container['message_type']
                            )
                        except Exception as e:
                            if 'state_container' in locals() and state_container and state_container.get('sending_task'):
                                state_container['sending_task'].cancel()
                            text_response = None
                            text_failure_reason = _format_api_error(e)

                        response_text = "Here is the image you requested."
                        was_blocked = False
                        if not text_response or not text_response.candidates:
                            reason = "Unknown Error"
                            if text_response and text_response.prompt_feedback and text_response.prompt_feedback.block_reason:
                                reason = text_response.prompt_feedback.block_reason.name.replace('_', ' ').title()
                            elif text_failure_reason:
                                reason = text_failure_reason

                            custom_main = profile_settings.get("error_response", ERR_GENERAL_ERROR)
                            response_text = custom_main
                            turn_warnings.append(ERR_SAFETY_BLOCK.format(reason=reason) if "Safety" in reason else WARN_MAIN_MODEL_FAILED.format(reason=reason))
                            was_blocked = True
                        elif text_response.candidates[0].finish_reason.name == 'STOP':
                            raw_text = getattr(text_response, 'text', "")
                            if hasattr(text_response, 'raw') and text_response.raw.candidates and hasattr(text_response.raw.candidates[0], 'grounding_metadata'):
                                raw_text = _add_inline_citations(raw_text, text_response.raw.candidates[0].grounding_metadata)

                            raw_text, _ = self.cog.generation_service._extract_and_apply_neuro_state(raw_text, package['effective_profile_owner_id'], package['effective_profile_name'])

                            response_text = _scrub_response_text(raw_text.strip(), participant_names=[package['bot_display_name']])

                            if not response_text:
                                custom_main = profile_settings.get("error_response", ERR_GENERAL_ERROR)
                                response_text = custom_main
                                turn_warnings.append(ERR_SAFETY_BLOCK.format(reason=ERR_REASON_EMPTY_RESPONSE))
                                was_blocked = True
                            else:
                                grounding_sources = package.get("grounding_sources") or []
                                if hasattr(text_response, 'raw') and text_response.raw.candidates:
                                    if hasattr(text_response.raw.candidates[0], 'grounding_metadata'):
                                        metadata = text_response.raw.candidates[0].grounding_metadata
                                        if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks is not None:
                                            for chunk in metadata.grounding_chunks:
                                                if hasattr(chunk, 'web'):
                                                    grounding_sources.append({'uri': chunk.web.uri, 'title': chunk.web.title})

                                    if hasattr(text_response.raw.candidates[0], 'url_context_metadata'):
                                        url_metadata = text_response.raw.candidates[0].url_context_metadata
                                        if hasattr(url_metadata, 'url_metadata') and url_metadata.url_metadata is not None:
                                            for u in url_metadata.url_metadata:
                                                if hasattr(u, 'retrieved_url') and u.retrieved_url:
                                                    grounding_sources.append({'uri': u.retrieved_url, 'title': 'URL Context'})

                                sources_text_list = _format_citation_subtext(grounding_sources)

                        model_turn = {'role': 'model', 'parts': [response_text]}

                        # --- Update Placeholder ---
                        if not was_blocked and not is_child_bot and placeholder_message:
                            try: await placeholder_message.edit(content="-# Sending... (Uploading Media)")
                            except: pass

                        # --- Final Message Sending ---
                        if package.get('generated_image_path') and not package.get('failure_reason'):
                            image_file_to_send = discord.File(package['generated_image_path'], filename="generated_image.png")

                        final_response_text = response_text
                        is_realistic_typing = profile_settings.get("realistic_typing_enabled", False)

                    except Exception as e:
                        final_response_text = f"An unexpected error occurred in the finalization stage: {e}"; print(f"Error in finisher stage: {e}"); traceback.print_exc()

                    if is_child_bot:
                        if placeholder_message and hasattr(placeholder_message, 'id'):
                            try:
                                msg_to_del = await channel.fetch_message(placeholder_message.id)
                                await msg_to_del.delete()
                            except Exception: pass

                        correlation_id = str(uuid.uuid4())
                        self.cog.pending_child_confirmations[correlation_id] = {
                            "type": "single_profile", "user_turn": user_turn, "model_turn": model_turn,
                            "bot_id": package['bot_id'], "channel_id": channel.id, "turn_id": turn_id
                        }

                        # [UPDATED] Resolve Response Mode for Image Delivery
                        rmode = profile_settings.get("response_mode", "regular")

                        delivery_text = final_response_text
                        anchor_id = package.get("original_message_id")

                        # Handle text-based mention for child bots
                        if anchor_id and rmode == "mention":
                            try:
                                anchor_msg = await channel.fetch_message(anchor_id)
                                delivery_text = f"{anchor_msg.author.mention} {delivery_text}"
                            except: pass

                        reply_id = anchor_id if (anchor_id and rmode in ["reply", "mention_reply"]) else None
                        should_ping = (rmode == "mention_reply")

                        payload = {
                            "action": "send_message", "channel_id": channel.id, "content": delivery_text,
                            "correlation_id": correlation_id, "realistic_typing": is_realistic_typing,
                            "typing_cps": profile_settings.get("typing_cps", 30.0),
                            "typing_max_delay": profile_settings.get("typing_max_delay", 2.5),
                            "typing_mode": profile_settings.get("typing_mode", "sentence"),
                            "reply_to_id": reply_id, "ping": should_ping
                        }
                        if image_file_to_send:
                            # Hand over the path; execute_send opens it. Encoding a
                            # multi-megabyte PNG to base64 to move it across an
                            # in-process asyncio.Queue cost ~4x its size in live
                            # copies and bought nothing.
                            payload["attachment"] = {
                                "filename": "generated_image.png",
                                "path": package['generated_image_path'],
                            }
                        await self.cog.manager_queue.put({"action": "send_to_child", "bot_id": package['bot_id'], "payload": payload})
                    else:
                        # [UPDATED] Fix undefined 'i' by resolving anchor_message from package
                        anchor_msg = None
                        anchor_id = package.get("original_message_id")
                        if anchor_id:
                            try: anchor_msg = await channel.fetch_message(anchor_id)
                            except: pass

                        sent_messages = await self.cog.generation_service._send_channel_message(
                            channel, final_response_text, target_message_to_edit=placeholder_message,
                            profile_owner_id_for_appearance=package['effective_profile_owner_id'],
                            profile_name_for_appearance=package['effective_profile_name'],
                            file=image_file_to_send, reply_to=anchor_msg
                        )

                    for source_msg in sources_text_list:
                        if is_child_bot:
                            source_payload = {
                                "action": "send_message", "channel_id": channel.id,
                                "content": source_msg, "realistic_typing": False,
                                "reply_to_id": None, "ping": False
                            }
                            await self.cog.manager_queue.put({"action": "send_to_child", "bot_id": package['bot_id'], "payload": source_payload})
                        else:
                            await self.cog.generation_service._send_channel_message(
                                channel, source_msg, target_message_to_edit=None, bypass_typing=True,
                                profile_owner_id_for_appearance=package['effective_profile_owner_id'],
                                profile_name_for_appearance=package['effective_profile_name']
                            )

                    await self.cog.generation_service._dispatch_warnings(channel, 'child_bot' if is_child_bot else 'webhook', package.get('bot_id'), turn_warnings, package['effective_profile_owner_id'], package['effective_profile_name'])

                # --- Aggressive Memory Cleanup ---
                if image_file_to_send:
                    image_file_to_send.close()
                    del image_file_to_send

                if package.get('generated_image_path') and os.path.exists(package['generated_image_path']):
                    os.remove(package['generated_image_path'])

                # Delete the dictionary itself
                del package

                self.cog.text_request_queue.task_done()
            except asyncio.CancelledError:
                break
            except RuntimeError as e:
                if "Session is closed" in str(e):
                    break
                print(f"Error in image finisher worker: {e}"); traceback.print_exc()
            except Exception as e:
                print(f"Error in image finisher worker: {e}"); traceback.print_exc()
                # Ensure typing is stopped on error for child bots
                if 'package' in locals() and package and package.get("is_child_bot"):
                    try:
                        await self.cog.manager_queue.put({
                            "action": "send_to_child", "bot_id": package['bot_id'],
                            "payload": {"action": "stop_typing", "channel_id": package['channel_id']}
                        })
                    except Exception as e_stop:
                        print(f"Failed to send stop_typing on error: {e_stop}")
                if 'placeholder_message' in locals() and placeholder_message:
                    try:
                        await placeholder_message.delete()
                    except: pass
                if 'state_container' in locals() and state_container and state_container.get('sending_task'):
                    state_container['sending_task'].cancel()

    async def _image_gen_worker(self, worker_id: int):
        """Pre-fetches image generation for text-only prompts."""
        while True:
            try:
                # [FIXED] Unpack Priority Tuple
                item = await self.cog.image_request_queue.get()
                if item is None: break

                priority, _, request_data = item

                # If a reference image is present, this request bypasses pre-fetching.
                if request_data.get("reference_image_urls"):
                    await self.cog.text_request_queue.put((priority, time.time(), request_data))
                    self.cog.image_request_queue.task_done()
                    continue

                # --- Pre-fetch Logic ---
                image_data, failure_reason, response = None, None, None
                # response is reset per request, not merely on the error path:
                # these workers loop inside one frame, so a generation that
                # raises before rebinding it would otherwise leave the
                # *previous* request's response to be closed here.
                try:
                    api_key = self.cog.storage_manager._get_api_key_for_guild(request_data['guild_id'])
                    if not api_key: raise ValueError("Server API key is not configured.")

                    img_model_raw = request_data.get("image_generation_model", DEFAULT_IMAGE_MODEL)
                    img_fallback_raw = request_data.get("image_generation_fallback_model")
                    image_model = self.build_image_model(
                        img_model_raw, api_key, request_data['system_instruction'],
                        request_data['safety_settings'], request_data.get('image_output'))

                    status = "api_error"
                    try:
                        async def _attempt(raw_name, _is_fallback):
                            nonlocal image_model
                            image_model = self.build_image_model(
                                raw_name, api_key, request_data['system_instruction'],
                                request_data['safety_settings'], request_data.get('image_output'))
                            return await image_model.generate_content_async([{'role': 'user', 'parts': [request_data['prompt_text']]}])

                        response, _used, _was_fallback = await self.cog.api_service.run_with_fallback(
                            img_model_raw, img_fallback_raw, _attempt, label="Image generation")
                        status = "blocked_by_safety" if not response.candidates else "success"
                    finally:
                        self.cog._log_api_call(user_id=request_data['author_id'], guild_id=request_data['guild_id'], context="image_generation_prefetch", model_used=image_model, status=status)

                    if not response.candidates:
                        reason = "Safety Filter";
                        if response.prompt_feedback and response.prompt_feedback.block_reason: reason = response.prompt_feedback.block_reason.name.replace('_', ' ').title()
                        failure_reason = f"the safety filter ({reason})"
                    else:
                        candidate = response.candidates[0]
                        if candidate.finish_reason.name != 'STOP':
                            failure_reason = f"the process being stopped for reason: **{candidate.finish_reason.name.replace('_', ' ').title()}**"
                        else:
                            image_data = next((part.inline_data.data for part in candidate.content.parts if getattr(part, 'inline_data', None) and part.inline_data.mime_type.startswith('image/')), None)
                            if not image_data: failure_reason = "an unknown issue (the model returned no image data)"
                except Exception as e:
                    failure_reason = f"an unexpected error: `{e}`"

                if image_data:
                    # Takes ownership of the file blob_stream already wrote; only a
                    # sub-threshold image is still bytes needing a write here.
                    request_data['generated_image_path'] = await self.cog.api_service.materialise_inline_data(
                        response, image_data, ".png")
                # image_data and response are only rebound when the *next* request
                # arrives, and this worker spends most of its life blocked on the
                # queue below -- so without dropping them here, whatever the last
                # generation returned stays resident for the whole idle period. The
                # close() also unlinks a blob no request took.
                image_data = None
                if response is not None:
                    try:
                        response.close()
                    except Exception:
                        pass
                    response = None
                if request_data.get('generated_image_path'):
                    maybe_trim_malloc()

                request_data['failure_reason'] = failure_reason

                # Pass priority to next stage
                await self.cog.text_request_queue.put((priority, time.time(), request_data))
                self.cog.image_request_queue.task_done()
            except asyncio.CancelledError:
                break
            except RuntimeError as e:
                if "Session is closed" in str(e):
                    break
                print(f"Error in image generation worker #{worker_id}: {e}"); traceback.print_exc()
            except Exception as e:
                print(f"Error in image generation worker #{worker_id}: {e}"); traceback.print_exc()

    async def _process_text_attachments(self, attachments: List[Any], client: httpx.AsyncClient) -> str:
        text_blocks = []
        text_extensions = ('.txt', '.log', '.md', '.csv', '.json', '.py', '.js', '.html', '.css', '.xml')
        
        count = 0
        for att in attachments:
            if count >= 2: break
            
            if isinstance(att, dict):
                url = att.get('url')
                filename = att.get('filename', 'attachment.txt')
                content_type = (att.get('content_type') or '').lower()
            else:
                url = att.url
                filename = att.filename
                content_type = (att.content_type or '').lower()
                
            is_text = content_type.startswith("text/") or filename.lower().endswith(text_extensions)
            if not is_text or not url:
                continue
                
            try:
                resp = await client.get(url, follow_redirects=True, timeout=10.0)
                resp.raise_for_status()
                
                raw_bytes = resp.content
                if len(raw_bytes) > 5 * 1024 * 1024:
                    raw_bytes = raw_bytes[:5 * 1024 * 1024]
                    
                decoded = raw_bytes.decode('utf-8', errors='replace')
                
                if '\x00' in decoded[:1000]:
                    continue
                    
                clean_text = decoded.strip()
                if not clean_text:
                    continue
                    
                if len(clean_text) > 40000:
                    clean_text = clean_text[:40000] + "\n[Content truncated at 40,000 characters]"
                    
                text_blocks.append(f"<text_attachment filename='{filename}'>\n{clean_text}\n</text_attachment>")
                count += 1
            except Exception as e:
                print(f"Failed to fetch or process text attachment {filename}: {e}")
                
        return "\n\n".join(text_blocks)

    def _get_image_gen_system_instruction(self, owner_id: int, profile_name: str) -> Optional[str]:
        index = self.cog.profile_manager._get_user_index(owner_id)
        is_borrowed = profile_name in index.get("borrowed", [])

        effective_owner_id = owner_id
        effective_profile_name = profile_name
        if is_borrowed:
            borrowed_data = self.cog.profile_manager._get_profile_config(owner_id, profile_name, True) or {}
            effective_owner_id = int(borrowed_data.get("original_owner_id", owner_id))
            effective_profile_name = borrowed_data.get("original_profile_name", profile_name)

        source_prompts = self.cog.profile_manager._get_profile_prompts(effective_owner_id, effective_profile_name)

        if not source_prompts:
            return None

        # Get the general image style prompt
        encrypted_style_prompt = source_prompts.get("image_generation_prompt")
        style_prompt = self.cog.storage_manager._decrypt_data(encrypted_style_prompt) if encrypted_style_prompt else ""
        
        return style_prompt.strip() or None

    async def _handle_image_generation_request(self, message: discord.Message, prompt_content: str):
        try:
            effective_profile_owner_id = message.author.id
            effective_profile_name = self.cog.session_manager._get_active_user_profile_name_for_channel(effective_profile_owner_id, message.channel.id)

            index = self.cog.profile_manager._get_user_index(effective_profile_owner_id)
            is_borrowed = effective_profile_name in index.get("borrowed", [])
            profile_data = self.cog.profile_manager._get_profile_config(effective_profile_owner_id, effective_profile_name, is_borrowed) or {}

            if not profile_data.get("image_generation_enabled", False):
                return

            if self.cog.image_request_queue.full():
                await message.reply("The image generation backlog is currently full. Please try again in a moment.", delete_after=10)
                return

            placeholder_message = None
            # Check if the finisher is busy. If it is, we are queued.
            if self.cog.image_gen_semaphore.locked():
                qsize = self.cog.image_request_queue.qsize()
                await message.reply(f"Your image generation request has been queued. You are #{qsize + 1} in line.", delete_after=10)
            
            image_prefixes = ("!image", "!imagine")
            used_prefix = next((p for p in image_prefixes if prompt_content.lower().startswith(p)), "!image")
            prompt_text = prompt_content[len(used_prefix):].strip()
            if not prompt_text:
                await message.reply(f"Please provide a prompt after `{used_prefix}`.", delete_after=10)
                return

            effective_profile_owner_id = message.author.id
            effective_profile_name = self.cog.session_manager._get_active_user_profile_name_for_channel(effective_profile_owner_id, message.channel.id)

            # If the finisher is NOT busy and the queue is currently empty, we are first in line.
            # Send the placeholder immediately.
            if not self.cog.image_gen_semaphore.locked() and self.cog.image_request_queue.empty():
                 placeholders = await self.cog.generation_service._send_channel_message(
                    message.channel, f"{PLACEHOLDER_EMOJI}",
                    profile_owner_id_for_appearance=effective_profile_owner_id,
                    profile_name_for_appearance=effective_profile_name
                )
                 placeholder_message = placeholders[0] if placeholders else None

            index = self.cog.profile_manager._get_user_index(effective_profile_owner_id)
            is_borrowed = effective_profile_name in index.get("borrowed", [])
            profile_data = self.cog.profile_manager._get_profile_config(effective_profile_owner_id, effective_profile_name, is_borrowed) or {}
            dynamic_safety_settings = _resolve_safety_settings(message.channel, profile_data)

            # Get appearance text
            source_owner_id = effective_profile_owner_id
            source_profile_name = effective_profile_name
            if is_borrowed:
                borrowed_data = self.cog.profile_manager._get_profile_config(effective_profile_owner_id, effective_profile_name, True) or {}
                source_owner_id = int(borrowed_data.get("original_owner_id", effective_profile_owner_id))
                source_profile_name = borrowed_data.get("original_profile_name", effective_profile_name)
            
            source_prompts = self.cog.profile_manager._get_profile_prompts(source_owner_id, source_profile_name) or {}
            persona = source_prompts.get("persona", {})
            appearance_lines_encrypted = persona.get("appearance", [])
            appearance_text = "\n".join([self.cog.storage_manager._decrypt_data(line) for line in appearance_lines_encrypted])

            bot_display_name = self.cog.bot.user.display_name
            appearance_data = self.cog.user_appearances.get(str(effective_profile_owner_id), {}).get(effective_profile_name, {})
            if appearance_data and appearance_data.get("custom_display_name"): bot_display_name = appearance_data.get("custom_display_name")

            final_prompt_text = prompt_text
            if appearance_text.strip():
                prompt_lower = prompt_text.lower()
                second_person_pronouns = ["you", "your", "yourself", "u", "ur"]
                # Check for pronouns or the profile's names
                if any(pronoun in prompt_lower.split() for pronoun in second_person_pronouns) or \
                   bot_display_name.lower() in prompt_lower or \
                   effective_profile_name.lower() in prompt_lower:
                    appearance_template = self.cog.global_prompts.get("IMAGE_APPEARANCE", DEFAULT_IMAGE_APPEARANCE)
                    final_prompt_text = appearance_template.format(appearance=appearance_text.strip(), prompt=prompt_text)

            system_instruction = self._get_image_gen_system_instruction(effective_profile_owner_id, effective_profile_name)
            appearance_data = self.cog.user_appearances.get(str(effective_profile_owner_id), {}).get(effective_profile_name, {})
            if appearance_data and appearance_data.get("custom_display_name"): bot_display_name = appearance_data.get("custom_display_name")

            reference_image_urls = []
            if message.reference and message.reference.message_id:
                ref_msg = message.reference.resolved
                if not ref_msg or not isinstance(ref_msg, discord.Message):
                    try:
                        ref_msg = await message.channel.fetch_message(message.reference.message_id)
                    except Exception: pass
                
                if ref_msg and isinstance(ref_msg, discord.Message):
                    for attachment in ref_msg.attachments:
                        if attachment.content_type and attachment.content_type.startswith("image/"):
                            reference_image_urls.append({"url": attachment.url, "mime_type": attachment.content_type})
                            if len(reference_image_urls) >= 2: break
            
            if len(reference_image_urls) < 10 and message.attachments:
                for attachment in message.attachments:
                    if attachment.content_type and attachment.content_type.startswith("image/"):
                        reference_image_urls.append({"url": attachment.url, "mime_type": attachment.content_type})
                        if len(reference_image_urls) >= 10: break

            # Define local variables required for grounding logic
            guild_id = message.guild.id
            channel_id = message.channel.id
            owner_id = effective_profile_owner_id
            profile_name = effective_profile_name

            grounding_sources = []
            grounding_mode = profile_data.get("grounding_mode", "off")
            if isinstance(grounding_mode, bool): grounding_mode = "on" if grounding_mode else "off"

            if grounding_mode in ["on", "on+"]:
                session_key = (channel_id, owner_id, profile_name)
                img_session = self.cog.multi_profile_channels.get(channel_id) or {}

                stm_len = int(profile_data.get("stm_length", defaultConfig.CHATBOT_MEMORY_LENGTH))
                grounding_stm = min(10, stm_len)
                history_for_grounding = []
                if grounding_stm > 0:
                    g_bot_pid = self.cog.profile_manager._get_pid_from_name_any(owner_id, profile_name)
                    history_for_grounding = self.cog.session_manager._build_history_for_participant(
                        img_session.get("unified_log", []), g_bot_pid, profile_data
                    )[-grounding_stm:]

                mapping_key = self.cog.session_manager._get_mapping_key_for_session(session_key, 'multi')
                grounding_result = await self.cog.tools_service._get_hybrid_grounding_context(prompt_text, guild_id, history_for_grounding, mapping_key, safety_settings=dynamic_safety_settings, is_for_image=True, warning_channel=message.channel)
                if grounding_result:
                    grounding_context, sources, *_ = grounding_result
                    if grounding_context:
                        grounding_template = self.cog.global_prompts.get("IMAGE_GROUNDING", DEFAULT_IMAGE_GROUNDING)
                        final_prompt_text = grounding_template.format(prompt=prompt_text, grounding=grounding_context)
                        grounding_sources = sources

            request_data = {
                "is_child_bot": False, "author_id": message.author.id, "channel_id": message.channel.id, "guild_id": message.guild.id,
                "original_message_id": message.id, "original_content": message.content, "prompt_text": final_prompt_text, 
                "effective_profile_owner_id": effective_profile_owner_id, "effective_profile_name": effective_profile_name, 
                "bot_display_name": bot_display_name, "safety_settings": dynamic_safety_settings,
                "system_instruction": system_instruction, "reference_image_urls": reference_image_urls, "placeholder_message": placeholder_message,
                "grounding_sources": grounding_sources, "grounding_mode": grounding_mode,
                "image_generation_model": profile_data.get("image_generation_model", DEFAULT_IMAGE_MODEL),
                "image_generation_fallback_model": profile_data.get("image_generation_fallback_model"),
                # Carried rather than re-read off the profile in the worker: the request
                # may sit in the queue behind others while its owner edits the profile,
                # and the image should come back the shape it was asked for.
                "image_output": {k: profile_data.get(k) for k in IMAGE_OUTPUT_KEYS},
            }
            
            await self.cog.image_request_queue.put((IMAGE_QUEUE_PRIORITY, time.time(), request_data))

        except Exception as e:
            await message.reply(f"An error occurred while queueing your request: {e}", delete_after=10)
            traceback.print_exc()

    def _purge_channel_image_requests(self, channel_id: int):
        """Purges any pending image generation requests for a given channel from queues."""
        items = []
        while not self.cog.image_request_queue.empty():
            try:
                item = self.cog.image_request_queue.get_nowait()
                _, _, req_data = item
                if req_data.get('channel_id') != channel_id:
                    items.append(item)
                else:
                    self.cog.image_request_queue.task_done()
            except (asyncio.QueueEmpty, ValueError):
                break
        for item in items:
            self.cog.image_request_queue.put_nowait(item)

        text_items = []
        while not self.cog.text_request_queue.empty():
            try:
                item = self.cog.text_request_queue.get_nowait()
                _, _, req_data = item
                if req_data.get('channel_id') != channel_id:
                    text_items.append(item)
                else:
                    self.cog.text_request_queue.task_done()
            except (asyncio.QueueEmpty, ValueError):
                break
        for item in text_items:
            self.cog.text_request_queue.put_nowait(item)
