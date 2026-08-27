from ...utils.constants import PLACEHOLDER_EMOJI, DEFAULT_IMAGE_APPEARANCE, DEFAULT_IMAGE_MODEL
from ...utils.helpers import _format_api_error, _resolve_safety_settings
from ...utils.memory_tuning import maybe_trim_malloc
from ...utils import mem_probe


class ImageRoundMixin:
    """The single image generation a multi-profile round may run before its
    participants speak, so the generated image is available to every turn.
    """

    async def _run_image_generation_round(
        self, session, channel, generator_profile_key, image_gen_prompt,
        generator_display_name, first_participant, feedback_task, new_round_turn_data,
        generated_image_path_for_round=None, image_gen_placeholder_id=None, image_gen_error_msg=None,
    ):
        """Runs the round's one image generation and returns
        (generated_image_path_for_round, image_gen_placeholder_id, image_gen_error_msg).

        Lifted out of _multi_profile_worker rather than left inline: the response body
        carries the base64 payload and its memoised decode, and while this ran in the
        worker frame every one of those buffers stayed reachable for the whole
        participant loop below it. The explicit teardown in the finally block is kept,
        but returning from a frame of its own is what actually bounds their lifetime.

        The three values are passed in and returned so a profile with image generation
        disabled leaves the caller's state exactly as it found it.
        """
        gen_owner_id, gen_profile_name = generator_profile_key
        gen_idx = self.cog.profile_manager._get_user_index(gen_owner_id)
        gen_is_b = gen_profile_name in gen_idx.get("borrowed", [])
        gen_cfg = self.cog.profile_manager._get_profile_config(gen_owner_id, gen_profile_name, gen_is_b) or {}

        if not gen_cfg.get("image_generation_enabled", False):
            return generated_image_path_for_round, image_gen_placeholder_id, image_gen_error_msg

        # Hoisted out of the try so the finally below can still read them when
        # generation raises: _generate_with_heartbeat mutates the container it
        # is handed, so a placeholder it created is recorded there even on the
        # error path, and the response owns any blob file that has to be unlinked.
        image_state_container = None
        response = None
        try:
            api_key = self.cog.storage_manager._get_api_key_for_guild(channel.guild.id)
            if not api_key: raise ValueError("Server API key not configured.")

            img_model_raw = gen_cfg.get("image_generation_model", DEFAULT_IMAGE_MODEL)
            img_fallback_raw = gen_cfg.get("image_generation_fallback_model")

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
                    appearance_template = self.cog.global_prompts.get("IMAGE_APPEARANCE", DEFAULT_IMAGE_APPEARANCE)
                    final_prompt_text = appearance_template.format(appearance=appearance_text.strip(), prompt=image_gen_prompt)

            ref_images = []
            for _, _, turn_media in new_round_turn_data:
                for media in turn_media:
                    if media.get("mime_type", "").startswith("image/"):
                        ref_images.append(media)

            parts = [final_prompt_text]
            for ref in ref_images[:10]:
                parts.append({"url": ref["url"], "mime_type": ref.get("mime_type", "image/png")})

            # Determine safety
            dynamic_safety_settings = _resolve_safety_settings(channel, gen_cfg)

            # Built up front so the `finally` that logs the call has a model to name even
            # if the first attempt raises before rebinding it.
            image_model = self.cog.media_service.build_image_model(
                img_model_raw, api_key, system_instruction, dynamic_safety_settings, gen_cfg)

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

            async def _attempt(raw_name, _is_fallback):
                nonlocal image_model
                image_model = self.cog.media_service.build_image_model(
                    raw_name, api_key, system_instruction, dynamic_safety_settings, gen_cfg)
                # image_state_container is mutated in place by the heartbeat, so a retry
                # edits the placeholder the first attempt made rather than adding one.
                return await self._generate_with_heartbeat(
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

            with mem_probe.probe("  image gen: API call", peak=False):
                result, _used, _was_fallback = await self.cog.api_service.run_with_fallback(
                    img_model_raw, img_fallback_raw, _attempt, label="Image generation")
            response, image_state_container = result
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
                    img_data = next((part.inline_data.data for part in candidate.content.parts if getattr(part, 'inline_data', None) and part.inline_data.mime_type.startswith('image/')), None)
                    if img_data:
                        # Already on disk: the response streamed it there rather
                        # than through the heap (cogs/utils/blob_stream), so this
                        # is a rename. A small enough image is still bytes and
                        # gets written here, exactly as it always was.
                        with mem_probe.probe("  image gen: to file"):
                            generated_image_path_for_round = await self.cog.api_service.materialise_inline_data(
                                response, img_data, ".png")
                        img_data = None
                    else:
                        image_gen_error_msg = "no image data returned"

            self.cog._log_api_call(user_id=session.get('owner_id', 0), guild_id=channel.guild.id, context="image_generation_multi", model_used=image_model, status=status)

        except Exception as e:
            image_gen_error_msg = _format_api_error(e)
            print(f"Error generating image in multi-profile round: {e}")
        finally:
            # The heartbeat spawns its own placeholder when the generator is a
            # child bot that sends none of its own — in that case feedback_task
            # is None, so this id exists nowhere else and the participant loop
            # below would never delete it, leaving a stranded "Still
            # generating..." message in the channel.
            if image_state_container:
                image_gen_placeholder_id = image_state_container.get('msg_a_id')

            # The response no longer carries the image -- blob_stream diverted it
            # to a file on the way off the socket -- so this is now about the file
            # rather than the bytes: close() unlinks any blob the round did not
            # take, which is what a safety block or a fallback retry leaves behind.
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
            response = None
            candidate = None
            image_state_container = None

            # Kept, smaller in scope than it was. The multi-megabyte frees this
            # used to answer for are gone with the buffered read, but the round
            # still churns the allocator through the skeleton parse and the File
            # API upload of the generated image, and an arena that is never
            # trimmed is how the resident set ratchets. Rate-limited, so a round
            # that also moves media through the File API trims once, not twice.
            maybe_trim_malloc()

        return generated_image_path_for_round, image_gen_placeholder_id, image_gen_error_msg
