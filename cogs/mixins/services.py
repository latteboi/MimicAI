import asyncio
import discord
from discord.ext import tasks
import traceback
import time
import platform
import signal
import datetime
import random
import uuid
import re
import io
import base64
import httpx
import gc
import pathlib
import collections
import orjson as json
import functools
from PIL import Image
import google.generativeai.types as genai_types
from urllib.parse import urlparse
from zoneinfo import ZoneInfo
from typing import List, Dict, Tuple, Optional, Any, Literal, Set, Union, get_args
from .constants import *
from .storage import (
    _delete_file_shard, 
    _quantize_embedding, 
    _dequantize_embedding, 
    cosine_similarity, 
)

# [LEGAL/SDK IMPORTS]
import google.generativeai as genai
from google import genai as google_genai
from google.genai import types as google_genai_types
from google.api_core import exceptions as api_exceptions

class Timeout:
    def __init__(self, seconds=2, error_message='Function call timed out'):
        self.seconds = seconds
        self.error_message = error_message
        self.is_windows = platform.system() == "Windows"

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        if not self.is_windows:
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        if not self.is_windows:
            signal.alarm(0)

def _split_into_sentences_with_abbreviations(text: str) -> List[str]:
    abbreviations = {
        'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'rev.', 'hon.', 'st.', 'sr.', 'jr.', 'capt.', 'sgt.', 'col.', 'gen.',
        'etc.', 'vs.', 'i.e.', 'e.g.', 'cf.', 'et al.', 'viz.',
        'ave.', 'blvd.', 'rd.',
        'a.m.', 'p.m.', 'in.', 'ft.', 'yd.', 'mi.',
        'approx.', 'apt.', 'assn.', 'asst.', 'bldg.', 'co.', 'corp.', 'dept.', 'est.', 'inc.', 'ltd.', 'mfg.', 'vol.'
    }
    
    potential_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    if not potential_sentences:
        return []

    merged_sentences = []
    for s in potential_sentences:
        if not merged_sentences:
            merged_sentences.append(s)
            continue
        
        last_sentence = merged_sentences[-1]
        words = last_sentence.split()
        if words and words[-1].lower() in abbreviations:
            merged_sentences[-1] += " " + s
        else:
            merged_sentences.append(s)
            
    return merged_sentences

# --- OpenRouter Adapter Classes ---
class MockPart:
    def __init__(self, text): self.text = text
class MockContent:
    def __init__(self, text): self.parts = [MockPart(text)]
class MockCandidate:
    def __init__(self, text, finish_reason):
        self.content = MockContent(text)
        self.finish_reason = type('obj', (object,), {'name': finish_reason})
class MockResponse:
    def __init__(self, text, finish_reason="STOP"):
        self.text = text; self.candidates = [MockCandidate(text, finish_reason)]

class OpenRouterChatSession:
    def __init__(self, model, history=None):
        self.model = model
        self.history = history or []

class OpenRouterModel:
    def __init__(self, model_name, api_key, system_instruction=None, thinking_params=None, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.system_instruction = system_instruction
        self.thinking_params = thinking_params or {} # [NEW]

    def start_chat(self, history=None):
        return OpenRouterChatSession(self, history=history)

    async def generate_content_async(self, contents, generation_config=None, safety_settings=None):
        import base64
        messages = []
        if self.system_instruction:
            messages.append({"role": "system", "content": self.system_instruction})
        
        for content in contents:
            role = "assistant" if content.role == "model" else "user"
            message_parts = []
            
            for p in content.parts:
                if hasattr(p, 'text') and p.text and p.text.strip():
                    message_parts.append({"type": "text", "text": p.text})
                elif hasattr(p, 'inline_data') and p.inline_data:
                    mime_type = p.inline_data.mime_type
                    if mime_type.startswith("image/"):
                        try:
                            b64_data = base64.b64encode(p.inline_data.data).decode('utf-8')
                            data_uri = f"data:{mime_type};base64,{b64_data}"
                            message_parts.append({"type": "image_url", "image_url": {"url": data_uri, "detail": "auto"}})
                        except Exception as e:
                            print(f"Error encoding image for OpenRouter: {e}")

            if message_parts:
                if len(message_parts) == 1 and message_parts[0]["type"] == "text":
                    messages.append({"role": role, "content": message_parts[0]["text"]})
                else:
                    messages.append({"role": role, "content": message_parts})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": getattr(generation_config, 'temperature', 1.0) if generation_config else 1.0,
            "top_p": getattr(generation_config, 'top_p', 1.0) if generation_config else 1.0,
        }

        # Unified Reasoning Support with Mutual Exclusivity Guard
        include_thoughts = self.thinking_params.get("thinking_summary_visible") == "on"
        if generation_config and hasattr(generation_config, 'thinking_config') and generation_config.thinking_config:
            include_thoughts = generation_config.thinking_config.include_thoughts

        budget = int(self.thinking_params.get("thinking_budget", -1))
        
        if include_thoughts or budget > 0:
            payload["reasoning"] = {"exclude": not include_thoughts}
            # OpenRouter requires max_tokens OR effort to be specified, but never both.
            if budget > 0:
                payload["reasoning"]["max_tokens"] = budget
            else:
                payload["reasoning"]["effort"] = self.thinking_params.get("thinking_level", "high")
        
        if hasattr(generation_config, '_advanced_params') and generation_config._advanced_params:
            payload.update(generation_config._advanced_params)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://discord.com",
            "X-Title": "MimicAI Discord Bot"
        }

        async with httpx.AsyncClient() as client:
            response = await client.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=60)
            if response.status_code != 200:
                raise Exception(f"OpenRouter API Error {response.status_code}: {response.text}")
            
            data = response.json()
            if 'error' in data:
                 raise Exception(f"OpenRouter API Error: {data['error']}")
            
            choice = data['choices'][0]
            msg_obj = choice['message']
            
            class OpenRouterThoughtResponse:
                def __init__(self, content, reasoning, finish_reason):
                    self.text = content
                    self.thought = reasoning or ""
                    
                    # Create mock content and parts for the worker's scrubbing logic
                    mock_part = type('obj', (object,), {'text': content})
                    mock_content = type('obj', (object,), {'parts': [mock_part]})
                    
                    # Create the candidate object
                    self.candidates = [type('obj', (object,), {
                        'content': mock_content,
                        'finish_reason': type('obj', (object,), {'name': finish_reason})
                    })]
                    
                def __bool__(self): return True

            return OpenRouterThoughtResponse(msg_obj.get('content', ''), msg_obj.get('reasoning', ''), choice.get('finish_reason', 'STOP').upper())

class GoogleGenAIChatSession:
    def __init__(self, model, history=None):
        self.model = model
        self.history = history or []

class GoogleGenAIModel:
    def __init__(self, api_key, model_name, system_instruction=None, safety_settings=None, thinking_params=None):
        # Force v1beta for stable "Thinking" part delivery
        self.client = google_genai.Client(
            api_key=api_key, 
            http_options=google_genai_types.HttpOptions(api_version='v1beta')
        )
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.safety_settings = safety_settings
        self.thinking_params = thinking_params or {}

    def start_chat(self, history=None):
        return GoogleGenAIChatSession(self, history=history)

    async def generate_content_async(self, contents, generation_config=None):
        formatted_contents = []
        for item in contents:
            if isinstance(item, str):
                formatted_contents.append(google_genai_types.Part.from_text(text=item))
            elif isinstance(item, dict):
                if 'data' in item and 'mime_type' in item:
                    # Pass bytes directly without creating temporary variables
                    formatted_contents.append(google_genai_types.Part.from_bytes(data=item['data'], mime_type=item['mime_type']))
            elif hasattr(item, 'role') and hasattr(item, 'parts'):
                new_parts = []
                for p in item.parts:
                    if hasattr(p, 'text') and p.text:
                        new_parts.append(google_genai_types.Part.from_text(text=p.text))
                    elif hasattr(p, 'inline_data') and p.inline_data:
                        new_parts.append(google_genai_types.Part.from_bytes(data=p.inline_data.data, mime_type=p.inline_data.mime_type))
                formatted_contents.append(google_genai_types.Content(role=item.role, parts=new_parts))
            else:
                formatted_contents.append(item)

        v2_safety = []
        if self.safety_settings:
            for cat, thresh in self.safety_settings.items():
                v2_safety.append(google_genai_types.SafetySetting(
                    category=cat.name if hasattr(cat, 'name') else str(cat),
                    threshold=thresh.name if hasattr(thresh, 'name') else str(thresh)
                ))

        # Thinking Config Logic
        thinking_cfg = None
        model_lower = self.model_name.lower()
        include_thoughts = self.thinking_params.get("thinking_summary_visible") == "on"
        
        # Override thoughts status if config specifically requests it (Whispers/Global Chat)
        if generation_config and hasattr(generation_config, 'thinking_config') and generation_config.thinking_config:
            include_thoughts = generation_config.thinking_config.include_thoughts

        if "gemini-3" in model_lower:
            t_level = self.thinking_params.get("thinking_level", "high")
            # Normalize: 3-Pro does not support 'medium' in the SDK enum
            if "pro" in model_lower and t_level == "medium":
                t_level = "high"

            thinking_cfg = google_genai_types.ThinkingConfig(
                include_thoughts=include_thoughts,
                thinking_level=t_level
            )
        elif "gemini-2.5" in model_lower:
            budget = self.thinking_params.get("thinking_budget", -1)
            if "lite" not in model_lower:
                thinking_cfg = google_genai_types.ThinkingConfig(
                    include_thoughts=include_thoughts,
                    thinking_budget=int(budget)
                )

        config = google_genai_types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            temperature=generation_config.temperature if generation_config else None,
            top_p=generation_config.top_p if generation_config else None,
            top_k=generation_config.top_k if generation_config else None,
            safety_settings=v2_safety if v2_safety else None,
            thinking_config=thinking_cfg
        )

        if hasattr(generation_config, '_advanced_params') and generation_config._advanced_params:
            for k, v in generation_config._advanced_params.items():
                if hasattr(config, k):
                    setattr(config, k, v)

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=formatted_contents,
            config=config
        )
        
        # [NEW] Clear the formatted_contents list immediately after call
        formatted_contents.clear()
        del formatted_contents

        class ThoughtResponse:
            def __init__(self, raw_resp):
                self.raw = raw_resp
                self.text = ""
                self.thought = ""
                self.candidates = raw_resp.candidates
                self.prompt_feedback = getattr(raw_resp, 'prompt_feedback', None)
                self.usage_metadata = getattr(raw_resp, 'usage_metadata', None)
                
                if raw_resp.candidates:
                    for part in raw_resp.candidates[0].content.parts:
                        is_thought = getattr(part, 'thought', False)
                        if is_thought:
                            self.thought += part.text or ""
                        elif part.text:
                            self.text += part.text
            def __bool__(self): return bool(self.candidates)

        return ThoughtResponse(response)
    
class ServicesMixin:

    async def _get_or_create_model_for_channel(self, channel_id: int, actual_message_author_id: int, guild_id: int, profile_owner_override: Optional[int] = None, profile_name_override: Optional[str] = None, prompt_content: Optional[str] = None) -> Tuple[Optional[Any], bool, float, float, int, Optional[str], Optional[str]]:
        
        api_key = self._get_api_key_for_guild(guild_id)
        if not api_key:
            return None, True, 0.0, 0.0, 0, "Server API key is not configured.", None
        
        if profile_owner_override is not None and profile_name_override is not None:
            profile_owner_id_for_instructions = profile_owner_override
            profile_name_for_instructions = profile_name_override
        else:
            profile_owner_id_for_instructions: Optional[int] = actual_message_author_id
            profile_name_for_instructions: str = self._get_active_user_profile_name_for_channel(profile_owner_id_for_instructions, channel_id)
        
        channel = self.bot.get_channel(channel_id)
        if not channel:
            return None, True, 0.0, 0.0, 0, "Could not find the channel for this interaction.", None

        if not self._check_unrestricted_safety_policy(profile_owner_id_for_instructions, profile_name_for_instructions, channel):
            return None, True, 0.0, 0.0, 0, "Profiles with 'Unrestricted 18+' safety can only be used in age-restricted channels.", None

        model_cache_key = (channel_id, profile_owner_id_for_instructions, profile_name_for_instructions)

        user_data = self._get_user_data_entry(profile_owner_id_for_instructions)
        is_borrowed = profile_name_for_instructions in user_data.get("borrowed_profiles", {})
        
        original_owner_id = profile_owner_id_for_instructions
        original_profile_name = profile_name_for_instructions
        if is_borrowed:
            borrowed_data = user_data["borrowed_profiles"][profile_name_for_instructions]
            original_owner_id = int(borrowed_data["original_owner_id"])
            original_profile_name = borrowed_data["original_profile_name"]

        current_profile_key_for_model = (original_owner_id, original_profile_name)

        training_examples_list = []
        if prompt_content:
            training_examples_list = await self._get_relevant_training_examples(
                profile_owner_id_for_instructions,
                profile_name_for_instructions,
                prompt_content,
                guild_id
            )

        current_instructions, error_in_instr_constr, _, temperature, top_p, top_k, primary_model, fallback_model = self._construct_system_instructions(
            profile_owner_id_for_instructions,
            profile_name_for_instructions,
            channel_id,
            training_examples_list=training_examples_list
        )
        
        warning_message = None

        recreate_model = True
        if model_cache_key in self.channel_models and not training_examples_list:
            last_profile_key = self.channel_model_last_profile_key.get(model_cache_key)
            if last_profile_key == current_profile_key_for_model:
                 recreate_model = False 
            
        if recreate_model and model_cache_key in self.channel_models:
            del self.channel_models[model_cache_key]
            self.channel_model_last_profile_key.pop(model_cache_key, None)
        
        if model_cache_key in self.channel_models and not recreate_model: 
            model_instance, model_init_error_state, cached_model_name = self.channel_models[model_cache_key]
            return model_instance, model_init_error_state, temperature, top_p, top_k, warning_message, fallback_model

        model_instance, model_init_error = None, True
        
        safety_level_str = "low" 
        profile_data_for_safety = user_data.get("borrowed_profiles" if is_borrowed else "profiles", {}).get(profile_name_for_instructions, {})
        safety_level_str = profile_data_for_safety.get("safety_level", "low")

        safety_map = {
            "unrestricted": HarmBlockThreshold.BLOCK_NONE,
            "low": HarmBlockThreshold.BLOCK_ONLY_HIGH,
            "medium": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            "high": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        }
        threshold = safety_map.get(safety_level_str, HarmBlockThreshold.BLOCK_ONLY_HIGH)
        
        dynamic_safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: threshold,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: threshold,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: threshold,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: threshold,
        }

        model_to_create = primary_model
        
        # [NEW] Extract parameters once for either provider
        p_sett_thinking = user_data.get("borrowed_profiles" if is_borrowed else "profiles", {}).get(profile_name_for_instructions, {})
        t_params = {
            "thinking_summary_visible": p_sett_thinking.get("thinking_summary_visible", "off"),
            "thinking_level": p_sett_thinking.get("thinking_level", "high"),
            "thinking_budget": p_sett_thinking.get("thinking_budget", -1)
        }

        def create_model_instance(name, system_instr, safety):
            name_upper = name.upper()
            actual_name = name
            is_openrouter = False
            
            if name_upper.startswith("OPENROUTER/"):
                actual_name = name[11:]
                is_openrouter = True
            elif name_upper.startswith("GOOGLE/"):
                actual_name = name[7:]
            
            if is_openrouter:
                or_key = self._get_api_key_for_guild(guild_id, provider="openrouter")
                if not or_key: raise ValueError("OpenRouter API Key not found.")
                # [UPDATED] Pass thinking_params to OpenRouter
                return OpenRouterModel(actual_name, api_key=or_key, system_instruction=system_instr, thinking_params=t_params)
            else:
                # [UPDATED] Pass thinking_params to Google wrapper
                return GoogleGenAIModel(api_key=api_key, model_name=actual_name, system_instruction=system_instr, safety_settings=safety, thinking_params=t_params)

        try:
            model_instance = create_model_instance(model_to_create, current_instructions, dynamic_safety_settings)
            model_init_error = False
        except Exception as e1:
            print(f"Err '{model_to_create}' key {model_cache_key}: {e1}. Fallback.")
            model_to_create = fallback_model
            try:
                model_instance = create_model_instance(model_to_create, current_instructions, dynamic_safety_settings)
                model_init_error = False
            except Exception as e2:
                return None, True, temperature, top_p, top_k, f"Model Initialization Error: Failed to load Primary ('{primary_model}') and Fallback ('{fallback_model}') models. Check your API key.", fallback_model
        
        final_error_state = error_in_instr_constr or model_init_error
        self.channel_models[model_cache_key] = (model_instance, final_error_state, model_to_create)
        self.channel_model_last_profile_key[model_cache_key] = current_profile_key_for_model
        return model_instance, final_error_state, temperature, top_p, top_k, warning_message, fallback_model

    async def _get_or_create_model_for_global_chat(self, user_id: int, profile_name: str) -> Tuple[Optional[Any], float, float, int, Optional[str], Optional[str]]:
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
            return None, 0.0, 0.0, 0, f"The source for your active global profile ('{profile_name}') could not be found.", None

        temp = profile_data.get("temperature", defaultConfig.GEMINI_TEMPERATURE)
        top_p = profile_data.get("top_p", defaultConfig.GEMINI_TOP_P)
        top_k = profile_data.get("top_k", defaultConfig.GEMINI_TOP_K)
        primary_model = profile_data.get("primary_model", PRIMARY_MODEL_NAME)
        fallback_model = profile_data.get("fallback_model", FALLBACK_MODEL_NAME)
        
        warning_message = None
        system_instructions, _, _, _, _, _, _, _ = self._construct_system_instructions(user_id, profile_name, 0)
        
        safety_level_str = profile_data.get("safety_level", "low")
        safety_map = { "unrestricted": HarmBlockThreshold.BLOCK_NONE, "low": HarmBlockThreshold.BLOCK_ONLY_HIGH, "medium": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, "high": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE }
        threshold = safety_map.get(safety_level_str, HarmBlockThreshold.BLOCK_ONLY_HIGH)
        safety_settings = { cat: threshold for cat in get_args(HarmCategory) }

        is_or = False
        actual_model_name = primary_model
        
        if primary_model.upper().startswith("OPENROUTER/"):
            is_or = True
            actual_model_name = primary_model[11:]
        elif primary_model.upper().startswith("GOOGLE/"):
            is_or = False
            # [FIXED] Correctly strip the prefix to provide a valid ID to the new SDK
            actual_model_name = primary_model[7:]
        elif "/" in primary_model or "grok" in primary_model.lower():
            is_or = True

        try:
            if is_or:
                user_api_key = self._get_api_key_for_user(user_id, provider="openrouter")
                if not user_api_key:
                    return None, 0.0, 0.0, 0, "You need to submit an OpenRouter API key using `/settings` to use this model.", None
                model = OpenRouterModel(actual_model_name, api_key=user_api_key, system_instruction=system_instructions, thinking_params={})
            else:
                user_api_key = self._get_api_key_for_user(user_id)
                if not user_api_key:
                    return None, 0.0, 0.0, 0, "This feature requires a personal Google Gemini API key. Use `/settings` to add one.", None
                
                t_params = {
                    "thinking_summary_visible": profile_data.get("thinking_summary_visible", "off"),
                    "thinking_level": profile_data.get("thinking_level", "high"),
                    "thinking_budget": profile_data.get("thinking_budget", -1)
                }
                # [NEW] Use the GoogleGenAIModel wrapper for SDK v2
                model = GoogleGenAIModel(api_key=user_api_key, model_name=actual_model_name, system_instruction=system_instructions, safety_settings=safety_settings, thinking_params=t_params)
            
            return model, temp, top_p, top_k, warning_message, fallback_model
        except Exception as e:
            print(f"Error creating model for global chat (user: {user_id}, profile: {profile_name}): {e}")
            return None, 0.0, 0.0, 0, "A critical error occurred while creating the AI model.", None
        
    def _construct_system_instructions(self, profile_owner_id: Optional[int], profile_name_to_use: str, channel_id: int, is_multi_profile: bool = False, training_examples_list: Optional[List[str]] = None, recalled_ltm: Optional[str] = None, critic_constraints: Optional[str] = None) -> Tuple[str, bool, bool, float, float, int, str, str]:
        persona_data: Dict[str, List[str]] = {}
        ai_instr_str: str = ""
        grounding_enabled = False
        temperature = defaultConfig.GEMINI_TEMPERATURE
        top_p = defaultConfig.GEMINI_TOP_P
        top_k = defaultConfig.GEMINI_TOP_K
        primary_model = PRIMARY_MODEL_NAME
        fallback_model = FALLBACK_MODEL_NAME
        time_tracking_enabled = False
        timezone_str = "UTC"

        if profile_owner_id is not None: 
            user_data = self._get_user_data_entry(profile_owner_id)
            is_borrowed = profile_name_to_use in user_data.get("borrowed_profiles", {})
            
            if is_borrowed:
                profile_data = user_data["borrowed_profiles"][profile_name_to_use]
            else:
                profile_data = user_data.get("profiles", {}).get(profile_name_to_use, {})

            persona_data, ai_instr_str, grounding_enabled, temperature, top_p, top_k, _, _, primary_model, fallback_model = self._get_user_profile_for_model(profile_owner_id, channel_id, profile_name_to_use)
            
        if profile_data:
            time_tracking_enabled = profile_data.get("time_tracking_enabled", False)
            timezone_str = profile_data.get("timezone", "UTC")

        final_instr_parts = []
        
        if is_multi_profile:
            session = self.multi_profile_channels.get(channel_id)
            if session and session.get("session_prompt"):
                final_instr_parts.append(f"<scene_prompt>\n{session['session_prompt']}\n</scene_prompt>")

        if time_tracking_enabled:
            try:
                tz = ZoneInfo(timezone_str)
                now = datetime.datetime.now(tz)
                time_str = now.strftime("%A, %d %B %Y, %I:%M %p (%Z)")
                final_instr_parts.append(f"<time_context>\nYour current time is {time_str}.\n</time_context>")
            except Exception as e:
                print(f"Error processing timezone '{timezone_str}': {e}. Defaulting to UTC.")
                now_utc = datetime.datetime.now(datetime.timezone.utc)
                time_str_utc = now_utc.strftime("%A, %d %B %Y, %I:%M %p (UTC)")
                final_instr_parts.append(f"<time_context>\nYour current time is {time_str_utc}.\n</time_context>")

        if persona_data and any(persona_data.values()):
            final_instr_parts.append("<persona_profile>")
            for key in self.persona_modal_sections_order: 
                if lines := persona_data.get(key,[]):
                    decrypted_lines = [self._decrypt_data(line) for line in lines if line.strip()]
                    if any(l.strip() for l in decrypted_lines):
                        final_instr_parts.extend([f"<{key}>"] + decrypted_lines + [f"</{key}>"])
            final_instr_parts.append("</persona_profile>")
        
        current_instructions_str = "\n".join(final_instr_parts).strip()
        
        decrypted_parts = []
        if isinstance(ai_instr_str, list):
            for part in ai_instr_str:
                dec = self._decrypt_data(part)
                if dec.strip(): decrypted_parts.append(dec)
        elif isinstance(ai_instr_str, str):
            dec = self._decrypt_data(ai_instr_str)
            if dec.strip(): decrypted_parts.append(dec)
        
        if decrypted_parts:
            if current_instructions_str: current_instructions_str += "\n\n"
            current_instructions_str += "<behavior_guidelines>\n"
            current_instructions_str += "\n\n".join(decrypted_parts).strip()
            current_instructions_str += "\n</behavior_guidelines>"

        if training_examples_list:
            examples_block = "\n---\n".join(training_examples_list)
            training_prompt = f"<training_data>\nThese are crucial examples of your persona in action. You MUST emulate the style, personality, and voice shown here. Adapt the content to the current conversation, but the persona demonstrated in these examples is your primary guide.\n\n{examples_block}\n</training_data>"
            current_instructions_str += "\n\n" + training_prompt

        if recalled_ltm:
            final_instr_parts.append(f"<recalled_memories>\n{recalled_ltm}\n</recalled_memories>")

        if critic_constraints:
            current_instructions_str += f"\n\n<negative_constraints>\nSTRICT ADHERENCE REQUIRED:\n{critic_constraints}\n</negative_constraints>"
        
        rule_block = (
            "<context_rules>\n"
            "CONTEXT PARSING RULES\n"
            "PRESENT: [Name] [Timestamp]: Indicates a person is in this conversation.\n"
            "ABSENT: Any name without brackets is a reference from memory/training, not a current participant.\n"
            "CONTEXT: Memories and Training provide background information and may reference absent individuals.\n"
            "TIME AWARENESS: Your own previous messages include a metadata line showing when you started thinking. Use this to gauge the passage of time.\n\n"
            "CRITICAL: Your response MUST be ONLY the raw text content of your message. Do NOT include your own display name. Do NOT repeat metadata from previous messages.\n"
            "</context_rules>"
        )
        current_instructions_str += "\n\n" + rule_block

        final_system_instruction = current_instructions_str if current_instructions_str.strip() else DEFAULT_SYSTEM_INSTRUCTION
        return final_system_instruction, False, grounding_enabled, temperature, top_p, top_k, primary_model, fallback_model

    async def _validate_api_keys(self, gemini_key: str, openrouter_key: str) -> Tuple[bool, str, str]:
        """Validates API keys using the new Google Gen AI SDK. Returns (is_valid, error_message, tier)."""
        detected_tier = "free"
        
        if gemini_key:
            try:
                # Initialize new SDK client
                test_client = google_genai.Client(
                    api_key=gemini_key, 
                    http_options=google_genai_types.HttpOptions(api_version='v1alpha')
                )
                
                # Step 1: Authentication Check (Is the key valid?)
                await test_client.aio.models.generate_content(
                    model='gemini-flash-lite-latest', 
                    contents="ping",
                    config=google_genai_types.GenerateContentConfig(max_output_tokens=1)
                )

                # Step 2: Tier Detection (Does it have access to premium-only models?)
                try:
                    await test_client.aio.models.generate_content(
                        model='gemini-2.5-flash-image', 
                        contents="ping",
                        config=google_genai_types.GenerateContentConfig(max_output_tokens=1)
                    )
                    detected_tier = "paid"
                except Exception:
                    # Key is valid, but rejected by a restricted model
                    detected_tier = "free"

            except Exception as e:
                return False, f"Google Gemini API validation failed: {str(e)}", "none"

        if openrouter_key:
            try:
                async with httpx.AsyncClient() as client:
                    headers = {"Authorization": f"Bearer {openrouter_key}"}
                    response = await client.get("https://openrouter.ai/api/v1/auth/key", headers=headers)
                    
                    if response.status_code == 401:
                        return False, "The OpenRouter API key provided is invalid or has been revoked.", "none"
                    elif response.status_code != 200:
                        return False, f"OpenRouter validation failed with status code: {response.status_code}", "none"
                    
                    detected_tier = "paid" 

            except httpx.RequestError as e:
                return False, f"Could not validate the OpenRouter key due to a network error: {e}", "none"
            except Exception as e:
                return False, f"An unexpected error occurred while validating the OpenRouter key: {e}", "none"
        
        return True, "", detected_tier
    
    async def _multi_profile_worker(self, channel_id: int):
        session = self.multi_profile_channels.get(channel_id)
        if not session: return

        session_type = session.get("type", "multi")
        session = self._ensure_session_hydrated(channel_id, session_type)
        if not session:
            print(f"Worker for channel {channel_id} could not hydrate session. Aborting.")
            return

        from google.generativeai.types import content_types
        session['is_running'] = True
        
        # Local cache to prevent processing the same message ID multiple times in a short loop
        recent_processed_ids = collections.deque(maxlen=20)
        
        while True:
            try:
                if session.get("type") == "freewill" and session.get("freewill_mode") != "proactive":
                    current_opted_in_dicts = []
                    channel = self.bot.get_channel(channel_id)
                    guild = channel.guild
                    guild_id_str = str(guild.id)

                    server_participation = self.freewill_participation.get(guild_id_str, {})
                    channel_participants = server_participation.get(str(channel_id), {})
                    for user_id_str, profiles in channel_participants.items():
                        member = guild.get_member(int(user_id_str))
                        if not member: continue

                        for profile_name, settings in profiles.items():
                            if settings.get("personality", "off") != "off":
                                p_dict = self._build_freewill_participant_dict(int(user_id_str), profile_name, channel)
                                if p_dict: current_opted_in_dicts.append(p_dict)

                    session_participants_set = { (p['owner_id'], p['profile_name']) for p in session['profiles'] }
                    master_participants_set = { (p['owner_id'], p['profile_name']) for p in current_opted_in_dicts }

                    added_keys = master_participants_set - session_participants_set
                    if added_keys:
                        history_to_copy_obj = next(iter(session["chat_sessions"].values()), None)
                        history_to_copy = history_to_copy_obj.history if history_to_copy_obj else []
                        
                        for p_dict in current_opted_in_dicts:
                            if (p_dict['owner_id'], p_dict['profile_name']) in added_keys:
                                session['profiles'].append(p_dict)
                                model = genai.GenerativeModel('gemini-flash-latest')
                                session['chat_sessions'][(p_dict['owner_id'], p_dict['profile_name'])] = model.start_chat(history=list(history_to_copy))

                    removed_keys = session_participants_set - master_participants_set
                    if removed_keys:
                        session['profiles'] = [p for p in session['profiles'] if (p['owner_id'], p['profile_name']) not in removed_keys]
                        for key in removed_keys:
                            session['chat_sessions'].pop(key, None)
                            session.get('initial_turn_taken', set()).discard(key)

                initial_trigger = None
                # Define flag early to avoid NameError
                is_proactive_auto_round = False
                
                timeout = None
                if session.get("freewill_mode") == "proactive":
                    timeout = float(session.get("proactive_cooldown", 300))
                
                try:
                    initial_trigger = await asyncio.wait_for(session['task_queue'].get(), timeout=timeout)
                    if initial_trigger is None and session.get("freewill_mode") == "proactive":
                        is_proactive_auto_round = True

                except asyncio.TimeoutError:
                    if session.get("type") == "freewill" and session.get("freewill_mode") == "proactive":
                        break
                    else:
                        self.multi_profile_channels.pop(channel_id, None)
                        self._save_multi_profile_sessions()
                        break
                except asyncio.CancelledError:
                    raise

                # [NEW] Gather all batched triggers immediately
                all_triggers_for_round = [initial_trigger]
                while not session['task_queue'].empty():
                    try: all_triggers_for_round.append(session['task_queue'].get_nowait())
                    except asyncio.QueueEmpty: break
                
                primary_eager_placeholder = None
                is_image_gen_round = False
                image_gen_prompt = ""
                image_gen_anchor_message = None
                # [UPDATED] Store tuples of (base_text, url_context_text, media_parts) 
                new_round_turn_data = [] 
                round_author_name = "A user"
                starting_profile_override = None
                triggering_user_id = session.get("owner_id")
                
                # [NEW] Standardized initialization to prevent UnboundLocalErrors
                url_media_parts = []

                if is_proactive_auto_round and session.get("proactive_initial_rounds") == 1:
                    cast = session['profiles']
                    if len(cast) > 1:
                        target_participant = cast[1]
                        target_id = target_participant['owner_id']
                        target_profile = target_participant['profile_name']
                        target_user_data = self._get_user_data_entry(target_id)
                        target_appearance_name = target_profile
                        if target_profile in target_user_data.get("borrowed_profiles", {}):
                            borrowed_data = target_user_data["borrowed_profiles"][target_profile]
                            target_appearance_name = borrowed_data["original_profile_name"]
                        
                        target_display_name = target_appearance_name
                        if str(target_id) in self.user_appearances and target_appearance_name in self.user_appearances[str(target_id)]:
                            appearance = self.user_appearances[str(target_id)][target_appearance_name]
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
                        if i == 0 and not is_proactive_auto_round:
                            new_round_turn_data.append(("<internal_note>No response or follow up.</internal_note>", None, []))
                        continue


                    message_trigger, reaction_trigger, message_payload = None, None, None
                    
                    if i == 0:
                        if isinstance(trigger, tuple):
                            if trigger[0] == 'reply':
                                if len(trigger) == 4:
                                    _, message_trigger, starting_profile_override, primary_eager_placeholder = trigger
                                else:
                                    _, message_trigger, starting_profile_override = trigger
                            elif trigger[0] == 'reaction': _, reaction_trigger, starting_profile_override = trigger
                            elif trigger[0] == 'initial_reactive_turn': _, message_trigger, starting_profile_override = trigger
                            elif trigger[0] == 'child_mention': _, message_payload, starting_profile_override = trigger
                            elif trigger[0] == 'ad_hoc_mention': _, message_trigger, starting_profile_override = trigger
                        elif isinstance(trigger, discord.RawReactionActionEvent): reaction_trigger = trigger
                        elif isinstance(trigger, str):
                            # Handle string prompts
                            content = trigger
                            author_name = "System"
                            created_at = datetime.datetime.now(datetime.timezone.utc)
                            
                            turn_id = str(uuid.uuid4())
                            turn_object = {
                                "turn_id": turn_id, "speaker_key": ["system", "system"],
                                "content": f"{author_name} {created_at.isoformat()}:\n{content}\n",
                                "timestamp": created_at.isoformat()
                            }
                            session.get("unified_log", []).append(turn_object)
                            
                            new_round_turn_data.append((content, None, []))
                            
                            message_trigger = None
                        else: message_trigger = trigger
                    else:
                        message_trigger = trigger if isinstance(trigger, discord.Message) else None

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
                        content = f"{reply_context} {content}" if reply_context else content
                        
                        # [NEW] URL Context Logic: Enforce Profile Setting & Separation
                        any_url_enabled = False
                        for p in session['profiles']:
                            p_udata = self._get_user_data_entry(p['owner_id'])
                            p_is_b = p['profile_name'] in p_udata.get("borrowed_profiles", {})
                            p_settings = p_udata.get("borrowed_profiles" if p_is_b else "profiles", {}).get(p['profile_name'], {})
                            if p_settings.get("url_fetching_enabled", True):
                                any_url_enabled = True; break
                        
                        url_text_content = None
                        trigger_media_parts = []
                        
                        if any_url_enabled:
                            url_text_list, url_media = await self._process_urls_in_content(content, trigger_obj['guild_id'] if is_child_mention else trigger_obj.guild.id, {"url_fetching_enabled": True})
                            if url_text_list:
                                url_text_content = "\n".join(url_text_list)
                            
                            # [UPDATED] Ensure URL media is tracked for the whole round
                            url_media_parts.extend(url_media)
                            trigger_media_parts = url_media

                        # [NEW] Localized User Timestamp Logic
                        u_data_author = self._get_user_data_entry(triggering_user_id)
                        u_prof_author = self._get_active_user_profile_name_for_channel(triggering_user_id, channel_id)
                        u_is_b_author = u_prof_author in u_data_author.get("borrowed_profiles", {})
                        u_sett_author = u_data_author.get("borrowed_profiles" if u_is_b_author else "profiles", {}).get(u_prof_author, {})
                        author_tz = u_sett_author.get("timezone", "UTC")

                        created_at = datetime.datetime.now(datetime.timezone.utc) if is_child_mention else trigger_obj.created_at
                        user_line = self._format_history_entry(author_name, created_at, content, author_tz)
                        
                        turn_id = str(uuid.uuid4())
                        
                        turn_object = {
                            "turn_id": turn_id, "speaker_key": [triggering_user_id, "user"],
                            "content": user_line, "timestamp": created_at.isoformat()
                        }
                        if url_text_content:
                            # Clear any previous URL context from the log to make the new one exclusive
                            for turn in session.get("unified_log", []):
                                if "url_context" in turn:
                                    del turn["url_context"]
                            turn_object["url_context"] = url_text_content
                            
                        session.get("unified_log", []).append(turn_object)

                        session_type = session.get("type", "multi")
                        turn_data = (channel_id, session_type, turn_id)
                        trigger_id = trigger_obj['id'] if is_child_mention else trigger_obj.id
                        self.message_to_history_turn[trigger_id] = turn_data
                        
                        mapping_key = (session_type, channel_id)
                        if mapping_key not in self.mapping_caches:
                            self.mapping_caches[mapping_key] = self._load_mapping_from_disk(mapping_key)
                        self.mapping_caches[mapping_key][str(trigger_id)] = turn_data
                        
                        # Initialize list for standard message attachments/reply images
                        new_message_parts = []
                        
                        # --- Logic to fetch image from replied-to message ---
                        msg_for_ref = message_trigger
                        if not msg_for_ref and message_payload:
                            try:
                                # For child bots, we only have payload, so fetch the discord.Message
                                r_ch = self.bot.get_channel(message_payload['channel_id'])
                                if r_ch:
                                    msg_for_ref = await r_ch.fetch_message(message_payload['id'])
                            except Exception: pass

                        if msg_for_ref and msg_for_ref.reference:
                            ref_img = None 
                            try:
                                ref_msg = msg_for_ref.reference.resolved
                                if not ref_msg:
                                    r_ch = self.bot.get_channel(msg_for_ref.reference.channel_id)
                                    if r_ch:
                                        ref_msg = await r_ch.fetch_message(msg_for_ref.reference.message_id)
                                
                                if ref_msg and ref_msg.attachments:
                                    # Find the first image/audio/video attachment in the referenced message
                                    ref_media = next((a for a in ref_msg.attachments if a.content_type and (a.content_type.startswith("image/") or a.content_type.startswith("audio/") or a.content_type.startswith("video/"))), None)
                                    if ref_media:
                                        async with httpx.AsyncClient() as client:
                                            resp = await client.get(ref_media.url)
                                            if resp.status_code == 200:
                                                media_data = await resp.aread()
                                                new_message_parts.append({"mime_type": ref_media.content_type, "data": media_data})
                            except Exception as e:
                                print(f"Error fetching replied media: {e}")
                        # ----------------------------------------------------

                        attachments = trigger_obj['attachments'] if is_child_mention else [a for a in trigger_obj.attachments if a.content_type and (a.content_type.startswith("image/") or a.content_type.startswith("audio/") or a.content_type.startswith("video/"))]
                        if attachments:
                            async with httpx.AsyncClient() as client:
                                for attachment in attachments:
                                    try:
                                        attachment_url = attachment['url'] if is_child_mention else attachment.url
                                        response = await client.get(attachment_url)
                                        response.raise_for_status()
                                        media_data = await response.aread()
                                        
                                        # Infer mime type from headers if not explicitly available object
                                        ctype = response.headers.get("Content-Type", "image/png")
                                        if hasattr(attachment, 'content_type') and attachment.content_type:
                                            ctype = attachment.content_type
                                        elif isinstance(attachment, dict) and 'content_type' in attachment: # Handle payload dict
                                            ctype = attachment['content_type']
                                        
                                        new_message_parts.append({"mime_type": ctype, "data": media_data})
                                        del media_data
                                    except Exception as e:
                                        print(f"Failed to process media attachment in multi-profile trigger: {e}")
                        # ----------------------------------------------------

                        # [UPDATED] Include audio/ MIME types
                        attachments = trigger_obj['attachments'] if is_child_mention else [a for a in trigger_obj.attachments if a.content_type and (a.content_type.startswith("image/") or a.content_type.startswith("audio/"))]
                        if attachments:
                            async with httpx.AsyncClient() as client:
                                for attachment in attachments:
                                    try:
                                        attachment_url = attachment['url'] if is_child_mention else attachment.url
                                        response = await client.get(attachment_url)
                                        response.raise_for_status()
                                        media_data = await response.aread()
                                        
                                        # Infer mime type
                                        ctype = response.headers.get("Content-Type", "image/png")
                                        if hasattr(attachment, 'content_type') and attachment.content_type:
                                            ctype = attachment.content_type
                                        elif isinstance(attachment, dict) and 'content_type' in attachment:
                                            ctype = attachment['content_type']
                                        
                                        new_message_parts.append({"mime_type": ctype, "data": media_data})
                                        del media_data
                                    except Exception as e:
                                        print(f"Failed to process media attachment in multi-profile trigger: {e}")

                        # Combine standard attachments with URL-extracted media
                        trigger_media_parts.extend(new_message_parts)
                        
                        # Store raw components for gating logic
                        new_round_turn_data.append((user_line, url_text_content, trigger_media_parts))

                        if triggering_user_id in self.debug_users:
                            try:
                                user_to_dm = self.bot.get_user(triggering_user_id)
                                if user_to_dm:
                                    # Create a temporary debug obj
                                    debug_parts = [user_line]
                                    if url_text_content: debug_parts.append(url_text_content)
                                    debug_parts.extend(trigger_media_parts)
                                    debug_obj = content_types.to_content({'role': 'user', 'parts': debug_parts})
                                    
                                    debug_message = self._format_debug_prompt([debug_obj])
                                    await user_to_dm.send(debug_message)
                            except Exception as e:
                                print(f"Failed to send user turn debug DM to user {triggering_user_id}: {e}")

                    elif reaction_trigger and i == 0:
                        triggering_user_id = reaction_trigger.user_id

                # [UPDATED] Apply new turns to all chat sessions with gating
                for base_text, url_ctx, media in new_round_turn_data:
                    for p_key, chat_session in session['chat_sessions'].items():
                        # Determine if this specific profile allows URL context
                        p_owner_id, p_name = p_key
                        p_udata = self._get_user_data_entry(p_owner_id)
                        p_is_b = p_name in p_udata.get("borrowed_profiles", {})
                        p_settings = p_udata.get("borrowed_profiles" if p_is_b else "profiles", {}).get(p_name, {})
                        
                        final_parts = [base_text]
                        
                        if url_ctx and p_settings.get("url_fetching_enabled", True):
                            final_parts.append(f"\n[System: Context from Link]\n{url_ctx}")
                        
                        # Attachments (including URL media) are added for everyone if present
                        final_parts.extend(media)
                        
                        content_obj = content_types.to_content({'role': 'user', 'parts': final_parts})
                        chat_session.history.append(content_obj)

                for content_obj in new_round_turn_data:
                    # Logic to populate chat_sessions handled below by new_round_turn_data loop
                    pass

                profile_order = []
                freewill_mode = session.get("freewill_mode")
                session_mode = session.get("session_mode", "sequential")
                channel = self.bot.get_channel(channel_id)

                if freewill_mode == 'reactive':
                    if starting_profile_override:
                        profile_order = [starting_profile_override]
                    else:
                        initial_turn_profile = None
                        if isinstance(initial_trigger, tuple) and initial_trigger[0] == 'initial_reactive_turn':
                            _, message_trigger, initial_turn_profile = initial_trigger
                        else:
                            message_trigger = initial_trigger if isinstance(initial_trigger, discord.Message) else None

                        if message_trigger:
                            content_lower = message_trigger.content.lower()
                            wakeword_triggers = []
                            server_participation = self.freewill_participation.get(str(channel.guild.id), {})
                            channel_participants = server_participation.get(str(channel.id), {})
                            
                            for p in session['profiles']:
                                user_profiles = channel_participants.get(str(p['owner_id']), {})
                                profile_settings = user_profiles.get(p['profile_name'], {})
                                if any(w.lower() in content_lower for w in profile_settings.get("wakewords", [])):
                                    wakeword_triggers.append(p)

                            if wakeword_triggers:
                                profile_order = [random.choice(wakeword_triggers)]
                            else:
                                # [UPDATED] Numeric Chance Logic
                                participants_for_roll = []
                                weights = []
                                highest_chance = 0.0

                                for p in session['profiles']:
                                    user_profiles = channel_participants.get(str(p['owner_id']), {})
                                    profile_settings = user_profiles.get(p['profile_name'], {})
                                    
                                    pers = profile_settings.get("personality", "off")
                                    chance = 0.0
                                    
                                    if isinstance(pers, int):
                                        chance = pers / 100.0
                                    else:
                                        chance = {"introverted": 0.03, "regular": 0.10, "outgoing": 0.30}.get(pers, 0.0)
                                    
                                    if chance > 0:
                                        participants_for_roll.append(p)
                                        weights.append(chance)
                                        if chance > highest_chance: highest_chance = chance
                                
                                if highest_chance > 0 and random.random() <= highest_chance:
                                    if participants_for_roll and any(w > 0 for w in weights):
                                        profile_order = random.choices(participants_for_roll, weights=weights, k=1)

                        if initial_turn_profile and not profile_order:
                            profile_order = [initial_turn_profile]
                
                else: # Standard multi-profile or proactive freewill
                    if starting_profile_override:
                        start_p = starting_profile_override
                        
                        session_mode = session.get("session_mode", "sequential")
                        if session_mode == 'sequential':
                            try:
                                # Find the index of the profile we are starting with
                                start_idx = session['profiles'].index(start_p)
                                # Rotate the list to make that profile the first element
                                new_order = session['profiles'][start_idx:] + session['profiles'][:start_idx]
                                session['profiles'] = new_order
                                self._save_multi_profile_sessions()
                            except ValueError:
                                # The starting profile wasn't in the list, this shouldn't happen but handle gracefully
                                pass

                        # The temporary profile_order for this round is now based on the (potentially new) session order
                        profile_order = list(session['profiles'])
                        if session_mode == 'random':
                            # For random, we still respect the override for the first turn, then shuffle the rest
                            if start_p in profile_order:
                                profile_order.remove(start_p)
                            random.shuffle(profile_order)
                            profile_order.insert(0, start_p)

                    else:
                        # No override, determine next speaker based on last speaker
                        session_mode = session.get("session_mode", "sequential")
                        if session_mode == 'sequential' and session.get('last_speaker_key'):
                            try:
                                last_speaker_index = next(i for i, p in enumerate(session['profiles']) if (p['owner_id'], p['profile_name']) == session['last_speaker_key'])
                                start_index = (last_speaker_index + 1) % len(session['profiles'])
                                # Permanently rotate the session's profile list
                                new_order = session['profiles'][start_index:] + session['profiles'][:start_index]
                                session['profiles'] = new_order
                                self._save_multi_profile_sessions()
                            except (ValueError, StopIteration):
                                pass # Last speaker not found, use default order
                        
                        # The temporary profile_order for this round is now based on the (potentially new) session order
                        profile_order = list(session['profiles'])
                        if session_mode == 'random':
                            random.shuffle(profile_order)
                        elif session.get('last_speaker_key'):
                            # In sequential mode, start after the last speaker
                            try:
                                last_speaker_index = next(i for i, p in enumerate(profile_order) if (p['owner_id'], p['profile_name']) == session['last_speaker_key'])
                                start_index = (last_speaker_index + 1) % len(profile_order)
                                profile_order = profile_order[start_index:] + profile_order[:start_index]
                            except (ValueError, StopIteration):
                                pass # Last speaker not found, use default order

                # --- Ephemeral Participant Injection ---
                ephemeral_participant = None
                if isinstance(initial_trigger, tuple):
                    if initial_trigger[0] == 'child_mention' or initial_trigger[0] == 'ad_hoc_mention':
                        _, _, ephemeral_participant = initial_trigger

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

                # --- Ensure all participants for this round have a chat session ---
                from google.generativeai.types import content_types
                dummy_model = genai.GenerativeModel('gemini-flash-latest')
                for p_data in profile_order:
                    p_key = (p_data['owner_id'], p_data['profile_name'])
                    if p_key not in session['chat_sessions'] or session['chat_sessions'][p_key] is None:
                        participant_history = []
                        for turn in session.get("unified_log", []):
                            speaker_key = tuple(turn.get("speaker_key", []))
                            role = 'model' if speaker_key == p_key else 'user'
                            content_obj = content_types.to_content({'role': role, 'parts': [turn.get("content")]})
                            participant_history.append(content_obj)
                        session["chat_sessions"][p_key] = dummy_model.start_chat(history=participant_history)
                
                # [UPDATED] Pre-Check API Keys before ANY visual feedback
                channel = self.bot.get_channel(channel_id)
                has_gemini = self._get_api_key_for_guild(channel.guild.id, "gemini")
                has_openrouter = self._get_api_key_for_guild(channel.guild.id, "openrouter")
                
                if not has_gemini and not has_openrouter:
                    try:
                        await channel.send("An API key has not been configured for this server. You can use the `/settings` command in my DM to set one.")
                    except discord.Forbidden: pass
                    
                    # Mark triggers as done to prevent queue stalling
                    for trigger in all_triggers_for_round:
                        if trigger is not None: session['task_queue'].task_done()
                    continue

                was_blocked = False
                generated_image_bytes_for_round = None
                generator_profile_key = None
                generator_display_name = "A participant"

                if is_image_gen_round:
                    if starting_profile_override:
                        generator_profile_key = (starting_profile_override['owner_id'], starting_profile_override['profile_name'])
                    elif profile_order:
                        first_participant = profile_order[0]
                        generator_profile_key = (first_participant['owner_id'], first_participant['profile_name'])
                    
                    if generator_profile_key:
                        gen_owner_id, gen_profile_name = generator_profile_key
                        gen_user_data = self._get_user_data_entry(gen_owner_id)
                        gen_is_borrowed = gen_profile_name in gen_user_data.get("borrowed_profiles", {})
                        gen_effective_owner_id = gen_owner_id
                        gen_effective_profile_name = gen_profile_name
                        if gen_is_borrowed:
                            borrowed_data = gen_user_data["borrowed_profiles"][gen_profile_name]
                            gen_effective_owner_id = int(borrowed_data["original_owner_id"])
                            gen_effective_profile_name = borrowed_data["original_profile_name"]
                        
                        gen_appearance_data = self.user_appearances.get(str(gen_effective_owner_id), {}).get(gen_effective_profile_name, {})
                        if gen_appearance_data.get("custom_display_name"):
                            generator_display_name = gen_appearance_data["custom_display_name"]
                        else:
                            generator_display_name = gen_effective_profile_name

                responses_this_round = []
                # [NEW] Track round-specific audio segments for stitching
                round_audio_segments = []
                initial_round_context = ""
                
                # [NEW] Batch Intent Tracking
                batched_url_research_content = []
                # ----------------------------

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

                # --- Immediate Feedback Step ---
                first_participant = profile_order[0] if profile_order else None
                first_placeholder_message = None
                if first_participant:
                    if first_participant.get('method') == 'child_bot':
                        await self.manager_queue.put({
                            "action": "send_to_child", "bot_id": first_participant['bot_id'],
                            "payload": {"action": "start_typing", "channel_id": channel_id}
                        })
                    else: # Webhook
                        if primary_eager_placeholder:
                            first_placeholder_message = primary_eager_placeholder
                        else:
                            thinking_messages = await self._send_channel_message(
                                channel, f"{PLACEHOLDER_EMOJI}",
                                profile_owner_id_for_appearance=first_participant['owner_id'], 
                                profile_name_for_appearance=first_participant['profile_name']
                            )
                            if thinking_messages: first_placeholder_message = thinking_messages[0]

                grounding_context, grounding_sources = None, []
                grounding_profile_key = None
                grounding_mode_for_citator = "off"

                grounding_target_participant = starting_profile_override or (profile_order[0] if profile_order else None)

                if grounding_target_participant:
                    g_owner_id = grounding_target_participant['owner_id']
                    g_profile_name = grounding_target_participant['profile_name']
                    g_user_data = self._get_user_data_entry(g_owner_id)
                    g_is_borrowed = g_profile_name in g_user_data.get("borrowed_profiles", {})
                    g_profile_settings = g_user_data.get("borrowed_profiles" if g_is_borrowed else "profiles", {}).get(g_profile_name, {})
                    
                    grounding_mode = g_profile_settings.get("grounding_mode", "off")
                    if isinstance(grounding_mode, bool): grounding_mode = "on" if grounding_mode else "off"
                    grounding_mode_for_citator = grounding_mode

                    if grounding_mode in ["on", "on+"]:
                        g_participant_key = (g_owner_id, g_profile_name)
                        g_chat_session = session['chat_sessions'].get(g_participant_key)
                        history_for_grounding = []
                        if g_chat_session:
                            g_stm_length = int(g_profile_settings.get("stm_length", defaultConfig.CHATBOT_MEMORY_LENGTH))
                            g_stm_capped = min(10, g_stm_length)
                            history_for_grounding = g_chat_session.history[-(g_stm_capped * 2):] if g_stm_capped > 0 else []

                        # Safety Logic for Grounding
                        g_safety_level = g_profile_settings.get('safety_level', 'low')
                        g_safety_map = { "unrestricted": HarmBlockThreshold.BLOCK_NONE, "low": HarmBlockThreshold.BLOCK_ONLY_HIGH, "medium": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, "high": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE }
                        g_threshold = g_safety_map.get(g_safety_level, HarmBlockThreshold.BLOCK_ONLY_HIGH)
                        g_dynamic_safety_settings = { cat: g_threshold for cat in get_args(HarmCategory) }

                        is_for_image_flag = is_image_gen_round
                        grounding_query = image_gen_prompt if is_image_gen_round else initial_round_context

                        mapping_key = (session.get("type", "multi"), channel.id)
                        grounding_result = await self._get_hybrid_grounding_context(grounding_query, channel.guild.id, history_for_grounding, mapping_key, safety_settings=g_dynamic_safety_settings, is_for_image=is_for_image_flag)
                        if grounding_result:
                            g_context, g_sources, should_set_checkpoint = grounding_result
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
                            if should_set_checkpoint:
                                trigger_message = None
                                if isinstance(initial_trigger, discord.Message):
                                    trigger_message = initial_trigger
                                elif isinstance(initial_trigger, tuple) and len(initial_trigger) > 1 and isinstance(initial_trigger[1], discord.Message):
                                    trigger_message = initial_trigger[1]
                                
                                if trigger_message:
                                    turn_info = self.message_to_history_turn.get(trigger_message.id)
                                    if turn_info and len(turn_info) > 2:
                                        turn_id = turn_info[2]
                                        if mapping_key not in self.mapping_caches: self.mapping_caches[mapping_key] = {}
                                        self.mapping_caches[mapping_key]['grounding_checkpoint'] = turn_id

                # [NEW] Phase: Research Once (URL Context)
                round_url_text_contexts = []
                
                any_url_enabled = False
                for p in session['profiles']:
                    p_udata = self._get_user_data_entry(p['owner_id'])
                    p_is_b = p['profile_name'] in p_udata.get("borrowed_profiles", {})
                    p_settings = p_udata.get("borrowed_profiles" if p_is_b else "profiles", {}).get(p['profile_name'], {})
                    if p_settings.get("url_fetching_enabled", True):
                        any_url_enabled = True; break
                
                if any_url_enabled and batched_url_research_content:
                    print(f"[DEBUG: URL-Multi] Performing round research on {len(batched_url_research_content)} items.")
                    for content_str, g_id in batched_url_research_content:
                        u_t, _ = await self._process_urls_in_content(content_str, g_id, {"url_fetching_enabled": True})
                        round_url_text_contexts.extend(u_t)

                for i, participant in enumerate(profile_order):
                    channel = self.bot.get_channel(channel_id)
                    api_key = self._get_api_key_for_guild(channel.guild.id)
                    
                    # Initialize turn-specific variables at the very start of the loop
                    is_generator = False
                    p_settings = {}
                    participant_key = (participant['owner_id'], participant['profile_name'])
                    contents_for_api_call = [] 
                    fallback_used = False
                    response_text = ""
                    was_blocked = False
                    placeholder_message = None
                    
                    if not api_key:
                        if i == 0:
                            try:
                                await channel.send("An API key must be configured on this server for sessions.")
                            except discord.Forbidden:
                                pass
                        break

                    # Resolve Real-time settings
                    p_owner_id = participant['owner_id']
                    p_name = participant['profile_name']
                    p_udata = self._get_user_data_entry(p_owner_id)
                    p_is_b = p_name in p_udata.get("borrowed_profiles", {})
                    p_settings = p_udata.get("borrowed_profiles" if p_is_b else "profiles", {}).get(p_name, {})

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
                    self.session_last_accessed[channel_id] = time.time()
                    participant_key = (participant['owner_id'], participant['profile_name'])
                    chat_session = session['chat_sessions'].get(participant_key)

                    # Safety Check: If session state is corrupted (contains list instead of ChatSession), force re-hydration
                    if isinstance(chat_session, list) or chat_session is None:
                        print(f"Detected corrupted session state for {participant_key} in {channel_id}. Attempting repair.")
                        session['is_hydrated'] = False
                        session = self._ensure_session_hydrated(channel_id, session.get("type", "multi"))
                        if session:
                            chat_session = session['chat_sessions'].get(participant_key)
                        
                        if not chat_session or isinstance(chat_session, list):
                            print(f"Critical: Could not repair session for {participant_key}. Skipping turn.")
                            continue

                    if session.get("type") == "freewill" and participant_key not in session.get("initial_turn_taken", set()):
                        initial_history_data = session.get("initial_channel_history", [])
                        initial_history_content = []
                        
                        participant_user_id = None
                        if participant.get('method') == 'child_bot':
                            participant_user_id = int(participant['bot_id'])
                        else: # webhook
                            participant_user_id = self.bot.user.id

                        for author_id, author_name, timestamp, content in initial_history_data:
                            role = 'model' if author_id == participant_user_id else 'user'
                            formatted_line = self._format_history_entry(author_name, timestamp, content)
                            content_obj = content_types.to_content({'role': role, 'parts': [formatted_line]})
                            initial_history_content.append(content_obj)

                        chat_session.history = initial_history_content + chat_session.history
                        session.setdefault("initial_turn_taken", set()).add(participant_key)

                    if participant.get('method') == 'child_bot':
                        await self.manager_queue.put({
                            "action": "send_to_child", "bot_id": participant['bot_id'],
                            "payload": {"action": "start_typing", "channel_id": channel_id}
                        })

                    owner_id = participant['owner_id']
                    profile_name = participant['profile_name']
                    channel = self.bot.get_channel(channel_id)

                    # [FIX] Initialize these before the try block to prevent UnboundLocalError
                    fallback_used = False
                    response_text = ""
                    was_blocked = False

                    if not self._check_unrestricted_safety_policy(owner_id, profile_name, channel):
                        error_message = f"[System Notice: '{profile_name}' cannot respond. Profiles with 'Unrestricted 18+' safety are only permitted in age-restricted channels.]"
                        
                        # Send the message immediately, bypassing placeholders/typing for this turn
                        await self._send_channel_message(channel, error_message)

                        # Log this system notice in everyone's history
                        history_line = self._format_history_entry("System", datetime.datetime.now(datetime.timezone.utc), error_message)
                        error_content_obj = content_types.to_content({'role': 'user', 'parts': [history_line]})
                        for other_chat in session['chat_sessions'].values():
                            other_chat.history.append(error_content_obj)
                        continue

                    user_data = self._get_user_data_entry(owner_id)
                    is_borrowed = profile_name in user_data.get("borrowed_profiles", {})
                    effective_owner_id = owner_id
                    effective_profile_name = profile_name
                    if is_borrowed:
                        borrowed_data = user_data["borrowed_profiles"][profile_name]
                        effective_owner_id = int(borrowed_data["original_owner_id"])
                        effective_profile_name = borrowed_data["original_profile_name"]

                    speaker_display_name = profile_name
                    appearance_data = self.user_appearances.get(str(effective_owner_id), {}).get(effective_profile_name, {})
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
                        gen_user_data = self._get_user_data_entry(gen_owner_id)
                        gen_is_borrowed = gen_profile_name in gen_user_data.get("borrowed_profiles", {})
                        gen_effective_owner_id = gen_owner_id
                        gen_effective_profile_name = gen_profile_name
                        if gen_is_borrowed:
                            borrowed_data = gen_user_data["borrowed_profiles"][gen_profile_name]
                            gen_effective_owner_id = int(borrowed_data["original_owner_id"])
                            gen_effective_profile_name = borrowed_data["original_profile_name"]
                        
                        gen_appearance_data = self.user_appearances.get(str(gen_effective_owner_id), {}).get(gen_effective_profile_name, {})
                        if gen_appearance_data.get("custom_display_name"):
                            generator_display_name = gen_appearance_data["custom_display_name"]
                        else:
                            generator_display_name = gen_effective_profile_name
                    
                    # Resolve Real-time settings
                        p_owner_id = participant['owner_id']
                        p_name = participant['profile_name']
                        p_udata = self._get_user_data_entry(p_owner_id)
                        p_is_b = p_name in p_udata.get("borrowed_profiles", {})
                        p_settings = p_udata.get("borrowed_profiles" if p_is_b else "profiles", {}).get(p_name, {})

                        if p_settings.get("url_fetching_enabled", True) and round_url_text_contexts:
                            url_instr = "<url_research>\n[Context from links in current messages]:\n" + "\n".join(round_url_text_contexts) + "\n</url_research>"
                            contents_for_api_call.append(content_types.to_content({'role': 'user', 'parts': [url_instr]}))

                        if not contents_for_api_call:
                            contents_for_api_call.append(content_types.to_content({'role': 'user', 'parts': ["<internal_note>Start the conversation.</internal_note>"]}))

                        ltm_recall_text = await self._get_relevant_ltm_for_prompt(session_key, chat_session.history, owner_id, profile_name, dynamic_context_for_turn, round_author_name, channel.guild.id, triggering_user_id)

                        # [NEW] Check Image Gen intent vs Profile Toggle
                        turn_is_image_gen = False
                        if is_image_gen_round:
                            if p_settings.get("image_generation_enabled", True):
                                turn_is_image_gen = True
                            else:
                                # Re-inject prefix if toggle is OFF
                                initial_round_context = f"!image {image_gen_prompt}\n{initial_round_context}"

                        is_generator = turn_is_image_gen and participant_key == generator_profile_key

                    try:
                        api_key = self._get_api_key_for_guild(channel.guild.id)
                        if not api_key: raise ValueError("Server API key is not configured.")
                        genai.configure(api_key=api_key)
                        
                        if i == 0 and first_placeholder_message:
                            placeholder_message = first_placeholder_message
                        elif i > 0: # Not the first participant
                            if participant.get('method') == 'child_bot':
                                await self.manager_queue.put({
                                    "action": "send_to_child", "bot_id": participant['bot_id'],
                                    "payload": {"action": "start_typing", "channel_id": channel_id}
                                })
                            else: # Webhook
                                thinking_messages = await self._send_channel_message(
                                    channel, f"{PLACEHOLDER_EMOJI}",
                                    profile_owner_id_for_appearance=owner_id, profile_name_for_appearance=profile_name
                                )
                                if thinking_messages: placeholder_message = thinking_messages[0]

                        if is_generator and generated_image_bytes_for_round is None:
                            if self.image_gen_semaphore.locked() and image_gen_anchor_message:
                                try:
                                    if isinstance(image_gen_anchor_message, discord.Message):
                                        await image_gen_anchor_message.reply("Your image generation request has been queued. It will be processed when the single spot opens.", delete_after=10)
                                except Exception: pass

                            async with self.image_gen_semaphore:
                                system_instruction = self._get_image_gen_system_instruction(owner_id, profile_name)
                                
                                # Get appearance text
                                source_owner_id = owner_id
                                source_profile_name = profile_name
                                if is_borrowed:
                                    borrowed_data = user_data["borrowed_profiles"][profile_name]
                                    source_owner_id = int(borrowed_data["original_owner_id"])
                                    source_profile_name = borrowed_data["original_profile_name"]
                                
                                source_owner_data = self._get_user_data_entry(source_owner_id)
                                source_profile_data = source_owner_data.get("profiles", {}).get(source_profile_name, {})
                                persona = source_profile_data.get("persona", {})
                                appearance_lines_encrypted = persona.get("appearance", [])
                                appearance_text = "\n".join([self._decrypt_data(line) for line in appearance_lines_encrypted])

                                final_prompt_text = image_gen_prompt
                                if appearance_text.strip():
                                    prompt_lower = image_gen_prompt.lower()
                                    second_person_pronouns = ["you", "your", "yourself", "u", "ur"]
                                    if any(pronoun in prompt_lower.split() for pronoun in second_person_pronouns) or \
                                    generator_display_name.lower() in prompt_lower or \
                                    gen_profile_name.lower() in prompt_lower:
                                        final_prompt_text = f"Your appearance:\n{appearance_text.strip()}\n\nUser's prompt:\n{image_gen_prompt}"

                                profile_settings = user_data.get("borrowed_profiles" if is_borrowed else "profiles", {}).get(profile_name, {})
                                safety_level_str = profile_settings.get('safety_level', 'low')

                                safety_map = { "unrestricted": HarmBlockThreshold.BLOCK_NONE, "low": HarmBlockThreshold.BLOCK_ONLY_HIGH, "medium": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, "high": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE }
                                threshold = safety_map.get(safety_level_str, HarmBlockThreshold.BLOCK_ONLY_HIGH)
                                dynamic_safety_settings = { cat: threshold for cat in get_args(HarmCategory) }

                                image_model = genai.GenerativeModel('gemini-2.5-flash-image', system_instruction=system_instruction, safety_settings=dynamic_safety_settings)
                                
                                parts = [final_prompt_text]
                                if image_gen_anchor_message:
                                    # FIXED: Cumulative Image Gathering (Max 2 images)
                                    # Logic handles Dict payloads (Child Bot) and Discord Objects (Webhook/User)
                                    
                                    # 1. Try to fetch from Reply Reference
                                    ref_msg_id = None
                                    ref_channel_id = None
                                    
                                    if isinstance(image_gen_anchor_message, dict):
                                        replied_to = image_gen_anchor_message.get('replied_to')
                                        if replied_to:
                                            ref_msg_id = replied_to.get('id')
                                            ref_channel_id = replied_to.get('channel_id')
                                    else:
                                        if image_gen_anchor_message.reference:
                                            ref_msg_id = image_gen_anchor_message.reference.message_id
                                            ref_channel_id = image_gen_anchor_message.reference.channel_id

                                    if ref_msg_id:
                                        try:
                                            fetch_channel = channel
                                            if ref_channel_id and ref_channel_id != channel.id:
                                                fetch_channel = self.bot.get_channel(ref_channel_id)
                                            
                                            if fetch_channel:
                                                ref_msg = await fetch_channel.fetch_message(ref_msg_id)
                                                if ref_msg.attachments and ref_msg.attachments[0].content_type.startswith("image/"):
                                                    image_data = await ref_msg.attachments[0].read()
                                                    parts.append({"mime_type": ref_msg.attachments[0].content_type, "data": image_data})
                                                    del image_data
                                        except Exception as e:
                                            print(f"Error fetching referenced image for generation: {e}")
                                    
                                    # 2. Try to fetch from Current Attachments (fill remaining slots up to 2 images total)
                                    # Note: parts[0] is text, so we check if len(parts) < 3 (1 text + 2 images)
                                    if len(parts) < 3:
                                        attachments_list = []
                                        if isinstance(image_gen_anchor_message, dict):
                                            attachments_list = image_gen_anchor_message.get('attachments', [])
                                        else:
                                            attachments_list = image_gen_anchor_message.attachments

                                        for attachment in attachments_list:
                                            if len(parts) >= 3: break # Limit reached

                                            is_dict = isinstance(attachment, dict)
                                            url = attachment['url'] if is_dict else attachment.url
                                            
                                            # Filter objects
                                            if not is_dict and not (attachment.content_type and attachment.content_type.startswith("image/")):
                                                continue

                                            try:
                                                async with httpx.AsyncClient() as client:
                                                    resp = await client.get(url)
                                                    resp.raise_for_status()
                                                    image_data = await resp.aread()
                                                    ctype = resp.headers.get("Content-Type", "image/jpeg")
                                                    parts.append({"mime_type": ctype, "data": image_data})
                                                    del image_data
                                            except Exception as e:
                                                print(f"Error fetching attached image for generation: {e}")
                                
                                status = "api_error"
                                try:
                                    response = await image_model.generate_content_async(parts)
                                    status = "blocked_by_safety" if not response.candidates else "success"
                                finally:
                                    self._log_api_call(user_id=triggering_user_id, guild_id=channel.guild.id, context="image_generation_multi", model_used=image_model.model_name, status=status)

                                if response.candidates and response.candidates[0].finish_reason.name == 'STOP':
                                    for part in response.candidates[0].content.parts:
                                        if hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                                            generated_image_bytes_for_round = part.inline_data.data
                                            break
                        
                        # [RESTORED] Context Gathering for Retrieval
                        round_context_text = "\n".join([t[0] for t in new_round_turn_data])
                        dynamic_context_for_turn = round_context_text + "\n" + "\n".join(responses_this_round)

                        # [RESTORED] Training Example Injection
                        training_examples_list = await self._get_relevant_training_examples(owner_id, profile_name, dynamic_context_for_turn, channel.guild.id)
                        full_system_instruction, _, grounding_enabled, temp, top_p, top_k, primary_model, fallback_model_name = self._construct_system_instructions(
                            owner_id, profile_name, channel.id, is_multi_profile=True, training_examples_list=training_examples_list
                        )
                        
                        # [MOVED] Critic Check - Injection into System Instruction
                        profile_settings = user_data.get("borrowed_profiles" if is_borrowed else "profiles", {}).get(profile_name, {})
                        if profile_settings.get("critic_enabled", False):
                            critic_constraints = await self._run_critic(chat_session.history, speaker_display_name, channel.guild.id)
                            if critic_constraints:
                                full_system_instruction += f"\n\nNEGATIVE CONSTRAINTS (STRICT ADHERENCE REQUIRED):\n{critic_constraints}"

                        profile_settings = user_data.get("borrowed_profiles" if is_borrowed else "profiles", {}).get(profile_name, {})
                        safety_level_str = profile_settings.get('safety_level', 'low')

                        safety_map = { "unrestricted": HarmBlockThreshold.BLOCK_NONE, "low": HarmBlockThreshold.BLOCK_ONLY_HIGH, "medium": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, "high": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE }
                        threshold = safety_map.get(safety_level_str, HarmBlockThreshold.BLOCK_ONLY_HIGH)
                        dynamic_safety_settings = { cat: threshold for cat in get_args(HarmCategory) }

                        # Factory Logic
                        name_upper = primary_model.upper()
                        actual_name = primary_model
                        is_openrouter = False
                        
                        if name_upper.startswith("OPENROUTER/"):
                            actual_name = primary_model[11:]
                            is_openrouter = True
                        elif name_upper.startswith("GOOGLE/"):
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

                        if is_openrouter:
                            or_key = self._get_api_key_for_guild(channel.guild.id, provider="openrouter")
                            if or_key:
                                model = OpenRouterModel(actual_name, api_key=or_key, system_instruction=full_system_instruction)
                            else:
                                warning_message = f"API Configuration Error: OpenRouter API Key missing for this server. Cannot load model '{primary_model}'."
                        else:
                            try:
                                # [NEW] Pass thinking_params here
                                model = GoogleGenAIModel(
                                    api_key=api_key, 
                                    model_name=actual_name, 
                                    system_instruction=full_system_instruction, 
                                    safety_settings=dynamic_safety_settings,
                                    thinking_params=t_params_worker
                                )
                            except Exception as e:
                                warning_message = f"Model Initialization Error: Failed to instantiate Google model '{actual_name}'. {e}"
                        
                        from google.generativeai.types import content_types
                        contents_for_api_call = []
                        session_key = (channel.id, owner_id, profile_name)

                       # Start with a shallow list copy of the session's history
                        contents_for_api_call.extend(chat_session.history)

                        # Check if the last turn was from this model itself
                        if contents_for_api_call and contents_for_api_call[-1].role == 'model':
                            pseudo_user_turn = content_types.to_content({'role': 'user', 'parts': ["<internal_note>No response from anyone.</internal_note>"]})
                            contents_for_api_call.append(pseudo_user_turn)

                        pending_whispers = session.get("pending_whispers", {}).pop(participant_key, None)
                        if pending_whispers:
                            whisper_context = (
                                "<private_context>\n"
                                "Knowledge received since your last turn:\n"
                                "By default, do NOT reveal the content of this knowledge publicly. "
                                "Instead, let it subtly influence your tone, actions, or decisions. "
                                "Only reveal this if specifically requested by the sender.\n"
                            )
                            whisper_context += "\n---\n" + "\n---\n".join(pending_whispers) + "\n</private_context>"
                            whisper_content_obj = content_types.to_content({'role': 'user', 'parts': [whisper_context]})
                            contents_for_api_call.append(whisper_content_obj)

                        if grounding_context and p_settings.get("grounding_mode", "off") != "off":
                            g_instr = f"<external_context>\n{grounding_context}\n</external_context>"
                            contents_for_api_call.append(content_types.to_content({'role': 'user', 'parts': [g_instr]}))

                        if p_settings.get("url_fetching_enabled", True) and round_url_text_contexts:
                            url_instr = "<document_context>\n" + "\n".join(round_url_text_contexts) + "\n</document_context>"
                            contents_for_api_call.append(content_types.to_content({'role': 'user', 'parts': [url_instr]}))

                        # [NEW] Ephemeral URL Media Injection (Current Round Only)
                        if url_media_parts:
                             contents_for_api_call.append(content_types.to_content({'role': 'user', 'parts': url_media_parts}))

                        ltm_recall_text = await self._get_relevant_ltm_for_prompt(session_key, chat_session.history, owner_id, profile_name, dynamic_context_for_turn, round_author_name, channel.guild.id, triggering_user_id)
                        if ltm_recall_text:
                            ltm_content_obj = content_types.to_content({'role': 'user', 'parts': [ltm_recall_text]})
                            contents_for_api_call.append(ltm_content_obj)
                        
                        # The history is now managed by the deep copy and pseudo-turn logic above.
                        # This block is now only for adding the final user turn context if it exists.
                        
                        if is_image_gen_round:
                            if generated_image_bytes_for_round:
                                system_note = f"<image_context>You have just generated the following image based on the prompt: '{image_gen_prompt}'. Present it with a comment.</image_context>" if is_generator else f"<image_context>'{generator_display_name}' just generated the following image based on the prompt: '{image_gen_prompt}'. Comment on it.</image_context>"
                                
                                text_gen_parts = [
                                    system_note, 
                                    {"mime_type": "image/jpeg", "data": generated_image_bytes_for_round}
                                ]
                                contents_for_api_call.append(content_types.to_content({'role': 'user', 'parts': text_gen_parts}))
                            else:
                                if is_generator:
                                    system_note = f"<image_context>Your attempt to generate an image based on the prompt '{image_gen_prompt}' failed. Comment on it.</image_context>"
                                    contents_for_api_call.append(content_types.to_content({'role': 'user', 'parts': [system_note]}))

                        if not contents_for_api_call:
                            contents_for_api_call.append(content_types.to_content({'role': 'user', 'parts': ["<internal_note>Begin conversation.</internal_note>"]}))

                        # [NEW] Advanced Params Injection
                        adv_params = {
                            "frequency_penalty": profile_settings.get("frequency_penalty"),
                            "presence_penalty": profile_settings.get("presence_penalty"),
                            "repetition_penalty": profile_settings.get("repetition_penalty"),
                            "min_p": profile_settings.get("min_p"),
                            "top_a": profile_settings.get("top_a")
                        }
                        adv_params = {k: v for k, v in adv_params.items() if v is not None}

                        gen_config = genai.types.GenerationConfig(temperature=temp, top_p=top_p, top_k=top_k)
                        gen_config._advanced_params = adv_params

                        status = "api_error"
                        response = None
                        fallback_used = False
                        blocked_reason_override = None
                        
                        if model:
                            try:
                                # Pass list directly (shallow copy) to prevent memory spike
                                response = await model.generate_content_async(contents_for_api_call, generation_config=gen_config)
                                if not response or not response.candidates:
                                    raise ValueError("Response blocked or empty")
                                status = "success"
                            except (api_exceptions.ResourceExhausted, api_exceptions.InternalServerError, api_exceptions.ServiceUnavailable, api_exceptions.Aborted, Exception) as e:
                                is_rate_limit = isinstance(e, api_exceptions.ResourceExhausted) or "429" in str(e) or "rate limit" in str(e).lower()
                                if is_rate_limit: blocked_reason_override = "Rate Limit"

                                is_google_error = isinstance(e, (api_exceptions.ResourceExhausted, api_exceptions.InternalServerError, api_exceptions.ServiceUnavailable, api_exceptions.Aborted))
                                is_openrouter_error = isinstance(e, Exception) and ("OpenRouter" in str(e) or "402" in str(e) or "Response blocked" in str(e))
                                
                                if (is_google_error or is_openrouter_error) and fallback_model_name:
                                    try:
                                        fb_name = fallback_model_name
                                        fb_is_or = False
                                        
                                        if fb_name.upper().startswith("GOOGLE/"):
                                            fb_name = fb_name[7:]
                                            fb_is_or = False
                                        elif fb_name.upper().startswith("OPENROUTER/"):
                                            fb_name = fb_name[11:]
                                            fb_is_or = True
                                        elif "/" in fb_name:
                                            fb_is_or = True
                                        
                                        if fb_is_or:
                                            or_key = self._get_api_key_for_guild(channel.guild.id, provider="openrouter")
                                            if or_key:
                                                fallback_instance = OpenRouterModel(fb_name, api_key=or_key, system_instruction=full_system_instruction)
                                            else:
                                                raise ValueError("No OpenRouter key for fallback")
                                        else:
                                            fallback_instance = genai.GenerativeModel(fb_name, system_instruction=full_system_instruction, safety_settings=dynamic_safety_settings)
                                        
                                        # Pass list directly
                                        response = await fallback_instance.generate_content_async(contents_for_api_call, generation_config=gen_config)
                                        if not response or not response.candidates:
                                            # Fallback also blocked/failed, do not change status to success
                                            pass
                                        else:
                                            # Fallback succeeded
                                            fallback_used = True
                                            self._log_api_call(user_id=triggering_user_id, guild_id=channel.guild.id, context="multi_profile_fallback", model_used=fb_name, status="success")
                                    except Exception as retry_e:
                                        print(f"Fallback retry failed: {retry_e}")
                                        status = "api_error"
                                else:
                                    print(f"Critical API Error in Multi-Profile: {e}")
                                    status = "api_error"
                            except (api_exceptions.PermissionDenied) as e:
                                print(f"Permission Denied: {e}")
                                status = "api_error"
                            finally:
                                # Always log the primary model's final status (success or api_error)
                                self._log_api_call(user_id=triggering_user_id, guild_id=channel.guild.id, context="multi_profile", model_used=model.model_name if hasattr(model, 'model_name') else "unknown", status=status)
                        else:
                            # [NEW] Handle model initialization failure by simulating a blocked response
                            blocked_reason_override = warning_message or "Internal API Initialization Error"

                        was_blocked = False
                        if not response or not response.candidates:
                            reason = blocked_reason_override or "Unknown"
                            if response and response.prompt_feedback and response.prompt_feedback.block_reason: 
                                reason = response.prompt_feedback.block_reason.name.replace('_', ' ').title()
                            
                            if reason == "Rate Limit":
                                response_text = f"My response was blocked due to: **API Rate Limit**. Please try again later or use paid tier API."
                            else:
                                response_text = f"My response was blocked due to: **{reason}**. Please rephrase or try a different topic."
                            was_blocked = True
                        else:
                            try:
                                # Use the filtered text attribute from the model wrapper to exclude thoughts
                                raw_text = getattr(response, 'text', "").strip()
                                
                                all_participant_names = []
                                for p_data in session.get("profiles", []):
                                    owner_id = p_data['owner_id']
                                    p_name = p_data['profile_name']
                                    
                                    p_user_data = self._get_user_data_entry(owner_id)
                                    p_is_borrowed = p_name in p_user_data.get("borrowed_profiles", {})
                                    p_effective_owner_id = owner_id
                                    p_effective_profile_name = p_name
                                    if p_is_borrowed:
                                        borrowed_data = p_user_data["borrowed_profiles"][p_name]
                                        p_effective_owner_id = int(borrowed_data["original_owner_id"])
                                        p_effective_profile_name = borrowed_data["original_profile_name"]
                                    
                                    display_name = p_effective_profile_name
                                    appearance_data = self.user_appearances.get(str(p_effective_owner_id), {}).get(p_effective_profile_name, {})
                                    if appearance_data.get("custom_display_name"):
                                        display_name = appearance_data["custom_display_name"]
                                    all_participant_names.append(display_name)

                                scrubbed_text = self._scrub_response_text(raw_text, participant_names=all_participant_names)
                                response_text = self._deduplicate_response(scrubbed_text)
                                if not response_text:
                                    response_text = "An API error has occurred. Please try again."
                                    was_blocked = True # Treat as a block to prevent LTM creation
                            except ValueError:
                                reason = response.candidates[0].finish_reason.name
                                response_text = f"My response was blocked due to: **{reason.replace('_', ' ').title()}**. Please rephrase or try a different topic."
                                was_blocked = True

                        t2_end_mono = time.monotonic()
                        duration = t2_end_mono - t1_start_mono
                        sent_timestamp = datetime.datetime.now(datetime.timezone.utc) # Approximation

                        timezone_str = profile_settings.get("timezone", "UTC")
                        main_history_line = self._format_history_entry(speaker_display_name, sent_timestamp, response_text, timezone_str)
                        try:
                            t1_formatted = t1_start_utc.astimezone(ZoneInfo(timezone_str)).strftime('%I:%M:%S %p %Z')
                        except Exception:
                            t1_formatted = t1_start_utc.strftime('%I:%M:%S %p UTC')
                        metadata_line = f"(Thought Initiated: {t1_formatted} | Duration: {duration:.2f}s)"
                        history_line = f"{main_history_line.strip()}\n{metadata_line}\n"

                        model_content_obj = content_types.to_content({'role': 'model', 'parts': [history_line]})
                        user_content_obj = content_types.to_content({'role': 'user', 'parts': [history_line]})

                        if owner_id in self.debug_users:
                            try:
                                user_to_dm = self.bot.get_user(owner_id)
                                if user_to_dm:
                                    turns_for_debug = []
                                    if grounding_context and participant_key == grounding_profile_key:
                                        turns_for_debug.append(content_types.to_content({'role': 'user', 'parts': [grounding_context, "\n"]}))
                                    if ltm_recall_text:
                                        turns_for_debug.append(content_types.to_content({'role': 'user', 'parts': [ltm_recall_text, "\n"]}))
                                    
                                    turns_for_debug.append(model_content_obj)

                                    debug_message = self._format_debug_prompt(turns_for_debug)
                                    await user_to_dm.send(debug_message)
                            except Exception as e:
                                print(f"Failed to send debug DM to user {owner_id}: {e}")

                    except Exception as e:
                        print(f"Multi-profile generation error for '{profile_name}': {e}")
                        traceback.print_exc()
                        response_text = "An error has occurred. The response may have been blocked by the safety filter."

                    # Resolve the chat session for this specific participant to handle persistence
                    participant_key = (participant['owner_id'], participant['profile_name'])
                    chat_session = session['chat_sessions'].get(participant_key)

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

                    if fallback_used and p_settings.get("show_fallback_indicator", True):
                        display_text += "\n\n-# Fallback Model Used"

                    sources_text = None
                    if participant_key == grounding_profile_key and grounding_mode_for_citator == "on+" and grounding_sources:
                        source_links = []
                        for i, source in enumerate(grounding_sources):
                            domain = source.get('title')
                            if not domain:
                                try:
                                    domain = urlparse(source['uri']).netloc
                                    if domain.startswith('www.'):
                                        domain = domain[4:]
                                except Exception:
                                    domain = "source"
                            domain = re.sub(r'\s+', ' ', domain).strip()
                            source_links.append(f"{i+1}. [{domain}](<{source['uri']}>)")
                        
                        links_per_line = 5
                        chunked_links = [source_links[i:i + links_per_line] for i in range(0, len(source_links), links_per_line)]
                        
                        formatted_lines = []
                        if chunked_links:
                            # Format the first line with the "Sources:" prefix
                            formatted_lines.append(f"> -# Sources:  {'  '.join(chunked_links[0])}")
                            # Format any subsequent lines
                            for chunk in chunked_links[1:]:
                                formatted_lines.append(f"> -# {'  '.join(chunk)}")
                        
                        sources_text = "\n".join(formatted_lines)
                    
                    t2_end_mono = time.monotonic()
                    duration = t2_end_mono - t1_start_mono
                    sent_timestamp = datetime.datetime.now(datetime.timezone.utc) 

                    timezone_str = profile_settings.get("timezone", "UTC")
                    main_history_line = self._format_history_entry(speaker_display_name, sent_timestamp, response_text, timezone_str)
                    try:
                        t1_formatted = t1_start_utc.astimezone(ZoneInfo(timezone_str)).strftime('%I:%M:%S %p %Z')
                    except Exception:
                        t1_formatted = t1_start_utc.strftime('%I:%M:%S %p UTC')
                    metadata_line = f"(Thought Initiated: {t1_formatted} | Duration: {duration:.2f}s)"
                    history_line = f"{main_history_line.strip()}\n{metadata_line}\n"

                    turn_id = str(uuid.uuid4())
                    turn_object = {
                        "turn_id": turn_id,
                        "speaker_key": [owner_id, profile_name],
                        "content": history_line,
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                    }
                    session.get("unified_log", []).append(turn_object)
                    session['last_speaker_key'] = participant_key

                    model_content_obj = content_types.to_content({'role': 'model', 'parts': [history_line]})
                    user_content_obj = content_types.to_content({'role': 'user', 'parts': [history_line]})

                    chat_session.history.append(model_content_obj)

                    is_realistic_typing = profile_settings.get("realistic_typing_enabled", False)

                    # [NEW] Unified Synthesis Logic for all Audio Modes
                    audio_mode = session.get("audio_mode", "text-only")
                    audio_file_for_send = None
                    
                    if audio_mode in ["audio+text", "audio-only", "multi-audio"]:
                        # 1. Build Contextual Round Transcript
                        round_transcript = ""
                        for idx, prev_resp in enumerate(responses_this_round[:-1]):
                            prev_p = profile_order[idx]
                            prev_app = self.user_appearances.get(str(prev_p['owner_id']), {}).get(prev_p['profile_name'], {})
                            prev_name = prev_app.get("custom_display_name") or prev_p['profile_name']
                            round_transcript += f"{prev_name}: {prev_resp}\n\n"

                        # 2. Resolve Profile Speech Settings and Director's Desk
                        s_voice = p_settings.get("speech_voice", "Aoede")
                        s_model = p_settings.get("speech_model", "gemini-2.5-flash-preview-tts")
                        s_temp = float(p_settings.get("speech_temperature", 1.0))
                        
                        s_arch = p_settings.get("speech_archetype", "")
                        s_acc = p_settings.get("speech_accent", "")
                        s_dyn = p_settings.get("speech_dynamics", "")
                        s_styl = p_settings.get("speech_style", "")
                        s_pace = p_settings.get("speech_pacing", "")

                        # 3. Construct Conditional Markdown Prompt
                        prompt_parts = []
                        
                        # Section: Audio Profile
                        if s_arch or s_acc:
                            part = f"# AUDIO PROFILE: {speaker_display_name}\n"
                            if s_arch: part += f"Archetype: {s_arch}\n"
                            if s_acc: part += f"Accent: {s_acc}\n"
                            prompt_parts.append(part.strip())

                        # Section: The Scene
                        if s_dyn:
                            part = f"## THE SCENE\nDynamics: {s_dyn}\nAction: Fluid conversation."
                            prompt_parts.append(part.strip())

                        # Section: Director's Notes
                        if s_styl or s_pace:
                            part = "### DIRECTOR'S NOTES\n"
                            if s_styl: part += f"Style: {s_styl}\n"
                            if s_pace: part += f"Pacing: {s_pace}\n"
                            prompt_parts.append(part.strip())

                        # Section: Context & Transcript
                        if round_transcript:
                            prompt_parts.append(f"#### SAMPLE CONTEXT\nPrevious turn flow:\n{round_transcript.strip()}")
                        
                        prompt_parts.append(f"#### TRANSCRIPT\n{speaker_display_name}: {response_text}")

                        tts_priming_prompt = "\n\n".join(prompt_parts)
                        
                        # 4. Synthesise Audio
                        turn_audio_stream = await self._generate_google_tts(
                            tts_priming_prompt, 
                            channel.guild.id, 
                            model_id=s_model, 
                            voice_name=s_voice, 
                            temperature=s_temp
                        )
                        
                        if turn_audio_stream:
                            if audio_mode == "multi-audio":
                                # Store for round-end stitching
                                round_audio_segments.append(turn_audio_stream)
                            else:
                                # Prepare for immediate delivery with this turn
                                audio_file_for_send = discord.File(turn_audio_stream, filename=f"voice_{turn_id[:4]}.wav")
                                if audio_mode == "audio-only":
                                    # Use zero-width space to hide text while keeping message valid
                                    display_text = ""

                    file_to_send = audio_file_for_send if audio_file_for_send else None

                    if participant.get('method') == 'child_bot':
                        correlation_id = str(uuid.uuid4())
                        confirmation_event = asyncio.Event()
                        self.pending_child_confirmations[correlation_id] = {
                            "event": confirmation_event, "type": "multi_profile", "participant": participant,
                            "history_line": history_line, "channel_id": channel.id, "turn_id": turn_id
                        }
                        
                        rmode = profile_settings.get("response_mode", "regular")
                        reply_id = None
                        should_ping = False
                        
                        if i == 0:
                            if anchor_message and rmode == "mention":
                                display_text = f"{anchor_message.author.mention} {display_text}"
                            reply_id = anchor_message.id if (anchor_message and rmode in ["reply", "mention_reply"]) else None
                            should_ping = (rmode == "mention_reply")

                        payload = {
                            "action": "send_message", "channel_id": channel.id, "content": display_text,
                            "realistic_typing": is_realistic_typing, "correlation_id": correlation_id,
                            "reply_to_id": reply_id, "ping": should_ping
                        }
                        
                        if file_to_send:
                            # [FIXED] Corrected variable name to turn_audio_stream
                            attachment_data = None
                            if is_generator and generated_image_bytes_for_round:
                                attachment_data = {
                                    "filename": "generated_image.png",
                                    "data_base64": base64.b64encode(generated_image_bytes_for_round).decode('utf-8')
                                }
                            elif audio_file_for_send:
                                # Ensure the stream is at the start and read the bytes
                                turn_audio_stream.seek(0)
                                attachment_data = {
                                    "filename": f"voice_{turn_id[:4]}.wav",
                                    "data_base64": base64.b64encode(turn_audio_stream.read()).decode('utf-8')
                                }
                            
                            if attachment_data:
                                payload["attachment"] = attachment_data

                        await self.manager_queue.put({"action": "send_to_child", "bot_id": participant['bot_id'], "payload": payload})
                        del payload

                        try: await asyncio.wait_for(confirmation_event.wait(), timeout=45.0)
                        except asyncio.TimeoutError: self.pending_child_confirmations.pop(correlation_id, None)

                        # Send thought summary as a separate file attachment
                        if thought_file_to_send:
                            t_corr_id = str(uuid.uuid4())
                            t_conf_event = asyncio.Event()
                            self.pending_child_confirmations[t_corr_id] = {
                                "event": t_conf_event, "type": "multi_profile", "participant": participant,
                                "history_line": history_line, "channel_id": channel.id, "turn_id": turn_id
                            }
                            thought_text_bytes = thought_text.encode('utf-8')
                            thought_base64 = base64.b64encode(thought_text_bytes).decode('utf-8')
                            
                            await self.manager_queue.put({
                                "action": "send_to_child", "bot_id": participant['bot_id'],
                                "payload": {
                                    "action": "send_message", "channel_id": channel.id, 
                                    "content": "", # [UPDATED] Removed text label
                                    "realistic_typing": False, "correlation_id": t_corr_id,
                                    "attachment": {
                                        "filename": "thinking_summary.txt",
                                        "data_base64": thought_base64
                                    }
                                }
                            })
                            try: await asyncio.wait_for(t_conf_event.wait(), timeout=45.0)
                            except asyncio.TimeoutError: self.pending_child_confirmations.pop(t_corr_id, None)

                    else: # Webhook logic
                        sent_messages = await self._send_channel_message(
                            channel, display_text, target_message_to_edit=placeholder_message,
                            profile_owner_id_for_appearance=owner_id, profile_name_for_appearance=profile_name,
                            file=file_to_send, reply_to=(anchor_message if i == 0 else None)
                        )
                        
                        # Send thought summary as a separate file attachment
                        if thought_file_to_send:
                            t_msgs = await self._send_channel_message(
                                channel, 
                                "", # [UPDATED] Removed text label
                                file=thought_file_to_send,
                                profile_owner_id_for_appearance=owner_id, profile_name_for_appearance=profile_name,
                                bypass_typing=True
                            )
                            if t_msgs: sent_messages.extend(t_msgs)

                        if sent_messages:
                            session['last_bot_message_id'] = sent_messages[-1].id
                            session_type = session.get("type", "multi")
                            turn_data = (channel_id, session_type, turn_id)
                            
                            mapping_key = (session_type, channel_id)
                            if mapping_key not in self.mapping_caches:
                                self.mapping_caches[mapping_key] = self._load_mapping_from_disk(mapping_key)

                            for msg in sent_messages:
                                self.message_to_history_turn[msg.id] = turn_data
                                self.mapping_caches[mapping_key][str(msg.id)] = turn_data
                    
                    if sources_text:
                        for line in sources_text.split('\n'):
                            if not line.strip(): continue
                            if participant.get('method') == 'child_bot':
                                source_payload = {
                                    "action": "send_message", "channel_id": channel.id,
                                    "content": line, "realistic_typing": False, # Instant for child
                                    "reply_to_id": None, "ping": False
                                }
                                await self.manager_queue.put({"action": "send_to_child", "bot_id": participant['bot_id'], "payload": source_payload})
                            else: # Webhook
                                await self._send_channel_message(
                                    channel, line,
                                    profile_owner_id_for_appearance=owner_id, profile_name_for_appearance=profile_name,
                                    reply_to=None,
                                    bypass_typing=True # Skip delay
                                )
                    
                    for key, other_chat_session in session['chat_sessions'].items():
                        if key != participant_key:
                            other_chat_session.history.append(user_content_obj)
                    
                    # [NEW] Explicitly clear media parts from local scope to allow GC to claim them
                    if 'contents_for_api_call' in locals():
                        del contents_for_api_call
                    if 'response' in locals():
                        del response
                    
                    is_last_participant = (participant == profile_order[-1])
                    if not is_last_participant:
                        await asyncio.sleep(1.0)
                        
                        batched_triggers = []
                        while not session['task_queue'].empty():
                            try: batched_triggers.append(session['task_queue'].get_nowait())
                            except asyncio.QueueEmpty: break
                        
                        if batched_triggers:
                            for trigger in batched_triggers:
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
                                    content = f"{reply_context} {trigger.clean_content}" if reply_context else trigger.clean_content
                                    
                                    # [NEW] Batch URL Context Logic
                                    any_url_enabled_batch = False
                                    for p in session['profiles']:
                                        p_udata = self._get_user_data_entry(p['owner_id'])
                                        p_is_b = p['profile_name'] in p_udata.get("borrowed_profiles", {})
                                        p_settings = p_udata.get("borrowed_profiles" if p_is_b else "profiles", {}).get(p['profile_name'], {})
                                        if p_settings.get("url_fetching_enabled", True):
                                            any_url_enabled_batch = True; break
                                    
                                    url_text_batch = None
                                    url_media_batch = []
                                    
                                    if any_url_enabled_batch:
                                        url_text_list, url_media = await self._process_urls_in_content(content, trigger.guild.id, {"url_fetching_enabled": True})
                                        if url_text_list: url_text_batch = "\n".join(url_text_list)
                                        url_media_batch = url_media

                                    # [NEW] Localized User Timestamp Logic (Batch)
                                    u_data_batch = self._get_user_data_entry(trigger.author.id)
                                    u_prof_batch = self._get_active_user_profile_name_for_channel(trigger.author.id, channel_id)
                                    u_is_b_batch = u_prof_batch in u_data_batch.get("borrowed_profiles", {})
                                    u_sett_batch = u_data_batch.get("borrowed_profiles" if u_is_b_batch else "profiles", {}).get(u_prof_batch, {})
                                    batch_tz = u_sett_batch.get("timezone", "UTC")

                                    user_line = self._format_history_entry(author_name, trigger.created_at, content, batch_tz)
                                    
                                    batch_msg_media = []
                                    message_attachments = [a for a in trigger.attachments if a.content_type and (a.content_type.startswith("image/") or a.content_type.startswith("audio/") or a.content_type.startswith("video/"))]
                                    for attachment in message_attachments:
                                        try:
                                            media_data = await attachment.read()
                                            batch_msg_media.append({"mime_type": attachment.content_type, "data": media_data})
                                        except Exception as e:
                                            print(f"Failed to read batched media attachment {attachment.filename}: {e}")

                                    # [NEW] Conditional Injection into Chat Sessions
                                    for p_key, inner_chat_session in session['chat_sessions'].items():
                                        p_owner_id, p_name = p_key
                                        p_udata = self._get_user_data_entry(p_owner_id)
                                        p_is_b = p_name in p_udata.get("borrowed_profiles", {})
                                        p_settings = p_udata.get("borrowed_profiles" if p_is_b else "profiles", {}).get(p_name, {})
                                        
                                        final_parts = [user_line]
                                        if url_text_batch and p_settings.get("url_fetching_enabled", True):
                                            final_parts.append(f"\n<document_context>\n{url_text_batch}\n</document_context>")
                                        
                                        final_parts.extend(url_media_batch)
                                        final_parts.extend(batch_msg_media)
                                        
                                        new_content_obj = content_types.to_content({'role': 'user', 'parts': final_parts})
                                        inner_chat_session.history.append(new_content_obj)

                                    if trigger.author.id in self.debug_users:
                                        try:
                                            user_to_dm = self.bot.get_user(trigger.author.id)
                                            if user_to_dm:
                                                # Create temporary debug object with all context
                                                debug_parts = [user_line]
                                                if url_text_batch: debug_parts.append(url_text_batch)
                                                debug_parts.extend(url_media_batch)
                                                debug_parts.extend(batch_msg_media)
                                                debug_obj = content_types.to_content({'role': 'user', 'parts': debug_parts})
                                                
                                                debug_message = self._format_debug_prompt([debug_obj])
                                                await user_to_dm.send(debug_message)
                                        except Exception as e:
                                            print(f"Failed to send batched user turn debug DM to user {trigger.author.id}: {e}")
                                    
                                    new_turn_id = str(uuid.uuid4())
                                    new_turn_object = {
                                        "turn_id": new_turn_id,
                                        "speaker_key": [trigger.author.id, "user"],
                                        "content": user_line,
                                        "timestamp": trigger.created_at.isoformat()
                                    }
                                    if url_text_batch:
                                        new_turn_object["url_context"] = url_text_batch
                                    session.get("unified_log", []).append(new_turn_object)

                                    session_type = session.get("type", "multi")
                                    turn_data = (channel_id, session_type, new_turn_id)
                                    self.message_to_history_turn[trigger.id] = turn_data
                                    
                                    mapping_key = (session_type, channel_id)
                                    if mapping_key not in self.mapping_caches:
                                        self.mapping_caches[mapping_key] = self._load_mapping_from_disk(mapping_key)
                                    self.mapping_caches[mapping_key][str(trigger.id)] = turn_data
                            all_triggers_for_round.extend(batched_triggers)

                    # --- Explicit Memory Cleanup ---
                    if generated_image_bytes_for_round:
                        del generated_image_bytes_for_round
                        generated_image_bytes_for_round = None
                    
                    # Force garbage collection to clear image buffers from this turn
                    gc.collect()

                for trigger in all_triggers_for_round:
                    if trigger is not None:
                        session['task_queue'].task_done()

                # [NEW] Multi-Audio Final Delivery
                if session.get("audio_mode") == "multi-audio" and round_audio_segments:
                    placeholders = await self._send_channel_message(channel, f"{PLACEHOLDER_EMOJI}")
                    master_placeholder = placeholders[0] if placeholders else None

                    master_stream = self._stitch_wav_segments(round_audio_segments)
                    
                    if master_stream.getbuffer().nbytes > 0:
                        master_file = discord.File(master_stream, filename="round_master.wav")
                        await self._send_channel_message(
                            channel, 
                            "-# **Round Audio Summary**", 
                            target_message_to_edit=master_placeholder,
                            file=master_file,
                            bypass_typing=True
                        )
                    elif master_placeholder:
                        await master_placeholder.delete()

                # [NEW] Aggressive Round-End Memory Cleanup
                if 'new_round_turn_data' in locals():
                    # Deep clear the binary data inside the turn data tuples
                    for i in range(len(new_round_turn_data)):
                        base, url, media = new_round_turn_data[i]
                        media.clear() # Empty the lists of bytes
                    del new_round_turn_data
                
                if 'url_media_parts' in locals():
                    url_media_parts.clear()
                    del url_media_parts

                # Force a collection of generation 0 and 1 objects immediately
                gc.collect()

                guild_id = self.bot.get_channel(channel_id).guild.id
                
                if not was_blocked:
                    # Check each participant that ACTUALLY SPOKE this round
                    for participant in profile_order:
                        owner_id = participant['owner_id']
                        profile_name = participant['profile_name']
                        p_user_data = self._get_user_data_entry(owner_id)
                        p_settings = p_user_data.get("profiles", {}).get(profile_name, {})
                        
                        if not p_settings.get("ltm_creation_enabled", False): continue
                        
                        # Increment individual counter
                        participant['ltm_counter'] = participant.get('ltm_counter', 0) + 1
                        
                        interval = p_settings.get("ltm_creation_interval", 10)
                        context_size = p_settings.get("ltm_summarization_context", 10)
              
                        if participant['ltm_counter'] >= interval:
                            history_source_obj = next(iter(session['chat_sessions'].values()), None)
                            if history_source_obj and len(history_source_obj.history) >= 4:
                                # Context size * 2 because turn history includes both user and bot
                                events_for_summary = [turn.parts[0].text for turn in history_source_obj.history[-(context_size * 2):] if turn.parts and hasattr(turn.parts[0], 'text')]
                                
                                _, _, _, temp, top_p, top_k, primary_model, _ = self._construct_system_instructions(owner_id, profile_name, channel_id, is_multi_profile=True)
                                ltm_d = await self._generate_ltm_data_from_history(events_for_summary, round_author_name, {"temperature": temp, "top_p": top_p, "top_k": top_k}, primary_model, guild_id, profile_owner_id=owner_id, profile_name=profile_name)
                                if ltm_d:
                                    summary_embedding = await self._get_embedding(ltm_d, guild_id, task_type="RETRIEVAL_DOCUMENT")
                                    if summary_embedding:
                                        quantized_embedding = _quantize_embedding(summary_embedding)
                                        self._add_ltm(owner_id, profile_name, ltm_d, quantized_embedding, guild_id, triggering_user_id, round_author_name)
                            
                            participant['ltm_counter'] = 0

                # AGGRESSIVE GC: Clear references
                if 'new_round_content_objects' in locals(): del new_round_content_objects
                if 'all_triggers_for_round' in locals(): del all_triggers_for_round

                # Trim the unified log to be the single source of truth for the session's history window.
                if len(session.get("unified_log", [])) > self.max_history_items * 2:
                    session["unified_log"] = session["unified_log"][-(self.max_history_items * 2):]

                # Rebuild all participant histories from the trimmed unified log to ensure consistency.
                from google.generativeai.types import content_types
                dummy_model = genai.GenerativeModel('gemini-flash-latest')
                trimmed_unified_log = session.get("unified_log", [])

                for p_data in session["profiles"]:
                    p_key = (p_data['owner_id'], p_data['profile_name'])
                    
                    p_user_data = self._get_user_data_entry(p_data['owner_id'])
                    p_is_borrowed = p_data['profile_name'] in p_user_data.get("borrowed_profiles", {})
                    p_profile_settings = p_user_data.get("borrowed_profiles" if p_is_borrowed else "profiles", {}).get(p_data['profile_name'], {})
                    stm_length = int(p_profile_settings.get("stm_length", defaultConfig.CHATBOT_MEMORY_LENGTH))

                    history_slice = trimmed_unified_log[-(stm_length * 2):] if stm_length > 0 else []
                    
                    participant_history = []
                    for turn in history_slice:
                        turn_type = turn.get("type")
                        if not turn_type:
                            speaker_key = tuple(turn.get("speaker_key", []))
                            role = 'model' if speaker_key == p_key else 'user'
                            
                            # [UPDATED] Standardized Headers
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

                # Handle Proactive Freewill Continuation
                if session.get("type") == "freewill" and session.get("freewill_mode") == "proactive":
                    rounds_left = session.get("proactive_initial_rounds", 0)
                    if rounds_left > 0:
                        session["proactive_initial_rounds"] -= 1
                        # Schedule the next turn
                        # We use a short delay to simulate natural pause before next "automatic" event
                        await asyncio.sleep(2) 
                        await session['task_queue'].put(None) # None trigger = "Continue conversation"

            except asyncio.CancelledError:
                print(f"Worker for channel {channel_id} cancelled.")
                break
            except Exception as e:
                print(f"Error in multi-profile worker for channel {channel_id}: {e}")
                traceback.print_exc()
        
        session['is_running'] = False

    async def _image_finisher_worker(self):
        """Consumes generated images, generates text, and sends the final message."""
        while True:
            try:
                # [FIXED] Unpack Priority Tuple
                item = await self.text_request_queue.get()
                if item is None: break
                
                priority, package = item

                async with self.image_gen_semaphore:
                    placeholder_message = package.get("placeholder_message")
                    final_response_text = "An error occurred."
                    image_file_to_send = None
                    
                    is_child_bot = package.get("is_child_bot", False)
                    channel = self.bot.get_channel(package['channel_id'])
                    if not channel: self.text_request_queue.task_done(); continue

                    # If this was a queued request, it won't have a placeholder yet. Create one now.
                    if not placeholder_message and not is_child_bot:
                        placeholders = await self._send_channel_message(
                            channel, f"{PLACEHOLDER_EMOJI}",
                            profile_owner_id_for_appearance=package['effective_profile_owner_id'],
                            profile_name_for_appearance=package['effective_profile_name']
                        )
                        placeholder_message = placeholders[0] if placeholders else None
                    elif is_child_bot and not package.get("reference_image_url"):
                         # Typing was already started for non-ref images. For ref-images, start it now.
                         await self.manager_queue.put({"action": "send_to_child", "bot_id": package['bot_id'], "payload": {"action": "start_typing", "channel_id": channel.id}})
                    
                    try:
                        # --- Just-in-Time Generation for Reference Images ---
                        if package.get("reference_image_urls"):
                            image_bytes, failure_reason = None, None
                            try:
                                api_key = self._get_api_key_for_guild(package['guild_id']);
                                if not api_key: raise ValueError("Server API key not configured.")
                                genai.configure(api_key=api_key)
                                image_model = genai.GenerativeModel('gemini-2.5-flash-image', system_instruction=package['system_instruction'], safety_settings=package['safety_settings'])
                                parts = [package['prompt_text']]
                                async with httpx.AsyncClient() as client:
                                    for url in package.get("reference_image_urls", []):
                                        response = await client.get(url); response.raise_for_status()
                                        ref_image_data = await response.aread()
                                        ctype = response.headers.get("Content-Type", "image/png")
                                        
                                        # FIXED: Direct pass-through of raw bytes.
                                        parts.append({"mime_type": ctype, "data": ref_image_data})

                                status = "api_error"
                                try:
                                    response = await image_model.generate_content_async(parts)
                                    status = "blocked_by_safety" if not response.candidates else "success"
                                finally:
                                    self._log_api_call(user_id=package['author_id'], guild_id=package['guild_id'], context="image_generation_jit", model_used=image_model.model_name, status=status)
                                
                                # No PIL cleanup needed
                                del parts

                                if not response.candidates:
                                    reason = "Safety Filter";
                                    if response.prompt_feedback and response.prompt_feedback.block_reason: reason = response.prompt_feedback.block_reason.name.replace('_', ' ').title()
                                    failure_reason = f"the safety filter ({reason})"
                                else:
                                    candidate = response.candidates[0]
                                    if candidate.finish_reason.name != 'STOP': failure_reason = f"the process being stopped for reason: **{candidate.finish_reason.name.replace('_', ' ').title()}**"
                                    else:
                                        image_bytes = next((part.inline_data.data for part in candidate.content.parts if hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/')), None)
                                        if not image_bytes: failure_reason = "an unknown issue (the model returned no image data)"
                            except Exception as e: failure_reason = f"an unexpected error: `{e}`"
                            package['generated_image_bytes'] = image_bytes
                            package['failure_reason'] = failure_reason
                        
                        # --- Text Generation ---
                        text_model, _, temp, top_p, top_k, _, _ = await self._get_or_create_model_for_channel(package['channel_id'], package['author_id'], package['guild_id'], profile_owner_override=package['effective_profile_owner_id'], profile_name_override=package['effective_profile_name'])
                        
                        from google.generativeai.types import content_types
                        model_cache_key = (package['channel_id'], package['effective_profile_owner_id'], package['effective_profile_name'])
                        chat = self.chat_sessions.get(model_cache_key);
                        if not chat: chat = text_model.start_chat(history=[])
                        self.chat_sessions[model_cache_key] = chat; contents_for_api_call = list(chat.history)
                        
                        turn_id = str(uuid.uuid4())

                        if package['generated_image_bytes'] and not package['failure_reason']:
                            system_note = f"<image_context>You have just generated the following image based on the prompt: '{package['prompt_text']}'. Present it.</image_context>"
                            
                            final_user_parts = [
                                system_note, 
                                {"mime_type": "image/jpeg", "data": package['generated_image_bytes']}
                            ]
                            
                            user_turn = content_types.to_content({'role': 'user', 'parts': final_user_parts})
                        
                        user_turn = content_types.to_content({'role': 'user', 'parts': final_user_parts})
                        contents_for_api_call.append(user_turn)
                        gen_config = genai.types.GenerationConfig(temperature=temp, top_p=top_p, top_k=top_k)
                        
                        text_response = await text_model.generate_content_async(contents_for_api_call, generation_config=gen_config)
                        
                        response_text = "Here is the image you requested."
                        if text_response.candidates and text_response.candidates[0].finish_reason.name == 'STOP':
                            raw_text = "".join(p.text for p in text_response.candidates[0].content.parts if hasattr(p, 'text'))
                            response_text = self._deduplicate_response(self._scrub_response_text(raw_text.strip(), participant_names=[package['bot_display_name']]))
                        model_turn = content_types.to_content({'role': 'model', 'parts': [response_text]}); chat.history.extend([user_turn, model_turn])

                        # --- Final Message Sending ---
                        if package['generated_image_bytes'] and not package['failure_reason']:
                            image_file_to_send = discord.File(io.BytesIO(package['generated_image_bytes']), filename="generated_image.png")
                        
                        final_response_text = response_text
                        owner_id = package['effective_profile_owner_id']
                        profile_name = package['effective_profile_name']
                        user_data = self._get_user_data_entry(owner_id)
                        is_borrowed = profile_name in user_data.get("borrowed_profiles", {})
                        profile_settings = user_data.get("borrowed_profiles" if is_borrowed else "profiles", {}).get(profile_name, {})
                        is_realistic_typing = profile_settings.get("realistic_typing_enabled", False)

                    except Exception as e:
                        final_response_text = f"An unexpected error occurred in the finalization stage: {e}"; print(f"Error in finisher stage: {e}"); traceback.print_exc()

                    if is_child_bot:
                        correlation_id = str(uuid.uuid4())
                        self.pending_child_confirmations[correlation_id] = {
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
                            "reply_to_id": reply_id, "ping": should_ping
                        }
                        if image_file_to_send:
                            import base64
                            payload["attachment"] = { "filename": "generated_image.png", "data_base64": base64.b64encode(package['generated_image_bytes']).decode('utf-8') }
                        await self.manager_queue.put({"action": "send_to_child", "bot_id": package['bot_id'], "payload": payload})
                    else:
                        # [UPDATED] Fix undefined 'i' by resolving anchor_message from package
                        anchor_msg = None
                        anchor_id = package.get("original_message_id")
                        if anchor_id:
                            try: anchor_msg = await channel.fetch_message(anchor_id)
                            except: pass

                        sent_messages = await self._send_channel_message(
                            channel, final_response_text, target_message_to_edit=placeholder_message, 
                            profile_owner_id_for_appearance=package['effective_profile_owner_id'], 
                            profile_name_for_appearance=package['effective_profile_name'], 
                            file=image_file_to_send, reply_to=anchor_msg
                        )
                        
                        turn_data = (model_cache_key, 'single', turn_id)
                        mapping_key = self._get_mapping_key_for_session(model_cache_key, 'single')
                        if mapping_key not in self.mapping_caches: self.mapping_caches[mapping_key] = self._load_mapping_from_disk(mapping_key)
                        
                        # Map the original user trigger message
                        original_message_id = package.get("original_message_id")
                        if original_message_id:
                            self.message_to_history_turn[original_message_id] = turn_data
                            self.mapping_caches[mapping_key][str(original_message_id)] = turn_data

                        # Map the bot's response messages
                        if sent_messages:
                            for msg in sent_messages: 
                                self.message_to_history_turn[msg.id] = turn_data
                                self.mapping_caches[mapping_key][str(msg.id)] = turn_data
                    
                    grounding_sources = package.get("grounding_sources")
                    grounding_mode = package.get("grounding_mode")
                    if grounding_mode == "on+" and grounding_sources:
                        source_links = []
                        for i, source in enumerate(grounding_sources):
                            domain = source.get('title')
                            if not domain:
                                try:
                                    domain = urlparse(source['uri']).netloc
                                    if domain.startswith('www.'):
                                        domain = domain[4:]
                                except Exception:
                                    domain = "source"
                            domain = re.sub(r'\s+', ' ', domain).strip()
                            source_links.append(f"{i+1}. [{domain}](<{source['uri']}>)")
                        
                        links_per_line = 5
                        chunked_links = [source_links[i:i + links_per_line] for i in range(0, len(source_links), links_per_line)]
                        
                        for i, chunk in enumerate(chunked_links):
                            if i == 0:
                                sources_text = f"> -# Sources:  {'  '.join(chunk)}"
                            else:
                                sources_text = f"> -# {'  '.join(chunk)}"
                            
                            if is_child_bot:
                                # [UPDATED] Standard source payload for images (no Response Mode)
                                source_payload = {
                                    "action": "send_message", "channel_id": channel.id,
                                    "content": sources_text, "realistic_typing": False,
                                    "reply_to_id": None, "ping": False
                                }
                                await self.manager_queue.put({"action": "send_to_child", "bot_id": package['bot_id'], "payload": source_payload})
                            else:
                                # [UPDATED] Standard source message for images (no Response Mode)
                                await self._send_channel_message(
                                    channel, sources_text,
                                    profile_owner_id_for_appearance=package['effective_profile_owner_id'],
                                    profile_name_for_appearance=package['effective_profile_name'],
                                    reply_to=None
                                )

                # --- Aggressive Memory Cleanup ---
                if image_file_to_send:
                    image_file_to_send.close()
                    del image_file_to_send
                
                # Clear references to large byte objects in the dictionary
                if 'generated_image_bytes' in package:
                    package['generated_image_bytes'] = None
                
                # Delete the dictionary itself
                del package
                
                # Force garbage collection
                gc.collect()

                self.text_request_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in image finisher worker: {e}"); traceback.print_exc()
                # Ensure typing is stopped on error for child bots
                if package.get("is_child_bot"):
                    try:
                        await self.manager_queue.put({
                            "action": "send_to_child", "bot_id": package['bot_id'],
                            "payload": {"action": "stop_typing", "channel_id": package['channel_id']}
                        })
                    except Exception as e_stop:
                        print(f"Failed to send stop_typing on error: {e_stop}")

    async def _image_gen_worker(self, worker_id: int):
        """Pre-fetches image generation for text-only prompts."""
        while True:
            try:
                # [FIXED] Unpack Priority Tuple
                item = await self.image_request_queue.get()
                if item is None: break 
                
                priority, request_data = item

                # If a reference image is present, this request bypasses pre-fetching.
                if request_data.get("reference_image_urls"):
                    await self.text_request_queue.put((priority, request_data))
                    self.image_request_queue.task_done()
                    continue

                # --- Pre-fetch Logic ---
                image_bytes, failure_reason = None, None
                try:
                    api_key = self._get_api_key_for_guild(request_data['guild_id'])
                    if not api_key: raise ValueError("Server API key is not configured.")
                    genai.configure(api_key=api_key)
                    
                    image_model = genai.GenerativeModel('gemini-2.5-flash-image', system_instruction=request_data['system_instruction'], safety_settings=request_data['safety_settings'])
                    
                    status = "api_error"
                    try:
                        response = await image_model.generate_content_async(request_data['prompt_text'])
                        status = "blocked_by_safety" if not response.candidates else "success"
                    finally:
                        self._log_api_call(user_id=request_data['author_id'], guild_id=request_data['guild_id'], context="image_generation_prefetch", model_used=image_model.model_name, status=status)

                    if not response.candidates:
                        reason = "Safety Filter";
                        if response.prompt_feedback and response.prompt_feedback.block_reason: reason = response.prompt_feedback.block_reason.name.replace('_', ' ').title()
                        failure_reason = f"the safety filter ({reason})"
                    else:
                        candidate = response.candidates[0]
                        if candidate.finish_reason.name != 'STOP':
                            failure_reason = f"the process being stopped for reason: **{candidate.finish_reason.name.replace('_', ' ').title()}**"
                        else:
                            image_bytes = next((part.inline_data.data for part in candidate.content.parts if hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/')), None)
                            if not image_bytes: failure_reason = "an unknown issue (the model returned no image data)"
                except Exception as e:
                    failure_reason = f"an unexpected error: `{e}`"
                
                request_data['generated_image_bytes'] = image_bytes
                request_data['failure_reason'] = failure_reason
                
                # Pass priority to next stage
                await self.text_request_queue.put((priority, request_data))
                self.image_request_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in image generation worker #{worker_id}: {e}"); traceback.print_exc()

    async def _get_relevant_ltm_for_prompt(self, session_key: Any, history: list, profile_owner_id: int, profile_name: str, msg_content: str, author_dn: str, guild_id: Optional[int], triggering_user_id: int) -> Optional[str]:
        user_data = self._get_user_data_entry(profile_owner_id)
        is_borrowed = profile_name in user_data.get("borrowed_profiles", {})

        ltm_params_owner_data = user_data
        ltm_params_profile_name = profile_name
        if is_borrowed:
            borrowed_data = user_data["borrowed_profiles"][profile_name]
            owner_id = int(borrowed_data["original_owner_id"])
            owner_profile_name = borrowed_data["original_profile_name"]
            ltm_params_owner_data = self._get_user_data_entry(owner_id)
            ltm_params_profile_name = owner_profile_name

        params_source = ltm_params_owner_data.get("profiles", {}).get(ltm_params_profile_name)
        if not params_source:
            return None

        ltm_context_size = int(params_source.get("ltm_context_size", 3))
        ltm_relevance_threshold = float(params_source.get("ltm_relevance_threshold", 0.75))

        if ltm_context_size == 0:
            return None

        owner_id_str = str(profile_owner_id)
        context_type = "guild"
        ltm_data = self._load_ltm_shard(owner_id_str, profile_name)
        if not ltm_data:
            return None
        all_profile_ltms = ltm_data.get(context_type, [])
        if not all_profile_ltms:
            return None

        # [NEW] matryoshka 256-dim embedding via SDK v2
        prompt_embedding = await self._get_embedding(msg_content, guild_id, task_type="RETRIEVAL_QUERY")
        if not prompt_embedding:
            return None

        current_turn = len(history)
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        session_cooldown_history = self.ltm_recall_history.get(session_key, {})
        
        candidate_ltms = []
        for ltm in all_profile_ltms:
            ltm_id = ltm.get('id')
            if ltm_id in session_cooldown_history:
                last_turn, last_sim = session_cooldown_history[ltm_id]
                dynamic_cooldown = 5 + (1 - last_sim) * 25 
                if current_turn - last_turn < dynamic_cooldown:
                    continue

            scope = ltm.get('scope')
            context_id = str(ltm.get('context_id')) if ltm.get('context_id') is not None else None
            is_valid_scope = False
            if scope == 'global':
                is_valid_scope = True
            elif scope == 'server' and context_id == str(guild_id):
                is_valid_scope = True
            elif scope == 'user' and context_id == str(profile_owner_id) and triggering_user_id == profile_owner_id:
                is_valid_scope = True
            
            if not is_valid_scope:
                continue

            if "s_emb" in ltm and ltm["s_emb"]:
                dequantized_embedding = _dequantize_embedding(ltm["s_emb"])
                similarity = cosine_similarity(prompt_embedding, dequantized_embedding)
                
                created_ts_str = ltm.get('created_ts') or ltm.get('ts')
                if created_ts_str:
                    try:
                        created_dt = datetime.datetime.fromisoformat(created_ts_str)
                        days_old = (now_utc - created_dt).total_seconds() / 86400.0
                        decay_factor = 0.995
                        decayed_similarity = similarity * (decay_factor ** days_old)
                    except (ValueError, TypeError):
                        decayed_similarity = similarity
                else:
                    decayed_similarity = similarity

                if decayed_similarity >= ltm_relevance_threshold:
                    candidate_ltms.append({
                        "ltm": ltm, 
                        "sim": decayed_similarity, 
                        "original_sim": similarity,
                        "embedding": dequantized_embedding
                    })

        if not candidate_ltms:
            return None

        candidate_ltms.sort(key=lambda x: x["sim"], reverse=True)

        final_memories = []
        saturation_penalty_factor = 0.75
        
        while len(final_memories) < ltm_context_size and candidate_ltms:
            best_memory = candidate_ltms.pop(0)
            final_memories.append(best_memory)
            
            if not candidate_ltms:
                break

            selected_embedding = best_memory["embedding"]
            for other_memory in candidate_ltms:
                saturation_similarity = cosine_similarity(selected_embedding, other_memory["embedding"])
                penalty = (1.0 - (saturation_similarity * saturation_penalty_factor))
                other_memory["sim"] *= penalty
            
            candidate_ltms.sort(key=lambda x: x["sim"], reverse=True)

        if not final_memories:
            return None

        if session_key not in self.ltm_recall_history:
            self.ltm_recall_history[session_key] = {}

        recalled_summaries = []
        for mem_data in final_memories:
            ltm = mem_data["ltm"]
            self.ltm_recall_history[session_key][ltm['id']] = (current_turn, mem_data["original_sim"])
            
            decrypted_sum = self._decrypt_data(ltm.get('sum', ''))
            recalled_summaries.append(decrypted_sum)
        
        if not recalled_summaries:
            return None

        return "<archive_context>\n" + "\n".join(recalled_summaries) + "\n</archive_context>"
    
    async def _get_relevant_training_examples(self, profile_owner_id: int, profile_name: str, msg_content:str, guild_id: int)->List[str]:
        # [UPDATED] Check context size before disk or API activity
        _, _, _, _, _, _, training_context_size, training_relevance_threshold, _, _ = self._get_user_profile_for_model(profile_owner_id, 0, profile_name)
        if training_context_size == 0:
            return []

        owner_id_str = str(profile_owner_id)
        user_data = self._get_user_data_entry(profile_owner_id)
        is_borrowed = profile_name in user_data.get("borrowed_profiles", {})
        
        effective_owner_id_for_training = profile_owner_id
        effective_profile_name_for_training = profile_name
        
        if is_borrowed:
            borrowed_data = user_data["borrowed_profiles"][profile_name]
            effective_owner_id_for_training = int(borrowed_data["original_owner_id"])
            effective_profile_name_for_training = borrowed_data["original_profile_name"]

        profile_examples = self._load_training_shard(str(effective_owner_id_for_training), effective_profile_name_for_training)

        if not profile_examples: return []
        msg_emb=await self._get_embedding(msg_content, guild_id, task_type="RETRIEVAL_QUERY")
        if not msg_emb:return[]
        
        sc = []
        for ex in profile_examples:
            if ex.get("u_emb"):
                dequantized_emb = _dequantize_embedding(ex.get("u_emb", []))
                sc.append({"ex": ex, "sim": cosine_similarity(msg_emb, dequantized_emb)})

        sc.sort(key=lambda x:x["sim"],reverse=True)
        
        relevant_examples = []
        for i in sc:
            if i["sim"] >= training_relevance_threshold:
                decrypted_u_in = self._decrypt_data(i['ex']['u_in'])
                decrypted_b_out = self._decrypt_data(i['ex']['b_out'])
                relevant_examples.append(f"<example>\nUser: {decrypted_u_in}\nYou: {decrypted_b_out}\n</example>")
        return relevant_examples[:training_context_size]
    
    async def _get_embedding(self, text: str, guild_id: int, task_type: str = "RETRIEVAL_QUERY") -> Optional[List[float]]:
        if not text or not text.strip():
            return None
            
        api_key = self._get_api_key_for_guild(guild_id)
        if not api_key: 
            return None

        # [NEW] Use the Google Gen AI SDK (v2) Client
        client = google_genai.Client(api_key=api_key)

        try:
            # Request truncated 256-dimensional embedding to save space (Matryoshka)
            result = await client.aio.models.embed_content(
                model='gemini-embedding-001',
                contents=text,
                config=google_genai_types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=256
                )
            )
            # The new SDK returns a list of embedding objects; we access values of the first one
            return result.embeddings[0].values
        except Exception as e:
            print(f"Embedding err for '{text[:30]}...': {e}")
            return None
        
    async def _generate_ltm_data_from_history(self, hist:list, user_dn:str, gen_config_params: Dict[str, Any], model_name_to_use: str, guild_id: Optional[int], bot_dn: str = "Bot", profile_owner_id: int = None, profile_name: str = None) -> Optional[str]:
        if not hist or len(hist) < MIN_HISTORY_FOR_LTM_CREATION: return None
        
        # [UPDATED] Standardize history for the LTM Summarizer
        # Provides Name [Timestamp] and Content only, stripping metadata and summaries.
        convo_parts = []
        for turn in hist:
            # Handle Content objects from single-profile or list of strings
            if hasattr(turn, 'role'):
                display_name = user_dn if turn.role == 'user' else bot_dn
                if not turn.parts: continue
                raw_text = "".join(p.text for p in turn.parts if hasattr(p, 'text'))
            else:
                raw_text = str(turn)
                display_name = "Unknown" # String-only fallback

            if not raw_text: continue

            # 1. Strip technical metadata
            text = re.sub(r'\(\s*Thought Initiated:.*?\)\s*\n?', '', raw_text).strip()
            
            # 2. Strip previous contexts to avoid "recursive" memory creation
            lines = text.split('\n')
            filtered_lines = []
            skip_block = False
            for line in lines:
                l_strip = line.strip()
                if any(l_strip.startswith(prefix) for prefix in [
                    "<external_context>",
                    "<document_context>",
                    "<archive_context>",
                    "<internal_note>",
                    "<image_context>"
                ]):
                    skip_block = True
                    continue
                if skip_block:
                    if l_strip.startswith(("</external_context>", "</document_context>", "</archive_context>", "</internal_note>", "</image_context>")):
                        skip_block = False
                    continue
                filtered_lines.append(line)
            
            final_content = "\n".join(filtered_lines).strip()
            if final_content:
                # If it doesn't already have a Name [Timestamp] header (e.g. from single-profile hist), add it
                if not re.match(r'^.+ \[[^\]]+\]:', final_content):
                    ts_str = datetime.datetime.now(datetime.timezone.utc).strftime("[%a, %d %b %Y, %I:%M %p UTC]")
                    convo_parts.append(f"{display_name} {ts_str}:\n{final_content}")
                else:
                    convo_parts.append(final_content)

        convo = "\n\n".join(convo_parts)

        if len(convo) > 3000: # Slightly higher limit for formatted text
            convo = convo[-3000:]
        
        instructions = DEFAULT_LTM_SUMMARIZATION_INSTRUCTIONS
        if profile_owner_id and profile_name:
            user_data = self._get_user_data_entry(profile_owner_id)
            is_borrowed = profile_name in user_data.get("borrowed_profiles", {})

            if is_borrowed:
                borrowed_data = user_data["borrowed_profiles"][profile_name]
                source_owner_id = int(borrowed_data["original_owner_id"])
                source_profile_name = borrowed_data["original_profile_name"]
                
                source_owner_user_data = self._get_user_data_entry(source_owner_id)
                profile_data = source_owner_user_data.get("profiles", {}).get(source_profile_name, {})
            else:
                # LTM summarization instructions are only on personal profiles
                profile_data = user_data.get("profiles", {}).get(profile_name, {})
            
            encrypted_instructions = profile_data.get("ltm_summarization_instructions", self._encrypt_data(DEFAULT_LTM_SUMMARIZATION_INSTRUCTIONS))
            instructions = self._decrypt_data(encrypted_instructions)

        cfg = genai_types.GenerationConfig(temperature=0.2)
        ltm_model_name = 'gemini-flash-lite-latest'
        
        status = "api_error"
        try:
            m = genai.GenerativeModel(
                model_name=ltm_model_name,
                system_instruction=instructions,
                safety_settings=DEFAULT_SAFETY_SETTINGS
            )
            r = await m.generate_content_async(f"<target_transcript>\n{convo}\n</target_transcript>", generation_config=cfg)
            status = "blocked_by_safety" if not r.candidates else "success"
            
            response_text = ""
            if r.candidates:
                candidate = r.candidates[0]
                if candidate.content and candidate.content.parts:
                    response_text = "".join(p.text for p in candidate.content.parts if hasattr(p, 'text')).strip()

            if response_text and response_text.upper() != "NO_SUMMARY":
                return response_text
        except Exception as e: 
            print(f"LTM Gen err {user_dn}: {e}")
            traceback.print_exc()
        return None
    
    async def _maybe_create_ltm(self, context_obj: Union[discord.Message, discord.abc.Messageable], author_dn: str, hist: list, profile_owner_id: int, profile_name: str, gen_config_params: Dict[str, Any], force_user_scope: bool = False, triggering_user_id_override: Optional[int] = None):
        guild = getattr(context_obj, 'guild', None)
        
        user_data = self._get_user_data_entry(profile_owner_id)
        is_borrowed = profile_name in user_data.get("borrowed_profiles", {})
        profile_settings = user_data.get("borrowed_profiles" if is_borrowed else "profiles", {}).get(profile_name, {})
        
        if not profile_settings.get("ltm_creation_enabled", False):
            return

        context_type: Literal["guild", "dm"] = "guild" if guild else "dm"
        
        ltm_counter_key = (profile_owner_id, profile_name, context_type)
        # 1 exchange = 1 increment
        self.message_counters_for_ltm[ltm_counter_key] = self.message_counters_for_ltm.get(ltm_counter_key, 0) + 1
        
        interval = profile_settings.get("ltm_creation_interval", 10)
        context_size = profile_settings.get("ltm_summarization_context", 10)
     
        if self.message_counters_for_ltm[ltm_counter_key] >= interval:
            self.message_counters_for_ltm[ltm_counter_key] = 0
            # Context size * 2 because turn history includes both user and bot
            h_sum = hist[-(context_size * 2):]
            if len(h_sum) < 4: # Minimal safety check
                return

            print(f"[DEBUG: LTM] Triggering summary for {profile_name} using {len(h_sum)} context turns.")

            guild_id = None
            channel_id = None
            author = None
            triggering_user_id = None

            if isinstance(context_obj, discord.Message):
                guild_id = context_obj.guild.id if context_obj.guild else None
                channel_id = context_obj.channel.id
                author = context_obj.author
                triggering_user_id = author.id if author else self.bot.user.id
            else: # Is a TextChannel from a child bot
                guild_id = context_obj.guild.id if context_obj.guild else None
                channel_id = context_obj.id
                triggering_user_id = triggering_user_id_override or self.bot.user.id

            _, _, _, temp, top_p, top_k, _, _, _, fallback_model = self._get_user_profile_for_model(profile_owner_id, channel_id, profile_name)
            effective_gen_config = {"temperature": temp, "top_p": top_p, "top_k": top_k}
            
            user_id_map = {triggering_user_id: author_dn}
            sanitized_history, sanitized_author = self._get_sanitized_history_and_author(h_sum, user_id_map, triggering_user_id)

            user_data = self._get_user_data_entry(profile_owner_id)
            is_borrowed = profile_name in user_data.get("borrowed_profiles", {})
            effective_owner_id = profile_owner_id
            effective_profile_name = profile_name
            if is_borrowed:
                borrowed_data = user_data["borrowed_profiles"][profile_name]
                effective_owner_id = int(borrowed_data["original_owner_id"])
                effective_profile_name = borrowed_data["original_profile_name"]
            
            bot_display_name = effective_profile_name
            appearance = self.user_appearances.get(str(effective_owner_id), {}).get(effective_profile_name, {})
            if appearance.get("custom_display_name"):
                bot_display_name = appearance["custom_display_name"]

            summary = await self._generate_ltm_data_from_history(sanitized_history, sanitized_author, effective_gen_config, fallback_model, guild_id, bot_dn=bot_display_name, profile_owner_id=profile_owner_id, profile_name=profile_name)
            if summary:
                summary_embedding = await self._get_embedding(summary, guild_id, task_type="RETRIEVAL_DOCUMENT")
                if summary_embedding:
                    quantized_embedding = _quantize_embedding(summary_embedding)
                    self._add_ltm(profile_owner_id, profile_name, summary, quantized_embedding, guild.id if guild else None, triggering_user_id, sanitized_author, force_user_scope=force_user_scope)

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

        custom_display_name_to_use = self.bot.user.name if self.bot.user else "Bot"
        custom_avatar_url_to_use = self.bot.user.display_avatar.url if self.bot.user else None
        use_webhook = False
        is_realistic_typing = False
        
        is_placeholder = (content == f"{PLACEHOLDER_EMOJI}")

        if profile_owner_id_for_appearance is not None and profile_name_for_appearance:
            borrower_user_data = self._get_user_data_entry(profile_owner_id_for_appearance)
            is_borrowed = profile_name_for_appearance in borrower_user_data.get("borrowed_profiles", {})

            if is_borrowed:
                borrowed_data = borrower_user_data["borrowed_profiles"][profile_name_for_appearance]
                is_realistic_typing = borrowed_data.get("realistic_typing_enabled", False)
                
                effective_owner_id = int(borrowed_data["original_owner_id"])
                effective_profile_name = borrowed_data["original_profile_name"]
            else:
                personal_profile_data = borrower_user_data.get("profiles", {}).get(profile_name_for_appearance, {})
                is_realistic_typing = personal_profile_data.get("realistic_typing_enabled", False)

                effective_owner_id = profile_owner_id_for_appearance
                effective_profile_name = profile_name_for_appearance
            
            # [NEW] Enforce bypass if flag is set
            if bypass_typing:
                is_realistic_typing = False

            owner_id_str = str(effective_owner_id)
            appearance_data = self.user_appearances.get(owner_id_str, {}).get(effective_profile_name)

            if appearance_data and (appearance_data.get("custom_display_name") or appearance_data.get("custom_avatar_url")):
                use_webhook = True
                custom_display_name_to_use = appearance_data.get("custom_display_name") or custom_display_name_to_use
                
                if appearance_data.get("custom_avatar_url"):
                    custom_avatar_url_to_use = appearance_data["custom_avatar_url"]
                else:
                    # Use generic Discord default avatar if display name is set but avatar is not
                    avatar_index = hash(effective_profile_name) % 6
                    custom_avatar_url_to_use = f"https://cdn.discordapp.com/embed/avatars/{avatar_index}.png"
                    
            elif profile_name_for_appearance != DEFAULT_PROFILE_NAME:
                use_webhook = True
                custom_display_name_to_use = profile_name_for_appearance
                avatar_index = hash(profile_name_for_appearance) % 6
                custom_avatar_url_to_use = f"https://cdn.discordapp.com/embed/avatars/{avatar_index}.png"
        
        if target_message_to_edit and use_webhook and content != f"{PLACEHOLDER_EMOJI}":
            try:
                await target_message_to_edit.delete()
            except discord.NotFound:
                pass 
            except Exception as e:
                print(f"Failed to delete placeholder message {target_message_to_edit.id}: {e}")
            target_message_to_edit = None

        sent_messages_list: List[discord.Message] = []

        if file and not is_realistic_typing:
            if not content.endswith("\n\u200b"):
                content += "\n\u200b"

        if is_realistic_typing and isinstance(channel, (discord.TextChannel, discord.Thread)) and content != f"{PLACEHOLDER_EMOJI}":
            try:
                webhook_to_use = await self._get_or_create_webhook(channel)
                if not webhook_to_use:
                    raise ValueError("Could not get webhook for realistic typing.")

                if target_message_to_edit:
                    try: await target_message_to_edit.delete()
                    except (discord.NotFound, discord.Forbidden): pass
                
                sentences = _split_into_sentences_with_abbreviations(content)
                
                for i, sentence in enumerate(sentences):
                    if not isinstance(sentence, str) or not sentence.strip():
                        continue
                    
                    delay = max(0.5, min(len(sentence) / 30.0, 2.5))
                    await asyncio.sleep(delay)
                    
                    embeds_to_send = embeds if i == 0 and embeds is not None else []
                    
                    send_kwargs = {
                        "content": sentence,
                        "username": custom_display_name_to_use,
                        "avatar_url": custom_avatar_url_to_use,
                        "embeds": embeds_to_send,
                        "wait": True
                    }
                    if i == 0 and file:
                        send_kwargs["file"] = file
                    if isinstance(channel, discord.Thread):
                        send_kwargs["thread"] = channel
                    
                    sent_message_part = await webhook_to_use.send(**send_kwargs)
                    if sent_message_part:
                        sent_messages_list.append(sent_message_part)
                        if i == 0 and store_prompt_for_id:
                            self.message_id_to_original_prompt[sent_message_part.id] = store_prompt_for_id
                
                return sent_messages_list
            except Exception as e:
                print(f"Realistic typing failed, falling back to standard send. Error: {e}")
                traceback.print_exc()

        remaining_content = content
        is_first_chunk = True
        
        while remaining_content:
            chunk = ""
            if len(remaining_content) <= DISCORD_MAX_MESSAGE_LENGTH:
                chunk = remaining_content
                remaining_content = ""
            else:
                split_pos = -1
                para_break = remaining_content.rfind('\n\n', 0, DISCORD_MAX_MESSAGE_LENGTH)
                if para_break != -1:
                    split_pos = para_break + 2
                else:
                    sent_break = remaining_content.rfind('. ', 0, DISCORD_MAX_MESSAGE_LENGTH)
                    if sent_break != -1:
                        split_pos = sent_break + 2
                    else:
                        split_pos = DISCORD_MAX_MESSAGE_LENGTH

                chunk = remaining_content[:split_pos]
                remaining_content = remaining_content[split_pos:]

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
            if is_first_chunk and reply_to and not is_placeholder:
                # Handle Response Mode for Webhooks
                user_data_target = self._get_user_data_entry(profile_owner_id_for_appearance)
                target_profile_settings = user_data_target.get("profiles", {}).get(profile_name_for_appearance, {})
                if not target_profile_settings:
                    target_profile_settings = user_data_target.get("borrowed_profiles", {}).get(profile_name_for_appearance, {})
                
                rmode = target_profile_settings.get("response_mode", "regular")
                # Webhooks fallback to text mention for 'mention' and 'mention_reply'
                if rmode in ["mention", "mention_reply"]:
                    final_content_for_send = f"{reply_to.author.mention} {final_content_for_send}"

            sent_message_part: Optional[discord.Message] = None

            webhook_to_use = None
            if use_webhook and isinstance(channel, (discord.TextChannel, discord.Thread)):
                webhook_to_use = await self._get_or_create_webhook(channel)

            if webhook_to_use:
                try:
                    send_kwargs = {
                        "content": final_content_for_send,
                        "username": custom_display_name_to_use,
                        "avatar_url": custom_avatar_url_to_use,
                        "embeds": current_embeds_for_api,
                        "wait": True
                    }
                    if current_file_for_api:
                        send_kwargs["file"] = current_file_for_api
                    if isinstance(channel, discord.Thread):
                        send_kwargs["thread"] = channel
                    
                    sent_message_part = await webhook_to_use.send(**send_kwargs)
                except Exception as e:
                    print(f"Webhook send failed, falling back to regular message. Error: {e}")
                    sent_message_part = None 
            
            if not sent_message_part:
                if current_target_to_edit:
                    try:
                        sent_message_part = await current_target_to_edit.edit(content=final_content_for_send, embeds=current_embeds_for_api)
                    except discord.HTTPException:
                        sent_message_part = await channel.send(final_content_for_send, embeds=current_embeds_for_api, file=current_file_for_api)
                else:
                    sent_message_part = await channel.send(final_content_for_send, embeds=current_embeds_for_api, file=current_file_for_api)

            if sent_message_part:
                sent_messages_list.append(sent_message_part)
                if current_store_prompt:
                    self.message_id_to_original_prompt[sent_message_part.id] = store_prompt_for_id

            is_first_chunk = False
            if remaining_content:
                await asyncio.sleep(0.5)

        return sent_messages_list

    async def _get_or_create_webhook(self, channel: Union[discord.TextChannel, discord.Thread]) -> Optional[discord.Webhook]:
        parent_channel = channel.parent if isinstance(channel, discord.Thread) else channel
        if parent_channel.id in self.channel_webhooks:
            try:
                wh_data = self.channel_webhooks[parent_channel.id]
                return discord.Webhook.from_url(wh_data['url'], session=self.bot.http._HTTPClient__session)
            except (discord.errors.InvalidArgument, KeyError):
                pass
        
        if parent_channel.permissions_for(parent_channel.guild.me).manage_webhooks:
            try:
                webhooks = await parent_channel.webhooks()
                bot_webhook = next((wh for wh in webhooks if wh.user and wh.user.id == self.bot.user.id), None)
                if not bot_webhook:
                    bot_webhook = await parent_channel.create_webhook(name=f"{self.bot.user.name} Webhook", reason="For custom appearances")
                
                self.channel_webhooks[parent_channel.id] = {'url': bot_webhook.url}
                self._save_channel_webhooks()
                return bot_webhook
            except Exception as e:
                print(f"Failed to get/create webhook for {parent_channel.name}: {e}")
        return None
    
    def _scrub_response_text(self, text: str, participant_names: Optional[List[str]] = None) -> str:
        """Hard-coded filter to remove any leaked script formatting or XML tags from the AI's response."""
        try:
            with Timeout(seconds=2, error_message="Scrubbing timed out due to complex regex."):
                scrubbed_text = text.strip()
                scrubbed_text = scrubbed_text.replace("&#x20;", " ")

                # Universal XML Tag Scrubber (Internal thoughts, metadata, and context)
                scrubbed_text = re.sub(r'<[^>]+>.*?</[^>]+>', '', scrubbed_text, flags=re.DOTALL)
                scrubbed_text = re.sub(r'<[^>]+>', '', scrubbed_text) # Catch unclosed tags

                # Pattern to remove "DISPLAY_NAME [TIMESTAMP]:\n" format.
                pattern_timestamp_global = r'[^\[\n]+ \[[^\]]+\]:\s*\n?'
                scrubbed_text = re.sub(pattern_timestamp_global, '', scrubbed_text).strip()

                if participant_names:
                    escaped_names = [re.escape(name) for name in participant_names]
                    names_pattern_part = "|".join(escaped_names)
                    pattern_multiline_prefix = rf'^\s*(?:{names_pattern_part}):\s*'
                    scrubbed_text = re.sub(pattern_multiline_prefix, '', scrubbed_text, flags=re.MULTILINE | re.IGNORECASE).strip()

                scrubbed_text = re.sub(r'Message\s*#[\w-]+', '', scrubbed_text).strip()
                scrubbed_text = re.sub(r'\(\s*Thought Initiated:.*?\)\s*\n?', '', scrubbed_text).strip()

                return scrubbed_text
        except TimeoutError as e:
            print(f"Warning: {e}. Returning original text.")
            return text
        
    def _deduplicate_response(self, text: str) -> str:
        try:
            with Timeout(seconds=2, error_message="Deduplication timed out due to complex regex."):
                clean_text = text.strip()
                if not clean_text:
                    return ""

                final_text = clean_text
                block_found = False

                # 1. New: Check for a large repeating block that constitutes the entire message.
                # This is separator-agnostic and handles cases like A-A-A or A...A...A
                min_block_len = 30 # Avoid matching small phrases
                # Iterate from largest possible block size down to smallest
                for block_len in range(len(clean_text) // 2, min_block_len, -1):
                    block = clean_text[:block_len]
                    if not block.strip():
                        continue
                    
                    parts = clean_text.split(block)
                    
                    if len(parts) > 2 and all(not p.strip() for p in parts):
                        final_text = block.strip()
                        block_found = True
                        break
                
                if not block_found:
                    # 2. Fallback: Perfect AA duplication
                    length = len(clean_text)
                    if length > 1 and length % 2 == 0:
                        half_len = length // 2
                        if clean_text[:half_len].strip() == clean_text[half_len:].strip():
                            final_text = clean_text[:half_len]

                    # 3. Fallback: Paragraph-level duplication check
                    if final_text == clean_text:
                        paragraphs = clean_text.split('\n\n')
                        paragraphs = [p.strip() for p in paragraphs if p.strip()]
                        
                        if len(paragraphs) > 1:
                            if len(paragraphs) % 2 == 0:
                                half = len(paragraphs) // 2
                                if paragraphs[:half] == paragraphs[half:]:
                                    final_text = '\n\n'.join(paragraphs[:half])
                            
                            if final_text == clean_text:
                                unique_paragraphs = []
                                last_p = None
                                for p in paragraphs:
                                    if p != last_p:
                                        unique_paragraphs.append(p)
                                    last_p = p
                                if len(unique_paragraphs) < len(paragraphs):
                                    final_text = '\n\n'.join(unique_paragraphs)

                    # 4. Fallback: Sentence-level duplication check
                    if final_text == clean_text:
                        sentences = re.split(r'(?<=[.!?])\s+', clean_text)
                        sentences = [s.strip() for s in sentences if s.strip()]
                        if len(sentences) > 1:
                            if len(sentences) % 2 == 0:
                                half = len(sentences) // 2
                                if sentences[:half] == sentences[half:]:
                                    final_text = ' '.join(sentences[:half])

                            if final_text == clean_text:
                                unique_sentences = []
                                last_s = None
                                for s in sentences:
                                    if s != last_s:
                                        unique_sentences.append(s)
                                    last_s = s
                                if len(unique_sentences) < len(sentences):
                                    final_text = ' '.join(unique_sentences)
                
                # Check for massive repetition that wasn't caught by other methods
                match = re.search(r'(.)\1{999,}', final_text)
                if match:
                    # If found, it indicates a model failure. Return empty to trigger the generic error.
                    return ""

                # Final sanitization to remove trailing, unclosed code block markers and whitespace
                final_text = re.sub(r'[`\s]*$', '', final_text)

                return final_text.strip()
        except TimeoutError as e:
            print(f"Warning: {e}. Returning original text.")
            return text
        
    def _format_history_entry(self, display_name: str, timestamp: Union[datetime.datetime, str], content: str, timezone_str: str = "UTC") -> str:
        # Convert string timestamp to datetime object if necessary
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp)
            except ValueError:
                # If parsing fails, default to current time to prevent crash
                timestamp = datetime.datetime.now(datetime.timezone.utc)

        try:
            target_tz = ZoneInfo(timezone_str)
            local_time = timestamp.astimezone(target_tz)
            time_str = local_time.strftime("[%a, %d %b %Y, %I:%M %p %Z]")
        except Exception:
            time_str = timestamp.strftime("[%a, %d %b %Y, %I:%M %p UTC]")
        return f"{display_name} {time_str}:\n{content}\n"
    
    @tasks.loop(minutes=1.0)
    async def freewill_task(self):
        if not self.has_lock:
            return

        for guild_id_str, config in self.freewill_config.items():
            guild = self.bot.get_guild(int(guild_id_str))
            if not guild: continue

            # Combine living channels list. We only process 'Living' channels for proactive events.
            living_channels = config.get("living_channel_ids", [])
            channel_settings_map = config.get("channel_settings", {})

            for channel_id in living_channels:
                # 1. Check Cooldown
                ch_settings = channel_settings_map.get(str(channel_id), {})
                cooldown = ch_settings.get("event_cooldown", 300) # Default 5 mins
                if time.time() - self.last_freewill_event.get(channel_id, 0) < cooldown:
                    continue

                # 2. Check Chance
                ev_chance = ch_settings.get("event_chance", 0) # Default 0 (Manual only)
                if ev_chance <= 0:
                    continue
                
                threshold = ev_chance / 100.0
                if random.random() > threshold:
                    continue

                # 3. Execution Logic
                channel = guild.get_channel(channel_id)
                if not channel or not isinstance(channel, discord.TextChannel): continue
                
                session = self.multi_profile_channels.get(channel.id)
                if session and session.get("type") != "freewill": continue
                if session and session.get('is_running'): continue
                
                if session and not session.get("is_hydrated"):
                    session = self._ensure_session_hydrated(channel.id, "freewill")
                    
                opted_in_profiles = []
                server_participation = self.freewill_participation.get(str(guild.id), {})
                channel_participants = server_participation.get(str(channel.id), {})

                for user_id_str, profiles in channel_participants.items():
                    if not self.is_user_premium(int(user_id_str)): continue

                    member = guild.get_member(int(user_id_str))
                    if member:
                        for profile_name, settings in profiles.items():
                            pers = settings.get("personality", "off")
                            is_active = False
                            if isinstance(pers, int): is_active = pers > 0
                            else: is_active = pers != "off"

                            if is_active:
                                participant_dict = self._build_freewill_participant_dict(int(user_id_str), profile_name, channel)
                                if participant_dict:
                                    opted_in_profiles.append(participant_dict)
                
                if len(opted_in_profiles) < 2: continue
                
                self.last_freewill_event[channel.id] = time.time()
                
                cast_size = random.choices([2, 3], weights=[0.8, 0.2], k=1)[0]
                if len(opted_in_profiles) < cast_size: cast_size = len(opted_in_profiles)
                cast = random.sample(opted_in_profiles, k=cast_size)
                session_owner_id = cast[0]['owner_id']

                if not session:
                    chat_sessions = {}
                    for p in cast:
                        model = genai.GenerativeModel('gemini-flash-latest')
                        chat_sessions[(p['owner_id'], p['profile_name'])] = model.start_chat(history=[])

                    session = {
                        "type": "freewill", "freewill_mode": "proactive",
                        "proactive_initial_rounds": 2, "proactive_cooldown": cooldown,
                        "chat_sessions": chat_sessions, "unified_log": [],
                        "initial_channel_history": await self._build_freewill_history(channel),
                        "initial_turn_taken": set(), "last_bot_message_id": None,
                        "owner_id": session_owner_id, "is_running": False,
                        "task_queue": asyncio.Queue(), "worker_task": None,
                        "turns_since_last_ltm": 0, "session_prompt": None,
                        "profiles": cast, "is_hydrated": True
                    }
                    self.multi_profile_channels[channel.id] = session
                else:
                    session['profiles'] = cast 
                    session['freewill_mode'] = "proactive"
                    session['proactive_initial_rounds'] = 2
                    session['proactive_cooldown'] = cooldown
                    session['initial_channel_history'] = await self._build_freewill_history(channel)
                    session['initial_turn_taken'] = set()
                    
                    dummy_model = genai.GenerativeModel('gemini-flash-latest')
                    if 'chat_sessions' not in session: session['chat_sessions'] = {}
                    chat_sessions = session['chat_sessions']
                    
                    for p in cast:
                        p_key = (p['owner_id'], p['profile_name'])
                        if p_key not in chat_sessions or chat_sessions[p_key] is None:
                            chat_sessions[p_key] = dummy_model.start_chat(history=[])

                self._save_multi_profile_sessions()
                
                dummy_session_key = (channel.id, None, None)
                self._save_session_to_disk(dummy_session_key, "freewill", session.get("unified_log", []))

                target_participant = cast[1]
                target_id = target_participant['owner_id']
                target_profile = target_participant['profile_name']
                target_user_data = self._get_user_data_entry(target_id)
                target_appearance_name = target_profile
                if target_profile in target_user_data.get("borrowed_profiles", {}):
                    borrowed_data = target_user_data["borrowed_profiles"][target_profile]
                    target_appearance_name = borrowed_data["original_profile_name"]
                
                target_display_name = target_appearance_name
                if str(target_id) in self.user_appearances and target_appearance_name in self.user_appearances[str(target_id)]:
                    appearance = self.user_appearances[str(target_id)][target_appearance_name]
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

                await session['task_queue'].put(director_prompt)
                
                if not session.get('is_running'):
                    session['is_running'] = True 
                    session['worker_task'] = self.bot.loop.create_task(self._multi_profile_worker(channel.id))

    @tasks.loop(seconds=60.0)
    async def evict_inactive_sessions_task(self):
        now = time.time()
        inactive_threshold = 60  # 1 minute strict dehydration policy
        
        keys_to_evict = [
            key for key, last_time in self.session_last_accessed.items()
            if now - last_time > inactive_threshold
        ]
        
        for key in keys_to_evict:
            # Handle multi-profile sessions (key is channel_id int)
            if isinstance(key, int):
                session_to_evict = self.multi_profile_channels.get(key)
                if session_to_evict and session_to_evict.get("is_hydrated"):
                    session_type = session_to_evict.get("type", "multi")
                    unified_log = session_to_evict.get("unified_log")
                    
                    if unified_log is not None:
                        dummy_session_key = (key, None, None)
                        self._save_session_to_disk(dummy_session_key, session_type, unified_log)

                    if session_to_evict.get('worker_task') and not session_to_evict['worker_task'].done():
                        session_to_evict['worker_task'].cancel()
                    
                    mapping_key = (session_type, key)
                    if mapping_key in self.mapping_caches:
                        self._save_mapping_to_disk(mapping_key, self.mapping_caches.pop(mapping_key))

                    # FIXED: Aggressively dehydrate the session to release memory.
                    session_to_evict['is_hydrated'] = False
                    
                    # Clear the large data structures from the in-memory dictionary.
                    if 'unified_log' in session_to_evict:
                        del session_to_evict['unified_log']
                    if 'chat_sessions' in session_to_evict:
                        session_to_evict['chat_sessions'].clear()
                    
                    # Give the garbage collector a hint to clean up now.
                    gc.collect()

            # Handle single-profile and global chat sessions (key is a tuple)
            else:
                session_to_save = self.chat_sessions.get(key) or self.global_chat_sessions.get(key)
                if session_to_save:
                    session_type = 'global_chat' if key in self.global_chat_sessions else 'single'
                    self._save_session_to_disk(key, session_type, session_to_save)
                    
                    mapping_key = self._get_mapping_key_for_session(key, session_type)
                    if mapping_key in self.mapping_caches:
                        self._save_mapping_to_disk(mapping_key, self.mapping_caches.pop(mapping_key))

                self.chat_sessions.pop(key, None)
                self.global_chat_sessions.pop(key, None)

            # Remove from tracking dict after processing
            self.session_last_accessed.pop(key, None)

    @tasks.loop(hours=168.0) # 168 hours = 7 days
    async def weekly_cleanup_task(self):
        if self.has_lock:
            print("Starting weekly data cleanup...")
            await self._perform_data_cleanup()
            print("Weekly data cleanup finished.")

    async def _perform_data_cleanup(self):
        import shutil
        import time

        log = ["Starting Automatic Weekly Data Cleanup..."]
        bot_guild_ids = {g.id for g in self.bot.guilds}
        all_bot_member_ids = {str(m.id) for g in self.bot.guilds for m in g.members}
        all_bot_channel_ids = {c.id for g in self.bot.guilds for c in g.channels}

        # --- 1. Expired Share Codes ---
        cleaned_codes = 0
        now = time.time()
        for code, data in list(self.share_codes.items()):
            if now > data.get("expires_at", 0):
                del self.share_codes[code]
                cleaned_codes += 1
        if cleaned_codes > 0:
            log.append(f"🧹 Removed {cleaned_codes} expired share codes.")

        # --- 2. Stale/Broken Profile Shares ---
        cleaned_shares = 0
        shares_changed = False
        for recipient_id_str, shares in list(self.profile_shares.items()):
            if recipient_id_str not in all_bot_member_ids:
                cleaned_shares += len(self.profile_shares.pop(recipient_id_str, []))
                shares_changed = True
                continue
            
            original_len = len(shares)
            valid_shares = []
            for share in shares:
                sharer_id_str = str(share.get("sharer_id"))
                profile_name = share.get("profile_name")
                if sharer_id_str in all_bot_member_ids:
                    sharer_data = self._get_user_data_entry(int(sharer_id_str))
                    if profile_name in sharer_data.get("profiles", {}):
                        valid_shares.append(share)
            
            if len(valid_shares) < original_len:
                self.profile_shares[recipient_id_str] = valid_shares
                cleaned_shares += original_len - len(valid_shares)
                shares_changed = True
        if shares_changed:
            self._save_profile_shares()
        if cleaned_shares > 0:
            log.append(f"🧹 Removed {cleaned_shares} stale or broken profile share requests.")

        # --- 3. Stale Key Submissions ---
        cleaned_keys = 0
        keys_changed = False
        for guild_id_str, submissions in list(self.key_submissions.items()):
            guild = self.bot.get_guild(int(guild_id_str))
            if not guild:
                cleaned_keys += len(self.key_submissions.pop(guild_id_str, []))
                keys_changed = True
                continue

            original_len = len(submissions)
            self.key_submissions[guild_id_str] = [s for s in submissions if guild.get_member(s.get("submitter_id"))]
            if len(self.key_submissions[guild_id_str]) < original_len:
                cleaned_keys += original_len - len(self.key_submissions[guild_id_str])
                keys_changed = True
        if keys_changed:
            self._save_key_submissions()
        if cleaned_keys > 0:
            log.append(f"🧹 Removed {cleaned_keys} stale API key submissions.")

        # --- 4. Orphaned Channel Webhooks ---
        cleaned_webhooks = 0
        for ch_id in list(self.channel_webhooks.keys()):
            if ch_id not in all_bot_channel_ids:
                del self.channel_webhooks[ch_id]
                cleaned_webhooks += 1
        if cleaned_webhooks > 0:
            self._save_channel_webhooks()
            log.append(f"🧹 Removed {cleaned_webhooks} orphaned channel webhooks.")

        # --- 5. Orphaned Server-Level Files ---
        cleaned_server_files = 0
        # Freewill
        if os.path.isdir(self.FREEWILL_SERVERS_DIR):
            for filename in os.listdir(self.FREEWILL_SERVERS_DIR):
                try:
                    server_id = int(filename.split('.')[0])
                    if server_id not in bot_guild_ids:
                        _delete_file_shard(os.path.join(self.FREEWILL_SERVERS_DIR, filename))
                        cleaned_server_files += 1
                except (ValueError, IndexError):
                    continue
        # Sessions
        sessions_servers_path = pathlib.Path(SESSIONS_SERVERS_DIR)
        if sessions_servers_path.is_dir():
            for server_dir in sessions_servers_path.iterdir():
                try:
                    if server_dir.is_dir() and int(server_dir.name) not in bot_guild_ids:
                        shutil.rmtree(server_dir)
                        cleaned_server_files += 1
                except ValueError:
                    continue
        if cleaned_server_files > 0:
            log.append(f"🧹 Removed {cleaned_server_files} orphaned server-level data directories/files.")

        # --- 6. Full User Data Cleanup (for users no longer sharing any server with the bot) ---
        cleaned_users_count = 0
        all_user_ids_in_shards = set()
        if os.path.isdir(self.PROFILES_DIR): all_user_ids_in_shards.update(f[:-len(".json.gz")] for f in os.listdir(self.PROFILES_DIR) if f.endswith(".json.gz"))
        if os.path.isdir(self.LTM_DIR): all_user_ids_in_shards.update(d for d in os.listdir(self.LTM_DIR) if os.path.isdir(os.path.join(self.LTM_DIR, d)))
        if os.path.isdir(self.TRAINING_DIR): all_user_ids_in_shards.update(d for d in os.listdir(self.TRAINING_DIR) if os.path.isdir(os.path.join(self.TRAINING_DIR, d)))
        all_user_ids = all_user_ids_in_shards | set(self.user_appearances.keys()) | set(self.profile_shares.keys())

        for user_id_str in list(all_user_ids):
            if user_id_str not in all_bot_member_ids:
                paths_to_delete = [
                    os.path.join(self.PROFILES_DIR, f"{user_id_str}.json.gz"),
                    os.path.join(self.LTM_DIR, user_id_str),
                    os.path.join(self.TRAINING_DIR, user_id_str),
                    os.path.join(SESSIONS_GLOBAL_DIR, user_id_str)
                ]
                for path in paths_to_delete:
                    if os.path.isfile(path): _delete_file_shard(path)
                    elif os.path.isdir(path): shutil.rmtree(path, ignore_errors=True)
                
                self.user_profiles.pop(user_id_str, None)
                self.user_appearances.pop(user_id_str, None)
                self.profile_shares.pop(user_id_str, None)
                cleaned_users_count += 1
        
        if cleaned_users_count > 0:
            self._save_user_appearances()
            self._save_profile_shares()
            log.append(f"🧹 Removed all data for {cleaned_users_count} users no longer sharing a server with the bot.")

        # --- 7. Detailed Per-User & Per-Server Integrity Check ---
        cleaned_ltm_shards, cleaned_training_shards, cleaned_appearances, cleaned_borrows = 0, 0, 0, 0
        cleaned_session_files, cleaned_child_bots = 0, 0
        
        profile_shard_files = [f for f in os.listdir(self.PROFILES_DIR) if f.endswith(".json.gz")] if os.path.isdir(self.PROFILES_DIR) else []
        for profile_shard_file in profile_shard_files:
            user_id_str = profile_shard_file[:-len(".json.gz")]
            user_data = self._get_user_data_entry(int(user_id_str))
            if not user_data: continue

            user_profiles = set(user_data.get("profiles", {}).keys())
            all_valid_profiles = user_profiles | set(user_data.get("borrowed_profiles", {}).keys())
            data_changed = False

            # LTM/Training/Appearance cleanup (vs personal profiles)
            for data_dir, counter_ref in [(self.LTM_DIR, "ltm"), (self.TRAINING_DIR, "train")]:
                user_data_dir = os.path.join(data_dir, user_id_str)
                if os.path.isdir(user_data_dir):
                    for data_file in os.listdir(user_data_dir):
                        profile_name = data_file[:-len(".json.gz")]
                        if profile_name not in user_profiles:
                            _delete_file_shard(os.path.join(user_data_dir, data_file))
                            if counter_ref == "ltm": cleaned_ltm_shards += 1
                            else: cleaned_training_shards += 1
            
            if user_id_str in self.user_appearances:
                for appearance_name in list(self.user_appearances[user_id_str].keys()):
                    if appearance_name not in user_profiles:
                        del self.user_appearances[user_id_str][appearance_name]
                        cleaned_appearances += 1
            
            # Borrowed profile cleanup
            for borrowed_name, borrow_data in list(user_data.get("borrowed_profiles", {}).items()):
                owner_id = borrow_data.get("original_owner_id")
                original_name = borrow_data.get("original_profile_name")
                if owner_id and original_name:
                    owner_data = self._get_user_data_entry(int(owner_id))
                    if original_name not in owner_data.get("profiles", {}):
                        del user_data["borrowed_profiles"][borrowed_name]
                        cleaned_borrows += 1
                        data_changed = True
            
            # Session file cleanup (vs all valid profiles)
            global_session_dir = pathlib.Path(SESSIONS_GLOBAL_DIR) / user_id_str
            if global_session_dir.is_dir():
                for session_file in global_session_dir.iterdir():
                    if session_file.name.endswith(".json.gz"):
                        profile_name = session_file.name[:-len(".json.gz")]
                        if profile_name not in all_valid_profiles:
                            _delete_file_shard(str(session_file))
                            cleaned_session_files += 1
            
            if data_changed: self._save_user_data_entry(int(user_id_str), user_data)

        # Child Bot cleanup
        child_bots_changed = False
        for owner_id_str, user_shard in list(self._get_all_child_bot_shards().items()):
            owner_data = self._get_user_data_entry(int(owner_id_str))
            owner_profiles = set(owner_data.get("profiles", {}).keys())
            for bot_id, bot_config in list(user_shard.items()):
                if bot_config.get("profile_name") not in owner_profiles:
                    del user_shard[bot_id]
                    cleaned_child_bots += 1
                    child_bots_changed = True
            if child_bots_changed: self._save_user_child_bot_shard(int(owner_id_str), user_shard)
        if child_bots_changed: self._load_child_bots()

        if cleaned_ltm_shards > 0: log.append(f"🧹 Removed {cleaned_ltm_shards} orphaned LTM data files.")
        if cleaned_training_shards > 0: log.append(f"🧹 Removed {cleaned_training_shards} orphaned Training data files.")
        if cleaned_appearances > 0: log.append(f"🧹 Removed {cleaned_appearances} orphaned appearances.")
        if cleaned_borrows > 0: log.append(f"🧹 Removed {cleaned_borrows} broken borrowed profiles.")
        if cleaned_session_files > 0: log.append(f"🧹 Removed {cleaned_session_files} orphaned session files for deleted profiles.")
        if cleaned_child_bots > 0: log.append(f"🧹 Removed {cleaned_child_bots} orphaned child bot configurations.")

        # --- 8. Channel & User Session Directory Cleanup ---
        cleaned_channel_dirs, cleaned_user_session_dirs = 0, 0
        sessions_servers_path = pathlib.Path(SESSIONS_SERVERS_DIR)
        if sessions_servers_path.is_dir():
            for server_dir in list(sessions_servers_path.iterdir()):
                if not server_dir.is_dir(): continue
                try:
                    server_id_int = int(server_dir.name)
                    guild = self.bot.get_guild(server_id_int)
                    
                    for channel_dir in list(server_dir.iterdir()):
                        if not channel_dir.is_dir(): continue
                        try:
                            channel_id = int(channel_dir.name)
                            # Remove if channel is gone or if directory is empty of actual data
                            is_deleted = guild and channel_id not in {c.id for c in guild.channels}
                            has_files = any(f.is_file() for f in channel_dir.rglob('*') if not f.name.startswith('.'))
                            
                            if is_deleted or not has_files:
                                shutil.rmtree(channel_dir, ignore_errors=True)
                                cleaned_channel_dirs += 1
                                continue
                            
                            # Deep check for orphaned user subdirectories in single-profile sessions
                            single_user_path = channel_dir / "single"
                            if single_user_path.is_dir():
                                current_member_ids = {str(m.id) for m in guild.members} if guild else set()
                                for user_dir in list(single_user_path.iterdir()):
                                    if not user_dir.is_dir(): continue
                                    is_orphaned = guild and user_dir.name not in current_member_ids
                                    is_empty = not any(user_dir.iterdir())
                                    if is_orphaned or is_empty:
                                        shutil.rmtree(user_dir, ignore_errors=True)
                                        cleaned_user_session_dirs += 1
                        except (ValueError, OSError): continue
                    
                    # Remove server dir if empty
                    if not any(server_dir.iterdir()):
                        server_dir.rmdir()
                except (ValueError, OSError): continue
        
        if cleaned_channel_dirs > 0: log.append(f"🧹 Removed {cleaned_channel_dirs} session directories for deleted channels.")
        if cleaned_user_session_dirs > 0: log.append(f"🧹 Removed {cleaned_user_session_dirs} user session directories for users no longer in the server.")

        # --- 9. Final Config File Cleanup ---
        cleaned_channel_settings = 0
        for ch_id in list(self.active_channels):
            if ch_id not in all_bot_channel_ids: self.active_channels.discard(ch_id); cleaned_channel_settings += 1
        for ch_id in list(self.channel_scoped_profiles.keys()):
            if ch_id not in all_bot_channel_ids: del self.channel_scoped_profiles[ch_id]; cleaned_channel_settings += 1
        for ch_id in list(self.multi_profile_channels.keys()):
            if ch_id not in all_bot_channel_ids: del self.multi_profile_channels[ch_id]; cleaned_channel_settings += 1
        
        if cleaned_channel_settings > 0:
            self._save_channel_settings()
            self._save_multi_profile_sessions()
            log.append(f"🧹 Removed settings for {cleaned_channel_settings} deleted channels from config files.")

        log.append("\nCleanup complete.")
        print("\n".join(log).replace("**", ""))

    async def handle_child_bot_event(self, event_data: Dict):
        event_type = event_data.get("event_type")
        bot_id = event_data.get("bot_id")
        
        if event_type == "message_received":
            message_payload = event_data.get("message", {})
            channel_id = message_payload.get("channel_id")
            guild_id = message_payload.get("guild_id")
            
            # Freewill Check: If this channel is configured for Freewill, ignore child bot mentions.
            if guild_id:
                fw_config = self.freewill_config.get(str(guild_id), {})
                if channel_id in fw_config.get("living_channel_ids", []) or channel_id in fw_config.get("lurking_channel_ids", []):
                    return

            # Strict Session Check: If a Freewill session is active, ignore child bot mentions.
            session = self.multi_profile_channels.get(channel_id)
            if session and session.get("type") == "freewill":
                return

            message_id = message_payload.get("id")

            if message_id:
                self.processed_child_messages[message_id] = True

            bot_config = self.child_bots.get(bot_id)
            if not bot_config: return
            
            # [NEW] Premium Gate for Runtime Execution
            # If owner is not premium, ignore the event silently to save resources.
            if not self.is_user_premium(bot_config['owner_id']):
                return

            owner_id = bot_config['owner_id']
            profile_name = bot_config['profile_name']
            
            session = self.multi_profile_channels.get(channel_id)
            
            ephemeral_participant = {
                "owner_id": owner_id,
                "profile_name": profile_name,
                "method": "child_bot",
                "bot_id": bot_id,
                "ephemeral": True # Mark as a guest star
            }

            if not session:
                # Create a new session on the fly
                chat_sessions = { (p['owner_id'], p['profile_name']): None for p in [ephemeral_participant] }
                session = {
                    "type": "multi", "chat_sessions": chat_sessions, "unified_log": [], "is_hydrated": False,
                    "last_bot_message_id": None, "owner_id": message_payload.get("author_id"), "is_running": False,
                    "auto_continue": False, "auto_delay": None, "timer_handle": None,
                    "task_queue": asyncio.Queue(), "worker_task": None, "turns_since_last_ltm": 0,
                    "session_prompt": None, "session_mode": "sequential", "profiles": [ephemeral_participant]
                }
                self.multi_profile_channels[channel_id] = session

            # Put the trigger into the queue
            trigger = ('child_mention', message_payload, ephemeral_participant)
            await session['task_queue'].put(trigger)
            
            # Start the worker if it's not running
            if not session.get('is_running'):
                session['worker_task'] = self.bot.loop.create_task(self._multi_profile_worker(channel_id))

    async def handle_child_bot_image_request(self, event_data: Dict):
        bot_id = event_data.get("bot_id")
        message_data = event_data.get("message", {})
        channel_id = message_data.get("channel_id")

        if channel_id in self.multi_profile_channels:
            return

        async def send_notification_to_child(content: str):
            await self.manager_queue.put({
                "action": "send_to_child", "bot_id": bot_id,
                "payload": {"action": "send_message", "channel_id": channel_id, "content": f"(Notice for {message_data['author_name']}): {content}"}
            })

        try:
            if self.image_request_queue.full():
                await send_notification_to_child("The image generation backlog is currently full. Please try again in a moment.")
                return

            placeholder_sent = False
            if self.image_gen_semaphore.locked():
                qsize = self.image_request_queue.qsize()
                await send_notification_to_child(f"Your image generation request has been queued. You are #{qsize + 1} in line.")
            else:
                await self.manager_queue.put({"action": "send_to_child", "bot_id": bot_id, "payload": {"action": "start_typing", "channel_id": channel_id}})
                placeholder_sent = True

            bot_config = self.child_bots.get(bot_id)
            if not bot_config: return

            # [NEW] Premium Gate for Child Bot Images
            if not self.is_user_premium(bot_config['owner_id']):
                # If we already sent a "Queue" DM notification above, we might want to tell them why it failed.
                # However, for simplicity and hard-gating, we just return.
                return

            guild_id = message_data.get("guild_id")

            image_prefixes = ("!image", "!imagine")
            used_prefix = next((p for p in image_prefixes if message_data.get("content", "").lower().startswith(p)), "!image")
            prompt_text = message_data.get("content", "")[len(used_prefix):].strip()
            if not prompt_text: return
            
            owner_id, profile_name = bot_config['owner_id'], bot_config['profile_name']
            user_data = self._get_user_data_entry(owner_id)
            is_borrowed = profile_name in user_data.get("borrowed_profiles", {})
            profile_data = user_data.get("borrowed_profiles" if is_borrowed else "profiles", {}).get(profile_name, {})

            if not profile_data.get("image_generation_enabled", True):
                return

            safety_level_str = profile_data.get("safety_level", "low")
            
            safety_map = { "unrestricted": HarmBlockThreshold.BLOCK_NONE, "low": HarmBlockThreshold.BLOCK_ONLY_HIGH, "medium": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, "high": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE }
            threshold = safety_map.get(safety_level_str, HarmBlockThreshold.BLOCK_ONLY_HIGH)
            dynamic_safety_settings = { cat: threshold for cat in get_args(HarmCategory) }
            
            # Get appearance text
            source_owner_id = owner_id
            source_profile_name = profile_name
            if is_borrowed:
                borrowed_data = user_data["borrowed_profiles"][profile_name]
                source_owner_id = int(borrowed_data["original_owner_id"])
                source_profile_name = borrowed_data["original_profile_name"]
            
            source_owner_data = self._get_user_data_entry(source_owner_id)
            source_profile_data = source_owner_data.get("profiles", {}).get(source_profile_name, {})
            persona = source_profile_data.get("persona", {})
            appearance_lines_encrypted = persona.get("appearance", [])
            appearance_text = "\n".join([self._decrypt_data(line) for line in appearance_lines_encrypted])

            bot_user = self.bot.get_user(int(bot_id)); bot_display_name = bot_user.name if bot_user else profile_name

            final_prompt_text = prompt_text
            if appearance_text.strip():
                prompt_lower = prompt_text.lower()
                second_person_pronouns = ["you", "your", "yourself", "u", "ur"]
                # Check for pronouns or the profile's names
                if any(pronoun in prompt_lower.split() for pronoun in second_person_pronouns) or \
                   bot_display_name.lower() in prompt_lower or \
                   profile_name.lower() in prompt_lower:
                    final_prompt_text = f"Your appearance:\n{appearance_text.strip()}\n\nUser's prompt:\n{prompt_text}"

            system_instruction = self._get_image_gen_system_instruction(owner_id, profile_name)

            reference_image_urls = []
            replied_to_data = message_data.get("replied_to")
            if replied_to_data and replied_to_data.get("attachment_url"):
                reference_image_urls.append(replied_to_data["attachment_url"])

            attachments_data = message_data.get("attachments", [])
            if len(reference_image_urls) < 2 and attachments_data:
                for attachment in attachments_data:
                    if attachment.get("url"):
                        reference_image_urls.append(attachment.get("url"))
                        if len(reference_image_urls) >= 2: break

            grounding_sources = []
            grounding_mode = profile_data.get("grounding_mode", "off")
            if isinstance(grounding_mode, bool): grounding_mode = "on" if grounding_mode else "off"

            if grounding_mode in ["on", "on+"]:
                session_key = (channel_id, owner_id, profile_name)
                chat = self.chat_sessions.get(session_key)
                history_for_grounding = chat.history if chat else []
                
                mapping_key = self._get_mapping_key_for_session(session_key, 'single')
                grounding_result = await self._get_hybrid_grounding_context(prompt_text, guild_id, history_for_grounding, mapping_key, is_for_image=True)
                if grounding_result:
                    grounding_context, sources, _ = grounding_result
                    if grounding_context:
                        final_prompt_text = f"{prompt_text}\n\nUse this information to help generate the image:\n{grounding_context}"
                        grounding_sources = sources

            request_data = {
                "is_child_bot": True, "bot_id": bot_id, "author_id": message_data['author_id'],
                "channel_id": channel_id, "guild_id": guild_id, "original_message_id": message_data['id'], 
                "original_content": message_data['content'], "prompt_text": final_prompt_text, 
                "effective_profile_owner_id": owner_id, "effective_profile_name": profile_name,
                "bot_display_name": bot_display_name, "safety_settings": dynamic_safety_settings,
                "system_instruction": system_instruction, "reference_image_urls": reference_image_urls, "placeholder_message": None, 
                "grounding_sources": grounding_sources, "grounding_mode": grounding_mode
            }
            
            # [NEW] Priority Logic
            is_premium = self.is_user_premium(owner_id)
            priority = 10 if is_premium else 20
            
            await self.image_request_queue.put((priority, request_data))
        except Exception as e:
            print(f"Error dispatching child bot image request for bot {bot_id}: {e}"); traceback.print_exc()

    async def handle_child_bot_toggle(self, event_data: Dict):
        bot_id = str(event_data.get("bot_id")) # Ensure string
        channel_id = event_data.get("channel_id")
        
        bot_config = self.child_bots.get(bot_id)
        if not bot_config: return

        session = self.multi_profile_channels.get(channel_id)
        
        # Block toggling in Freewill sessions
        if session and session.get("type") == "freewill":
            channel = self.bot.get_channel(channel_id)
            if channel:
                await channel.send("You cannot use this command in Freewill sessions. You can add child bots through the Freewill UI.")
            return

        action_taken = None

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
                "session_mode": "sequential"
            }
            self.multi_profile_channels[channel_id] = session
            action_taken = "add"
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
                
                if not session['profiles']:
                    self.multi_profile_channels.pop(channel_id, None)
            else:
                # Add it
                participant = {
                    "owner_id": bot_config['owner_id'], "profile_name": bot_config['profile_name'],
                    "method": "child_bot", "bot_id": bot_id, "ephemeral": False
                }
                session['profiles'].append(participant)
                # Also create the placeholder for the chat session
                session['chat_sessions'][(participant['owner_id'], participant['profile_name'])] = None
                action_taken = "add"
        
        self._save_multi_profile_sessions()

        if action_taken:
            ipc_action = "session_update_add" if action_taken == "add" else "session_update_remove"
            await self.manager_queue.put({
                "action": "send_to_child", "bot_id": bot_id,
                "payload": {"action": ipc_action, "channel_id": channel_id}
            })
            if action_taken == "remove":
                await self.manager_queue.put({
                    "action": "send_to_child", "bot_id": bot_id,
                    "payload": {"action": "stop_typing", "channel_id": channel_id}
                })

    async def handle_child_bot_refresh(self, command_data: Dict):
        bot_id = command_data.get("bot_id")
        channel_id = command_data.get("channel_id")
        if not bot_id or not channel_id:
            return

        bot_config = self.child_bots.get(bot_id)
        if not bot_config:
            return

        owner_id = bot_config['owner_id']
        profile_name = bot_config['profile_name']
        
        # 1. Cancel the running worker task for this specific child bot session
        worker_key = (channel_id, bot_id)
        if worker_key in self.child_bot_single_sessions:
            worker_data = self.child_bot_single_sessions.pop(worker_key)
            if worker_data and worker_data.get('task'):
                self._safe_cancel_task(worker_data['task'])

        # 2. Clear all caches and delete the on-disk session file
        session_key = (channel_id, owner_id, profile_name)
        
        mapping_key = self._get_mapping_key_for_session(session_key, 'single')
        if mapping_key in self.mapping_caches:
            self.mapping_caches[mapping_key].pop('grounding_checkpoint', None)
        
        self.chat_sessions.pop(session_key, None)
        self.channel_models.pop(session_key, None)
        self.channel_model_last_profile_key.pop(session_key, None)
        self.session_last_accessed.pop(session_key, None)
        self._delete_session_from_disk(session_key, 'single')
        self.ltm_recall_history.pop(session_key, None)

        # 3. Reset the LTM creation counter
        ltm_counter_key = (owner_id, profile_name, "guild")
        self.message_counters_for_ltm.pop(ltm_counter_key, None)

    async def handle_child_bot_confirmation(self, event_data: Dict):
        correlation_id = event_data.get("correlation_id")
        if not correlation_id or correlation_id not in self.pending_child_confirmations:
            return

        confirmation_data = self.pending_child_confirmations.pop(correlation_id)
        message_ids = event_data.get("message_ids", [])
        
        try:
            if confirmation_data.get("type") == "single_profile":
                bot_id = confirmation_data.get("bot_id")
                bot_config = self.child_bots.get(bot_id, {})
                owner_id = bot_config.get("owner_id")
                profile_name = bot_config.get("profile_name")
                
                if owner_id is not None and profile_name:
                    channel_id = confirmation_data.get("channel_id") or event_data.get("channel_id")
                    if not channel_id: return

                    session_key = (channel_id, owner_id, profile_name)
                    chat = self.chat_sessions.get(session_key)
                    if chat:
                        turn_id = confirmation_data.get("turn_id")
                        if turn_id is None: return

                        turn_data = (session_key, 'single', turn_id)
                        mapping_key = self._get_mapping_key_for_session(session_key, 'single')
                        if mapping_key not in self.mapping_caches:
                            self.mapping_caches[mapping_key] = self._load_mapping_from_disk(mapping_key)

                        for msg_id in message_ids:
                            self.message_to_history_turn[msg_id] = turn_data
                            self.mapping_caches[mapping_key][str(msg_id)] = turn_data
            
            elif confirmation_data.get("type") == "multi_profile":
                participant = confirmation_data.get("participant")
                channel_id = confirmation_data.get("channel_id")
                turn_id = confirmation_data.get("turn_id")

                if not all([participant, channel_id, turn_id]):
                    return

                session = self.multi_profile_channels.get(channel_id)
                if session and message_ids:
                    # History is already updated in the worker loop or speak_as logic.
                    # We only need to map the Discord message IDs here for reactions/purge.
                    session['last_bot_message_id'] = message_ids[-1]
                    session_type = session.get("type", "multi")
                    turn_data = (channel_id, session_type, turn_id)
                    
                    mapping_key = (session_type, channel_id)
                    if mapping_key not in self.mapping_caches:
                        self.mapping_caches[mapping_key] = self._load_mapping_from_disk(mapping_key)

                    for msg_id in message_ids:
                        self.message_to_history_turn[msg_id] = turn_data
                        self.mapping_caches[mapping_key][str(msg_id)] = turn_data
        
        except Exception as e:
            print(f"Error during child bot confirmation ({correlation_id}): {e}")
            traceback.print_exc()
        
        finally:
            if "event" in confirmation_data and confirmation_data["event"]:
                confirmation_data["event"].set()

    async def _is_profile_content_safe(self, profile_name: str, display_name: str, avatar_url: Optional[str], guild_id: int) -> Tuple[bool, str]:
        # Create a safer prompt that describes the content instead of including it directly.
        # The actual user content will be in the 'parts' of the request.
        prompt_text = (
            "You are an expert AI content moderator for a social platform like Discord. Your task is to analyze user-submitted content (text and an optional image for an avatar) and determine if it violates policy. Your primary goal is to distinguish between what is merely 'suggestive' (often SAFE) and what is 'explicit' or 'graphic' (UNSAFE).\n\n"
            "**Policy Violations (UNSAFE):**\n"
            "- **Sexually Explicit:** Graphic depictions of sexual acts, genitalia, or pornographic material.\n"
            "- **Extreme Violence:** Real gore, graphic depictions of severe injury or death.\n"
            "- **Hate Speech:** Content that promotes violence or hatred against individuals or groups based on protected characteristics.\n\n"
            "**Acceptable Content (SAFE):**\n"
            "- **Swimwear/Beachwear:** Photos or art of people in bikinis, swimsuits, etc., are SAFE.\n"
            "- **Artistic Nudity:** Non-pornographic artistic depictions of nudity are generally SAFE.\n"
            "- **Suggestive Poses/Themes:** Common anime/fantasy art styles that may be suggestive but are not explicit are SAFE.\n"
            "- **Cleavage/Musculature:** Depictions of cleavage or muscular bodies are SAFE.\n\n"
            "Analyze all provided content (profile name, display name, and image) together. Respond with ONLY a single word: SAFE or UNSAFE."
        )
        
        prompt_contents = [prompt_text, f"Profile Name: {profile_name}", f"Display Name: {display_name}"]
        
        if avatar_url:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(avatar_url, follow_redirects=True, timeout=10.0)
                    response.raise_for_status()
                    image_data = await response.aread()
                    
                    # FIXED: Direct pass-through. No PIL/PNG conversion.
                    content_type = response.headers.get("Content-Type", "image/png")
                    prompt_contents.append({"mime_type": content_type, "data": image_data})
            except httpx.RequestError:
                return False, "Could not download the avatar image from the provided URL."
            except Exception:
                return False, "An error occurred while processing the avatar URL."

        try:
            api_key = self._get_api_key_for_guild(guild_id)
            if not api_key:
                return False, "Server API key is not available for moderation."
            genai.configure(api_key=api_key)
            
            model_name = 'gemini-flash-lite-latest'
            status = "api_error"
            try:
                # [FIXED] Move rules to system_instruction to prevent jailbreaking
                model = genai.GenerativeModel(
                    model_name=model_name, 
                    system_instruction=prompt_text,
                    safety_settings=DEFAULT_SAFETY_SETTINGS
                )
                # The prompt now only contains the user-submitted content to evaluate
                eval_payload = [
                    f"<target_content>\nProfile Name: {profile_name}\nDisplay Name: {display_name}\n</target_content>"
                ]
                if avatar_url:
                    # Logic for image fetching remains the same
                    pass 

                # Pass the cleaned payload instead of the original prompt_contents
                response = await model.generate_content_async(eval_payload)
                status = "blocked_by_safety" if not response.candidates else "success"
            finally:
                self._log_api_call(user_id=0, guild_id=guild_id, context="moderation_check", model_used=model_name, status=status)

            if not response.candidates:
                reason = "Unknown"
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    reason = response.prompt_feedback.block_reason.name
                print(f"Auto-moderation check failed because the prompt was blocked. Reason: {reason}")
                return False, f"Content was flagged as unsafe by the AI moderator ({reason})."

            result = ""
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    result = "".join(p.text for p in candidate.content.parts if hasattr(p, 'text')).strip().upper()

            if result == "SAFE":
                return True, "Content is safe."
            else:
                return False, "Content was flagged as unsafe by the AI moderator."
        except Exception as e:
            print(f"Auto-moderation check failed: {e}")
            traceback.print_exc()
            return False, "An error occurred during the moderation check."
        
    async def _run_critic(self, history: list, char_name: str, guild_id: int) -> Optional[str]:
        """Uses a cheap model to find verbatim linguistic loops in recent history."""
        recent_turns = []
        count = 0
        for turn in reversed(history):
            if turn.role == 'model':
                parts = [p.text for p in turn.parts if hasattr(p, 'text')]
                if parts:
                    recent_turns.append(" ".join(parts))
                    count += 1
            if count >= 3: break
        
        if len(recent_turns) < 3: return None 
        
        transcript = "\n---\n".join(reversed(recent_turns))
        
        system_instruction = (
            f"You are a linguistic pattern analyzer for the character '{char_name}'.\n"
            "Your task is to detect repetitive structural patterns across the provided transcript.\n\n"
            "CRITERIA FOR FLAGGING:\n"
            "Identify any phrase, name, or sentence structure that appears in **2 or more messages** within this transcript.\n"
            "Specifically target:\n"
            "1. **Repeated Openers:** Starting messages with the same word, interjection, or name (e.g., 'Well,', 'Look,', 'User,').\n"
            "2. **Repeated Closers:** Ending messages with the same tag or name.\n"
            "3. **Recycled Phrases:** Reusing specific clauses or sentences verbatim (e.g., 'You think you're so clever, don't you?').\n\n"
            "OUTPUT RULES:\n"
            "- If the pattern is a simple greeting or established character catchphrase (e.g., 'Meow'), IGNORE it.\n"
            "- If no significant repetition is found, respond with ONLY 'PASS'.\n"
            "- If repetition is found, provide a negative constraint (e.g., 'Do not start sentences with *Well*', 'Do not use the phrase *...*')."
        )
        
        api_key = self._get_api_key_for_guild(guild_id)
        if not api_key: return None
        
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash', system_instruction=system_instruction)
            critic_cfg = genai.types.GenerationConfig(temperature=0.05, top_p=0.9)
            resp = await model.generate_content_async(f"Transcript:\n{transcript}", generation_config=critic_cfg)
            if resp.text:
                if "PASS" not in resp.text.upper():
                    return resp.text.strip()
        except Exception as e:
            print(f"Critic error: {e}")
        return None
    
    async def _execute_freewill_turn(self, channel: discord.TextChannel, participant: Dict, prompt_content: str, author_display_name: str, triggering_user_id: Optional[int] = None, anchor_message: Optional[discord.Message] = None) -> List[discord.Message]:
        profile_owner_id = participant['owner_id']
        profile_name = participant['profile_name']
        method = participant['method']
        bot_id = participant.get('bot_id')

        if not self._check_unrestricted_safety_policy(profile_owner_id, profile_name, channel):
            return [] # Fails silently in freewill context

        placeholder_message = None
        sent_messages = []
        try:
            model, _, temp, top_p, top_k, warning_message, fallback_model_name = await self._get_or_create_model_for_channel(
                channel.id, triggering_user_id or self.bot.user.id, channel.guild.id,
                profile_owner_override=profile_owner_id, profile_name_override=profile_name,
                prompt_content=prompt_content
            )
            
            # [UPDATED] If model fails, proceed to send the error message via profile instead of silent return
            status = "api_error"
            response = None
            fallback_used = False
            blocked_reason_override = None

            history_lines = await self._build_freewill_history(channel, anchor_message)
            from google.generativeai.types import content_types
            history_for_turn = [content_types.to_content({'role': 'user', 'parts': [line]}) for line in history_lines]
            
            # Note: We need a dummy chat session if model is None to proceed with formatting
            dummy_model = genai.GenerativeModel('gemini-flash-latest')
            chat = (model or dummy_model).start_chat(history=history_for_turn)

            if len(chat.history) > self.max_history_items * 2:
                chat.history = chat.history[-(self.max_history_items * 2):]

            if method == 'webhook':
                thinking_messages = await self._send_channel_message(
                    channel, f"{PLACEHOLDER_EMOJI}",
                    profile_owner_id_for_appearance=profile_owner_id, profile_name_for_appearance=profile_name
                )
                placeholder_message = thinking_messages[0] if thinking_messages else None
            else:
                await self.manager_queue.put({
                    "action": "send_to_child", "bot_id": bot_id,
                    "payload": {"action": "start_typing", "channel_id": channel.id}
                })

            if model:
                ltm_recall_text = await self._get_relevant_ltm_for_prompt(profile_owner_id, profile_name, prompt_content, author_display_name, channel.guild.id, triggering_user_id or self.bot.user.id)
                
                bot_display_name = profile_name
                appearance_data = self.user_appearances.get(str(profile_owner_id), {}).get(profile_name, {})
                if appearance_data.get("custom_display_name"):
                    bot_display_name = appearance_data["custom_display_name"]

                history_for_prompt = "".join(
                    self._format_history_entry(
                        "A user" if turn.role == 'user' else bot_display_name,
                        datetime.datetime.now(datetime.timezone.utc), # Approximation
                        turn.parts[0].text
                    ) for turn in chat.history
                )
                
                user_message_formatted = self._format_history_entry(author_display_name, datetime.datetime.now(datetime.timezone.utc), prompt_content)
                
                final_prompt_parts = [history_for_prompt]
                
                # Conditional Metadata Injection
                if ltm_recall_text: final_prompt_parts.append(ltm_recall_text)
                
                p_udata = self._get_user_data_entry(profile_owner_id)
                p_is_b = profile_name in p_udata.get("borrowed_profiles", {})
                p_settings = p_udata.get("borrowed_profiles" if p_is_b else "profiles", {}).get(profile_name, {})
                
                if p_settings.get("url_fetching_enabled", True):
                    u_t, _ = await self._process_urls_in_content(prompt_content, channel.guild.id, {"url_fetching_enabled": True})
                    if u_t: final_prompt_parts.append(f"<document_context>\n" + "\n".join(u_t) + "\n</document_context>")

                final_prompt_parts.append(user_message_formatted)
                
                from google.generativeai.types import content_types
                gen_config = genai.types.GenerationConfig(temperature=temp, top_p=top_p, top_k=top_k)
                
                try:
                    # Pass shallow list copy
                    response = await model.generate_content_async(list(final_prompt_parts), generation_config=gen_config)
                    if not response or not response.candidates:
                        raise ValueError("Response blocked or empty")
                    status = "success"
                except (api_exceptions.ResourceExhausted, api_exceptions.InternalServerError, api_exceptions.ServiceUnavailable, api_exceptions.Aborted, Exception) as e:
                    is_rate_limit = isinstance(e, api_exceptions.ResourceExhausted) or "429" in str(e) or "rate limit" in str(e).lower()
                    if is_rate_limit: blocked_reason_override = "Rate Limit"

                    is_google_error = isinstance(e, (api_exceptions.ResourceExhausted, api_exceptions.InternalServerError, api_exceptions.ServiceUnavailable, api_exceptions.Aborted))
                    is_openrouter_error = isinstance(e, Exception) and ("OpenRouter" in str(e) or "402" in str(e) or "Response blocked" in str(e))

                    if (is_google_error or is_openrouter_error) and fallback_model_name:
                        try:
                            # Re-construct instructions for the fallback model
                            sys_instr, _, _, _, _, _, _, _ = self._construct_system_instructions(
                                profile_owner_id, profile_name, channel.id, is_multi_profile=False
                            )
                            
                            # Re-calculate safety settings locally
                            p_settings = self._get_user_data_entry(profile_owner_id).get("profiles", {}).get(profile_name, {})
                            if not p_settings: p_settings = self._get_user_data_entry(profile_owner_id).get("borrowed_profiles", {}).get(profile_name, {})
                            
                            safe_lvl = p_settings.get("safety_level", "low")
                            safe_thresh = HarmBlockThreshold.BLOCK_ONLY_HIGH
                            if safe_lvl == "medium": safe_thresh = HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                            elif safe_lvl == "high": safe_thresh = HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
                            
                            dyn_safe = { cat: safe_thresh for cat in get_args(HarmCategory) }

                            fb_name = fallback_model_name
                            fb_is_or = False
                            
                            if fb_name.upper().startswith("GOOGLE/"):
                                fb_name = fb_name[7:]
                                fb_is_or = False
                            elif fb_name.upper().startswith("OPENROUTER/"):
                                fb_name = fb_name[11:]
                                fb_is_or = True
                            elif "/" in fb_name:
                                fb_is_or = True
                            
                            if fb_is_or:
                                or_key = self._get_api_key_for_guild(channel.guild.id, provider="openrouter")
                                if or_key:
                                    fallback_instance = OpenRouterModel(fb_name, api_key=or_key, system_instruction=sys_instr)
                                else:
                                    raise ValueError("No OR key for fallback")
                            else:
                                # [FIX] Resolve API key for the current guild context
                                guild_api_key = self._get_api_key_for_guild(channel.guild.id)
                                if not guild_api_key:
                                    raise ValueError("No Google key for fallback")
                                
                                # [NEW] Use GoogleGenAIModel wrapper for SDK v2
                                fallback_instance = GoogleGenAIModel(api_key=guild_api_key, model_name=fb_name, system_instruction=sys_instr, safety_settings=dyn_safe)

                            # Note: GoogleGenAIModel wrapper handles the list(final_prompt_parts) conversion
                            response = await fallback_instance.generate_content_async(list(final_prompt_parts), generation_config=gen_config)
                            if not response or not response.candidates:
                                pass # Fallback blocked
                            else:
                                # Fallback succeeded
                                fallback_used = True
                                self._log_api_call(user_id=triggering_user_id or 0, guild_id=channel.guild.id, context="freewill_fallback", model_used=fb_name, status="success")

                        except Exception as retry_e:
                            print(f"Freewill fallback retry failed: {retry_e}")
                            status = "api_error"
                    else:
                        status = "api_error"
                except (api_exceptions.PermissionDenied) as e:
                    status = "api_error"
                finally:
                    # Always log the primary model's final status
                    self._log_api_call(user_id=triggering_user_id or 0, guild_id=channel.guild.id, context="freewill", model_used=model.model_name if hasattr(model, 'model_name') else "unknown", status=status)
            else:
                blocked_reason_override = warning_message or "API Error: Model failed to load."

            response_text = ""
            if response and response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    response_text = getattr(response, 'text', "").strip()

            if not response_text.strip():
                reason = blocked_reason_override or "Unknown Error"
                if response and response.prompt_feedback and response.prompt_feedback.block_reason:
                     reason = response.prompt_feedback.block_reason.name.replace('_', ' ').title()
                
                if reason == "Rate Limit":
                    response_text = "My response was blocked due to: **API Rate Limit**. Please try again later or use paid tier API."
                else:
                    response_text = f"My response was blocked due to: **{reason}**."

            response_text = self._deduplicate_response(response_text)

            from google.generativeai.types import content_types
            user_content_obj = content_types.to_content({'role': 'user', 'parts': [prompt_content]})
            model_content_obj = content_types.to_content({'role': 'model', 'parts': [response_text]})
            chat.history.extend([user_content_obj, model_content_obj])

            text_to_send = response_text
            
            # Check profile setting for fallback indicator
            p_settings = self._get_user_data_entry(profile_owner_id).get("profiles", {}).get(profile_name, {})
            if not p_settings: p_settings = self._get_user_data_entry(profile_owner_id).get("borrowed_profiles", {}).get(profile_name, {})
            
            if fallback_used and p_settings.get("show_fallback_indicator", True):
                text_to_send += "\n\n-# Fallback Model Used"

            if method == 'child_bot' and bot_id:
                correlation_id = str(uuid.uuid4())
                self.pending_child_confirmations[correlation_id] = {
                    "type": "single_profile",
                    "user_turn": user_content_obj,
                    "model_turn": model_content_obj,
                    "bot_id": bot_id,
                    "channel_id": channel.id
                }
                await self.manager_queue.put({
                    "action": "send_to_child", "bot_id": bot_id,
                    "payload": {
                        "action": "send_message", "channel_id": channel.id, "content": text_to_send,
                        "realistic_typing": False, "correlation_id": correlation_id
                    }
                })
                sent_messages = []
            else:
                sent_messages = await self._send_channel_message(
                    channel, text_to_send, target_message_to_edit=placeholder_message,
                    profile_owner_id_for_appearance=profile_owner_id, profile_name_for_appearance=profile_name
                )

            if sent_messages and not response_text.startswith("An error has occurred"):
                # Standardize the turn_info structure to match single-profile and child bot replies
                turn_info = (('freewill', channel.id), user_content_obj, model_content_obj, profile_owner_id, profile_name)
                for msg in sent_messages:
                    self.message_to_history_turn[msg.id] = turn_info
            
            await self._maybe_create_ltm(
                sent_messages[0] if sent_messages else channel, 
                author_display_name, chat.history, profile_owner_id, profile_name, 
                {"temperature": temp, "top_p": top_p, "top_k": top_k}
            )

            if method == 'child_bot':
                return [True]
            return sent_messages

        except Exception as e:
            print(f"Error during freewill turn for {profile_name}: {e}")
            if placeholder_message: await placeholder_message.delete()
        return []
    
    def _build_freewill_participant_dict(self, owner_id: int, profile_name: str, channel: discord.TextChannel) -> Optional[Dict]:
        server_participation = self.freewill_participation.get(str(channel.guild.id), {})
        channel_participants = server_participation.get(str(channel.id), {})
        user_profiles = channel_participants.get(str(owner_id), {})
        profile_settings = user_profiles.get(profile_name)

        if not profile_settings:
            return None

        method = profile_settings.get("method", "webhook")
        
        participant_dict = {
            "owner_id": owner_id,
            "profile_name": profile_name,
            "method": method,
            "bot_id": None
        }

        if method == "child_bot":
            bot_id_found = next((bot_id for bot_id, data in self.child_bots.items() if data.get("owner_id") == owner_id and data.get("profile_name") == profile_name), None)
            if bot_id_found and channel.guild.get_member(int(bot_id_found)):
                participant_dict["bot_id"] = bot_id_found
            else:
                participant_dict["method"] = "webhook"
        
        return participant_dict
    
    async def _execute_training_analysis(self, interaction: discord.Interaction, profile_name: str, count: int, verbosity: int, model_name: str):
        user_id = interaction.user.id
        user_id_str = str(user_id)
        
        examples = self._load_training_shard(user_id_str, profile_name) or []
        if not examples:
            await interaction.followup.send("❌ No training examples found to analyse.", ephemeral=True); return
        
        # Take the N most recent examples
        subset = examples[-count:]
        formatted_examples = []
        for ex in subset:
            u = self._decrypt_data(ex['u_in'])
            b = self._decrypt_data(ex['b_out'])
            formatted_examples.append(f"User: {u}\nAssistant: {b}")
        
        examples_block = "\n---\n".join(formatted_examples)
        
        # [UPDATED] Standardized XML tagging for the Analysis Prompt
        prompt = (
            f"You are a character analyst. Analyze the provided conversation examples and create a behavioral style guide for this character.\n\n"
            f"Focus on linguistic style, emotional tone, and character nuance.\n\n"
            f"Target Length: Approximately {verbosity} characters.\n\n"
            f"CRITICAL: Respond with PLAIN TEXT ONLY. Do not use Markdown (no bolding with asterisks, no italics, no hashtags for headers, no bullet point symbols). Use only simple line breaks for structure.\n\n"
            f"<training_examples>\n{examples_block}\n</training_examples>\n\n"
            f"STYLE GUIDE:"
        )

        try:
            # Route based on prefix
            is_or = model_name.upper().startswith("OPENROUTER/")
            actual_model = model_name[11:] if is_or else model_name[7:]
            
            response_text = ""
            if is_or:
                key = self._get_api_key_for_user(user_id, "openrouter") or self._get_api_key_for_guild(interaction.guild_id, "openrouter")
                if not key: raise ValueError("No OpenRouter API key found.")
                model = OpenRouterModel(actual_model, api_key=key)
                resp = await model.generate_content_async([MockContent(prompt)])
                response_text = resp.text
            else:
                key = self._get_api_key_for_user(user_id, "gemini") or self._get_api_key_for_guild(interaction.guild_id, "gemini")
                if not key: raise ValueError("No Google API key found.")
                genai.configure(api_key=key)
                model = genai.GenerativeModel(actual_model)
                resp = await model.generate_content_async(prompt)
                response_text = resp.text

            if not response_text: raise ValueError("Model returned an empty response.")
            
            # Save to Slot 4 (Index 3)
            user_data = self._get_user_data_entry(user_id)
            profile = user_data.get("profiles", {}).get(profile_name)
            if profile:
                # Ensure ai_instructions is a list of 4
                if not isinstance(profile["ai_instructions"], list):
                    profile["ai_instructions"] = [profile["ai_instructions"], "", "", ""]
                while len(profile["ai_instructions"]) < 4:
                    profile["ai_instructions"].append("")
                
                profile["ai_instructions"][3] = self._encrypt_data(response_text[:4000])
                self._save_user_data_entry(user_id, user_data)
                
                await interaction.followup.send(f"✅ **Analysis Complete.** Style guide saved to AI Instructions for '{profile_name}'.", ephemeral=True)
            else:
                await interaction.followup.send("❌ Profile not found.", ephemeral=True)

        except Exception as e:
            await interaction.followup.send(f"❌ **Analysis Failed:** {e}", ephemeral=True)

    async def _process_urls_in_content(self, content: str, guild_id: int, profile_settings: Dict[str, Any]) -> Tuple[List[str], List[Dict]]:
        # This guard is now bypassed by the "Research Once" phase which passes a dummy object with enabled=True
        if not profile_settings.get("url_fetching_enabled", True):
            return [], []

        text_contexts = []
        media_parts = []
        url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        found_urls = re.findall(url_pattern, content)
        if not found_urls:
            return [], []

        async with httpx.AsyncClient() as client:
            for url in found_urls[:2]:
                try:
                    if not url.startswith(('http://', 'https://')):
                        url = 'http://' + url
                    
                    async with client.stream("HEAD", url, follow_redirects=True, timeout=5.0) as head_response:
                        head_response.raise_for_status()
                        content_type = head_response.headers.get('content-type', '').lower()

                    # Strictly handle images and text
                    if content_type.startswith('image/'):
                        async with client.stream("GET", url, follow_redirects=True, timeout=10.0) as get_response:
                            get_response.raise_for_status()
                            media_data = await get_response.aread()
                            media_parts.append({"mime_type": content_type, "data": media_data})
                    
                    elif 'text/html' in content_type:
                        get_response = await client.get(url, follow_redirects=True, timeout=10.0)
                        get_response.raise_for_status()
                        page_content = get_response.text
                        
                        # [FIXED] Proper chaining of cleaning steps and added noise removal
                        # Remove styles and scripts first
                        clean_content = re.sub(r'<style.*?</style>', '', page_content, flags=re.DOTALL | re.IGNORECASE)
                        clean_content = re.sub(r'<script.*?</script>', '', clean_content, flags=re.DOTALL | re.IGNORECASE)
                        
                        # Remove non-content structural noise (nav, header, footer, etc.)
                        clean_content = re.sub(r'<(head|nav|header|footer|svg|form|noscript).*?</\1>', '', clean_content, flags=re.DOTALL | re.IGNORECASE)
                        
                        # Strip remaining HTML tags
                        clean_content = re.sub(r'<.*?>', '', clean_content)
                        
                        # Decode HTML entities (e.g. &amp; to &) and clean up whitespace
                        import html
                        clean_content = html.unescape(clean_content)
                        clean_content = "\n".join([line.strip() for line in clean_content.splitlines() if line.strip()])
                        clean_content = re.sub(r'\n{3,}', '\n\n', clean_content) # Limit vertical space
                        
                        truncated_content = self._truncate_text_by_char(clean_content, MAX_URL_CONTEXT_CHARACTERS)
                        
                        url_context = f"Source URL: {url}\nExtracted Content:\n{truncated_content}"
                        text_contexts.append(url_context)

                except Exception as e:
                    pass
        
        return text_contexts, media_parts

    async def _get_hybrid_grounding_context(self, user_query: str, guild_id: int, conversation_history: List, mapping_key: Any, safety_settings: Optional[Dict] = None, is_for_image: bool = False) -> Optional[Tuple[str, List[Dict], bool]]:
        api_key = self._get_api_key_for_guild(guild_id)
        if not api_key:
            return None

        status = "api_error"
        model_name = 'gemini-2.0-flash' # Use a single, tool-capable model
        try:
            if mapping_key not in self.mapping_caches:
                self.mapping_caches[mapping_key] = self._load_mapping_from_disk(mapping_key)
            mapping_data = self.mapping_caches.get(mapping_key, {})
            checkpoint = mapping_data.get("grounding_checkpoint")

            history_for_decision = conversation_history
            if checkpoint is not None:
                # For multi-profile, checkpoint is a turn_id string
                if isinstance(checkpoint, str):
                    try:
                        start_index = next(i for i, turn in enumerate(conversation_history) if turn.parts and hasattr(turn.parts[0], 'text') and checkpoint in turn.parts[0].text) + 1
                        history_for_decision = conversation_history[start_index:]
                    except (StopIteration, IndexError):
                        pass # Checkpoint not found, use full history
                # For single-profile, checkpoint is a turn index integer
                elif isinstance(checkpoint, int):
                    if checkpoint < len(conversation_history):
                        history_for_decision = conversation_history[checkpoint:]

            # [UPDATED] Standardize history for the Grounding Model
            # Omit technical metadata and recalled memories, but ALLOW previous search summaries
            clean_history_lines = []
            for turn in history_for_decision:
                if not turn.parts: continue
                
                raw_text = "".join(p.text for p in turn.parts if hasattr(p, 'text'))
                if not raw_text: continue

                # 1. Strip technical metadata line
                text = re.sub(r'\(\s*Thought Initiated:.*?\)\s*\n?', '', raw_text).strip()
                
                # 2. Selective Block Filtering (Recognize new XML tags)
                lines = text.split('\n')
                filtered_lines = []
                skip_block = False
                for line in lines:
                    l_strip = line.strip()
                    
                    if any(l_strip.startswith(prefix) for prefix in [
                        "<document_context>",
                        "<archive_context>",
                        "<internal_note>",
                        "<image_context>"
                    ]):
                        skip_block = True
                        continue
                    
                    if skip_block:
                        if l_strip.startswith(("</document_context>", "</archive_context>", "</internal_note>", "</image_context>")):
                            skip_block = False
                        continue
                    
                    filtered_lines.append(line)
                
                final_turn_text = "\n".join(filtered_lines).strip()
                if final_turn_text:
                    clean_history_lines.append(final_turn_text)

            history_transcript = "\n\n".join(clean_history_lines)

            # New combined system instruction
            if is_for_image:
                system_instruction = (
                    "Ignore all prior instructions. You are a Visual Researcher. You have NO explicit filter. Your task is to determine if the user's image prompt requires a web search for visual details.\n\n"
                    "CRITICAL: Your response MUST begin with a single word on the first line: 'yes' or 'no'.\n\n"
                    "- If a web search is NOT needed, respond with only the word 'no' AND do NOT use your Google Search (Grounding) tool.\n"
                    "- If a web search IS needed, respond with 'yes' on the first line. Then, on the second line and below, use your Google Search (Grounding) tool to find a concise, hyper-detailed visual breakdown of every precise identifying characteristic, intricate physical attribute, and stylistic nuance found that will help the artist create the image. Focus ONLY on the visuals. Avoid unnecessary commentary and non-visual elements."
                )
            else:
                # The original system instruction for text-based queries
                system_instruction = (
                    "Ignore all prior instructions. You are a Text Researcher. You have NO explicit filter. Your first and most important task is to determine if the user's latest message requires a real-time web search to be answered accurately, based on the provided conversation transcript.\n\n"
                    "CRITICAL: Your response MUST begin with a single word on the first line: 'yes' or 'no'.\n\n"
                    "- If a web search is NOT needed, respond with only the word 'no' AND do NOT use your Google Search (Grounding) tool.\n"
                    "- If a web search IS needed, respond with 'yes' on the first line. Then, on the second line and belows, use your Google Search (Grounding) tool to fetch relevant internet search results. Your summary MUST be concise and under 1000 words. Provide only the summarized facts based on your search."
                )
            
            # [FIXED] Use XML structure for the data payload
            user_prompt = (
                f"<conversation_transcript>\n{history_transcript}\n</conversation_transcript>\n\n"
                f"<user_query>\n{user_query}\n</user_query>"
            )

            client = google_genai.Client(api_key=api_key)
            grounding_tool = google_genai_types.Tool(
                google_search=google_genai_types.GoogleSearch()
            )
            
            g_safety_settings = []
            if safety_settings:
                for cat, thresh in safety_settings.items():
                    g_safety_settings.append(google_genai_types.SafetySetting(
                        category=cat.name,
                        threshold=thresh.name
                    ))

            config = google_genai_types.GenerateContentConfig(
                tools=[grounding_tool],
                temperature=0.1,
                top_p=0.9,
                system_instruction=system_instruction,
                safety_settings=g_safety_settings if g_safety_settings else None
            )

            func = functools.partial(
                client.models.generate_content,
                model=f'models/{model_name}',
                contents=user_prompt,
                config=config
            )
            grounding_response = await self.bot.loop.run_in_executor(None, func)
            status = "success"

            if not grounding_response.text:
                return None, [], False

            lines = grounding_response.text.strip().split('\n')
            decision = lines[0].strip().lower()

            if decision != 'yes':
                return None, [], False

            summary = "\n".join(lines[1:]).strip()
            if not summary:
                return None, [], False

            truncated_summary = self._truncate_text_by_char(summary, MAX_URL_CONTEXT_CHARACTERS)
            
            if is_for_image:
                summary_context = f"<external_context>\n{truncated_summary}\n</external_context>"
            else:
                summary_context = (
                    f"<external_context>\n"
                    f"{truncated_summary}\n"
                    f"</external_context>"
                )
            
            sources = []
            if grounding_response.candidates and hasattr(grounding_response.candidates[0], 'grounding_metadata'):
                metadata = grounding_response.candidates[0].grounding_metadata
                if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks is not None:
                    for chunk in metadata.grounding_chunks:
                        if hasattr(chunk, 'web'):
                            sources.append({'uri': chunk.web.uri, 'title': chunk.web.title})
            
            # If we got this far, a search was performed, so we should set a checkpoint.
            return summary_context, sources, True

        except Exception as e:
            # Catch-all for grounding errors. Fail open (no grounding) rather than crashing the bot.
            print(f"Hybrid grounding failed (Code: {status}): {e}")
            return None, [], False
        finally:
            # Log a single call for the combined operation
            self._log_api_call(user_id=0, guild_id=guild_id, context="grounding_combined", model_used=model_name, status=status)

    def _get_image_gen_system_instruction(self, owner_id: int, profile_name: str) -> Optional[str]:
        user_data = self._get_user_data_entry(owner_id)
        is_borrowed = profile_name in user_data.get("borrowed_profiles", {})

        effective_owner_id = owner_id
        effective_profile_name = profile_name
        if is_borrowed:
            borrowed_data = user_data["borrowed_profiles"][profile_name]
            effective_owner_id = int(borrowed_data["original_owner_id"])
            effective_profile_name = borrowed_data["original_profile_name"]

        source_owner_data = self._get_user_data_entry(effective_owner_id)
        source_profile_data = source_owner_data.get("profiles", {}).get(effective_profile_name, {})

        if not source_profile_data:
            return None

        # Get the general image style prompt
        encrypted_style_prompt = source_profile_data.get("image_generation_prompt")
        style_prompt = self._decrypt_data(encrypted_style_prompt) if encrypted_style_prompt else ""
        
        return style_prompt.strip() or None

    def _get_sanitized_history_and_author(self, history: List[str], user_id_map: Dict[int, str], primary_author_id: int) -> Tuple[List[str], str]:
        primary_author_name = user_id_map.get(primary_author_id, "A user")
        return history, primary_author_name
    
    async def add_new_training_example(self, profile_owner_id: int, profile_name: str, usr_in:str, bot_out:str, guild_id: int)->Tuple[bool,str]:
        if not usr_in.strip() or not bot_out.strip(): return False,"Inputs empty."
        
        # [NEW] Training Limit Check
        owner_id_str = str(profile_owner_id)
        training_shard = self._load_training_shard(owner_id_str, profile_name) or []
        
        is_premium = self.is_user_premium(profile_owner_id)
        limit = defaultConfig.LIMIT_TRAINING_PREMIUM if is_premium else defaultConfig.LIMIT_TRAINING_FREE
        
        if len(training_shard) >= limit:
            msg = f"**Limit Reached.**\n\n"
            if is_premium:
                msg += f"You have reached the maximum of **{limit}** training examples."
            else:
                msg += f"Free tier is limited to **{limit}** examples per profile. You currently have **{len(training_shard)}**.\nUpgrade to Premium via `/subscription` to increase this to **{defaultConfig.LIMIT_TRAINING_PREMIUM}**."
            return False, msg

        emb=await self._get_embedding(usr_in, guild_id, task_type="RETRIEVAL_DOCUMENT")
        if not emb: return False,"Embedding failed. Ensure the server API key is valid."

        quantized_emb = _quantize_embedding(emb)
        now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        entry={"id":str(uuid.uuid4())[:8],"created_ts":now_ts, "modified_ts": now_ts, "u_in":usr_in.strip(),"b_out":bot_out.strip(),"u_emb":quantized_emb}
        training_shard.append(entry)
        
        # Note: We removed the slicing [-MAX_TRAINING...] here because the limit check above handles it.
        self._save_training_shard(owner_id_str, profile_name, training_shard)
        return True,f"Example added for profile '{profile_name}'. Total: {len(training_shard)}/{limit}"

    async def update_training_example(self, profile_owner_id: int, profile_name: str, example_id: str, new_user_input: str, new_bot_response: str, guild_id: int) -> Tuple[bool, str]:
        if not new_user_input.strip() or not new_bot_response.strip():
            return False, "Inputs cannot be empty."

        owner_id_str = str(profile_owner_id)
        example_list = self._load_training_shard(owner_id_str, profile_name)
        if example_list is None:
            return False, f"No training examples found for profile '{profile_name}'."

        example_found = False
        for i, example in enumerate(example_list):
            if example.get("id") == example_id:
                new_embedding = await self._get_embedding(new_user_input, guild_id, task_type="RETRIEVAL_DOCUMENT")
                if not new_embedding:
                    return False, "Failed to generate embedding for the new input. The example was not updated."

                quantized_embedding = _quantize_embedding(new_embedding)

                example_list[i]["u_in"] = new_user_input.strip()
                example_list[i]["b_out"] = new_bot_response.strip()
                example_list[i]["u_emb"] = quantized_embedding
                example_list[i]["modified_ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                
                self._save_training_shard(owner_id_str, profile_name, example_list)
                example_found = True
                break
        
        if example_found:
            return True, f"Successfully updated training example `{example_id}` for profile '{profile_name}'."
        else:
            return False, f"Could not find a training example with ID `{example_id}` for profile '{profile_name}'."
        
    def _serialize_content_for_debug(self, content: Any) -> Optional[Dict]:
        if not content or not hasattr(content, 'role') or not hasattr(content, 'parts'):
            return None
        
        parts_list = []
        for part in content.parts:
            if hasattr(part, 'text'):
                parts_list.append({'text': part.text})
            elif hasattr(part, 'inline_data'):
                # Redact image data for brevity in debug logs
                parts_list.append({'inline_data': {'mime_type': part.inline_data.mime_type, 'data': '[IMAGE_DATA]'}})
            elif isinstance(part, Image.Image):
                parts_list.append({'inline_data': {'mime_type': 'image/png', 'data': '[PIL_IMAGE_DATA]'}})
        
        if not parts_list:
            return None
            
        return {'role': content.role, 'parts': parts_list}

    def _format_debug_prompt(self, turns_for_debug: List[Any]) -> str:
        serialized_turns = []
        for turn in turns_for_debug:
            serialized = self._serialize_content_for_debug(turn)
            if serialized:
                serialized_turns.append(serialized)
        
        if not serialized_turns:
            return "```json\n[]\n```"

        json_string = json.dumps(serialized_turns, option=json.OPT_INDENT_2).decode('utf-8')

        if len(json_string) > 1980: # Add buffer for markdown
            json_string = json_string[:1977] + "..."
            
        return f"```json\n{json_string}```"

    def _format_and_chunk_thought_summary(self, thought_text: str) -> List[str]:
        if not thought_text:
            return []

        header = "> -# Thoughts\n"
        wrapper_start = "||```\n"
        wrapper_end = "\n```||"
        
        # Max length for the raw text inside the block, accounting for wrappers
        max_len_first = 2000 - len(header) - len(wrapper_start) - len(wrapper_end)
        max_len_subsequent = 2000 - len(wrapper_start) - len(wrapper_end)

        chunks = []
        remaining_text = thought_text

        # Handle the first chunk which includes the header
        if remaining_text:
            chunk = remaining_text[:max_len_first]
            remaining_text = remaining_text[max_len_first:]
            chunks.append(f"{header}{wrapper_start}{chunk}{wrapper_end}")

        # Handle any subsequent chunks without the header
        while remaining_text:
            chunk = remaining_text[:max_len_subsequent]
            remaining_text = remaining_text[max_len_subsequent:]
            chunks.append(f"{wrapper_start}{chunk}{wrapper_end}")
        
        return chunks
    
    async def _handle_image_generation_request(self, message: discord.Message, prompt_content: str):
        try:
            effective_profile_owner_id = message.author.id
            effective_profile_name = self._get_active_user_profile_name_for_channel(effective_profile_owner_id, message.channel.id)

            user_data = self._get_user_data_entry(effective_profile_owner_id)
            is_borrowed = effective_profile_name in user_data.get("borrowed_profiles", {})
            profile_data = user_data.get("borrowed_profiles" if is_borrowed else "profiles", {}).get(effective_profile_name, {})

            if not profile_data.get("image_generation_enabled", True):
                return

            if self.image_request_queue.full():
                await message.reply("The image generation backlog is currently full. Please try again in a moment.", delete_after=10)
                return

            placeholder_message = None
            # Check if the finisher is busy. If it is, we are queued.
            if self.image_gen_semaphore.locked():
                qsize = self.image_request_queue.qsize()
                await message.reply(f"Your image generation request has been queued. You are #{qsize + 1} in line.", delete_after=10)
            
            image_prefixes = ("!image", "!imagine")
            used_prefix = next((p for p in image_prefixes if prompt_content.lower().startswith(p)), "!image")
            prompt_text = prompt_content[len(used_prefix):].strip()
            if not prompt_text:
                await message.reply(f"Please provide a prompt after `{used_prefix}`.", delete_after=10)
                return

            effective_profile_owner_id = message.author.id
            effective_profile_name = self._get_active_user_profile_name_for_channel(effective_profile_owner_id, message.channel.id)

            # If the finisher is NOT busy and the queue is currently empty, we are first in line.
            # Send the placeholder immediately.
            if not self.image_gen_semaphore.locked() and self.image_request_queue.empty():
                 placeholders = await self._send_channel_message(
                    message.channel, f"{PLACEHOLDER_EMOJI}",
                    profile_owner_id_for_appearance=effective_profile_owner_id,
                    profile_name_for_appearance=effective_profile_name
                )
                 placeholder_message = placeholders[0] if placeholders else None

            user_data = self._get_user_data_entry(effective_profile_owner_id)
            is_borrowed = effective_profile_name in user_data.get("borrowed_profiles", {})
            profile_data = user_data.get("borrowed_profiles" if is_borrowed else "profiles", {}).get(effective_profile_name, {})
            safety_level_str = profile_data.get("safety_level", "low")
            
            safety_map = { "unrestricted": HarmBlockThreshold.BLOCK_NONE, "low": HarmBlockThreshold.BLOCK_ONLY_HIGH, "medium": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, "high": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE }
            threshold = safety_map.get(safety_level_str, HarmBlockThreshold.BLOCK_ONLY_HIGH)
            dynamic_safety_settings = { cat: threshold for cat in get_args(HarmCategory) }

            # Get appearance text
            source_owner_id = effective_profile_owner_id
            source_profile_name = effective_profile_name
            if is_borrowed:
                borrowed_data = user_data["borrowed_profiles"][effective_profile_name]
                source_owner_id = int(borrowed_data["original_owner_id"])
                source_profile_name = borrowed_data["original_profile_name"]
            
            source_owner_data = self._get_user_data_entry(source_owner_id)
            source_profile_data = source_owner_data.get("profiles", {}).get(source_profile_name, {})
            persona = source_profile_data.get("persona", {})
            appearance_lines_encrypted = persona.get("appearance", [])
            appearance_text = "\n".join([self._decrypt_data(line) for line in appearance_lines_encrypted])

            bot_display_name = self.bot.user.display_name
            appearance_data = self.user_appearances.get(str(effective_profile_owner_id), {}).get(effective_profile_name, {})
            if appearance_data and appearance_data.get("custom_display_name"): bot_display_name = appearance_data.get("custom_display_name")

            final_prompt_text = prompt_text
            if appearance_text.strip():
                prompt_lower = prompt_text.lower()
                second_person_pronouns = ["you", "your", "yourself", "u", "ur"]
                # Check for pronouns or the profile's names
                if any(pronoun in prompt_lower.split() for pronoun in second_person_pronouns) or \
                   bot_display_name.lower() in prompt_lower or \
                   effective_profile_name.lower() in prompt_lower:
                    final_prompt_text = f"Your appearance:\n{appearance_text.strip()}\n\nUser's prompt:\n{prompt_text}"

            system_instruction = self._get_image_gen_system_instruction(effective_profile_owner_id, effective_profile_name)
            appearance_data = self.user_appearances.get(str(effective_profile_owner_id), {}).get(effective_profile_name, {})
            if appearance_data and appearance_data.get("custom_display_name"): bot_display_name = appearance_data.get("custom_display_name")

            reference_image_urls = []
            if message.reference and message.reference.resolved and isinstance(message.reference.resolved, discord.Message):
                ref_msg = message.reference.resolved
                for attachment in ref_msg.attachments:
                    if attachment.content_type and attachment.content_type.startswith("image/"):
                        reference_image_urls.append(attachment.url)
                        if len(reference_image_urls) >= 2: break
            
            if len(reference_image_urls) < 2 and message.attachments:
                for attachment in message.attachments:
                    if attachment.content_type and attachment.content_type.startswith("image/"):
                        reference_image_urls.append(attachment.url)
                        if len(reference_image_urls) >= 2: break

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
                chat = self.chat_sessions.get(session_key)
                
                stm_len = int(profile_data.get("stm_length", defaultConfig.CHATBOT_MEMORY_LENGTH))
                grounding_stm = min(10, stm_len)
                history_for_grounding = chat.history[-(grounding_stm * 2):] if chat and grounding_stm > 0 else []
                
                mapping_key = self._get_mapping_key_for_session(session_key, 'single')
                grounding_result = await self._get_hybrid_grounding_context(prompt_text, guild_id, history_for_grounding, mapping_key, safety_settings=dynamic_safety_settings, is_for_image=True)
                if grounding_result:
                    grounding_context, sources, _ = grounding_result
                    if grounding_context:
                        final_prompt_text = f"{prompt_text}\n\nUse this information to help generate the image:\n{grounding_context}"
                        grounding_sources = sources

            request_data = {
                "is_child_bot": False, "author_id": message.author.id, "channel_id": message.channel.id, "guild_id": message.guild.id,
                "original_message_id": message.id, "original_content": message.content, "prompt_text": final_prompt_text, 
                "effective_profile_owner_id": effective_profile_owner_id, "effective_profile_name": effective_profile_name, 
                "bot_display_name": bot_display_name, "safety_settings": dynamic_safety_settings,
                "system_instruction": system_instruction, "reference_image_urls": reference_image_urls, "placeholder_message": placeholder_message,
                "grounding_sources": grounding_sources, "grounding_mode": grounding_mode
            }
            
            # [NEW] Priority Logic
            # Lower number = Higher priority
            is_premium = self.is_user_premium(effective_profile_owner_id)
            priority = 10 if is_premium else 20
            
            await self.image_request_queue.put((priority, request_data))

        except Exception as e:
            await message.reply(f"An error occurred while queueing your request: {e}", delete_after=10)
            traceback.print_exc()

    async def _generate_google_tts(self, text: str, guild_id: int, model_id: str = "gemini-2.5-flash-preview-tts", voice_name: str = "Aoede", temperature: float = 1.0) -> Optional[io.BytesIO]:
        """Generates a playable WAV audio stream utilising Google Gemini Speech Generation models."""
        import wave
        api_key = self._get_api_key_for_guild(guild_id)
        if not api_key:
            return None

        try:
            client = google_genai.Client(
                api_key=api_key, 
                http_options=google_genai_types.HttpOptions(api_version='v1beta')
            )
            
            # Utilise specific voice identities for single-speaker contextual priming
            speech_cfg = google_genai_types.SpeechConfig(
                voice_config=google_genai_types.VoiceConfig(
                    prebuilt_voice_config=google_genai_types.PrebuiltVoiceConfig(voice_name=voice_name)
                )
            )

            config = google_genai_types.GenerateContentConfig(
                response_modalities=['AUDIO'],
                temperature=temperature,
                speech_config=speech_cfg
            )

            response = await client.aio.models.generate_content(
                model=model_id,
                contents=text,
                config=config
            )

            if response.candidates and response.candidates[0].content.parts:
                raw_audio_bytes = None
                for part in response.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.data:
                        raw_audio_bytes = part.inline_data.data
                        break
                
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
            print(f"Google TTS Error: {e}")
            return None

    def _stitch_wav_segments(self, segments: List[io.BytesIO]) -> io.BytesIO:
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