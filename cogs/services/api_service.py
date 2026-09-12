import os
import time
import asyncio
import base64
import datetime
import httpx
import orjson as json
from collections import OrderedDict
from typing import get_args, Any, Dict, List, Optional, Tuple
from discord.ext import tasks

from ..utils.constants import (
    OLLAMA_LOCAL_URL, MODELS_DATA_DIR, PRICING_CACHE_FILE, IMAGE_MODELS, AUDIO_MODELS,
    ALLOWED_MODELS, defaultConfig, PRIMARY_MODEL_NAME, FALLBACK_MODEL_NAME,
    IMAGE_MODEL_KEYS, AUDIO_MODEL_KEYS, DEFAULT_SPEECH_VOICE,
    THINKING_LEVELS_TO_GOOGLE, THINKING_LEVELS_TO_GOOGLE_BINARY,
    THINKING_LEVELS_TO_OLLAMA, GEMINI_FREE_TIER_BLOCKED, OLLAMA_OWNER_ONLY,
)
from ..utils.data_policy import openrouter_data_collection
from ..managers.storage_manager import IOManager
from ..utils.blob_stream import InlineBlobExtractor, is_blob_sentinel, sentinel_path
from ..utils.helpers import (_resolve_safety_settings, is_real_model,
                            google_thinking_caps, resolve_media_resolution,
                            resolve_native_tools, resolve_openrouter_image_detail,
                            resolve_openrouter_service_tier, resolve_thinking_params)
from ..utils.http_client import get_shared_client
from ..utils.net_guard import safe_stream
from ..utils.memory_tuning import maybe_trim_malloc
from ..utils import mem_probe

# How long a key that just got rate-limited is skipped for. storage_manager's
# _get_api_key_for_guild/_get_api_key_for_user consult cog.api_key_cooldowns before
# handing a key back out, so this is what stops a 429'd BYO key from being retried
# on the very next turn instead of backing off.

# The adapters moved to `services/api/`; these names are re-exported because
# MimicCog, profile_manager, memory_manager, media_service, tools_service and
# help_service all import them from here.
from .api.embeddings import get_embedding_vector
from .api.google_rest import (
    GoogleRESTModel, GoogleRESTResponse, close_google_rest_client,
    generate_google_tts_audio, get_google_rest_client, materialise_inline_data,
)
from .api.ollama import OllamaModel, OllamaResponse
from .api.openrouter import OpenRouterModel
from .api.openrouter_catalogue import BROWSE_POPULAR, OpenRouterCatalogue
from .api.rest_view import _BlobRef, _EnumStr, _RestView

_KEY_COOLDOWN_SECONDS = 60.0





def _cooldown_key_on_rate_limit(cog, api_key: Optional[str], error: BaseException) -> None:
    if not api_key:
        return
    err_str = str(error)
    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
        cog.api_key_cooldowns[api_key] = time.time() + _KEY_COOLDOWN_SECONDS


def _with_key_cooldown_tracking(cog, model, api_key: str):
    """Wraps model.generate_content_async so a rate-limit response cools the BYO key down."""
    original = model.generate_content_async

    async def _tracked(*args, **kwargs):
        try:
            return await original(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _cooldown_key_on_rate_limit(cog, api_key, e)
            raise

    model.generate_content_async = _tracked
    return model


# --- Google adapter routing ----------------------------------------------------
#
# Retained as an alias because ten construction sites across six files import this
# name. Migration 2 is complete and GoogleRESTModel is the only Google adapter, so
# there is nothing left to route — but renaming those ten sites is a separate diff
# from deleting the SDK, and this file is the one that had to change.
GoogleGenAIModel = GoogleRESTModel


class APIService:
    """Owns model-instantiation routing: resolves a raw model name (GOOGLE/, OPENROUTER/,
    OLLAMA/, or bare) and the caller's API key into the correct adapter instance.

    Holds a back-reference to the parent cog for state/logic not yet migrated
    (API key resolution), per the transitional Dependency Injection pattern in
    CLAUDE.md.
    """

    #: Exposed here so call sites reach it through `self.cog.api_service` like
    #: everything else on this service, rather than adding an import edge from
    #: media_service and the generation mixins back into this module.
    materialise_inline_data = staticmethod(materialise_inline_data)

    def __init__(self, cog):
        self.cog = cog
        #: Every text-output OpenRouter model, loaded by the first pricing sync and
        #: refreshed by each one after. See api/openrouter_catalogue.
        self.catalogue = OpenRouterCatalogue(MODELS_DATA_DIR)
        self._catalogue_loaded = False
        #: The pricing table, held in memory. `_get_model_pricing` used to re-read and
        #: parse the whole file per call, and the audit screens call it once per turn.
        self._pricing_rates: Optional[Dict[str, Dict[str, float]]] = None
        #: host -> (answered, model names) from each Ollama host's last probe, oldest
        #: first. Shared by every picker -- see probe_ollama.
        self._ollama_probes: "OrderedDict[str, Tuple[bool, List[str]]]" = OrderedDict()

    def _instantiate_model(self, raw_model_name: str, guild_id, user_id, system_instruction=None, safety_settings=None, thinking_params=None, tools=None, profile_settings=None, openrouter_key_error: str = None, google_key_error: str = None, use_broad_openrouter_heuristic: bool = True, config_owner_id=None, policy_guild_id=None, conversation: bool = False):
        """One adapter for `raw_model_name`, keyed and policed for where it will run.

        `guild_id` picks whose key pays: the server's, or `user_id`'s own when None. A
        server's traffic answers to that server's data policy.

        `conversation=True` marks a conversation on `user_id`'s own key -- Global Chat, the
        only one -- which answers to the policy of `policy_guild_id`, the server its card
        was opened in, and is closed when that is None: a DM or a group DM has no server for
        the bot owner to open. Without it a personal key carries only its owner's command
        input, which nothing gates. `policy_guild_id` means nothing otherwise, so passing it
        alone is refused rather than silently ignored.

        `config_owner_id` is the owner of the config that chose this model. Ollama is
        refused unless that is the bot owner, and refused when it is not given.
        """
        # System prefixes 'GOOGLE/', 'OPENROUTER/', and 'OLLAMA/' are strictly case-sensitive.
        # OpenRouter hosts models under lowercase creator namespaces like 'google/gemini-2.5-flash'.
        actual_name = raw_model_name
        is_openrouter = False
        is_ollama = False

        if raw_model_name.startswith("OPENROUTER/"):
            actual_name = raw_model_name[11:]
            is_openrouter = True
        elif raw_model_name.startswith("OLLAMA/"):
            actual_name = raw_model_name[7:]
            is_ollama = True
        elif raw_model_name.startswith("GOOGLE/"):
            actual_name = raw_model_name[7:]
        elif "/" in raw_model_name or (use_broad_openrouter_heuristic and ("grok" in raw_model_name.lower() or "anthropic" in raw_model_name.lower())):
            is_openrouter = True

        t_params = thinking_params or {}
        p_settings = profile_settings or {}
        # One setting, two wire shapes: Google takes a request-level mediaResolution
        # enum, OpenRouter the coarser per-part `detail` hint. Ollama has neither and
        # is handed nothing. Resolved here rather than at each call site because this
        # is the one constructor every provider goes through, and the profile config
        # it needs is already in hand.
        media_res = resolve_media_resolution(p_settings)
        image_detail = resolve_openrouter_image_detail(p_settings)
        service_tier = resolve_openrouter_service_tier(p_settings)

        if policy_guild_id is not None and not conversation:
            raise TypeError("policy_guild_id is only read for a conversation")
        policy_guild = policy_guild_id if conversation else guild_id
        gated = conversation or bool(guild_id)
        storage = self.cog.storage_manager

        if is_openrouter:
            api_key = storage._get_api_key_for_guild(guild_id, "openrouter") if guild_id else storage._get_api_key_for_user(user_id, "openrouter")
            if not api_key: raise ValueError(openrouter_key_error or "OpenRouter API Key not found. Run `/start` to set one up, or `/settings` if you know your way around.")
            data_collection = (openrouter_data_collection(
                self.cog.server_manager._get_server_index(str(policy_guild)) if policy_guild else None)
                if gated else None)
            model = OpenRouterModel(actual_name, api_key=api_key, system_instruction=system_instruction, thinking_params=t_params, image_detail=image_detail, service_tier=service_tier, data_collection=data_collection)
            return _with_key_cooldown_tracking(self.cog, model, api_key)
        elif is_ollama:
            if not self.cog.profile_manager.may_use_ollama(config_owner_id):
                raise ValueError(OLLAMA_OWNER_ONLY)
            ollama_host = p_settings.get("ollama_host_url", OLLAMA_LOCAL_URL)
            return OllamaModel(actual_name, api_url=ollama_host, system_instruction=system_instruction, thinking_params=t_params)
        else:
            if guild_id:
                api_key = storage._get_api_key_for_guild(guild_id)
                if not api_key and storage.gemini_blocked_for_guild(guild_id):
                    raise ValueError(GEMINI_FREE_TIER_BLOCKED)
            else:
                if conversation and not storage.personal_gemini_allowed_in_conversation(user_id, policy_guild):
                    raise ValueError(GEMINI_FREE_TIER_BLOCKED)
                api_key = storage._get_api_key_for_user(user_id)
            if not api_key: raise ValueError(google_key_error or "Google API Key not found. Run `/start` to set one up, or `/settings` if you know your way around.")
            model = GoogleGenAIModel(api_key=api_key, model_name=actual_name, system_instruction=system_instruction, safety_settings=safety_settings, thinking_params=t_params, tools=tools, media_resolution=media_res)
            return _with_key_cooldown_tracking(self.cog, model, api_key)

    async def run_with_fallback(self, primary: str, fallback: Optional[str], attempt,
                                *, label: str = "utility"):
        """Runs one utility generation on `primary`, retrying once on `fallback`.

        `attempt(model_name, is_fallback)` owns the construction and the response
        handling; this owns only which name to try and when to stop. That puts the five
        utility paths -- critic, grounding, LTM, image and speech -- on one retry policy
        without forcing them into one call shape, which they genuinely do not share: one
        returns audio bytes, one drives a heartbeat, three return a candidate list.

        Only exceptions retry. An empty or safety-blocked response is a decision about
        the content, not a statement about the model being unavailable, and re-rolling
        it on a second model spends another call to be refused again.

        A fallback equal to the primary is skipped rather than tried twice, which is
        what makes the shipped defaults cost nothing: every utility fallback defaults to
        the same model as its primary, so the retry only becomes live once someone
        actually changes one of them.

        Returns (result, model_used, used_fallback). If both fail the second error is
        raised -- callers report the error they are handed, and the fallback's is the
        one that actually ended the attempt.
        """
        attempts = [(primary, False)]
        if is_real_model(fallback) and fallback != primary:
            attempts.append((fallback, True))

        last_error = None
        for name, is_fallback in attempts:
            try:
                return await attempt(name, is_fallback), name, is_fallback
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_error = e
                if not is_fallback and len(attempts) > 1:
                    print(f"{label}: primary '{name}' failed "
                          f"({type(e).__name__}: {e}); retrying on '{attempts[1][0]}'.")
        raise last_error

    def get_top_models(self, provider: str, target_config_key: str,
                       ollama_host: Optional[str] = None) -> List[str]:
        if target_config_key in IMAGE_MODEL_KEYS: return list(get_args(IMAGE_MODELS))
        if target_config_key in AUDIO_MODEL_KEYS: return list(get_args(AUDIO_MODELS))
        if provider == 'google': return list(get_args(ALLOWED_MODELS))
        elif provider == 'ollama': return list((self.last_ollama_probe(ollama_host) or (False, []))[1])
        return self.catalogue.browse(BROWSE_POPULAR, show_training=False)[0]

    #: Ollama hosts whose last probe is kept. Only the bot owner can use Ollama, so this is
    #: one or two hosts; the bound is for a host URL retyped over and over.
    _OLLAMA_PROBES_KEPT = 8

    @staticmethod
    def _ollama_host(host_url: Optional[str]) -> str:
        return (host_url or OLLAMA_LOCAL_URL).rstrip('/')

    def last_ollama_probe(self, host_url: Optional[str]) -> Optional[Tuple[bool, List[str]]]:
        """(answered, model names) from this Ollama host's last probe, or None if it has not
        been probed since the bot started. Never dials."""
        return self._ollama_probes.get(self._ollama_host(host_url))

    async def probe_ollama(self, host_url: Optional[str]) -> Tuple[bool, List[str]]:
        """Asks an Ollama host which models it has pulled, and keeps the answer.

        Kept per host rather than per picker, so a picker opening its Ollama tab shows the
        last answer at once while it asks again -- see ModelPickerMixin._add_api_buttons.
        """
        host = self._ollama_host(host_url)
        try:
            resp = await get_shared_client().get(f"{host}/api/tags", timeout=2.0)
            answered = resp.status_code == 200
            models = [m['name'] for m in resp.json().get('models', [])] if answered else []
        except Exception:
            answered, models = False, []
        self._ollama_probes[host] = (answered, models)
        self._ollama_probes.move_to_end(host)
        while len(self._ollama_probes) > self._OLLAMA_PROBES_KEPT:
            self._ollama_probes.popitem(last=False)
        return answered, models

    def record_openrouter_use(self, model_id: str) -> None:
        """Counts a successful OpenRouter call for Most Popular, writing it now and then.

        This used to read, parse and rewrite the count file on the event loop after every
        successful turn, with a truncating write a crash could empty.
        """
        if not self._catalogue_loaded:
            self.catalogue._load_usage()
        if self.catalogue.record_use(model_id):
            try:
                asyncio.get_running_loop().create_task(asyncio.to_thread(self.catalogue.flush_usage))
            except RuntimeError:
                self.catalogue.flush_usage()

    async def _get_or_create_model_for_channel(self, channel_id: int, actual_message_author_id: int, guild_id: int, profile_owner_override: Optional[int] = None, profile_name_override: Optional[str] = None, prompt_content: Optional[str] = None) -> Tuple[Optional[Any], bool, float, float, int, Optional[str], Optional[str]]:
        
        api_key = self.cog.storage_manager._get_api_key_for_guild(guild_id)
        
        if profile_owner_override is not None and profile_name_override is not None:
            profile_owner_id_for_instructions = profile_owner_override
            profile_name_for_instructions = profile_name_override
        else:
            profile_owner_id_for_instructions: Optional[int] = actual_message_author_id
            profile_name_for_instructions: str = self.cog.session_manager._get_active_user_profile_name_for_channel(profile_owner_id_for_instructions, channel_id)
        
        channel = self.cog.bot.get_channel(channel_id)
        if not channel:
            return None, True, 0.0, 0.0, 0, "Could not find the channel for this interaction.", None

        if not self.cog.profile_manager._check_unrestricted_safety_policy(profile_owner_id_for_instructions, profile_name_for_instructions, channel):
            return None, True, 0.0, 0.0, 0, "This character's content rating is Adult 18+, which only runs in age-restricted channels.", None

        model_cache_key = (channel_id, profile_owner_id_for_instructions, profile_name_for_instructions)

        user_index = self.cog.profile_manager._get_user_index(profile_owner_id_for_instructions)
        is_borrowed = profile_name_for_instructions in user_index.get("borrowed", [])

        original_owner_id, original_profile_name = self.cog.profile_manager._resolve_effective_profile(profile_owner_id_for_instructions, profile_name_for_instructions)

        current_profile_key_for_model = (original_owner_id, original_profile_name)

        training_examples_list = []
        if prompt_content:
            training_examples_list = await self.cog.memory_manager._get_relevant_training_examples(
                profile_owner_id_for_instructions,
                profile_name_for_instructions,
                prompt_content,
                guild_id
            )

        current_instructions, error_in_instr_constr, _, temperature, top_p, top_k, primary_model, fallback_model = self.cog.generation_service._construct_system_instructions(
            profile_owner_id_for_instructions,
            profile_name_for_instructions,
            channel_id,
            training_examples_list=training_examples_list
        )
        
        # Either provider will do: this used to test the Gemini slot alone, so an
        # OpenRouter-only server was told it had no key at all.
        if (not api_key and not primary_model.upper().startswith("OLLAMA/")
                and not self.cog.storage_manager._get_api_key_for_guild(guild_id, "openrouter")):
            if self.cog.storage_manager.gemini_blocked_for_guild(guild_id):
                return None, True, 0.0, 0.0, 0, GEMINI_FREE_TIER_BLOCKED, None
            return None, True, 0.0, 0.0, 0, "Server API key is not configured.", None
        
        warning_message = None

        recreate_model = True
        if model_cache_key in self.cog.channel_models and not training_examples_list:
            last_profile_key = self.cog.channel_model_last_profile_key.get(model_cache_key)
            if last_profile_key == current_profile_key_for_model:
                 recreate_model = False 
            
        if recreate_model and model_cache_key in self.cog.channel_models:
            del self.cog.channel_models[model_cache_key]
            self.cog.channel_model_last_profile_key.pop(model_cache_key, None)
        
        if model_cache_key in self.cog.channel_models and not recreate_model: 
            model_instance, model_init_error_state, cached_model_name = self.cog.channel_models[model_cache_key]
            return model_instance, model_init_error_state, temperature, top_p, top_k, warning_message, fallback_model

        model_instance, model_init_error = None, True
        
        profile_data_for_safety = self.cog.profile_manager._get_profile_config(profile_owner_id_for_instructions, profile_name_for_instructions, is_borrowed) or {}
        dynamic_safety_settings = _resolve_safety_settings(channel, profile_data_for_safety)

        model_to_create = primary_model
        
        # Extract parameters once for either provider
        p_sett_thinking = self.cog.profile_manager._get_profile_config(profile_owner_id_for_instructions, profile_name_for_instructions, is_borrowed) or {}
        t_params = resolve_thinking_params(p_sett_thinking, "response")

        model_tools = resolve_native_tools(p_sett_thinking)

        try:
            model_instance = self._instantiate_model(model_to_create, guild_id, profile_owner_id_for_instructions, current_instructions, dynamic_safety_settings, t_params, model_tools, p_sett_thinking, config_owner_id=profile_owner_id_for_instructions)
            model_init_error = False
        except Exception as e1:
            print(f"Err '{model_to_create}' key {model_cache_key}: {e1}. Fallback.")
            model_to_create = fallback_model
            # Its own resolution, not the primary's: a cheap standby behind an
            # expensive primary is the usual shape, and one shared effort made the
            # standby either waste what it was chosen to save or under-think a request
            # the primary was tuned for. Unset still inherits the primary.
            t_params_fb = resolve_thinking_params(p_sett_thinking, "response", "fallback")
            try:
                model_instance = self._instantiate_model(model_to_create, guild_id, profile_owner_id_for_instructions, current_instructions, dynamic_safety_settings, t_params_fb, model_tools, p_sett_thinking, config_owner_id=profile_owner_id_for_instructions)
                model_init_error = False
            except Exception as e2:
                return None, True, temperature, top_p, top_k, f"Model Initialization Error: Failed to load Primary ('{primary_model}') and Fallback ('{fallback_model}') models. Check your API key.", fallback_model
        
        final_error_state = error_in_instr_constr or model_init_error
        self.cog.channel_models[model_cache_key] = (model_instance, final_error_state, model_to_create)
        self.cog.channel_model_last_profile_key[model_cache_key] = current_profile_key_for_model
        return model_instance, final_error_state, temperature, top_p, top_k, warning_message, fallback_model

    async def _get_or_create_model_for_global_chat(self, user_id: int, profile_name: str, policy_guild_id: Optional[int] = None) -> Tuple[Optional[Any], float, float, int, Optional[str], Optional[str]]:
        source_owner_id, source_profile_name = self.cog.profile_manager._resolve_effective_profile(user_id, profile_name)
        
        profile_data = self.cog.profile_manager._get_profile_config(source_owner_id, source_profile_name, False)
        if not profile_data:
            return None, 0.0, 0.0, 0, f"The source for your active global profile ('{profile_name}') could not be found.", None

        temp = profile_data.get("temperature", defaultConfig.GEMINI_TEMPERATURE)
        top_p = profile_data.get("top_p", defaultConfig.GEMINI_TOP_P)
        top_k = profile_data.get("top_k", defaultConfig.GEMINI_TOP_K)
        primary_model = profile_data.get("primary_model", PRIMARY_MODEL_NAME)
        fallback_model = profile_data.get("fallback_model", FALLBACK_MODEL_NAME)
        
        warning_message = None
        system_instructions, _, _, _, _, _, _, _ = self.cog.generation_service._construct_system_instructions(user_id, profile_name, 0)
        
        user_api_key = self.cog.storage_manager._get_api_key_for_user(user_id, "gemini")
        or_key = self.cog.storage_manager._get_api_key_for_user(user_id, "openrouter")
        
        if not user_api_key and not or_key and not primary_model.upper().startswith("OLLAMA/"):
            return None, 0.0, 0.0, 0, "This feature requires a personal API key. Use `/settings` to add one.", None
        
        # Global chat runs in a DM, which can never be age-restricted -- and an
        # adult-rated profile is refused from the feature outright. Passing the
        # absent channel resolves to BLOCK_ONLY_HIGH, which is what this path
        # already sent.
        safety_settings = _resolve_safety_settings(None, profile_data)

        try:
            t_params = resolve_thinking_params(profile_data, "response")
            
            model_tools = None
            if not primary_model.upper().startswith(("OPENROUTER/", "OLLAMA/")) and "/" not in primary_model:
                model_tools = resolve_native_tools(profile_data)
                    
            model = self._instantiate_model(primary_model, None, user_id, system_instructions, safety_settings, t_params, model_tools, profile_data,
                                            config_owner_id=source_owner_id, policy_guild_id=policy_guild_id,
                                            conversation=True)
            
            return model, temp, top_p, top_k, warning_message, fallback_model
        except ValueError as e:
            # The factory's refusals -- a missing key, a data policy, Ollama -- are
            # already phrased for the user.
            return None, 0.0, 0.0, 0, str(e), None
        except Exception as e:
            print(f"Error creating model for global chat (user: {user_id}, profile: {profile_name}): {e}")
            return None, 0.0, 0.0, 0, "A critical error occurred while creating the AI model.", None
        

    async def _validate_api_keys(self, gemini_key: str, openrouter_key: str) -> Tuple[bool, str, str]:
        """Validates API keys against the REST API. Returns (is_valid, error_message, tier).

        The tier describes whether the **user's Google key has billing enabled** —
        image models are rejected on unbilled keys. It is not a user tier.

        Kept on v1alpha, as it was under the SDK.
        """
        detected_tier = "free"

        if gemini_key:
            async def _ping(model_name: str):
                client = get_google_rest_client()
                return await client.post(
                    f"/v1alpha/models/{model_name}:generateContent",
                    content=json.dumps({
                        "contents": [{"role": "user", "parts": [{"text": "ping"}]}],
                        "generationConfig": {"maxOutputTokens": 1},
                    }),
                    headers={"x-goog-api-key": gemini_key, "Content-Type": "application/json"},
                )

            try:
                # Step 1: Authentication Check (Is the key valid?)
                auth_resp = await _ping('gemini-flash-lite-latest')
                if auth_resp.status_code != 200:
                    return False, f"Google Gemini API validation failed: {auth_resp.status_code}: {auth_resp.text}", "none"

                # Step 2: Billing Detection (Does it have access to image models?)
                billing_resp = await _ping('gemini-3.1-flash-image')
                detected_tier = "paid" if billing_resp.status_code == 200 else "free"

            except Exception as e:
                return False, f"Google Gemini API validation failed: {str(e)}", "none"

        if openrouter_key:
            try:
                client = get_shared_client()
                headers = {"Authorization": f"Bearer {openrouter_key}"}
                # 5s explicitly: this used to ride on httpx.AsyncClient's own
                # default, which the shared client raises to 30s.
                response = await client.get("https://openrouter.ai/api/v1/auth/key", headers=headers, timeout=5.0)
                
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
    

    #: Official Google Gemini standard-tier pricing, per 1M tokens in USD. OpenRouter's
    #: come from its own listing on every sync.
    GOOGLE_RATES = {
        "GOOGLE/gemini-3.7-flash": {"input_1m": 0.75, "output_1m": 3.75},
        "GOOGLE/gemini-3.6-flash": {"input_1m": 1.50, "output_1m": 7.50},
        "GOOGLE/gemini-3.5-flash": {"input_1m": 1.50, "output_1m": 9.00},
        "GOOGLE/gemini-3.5-flash-lite": {"input_1m": 0.30, "output_1m": 2.50},
        "GOOGLE/gemini-3.1-pro-preview": {"input_1m": 2.00, "output_1m": 12.00},
        "GOOGLE/gemini-3.1-flash-lite": {"input_1m": 0.25, "output_1m": 1.50},
        "GOOGLE/gemini-3-flash-preview": {"input_1m": 0.50, "output_1m": 3.00},
        "GOOGLE/gemini-2.5-pro": {"input_1m": 1.25, "output_1m": 10.00},
        "GOOGLE/gemini-2.5-flash": {"input_1m": 0.30, "output_1m": 2.50},
        "GOOGLE/gemini-2.5-flash-lite": {"input_1m": 0.10, "output_1m": 0.40},
        "GOOGLE/gemini-flash-latest": {"input_1m": 1.50, "output_1m": 7.50},
        "GOOGLE/gemini-pro-latest": {"input_1m": 2.00, "output_1m": 12.00},
        "GOOGLE/gemini-flash-lite-latest": {"input_1m": 0.30, "output_1m": 2.50},
    }

    @tasks.loop(hours=24)
    async def pricing_sync_task(self):
        """Refreshes the OpenRouter catalogue and the pricing table, once at boot and daily.

        Two documented public endpoints -- the model listing, sorted by popularity because
        that order is the only ranking OpenRouter publishes, and the zero-retention
        endpoint list -- and one documented authenticated one, the listing as the bot
        owner's account sees it, which is how the catalogue tells which models some host
        serves without training. Parsing happens in a thread; the listing is ~700 KB.
        """
        try:
            os.makedirs(MODELS_DATA_DIR, exist_ok=True)
            if not self._catalogue_loaded:
                await asyncio.to_thread(self.catalogue.load)
                self._catalogue_loaded = True

            previous = self._pricing_rates or await asyncio.to_thread(self._read_pricing_file)
            rates = dict(self.GOOGLE_RATES)
            try:
                client = get_shared_client()
                models_resp = await client.get(
                    "https://openrouter.ai/api/v1/models?sort=most-popular", timeout=20.0)
                zdr_resp = await client.get("https://openrouter.ai/api/v1/endpoints/zdr", timeout=20.0)
                if models_resp.status_code != 200:
                    raise RuntimeError(f"HTTP {models_resp.status_code}")
                account_body, account_problem = await self._owner_model_listing(client)
                rates.update(await asyncio.to_thread(
                    self.catalogue.apply_listing, models_resp.content,
                    zdr_resp.content if zdr_resp.status_code == 200 else None,
                    None, account_body))
                if account_body is not None and not self.catalogue.training_current:
                    account_problem = ("the listing could not tell -- turn off providers that may "
                                       "train on inputs in the account's OpenRouter privacy settings")
                if account_problem:
                    print(f"Warning: could not tell which OpenRouter models avoid training "
                          f"({account_problem}); models not known to avoid it stay offered to "
                          "the bot owner alone.")
            except Exception as e:
                # Keep yesterday's OpenRouter prices rather than pricing every turn at zero.
                rates.update({k: v for k, v in (previous or {}).items() if k.startswith("OPENROUTER/")})
                print(f"Warning: Failed to fetch the OpenRouter catalogue: {e}")

            cache_data = {
                "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "rates": rates
            }
            await asyncio.to_thread(IOManager.write_json, cache_data, PRICING_CACHE_FILE)
            self._pricing_rates = rates

        except Exception as e:
            print(f"Error in pricing_sync_task: {e}")

    async def _owner_model_listing(self, client) -> Tuple[Optional[bytes], Optional[str]]:
        """`/models/user` as the bot owner's OpenRouter account sees it, or why there is none.

        No key is not a problem to log: that operator offers everyone else zero-retention
        models only, and their data policy screen says why.
        """
        try:
            key = self.cog.storage_manager._get_api_key_for_user(
                int(defaultConfig.DISCORD_OWNER_ID), "openrouter")
        except (TypeError, ValueError):
            return None, None
        if not key:
            return None, None
        try:
            resp = await client.get("https://openrouter.ai/api/v1/models/user?limit=1000",
                                    headers={"Authorization": f"Bearer {key}"}, timeout=20.0)
        except httpx.HTTPError as e:
            return None, type(e).__name__
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
        return resp.content, None

    @staticmethod
    def _read_pricing_file() -> Dict[str, Dict[str, float]]:
        cache = IOManager.read_json(PRICING_CACHE_FILE) or {}
        return cache.get("rates") or {}

    def _get_model_pricing(self, model_name: str) -> Tuple[float, float]:
        try:
            if self._pricing_rates is None:
                # Before the first sync lands. Read once, not per call.
                self._pricing_rates = self._read_pricing_file()
            rates = self._pricing_rates
            mapped_name = model_name
            if not mapped_name.startswith(("GOOGLE/", "OPENROUTER/", "OLLAMA/")):
                if "/" in mapped_name:
                    mapped_name = f"OPENROUTER/{mapped_name}"
                else:
                    mapped_name = f"GOOGLE/{mapped_name}"

            pricing = rates.get(mapped_name)
            if pricing:
                return float(pricing.get("input_1m", 0.0)), float(pricing.get("output_1m", 0.0))
        except Exception:
            pass
        return 0.0, 0.0

    def _calculate_turn_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        input_rate, output_rate = self._get_model_pricing(model_name)
        cost_input = (input_tokens / 1000000) * input_rate
        cost_output = (output_tokens / 1000000) * output_rate
        return cost_input + cost_output

    def turn_cost(self, meta: Dict[str, Any]) -> Tuple[float, bool]:
        """One recorded turn's cost, and whether the provider billed it or we guessed.

        The rate table is keyed on a listed model id at standard rates, so it cannot
        price a flex or priority route, a `:floor`/`:nitro` variant (neither is a listed
        id, so the lookup misses entirely and reports zero), or a cached-prompt rebate.
        OpenRouter returns what it actually charged; when that is on the turn it is the
        answer, and the estimate is only for turns from providers that report none.
        """
        billed = meta.get("cost")
        if isinstance(billed, (int, float)) and not isinstance(billed, bool):
            return float(billed), True
        return self._calculate_turn_cost(meta.get("model", "") or "",
                                         meta.get("input_tokens", 0) or 0,
                                         meta.get("output_tokens", 0) or 0), False
