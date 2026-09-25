import os
import re
import math
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
    ALLOWED_MODELS, defaultConfig,
    IMAGE_MODEL_KEYS, AUDIO_MODEL_KEYS, DEFAULT_SPEECH_VOICE,
    THINKING_LEVELS_TO_GOOGLE, THINKING_LEVELS_TO_GOOGLE_BINARY,
    THINKING_LEVELS_TO_OLLAMA, GEMINI_FREE_TIER_BLOCKED, OLLAMA_OWNER_ONLY,
    OPENROUTER_DATA_POLICY_BLOCKED, IMAGE_MODEL_NO_OLLAMA, SPEECH_MODEL_NO_OLLAMA,
    OPENROUTER_SERVER_TOOLS,
    API_KEY_COOLING_DOWN, NO_MODEL_SET, KEY_CHECK_MODEL_MISSING,
)
from ..utils.data_policy import openrouter_data_collection
from ..managers.storage_manager import IOManager
from ..utils.blob_stream import InlineBlobExtractor, is_blob_sentinel, sentinel_path
from ..utils.helpers import (_format_api_error, _resolve_safety_settings,
                            billable_output_tokens, is_real_model,
                            google_thinking_caps, resolve_image_output_params,
                            resolve_image_tools, resolve_media_resolution,
                            resolve_native_tools, resolve_openrouter_image_detail,
                            resolve_openrouter_endpoint, resolve_openrouter_service_tier,
                            resolve_thinking_params, system_model)
from ..utils.http_client import get_shared_client
from ..utils.user_defaults import model_chain
from ..utils.net_guard import safe_stream
from ..utils.memory_tuning import maybe_trim_malloc
from ..utils import mem_probe

# The adapters moved to `services/api/`; these names are re-exported because
# MimicCog, profile_manager, memory_manager, media_service, tools_service and
# help_service all import them from here.
from .api.embeddings import get_embedding_vector
from .api.google_rest import (
    GoogleRESTModel, GoogleRESTResponse, GoogleSpeechModel, close_google_rest_client,
    generate_google_tts_audio, get_google_rest_client, materialise_inline_data,
)
from .api.ollama import OllamaModel, OllamaResponse
from .api.function_calls import as_google_tool
from .generation.tool_loop import declarations, declared_on, functions_for
from .api.openrouter import OpenRouterModel
from .api.openrouter_catalogue import BROWSE_POPULAR, OpenRouterCatalogue
from .api.openrouter_endpoints import ENDPOINTS_URL, Endpoint, base_model_id, parse_endpoints
from .api.openrouter_image_catalogue import OpenRouterImageCatalogue
from .api.openrouter_images import OpenRouterImageModel
from .api.openrouter_speech import OpenRouterSpeechModel
from .api.openrouter_speech_catalogue import OpenRouterSpeechCatalogue
from .api.rest_view import _BlobRef, _EnumStr, _RestView

# A rate-limited model rests on the key that was refused, so the next turn does not send
# it straight back into the same 429. `cog.api_key_cooldowns` maps (key, model id) to when
# the rest ends, and `_instantiate_model` is the one place that reads it.
#
# Per model, not per key: Google counts its quotas per model, and an OpenRouter 429 is most
# often one model's host turning traffic away. A whole-key rest also took the fallback down
# with the primary -- the fallback is built after the primary fails, so it found its key
# already resting, in exactly the case a fallback exists for.
#
# The provider's own hint sets the length when it gives one: Google's `retryDelay`, or the
# reset time OpenRouter reports for its own limits. A daily quota rests the longest, because
# nothing will be accepted before it resets.
_RATE_LIMIT_REST_DEFAULT = 10.0
_RATE_LIMIT_REST_MIN = 2.0
_RATE_LIMIT_REST_MAX = 600.0

_GOOGLE_RETRY_DELAY = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')
# Quoted either way: OpenRouter's error body arrives as JSON text, or as the repr of a dict
# when the error came inside a 200.
_OPENROUTER_RATE_LIMIT_RESET = re.compile(r'''["']X-RateLimit-Reset["']\s*:\s*["']?(\d+)''', re.IGNORECASE)
# What an upstream host asks for when the 429 is its own rather than OpenRouter's, which
# is most of them on a free model: `retry_after_seconds`, and the same number again in a
# `Retry-After` header OpenRouter passes through. Either spelling, whichever comes first.
_UPSTREAM_RETRY_AFTER = re.compile(
    r'''["'](?:retry_after_seconds|Retry-After)["']\s*:\s*["']?(\d+(?:\.\d+)?)''', re.IGNORECASE)


def _rate_limit_rest_seconds(error: BaseException, now: Optional[float] = None) -> Optional[float]:
    """How long to rest a model after `error`, or None when it was not a rate limit."""
    err_str = str(error)
    if "429" not in err_str and "RESOURCE_EXHAUSTED" not in err_str:
        return None
    if "PerDay" in err_str:
        return _RATE_LIMIT_REST_MAX
    rest = _RATE_LIMIT_REST_DEFAULT
    if match := _GOOGLE_RETRY_DELAY.search(err_str):
        rest = float(match.group(1))
    elif match := _OPENROUTER_RATE_LIMIT_RESET.search(err_str):
        reset = int(match.group(1))
        # Milliseconds since the epoch; seconds are accepted too.
        reset_at = reset / 1000 if reset > 10_000_000_000 else reset
        rest = reset_at - (time.time() if now is None else now)
    elif match := _UPSTREAM_RETRY_AFTER.search(err_str):
        # Floored at the default, never shortened by it: the host is answering for
        # itself, and the pool behind a free model is shared with everyone else asking
        # the same question. A hint longer than the default is taken at its word.
        rest = max(float(match.group(1)), _RATE_LIMIT_REST_DEFAULT)
    return min(max(rest, _RATE_LIMIT_REST_MIN), _RATE_LIMIT_REST_MAX)


def _rest_model_on_rate_limit(cog, api_key: Optional[str], model_id: str, error: BaseException) -> None:
    if not api_key:
        return
    rest = _rate_limit_rest_seconds(error)
    if rest is not None:
        cog.api_key_cooldowns[(api_key, model_id)] = time.time() + rest


def _rest_ends(cog, api_key: str, model_id: str) -> float:
    """When `model_id` may next be sent on `api_key`, or 0.0 if it may be sent now."""
    ends = cog.api_key_cooldowns.get((api_key, model_id), 0.0)
    if ends and ends <= time.time():
        cog.api_key_cooldowns.pop((api_key, model_id), None)
        return 0.0
    return ends


def _with_key_cooldown_tracking(cog, model, api_key: str, model_id: str,
                                method: str = "generate_content_async"):
    """Wraps the model's call method so a rate-limit response rests this model on the key.

    `method` is `generate_content_async` for every adapter but speech, whose one call is
    `synthesise`.
    """
    original = getattr(model, method)

    async def _tracked(*args, **kwargs):
        try:
            return await original(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _rest_model_on_rate_limit(cog, api_key, model_id, e)
            raise

    setattr(model, method, _tracked)
    return model


def _refusal(message: str) -> ValueError:
    """A refusal already phrased for the user, carried whole as `formatted_reason`:
    `_format_api_error` would cut a plain message at 80 characters."""
    error = ValueError(message)
    error.formatted_reason = message
    return error


class MissingKeyError(ValueError):
    """No key for this provider where the call would be billed, raised before any request.

    Its own type so `run_with_fallback` can tell it apart: a Final Fallback on a
    provider nobody here holds is skipped this way on every call, and its "key not found"
    must not stand in for the failure that actually ended the chain.
    """


def _refuse_if_resting(cog, api_key: str, model_id: str) -> None:
    ends = _rest_ends(cog, api_key, model_id)
    if ends:
        error = _refusal(API_KEY_COOLING_DOWN.format(model=model_id, ends=f"<t:{math.ceil(ends)}:R>"))
        #: Read by callers that treat a rate limit as routine rather than as a fault.
        error.rate_limited = True
        raise error


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
        #: The image models the same sync lists -- see api/openrouter_image_catalogue. Loaded
        #: and refreshed with the text catalogue; `_instantiate_model` reads it to decide
        #: whether an OpenRouter image model may serve a server at all.
        self.image_catalogue = OpenRouterImageCatalogue(MODELS_DATA_DIR)
        #: The speech models the same sync lists -- see api/openrouter_speech_catalogue. Read
        #: by the factory for the same decision, and for the voices each model takes.
        self.speech_catalogue = OpenRouterSpeechCatalogue(MODELS_DATA_DIR)
        self._catalogue_loaded = False
        #: The pricing table, held in memory. `_get_model_pricing` used to re-read and
        #: parse the whole file per call, and the audit screens call it once per turn.
        self._pricing_rates: Optional[Dict[str, Dict[str, float]]] = None
        #: host -> (answered, model names) from each Ollama host's last probe, oldest
        #: first. Shared by every picker -- see probe_ollama.
        self._ollama_probes: "OrderedDict[str, Tuple[bool, List[str]]]" = OrderedDict()
        #: model id -> (monotonic time read, endpoints), oldest first. Read on demand by
        #: the Hosts screen -- see openrouter_endpoints.
        self._endpoint_listings: "OrderedDict[str, Tuple[float, Tuple[Endpoint, ...]]]" = OrderedDict()

    def _instantiate_model(self, raw_model_name: str, guild_id, user_id, system_instruction=None, safety_settings=None, thinking_params=None, tools=None, profile_settings=None, openrouter_key_error: str = None, google_key_error: str = None, use_broad_openrouter_heuristic: bool = True, config_owner_id=None, policy_guild_id=None, conversation: bool = False, image_config=None, speech: bool = False, functions=()):
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

        `tools` are the native tools in Google's spelling (`google_search`, `url_context`);
        OpenRouter is sent its own server tools for the same jobs (OPENROUTER_SERVER_TOOLS)
        and Ollama nothing. `functions` are the character's own -- the `tool_loop.functions_for`
        tuple the caller also handed the prompt builder -- declared on every provider that
        can carry one, which is why they are a separate argument rather than one list to
        be sorted out downstream. Ollama takes neither. The model keeps the tuple as
        `model.functions`, and `tool_loop.run` answers exactly that set.

        `functions` sits last, away from the `tools` it belongs beside, because a dozen
        callers pass the first eight arguments positionally: inserting a parameter in the
        middle silently rebinds `profile_settings` at every one of them.

        `image_config` makes it an image model: the profile's image output and sampling
        keys, resolved here for the model named. Every image path builds through here
        (MediaService.build_image_model), because an OpenRouter image request cannot carry
        `data_collection` -- the Image API takes no such field. Where a text model is told to
        deny training hosts, an image model is refused outright unless the image catalogue
        knows its one host does not train; an id the catalogue does not list is refused
        too, since there is no per-request deny to fall back on.

        `speech=True` makes it a speech model: an adapter whose `synthesise(transcript,
        directed_prompt, voice_name, temperature)` returns the path of an audio file. Built
        here for the reason images are -- the key gate and the rate-limit cooldown live in
        this one place, and speech that resolved its own key was subject to neither. An
        OpenRouter speech model is judged as an image model is: its endpoint takes no
        `data_collection` either, so the speech catalogue decides.
        """
        if speech and image_config is not None:
            raise TypeError("a model is an image model or a speech model, not both")
        if not is_real_model(raw_model_name):
            # A profile made before its owner chose a provider has none. MissingKeyError,
            # so a Final Fallback behind it is still tried and this is not what is reported.
            raise MissingKeyError(NO_MODEL_SET)
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
            if not api_key: raise MissingKeyError(openrouter_key_error or "OpenRouter API Key not found. Run `/start` to set one up, or `/settings` if you know your way around.")
            _refuse_if_resting(self.cog, api_key, actual_name)
            data_collection = (openrouter_data_collection(
                self.cog.server_manager._get_server_index(str(policy_guild)) if policy_guild else None)
                if gated else None)
            if speech:
                # No `data_collection` to send here either, so the model itself is judged.
                if data_collection == "deny" and not self.speech_catalogue.is_open(actual_name):
                    raise _refusal(OPENROUTER_DATA_POLICY_BLOCKED)
                model = OpenRouterSpeechModel(actual_name, api_key,
                                              voices=self.speech_catalogue.voices(actual_name),
                                              clones=self.speech_catalogue.clones(actual_name))
                return _with_key_cooldown_tracking(self.cog, model, api_key, actual_name, "synthesise")
            if image_config is not None:
                # No `data_collection` to send, so the model itself is judged -- see above.
                if data_collection == "deny" and not self.image_catalogue.is_open(actual_name):
                    raise _refusal(OPENROUTER_DATA_POLICY_BLOCKED)
                model = OpenRouterImageModel(
                    actual_name, api_key=api_key, system_instruction=system_instruction,
                    image_params=resolve_image_output_params(image_config, f"OPENROUTER/{actual_name}"))
                return _with_key_cooldown_tracking(self.cog, model, api_key, actual_name)
            model = OpenRouterModel(actual_name, api_key=api_key, system_instruction=system_instruction, thinking_params=t_params, image_detail=image_detail, service_tier=service_tier, data_collection=data_collection,
                                    endpoint=resolve_openrouter_endpoint(p_settings, actual_name),
                                    tools=declarations(functions),
                                    server_tools=[OPENROUTER_SERVER_TOOLS[k] for t in tools or ()
                                                  for k in t if k in OPENROUTER_SERVER_TOOLS])
            model.functions = tuple(functions)
            return _with_key_cooldown_tracking(self.cog, model, api_key, actual_name)
        elif is_ollama:
            if image_config is not None:
                raise ValueError(IMAGE_MODEL_NO_OLLAMA)
            if speech:
                raise ValueError(SPEECH_MODEL_NO_OLLAMA)
            if not self.cog.profile_manager.may_use_ollama(config_owner_id):
                raise ValueError(OLLAMA_OWNER_ONLY)
            ollama_host = p_settings.get("ollama_host_url", OLLAMA_LOCAL_URL)
            return OllamaModel(actual_name, api_url=ollama_host, system_instruction=system_instruction, thinking_params=t_params)
        else:
            if guild_id:
                api_key = storage._get_api_key_for_guild(guild_id)
                if not api_key and storage.gemini_blocked_for_guild(guild_id):
                    raise _refusal(GEMINI_FREE_TIER_BLOCKED)
            else:
                if conversation and not storage.personal_gemini_allowed_in_conversation(user_id, policy_guild):
                    raise _refusal(GEMINI_FREE_TIER_BLOCKED)
                api_key = storage._get_api_key_for_user(user_id)
            if not api_key: raise MissingKeyError(google_key_error or "Google API Key not found. Run `/start` to set one up, or `/settings` if you know your way around.")
            _refuse_if_resting(self.cog, api_key, actual_name)
            if speech:
                return _with_key_cooldown_tracking(
                    self.cog, GoogleSpeechModel(actual_name, api_key), api_key, actual_name, "synthesise")
            if image_config is not None:
                # No thinking config and no mediaResolution, which an image request rejects
                # or ignores: its options and search tool are resolved for this model instead.
                model = GoogleGenAIModel(api_key=api_key, model_name=actual_name,
                                         system_instruction=system_instruction,
                                         safety_settings=safety_settings,
                                         tools=resolve_image_tools(image_config, raw_model_name),
                                         image_params=resolve_image_output_params(image_config, raw_model_name))
            else:
                # The native tools and the declarations ride in the same `tools` array
                # here -- Google takes one list holding both kinds -- which is exactly
                # the merge the two arguments exist to keep out of the callers.
                google_tools = list(tools or ())
                declared = as_google_tool(declarations(functions))
                if declared:
                    google_tools.append(declared)
                model = GoogleGenAIModel(api_key=api_key, model_name=actual_name, system_instruction=system_instruction, safety_settings=safety_settings, thinking_params=t_params, tools=google_tools or None, media_resolution=media_res)
                model.functions = tuple(functions)
            return _with_key_cooldown_tracking(self.cog, model, api_key, actual_name)

    async def run_with_fallback(self, primary: str, fallback, attempt,
                                *, label: str = "utility"):
        """Runs one utility generation on `primary`, retrying once on `fallback`.

        `fallback` is a name, None, or a tuple of names tried in order -- a category's
        Fallback then its Final Fallback (see `model_chain`), or the attachment
        describer's free model, then the same model paid, then Google.

        `attempt(model_name, is_fallback)` owns the construction and the response
        handling; this owns only which name to try and when to stop. That puts the five
        utility paths -- critic, grounding, LTM, image and speech -- on one retry policy
        without forcing them into one call shape, which they genuinely do not share: one
        returns an audio file's path, one drives a heartbeat, three return a candidate list.

        Only exceptions retry. An empty or safety-blocked response is a decision about
        the content, not a statement about the model being unavailable, and re-rolling
        it on a second model spends another call to be refused again. An exception marked
        `retryable = False` is that same decision raised rather than returned -- speech
        has no response object to carry a block reason -- and is re-raised at once.

        A fallback equal to the primary is skipped rather than tried twice, which is
        what makes the shipped defaults cost nothing: every utility fallback defaults to
        the same model as its primary, so the retry only becomes live once someone
        actually changes one of them.

        Returns (result, model_used, used_fallback). If every model fails the last error
        is raised -- callers report the error they are handed, and the last fallback's is
        the one that actually ended the attempt. A MissingKeyError never replaces an
        earlier error: it only says a provider was absent, not why the chain failed.
        """
        attempts = [(primary, False)]
        for name in (fallback if isinstance(fallback, tuple) else (fallback,)):
            if is_real_model(name) and all(name != tried for tried, _ in attempts):
                attempts.append((name, True))

        last_error = None
        for i, (name, is_fallback) in enumerate(attempts):
            try:
                return await attempt(name, is_fallback), name, is_fallback
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if getattr(e, "retryable", True) is False:
                    raise
                if last_error is None or not isinstance(e, MissingKeyError):
                    last_error = e
                if i + 1 < len(attempts):
                    # The phrased reason, not the exception: a provider's error body is a
                    # JSON document, and a journal on a 1 GB box is not where it belongs.
                    print(f"{label}: {'fallback' if is_fallback else 'primary'} '{name}' failed "
                          f"({type(e).__name__}: {_format_api_error(e)}); "
                          f"retrying on '{attempts[i + 1][0]}'.")
        raise last_error

    def model_chain(self, config: Optional[Dict[str, Any]], primary_key: str,
                    owner_id: Optional[int]) -> Tuple[str, Tuple[str, ...]]:
        """`user_defaults.model_chain` under the provider preference of `config`'s owner."""
        return model_chain(config, primary_key,
                           self.cog.profile_manager.provider_preference(owner_id))

    def get_top_models(self, provider: str, target_config_key: str,
                       ollama_host: Optional[str] = None) -> List[str]:
        if target_config_key in IMAGE_MODEL_KEYS:
            if provider == 'openrouter':
                return self.image_catalogue.browse(BROWSE_POPULAR, show_training=False)[0]
            return list(get_args(IMAGE_MODELS))
        if target_config_key in AUDIO_MODEL_KEYS:
            if provider == 'openrouter':
                return self.speech_catalogue.browse(BROWSE_POPULAR, show_training=False)[0]
            return list(get_args(AUDIO_MODELS))
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

    #: Endpoint listings kept, and for how long one is shown before it is asked again.
    #: Short, because the screen quotes uptime; few, because pins are set on a handful.
    _ENDPOINT_LISTINGS_KEPT = 32
    _ENDPOINT_LISTING_TTL = 600.0
    #: A model id as it may appear in the listing's path. A typed custom id can hold
    #: anything, and this one is interpolated into a URL.
    _LISTABLE_MODEL_ID = re.compile(r"[\w.-]+/[\w.-]+")

    def cached_openrouter_endpoints(self, model_id: str) -> Optional[Tuple[Endpoint, ...]]:
        """The last listing for `model_id` while it is still fresh, else None. Never asks."""
        entry = self._endpoint_listings.get(base_model_id(model_id))
        if entry is None or time.monotonic() - entry[0] > self._ENDPOINT_LISTING_TTL:
            return None
        return entry[1]

    async def openrouter_endpoints(self, model_id: str) -> Optional[Tuple[Endpoint, ...]]:
        """Every endpoint OpenRouter lists for one text model; None if it could not say.

        One small request per model, asked when the Hosts screen opens rather than for
        every model in the daily sync -- see api/openrouter_endpoints.
        """
        key = base_model_id(model_id)
        cached = self.cached_openrouter_endpoints(key)
        if cached is not None:
            return cached
        if not self._LISTABLE_MODEL_ID.fullmatch(key):
            return None
        try:
            resp = await get_shared_client().get(ENDPOINTS_URL.format(key), timeout=10.0)
        except httpx.HTTPError:
            return None
        endpoints = parse_endpoints(resp.content) if resp.status_code == 200 else None
        if endpoints is None:
            return None
        self._endpoint_listings[key] = (time.monotonic(), endpoints)
        self._endpoint_listings.move_to_end(key)
        while len(self._endpoint_listings) > self._ENDPOINT_LISTINGS_KEPT:
            self._endpoint_listings.popitem(last=False)
        return endpoints

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

        p_settings = self.cog.profile_manager._get_profile_config(profile_owner_id_for_instructions, profile_name_for_instructions, is_borrowed) or {}
        # One tuple for the prompt and the model, so the character is told about
        # exactly what it is sent. Every caller of this runs `tool_loop.run`.
        functions = functions_for(p_settings, has_server=bool(guild_id))

        current_instructions, error_in_instr_constr, _, temperature, top_p, top_k, primary_model, fallback_model = self.cog.generation_service._construct_system_instructions(
            profile_owner_id_for_instructions,
            profile_name_for_instructions,
            channel_id,
            training_examples_list=training_examples_list,
            functions=functions,
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
            # A model declaring other functions than the prompt just described is not
            # the same model, whatever the key says: the character would be told about
            # one set and sent another.
            cached_functions = declared_on(self.cog.channel_models[model_cache_key][0])
            if last_profile_key == current_profile_key_for_model and cached_functions == functions:
                 recreate_model = False 
            
        if recreate_model and model_cache_key in self.cog.channel_models:
            del self.cog.channel_models[model_cache_key]
            self.cog.channel_model_last_profile_key.pop(model_cache_key, None)
        
        if model_cache_key in self.cog.channel_models and not recreate_model:
            model_instance, model_init_error_state, cached_model_name = self.cog.channel_models[model_cache_key]
            # The instruction built above, not the one the model was cached with. That
            # one carried the <current_time> and <birthday_context> of whenever the entry
            # was made, so a whisper answered on a clock frozen at the first whisper and
            # never learnt of a birthday that came round, or was set, after it. Every
            # adapter reads this attribute when it builds a request.
            if model_instance is not None:
                model_instance.system_instruction = current_instructions
            return model_instance, model_init_error_state, temperature, top_p, top_k, warning_message, fallback_model

        model_instance, model_init_error = None, True
        
        dynamic_safety_settings = _resolve_safety_settings(channel, p_settings)

        model_to_create = primary_model
        
        # Extract parameters once for either provider
        p_sett_thinking = p_settings
        t_params = resolve_thinking_params(p_sett_thinking, "response")

        model_tools = resolve_native_tools(p_sett_thinking)

        try:
            model_instance = self._instantiate_model(model_to_create, guild_id, profile_owner_id_for_instructions, current_instructions, dynamic_safety_settings, t_params, model_tools, p_sett_thinking, config_owner_id=profile_owner_id_for_instructions, functions=functions)
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
                model_instance = self._instantiate_model(model_to_create, guild_id, profile_owner_id_for_instructions, current_instructions, dynamic_safety_settings, t_params_fb, model_tools, p_sett_thinking, config_owner_id=profile_owner_id_for_instructions, functions=functions)
                model_init_error = False
            except Exception as e2:
                return None, True, temperature, top_p, top_k, f"Model Initialization Error: Failed to load Primary ('{primary_model}') and Fallback ('{fallback_model}') models. Check your API key.", fallback_model
        
        final_error_state = error_in_instr_constr or model_init_error
        self.cog.channel_models[model_cache_key] = (model_instance, final_error_state, model_to_create)
        self.cog.channel_model_last_profile_key[model_cache_key] = current_profile_key_for_model
        return model_instance, final_error_state, temperature, top_p, top_k, warning_message, fallback_model

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

            # Both are `/mod`'s to move, Google-only, and stored with the routing prefix.
            auth_model, tier_model = (system_model(self.cog, k).removeprefix("GOOGLE/")
                                      for k in ("key_check_model", "key_tier_probe_model"))
            try:
                # Step 1: Authentication Check (Is the key valid?)
                auth_resp = await _ping(auth_model)
                if auth_resp.status_code == 404:
                    return False, KEY_CHECK_MODEL_MISSING.format(model=auth_model), "none"
                if auth_resp.status_code != 200:
                    return False, f"Google Gemini API validation failed: {auth_resp.status_code}: {auth_resp.text}", "none"

                # Step 2: Billing Detection (Does it have access to image models?) An
                # unbilled key is refused with a 400 or a 429; a 404 is the probe model
                # gone, which used to read as "free" and so turned every key away.
                billing_resp = await _ping(tier_model)
                if billing_resp.status_code == 404:
                    return False, KEY_CHECK_MODEL_MISSING.format(model=tier_model), "none"
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
                await asyncio.to_thread(self.image_catalogue.load)
                await asyncio.to_thread(self.speech_catalogue.load)
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

            # After the text listing, because its training check is what lets the image
            # listing mean anything -- see api/openrouter_image_catalogue. A failure here
            # leaves yesterday's image models in place and touches no text price.
            try:
                await self._sync_image_catalogue(get_shared_client())
            except Exception as e:
                print(f"Warning: Failed to fetch the OpenRouter image catalogue: {e}")
            # Likewise after the text listing, and independent of the image one.
            try:
                await self._sync_speech_catalogue(get_shared_client())
            except Exception as e:
                print(f"Warning: Failed to fetch the OpenRouter speech catalogue: {e}")

            cache_data = {
                "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "rates": rates
            }
            await asyncio.to_thread(IOManager.write_json, cache_data, PRICING_CACHE_FILE)
            self._pricing_rates = rates

        except Exception as e:
            print(f"Error in pricing_sync_task: {e}")

    #: Endpoint lookups the image and speech syncs each run at once: a few dozen small
    #: requests a day, spread thin enough never to crowd a turn out of the shared client's
    #: connection pool.
    _ENDPOINT_CONCURRENCY = 4

    async def _endpoint_bodies(self, client, model_ids: List[str], url: str) -> Dict[str, Optional[bytes]]:
        """Each model's endpoints body, from `url` with its id filled in; None where it failed."""
        gate = asyncio.Semaphore(self._ENDPOINT_CONCURRENCY)

        async def fetch(model_id: str):
            async with gate:
                try:
                    resp = await client.get(url.format(model_id), timeout=20.0)
                except httpx.HTTPError:
                    return model_id, None
            return model_id, (resp.content if resp.status_code == 200 else None)

        return dict(await asyncio.gather(*(fetch(m) for m in model_ids)))

    async def _sync_image_catalogue(self, client) -> None:
        """Refreshes the OpenRouter image catalogue from its four documented endpoints."""
        models_resp = await client.get("https://openrouter.ai/api/v1/images/models", timeout=20.0)
        if models_resp.status_code != 200:
            raise RuntimeError(f"HTTP {models_resp.status_code}")
        ranking_resp = await client.get(
            "https://openrouter.ai/api/v1/models?output_modalities=image&sort=most-popular", timeout=20.0)
        model_ids = [m["id"] for m in (json.loads(models_resp.content).get("data") or [])
                     if isinstance(m, dict) and m.get("id")]
        endpoint_bodies = await self._endpoint_bodies(
            client, model_ids, "https://openrouter.ai/api/v1/images/models/{}/endpoints")
        account_body, _problem = await self._owner_model_listing(client, "output_modalities=image&limit=1000")
        await asyncio.to_thread(
            self.image_catalogue.apply_listing, models_resp.content,
            ranking_resp.content if ranking_resp.status_code == 200 else None,
            endpoint_bodies, account_body, self.catalogue.training_current)

    async def _sync_speech_catalogue(self, client) -> None:
        """Refreshes the OpenRouter speech catalogue from its three documented endpoints.

        The listing is sorted by popularity, so it is the ranking too. Run after the text
        sync, whose training check is what lets the speech account listing mean anything.
        """
        models_resp = await client.get(
            "https://openrouter.ai/api/v1/models?output_modalities=speech&sort=most-popular", timeout=20.0)
        if models_resp.status_code != 200:
            raise RuntimeError(f"HTTP {models_resp.status_code}")
        model_ids = [m["id"] for m in (json.loads(models_resp.content).get("data") or [])
                     if isinstance(m, dict) and m.get("id")]
        endpoint_bodies = await self._endpoint_bodies(
            client, model_ids, "https://openrouter.ai/api/v1/models/{}/endpoints")
        account_body, _problem = await self._owner_model_listing(client, "output_modalities=speech&limit=1000")
        await asyncio.to_thread(
            self.speech_catalogue.apply_listing, models_resp.content, endpoint_bodies,
            account_body, self.catalogue.training_current)

    async def _owner_model_listing(self, client, query: str = "limit=1000") -> Tuple[Optional[bytes], Optional[str]]:
        """`/models/user` as the bot owner's OpenRouter account sees it, or why there is none.

        `query` picks what it lists. OpenRouter lists text models unless told otherwise, and
        the image sync asks for `output_modalities=image`.

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
            resp = await client.get(f"https://openrouter.ai/api/v1/models/user?{query}",
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
                                         billable_output_tokens(meta)), False
