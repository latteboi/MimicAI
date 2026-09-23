"""Drafting a character from a concept: the prompt, the parse, the model and the save.

`/profile generate` is reached three ways -- the command itself, the /start wizard and the
Public Library's Generate button -- and all three arrive at the command's callback, which
drafts through here and shows the draft (`gui_generate.GeneratedProfileView`) before
anything is written. A draft lives only on that view: nothing touches disk until Save.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from ..utils.constants import (DEFAULT_PROFILE_GENERATOR_PROMPT, DEFAULT_SAFETY_SETTINGS,
                               LIBRARY_INTRO_MAX_CHARS, defaultConfig)
from ..utils.helpers import _format_api_error, resolve_thinking_params
from ..utils.user_defaults import (apply_defaults, model_chain, model_provider,
                                   model_slot_defaults)

#: A character is creative work. This was 0.3, a form-filling temperature, and every
#: concept came back as the same genre-average character.
GENERATOR_TEMPERATURE = 0.9

#: Persona section -> the header that carries it, in the persona modal's order.
PERSONA_SECTIONS = {
    "backstory": "persona_backstory",
    "personality_traits": "persona_personality_traits",
    "likes": "persona_likes",
    "dislikes": "persona_dislikes",
    "appearance": "persona_appearance",
}

SECTIONS = ("display_name", "library_intro", "placeholder_emoji",
            *PERSONA_SECTIONS.values(), "ai_instructions")

#: AppearanceModal's input cap, which is what a user editing the name afterwards meets.
DISPLAY_NAME_MAX_CHARS = 20

#: Webhook usernames Discord refuses, plus the two mentions AppearanceModal refuses.
_RESERVED_NAME_PARTS = ("clyde", "discord", "@everyone", "@here")

NO_KEY_MESSAGE = ("A personal API key is not configured, so I cannot generate a profile. "
                  "Run `/start` to add one.")

#: Appended after the operator's template rather than written into it, so a customised
#: PROFILE_GENERATOR prompt -- which may only use `{prompt}` -- can still be refined.
REVISION_TEMPLATE = (
    "\n\nThis is a revision. Your previous draft is below, followed by what the user wants "
    "changed. Write the complete profile again in the same section format, changing what "
    "the request asks for and keeping everything else.\n\n"
    "<previous_draft>\n{draft}\n</previous_draft>\n\n"
    "<revision_request>\n{request}\n</revision_request>"
)

_HEADER = re.compile(r"\[SECTION:\s*([A-Za-z_]+)\s*\]")
#: A line that is only decoration: what is left of `**[SECTION:x]**` or `## [SECTION:x]`
#: once the header is split out, or a code fence around the whole reply.
_DECORATION = re.compile(r"^(?:[-=`*#_\s]*|```[\w-]*)$")
_BULLET = re.compile(r"^(?:[-*•]|\d+[.)])\s+")


class ProfileGenerationError(Exception):
    """A draft could not be produced. The message is written for the user."""

    #: Read by `run_with_fallback`: False stops it trying the fallback model.
    retryable = True


def parse_sections(text: str) -> Dict[str, str]:
    """The known sections of a generated reply, keyed by header, one item per line.

    Tolerant of the decoration models add despite being told not to -- bold or heading
    marks around a header, a fence around the reply, a bullet before each line -- and of
    missing sections, which an operator's older template will not ask for at all.
    """
    parts = _HEADER.split(text or "")
    sections: Dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        key = parts[i].lower()
        if key not in SECTIONS:
            continue
        lines = [_BULLET.sub("", line.strip()) for line in parts[i + 1].splitlines()
                 if not _DECORATION.match(line)]
        sections[key] = "\n".join(line for line in lines if line)
    return sections


def clean_display_name(value: Optional[str]) -> Optional[str]:
    """The first line, or None. Over the cap is None rather than cut: trimmed to whole
    words, "Vesper Crane the Unbroken" became "Vesper Crane the"."""
    lines = (value or "").strip().splitlines()
    name = lines[0].strip().strip("\"'“”") if lines else ""
    if not name or len(name) > DISPLAY_NAME_MAX_CHARS:
        return None
    if any(part in name.lower() for part in _RESERVED_NAME_PARTS):
        return None
    return name


def clean_emoji(value: Optional[str]) -> Optional[str]:
    """One unicode emoji or None. Custom `<:name:id>` emoji are the user's to pick."""
    tokens = (value or "").split()
    token = tokens[0] if tokens else ""
    if not token or len(token) > 10 or any(c.isascii() for c in token):
        return None
    return token


def clean_intro(value: Optional[str]) -> Optional[str]:
    """Collapsed to one paragraph and cut at a word inside LIBRARY_INTRO_MAX_CHARS."""
    text = " ".join((value or "").split())
    if len(text) > LIBRARY_INTRO_MAX_CHARS:
        text = text[:LIBRARY_INTRO_MAX_CHARS - 1].rsplit(" ", 1)[0].rstrip(",;:- ") + "…"
    return text or None


def draft_from_sections(sections: Dict[str, str]) -> Dict[str, Any]:
    return {
        "display_name": clean_display_name(sections.get("display_name")),
        "library_intro": clean_intro(sections.get("library_intro")),
        "placeholder_emoji": clean_emoji(sections.get("placeholder_emoji")),
        "persona": {key: sections.get(header, "") for key, header in PERSONA_SECTIONS.items()},
        "ai_instructions": sections.get("ai_instructions", ""),
    }


def render_draft(draft: Dict[str, Any]) -> str:
    """A draft written back out in the section format the model produced it in."""
    values = {"display_name": draft.get("display_name"),
              "library_intro": draft.get("library_intro"),
              "placeholder_emoji": draft.get("placeholder_emoji"),
              "ai_instructions": draft.get("ai_instructions")}
    for key, header in PERSONA_SECTIONS.items():
        values[header] = (draft.get("persona") or {}).get(key)
    return "\n".join(f"[SECTION:{s}]\n{values[s] or ''}" for s in SECTIONS)


def build_prompt(template: str, concept: str, previous: Optional[Dict[str, Any]] = None,
                 request: Optional[str] = None) -> str:
    prompt = template.format(prompt=concept)
    if previous is not None:
        prompt += REVISION_TEMPLATE.format(draft=render_draft(previous),
                                           request=(request or "").strip())
    return prompt


def generator_models(cog, user_id: int) -> Tuple[str, Tuple[str, ...]]:
    """(primary, fallbacks) for a draft: the chain the new profile itself would run on.

    Built the way `_get_or_create_user_profile` builds a config -- the shipped models on the
    user's effective provider, their own defaults over them -- rescued onto a provider
    they hold, then the Final Fallback behind. Only models on a provider the user holds
    a key for are kept: a draft is billed to the user alone.
    """
    pm = cog.profile_manager
    provider = pm.effective_provider(user_id)
    shipped = model_slot_defaults(provider)
    config = {"primary_model": shipped["primary_model"], "fallback_model": shipped["fallback_model"]}
    apply_defaults(config, pm._get_user_defaults(user_id), borrowed=False)
    pm._rescue_unusable_models(user_id, config)

    primary, fallbacks = model_chain(config, "primary_model", provider)
    usable: List[str] = [m for m in dict.fromkeys((primary, *fallbacks))
                         if pm._user_holds_provider_key(user_id, model_provider(m))]
    if not usable:
        raise ProfileGenerationError(NO_KEY_MESSAGE)
    return usable[0], tuple(usable[1:])


async def generate_draft(cog, user_id: int, concept: str,
                         previous: Optional[Dict[str, Any]] = None,
                         request: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    """(draft, model used). `previous` and `request` together make it a refinement.

    Raises ProfileGenerationError, and nothing else, for anything the user should be told.
    """
    primary, fallbacks = generator_models(cog, user_id)
    template = cog.global_prompts.get("PROFILE_GENERATOR", DEFAULT_PROFILE_GENERATOR_PROMPT)
    try:
        prompt = build_prompt(template, concept, previous, request)
    except (KeyError, IndexError, ValueError):
        raise ProfileGenerationError(
            "The profile generator prompt on this bot is misconfigured. Let the bot operator know.")

    async def attempt(model_name: str, is_fallback: bool) -> Dict[str, Any]:
        model = cog.api_service._instantiate_model(
            model_name, None, user_id, None, DEFAULT_SAFETY_SETTINGS,
            resolve_thinking_params(None, "utility"), config_owner_id=user_id)
        response = await model.generate_content_async(
            [prompt], generation_config={"temperature": GENERATOR_TEMPERATURE})
        if not response or not response.candidates:
            # A decision about the concept, not the model being down: the fallback
            # would be asked the same thing, so this is not retried.
            blocked = ProfileGenerationError(
                "The model returned nothing, which usually means a safety filter refused "
                "the concept. Try rewording it.")
            blocked.retryable = False
            raise blocked
        try:
            text = response.text or ""
        except Exception:
            text = ""
        draft = draft_from_sections(parse_sections(text))
        if not draft["persona"]["personality_traits"]:
            raise ProfileGenerationError(
                "The model's reply was not in the expected format. Press Regenerate to try again.")
        return draft

    try:
        draft, used, _was_fallback = await cog.api_service.run_with_fallback(
            primary, fallbacks, attempt, label="Profile generator")
    except ProfileGenerationError:
        raise
    except Exception as e:
        raise ProfileGenerationError(_format_api_error(e)) from e
    return draft, used


def save_draft(cog, user_id: int, profile_name: str, draft: Dict[str, Any]) -> Optional[str]:
    """Write an accepted draft as a new personal profile. None on success, else the reason.

    Blocking file I/O; callers run it through `asyncio.to_thread`. The name and the
    profile limit are checked again because a draft can sit open for minutes.
    """
    pm = cog.profile_manager
    index = pm._get_user_index(user_id)
    if profile_name in index.get("personal", []) or profile_name in index.get("borrowed", []):
        return (f"You already have a profile named '{profile_name}'. "
                f"Generate the character again under another name.")
    if len(index.get("personal", [])) >= defaultConfig.LIMIT_PROFILES:
        return f"You have reached the maximum of {defaultConfig.LIMIT_PROFILES} personal profiles."

    profile = pm._get_or_create_user_profile(user_id, profile_name)
    if not profile:
        return "Failed to create the profile structure."

    encrypt = cog.storage_manager._encrypt_data
    prompts = profile.get("prompts") or {}
    prompts["persona"] = {key: [encrypt(line) for line in value.splitlines()]
                          for key, value in draft["persona"].items()}
    if not isinstance(prompts.get("ai_instructions"), list):
        prompts["ai_instructions"] = ["", "", "", ""]
    prompts["ai_instructions"][0] = encrypt(draft["ai_instructions"])
    pm._save_profile_prompts(user_id, profile_name, prompts)

    extras = {"custom_display_name": draft.get("display_name"),
              "placeholder_emoji": draft.get("placeholder_emoji"),
              "library_intro": draft.get("library_intro")}
    extras = {key: value for key, value in extras.items() if value}
    if extras:
        config = pm._get_profile_config(user_id, profile_name, False) or {}
        config.update(extras)
        pm._save_profile_config(user_id, profile_name, config, False)
    return None
