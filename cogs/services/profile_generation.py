"""Drafting a character from a concept: the prompt, the parse, the model and the save.

`/profile generate` is reached three ways -- the command itself, the /start wizard and the
Public Library's Generate button, from a concept, an image or both -- and all three arrive
at the command's callback, which
drafts through here and shows the draft (`gui_generate.GeneratedProfileView`) before
anything is written. A draft lives only on that view: nothing touches disk until Save.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from ..utils import fixed_examples, lorebook
from ..utils.character_card import prompt_text
from ..utils.constants import (AI_INSTRUCTIONS_PART_MAX_LENGTH, DEFAULT_CARD_IMPORT_PROMPT,
                               DEFAULT_PROFILE_GENERATOR_PROMPT, DEFAULT_SAFETY_SETTINGS,
                               GENERATOR_KEYS, LIBRARY_INTRO_MAX_CHARS,
                               PERSONA_TEXT_INPUT_MAX_LENGTH, defaultConfig)
from ..utils.helpers import (_format_api_error, resolve_auto_models, resolve_thinking_params,
                             system_model, system_settings)
from ..utils.user_defaults import apply_defaults, model_chain, model_provider

#: A character is creative work. This was 0.3, a form-filling temperature, and every
#: concept came back as the same genre-average character.
GENERATOR_TEMPERATURE = 0.9
#: Sorting a card into sections is filing: the author's words, moved, not rewritten.
CARD_IMPORT_TEMPERATURE = 0.3

#: /mod key -> (shipped template, temperature). Both take `{prompt}` and answer in sections.
TEMPLATES = {
    "PROFILE_GENERATOR": (DEFAULT_PROFILE_GENERATOR_PROMPT, GENERATOR_TEMPERATURE),
    "CARD_IMPORT": (DEFAULT_CARD_IMPORT_PROMPT, CARD_IMPORT_TEMPERATURE),
}

#: Persona section -> the header that carries it, in the persona modal's order.
PERSONA_SECTIONS = {
    "backstory": "persona_backstory",
    "personality_traits": "persona_personality_traits",
    "likes": "persona_likes",
    "dislikes": "persona_dislikes",
    "appearance": "persona_appearance",
}

SECTIONS = ("display_name", "library_intro", *PERSONA_SECTIONS.values(), "ai_instructions")

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

#: What the template is handed as `{prompt}` when an image comes with no concept.
IMAGE_CONCEPT = "the character in the attached image"

#: Appended after the operator's template for the same reason as REVISION_TEMPLATE.
IMAGE_NOTE = (
    "\n\nAn image of the character is attached. Take their appearance from what is visible, "
    "and let the rest of it -- expression, clothing, setting, art style -- suggest who they "
    "are. Where the concept and the image disagree, follow the concept. The character is "
    "fictional, inspired by the image: never identify or name a real person, even one you "
    "recognise."
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
        "persona": {key: sections.get(header, "") for key, header in PERSONA_SECTIONS.items()},
        "ai_instructions": sections.get("ai_instructions", ""),
    }


def render_draft(draft: Dict[str, Any]) -> str:
    """A draft written back out in the section format the model produced it in."""
    values = {"display_name": draft.get("display_name"),
              "library_intro": draft.get("library_intro"),
              "ai_instructions": draft.get("ai_instructions")}
    for key, header in PERSONA_SECTIONS.items():
        values[header] = (draft.get("persona") or {}).get(key)
    return "\n".join(f"[SECTION:{s}]\n{values[s] or ''}" for s in SECTIONS)


def build_prompt(template: str, concept: str, previous: Optional[Dict[str, Any]] = None,
                 request: Optional[str] = None, image: bool = False) -> str:
    prompt = template.format(prompt=concept)
    if image:
        prompt += IMAGE_NOTE
    if previous is not None:
        prompt += REVISION_TEMPLATE.format(draft=render_draft(previous),
                                           request=(request or "").strip())
    return prompt


def generator_models(cog, user_id: int) -> Tuple[str, Tuple[str, ...]]:
    """(primary, fallbacks) for a draft.

    The generator's System Models on the user's effective provider -- a draft needs a model
    before any provider is chosen, so a key counts here as it does not for a profile --
    their own defaults over them once they have chosen one, rescued onto a provider they
    hold. Only models on a provider the user holds a key for are kept: a draft is billed
    to the user alone.
    """
    pm = cog.profile_manager
    provider = pm.effective_provider(user_id)
    config = dict(zip(("primary_model", "fallback_model"),
                      (system_model(cog, key, provider) for key in GENERATOR_KEYS)))
    if pm.provider_preference(user_id):
        apply_defaults(config, pm._get_user_defaults(user_id), borrowed=False)
    pm._rescue_unusable_models(user_id, config)

    primary, fallbacks = model_chain(resolve_auto_models(cog, config), "primary_model", provider)
    usable: List[str] = [m for m in dict.fromkeys((primary, *fallbacks))
                         if pm._user_holds_provider_key(user_id, model_provider(m))]
    if not usable:
        raise ProfileGenerationError(NO_KEY_MESSAGE)
    return usable[0], tuple(usable[1:])


async def generate_draft(cog, user_id: int, concept: str,
                         previous: Optional[Dict[str, Any]] = None,
                         request: Optional[str] = None,
                         template_key: str = "PROFILE_GENERATOR",
                         image: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, Any], str]:
    """(draft, model used). `previous` and `request` together make it a refinement.
    `template_key` is one of TEMPLATES: a concept to design from, or a card to sort.
    `image` is a `{"url", "mime_type"}` part, shown to the model ahead of the prompt.

    Raises ProfileGenerationError, and nothing else, for anything the user should be told.
    """
    primary, fallbacks = generator_models(cog, user_id)
    # The generator tab's thinking and Hosts & Tier, from /mod, for the user's side --
    # whichever models their own defaults put in front of it.
    settings = system_settings(cog, "generator", cog.profile_manager.effective_provider(user_id))
    default, temperature = TEMPLATES[template_key]
    template = cog.global_prompts.get(template_key, default)
    try:
        prompt = build_prompt(template, concept, previous, request, image=bool(image))
    except (KeyError, IndexError, ValueError):
        raise ProfileGenerationError(
            "The profile generator prompt on this bot is misconfigured. Let the bot operator know.")
    # A list of turns, not parts: a bare part dict is read as a turn with no parts, and
    # the model drafts from the words "the attached image" without ever seeing it.
    contents = [{"role": "user", "parts": [image, prompt]}] if image else [prompt]

    async def attempt(model_name: str, is_fallback: bool) -> Dict[str, Any]:
        model = cog.api_service._instantiate_model(
            model_name, None, user_id, None, DEFAULT_SAFETY_SETTINGS,
            resolve_thinking_params(settings, "generator", "fallback" if is_fallback else "primary"),
            profile_settings=settings, config_owner_id=user_id)
        response = await model.generate_content_async(
            contents, generation_config={"temperature": temperature})
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
        # A format test, not a content one: a card may say nothing about a section.
        if not any(draft["persona"].values()):
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


async def draft_from_card(cog, user_id: int, card: Dict[str, Any]) -> Tuple[Dict[str, Any], str, Optional[str]]:
    """(draft, model that sorted it or "", why it was not sorted). The card's text sorted
    into sections by the user's own models -- or, when none will (no key, a refusal, a
    reply in no format), the card as written: description and scenario as backstory,
    personality as personality. An import never fails for want of a model."""
    try:
        draft, used = await generate_draft(cog, user_id, prompt_text(card), template_key="CARD_IMPORT")
        return draft, used, None
    except ProfileGenerationError as e:
        backstory = "\n\n".join(t for t in (card["description"], card["scenario"]) if t)
        persona = {key: "" for key in PERSONA_SECTIONS}
        persona.update(backstory=backstory, personality_traits=card["personality"])
        return ({"display_name": None, "library_intro": None, "persona": persona,
                 "ai_instructions": ""}, "", str(e))


def save_draft(cog, user_id: int, profile_name: str, draft: Dict[str, Any]) -> Optional[str]:
    """Write an accepted draft as a new personal profile. None on success, else the reason.

    Blocking file I/O; callers run it through `asyncio.to_thread`. The name and the
    profile limit are checked again because a draft can sit open for minutes. Every field
    is cut to the editor's own cap, which an imported card can run past. A card's
    lorebook, its fixed examples and its 18+ tag (`adult`) are written here too, before
    anyone can seat it.
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
    prompts["persona"] = {key: [encrypt(line) for line in value[:PERSONA_TEXT_INPUT_MAX_LENGTH].splitlines()]
                          for key, value in draft["persona"].items()}
    if not isinstance(prompts.get("ai_instructions"), list):
        prompts["ai_instructions"] = ["", "", "", ""]
    prompts["ai_instructions"][0] = encrypt(draft["ai_instructions"][:AI_INSTRUCTIONS_PART_MAX_LENGTH])
    if draft.get("lorebook"):
        prompts["lorebook"] = lorebook.normalise(draft["lorebook"])
    if draft.get("fixed_examples"):
        prompts["fixed_examples"] = fixed_examples.normalise(draft["fixed_examples"])
    pm._save_profile_prompts(user_id, profile_name, prompts)
    if draft.get("adult"):
        pm.set_owner_adult_declaration(user_id, profile_name, True)

    extras = {"custom_display_name": draft.get("display_name"),
              "custom_avatar_url": draft.get("avatar_url"),
              "library_intro": draft.get("library_intro")}
    extras = {key: value for key, value in extras.items() if value}
    if extras:
        config = pm._get_profile_config(user_id, profile_name, False) or {}
        config.update(extras)
        pm._save_profile_config(user_id, profile_name, config, False)
    return None
