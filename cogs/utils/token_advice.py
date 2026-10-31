"""Token Advice: what a profile's settings, or a session's, cost per reply.

Two screens read this -- `/profile manage` and `/session config` -- and it is pure, so
both say the same thing about the same setting. An item is a feature that is on and what
it costs, never a verdict that it should be off: a character that thinks hard or reads
links does so on purpose. `TOKEN_USAGE.md` is where the figures come from.

Each item carries its size, judged per reply: LARGE past ~2,000 tokens or for any extra
model call, MEDIUM from 500, SMALL below. A list is largest first, and otherwise in
TOKEN_USAGE.md's order.
"""
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from . import fixed_examples, lorebook
from .constants import GROUNDING_MODE_LABELS, MEDIA_RESOLUTIONS, ROUND_SETTLE_SECONDS, defaultConfig
from .helpers import (_estimate_text_tokens, resolve_critic_settings, resolve_grounding_mode,
                      resolve_media_resolution, resolve_thinking_params, resolve_url_mode)

LARGE, MEDIUM, SMALL = "🔴", "🟠", "🟡"
_RANK = {LARGE: 0, MEDIUM: 1, SMALL: 2}

#: Gemini 3's tokens for one received image at each Input Resolution, per
#: MEDIA_RESOLUTIONS. "" sends no field, which Gemini 3 reads as HIGH.
IMAGE_TOKENS = {"MEDIA_RESOLUTION_LOW": 280, "MEDIA_RESOLUTION_MEDIUM": 560,
                "MEDIA_RESOLUTION_HIGH": 1120, "MEDIA_RESOLUTION_ULTRA_HIGH": 2240, "": 1120}
#: Text a profile always sends earns an item of its own from here: a medium cost.
STANDING_ITEM_TOKENS = 500
#: A message this large in the recent chat earns an item: pasted text, mostly.
PASTED_ITEM_TOKENS = 2000
#: Replies per message past which the cast earns an item, and past which it is large.
REPLIES_ITEM, REPLIES_LARGE = 1.2, 2.0
#: Public messages the measured replies per message is taken over, and the fewest it needs.
REPLIES_SAMPLE, REPLIES_SAMPLE_MIN = 20, 5
#: Replies each character's recent cost is averaged over, and how far back the log is read.
CAST_SAMPLE, LOG_SCAN_MAX = 10, 400

FOOTER = "🔴 large · 🟠 medium · 🟡 small impact per reply. A token is about ¾ of a word."


class Advice(NamedTuple):
    size: str
    heading: str
    text: str
    #: Where it is set: a PROFILE_ACTIONS value, or a Session Config tab ("tab" or
    #: "config:<screen>"), which the screen turns into its own pointer.
    target: str


def size_of(tokens: float) -> str:
    return LARGE if tokens > 2000 else MEDIUM if tokens >= 500 else SMALL


def largest_first(items: List[Advice]) -> List[Advice]:
    return sorted(items, key=lambda a: _RANK[a.size])


def quiet_label(seconds: Any) -> str:
    """A cooldown as people say it: `5 min`, `90 s`, `2 h`."""
    seconds = int(float(seconds or 0))
    if seconds and seconds % 3600 == 0:
        return f"{seconds // 3600} h"
    return f"{seconds // 60} min" if seconds and seconds % 60 == 0 else f"{seconds} s"


def stm_window(config: Dict[str, Any]) -> Tuple[int, int]:
    """(STM Length, the most turns a prompt then carries): the window moves in steps of
    half the STM so it can be cached (`_build_history_for_participant`)."""
    stm = int(config.get("stm_length", defaultConfig.CHATBOT_MEMORY_LENGTH))
    return stm, stm + stm // 2 - 1


# --- a profile ------------------------------------------------------------------------

def standing_text(prompts: Dict[str, Any], decrypt: Callable[[str], str],
                  persona_order: Sequence[str]) -> List[Tuple[str, int, str]]:
    """(part, tokens, where it is set) for the text the profile writes that goes out with
    every reply, as prompt_builder assembles it. A part with nothing in it is left out."""
    def lines(value) -> str:
        items = value if isinstance(value, list) else [value]
        return "\n".join(decrypt(v) for v in items if isinstance(v, str) and v.strip())

    persona = prompts.get("persona") or {}
    parts = [
        ("Persona", "\n".join(lines(persona.get(key) or []) for key in persona_order), "edit_persona"),
        ("Instructions", lines(prompts.get("ai_instructions") or ""), "edit_instructions"),
        ("Style guide", decrypt(prompts.get("style_guide") or "") if prompts.get("style_guide_enabled") else "",
         "style_guide"),
        ("Always-on lore", "\n\n".join(lorebook.constant(lorebook.entries(prompts.get("lorebook")))),
         "manage_lore"),
        ("Fixed examples", fixed_examples.render(fixed_examples.entries(prompts.get("fixed_examples"))),
         "manage_fixed_examples"),
    ]
    return [(label, _estimate_text_tokens(text), target) for label, text, target in parts if text.strip()]


def profile_advice(config: Dict[str, Any], always_on_lore: List[str],
                   fixed: List[Dict[str, str]], cog=None) -> List[Advice]:
    """What this profile's settings cost, largest first. Only what is true of it.

    `always_on_lore` and `fixed` are the keyless entries' text and the fixed examples, or
    empty for a borrow: those are its source's, and not its to change.
    """
    items = []
    level = resolve_thinking_params(config, "response", cog=cog)["thinking_level"]
    if level in ("max", "xhigh", "high"):
        items.append(Advice(LARGE, f"Reasoning Level: {level}",
                            "Thinking is billed as output, the most expensive kind of token, and at "
                            "this level it can run to thousands of tokens a reply.", "thinking_params"))
    stm, window = stm_window(config)
    if stm > defaultConfig.CHATBOT_MEMORY_LENGTH:
        items.append(Advice(LARGE if stm >= 2 * defaultConfig.CHATBOT_MEMORY_LENGTH else MEDIUM,
                            f"Short-Term Memory: {stm} turns",
                            f"Every reply re-sends up to {window} turns of the conversation, with anything "
                            "pasted into them. Most of it is cached while the window holds still.",
                            "stm_params"))
    lore_tokens = _estimate_text_tokens("\n\n".join(always_on_lore))
    if lore_tokens >= STANDING_ITEM_TOKENS:
        items.append(Advice(size_of(lore_tokens),
                            f"Always-on lore: {len(always_on_lore)} entries, ~{lore_tokens:,} tokens",
                            "Sent with every reply, and cached. An entry with keywords is sent only when a "
                            "recent message mentions one.", "manage_lore"))
    fixed_tokens = _estimate_text_tokens(fixed_examples.render(fixed))
    if fixed_tokens >= STANDING_ITEM_TOKENS:
        items.append(Advice(size_of(fixed_tokens), f"Fixed examples: {len(fixed)}, ~{fixed_tokens:,} tokens",
                            "Sent with every reply, and cached. A training example is sent only when a "
                            "message resembles it.", "manage_fixed_examples"))
    if resolve_url_mode(config) != "off":
        items.append(Advice(LARGE, "URL Context: On",
                            "A linked page is up to ~4,000 tokens, two a message, and is re-sent with every "
                            "reply while its message is in Short-Term Memory.", "url_toggle"))
    resolution = resolve_media_resolution(config)
    per_image = IMAGE_TOKENS[resolution]
    if per_image > IMAGE_TOKENS["MEDIA_RESOLUTION_MEDIUM"]:
        label = next(l for v, l, _d in MEDIA_RESOLUTIONS if v == resolution)
        items.append(Advice(size_of(per_image), f"Images read at {label}",
                            f"About {per_image:,} tokens an image on Gemini 3, for each character that "
                            "replies. Medium is 560.", "media_resolution"))
    if resolve_grounding_mode(config) == "rag":
        items.append(Advice(LARGE, f"Grounding: {GROUNDING_MODE_LABELS['rag']}",
                            "A gate model reads the conversation before every round to decide whether to "
                            "search, and a researcher runs when it does.", "grounding"))
    if resolve_critic_settings(config)["mode"] == "full":
        items.append(Advice(LARGE, "Anti-Repetition Critic: Full",
                            "An extra model call before most replies, reading the last few turns.", "critic"))
    if config.get("speech_tts_enabled"):
        items.append(Advice(LARGE, "Speech TTS: On",
                            "In a session with Text-to-Speech on, every reply is voiced too, billed per "
                            "second of audio on top of its text.", "speech_settings"))
    return largest_first(items)


# --- a session ------------------------------------------------------------------------

def _public(turn: Dict[str, Any]) -> bool:
    return not turn.get("type") and not turn.get("is_hidden") and not turn.get("compacted")


def _is_reply(turn: Dict[str, Any]) -> bool:
    return (_public(turn) and not turn.get("is_user") and bool(turn.get("profile_name"))
            and turn.get("speaker_pid") != "SYSTEM")


def replies_per_message(log: List[Dict[str, Any]]) -> Optional[Tuple[float, int]]:
    """(replies per public message, messages counted) over the newest REPLIES_SAMPLE, or
    None with too few to say. Rounds nobody asked for count their replies too: they are
    what the cast costs."""
    messages = replies = 0
    for turn in reversed(log[-LOG_SCAN_MAX:]):
        if _is_reply(turn):
            replies += 1
        elif _public(turn) and turn.get("is_user"):
            messages += 1
            if messages >= REPLIES_SAMPLE:
                break
    return (replies / messages, messages) if messages >= REPLIES_SAMPLE_MIN else None


def cast_costs(log: List[Dict[str, Any]], seats: Sequence[Dict[str, Any]]) -> List[Tuple[str, Optional[Dict[str, int]]]]:
    """(name, its average reply) for each seat over its newest CAST_SAMPLE replies, or None
    for a seat with none: input, how much of it was cached, output and thinking."""
    keys = [(seat.get("owner_id"), seat.get("profile_name")) for seat in seats]
    metas: Dict[tuple, List[Dict[str, Any]]] = {key: [] for key in keys}
    for turn in reversed(log[-LOG_SCAN_MAX:]):
        key = (turn.get("owner_id"), turn.get("profile_name"))
        if key in metas and len(metas[key]) < CAST_SAMPLE and _is_reply(turn) and turn.get("meta"):
            metas[key].append(turn["meta"])

    def mean(rows, *fields) -> int:
        return round(sum(sum(int(r.get(f) or 0) for f in fields) for r in rows) / len(rows))

    # Output as it is billed (`billable_output_tokens`): Google counts thinking apart from
    # the reply, OpenRouter inside it, as `reasoning_tokens`.
    return [(name, {"input": mean(rows, "input_tokens"), "cached": mean(rows, "cached_tokens"),
                    "output": mean(rows, "output_tokens", "thinking_tokens"),
                    "thinking": mean(rows, "thinking_tokens") or mean(rows, "reasoning_tokens")}
             if (rows := metas[(owner, name)]) else None)
            for owner, name in keys]


def session_advice(session: Dict[str, Any], seats: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]], *,
                   settle: Tuple[float, float, float], compaction: Dict[str, Any],
                   director: bool, custom_emojis: bool) -> List[Advice]:
    """What this session's settings cost, largest first. `seats` is each seated
    participant with its profile's config; `settle`, `compaction` and `director` as the
    session reads them; `custom_emojis` whether the server has any to send."""
    items = []
    log = session.get("unified_log") or []
    answering = [seat for seat, _cfg in seats if not seat.get("is_skipped")]
    limit = int(session.get("max_responses", 10))

    measured = replies_per_message(log)
    if measured:
        rate, basis = measured[0], f"Measured over the last {measured[1]} messages."
    else:
        rate = min(limit, sum(seat.get("chance", 100) for seat in answering) / 100)
        basis = "Worked out from their chances, with too few messages yet to measure."
    if len(answering) > 1 and rate > REPLIES_ITEM:
        chances = sorted({seat.get("chance", 100) for seat in answering})
        chance = f"all at {chances[0]}%" if len(chances) == 1 else f"at {chances[0]}–{chances[-1]}%"
        items.append(Advice(LARGE if rate >= REPLIES_LARGE else MEDIUM,
                            f"Each message gets about {rate:.1f} replies",
                            f"{len(answering)} characters, {chance} Chance, Response Limit {limit}. {basis} "
                            "Every reply sends that character's whole prompt, and names and wakewords can "
                            "bring more in.", "reactivity"))

    pro = session.get("proactivity") or {}
    if pro.get("enabled"):
        cooldown = max(60.0, float(pro.get("cooldown", 300)))
        chance = pro.get("chance", 10)
        per_day = 86400 / cooldown * chance / 100
        items.append(Advice(LARGE, f"Proactivity: {chance}% after every {quiet_label(cooldown)} of quiet",
                            f"With nobody talking, about {per_day:,.0f} rounds a day, each one a full round "
                            "for the cast, and they keep coming with nobody there."
                            + (" The AI Director adds a call before each." if director else ""),
                            "proactivity"))

    voiced = [seat for seat, cfg in seats if cfg.get("speech_tts_enabled")]
    if session.get("audio_mode", "off") == "on" and voiced:
        items.append(Advice(LARGE, "Text-to-Speech: On",
                            f"{len(voiced)} of the cast {'is' if len(voiced) == 1 else 'are'} voiced: each of "
                            "their replies is billed per second of audio, on top of its text.", "config:tts"))

    # Only what some seat's window still holds is re-sent.
    window = max((stm_window(cfg)[1] for _seat, cfg in seats), default=0)
    recent = [t for t in log[-LOG_SCAN_MAX:] if _public(t)][-window:] if window else []
    biggest = max((_estimate_text_tokens(str(t.get("content") or "")) for t in recent if t.get("is_user")),
                  default=0)
    if biggest > PASTED_ITEM_TOKENS:
        items.append(Advice(LARGE, f"A recent message holds ~{biggest:,} tokens",
                            "Pasted text is re-sent with every reply, for every character, until its "
                            "message leaves Short-Term Memory. Compact Now folds it into the synopsis.",
                            "compaction"))

    prompt_tokens = _estimate_text_tokens(session.get("session_prompt") or "")
    if prompt_tokens >= STANDING_ITEM_TOKENS:
        items.append(Advice(size_of(prompt_tokens), f"Master Prompt: ~{prompt_tokens:,} tokens",
                            "Sent to every character on every call, and cached until it is edited.",
                            "config:prompt"))

    # Mostly a saving: folded turns leave every prompt. Worth an item only where it is not
    # one for somebody, or where its synopsis is a cost of its own.
    if compaction.get("enabled"):
        synopsis = round(compaction.get("max_words", 220) * 4 / 3)
        # Against the idle mark, the shallower of the two: a window short of it is short of
        # every fold, where one short only of `threshold` is still shortened by quiet ones.
        unshortened = [str(seat.get("profile_name") or "?") for seat, cfg in seats
                       if stm_window(cfg)[1] <= compaction["idle_threshold"] - compaction["chunk"]]
        if unshortened or synopsis >= STANDING_ITEM_TOKENS:
            text = (f"Once {compaction['threshold']} turns are live, or {compaction['idle_threshold']} and "
                    f"the channel goes quiet, a summariser call folds the oldest "
                    f"{compaction['chunk']} into the synopsis, up to ~{synopsis:,} tokens, which then goes to "
                    "every character on every call.")
            if unshortened:
                names = ", ".join(unshortened[:3]) + (" and others" if len(unshortened) > 3 else "")
                text += (f" {names} {'has' if len(unshortened) == 1 else 'have'} a Short-Term Memory too "
                         "short to reach the folded turns, so a fold shortens nothing of theirs.")
            items.append(Advice(size_of(synopsis) if not unshortened else MEDIUM, "Compaction: On", text,
                                "compaction"))

    if settle[0] < ROUND_SETTLE_SECONDS:
        items.append(Advice(MEDIUM, f"Settle Window: {settle[0]:g} s quiet",
                            "A quick run of messages gets a round each, where a longer wait answers them "
                            "together.", "config:settle"))

    if session.get("server_context", True) and custom_emojis:
        items.append(Advice(SMALL, "Server Context: On",
                            "Custom emojis people post are sent as pictures, up to 4 a round, to each "
                            "character that replies, at its own image setting.", "config:server"))
    return largest_first(items)
