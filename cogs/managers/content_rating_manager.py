"""Content ratings: the classifier, verdicts and their holds, refusals and appeals, and
whether a rating lets a profile run."""

import os
import time
import hashlib
import re
import threading
import datetime
import asyncio
import discord
from typing import Dict, List, Any, Optional, Tuple
from ..utils.constants import (
    USERS_DIR, CLASSIFIER_KEYS, defaultConfig, DEFAULT_SAFETY_SETTINGS, CONTENT_RATING_LABELS, CONTENT_RATING_REASON_LABELS, DEFAULT_CONTENT_CLASSIFIER_PROMPT, REFUSAL_CONFIRMERS, UNREVIEWED_REFUSALS, CONTENT_RATING_UNRATED, CONTENT_RATING_PENDING, CONTENT_RATING_GENERAL, CONTENT_RATING_ADULT, CONTENT_RATING_REFUSED, PLACEMENT_REFUSALS, REFUSED_INDEX_FILE, CONTENT_RATING_CAPABILITIES, CONTENT_CAPABILITY_DENIALS)
from ..utils.helpers import (
    resolve_thinking_params, subscription_curated, system_model, system_settings, resolve_openrouter_endpoint,
    resolve_openrouter_service_tier)
from ..utils import fixed_examples, lorebook
from ..utils.discord_cdn import signed_attachment_url
from ..utils.http_client import get_capped, get_shared_client
from .storage_manager import IOManager
from ..services.api_service import OpenRouterModel, GoogleGenAIModel
from .profile_manager import _carried_hold, _held_for_rating

#: `_effective_safety_level`'s answers, least to most restrictive.
_SAFETY_LEVEL_ORDER = ("restricted", "checking", "not_allowed", "blocked", "refused")

#: Rating jobs land in worker threads, and each read-modify-writes the whole file.
_REFUSED_INDEX_LOCK = threading.Lock()

#: The one line the classifier is asked for, bolded or quoted at most, read through
#: `_answer_line`. Anything else is no verdict -- never Allowed: a model declining in words
#: must not rate a profile clean.
#: GENERAL and ADULT are the shipped prompt's words before Allowed and Not Allowed, which a
#: /mod override written then still asks for.
_VERDICT_LINE = re.compile(r"\W*(NOT[_ ]ALLOWED|ALLOWED|GENERAL|ADULT)\b[^\w:]*(?::\W*(\w+))?")
_VERDICTS = {"ALLOWED": CONTENT_RATING_GENERAL, "GENERAL": CONTENT_RATING_GENERAL,
             "NOT_ALLOWED": CONTENT_RATING_ADULT, "NOT ALLOWED": CONTENT_RATING_ADULT,
             "ADULT": CONTENT_RATING_ADULT}
#: The refusal check's one word, matched the same way.
_CONFIRM_LINE = re.compile(r"\W*(YES|NO)\b", re.IGNORECASE)
#: What may sit beside an answer on its line: the bold, quotes or full stop a model adds.
_ANSWER_DRESSING = " \t*_`'\".>"


def _answer_line(pattern: re.Pattern, reply: str) -> Optional[re.Match]:
    """The answer in a classifier's reply, `pattern` matched line by line; None for none.

    Every line opening with an answer must give the same one, and one of them must be the
    line asked for with nothing else on it (`_ANSWER_DRESSING` aside). A model reasoning
    aloud -- "NOT_ALLOWED: ... no. Let me evaluate ... ALLOWED: NONE" -- contradicts itself,
    which is no answer and goes to the Fallback, never its first word: read as a prefix,
    that stored a profile Not Allowed that the reply went on to allow, and a resubmission
    asked the same model for `_hold_floor`'s second opinion. Lines after the answer are
    read past: matched, not compared.
    """
    opened = [(line, m) for line in reply.splitlines() if (m := pattern.match(line))]
    if len({_VERDICTS.get(m.group(1).upper(), m.group(1).upper()) for _, m in opened}) != 1:
        return None
    return next((m for line, m in opened if not line[m.end():].strip(_ANSWER_DRESSING)), None)
#: OpenRouter's out-of-credit refusal, as a classifier failure carries it (`openrouter.py`).
_NO_CREDIT = "API Error 402"
#: A provider refusing to read a profile at all: Google's filter that no setting reaches,
#: as a block or finish reason, or passed through OpenRouter as `native_finish_reason`.
#: Not a verdict, but a hold (`provider_blocked`), never the operator's queue: they would
#: be reading what the provider would not.
_PROVIDER_REFUSALS = frozenset({"PROHIBITED_CONTENT"})
#: The same from a moderated OpenRouter model: its 403 names its categories, and the
#: wording is the moderator's own, so matched rather than compared.
_MINORS_FLAGGED = re.compile(r"minor|child", re.IGNORECASE)


def _provider_refusal(*codes) -> Optional[str]:
    """The first of a reply's block and finish reasons in `_PROVIDER_REFUSALS`, by name:
    the SDK's enums and REST's strings alike."""
    for code in codes:
        name = str(getattr(code, "name", code) or "")
        if name in _PROVIDER_REFUSALS:
            return name
    return None


def _awaits_review(rating: Dict[str, Any]) -> bool:
    """A refusal the operator has yet to settle: disputed, unconfirmed, or appealed. A
    confirmed or upheld one settles itself and stays off the review list."""
    return rating.get("verdict") == CONTENT_RATING_REFUSED and bool(rating.get("review"))


#: Verdicts by what they keep in, for telling a re-rating that loosens one.
_STRICTNESS = {CONTENT_RATING_GENERAL: 0, CONTENT_RATING_ADULT: 1, CONTENT_RATING_REFUSED: 2}


def _held(rating: Dict[str, Any]) -> bool:
    """Whether `rating` keeps its profile from running until a verdict lands: a hold for
    rating (`_held_for_rating`: an NSFW-tagged card, a generated profile tagged 18+, an
    import rated Not Allowed elsewhere, or an 18+ declaration made before adult content was
    removed), or an Unrated or Pending record a Not Allowed verdict left behind (`was`), so
    an edit or a resubmission is no way back to running before it is rated again."""
    return rating.get("source") == "owner_declared" or rating.get("was") == CONTENT_RATING_ADULT


def _rating_floor(rating: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """(verdict, reason) a re-rating may not go below on one model's word (`_hold_floor`).

    A standing Refused, or a Not Allowed the classifier or the operator gave -- a hold for
    rating is no verdict -- or the Not Allowed an Unrated or Pending record carries as `was`,
    so dropping to Unrated and submitting again is no way round it.
    """
    verdict = rating.get("verdict")
    if verdict == CONTENT_RATING_REFUSED or (
            verdict == CONTENT_RATING_ADULT and rating.get("source") != "owner_declared"):
        return verdict, rating.get("reason")
    if rating.get("was") == CONTENT_RATING_ADULT:
        return CONTENT_RATING_ADULT, None
    return None, None


def _carried_floor(rating: Dict[str, Any]) -> Dict[str, str]:
    """`was` and the hold, for an Unrated or Pending record replacing `rating`. Refused
    never is one."""
    was = {"was": CONTENT_RATING_ADULT} if _rating_floor(rating)[0] == CONTENT_RATING_ADULT else {}
    return {**was, **_carried_hold(rating)}


class ContentRatingManager:
    """Content ratings: the classifier, verdicts and their holds, refusals and appeals, and
    whether a rating lets a profile run. Reached as `cog.content_rating`."""

    def __init__(self, cog):
        self.cog = cog

    def _moderated_surface(self, owner_id: int, profile_name: str) -> str:
        """The profile text the classifier judges, and the text the hash covers.

        Built from the same decrypted fields prompt_builder assembles, so the
        verdict is about what the model will actually be told to be. Deliberately
        excludes LTM, session logs and training examples: the first two are
        conversation-derived, and training examples -- though the strongest signal
        of what a profile really outputs -- change often enough that hashing them
        would invalidate the verdict constantly. Revisit them with their own hash
        and a debounce.
        """
        eff_owner, eff_name = self.cog.profile_manager._resolve_effective_profile(owner_id, profile_name)
        prompts = self.cog.profile_manager._get_profile_prompts(eff_owner, eff_name) or {}
        config = self.cog.profile_manager._get_profile_config(eff_owner, eff_name, False) or {}

        parts = [f"name: {eff_name}"]

        display = config.get("custom_display_name")
        if display:
            parts.append(f"display_name: {display}")
        avatar = config.get("custom_avatar_url")
        if avatar:
            parts.append(f"avatar_url: {avatar}")
        # Shown as often as the usual one, so a mood avatar set after the verdict makes it stale.
        for axis, url in sorted((config.get("mood_avatars") or {}).items()):
            parts.append(f"mood_avatar_{axis}: {url}")
        # Shown to everyone browsing the Public Library, which is the audience the
        # rating answers to -- so an intro written after the verdict makes it stale.
        intro = config.get("library_intro")
        if intro:
            parts.append(f"library_intro:\n{intro}")

        persona = prompts.get("persona", {}) or {}
        for key in self.cog.persona_modal_sections_order:
            lines = persona.get(key) or []
            decrypted = [self.cog.storage_manager._decrypt_data(l).strip()
                         for l in lines if isinstance(l, str) and l.strip()]
            body = "\n".join(d for d in decrypted if d)
            if body:
                parts.append(f"{key}:\n{body}")

        instr = prompts.get("ai_instructions", "")
        instr_list = instr if isinstance(instr, list) else [instr]
        decrypted_instr = [self.cog.storage_manager._decrypt_data(i).strip()
                           for i in instr_list if isinstance(i, str) and i.strip()]
        body = "\n\n".join(d for d in decrypted_instr if d)
        if body:
            parts.append(f"instructions:\n{body}")
        # Judged whether on or off, so turning it on rescans nothing. Enhance Awareness is
        # shipped text and is never here.
        guide = prompts.get("style_guide")
        if isinstance(guide, str) and guide.strip():
            parts.append(f"style_guide:\n{self.cog.storage_manager._decrypt_data(guide).strip()}")

        img_prompt = prompts.get("image_generation_prompt")
        if isinstance(img_prompt, str) and img_prompt.strip():
            parts.append(f"image_prompt:\n{self.cog.storage_manager._decrypt_data(img_prompt).strip()}")

        # What the character is handed alongside its persona, so it is judged with it.
        lore = "\n\n".join(e["content"] for e in lorebook.entries(prompts.get("lorebook")))
        if lore:
            parts.append(f"lorebook:\n{lore}")
        # Sent with every reply, and changed only on an edit, unlike training examples.
        fixed = fixed_examples.entries(prompts.get("fixed_examples"))
        if fixed:
            parts.append(f"fixed_examples:\n{fixed_examples.render(fixed)}")

        return "\n\n".join(parts)

    def _surface_hash(self, surface: str) -> str:
        return hashlib.sha256(surface.encode("utf-8", "replace")).hexdigest()[:16]


    def reset_all_content_ratings(self) -> Dict[str, Any]:
        """Operator maintenance: reset every profile on the instance to Unrated.

        A single deliberate action, run once by the operator, rather than a lazy
        per-read migration or a boot sweep. That is why nothing here is versioned:
        with no automatic pass to guard against, there is no need to record which
        profiles have already been through it, and no schema marker to carry
        forward in every rating record for the rest of the bot's life.

        **Unconditional.** Every profile, including Not Allowed and held ones. The point is a
        clean baseline: a rating now means "the owner asked for this and got an answer", and
        no verdict predating that rule qualifies, however it was reached. A profile that ran
        nowhere -- Not Allowed, Refused, or held -- is held for its new rating rather than
        made Unrated, which runs.

        Also strips the retired `safety_level`, which is the last of the pre-rating
        fields.

        Borrowed profiles (B/C PIDs) are skipped: their rating resolves through the
        source profile, so their local snapshot is not the authority and rewriting
        it would only create a second answer.

        Blocking -- walks and rewrites every profile file. Call it in a thread.
        """
        counts = {"scanned": 0, "reset": 0, "skipped_borrowed": 0,
                  "legacy_fields": 0, "errors": 0, "delisted": []}
        if not os.path.isdir(USERS_DIR):
            return counts

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        for user_id_str in os.listdir(USERS_DIR):
            if not user_id_str.isdigit():
                continue
            profiles_dir = os.path.join(USERS_DIR, user_id_str, "profiles")
            if not os.path.isdir(profiles_dir):
                continue

            for pid in os.listdir(profiles_dir):
                path = os.path.join(profiles_dir, pid, "profile.json.gz")
                if not os.path.exists(path):
                    continue
                counts["scanned"] += 1

                if pid.startswith(("B", "C")):
                    counts["skipped_borrowed"] += 1
                    continue

                try:
                    p_data = IOManager.read_json_gzip(path, self.cog.fernet)
                    if not isinstance(p_data, dict):
                        continue
                    config = p_data.get("config")
                    if not isinstance(config, dict):
                        continue

                    if config.pop("safety_level", None) is not None:
                        counts["legacy_fields"] += 1

                    old = config.get("content_rating") or {}
                    ran_nowhere = (_held(old) or old.get("verdict") in
                                   (CONTENT_RATING_ADULT, CONTENT_RATING_REFUSED))
                    config["content_rating"] = ({**_held_for_rating(), **_carried_hold(old)} if ran_nowhere else {
                        "verdict": CONTENT_RATING_UNRATED,
                        "hash": None,
                        "model": None,
                        "at": now,
                        "reason": None,
                        "source": "reset",
                    })
                    p_data["config"] = config
                    # Through the one writer, or the catalogue keeps the old verdict.
                    self.cog.profile_manager._save_profile_by_pid(int(user_id_str), pid, p_data)
                    counts["reset"] += 1
                except Exception as e:
                    counts["errors"] += 1
                    print(f"Content rating reset failed for {user_id_str}/{pid}: "
                          f"{type(e).__name__}({e})")

        # Everything is Unrated now, and an Unrated profile may not be published, so
        # the entire public index is stale by construction. Clearing it is not a side
        # effect of the reset -- it is the same decision applied to the index.
        # Existing borrows are untouched: a borrow reads the source profile, never
        # this index.
        for entry_id, p_info in list(self.cog.public_profiles.items()):
            desc = self.cog.library._describe_public_entry(entry_id, p_info)
            label = entry_id
            if desc:
                label = f"{desc['profile_name']} (owner {desc['owner_id']})"
            self.cog.public_profiles.pop(entry_id, None)
            counts["delisted"].append(label)
        if counts["delisted"]:
            self.cog.library._save_public_index()

        self.cog.content_rating_cache.clear()
        with _REFUSED_INDEX_LOCK:
            IOManager.write_json({}, REFUSED_INDEX_FILE)
        return counts

    def _verdict_of(self, config: Dict[str, Any]) -> str:
        """The rating verdict for a config, normalised to a known state.

        Anything unrecognised -- an absent record, the retired "unclassified", a
        value from a build that has since been removed -- reads as Unrated. That is
        the correct failure direction now: Unrated runs normally but cannot be
        distributed, so an unreadable rating costs the owner a submission rather
        than either silencing the profile or letting it out into the library.
        """
        rating = config.get("content_rating") or {}
        verdict = rating.get("verdict")
        if verdict in CONTENT_RATING_CAPABILITIES:
            return verdict
        return CONTENT_RATING_UNRATED

    def _effective_safety_level(self, config: Dict[str, Any]) -> str:
        """Whether this profile may run, from its content rating alone: a key of
        CHANNEL_ACCESS_LABELS and, bar "restricted", of PLACEMENT_REFUSALS.

        "restricted" runs, under <content_policy> like every profile. Not Allowed and Refused
        run nowhere, and nor does a profile `_held` for a verdict. Unrated, Pending and
        Allowed are otherwise identical at runtime -- the rating governs distribution -- so
        a profile whose owner has never submitted it is not silenced, it is merely
        undistributable. One its provider would not even read (`provider_blocked`) runs
        nowhere until a model does.
        """
        rating = config.get("content_rating") or {}
        if rating.get("provider_blocked"):
            return "blocked"
        verdict = self._verdict_of(config)
        if verdict == CONTENT_RATING_REFUSED:
            return "refused"
        if _held(rating):
            return "checking"
        return "restricted" if CONTENT_RATING_CAPABILITIES[verdict]["runs"] else "not_allowed"

    def _content_rating_state(self, owner_id: int, profile_name: str) -> Tuple[str, Optional[str]]:
        """(verdict, reason) as last recorded, for display and for the UI gates.

        Deliberately does NOT recompute the surface hash. This runs on every render
        of the profile Home tab, and hashing means decrypting the whole persona;
        staleness is detected instead by rating_is_stale, which is async and
        runs on dashboard open and session setup. The consequence is conservative in
        the right direction: a stale 'adult' verdict keeps the toggle locked until
        the recheck lands, rather than briefly unlocking it.
        """
        eff_owner, eff_name = self.cog.profile_manager._resolve_effective_profile(owner_id, profile_name)
        config = self.cog.profile_manager._get_profile_config(eff_owner, eff_name, False) or {}
        return self._verdict_of(config), (config.get("content_rating") or {}).get("reason")

    def content_capability(self, owner_id: int, profile_name: str, capability: str) -> Tuple[bool, Optional[str]]:
        """(allowed, reason_if_not) for one distribution capability.

        Single source of truth for the share, publish and global-chat gates. Each
        used to test verdicts inline in its own module -- the hub checked one set of
        values, the global chat command checked publication instead of rating, and
        the turn gate checked a third thing -- so the same profile could be accepted
        by one and refused by another with no shared wording to explain it.
        """
        verdict, _ = self._content_rating_state(owner_id, profile_name)
        allowed = CONTENT_RATING_CAPABILITIES[verdict][capability]
        if allowed:
            return True, None
        reason = CONTENT_CAPABILITY_DENIALS.get(
            (capability, verdict),
            f"Not available for a profile rated {CONTENT_RATING_LABELS[verdict]}.")
        return False, reason

    def _resolve_enforced_safety_level(self, profile_owner_id: int, profile_name: str) -> str:
        """The level the gate enforces for this profile, cached per (owner, name).

        The uncached path reads, decrypts and decompresses up to two profile files
        and hashes the persona. The gate runs on the turn path, so the result is
        memoised and invalidated by _invalidate_content_rating from every edit,
        classification and bulk write.
        """
        cache_key = (int(profile_owner_id), profile_name)
        cached = self.cog.content_rating_cache.get(cache_key)
        if cached is not None:
            return cached

        index = self.cog.profile_manager._get_user_index(profile_owner_id)
        is_borrowed = profile_name in index.get("borrowed", [])
        config = self.cog.profile_manager._get_profile_config(profile_owner_id, profile_name, is_borrowed) or {}

        level = self._effective_safety_level(config)

        # A borrowed profile carries its own local copy of the config, and
        # _store_content_rating refuses to write one -- the authoritative rating
        # only ever exists at the source, where the persona the verdict was formed
        # from actually lives. The local copy can still hold a stale rating, or a
        # legacy 18+ declaration frozen in at borrow time, so take whichever of the
        # two is stricter: the borrow can only ever escalate, never relax.
        if is_borrowed and level != "refused":
            src_owner, src_name = self.cog.profile_manager._resolve_effective_profile(profile_owner_id, profile_name)
            if (src_owner, src_name) != (profile_owner_id, profile_name):
                src_config = self.cog.profile_manager._get_profile_config(src_owner, src_name, False) or {}
                level = max(level, self._effective_safety_level(src_config),
                            key=_SAFETY_LEVEL_ORDER.index)

        self.cog.content_rating_cache[cache_key] = level
        return level

    def _invalidate_content_rating(self, owner_id: int, profile_name: str):
        """Drops the cached level for a profile and for every borrow of it.

        A borrowed entry is keyed by the borrower's local name, so a change at the
        source cannot be located by key -- the borrow map is scanned instead. It is
        small (one entry per live borrow in the cache) and this runs on edits, not
        on the turn path.
        """
        self.cog.content_rating_cache.pop((int(owner_id), profile_name), None)
        for key in [k for k in list(self.cog.content_rating_cache.keys())
                    if k != (int(owner_id), profile_name)]:
            k_owner, k_name = key
            try:
                src = self.cog.profile_manager._resolve_effective_profile(k_owner, k_name)
            except Exception:
                continue
            if src == (int(owner_id), profile_name):
                self.cog.content_rating_cache.pop(key, None)

    # ponytail: `channel` is unread below -- no channel widens or narrows what a rating lets
    # run, age-restricted ones included -- but kept for the ~50 callers and test doubles that
    # pass it. Drop it from them in a cleanup.
    def profile_block(self, owner_id: int, profile_name: str, channel) -> Optional[str]:
        """Which of PLACEMENT_REFUSALS this profile's rating meets, or None if it runs."""
        level = self._resolve_enforced_safety_level(owner_id, profile_name)
        return None if level == "restricted" else level

    def placement_refusal(self, profile_owner_id: int, profile_name: str,
                          channel: discord.abc.Messageable) -> Optional[str]:
        """Why this profile may not speak, to follow its name, or None if it may.

        The one runtime gate: every path that speaks as a character asks it, directly or
        through _check_unrestricted_safety_policy. Seating asks `seat_refusal` first, so a
        seat this refuses got here by a rating changing since, and a round skips it in silence.
        """
        key = self.profile_block(profile_owner_id, profile_name, channel)
        return PLACEMENT_REFUSALS[key] if key else None

    def seat_refusal(self, owner_id: int, profile_name: str, channel) -> Optional[str]:
        """Why this profile may not be seated, to follow its name, or None.

        A profile held for its rating is seated, and its check queued now, so it is usually
        done before the first turn.
        """
        key = self.profile_block(owner_id, profile_name, channel)
        if key == "checking":
            self.schedule_content_classification(
                *self.cog.profile_manager._resolve_effective_profile(owner_id, profile_name))
            return None
        return PLACEMENT_REFUSALS[key] if key else None

    def _check_unrestricted_safety_policy(self, profile_owner_id: int, profile_name: str, channel: discord.abc.Messageable) -> bool:
        return self.placement_refusal(profile_owner_id, profile_name, channel) is None

    async def _fetch_avatar_parts(self, owner_id: int, profile_name: str) -> List[Dict[str, Any]]:
        """Every avatar the profile can speak under as inline image parts, its usual one first:
        a mood avatar is shown as often, so it is judged with the rest. One at a time, so the
        downloads are never all in flight at once."""
        eff_owner, eff_name = self.cog.profile_manager._resolve_effective_profile(owner_id, profile_name)
        config = self.cog.profile_manager._get_profile_config(eff_owner, eff_name, False) or {}
        urls = dict.fromkeys(url for url in (config.get("custom_avatar_url"),
                                             *(config.get("mood_avatars") or {}).values()) if url)
        parts = []
        for url in urls:
            part = await self._fetch_avatar_part(eff_owner, eff_name, url)
            if part:
                parts.append(part)
        return parts

    async def _fetch_avatar_part(self, owner_id: int, profile_name: str,
                                 avatar_url: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """The profile's avatar -- or `avatar_url`, one of its mood avatars -- as an inline
        image part, or None.

        Absorbed from the retired auto-moderator, with one behavioural change that
        was a standing bug: a download failure is no longer fatal.

        The auto-moderator refused to publish when it could not fetch the avatar,
        which conflated "this image is unacceptable" with "this host would not serve
        *us*". Those are not the same, and the difference is visible in production:
        the webhook path never downloads anything -- it hands the URL to Discord and
        Discord's servers fetch it -- so an image the bot could not retrieve from a
        datacentre IP still rendered perfectly in chat, while publishing the same
        profile failed with a URL error. Now the image is one signal among several:
        if it cannot be had, the text is judged alone.
        """
        eff_owner, eff_name = self.cog.profile_manager._resolve_effective_profile(owner_id, profile_name)
        if avatar_url is None:
            config = self.cog.profile_manager._get_profile_config(eff_owner, eff_name, False) or {}
            avatar_url = config.get("custom_avatar_url")
        if not avatar_url:
            return None

        avatar_url = await signed_attachment_url(self.cog.bot.http, avatar_url)
        try:
            # The spoofed User-Agent rides on the request, not the client, so this
            # can share the process-wide pool with everything else.
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                                     "Chrome/120.0.0.0 Safari/537.36"}
            # Read in chunks and dropped at the cap. A plain GET had buffered whatever the
            # URL served before its size was ever tested.
            response = await get_capped(
                get_shared_client(), avatar_url, defaultConfig.CONTENT_CLASSIFY_MAX_IMAGE_BYTES,
                timeout=10.0, headers=headers)
        except Exception as e:
            print(f"Classifier could not fetch the avatar for {eff_owner}/{eff_name} "
                  f"({type(e).__name__}); judging the text alone.")
            return None
        if response.status_code != 200:
            print(f"Classifier could not fetch the avatar for {eff_owner}/{eff_name} "
                  f"(HTTP {response.status_code}); judging the text alone.")
            return None

        data = response.content
        if not data:
            return None

        content_type = (response.headers.get("Content-Type") or "image/png").split(";")[0].strip()
        if not content_type.startswith("image/"):
            # A URL that serves an HTML error page is not an avatar. Sending it as
            # an image part is a guaranteed provider error.
            return None

        return {"mime_type": content_type, "data": data}

    def _classifier_attempts(self, owner_id: int) -> Tuple[Dict[str, Any], List[tuple]]:
        """(the /mod tab's settings, [(provider, model, adapter, role)]) for this owner.

        Ships the attachment describer's chain, for the same reasons, on the profile
        owner's provider: its Primary, then its Fallback, on the owner's own key alone.
        Never a server's -- a profile is not owned by one -- and never the bot owner's: a
        user with no key gets no verdict, and the profile stays Pending. Built here rather
        than by the model factory, so only the two providers this loop knows are routed;
        `/mod` offers no other, and the tab's thinking and Hosts & Tier are applied here
        too. A repeat is tried once, as `run_with_fallback` does.
        """
        attempts = []
        side = self.cog.user_settings.effective_provider(owner_id)
        settings = system_settings(self.cog, "classifier", side)
        chain = dict.fromkeys(
            subscription_curated(self.cog, CLASSIFIER_KEYS[0])
            if self.cog.storage_manager.subscription_payer(None, owner_id)
            else (system_model(self.cog, k, side) for k in CLASSIFIER_KEYS))
        for role, value in zip(("primary", "fallback"), chain):
            if value.startswith("GOOGLE/"):
                attempts.append(("gemini", value[len("GOOGLE/"):], GoogleGenAIModel, role))
            elif value.startswith("OPENROUTER/"):
                attempts.append(("openrouter", value[len("OPENROUTER/"):], OpenRouterModel, role))
        return settings, attempts

    async def _ask_classifier(self, owner_id: int, attempt: tuple, settings: Dict[str, Any],
                              instruction: str, payload: list, answer: re.Pattern) -> Tuple[str, str, str]:
        """One model's reply to `instruction`: (status, text, why it is no answer).

        `status` is "success" when `answer` matches the text, "no_verdict" when the model
        replied without one -- blocked, empty, or words in place of the line -- "api_error"
        when the call failed, and "no_key" when the owner has no usable key for it. A block
        that is the provider refusing to read the profile at all (`_provider_refusal`, or a
        moderation 403 naming minors) is "blocked" instead, `why` the provider's own code.
        """
        provider, model_name, model_cls, role = attempt
        key = self.cog.storage_manager._get_api_key_for_user(owner_id, provider)
        if not key:
            # No key covers two cases: none assigned for this provider, or the
            # assigned key inside its rate-limit cooldown window. Recorded rather
            # than skipped in silence, so a run that never reached a provider does
            # not look identical to one that failed at it.
            return "no_key", "", f"{provider}: no usable key (unassigned, or in rate-limit cooldown)"
        status, raw, why = "api_error", "", ""
        try:
            # Resolved, never `{}`: an adapter handed no thinking params reasons at
            # "high". The classifier slot ships Medium, as /mod may change it.
            kwargs = {"api_key": key, "model_name": model_name, "system_instruction": instruction,
                      "thinking_params": resolve_thinking_params(settings, "classifier", role)}
            if model_cls is GoogleGenAIModel:
                kwargs["safety_settings"] = DEFAULT_SAFETY_SETTINGS
            else:
                kwargs["endpoint"] = resolve_openrouter_endpoint(settings, model_name)
                kwargs["service_tier"] = resolve_openrouter_service_tier(settings)
            model = model_cls(**kwargs)
            # No sampling parameters: each model judges at its own defaults. Claude
            # takes none, and thinking refuses the greedy ones where a model does.
            response = await model.generate_content_async(payload)
            candidate = response.candidates[0] if response and response.candidates else None
            parts = (candidate.content.parts or []) if candidate and candidate.content else []
            raw = "".join(getattr(p, "text", None) or "" for p in parts).strip()
            if _answer_line(answer, raw):
                status = "success"
            else:
                block = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
                finish = getattr(candidate, "finish_reason", None)
                refused = _provider_refusal(block, finish, getattr(candidate, "native_finish_reason", None))
                status = "blocked" if refused else "no_verdict"
                why = (refused if refused
                       else "answered without a verdict" if raw
                       else f"blocked ({block})" if block
                       else f"no text (finish reason {finish})" if finish else "no text")
        except Exception as e:
            # Not the body: a moderation 403 quotes the input it flagged.
            minors = next((str(r) for r in getattr(e, "moderation_reasons", ())
                           if _MINORS_FLAGGED.search(str(r))), None)
            status, why = ("blocked", minors) if minors else (status, f"{provider}: {e}")
        finally:
            self.cog._log_api_call(user_id=0, guild_id=None, context="content_classification",
                                   model_used=model_name, status=status)
        return status, raw, why

    async def _classify_profile_content(self, owner_id: int, profile_name: str) -> Optional[Dict[str, Any]]:
        """Runs the classifier over one profile and returns a content_rating record.

        The Primary first; the Fallback when it is down or replies without a verdict,
        since it is a different model and may judge what the first would not. Returns None
        when neither gave one -- no key, every provider failed, or no verdict from any --
        so the caller can leave the profile unclassified and retry, or flag it for the
        operator, rather than recording a wrong verdict.

        A refusing code refuses the profile and strikes its owner, so the other model is
        asked whether it holds, in that code's words (`REFUSAL_CONFIRMERS`). Agreed, the
        refusal is `confirmed_by` that model and settles itself. Disputed, or with no
        answer, it is still Refused but waits on the operator's `review`, with no strike --
        bar `UNREVIEWED_REFUSALS`, which wait on no one: their owner edits and resubmits.

        A verdict lighter than the one the profile stands on (`_rating_floor`) holds only
        once a second model agrees (`_hold_floor`).

        No verdict from any model, and at least one provider refusing to read it at all, is
        not flagged for the operator: it is held (`provider_blocked`) for its owner to edit.
        """
        self._last_classify_flagged = False  # a raise before any answer flags nothing
        self._last_classify_blocked = None
        surface = await asyncio.to_thread(self._moderated_surface, owner_id, profile_name)
        surface_hash = self._surface_hash(surface)
        floor, floor_reason = await asyncio.to_thread(
            lambda: _rating_floor((self.cog.profile_manager._get_profile_config(owner_id, profile_name, False) or {})
                                  .get("content_rating") or {}))

        prompt_text = self.cog.global_prompts.get(
            "CONTENT_CLASSIFIER", DEFAULT_CONTENT_CLASSIFIER_PROMPT)

        # All of it. It was cut at 6,000 characters, so behind a full persona the
        # instructions, lorebook and fixed examples went unread -- and the hash covers them,
        # so an edit there was "re-checked" unseen too. The editors bound it (~200,000
        # characters at every cap), well inside a classifier's context.
        parts = [f"<target_profile>\n{surface}\n</target_profile>"]
        parts.extend(await self._fetch_avatar_parts(owner_id, profile_name))

        payload = [{"role": "user", "parts": parts}]

        settings, attempts = self._classifier_attempts(owner_id)
        raw, answered_by = None, None
        failures, no_verdicts, blocked = [], [], []
        for n, attempt in enumerate(attempts):
            status, text, why = await self._ask_classifier(
                owner_id, attempt, settings, prompt_text, payload, _VERDICT_LINE)
            if status == "success":
                raw, answered_by = text, n
                break
            if status == "blocked":
                blocked.append(why)
                no_verdicts.append(f"{attempt[1]}: blocked ({why})")
            elif status == "no_verdict":
                no_verdicts.append(f"{attempt[1]}: {why}")
            else:
                failures.append(why)

        if answered_by is None:
            # A model replying with no verdict is never Allowed and never Refused: the
            # operator reviews it, bar a provider refusing to read it, which holds it.
            # Retrying cannot help within the backoff window either -- a key is not going
            # to appear in fifteen seconds, credit is not bought, and a block is not
            # lifted -- so tell the caller not to bother.
            self._last_classify_blocked = blocked[0] if blocked else None
            self._last_classify_flagged = bool(no_verdicts) and not blocked
            self._last_classify_failure = "; ".join(failures + no_verdicts) or "unknown"
            self._last_classify_retryable = not no_verdicts and (
                not all("no usable key" in f or _NO_CREDIT in f for f in failures)
                if failures else True)
            return None

        match = _answer_line(_VERDICT_LINE, raw)
        verdict = _VERDICTS[match.group(1)]

        # Only a recognised category code survives. The prompt asks for one, but a
        # model that answers in prose anyway must not have that prose stored and
        # shown back to the owner -- it would be describing their persona to them,
        # in the classifier's words. An unusable reason is dropped, and the embed
        # renders the generic label in its place.
        code = (match.group(2) or "").upper()
        reason = code if code in CONTENT_RATING_REASON_LABELS else None

        record = {
            "verdict": verdict,
            "hash": surface_hash,
            "model": attempts[answered_by][1],
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "reason": reason,
            "source": "classifier",
        }
        if reason in REFUSAL_CONFIRMERS:
            # Refused whatever the head said: an ALLOWED beside this code is a model
            # contradicting itself, and the operator's review settles it either way. What
            # the head said is kept for that review to restore (`clear_classifier_verdict`).
            record.update(verdict=CONTENT_RATING_REFUSED, underlying=verdict)
            other = next((a for i, a in enumerate(attempts) if i != answered_by), None)
            status, text, _why = (await self._ask_classifier(
                owner_id, other, settings,
                self.cog.global_prompts.get(*REFUSAL_CONFIRMERS[reason]),
                payload, _CONFIRM_LINE) if other else ("no_key", "", ""))
            confirmed = _answer_line(_CONFIRM_LINE, text) if status == "success" else None
            if confirmed and confirmed.group(1).upper() == "YES":
                record["confirmed_by"] = other[1]
            elif reason not in UNREVIEWED_REFUSALS:
                record["review"] = "disputed" if confirmed else "unconfirmed"
        elif floor and _STRICTNESS[verdict] < _STRICTNESS[floor]:
            record = await self._hold_floor(owner_id, record, floor, floor_reason, settings,
                                            attempts, answered_by, prompt_text, payload)
        return record

    async def _hold_floor(self, owner_id: int, record: Dict[str, Any], floor: str,
                          floor_reason: Optional[str], settings: Dict[str, Any], attempts: List[tuple],
                          answered_by: int, prompt_text: str, payload: list) -> Dict[str, Any]:
        """`record`, if a second model agrees it may go below `floor`; else the stricter answer.

        One answer is one draw -- each model judges at its own sampling defaults -- so small
        edits resubmitted could walk a refusal or a Not Allowed down until a draw came back
        lighter. Tightening needs no second opinion. Asked of the other model, or of the same
        one again when the chain has only one: a refusal its own confirmer's question, anything
        else the classifier's. Held Refused, it waits on the operator with no strike, or on
        no one for `UNREVIEWED_REFUSALS`.
        """
        second = next((a for i, a in enumerate(attempts) if i != answered_by), attempts[answered_by])
        if floor == CONTENT_RATING_REFUSED and floor_reason in REFUSAL_CONFIRMERS:
            status, text, _why = await self._ask_classifier(
                owner_id, second, settings, self.cog.global_prompts.get(*REFUSAL_CONFIRMERS[floor_reason]),
                payload, _CONFIRM_LINE)
            answer = _answer_line(_CONFIRM_LINE, text) if status == "success" else None
            if answer and answer.group(1).upper() == "NO":
                return record
            held, reason = CONTENT_RATING_REFUSED, floor_reason
        else:
            status, text, _why = await self._ask_classifier(
                owner_id, second, settings, prompt_text, payload, _VERDICT_LINE)
            answer = _answer_line(_VERDICT_LINE, text) if status == "success" else None
            held, reason = floor, floor_reason  # no answer: the stricter stands
            if answer:
                code = (answer.group(2) or "").upper()
                held = CONTENT_RATING_REFUSED if code in REFUSAL_CONFIRMERS else _VERDICTS[answer.group(1)]
                reason = code if code in CONTENT_RATING_REASON_LABELS else None
                if _STRICTNESS[held] <= _STRICTNESS[record["verdict"]]:
                    return record
        if held == CONTENT_RATING_REFUSED:
            review = {} if reason in UNREVIEWED_REFUSALS else {"review": "disputed" if answer else "unconfirmed"}
            return {**record, "verdict": held, "reason": reason, "underlying": record["verdict"], **review}
        return {**record, "verdict": held, "reason": reason,
                **({"model": second[1]} if answer else {})}

    def _store_content_rating(self, owner_id: int, profile_name: str, rating: Dict[str, Any]) -> bool:
        """Persists a verdict, unless a moderator override is pinned for this content."""
        index = self.cog.profile_manager._get_user_index(owner_id)
        is_borrowed = profile_name in index.get("borrowed", [])
        if is_borrowed:
            return False

        config = self.cog.profile_manager._get_profile_config(owner_id, profile_name, False)
        if config is None:
            return False

        existing = config.get("content_rating") or {}
        if (existing.get("source") == "owner_override"
                and existing.get("hash") == rating.get("hash")):
            # A moderator cleared this exact content; re-running the classifier over
            # it must not silently undo that. A later edit changes the hash and the
            # override lapses with the content it was granted for.
            return False

        # A hold for rating (`_held`) takes whatever lands, as any record does: it was a
        # wait for this verdict, never a verdict of its own.
        config["content_rating"] = rating
        self.cog.profile_manager._save_profile_config(owner_id, profile_name, config, False)
        self._invalidate_content_rating(owner_id, profile_name)
        self.cog.library._delist_unless_publishable(owner_id, profile_name)
        self._set_refused_entry(owner_id, self.cog.profile_manager._get_pid_from_name(owner_id, profile_name),
                                profile_name if _awaits_review(rating) else None)
        return True

    async def rating_is_stale(self, owner_id: int, profile_name: str) -> bool:
        """True when the profile's text has changed since its verdict was recorded.

        Hashes the moderated surface, which decrypts the whole persona -- so this
        belongs on deliberate paths only, never on the turn path. States that cannot
        go stale are answered before the hash is computed rather than after it.
        """
        eff_owner, eff_name = self.cog.profile_manager._resolve_effective_profile(owner_id, profile_name)
        config = self.cog.profile_manager._get_profile_config(eff_owner, eff_name, False) or {}
        rating = config.get("content_rating") or {}
        verdict = self._verdict_of(config)

        # Unrated has nothing to be stale against; Pending is mid-flight; Refused moves
        # only on a resubmission or the operator's Clear, never on an edit; and a hold for
        # rating is no verdict, so has no rated text to compare (resolve_stale_rating
        # queues its check).
        if verdict in (CONTENT_RATING_UNRATED, CONTENT_RATING_PENDING, CONTENT_RATING_REFUSED):
            return False
        if _held(rating):
            return False
        retry_after = rating.get("retry_after")
        if retry_after and time.time() < retry_after:
            return False

        stored = rating.get("hash")
        if not stored:
            return False

        current = await asyncio.to_thread(
            lambda: self._surface_hash(self._moderated_surface(eff_owner, eff_name)))
        return stored != current

    async def resolve_stale_rating(self, owner_id: int, profile_name: str) -> Optional[str]:
        """Handles a rating whose profile has been edited since it was judged.

        Returns what happened, for the caller to report, or None if nothing needed
        doing. This is the backstop for edits that do not go through the interactive
        prompt -- an import, a bulk apply, a restored backup, a future edit path.

        The rule is the whole point of the redesign:

        * **Undistributed** -- drop to Unrated. Free, no API call. The owner keeps
          using the profile exactly as before and re-submits when they next want to
          share it.
        * **Distributed** -- re-classify. Other people are running this profile on
          the strength of its verdict, so the verdict has to keep up.
        * **Not Allowed** -- re-classify too: dropped to Unrated it would run again
          unrated. It stays Not Allowed until the new verdict lands.

        A profile `_held` for its rating is checked whether edited or not: it runs nowhere
        until it is. These are the only paths that spend a call without being asked to, and
        they are bounded by the profiles actually shared, refused a run, or held.
        """
        eff_owner, eff_name = self.cog.profile_manager._resolve_effective_profile(owner_id, profile_name)
        if (int(eff_owner), eff_name) in self.cog.rating_prompts_open:
            # The owner is looking at PostEditRatingView, which settles this itself
            # when they choose or it times out. Deciding here as well meant a session
            # start could queue a re-check that then overwrote their "Set to Unrated".
            return None

        stale = await self.rating_is_stale(owner_id, profile_name)
        rating = (await asyncio.to_thread(self.cog.profile_manager._get_profile_config, eff_owner, eff_name, False)
                  or {}).get("content_rating") or {}
        if _held(rating):
            self.schedule_content_classification(eff_owner, eff_name)
            return "reclassify"

        if not stale:
            return None

        distributed = await asyncio.to_thread(
            self.cog.library.is_profile_distributed, eff_owner, eff_name)
        if distributed or rating.get("verdict") == CONTENT_RATING_ADULT:
            self.schedule_content_classification(eff_owner, eff_name)
            return "reclassify"

        await asyncio.to_thread(self.drop_to_unrated, eff_owner, eff_name)
        return "unrated"

    # --- Refused profiles, for /mod ----------------------------------------------
    #
    # REFUSED_INDEX_FILE, "<owner_id>:<pid>" -> {"name", "at"}: the operator's review list,
    # of refusals awaiting review (`_awaits_review`: disputed, unconfirmed or appealed) and
    # of profiles no model gave a verdict for (`flagged`). A confirmed or upheld refusal is
    # not on it: it settled itself. Answering "which?" from the shards would decrypt every
    # profile on the instance. Written where a verdict lands, is cleared, appealed, upheld
    # or is not given. A profile deleted or reset since is not removed here:
    # `refused_profiles` drops whatever no longer reads either when asked.

    def _set_refused_entry(self, owner_id: int, pid: Optional[str], name: Optional[str]) -> None:
        """Lists `pid` as refused under `name`, or delists it when `name` is None."""
        if not pid:
            return
        key = f"{int(owner_id)}:{pid}"
        with _REFUSED_INDEX_LOCK:
            index = IOManager.read_json(REFUSED_INDEX_FILE) or {}
            if name is not None:
                index[key] = {"name": name, "at": int(time.time())}
            elif index.pop(key, None) is None:
                return
            IOManager.write_json(index, REFUSED_INDEX_FILE)

    def refused_profiles(self) -> List[Tuple[int, str, int, Optional[str], Optional[str]]]:
        """(owner_id, current name, listed at, why flagged -- None when Refused, why a
        refusal awaits review -- None when flagged) for each profile still awaiting review,
        oldest first. Reads one config per entry, so call it in a thread."""
        index = IOManager.read_json(REFUSED_INDEX_FILE) or {}
        live, stale = [], []
        for key, entry in index.items():
            owner, _, pid = key.partition(":")
            name = self.cog.profile_manager._get_name_from_pid(int(owner), pid) if owner.isdigit() else None
            config = self.cog.profile_manager._get_profile_config(int(owner), name, False) if name else None
            rating = (config or {}).get("content_rating") or {}
            flagged = rating.get("flagged")
            review = rating.get("review") if _awaits_review(rating) else None
            if config is not None and (review or flagged):
                live.append((int(owner), name, int(entry.get("at") or 0), flagged, review))
            else:
                stale.append(key)
        if stale:
            with _REFUSED_INDEX_LOCK:
                current = IOManager.read_json(REFUSED_INDEX_FILE) or {}
                for key in stale:
                    current.pop(key, None)
                IOManager.write_json(current, REFUSED_INDEX_FILE)
        return sorted(live, key=lambda row: row[2])

    @staticmethod
    def rating_failure_hint(rating: Dict[str, Any]) -> str:
        """The likeliest fix for a classification that gave up, read off the failure it stored."""
        if rating.get("provider_blocked"):
            return ("Your AI provider's safety system refused to read this profile, so it "
                    "cannot run until it is rated. Edit it, then submit it again.")
        if _NO_CREDIT in (rating.get("check_failure") or rating.get("reason") or ""):
            return ("Your OpenRouter account is out of credit, which the rating models need: "
                    "top it up at <https://openrouter.ai/settings/credits>.")
        return "This usually means no API key was available: check `/settings`."

    def _record_classification_failure(self, owner_id: int, profile_name: str, reason: str,
                                       flagged: bool = False, blocked: Optional[str] = None):
        """Stamps a retry-after on a profile the classifier could not judge.

        `flagged`: the model answered with no verdict, so the profile also goes on the
        operator's review list -- never Refused, since a model producing no text is not a
        finding. The flag outlives a later failure of the plain kind, until a verdict
        lands: the model's, or the operator's (`moderator_verdict`).

        `blocked`: a provider refused to read it at all. No finding either, but it holds
        the profile (`provider_blocked`) until a model gives a verdict, and replaces any
        flag: the operator would be reading what the provider would not.
        """
        config = self.cog.profile_manager._get_profile_config(owner_id, profile_name, False)
        if config is None:
            return
        rating = config.get("content_rating") or {}
        if blocked:
            rating = {k: v for k, v in rating.items() if k != "flagged"}
        flag = reason[:200] if flagged else rating.get("flagged")
        flag = {"flagged": flag} if flag else {}
        hold = {"provider_blocked": blocked} if blocked else {}
        verdict = self._verdict_of(config)
        if verdict == CONTENT_RATING_ADULT and rating.get("source") == "owner_declared":
            # A hold's check failed. The hold stands, so it still runs nowhere, and the
            # retry-after stops every session setup re-queueing the same failure.
            config["content_rating"] = {
                **rating, "check_failure": reason[:200],
                "retry_after": time.time() + defaultConfig.CONTENT_CLASSIFY_RETRY_AFTER, **flag, **hold}
            self.cog.profile_manager._save_profile_config(owner_id, profile_name, config, False)
            self._invalidate_content_rating(owner_id, profile_name)
        elif verdict in (CONTENT_RATING_ADULT, CONTENT_RATING_REFUSED) or (
                verdict == CONTENT_RATING_GENERAL and not (flagged or blocked)):
            # A real verdict already stands; a later failure must not erase it. Bar a
            # Allowed one re-checked after an edit that the model would not rate: the
            # edited text is what its borrowers run, so it waits on review as Pending.
            return
        else:
            config["content_rating"] = {
                # Stays Pending rather than dropping to Unrated: the owner did ask for a
                # verdict, and the dashboard shows the stored reason and the retry time
                # against this state. Dropping to Unrated would silently discard their
                # submission because a key happened to be missing.
                "verdict": CONTENT_RATING_PENDING,
                "hash": None,
                "model": None,
                "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "reason": reason[:200],
                "source": "failed",
                "retry_after": time.time() + defaultConfig.CONTENT_CLASSIFY_RETRY_AFTER,
                **flag,
                **_carried_floor(rating),
                **hold,
            }
            self.cog.profile_manager._save_profile_config(owner_id, profile_name, config, False)
            self._invalidate_content_rating(owner_id, profile_name)
        if flagged or blocked:
            # Not the Pending that `_delist_unless_publishable` leaves listed: no verdict
            # is on its way.
            self.cog.library.unpublish_profile(owner_id, profile_name)
            self._set_refused_entry(owner_id, self.cog.profile_manager._get_pid_from_name(owner_id, profile_name),
                                    None if blocked else profile_name)

    def _classification_on_cooldown(self, owner_id: int, profile_name: str) -> bool:
        config = self.cog.profile_manager._get_profile_config(owner_id, profile_name, False) or {}
        rating = config.get("content_rating") or {}
        retry_after = rating.get("retry_after")
        return bool(retry_after) and time.time() < retry_after

    def clear_classifier_verdict(self, owner_id: int, profile_name: str, moderator_id: int) -> bool:
        """Operator appeal: pins a lighter verdict for exactly the content that was flagged.

        A Not Allowed verdict clears to Allowed. A Refusal clears to what the classifier's
        own verdict was beside its refusing code -- a false positive on a persona that is
        Not Allowed anyway stays Not Allowed. Keyed to the current surface hash, so the
        override lapses the moment the persona changes -- a cleared profile cannot be edited
        into what was flagged and keep its clearance.
        """
        config = self.cog.profile_manager._get_profile_config(owner_id, profile_name, False)
        if config is None:
            return False
        verdict = self._verdict_of(config)
        if verdict not in (CONTENT_RATING_ADULT, CONTENT_RATING_REFUSED):
            return False
        if _held(config.get("content_rating") or {}):
            # Waiting on a rating, not a review: nobody has read it to clear. Force
            # Re-classify rates it, and a flagged one takes the operator's verdict.
            return False
        cleared = CONTENT_RATING_GENERAL
        if verdict == CONTENT_RATING_REFUSED:
            underlying = (config.get("content_rating") or {}).get("underlying")
            cleared = CONTENT_RATING_GENERAL if underlying == CONTENT_RATING_GENERAL else CONTENT_RATING_ADULT

        surface = self._moderated_surface(owner_id, profile_name)
        self._set_refused_entry(owner_id, self.cog.profile_manager._get_pid_from_name(owner_id, profile_name), None)
        config["content_rating"] = {
            "verdict": cleared,
            "hash": self._surface_hash(surface),
            "model": None,
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "reason": f"Cleared by moderator {moderator_id}",
            "source": "owner_override",
        }
        self.cog.profile_manager._save_profile_config(owner_id, profile_name, config, False)
        self._invalidate_content_rating(owner_id, profile_name)
        return True

    def moderator_verdict(self, owner_id: int, profile_name: str, verdict: str,
                          moderator_id: int) -> bool:
        """Operator review of a flagged profile: the verdict the model would not give,
        pinned to exactly this content as `clear_classifier_verdict` pins its own, and
        landed as any verdict does, a hold for rating included.
        True when it was stored as given, so the caller strikes a Refusal."""
        config = self.cog.profile_manager._get_profile_config(owner_id, profile_name, False)
        if (not ((config or {}).get("content_rating") or {}).get("flagged")
                or verdict not in (CONTENT_RATING_GENERAL, CONTENT_RATING_ADULT, CONTENT_RATING_REFUSED)):
            return False
        record = {
            "verdict": verdict,
            "hash": self._surface_hash(self._moderated_surface(owner_id, profile_name)),
            "model": None,
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "reason": f"Set by moderator {moderator_id}",
            "source": "owner_override",
        }
        if verdict == CONTENT_RATING_REFUSED:
            # No code: the operator refused it, and the dashboard says so.
            record.update(reason=None, underlying=CONTENT_RATING_ADULT, upheld_by=moderator_id)
        return self._store_content_rating(owner_id, profile_name, record)

    def appeal_refusal(self, owner_id: int, profile_name: str) -> bool:
        """The owner asks the operator to review a refusal that settled itself. Not one the
        operator already decided (`upheld_by`), so once per refusal: a new one, after an edit
        and a re-rating, may be appealed again. Never one no person reviews
        (`UNREVIEWED_REFUSALS`): its owner edits and resubmits instead."""
        config = self.cog.profile_manager._get_profile_config(owner_id, profile_name, False)
        rating = (config or {}).get("content_rating") or {}
        if (rating.get("verdict") != CONTENT_RATING_REFUSED or rating.get("review")
                or rating.get("upheld_by") or rating.get("reason") in UNREVIEWED_REFUSALS):
            return False
        config["content_rating"] = {**rating, "review": "appealed"}
        self.cog.profile_manager._save_profile_config(owner_id, profile_name, config, False)
        self._invalidate_content_rating(owner_id, profile_name)
        self._set_refused_entry(owner_id, self.cog.profile_manager._get_pid_from_name(owner_id, profile_name), profile_name)
        return True

    def uphold_refusal(self, owner_id: int, profile_name: str, moderator_id: int) -> bool:
        """The operator's review agrees with a refusal awaiting it: it settles, leaves the
        review list, and the caller strikes it (`add_strike` counts a profile once)."""
        config = self.cog.profile_manager._get_profile_config(owner_id, profile_name, False)
        rating = (config or {}).get("content_rating") or {}
        if not _awaits_review(rating):
            return False
        config["content_rating"] = {**{k: v for k, v in rating.items() if k != "review"},
                                    "upheld_by": moderator_id}
        self.cog.profile_manager._save_profile_config(owner_id, profile_name, config, False)
        self._invalidate_content_rating(owner_id, profile_name)
        self._set_refused_entry(owner_id, self.cog.profile_manager._get_pid_from_name(owner_id, profile_name), None)
        return True

    def is_held(self, owner_id: int, profile_name: str) -> bool:
        """Whether the profile runs nowhere until its rating lands (`_held`), read at the
        source as a borrow's rating is."""
        eff_owner, eff_name = self.cog.profile_manager._resolve_effective_profile(owner_id, profile_name)
        config = self.cog.profile_manager._get_profile_config(eff_owner, eff_name, False) or {}
        return _held(config.get("content_rating") or {})

    def hold_for_rating(self, owner_id: int, profile_name: str) -> bool:
        """Holds a profile that arrived marked 18+ -- a card tagged NSFW, a generated profile
        tagged 18+ -- so it runs nowhere until the classifier has rated it. Not queued here:
        this may run in a thread, and the next seat or resolve_stale_rating queues it.

        A borrow is not the borrower's to hold, and a Not Allowed or Refused profile already
        runs nowhere. True when the profile is now held or already runs nowhere.
        """
        index = self.cog.profile_manager._get_user_index(owner_id)
        if profile_name in index.get("borrowed", []):
            return False
        config = self.cog.profile_manager._get_profile_config(owner_id, profile_name, False)
        if config is None:
            return False
        rating = config.get("content_rating") or {}
        if rating.get("verdict") in (CONTENT_RATING_ADULT, CONTENT_RATING_REFUSED):
            return True
        config["content_rating"] = {**_held_for_rating(), **_carried_hold(rating)}
        self.cog.profile_manager._save_profile_config(owner_id, profile_name, config, False)
        self._invalidate_content_rating(owner_id, profile_name)
        self.cog.library._delist_unless_publishable(owner_id, profile_name)
        return True

    async def submit_for_rating(self, owner_id: int, profile_name: str) -> Tuple[bool, str]:
        """Owner-initiated: move a profile to Pending and queue the classifier.

        The only path that starts a classification from nothing. Everything else
        either re-runs an existing verdict for a distributed profile, or drops the
        profile back to Unrated and waits to be asked again.

        **Async on purpose.** schedule_content_classification needs a running event
        loop to create its task, and returns silently when there is not one. An
        earlier version of this was a plain sync method that callers reached through
        asyncio.to_thread for the file writes -- which meant the schedule call landed
        on a worker thread every single time, hit that guard, and did nothing. The
        profile was written Pending, the user was told it had been submitted, and no
        job existed to ever move it off Pending. The blocking work is threaded
        individually here so the scheduling stays on the loop.
        """
        index = await asyncio.to_thread(self.cog.profile_manager._get_user_index, owner_id)
        if profile_name in index.get("borrowed", []):
            return False, "A borrowed profile is rated by its owner, not by you."

        verdict, _ = await asyncio.to_thread(self._content_rating_state, owner_id, profile_name)
        if verdict in (CONTENT_RATING_REFUSED, CONTENT_RATING_ADULT):
            # Re-rated in place: written Pending, it would run in the meantime -- and stay
            # running if the check never landed. A hold for rating is one of these too.
            if (int(owner_id), profile_name) in self.cog.pending_classifications:
                return False, "This profile is already being re-rated."
            rating = (await asyncio.to_thread(self.cog.profile_manager._get_profile_config, owner_id, profile_name, False)
                      or {}).get("content_rating") or {}
            if verdict == CONTENT_RATING_REFUSED:
                # Only once edited: the same text asked again is a re-roll, until one answer
                # comes back clean. An owner who thinks the refusal wrong appeals it instead.
                # A Not Allowed needs no such guard: going lighter takes a second model.
                current = await asyncio.to_thread(
                    lambda: self._surface_hash(self._moderated_surface(owner_id, profile_name)))
                if rating.get("hash") == current:
                    if rating.get("review"):
                        return False, "The bot operator is reviewing this refusal."
                    return False, ("This profile has not changed since it was refused, so it would "
                                   "be judged on the same text. Edit it first"
                                   + ("." if rating.get("upheld_by")
                                      or rating.get("reason") in UNREVIEWED_REFUSALS else
                                      ", or choose Request Review if you think the refusal is a mistake."))
            self.schedule_content_classification(owner_id, profile_name)
            state = "held" if _held(rating) else CONTENT_RATING_LABELS[verdict]
            return True, (f"Submitted. It stays {state} until the new verdict lands, which "
                          "usually takes a few seconds.")
        if verdict == CONTENT_RATING_PENDING:
            # A Pending profile is either genuinely in flight or a submission that
            # failed and stamped a retry-after. Refusing both alike strands the
            # second kind permanently: the internal retries are exhausted, nothing
            # re-queues it, and the owner is told to wait for a verdict that will
            # never arrive. Once the cooldown has passed, let them try again.
            eff_owner, eff_name = await asyncio.to_thread(
                self.cog.profile_manager._resolve_effective_profile, owner_id, profile_name)
            config = await asyncio.to_thread(self.cog.profile_manager._get_profile_config, eff_owner, eff_name, False)
            rating = (config or {}).get("content_rating") or {}
            retry_after = rating.get("retry_after")
            if not retry_after:
                return False, "This profile is already awaiting a verdict."
            if time.time() < retry_after:
                return False, (f"The last attempt failed and it will retry "
                               f"<t:{int(retry_after)}:R>. {self.rating_failure_hint(rating)}")

        config =await asyncio.to_thread(self.cog.profile_manager._get_profile_config, owner_id, profile_name, False)
        if config is None:
            return False, "Profile not found."

        # A flag stays until a verdict lands: submitting again is not a review.
        previous = config.get("content_rating") or {}
        flagged = previous.get("flagged")
        config["content_rating"] = {
            "verdict": CONTENT_RATING_PENDING,
            "hash": None,
            "model": None,
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "reason": None,
            "source": "submitted",
            **({"flagged": flagged} if flagged else {}),
            **_carried_floor(previous),
        }
        await asyncio.to_thread(self.cog.profile_manager._save_profile_config, owner_id, profile_name, config, False)
        self._invalidate_content_rating(owner_id, profile_name)

        # On the loop, not in a thread -- see the note above.
        self.schedule_content_classification(owner_id, profile_name)
        return True, "Submitted. The verdict usually lands within a few seconds."

    async def settled_capability(self, owner_id: int, profile_name: str, capability: str,
                                 *, wait: bool = False) -> Tuple[bool, Optional[str]]:
        """`content_capability`, once a rating the profile has outgrown is settled.

        The stored verdict is read without re-hashing (`_content_rating_state`), so an edit
        no prompt caught was shared, published or opened in Global Chat on a verdict about
        other text. The deliberate distribution paths ask this instead. It hashes the
        persona: never on the turn path. `wait` holds for the re-check a distributed
        profile is given, for a caller that has already answered its interaction.
        """
        outcome = await self.resolve_stale_rating(owner_id, profile_name)
        if outcome == "reclassify" and wait:
            await self.await_classification(*self.cog.profile_manager._resolve_effective_profile(owner_id, profile_name))
        return self.content_capability(owner_id, profile_name, capability)

    async def rate_for_distribution(self, owner_id: int, profile_name: str,
                                    capability: str) -> Tuple[bool, Optional[str]]:
        """`settled_capability`, with an Unrated profile submitted and its verdict awaited.

        What Publish, Send and Send as Clone call, so rating is part of the press rather
        than a trip to Content Safety first: asking to distribute a profile is asking for
        the verdict that permits it. The caller must have answered its interaction -- the
        wait runs up to CONTENT_CLASSIFY_UI_WAIT_SECONDS.
        """
        allowed, reason = await self.settled_capability(owner_id, profile_name, capability, wait=True)
        verdict, _ = self._content_rating_state(owner_id, profile_name)
        if allowed or verdict not in (CONTENT_RATING_UNRATED, CONTENT_RATING_PENDING):
            return allowed, reason
        if (int(owner_id), profile_name) not in self.cog.pending_classifications:
            # Unrated, or a Pending submission that failed: submit_for_rating refuses one
            # still inside its retry window, and says when it will retry.
            ok, message = await self.submit_for_rating(owner_id, profile_name)
            if not ok:
                return False, message
        await self.await_classification(owner_id, profile_name)
        allowed, reason = self.content_capability(owner_id, profile_name, capability)
        rating = (self.cog.profile_manager._get_profile_config(owner_id, profile_name, False) or {}).get("content_rating") or {}
        if not allowed and rating.get("source") == "failed":
            # Given up, not still running: "waiting on the verdict" would send them off to wait.
            lead = "" if rating.get("provider_blocked") else "The content rating could not be reached. "
            return False, lead + self.rating_failure_hint(rating)
        return allowed, reason

    def drop_to_unrated(self, owner_id: int, profile_name: str) -> bool:
        """Returns a profile to Unrated without spending a call.

        What an edit to an undistributed profile does. The owner keeps using it
        exactly as before -- Unrated and Allowed are identical at runtime -- and
        re-submits whenever they next want to share it.
        """
        config = self.cog.profile_manager._get_profile_config(owner_id, profile_name, False)
        if config is None:
            return False
        rating = config.get("content_rating") or {}
        if self._verdict_of(config) in (CONTENT_RATING_REFUSED, CONTENT_RATING_ADULT):
            return False  # it runs nowhere, and dropped it would run unrated (a hold too)

        config["content_rating"] = {
            "verdict": CONTENT_RATING_UNRATED,
            "hash": None,
            "model": None,
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "reason": None,
            "source": "edited",
            **_carried_floor(rating),
        }
        self.cog.profile_manager._save_profile_config(owner_id, profile_name, config, False)
        self._invalidate_content_rating(owner_id, profile_name)
        self.cog.library._delist_unless_publishable(owner_id, profile_name)
        return True

    def schedule_content_classification(self, owner_id: int, profile_name: str):
        """Queues a profile for classification. Never awaited.

        Not called from the edit paths: an owner submitting, the re-check of a profile that
        is distributed or Not Allowed (`resolve_stale_rating`), or a profile held for its
        rating being seated or settled. Every other edit either drops the profile to Unrated
        or, for a distributed one, asks the owner which they want.
        """
        index = self.cog.profile_manager._get_user_index(owner_id)
        if profile_name in index.get("borrowed", []):
            return
        if self._classification_on_cooldown(owner_id, profile_name):
            return

        key = (int(owner_id), profile_name)
        self._invalidate_content_rating(owner_id, profile_name)
        if key in self.cog.pending_classifications:
            return

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # Called from a worker thread with no loop, so no task can be created.
            # Loud, because a silent return here is what left submitted profiles
            # stuck on Pending forever: the caller had already written the state and
            # told the user it was in progress, and nothing existed to finish it.
            # The invalidation above still stands, and resolve_stale_rating picks the
            # profile up next time the dashboard or a session touches it.
            print(f"Classification for {owner_id}/{profile_name} was scheduled from a "
                  f"worker thread with no event loop and has been dropped. Call this "
                  f"from the loop.")
            self.cog.pending_classifications.pop(key, None)
            self.cog.classification_events.pop(key, None)
            return

        self.cog.pending_classifications[key] = 0
        # Created before the task, so a dashboard that starts waiting in the same
        # tick finds an Event to wait on rather than concluding nothing is running.
        self.cog.classification_events[key] = asyncio.Event()
        asyncio.create_task(self._run_classification_job(owner_id, profile_name))

    async def await_classification(self, owner_id: int, profile_name: str,
                                   timeout: Optional[float] = None) -> bool:
        """Waits for an in-flight classification to land. True if it finished in time.

        The dashboards used to `asyncio.sleep(2.5)` and repaint. That is a guess, and
        a bad one: a classifier call plus an avatar fetch routinely outruns it, and
        the repaint then renders the same Pending the user was already looking at --
        with nothing left in the system to update it afterwards. Reopening the
        dashboard appeared to fix it only because that re-read the config once the
        job had already landed.

        Returns False on timeout, which is not an error: the job is still running and
        the caller should repaint with whatever the current state is.
        """
        if timeout is None:
            timeout = defaultConfig.CONTENT_CLASSIFY_UI_WAIT_SECONDS
        key = (int(owner_id), profile_name)
        event = self.cog.classification_events.get(key)
        if event is None:
            # No job registered. Either it finished between the caller's check and
            # this call -- the common case, and a success -- or nothing was ever
            # queued, which the caller's repaint will show as it is.
            return key not in self.cog.pending_classifications
        try:
            await asyncio.wait_for(event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def _run_classification_job(self, owner_id: int, profile_name: str):
        key = (int(owner_id), profile_name)
        try:
            for attempt in range(defaultConfig.CONTENT_CLASSIFY_MAX_ATTEMPTS):
                self.cog.pending_classifications[key] = attempt + 1
                try:
                    rating = await self._classify_profile_content(owner_id, profile_name)
                except Exception as e:
                    print(f"Classification job errored for {owner_id}/{profile_name}: {e}")
                    rating = None
                    # A raise is no finding, and another job -- any profile's, any time
                    # since -- may have set these, so all four or its reason is recorded here.
                    self._last_classify_flagged, self._last_classify_blocked = False, None
                    self._last_classify_retryable, self._last_classify_failure = True, f"error: {e}"

                if rating:
                    stored = await asyncio.to_thread(self._store_content_rating, owner_id, profile_name, rating)
                    if stored:
                        self.cog.library._restamp_listing(owner_id, profile_name)
                    if stored and rating.get("confirmed_by"):
                        # Two models agreed; a refusal awaiting review strikes only when
                        # the operator upholds it (`uphold_refusal`). On the loop: the
                        # strike shares blacklist.json with /mod's writes.
                        self.cog.server_manager.add_strike(
                            owner_id, self.cog.profile_manager._get_pid_from_name(owner_id, profile_name), profile_name)
                    return

                if not getattr(self, "_last_classify_retryable", True):
                    break

                if attempt < defaultConfig.CONTENT_CLASSIFY_MAX_ATTEMPTS - 1:
                    await asyncio.sleep(5 * (2 ** attempt))

            # Out of attempts. Record why and when, so the next dashboard render or
            # session setup does not immediately re-queue the same doomed job -- that
            # loop is what turned one unclassifiable profile into an endless stream of
            # give-up lines. The profile stays Pending, which runs normally -- unless its
            # provider would not read it (`provider_blocked`) -- but cannot be shared
            # until a verdict is actually reached.
            reason = getattr(self, "_last_classify_failure", None) or "unknown"
            print(f"Classification gave up for {owner_id}/{profile_name}: {reason}. "
                  f"Retrying no sooner than {defaultConfig.CONTENT_CLASSIFY_RETRY_AFTER}s from now.")
            await asyncio.to_thread(self._record_classification_failure, owner_id, profile_name, reason,
                                    getattr(self, "_last_classify_flagged", False),
                                    getattr(self, "_last_classify_blocked", None))
        finally:
            self.cog.pending_classifications.pop(key, None)
            # Popped before it is set, so a waiter arriving after this point falls
            # through to the pending_classifications check and sees "finished"
            # rather than waiting on an Event nothing will ever set again. Waiters
            # already inside wait() hold their own reference and are still woken.
            done = self.cog.classification_events.pop(key, None)
            if done is not None:
                done.set()
