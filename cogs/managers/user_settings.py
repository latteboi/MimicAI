"""A user's own settings, kept in their index.json beside the profile maps: About Me (timezone,
provider preference, registration) and Override Defaults."""

import os
import time
from typing import Dict, List, Any, Optional
from ..utils.user_defaults import (
    model_provider, model_slot_defaults, other_provider, sanitise_defaults)
from ..utils.helpers import is_real_model
from ..utils.constants import USERS_DIR, defaultConfig, MODEL_PROVIDERS, PROVIDER_CHOICES, PROVIDER_SUBSCRIPTION
from ..utils.helpers import birth_time, is_real_model



class UserSettings:
    """A user's own settings, kept in their index.json beside the profile maps: About Me
    (timezone, provider preference, registration) and Override Defaults. Reached as
    `cog.user_settings`."""

    def __init__(self, cog):
        self.cog = cog

    def _get_user_defaults(self, user_id: int) -> Dict[str, Any]:
        """The user's standing profile defaults, filtered to what is still defaultable.

        Returned as a copy: `_get_user_index` hands back the cached index by reference,
        and the settings screen edits what it is given.
        """
        index = self.cog.profile_manager._get_user_index(user_id)
        return dict(sanitise_defaults(index.get("defaults")))

    def _save_user_defaults(self, user_id: int, defaults: Dict[str, Any]):
        """Replaces the stored defaults. Sparse -- only keys the user actually chose.

        An empty dict removes the key entirely rather than storing `{}`, so "never
        opened this screen" and "opened it and cleared everything" look the same on
        disk, which is what they mean.
        """
        index = self.cog.profile_manager._get_user_index(user_id)
        cleaned = sanitise_defaults(defaults)
        if cleaned:
            index["defaults"] = cleaned
        else:
            index.pop("defaults", None)
        self.cog.profile_manager._save_user_index(user_id, index)

    # --- About Me: the user's own settings, not a character's ----------------------
    #
    # `index.json["about"]`, a sibling of "defaults" rather than a key inside it:
    # "defaults" is fed to apply_defaults and means "stamp this onto profiles", and
    # its contents are derived from PROFILE_ACTIONS by defaultable_keys(). A user's
    # timezone is not a profile setting and must not become a fourth hand-written
    # list of keys.
    #
    # Plaintext in index.json, which is already cached in cog.user_indices, because
    # the timezone is read on the turn path -- once per user message, to format that
    # message's history line. It used to be read off whichever profile the user
    # happened to have active in the channel, through _get_profile_config, which
    # caches nothing and so paid a decrypt, a zstd decompress and a parse for
    # one string. Worse, it was the wrong string: a user's clock is the user's, and
    # a user with no active profile silently got UTC.
    #
    # Sparse, for the same reason "defaults" is: absent means "never set", which must
    # stay distinguishable from a value that happens to match today's fallback.

    def get_user_about(self, user_id: int) -> Dict[str, Any]:
        """The user's own settings. Sparse, so an absent key means "never set".

        Returned as a copy: `_get_user_index` hands back the cached index by
        reference, and the settings screen edits what it is given.

        No id is nobody. A session with no `owner_id` once asked for its owner's clock
        here, and the index read then wrote one -- `users/None/`.
        """
        if user_id is None:
            return {}
        about = self.cog.profile_manager._get_user_index(user_id).get("about")
        return dict(about) if isinstance(about, dict) else {}

    def save_user_about(self, user_id: int, about: Dict[str, Any]):
        """Replaces the stored About Me block, dropping empties to keep it sparse."""
        index = self.cog.profile_manager._get_user_index(user_id)
        cleaned = {k: v for k, v in (about or {}).items() if v not in (None, "")}
        if cleaned:
            index["about"] = cleaned
        else:
            index.pop("about", None)
        self.cog.profile_manager._save_user_index(user_id, index)

    def user_timezone(self, user_id: int) -> str:
        """The timezone this user's own messages are timestamped in.

        The single answer for the three history-formatting paths -- a new turn
        (`triggers`), a batched turn (`generation_service`) and a message edit
        (`event_listeners`) -- which had each grown the same four lines resolving it
        off the user's active profile.

        Not to be confused with a profile's `timezone`, which is the character's
        clock and stays where it is: that one is what `<current_time>` and a
        speaker's own history line are built from.
        """
        return self.get_user_about(user_id).get("timezone") or "UTC"

    def provider_preference(self, user_id: Optional[int]) -> Optional[str]:
        """The provider whose models this user's profiles ship on: "gemini", "openrouter" or
        "none", else None. The subscription runs on OpenRouter, so it reads as "openrouter"
        here; `provider_choice` is what was chosen.

        Cheap enough for the turn path -- the index is cached -- which is why the chain
        reads this and not `effective_provider`.
        """
        choice = self.provider_choice(user_id)
        return "openrouter" if choice == PROVIDER_SUBSCRIPTION else choice

    def provider_choice(self, user_id: Optional[int]) -> Optional[str]:
        """What the user chose in `/start` or Override Defaults, as those screens show it."""
        return self.get_user_about(user_id).get("provider") if user_id else None

    def set_provider_preference(self, user_id: int, provider: str) -> None:
        """The first call, from `/start`, is registering. See `is_registered`."""
        if provider not in PROVIDER_CHOICES:
            raise ValueError(f"Not a provider choice: {provider!r}")
        about = self.get_user_about(user_id)
        if about.get("provider") not in PROVIDER_CHOICES:
            # The first choice registers, and stamps when -- a later switch never moves it.
            # A directory that predates /start keeps its own date, not today's.
            about["registered_at"] = int(birth_time(os.path.join(USERS_DIR, str(user_id))) or time.time())
        about["provider"] = provider
        self.save_user_about(user_id, about)

    def is_registered(self, user_id: Optional[int]) -> bool:
        """Whether this user has set up through `/start` -- the one way in.

        Every path that creates a user's own data asks this first and answers
        NOT_REGISTERED: a profile, a borrow, a key, a setting, an accepted share. Until
        then nothing is stored under their id, whatever they read or say in a session.
        """
        return self.get_user_about(user_id).get("provider") in PROVIDER_CHOICES

    def effective_provider(self, user_id: int) -> str:
        """The provider this user's new models ship on: the preferred one, unless they
        hold a key for the other alone -- then that one, preference or not, since a
        profile made on a provider they cannot reach says nothing at all.

        Read on creation, borrow and generation only: it decrypts the key file.
        """
        preferred = self.provider_preference(user_id)
        if preferred not in MODEL_PROVIDERS:
            preferred = "gemini"
        other = other_provider(preferred)
        if (not self._user_holds_provider_key(user_id, preferred)
                and self._user_holds_provider_key(user_id, other)):
            return other
        return preferred

    def _user_holds_provider_key(self, user_id: int, provider: str) -> bool:
        """Whether the user has a key of this provider in any of their four slots.

        Deliberately looser than `_get_api_key_for_user`, which also requires the key
        to be *assigned*. This is used at borrow time to decide whether a profile has
        any chance of running, and an unassigned key is a step in `/start`, not a
        reason to rewrite someone's model choice out from under them.

        Ollama needs no key -- it is a URL the bot dials -- so it holds for whoever may
        use it at all, which is the bot owner alone (see `may_use_ollama`).
        """
        if provider == "ollama":
            return self.may_use_ollama(user_id)
        # The MimicAI subscription is an OpenRouter key the user holds, though no slot does.
        if provider == "openrouter" and self.cog.subscriptions.active(user_id):
            return True
        try:
            keys_data = self.cog.storage_manager._get_user_keys_data(user_id) or {}
        except Exception:
            return False
        return any(slot.get("provider") == provider and slot.get("key")
                   for slot in (keys_data.get("slots") or {}).values()
                   if isinstance(slot, dict))

    @staticmethod
    def may_use_ollama(user_id: Optional[int]) -> bool:
        """Only the bot owner may configure or run Ollama models.

        An Ollama host receives every message the profile sees, other members' included,
        so it has to be the instance operator's own machine -- see OLLAMA_OWNER_ONLY. The
        model factory asks this with the owner of the config that chose the model, and
        refuses when it cannot tell whose that is.
        """
        try:
            return user_id is not None and int(user_id) == int(defaultConfig.DISCORD_OWNER_ID)
        except (TypeError, ValueError):
            return False

    def _rescue_unusable_models(self, user_id: int, config: Dict[str, Any]) -> List[str]:
        """Repoints model slots naming a provider this user has no key for.

        A borrow arrives as a copy of the author's config, model choices included. An
        author on OpenRouter therefore handed every borrower a profile that could not
        generate a single word, and the failure surfaced as "OpenRouter API Key not
        found" to someone who had just finished adding a Google key -- an error about
        a decision they never made.

        Only ever moves a slot *to* a provider the user does hold, so a user with no
        Google key is not rescued from OpenRouter onto a Google model they equally
        cannot run. If neither side is usable the author's value stays: there is no
        right answer, and churning it would only hide where the problem came from.

        The target is the slot's shipped value on the user's effective provider, so an
        OpenRouter-only borrower lands on the models OpenRouter ships.
        """
        rescued = []
        for slot, platform_value in model_slot_defaults(self.effective_provider(user_id)).items():
            value = config.get(slot)
            if not is_real_model(value):
                continue
            if self._user_holds_provider_key(user_id, model_provider(value)):
                continue
            if not self._user_holds_provider_key(user_id, model_provider(platform_value)):
                continue
            config[slot] = platform_value
            rescued.append(slot)
        return sorted(rescued)
