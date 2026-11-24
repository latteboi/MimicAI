import os
import re
import secrets
import shutil
import datetime
import asyncio
from discord.ext import tasks
import discord
from typing import Dict, List, Any, Optional, Tuple
from ..utils.user_defaults import (
    MODEL_SLOT_PAIRS, apply_defaults, model_slot_defaults, platform_model_defaults)
from ..utils.constants import (
    USERS_DIR, PROFILE_NAME_SIDECAR, PID_CLASS_PREFIXES, PROFILE_MESSAGE_MAX_LENGTH, defaultConfig, PRIMARY_MODEL_NAME, FALLBACK_MODEL_NAME, DEFAULT_LTM_SUMMARIZATION_INSTRUCTIONS, DEFAULT_ANTI_REPETITION_PROMPT, NO_FALLBACK, IMAGE_MODEL_KEYS, AUDIO_MODEL_KEYS, CONTENT_RATING_UNRATED, CONTENT_RATING_ADULT, DEFAULT_IMAGE_MODEL, DEFAULT_SPEECH_MODEL, DEFAULT_SPEECH_VOICE, NEW_PROFILE_SPEECH_TEMPERATURE, UNREADABLE_MEDIA_DEFAULT, SYSTEM_PROFILE_MODEL, SYSTEM_PROFILE_FALLBACK_MODEL, MODEL_PROVIDERS)
from ..utils.helpers import is_shipped_ltm_prompt, ping_people, pick_mood_avatar
from ..utils import signatures
from ..utils.discord_cdn import is_web_url, unsigned_attachment_url
from .storage_manager import IOManager
try:
    import orjson as json
except ImportError:
    import json


#: What _accept_share_request used to write as a borrow's source PID when it could
#: not determine one. It is not a PID: every borrow that landed on it keys the same
#: "<owner>:00000000" entry in borrows.json, so _borrowers_of returns the borrowers
#: of unrelated profiles and a cascade delete reaches borrows of a profile that was
#: never deleted. Recognised here only so a borrow's readers can refuse it, until
#: `_validate_and_clean_borrowed_profiles` reaps the borrows still carrying it.
_UNKNOWN_SOURCE_PID = "00000000"

#: Model slots no provider preference fills: image and speech bill per use, so their owner
#: picks one. A new profile writes None to them, and a borrow keeps its author's.
_UNSHIPPED_MODEL_KEYS = IMAGE_MODEL_KEYS | AUDIO_MODEL_KEYS


def _carried_hold(rating: Dict[str, Any]) -> Dict[str, str]:
    """A provider's refusal to read the profile, for any record replacing `rating` but a
    verdict: only a model reading it lifts the hold, so no rewrite of the record may."""
    return {"provider_blocked": rating["provider_blocked"]} if rating.get("provider_blocked") else {}


def _held_for_rating() -> Dict[str, Any]:
    """A profile that arrived marked 18+, held as a content_rating until the classifier rates
    it: it runs nowhere meanwhile (`content_rating_manager._held`), and the verdict replaces
    it. `owner_declared` is the source an 18+ declaration was stored under before adult
    content was removed, which old records still carry. The caller, or the next
    resolve_stale_rating, queues the check."""
    return {"verdict": CONTENT_RATING_ADULT, "hash": None, "model": None,
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "reason": "Held for its content rating", "source": "owner_declared"}


class ProfileManager:
    """Owns profile CRUD, personal/borrowed inheritance resolution, private shares, and cloning.

    Holds a back-reference to the parent cog for state/logic not yet migrated
    (fernet, the generic shard system, and shared instance caches),
    per the transitional Dependency Injection pattern in CLAUDE.md.
    """

    def __init__(self, cog):
        self.cog = cog

    # Name resolution across the profile classes checks the user's OWN profiles
    # before the global System ones. It used to be the other way around, which meant
    # a user who created a personal profile sharing a name with any System profile
    # -- 'mimicguide' being one that ships -- had their own profile shadowed
    # everywhere a name was resolved: edits landed on it, but generation, prompts and
    # appearance all read the System profile instead. No creation path checks the
    # System index for collisions, so the name was accepted and the profile then
    # quietly did not exist. System stays the fallback, so a name with no personal
    # profile behind it still resolves to it exactly as before.

    def _system_index(self) -> Dict[str, str]:
        """The System name -> PID map. Always the bot owner's index, never the caller's.

        System profiles are stored in the bot owner's tree like any other profile --
        an X-prefixed PID under users/<owner>/profiles/ -- but they are addressable
        by everyone. Every reader must therefore look them up in the OWNER's index;
        a member's own index has an empty "system" map and always will.

        This is the single point that knows that. Sites used to re-derive it three
        different ways -- the caller's index, the owner's index, and a
        pid.startswith("X") test -- and the ones reading the caller's index silently
        found nothing, which is why System profiles were invisible to everyone but
        the owner in listings and autocomplete while _resolve_effective_profile
        happily granted access to them.
        """
        owner_index = self._get_user_index(int(defaultConfig.DISCORD_OWNER_ID))
        system = owner_index.get("system", {})
        return system if isinstance(system, dict) else {}

    def _is_system_name(self, user_id: int, profile_name: str) -> bool:
        """Whether `profile_name` resolves to a System profile *for this user*.

        Personal and borrowed shadow System, matching _resolve_effective_profile, so
        discovery and resolution cannot disagree about which profile a name means.
        Callers that test the System index on its own get this wrong for a user who
        owns a profile sharing a System name.
        """
        index = self._get_user_index(user_id)
        if profile_name in index.get("personal", {}) or profile_name in index.get("borrowed", {}):
            return False
        return profile_name in self._system_index()

    def _resolve_owner_and_pid(self, user_id: int, profile_name: str,
                               is_borrowed: bool = False, strict: bool = False) -> Tuple[int, Optional[str]]:
        """(owner whose tree holds the shard, PID within it).

        Resolving the PID alone is not enough for a System profile: the shard lives
        under the bot owner's id, so a caller's id would build a path that does not
        exist. Owner and PID have to be resolved together, which is what every
        _get_profile / _save_profile now does.

        A borrow short-circuits -- the borrow shard genuinely belongs to the
        borrower, and its config is what names the source.

        A name that maps to nothing comes back as the PID itself, unless `strict`, when
        it comes back None. Soft is for reads: `_borrow_source` answers a vanished
        source with its PID as the name. Writes are strict, as `_get_shard_path`'s are:
        soft, a stale name minted `profiles/<name>/profile.json.gz`, and the next
        rebuild adopted that as a profile nobody made.
        """
        index = self._get_user_index(user_id)
        miss = None if strict else profile_name

        # Both maps are dicts: `_get_user_index` repairs any other shape before handing it out.
        if is_borrowed:
            return user_id, index.get("borrowed", {}).get(profile_name, miss)

        personal = index.get("personal", {})
        if profile_name in personal:
            return user_id, personal[profile_name]

        system = self._system_index()
        if profile_name in system and profile_name not in index.get("borrowed", {}):
            return int(defaultConfig.DISCORD_OWNER_ID), system[profile_name]

        return user_id, miss

    def _resolve_prompt_owner_and_pid(self, user_id: int, profile_name: str,
                                      strict: bool = False) -> Tuple[int, Optional[str]]:
        """(owner whose tree holds the shard, PID) without needing an is_borrowed flag.

        The prompt accessors take a name and no class, so they search personal,
        then borrowed, then System -- a borrow keeps its own copy of the prompts,
        which is why it is consulted before the System fallback. `strict` as
        `_resolve_owner_and_pid`'s.
        """
        index = self._get_user_index(user_id)
        for key in ("personal", "borrowed"):
            mapping = index.get(key, {})
            if isinstance(mapping, dict) and profile_name in mapping:
                return user_id, mapping[profile_name]

        system = self._system_index()
        if profile_name in system:
            return int(defaultConfig.DISCORD_OWNER_ID), system[profile_name]

        return user_id, None if strict else profile_name

    def _get_pid_from_name(self, user_id: int, profile_name: str) -> Optional[str]:
        """The PID this name resolves to for this user, or None if it resolves to none.

        The strict half of the pair. `_get_pid_from_name_any` answers the same
        question but substitutes the name itself on a miss, which is right for the
        display and comparison sites that only need *a* stable token -- and wrong
        anywhere the answer is stored or compared against a real PID, because a
        name-shaped PID is indistinguishable from a real one downstream and turns a
        renamed or deleted profile into silently mismatched data instead of an error.

        Personal, then borrowed, then System -- the same precedence
        `_resolve_effective_profile` applies, so discovery and resolution agree.
        """
        index = self._get_user_index(user_id)
        if profile_name in index.get("personal", {}):
            return index["personal"][profile_name]
        if profile_name in index.get("borrowed", {}):
            return index["borrowed"][profile_name]
        system = self._system_index()
        if profile_name in system:
            return system[profile_name]
        return None

    def _get_pid_from_name_any(self, user_id: int, profile_name: str) -> str:
        """As `_get_pid_from_name`, but falls back to the name itself on a miss.

        Deliberately soft, and kept that way: the turn path compares this against
        `speaker_pid` and a session whose name no longer resolves should degrade
        rather than raise. Use `_get_pid_from_name` wherever a miss must be visible.
        """
        return self._get_pid_from_name(user_id, profile_name) or profile_name

    def _get_name_from_pid(self, user_id: int, target_pid: str,
                           include_borrowed: bool = False) -> Optional[str]:
        """The local name a PID maps to, across the classes the owner can share.

        System profiles are included because _accept_share_request resolves the
        shared profile's current name through here: searching only 'personal' meant
        a shared System profile fell through to the requester's cached name, which
        is stale the moment the bot owner renames it. Borrowed profiles are
        deliberately excluded -- a borrow is not the borrower's to re-share.

        `include_borrowed` is for the one caller that is not re-sharing anything:
        repairing a seated participant's cached name from its PID. A borrow's local
        name is the borrower's own, and a session that seats one has exactly as much
        right to the current spelling as it does for a personal profile.
        """
        index = self._get_user_index(user_id)
        categories = ("personal", "borrowed", "system") if include_borrowed else ("personal", "system")
        for category in categories:
            mapping = index.get(category, {})
            if isinstance(mapping, dict):
                for name, pid in mapping.items():
                    if pid == target_pid: return name
        return None

    def _is_valid_profile_name(self, name: str) -> tuple[bool, str]:
        if not name or not name.strip():
            return False, "Profile name cannot be empty."
        if len(name) > 20:
            return False, "Profile name must be 20 characters or fewer."
        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            return False, "Profile name can only contain letters, numbers, underscores, and hyphens (no spaces)."
        reserved = [
            "clyde", "system", "user", "none", "all", "everyone", "here",
            "discord", "null", "undefined", "true", "false", "root",
            "mimic", "mimicai", "mimica", "bot", "admin", "mod", "help",
            "global", "bulk"
        ]
        if name.lower() in reserved:
            return False, "This name is a reserved system keyword and cannot be used."
        return True, ""

    def _get_profile_id(self, user_id: int, profile_name: str) -> str:
        index = self._get_user_index(user_id)
        is_borrowed = profile_name in index.get("borrowed", [])
        if is_borrowed:
            source = self._borrow_source(user_id, profile_name)
            if source is None:
                return self._get_pid_from_name_any(user_id, profile_name)
            return source[1]
        else:
            return self._get_pid_from_name_any(user_id, profile_name)

    def _get_profile(self, user_id: int, profile_name: str, is_borrowed: bool = False) -> Optional[Dict[str, Any]]:
        if not profile_name:
            return None
        eff_owner, pid = self._resolve_owner_and_pid(user_id, profile_name, is_borrowed)
        if not pid:
            return None
        return self._get_profile_by_pid(eff_owner, pid)

    def _save_profile(self, user_id: int, profile_name: str, data: Dict[str, Any], is_borrowed: bool = False):
        if not profile_name:
            return
        eff_owner, pid = self._resolve_owner_and_pid(user_id, profile_name, is_borrowed, strict=True)
        if not pid:
            print(f"Profile save skipped: '{profile_name}' resolves to no profile for {user_id}.")
            return
        # A resolved owner that is not the caller means a System profile reached
        # here under a member's id. Reads resolve across trees on purpose -- that is
        # what makes System profiles usable by everyone -- but writes must not: the
        # UI already refuses them (ProfileManageView.is_read_only), and every
        # legitimate System write arrives with the bot owner's id because
        # ProfileManageView rewrites it. Before resolution crossed trees this landed
        # in a junk directory and was harmless; now it would land on the real shard.
        if eff_owner != user_id:
            return
        self._save_profile_by_pid(eff_owner, pid, data)

    def _get_profile_by_pid(self, user_id: int, pid: str) -> Optional[Dict[str, Any]]:
        if not pid:
            return None
        path = os.path.join(USERS_DIR, str(user_id), "profiles", pid, "profile.json.gz")
        # Cached: this is the hot read of the whole system -- _get_profile_config lands
        # here several times per turn and had no cache of its own. The cache holds
        # decrypted bytes and re-parses per call, so the dict returned here is still
        # this caller's own to mutate.
        return IOManager.read_json_gzip_cached(path, self.cog.fernet)

    def _save_profile_by_pid(self, user_id: int, pid: str, data: Dict[str, Any]):
        if not pid:
            return
        p_dir = os.path.join(USERS_DIR, str(user_id), "profiles", pid)
        os.makedirs(p_dir, exist_ok=True)
        path = os.path.join(p_dir, "profile.json.gz")
        IOManager.write_json_gzip(data, path, self.cog.fernet)
        # After the shard, never before: the sidecar exists to describe a shard that
        # is already on disk, and a crash between the two must leave the name file
        # trailing the profile rather than announcing one that was never written.
        self._write_name_sidecar(user_id, pid, data.get("name"), self._pid_class(pid))
        self._update_card(user_id, pid, data.get("config") or {})
        self.cog.library._update_listing(user_id, pid, data.get("config") or {})

    # --- The name sidecar ----------------------------------------------------------
    #
    # index.json is derived state: every name -> PID entry in it can be reconstructed
    # from the profiles directory. What made a reconstruction expensive was that the
    # name lived only inside profile.json.gz, so _repair_user_index had to decrypt and
    # zstd-inflate every shard a user owned just to read one string out of each.
    # That cost is the reason repair was something to be avoided rather than simply
    # done, and the reason _index_is_consistent exists at all.
    #
    # The class no longer needs reading either -- it is the PID's first letter, see
    # PID_CLASS_PREFIXES -- so the name is the only irreducible fact, and a few hundred
    # plaintext bytes beside the shard makes rebuilding it cheap enough to stop
    # tiptoeing around.

    @staticmethod
    def _name_sidecar_path(user_id: Any, pid: str) -> str:
        return os.path.join(USERS_DIR, str(user_id), "profiles", pid, PROFILE_NAME_SIDECAR)

    def _write_name_sidecar(self, user_id: int, pid: str, name: Optional[str],
                            category: Optional[str] = None):
        """Record a profile's name and class beside its shard. Best-effort.

        A failure here is not worth failing a save over: the sidecar is a speed-up,
        and the readers fall back to opening the shard whenever it is missing or
        unreadable -- then write it, so a gap heals on the next rebuild.
        """
        if not name:
            return
        payload = {"name": name}
        if category:
            payload["class"] = category
        try:
            IOManager.write_json(payload, self._name_sidecar_path(user_id, pid))
        except Exception as e:
            print(f"Name sidecar: write failed for {user_id}/{pid} ({e})")

    def _describe_shard(self, user_id: int, pid: str,
                        backfill: bool = True) -> Optional[Tuple[str, str]]:
        """(display name, index class) for one PID, without opening the shard if possible.

        The two facts index.json holds about a profile. The sidecar answers both; a
        well-formed PID answers the class on its own. Only a profile with neither --
        one written before the sidecar existed -- costs a decrypt, and `backfill`
        then writes the sidecar so it is the last time that profile costs one.

        Returns None when there is no readable profile at this PID at all, which is
        how a caller tells an orphaned directory from a real one.
        """
        sidecar = IOManager.read_json(self._name_sidecar_path(user_id, pid))
        if isinstance(sidecar, dict) and sidecar.get("name"):
            category = sidecar.get("class") or self._pid_class(pid)
            if category:
                return sidecar["name"], category

        profile_data = self._get_profile_by_pid(user_id, pid)
        if not isinstance(profile_data, dict):
            return None
        name = profile_data.get("name") or pid
        category = self._pid_class(pid) or "personal"
        if backfill:
            self._write_name_sidecar(user_id, pid, name, category)
        return name, category

    # --- The catalogue -------------------------------------------------------------
    #
    # What a list shows for a profile -- display name and rating verdict, and for a borrow
    # where its source is -- kept in index.json, so an autocomplete or a dropdown over a
    # hundred profiles reads one file rather than opening each (~0.07 ms against 8-29 ms
    # for 100). Keyed by PID: a rename never touches it. A dict in the cached index; one
    # Fernet token on disk, so display names and verdicts stay encrypted at rest.
    #
    # Copies, written by _save_profile_by_pid alone. A card that is missing costs one
    # profile read, never a wrong answer, and nothing that lets a profile out reads it:
    # the gates read the profile itself.

    @staticmethod
    def _source_pointer(config: Dict[str, Any]) -> Optional[list]:
        """A borrow's [owner, source PID], or None when it names no usable source."""
        owner = config.get("original_owner_id")
        pid = config.get("original_pid")
        if not owner or not pid or pid == _UNKNOWN_SOURCE_PID:
            return None
        return [int(owner), pid]

    def _card_of(self, pid: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """One profile's catalogue entry, sparse: "d" display name, "r" verdict when not
        Unrated, "src" for a borrow -- whose name and verdict are its source's card."""
        if self._pid_class(pid) == "borrowed":
            src = self._source_pointer(config)
            return {"src": src} if src else {}
        card: Dict[str, Any] = {}
        if config.get("custom_display_name"):
            card["d"] = str(config["custom_display_name"])
        verdict = self.cog.content_rating._verdict_of(config)
        if verdict != CONTENT_RATING_UNRATED:
            card["r"] = verdict
        return card

    def _open_catalogue(self, value: Any) -> Dict[str, Any]:
        """The catalogue as stored -- a token, or a dict with no key configured -- opened.
        One that will not open is empty: the next repair refills it."""
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and self.cog.fernet:
            try:
                opened = json.loads(self.cog.fernet.decrypt(value.encode()))
                if isinstance(opened, dict):
                    return opened
            except Exception:
                pass
        return {}

    def _card(self, owner_id: int, pid: Optional[str]) -> Optional[Dict[str, Any]]:
        """The catalogue entry for a PID in this owner's tree, or None if it has none yet."""
        if not pid:
            return None
        return self._get_user_index(owner_id).get("catalogue", {}).get(pid)

    def _update_card(self, user_id: int, pid: str, config: Dict[str, Any]):
        """Refresh one card, saving the index only when it changed: this runs on every
        profile save, and most change nothing a list shows. Best-effort, as the sidecar
        is -- the shard is already written, and a missing card only costs a read."""
        try:
            card = self._card_of(pid, config)
            index = self._get_user_index(user_id)
            catalogue = index.setdefault("catalogue", {})
            if catalogue.get(pid) != card:
                catalogue[pid] = card
                self._save_user_index(user_id, index)
        except Exception as e:
            print(f"Catalogue: card update failed for {user_id}/{pid} ({e})")

    def catalogue_entry(self, user_id: int, profile_name: str) -> Tuple[Optional[str], str]:
        """(display name or None, verdict) a list shows for this name -- a borrow's are its
        source's. Index reads only, bar a profile the catalogue has not caught up with.

        For lists and autocomplete. Never a gate: a path that lets a profile out reads the
        profile (`settled_capability`).
        """
        eff_owner, eff_name = self._resolve_effective_profile(user_id, profile_name)
        tree_owner, pid = self._resolve_owner_and_pid(eff_owner, eff_name, strict=True)
        card = self._card(tree_owner, pid)
        if card is None:
            card = self._card_of(pid or "", self._get_profile_config(eff_owner, eff_name, False) or {})
        return card.get("d"), card.get("r", CONTENT_RATING_UNRATED)

    #: A minted PID (`_mint_pid`) is a class letter plus 15 uppercase hex digits.
    _PID_BODY = frozenset("0123456789ABCDEF")

    def _mint_pid(self, owner_id: int, letter: str) -> str:
        """A new PID of class `letter`, drawn again while `owner_id` already has a folder by
        that name (`_save_profile_by_pid` would write over that profile) or the Public Library
        lists it (`public_profiles` is keyed on the bare PID, so publishing would replace that
        listing). Otherwise a PID is unique across owners by odds alone (60 bits), and
        `_pid_owners` copes with two."""
        while True:
            pid = letter + secrets.token_hex(8)[:15].upper()
            if (pid not in self.cog.public_profiles
                    and not os.path.exists(os.path.join(USERS_DIR, str(owner_id), "profiles", pid))):
                return pid

    @classmethod
    def _is_pid(cls, pid: str) -> bool:
        """Whether this folder name is a minted PID rather than a pre-PID profile name.

        The check has to be the full shape, not the first letter. Profile names are
        also legal folder names in the pre-PID population, and a user with a profile
        called "Bob" would otherwise be read as a borrow and one called "Xavier" as a
        System profile.
        """
        return (isinstance(pid, str) and len(pid) == 16
                and pid[0] in PID_CLASS_PREFIXES
                and all(c in cls._PID_BODY for c in pid[1:]))

    @classmethod
    def _pid_class(cls, pid: str) -> Optional[str]:
        """Which index.json map owns this PID: "personal", "borrowed" or "system" -- its
        letter. None for a folder that is not a minted PID."""
        return PID_CLASS_PREFIXES[pid[0]] if cls._is_pid(pid) else None

    def _set_child_bot_config(self, user_id: int, profile_name: str, bot_config: Dict[str, Any]):
        p_data = self._get_profile(user_id, profile_name, is_borrowed=False)
        if p_data is not None:
            p_data["child_bot"] = bot_config
            self._save_profile(user_id, profile_name, p_data, is_borrowed=False)

    def _delete_child_bot_config(self, user_id: int, profile_name: str):
        p_data = self._get_profile(user_id, profile_name, is_borrowed=False)
        if p_data is not None and p_data.get("child_bot"):
            p_data["child_bot"] = None
            self._save_profile(user_id, profile_name, p_data, is_borrowed=False)

    def _update_child_bot_presence(self, user_id: int, profile_name: str, presence_update: Dict[str, Any]) -> Dict[str, Any]:
        p_data = self._get_profile(user_id, profile_name, is_borrowed=False)
        if not p_data:
            return {}
        bot_cfg = p_data.get("child_bot") or {}
        current_presence = bot_cfg.get("presence", {})
        current_presence.update(presence_update)
        bot_cfg["presence"] = current_presence
        p_data["child_bot"] = bot_cfg
        self._save_profile(user_id, profile_name, p_data, is_borrowed=False)
        return current_presence

    def _rename_profile(self, user_id: int, old_name: str, new_name: str, is_borrowed: bool = False) -> bool:
        user_index = self._get_user_index(user_id)
        list_key = "borrowed" if is_borrowed else "personal"
        
        if old_name not in user_index.get(list_key, {}):
            return False
            
        pid = user_index[list_key].pop(old_name)
        user_index[list_key][new_name] = pid

        # Shard first, index second, as everywhere else. The shard's name -- and the
        # sidecar beside it -- is what a rebuild believes, so writing the index first
        # meant a crash in between left a rebuild able to undo the rename.
        #
        # Nothing else moves: the PID is the directory name, so memories, training
        # examples and the borrow index all stay put. Keyed by local name, the borrow
        # index used to see this as a borrow lost and a borrow gained, and decrypt the
        # shard to re-derive a source that had not changed.
        p_data = self._get_profile_by_pid(user_id, pid)
        if p_data:
            p_data["name"] = new_name
            self._save_profile_by_pid(user_id, pid, p_data)

        self._save_user_index(user_id, user_index)

        # The name is part of the classified surface, and the cache is keyed by it,
        # so both entries must go -- the stale key would otherwise linger until the
        # LRU evicted it.
        self.cog.content_rating._invalidate_content_rating(user_id, old_name)

        # What holds the name in memory, carried across. The child bot registry is
        # rebuilt only on a reconnect, so until then the bot answered under the old
        # name -- no config, no persona -- and `/session swap` could not find it.
        for p in (p for s in self.cog.multi_profile_channels.values() for p in s.get("profiles", [])):
            if p["owner_id"] == user_id and p["profile_name"] == old_name:
                p["profile_name"] = new_name
        by_name = self.cog.child_bots_by_owner_profile
        for bot_id, bot in self.cog.child_bots.items():
            if bot.get("owner_id") == user_id and bot.get("pid") == pid:
                bot["profile_name"] = new_name
                by_name.pop((user_id, old_name), None)
                by_name[(user_id, new_name)] = bot_id
        old_key, new_key = ("global", user_id, old_name), ("global", user_id, new_name)
        for cache in (self.cog.global_chat_sessions, self.cog.session_last_accessed):
            if old_key in cache:
                cache[new_key] = cache.pop(old_key)
        return True

    async def delete_profiles(self, user_id: int, names: List[str]) -> int:
        """Delete these profiles of one user, a personal one with every borrow of it.
        Returns how many went.

        The one delete, as `_rename_profile` is the one rename, and it forgets what holds
        the name in memory for the same reason. Left behind, a linked child bot kept its
        gateway connection and answered with no persona until a restart, and a new profile
        given the old name wore the deleted one's avatar, rating and Global Chat. A seat in
        a session is `_ensure_session_hydrated`'s to drop, by PID, on its next use.

        The index is edited here on the loop -- `_get_user_index` hands back the dict every
        handler shares -- and saved only after the directories are gone: shard first, index
        second, or a crash in between leaves a shard nothing names, which a rebuild adopts.
        The cascade, the removals and the save run in a thread: every borrower is an
        rmtree and an index write.
        """
        index = self._get_user_index(user_id)
        doomed, sources = [], []
        for name in names:
            category = "borrowed" if name in index.get("borrowed", {}) else "personal"
            if name not in index.get(category, {}):
                continue
            # Before the pop: the borrows' cached ratings are found by resolving them here.
            self.cog.content_rating._invalidate_content_rating(user_id, name)
            pid = index[category].pop(name)
            if category == "personal":
                sources.append(pid)
            doomed.append(pid)
            self.cog.user_appearances.get(str(user_id), {}).pop(name, None)
            key = ("global", user_id, name)
            for cache in (self.cog.global_chat_sessions, self.cog.session_last_accessed,
                          self.cog.dirty_sessions):
                cache.pop(key, None)
        if not doomed:
            return 0

        for bot_id, bot in list(self.cog.child_bots.items()):
            if bot.get("owner_id") == user_id and bot.get("pid") in doomed:
                self.cog.child_bots.pop(bot_id, None)
                self.cog.child_bots_by_owner_profile.pop((user_id, bot.get("profile_name")), None)
                await self.cog.manager_queue.put({"action": "shutdown_bot", "bot_id": bot_id})

        profiles_dir = os.path.join(USERS_DIR, str(user_id), "profiles")

        def _remove():
            for pid in sources:
                self.cog.sharing._cascade_delete_borrowed_profiles(user_id, pid)
            for pid in doomed:
                shutil.rmtree(os.path.join(profiles_dir, pid), ignore_errors=True)
            self._save_user_index(user_id, index)
        await asyncio.to_thread(_remove)
        return len(doomed)

    def _duplicate_profile(self, user_id: int, source_name: str, target_name: str) -> Tuple[bool, str]:
        user_index = self._get_user_index(user_id)
        if len(user_index.get("personal", {})) >= defaultConfig.LIMIT_PROFILES:
            return False, "Limit reached."

        source_pid = self._get_pid_from_name_any(user_id, source_name)
        source_data = self._get_profile_by_pid(user_id, source_pid)
        if not source_data:
            return False, "Source profile data not found."

        new_pid = self._mint_pid(user_id, "A")
        new_data = {
            "name": target_name,
            "config": source_data.get("config", {}).copy(),
            "prompts": source_data.get("prompts", {}).copy(),
            "child_bot": None
        }
        new_data["config"]["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        self._save_profile_by_pid(user_id, new_pid, new_data)

        if not isinstance(user_index.get("personal"), dict):
            legacy_personal = user_index.get("personal", [])
            user_index["personal"] = {}
            if isinstance(legacy_personal, list):
                for p_name in legacy_personal:
                    user_index["personal"][p_name] = p_name

        user_index["personal"][target_name] = new_pid
        self._save_user_index(user_id, user_index)

        self.cog.memory_manager._copy_ltm_shard(str(user_id), source_name, target_name)
        self.cog.memory_manager._copy_training_shard(str(user_id), source_name, target_name)
        return True, f"Duplicated to '{target_name}'."

    #: The keys of index.json a rebuild reconstructs from the profiles directory.
    #: Everything else in the file is carried through untouched -- see
    #: _repair_user_index for why the list is of what is derived rather than of what
    #: is preserved.
    _DERIVED_INDEX_KEYS = ("personal", "borrowed", "system")

    def _repair_user_index(self, user_id: int) -> Dict[str, Any]:
        """Rebuild the name -> PID maps in a user's index.json from their profiles directory.

        Rebuilds *only* `_DERIVED_INDEX_KEYS` and leaves the rest of the file alone.
        This used to be the other way round -- a fresh dict naming the keys worth
        keeping -- so every non-derivable key added later ("defaults", "about", and
        whatever comes next) had to be remembered here or a repair silently deleted a
        user's settings. Preserving by default makes that impossible to get wrong.

        The three maps are cleared before the scan refills them. "system" used to be
        seeded from the old index and only added to, so a System profile whose folder
        was gone stayed named in the map forever -- and since _index_is_consistent
        counts it, that one stale entry failed the check on every pass and bought a
        full repair every hour, for good.

        Cheap now: _describe_shard answers from the plaintext sidecar, so this touches
        an encrypted shard only for profiles written before the sidecar existed, and
        backfills those as it goes.
        """
        user_id_str = str(user_id)
        index = IOManager.read_json(os.path.join(USERS_DIR, user_id_str, "index.json")) or {}
        for key in self._DERIVED_INDEX_KEYS:
            index[key] = {}
        # Kept, and filled for any profile without a card; _save_user_index drops the ones
        # whose profile has gone.
        catalogue = index["catalogue"] = self._open_catalogue(index.get("catalogue"))
        profiles_dir = os.path.join(USERS_DIR, user_id_str, "profiles")

        if os.path.isdir(profiles_dir):
            for pid_folder in os.listdir(profiles_dir):
                p_dir = os.path.join(profiles_dir, pid_folder)
                if not os.path.isdir(p_dir):
                    continue

                # The shard, not the sidecar, is what makes a directory a profile. A
                # directory holding only memories -- see the note on _get_shard_path --
                # is an orphan, and naming it in the index would make it resolvable.
                if not os.path.exists(os.path.join(p_dir, "profile.json.gz")):
                    continue

                described = self._describe_shard(user_id, pid_folder)
                if described is None:
                    continue
                p_name, category = described
                index.setdefault(category, {})[p_name] = pid_folder

                if pid_folder not in catalogue:
                    # Uncached: a sweep has no business in the turn path's shard cache.
                    data = IOManager.read_json_gzip(
                        os.path.join(p_dir, "profile.json.gz"), self.cog.fernet)
                    if isinstance(data, dict):
                        catalogue[pid_folder] = self._card_of(pid_folder, data.get("config") or {})

        # Runs whether or not there was a profiles directory to scan: a user with no
        # profiles still has a key flag, and the branches that used to bracket this
        # had drifted into two copies of the same four lines.
        self._refresh_key_flag(user_id_str, index)
        self._save_user_index(user_id, index)

        return index

    @staticmethod
    def _profiles_dir_stamp(user_id_str: str) -> Optional[int]:
        """mtime_ns of the user's profiles/ directory, or None when there is none.

        A directory's own mtime moves when an entry is created or removed in it, and
        not when a file inside one of its children is rewritten -- which is exactly
        the granularity index.json cares about. Creating or deleting a profile bumps
        it; editing one does not.

        A rename does not bump it either, but a rename writes index.json through
        _rename_profile, and _save_user_index re-stamps. So the stamp only has to
        catch what happens *outside* this process: a restore from backup, a hand-
        edited directory, another instance.
        """
        try:
            return os.stat(os.path.join(USERS_DIR, user_id_str, "profiles")).st_mtime_ns
        except OSError:
            return None

    @staticmethod
    def _key_file_stamp(user_id_str: str) -> Optional[List[int]]:
        """(mtime_ns, size) of the user's key file, or None when there is no file.

        Same stat-stamp trick the LTM vector cache uses: it changes whenever the file
        is rewritten, and reading it costs a stat rather than a decrypt.
        """
        try:
            st = os.stat(os.path.join(USERS_DIR, user_id_str, "keys.json.gz"))
        except OSError:
            return None
        return [st.st_mtime_ns, st.st_size]

    def _refresh_key_flag(self, user_id_str: str, index: Dict[str, Any]):
        """Recompute has_personal_key from the key file, and stamp what it was read from.

        The stamp is what lets _index_is_consistent verify the flag with a stat. Every
        writer of has_personal_key must come through here, or it will leave a flag that
        disagrees with its stamp and buy a full repair on the next hourly pass.
        """
        keys_path = os.path.join(USERS_DIR, user_id_str, "keys.json.gz")
        index["has_personal_key"] = False
        if os.path.exists(keys_path):
            keys_data = IOManager.read_json_gzip(keys_path, self.cog.fernet)
            if keys_data and (keys_data.get("key") or keys_data.get("slots")):
                index["has_personal_key"] = True
        index["key_file_stamp"] = self._key_file_stamp(user_id_str)

    def _index_is_consistent(self, user_id_str: str) -> bool:
        """Cheap pre-check for _repair_all_user_indices.

        A repair opens, decrypts and zstd-decompresses every profile shard the
        user owns. That is the right cost when the index is actually wrong, and pure
        waste when it is not -- and this runs on every boot AND hourly, so the common
        case is "nothing has changed since the last one".

        Everything checked here is a directory listing plus one small file, so the
        check is orders of magnitude cheaper than the repair it avoids. It compares:
          - the index parses and has the expected shape
          - it names exactly as many profiles as there are shards on disk
          - its has_personal_key flag still matches the key file

        Deliberately conservative: anything unexpected returns False and the full
        repair runs, so a false negative only costs the work we do today.
        """
        try:
            index = IOManager.read_json(os.path.join(USERS_DIR, user_id_str, "index.json"))
            if not index or not isinstance(index.get("personal"), dict) or not isinstance(index.get("borrowed"), dict):
                return False

            # The cheapest test first: if profiles/ has not gained or lost an entry
            # since the index was written, no profile was created or deleted, and the
            # set comparison below cannot fail. One stat instead of a listdir.
            #
            # Presence is the test, not truth -- as with key_file_stamp, None is the
            # honest stamp of a user with no profiles directory. An index written
            # before this existed carries no stamp at all, fails once, and is stamped
            # by the repair.
            if "profiles_stamp" not in index:
                return False
            if index["profiles_stamp"] != self._profiles_dir_stamp(user_id_str):
                return False

            profiles_dir = os.path.join(USERS_DIR, user_id_str, "profiles")
            pids_on_disk = set()
            if os.path.isdir(profiles_dir):
                for pid_folder in os.listdir(profiles_dir):
                    if os.path.exists(os.path.join(profiles_dir, pid_folder, "profile.json.gz")):
                        pids_on_disk.add(pid_folder)

            # Sets, not counts. The folder names on disk *are* the index's values, so
            # comparing them costs the same listdir and catches what a count cannot:
            # one profile created and another deleted between passes, a folder renamed
            # by hand, or the same PID named under two entries.
            pids_in_index = set()
            for key in self._DERIVED_INDEX_KEYS:
                mapping = index.get(key)
                if isinstance(mapping, dict):
                    pids_in_index.update(mapping.values())
            if pids_in_index != pids_on_disk:
                return False

            # A profile with no card predates the catalogue (or a restore dropped it): the
            # repair is what fills it, which makes the first boot after a deploy the
            # migration. One small decrypt per user.
            if set(self._open_catalogue(index.get("catalogue"))) != pids_on_disk:
                return False

            # Nothing clears has_personal_key when a key is deleted -- only a repair
            # does -- so it has to be verified here or a stale True would survive.
            #
            # Verified against a stat rather than the file's contents. This ran per
            # user on every boot and then hourly, and opening keys.json.gz means a
            # decrypt, a zstd decompress and a parse -- for a flag that only
            # changes when the file does. The stamp is written beside the flag by
            # _refresh_key_flag; if it still matches the file, the flag was computed
            # from these exact bytes and needs no re-reading.
            #
            # Tested for presence, not truth: None is the stamp of a user with no key
            # file at all, which is the common case and a perfectly consistent one.
            # Treating it as "unstamped" put every keyless user through a full repair
            # every hour.
            #
            # A genuinely absent key is an index written before this existed, so it
            # fails once and the repair stamps it -- the same cost as every hour used
            # to be, paid a single time.
            if "key_file_stamp" not in index:
                return False
            return index["key_file_stamp"] == self._key_file_stamp(user_id_str)
        except Exception:
            return False

    def _repair_all_user_indices(self, force: bool = False):
        """Scans the USERS_DIR for user folders and runs self-repair on each user's index.json.

        Skips users whose index already checks out (see _index_is_consistent) unless
        force=True, because this runs on every boot and then hourly thereafter.
        """
        if not os.path.isdir(USERS_DIR):
            return
        for user_id_str in os.listdir(USERS_DIR):
            if user_id_str.isdigit():
                try:
                    if not force and self._index_is_consistent(user_id_str):
                        continue
                    user_id = int(user_id_str)
                    self._repair_user_index(user_id)
                except Exception as e:
                    print(f"Error repairing index for user {user_id_str}: {e}")

        # A forced repair is the "something is wrong, rebuild it" path, and the borrow
        # reverse index is derived state like any other. The unforced hourly pass
        # leaves it alone: it is maintained incrementally by _save_user_index, and a
        # full scan is exactly the cost it exists to avoid.
        if force:
            self.cog.borrows._rebuild_borrow_index()

    def _get_user_index(self, user_id: int) -> Dict[str, Any]:
        user_id_str = str(user_id)
        if user_id_str in self.cog.user_indices: return self.cog.user_indices[user_id_str]

        path = os.path.join(USERS_DIR, user_id_str, "index.json")
        index = IOManager.read_json(path)

        if index is None and not os.path.isdir(os.path.join(USERS_DIR, user_id_str)):
            # Someone the bot holds nothing for: a keystroke of autocomplete, a message in
            # a session channel, a look at /start. There is nothing to repair, and the
            # repair's save would mint a directory for every passer-by. The first thing
            # they actually keep -- a profile, a key, a setting -- saves the index.
            index = {key: {} for key in self._DERIVED_INDEX_KEYS}
        elif not index or not isinstance(index.get("personal"), dict) or not isinstance(index.get("borrowed"), dict):
            index = self._repair_user_index(user_id)

        index["catalogue"] = self._open_catalogue(index.get("catalogue"))
        self.cog.user_indices[user_id_str] = index
        # Cache miss only, and only for an index good enough to skip the repair above
        # -- a repair already saves, which reconciles. This is the path that catches an
        # index.json written by something other than _save_user_index: a restore from
        # backup, or the other instance while this one held no lock. Costs one set
        # comparison per LRU miss; a borrow the index has not seen is the direction
        # that would wrongly report a shared profile as private.
        try:
            self.cog.borrows._reconcile_borrow_index(user_id, index)
        except Exception as e:
            print(f"Borrow index: reconcile failed for user {user_id_str} ({e})")
        return index

    def _save_user_index(self, user_id: int, data: Dict[str, Any]):
        user_id_str = str(user_id)
        path = os.path.join(USERS_DIR, user_id_str, "index.json")
        # Stamped here because this is the only writer, and stamped *before* the write
        # so the value describes the directory this index was built against. Every
        # caller that creates or deletes a profile now writes the shard first (see the
        # ordering note on _get_or_create_user_profile), so by the time we get here
        # profiles/ already holds what the index is about to claim.
        data["profiles_stamp"] = self._profiles_dir_stamp(user_id_str)
        # Pruned to the profiles on disk: every delete ends here, so no card outlives its
        # profile. Sealed for disk only -- the cached index keeps the dict.
        profiles_dir = os.path.join(USERS_DIR, user_id_str, "profiles")
        on_disk = set(os.listdir(profiles_dir)) if os.path.isdir(profiles_dir) else set()
        catalogue = data["catalogue"] = {pid: card for pid, card
                                         in self._open_catalogue(data.get("catalogue")).items()
                                         if pid in on_disk}
        sealed = (self.cog.fernet.encrypt(json.dumps(catalogue)).decode()
                  if self.cog.fernet else catalogue)
        IOManager.write_json({**data, "catalogue": sealed}, path)
        self.cog.user_indices[user_id_str] = data
        # The appearance cache is keyed by name, and every rename, delete and creation
        # ends here. Kept past one, a name handed to a different profile -- a rename's
        # old name reused, a deleted profile's name taken -- wore the previous holder's
        # face until the LRU let it go. Rebuilt from config on the next read.
        self.cog.user_appearances.pop(user_id_str, None)
        # Every path that adds or removes a borrow ends here, which is why the
        # reverse index is maintained here rather than at each of them.
        try:
            self.cog.borrows._reconcile_borrow_index(user_id, data)
        except Exception as e:
            print(f"Borrow index: reconcile failed for user {user_id_str} ({e})")

    def _get_profile_config(self, user_id: int, profile_name: str, is_borrowed: bool = False) -> Optional[Dict[str, Any]]:
        p_data = self._get_profile(user_id, profile_name, is_borrowed)
        if p_data is not None:
            return p_data.get("config", {})
        return None
    def _save_profile_config(self, user_id: int, profile_name: str, data: Dict[str, Any], is_borrowed: bool = False):
        p_data = self._get_profile(user_id, profile_name, is_borrowed)
        if p_data is None:
            p_data = {
                "name": profile_name,
                "config": data,
                "prompts": {},
                "child_bot": None
            }
        else:
            p_data["name"] = profile_name
            p_data["config"] = data
        self._save_profile(user_id, profile_name, p_data, is_borrowed)

    def _borrow_source(self, user_id: int, profile_name: str) -> Optional[Tuple[int, str, str]]:
        """(owner, PID, current name) of the profile this borrow was made from, or None
        when the borrow's config names no usable source --
        `_validate_and_clean_borrowed_profiles` reaps those.

        Found through `original_pid`. `original_profile_name` is a snapshot taken when the
        borrow was made: once the author renamed the profile it named nothing, and once
        the old name was reused it named a different character, which the borrow then
        spoke as. The name comes fresh from the owner's index; a source no longer in it
        is answered with its PID as the name, which no profile name can equal (names are
        lower case), so it resolves to that source's own folder or to nothing.

        The pointer is read off the catalogue, so resolving a borrow opens nothing; the
        borrow's own config only for a card not yet written.
        """
        card = self._card(user_id, self._get_user_index(user_id).get("borrowed", {}).get(profile_name))
        if card is not None:
            src = card.get("src")
        else:
            src = self._source_pointer(self._get_profile_config(user_id, profile_name, True) or {})
        # [owner, pid]; a card written before the snapshot went is [owner, pid|None, name].
        if not src or not src[1]:
            return None
        owner, pid = src[0], src[1]
        return owner, pid, self._get_name_from_pid(owner, pid) or pid

    def signature_token(self, guild_id: int, owner_id: int, profile_name: str) -> str:
        """The message signature for `owner_id` running `profile_name` in this server, or
        "" when there is nothing to sign with (no key, or a pre-PID profile).

        Who ran it, and what: a borrow names its source's PID, which is what would
        otherwise be lost with the borrow; the borrower is the user ID beside it.
        """
        root = getattr(self.cog.fernet, "signing", None)
        if root is None:
            return ""
        pid = self._get_pid_from_name(owner_id, profile_name)
        if pid and pid[0] in "BC":
            source = self._borrow_source(owner_id, profile_name)
            if source and source[1]:
                pid = source[1]
        if not self._is_pid(pid):
            return ""
        return signatures.sign(root, guild_id, owner_id, pid)

    def signed(self, channel: Any, signer: Optional[Tuple[int, str]], text: Any) -> Any:
        """`text` as it goes on the wire as `signer`, an (owner ID, profile name) pair: any
        token in it stripped and that profile's appended. None for a placeholder or status
        line, which says nothing of the character's and is replaced once the reply lands.
        """
        if not isinstance(text, str):
            return text
        if signer is None or signer[0] is None or not signer[1]:
            return signatures.strip(text)
        guild = getattr(channel, "guild", None)
        # Its pings too, here and never in the log (`ping_people`); left as typed when a
        # mention's extra characters would not fit the room every part was cut to.
        session = self.cog.multi_profile_channels.get(getattr(channel, "id", None)) or {}
        text = signatures.strip(text)
        pinged = ping_people(text, session.get("unified_log"), guild)
        if len(pinged) <= max(len(text), PROFILE_MESSAGE_MAX_LENGTH):
            text = pinged
        return signatures.attach(text, self.signature_token(
            guild.id if guild else 0, int(signer[0]), signer[1]))

    def describe_signature(self, user_id: int, pid: str) -> str:
        """What a decoded signature's PID names now, for the operator: the runner's own
        profile, a System profile, the source of one of their borrows, or nothing."""
        index = self._get_user_index(user_id)
        for category in ("personal", "borrowed"):
            for name, own_pid in (index.get(category) or {}).items():
                if own_pid == pid:
                    return f"**{name}**, their own ({category})."
        for name, own_pid in self._system_index().items():
            if own_pid == pid:
                return f"**{name}**, a System profile."
        for name in index.get("borrowed") or {}:
            source = self._borrow_source(user_id, name)
            if source and source[1] == pid:
                return f"**{source[2]}** by <@{source[0]}> ({source[0]}), borrowed as **{name}**."
        # Not theirs and not among their borrows now: a borrow they have since removed,
        # whose source is still wherever its folder is.
        held = [f"**{self._get_name_from_pid(owner, pid) or 'an unindexed profile'}** by <@{owner}> ({owner})"
                for owner in self._pid_owners(pid) if owner != int(user_id)]
        if held:
            return f"{' or '.join(held)}, which they borrowed and have since removed."
        if not os.path.isdir(os.path.join(USERS_DIR, str(user_id))):
            return "Nothing: the profile is gone, and so is all of that account's data here."
        return "Nothing now: the profile has been deleted since."

    @staticmethod
    def _pid_owners(pid: str) -> List[int]:
        """Every account holding a profile folder named `pid`.

        A signature names a borrow by its source's PID alone, with no room for the
        source's owner. `_mint_pid` keeps a PID unique per owner only, so this can
        answer with two.
        """
        # ponytail: one isdir per account and no shard opened, for an operator's trace.
        # An index, if this is ever asked per message.
        try:
            accounts = os.listdir(USERS_DIR)
        except OSError:
            return []
        return [int(u) for u in accounts
                if u.isdigit() and os.path.isdir(os.path.join(USERS_DIR, u, "profiles", pid))]

    def _resolve_effective_profile(self, user_id: int, profile_name: str) -> Tuple[int, str]:
        index = self._get_user_index(user_id)

        # Own profiles first, System as the fallback -- see the note above
        # _get_pid_from_name. The System branch used to run before this one, so a
        # user's personal profile sharing a name with a System profile resolved to
        # the System profile and never ran.
        if profile_name in index.get("borrowed", []):
            source = self._borrow_source(user_id, profile_name)
            if source is None:
                return user_id, profile_name
            return source[0], source[2]
        if profile_name in index.get("personal", []):
            return user_id, profile_name

        if profile_name in self._system_index():
            return int(defaultConfig.DISCORD_OWNER_ID), profile_name

        return user_id, profile_name

    def author_config(self, owner_id: int, profile_name: str) -> Dict[str, Any]:
        """The config of the profile whose prompts this name speaks with.

        A borrow's own config is its borrower's. A setting that only means something
        beside the prompt -- Line Spacing shapes the format a prompt asks for -- is read
        where the prompt is: the source, which is what `_resolve_effective_profile` finds.
        """
        return self._get_profile_config(*self._resolve_effective_profile(owner_id, profile_name), False) or {}

    @staticmethod
    def _appearance_entry(config: Dict[str, Any]) -> Dict[str, Any]:
        """What `cog.user_appearances` holds for one profile. Every writer builds it here, so
        an entry written by one edit never drops what another edit set."""
        return {"custom_display_name": config.get("custom_display_name"),
                "custom_avatar_url": config.get("custom_avatar_url"),
                "mood_avatars": config.get("mood_avatars") or {},
                "mood_avatar_threshold": config.get("mood_avatar_threshold")}

    def _get_user_appearance(self, owner_id: int, profile_name: str) -> Dict[str, Optional[str]]:
        """The name and avatar a profile shows, with the avatar ready for Discord to draw.

        The cache keeps the avatar as typed, which is what Appearance shows back. A
        Discord attachment is handed out without its signature, which expires a day
        after issue -- see `cogs/utils/discord_cdn.py`. Anything that downloads the
        avatar itself reads the config and signs it.

        The avatar is the mood avatar `pick_mood_avatar` chooses where one applies, read off
        the neuro state of the profile named here -- so a caller drawing a borrow's avatar
        passes the borrow, not its source, whose appearance this resolves either way.
        """
        eff_owner_id, eff_name = self._resolve_effective_profile(owner_id, profile_name)
        owner_id_str = str(eff_owner_id)
        if owner_id_str in self.cog.user_appearances and eff_name in self.cog.user_appearances[owner_id_str]:
            data = self.cog.user_appearances[owner_id_str][eff_name]
        else:
            config = self._get_profile_config(eff_owner_id, eff_name, False) or {}
            data = self._appearance_entry(config)
            self.cog.user_appearances.setdefault(owner_id_str, {})[eff_name] = data

        avatar = data.get("custom_avatar_url")
        if data.get("mood_avatars"):
            # Read only for a profile that has mood avatars: everyone else pays nothing.
            is_borrowed = profile_name in self._get_user_index(owner_id).get("borrowed", [])
            speaker = self._get_profile_config(owner_id, profile_name, is_borrowed)
            moods = {axis: url for axis, url in data["mood_avatars"].items() if is_web_url(url)}
            # ponytail: in memory, so a restart forgets the mood showing and may switch once.
            key = (str(owner_id), profile_name)
            shown = self.cog.mood_avatars_shown[key] = pick_mood_avatar(
                moods, data.get("mood_avatar_threshold"), speaker, self.cog.mood_avatars_shown.get(key))
            avatar = moods.get(shown) or avatar
        # A link saved before the modals checked one would fail every embed and message it rides.
        drawable = unsigned_attachment_url(avatar) if is_web_url(avatar) else None
        return data if drawable == data.get("custom_avatar_url") else {**data, "custom_avatar_url": drawable}

    def _get_profile_prompts(self, user_id: int, profile_name: str) -> Optional[Dict[str, Any]]:
        eff_owner, pid = self._resolve_prompt_owner_and_pid(user_id, profile_name)
        if not pid: return None
        p_data = self._get_profile_by_pid(eff_owner, pid)
        if p_data is not None:
            return p_data.get("prompts", {})
        return None

    def effective_prompts(self, user_id: int, profile_name: str) -> Dict[str, Any]:
        """The prompts this name speaks with -- its lorebook, Enhance Awareness, style guide.
        A borrow's are its source's, as its persona is: a borrow's own prompts are written
        empty."""
        eff_owner, eff_name = self._resolve_effective_profile(user_id, profile_name)
        return self._get_profile_prompts(eff_owner, eff_name) or {}

    def _save_profile_prompts(self, user_id: int, profile_name: str, data: Dict[str, Any]):
        """Persists prompts and invalidates any cached content rating.

        Every persona, instruction and image-prompt write funnels through here.

        Deliberately does NOT classify. This hook used to fire the classifier on
        every persona and instruction save and was the dominant source of classifier
        calls -- a profile could be judged dozens of times over its life without its
        owner ever asking for a verdict. The edit now only invalidates: the caller
        decides what to do about the rating going stale, interactively where there
        is a user to ask, and via resolve_stale_rating everywhere else.
        """
        eff_owner, pid = self._resolve_prompt_owner_and_pid(user_id, profile_name, strict=True)
        if not pid:
            print(f"Prompt save skipped: '{profile_name}' resolves to no profile for {user_id}.")
            return
        # See the note in _save_profile: reads cross into the bot owner's tree for a
        # System profile, writes do not.
        if eff_owner != user_id: return
        p_data = self._get_profile_by_pid(eff_owner, pid)
        if p_data is None:
            p_data = {
                "name": profile_name,
                "config": {},
                "prompts": data,
                "child_bot": None
            }
        else:
            p_data["prompts"] = data
        self._save_profile_by_pid(eff_owner, pid, p_data)
        self.cog.content_rating._invalidate_content_rating(user_id, profile_name)

    def _get_or_create_user_profile(self, user_id: int, profile_name: str) -> Optional[Dict[str, Any]]:
        profile_name = profile_name.lower().strip()

        index = self._get_user_index(user_id)

        if profile_name not in index.get("personal", []):
            if len(index.get("personal", [])) >= defaultConfig.LIMIT_PROFILES:
                return None

            pid = self._mint_pid(user_id, "A")
            index["personal"][profile_name] = pid

            config = {
                "grounding_enabled": False, "stm_length": defaultConfig.CHATBOT_MEMORY_LENGTH,
                "temperature": defaultConfig.GEMINI_TEMPERATURE, "top_p": defaultConfig.GEMINI_TOP_P,
                "top_k": defaultConfig.GEMINI_TOP_K, "training_context_size": defaultConfig.TRAINING_CONTEXT_SIZE,
                "training_relevance_threshold": defaultConfig.TRAINING_RELEVANCE_THRESHOLD,
                "ltm_context_size": 3, "ltm_relevance_threshold": 0.75, "ltm_creation_interval": 10,
                "primary_model": PRIMARY_MODEL_NAME, "fallback_model": FALLBACK_MODEL_NAME,
                "timezone": "UTC",
                "realistic_typing_enabled": False, "ltm_creation_enabled": False,
                # Written explicitly, and False, where absence means on: recall has been
                # unconditional for the life of the setting, so every profile already on
                # disk has no key and must keep recalling. See ltm_auto_recall_enabled.
                "ltm_recall_enabled": False,
                "ltm_recall_tool_enabled": False,
                "image_generation_enabled": False,
                # Empty means "send no such field" -- see
                # MediaService.resolve_image_output_params. Not defaulted to a ratio,
                # because every image model already has one and ours would only
                # override it on profiles nobody configured.
                "image_aspect_ratio": "", "image_size": "", "image_thinking_level": "",
                "image_quality": "",
                # Same "empty means send nothing" rule: no search tool on an image
                # request, and no sampling overrides, unless the profile asks.
                "image_grounding_mode": "",
                "image_temperature": "", "image_top_p": "", "image_top_k": "",
                # What happens to an attachment neither of this profile's models can
                # read. The floor, and free: see UNREADABLE_MEDIA_MODES.
                "unreadable_media_mode": UNREADABLE_MEDIA_DEFAULT,
                # Medium, not Model default: Gemini 3 reads that as HIGH, twice the
                # tokens an image for detail a character rarely needs. Seeded, so a
                # profile made before keeps what it had.
                "media_input_resolution": "MEDIA_RESOLUTION_MEDIUM",
                "url_fetching_enabled": False, "response_mode": "regular", "thinking_summary_visible": "off",
                "thinking_level": "low", "thinking_budget": -1,
                "error_response": "An error has occurred.", "speech_tts_enabled": False, "speech_voice": DEFAULT_SPEECH_VOICE,
                "speech_temperature": NEW_PROFILE_SPEECH_TEMPERATURE,
                "neuro_engine_enabled": False, "neuro_state": {"dopamine": 50, "cortisol": 20, "oxytocin": 50, "adrenaline": 20},
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }

            # The provider chosen in `/start` ships its models. `none` ships none: every
            # model slot is NO_FALLBACK -- "None" on the dashboard, refused by
            # `_instantiate_model` -- until the user's own defaults name one.
            preferred = self.cog.user_settings.provider_preference(user_id)
            if preferred in MODEL_PROVIDERS:
                # Routed onto the chosen provider: only the slots it moves are written, so a
                # Google profile keeps absent slots absent and following the shipped value.
                platform = platform_model_defaults()
                config.update({k: v for k, v in model_slot_defaults(preferred).items()
                               if v != platform.get(k)})
            else:
                config.update({k: NO_FALLBACK for pair in MODEL_SLOT_PAIRS.items() for k in pair})
            # Written, since absent runs the shipped chain, and before the owner's own
            # defaults, which may name one.
            config.update(dict.fromkeys(_UNSHIPPED_MODEL_KEYS, NO_FALLBACK))
            # The template above is what the bot ships; anything the user has set as
            # their own default wins over it. New profiles take every defaultable key,
            # including the ones a borrow may not have -- there is no author here whose
            # tuning a standing preference could be overwriting.
            apply_defaults(config, self.cog.user_settings._get_user_defaults(user_id), borrowed=False)

            # No "ltm_summarization_instructions": a seeded copy of the default freezes
            # this profile on whichever wording shipped today, so /mod's override and
            # every later improvement would govern nothing but profiles made after it.
            # Absent resolves live -- see resolve_ltm_summarization_instructions.
            prompts = {
                "persona": {}, "ai_instructions": ["", "", "", ""], "image_generation_prompt": None,
            }

            unified_profile = {
                "name": profile_name,
                "config": config,
                "prompts": prompts,
                "child_bot": None
            }
            # Shard first, index second. See the ordering rule on _save_user_index:
            # a crash between the two must leave a profile the index has not named
            # yet -- which a rebuild adopts -- rather than a name pointing at a
            # directory that was never created, which a rebuild would delete.
            self._save_profile_by_pid(user_id, pid, unified_profile)
            self._save_user_index(user_id, index)
            return {"config": config, "prompts": prompts}

        return {"config": self._get_profile_config(user_id, profile_name), "prompts": self._get_profile_prompts(user_id, profile_name)}

    def _get_or_create_system_profile(self, profile_name: str) -> Optional[Dict[str, Any]]:
        user_id = int(defaultConfig.DISCORD_OWNER_ID)
        profile_name = profile_name.lower().strip()

        index = self._get_user_index(user_id)
        if "system" not in index:
            index["system"] = {}

        if profile_name not in index["system"]:
            pid = self._mint_pid(user_id, "X")
            index["system"][profile_name] = pid

            config = {
                "grounding_enabled": False, "stm_length": defaultConfig.CHATBOT_MEMORY_LENGTH,
                "temperature": 0.2, "top_p": 0.9,
                "top_k": 40, "training_context_size": 0,
                "training_relevance_threshold": 0.0,
                "ltm_context_size": 0, "ltm_relevance_threshold": 1.0, "ltm_creation_interval": 100,
                "primary_model": SYSTEM_PROFILE_MODEL, "fallback_model": SYSTEM_PROFILE_FALLBACK_MODEL,
                "timezone": "UTC", "generation_metadata_enabled": False,
                "realistic_typing_enabled": False, "ltm_creation_enabled": False,
                # Written explicitly, and False, where absence means on: recall has been
                # unconditional for the life of the setting, so every profile already on
                # disk has no key and must keep recalling. See ltm_auto_recall_enabled.
                "ltm_recall_enabled": False,
                "ltm_recall_tool_enabled": False,
                "image_generation_enabled": False, "image_generation_model": DEFAULT_IMAGE_MODEL,
                # Empty means "send no such field" -- see
                # MediaService.resolve_image_output_params. Not defaulted to a ratio,
                # because every image model already has one and ours would only
                # override it on profiles nobody configured.
                "image_aspect_ratio": "", "image_size": "", "image_thinking_level": "",
                "image_quality": "",
                # Same "empty means send nothing" rule: no search tool on an image
                # request, and no sampling overrides, unless the profile asks.
                "image_grounding_mode": "",
                "image_temperature": "", "image_top_p": "", "image_top_k": "",
                # What happens to an attachment neither of this profile's models can
                # read. The floor, and free: see UNREADABLE_MEDIA_MODES.
                "unreadable_media_mode": UNREADABLE_MEDIA_DEFAULT,
                # Medium, not Model default: Gemini 3 reads that as HIGH, twice the
                # tokens an image for detail a character rarely needs. Seeded, so a
                # profile made before keeps what it had.
                "media_input_resolution": "MEDIA_RESOLUTION_MEDIUM",
                "url_fetching_enabled": False, "response_mode": "regular", "thinking_summary_visible": "off",
                "thinking_level": "low", "thinking_budget": -1,
                "error_response": "An error has occurred.", "speech_tts_enabled": False, "speech_voice": DEFAULT_SPEECH_VOICE,
                "speech_model": DEFAULT_SPEECH_MODEL, "speech_temperature": NEW_PROFILE_SPEECH_TEMPERATURE,
                "neuro_engine_enabled": False, "neuro_state": {"dopamine": 50, "cortisol": 20, "oxytocin": 50, "adrenaline": 20},
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "help_mode_enabled": False
            }

            prompts = {
                "persona": {
                    "backstory": [self.cog.storage_manager._encrypt_data("You are MimicGuide, the official technical assistant for the MimicAI Discord Bot.")],
                    "personality_traits": [self.cog.storage_manager._encrypt_data("Helpful, precise, and highly technical.")],
                    "likes": [], "dislikes": [], "appearance": []
                },
                "ai_instructions": [self.cog.storage_manager._encrypt_data("Answer questions concisely using the provided documentation. If you do not know the answer, state that you do not know."), "", "", ""],
                "image_generation_prompt": None,
                # Unseeded, as above.
            }

            unified_profile = {
                "name": profile_name,
                "config": config,
                "prompts": prompts,
                "child_bot": None
            }
            # Shard first, index second -- see the ordering rule on _save_user_index.
            self._save_profile_by_pid(user_id, pid, unified_profile)
            self._save_user_index(user_id, index)

        return {"config": self._get_profile_config(user_id, profile_name), "prompts": self._get_profile_prompts(user_id, profile_name)}

    @tasks.loop(hours=1.0)
    async def hourly_self_repair_task(self):
        if self.cog.has_lock:
            await asyncio.to_thread(self._repair_all_user_indices)

    def _default_ltm_summarization_instructions(self) -> str:
        """The LTM summariser prompt a profile gets when it has none of its own.

        /mod's "LTM Summarization" entry writes global_prompts, which nothing read
        before -- summarisation resolves the prompt per-profile. This makes the
        global the source of that per-profile default, so it governs every profile
        that has not been given text of its own.
        """
        return self.cog.global_prompts.get(
            "LTM_SUMMARIZATION_INSTRUCTIONS", DEFAULT_LTM_SUMMARIZATION_INSTRUCTIONS)

    def resolve_ltm_summarization_instructions(self, owner_id: Optional[int],
                                               profile_name: Optional[str]) -> str:
        """Profile prompt, else /mod's instance-wide override, else the shipped default.

        Resolved, not seeded, as a shipped default prompt always is -- and this one has
        history to undo. Profiles used to be
        created with the default *copied* into their prompts, so a better default would
        have reached new profiles only, and every character anyone already had would
        have stayed on the text that shipped the day it was made. Nothing seeds it any
        more, and a stored value that is byte-for-byte one of those seeded copies is
        treated as the absence it always was. Text an owner actually wrote is untouched,
        because it cannot match.
        """
        default = self._default_ltm_summarization_instructions()
        if not owner_id or not profile_name:
            return default

        source_owner_id, source_profile_name = self._resolve_effective_profile(owner_id, profile_name)
        prompts = self._get_profile_prompts(source_owner_id, source_profile_name) or {}
        encrypted = prompts.get("ltm_summarization_instructions")
        if not encrypted:
            return default
        # A stored-but-blank value must not yield a blank system instruction.
        stored = self.cog.storage_manager._decrypt_data(encrypted).strip()
        if not stored or is_shipped_ltm_prompt(stored, default):
            return default
        return stored

    def _default_critic_instructions(self) -> str:
        """The Anti-Repetition Critic prompt a session gets when it has none of its own: /mod's
        ANTI_REPETITION, else the shipped one. Not seeded, so an edit in /mod reaches every
        session that never wrote its own."""
        return self.cog.global_prompts.get("ANTI_REPETITION", DEFAULT_ANTI_REPETITION_PROMPT)


    def _invalidate_channel_model_cache(self, key: Tuple[int, int]):
        if key in self.cog.channel_models: del self.cog.channel_models[key]
        self.cog.channel_model_last_profile_key.pop(key, None)

    async def update_profile_advanced_params(self, user_id: int, profile_name: str, params: Dict[str, Any], channel_id_context: int, is_borrowed: bool) -> bool:
        if not self.cog.has_lock: return False

        def _sync_update():
            target = self._get_profile_config(user_id, profile_name, is_borrowed)
            if not target: return None

            for k, v in params.items():
                if v is None:
                    if k in target: del target[k] # Reset to default (remove override)
                else:
                    target[k] = v

            self._save_profile_config(user_id, profile_name, target, is_borrowed)
            return self.cog.session_manager._get_active_user_profile_name_for_channel(user_id, channel_id_context) == profile_name

        is_active = await asyncio.to_thread(_sync_update)
        if is_active is None: return False

        if is_active:
            self._invalidate_channel_model_cache((channel_id_context, user_id))

        return True
    

    def _may_edit_prompts(self, user_id: int, profile_name: str) -> bool:
        """Whether this user may write this profile's persona and instructions.

        Their own personal profile, or -- for the bot owner -- a System profile,
        whose shard sits in their tree but whose name is in "system" and never in
        "personal": the bare personal test that used to gate the two updaters
        rejected every persona and instruction edit on a System profile, including
        the bot owner's, the only person the dashboard offers the form to.

        Anyone else is refused here rather than by _save_profile_prompts, which
        drops a cross-tree write silently and would leave the caller reporting a
        success that never happened. A borrow is excluded too -- its prompts are
        the source's, which is why the persona tab is `_own`.
        """
        index = self._get_user_index(user_id)
        if profile_name in index.get("personal", []):
            return True
        return (user_id == int(defaultConfig.DISCORD_OWNER_ID)
                and self._is_system_name(user_id, profile_name))

    async def update_user_profile_persona(self, user_id: int, profile_name: str, persona_data: Dict[str, List[str]], channel_id_context: int) -> bool:
        if not self.cog.has_lock: return False

        def _sync_update():
            if not self._may_edit_prompts(user_id, profile_name): return None

            prompts = self._get_profile_prompts(user_id, profile_name) or {}
            prompts["persona"] = persona_data
            self._save_profile_prompts(user_id, profile_name, prompts)

            return self.cog.session_manager._get_active_user_profile_name_for_channel(user_id, channel_id_context) == profile_name

        is_active = await asyncio.to_thread(_sync_update)
        if is_active is None: return False

        if is_active:
            self._invalidate_channel_model_cache((channel_id_context, user_id))
        return True

    async def update_user_profile_ai_instructions(self, user_id: int, profile_name: str, instructions: List[str], channel_id_context: int, ticks: Optional[Dict[str, bool]] = None) -> bool:
        """`ticks` is `{key: bool}` for the Instructions boxes (`INSTRUCTION_TICKS`)."""
        if not self.cog.has_lock: return False

        def _sync_update():
            if not self._may_edit_prompts(user_id, profile_name): return None

            prompts = self._get_profile_prompts(user_id, profile_name) or {}
            prompts["ai_instructions"] = instructions
            prompts.update(ticks or {})
            self._save_profile_prompts(user_id, profile_name, prompts)

            return self.cog.session_manager._get_active_user_profile_name_for_channel(user_id, channel_id_context) == profile_name

        is_active = await asyncio.to_thread(_sync_update)
        if is_active is None: return False

        if is_active:
            self._invalidate_channel_model_cache((channel_id_context, user_id))
        return True
        

    async def update_profile_generation_params(self, user_id: int, profile_name: str, params: Dict[str, Any], channel_id_context: int, is_borrowed: bool) -> bool:
        if not self.cog.has_lock: return False

        def _sync_update():
            profile = self._get_profile_config(user_id, profile_name, is_borrowed)
            if not profile: return None

            if "temperature" in params: profile["temperature"] = params["temperature"]
            if "top_p" in params: profile["top_p"] = params["top_p"]
            if "top_k" in params: profile["top_k"] = params["top_k"]
            if "stm_length" in params: profile["stm_length"] = params["stm_length"]

            self._save_profile_config(user_id, profile_name, profile, is_borrowed)
            return self.cog.session_manager._get_active_user_profile_name_for_channel(user_id, channel_id_context) == profile_name

        is_active = await asyncio.to_thread(_sync_update)
        if is_active is None: return False

        if is_active:
            self._invalidate_channel_model_cache((channel_id_context, user_id))
        return True

    async def update_profile_training_params(self, user_id: int, profile_name: str, params: Dict[str, Any]) -> bool:
        if not self.cog.has_lock: return False

        def _sync_update():
            profile = self._get_profile_config(user_id, profile_name, False)
            if not profile: return False

            if "training_context_size" in params: profile["training_context_size"] = params["training_context_size"]
            if "training_relevance_threshold" in params: profile["training_relevance_threshold"] = params["training_relevance_threshold"]
            self._save_profile_config(user_id, profile_name, profile, False)
            return True

        return await asyncio.to_thread(_sync_update)


    async def update_profile_ltm_params(self, user_id: int, profile_name: str, params: Dict[str, Any]) -> bool:
        if not self.cog.has_lock: return False

        def _sync_update():
            index = self._get_user_index(user_id)
            is_borrowed = profile_name in index.get("borrowed", [])
            profile = self._get_profile_config(user_id, profile_name, is_borrowed)
            if not profile: return False

            if "ltm_context_size" in params: profile["ltm_context_size"] = params["ltm_context_size"]
            if "ltm_relevance_threshold" in params: profile["ltm_relevance_threshold"] = params["ltm_relevance_threshold"]
            if "ltm_creation_interval" in params: profile["ltm_creation_interval"] = params["ltm_creation_interval"]

            self._save_profile_config(user_id, profile_name, profile, is_borrowed)
            return True

        return await asyncio.to_thread(_sync_update)


    async def update_profile_models(self, user_id: int, profile_name: str, primary_model: Optional[str], fallback_model: Optional[str], is_borrowed: bool, channel_id_context: int, show_fallback_indicator: Optional[bool] = None) -> bool:
        if not self.cog.has_lock: return False

        def _sync_update():
            profile = self._get_profile_config(user_id, profile_name, is_borrowed)
            if not profile: return False

            if primary_model: profile["primary_model"] = primary_model
            if fallback_model: profile["fallback_model"] = fallback_model
            if show_fallback_indicator is not None: profile["show_fallback_indicator"] = show_fallback_indicator

            self._save_profile_config(user_id, profile_name, profile, is_borrowed)
            return True

        if not await asyncio.to_thread(_sync_update):
            return False

        keys_to_delete = []
        for key in list(self.cog.channel_models.keys()):
            key_user_id = None
            if isinstance(key, tuple) and len(key) == 2:
                key_user_id = key[1]
            
            if key_user_id == user_id:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            self._invalidate_channel_model_cache(key)

        return True

    async def _validate_active_profile(self, user_id: int, channel: discord.abc.Messageable) -> bool:
        index = self._get_user_index(user_id)
        active_profile_name = self.cog.session_manager._get_active_user_profile_name_for_channel(user_id, channel.id)
        if not active_profile_name: return True

        if active_profile_name in index.get("borrowed", []):
            # Whether the source still exists, asked of its PID. This asked whether the
            # borrow's snapshot name was among the author's *personal* profiles, and
            # deleted the borrow -- memories and all -- when it was not: so every borrow
            # went on its holder's next message once the author renamed the source, and
            # every borrow of a System profile, whose name is never a personal one.
            source = self._borrow_source(user_id, active_profile_name)
            if source is None:
                source_exists = False
            elif source[1] is not None:
                source_exists = self._get_name_from_pid(source[0], source[1]) is not None
            else:
                owner_index = self._get_user_index(source[0])
                source_exists = (source[2] in owner_index.get("personal", [])
                                 or source[2] in owner_index.get("system", {}))
            if not source_exists:
                if isinstance(index["borrowed"], dict):
                    pid = index["borrowed"].pop(active_profile_name, active_profile_name)
                else:
                    index["borrowed"].remove(active_profile_name)
                    pid = active_profile_name
                
                self._save_user_index(user_id, index)
                
                channel_obj = self.cog.bot.get_channel(channel.id) if hasattr(channel, 'id') else channel
                server_id_str = str(channel_obj.guild.id) if channel_obj and getattr(channel_obj, 'guild', None) else "dm"
                server_index = self.cog.server_manager._get_server_index(server_id_str)
                if str(channel.id) in server_index.setdefault("user_active_profiles", {}).setdefault(str(user_id), {}):
                    del server_index["user_active_profiles"][str(user_id)][str(channel.id)]
                self.cog.server_manager._save_server_index(server_id_str, server_index)
                
                try:
                    p_dir = os.path.join(self.cog.USERS_DIR, str(user_id), "profiles", pid)
                    shutil.rmtree(p_dir, ignore_errors=True)
                    await channel.send(f"<@{user_id}>, the borrowed profile '{active_profile_name}' is broken because the original was deleted. It has been removed from your list and your active profile in this channel has been reset.")
                except discord.Forbidden:
                    pass
                return False
        return True


def attach_profile_managers(cog) -> ProfileManager:
    """Put ProfileManager and the managers split out of it on `cog`, and return the first.
    The one place they are wired: MimicCog and the tests both call it."""
    from .user_settings import UserSettings
    from .borrow_index import BorrowIndex
    from .sharing_manager import SharingManager
    from .library_manager import LibraryManager
    from .content_rating_manager import ContentRatingManager
    from .voice_sample_manager import VoiceSampleManager
    from .persona_content import PersonaContent
    from ..services.transfer_service import TransferService
    from ..gui.profile_dashboard import ProfileDashboard
    cog.profile_manager = ProfileManager(cog)
    cog.user_settings = UserSettings(cog)
    cog.borrows = BorrowIndex(cog)
    cog.sharing = SharingManager(cog)
    cog.library = LibraryManager(cog)
    cog.content_rating = ContentRatingManager(cog)
    cog.voice_samples = VoiceSampleManager(cog)
    cog.persona_content = PersonaContent(cog)
    cog.transfer = TransferService(cog)
    cog.profile_dashboard = ProfileDashboard(cog)
    return cog.profile_manager
