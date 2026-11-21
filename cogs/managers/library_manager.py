"""The Public Library: the public index, a listing's text, publishing and unpublishing, and
whether a profile is distributed."""

import os
import time
import threading
import datetime
from typing import Dict, List, Any, Optional, Tuple
from ..utils.constants import (
    USERS_DIR, PUBLIC_PROFILES_DIR, CONTENT_RATING_PENDING, CONTENT_RATING_CAPABILITIES, LIMIT_PUBLISHES_PER_DAY)
from ..utils.helpers import birth_time
from .storage_manager import IOManager

#: A rating job delists from a worker thread while the loop publishes: serialised inside
#: this lock, the last os.replace always carries the newest dict, never an older snapshot.
_PUBLIC_INDEX_LOCK = threading.Lock()


class LibraryManager:
    """The Public Library: the public index, a listing's text, publishing and unpublishing,
    and whether a profile is distributed. Reached as `cog.library`."""

    def __init__(self, cog):
        self.cog = cog

    def _load_public_profiles(self):
        index_path = os.path.join(PUBLIC_PROFILES_DIR, "index.json")
        data = IOManager.read_json(index_path)
        if data is None and os.path.exists(index_path):
            # A failed open, not bad bytes (those were quarantined): the listings exist
            # nowhere else, and starting empty meant the next save wrote over all of them.
            raise RuntimeError(f"Public index unreadable, refusing to start empty: {index_path}")
        self.cog.public_profiles = data if isinstance(data, dict) else {}

    def _save_public_index(self):
        index_path = os.path.join(PUBLIC_PROFILES_DIR, "index.json")
        with _PUBLIC_INDEX_LOCK:
            IOManager.write_json(self.cog.public_profiles, index_path)

    def _describe_public_entry(self, entry_id: str, p_info: Any) -> Optional[Dict[str, Any]]:
        """Normalises one public-index entry into a single shape.

        The index holds two formats. Publishing writes a plain "owner_id:pid"
        string; older entries are a dict carrying the display name and timestamp.
        Four separate readers branched on the type themselves, and every string
        branch looked the profile name up in a `profiles/<pid>/name.txt` that
        nothing in the codebase has ever written -- so a string entry always
        rendered as "Unknown" and, worse, silently vanished from the
        already-published set that apply_public diffs against. The name is
        resolved from the owner's index here, which is where it actually lives.

        Returns None for a malformed entry so callers can skip it.
        """
        name = None
        published_at = updated_at = 0
        status = "active"

        if isinstance(p_info, str) and ":" in p_info:
            owner_str, pid = p_info.split(":", 1)
        elif isinstance(p_info, dict):
            owner_str = str(p_info.get("owner_id"))
            pid = p_info.get("original_pid") or p_info.get("original_profile_id")
            name = p_info.get("original_profile_name")
            published_at = p_info.get("published_at") or 0
            updated_at = p_info.get("updated_at") or 0
            status = p_info.get("status", "active")
        else:
            return None

        if not pid or not owner_str:
            return None
        try:
            owner_id = int(owner_str)
        except (TypeError, ValueError):
            return None

        # The owner's index is the authority on whether the profile still exists.
        # A pid that is not in it was deleted, unpublished by hand, or belonged to a
        # user whose data was wiped -- the entry is a tombstone. It used to render
        # as the literal string "Unknown" and sit in the hub forever.
        resolved = self.cog.profile_manager._get_name_from_pid(owner_id, pid)
        orphaned = resolved is None
        # The live name first: the stored one is a snapshot from publishing, so a profile
        # renamed since was listed under a name its owner no longer uses. The snapshot
        # only labels a tombstone.
        name = resolved or (name if name and name != "Unknown" else None)

        if not published_at:
            # An "owner_id:pid" entry carries no time. Its profile directory's birth, never
            # the profile file's mtime: every save rewrites that file, so a profile in use
            # kept jumping to the top of Newest as if republished.
            published_at = birth_time(os.path.join(USERS_DIR, str(owner_id), "profiles", pid)) or 0

        return {
            "id": entry_id,
            "owner_id": owner_id,
            "original_pid": pid,
            "profile_name": name or "Unknown",
            "published_at": published_at,
            # A new listing is an update too; a republish with nothing edited is not.
            "updated_at": max(updated_at, published_at),
            "status": status,
            "orphaned": orphaned,
        }

    def _iter_public_entries(self, owner_id: Optional[int] = None, include_orphaned: bool = False):
        """Yields normalised public entries, optionally filtered to one owner.

        Orphans are withheld by default so a tombstone can never be listed,
        borrowed, or counted, even between prunes.

        Filtered by owner *before* describing: describing reads the owner's index, so
        filtering after made every one-owner call pay for the whole catalogue.
        """
        wanted = None if owner_id is None else int(owner_id)
        for entry_id, p_info in list(self.cog.public_profiles.items()):
            if wanted is not None and self._public_entry_owner(p_info) != wanted:
                continue
            desc = self._describe_public_entry(entry_id, p_info)
            if not desc:
                continue
            if desc["orphaned"] and not include_orphaned:
                continue
            yield desc

    @staticmethod
    def _public_entry_owner(p_info: Any) -> Optional[int]:
        """The owner a public entry names, read off the entry alone -- no index lookup."""
        owner = (p_info.split(":", 1)[0] if isinstance(p_info, str)
                 else p_info.get("owner_id") if isinstance(p_info, dict) else None)
        try:
            return int(owner)
        except (TypeError, ValueError):
            return None

    def _prune_public_index(self) -> List[Dict[str, Any]]:
        """Drops public entries whose source profile no longer exists, or whose rating
        no longer allows a listing.

        Two independent checks for existence, because either can fail on its own: the
        pid must still be in the owner's personal index, and the profile file must still
        be on disk. Deletion normally cascades through
        _cascade_delete_borrowed_profiles, so anything reaching here survived a
        crash, a manual edit, or a delete path that predates that cascade.

        The rating is read off the file, as `_delist_unless_publishable` reads it, for a
        listing that no write delisted: one granted under a verdict a later build
        removed (the bot owner's Exempt) reads Unrated now, and a Library borrow does
        not ask the rating again. One decrypt per listing, at boot.

        The same read refreshes every kept listing's text, filed as a dict under its own
        PID (see `_update_listing`): a legacy "owner_id:pid" string or a key that is not the
        PID is converted here, and a second entry for the same PID dropped.

        Returns the removed entries so a caller can report them.
        """
        removed = []
        changed = False
        for entry_id, p_info in list(self.cog.public_profiles.items()):
            desc = self._describe_public_entry(entry_id, p_info)

            if desc is None:
                # Unparseable entry -- neither a pointer string nor a dict.
                removed.append({"id": entry_id, "profile_name": "<malformed>", "owner_id": None})
                self.cog.public_profiles.pop(entry_id, None)
                continue

            if not desc["orphaned"]:
                profile_path = os.path.join(
                    USERS_DIR, str(desc["owner_id"]), "profiles", desc["original_pid"], "profile.json.gz")
                if os.path.exists(profile_path):
                    p_data = IOManager.read_json_gzip(profile_path, self.cog.fernet) or {}
                    config = p_data.get("config") or {}
                    verdict = self.cog.content_rating._verdict_of(config)
                    if verdict == CONTENT_RATING_PENDING or CONTENT_RATING_CAPABILITIES[verdict]["publish"]:
                        pid = desc["original_pid"]
                        entry = p_info if isinstance(p_info, dict) else {
                            "owner_id": desc["owner_id"], "original_pid": pid}
                        filed = self._with_listing_text(entry, config)
                        if entry_id != pid:
                            self.cog.public_profiles.pop(entry_id, None)
                            changed = True
                            if pid in self.cog.public_profiles:
                                continue
                        if self.cog.public_profiles.get(pid) != filed:
                            self.cog.public_profiles[pid] = filed
                            changed = True
                        continue

            removed.append(desc)
            self.cog.public_profiles.pop(entry_id, None)

        if removed or changed:
            self._save_public_index()
        return removed

    def _find_public_entry_id(self, owner_id: int, pid: str) -> Optional[str]:
        """The public-index key for a given owner/pid, or None if not published."""
        for desc in self._iter_public_entries(owner_id):
            if desc["original_pid"] == pid:
                return desc["id"]
        return None

    def _delist_unless_publishable(self, owner_id: int, profile_name: str) -> bool:
        """Takes a profile out of the Public Library once its rating no longer allows it.

        Called by every write that can leave a listed profile Unrated or Not Allowed. Nothing
        else re-reads the rating of a listing, so without this the Library went on
        offering a profile its own rating said could not be shared. True if an entry
        was removed. Pending is left listed: it is a verdict on its way, and the write
        that lands it comes back through here. Existing borrows are untouched: they read
        the source profile, never this index. Blocking; the callers run in a thread.
        """
        verdict, _ = self.cog.content_rating._content_rating_state(owner_id, profile_name)
        if verdict == CONTENT_RATING_PENDING or CONTENT_RATING_CAPABILITIES[verdict]["publish"]:
            return False
        return self.unpublish_profile(owner_id, profile_name)

    def unpublish_profile(self, owner_id: int, profile_name: str) -> bool:
        """Takes a profile out of the Public Library. True if it was listed.

        By PID: an entry's stored name is a snapshot from publishing, so a profile renamed
        since could not be found by name. Existing borrows are untouched. Blocking.

        When it was first published is kept, in About Me's sparse block, for `list_public`:
        published again, it returns to its place in Newest rather than taking the top.
        """
        pid = self.cog.profile_manager._get_pid_from_name(owner_id, profile_name)
        if not pid:
            return False
        entries = [d for d in self._iter_public_entries(owner_id, include_orphaned=True)
                   if d["original_pid"] == pid]
        for d in entries:
            self.cog.public_profiles.pop(d["id"], None)
        if entries:
            self._save_public_index()
            about = self.cog.user_settings.get_user_about(owner_id)
            # Pruned to profiles the owner still has, so it never outgrows them.
            first = {p: t for p, t in (about.get("first_published") or {}).items()
                     if self.cog.profile_manager._get_name_from_pid(owner_id, p) is not None}
            first[pid] = first.get(pid) or min(int(d["published_at"]) for d in entries)
            about["first_published"] = first
            self.cog.user_settings.save_user_about(owner_id, about)
        return bool(entries)

    # --- A listing's text ------------------------------------------------------------
    #
    # What the Library searches and shows -- display name and intro -- is copied into the
    # listing's own public-index entry, which is in memory whole, so a search or a page of
    # the dropdown opens no profile. Public by definition, so plaintext. Written by
    # _save_profile_by_pid on every save of a listed profile, by list_public when one is
    # listed, and refreshed for every listing by the boot sweep (_prune_public_index), which
    # also files each under its own PID: what lets a save find its listing in one lookup.

    _LISTING_TEXT_KEYS = ("display_name", "intro")

    @staticmethod
    def _listing_text(config: Dict[str, Any]) -> Dict[str, str]:
        text = {"display_name": config.get("custom_display_name"), "intro": config.get("library_intro")}
        return {k: str(v) for k, v in text.items() if v}

    def _with_listing_text(self, entry: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        return {**{k: v for k, v in entry.items() if k not in self._LISTING_TEXT_KEYS},
                **self._listing_text(config)}

    def _update_listing(self, owner_id: int, pid: str, config: Dict[str, Any]):
        """Refresh a listed profile's text, saving the public index only when it changed.
        One lookup for the unlisted: this runs on every profile save. Best-effort.

        In place, never re-inserted: this runs in a save's worker thread, and writing the
        entry back put a listing an unpublish had just taken down back in the Library."""
        try:
            entry = self.cog.public_profiles.get(pid)
            if not isinstance(entry, dict) or self._public_entry_owner(entry) != int(owner_id):
                return
            text = self._listing_text(config)
            if {k: entry[k] for k in self._LISTING_TEXT_KEYS if k in entry} != text:
                for k in self._LISTING_TEXT_KEYS:
                    entry.pop(k, None)
                entry.update(text)
                self._save_public_index()
        except Exception as e:
            print(f"Public index: listing text update failed for {owner_id}/{pid} ({e})")

    def _listing(self, owner_id: int, pid: str) -> Dict[str, Any]:
        entry = self.cog.public_profiles.get(pid)
        return entry if isinstance(entry, dict) and self._public_entry_owner(entry) == int(owner_id) else {}

    def library_search_text(self, owner_id: int, pid: str) -> str:
        """A listing's display name and intro, lower-cased, for the Library's search."""
        listing = self._listing(owner_id, pid)
        return "\n".join(listing[k] for k in self._LISTING_TEXT_KEYS if k in listing).lower()

    def library_display_name(self, owner_id: int, pid: str) -> str:
        """A listing's display name, or "" for none."""
        return self._listing(owner_id, pid).get("display_name", "")

    def distribution_of(self, owner_id: int, profile_name: str) -> Tuple[bool, int]:
        """(listed in the Public Library, number of users borrowing it), for copy that
        has to say which. `is_profile_distributed` is the yes/no form."""
        eff_owner, eff_name = self.cog.profile_manager._resolve_effective_profile(owner_id, profile_name)
        pid = self.cog.profile_manager._get_pid_from_name(eff_owner, eff_name)
        borrowers = {borrower_id for borrower_id, _ in self.cog.borrows._borrowers_of(eff_owner, pid)} if pid else set()
        return self._is_profile_public(owner_id, profile_name), len(borrowers)

    def _is_profile_public(self, user_id: int, profile_name: str) -> bool:
        """Whether the profile this name speaks as -- a borrow's source -- is listed. One
        lookup: every listing is filed under its own PID (`_prune_public_index`)."""
        if profile_name in self.cog.profile_manager._get_user_index(user_id).get("borrowed", {}):
            source = self.cog.profile_manager._borrow_source(user_id, profile_name)
            if source is None:
                return False
            owner, pid = source[0], source[1]
        else:
            owner, pid = self.cog.profile_manager._resolve_prompt_owner_and_pid(user_id, profile_name)
        return bool(pid and self._listing(owner, pid))

    def is_profile_distributed(self, owner_id: int, profile_name: str) -> bool:
        """True when someone other than the owner currently holds this profile.

        Published to the library, or borrowed by at least one user. This is the line
        the whole scheme turns on: an undistributed profile is nobody else's problem
        and is never classified without being asked for, while a distributed one has
        to keep its verdict honest because other people are running it.

        Two dict lookups against the borrow reverse index. It used to walk every user
        directory and decrypt every borrow config, which is why the index exists.
        """
        if self._is_profile_public(owner_id, profile_name):
            return True

        eff_owner, eff_name = self.cog.profile_manager._resolve_effective_profile(owner_id, profile_name)
        # Strict: a name that resolves to no PID has no borrowers to find, and the
        # soft resolver's name-shaped answer would be looked up as though it were one.
        pid = self.cog.profile_manager._get_pid_from_name(eff_owner, eff_name)
        return bool(self.cog.borrows._borrowers_of(eff_owner, pid))

    async def publish_profile(self, owner_id: int, profile_name: str) -> Tuple[bool, Optional[str]]:
        """Lists one of the owner's profiles in the Public Library, rating it first if it
        has no verdict. (listed, reason_if_not); one already listed is a success."""
        if self._is_profile_public(owner_id, profile_name):
            return True, None
        left, refusal = self.publishes_left(owner_id)
        if not left:
            return False, refusal
        allowed, reason = await self.cog.content_rating.rate_for_distribution(owner_id, profile_name, "publish")
        if not allowed:
            return False, reason
        if not self.list_public(owner_id, profile_name):
            return False, "Profile not found."
        self._save_public_index()
        self.note_publishes(owner_id, 1)
        return True, None

    def list_public(self, owner_id: int, profile_name: str) -> Optional[str]:
        """Puts one profile in the public index; its PID, or None if it has none. The
        caller saves the index.

        Newest sorts by when it was first published, which a republish keeps
        (`unpublish_profile`), so unlisting and listing again buys no place. Recently
        Updated sorts by when its rating last landed: an edit cannot be listed unrated.
        """
        pid = self.cog.profile_manager._get_pid_from_name(owner_id, profile_name)
        if not pid:
            return None
        first = (self.cog.user_settings.get_user_about(owner_id).get("first_published") or {}).get(pid)
        self.cog.public_profiles[pid] = {"owner_id": int(owner_id), "original_pid": pid,
                                         "published_at": first or int(time.time()),
                                         "updated_at": self._rated_at(owner_id, profile_name),
                                         **self._listing_text(
                                             self.cog.profile_manager._get_profile_config(owner_id, profile_name, False) or {})}
        return pid

    def _rated_at(self, owner_id: int, profile_name: str) -> int:
        """When the profile's rating landed, in epoch seconds; 0 if it has none."""
        rating = (self.cog.profile_manager._get_profile_config(owner_id, profile_name, False) or {}).get("content_rating") or {}
        try:
            return int(datetime.datetime.fromisoformat(rating["at"]).timestamp())
        except (KeyError, TypeError, ValueError):
            return 0

    def _restamp_listing(self, owner_id: int, profile_name: str) -> None:
        """Moves a listed profile's Recently Updated time to its new rating's. In place,
        never re-inserted, as `_update_listing`: an unpublish in a worker thread may have
        taken the entry since it was found."""
        pid = self.cog.profile_manager._get_pid_from_name(owner_id, profile_name)
        entries = [self.cog.public_profiles.get(d["id"]) for d in self._iter_public_entries(owner_id)
                   if d["original_pid"] == pid]
        entries = [e for e in entries if isinstance(e, dict)]
        rated_at = self._rated_at(owner_id, profile_name) if entries else 0
        for entry in entries:
            entry["updated_at"] = rated_at
        if entries:
            self._save_public_index()

    # Publishes in the last 24 hours, as epoch seconds, in About Me's sparse block beside
    # the share settings: `published_times` only while it holds one.

    def _recent_publishes(self, owner_id: int) -> List[int]:
        cutoff = time.time() - 86400
        return [t for t in self.cog.user_settings.get_user_about(owner_id).get("published_times") or [] if t > cutoff]

    def publishes_left(self, owner_id: int) -> Tuple[int, str]:
        """(listings this user may still make, why one past those is refused). A rolling
        day: each publish holds its slot for 24 hours."""
        recent = self._recent_publishes(owner_id)
        frees = int(min(recent, default=time.time())) + 86400
        return (max(0, LIMIT_PUBLISHES_PER_DAY - len(recent)),
                f"You can publish {LIMIT_PUBLISHES_PER_DAY} profiles a day; the next slot opens "
                f"<t:{frees}:R>.")

    def note_publishes(self, owner_id: int, count: int) -> None:
        if count:
            about = self.cog.user_settings.get_user_about(owner_id)
            about["published_times"] = self._recent_publishes(owner_id) + [int(time.time())] * count
            self.cog.user_settings.save_user_about(owner_id, about)
