"""Profile shares and the borrow lifecycle: the share queue, accepting a share or a Library
borrow, clones, converting a borrow, and reaping borrows whose source is gone."""

import os
import shutil
import datetime
import asyncio
import traceback
import discord
from typing import Dict, List, Any, Optional, Set, Tuple
from ..utils.user_defaults import apply_defaults, model_slot_defaults, setting_label
from ..utils.constants import (
    USERS_DIR, defaultConfig, CONTENT_RATING_ADULT, CONTENT_RATING_CAPABILITIES, MODEL_PROVIDERS, PROVIDER_CHOICES, NOT_REGISTERED)
from .storage_manager import IOManager
from ..gui.base_components import dismissible, text_embed
from .profile_manager import _UNKNOWN_SOURCE_PID, _UNSHIPPED_MODEL_KEYS



class SharingManager:
    """Profile shares and the borrow lifecycle: the share queue, accepting a share or a
    Library borrow, clones, converting a borrow, and reaping borrows whose source is
    gone. Reached as `cog.sharing`."""

    def __init__(self, cog):
        self.cog = cog

    def _load_profile_shares(self):
        self.cog.profile_shares = {}
        if not os.path.isdir(USERS_DIR):
            return
        for user_id_str in os.listdir(USERS_DIR):
            if not user_id_str.isdigit(): continue
            file_path = os.path.join(USERS_DIR, user_id_str, "shares.json.gz")
            if os.path.exists(file_path):
                data = IOManager.read_json_gzip(file_path, self.cog.fernet)
                if data:
                    self.cog.profile_shares[user_id_str] = data

    def _save_profile_share_shard(self, recipient_id_str: str, data: List):
        if not data:
            self.cog.storage_manager._delete_shard("profile_shares", recipient_id_str)
        else:
            self.cog.storage_manager._save_shard("profile_shares", recipient_id_str, data)

    # --- Who may share with whom ---------------------------------------------------
    #
    # A recipient's answer lives beside the rest of their own settings, in About Me's
    # sparse block: `shares_closed` only while closed, `blocked_sharers` only while it
    # names someone. A share is only ever offered, never delivered: it waits in Incoming
    # Shares, and nothing is sent to anyone to say so.

    def accepts_share(self, recipient_id: int, sharer_id: int) -> bool:
        """Whether `sharer_id` may offer `recipient_id` a profile: someone registered, open
        to shares, who has not blocked them. The sender is told only that it could not be
        sent -- never which of the three it was."""
        about = self.cog.user_settings.get_user_about(recipient_id)
        return (about.get("provider") in PROVIDER_CHOICES and not about.get("shares_closed")
                and int(sharer_id) not in (about.get("blocked_sharers") or []))

    def set_shares_closed(self, user_id: int, closed: bool):
        about = self.cog.user_settings.get_user_about(user_id)
        about["shares_closed"] = True if closed else None
        self.cog.user_settings.save_user_about(user_id, about)

    def set_blocked_sharers(self, user_id: int, sharer_ids) -> List[int]:
        """Replaces the block list, and takes back whatever the blocked had waiting."""
        blocked = sorted({int(i) for i in sharer_ids} - {int(user_id)})
        about = self.cog.user_settings.get_user_about(user_id)
        about["blocked_sharers"] = blocked or None
        self.cog.user_settings.save_user_about(user_id, about)
        self._drop_pending_shares(user_id, lambda s: s.get("sharer_id") in blocked)
        return blocked

    def _pending_shares(self, recipient_id: int) -> List[Dict[str, Any]]:
        return list(self.cog.profile_shares.get(str(recipient_id)) or [])

    def _drop_pending_shares(self, recipient_id: int, drop) -> None:
        """Removes the shares `drop(share)` names from one recipient's queue, and saves."""
        key = str(recipient_id)
        shares = self._pending_shares(recipient_id)
        kept = [s for s in shares if not drop(s)]
        if len(kept) == len(shares):
            return
        if kept:
            self.cog.profile_shares[key] = kept
        else:
            self.cog.profile_shares.pop(key, None)
        self._save_profile_share_shard(key, kept)

    @staticmethod
    def _is_share_of(share: Dict[str, Any], sharer_id: int, target_pid: Optional[str],
                     fallback_name: str, kind: Optional[str] = None) -> bool:
        """`kind` is sparse: None is a borrow offer, "clone" a clone offer. A waiting
        offer of one kind never answers for the other."""
        if share.get("sharer_id") != sharer_id or share.get("kind") != kind:
            return False
        if target_pid:
            return share.get("original_pid") == target_pid
        return share.get("profile_name") == fallback_name

    def _borrower_defaults(self, user_id: int, config: Dict[str, Any]) -> Dict[str, Any]:
        """What this user's defaults would change on a borrow of `config`, as {key: value}.

        The models their provider ships, then Override Defaults' borrow-scoped keys -- a
        borrow made theirs the way a new profile is. Empty until they choose a provider.
        """
        preferred = self.cog.user_settings.provider_preference(user_id)
        if not preferred:
            return {}
        # No provider ships no models: only their own defaults move a slot.
        shipped = model_slot_defaults(preferred) if preferred in MODEL_PROVIDERS else {}
        mine = {**config, **{k: v for k, v in shipped.items() if k not in _UNSHIPPED_MODEL_KEYS}}
        apply_defaults(mine, self.cog.user_settings._get_user_defaults(user_id), borrowed=True)
        return {k: v for k, v in mine.items() if config.get(k) != v}

    async def _ask_borrow_defaults(self, interaction: discord.Interaction, profile_name: str,
                                   changes: Dict[str, Any]) -> bool:
        """Whether the borrower takes their defaults over the author's. No answer is no."""
        from ..gui.gui_hub import BorrowDefaultsView
        labels = sorted({setting_label(k) for k in changes})
        named = ", ".join(f"**{label}**" for label in labels[:6])
        if len(labels) > 6:
            named += f" and {len(labels) - 6} more"
        view = BorrowDefaultsView(self.cog)
        message = await interaction.followup.send(
            embed=text_embed(f"Borrowing `{profile_name}`: your defaults would change {named}. "
            f"Use yours, or keep what its author chose?"), view=view, ephemeral=True, wait=True)
        if await view.wait():
            try:
                await message.edit(embed=dismissible(text_embed("No answer, so the original's settings were kept.")),
                                   view=None)
            except discord.HTTPException:
                pass
        return view.use_mine

    def _validate_and_clean_borrowed_profiles(self, user_id: int) -> int:
        """
        Scans a user's borrowed profiles. If the source profile no longer exists
        (deleted by owner), it removes the borrowed entry.
        Returns the number of profiles removed.

        Blocking -- a read per borrow -- so callers run it in a thread, where two may
        clean the same user at once: hence the snapshot and the pop that names nothing.
        """
        index = self.cog.profile_manager._get_user_index(user_id)
        borrowed = index.get("borrowed", {})
        if not borrowed:
            return 0

        # Group by owner to minimize I/O
        profiles_by_owner = {}
        ids_to_remove = []
        #: local name -> owner, for the ones whose source is gone: the borrower is told.
        sourceless: Dict[str, str] = {}
        for local_name in list(borrowed):
            b_config = self.cog.profile_manager._get_profile_config(user_id, local_name, True)
            if not b_config:
                # Index entry with no readable profile behind it -- the directory was
                # removed, the file is corrupt, or a save was interrupted. This used
                # to be skipped silently, so the dead name stayed in the index
                # forever and kept passing the session-hydration existence check,
                # which only looks the name up in the index.
                ids_to_remove.append(local_name)
                continue

            src = self.cog.profile_manager._source_pointer(b_config)
            if src:
                profiles_by_owner.setdefault(str(src[0]), []).append((local_name, src[1]))
            else:
                # A borrowed profile with no source PID cannot be resolved or
                # refreshed; nothing can ever validate it again. The source *name* is
                # not a substitute -- it is a snapshot taken at borrow time, so an
                # owner's rename makes it name nothing.
                ids_to_remove.append(local_name)

        removed_count = 0

        for owner_id_str, items in profiles_by_owner.items():
            owner_index = self.cog.profile_manager._get_user_index(int(owner_id_str))
            # Personal and System both: a System profile can be shared, and validating
            # a borrow of one against the personal map alone condemned it.
            valid_pids = set()
            for category in ("personal", "system"):
                mapping = owner_index.get(category)
                if isinstance(mapping, dict):
                    valid_pids.update(mapping.values())

            for local_name, o_pid in items:
                if o_pid not in valid_pids:
                    ids_to_remove.append(local_name)
                    sourceless[local_name] = owner_id_str

        if ids_to_remove:
            gone = []
            for local_name in ids_to_remove:
                # No name fallback: a name is never a folder, and a racing clean got it first.
                pid = borrowed.pop(local_name, None)
                if pid:
                    shutil.rmtree(os.path.join(USERS_DIR, str(user_id), "profiles", pid),
                                  ignore_errors=True)
                    if local_name in sourceless:
                        gone.append((local_name, sourceless[local_name]))

            self._note_ended(index, gone, "deleted")
            self.cog.profile_manager._save_user_index(user_id, index)
            removed_count = len(ids_to_remove)

        return removed_count

    def _cascade_delete_borrowed_profiles(self, original_owner_id: int, deleted_pid: str) -> List[Tuple[int, str]]:
        """Instantly removes all borrowed variants linked to a deleted personal profile across
        the entire system. Returns `_end_borrows`'s (borrower, borrow PID) for each."""
        # Proactively remove the deleted profile from the global public database index.
        # By PID alone: matching the name as well removed the listing of any other profile
        # of this owner's that had once been called the same.
        public_ids_to_del = [d["id"] for d in self.cog.library._iter_public_entries(original_owner_id, include_orphaned=True)
                             if d["original_pid"] == deleted_pid]
        if public_ids_to_del:
            for pub_id in public_ids_to_del:
                self.cog.public_profiles.pop(pub_id, None)
            self.cog.library._save_public_index()

        return self._end_borrows(original_owner_id, deleted_pid, "deleted")

    #: Ended borrows a user is told of at once (`take_ended_borrows`); the oldest go first.
    ENDED_BORROW_NOTES = 25

    def _note_ended(self, index: Dict[str, Any], ended: List[Tuple[str, Any]], why: str) -> None:
        """Adds (local name, owner) borrows to the notes `take_ended_borrows` reads, in the
        About Me of `index`, for the save the caller makes anyway. Every path that ends
        someone else's borrow writes one, or its borrower never learns why it vanished."""
        if not ended:
            return
        about = index["about"] = index.get("about") if isinstance(index.get("about"), dict) else {}
        notes = (about.get("ended_borrows") or []) + [
            {"name": name, "owner": str(owner), "why": why} for name, owner in ended]
        about["ended_borrows"] = notes[-self.ENDED_BORROW_NOTES:]

    def _end_borrows(self, owner_id: int, source_pid: str, why: str,
                     only: Optional[Set[str]] = None) -> List[Tuple[int, str]]:
        """Removes the borrows of one source -- every one, or those whose PIDs are in `only`
        -- each borrower's copy with the memories it made, and leaves each borrower a note
        `/profile hub` shows them. `why` is "deleted" or "withdrawn". Returns (borrower,
        borrow PID) for each that went: the caller drops their seats (`drop_missing_seats`).

        Only `_borrowers_of` names what goes, so `only` cannot reach past this source's
        borrows whatever it holds. Blocking.
        """
        if not os.path.isdir(USERS_DIR):
            return []

        # The reverse index names the borrowers outright, so only they are touched --
        # this used to open and decrypt every borrow config every user owned. A stale
        # entry is tolerated rather than verified: a PID the borrower no longer holds
        # is skipped below, and a borrow the index has somehow missed is caught by
        # _validate_and_clean_borrowed_profiles and the daily integrity sweep, both of
        # which already exist to reap borrows whose source is gone.
        by_user: Dict[int, List[str]] = {}
        for borrower_id, borrow_pid in self.cog.borrows._borrowers_of(owner_id, source_pid):
            if only is None or borrow_pid in only:
                by_user.setdefault(borrower_id, []).append(borrow_pid)

        ended: List[Tuple[int, str]] = []
        for uid, to_delete in by_user.items():
            user_id_str = str(uid)
            try:
                index = self.cog.profile_manager._get_user_index(uid)
                borrowed = index.get("borrowed", {})
                gone = []

                # The index is keyed by the borrower's local name; the reverse index
                # hands us PIDs. Inverting once is cheaper than a scan per PID, and the
                # PID is what identifies the directory to remove -- a name lookup that
                # missed used to fall through to rmtree'ing a name-shaped path.
                # Snapshot: this runs in a thread while the loop may add a borrow.
                by_pid = {pid: name for name, pid in list(borrowed.items())}

                for borrow_pid in to_delete:
                    local_name = by_pid.get(borrow_pid)
                    if local_name is None:
                        continue
                    borrowed.pop(local_name, None)
                    shutil.rmtree(os.path.join(USERS_DIR, user_id_str, "profiles", borrow_pid),
                                  ignore_errors=True)
                    gone.append(local_name)
                    ended.append((uid, borrow_pid))

                if gone:
                    self._note_ended(index, [(name, owner_id) for name in gone], why)
                    self.cog.profile_manager._save_user_index(uid, index)
                else:
                    # Nothing to save, so the reconcile in _save_user_index will not
                    # run and the dead entries would survive in the index.
                    for borrow_pid in to_delete:
                        self.cog.borrows._borrow_index_remove(user_id_str, borrow_pid)
                    self.cog.borrows._save_borrow_index()
            except Exception as e:
                print(f"Error ending borrows for user {user_id_str}: {e}")
        return ended

    async def withdraw_borrows(self, owner_id: int, profile_name: str, borrow_pids: Set[str],
                               unpublish: bool = False) -> int:
        """Ends these borrows of one of the owner's profiles, and their seats in live sessions.
        With `unpublish` the listing goes first, so nobody borrows it again in between.
        Returns how many went."""
        def _withdraw():
            pid = self.cog.profile_manager._get_pid_from_name(owner_id, profile_name)
            if not pid:
                return []
            if unpublish:
                self.cog.library.unpublish_profile(owner_id, profile_name)
            return self._end_borrows(owner_id, pid, "withdrawn", only=set(borrow_pids))
        ended = await asyncio.to_thread(_withdraw)
        await self.cog.session_manager.drop_missing_seats({borrower for borrower, _ in ended})
        return len(ended)

    def take_ended_borrows(self, user_id: int) -> List[Dict[str, str]]:
        """This user's borrows that ended since they last opened `/profile hub`, cleared as
        they are read. Blocking."""
        about = self.cog.user_settings.get_user_about(user_id)
        notes = about.pop("ended_borrows", None)
        if notes:
            self.cog.user_settings.save_user_about(user_id, about)
        return notes or []


    async def _accept_share_request(self, interaction: discord.Interaction, sharer_id: int, target_pid: Optional[str], fallback_name: str, desired_name: str, is_public_borrow: bool = False) -> bool:
        """Creates the borrow. True if it was made; otherwise the reason has been sent.

        A private borrow needs a share still waiting for this user from this sharer: there
        is no other way to be handed one, and blocking the sharer took theirs back.
        """
        recipient_id = interaction.user.id
        if not self.cog.user_settings.is_registered(recipient_id):
            await interaction.followup.send(embed=text_embed(NOT_REGISTERED), ephemeral=True)
            return False
        if not is_public_borrow and not any(
                self._is_share_of(s, sharer_id, target_pid, fallback_name)
                for s in self._pending_shares(recipient_id)):
            await interaction.followup.send(embed=text_embed("That share is no longer waiting for you."), ephemeral=True)
            return False
        # A Library page is a snapshot from when it opened: a listing its owner has since
        # taken down stays on it, Borrow and all, until it is repainted.
        if is_public_borrow and not (target_pid and self.cog.library._listing(sharer_id, target_pid)):
            await interaction.followup.send(embed=text_embed("That profile is no longer in the Public Library."), ephemeral=True)
            return False
        def _sync_prepare():
            current_name = self.cog.profile_manager._get_name_from_pid(sharer_id, target_pid) if target_pid else fallback_name
            if not current_name: current_name = fallback_name

            owner_profile_data = self.cog.profile_manager._get_profile_config(sharer_id, current_name, False)
            if not owner_profile_data:
                return None
            # Not the author's error message: it is posted under the character's name and no
            # rating reads it (`_moderated_surface`), so a listing carried unrated text into
            # every borrower's servers. The borrower writes their own under Misc.
            owner_profile_data.pop("error_response", None)

            # Checked here, when the borrow is made, as well as where the share was
            # offered: a request or code can wait while the rating changes under it,
            # and a Library listing can outlive the rating it was published on.
            verdict, _ = self.cog.content_rating._content_rating_state(sharer_id, current_name)
            capability = "publish" if is_public_borrow else "share"
            if not CONTENT_RATING_CAPABILITIES[verdict][capability]:
                if is_public_borrow:
                    # A listing left over from before its rating changed; nobody else
                    # should have to find that out by trying.
                    self.cog.library._delist_unless_publishable(sharer_id, current_name)
                return current_name, verdict

            index = self.cog.profile_manager._get_user_index(interaction.user.id)
            current_borrowed = len(index.get("borrowed", {})) if isinstance(index.get("borrowed"), dict) else len(index.get("borrowed", []))
            return current_name, owner_profile_data, index, current_borrowed

        prep = await asyncio.to_thread(_sync_prepare)
        if prep is None:
            await interaction.followup.send(embed=text_embed("The shared profile seems to no longer exist."), ephemeral=True)
            return False
        if len(prep) == 2:
            refused_name, verdict = prep
            reason = ("A profile rated Not Allowed cannot be shared." if verdict == CONTENT_RATING_ADULT
                      else "It is not rated for sharing right now. Its owner needs to have it rated first.")
            await interaction.followup.send(
                embed=text_embed(f"❌ **'{refused_name}' cannot be borrowed.** {reason}"), ephemeral=True)
            return False
        current_name, owner_profile_data, index, current_borrowed = prep

        limit = defaultConfig.LIMIT_BORROWED
        if current_borrowed >= limit:
            await interaction.followup.send(embed=text_embed(f"Limit Reached. You have {current_borrowed}/{limit} borrowed profiles."), ephemeral=True)
            return False

        # Asked only where it would change something: defaults are off until the user
        # chooses a provider, and a profile already matching them leaves nothing to decide.
        mine = self._borrower_defaults(interaction.user.id, owner_profile_data)
        if mine and not await self._ask_borrow_defaults(interaction, current_name, mine):
            mine = {}

        # Collected inside the threaded save, read after it: what the borrow arrived
        # with is not what the library showed, and the user is owed the difference.
        adjustments: List[Tuple[str, str]] = []

        def _sync_save():
            # The source PID, resolved rather than defaulted. This used to fall back to
            # the literal "00000000" -- see _UNKNOWN_SOURCE_PID for what that did to the
            # reverse index -- and then to a `profile_id` copied into the config, which
            # was only ever consulted when the index had no PID, and so named nothing.
            # A borrow whose source cannot be named is not a borrow that
            # should be created: it can never be cascade-deleted, never counts towards
            # the source being distributed, and can never be refreshed.
            target_original_pid = target_pid or self.cog.profile_manager._get_pid_from_name(sharer_id, current_name)
            if not target_original_pid or target_original_pid == _UNKNOWN_SOURCE_PID:
                return False

            snapshot_data = owner_profile_data.copy()
            snapshot_data.update({
                "original_owner_id": str(sharer_id),
                "original_pid": target_original_pid,
                "original_profile_name": current_name,
                "borrowed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "ltm_creation_enabled": False
            })

            # Two passes over the inherited config, in this order.
            #
            # First the borrower's own defaults, if they chose them -- only the keys
            # whose action row is scope="all", see user_defaults. A borrow is somebody
            # else's writing, and a standing preference for, say, temperature 1.1 must
            # not silently retune a persona built at 0.4.
            #
            # Then the rescue, which is not a preference at all: it repoints model
            # slots naming a provider this user holds no key for. Order matters --
            # an explicit default is checked by the rescue like any other value, so
            # choosing an OpenRouter default without an OpenRouter key cannot produce
            # a profile that is dead on arrival.
            snapshot_data.update(mine)
            rescued = self.cog.user_settings._rescue_unusable_models(interaction.user.id, snapshot_data)
            adjustments.extend(("rescue", k) for k in rescued)

            # The letter is how the borrow arrived -- 'C' through the Public Library, 'B'
            # through a private share -- and `borrow_origin` reads it. No borrow predates
            # 'C', so every one carries the right letter. Provenance, not state: a PID is
            # the folder name and never changes, so a C stays a C after its source is
            # unpublished. Whether the source is listed now is _find_public_entry_id's.
            pid = self.cog.profile_manager._mint_pid(interaction.user.id, 'C' if is_public_borrow else 'B')

            if "borrowed" not in index or not isinstance(index["borrowed"], dict):
                index["borrowed"] = {}

            index["borrowed"][desired_name] = pid

            unified_borrowed = {
                "name": desired_name,
                "config": snapshot_data,
                "prompts": {},
                "child_bot": None
            }
            # Shard first, index second, and here the ordering is load-bearing beyond
            # crash safety: _save_user_index runs _reconcile_borrow_index, which reads
            # the new borrow's shard to learn what it points at. Saving the index first
            # meant that read found no file, _borrow_key_for_config got an empty config
            # and returned None, and the borrow was never written to borrows.json at
            # all -- so is_profile_distributed reported a freshly borrowed profile as
            # private until the borrower's index happened to fall out of the LRU.
            self.cog.profile_manager._save_profile_by_pid(interaction.user.id, pid, unified_borrowed)
            self.cog.profile_manager._save_user_index(interaction.user.id, index)
            return True

        if not await asyncio.to_thread(_sync_save):
            await interaction.followup.send(
                embed=text_embed("That profile could not be traced back to its owner, so it cannot be "
                "borrowed. Ask them to re-share it."), ephemeral=True)
            return False

        if not is_public_borrow:
            await self._reject_share_request(interaction, sharer_id, target_pid, fallback_name)

        await self._report_borrow_adjustments(interaction, desired_name, adjustments)
        return True

    async def _report_borrow_adjustments(self, interaction: discord.Interaction,
                                         profile_name: str,
                                         adjustments: List[Tuple[str, str]]):
        """Tells the borrower which inherited settings did not survive the borrow.

        Silent on the common case, where nothing was changed. It exists for a model
        slot repointed because its provider is one this user cannot reach, which
        leaves the profile behaving differently from the one in the library -- and a
        difference nobody announced reads as the borrow being broken. Their own
        defaults are not reported: they were asked (`_ask_borrow_defaults`).
        """
        if not adjustments:
            return

        lines = []
        rescued = [k for kind, k in adjustments if kind == "rescue"]

        if rescued:
            lines.append(
                f"⚠️ `{profile_name}` was built around a provider you have no key for. "
                f"Repointed to the bot's defaults: "
                + ", ".join(f"**{setting_label(k)}**" for k in rescued)
                + ".\nAdd the missing key in `/settings`, then set the models back in "
                  f"`/profile manage profile_name:{profile_name}`.")

        try:
            await interaction.followup.send(embed=text_embed("\n\n".join(lines)), ephemeral=True)
        except Exception:
            # A borrow that succeeded must not be reported as failed because its
            # explanatory note could not be delivered.
            pass

    async def _reject_share_request(self, interaction: discord.Interaction, sharer_id: int, target_pid: Optional[str], fallback_name: str, kind: Optional[str] = None):
        """Takes one share off the recipient's queue -- accepted or declined, nobody is told."""
        self._drop_pending_shares(
            interaction.user.id, lambda s: self._is_share_of(s, sharer_id, target_pid, fallback_name, kind))

    async def _execute_clone_handshake(self, owner_id: int, source_pid: str, recipient_id: int, desired_name: str) -> Tuple[bool, str]:
        """Makes the clone a waiting clone offer names, and takes the offer off the queue.

        The offer is the only authorisation, as a private borrow's is: it was sent
        through `accepts_share`, and blocking its sender took it back.
        """
        if not self.cog.user_settings.is_registered(recipient_id):
            return False, NOT_REGISTERED
        if not any(self._is_share_of(s, owner_id, source_pid, "", "clone")
                   for s in self._pending_shares(recipient_id)):
            return False, "That clone is no longer waiting for you."

        def _sync_clone():
            owner_id_str = str(owner_id)
            recip_id_str = str(recipient_id)

            src_dir = os.path.join(self.cog.USERS_DIR, owner_id_str, "profiles", source_pid)

            if not os.path.exists(src_dir):
                return False, "Source profile data no longer exists."

            # A clone hands another user the whole persona, so it is sharing, and is
            # gated as sharing is -- again here because the offer can outlive the rating.
            source_name = self.cog.profile_manager._get_name_from_pid(owner_id, source_pid)
            if not source_name:
                return False, "Source profile data no longer exists."
            allowed, reason = self.cog.content_rating.content_capability(owner_id, source_name, "share")
            if not allowed:
                return False, f"❌ **This profile cannot be cloned.** {reason}"

            index = self.cog.profile_manager._get_user_index(recipient_id)
            if len(index.get("personal", [])) >= defaultConfig.LIMIT_PROFILES:
                return False, "You have reached your personal profile limit."
            # Writing the index entry over an existing one would orphan that profile.
            if desired_name in index.get("personal", []) or desired_name in index.get("borrowed", []):
                return False, f"You already have a profile named '{desired_name}'."

            new_pid = self.cog.profile_manager._mint_pid(recipient_id, "A")
            recip_dir = os.path.join(self.cog.USERS_DIR, recip_id_str, "profiles", new_pid)

            try:
                src_data = self.cog.profile_manager._get_profile_by_pid(owner_id, source_pid) or {}
                config_data = src_data.get("config", {}).copy()
                config_data["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

                new_data = {
                    "name": desired_name,
                    "config": config_data,
                    "prompts": src_data.get("prompts", {}).copy(),
                    "child_bot": None
                }
                self.cog.profile_manager._save_profile_by_pid(recipient_id, new_pid, new_data)

                self.cog.user_appearances.setdefault(recip_id_str, {})[desired_name] = \
                    self.cog.profile_manager._appearance_entry(config_data)

                if not isinstance(index.get("personal"), dict):
                    legacy_personal = index.get("personal", [])
                    index["personal"] = {}
                    if isinstance(legacy_personal, list):
                        for p_name in legacy_personal:
                            index["personal"][p_name] = p_name

                index["personal"][desired_name] = new_pid
                self.cog.profile_manager._save_user_index(recipient_id, index)
                return True, f"✅ Profile cloned successfully as '{desired_name}'!"

            except Exception as e:
                print(f"Error executing clone handshake: {e}")
                traceback.print_exc()
                shutil.rmtree(recip_dir, ignore_errors=True)
                return False, f"An unexpected error occurred during cloning: {e}"

        ok, message = await asyncio.to_thread(_sync_clone)
        if ok:
            self._drop_pending_shares(
                recipient_id, lambda s: self._is_share_of(s, owner_id, source_pid, "", "clone"))
        return ok, message


    async def _convert_copy_profile(self, user_id: int, source_name: str, target_name: str, to_system: bool) -> Tuple[bool, str]:
        owner_id = int(defaultConfig.DISCORD_OWNER_ID)
        if user_id != owner_id:
            return False, "Only the Bot Owner can perform system profile conversions."

        is_valid, err_msg = self.cog.profile_manager._is_valid_profile_name(target_name)
        if not is_valid:
            return False, err_msg

        def _sync_convert():
            owner_index = self.cog.profile_manager._get_user_index(owner_id)

            if to_system:
                if source_name not in owner_index.get("personal", {}):
                    return False, f"Source personal profile '{source_name}' not found."
                if target_name in owner_index.get("system", {}):
                    return False, f"A system profile named '{target_name}' already exists."
                source_pid = owner_index["personal"][source_name]
                target_pid = self.cog.profile_manager._mint_pid(owner_id, "X")
                target_category = "system"
            else:
                if source_name not in owner_index.get("system", {}):
                    return False, f"Source system profile '{source_name}' not found."
                if target_name in owner_index.get("personal", {}) or target_name in owner_index.get("borrowed", {}):
                    return False, f"A personal profile named '{target_name}' already exists."
                source_pid = owner_index["system"][source_name]
                target_pid = self.cog.profile_manager._mint_pid(owner_id, "A")
                target_category = "personal"

            source_dir = os.path.join(self.cog.USERS_DIR, str(owner_id), "profiles", source_pid)
            target_dir = os.path.join(self.cog.USERS_DIR, str(owner_id), "profiles", target_pid)

            if not os.path.exists(source_dir):
                return False, "Source profile directory does not exist on disk."

            try:
                os.makedirs(target_dir, exist_ok=True)
                src_data = self.cog.profile_manager._get_profile_by_pid(owner_id, source_pid) or {}
                config_data = src_data.get("config", {}).copy()
                config_data["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

                target_data = {
                    "name": target_name,
                    "config": config_data,
                    "prompts": src_data.get("prompts", {}).copy(),
                    "child_bot": None
                }
                self.cog.profile_manager._save_profile_by_pid(owner_id, target_pid, target_data)

                src_file = os.path.join(source_dir, "training.json.gz")
                if os.path.exists(src_file):
                    shutil.copy2(src_file, os.path.join(target_dir, "training.json.gz"))
                if os.path.isdir(os.path.join(source_dir, "ltm")):
                    shutil.copytree(os.path.join(source_dir, "ltm"), os.path.join(target_dir, "ltm"),
                                    dirs_exist_ok=True)
                    self.cog.memory_manager._index_ltm_copy(os.path.join(source_dir, "ltm"),
                                                            os.path.join(target_dir, "ltm"))

                if target_category not in owner_index or not isinstance(owner_index[target_category], dict):
                    owner_index[target_category] = {}

                owner_index[target_category][target_name] = target_pid
                self.cog.profile_manager._save_user_index(owner_id, owner_index)

                return True, f"Successfully copied '{source_name}' to {target_category.title()} Profile '{target_name}'."
            except Exception as e:
                shutil.rmtree(target_dir, ignore_errors=True)
                return False, f"An unexpected error occurred during copy: {e}"

        return await asyncio.to_thread(_sync_convert)
