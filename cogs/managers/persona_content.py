"""What a persona carries beside its prompts: lorebook entries, fixed examples, bonds and the
character's birthday."""

import os
from typing import Dict, Iterable, List, Any, Optional, Tuple
from ..utils.constants import LIMIT_FIXED_EXAMPLES, LIMIT_LORE_ENTRIES
from ..utils.helpers import _get_user_hash, _legacy_user_hash
from ..utils import fixed_examples, lorebook
from ..utils.bonds import apply_updates as apply_bond_updates



class PersonaContent:
    """What a persona carries beside its prompts: lorebook entries, fixed examples, bonds
    and the character's birthday. Reached as `cog.persona_content`."""

    def __init__(self, cog):
        self.cog = cog

    def character_birthday(self, user_id: int, profile_name: str) -> Any:
        """The stored birthday of the character this name speaks as, unvalidated, or None.

        Not `_get_profile_prompts` alone: that reads a borrow's own shard, and a borrow is
        written with empty prompts, so every borrowed character went without its birthday.
        A borrow's is its source's, read by PID -- see `_borrow_source`. A personal or
        System profile reads its own.
        """
        index = self.cog.profile_manager._get_user_index(user_id)
        if profile_name in index.get("borrowed", []):
            source = self.cog.profile_manager._borrow_source(user_id, profile_name)
            if source is None or source[1] is None:
                return None
            data = self.cog.profile_manager._get_profile_by_pid(source[0], source[1]) or {}
            return (data.get("prompts") or {}).get("birthday")
        return (self.cog.profile_manager._get_profile_prompts(user_id, profile_name) or {}).get("birthday")

    def lorebook(self, user_id: int, profile_name: str) -> List[Dict[str, Any]]:
        """This profile's own lorebook, as its editor shows it. Blocking."""
        return lorebook.entries((self.cog.profile_manager._get_profile_prompts(user_id, profile_name) or {}).get("lorebook"))

    def put_lore_entry(self, user_id: int, profile_name: str, entry_id: Optional[str],
                       keys: str, content: str) -> Tuple[bool, str]:
        """Adds an entry, or with `entry_id` rewrites that one. Blocking: run it in a thread."""
        prompts = self.cog.profile_manager._get_profile_prompts(user_id, profile_name)
        if prompts is None:
            return False, "That profile could not be found."
        book = lorebook.entries(prompts.get("lorebook"))
        entry = {"id": entry_id or lorebook.new_id(), "keys": keys, "content": content}
        if entry_id:
            if not any(e.get("id") == entry_id for e in book):
                return False, "That entry no longer exists."
            book = [entry if e.get("id") == entry_id else e for e in book]
        elif len(book) >= LIMIT_LORE_ENTRIES:
            return False, f"This lorebook is full ({LIMIT_LORE_ENTRIES} entries). Delete one first."
        else:
            book.append(entry)
        prompts["lorebook"] = lorebook.normalise(book)
        self.cog.profile_manager._save_profile_prompts(user_id, profile_name, prompts)
        return True, "Lore entry saved." if entry_id else "Lore entry added."

    def add_lore_entries(self, user_id: int, profile_name: str, new: List[Dict[str, Any]]) -> str:
        """An imported book's entries, after the profile's own: one it already holds word for
        word is skipped, and as many as fit. Only on a personal profile. Blocking."""
        if profile_name not in self.cog.profile_manager._get_user_index(user_id).get("personal", []):
            return "❌ You can only add a lorebook to a personal profile you own."
        prompts = self.cog.profile_manager._get_profile_prompts(user_id, profile_name)
        if prompts is None:
            return "❌ That profile could not be found."
        book = lorebook.entries(prompts.get("lorebook"))
        held = {e["content"] for e in book}
        fresh = [e for e in new if e["content"] not in held]
        added = fresh[:max(0, LIMIT_LORE_ENTRIES - len(book))]
        if added:
            prompts["lorebook"] = lorebook.normalise(book + added)
            self.cog.profile_manager._save_profile_prompts(user_id, profile_name, prompts)
        message = f"Added {len(added)} lore entr{'y' if len(added) == 1 else 'ies'} to **{profile_name}**."
        if len(new) > len(fresh):
            message += f" {len(new) - len(fresh)} it already had were skipped."
        if len(fresh) > len(added):
            message += f" {len(fresh) - len(added)} did not fit: a lorebook holds {LIMIT_LORE_ENTRIES}."
        return message

    def add_fixed_examples(self, user_id: int, profile_name: str, new: List[Dict[str, str]]) -> str:
        """An imported file's fixed examples, after the profile's own, as `add_lore_entries`
        adds a book: one it already holds word for word is skipped, and as many as fit. Only on
        a personal profile -- they live in its prompts, which a borrow never reads. Blocking."""
        if profile_name not in self.cog.profile_manager._get_user_index(user_id).get("personal", []):
            return "❌ You can only add fixed examples to a personal profile you own."
        prompts = self.cog.profile_manager._get_profile_prompts(user_id, profile_name)
        if prompts is None:
            return "❌ That profile could not be found."
        held = fixed_examples.entries(prompts.get("fixed_examples"))
        have = {(e.get("said", ""), e["reply"]) for e in held}
        fresh = [e for e in new if (e["said"], e["reply"]) not in have]
        added = fresh[:max(0, LIMIT_FIXED_EXAMPLES - len(held))]
        if added:
            prompts["fixed_examples"] = fixed_examples.normalise(held + added)
            self.cog.profile_manager._save_profile_prompts(user_id, profile_name, prompts)
        message = f"Added {len(added)} fixed example{'' if len(added) == 1 else 's'} to **{profile_name}**."
        if len(new) > len(fresh):
            message += f" {len(new) - len(fresh)} it already had were skipped."
        if len(fresh) > len(added):
            message += (f" {len(fresh) - len(added)} did not fit: a profile holds {LIMIT_FIXED_EXAMPLES} "
                        "fixed examples. Import the rest as training examples, sent only when a message resembles them.")
        return message

    def fixed_examples(self, user_id: int, profile_name: str) -> List[Dict[str, str]]:
        """This profile's own fixed examples, as their editor shows them. Blocking."""
        return fixed_examples.entries(
            (self.cog.profile_manager._get_profile_prompts(user_id, profile_name) or {}).get("fixed_examples"))

    def put_fixed_example(self, user_id: int, profile_name: str, entry_id: Optional[str],
                          said: str, reply: str) -> Tuple[bool, str]:
        """Adds an example, or with `entry_id` rewrites that one. Blocking: run it in a thread."""
        prompts = self.cog.profile_manager._get_profile_prompts(user_id, profile_name)
        if prompts is None:
            return False, "That profile could not be found."
        held = fixed_examples.entries(prompts.get("fixed_examples"))
        entry = {"id": entry_id or fixed_examples.new_id(), "said": said, "reply": reply}
        if entry_id:
            if not any(e.get("id") == entry_id for e in held):
                return False, "That example no longer exists."
            held = [entry if e.get("id") == entry_id else e for e in held]
        elif len(held) >= LIMIT_FIXED_EXAMPLES:
            return False, f"This profile has {LIMIT_FIXED_EXAMPLES} fixed examples, the most it can hold. Delete one first."
        else:
            held.append(entry)
        prompts["fixed_examples"] = fixed_examples.normalise(held)
        self.cog.profile_manager._save_profile_prompts(user_id, profile_name, prompts)
        return True, "Fixed example saved." if entry_id else "Fixed example added."

    def delete_fixed_examples(self, user_id: int, profile_name: str, ids: List[str]) -> int:
        """Removes the examples with these ids and returns how many went. Blocking."""
        prompts = self.cog.profile_manager._get_profile_prompts(user_id, profile_name)
        if prompts is None:
            return 0
        held = fixed_examples.entries(prompts.get("fixed_examples"))
        drop = set(ids)
        kept = [e for e in held if e.get("id") not in drop]
        if len(kept) != len(held):
            prompts["fixed_examples"] = kept
            self.cog.profile_manager._save_profile_prompts(user_id, profile_name, prompts)
        return len(held) - len(kept)

    def bonds(self, user_id: int, profile_name: str, people: Iterable[int] = ()) -> Dict[str, List[int]]:
        """What this profile feels about each person it has scored: ID -> [affection,
        resentment]. Its own, a borrow's included, as its neuro levels are.

        `people` are Discord IDs about to be looked up: a bond any of them has under the ID
        shown before v0.8.4 (`_legacy_user_hash`) moves to today's and is saved. Blocking.
        """
        data = self.cog.storage_manager._load_shard("bonds", user_id, profile_name)
        data = data if isinstance(data, dict) else {}
        moved = False
        for uid in people:
            old, new = _legacy_user_hash(uid), _get_user_hash(uid)
            if old in data and new not in data:
                data[new] = data.pop(old)
                moved = True
        if moved and self._holds_folder(user_id, profile_name):
            self.cog.storage_manager._save_shard("bonds", user_id, data, profile_name)
        return data

    def _holds_folder(self, user_id: int, profile_name: str) -> bool:
        """Whether this owner holds the profile's folder: a System profile read from someone
        else's seat has none here, and a bonds write would mint one."""
        path = self.cog.storage_manager._get_shard_path("bonds", user_id, profile_name, strict=True)
        return bool(path) and os.path.exists(os.path.join(os.path.dirname(path), "profile.json.gz"))

    def apply_bonds(self, user_id: int, profile_name: str, updates: Dict[str, Dict[str, int]]) -> None:
        """A reply's `<bond_update>`, applied and saved, only where this owner holds the
        profile's folder (`_holds_folder`). Blocking, like the neuro save beside it."""
        if not self._holds_folder(user_id, profile_name):
            return
        self.cog.storage_manager._save_shard(
            "bonds", user_id, apply_bond_updates(self.bonds(user_id, profile_name), updates), profile_name)

    def reset_bonds(self, user_id: int, profile_name: str) -> None:
        self.cog.storage_manager._delete_shard("bonds", user_id, profile_name)

    def delete_lore_entries(self, user_id: int, profile_name: str, ids: List[str]) -> int:
        """Removes the entries with these ids and returns how many went. Blocking."""
        prompts = self.cog.profile_manager._get_profile_prompts(user_id, profile_name)
        if prompts is None:
            return 0
        book = lorebook.entries(prompts.get("lorebook"))
        drop = set(ids)
        kept = [e for e in book if e.get("id") not in drop]
        if len(kept) != len(book):
            prompts["lorebook"] = kept
            self.cog.profile_manager._save_profile_prompts(user_id, profile_name, prompts)
        return len(book) - len(kept)
