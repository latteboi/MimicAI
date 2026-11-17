"""A profile's voice samples and their consent records."""

import os
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from ..utils.constants import (
    USERS_DIR, defaultConfig, VOICE_SAMPLE_CONSENT_FILES, VOICE_SAMPLE_SLOT_FILES, VOICE_SAMPLE_SLOT_KEY, VOICE_SAMPLE_SLOTS)
from .storage_manager import IOManager



class VoiceSampleManager:
    """A profile's voice samples and their consent records. Reached as `cog.voice_samples`."""

    def __init__(self, cog):
        self.cog = cog

    # --- Voice samples -------------------------------------------------------------------

    def _voice_sample_dir(self, user_id: int, profile_name: str) -> Optional[str]:
        """The profile directory holding the voice sample this profile speaks with, or None.

        A borrow speaks with its source's, found through `original_pid` -- the name it
        carries is a snapshot. A personal or System profile has its own.
        """
        index = self.cog.profile_manager._get_user_index(user_id)
        if profile_name in index.get("borrowed", []):
            src = self.cog.profile_manager._source_pointer(self.cog.profile_manager._get_profile_config(user_id, profile_name, True) or {})
            if not src:
                return None
            owner, pid = src
            return os.path.join(USERS_DIR, str(owner), "profiles", pid)
        pid = self.cog.profile_manager._get_pid_from_name(user_id, profile_name)
        if pid is None:
            return None
        owner = user_id if profile_name in index.get("personal", []) else int(defaultConfig.DISCORD_OWNER_ID)
        return os.path.join(USERS_DIR, str(owner), "profiles", pid)

    def may_set_voice_sample(self, user_id: int, profile_name: str) -> bool:
        """Only a profile's owner gives it a voice: a personal profile, or a System one for the bot owner."""
        index = self.cog.profile_manager._get_user_index(user_id)
        if profile_name in index.get("personal", []):
            return True
        return (profile_name not in index.get("borrowed", [])
                and profile_name in self.cog.profile_manager._system_index()
                and str(user_id) == str(defaultConfig.DISCORD_OWNER_ID))

    @staticmethod
    def _valid_voice_slot(value) -> Optional[int]:
        return value if isinstance(value, int) and 1 <= value <= VOICE_SAMPLE_SLOTS else None

    def voice_sample_slot(self, user_id: int, profile_name: str) -> int:
        """The slot this profile speaks with: its own choice, else a borrow's source's, else 1.

        Exactly one slot is always selected, and only it is sent; an empty one means the
        preset voice. A borrow copies its author's choice when it is made, and one made
        before the author chose follows the author's current choice rather than slot 1.
        """
        index = self.cog.profile_manager._get_user_index(user_id)
        is_borrowed = profile_name in index.get("borrowed", [])
        config = self.cog.profile_manager._get_profile_config(user_id, profile_name, is_borrowed) or {}
        slot = self._valid_voice_slot(config.get(VOICE_SAMPLE_SLOT_KEY))
        if slot is None and is_borrowed:
            src = self.cog.profile_manager._source_pointer(config)
            if src:
                source = self.cog.profile_manager._get_profile_by_pid(*src) or {}
                slot = self._valid_voice_slot((source.get("config") or {}).get(VOICE_SAMPLE_SLOT_KEY))
        return slot or 1

    async def voice_sample_summary(self, user_id: int, profile_name: str) -> Tuple[int, int, bool]:
        """(slots filled, selected slot, whether the selected one is filled), from stats of the records.

        For the dashboards, which say whether samples exist, never what they hold.
        """
        slot = self.voice_sample_slot(user_id, profile_name)
        p_dir = self._voice_sample_dir(user_id, profile_name)
        if not p_dir:
            return 0, slot, False

        def _stat():
            return [os.path.exists(os.path.join(p_dir, record)) for _audio, record in VOICE_SAMPLE_SLOT_FILES]

        filled = await asyncio.to_thread(_stat)
        return sum(filled), slot, filled[slot - 1]

    async def voice_sample_records(self, user_id: int, profile_name: str) -> List[Optional[Dict[str, Any]]]:
        """What each slot's voice sample is and who vouched for it, None for an empty slot."""
        p_dir = self._voice_sample_dir(user_id, profile_name)
        if not p_dir:
            return [None] * VOICE_SAMPLE_SLOTS
        fernet = self.cog.fernet

        def _read():
            return [IOManager.read_json_gzip(os.path.join(p_dir, record), fernet)
                    for _audio, record in VOICE_SAMPLE_SLOT_FILES]

        return await asyncio.to_thread(_read)

    async def voice_sample_record(self, user_id: int, profile_name: str) -> Optional[Dict[str, Any]]:
        """The selected slot's record, None for an empty one: one small read, for a caller that
        needs to know what the sample is -- and whether it has a consent recording -- before
        paying to decrypt it."""
        p_dir = self._voice_sample_dir(user_id, profile_name)
        if not p_dir:
            return None
        _audio, record_file = VOICE_SAMPLE_SLOT_FILES[self.voice_sample_slot(user_id, profile_name) - 1]
        return await asyncio.to_thread(IOManager.read_json_gzip, os.path.join(p_dir, record_file), self.cog.fernet)

    async def save_voice_sample(self, user_id: int, profile_name: str, audio: bytes, *, slot: int,
                                mime_type: str, filename: str, transcript: Optional[str],
                                consent: Optional[bytes] = None, consent_mime_type: Optional[str] = None,
                                consent_filename: Optional[str] = None) -> bool:
        """Stores a voice sample in `slot` with the consent given for it; False if it is not the user's to give.

        `consent` is the speaker's recorded consent, which Gemini needs to clone the voice.
        A sample saved without one drops the slot's old one: it spoke for another recording.
        """
        if not self.may_set_voice_sample(user_id, profile_name) or not self._valid_voice_slot(slot):
            return False
        p_dir = self._voice_sample_dir(user_id, profile_name)
        if not p_dir:
            return False
        record = {"mime_type": mime_type, "filename": filename, "size": len(audio),
                  "transcript": (transcript or "").strip(),
                  # Who said the voice is theirs, or theirs to use, and when.
                  "consented_by": user_id, "consented_at": int(time.time())}
        if consent:
            record.update(consent_mime_type=consent_mime_type or "audio/wav",
                          consent_filename=consent_filename or "consent")
        fernet = self.cog.fernet
        audio_file, record_file = VOICE_SAMPLE_SLOT_FILES[slot - 1]
        consent_path = os.path.join(p_dir, VOICE_SAMPLE_CONSENT_FILES[slot - 1])

        def _write():
            # The audio first: a crash between the two orphans a file, never a record that
            # names nothing.
            IOManager.write_blob(audio, os.path.join(p_dir, audio_file), fernet)
            if consent:
                IOManager.write_blob(consent, consent_path, fernet)
            else:
                try:
                    os.remove(consent_path)
                except FileNotFoundError:
                    pass
            IOManager.write_json_gzip(record, os.path.join(p_dir, record_file), fernet)

        await asyncio.to_thread(_write)
        return True

    async def delete_voice_sample(self, user_id: int, profile_name: str, slot: int) -> bool:
        if not self.may_set_voice_sample(user_id, profile_name) or not self._valid_voice_slot(slot):
            return False
        p_dir = self._voice_sample_dir(user_id, profile_name)
        if not p_dir:
            return False
        audio_file, record_file = VOICE_SAMPLE_SLOT_FILES[slot - 1]

        def _delete():
            # The record first, for the reason the audio is written first.
            for name in (record_file, audio_file, VOICE_SAMPLE_CONSENT_FILES[slot - 1]):
                try:
                    os.remove(os.path.join(p_dir, name))
                except FileNotFoundError:
                    pass

        await asyncio.to_thread(_delete)
        return True

    async def materialise_voice_sample(self, user_id: int, profile_name: str,
                                       with_consent: bool = False) -> Optional[Dict[str, Any]]:
        """The selected slot's voice sample decrypted to a temp file the caller removes, or None.

        `{"path", "mime_type", "transcript"}`, the shape the OpenRouter speech adapter sends.
        Only the selected slot, never the others beside it. Decrypted per request rather
        than kept: the plaintext of someone's voice sits in the temp directory only while
        one request takes. `with_consent` adds the consent recording as `consent_path` and
        `consent_mime_type` -- a second file the caller removes -- and is None for a slot
        that has none.
        """
        p_dir = self._voice_sample_dir(user_id, profile_name)
        if not p_dir:
            return None
        fernet = self.cog.fernet
        slot = self.voice_sample_slot(user_id, profile_name)
        audio_file, record_file = VOICE_SAMPLE_SLOT_FILES[slot - 1]

        def _load():
            import tempfile
            record = IOManager.read_json_gzip(os.path.join(p_dir, record_file), fernet)
            if not record:
                return None
            audio = IOManager.read_blob(os.path.join(p_dir, audio_file), fernet)
            if not audio:
                return None
            consent = None
            if with_consent:
                consent = record.get("consent_mime_type") and IOManager.read_blob(
                    os.path.join(p_dir, VOICE_SAMPLE_CONSENT_FILES[slot - 1]), fernet)
                if not consent:
                    return None
            sample = {"mime_type": record.get("mime_type") or "audio/wav",
                      "transcript": record.get("transcript") or ""}
            for key, data in (("path", audio), ("consent_path", consent)):
                if data:
                    fd, sample[key] = tempfile.mkstemp(suffix=".voice")
                    with os.fdopen(fd, "wb") as f:
                        f.write(data)
            if consent:
                sample["consent_mime_type"] = record["consent_mime_type"]
            return sample

        return await asyncio.to_thread(_load)
