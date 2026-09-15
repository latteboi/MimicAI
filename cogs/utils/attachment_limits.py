"""The one size test for a file someone sends the bot, made before anything downloads it.

Discord states an attachment's size up front, so the test costs nothing, and it is the only
point where it can: a 500 MB video streamed to disk is safe for memory, and is still uploaded
to the provider and billed by its length. `_stream_to_tempfile` enforces the same limit while
it downloads, for media that arrives with no stated size.
"""
from typing import Any, Optional

from .constants import ATTACHMENT_SKIPPED_NOTE, defaultConfig

_MB = 1024 * 1024


def attachment_size(attachment: Any) -> Optional[int]:
    """Bytes, from a `discord.Attachment` or a child bot's attachment dict; None if unstated."""
    if isinstance(attachment, dict):
        size = attachment.get("size")
    else:
        size = getattr(attachment, "size", None)
    return size if isinstance(size, int) else None


def over_attachment_limit(attachment: Any) -> bool:
    size = attachment_size(attachment)
    return size is not None and size > defaultConfig.LIMIT_ATTACHMENT_BYTES


def skipped_attachment_note(attachment: Any) -> str:
    """What a turn says in place of a file too large to read, so the character knows one
    was sent rather than answering as if nothing was."""
    if isinstance(attachment, dict):
        filename = attachment.get("filename")
    else:
        filename = getattr(attachment, "filename", None)
    return ATTACHMENT_SKIPPED_NOTE.format(
        limit=defaultConfig.LIMIT_ATTACHMENT_BYTES // _MB,
        filename=filename or "attachment",
        size=f"{(attachment_size(attachment) or 0) / _MB:.1f}")
