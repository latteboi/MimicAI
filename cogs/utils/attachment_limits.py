"""The one size test for a file someone sends the bot, made before anything downloads it.

Discord states an attachment's size up front, so the test costs nothing, and it is the only
point where it can: a 500 MB video streamed to disk is safe for memory, and is still uploaded
to the provider and billed by its length. `_stream_to_tempfile` enforces the same limit while
it downloads, for media that arrives with no stated size.
"""
import os
from typing import Any, Optional

from .constants import (ATTACHMENT_SKIPPED_NOTE, DOCUMENT_MIME_TYPES, ERR_REASON_IMAGE_TOO_LARGE,
                        defaultConfig)
from .helpers import attachment_mime

_MB = 1024 * 1024
# Discord's default since 2026-09-03, for bots and webhooks too. discord.py (2.7.1 and
# master) still reports 10 MiB for an unboosted or Level 1 server.
_DISCORD_DEFAULT_UPLOAD_BYTES = 20 * _MB


def upload_limit_bytes(guild: Any) -> int:
    """The largest file the bot may post in `guild` -- the outbound limit, not the one above."""
    return max(guild.filesize_limit, _DISCORD_DEFAULT_UPLOAD_BYTES)


def refuse_oversized_upload(path: Optional[str], guild: Any) -> Optional[str]:
    """Deletes a generated image too large to post in `guild` and says why; None if it fits.

    Asked before the character sees the image, so it is told the picture failed instead of
    presenting one Discord then drops -- after a 20 MB upload -- with only a console line.
    """
    if not path:
        return None
    size = os.path.getsize(path)
    limit = upload_limit_bytes(guild) if guild is not None else _DISCORD_DEFAULT_UPLOAD_BYTES
    if size <= limit:
        return None
    os.remove(path)
    return ERR_REASON_IMAGE_TOO_LARGE.format(size=f"{size / _MB:.1f}", limit=limit // _MB)


def attachment_size(attachment: Any) -> Optional[int]:
    """Bytes, from a `discord.Attachment` or a child bot's attachment dict; None if unstated."""
    if isinstance(attachment, dict):
        size = attachment.get("size")
    else:
        size = getattr(attachment, "size", None)
    return size if isinstance(size, int) else None


def attachment_limit_bytes(attachment: Any) -> int:
    """The largest this particular file may be.

    A document gets the lower cap. It is the one attachment whose cost does not scale
    with its length -- OpenRouter bills a PDF by the page it parses, and the round hands
    the same file to every seated character -- so a size that is trivial to upload can be
    expensive to read, several times over.
    """
    if attachment_mime(attachment) in DOCUMENT_MIME_TYPES:
        return defaultConfig.LIMIT_DOCUMENT_BYTES
    return defaultConfig.LIMIT_ATTACHMENT_BYTES


def over_attachment_limit(attachment: Any) -> bool:
    size = attachment_size(attachment)
    return size is not None and size > attachment_limit_bytes(attachment)


def skipped_attachment_note(attachment: Any) -> str:
    """What a turn says in place of a file too large to read, so the character knows one
    was sent rather than answering as if nothing was."""
    if isinstance(attachment, dict):
        filename = attachment.get("filename")
    else:
        filename = getattr(attachment, "filename", None)
    return ATTACHMENT_SKIPPED_NOTE.format(
        # The cap this file was actually judged against, which is the lower one for a
        # document. Quoting 25 MB at someone whose 6 MB PDF was refused reads as a bug.
        limit=attachment_limit_bytes(attachment) // _MB,
        filename=filename or "attachment",
        size=f"{(attachment_size(attachment) or 0) / _MB:.1f}")
