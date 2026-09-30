import re
from typing import Tuple

from ...utils.helpers import _scrub_response_text

#: The neuro engine's report, as the prompt asks for it, and bare, as a model that drops
#: the tag writes it. One pair for the parser in `_neuro_state_from_text` and the scrub
#: below: a marker one of them recognises and the other does not is either state lost or
#: numbers said out loud.
NEURO_TAG_PATTERN = re.compile(r'<neuro_update>\s*(.*?)\s*</neuro_update>', re.IGNORECASE | re.DOTALL)
NEURO_BARE_PATTERN = re.compile(r'(?:D:\d{1,3}\s*\|\s*C:\d{1,3}\s*\|\s*O:\d{1,3}\s*\|\s*A:\d{1,3})',
                                re.IGNORECASE)


#: The marker a character ends a reply with to have a memory written (`ltm_flag_enabled`).
#: The bare and the paired spellings both, since a model told `<memory_flag/>` writes any
#: of the three; `_scrub_response_text` cannot take this one out itself, because its tag
#: patterns want a `>` or whitespace straight after the name and `/` is neither.
MEMORY_FLAG_PATTERN = re.compile(r'<memory_flag\s*/?>(?:\s*</memory_flag>)?', re.IGNORECASE)


def take_memory_flag(raw_text: str) -> Tuple[str, bool]:
    """The reply without any `<memory_flag/>`, and whether it had one.

    Always removed, honoured or not: the tag is a protocol marker and must never reach
    the channel or the log, whichever path the reply came down.
    """
    clean = MEMORY_FLAG_PATTERN.sub('', raw_text)
    return (clean.strip(), True) if clean != raw_text else (raw_text, False)


def _strip_neuro_update_and_scrub(raw_text: str, participant_names) -> str:
    """Strips <neuro_update> blocks, D:/C:/O:/A: state headers and a memory flag before running the standard response scrub."""
    temp_clean = NEURO_TAG_PATTERN.sub('', raw_text)
    temp_clean = NEURO_BARE_PATTERN.sub('', temp_clean)
    temp_clean = MEMORY_FLAG_PATTERN.sub('', temp_clean)
    return _scrub_response_text(temp_clean, participant_names=participant_names)
