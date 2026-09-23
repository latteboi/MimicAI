import re

from ...utils.helpers import _scrub_response_text

#: The neuro engine's report, as the prompt asks for it, and bare, as a model that drops
#: the tag writes it. One pair for the parser in `_neuro_state_from_text` and the scrub
#: below: a marker one of them recognises and the other does not is either state lost or
#: numbers said out loud.
NEURO_TAG_PATTERN = re.compile(r'<neuro_update>\s*(.*?)\s*</neuro_update>', re.IGNORECASE | re.DOTALL)
NEURO_BARE_PATTERN = re.compile(r'(?:D:\d{1,3}\s*\|\s*C:\d{1,3}\s*\|\s*O:\d{1,3}\s*\|\s*A:\d{1,3})',
                                re.IGNORECASE)


def _strip_neuro_update_and_scrub(raw_text: str, participant_names) -> str:
    """Strips <neuro_update> blocks and D:/C:/O:/A: state headers before running the standard response scrub."""
    temp_clean = NEURO_TAG_PATTERN.sub('', raw_text)
    temp_clean = NEURO_BARE_PATTERN.sub('', temp_clean)
    return _scrub_response_text(temp_clean, participant_names=participant_names)
