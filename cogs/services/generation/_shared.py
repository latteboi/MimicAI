import re
from typing import Dict

from ...utils.constants import HarmBlockThreshold, HarmCategory, HARM_CATEGORIES
from ...utils.helpers import _scrub_response_text


def _strip_neuro_update_and_scrub(raw_text: str, participant_names) -> str:
    """Strips <neuro_update> blocks and D:/C:/O:/A: state headers before running the standard response scrub."""
    temp_clean = re.sub(r'<neuro_update>\s*(.*?)\s*</neuro_update>', '', raw_text, flags=re.IGNORECASE | re.DOTALL)
    temp_clean = re.sub(r'(?:D:\d{1,3}\s*\|\s*C:\d{1,3}\s*\|\s*O:\d{1,3}\s*\|\s*A:\d{1,3})', '', temp_clean, flags=re.IGNORECASE)
    return _scrub_response_text(temp_clean, participant_names=participant_names)


def _resolve_safety_settings(safety_level_str: str) -> Dict[HarmCategory, HarmBlockThreshold]:
    safety_map = {
        "unrestricted": HarmBlockThreshold.BLOCK_NONE,
        "low": HarmBlockThreshold.BLOCK_ONLY_HIGH,
        "medium": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        "high": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    }
    threshold = safety_map.get(safety_level_str, HarmBlockThreshold.BLOCK_ONLY_HIGH)
    return { cat: threshold for cat in HARM_CATEGORIES }
