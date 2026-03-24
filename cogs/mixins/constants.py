import os
from typing import Literal
import configs.DefaultConfig as defaultConfig
from google.genai.types import HarmCategory, HarmBlockThreshold

PRIMARY_MODEL_NAME = 'gemini-flash-latest'
FALLBACK_MODEL_NAME = 'gemini-flash-lite-latest'
DEFAULT_SYSTEM_INSTRUCTION = "."
DEFAULT_PROFILE_NAME = "mimic"

# Define the allowed models for the new command
ALLOWED_MODELS = Literal[
    'gemini-pro-latest', 'gemini-flash-latest', 'gemini-flash-lite-latest',
    'gemini-3.1-pro-preview', 'gemini-3.1-flash-lite-preview', 'gemini-3-flash-preview', 'gemini-2.5-flash-preview-09-2025', 'gemini-2.5-flash-lite-preview-09-2025',
    'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite',
    'gemini-2.0-flash', 'gemini-2.0-flash-lite'
]

COGS_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(COGS_BASE, "data")

SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")
SESSIONS_GLOBAL_DIR = os.path.join(SESSIONS_DIR, "global_chat")
SESSIONS_SERVERS_DIR = os.path.join(SESSIONS_DIR, "servers") # Legacy path kept for migration

SERVERS_DIR = os.path.join(DATA_DIR, "servers")
FREEWILL_SERVERS_DIR = SERVERS_DIR # Alias for backwards compatibility during transition

LEGACY_PROFILES_DIR = os.path.join(DATA_DIR, "profiles")
LEGACY_LTM_DIR = os.path.join(DATA_DIR, "ltm")
LEGACY_TRAINING_DIR = os.path.join(DATA_DIR, "training")
LEGACY_CHILD_BOTS_DIR = os.path.join(DATA_DIR, "child_bots")
LEGACY_GLOBAL_CHAT_DIR = os.path.join(SESSIONS_DIR, "global_chat")

PUBLIC_PROFILES_DIR = os.path.join(DATA_DIR, "public_profiles")
FREEWILL_SERVERS_DIR = os.path.join(DATA_DIR, "servers")
CHILD_BOTS_DIR = os.path.join(DATA_DIR, "child_bots")
COG_LOCK_FILE_PATH = os.path.join(os.path.dirname(__file__), "gemini_agent.lock")  
USERS_DIR = os.path.join(DATA_DIR, "users")
APPEARANCES_DIR = os.path.join(USERS_DIR, "appearances") # Retained temporarily for Phase 3 migration
BLACKLIST_FILE_PATH = os.path.join(USERS_DIR, "blacklist.json")

LEGACY_SHARES_DIR = os.path.join(USERS_DIR, "shares")
LEGACY_PERSONAL_KEYS_DIR = os.path.join(USERS_DIR, "personal_keys")

COG_LOCK_FILE_PATH = os.path.join(COGS_BASE, "gemini_agent.lock")

EMBEDDING_MODEL_NAME = 'models/gemini-embedding-001'
DISCORD_MAX_MESSAGE_LENGTH = 2000
PLEASE_TRY_AGAIN_ERROR_MESSAGE = 'There was an issue with your question please try again...'
MAX_LTM_COUNT_PER_PROFILE_CONTEXT = 1000
STM_LIMIT_MAX = 50
LTM_INJECTION_PROBABILITY = 1
LTM_CREATION_INTERVAL = 10
MIN_HISTORY_FOR_LTM_CREATION = 6
MAX_TRAINING_EXAMPLES_PER_PROFILE = 50
PERSONA_TEXT_INPUT_MAX_LENGTH = 4000
AI_INSTRUCTIONS_PART_MAX_LENGTH = 4000
PLACEHOLDER_EMOJI = defaultConfig.PLACEHOLDER_EMOJI
LOCK_STALE_THRESHOLD_SECONDS = 60 
LOCK_REFRESH_INTERVAL_SECONDS = 30 
CHAT_SESSION_CACHE_MAX_SIZE = 5 
PROMPT_CACHE_MAX_SIZE = 20
MAX_USER_PROFILES = 50
MAX_BORROWED_PROFILES = 50
MAX_USER_APPEARANCES = 50
MAX_MULTI_PROFILES = 10
DROPDOWN_MAX_OPTIONS = 25
SHARE_CODE_EXPIRATION_SECONDS = 300
MAX_URL_CONTEXT_CHARACTERS = 16000 # Approx 4000 tokens
MAX_GROUNDING_SUMMARY_CHARACTERS = 2000 # Approx 500 tokens

REGENERATE_EMOJI = "🔁"
NEXT_SPEAKER_EMOJI = "⏯️"
CONTINUE_ROUND_EMOJI = "🍿"
MUTE_TURN_EMOJI = ["🔇", "🔕"]
SKIP_PARTICIPANT_EMOJI = ["❌", "✖️"]

DEFAULT_LTM_SUMMARIZATION_INSTRUCTIONS = (
    "You are a memory consolidation AI. Your task is to analyze a conversation excerpt and create a concise, third-person summary of the most important information to be stored as a long-term memory.\n\n"
    "Focus on capturing the following:\n"
    "- Key facts, revelations, or personal details shared by participants.\n"
    "- Significant events, decisions made, or future plans established.\n"
    "- The evolution of relationships (e.g., agreements, disagreements, alliances).\n"
    "- Strongly stated opinions, preferences, likes, or dislikes.\n\n"
    "What to exclude:\n"
    "- Do not include generic greetings, farewells, or conversational filler (e.g., 'hello', 'how are you').\n"
    "- Ignore simple questions that were immediately answered; focus on the resulting information.\n\n"
    "CRITICAL CONSTRAINTS:\n"
    "- The summary MUST be written in the third person.\n"
    "- The summary MUST explicitly identify the participants by name for every action or detail recorded.\n"
    "- The summary MUST be under 500 characters.\n"
    "- If the excerpt contains no new, meaningful information worth remembering, respond ONLY with the text 'NO_SUMMARY'."
)

DEFAULT_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

class GoogleGenAIChatSession:
    def __init__(self, history=None):
        self.history = history or []

# --- Standardised Error & Warning Messages ---
WARN_FALLBACK_USED = "**Fallback Model Used**"
WARN_MAIN_MODEL_FAILED = "**Main Model Failed** ({reason})"
WARN_BOTH_MODELS_FAILED = "**Main & Fallback Model Failed** ({reason})"
WARN_VOICE_SYNTHESIS_FAILED = "**Text-To-Speech Failed** ({reason})"
WARN_URL_FETCHING_FAILED = "**URL Fetching Failed** ({reason})"
WARN_GROUNDING_FAILED = "**Grounding Failed** ({reason})"
WARN_IMAGE_GEN_FAILED = "**Image Generation Failed** ({reason})"

ERR_GENERAL_ERROR = "An error has occurred."
ERR_SAFETY_BLOCK = "Safety Filter ({reason})"
ERR_RATE_LIMIT = "API Rate Limit"
ERR_UNKNOWN = "Unknown Error"

ERR_REASON_UNSUPPORTED_IMAGE = "Images Unsupported"
ERR_REASON_UNSUPPORTED_AUDIO = "Audio Unsupported"
ERR_REASON_UNSUPPORTED_VIDEO = "Video Unsupported"
ERR_REASON_EMPTY_RESPONSE = "AI produced no text content"
ERR_REASON_REPETITIVE_CONTENT = "Model Collapse"
ERR_REASON_PROVIDER_ERROR = "Provider Error"
ERR_REASON_TIMEOUT_MAIN = "Timed-out"
ERR_REASON_TIMEOUT_FALLBACK = "Fallback Timed-out"
ERR_REASON_TIMEOUT_BOTH = "Timed-out"