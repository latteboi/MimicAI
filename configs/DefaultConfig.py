import os
from cryptography.fernet import Fernet
from dotenv import load_dotenv

# Load variables from .env file if it exists
load_dotenv()

# --- Secret Retrieval Logic (Hybrid) ---
def get_config_value(key_name: str, default: str = None) -> str | None:
    """Checks environment variables first, then falls back to Google Secret Manager."""
    # 1. Check local environment / .env
    val = os.getenv(key_name)
    if val:
        return val

    # 2. Fallback to Google Secret Manager
    try:
        from google.cloud import secretmanager
        from google.api_core.exceptions import NotFound, GoogleAPICallError
        
        project_id = os.getenv('GCP_PROJECT_ID')
        if not project_id:
            return default
            
        client = secretmanager.SecretManagerServiceClient()
        # Secret Manager uses lowercase/underscores typically; env uses uppercase.
        # We try both to be flexible.
        for name in [key_name.lower(), key_name.upper()]:
            resource_name = f"projects/{project_id}/secrets/{name}/versions/latest"
            try:
                response = client.access_secret_version(request={"name": resource_name})
                return response.payload.data.decode("UTF-8")
            except (NotFound, GoogleAPICallError):
                continue
    except ImportError:
        pass # Google Cloud library not installed

    return default

# --- Core Bot Credentials ---
DISCORD_SDK = get_config_value("DISCORD_SDK")
DISCORD_OWNER_ID = get_config_value("DISCORD_OWNER_ID")

# --- Self-Hosted Configuration ---
ALL_USERS_PREMIUM = True
PLACEHOLDER_EMOJI = get_config_value("PLACEHOLDER_EMOJI", "⏳")

# --- Encryption Key Handling ---
ENCRYPTION_KEY_RAW = get_config_value("ENCRYPTION_KEY")
if not ENCRYPTION_KEY_RAW:
    print("WARNING: No ENCRYPTION_KEY found. Generating a temporary session key.")
    ENCRYPTION_KEY = Fernet.generate_key()
else:
    key_val = ENCRYPTION_KEY_RAW.strip()
    ENCRYPTION_KEY = key_val.encode() if isinstance(key_val, str) else key_val

# --- Feature Limits (Self-Hosted Defaults) ---
LIMIT_PROFILES_FREE = 5
LIMIT_PROFILES_PREMIUM = 100

LIMIT_BORROWED_FREE = 5
LIMIT_BORROWED_PREMIUM = 100

LIMIT_LTM_FREE = 50
LIMIT_LTM_PREMIUM = 5000

LIMIT_TRAINING_FREE = 10
LIMIT_TRAINING_PREMIUM = 100

# --- Global AI Parameters ---
CHATBOT_MEMORY_LENGTH = 20
GEMINI_TEMPERATURE = 1.0
GEMINI_TOP_P = 0.95
GEMINI_TOP_K = 0
TRAINING_CONTEXT_SIZE = 5
TRAINING_RELEVANCE_THRESHOLD = 0.1

# --- Hub News ---
MIMIC_NEWS = "MimicAI Self-Hosted Edition."

# --- Safety Checks ---
if not DISCORD_SDK:
    print("CRITICAL ERROR: DISCORD_SDK is missing from Env and Secret Manager.")
if not DISCORD_OWNER_ID:
    print("CRITICAL ERROR: DISCORD_OWNER_ID is missing from Env and Secret Manager.")