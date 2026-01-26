import os
from cryptography.fernet import Fernet
from dotenv import load_dotenv

# Ensure environment is loaded
load_dotenv()

# --- Core Bot Credentials ---
DISCORD_SDK = os.getenv("DISCORD_SDK")
DISCORD_OWNER_ID = os.getenv("DISCORD_OWNER_ID")

# --- Self-Hosted Configuration ---
ALL_USERS_PREMIUM = True
# Custom indicator for when the bot is processing a response
PLACEHOLDER_EMOJI = os.getenv("PLACEHOLDER_EMOJI", "⏳")

# --- Encryption Key Handling ---
# CRITICAL: You must set this to your old key to keep your existing data!
ENCRYPTION_KEY_RAW = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY_RAW:
    print("WARNING: No ENCRYPTION_KEY found in environment. Generating a temporary session key (Old data will not be readable).")
    ENCRYPTION_KEY = Fernet.generate_key()
else:
    # Handle both string and byte formats from environment
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
    print("CRITICAL ERROR: DISCORD_SDK environment variable is missing.")
if not DISCORD_OWNER_ID:
    print("CRITICAL ERROR: DISCORD_OWNER_ID environment variable is missing.")