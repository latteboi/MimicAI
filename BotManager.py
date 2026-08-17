import asyncio
import discord
from discord.ext import commands
import orjson as json
from cryptography.fernet import Fernet
import signal
import platform
import os
import sys
from dotenv import load_dotenv

# Load variables from .env file if it exists
load_dotenv()

# Pre-emptively suppress warnings if Manual Auth is flagged
if os.getenv("MANUAL_AUTH_MODE", "False").lower() == "true":
    if not os.getenv("DISCORD_SDK"): os.environ["DISCORD_SDK"] = "MANUAL_MODE_PENDING"
    if not os.getenv("DISCORD_OWNER_ID"): os.environ["DISCORD_OWNER_ID"] = "MANUAL_MODE_PENDING"
    if not os.getenv("ENCRYPTION_KEY"): os.environ["ENCRYPTION_KEY"] = "MANUAL_MODE_PENDING"

from cogs.utils.constants import defaultConfig

# --- Global Queue for Cog Dispatch ---
manager_queue = asyncio.Queue()

# --- Intents Setup ---
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.guilds = True

# --- Bot Initialization ---
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

# --- Event: Bot Ready ---
@bot.event
async def on_ready():
    print(f"Bot is online as {bot.user.name} (ID: {bot.user.id})")
    print("Attempting to sync application (slash) commands...")
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} application commands globally.")
    except Exception as e:
        print(f"Failed to sync application commands: {e}")
    print("Bot setup complete and ready for commands.")

# --- Main asynchronous function to setup and run the bot ---
async def main():
    import hashlib
    
    manual_auth = os.getenv("MANUAL_AUTH_MODE", "False").lower() == "true"
    
    def _clean_key(val):
        if not val: return ""
        import re
        if isinstance(val, bytes):
            val = val.decode('utf-8')
        return re.sub(r'\s+', '', str(val).replace('"', '').replace("'", ""))
    
    if manual_auth:
        print("\n--- Manual Authentication Mode ---")
        defaultConfig.DISCORD_SDK = _clean_key(input("Enter DISCORD_SDK: "))
        defaultConfig.DISCORD_OWNER_ID = _clean_key(input("Enter DISCORD_OWNER_ID: "))
        defaultConfig.ENCRYPTION_KEY = _clean_key(input("Enter ENCRYPTION_KEY: "))
        print("----------------------------------\n")
    else:
        defaultConfig.DISCORD_SDK = _clean_key(defaultConfig.DISCORD_SDK)
        defaultConfig.DISCORD_OWNER_ID = _clean_key(defaultConfig.DISCORD_OWNER_ID)
        defaultConfig.ENCRYPTION_KEY = _clean_key(defaultConfig.ENCRYPTION_KEY)
        
        if not defaultConfig.DISCORD_SDK or not defaultConfig.ENCRYPTION_KEY:
            print("CRITICAL ERROR: MANUAL_AUTH_MODE is False, but your keys are missing from the .env file and GCP Secret Manager.")
            return

    # System Lock Verification
    lock_path = os.path.join(os.path.dirname(__file__), "cogs", "data", "system_lock.json")
    if os.path.exists(lock_path):
        try:
            with open(lock_path, "rb") as f:
                lock_data = json.loads(f.read())
            
            s_h = hashlib.sha256(defaultConfig.DISCORD_SDK.encode()).hexdigest()
            o_h = hashlib.sha256(defaultConfig.DISCORD_OWNER_ID.encode()).hexdigest()
            k_h = hashlib.sha256(defaultConfig.ENCRYPTION_KEY.encode()).hexdigest()
            
            mismatches = []
            # We no longer strictly lock the DISCORD_SDK token, as it doesn't encrypt data and can be reset.
            if lock_data.get("owner_hash") != o_h: mismatches.append("DISCORD_OWNER_ID")
            if lock_data.get("key_hash") != k_h: mismatches.append("ENCRYPTION_KEY")
            
            if mismatches:
                print(f"\nCRITICAL ERROR: Authentication mismatch in: {', '.join(mismatches)}")
                print("The provided credentials do not match the original setup. Bot startup aborted to prevent data corruption.")
                return
                
            # If only the SDK token changed, update the lock file silently
            if lock_data.get("sdk_hash") != s_h:
                lock_data["sdk_hash"] = s_h
                try:
                    with open(lock_path, "wb") as f:
                        f.write(json.dumps(lock_data))
                except Exception as e:
                    print(f"Warning: Failed to update system lock file with new SDK hash: {e}")
                    
        except Exception as e:
            print(f"\nCRITICAL ERROR: Failed to read system lock file: {e}")
            return
    else:
        # Create lock if missing (first run recovery)
        try:
            lock_data = {
                "sdk_hash": hashlib.sha256(defaultConfig.DISCORD_SDK.encode()).hexdigest(),
                "owner_hash": hashlib.sha256(defaultConfig.DISCORD_OWNER_ID.encode()).hexdigest(),
                "key_hash": hashlib.sha256(defaultConfig.ENCRYPTION_KEY.encode()).hexdigest()
            }
            os.makedirs(os.path.dirname(lock_path), exist_ok=True)
            with open(lock_path, "wb") as f:
                f.write(json.dumps(lock_data))
        except Exception as e:
            print(f"\nCRITICAL ERROR: Failed to create system lock file: {e}")
            return

    # Load encryption key
    global fernet
    try:
        fernet = Fernet(defaultConfig.ENCRYPTION_KEY)
    except Exception as e:
        print(f"\nCRITICAL ERROR: Failed to load encryption key: {e}")
        return

    # Load child bot configurations
    print("Loading child bot configurations...")
    users_dir = os.path.join(os.path.dirname(__file__), "cogs", "data", "users")
    bot.child_bot_config = {}
    if os.path.isdir(users_dir):
        from cogs.managers.storage_manager import IOManager
        for user_id_str in os.listdir(users_dir):
            if not user_id_str.isdigit(): continue
            profiles_dir = os.path.join(users_dir, user_id_str, "profiles")
            if not os.path.isdir(profiles_dir): continue
            for pid_folder in os.listdir(profiles_dir):
                bot_file = os.path.join(profiles_dir, pid_folder, "child_bot.json.gz")
                if os.path.exists(bot_file):
                    bot_data = IOManager.read_json_gzip(bot_file, encrypted=False)
                    if bot_data and "bot_id" in bot_data:
                        bot_data["owner_id"] = int(user_id_str)
                        bot_data["pid"] = pid_folder
                        name_file = os.path.join(profiles_dir, pid_folder, "name.txt")
                        if os.path.exists(name_file):
                            with open(name_file, 'r', encoding='utf-8') as nf:
                                bot_data["profile_name"] = nf.read().strip()
                        bot.child_bot_config[bot_data["bot_id"]] = bot_data
    
    # Attach manager queue to bot object for cog access
    bot.manager_queue = manager_queue

    # Load main bot cog
    print("Loading main bot cogs...")
    await bot.load_extension("cogs.MimicCog")
    print("MimicCog loaded successfully.")

    # Start the main bot
    async def runner():
        async with bot:
            await bot.start(defaultConfig.DISCORD_SDK)

    # Handle graceful shutdown on signals
    if platform.system() != "Windows":
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(bot.close()))

    try:
        print("Starting main bot...")
        await runner()
    finally:
        # Graceful shutdown
        print("Shutting down bot instance...")
        mimic_cog = bot.get_cog("MimicCog")
        if mimic_cog and hasattr(mimic_cog, "child_bot_manager"):
            await mimic_cog.child_bot_manager.shutdown_all()
        print("All processes terminated. Exiting.")

# --- Run the bot ---
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Initiating shutdown.")
    except Exception:
        import traceback
        print(f"An error occurred during bot operation:")
        traceback.print_exc()