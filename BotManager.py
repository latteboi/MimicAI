# Allocator tuning runs before anything else is imported: M_ARENA_MAX only governs
# arenas that do not exist yet, and pinning M_MMAP_THRESHOLD early keeps every
# large buffer the bot ever allocates on the mmap path, where freeing it actually
# returns the pages. See cogs/utils/memory_tuning.py for why this is needed at all.
# No-op off glibc.
from cogs.utils.memory_tuning import tune_allocator

tune_allocator()

import asyncio
import discord
from discord.ext import commands
import orjson as json
from cryptography.fernet import Fernet
import signal
import platform
import os
import sys
import faulthandler
from dotenv import load_dotenv

# Dump a Python traceback to stderr if the interpreter takes a fatal signal
# (SIGSEGV/SIGBUS/SIGFPE/SIGABRT). A native crash inside a C extension otherwise
# leaves nothing in the journal but "code=dumped, status=11/SEGV", which does not
# say which call was in flight. Costs nothing at runtime; stderr is captured by
# systemd, so the traceback lands in `journalctl -u mimicai`.
faulthandler.enable()

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
# max_messages=None disables discord.py's cache of the last 1000 Message objects.
# Nothing here reads it: every delete/edit listener uses the on_raw_* variants, which
# do not consult the cache, and message.reference.resolved is built by discord.py from
# the gateway payload's `referenced_message` field rather than from the cache. Left on,
# the deque fills with full Message objects -- author, embeds, attachments, reactions --
# and never shrinks, which on a 24/7 e2-micro is the one baseline term that grows.
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None, max_messages=None)

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

    # Child bot configs are NOT preloaded here any more. This block used to open,
    # Fernet-decrypt and zstd-decompress every profile.json.gz on disk to build
    # bot.child_bot_config -- a full scan of every profile, before the cog even loads.
    # ChildBotManager._load_child_bots() already derives the same mapping into
    # cog.child_bots, and that was the only mapping anything actually launched from;
    # bot.child_bot_config had exactly one reader (the shutdown confirmation view),
    # which now reads cog.child_bots instead.
    #
    # The legacy child_bot.json.gz fallback went with it. _load_child_bots never
    # honoured that format, so a legacy child bot could be listed here but never
    # launched -- dropping it removes a scan, not a capability.

    # Cap the executor that backs every asyncio.to_thread call. The default is
    # min(32, cpu_count + 4), which is 5 here and buys nothing on a 0.25 vCPU
    # baseline -- the work sent there is IOManager's Fernet+zstd pipeline, which
    # is GIL-bound anyway. Each worker thread costs an 8 MB stack reservation, its
    # own glibc arena (so its own fragmentation high-water mark), and its own
    # thread-local ZstdCompressor/ZstdDecompressor pair, which are never released.
    # Two is enough to keep a read overlapping a write.
    from concurrent.futures import ThreadPoolExecutor

    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=2, thread_name_prefix="mimic-io")
    )

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