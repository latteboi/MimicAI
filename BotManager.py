# --- MimicAI & Assistant Workflow Documentation ---
# The development workflow with the AI assistant for MimicAI is strictly modular and compliance-based.
# 1. No Interpretation: The assistant must not guess intent, refactor existing stable code, or add unsolicited features.
# 2. Strict Scope: Only the exact requested logic is to be modified. Large rewrites of functions are NOT recommended.
# 3. Literal Formatting: The exact spacing, capitalization, and naming conventions must be matched flawlessly.
# 4. Ephemeral Edit Location Reporting: Code replacements are provided in isolated blocks wrapped with precise
#    "Location of change:" identifiers (e.g., `Class.method` or specific code anchors) to allow rapid copy-pasting.
# 5. Technical Debt Avoidance: Band-aid fixes are rejected in favor of addressing the root mechanical failure.
# This workflow guarantees stable iterations without breaking the complex asynchronous state management of the bot.
# --------------------------------------------------

import os
import sys
from dotenv import load_dotenv

# Load variables from .env file if it exists
load_dotenv()

# Pre-emptively suppress DefaultConfig warnings if Manual Auth is flagged
if os.getenv("MANUAL_AUTH_MODE", "False").lower() == "true":
    if not os.getenv("DISCORD_SDK"): os.environ["DISCORD_SDK"] = "MANUAL_MODE_PENDING"
    if not os.getenv("DISCORD_OWNER_ID"): os.environ["DISCORD_OWNER_ID"] = "MANUAL_MODE_PENDING"
    if not os.getenv("ENCRYPTION_KEY"): os.environ["ENCRYPTION_KEY"] = "MANUAL_MODE_PENDING"

import configs.DefaultConfig as defaultConfig
import asyncio
import discord
from discord.ext import commands
import subprocess
import orjson as json
from cryptography.fernet import Fernet
import websockets
import signal
import platform

# --- Global State for Orchestration ---
load_dotenv()

# --- Global State for Orchestration ---
active_child_processes = {}  # Maps bot_id_str -> subprocess.Popen object
ipc_connections = {}         # Maps bot_id_str -> websocket connection
manager_queue = asyncio.Queue()
IPC_HOST = "127.0.0.1"
IPC_PORT = 8765

# --- IPC Server Logic ---
async def ipc_server_handler(websocket, path=None):
    # We only expect ONE connection now: The Hive.
    # But we keep the dict structure for compatibility with GeminiCog sending to "bot_id"
    try:
        # Wait for identification
        msg = await websocket.recv()
        data = json.loads(msg)
        
        if data.get('action') == 'identify_hive':
            print("IPC: Hive Connected.")
            
            # Ensure the main bot is logged in so we have its user ID
            await bot.wait_until_ready()
            
            # Register this SINGLE socket as the route for ALL known child bots
            # This is a bit of a hack to make GeminiCog's "send_to_child" logic work seamlessly.
            # GeminiCog sends to "12345", we need to know "12345" lives on this socket.
            # Since we only have one Hive, we can just map 'hive' -> websocket
            ipc_connections['hive'] = websocket
            
            # Launch all configured bots
            print(f"IPC: Launching {len(bot.child_bot_config)} bots into the Hive...")
            for bot_id, config in bot.child_bot_config.items():
                try:
                    token = fernet.decrypt(config['token_encrypted'].encode()).decode()
                    await websocket.send(json.dumps({
                        "action": "launch",
                        "bot_id": bot_id,
                        "token": token,
                        "parent_id": bot.user.id,
                        "parent_name": bot.user.name,
                        "owner_id": config.get("owner_id"),
                        "profile_name": config.get("profile_name"),
                        "profile_id": config.get("pid")
                    }))
                except Exception as e:
                    print(f"Failed to launch {bot_id}: {e}")

            # Listen for events from Hive
            async for message in websocket:
                event_data = json.loads(message)
                gemini_cog = bot.get_cog("GeminiAgent")
                if gemini_cog:
                    event_type = event_data.get("event_type")
                    action = event_data.get("action")

                    if event_type == "message_sent_confirmation":
                        asyncio.create_task(gemini_cog.handle_child_bot_confirmation(event_data))
                    elif action == "toggle_session_participation":
                        asyncio.create_task(gemini_cog.handle_child_bot_toggle(event_data))

    except websockets.exceptions.ConnectionClosed:
        print("IPC: Hive disconnected.")
        ipc_connections.pop('hive', None)

# --- Manager Task for Real-Time Actions ---
async def manager_task():
    print("Manager task started.")
    while True:
        command = await manager_queue.get()
        action = command.get('action')
        
        ws = ipc_connections.get('hive')

        if action == 'launch_bot':
            # New bot created at runtime
            bot_id = command.get('bot_id')
            token = command.get('token')
            config = command.get('config')
            bot.child_bot_config[bot_id] = config
            
            if ws:
                await ws.send(json.dumps({
                    "action": "launch",
                    "bot_id": bot_id,
                    "token": token,
                    "parent_id": bot.user.id,
                    "parent_name": bot.user.name,
                    "owner_id": config.get("owner_id"),
                    "profile_name": config.get("profile_name"),
                    "profile_id": config.get("pid")
                }))

        elif action == 'shutdown_bot':
            bot_id = command.get('bot_id')
            bot.child_bot_config.pop(bot_id, None)
            if ws:
                await ws.send(json.dumps({
                    "action": "shutdown",
                    "bot_id": bot_id
                }))
        
        elif action == 'send_to_child':
            # Route to Hive
            if ws:
                # Wrap the payload to tell Hive which bot it's for
                wrapper = {
                    "action": "send_to_child",
                    "bot_id": command.get('bot_id'),
                    "payload": command.get('payload')
                }
                await ws.send(json.dumps(wrapper))

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
    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    manual_auth = os.getenv("MANUAL_AUTH_MODE", "False").lower() == "true"
    
    def _clean_key(val):
        if not val: return ""
        import re
        return re.sub(r'\s+', '', str(val).replace('"', '').replace("'", ""))
    
    if manual_auth:
        print("\n--- Manual Authentication Mode ---")
        defaultConfig.DISCORD_SDK = _clean_key(input("Enter DISCORD_SDK: "))
        defaultConfig.DISCORD_OWNER_ID = _clean_key(input("Enter DISCORD_OWNER_ID: "))
        defaultConfig.ENCRYPTION_KEY = _clean_key(input("Enter ENCRYPTION_KEY: "))
        print("----------------------------------\n")
    else:
        gcp_project_id = os.getenv("GCP_PROJECT_ID")
        if gcp_project_id:
            gcp_project_id = gcp_project_id.strip()
            print(f"GCP_PROJECT_ID detected ({gcp_project_id}). Fetching keys from Google Cloud Secret Manager...")
            try:
                from google.cloud import secretmanager
                client = secretmanager.SecretManagerServiceClient()
                
                def fetch_secret(secret_name):
                    # Try exact name first, then lowercase as fallback
                    for name_variant in [secret_name, secret_name.lower()]:
                        try:
                            resource_name = f"projects/{gcp_project_id}/secrets/{name_variant}/versions/latest"
                            response = client.access_secret_version(request={"name": resource_name})
                            return response.payload.data.decode("UTF-8").strip()
                        except Exception:
                            continue
                    return None

                # Diagnostics: List available secrets if a primary fetch fails
                def debug_secrets():
                    try:
                        parent = f"projects/{gcp_project_id}"
                        print(f"DEBUG: Listing all secrets in project {gcp_project_id}...")
                        for secret in client.list_secrets(request={"parent": parent}):
                            print(f" - Found secret: {secret.name.split('/')[-1]}")
                    except Exception as e:
                        print(f"DEBUG: Could not list secrets: {e}")

                s_val = fetch_secret("DISCORD_SDK")
                o_val = fetch_secret("DISCORD_OWNER_ID")
                k_val = fetch_secret("ENCRYPTION_KEY")

                if not s_val or not o_val or not k_val:
                    print("ERROR: One or more secrets could not be found.")
                    debug_secrets()

                if not os.getenv("DISCORD_SDK"): os.environ["DISCORD_SDK"] = s_val or ""
                if not os.getenv("DISCORD_OWNER_ID"): os.environ["DISCORD_OWNER_ID"] = o_val or ""
                if not os.getenv("ENCRYPTION_KEY"): os.environ["ENCRYPTION_KEY"] = k_val or ""
                
            except ImportError:
                print("CRITICAL ERROR: 'google-cloud-secret-manager' library not found. Install via 'pip install google-cloud-secret-manager'.")
            except Exception as e:
                print(f"WARNING: GCP Secret Manager initialization failed: {e}")

        # Force re-read from environment and clean
        defaultConfig.DISCORD_SDK = _clean_key(os.getenv("DISCORD_SDK"))
        defaultConfig.DISCORD_OWNER_ID = _clean_key(os.getenv("DISCORD_OWNER_ID"))
        defaultConfig.ENCRYPTION_KEY = _clean_key(os.getenv("ENCRYPTION_KEY"))
        
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
            if lock_data.get("sdk_hash") != s_h: mismatches.append("DISCORD_SDK")
            if lock_data.get("owner_hash") != o_h: mismatches.append("DISCORD_OWNER_ID")
            if lock_data.get("key_hash") != k_h: mismatches.append("ENCRYPTION_KEY")
            
            if mismatches:
                print(f"\nCRITICAL ERROR: Authentication mismatch in: {', '.join(mismatches)}")
                print("The provided credentials do not match the original setup. Bot startup aborted to prevent data corruption.")
                return
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
        import gzip
        for user_id_str in os.listdir(users_dir):
            if not user_id_str.isdigit(): continue
            profiles_dir = os.path.join(users_dir, user_id_str, "profiles")
            if not os.path.isdir(profiles_dir): continue
            for pid_folder in os.listdir(profiles_dir):
                bot_file = os.path.join(profiles_dir, pid_folder, "child_bot.json.gz")
                if os.path.exists(bot_file):
                    try:
                        with gzip.open(bot_file, 'rb') as f:
                            bot_data = json.loads(f.read())
                        if "bot_id" in bot_data:
                            bot_data["owner_id"] = int(user_id_str)
                            bot_data["pid"] = pid_folder
                            name_file = os.path.join(profiles_dir, pid_folder, "name.txt")
                            if os.path.exists(name_file):
                                with open(name_file, 'r', encoding='utf-8') as nf:
                                    bot_data["profile_name"] = nf.read().strip()
                            bot.child_bot_config[bot_data["bot_id"]] = bot_data
                    except Exception as e:
                        print(f"Failed to load child bot for {pid_folder}: {e}")
    
    # Attach manager queue to bot object for cog access
    bot.manager_queue = manager_queue

    # Load main bot cog
    print("Loading main bot cogs...")
    await bot.load_extension("cogs.GeminiCog")
    print("GeminiCog loaded successfully.")

    # Start IPC Server and Manager Task
    ipc_server = await websockets.serve(ipc_server_handler, IPC_HOST, IPC_PORT, max_size=2**24)
    print(f"IPC Server started on ws://{IPC_HOST}:{IPC_PORT}")
    manager_task_handle = asyncio.create_task(manager_task())

    # Initial launch of child bots
    print("Launching Hive process...")
    # Use absolute path for Windows compatibility
    script_path = os.path.join(os.path.dirname(__file__), 'child_bot.py')
    proc = subprocess.Popen([sys.executable, script_path, f"ws://{IPC_HOST}:{IPC_PORT}"])
    active_child_processes['hive'] = proc

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
        print("Shutting down main bot and child processes...")
        if not ipc_server.is_serving():
            ipc_server.close()
            await ipc_server.wait_closed()
        
        manager_task_handle.cancel()
        
        for bot_id, proc in list(active_child_processes.items()):
            print(f" - Terminating child bot {bot_id}...")
            proc.terminate()
        
        # Wait for all processes to terminate
        for bot_id, proc in list(active_child_processes.items()):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"   - Child bot {bot_id} did not terminate gracefully, killing.")
                proc.kill()
        
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