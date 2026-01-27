import configs.DefaultConfig as defaultConfig
import asyncio
import discord
from discord.ext import commands
import subprocess
import orjson as json
import os
import sys
from cryptography.fernet import Fernet
import websockets
import signal
import platform
from dotenv import load_dotenv

# Load variables from .env file if it exists
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
                        "parent_id": bot.user.id
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
                    "parent_id": bot.user.id
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
    # Load encryption key
    global fernet
    if defaultConfig.ENCRYPTION_KEY:
        fernet = Fernet(defaultConfig.ENCRYPTION_KEY)
    else:
        print("CRITICAL: Encryption key not found. Child bot system will not function.")
        return

    # Load child bot configurations
    print("Loading child bot configurations...")
    child_bots_dir = os.path.join(os.path.dirname(__file__), "cogs", "data", "child_bots")
    bot.child_bot_config = {}
    if os.path.isdir(child_bots_dir):
        import gzip
        for filename in os.listdir(child_bots_dir):
            if filename.endswith(".json.gz"):
                file_path = os.path.join(child_bots_dir, filename)
                owner_id = filename[:-len(".json.gz")]
                with gzip.open(file_path, 'rb') as f:
                    user_bot_data = json.loads(f.read())
                for bot_id, data in user_bot_data.items():
                    data['owner_id'] = int(owner_id)
                    bot.child_bot_config[bot_id] = data
    
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