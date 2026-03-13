import os
import subprocess
import sys
import venv
import platform

def get_venv_python():
    """Returns the path to the python executable within the venv based on OS."""
    if platform.system() == "Windows":
        return os.path.join(".venv", "Scripts", "python.exe")
    return os.path.join(".venv", "bin", "python")

def run_pip(python_exe, command):
    """Runs pip using the specified python executable."""
    try:
        subprocess.check_call([python_exe, "-m", "pip", "install"] + command)
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def setup_mimic():
    print("--- MimicAI Self-Hosted Setup ---")

    print("\n[0/5] Configuration Mode")
    print("Manual Authentication Mode stores session keys in volatile memory only.")
    print("It requires manual STDIN injection on every boot.")
    print("Do NOT use unless you understand the service implications.")
    manual_auth = input("Enable Manual Authentication Mode? [y/N]: ").strip().lower() == 'y'

    # 0. Linux-specific preparation
    if platform.system() == "Linux":
        print("\n[1/5] Checking Linux system dependencies...")
        # Note: venv is now imported at the top level
        
        # Check for build-essential if the user is on a very stripped instance
        if subprocess.run(["which", "gcc"], capture_output=True).returncode != 0:
            print(" ! Warning: GCC compiler not found. Some libraries may fail to install.")
            print(" ! Recommendation: sudo apt install build-essential -y")

    # 1. Create Virtual Environment
    venv_dir = ".venv"
    # Determine the expected activation script path to verify health
    activate_script = os.path.join(venv_dir, "Scripts", "activate") if platform.system() == "Windows" else os.path.join(venv_dir, "bin", "activate")
    
    if os.path.exists(venv_dir) and not os.path.exists(activate_script):
        print(f"\n[2/5] Found broken virtual environment folder. Deleting and recreating...")
        import shutil
        try:
            shutil.rmtree(venv_dir, ignore_errors=True)
        except Exception as e:
            print(f" ! Error cleaning up broken environment: {e}")
            sys.exit(1)

    if not os.path.exists(venv_dir):
        print("\n[2/5] Creating local virtual environment (.venv)...")
        venv.create(venv_dir, with_pip=True)
        print(" - Virtual environment created.")
    else:
        print("\n[2/5] Virtual environment already exists. Skipping creation.")

    venv_python = get_venv_python()

    # Verify pip is actually functional
    try:
        subprocess.check_call([venv_python, "-m", "pip", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(" - Pip is missing inside the virtual environment. Attempting to repair...")
        try:
            subprocess.check_call([venv_python, "-m", "ensurepip", "--upgrade"])
            print(" - Pip restored successfully.")
        except subprocess.CalledProcessError:
            print(" ! Critical Error: Failed to install pip automatically.")
            print(" ! Please delete the .venv folder and run: sudo apt install python3-pip python3-venv -y")
            sys.exit(1)

    # 2. Install Dependencies inside Venv
    print("\n[3/5] Installing required libraries into the environment...")
    # Standard dependencies list
    deps = ["discord.py", "google-genai", "google-generativeai", "websockets", "orjson", "cryptography", "aiohttp", "httpx", "numpy", "python-dotenv", "Pillow", "tzdata"]
    
    if os.path.exists("requirements.txt"):
        run_pip(venv_python, ["-r", "requirements.txt"])
    else:
        run_pip(venv_python, deps)

    # 3. Create Directory Structure
    print("\n[4/5] Creating data directories...")
    directories = [
        "cogs/data/users",
        "cogs/data/servers",
        "cogs/data/public_profiles",
        "cogs/data/models"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f" - Created: {directory}")

    # 4. Handle Environment Variables
    print("\n[5/5] Configuring environment...")
    
    def _clean_key(val):
        if not val: return ""
        import re
        # Aggressively strip all quotes, spaces, tabs, and newlines
        return re.sub(r'\s+', '', str(val).replace('"', '').replace("'", ""))

    if not os.path.exists(".env"):
        # We run the key generation via the venv python to ensure cryptography is available
        print(" - Generating secure encryption key...")
        gen_key_code = "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
        fernet_key = _clean_key(subprocess.check_output([venv_python, "-c", gen_key_code]).decode())
        
        sdk_token = _clean_key(input("Enter your Discord Bot Token (SDK): "))
        owner_id = _clean_key(input("Enter your Discord User ID (Owner): "))
        
        if manual_auth:
            print("\n" + "!"*70)
            print("MANUAL AUTHENTICATION MODE ACTIVE")
            print("Your generated Fernet Encryption Key is:")
            print(f"\n{fernet_key}\n")
            print("SAVE THIS KEY NOW IN A SECURE LOCATION.")
            print("It will NOT be saved to disk. If you lose it, your data is permanently corrupted.")
            print("!"*70)
            input("\nPress Enter to confirm you have safely stored the key...")
        
        import hashlib
        import json
        lock_data = {
            "sdk_hash": hashlib.sha256(sdk_token.encode()).hexdigest(),
            "owner_hash": hashlib.sha256(owner_id.encode()).hexdigest(),
            "key_hash": hashlib.sha256(fernet_key.encode()).hexdigest()
        }
        os.makedirs("cogs/data", exist_ok=True)
        with open("cogs/data/system_lock.json", "w") as f:
            json.dump(lock_data, f)
        
        env_content = (
            f"PLACEHOLDER_EMOJI=⌛\n"
            f"MANUAL_AUTH_MODE={'True' if manual_auth else 'False'}\n"
        )
        
        if not manual_auth:
            env_content += (
                f"DISCORD_SDK={sdk_token}\n"
                f"DISCORD_OWNER_ID={owner_id}\n"
                f"ENCRYPTION_KEY={fernet_key}\n"
            )
            
        with open(".env", "w", encoding="utf-8") as f:
            f.write(env_content)
        print("\nSUCCESS: .env file and system lock created.")
    else:
        print("\nINFO: .env file already exists. Configuration skipped to protect existing keys.")

    # 5. Linux Service Configuration (systemd)
    if platform.system() == "Linux":
        choice = input("\nDo you want to install MimicAI as a system service? (Auto-start on boot) [y/N]: ").strip().lower()
        if choice == 'y':
            import getpass
            username = getpass.getuser()
            current_dir = os.getcwd()
            venv_python_abs = os.path.abspath(venv_python)
            service_name = "mimicai.service"
            service_path = f"/etc/systemd/system/{service_name}"
            
            service_content = f"""[Unit]
Description=MimicAI Self-Hosted Bot
After=network.target

[Service]
Type=simple
User={username}
WorkingDirectory={current_dir}
ExecStart={venv_python_abs} BotManager.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
            try:
                # Write locally first
                with open(service_name, "w") as f:
                    f.write(service_content)
                
                print("\n - Requesting permissions to install service...")
                # Chain commands with sudo
                cmds = [
                    ["sudo", "mv", os.path.join(current_dir, service_name), service_path],
                    ["sudo", "chown", "root:root", service_path],
                    ["sudo", "systemctl", "daemon-reload"],
                    ["sudo", "systemctl", "enable", service_name],
                    ["sudo", "systemctl", "start", service_name]
                ]
                
                for cmd in cmds:
                    subprocess.check_call(cmd)
                
                print(f"SUCCESS: MimicAI service is active and set to run on boot.")
                print(f" - View logs: journalctl -u {service_name} -f")
            except subprocess.CalledProcessError:
                print(f"\n ! Automation failed (likely password rejected). Please run manually:")
                print(f"   sudo mv {current_dir}/{service_name} /etc/systemd/system/")
                print(f"   sudo systemctl daemon-reload && sudo systemctl enable {service_name} --now")
            except Exception as e:
                print(f" ! Error installing service: {e}")

    # Final Instructions
    activate_cmd = "source .venv/bin/activate" if platform.system() != "Windows" else ".venv\\Scripts\\activate"
    service_installed = (platform.system() == "Linux" and os.path.exists("/etc/systemd/system/mimicai.service"))
    
    print("\n" + "="*48)
    print("SETUP COMPLETE!")
    print("="*48)
    
    if service_installed:
        print("STATUS: MimicAI is now running in the background.")
        print(" - View logs:  journalctl -u mimicai -f")
        print(" - Stop bot:   sudo systemctl stop mimicai")
        print(" - Start bot:  sudo systemctl start mimicai")
    else:
        print(f"To run your bot, you MUST use the virtual environment:")
        if platform.system() == "Windows":
            print(f"\n1. Activate:    {activate_cmd}")
            print(f"2. Start Bot:   .venv\\Scripts\\python.exe BotManager.py")
        else:
            print(f"\n1. Activate:    {activate_cmd}")
            print(f"2. Start Bot:   python3 BotManager.py")
    
    print("\nNOTE: To uninstall, run 'python3 remove.py'")
    print("="*48)

if __name__ == "__main__":
    setup_mimic()