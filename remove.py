import os
import subprocess
import platform
import shutil

def remove_mimic():
    print("--- MimicAI Uninstall Utility ---")
    confirm = input("This will stop the bot and DELETE all local data/configs. Proceed? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        return

    # 1. Handle Linux Service
    if platform.system() == "Linux":
        service_name = "mimicai.service"
        service_path = f"/etc/systemd/system/{service_name}"
        if os.path.exists(service_path):
            print("\n - Removing systemd service...")
            try:
                cmds = [
                    ["sudo", "systemctl", "stop", service_name],
                    ["sudo", "systemctl", "disable", service_name],
                    ["sudo", "rm", service_path],
                    ["sudo", "systemctl", "daemon-reload"]
                ]
                for cmd in cmds:
                    subprocess.run(cmd, stderr=subprocess.DEVNULL)
                print("   - Service removed.")
            except Exception as e:
                print(f"   - Failed to remove service: {e}")

    # 2. Delete Virtual Environment
    if os.path.exists(".venv"):
        print(" - Deleting virtual environment (.venv)...")
        shutil.rmtree(".venv")

    # 3. Delete Data and Configs
    to_delete = [".env", "cogs/data", "mimicai.service"]
    for path in to_delete:
        if os.path.exists(path):
            print(f" - Deleting: {path}")
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    print("\n" + "="*48)
    print("UNINSTALL COMPLETE")
    print("="*48)

if __name__ == "__main__":
    remove_mimic()