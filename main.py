
import subprocess
import sys
from pathlib import Path

def run_script(script_path):
    try:
        print(f"Running {script_path}...")
        subprocess.run([sys.executable, script_path], check=True)
        print(f"Finished running {script_path}.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_path}: {e}")
        sys.exit(1)

def main():
    # Define the project root dynamically
    project_root = Path(__file__).parent

    # List of scripts in top-to-bottom order
    scripts = [
        "install_dependencies.py",
      
        "All.py",
    ]

    for script in scripts:
        script_path = project_root / script
        if script_path.exists():
            run_script(str(script_path))
        else:
            print(f"Script {script} not found in {project_root}.")
            sys.exit(1)

if __name__ == "__main__":
    main()
