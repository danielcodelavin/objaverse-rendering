import glob
import os
import subprocess
import sys

# --- CONFIGURATION ---
# Hardcode your paths and settings here.

# 1. Directory containing your downloaded .glb files.
#    The script will search this folder and all its subfolders.
INPUT_DIR = "/path/to/your/downloaded_glbs"

# 2. Directory where the rendered images will be saved.
OUTPUT_DIR = "/path/to/save/your/renders"

# 3. Full path to the Blender executable.
BLENDER_PATH = "/path/to/blender-3.2.2-linux-x64/blender"

# 4. Path to your rendering script.
RENDER_SCRIPT_PATH = "blender_script.py"
# --- END CONFIGURATION ---


def main():
    """Finds and renders all .glb files using the hardcoded settings."""
    
    object_paths = glob.glob(os.path.join(INPUT_DIR, '**', '*.glb'), recursive=True)

    if not object_paths:
        print(f"Error: No .glb files found in '{INPUT_DIR}'. Please check the path.")
        sys.exit(1)

    print(f"Found {len(object_paths)} objects to render.")

    for i, object_path in enumerate(object_paths):
        print(f"--- Processing {i+1}/{len(object_paths)}: {object_path} ---")

        # This command is structured to be identical to the one in your original distributed.py,
        # ensuring the rendering environment and output are exactly the same.
        command = (
            f"xvfb-run --auto-servernum --server-args='-screen 0 1280x720x24' "
            f"{BLENDER_PATH} -b -P {RENDER_SCRIPT_PATH} -- "
            f"--object_path '{object_path}' "
            f"--output_dir '{OUTPUT_DIR}'"
        )
        
        try:
            # Using shell=True to correctly process the full command string with xvfb-run.
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Blender failed to render {object_path}. Continuing with the next object. Error: {e}")
        except FileNotFoundError:
            print(f"Error: A file or command was not found. Please check your configuration paths.")
            sys.exit(1)

    print("--- Finished rendering all objects. ---")


if __name__ == "__main__":
    # Verify that the render script exists before starting
    if not os.path.exists(RENDER_SCRIPT_PATH):
        print(f"Error: Render script not found at '{RENDER_SCRIPT_PATH}'")
        sys.exit(1)
        
    main()