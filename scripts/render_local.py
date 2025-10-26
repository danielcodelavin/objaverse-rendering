import glob
import os
import subprocess
import sys

INPUT_DIR = "/home/stud/lavingal/storage/slurm/lavingal/LVSM/datasets/converted_glb_gso"


OUTPUT_DIR = "/home/stud/lavingal/storage/slurm/lavingal/LVSM/datasets/GSO_rendered/images"


BLENDER_PATH = "/storage/slurm/lavingal/lavingal/blender-3.2.2-linux-x64/blender"


RENDER_SCRIPT_PATH = "blender_script.py"


def main():
    """Finds and renders all .glb files using the hardcoded settings."""
    
    object_paths = glob.glob(os.path.join(INPUT_DIR, '**', '*.glb'), recursive=True)

    if not object_paths:
        print(f"Error: No .glb files found in '{INPUT_DIR}'. Please check the path.")
        sys.exit(1)

    print(f"Found {len(object_paths)} objects to render.")

    for i, object_path in enumerate(object_paths):
        print(f"--- Processing {i+1}/{len(object_paths)}: {object_path} ---")

       
        command = (
            f"xvfb-run --auto-servernum --server-args='-screen 0 1280x720x24' "
            f"{BLENDER_PATH} -b -P {RENDER_SCRIPT_PATH} -- "
            f"--object_path '{object_path}' "
            f"--output_dir '{OUTPUT_DIR}'"
        )
        
        try:
         
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Blender failed to render {object_path}. Continuing with the next object. Error: {e}")
        except FileNotFoundError:
            print(f"Error: A file or command was not found. Please check your configuration paths.")
            sys.exit(1)

    print("--- Finished rendering all objects. ---")


if __name__ == "__main__":
 
    if not os.path.exists(RENDER_SCRIPT_PATH):
        print(f"Error: Render script not found at '{RENDER_SCRIPT_PATH}'")
        sys.exit(1)
        
    main()