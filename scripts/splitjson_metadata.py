import json
from pathlib import Path
from typing import List
import os

def remove_urls_by_uid(uid_array: List[str], filepath: str) -> List[str]:

    path = Path(filepath)

   
    try:
        with path.open("r", encoding="utf-8") as f:
            urls: List[str] = json.load(f)
            if not isinstance(urls, list):
                raise ValueError("JSON file must contain a list of strings.")
    except FileNotFoundError:
        raise FileNotFoundError(f"{filepath!s} not found.")
    except json.JSONDecodeError:
        raise ValueError(f"{filepath!s} does not contain valid JSON.")

 
    def uid_from_url(url: str) -> str:
        return url.rsplit("/", 1)[-1].split(".glb", 1)[0]

   
    uid_set = set(uid_array)
    kept_urls = [url for url in urls if uid_from_url(url) not in uid_set]

   
    new_path = path.with_name(f"real_{path.name}")

   
    with new_path.open("w", encoding="utf-8") as f:
        json.dump(kept_urls, f, indent=2)

    return kept_urls



def load_metadata_filenames(folder):
    metadata_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".json"):
                metadata_files.append(file[:-5])
    return metadata_files

if __name__ == "__main__":
    METADATA_FOLDER ="/home/stud/lavingal/storage/slurm/lavingal/LVSM/datasets/objaverseplus/metadata"
    JSON_FILE = "/home/stud/lavingal/storage/slurm/lavingal/objaverse-rendering/scripts/truncated_input_models_path.json"
    bad_uids = load_metadata_filenames(METADATA_FOLDER)
    retained_urls = remove_urls_by_uid(bad_uids, JSON_FILE)
    print(f"Removed {len(bad_uids)} uids from {len(retained_urls)} urls.")
    print(f"Retained {len(retained_urls)} urls.")
    print(f"Saved to {JSON_FILE} and {JSON_FILE}.truncated")
    
