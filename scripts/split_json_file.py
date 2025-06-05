
import json
import os

# ───── CONFIG ────────────────────────────────────────────────────────────────
INPUT_PATH = "real_truncated_input_models_path.json"   # path to your JSON list
NUM_SPLITS = 2                    # number of output files to create
# ───────────────────────────────────────────────────────────────────────────────

# Load
with open(INPUT_PATH, 'r') as f:
    data = json.load(f)

# Validate
if not isinstance(data, list):
    raise RuntimeError(f"Expected a list in {INPUT_PATH}, got {type(data)}")
total = len(data)
if total == 0:
    print("Input list is empty — nothing to split.")
    exit(0)

# Compute chunk boundaries
dirname = os.path.dirname(INPUT_PATH) or '.'
basename = os.path.basename(INPUT_PATH)
base_size = total // NUM_SPLITS
remainder = total % NUM_SPLITS

start = 0
for i in range(NUM_SPLITS):
    size = base_size + (1 if i < remainder else 0)
    end = start + size
    chunk = data[start:end]
    if not chunk:
        print(f"Skipping empty chunk {i}")
    else:
        out_name = f"{i}_{basename}"
        out_path = os.path.join(dirname, out_name)
        with open(out_path, 'w') as out:
            json.dump(chunk, out, indent=2)
        print(f"Wrote {len(chunk)} items to {out_path}")
    start = end

print("Done splitting.")