from datasets import load_dataset
import objaverse, json, gzip

ds  = load_dataset("cindyxl/ObjaversePlusPlus", split="train")        # HF table  [oai_citation:0‡Hugging Face](https://huggingface.co/datasets/cindyxl/ObjaversePlusPlus?utm_source=chatgpt.com)
very_high_quality = ds.filter(lambda x: x["score"] >= 3)
  # quality ≥ 3 & single-object  [oai_citation:1‡Hugging Face](https://huggingface.co/papers/2504.07334?utm_source=chatgpt.com)
no_multi = very_high_quality.filter(lambda x: x["is_multi_object"] == "false")
uids = no_multi["UID"]                                                     

object_paths = objaverse._load_object_paths()                         # uid → 'glbs/0ab-...'  [oai_citation:2‡objaverse.allenai.org](https://objaverse.allenai.org/docs/objaverse-1.0/?utm_source=chatgpt.com)
urls = [f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_paths[u]}"
        for u in uids]

with open("input_models_path.json", "w") as f:
    json.dump(urls, f, indent=2)
print("wrote", len(urls), "URLs")