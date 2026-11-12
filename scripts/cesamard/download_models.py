"""Download models into the Hugging Face cache (safe to rerun; finished files are skipped).

  python scripts/cesamard/download_models.py            # the box-model candidates (round 1)
  python scripts/cesamard/download_models.py voters     # third-opinion VLM + photo-text relevance model
"""
import sys
import time

from huggingface_hub import snapshot_download

WEIGHTS = ["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"]
JOBS = {
    "default": [
        ("IDEA-Research/grounding-dino-base", ["*.json", "*.safetensors", "*.txt"]),
        ("florence-community/Florence-2-large", None),
        ("Qwen/Qwen3-VL-8B-Instruct", None),
    ],
    "voters": [
        ("google/siglip2-so400m-patch14-384", WEIGHTS),
        ("OpenGVLab/InternVL3_5-8B-HF", WEIGHTS),
    ],
}

for repo, patterns in JOBS[sys.argv[1] if len(sys.argv) > 1 else "default"]:
    t = time.time()
    path = snapshot_download(repo, allow_patterns=patterns, max_workers=8)
    print(f"DONE {repo} ({time.time() - t:.0f}s) -> {path}", flush=True)
print("ALL DONE", flush=True)
