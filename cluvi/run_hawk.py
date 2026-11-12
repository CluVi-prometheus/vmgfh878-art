"""Run cluvi.predict inside HAWK's patched runtime. Does in Python what HAWK's scripts/run.sh does in bash:
check that the patch manifest exists, then put HAWK's `src` first and its `.runtime/python` (patched transformers 4.52.0,
tokenizers 0.21.1, huggingface-hub 0.31.4, qwen-vl-utils 0.0.11) second on the import path. Written in Python because
run.sh joins paths with ':' and Windows Python expects ';' (and our project path has spaces and Korean characters).

  python -m cluvi.run_hawk --model heavy --split gold                    # E1
  python -m cluvi.run_hawk --model heavy --split gold --hawk-keep 0.20  # E2
  python -m cluvi.train_lora ...                                        # E3 LoRA training (calls use_runtime itself)

HAWK checkout: <project>/HAWK by default, or set the HAWK_DIR environment variable.
"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAWK = Path(os.environ.get("HAWK_DIR", ROOT / "HAWK"))
RUNTIME = HAWK / ".runtime" / "python"


def use_runtime(tag="run_hawk"):
    """Put HAWK's src + patched runtime first on the import path (before anything imports transformers)."""
    if not (RUNTIME / "hawk-patch-manifest.json").exists():
        sys.exit(f"HAWK runtime not found at {RUNTIME} (no hawk-patch-manifest.json). Build it first (design doc 8.1).")
    if "transformers" in sys.modules:
        sys.exit("transformers was imported before the runtime path was set")
    sys.path[:0] = [str(HAWK / "src"), str(RUNTIME)]          # same order as run.sh: src, then the patched runtime
    import transformers
    print(f"[{tag}] transformers {transformers.__version__} from {Path(transformers.__file__).parent}", flush=True)
    transformers.logging.set_verbosity_error()   # keep the terminal to progress lines (drops per-call generation notices)
    return transformers


def main():
    use_runtime()
    sys.argv = ["cluvi.predict", *sys.argv[1:]]
    runpy.run_module("cluvi.predict", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
