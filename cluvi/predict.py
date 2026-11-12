"""Run one model over one split ONCE and save every Card. All later comparisons (cascade, thresholds) reuse this file.

Team split (2026-11-11): train 881 (silver) / val 100 (silver, held out of train) / test 185 (human). dev 88 = no labels.

  python -m cluvi.predict --model mock  --split test                       # plumbing test, no GPU
  python -m cluvi.predict --model text  --split test                       # Light Text, CPU, seconds

VLM runs (heavy / light) must go through HAWK's patched transformers 4.52.0, so E1, E2 and E3 share one library stack.
cluvi.run_hawk does in Python what HAWK's scripts/run.sh does in bash (works on Windows paths):
  python -m cluvi.run_hawk --model heavy --split dev --limit 30                      # E0 format check (no labels)
  python -m cluvi.run_hawk --model heavy --split dev --start 30 --limit 30           # E0 holdout rows 31-60
  python -m cluvi.run_hawk --model heavy --split val                                 # E1 on val 100
  python -m cluvi.run_hawk --model heavy --split test                                # E1 on test 185
  python -m cluvi.run_hawk --model heavy --split test --hawk-keep 0.20               # E2 -> heavy_hawk200_test.jsonl
  python -m cluvi.run_hawk --model heavy --split val --adapter outputs/lora/e3l/epoch2   # E3-L -> heavy_lora-e3l-epoch2_val

The run's settings are written next to the predictions as <name>_config.json. No config file = not a valid result.
Resuming an existing file is allowed only if the settings are identical; otherwise the run stops (use another --tag).
Every review prints one progress line (position, elapsed, time per review, time left, finish clock).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from .common import PRED_DIR, PROMPT_VERSION, load_cards, load_split, prompt_text, save_cards
from .models import MAX_PIXELS, MIN_PIXELS, load_model

RUNTIME_TRANSFORMERS = "4.52.0"           # HAWK's validated (and patched) version
VOLATILE = {"started", "n"}               # config keys allowed to differ when resuming


def check_runtime(allow_any):
    """VLM runs must use HAWK's patched transformers 4.52.0 (loaded from its .runtime folder), so that E1 and E2 differ
    only in HAWK being on. Fails loudly instead of silently running on the venv's transformers."""
    import transformers
    ok = transformers.__version__ == RUNTIME_TRANSFORMERS and ".runtime" in Path(transformers.__file__).parts
    if not ok and not allow_any:
        sys.exit(f"transformers {transformers.__version__} from {transformers.__file__}: VLM runs need HAWK's patched "
                 f"{RUNTIME_TRANSFORMERS}. Start with `python -m cluvi.run_hawk ...` (or pass --any-transformers to debug).")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="mock | text | light | heavy")
    ap.add_argument("--split", default="test", help="train (881) | val (100) | test (185) | dev (88, no labels)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--start", type=int, default=0, help="skip the first N reviews (e.g. dev 30-60 as a holdout)")
    ap.add_argument("--min-pixels", type=int, default=MIN_PIXELS, help="pixel floor per photo (model card example)")
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS, help="pixel cap per photo (28x28 px = 1 token)")
    ap.add_argument("--hawk-keep", type=float,
                    help="official HAWK pruning with this keep ratio (0.20 = HAWK native-resolution setting)")
    ap.add_argument("--repetition-penalty", type=float,
                    help="override the shipped generation_config value (1.05); for the E0 diagnosis only")
    ap.add_argument("--adapter", help="LoRA adapter folder (E3, cluvi.train_lora); attached unmerged on the 4-bit model")
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--prompt", default=PROMPT_VERSION)
    ap.add_argument("--warmup", type=int, default=2, help="untimed calls before the run (excluded from results)")
    ap.add_argument("--tag", help="suffix for a variant; default is built from the settings (e.g. _hawk200)")
    ap.add_argument("--any-transformers", action="store_true", help="debug only: skip the 4.52.0 runtime check")
    a = ap.parse_args()
    vlm = a.model in ("light", "heavy")
    runtime_ok = check_runtime(a.any_transformers) if vlm else None

    if a.tag is None:      # settings that differ from E1 go into the file name, so variants never share a file
        ad = Path(a.adapter) if a.adapter else None
        a.tag = "".join([f"_lora-{ad.parent.name}-{ad.name}" if ad else "",
                         f"_hawk{round(a.hawk_keep * 1000)}" if a.hawk_keep is not None else "",
                         f"_px{a.max_pixels // 784}" if vlm and a.max_pixels != MAX_PIXELS else "",
                         f"_rp{round(a.repetition_penalty * 100)}" if a.repetition_penalty is not None else "",
                         f"_{a.prompt}" if a.prompt != PROMPT_VERSION else ""])
    samples = load_split(a.split)[a.start: a.start + a.limit if a.limit else None]
    name = f"{a.model}{a.tag}_{a.split}{f'_from{a.start}' if a.start else ''}{'_smoke' if a.limit else ''}"
    out, cfg_path = PRED_DIR / f"{name}.jsonl", PRED_DIR / f"{name}_config.json"

    kw = {"min_pixels": a.min_pixels, "max_pixels": a.max_pixels, "max_new_tokens": a.max_new_tokens,
          "prompt_version": a.prompt, "hawk_keep": a.hawk_keep, "repetition_penalty": a.repetition_penalty} if vlm else {}
    if vlm and a.adapter:
        kw["adapter"] = a.adapter                         # only passed when set: E1 / E2 calls stay byte-identical
    model = load_model(a.model, **kw)
    cfg = {"run_id": name, "split": a.split, "start": a.start, "n": len(samples), "warmup": a.warmup,
           "prompt_sha1": hashlib.sha1(prompt_text(a.prompt).encode()).hexdigest()[:12],
           "hawk_runtime": runtime_ok, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           **(model.run_config() if hasattr(model, "run_config") else {})}
    done = {}
    if out.exists():
        old = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else None
        diff = None if old is None else sorted(k for k in set(old) | set(cfg)
                                                if k not in VOLATILE and old.get(k) != cfg.get(k))
        if old is None or diff:
            sys.exit(f"{out.name} exists with different settings ({'no config file' if old is None else diff}). "
                     f"Use another --tag or delete the old file on purpose.")
        done = load_cards(out)
        print(f"resuming {out.name}: {len(done)} done", flush=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")

    for i, s in enumerate(samples[: a.warmup if vlm else 0], 1):
        print(f"[warm-up {i}/{a.warmup}] row {s.row_id} (not saved)", flush=True)
        model.analyze(s)                                   # warm-up: CUDA kernels / caches; result thrown away
    todo = sum(s.row_id not in done for s in samples)
    print(f"[{name}] {len(samples)} reviews, {len(samples) - todo} already done, {todo} to run", flush=True)
    t0, ran = time.time(), 0
    for n, s in enumerate(samples, 1):
        if s.row_id in done:
            continue
        c = done[s.row_id] = model.analyze(s)
        ran += 1
        if ran % 10 == 0 or n == len(samples):
            save_cards(list(done.values()), out)
        if vlm:
            print(progress_line(n, len(samples), ran, todo, time.time() - t0, c), flush=True)
    save_cards(list(done.values()), out)
    print(f"wrote {out}", flush=True)


def _hms(sec):
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def progress_line(n, total, ran, todo, elapsed, card):
    """One short line per review: position, elapsed, time per review, time left, finish clock, and what came out."""
    per = elapsed / ran
    left = per * (todo - ran)
    finish = time.strftime("%H:%M", time.localtime(time.time() + left))
    e2e = card.cost.get("e2e_ms")
    return (f"[{n}/{total} {100 * n / total:5.1f}%] elapsed {_hms(elapsed)} | {per:5.1f}s/review | "
            f"left ~{_hms(left)} (done ~{finish}) | row {card.row_id}: {card.group6 or 'INVALID'}"
            f"{'' if e2e is None else f' {e2e / 1000:.1f}s'}{' ' + ','.join(card.issues) if card.issues else ''}")


if __name__ == "__main__":
    main()
