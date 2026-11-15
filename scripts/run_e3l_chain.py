"""E3-L end to end, in the side terminal: probe -> adapter load check -> training -> full val run -> val verdict.
Stops before test (test 185 runs once, only after the user has seen the val verdict).

  .venv\\Scripts\\python.exe scripts\\run_e3l_chain.py      (run from the project folder)

  1/5 probe      cluvi.train_lora --probe 10: longest train inputs fit in VRAM? speed -> time estimate. If they do not
                 fit, training uses the fallback fixed in the proposal (--max-visual-tokens 4000, drops 75 rows).
  2/5 load check cluvi.run_hawk --split dev --limit 2 --adapter <probe adapter>: the inference path attaches the
                 adapter (112 LoRA modules) and still writes valid JSON. Training does not start if this fails.
  3/5 train      cluvi.train_lora (3 epochs, val head check after each epoch, resume point every 20 updates)
  4/5 val        cluvi.run_hawk --split val --adapter outputs/lora/e3l/epoch<N> (N = the pre-registered epoch rule)
  5/5 verdict    cluvi.judge_e3 (proposal's success criteria vs E1)

Ctrl-C stops it at any point; running it again continues (finished stages are skipped, training resumes from the
last saved step, the val run continues its file).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LORA = ROOT / "outputs" / "lora"
RUN = LORA / "e3l"
PREDS = ROOT / "outputs" / "preds"
LOG = LORA / "e3l_chain_log.txt"
EPOCHS = 3
ENV = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1"}
T0 = time.time()


def say(msg):
    line = f"[E3-L chain {time.strftime('%m-%d %H:%M')} +{(time.time() - T0) / 60:.0f}m] {msg}"
    print(line, flush=True)
    LORA.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(stage, args):
    """Run one stage, showing its own progress lines live; returns the exit code."""
    say(f"{stage} start: python {' '.join(args)}")
    p = subprocess.Popen([sys.executable, *args], cwd=ROOT, env=ENV, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
    with LOG.open("a", encoding="utf-8") as log:
        for line in p.stdout:
            if "UserWarning" in line or "type_of_target" in line or "flop_counter" in line:
                continue
            print(line, end="", flush=True)
            log.write(line)
    p.wait()
    say(f"{stage} exit code {p.returncode}")
    return p.returncode


def gpu_free_gb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.split(",")
    return (int(out[1]) - int(out[0])) / 1024


def main():
    say(f"start | GPU free {gpu_free_gb():.1f} GB")
    train_args = ["-m", "cluvi.train_lora"]
    started = (RUN / "train_config.json").exists()
    if started:
        cfg = json.loads((RUN / "train_config.json").read_text(encoding="utf-8"))
        if cfg["data"].get("max_visual_tokens"):
            train_args += ["--max-visual-tokens", str(cfg["data"]["max_visual_tokens"])]
        say("training already started earlier -> skip probe and load check")
    else:
        # ---- 1/5 probe ----
        if run("1/5 probe", ["-m", "cluvi.train_lora", "--probe", "10"]) != 0:
            sys.exit("[E3-L chain] probe failed -> stopped before training (see the lines above)")
        probe = json.loads((LORA / "_probe" / "probe.json").read_text(encoding="utf-8"))
        oom = [r["row_id"] for r in probe["rows"] if not r["ok"]]
        if oom:
            say(f"longest inputs did not fit ({oom}) -> fallback fixed in the proposal: --max-visual-tokens 4000")
            train_args += ["--max-visual-tokens", "4000"]
        else:
            worst = max(r["reserved_gb"] for r in probe["rows"])
            est = probe.get("estimate") or {}
            say(f"probe OK: all {len(probe['rows'])} fit, max reserved {worst} GB | estimate "
                f"{est.get('train_h', float('nan')):.1f} h training + val head checks")
        # ---- 2/5 adapter load check ----
        code = run("2/5 load check", ["-m", "cluvi.run_hawk", "--model", "heavy", "--split", "dev", "--limit", "2",
                                      "--warmup", "0", "--adapter", str(LORA / "_probe" / "adapter"),
                                      "--tag", "_lora-probe"])
        cfg_p = PREDS / "heavy_lora-probe_dev_smoke_config.json"
        cards = [json.loads(x) for x in (PREDS / "heavy_lora-probe_dev_smoke.jsonl").read_text(encoding="utf-8")
                 .splitlines() if x.strip()] if code == 0 else []
        mods = json.loads(cfg_p.read_text(encoding="utf-8")).get("lora_modules") if cfg_p.exists() else None
        parsed = sum("json_parse_failed" not in c["issues"] for c in cards)
        if code != 0 or mods != 112 or parsed < len(cards) or not cards:
            sys.exit(f"[E3-L chain] adapter load check failed (exit {code}, LoRA modules {mods}, parsed "
                     f"{parsed}/{len(cards)}) -> stopped before training")
        say(f"load check OK: {mods} LoRA modules attached, {parsed}/{len(cards)} replies are valid JSON")

    # ---- 3/5 train ----
    if (RUN / f"epoch{EPOCHS}" / "val_head.json").exists():
        say("training already finished -> skip")
    else:
        if (RUN / "last").exists():
            train_args.append("--resume")
        if run("3/5 train", train_args) != 0:
            sys.exit("[E3-L chain] training stopped. Run this script again to resume from the last saved step.")

    # ---- 4/5 full val run with the chosen epoch ----
    from cluvi.train_lora import pick_epoch
    ep = pick_epoch(RUN, EPOCHS)
    if ep is None:
        sys.exit("[E3-L chain] no val head results -> cannot pick an epoch")
    heads = {e: json.loads((RUN / f"epoch{e}" / "val_head.json").read_text(encoding="utf-8"))["metrics"]["all"]
             for e in range(1, EPOCHS + 1) if (RUN / f"epoch{e}" / "val_head.json").exists()}
    for e, m in heads.items():
        say(f"epoch {e} val head: G6 {m['group6_f1']:.3f} Vis {m['visibility_f1']:.3f} Scope {m['scope_f1']:.3f} "
            f"-> mean {m['select']:.3f}{'   <- chosen (rule: highest mean, tie -> earlier)' if e == ep else ''}")
    if run("4/5 val", ["-m", "cluvi.run_hawk", "--model", "heavy", "--split", "val",
                       "--adapter", str(RUN / f"epoch{ep}")]) != 0:
        sys.exit("[E3-L chain] val run stopped. Run this script again to continue it.")

    # ---- 5/5 verdict ----
    run("5/5 verdict", ["-m", "cluvi.judge_e3", str(PREDS / f"heavy_lora-e3l-epoch{ep}_val.jsonl")])
    say(f"done. test 185 is NOT run: it waits for the user's go after the val verdict. "
        f"(next: python -m cluvi.run_hawk --model heavy --split test --adapter {RUN / f'epoch{ep}'})")


if __name__ == "__main__":
    main()
