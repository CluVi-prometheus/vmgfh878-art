"""Wait for E1 (heavy_test.jsonl, 185 rows) to finish and free the GPU, then run E2 = E1 + HAWK keep 0.20 on val and test.

  .venv\\Scripts\\python.exe scripts\\run_e2_after_e1.py      (run from the project folder, in the side terminal)

Only one model fits on the GPU, so E2 must not start while E1 is still running. Ctrl-C stops it at any point;
running it again resumes (predict.py continues an unfinished file with the same settings).
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E1_TEST = ROOT / "outputs" / "preds" / "heavy_test.jsonl"
N_TEST = 185
E2_RUNS = [["--split", "val", "--hawk-keep", "0.20"], ["--split", "test", "--hawk-keep", "0.20"]]


def rows(path):
    return sum(1 for line in path.open(encoding="utf-8") if line.strip()) if path.exists() else 0


def gpu_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.split()[0])


t0 = time.time()
print(f"[E2 waiter] waiting for E1 test to finish ({E1_TEST.name}: {N_TEST} rows) ...", flush=True)
last = -1
while (n := rows(E1_TEST)) < N_TEST:
    if n != last:
        print(f"[E2 waiter] {time.strftime('%H:%M')} E1 test rows saved: {n}/{N_TEST} (saved every 10)", flush=True)
        last = n
    time.sleep(30)
while gpu_used_mib() > 2000:          # E1 process still holding the model
    time.sleep(10)
print(f"[E2 waiter] E1 done, GPU free after {(time.time() - t0) / 60:.0f} min. Starting E2.", flush=True)
for args in E2_RUNS:
    r = subprocess.run([sys.executable, "-m", "cluvi.run_hawk", "--model", "heavy", *args], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"[E2 waiter] stopped: {' '.join(args)} exited with {r.returncode}")
print("[E2 waiter] E2 val + test finished.", flush=True)
