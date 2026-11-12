"""Score every automatic labeler against the human blind sample (blind60_labels.json).

The sample is stratified by Claude-Qwen agreement k (make_blind.py); every rate is reported per stratum and
weighted back to main-set proportions. Human "ambiguous" answers count both the first and the second group as
right ("lenient"); "strict" uses the first group only. Excluded reviews are left out and counted.

  python eval_blind.py                       # all labelers found in auto/
  python eval_blind.py --labels <path>       # another blind labels file (testing)
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

LAB = Path(__file__).resolve().parents[2] / "data" / "cesamard" / "labeling"
AUTO = LAB / "auto"
LABELERS = {
    "claude": "main_claude_cmm_v1.3.json",
    "qwen-b": "main_qwen_v1.3b.json",
    "qwen-c": "main_qwen_v1.3c.json",
    "internvl": "main_internvl_v1.3b.json",
}
VISUAL = {"localized": True, "diffuse": True, "not_visible": False, "unrelated": False}


def main_issue(lab):
    issues = (lab or {}).get("issues", [])
    return ([i for i in issues if i.get("main")] or issues[:1] or [{}])[0]


def right(group, h):
    """Lenient: the human's first group, or the second one when they marked the review ambiguous."""
    return group is not None and group in (h["group"], h.get("alt_group"))


def wilson(k, n, z=1.96):
    if not n:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=LAB / "blind60_labels.json")
    args = ap.parse_args()
    human = {r: v for r, v in json.loads(args.labels.read_text(encoding="utf-8"))["labels"].items() if v.get("done")}
    strata = json.loads((AUTO / "blind60_strata.json").read_text(encoding="utf-8"))["strata"]
    k_of = {r: int(k) for k, s in strata.items() for r in s["rows"]}
    weight = {int(k): s["weight"] for k, s in strata.items()}
    excluded = [r for r, v in human.items() if v.get("exclude")]
    rows = [r for r in human if r not in excluded]
    print(f"blind sample: {len(human)} done, {len(excluded)} excluded, {len(rows)} scored | "
          f"human ambiguous {sum(human[r]['ambiguous'] for r in rows)} | photo {dict(Counter(human[r]['photo'] for r in rows))}")

    labelers = {n: json.loads((AUTO / f).read_text(encoding="utf-8"))["labels"] for n, f in LABELERS.items() if (AUTO / f).exists()}
    print(f"\n{'labeler':9s} {'k':>2s} {'n':>3s} {'group':>6s} {'lenient':>8s} {'visual':>7s}")
    for name, lab in labelers.items():
        tot = Counter()
        for k in (2, 1, 0):
            rs = [r for r in rows if k_of.get(r) == k]
            c = Counter()
            for r in rs:
                h, m = human[r], main_issue(lab.get(r))
                c["n"] += 1
                c["strict"] += m.get("group") == h["group"]
                c["lenient"] += right(m.get("group"), h)
                c["visual"] += VISUAL.get(m.get("vis"), None) == VISUAL[h["photo"]]
            for f in ("strict", "lenient", "visual"):
                tot[f] += weight[k] * c[f] / c["n"] if c["n"] else 0
            tot["w"] += weight[k] if c["n"] else 0
            if c["n"]:
                print(f"{name:9s} {k:2d} {c['n']:3d} {c['strict'] / c['n']:6.0%} {c['lenient'] / c['n']:8.0%} {c['visual'] / c['n']:7.0%}")
        if tot["w"]:
            print(f"{name:9s} {'all':>2s} {'':>3s} {tot['strict'] / tot['w']:6.0%} {tot['lenient'] / tot['w']:8.0%} "
                  f"{tot['visual'] / tot['w']:7.0%}   (weighted to main)")

    if "claude" in labelers:
        print("\nClaude's group (lenient) by agreement level, 95% Wilson interval:")
        for k in (2, 1, 0):
            rs = [r for r in rows if k_of.get(r) == k]
            ok = sum(right(main_issue(labelers["claude"].get(r)).get("group"), human[r]) for r in rs)
            lo, hi = wilson(ok, len(rs))
            print(f"  k={k}: {ok}/{len(rs)}  [{lo:.0%}, {hi:.0%}]")


if __name__ == "__main__":
    main()
