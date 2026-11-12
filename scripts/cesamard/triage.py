"""Combine the three automatic labelers and decide which reviews a human must check.

Labelers: Claude (auto/<set>_claude_cmm_*.json, primary: its label is the one kept), Qwen pass 1 (prompt v1.3b)
and Qwen pass 2 (prompt v1.3c). A review's agreement level k = how many Qwen passes give the same group as Claude.

  python triage.py gold      # calibration on the human gold set: Claude's accuracy by agreement level
  python triage.py main      # tier the main set -> main_auto_v1.3.json + auto/main_check_ids.txt

Tiers (main): "auto" = k >= MIN_AGREE, label accepted as is; "check" = everything else.
Calibration on gold (118, 2026-10-07): k=2 -> Claude's group right 76/78 (97%; 39/39 when not unsure, 37/39 when
unsure, so Claude's unsure flag is not used for the tier), k=1 -> 15/17 (88%), k=0 -> 16/23 (70%).
Boxes: Claude's box is kept; `box_agree` says whether a Qwen box on the same image overlaps it (IoU >= 0.3).
A box on a photo stored with an EXIF rotation also sends the review to "check": the labelers may not agree on which
way up the coordinates are (the label tool shows the photo rotated upright).
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "data" / "cesamard" / "labeling"
AUTO = LAB / "auto"
QWEN = ("v1.3b", "v1.3c")
MIN_AGREE = 2
BOX_IOU = 0.3


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))["labels"] if path.exists() else {}


def main_issue(lab):
    issues = (lab or {}).get("issues", [])
    return ([i for i in issues if i.get("main")] or issues[:1] or [{}])[0]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0


def box_agree(c, qs):
    """True/False when Claude has a box and some Qwen pass has one too; None when there is nothing to compare."""
    if not c.get("boxes"):
        return None
    q_boxes = [b for q in qs for b in q.get("boxes", [])]
    if not q_boxes:
        return None
    return any(b["img"] == cb["img"] and iou(b["bbox"], cb["bbox"]) >= BOX_IOU for b in q_boxes for cb in c["boxes"])


def exif_rotated(item):
    from PIL import Image
    return {im["idx"] for im in item["images"]
            if Image.open(ROOT / "data" / "cesamard" / im["src"]).getexif().get(274, 1) not in (0, 1)}


def agreement(c, qs):
    return sum(q.get("group") == c.get("group") for q in qs)


def calibrate():
    gold = load(LAB / "gold_v1.3.json")
    claude = {**load(AUTO / "round2_claude_cmm_v1.2.json"), **load(AUTO / "gold90_claude_cmm_v1.3.json")}
    qwen = {t: load(AUTO / f"gold_qwen_{t}.json") for t in QWEN}
    print(f"Qwen passes on gold: {[t for t in QWEN if qwen[t]]}")
    for subset in ("round2", "gold90", "all"):
        rows = [r for r, v in gold.items() if subset in ("all", v["source"])]
        by = {}
        for r in rows:
            c = main_issue(claude.get(r))
            k = agreement(c, [main_issue(qwen[t].get(r)) for t in QWEN if qwen[t]])
            g = gold[r]["issues"][0]
            s = by.setdefault(k, Counter())
            s["n"] += 1
            for f in ("group", "defect", "part", "vis"):
                s[f] += c.get(f) == g.get(f)
            s["unsure"] += bool(claude.get(r, {}).get("unsure"))
        print(f"\n[{subset}] Claude vs human, by number of Qwen passes agreeing on the group")
        print(f"  {'k':>2s} {'n':>4s} {'group':>6s} {'defect':>6s} {'part':>6s} {'vis':>6s} {'unsure':>6s}")
        for k in sorted(by, reverse=True):
            s = by[k]
            print(f"  {k:2d} {s['n']:4d} " + " ".join(f"{s[f] / s['n']:6.0%}" for f in ("group", "defect", "part", "vis"))
                  + f" {s['unsure']:6d}")
    print("\nnote: round2 Claude labels used guide v1.2; on gold90 the human checked Claude's drafts (anchoring).")


def tier_main():
    items = {str(x["row_id"]): x for x in json.loads((LAB / "main.json").read_text(encoding="utf-8"))}
    dups = json.loads((LAB / "main_duplicates.json").read_text(encoding="utf-8"))
    claude = load(AUTO / "main_claude_cmm_v1.3.json")
    qwen = {t: load(AUTO / f"main_qwen_{t}.json") for t in QWEN}
    missing = sorted(set(items) - set(claude), key=int)
    if missing:
        sys.exit(f"Claude labels missing for {len(missing)} reviews, e.g. {missing[:5]} (run merge_parts.py main v1.3)")

    out, stats = {}, Counter()
    for r in sorted(items, key=int):
        lab = claude[r]
        c = main_issue(lab)
        qs = [main_issue(qwen[t].get(r)) for t in QWEN if qwen[t]]
        k = agreement(c, qs)
        rotated_box = any(b["img"] in exif_rotated(items[r]) for b in c.get("boxes", []))
        tier = "auto" if k >= MIN_AGREE and not rotated_box else "check"
        out[r] = {"domain": items[r]["domain"], "duplicates": dups.get(r, []), "tier": tier, "qwen_agree": k,
                  "rotated_box": rotated_box,
                  "box_agree": box_agree(c, qs), "issues": [c], "marks": lab.get("marks", {}),
                  "unsure": lab.get("unsure", False), "note": lab.get("note", ""),
                  "qwen": {t: {f: q.get(f) for f in ("part", "group", "defect", "vis")} for t, q in zip(QWEN, qs)}}
        stats[tier] += 1
        stats[f"k={k}"] += 1
        stats[f"box_agree={out[r]['box_agree']}"] += 1
        stats["rotated_box"] += rotated_box
    meta = {"guide_version": "v1.3", "primary": "claude-cmm", "qwen": [t for t in QWEN if qwen[t]],
            "min_agree": MIN_AGREE, "box_iou": BOX_IOU, "n": len(out), "stats": dict(stats)}
    (LAB / "main_auto_v1.3.json").write_text(json.dumps({"meta": meta, "labels": out}, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    check = [r for r, v in out.items() if v["tier"] == "check"]
    (AUTO / "main_check_ids.txt").write_text("\n".join(check) + "\n")
    print(f"main_auto_v1.3.json: {len(out)} reviews | {dict(sorted(stats.items()))}")
    print(f"to check: {len(check)} -> auto/main_check_ids.txt")


if __name__ == "__main__":
    {"gold": calibrate, "main": tier_main}[sys.argv[1]]()
