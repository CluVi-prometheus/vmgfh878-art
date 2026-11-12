"""Compare labeler models on the clean evaluation set (evalset.json = round2 30 + blind60 60), same instructions.

Truth
  round2 30   gold_v1.3.json (source round2): part, part group, defect group, defect, visibility, boxes
  blind 53    blind60_final_labels.json (7 of 60 excluded by the human): defect group, visibility
Labelers
  fable, sonnet, opus   auto/evalset_<tag>_v1.3_part*.json (prompts/claude_eval_v1.3.md); opus's blind rows come from
                        its main run (same instructions, prompts/claude_main_v1.3.md)
  qwen                  Qwen3-VL-8B, prompt v1.3b (gold_qwen / main_qwen)
Token use per model is read from auto/evalset_usage.json ({tag: {"tokens": n, "reviews": n, "minutes": m}}) when present.

  python eval_models.py
"""
import json
from pathlib import Path

LAB = Path(__file__).resolve().parents[2] / "data" / "cesamard" / "labeling"
AUTO = LAB / "auto"


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))["labels"] if p.exists() else {}


def main_issue(lab):
    issues = (lab or {}).get("issues", [])
    return ([i for i in issues if i.get("main")] or issues[:1] or [{}])[0]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0


def labelers():
    out = {}
    for tag in ("fable", "sonnet", "opus"):
        lab = {}
        for p in sorted(AUTO.glob(f"evalset_{tag}_v1.3_part*.json")):
            lab.update(load(p))
        if tag == "opus":
            lab = {**load(AUTO / "main_claude_cmm_v1.3.json"), **lab}
        if lab:
            out[tag] = lab
    out["qwen"] = {**load(AUTO / "main_qwen_v1.3b.json"), **load(AUTO / "gold_qwen_v1.3b.json")}
    return out


def main():
    gold = {r: v for r, v in load(LAB / "gold_v1.3.json").items() if v["source"] == "round2"}
    blind = {r: v for r, v in load(LAB / "blind60_final_labels.json").items() if v.get("done") and not v.get("exclude")}
    usage = json.loads((AUTO / "evalset_usage.json").read_text(encoding="utf-8")) if (AUTO / "evalset_usage.json").exists() else {}
    labs = labelers()

    print(f"blind {len(blind)} reviews (group, visibility) | round2 {len(gold)} reviews (all fields)\n")
    head = ["model", "B group", "B vis", "R part", "R pgrp", "R group", "R defect", "R vis", "R point", "R iou.5", "tokens/review"]
    print(" ".join(f"{h:>9s}" for h in head))
    for name, lab in labs.items():
        missing = [r for r in list(blind) + list(gold) if r not in lab]
        b = [main_issue(lab.get(r)) for r in blind]
        bg = sum(m.get("group") == blind[r]["group"] for m, r in zip(b, blind))
        bv = sum(m.get("vis") == ("not_visible" if blind[r]["photo"] == "unrelated" else blind[r]["photo"]) for m, r in zip(b, blind))
        acc = {f: 0 for f in ("part", "scope", "group", "defect", "vis")}
        pt = hit = nbox = 0
        for r, g in gold.items():
            m, gi = main_issue(lab.get(r)), g["issues"][0]
            for f in acc:
                acc[f] += m.get(f) == gi.get(f)
            if gi["vis"] == "localized" and gi["boxes"]:
                nbox += 1
                mb = m.get("boxes") or []
                best_pt = best_iou = 0
                for b_ in mb:
                    gb = [x["bbox"] for x in gi["boxes"] if x["img"] == b_["img"]]
                    cx, cy = (b_["bbox"][0] + b_["bbox"][2]) / 2, (b_["bbox"][1] + b_["bbox"][3]) / 2
                    best_pt = max(best_pt, any(x[0] <= cx <= x[2] and x[1] <= cy <= x[3] for x in gb))
                    best_iou = max([best_iou] + [iou(b_["bbox"], x) for x in gb])
                pt += best_pt
                hit += best_iou >= 0.5
        u = usage.get(name, {})
        tpr = f"{u['tokens'] / u['reviews'] / 1000:.1f}k" if u.get("tokens") else "-"
        nb, ng = len(blind), len(gold)
        row = [name, f"{bg / nb:.0%}", f"{bv / nb:.0%}"] + [f"{acc[f] / ng:.0%}" for f in acc] + \
              [f"{pt}/{nbox}", f"{hit}/{nbox}", tpr]
        print(" ".join(f"{x:>9s}" for x in row) + (f"   (missing {len(missing)})" if missing else ""))
    print("\nB = blind sample (human, adjudicated), R = round2 (human blind, then adjudicated). "
          "R point/iou.5 = reviews with a human box whose best model box has its centre inside / IoU >= 0.5.")


if __name__ == "__main__":
    main()
