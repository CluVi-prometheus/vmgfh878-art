"""Assemble the gold set from the human-labeled rounds.

  round2 (blind, corrected to v1.3)  data/cesamard/labeling/round2_gold_v1.3.json
  gold90 (Claude draft, human-checked) data/cesamard/labeling/gold90_labels.json

Applies exclude.json (dropped reviews) and collapses reviews whose text is identical to a
single representative (lowest row_id), using any label override listed there.
Writes data/cesamard/labeling/gold_v1.3.json with a `source` field per review.

Usage:
  python build_gold.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data" / "cesamard"
LAB = DS / "labeling"
SOURCES = [
    ("round2", LAB / "round2.json", LAB / "round2_gold_v1.3.json", "blind+corrected"),
    ("gold90", LAB / "gold90.json", LAB / "gold90_labels.json", "draft-checked"),
]


def duplicate_groups():
    df = pd.read_csv(DS / "CESAMARD-Aspect_with_local_images.csv")
    c = df[df.Complaint == 1]
    groups = c.groupby("Review_S").row_id.apply(lambda s: sorted(map(str, s)))
    return {rid: g for g in groups if len(g) > 1 for rid in g}


def main():
    ex = json.loads((LAB / "exclude.json").read_text(encoding="utf-8"))
    excluded = set(ex["exclude"])
    overrides = ex["duplicates"]["label_overrides"]
    dups = duplicate_groups()

    gold, log = {}, []
    all_labels = {}
    for name, items_path, labels_path, how in SOURCES:
        items = {str(x["row_id"]): x for x in json.loads(items_path.read_text(encoding="utf-8"))}
        labels = json.loads(labels_path.read_text(encoding="utf-8"))["labels"]
        for row, lab in labels.items():
            if row in items and lab.get("done"):
                all_labels[row] = (name, how, items[row]["domain"], lab)

    for row, (name, how, domain, lab) in sorted(all_labels.items(), key=lambda x: int(x[0])):
        if row in excluded:
            log.append(f"#{row}: excluded ({ex['exclude'][row]})")
            continue
        group = dups.get(row, [row])
        rep = group[0]
        if rep != row and rep in all_labels:
            log.append(f"#{row}: duplicate of #{rep}, merged")
            continue
        if row in overrides:
            src = overrides[row]["from"]
            lab = {**lab, "issues": all_labels[src][3]["issues"], "override_from": src}
            log.append(f"#{row}: label taken from duplicate #{src}")
        gold[row] = {"source": name, "how": how, "domain": domain, "duplicates": [r for r in group if r != row],
                     "issues": lab["issues"][:1], "marks": lab.get("marks", {}), "unsure": lab.get("unsure", False),
                     "note": lab.get("note", ""), **({"draft": lab["draft"]} if "draft" in lab else {})}

    out = {"meta": {"guide_version": "v1.3", "n": len(gold), "sources": [s[0] for s in SOURCES], "log": log},
           "labels": gold}
    (LAB / "gold_v1.3.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for line in log:
        print(line)
    by = {}
    for v in gold.values():
        by[v["source"]] = by.get(v["source"], 0) + 1
    print(f"gold_v1.3.json: {len(gold)} reviews {by}")


if __name__ == "__main__":
    main()
