"""Validate one auto-labeler part file against its ID file (used by the labeling agents before they finish).

  python check_part.py main 07        # auto/main_claude_cmm_v1.3_part07.json vs auto/main_part07_ids.txt
  python check_part.py evalset 01 fable   # auto/evalset_fable_v1.3_part01.json (model comparison runs)

Checks: every ID present and no extras; exactly one issue with main=true; codes valid for the domain and scope;
part_text set when part=other; localized => at least one box, otherwise none; bbox inside 0-1 with x1<x2, y1<y2;
image indices exist. Prints the problems (exit 1) or a short summary (exit 0).
"""
import json
import sys
from collections import Counter
from pathlib import Path

from qwen_label import DEFECTS, OTHER_PARTS, PRODUCT_PARTS

LAB = Path(__file__).resolve().parents[2] / "data" / "cesamard" / "labeling"
VER = "v1.3"


def main():
    name, part = sys.argv[1], sys.argv[2]
    tag = sys.argv[3] if len(sys.argv) > 3 else "claude_cmm"
    ids = (LAB / "auto" / f"{name}_part{part}_ids.txt").read_text().split()
    items = {str(x["row_id"]): x for x in json.loads((LAB / f"{name}.json").read_text(encoding="utf-8"))}
    labels = json.loads((LAB / "auto" / f"{name}_{tag}_{VER}_part{part}.json").read_text(encoding="utf-8"))["labels"]

    errs = [f"missing #{r}" for r in ids if r not in labels] + [f"extra #{r}" for r in labels if r not in ids]
    for r in ids:
        if r not in labels:
            continue
        lab, item = labels[r], items[r]
        issues = lab.get("issues", [])
        if len(issues) != 1 or not issues[0].get("main"):
            errs.append(f"#{r}: need exactly one issue with main=true")
            continue
        i = issues[0]
        scopes = {"product": PRODUCT_PARTS.get(item["domain"], []) + ["whole"], **OTHER_PARTS, "other": ["other"]}
        if i.get("part") not in scopes.get(i.get("scope"), []):
            errs.append(f"#{r}: part {i.get('part')!r} not valid for scope {i.get('scope')!r} / {item['domain']}")
        if i.get("part") == "other" and not i.get("part_text"):
            errs.append(f"#{r}: part=other needs part_text")
        if i.get("defect") not in DEFECTS.get(i.get("group"), []):
            errs.append(f"#{r}: defect {i.get('defect')!r} not in group {i.get('group')!r}")
        if i.get("vis") not in ("localized", "diffuse", "not_visible"):
            errs.append(f"#{r}: vis {i.get('vis')!r}")
        boxes = i.get("boxes", [])
        if (i.get("vis") == "localized") != bool(boxes):
            errs.append(f"#{r}: localized needs >=1 box, other vis needs 0 boxes")
        n_img = len(item["images"])
        for b in boxes:
            x1, y1, x2, y2 = b.get("bbox", [0, 0, 0, 0])
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                errs.append(f"#{r}: bad bbox {b.get('bbox')}")
            if b.get("img") not in range(n_img):
                errs.append(f"#{r}: img {b.get('img')} not in 0..{n_img - 1}")
            if b.get("target") not in ("defect", "evidence"):
                errs.append(f"#{r}: target {b.get('target')!r}")
        for k in lab.get("marks", {}):
            if int(k) not in range(n_img):
                errs.append(f"#{r}: marks index {k} not in 0..{n_img - 1}")
    if errs:
        print("\n".join(errs))
        sys.exit(1)
    iss = [labels[r]["issues"][0] for r in ids]
    print(f"OK {len(ids)} reviews | vis {dict(Counter(i['vis'] for i in iss))} | "
          f"unsure {sum(labels[r].get('unsure', False) for r in ids)} | marks {sum(bool(labels[r].get('marks')) for r in ids)}")


if __name__ == "__main__":
    main()
