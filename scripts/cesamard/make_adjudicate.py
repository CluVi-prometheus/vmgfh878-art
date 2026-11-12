"""Adjudicate the blind sample: the reviews where the human's blind answer and Claude's differ.

In the blind round the human used "manufacturing" for any bad-quality product (torn in use, stopped working,
colour differs from the photo); on 2026-10-07 they confirmed the guide's rules instead (torn → damage, broken down →
function, differs from the ad → not_as_described, torn/crushed packaging → damage with part = packaging). Each
disagreement is shown again with both answers unlabeled and in random order, with those rules on screen.

  python make_adjudicate.py build    # -> blind60_adj.json (items with `candidates`), only differing reviews
  python label_server.py --set blind60_adj --page quick
  python make_adjudicate.py merge    # blind60_labels.json + blind60_adj_labels.json -> blind60_final_labels.json
"""
import json
import random
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[2] / "data" / "cesamard" / "labeling"
AUTO = LAB / "auto"
PHOTO_OF_VIS = {"localized": "localized", "diffuse": "diffuse", "not_visible": "not_visible"}
SEED = 11


def load(name):
    return json.loads((LAB / name).read_text(encoding="utf-8"))


def build():
    human = {r: v for r, v in load("blind60_labels.json")["labels"].items() if v.get("done") and not v.get("exclude")}
    claude = json.loads((AUTO / "main_claude_cmm_v1.3.json").read_text(encoding="utf-8"))["labels"]
    items = {str(x["row_id"]): x for x in load("blind60.json")}
    rng = random.Random(SEED)
    out = []
    for r, h in human.items():
        c = claude[r]["issues"][0]
        cand = {}
        for field, hv, cv in (("group", h["group"], c["group"]),
                              ("photo", "not_visible" if h["photo"] == "unrelated" else h["photo"], PHOTO_OF_VIS[c["vis"]])):
            pair = [hv, cv] if hv != cv else [hv]
            rng.shuffle(pair)
            cand[field] = pair
        if len(cand["group"]) > 1 or len(cand["photo"]) > 1:
            out.append({**items[r], "candidates": cand})
    rng.shuffle(out)
    (LAB / "blind60_adj.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"blind60_adj.json: {len(out)} reviews | group differs {sum(len(x['candidates']['group']) > 1 for x in out)}, "
          f"photo differs {sum(len(x['candidates']['photo']) > 1 for x in out)}")


def merge():
    base = load("blind60_labels.json")
    adj = {r: v for r, v in load("blind60_adj_labels.json")["labels"].items() if v.get("done")}
    labels = {}
    for r, v in base["labels"].items():
        a = adj.get(r)
        labels[r] = {**v, **({k: a[k] for k in ("group", "alt_group", "ambiguous", "photo", "exclude", "note")}
                             if a else {}), "adjudicated": bool(a), "blind": {k: v.get(k) for k in ("group", "photo", "exclude")}}
    out = {"meta": {"set": "blind60", "adjudicated": len(adj), "rules": "guide v1.3 + 2026-10-07 confirmations"},
           "labels": labels}
    (LAB / "blind60_final_labels.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    changed = sum(labels[r]["group"] != labels[r]["blind"]["group"] for r in adj)
    print(f"blind60_final_labels.json: {len(adj)} adjudicated, group changed on {changed}")


if __name__ == "__main__":
    {"build": build, "merge": merge}[sys.argv[1]]()
