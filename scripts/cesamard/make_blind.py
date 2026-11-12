"""Draw the blind evaluation sample from the main set (the human sees no model answer).

gold90 was labeled by checking Claude's drafts, so it leans toward Claude's answers. This sample replaces it for
measuring the labelers: 20 reviews from each Claude-Qwen agreement level k (2, 1, 0 in main_auto_v1.3.json),
so the disagreements are well covered; `weight` (stratum size / 20) restores the main-set proportions.

  python make_blind.py           # -> blind60.json (items, shuffled, Korean translation when available)
                                 #    auto/blind60_strata.json, auto/main_check_ko_part14_ids.txt (still untranslated)
  python label_server.py --set blind60 --page quick
"""
import glob
import json
import random
from pathlib import Path

LAB = Path(__file__).resolve().parents[2] / "data" / "cesamard" / "labeling"
AUTO = LAB / "auto"
PER_STRATUM = 20
SEED = 7


def main():
    auto = json.loads((LAB / "main_auto_v1.3.json").read_text(encoding="utf-8"))["labels"]
    items = {str(x["row_id"]): x for x in json.loads((LAB / "main.json").read_text(encoding="utf-8"))}
    ko = {}
    for p in glob.glob(str(AUTO / "main_check_ko_part*.json")):
        ko.update(json.loads(Path(p).read_text(encoding="utf-8")))

    rng = random.Random(SEED)
    strata, rows = {}, []
    for k in (2, 1, 0):
        pool = sorted((r for r, v in auto.items() if v["qwen_agree"] == k), key=int)
        pick = rng.sample(pool, PER_STRATUM)
        strata[str(k)] = {"n_main": len(pool), "weight": len(pool) / PER_STRATUM, "rows": pick}
        rows += pick
    rng.shuffle(rows)

    out = [{**items[r], "title_ko": ko.get(r, {}).get("title"), "text_ko": ko.get(r, {}).get("text")} for r in rows]
    (LAB / "blind60.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (AUTO / "blind60_strata.json").write_text(json.dumps({"seed": SEED, "strata": strata}, indent=1), encoding="utf-8")
    missing = [r for r in rows if r not in ko]
    if missing:
        (AUTO / "main_check_ko_part14_ids.txt").write_text("\n".join(missing) + "\n")
    by = {}
    for x in out:
        by[x["domain"]] = by.get(x["domain"], 0) + 1
    print(f"blind60.json: {len(out)} reviews {by} | untranslated {len(missing)}")


if __name__ == "__main__":
    main()
