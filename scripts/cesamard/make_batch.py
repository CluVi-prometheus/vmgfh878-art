"""Draw the next labeling batch from gold-150, skipping reviews used in earlier batches.

Each batch is labeled by a human in rounds; after a round, the guide may be revised.
A round labeled under a guide that later changed becomes a dev (auto-test) set.

Usage:
  python make_batch.py round2                 # 30 reviews: Electronics 9 / Fashion 9 / Edible 7 / Books 4 / Misc 1
  python make_batch.py round3 --seed 3
  python make_batch.py gold90 --all          # every gold review not used by an earlier batch

Writes data/cesamard/labeling/<name>.json. Merges <name>_ko.json (Korean literal
translations, by Claude) as title_ko / text_ko when present; rerun after adding it.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from make_pilot import DS, MAX_IMAGES, OUT, clean

COUNTS = {"Electronics": 9, "Fashion": 9, "Edible": 7, "Books": 4, "Misc": 1}
SKIP_SUFFIXES = ("_ko", "_labels", "_labels.bak")


def used_ids(exclude_name):
    ids = set()
    for f in OUT.glob("*.json"):
        if f.stem == exclude_name or f.stem.endswith(SKIP_SUFFIXES):
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(data, list):
            ids |= {x["row_id"] for x in data}
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--all", action="store_true", help="take every remaining gold review instead of a 30-review sample")
    args = ap.parse_args()

    gold = [int(x) for x in (OUT / "gold150_ids.txt").read_text().split()]
    used = used_ids(args.name)
    df = pd.read_csv(DS / "CESAMARD-Aspect_with_local_images.csv")
    pool = df[df.row_id.isin(gold) & ~df.row_id.isin(used)]
    man = pd.read_csv(DS / "images_hr_manifest.csv")
    man = man[man.status.isin(["ok", "ok_cached"])].sort_values(["row_id", "img_idx"])
    imgs = man.groupby("row_id")[["key", "local_path", "width", "height"]].apply(
        lambda g: g.to_dict("records"))

    if args.all:
        batch = pool.sample(frac=1, random_state=args.seed)
    else:
        batch = pd.concat(
            pool[pool.Domain == d].sample(min(n, (pool.Domain == d).sum()), random_state=args.seed)
            for d, n in COUNTS.items())
        batch = batch.sample(frac=1, random_state=args.seed)

    ko_path = OUT / f"{args.name}_ko.json"
    ko = json.loads(ko_path.read_text(encoding="utf-8")) if ko_path.exists() else {}
    items = []
    for _, r in batch.iterrows():
        ims = imgs[r.row_id]
        tr = ko.get(str(r.row_id), {})
        items.append({
            "row_id": int(r.row_id),
            "domain": r.Domain,
            "title": clean(r.Title),
            "text": clean(r.Review_S),
            "title_ko": tr.get("title"),
            "text_ko": tr.get("text"),
            "n_images_total": len(ims),
            "images": [
                {"idx": i, "key": im["key"], "src": im["local_path"],
                 "w": int(im["width"]), "h": int(im["height"])}
                for i, im in enumerate(ims[:MAX_IMAGES])
            ],
        })
    (OUT / f"{args.name}.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(args.name, batch.Domain.value_counts().to_dict(),
          "| images", sum(len(x["images"]) for x in items),
          "| excluded (earlier batches)", len(used & set(gold)),
          "| missing ko", sum(not x["text_ko"] for x in items))


if __name__ == "__main__":
    main()
