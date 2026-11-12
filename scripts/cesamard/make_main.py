"""Build the main labeling set: every complaint review with full-res images that is not in
gold-150 or dev-59, not excluded, and not a duplicate copy (identical text; the lowest
row_id is kept and the copies inherit its label later).

Writes data/cesamard/labeling/main.json (same item format as the other batches, no translations)
and main_duplicates.json ({representative row_id: [copy row_ids]}).

Usage:
  python make_main.py
"""
import json

import pandas as pd

from make_pilot import DEV_IDS, DS, MAX_IMAGES, OUT, clean


def main():
    df = pd.read_csv(DS / "CESAMARD-Aspect_with_local_images.csv")
    man = pd.read_csv(DS / "images_hr_manifest.csv")
    man = man[man.status.isin(["ok", "ok_cached"])].sort_values(["row_id", "img_idx"])
    imgs = man.groupby("row_id")[["key", "local_path", "width", "height"]].apply(lambda g: g.to_dict("records"))

    gold = {int(x) for x in (OUT / "gold150_ids.txt").read_text().split()}
    excluded = {int(x) for x in json.loads((OUT / "exclude.json").read_text(encoding="utf-8"))["exclude"]}
    c = df[(df.Complaint == 1) & df.row_id.isin(imgs.index)].sort_values("row_id")
    groups = c.groupby("Review_S").row_id.apply(list)
    rep_of = {r: g[0] for g in groups for r in g}

    pool = c[~c.row_id.isin(gold | set(DEV_IDS) | excluded)]
    reps = pool[pool.row_id.map(rep_of) == pool.row_id]
    copies = {int(r): [int(x) for x in groups[t] if x != r] for r, t in zip(reps.row_id, reps.Review_S) if len(groups[t]) > 1}

    items = []
    for _, r in reps.iterrows():
        ims = imgs[r.row_id]
        items.append({
            "row_id": int(r.row_id), "domain": r.Domain, "title": clean(r.Title), "text": clean(r.Review_S),
            "title_ko": None, "text_ko": None, "n_images_total": len(ims),
            "images": [{"idx": i, "key": im["key"], "src": im["local_path"], "w": int(im["width"]), "h": int(im["height"])}
                       for i, im in enumerate(ims[:MAX_IMAGES])],
        })
    (OUT / "main.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "main_duplicates.json").write_text(json.dumps(copies, indent=1), encoding="utf-8")
    print(f"main.json: {len(items)} reviews, {sum(len(x['images']) for x in items)} images | "
          f"{len(copies)} have duplicate copies ({sum(map(len, copies.values()))} copies) | "
          f"domains {reps.Domain.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
