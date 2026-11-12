"""Draw the gold-150 set and its blind pilot-30 subset for defect labeling.

gold 150 = complaint reviews with full-res images, excluding the 59 dev reviews,
stratified by domain (Electronics 46 / Fashion 45 / Edible 35 / Books 21 / Misc 3).
pilot 30 = a stratified subset of gold (9 / 9 / 7 / 4 / 1), labeled blind by a human.

Writes data/cesamard/labeling/{gold150_ids.txt, pilot30.json}.
If data/cesamard/labeling/pilot30_ko.json exists, its Korean translations
(literal, by Claude) are merged in as title_ko / text_ko.
"""
import html
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data" / "cesamard"
OUT = DS / "labeling"
SEED = 42
MAX_IMAGES = 4

DEV_IDS = [
    425, 279, 442, 432, 10, 229, 2259, 2525, 2559, 2704, 2837, 2963, 2363, 2853,
    3124, 3108, 2744, 2856, 1252, 829, 731, 1310, 2658, 2657,
    17, 443, 104, 267, 332, 2504, 2382, 2254, 3483, 1901, 2119, 3038,
    1135, 1217, 1160, 885, 1537, 1715, 2667,
    149, 308, 251, 2362, 2191, 3416, 3068, 2456, 2411, 2239, 2394, 3007,
    1351, 1155, 1063, 1109,
]
GOLD_N = {"Electronics": 46, "Fashion": 45, "Edible": 35, "Books": 21, "Misc": 3}
PILOT_N = {"Electronics": 9, "Fashion": 9, "Edible": 7, "Books": 4, "Misc": 1}


def clean(text):
    text = html.unescape(str(text)).replace("\\n", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def main():
    assert len(set(DEV_IDS)) == 59
    df = pd.read_csv(DS / "CESAMARD-Aspect_with_local_images.csv")
    man = pd.read_csv(DS / "images_hr_manifest.csv")
    man = man[man.status.isin(["ok", "ok_cached"])].sort_values(["row_id", "img_idx"])
    imgs = man.groupby("row_id")[["key", "local_path", "width", "height"]].apply(
        lambda g: g.to_dict("records"))

    pool = df[(df.Complaint == 1) & df.row_id.isin(imgs.index) & ~df.row_id.isin(DEV_IDS)]
    gold = pd.concat(
        pool[pool.Domain == d].sample(n, random_state=SEED) for d, n in GOLD_N.items())
    pilot = pd.concat(
        gold[gold.Domain == d].sample(n, random_state=SEED) for d, n in PILOT_N.items())
    pilot = pilot.sample(frac=1, random_state=SEED)  # mix domains in labeling order

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gold150_ids.txt").write_text(
        "\n".join(str(i) for i in sorted(gold.row_id)) + "\n", encoding="utf-8")

    ko_path = OUT / "pilot30_ko.json"
    ko = json.loads(ko_path.read_text(encoding="utf-8")) if ko_path.exists() else {}

    items = []
    for _, r in pilot.iterrows():
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
    (OUT / "pilot30.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")

    print("gold", gold.Domain.value_counts().to_dict())
    print("pilot", pilot.Domain.value_counts().to_dict())
    print("pilot images", sum(len(x["images"]) for x in items),
          "| reviews with >4 images", sum(x["n_images_total"] > MAX_IMAGES for x in items))


if __name__ == "__main__":
    main()
