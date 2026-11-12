"""Build the merged Hugging Face dataset: every CESAMARD-Aspect review with its original labels, the new defect
labels where they exist, and the photos.

One row per review (3,897). Original columns are kept; the new columns are empty for reviews that have no defect
label (non-complaints, dev reviews, complaints without full-resolution photos).

  defect_split   train       Claude Opus labels (main set minus the blind sample)
                 test        human gold (round2 blind + adjudicated, gold90 draft-checked)
                 test_blind  blind sample: Claude's full label + the human's blind/adjudicated group and visibility
                 dev         reviews used to write the guide (old labels, not released)
                 (empty)     no defect label
  Duplicate reviews (identical text) carry their representative's label and split (`duplicate_of`).

Configs pushed (split "full"; "all" is reserved by `datasets`):  default   with photos (Sequence(Image), ~4.3 GB)
                 text_only same rows, photo file names only

  python build_hf_dataset.py build                    # -> data/cesamard/hf_release/ (metadata.parquet, README.md, original/)
  python build_hf_dataset.py push --repo ORG/NAME     # after `hf auth login`; creates a private dataset repo
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data" / "cesamard"
LAB = DS / "labeling"
AUTO = LAB / "auto"
OUT = DS / "hf_release"
GROUP5 = {"damage": "physical_defect", "manufacturing": "physical_defect"}   # the boundary people disagree on most
VERSION = "v1.3"


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def aspects(row):
    out = []
    for a, c in (("Aspect Term", "Aspect Term Comp Analysis"), ("Aspect Term.1", "Aspect Term Comp Analysis.1"),
                 ("Aspect Term.2", "Aspect Term Comp Analysis.2"), ("Aspect Term.3", "Aspect Term Comp Analysis.3")):
        if pd.notna(row[a]):
            out.append({"aspect": str(row[a]).strip().lower().replace("sevice", "service"),
                        "complaint": None if pd.isna(row[c]) else int(row[c])})
    return out


def label_fields(issue, marks, source, split, **extra):
    g = issue["group"]
    return {"label_source": source, "defect_split": split, "part_scope": issue["scope"], "part": issue["part"],
            "part_text": issue.get("part_text") or None, "group6": g, "group5": GROUP5.get(g, g),
            "defect": issue["defect"], "visibility": issue["vis"],
            "boxes": [{"image_index": b["img"], "bbox": [round(v, 4) for v in b["bbox"]], "target": b["target"]}
                      for b in issue.get("boxes", [])],
            "customer_marks": sorted(int(k) for k, v in (marks or {}).items() if v), **extra}


def collect_labels(photo_keys):
    labels = {}

    def put(row, fields):
        assert row not in labels, f"#{row} labeled twice"
        labels[row] = fields

    # human gold 118 (+ its duplicate copies)
    for r, v in load(LAB / "gold_v1.3.json")["labels"].items():
        src = "human_round2" if v["source"] == "round2" else "human_gold90"
        f = label_fields(v["issues"][0], v.get("marks"), src, "test")
        put(r, f)
        for c in v.get("duplicates", []):
            put(c, {**f, "duplicate_of": int(r)})

    # Claude Opus on the main set, blind sample marked separately
    claude = load(AUTO / "main_claude_cmm_v1.3.json")["labels"]
    blind = load(LAB / "blind60_final_labels.json")["labels"]
    dups = load(LAB / "main_duplicates.json")
    for r, v in claude.items():
        if r in labels:   # same text as a gold review (e.g. #227 = gold #297): the human label wins, no train/test leak
            print(f"#{r}: duplicate of a gold review, Claude label not used")
            continue
        h = blind.get(r)
        extra = {"claude_unsure": bool(v.get("unsure"))}
        if h:
            hv = "not_visible" if h["photo"] == "unrelated" else h["photo"]
            extra.update({"human_check_excluded": bool(h["exclude"]),
                          "human_check_group6": None if h["exclude"] else h["group"],
                          "human_check_group5": None if h["exclude"] else GROUP5.get(h["group"], h["group"]),
                          "human_check_visibility": None if h["exclude"] else hv})
        f = label_fields(v["issues"][0], v.get("marks"), "claude_opus", "test_blind" if h else "train", **extra)
        put(r, f)
        for c in dups.get(r, []):
            same_photos = photo_keys.get(int(c), [])[:4] == photo_keys.get(int(r), [])[:4]
            put(str(c), {**f, "duplicate_of": int(r), **({} if same_photos else {"boxes": [], "boxes_dropped": True})})
    return labels


def build():
    df = pd.read_csv(DS / "CESAMARD-Aspect_with_local_images.csv")
    man = pd.read_csv(DS / "images_hr_manifest.csv")
    man = man[man.status.isin(["ok", "ok_cached"])].sort_values(["row_id", "img_idx"])
    photos = man.groupby("row_id").apply(lambda g: [Path(p).name for p in g.local_path], include_groups=False)
    photo_keys = man.groupby("row_id").key.apply(list).to_dict()
    labels = collect_labels(photo_keys)

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from make_pilot import DEV_IDS
    dev = {str(x) for x in DEV_IDS} | {str(x["row_id"]) for x in load(LAB / "pilot30.json")}
    ko = {}
    for f in ["round2_ko.json", "gold90_ko.json", "pilot30_ko.json"]:
        if (LAB / f).exists():
            ko.update({k: v for k, v in load(LAB / f).items() if not k.startswith("_")})
    for p in sorted(AUTO.glob("main_check_ko_part*.json")):
        ko.update(load(p))

    empty = {"label_source": None, "defect_split": None, "part_scope": None, "part": None, "part_text": None,
             "group6": None, "group5": None, "defect": None, "visibility": None, "boxes": [], "customer_marks": []}
    rows = []
    for _, r in df.iterrows():
        rid = str(r.row_id)
        lab = labels.get(rid) or {**empty, **({"defect_split": "dev"} if rid in dev else {})}
        rows.append({
            "row_id": int(r.row_id), "domain": r.Domain, "title": r.Title, "review": r.Review_S,
            "complaint": int(r.Complaint), "sarcastic": None if pd.isna(r.Sarcastic) else str(r.Sarcastic),
            "emotion": None if pd.isna(r.Emotion) else str(r.Emotion), "sentiment": int(r.Sentiment),
            "severity": None if pd.isna(r.Severity) else str(r.Severity),
            "aspects": aspects(r), "aspect_complaint": int(r.Complaint_A),
            "helpful_votes": None if pd.isna(r["Helpful Votes"]) else str(r["Helpful Votes"]),
            "review_date": None if pd.isna(r["Review Date"]) else str(r["Review Date"]),
            "page_url": None if pd.isna(r["#PAGE_URL"]) else str(r["#PAGE_URL"]),
            "image_urls": r.Image_urls, "image_files": photos.get(r.row_id, []),
            "title_ko": ko.get(rid, {}).get("title"), "text_ko": ko.get(rid, {}).get("text"),
            "duplicate_of": None, "claude_unsure": None, "human_check_excluded": None, "human_check_group6": None,
            "human_check_group5": None, "human_check_visibility": None, "boxes_dropped": None,
            **lab})
    meta = pd.DataFrame(rows)
    meta["duplicate_of"] = meta["duplicate_of"].astype("Int64")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "original").mkdir(exist_ok=True)
    meta.to_parquet(OUT / "metadata.parquet", index=False)
    shutil.copyfile(DS / "CESAMARD-Aspect_Dataset.csv", OUT / "original" / "CESAMARD-Aspect_Dataset.csv")
    shutil.copyfile(DS / "CESAMARD-Aspect_ReadMe.txt", OUT / "original" / "CESAMARD-Aspect_ReadMe.txt")
    shutil.copyfile(DS / "LICENSE", OUT / "original" / "LICENSE")
    s = meta.defect_split.value_counts(dropna=False).to_dict()
    print(f"{len(meta)} reviews | defect_split {s} | photos {sum(map(len, meta.image_files))} | "
          f"korean {meta.text_ko.notna().sum()} | duplicates {meta.duplicate_of.notna().sum()}")
    print("group6 (labeled):", meta.group6.value_counts().to_dict())
    print("visibility (labeled):", meta.visibility.value_counts().to_dict())
    print(f"wrote {OUT}")


def push(repo):
    from datasets import Dataset, Image, Sequence
    from huggingface_hub import HfApi
    meta = pd.read_parquet(OUT / "metadata.parquet")
    img_dir = DS / "images_hr"
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    text = Dataset.from_pandas(meta, preserve_index=False)
    text.push_to_hub(repo, config_name="text_only", split="full", private=True)
    full = text.map(lambda x: {"images": [str(img_dir / f) for f in x["image_files"]]})
    full = full.cast_column("images", Sequence(Image()))
    full.push_to_hub(repo, config_name="default", split="full", private=True, max_shard_size="500MB")
    api.upload_folder(folder_path=OUT / "original", path_in_repo="original", repo_id=repo, repo_type="dataset")
    # our dataset card replaces the generated one (its configs block points at data/ and text_only/)
    card = (OUT / "README.md").read_text(encoding="utf-8").replace("ORG/NAME", repo)
    api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=repo, repo_type="dataset")
    files = api.list_repo_files(repo, repo_type="dataset")
    print(f"pushed https://huggingface.co/datasets/{repo} (private) | {len(files)} files, e.g. {files[:6]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "push"])
    ap.add_argument("--repo")
    a = ap.parse_args()
    build() if a.cmd == "build" else push(a.repo)
