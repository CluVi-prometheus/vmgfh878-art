"""Merge the auto-labeler's per-part outputs for one set.

  auto/<set>_claude_cmm_<ver>_part*.json -> auto/<set>_claude_cmm_<ver>.json   (drafts)
  auto/<set>_ko_part*.json               -> <set>_ko.json                       (translations)

Then rerun make_batch.py for the set so the translations are merged into <set>.json.

Usage:
  python merge_parts.py gold90 v1.3
"""
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[2] / "data" / "cesamard" / "labeling"


def main():
    name, ver = sys.argv[1], sys.argv[2]
    items = {str(x["row_id"]) for x in json.loads((LAB / f"{name}.json").read_text(encoding="utf-8"))}
    auto = LAB / "auto"

    parts = sorted(auto.glob(f"{name}_claude_cmm_{ver}_part*.json"))
    labels, metas = {}, []
    for p in parts:
        d = json.loads(p.read_text(encoding="utf-8"))
        metas.append(d["meta"])
        labels.update(d["labels"])
    missing = items - set(labels)
    out = {"meta": {"set": name, "labeler": "claude-cmm", "guide_version": ver,
                    "model": sorted({m.get("model") for m in metas}), "parts": [p.name for p in parts]},
           "labels": {k: labels[k] for k in sorted(labels, key=int)}}
    (auto / f"{name}_claude_cmm_{ver}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    ko = {}
    for p in sorted(auto.glob(f"{name}_ko_part*.json")):
        ko.update(json.loads(p.read_text(encoding="utf-8")))
    ko = {"_note": "Claude 직역. 해석·판단은 넣지 않음. 개인정보는 번역에서 생략.", **{k: ko[k] for k in sorted(ko, key=int)}}
    (LAB / f"{name}_ko.json").write_text(json.dumps(ko, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"drafts: {len(labels)} reviews from {len(parts)} parts | missing {sorted(missing) or 'none'}")
    print(f"translations: {len(ko) - 1} reviews | missing {sorted(items - set(ko)) or 'none'}")


if __name__ == "__main__":
    main()
