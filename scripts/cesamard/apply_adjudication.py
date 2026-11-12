"""Apply a reviewed list of corrections to a blind label file, keeping the original intact.

Each change targets the review's main issue (issue 0). "boxes" is "keep", "clear",
or {"retarget": "defect"|"evidence"}. Box coordinates are never moved.

Usage:
  python apply_adjudication.py <blind_labels.json> <adjudication.json> <out_labels.json>
"""
import json
import sys
from pathlib import Path


def main():
    src, patch_path, out = map(Path, sys.argv[1:4])
    data = json.loads(src.read_text(encoding="utf-8"))
    patch = json.loads(patch_path.read_text(encoding="utf-8"))
    log = []
    for row, ch in patch["changes"].items():
        issue = data["labels"][row]["issues"][0]
        before = {k: issue.get(k) for k in ch["set"]}
        issue.update(ch["set"])
        if ch["boxes"] == "clear":
            issue["boxes"] = []
        elif isinstance(ch["boxes"], dict):
            for b in issue["boxes"]:
                b["target"] = ch["boxes"]["retarget"]
        if issue["vis"] != "localized":
            assert not issue["boxes"], f"#{row}: non-localized issue still has boxes"
        log.append({"row_id": row, "before": before, "after": ch["set"], "boxes": ch["boxes"], "reason": ch["reason"]})
    data["meta"].update({"guide_version": patch["guide_version"], "adjudicated_from": src.name,
                         "adjudication": patch_path.name, "n_changed": len(log)})
    data["meta"].pop("rev", None)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    for x in log:
        print(f"#{x['row_id']}: {x['before']} -> {x['after']} | boxes {x['boxes']}")
    print(f"wrote {out} ({len(log)} reviews changed)")


if __name__ == "__main__":
    main()
