"""Build the human check set for the main labeling from triage.py's output.

  main_auto_v1.3.json (tier "check")  ->  main_check.json              items, sorted by domain, with Korean translations
                                          auto/main_check_drafts.json  Claude's label as the draft, the Qwen answers as
                                                                       `alts`, the reason as `why` (shown by the label tool)
Translations come from auto/main_check_ko_part*.json ({row_id: {title, text}}) when present; rerun after adding them.

Usage:
  python make_check.py
  python label_server.py --set main_check --drafts data/cesamard/labeling/auto/main_check_drafts.json
"""
import json
from pathlib import Path

LAB = Path(__file__).resolve().parents[2] / "data" / "cesamard" / "labeling"
AUTO = LAB / "auto"
DOMAINS = ["Electronics", "Fashion", "Edible", "Books", "Misc"]
QWEN_NAMES = {"v1.3b": "Qwen 1차", "v1.3c": "Qwen 2차"}


def why(v):
    reasons = []
    if v["qwen_agree"] == 0:
        reasons.append("Qwen 두 번 모두 Claude와 다른 문제 유형을 골랐어요")
    elif v["qwen_agree"] == 1:
        reasons.append("Qwen 한 번만 Claude와 같은 문제 유형을 골랐어요")
    if v.get("rotated_box"):
        reasons.append("사진이 회전된 채 저장돼 있어 박스 위치를 꼭 확인해 주세요")
    return " / ".join(reasons)


def main():
    auto = json.loads((LAB / "main_auto_v1.3.json").read_text(encoding="utf-8"))["labels"]
    items = {str(x["row_id"]): x for x in json.loads((LAB / "main.json").read_text(encoding="utf-8"))}
    ko = {}
    for p in sorted(AUTO.glob("main_check_ko_part*.json")):
        ko.update(json.loads(p.read_text(encoding="utf-8")))

    rows = sorted((r for r, v in auto.items() if v["tier"] == "check"),
                  key=lambda r: (DOMAINS.index(items[r]["domain"]) if items[r]["domain"] in DOMAINS else 9, int(r)))
    out_items, drafts = [], {}
    for r in rows:
        tr = ko.get(r, {})
        out_items.append({**items[r], "title_ko": tr.get("title"), "text_ko": tr.get("text")})
        v = auto[r]
        drafts[r] = {"issues": v["issues"], "marks": v["marks"], "unsure": v["unsure"], "note": v["note"], "why": why(v),
                     "alts": [{"who": QWEN_NAMES.get(t, t), **a} for t, a in v["qwen"].items()]}
    (LAB / "main_check.json").write_text(json.dumps(out_items, ensure_ascii=False, indent=1), encoding="utf-8")
    (AUTO / "main_check_drafts.json").write_text(
        json.dumps({"meta": {"set": "main_check", "from": "main_auto_v1.3.json"}, "labels": drafts},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    by = {}
    for x in out_items:
        by[x["domain"]] = by.get(x["domain"], 0) + 1
    print(f"main_check.json: {len(out_items)} reviews {by} | missing ko {sum(not x['text_ko'] for x in out_items)}")


if __name__ == "__main__":
    main()
