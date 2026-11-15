"""E3-L val verdict: the success criteria fixed in reports/E3_실험_제안.html, checked against E1 on the same 100 val rows.

  python -m cluvi.judge_e3 outputs/preds/heavy_lora-e3l-epoch2_val.jsonl            # vs outputs/preds/heavy_val.jsonl

Criteria (val, silver labels):
  1 Group6 Macro-F1 >= E1 + 0.05, function recall >= 0.65, non_visual recall >= 0.45
  2 Visibility Macro-F1 >= 0.69 (simple-rule reference 0.688), not_visible recall >= 0.60
  3 Scope Macro-F1 >= 0.395 (R0)
  4 format kept: valid >= 95 %, review_span verbatim >= 90 %, reviews with a box problem <= E1
  5 sure 70 / unsure 30 reported apart: a gain only on the unsure rows = suspect "learned Claude's habits"
Also watched (risk list): localized recall (E1 0.87) for over-correction to not_visible, E2E latency of the adapter.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .common import OUT, PRED_DIR, load_cards, load_split
from .evaluate import label_pairs, paired_diff, score
from .lora_data import unsure_flags

BOX_ISSUES = ("bad_box", "box_out_of_range", "box_whole_picture", "box_copied", "localized_without_box",
              "box_but_not_localized")


def recall(rows, f, cls):
    pairs = [(g, p) for g, p in label_pairs(rows, f) if g == cls]
    return sum(g == p for g, p in pairs) / len(pairs) if pairs else None


def measures(cards, samples):
    rows = [(s, cards[s.row_id]) for s in samples if s.row_id in cards]
    m = score(cards, samples)
    spans = [c for _, c in rows if c.evidence.get("review_span")]
    e2e = [c.cost["e2e_ms"] for _, c in rows if isinstance(c.cost.get("e2e_ms"), (int, float))]
    return {"n": m["n"], "missing": m["missing"], "group6_f1": m.get("group6_f1"), "group6_acc": m.get("group6_acc"),
            "visibility_f1": m.get("visibility_f1"), "scope_f1": m.get("scope_f1"), "part_f1": m.get("part_f1"),
            "rec_function": recall(rows, "group6", "function"), "rec_non_visual": recall(rows, "group6", "non_visual"),
            "rec_not_visible": recall(rows, "visibility", "not_visible"),
            "rec_localized": recall(rows, "visibility", "localized"),
            "valid_rate": m["valid_rate"],
            "verbatim_rate": sum("review_span_not_verbatim" not in c.issues for c in spans) / len(spans) if spans else 0,
            "box_problem_reviews": sum(any(i in BOX_ISSUES for i in c.issues) for _, c in rows),
            "box_issues": dict(Counter(i for _, c in rows for i in c.issues if i in BOX_ISSUES)),
            "pointing": m.get("pointing"), "mean_iou": m.get("mean_iou"),
            "e2e_ms_median": float(np.median(e2e)) if e2e else None,
            "pred_group6": dict(Counter(c.group6 or "INVALID" for _, c in rows))}


def judge(e1, e3):
    """(criterion, E1 value, E3 value, threshold text, pass)."""
    t = []
    t.append(("Group6 Macro-F1", e1["group6_f1"], e3["group6_f1"], f">= {e1['group6_f1'] + 0.05:.3f} (E1 +0.05)",
              e3["group6_f1"] >= e1["group6_f1"] + 0.05))
    t.append(("function recall", e1["rec_function"], e3["rec_function"], ">= 0.65", e3["rec_function"] >= 0.65))
    t.append(("non_visual recall", e1["rec_non_visual"], e3["rec_non_visual"], ">= 0.45", e3["rec_non_visual"] >= 0.45))
    t.append(("Visibility Macro-F1", e1["visibility_f1"], e3["visibility_f1"], ">= 0.69", e3["visibility_f1"] >= 0.69))
    t.append(("not_visible recall", e1["rec_not_visible"], e3["rec_not_visible"], ">= 0.60",
              e3["rec_not_visible"] >= 0.60))
    t.append(("Scope Macro-F1", e1["scope_f1"], e3["scope_f1"], ">= 0.395", e3["scope_f1"] >= 0.395))
    t.append(("valid rate", e1["valid_rate"], e3["valid_rate"], ">= 0.95", e3["valid_rate"] >= 0.95))
    t.append(("review_span verbatim", e1["verbatim_rate"], e3["verbatim_rate"], ">= 0.90", e3["verbatim_rate"] >= 0.90))
    t.append(("reviews with box problem", e1["box_problem_reviews"], e3["box_problem_reviews"],
              f"<= {e1['box_problem_reviews']} (E1)", e3["box_problem_reviews"] <= e1["box_problem_reviews"]))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("e3", help="E3 predictions on val (jsonl)")
    ap.add_argument("--e1", default=str(PRED_DIR / "heavy_val.jsonl"))
    ap.add_argument("--json", default=str(OUT / "eval" / "e3l_val_judge.json"))
    a = ap.parse_args()
    val, flags = load_split("val"), unsure_flags()
    c1, c3 = load_cards(Path(a.e1)), load_cards(Path(a.e3))
    m1, m3 = measures(c1, val), measures(c3, val)
    if m3["missing"]:
        print(f"!! {m3['missing']} val reviews have no E3 prediction (partial run): verdict is provisional")
    rows = judge(m1, m3)
    f = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))     # noqa: E731
    print(f"\nE3-L val verdict ({Path(a.e3).stem} vs {Path(a.e1).stem}, n={m3['n']})")
    print(f"{'criterion':26s}{'E1':>9s}{'E3-L':>9s}   {'threshold':24s}pass")
    for name, v1, v3, th, ok in rows:
        print(f"{name:26s}{f(v1):>9s}{f(v3):>9s}   {th:24s}{'PASS' if ok else 'fail'}")
    n_pass = sum(r[4] for r in rows)
    print(f"-> {n_pass}/{len(rows)} criteria met")

    sub = {}
    for name, keep in (("sure", lambda s: not flags.get(s.row_id)), ("unsure", lambda s: flags.get(s.row_id))):
        part = [s for s in val if keep(s)]
        a1, a3 = measures(c1, part), measures(c3, part)
        sub[name] = {"n": len(part), "e1": a1, "e3": a3}
        print(f"{name:7s} (n={len(part):3d}): G6 acc {f(a1['group6_acc'])} -> {f(a3['group6_acc'])} | G6 F1 "
              f"{f(a1['group6_f1'])} -> {f(a3['group6_f1'])} | Vis F1 {f(a1['visibility_f1'])} -> {f(a3['visibility_f1'])}")
    gain = {k: (sub[k]["e3"]["group6_acc"] or 0) - (sub[k]["e1"]["group6_acc"] or 0) for k in sub}
    habit = gain["unsure"] > 0.02 and gain["sure"] <= 0
    print(f"sure-row gain {gain['sure']:+.3f}, unsure-row gain {gain['unsure']:+.3f} -> "
          f"{'SUSPECT: gain only on unsure rows (Claude habits)' if habit else 'ok (gain not only on unsure rows)'}")
    pd_ = paired_diff(c1, c3, val)
    print(f"paired Group6 Macro-F1 {pd_['diff']:+.3f} [95% CI {pd_['lo']:+.3f}, {pd_['hi']:+.3f}], "
          f"label flip rate {pd_['flip_rate']:.1%}")
    print(f"watch: localized recall {f(m1['rec_localized'])} -> {f(m3['rec_localized'])} | Pointing "
          f"{f(m1['pointing'])} -> {f(m3['pointing'])} | box issues {m1['box_issues']} -> {m3['box_issues']} | "
          f"E2E median {f(m1['e2e_ms_median'])} -> {f(m3['e2e_ms_median'])} ms")
    print(f"predicted group6 E1 {m1['pred_group6']}\n               E3 {m3['pred_group6']}")
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps({"e3": a.e3, "e1": a.e1, "e1_metrics": m1, "e3_metrics": m3,
                                        "criteria": [{"name": r[0], "e1": r[1], "e3": r[2], "threshold": r[3],
                                                      "pass": bool(r[4])} for r in rows],
                                        "n_pass": n_pass, "subsets": sub, "habit_suspect": habit, "paired_g6": pd_},
                                       ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
