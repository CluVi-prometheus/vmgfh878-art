"""Compare two label files for the same set (e.g. human vs Claude auto-labels).

Only reviews marked done in the reference file are compared. Issues are paired
within a review by maximizing a score (same defect 2, same group 1, same part 1;
pairs scoring 0 are never paired).

Usage:
  python compare_labels.py <reference_labels.json> <candidate_labels.json> [--items pilot30.json] [--main-only]
"""
import argparse
import itertools
import json
from collections import Counter
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))["labels"]


def score(a, b):
    s = 0
    if a.get("group") == b.get("group"):
        s += 1
        if a.get("defect") == b.get("defect"):
            s += 1
    if a.get("part") == b.get("part"):
        s += 1
    return s


def pair(ref, cand):
    """Best one-to-one pairing (issues per review are few, so brute force)."""
    best, best_pairs = -1, []
    small, large, flip = (ref, cand, False) if len(ref) <= len(cand) else (cand, ref, True)
    for perm in itertools.permutations(range(len(large)), len(small)):
        pairs = [(i, j) for i, j in enumerate(perm) if score(small[i], large[j]) > 0]
        total = sum(score(small[i], large[j]) for i, j in pairs)
        if total > best:
            best, best_pairs = total, pairs
    return [(j, i) if flip else (i, j) for i, j in best_pairs]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def inside(pt, box):
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--items", default=None)
    ap.add_argument("--main-only", action="store_true",
                    help="keep only the candidate's main issue (for references with one issue per review)")
    args = ap.parse_args()
    ref, cand = load(args.reference), load(args.candidate)
    if args.main_only:
        for v in cand.values():
            v["issues"] = [i for i in v["issues"] if i.get("main")][:1] or v["issues"][:1]
    domains = {}
    if args.items:
        domains = {str(x["row_id"]): x["domain"] for x in json.loads(Path(args.items).read_text(encoding="utf-8"))}

    rows = [k for k, v in ref.items() if v.get("done")]
    n_ref = n_cand = n_pair = 0
    agree = Counter()
    vis_conf = Counter()
    group_exact = 0
    group_tp = Counter(); group_fp = Counter(); group_fn = Counter()
    box_ious, box_point = [], []
    detail = []

    for r in rows:
        a, b = ref[r]["issues"], cand.get(r, {}).get("issues", [])
        n_ref += len(a); n_cand += len(b)
        ga, gb = {i["group"] for i in a}, {i["group"] for i in b}
        group_exact += ga == gb
        for g in ga | gb:
            if g in ga and g in gb: group_tp[g] += 1
            elif g in gb: group_fp[g] += 1
            else: group_fn[g] += 1
        pairs = pair(a, b)
        n_pair += len(pairs)
        for i, j in pairs:
            x, y = a[i], b[j]
            for f in ("part", "scope", "group", "defect", "vis"):
                agree[f] += x.get(f) == y.get(f)
            vis_conf[(x.get("vis"), y.get("vis"))] += 1
            if x.get("vis") == "localized" and y.get("vis") == "localized":
                for hb in x["boxes"]:
                    same_img = [cb for cb in y["boxes"] if cb["img"] == hb["img"]]
                    box_ious.append(max((iou(hb["bbox"], cb["bbox"]) for cb in same_img), default=0.0))
                    box_point.append(any(inside(((cb["bbox"][0] + cb["bbox"][2]) / 2, (cb["bbox"][1] + cb["bbox"][3]) / 2), hb["bbox"])
                                         for cb in same_img))
        fmt = lambda i: f"{i.get('part')}|{i.get('group')}>{i.get('defect')}|{i.get('vis')}({len(i.get('boxes', []))})"
        detail.append((r, domains.get(r, ""), " + ".join(map(fmt, a)) or "-", " + ".join(map(fmt, b)) or "-",
                       ga == gb, cand.get(r, {}).get("unsure"), ref[r].get("unsure")))

    pct = lambda n, d: f"{n}/{d} ({100 * n / d:.0f}%)" if d else "-"
    print(f"reviews compared: {len(rows)} | issues ref {n_ref}, cand {n_cand}, paired {n_pair}")
    print(f"review-level group set exact match: {pct(group_exact, len(rows))}")
    f1s = []
    for g in sorted(set(group_tp) | set(group_fp) | set(group_fn)):
        tp, fp, fn = group_tp[g], group_fp[g], group_fn[g]
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0
        f1s.append(f1)
        print(f"  group {g:18s} tp {tp} fp {fp} fn {fn}  F1 {f1:.2f}")
    print(f"  group macro-F1 {sum(f1s) / len(f1s):.2f}")
    print("paired-issue agreement:", ", ".join(f"{f} {pct(agree[f], n_pair)}" for f in ("part", "scope", "group", "defect", "vis")))
    print("vis (ref -> cand):", dict(vis_conf))
    if box_ious:
        print(f"boxes (ref boxes on issues both call localized): n={len(box_ious)}, "
              f"IoU>=0.5 {pct(sum(v >= 0.5 for v in box_ious), len(box_ious))}, mean IoU {sum(box_ious) / len(box_ious):.2f}, "
              f"pointing {pct(sum(box_point), len(box_point))}")
    print("\nrow | domain | group-set match | ref | cand | cand unsure | ref unsure")
    for r, d, x, y, m, cu, ru in detail:
        print(f"#{r} {d} {'OK ' if m else 'XX '}\n   ref : {x}\n   cand: {y}  {'[cand unsure]' if cu else ''}{'[ref unsure]' if ru else ''}")


if __name__ == "__main__":
    main()
