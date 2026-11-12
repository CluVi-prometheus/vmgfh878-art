"""One scoring function for every experiment: the 13 metrics of the initial-experiment design (accuracy AND cost on the
same rows). Definitions: reports/초기실험_설계_v0.1.md, section 4.

  python -m cluvi.evaluate outputs/preds/heavy_val.jsonl outputs/preds/heavy_test.jsonl
  python -m cluvi.evaluate outputs/preds/heavy_test.jsonl outputs/preds/heavy_hawk200_test.jsonl --base heavy_test
  python -m cluvi.evaluate outputs/preds/heavy_test.jsonl --format       # label-free checks (issues, evidence)
  python -m cluvi.evaluate outputs/preds/heavy_test.jsonl --confusion    # needs labels; on test only after E3 is frozen
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from .common import DEFECTS, load_cards, load_split

LABEL_FIELDS = {"scope": "part_scope", "part": "part", "group6": "group6", "group5": "group5", "fine": "defect",
                "visibility": "visibility"}


def iou(a, b):
    w, h = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def macro_f1(pairs):
    """Macro-F1 over the classes that appear in the gold answers. An invalid prediction counts as wrong."""
    g, p = zip(*pairs)
    return f1_score(g, p, labels=sorted(set(g)), average="macro", zero_division=0)


def label_pairs(rows, f):
    """(gold, predicted) per review. Each field is scored on its own: a missing or out-of-list value = "INVALID" = wrong."""
    return [(s.gold[f], getattr(c, f) or "INVALID") for s, c in rows if s.gold.get(f)]


def grounding(rows):
    """Reviews whose gold visibility is localized and that have gold boxes. Top-1 predicted box (the model is told to
    put the best one first) vs the gold boxes on the SAME photo. No box / wrong photo = IoU 0, miss.
      Mean IoU      mean of max-IoU over those reviews
      IoU@0.5       share with max-IoU >= 0.5 (RefCOCO-style Acc@0.5)
      Pointing Acc  share whose top-1 box centre falls inside a gold box on the same photo (pointing game)"""
    ious, hits, pts = [], 0, 0
    for s, c in rows:
        gb = s.gold.get("boxes") or []
        if s.gold.get("visibility") != "localized" or not gb:
            continue
        best, pt = 0.0, False
        if c.boxes:
            b = c.boxes[0]
            same = [g["bbox"] for g in gb if g["image_index"] == b["image_index"]]
            best = max((iou(b["bbox"], x) for x in same), default=0.0)
            cx, cy = (b["bbox"][0] + b["bbox"][2]) / 2, (b["bbox"][1] + b["bbox"][3]) / 2
            pt = any(x[0] <= cx <= x[2] and x[1] <= cy <= x[3] for x in same)
        ious.append(best)
        hits += best >= 0.5
        pts += pt
    if not ious:
        return {}
    n = len(ious)
    return {"n_ground": n, "mean_iou": float(np.mean(ious)), "iou50": hits / n, "pointing": pts / n}


def _cost(rows, key):
    return [c.cost[key] for _, c in rows if isinstance(c.cost.get(key), (int, float))]


def score(cards, samples):
    """cards: {row_id: Card}; samples: list[Sample]. Returns a flat dict of metrics."""
    rows = [(s, cards[s.row_id]) for s in samples if s.row_id in cards]
    n = len(rows)
    m = {"n": n, "missing": len(samples) - n, "valid_rate": sum(c.valid for _, c in rows) / max(n, 1)}
    for short, f in LABEL_FIELDS.items():
        pairs = label_pairs(rows, f)
        if pairs:
            m[f"{short}_f1"] = macro_f1(pairs)
            m[f"{short}_acc"] = sum(a == b for a, b in pairs) / len(pairs)
            m[f"{short}_n"] = len(pairs)
    m.update(grounding(rows))
    vis, full = sum(_cost(rows, "visual_tokens")), sum(_cost(rows, "visual_tokens_full"))
    m["retention_pct"] = 100 * vis / full if full else None
    m["visual_tokens"] = vis / max(n, 1)
    for k in ("prefill_ms", "e2e_ms", "vision_ms", "decode_ms", "output_tokens"):
        v = _cost(rows, k)
        if v:
            m[k] = float(np.median(v))
            m[k + "_p90"] = float(np.percentile(v, 90))
    v = _cost(rows, "peak_vram_gb")
    m["peak_vram_gb"] = max(v) if v else None
    m["vlm_call_rate"] = sum(c.cost.get("vlm_calls", 0) > 0 for _, c in rows) / max(n, 1)
    return m


def bootstrap_ci(cards, samples, short="group6", reps=2000, seed=0):
    """95% CI of one macro-F1 by resampling reviews."""
    rows = [(s, cards[s.row_id]) for s in samples if s.row_id in cards and s.gold.get(LABEL_FIELDS[short])]
    pairs = label_pairs(rows, LABEL_FIELDS[short])
    rng, vals = random.Random(seed), []
    for _ in range(reps):
        vals.append(macro_f1([pairs[rng.randrange(len(pairs))] for _ in pairs]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_diff(cards_a, cards_b, samples, short="group6", reps=2000, seed=0):
    """B - A on the same reviews: macro-F1 difference with a paired bootstrap 95% CI, plus how often the predicted
    label changed between the two runs (flip rate, needs no gold)."""
    f = LABEL_FIELDS[short]
    both = [s for s in samples if s.row_id in cards_a and s.row_id in cards_b]
    flip = sum((getattr(cards_a[s.row_id], f) or "INVALID") != (getattr(cards_b[s.row_id], f) or "INVALID")
               for s in both) / max(len(both), 1)
    rows = [(s, cards_a[s.row_id], cards_b[s.row_id]) for s in both if s.gold.get(f)]
    if not rows:
        return None                                    # the two runs share no labelled review (e.g. val vs test)
    pa = label_pairs([(s, a) for s, a, _ in rows], f)
    pb = label_pairs([(s, b) for s, _, b in rows], f)
    rng, vals = random.Random(seed), []
    for _ in range(reps):
        idx = [rng.randrange(len(rows)) for _ in rows]
        vals.append(macro_f1([pb[i] for i in idx]) - macro_f1([pa[i] for i in idx]))
    return {"diff": macro_f1(pb) - macro_f1(pa), "lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5)), "n": len(rows), "flip_rate": flip}


ENV_KEYS = ("model", "quant", "max_images", "add_vision_id", "system_prompt", "decoding", "batch_size",
            "attn_implementation", "gpu", "torch", "transformers", "qwen_vl_utils", "bitsandbytes", "hawk_runtime", "split")
KNOBS = {"hawk": ("hawk_keep_ratio", "hawk_commit"), "prompt": ("prompt_version", "prompt_sha1"),
         "pixels": ("min_pixels", "max_pixels")}


def pair_check(cfg_a, cfg_b):
    """A clean paired comparison changes exactly one knob and nothing in the environment."""
    if not cfg_a or not cfg_b:
        return "config file missing -> not a valid result"
    env = [k for k in ENV_KEYS if cfg_a.get(k) != cfg_b.get(k)]
    knobs = [name for name, keys in KNOBS.items() if any(cfg_a.get(k) != cfg_b.get(k) for k in keys)]
    if env or len(knobs) > 1:
        return f"NOT a clean pair: environment differs {env}, knobs changed {knobs}"
    return f"clean pair (changed: {knobs[0] if knobs else 'nothing'})"


COLS = [("n", "n"), ("valid_rate", "valid"),
        ("scope_f1", "Scope F1"), ("part_f1", "Part F1"), ("group6_f1", "G6 F1"), ("group5_f1", "G5 F1"),
        ("fine_f1", "Fine F1"), ("visibility_f1", "Vis F1"),
        ("mean_iou", "mIoU"), ("iou50", "IoU@.5"), ("pointing", "Point"),
        ("retention_pct", "Ret %"), ("prefill_ms", "Prefill ms"), ("e2e_ms", "E2E ms"), ("peak_vram_gb", "VRAM GB")]


def print_table(results):
    print(f"{'name':30s}" + "".join(f"{h:>11s}" for _, h in COLS))
    for name, m in results.items():
        cells = []
        for k, _ in COLS:
            v = m.get(k)
            cells.append(f"{v:11.0f}" if k in ("n", "prefill_ms", "e2e_ms", "retention_pct") and isinstance(v, (int, float))
                         else f"{v:11.3f}" if isinstance(v, float) else f"{'-' if v is None else v:>11}")
        print(f"{name[:30]:30s}" + "".join(cells))


def format_report(cards, samples):
    """Label-free checks (safe to look at on any split before E3 is decided)."""
    rows = [(s, cards[s.row_id]) for s in samples if s.row_id in cards]
    print("predicted group6:", dict(Counter(c.group6 or "INVALID" for _, c in rows)))
    print("format issues:", dict(Counter(i for _, c in rows for i in c.issues)))
    spans = [c for _, c in rows if c.evidence.get("review_span")]
    if spans:
        ok = sum("review_span_not_verbatim" not in c.issues for c in spans)
        print(f"review_span verbatim: {ok}/{len(spans)}")
    print(f"truncated outputs: {sum(bool(c.cost.get('truncated')) for _, c in rows)}/{len(rows)}, "
          f"near VRAM limit: {sum(bool(c.cost.get('near_vram_limit')) for _, c in rows)}")
    tab = Counter((c.visibility, c.evidence.get("claim_support")) for _, c in rows)
    print("predicted visibility x claim_support:", {f"{v}/{cs}": n for (v, cs), n in sorted(tab.items(), key=str)})


def confusion_report(cards, samples):
    """Needs gold labels. On test, look at this only after E3 is frozen (otherwise E3 is tuned on test)."""
    rows = [(s, cards[s.row_id]) for s in samples if s.row_id in cards and s.gold.get("group6")]
    if not rows:
        print("(no group6 labels in this split)")
        return
    groups = list(DEFECTS)
    g = [s.gold["group6"] for s, _ in rows]
    p = [c.group6 or "INVALID" for _, c in rows]
    cm = confusion_matrix(g, p, labels=groups + ["INVALID"])
    print("group6 confusion (rows = answer, cols = predicted; last col = invalid)")
    print(f"{'':18s}" + "".join(f"{x[:8]:>9s}" for x in groups + ["INVALID"]) + "   recall")
    for i, gl in enumerate(groups):
        tot = cm[i].sum()
        print(f"{gl:18s}" + "".join(f"{v:9d}" for v in cm[i]) + f"   {cm[i, i] / tot if tot else 0:.2f} (n={tot})")
    print("answer group6:", dict(Counter(g)))


def load_config(path):
    c = path.with_name(path.stem + "_config.json")
    return json.loads(c.read_text(encoding="utf-8")) if c.exists() else None


def split_of(path, cfg):
    if cfg and cfg.get("split"):
        return cfg["split"]
    s = path.stem
    return next((sp for sp in ("test", "val", "train", "dev") if f"_{sp}" in s), "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--base", help="name (file stem) of the reference run for paired differences")
    ap.add_argument("--margin", type=float, default=0.05, help="non-inferiority margin for Group6 Macro-F1 (E2 vs E1)")
    ap.add_argument("--format", action="store_true", help="label-free checks (issues, verbatim, distributions)")
    ap.add_argument("--confusion", action="store_true", help="group6 confusion matrix (test: only after E3 is frozen)")
    ap.add_argument("--json", help="also write the metrics here")
    a = ap.parse_args()
    res, runs = {}, {}
    for p in map(Path, a.paths):
        cfg = load_config(p)
        samples = load_split(split_of(p, cfg))
        if cfg and "_smoke" in p.stem:
            samples = samples[: cfg.get("n")]          # a --limit run only covers the first n reviews
        cards = load_cards(p)
        runs[p.stem] = (cards, samples, cfg)
        m = score(cards, samples)
        if m["missing"]:
            print(f"!! {p.stem}: {m['missing']} of {len(samples)} reviews have no prediction (partial run)")
        if cfg is None:
            print(f"!! {p.stem}: no _config.json -> not a valid result")
        if m.get("group6_n"):
            m["group6_f1_ci"] = bootstrap_ci(cards, samples)
        for sub in ("test", "test_blind"):            # test 185: report the two human parts separately
            part = [s for s in samples if s.subset == sub]
            if part and len(part) < len(samples):
                sm = score(cards, part)
                m[f"{sub}_group6_f1"], m[f"{sub}_visibility_f1"], m[f"{sub}_n"] = \
                    sm.get("group6_f1"), sm.get("visibility_f1"), sm.get("group6_n")
        res[p.stem] = m
    print_table(res)
    for name, m in res.items():
        if "group6_f1_ci" in m:
            lo, hi = m["group6_f1_ci"]
            print(f"{name}: Group6 Macro-F1 {m['group6_f1']:.3f} [95% CI {lo:.3f}, {hi:.3f}] (n={m['group6_n']}), "
                  f"acc {m['group6_acc']:.3f}")
        if "test_n" in m:
            f3 = lambda v: "-" if v is None else f"{v:.3f}"
            print(f"  test part (n={m['test_n']}): G6 F1 {f3(m['test_group6_f1'])}, Vis F1 {f3(m['test_visibility_f1'])} | "
                  f"test_blind part (n={m['test_blind_n']}): G6 F1 {f3(m['test_blind_group6_f1'])}, "
                  f"Vis F1 {f3(m['test_blind_visibility_f1'])}")
    if a.base:
        ca, samples, cfg_a = runs[a.base]
        for name, (cb, _, cfg_b) in runs.items():
            if name == a.base:
                continue
            r = paired_diff(ca, cb, samples)
            if r is None:
                print(f"{name} - {a.base}: skipped (no reviews in common)")
                continue
            verdict = ("not distinguishable at this n" if r["lo"] <= 0 <= r["hi"]
                       else "B higher" if r["lo"] > 0 else "B lower")
            print(f"{name} - {a.base}: Group6 Macro-F1 {r['diff']:+.3f} [paired 95% CI {r['lo']:+.3f}, {r['hi']:+.3f}] "
                  f"n={r['n']} -> {verdict}; non-inferior at -{a.margin}: {r['lo'] >= -a.margin}; "
                  f"label flip rate {r['flip_rate']:.1%}")
            print(f"  {pair_check(cfg_a, cfg_b)}")
    for name, (cards, samples, _) in runs.items():
        if a.format or a.confusion:
            print(f"\n===== {name}")
        if a.format:
            format_report(cards, samples)
        if a.confusion:
            confusion_report(cards, samples)
    if a.json:
        Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
