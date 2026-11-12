"""Cascade = simulated OFFLINE from saved predictions. No model is re-run: each sample walks the stages in order and
stops at the first stage whose conf["exit"] >= its threshold (the last stage always answers). The cost of every stage
it passed through is added up.

  python -m cluvi.cascade --stages text heavy                 # 2-stage
  python -m cluvi.cascade --stages text light heavy           # 3-stage
"""
import argparse
import itertools
from copy import deepcopy

from .common import PRED_DIR, load_cards, load_split
from .evaluate import print_table, score

TAUS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def run_cascade(stage_cards, taus):
    """stage_cards: list of {row_id: Card}, cheapest first. taus: one threshold per stage except the last."""
    out = {}
    for rid in stage_cards[-1]:
        cost = {"vlm_calls": 0, "visual_tokens": 0, "visual_tokens_full": 0, "prefill_ms": 0.0, "e2e_ms": 0.0}
        for i, cards in enumerate(stage_cards):
            c = cards[rid]
            for k in cost:
                cost[k] += c.cost.get(k, 0)
            if i == len(stage_cards) - 1 or (c.conf.get("exit") or 0) >= taus[i]:
                break
        c = deepcopy(c)
        c.cost, c.model = cost, f"exit@{c.model}"
        out[rid] = c
    return out


def oracle(stage_cards, samples):
    """Upper bound: the cheapest stage that gets group6 right (if none does, the last stage)."""
    gold = {s.row_id: s.gold["group6"] for s in samples}
    out = {}
    for rid in stage_cards[-1]:
        k = next((i for i, cards in enumerate(stage_cards) if cards[rid].group6 == gold.get(rid)), len(stage_cards) - 1)
        taus = [-1 if i == k else 2 for i in range(len(stage_cards) - 1)]    # -1 = always exit here, 2 = never
        out.update(run_cascade([{rid: cards[rid]} for cards in stage_cards], taus))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["text", "heavy"])
    ap.add_argument("--split", default="gold")
    a = ap.parse_args()
    samples = load_split(a.split)
    stage_cards = [load_cards(PRED_DIR / f"{st}_{a.split}.jsonl") for st in a.stages]
    common = set.intersection(*(set(c) for c in stage_cards))
    stage_cards = [{r: c for r, c in cards.items() if r in common} for cards in stage_cards]

    res = {f"{st} only": score(c, samples) for st, c in zip(a.stages, stage_cards)}
    for taus in itertools.product(TAUS, repeat=len(a.stages) - 1):
        res["cascade tau=" + ",".join(map(str, taus))] = score(run_cascade(stage_cards, taus), samples)
    res["oracle router (upper bound)"] = score(oracle(stage_cards, samples), samples)
    print_table(res)


if __name__ == "__main__":
    main()
