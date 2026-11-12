"""End to end on one complaint: review + photos -> analysis card -> agent answers a counselor's question.

  python -m cluvi.demo --model mock                       # no GPU
  python -m cluvi.demo --model heavy --row 3 --question "Can we offer a replacement?"
"""
import argparse
import json
from dataclasses import asdict

from .agent import run_agent
from .common import load_split
from .models import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mock", help="mock | light | heavy (the same VLM analyzes and drives the agent)")
    ap.add_argument("--row", type=int, help="row_id from the test split (default: the first one)")
    ap.add_argument("--question", default="What is wrong, and what can we offer the customer?")
    a = ap.parse_args()

    samples = load_split("test")
    s = next(x for x in samples if x.row_id == a.row) if a.row else samples[0]
    vlm = load_model(a.model)

    print(f"== complaint {s.row_id} ({s.domain}, {len(s.images)} photos)\n{s.title}\n{s.review[:300]}\n")
    card = vlm.analyze(s)
    print("== analysis card")
    print(json.dumps({k: v for k, v in asdict(card).items() if k != "raw"}, ensure_ascii=False, indent=1))
    print("== gold", json.dumps(s.gold, ensure_ascii=False))
    print(f"\n== agent  Q: {a.question}")
    out = run_agent(vlm, s, card, a.question)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
