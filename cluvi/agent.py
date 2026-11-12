"""Counselor agent: Planner -> Tool -> ... -> Answer -> Verifier. It reads the analysis Card; it never re-classifies.

The cascade only decides HOW the Card is made (which model). The agent sits after it and does not care.
Tools are mock for now (12/22~ the backend API replaces ORDERS / POLICY with real calls; the loop stays the same).
"""
import json
import re
from dataclasses import asdict

from .common import Card, Sample
from .models import open_images

POLICY = {   # mock CS policy per defect group: action, return window, whether photo evidence is required
    "damage": {"action": "replacement or refund", "window_days": 10, "needs_photo": True},
    "contamination": {"action": "refund + report to quality team", "window_days": 7, "needs_photo": True},
    "manufacturing": {"action": "replacement", "window_days": 10, "needs_photo": True},
    "not_as_described": {"action": "return pickup + refund", "window_days": 10, "needs_photo": True},
    "function": {"action": "warranty claim / technician visit", "window_days": 30, "needs_photo": False},
    "non_visual": {"action": "route to service or logistics team; no automatic refund", "window_days": 0, "needs_photo": False},
}


def get_order(row_id):
    return {"order_id": f"ORD-{row_id:05d}", "delivered_days_ago": row_id % 20, "price_inr": 299 + (row_id * 37) % 2700,
            "status": "delivered"}


PLANNER = """You help an e-commerce customer-service agent handle a complaint with photos.
Analysis card (from the vision model): {card}
Agent's question: {question}

Tools:
- get_order(): order and delivery info for this complaint
- get_policy(): the return/refund policy for this card's defect group
- look_again(question, image): ask the vision model about one photo (image = index)
{history}
Reply with JSON only, either {{"tool": "<name>", "args": {{...}}}} to call one tool, or {{"answer": "<reply to the agent, citing the card and tool results>"}}."""


def verify(card: Card, results: dict):
    """Rule checks on evidence. Warnings go to the human agent; they do not change the answer."""
    w = []
    pol, order = results.get("get_policy"), results.get("get_order")
    if card.group6 is None:
        w.append("analysis invalid: the model gave no defect group")
    if pol and pol["needs_photo"] and card.visibility == "not_visible":
        w.append("policy needs photo evidence but the defect is not visible in the photos -> check manually")
    if pol and order and order["delivered_days_ago"] > pol["window_days"]:
        w.append(f"return window passed ({order['delivered_days_ago']} days > {pol['window_days']})")
    return w


def run_agent(llm, sample: Sample, card: Card, question: str, max_steps=4):
    card_view = {k: v for k, v in asdict(card).items() if k not in ("raw", "cost")}
    results, history, cost = {}, "", {"vlm_calls": 0, "visual_tokens": 0}
    answer = None
    for _ in range(max_steps):
        reply, c, *_ = llm.chat(PLANNER.format(card=json.dumps(card_view), question=question, history=history), max_new_tokens=300)
        cost["vlm_calls"] += 1
        m = re.search(r"\{.*\}", reply, re.S)
        try:
            act = json.loads(m.group(0)) if m else {"answer": reply}
        except json.JSONDecodeError:
            act = {"answer": reply}
        if "answer" in act or "tool" not in act:
            answer = act.get("answer", reply)
            break
        name, args = act["tool"], act.get("args") or {}
        if name == "get_order":
            out = get_order(sample.row_id)
        elif name == "get_policy":
            out = POLICY.get(card.group6, {"action": "unknown group", "window_days": 0, "needs_photo": False})
        elif name == "look_again":
            imgs = open_images(sample)
            i = int(args.get("image", 0)) if str(args.get("image", 0)).isdigit() else 0
            out, c, *_ = llm.chat(str(args.get("question", "Describe the defect.")), imgs[i:i + 1] if imgs else [], max_new_tokens=150)
            cost["vlm_calls"] += 1
            cost["visual_tokens"] += c["visual_tokens"]
        else:
            out = f"unknown tool {name}"
        results[name] = out
        history += f"\nTOOL RESULT {name}: {json.dumps(out, ensure_ascii=False)}"
    return {"answer": answer or "(no answer within step limit)", "tools": results, "warnings": verify(card, results), "cost": cost}
