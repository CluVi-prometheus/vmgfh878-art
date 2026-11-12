"""E3-L training examples: E1's exact input + the answer's first three fields as the only target.

  input   same as E1: prompt vlm-v0.3.3, the first 4 photos (EXIF-upright), 256-1280 tokens per photo, Qwen chat
          template with numbered pictures -> built by the same functions inference uses (cluvi.models.encode_chat)
  target  the start of the reply exactly as E1 writes it (all 100 E1 val replies start with '```json\\n{\\n  "category"'):
              ```json
              {
                "category": "function/fit_size",
                "part": "whole",
                "visibility": "diffuse",          <- loss stops after this token; no end-of-reply token,
                                                     so the model keeps writing boxes + evidence as before
  weight  1.0, or 0.5 for the 239 train rows Claude marked unsure (claude_unsure)

Labels: train 881 = Claude Opus silver labels (metadata label_source = claude_opus for all 881).

  python -m cluvi.lora_data check --n 4      # CPU only: show which tokens get loss + boundary checks
  python -m cluvi.lora_data lengths          # CPU only: input length of every train review -> outputs/lora/train_lengths.json
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .common import META, OUT, PROMPT_VERSION, Sample, load_split, parts_for
from .models import MAX_IMAGES, MAX_PIXELS, MIN_PIXELS, encode_chat, open_images, review_prompt, visual_tokens_at

LORA_DIR = OUT / "lora"
UNSURE_WEIGHT = 0.5
FIELDS = ("category", "part", "visibility")
TAIL = ',\n  "boxes": ['    # what follows in E1's replies; used only to find the natural last token of the target


def head_values(s: Sample):
    part = s.gold["part"]
    if part not in {p for ps in parts_for(s.domain).values() for p in ps}:
        part = "other"      # prompt rule 3: "A part not in the list -> other" (train row 2654: Electronics body_fabric)
    return {"category": f"{s.gold['group6']}/{s.gold['defect']}", "part": part, "visibility": s.gold["visibility"]}


def head_text(values):
    return '```json\n{\n' + ",\n".join(f'  "{k}": "{values[k]}"' for k in FIELDS)


def head_tokens(tok, values):
    """(ids, field) of the target. field per token: 0 = fixed scaffold ("```json", keys, quotes), 1/2/3 = a token
    inside the category / part / visibility value. The text is tokenised together with what really follows, then cut
    after the token holding the closing quote of the visibility value, so the last token is the one the model would
    generate there (e.g. '",\\n'), not an artificial lone quote."""
    head = head_text(values)
    enc = tok(head + TAIL, add_special_tokens=False, return_offsets_mapping=True)
    spans, pos = [], 0
    for k in FIELDS:
        a = head.index(f'"{k}": "', pos) + len(f'"{k}": "')
        spans.append((a, a + len(values[k])))
        pos = a
    ids, field = [], []
    for i, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]):
        if a >= len(head):
            break
        ids.append(i)
        field.append(next((n for n, (x, y) in enumerate(spans, 1) if a < y and b > x), 0))
    return ids, field


def unsure_flags():
    df = pd.read_parquet(META, columns=["row_id", "claude_unsure"])
    return {int(r): bool(u) for r, u in zip(df.row_id, df.claude_unsure) if u is not None and u == u}


def encode_example(proc, s: Sample, prompt_version=PROMPT_VERSION, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
                   max_images=MAX_IMAGES):
    """One training example on CPU. Returns dict: enc (prompt + target tokens), n_prompt, target, field."""
    import torch
    imgs = open_images(s, max_images)
    prompt, _ = review_prompt(s, imgs, prompt_version, min_pixels, max_pixels)
    enc, _ = encode_chat(proc, prompt, imgs, min_pixels, max_pixels)
    ids, field = head_tokens(proc.tokenizer, head_values(s))
    n_prompt = enc["input_ids"].shape[1]
    t = torch.tensor([ids], dtype=enc["input_ids"].dtype)
    enc["input_ids"] = torch.cat([enc["input_ids"], t], 1)
    enc["attention_mask"] = torch.cat([enc["attention_mask"], torch.ones_like(t)], 1)
    return {"enc": enc, "n_prompt": n_prompt, "target": torch.tensor(ids), "field": torch.tensor(field),
            "n_visual": int(enc["image_grid_thw"].prod(-1).sum()) // 4 if "image_grid_thw" in enc else 0}


def photo_sizes(s: Sample, max_images=MAX_IMAGES):
    """(w, h) of the first photos after EXIF rotation, without decoding pixels."""
    from PIL import Image
    out = []
    for p in s.images[:max_images]:
        with Image.open(p) as im:
            w, h = im.size
            out.append((h, w) if im.getexif().get(0x0112) in (5, 6, 7, 8) else (w, h))
    return out


def lengths(proc, samples, prompt_version=PROMPT_VERSION):
    """Input tokens per review = text tokens of the full chat prompt + visual tokens (from photo sizes, no pixels)."""
    from .models import seen_sizes
    out = {}
    for s in samples:
        sizes = photo_sizes(s)
        text = review_prompt(s, [type("I", (), {"size": z})() for z in sizes], prompt_version)[0]
        msgs = [{"role": "user", "content": [{"type": "image"} for _ in sizes] + [{"type": "text", "text": text}]}]
        chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, add_vision_id=True)
        n_vis = visual_tokens_at(sizes)
        n_text = len(proc.tokenizer(chat, add_special_tokens=False)["input_ids"]) - len(sizes)   # one pad per photo
        out[s.row_id] = {"visual": n_vis, "text": n_text, "total": n_vis + n_text,
                         "seen": [list(x) for x in seen_sizes(sizes)]}
    return out


def _check(proc, n):
    """CPU checks, no model weights needed:
      1 target text == E1's reply start for the same labels (taken from E1 val replies)
      2 tokenising prompt and target separately == tokenising them as one text (no token merges across the seam)
      3 which tokens get loss, shown token by token"""
    from .common import load_cards
    tok = proc.tokenizer
    e1 = load_cards(OUT / "preds" / "heavy_val.jsonl")
    val = {s.row_id: s for s in load_split("val")}
    ok1 = 0
    for rid, c in list(e1.items())[:20]:
        if not c.valid:
            continue
        v = {"category": f"{c.group6}/{c.defect}", "part": c.part, "visibility": c.visibility}
        ids, _ = head_tokens(tok, v)
        real = tok(c.raw, add_special_tokens=False)["input_ids"][: len(ids)]
        ok1 += ids == real
        if ids != real:
            print(f"  [1] row {rid}: target tokens differ from E1's reply start\n      {tok.convert_ids_to_tokens(ids)}"
                  f"\n      {tok.convert_ids_to_tokens(real)}")
    print(f"[1] target == E1 reply start (same labels, token ids): {ok1}/{sum(c.valid for c in list(e1.values())[:20])}")

    train = load_split("train")
    flags = unsure_flags()
    picks = train[:n - 1] + [s for s in train if s.row_id == 2654]       # include the one remapped part code
    for s in picks:
        ex = encode_example(proc, s)
        enc, P, tgt, field = ex["enc"], ex["n_prompt"], ex["target"], ex["field"]
        imgs = open_images(s)
        prompt, _ = review_prompt(s, imgs)
        joint, _ = encode_chat(proc, prompt, imgs)
        same_prompt = joint["input_ids"].tolist() == enc["input_ids"][:, :P].tolist()
        # the last prompt tokens + the target, decoded to text and tokenised again in one go: must give the same ids
        seam = tok(tok.decode(enc["input_ids"][0, P - 8:]), add_special_tokens=False)["input_ids"] == \
            enc["input_ids"][0, P - 8:].tolist()
        w = UNSURE_WEIGHT if flags.get(s.row_id) else 1.0
        print(f"\n--- row {s.row_id} ({s.domain}, {len(imgs)} photo(s), unsure={flags.get(s.row_id)}, weight {w}) ---")
        print(f"  input {enc['input_ids'].shape[1]} tokens = prompt {P} (visual {ex['n_visual']}) + target {len(tgt)} "
              f"| prompt identical to inference path: {same_prompt} | no merge across seam: {seam}")
        print(f"  gold: {head_values(s)}")
        print("  prompt tail (no loss): " + repr(tok.decode(enc["input_ids"][0, P - 6:P])))
        names = ["scaffold", "category", "part", "visibility"]
        print("  loss tokens: " + " ".join(f"{tok.decode([i])!r}" + ("" if f == 0 else f"<{names[f][:3]}>")
                                            for i, f in zip(tgt.tolist(), field.tolist())))
        print(f"  loss tokens: {len(tgt)} (value tokens {int((field > 0).sum())}); "
              f"pixel_values {tuple(enc['pixel_values'].shape)}, grid {enc['image_grid_thw'].tolist()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "lengths"])
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    from .run_hawk import use_runtime
    use_runtime("lora_data")
    from transformers import AutoProcessor
    from .models import VLM_IDS
    proc = AutoProcessor.from_pretrained(VLM_IDS["heavy"][0], min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    if a.cmd == "check":
        _check(proc, a.n)
    else:
        train = load_split("train")
        ln = lengths(proc, train)
        LORA_DIR.mkdir(parents=True, exist_ok=True)
        (LORA_DIR / "train_lengths.json").write_text(json.dumps(ln), encoding="utf-8")
        tot = sorted(v["total"] for v in ln.values())
        vis = sorted(v["visual"] for v in ln.values())
        q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]
        print(f"train {len(tot)}: input tokens median {q(tot, .5)}, p90 {q(tot, .9)}, max {tot[-1]} | visual median "
              f"{q(vis, .5)}, p90 {q(vis, .9)}, max {vis[-1]}, >4000: {sum(v > 4000 for v in vis)}")
        top = sorted(ln.items(), key=lambda kv: -kv[1]["total"])[:10]
        print("10 longest (probe set): " + ", ".join(f"{k}:{v['total']}" for k, v in top))


if __name__ == "__main__":
    sys.exit(main())
