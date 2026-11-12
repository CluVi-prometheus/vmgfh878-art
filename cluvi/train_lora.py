"""E3-L: QLoRA on Qwen2.5-VL-7B that learns only the first three answer fields (category, part, visibility).

What changes vs E1: one LoRA adapter. Everything else is E1: the same 4-bit load (cluvi.models.QwenVLM), prompt
vlm-v0.3.3, photos, resizing, chat template, HAWK's transformers 4.52.0 runtime (HAWK itself off).
Settings = reports/E3_실험_제안.html: LoRA on the LLM's q/k/v/o_proj in all 28 layers (112 modules; the ViT's attention
is named attn.qkv / attn.proj and is not touched), r 8, alpha 16, dropout 0.1, AdamW lr 1e-4, cosine with 10 % warm-up,
3 epochs, batch 1 x accumulation 8, gradient checkpointing, ViT + lm_head frozen, loss weight 0.5 on Claude-unsure rows.
Training data and loss mask: cluvi/lora_data.py.

  python -m cluvi.train_lora --probe 10     # GPU check: 10 longest + 5 random train reviews -> VRAM, s/review, time
                                            #   estimate; then 1 update + save + val head on 3 rows (outputs/lora/_probe)
  python -m cluvi.train_lora                # full run -> outputs/lora/e3l/epoch{1,2,3}/ + val head check per epoch
  python -m cluvi.train_lora --resume       # continue from outputs/lora/e3l/last/ after a stop
  python -m cluvi.run_hawk --model heavy --split val --adapter outputs/lora/e3l/epoch2    # full E1-style run

Val head check: greedy decoding stopped right after the visibility field (48 new tokens). Greedy decoding with the same
repetition penalty writes the same first tokens as a full run, so these three labels are exactly what the full run
answers, in about a third of the time. It is used only to pick the epoch.
Epoch rule (fixed before training): the highest mean of val Group6, Visibility and Scope macro-F1; ties -> earlier epoch.
"""
import argparse
import hashlib
import json
import math
import random
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from .common import PRED_DIR, PROMPT_VERSION, Card, load_cards, load_split, parse_vlm, prompt_text
from .lora_data import LORA_DIR, UNSURE_WEIGHT, encode_example, unsure_flags
from .predict import _hms

TARGET = r"model\.language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
N_MODULES = 4 * 28
HEAD_NEW_TOKENS = 48
VOLATILE = {"started", "resumed"}


def log(out_dir, line):
    print(line, flush=True)
    if out_dir:
        with open(out_dir / "train_log.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")


def attach_lora(vlm, a):
    """Insert LoRA layers into the already-loaded E1 model and check that they landed where intended."""
    import torch
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(a.seed)                                   # LoRA A init is random (B starts at 0)
    cfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none", target_modules=TARGET)
    peft_model = get_peft_model(vlm.model, cfg)
    names = [n for n, m in vlm.model.named_modules() if hasattr(m, "lora_A")]
    trainable = [(n, p) for n, p in peft_model.named_parameters() if p.requires_grad]
    if len(names) != N_MODULES or any("language_model" not in n for n in names):
        sys.exit(f"LoRA landed on {len(names)} modules (expected {N_MODULES}, LLM only): {names[:3]} ...")
    if any("lora_" not in n for n, _ in trainable):
        sys.exit(f"non-LoRA parameters are trainable: {[n for n, _ in trainable if 'lora_' not in n][:3]}")
    vlm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    vlm.visual.gradient_checkpointing = False    # frozen ViT builds no graph; checkpointing would only re-run it
    total = sum(p.numel() for p in vlm.model.parameters())
    n_train = sum(p.numel() for _, p in trainable)
    return peft_model, {"lora_modules": len(names), "trainable_params": n_train, "trainable_share": n_train / total,
                        "adapter_dtype": str(trainable[0][1].dtype)}


def loss_on(vlm, ex):
    """Mean cross-entropy over the target tokens of one review, plus per-token hits (argmax == target).
    Logits are computed only at the target positions (not the whole ~3,000-token input) to save VRAM."""
    import torch.nn.functional as F
    dev, enc = vlm.model.device, ex["enc"]
    P, tgt = ex["n_prompt"], ex["target"].to(dev)
    out = vlm.model.model(input_ids=enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev),
                          pixel_values=enc["pixel_values"].to(dev), image_grid_thw=enc["image_grid_thw"].to(dev),
                          use_cache=False, return_dict=True)
    h = out.last_hidden_state[0, P - 1: P - 1 + len(tgt)]     # position t predicts token t+1
    logits = vlm.model.lm_head(h).float()
    return F.cross_entropy(logits, tgt), (logits.argmax(-1) == tgt).cpu()


def field_hits(hit, field):
    """1/0 per field (category, part, visibility): every token of that value predicted right (teacher-forced)."""
    return [bool(hit[field == f].all()) for f in (1, 2, 3)]


def prefetched(proc, seq, prompt_version):
    """Encode the next review on a CPU thread while the GPU works on the current one."""
    with ThreadPoolExecutor(1) as pool:
        futs = [pool.submit(encode_example, proc, seq[0], prompt_version)] if seq else []
        for i, s in enumerate(seq):
            if i + 1 < len(seq):
                futs.append(pool.submit(encode_example, proc, seq[i + 1], prompt_version))
            yield s, futs[i].result()
            futs[i] = None


# ---- val head check ----
HEAD_RE = {k: re.compile(rf'"{k}"\s*:\s*"([^"]*)"') for k in ("category", "part", "visibility")}


def head_card(reply, s, model_name):
    d = {k: m.group(1) for k, rx in HEAD_RE.items() if (m := rx.search(reply))}
    card = parse_vlm(json.dumps(d), s, Card(s.row_id, model_name), [])
    card.raw = reply
    return card


def head_metrics(cards, samples, flags):
    from .evaluate import label_pairs, score

    def recall(rows, f, cls):
        pairs = [(g, p) for g, p in label_pairs(rows, f) if g == cls]
        return sum(g == p for g, p in pairs) / len(pairs) if pairs else None

    def block(subset):
        m = score(cards, subset)
        rows = [(s, cards[s.row_id]) for s in subset if s.row_id in cards]
        out = {k: m.get(k) for k in ("n", "valid_rate", "group6_f1", "visibility_f1", "scope_f1", "part_f1",
                                     "group6_acc")}
        out.update({"rec_function": recall(rows, "group6", "function"),
                    "rec_non_visual": recall(rows, "group6", "non_visual"),
                    "rec_not_visible": recall(rows, "visibility", "not_visible"),
                    "rec_localized": recall(rows, "visibility", "localized")})
        out["select"] = sum(out[k] or 0 for k in ("group6_f1", "visibility_f1", "scope_f1")) / 3
        return out
    return {"all": block(samples), "sure": block([s for s in samples if not flags.get(s.row_id)]),
            "unsure": block([s for s in samples if flags.get(s.row_id)])}


def fmt_head(m):
    a = m["all"]
    f = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"     # noqa: E731
    return (f"G6 {f(a['group6_f1'])} | Vis {f(a['visibility_f1'])} | Scope {f(a['scope_f1'])} | select {f(a['select'])}"
            f" | rec function {f(a['rec_function'], 2)} non_visual {f(a['rec_non_visual'], 2)} not_visible "
            f"{f(a['rec_not_visible'], 2)} localized {f(a['rec_localized'], 2)} | valid {f(a['valid_rate'], 2)} | "
            f"G6 acc sure {f(m['sure'].get('group6_acc'))} / unsure {f(m['unsure'].get('group6_acc'))}")


def val_head(vlm, samples, flags, out_dir, tag):
    from .models import open_images, review_prompt
    vlm.model.eval()
    cards, t0 = {}, time.time()
    for i, s in enumerate(samples, 1):
        imgs = open_images(s, vlm.max_images)
        prompt, _ = review_prompt(s, imgs, vlm.prompt_version, vlm.min_pixels, vlm.max_pixels)
        cards[s.row_id] = head_card(vlm.chat(prompt, imgs, max_new_tokens=HEAD_NEW_TOKENS)[0], s, f"heavy+{tag}")
        if i % 20 == 0 or i == len(samples):
            el = time.time() - t0
            log(out_dir, f"  [val head {tag} {i}/{len(samples)}] {el / i:.1f}s/review | left ~{_hms(el / i * (len(samples) - i))}")
    vlm.model.train()
    m = head_metrics(cards, samples, flags)
    m["seconds"] = round(time.time() - t0, 1)
    return m, cards


# ---- state ----
def save_state(peft_model, opt, sched, step, skipped, path):
    import torch
    tmp = path.with_name(path.name + "_tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    peft_model.save_pretrained(tmp)
    torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "step": step, "skipped": skipped},
               tmp / "trainer_state.pt")
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)


def load_state(peft_model, opt, sched, path):
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    set_peft_model_state_dict(peft_model, load_file(str(path / "adapter_model.safetensors")))
    st = torch.load(path / "trainer_state.pt", weights_only=False)
    opt.load_state_dict(st["optimizer"])
    sched.load_state_dict(st["scheduler"])
    return st["step"], st["skipped"]


# ---- probe ----
def probe(a, vlm, peft_model, info):
    """Fit check before the long run: forward+backward on the longest train reviews (VRAM) and a few random ones
    (speed), then one real update, a save and a 3-review val head check, so the whole pipeline has run once."""
    import torch
    from transformers import get_cosine_schedule_with_warmup
    ln = json.loads((LORA_DIR / "train_lengths.json").read_text(encoding="utf-8"))
    train = load_split("train")
    longest = sorted(train, key=lambda s: -ln[str(s.row_id)]["total"])[: a.probe]
    rest = [s for s in train if s not in longest]
    rnd = random.Random(1).sample(rest, 6)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"[probe] LoRA {info} | GPU {total_gb:.1f} GB", flush=True)
    rows = []
    for k, s in enumerate(rnd[:1] + longest + rnd[1:]):
        ex = encode_example(vlm.proc, s, a.prompt)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        try:
            loss, _ = loss_on(vlm, ex)
            loss.backward()
            ok, lv = True, float(loss)
        except torch.cuda.OutOfMemoryError:
            ok, lv = False, None
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        peft_model.zero_grad(set_to_none=True)
        r = {"row_id": s.row_id, "tokens": int(ex["enc"]["input_ids"].shape[1]), "visual": ex["n_visual"], "ok": ok,
             "s": round(dt, 2), "loss": lv, "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
             "reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
        if not ok:
            torch.cuda.empty_cache()
        if k:                                               # the first one is a warm-up (CUDA kernels)
            rows.append(r)
        print(f"[probe {k}/{len(longest) + 5}{' warm-up' if not k else ''}] row {r['row_id']} | {r['tokens']} tokens "
              f"(visual {r['visual']}) | {'OK' if ok else 'OUT OF MEMORY'} | {r['s']}s | loss {lv} | "
              f"peak {r['peak_gb']} GB (reserved {r['reserved_gb']} / {total_gb:.1f})", flush=True)
    good = [r for r in rows if r["ok"]]
    est = None
    if len(good) >= 2:                                      # seconds ~ a + b * tokens, applied to every train review
        xs, ys = [r["tokens"] for r in good], [r["s"] for r in good]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(sum((x - mx) ** 2 for x in xs), 1e-9)
        per_epoch = sum(my + b * (v["total"] - mx) for v in ln.values())
        est = {"s_per_token": b, "epoch_h": per_epoch / 3600, "train_h": per_epoch * a.epochs / 3600}
        print(f"[probe] estimate: {per_epoch / 3600:.2f} h per epoch -> {a.epochs} epochs {est['train_h']:.2f} h "
              f"(+ val head checks ~{a.epochs * 100 * 4.5 / 3600:.1f} h)", flush=True)
    # one real update + save + val head on 3 rows: the full pipeline once
    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, 0, 1)
    for s in rnd[1:4]:
        loss, _ = loss_on(vlm, encode_example(vlm.proc, s, a.prompt))
        (loss / 3).backward()
    torch.nn.utils.clip_grad_norm_([p for p in peft_model.parameters() if p.requires_grad], 1.0)
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    pdir = LORA_DIR / "_probe"
    pdir.mkdir(parents=True, exist_ok=True)
    save_state(peft_model, opt, sched, 1, [], pdir / "adapter")
    val = load_split("val")[:3]
    m, cards = val_head(vlm, val, unsure_flags(), None, "probe")
    for s in val:
        print(f"[probe val head] row {s.row_id}: {cards[s.row_id].raw!r}", flush=True)
    (pdir / "probe.json").write_text(json.dumps({"rows": rows, "estimate": est, "lora": info}, indent=1),
                                     encoding="utf-8")
    worst = max((r["reserved_gb"] for r in rows), default=None)
    print(f"[probe] done | all fit: {all(r['ok'] for r in rows)} | max reserved {worst} / {total_gb:.1f} GB | "
          f"saved {pdir / 'adapter'} (test it with: python -m cluvi.run_hawk --model heavy --split dev --limit 2 "
          f"--adapter {pdir / 'adapter'})", flush=True)


# ---- training ----
def train(a, vlm, peft_model, info):
    import torch
    from transformers import get_cosine_schedule_with_warmup
    out_dir = LORA_DIR / a.run
    flags = unsure_flags()
    samples = load_split("train")
    if a.sure_only:
        samples = [s for s in samples if not flags.get(s.row_id)]
    excluded = []
    if a.max_visual_tokens:
        ln = json.loads((LORA_DIR / "train_lengths.json").read_text(encoding="utf-8"))
        excluded = sorted(s.row_id for s in samples if ln[str(s.row_id)]["visual"] > a.max_visual_tokens)
        samples = [s for s in samples if s.row_id not in set(excluded)]
    n, per_epoch = len(samples), math.ceil(len(samples) / a.accum)
    total_steps = per_epoch * a.epochs
    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, round(a.warmup * total_steps), total_steps)
    weight = {s.row_id: (a.unsure_weight if flags.get(s.row_id) else 1.0) for s in samples}

    cfg = {"run": a.run, "experiment": "E3-L", "base_vs_e1": "E1 + LoRA adapter only",
           "lora": {"r": a.r, "alpha": a.alpha, "dropout": a.dropout, "target_modules": TARGET, **info},
           "optim": {"name": "AdamW", "lr": a.lr, "weight_decay": 0.0, "schedule": "cosine", "warmup": a.warmup,
                     "max_grad_norm": 1.0, "epochs": a.epochs, "batch": 1, "accum": a.accum, "steps": total_steps},
           "loss": "mean CE over the head tokens (```json ... visibility value), x weight; logits only at those positions",
           "data": {"split": "train", "n": n, "unsure": sum(bool(flags.get(s.row_id)) for s in samples),
                    "unsure_weight": a.unsure_weight, "sure_only": a.sure_only,
                    "max_visual_tokens": a.max_visual_tokens, "excluded": excluded, "seed": a.seed},
           "gradient_checkpointing": "LLM only (non-reentrant); ViT frozen, no graph",
           "prompt_sha1": hashlib.sha1(prompt_text(a.prompt).encode()).hexdigest()[:12],
           "epoch_rule": "max mean(val G6, Vis, Scope macro-F1) on the val head check; tie -> earlier epoch",
           "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           **{f"e1_{k}": v for k, v in vlm.run_config().items()}}
    step, skipped = 0, []
    if a.resume:
        old = json.loads((out_dir / "train_config.json").read_text(encoding="utf-8"))
        diff = sorted(k for k in set(old) | set(cfg) if k not in VOLATILE and old.get(k) != cfg.get(k))
        if diff:
            sys.exit(f"--resume with different settings: {diff}")
        step, skipped = load_state(peft_model, opt, sched, out_dir / "last")
        cfg["started"], cfg["resumed"] = old["started"], time.strftime("%Y-%m-%d %H:%M:%S")
        log(out_dir, f"[{a.run}] resumed at step {step}/{total_steps}")
    elif out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"{out_dir} already has a run. Use --resume, or another --run name.")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")

    val = load_split("val")
    e1_val = load_cards(PRED_DIR / "heavy_val.jsonl")
    e1_m = head_metrics(e1_val, val, flags)
    log(out_dir, f"[{a.run}] {n} train reviews ({cfg['data']['unsure']} unsure x{a.unsure_weight}), {a.epochs} epochs, "
                 f"{per_epoch} updates/epoch, {total_steps} updates | LoRA {info['lora_modules']} modules, "
                 f"{info['trainable_params']:,} params ({100 * info['trainable_share']:.3f} %)")
    log(out_dir, f"[E1 val, same scoring] {fmt_head(e1_m)}")
    peft_model.train()
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    t_run, done_reviews, val_s, val_spent = time.time(), 0, None, 0.0
    recent = []                                                    # (loss, cat, part, vis) of the last 80 reviews
    while step < total_steps:
        epoch, g0 = divmod(step, per_epoch)
        order = list(range(n))
        random.Random(a.seed + epoch).shuffle(order)                # same order on resume
        groups = [order[i:i + a.accum] for i in range(0, n, a.accum)]
        seq = [samples[j] for g in groups[g0:] for j in g]
        stream = prefetched(vlm.proc, seq, a.prompt)
        torch.cuda.reset_peak_memory_stats()
        for gi in range(g0, len(groups)):
            t_step = time.time()
            for _ in groups[gi]:
                s, ex = next(stream)
                try:
                    loss, hit = loss_on(vlm, ex)
                    (loss * weight[s.row_id] / len(groups[gi])).backward()
                    recent = (recent + [(float(loss), *field_hits(hit, ex["field"]))])[-80:]
                except torch.cuda.OutOfMemoryError:
                    skipped.append(s.row_id)
                    torch.cuda.empty_cache()
                    log(out_dir, f"  out of memory on row {s.row_id} ({ex['enc']['input_ids'].shape[1]} tokens): skipped")
                    if len(skipped) > 5:
                        sys.exit("more than 5 reviews did not fit: rerun with --max-visual-tokens 4000")
                done_reviews += 1
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            per_review = (time.time() - t_run - val_spent) / max(done_reviews, 1)        # training time only
            left_val = 0 if a.no_val else (a.epochs - epoch) * (val_s or 100 * 4.5)    # val checks still to come
            left = per_review * (total_steps - step) * a.accum + left_val
            k = len(recent)
            ml = [sum(r[i] for r in recent) / k for i in range(4)] if k else [0] * 4
            line = (f"[ep {epoch + 1}/{a.epochs} | step {step}/{total_steps} {100 * step / total_steps:5.1f}%] "
                    f"loss {ml[0]:.3f} | head acc cat {ml[1]:.2f} part {ml[2]:.2f} vis {ml[3]:.2f} (last {k}) | "
                    f"lr {sched.get_last_lr()[0]:.1e} | grad {gnorm:.2f} | {time.time() - t_step:4.0f}s/step | "
                    f"elapsed {_hms(time.time() - t_run)} | left ~{_hms(left)} "
                    f"(done ~{time.strftime('%m-%d %H:%M', time.localtime(time.time() + left))}) | "
                    f"peak {torch.cuda.max_memory_reserved() / 2**30:.1f}/{total_gb:.1f} GB")
            log(out_dir, line)
            with open(out_dir / "steps.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "epoch": epoch + 1, "loss": ml[0], "acc": ml[1:], "lr":
                                    sched.get_last_lr()[0], "grad_norm": gnorm, "s": time.time() - t_step}) + "\n")
            if step % a.save_every == 0 and gi != len(groups) - 1:
                save_state(peft_model, opt, sched, step, skipped, out_dir / "last")
        # ---- end of epoch: keep this adapter, check it on val ----
        ep_dir = out_dir / f"epoch{epoch + 1}"
        peft_model.save_pretrained(ep_dir)
        log(out_dir, f"[{a.run}] epoch {epoch + 1} done -> {ep_dir}")
        if not a.no_val:
            t_v = time.time()
            m, cards = val_head(vlm, val, flags, out_dir, f"epoch{epoch + 1}")
            val_s = time.time() - t_v
            val_spent += val_s
            (ep_dir / "val_head.json").write_text(json.dumps({"epoch": epoch + 1, "metrics": m, "e1_metrics": e1_m,
                                                              "preds": {r: {"group6": c.group6, "part": c.part,
                                                                            "visibility": c.visibility, "raw": c.raw}
                                                                        for r, c in cards.items()}},
                                                             ensure_ascii=False, indent=1), encoding="utf-8")
            log(out_dir, f"[val head epoch {epoch + 1}] {fmt_head(m)}")
            log(out_dir, f"[val head E1     ] {fmt_head(e1_m)}")
        save_state(peft_model, opt, sched, step, skipped, out_dir / "last")
    pick = pick_epoch(out_dir, a.epochs)
    nxt = (f"chosen epoch (rule): {pick} -> next: python -m cluvi.run_hawk --model heavy --split val "
           f"--adapter {out_dir / f'epoch{pick}'}") if pick else "no val head check"
    log(out_dir, f"[{a.run}] finished in {_hms(time.time() - t_run)} | skipped {skipped or 'none'} | {nxt}")


def pick_epoch(out_dir, epochs):
    scores = {}
    for e in range(1, epochs + 1):
        f = out_dir / f"epoch{e}" / "val_head.json"
        if f.exists():
            scores[e] = json.loads(f.read_text(encoding="utf-8"))["metrics"]["all"]["select"]
    return max(scores, key=lambda e: (round(scores[e], 6), -e)) if scores else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="e3l", help="output folder name under outputs/lora/")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--unsure-weight", type=float, default=UNSURE_WEIGHT)
    ap.add_argument("--sure-only", action="store_true", help="variant: train on the 642 sure rows only")
    ap.add_argument("--max-visual-tokens", type=int, help="fallback if long inputs do not fit (proposal: 4000)")
    ap.add_argument("--prompt", default=PROMPT_VERSION)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=20, help="updates between resume points")
    ap.add_argument("--no-val", action="store_true", help="skip the val head check after each epoch")
    ap.add_argument("--probe", type=int, help="GPU fit check on the N longest train reviews, then exit")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    from .run_hawk import use_runtime
    use_runtime("train_lora")
    from .predict import check_runtime
    check_runtime(False)
    from .models import QwenVLM
    vlm = QwenVLM("heavy", prompt_version=a.prompt)            # exactly E1's load (NF4 LLM, bf16 ViT, sdpa)
    peft_model, info = attach_lora(vlm, a)
    probe(a, vlm, peft_model, info) if a.probe else train(a, vlm, peft_model, info)


if __name__ == "__main__":
    main()
