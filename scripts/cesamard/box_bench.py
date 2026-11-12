"""Box-finding benchmark (plan T4, box-only mode).

Every method gets the gold part and defect of each localized issue and must box the
defect in each image where the human drew a box. We score the method's top box:
  pointing  = its center lies inside a human box
  hit@0.5   = IoU with a human box >= 0.5
plus mean best IoU, seconds per image and peak VRAM.

Gold: gold_v1.3.json — round2 (blind, corrected) is the unbiased main set; gold90 (Claude draft checked
by a human) is larger but favors Claude, since its boxes started as Claude's; pilot30 (human v1.0) is secondary.

Usage:
  python box_bench.py run --method whole|center|claude|dino|florence|qwen
  python box_bench.py eval
Predictions go to data/cesamard/labeling/auto/boxes/<method>.json.
"""
import argparse
import json
import re
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data" / "cesamard"
LAB = DS / "labeling"
OUT = LAB / "auto" / "boxes"
GOLD_FILE = LAB / "gold_v1.3.json"           # round2 (blind+corrected) and gold90 (draft-checked), by `source`
ITEM_FILES = [LAB / "round2.json", LAB / "gold90.json", LAB / "pilot30.json"]
PILOT = LAB / "pilot30_labels.json"            # round 1, human v1.0 (secondary)
SUBSETS = ["round2", "gold90", "pilot30"]
CLAUDE = {"round2": LAB / "auto" / "round2_claude_cmm_v1.2.json",
          "gold90": LAB / "auto" / "gold90_claude_cmm_v1.3.json",   # the drafts the human checked: biased in Claude's favor
          "pilot30": LAB / "auto" / "pilot30_claude_cmm_v1.json"}
MAX_SIDE = 1024  # images are downscaled before inference; boxes are normalized so this does not change scoring

PART_EN = {
    "cover": "book cover", "corner": "book corner", "spine_binding": "book spine", "pages": "book pages",
    "print_surface": "printed page", "body_fabric": "fabric", "seam": "seam", "pocket": "pocket", "sleeve": "sleeve",
    "collar": "collar", "fastener": "zipper or button", "print_deco": "print", "lining": "inner lining", "hem": "hem",
    "housing": "device body", "display": "screen", "camera": "camera", "battery": "battery", "port": "port",
    "button": "button", "audio": "speaker", "internal": "internal part", "software": "screen",
    "contents": "food", "container": "container", "whole": "product", "outer_box": "shipping box",
    "product_package": "package", "seal": "seal", "charger_cable": "cable", "other_accessory": "accessory",
    "label": "label", "manual": "manual", "none": "product", "other": "product",
}
DEFECT_EN = {
    "tear_hole": "tear", "crush_break": "broken part", "scratch": "scratch", "opened_tampered": "opened seal",
    "pest": "insect", "mold_spoil": "mold", "foreign_object": "foreign object", "stain_dirt": "stain",
    "material": "damaged fabric", "stitching": "bad stitching", "print": "faded print", "color": "faded color",
    "display": "spot on the screen", "assembly": "loose part", "counterfeit_wrong": "wrong item",
    "missing": "missing part", "used": "signs of use", "label_expiry": "expiry date",
    "malfunction": "error message", "performance": "warning message", "fit_size": "size label",
    "design_usability": "part", "taste_smell": "food", "content": "page", "service": "product", "price": "price tag",
    "vague": "defect",
}


def labeled_reviews():
    """(set, row, label) for gold_v1.3 plus the secondary round-1 labels."""
    gold = json.loads(GOLD_FILE.read_text(encoding="utf-8"))["labels"]
    rows = [(v["source"], row, v) for row, v in gold.items()]
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))["labels"]
    rows += [("pilot30", row, v) for row, v in pilot.items() if v.get("done")]
    return rows


def instances():
    """One instance per (localized main issue, image the human boxed)."""
    out = []
    items = {}
    for f in ITEM_FILES:
        items.update({str(x["row_id"]): x for x in json.loads(f.read_text(encoding="utf-8"))})
    for name, row, lab in labeled_reviews():
        if lab["issues"]:
            iss = lab["issues"][0]
            if iss["vis"] != "localized" or not iss["boxes"]:
                continue
            for img in sorted({b["img"] for b in iss["boxes"]}):
                src = next(im["src"] for im in items[row]["images"] if im["idx"] == img)
                target = iss["boxes"][0].get("target", "defect")
                defect = "expiry date" if iss["defect"] == "label_expiry" else DEFECT_EN.get(iss["defect"], "defect")
                pt = iss.get("part_text") or ""
                part = pt if pt.isascii() and pt else PART_EN.get(iss["part"], "product")  # Korean free text → code name
                out.append({
                    "set": name, "row": row, "img": img, "src": src,
                    "gold": [b["bbox"] for b in iss["boxes"] if b["img"] == img],
                    "defect_en": defect, "part_en": part, "target": target,
                    "title": items[row]["title"],
                })
    return out


def load_image(src):
    im = Image.open(DS / src).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    return im


def norm_box(b, w, h):
    x1, y1, x2, y2 = b
    return [max(0, min(1, x1 / w)), max(0, min(1, y1 / h)), max(0, min(1, x2 / w)), max(0, min(1, y2 / h))]


# ---------------- methods ----------------
def run_trivial(insts, kind):
    box = [0, 0, 1, 1] if kind == "whole" else [0.25, 0.25, 0.75, 0.75]
    return {f"{i['set']}/{i['row']}/{i['img']}": {"boxes": [box], "sec": 0} for i in insts}


def run_claude(insts):
    preds = {}
    cache = {k: json.loads(p.read_text(encoding="utf-8"))["labels"] for k, p in CLAUDE.items()}
    for i in insts:
        lab = cache[i["set"]].get(i["row"], {"issues": []})
        issues = [x for x in lab["issues"] if x.get("main")] or lab["issues"][:1]
        boxes = [b["bbox"] for x in issues for b in x["boxes"] if b["img"] == i["img"]]
        preds[f"{i['set']}/{i['row']}/{i['img']}"] = {"boxes": boxes, "sec": None}
    return preds


def run_dino(insts):
    import torch
    from transformers import AutoProcessor, GroundingDinoForObjectDetection
    mid = "IDEA-Research/grounding-dino-base"
    proc = AutoProcessor.from_pretrained(mid)
    model = GroundingDinoForObjectDetection.from_pretrained(mid).to("cuda").eval()
    preds = {}
    for i in insts:
        im = load_image(i["src"])
        text = f"{i['defect_en']} ."
        t = time.time()
        with torch.no_grad():
            enc = proc(images=im, text=text, return_tensors="pt").to("cuda")
            out = model(**enc)
            res = proc.post_process_grounded_object_detection(
                out, enc.input_ids, threshold=0.2, text_threshold=0.2, target_sizes=[im.size[::-1]])[0]
        torch.cuda.synchronize()
        order = res["scores"].argsort(descending=True)
        boxes = [norm_box(res["boxes"][k].tolist(), *im.size) for k in order[:5]]
        preds[f"{i['set']}/{i['row']}/{i['img']}"] = {"boxes": boxes, "scores": res["scores"][order[:5]].tolist(),
                                                     "sec": time.time() - t, "query": text}
    return preds


def run_florence(insts):
    import torch
    from transformers import AutoProcessor, Florence2ForConditionalGeneration
    mid = "florence-community/Florence-2-large"
    proc = AutoProcessor.from_pretrained(mid)
    model = Florence2ForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.bfloat16).to("cuda").eval()
    task = "<CAPTION_TO_PHRASE_GROUNDING>"
    preds = {}
    for i in insts:
        im = load_image(i["src"])
        phrase = f"{i['defect_en']} on the {i['part_en']}"
        t = time.time()
        with torch.no_grad():
            enc = proc(text=task + phrase, images=im, return_tensors="pt").to("cuda", torch.bfloat16)
            gen = model.generate(**enc, max_new_tokens=256, num_beams=3, do_sample=False)
        torch.cuda.synchronize()
        text = proc.batch_decode(gen, skip_special_tokens=False)[0]
        parsed = proc.post_process_generation(text, task=task, image_size=im.size)
        boxes = [norm_box(b, *im.size) for b in parsed.get(task, {}).get("bboxes", [])][:5]
        preds[f"{i['set']}/{i['row']}/{i['img']}"] = {"boxes": boxes, "sec": time.time() - t, "query": phrase}
    return preds


def run_qwen(insts):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    mid = "Qwen/Qwen3-VL-8B-Instruct"
    proc = AutoProcessor.from_pretrained(mid)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        mid, device_map="cuda", torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    preds = {}
    for i in insts:
        im = load_image(i["src"])
        what = i["defect_en"] if i["target"] == "defect" else f"{i['defect_en']} (evidence of the problem)"
        prompt = (f"A customer complained: \"{i['title']}\". Locate the {what} on the {i['part_en']} in this image. "
                  "Output the bounding box as JSON: [{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"...\"}].")
        msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": prompt}]}]
        t = time.time()
        with torch.no_grad():
            enc = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to("cuda")
            gen = model.generate(**enc, max_new_tokens=128, do_sample=False)
        torch.cuda.synchronize()
        text = proc.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        nums = [[float(v) for v in m] for m in re.findall(r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]", text)]
        # Qwen3-VL grounding coordinates are relative, scaled to 0-1000
        boxes = [[max(0, min(1, v / 1000)) for v in b] for b in nums][:5]
        preds[f"{i['set']}/{i['row']}/{i['img']}"] = {"boxes": boxes, "sec": time.time() - t, "raw": text[:300]}
    return preds


# ---------------- scoring ----------------
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def evaluate():
    insts = instances()
    rows = []
    for f in sorted(OUT.glob("*.json")):
        preds = json.loads(f.read_text(encoding="utf-8"))["preds"]
        meta = json.loads(f.read_text(encoding="utf-8"))["meta"]
        for subset in SUBSETS:
            sub = [i for i in insts if i["set"] == subset]
            pt = hit = 0
            ious, secs, empty = [], [], 0
            for i in sub:
                p = preds.get(f"{i['set']}/{i['row']}/{i['img']}", {"boxes": []})
                if not p["boxes"]:
                    empty += 1
                    ious.append(0.0)
                    continue
                top = p["boxes"][0]
                best = max(iou(top, g) for g in i["gold"])
                ious.append(best)
                c = ((top[0] + top[2]) / 2, (top[1] + top[3]) / 2)
                pt += any(g[0] <= c[0] <= g[2] and g[1] <= c[1] <= g[3] for g in i["gold"])
                hit += best >= 0.5
                if p.get("sec") is not None:
                    secs.append(p["sec"])
            n = len(sub)
            if not n:
                continue
            rows.append((f.stem, subset, n, pt / n, hit / n, sum(ious) / n, empty,
                         (sum(secs) / len(secs)) if secs else None, meta.get("peak_vram_gb")))
    print(f"{'method':10s} {'set':8s} {'n':>3s} {'pointing':>9s} {'hit@0.5':>8s} {'meanIoU':>8s} {'nobox':>6s} {'sec/img':>8s} {'VRAM':>6s}")
    for m, s, n, p, h, mi, e, sec, vram in rows:
        print(f"{m:10s} {s:8s} {n:3d} {p:9.0%} {h:8.0%} {mi:8.2f} {e:6d} {('%.2f' % sec) if sec is not None else '-':>8s} "
              f"{('%.1f' % vram) if vram else '-':>6s}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--method", required=True, choices=["whole", "center", "claude", "dino", "florence", "qwen"])
    sub.add_parser("eval")
    sub.add_parser("list")
    args = ap.parse_args()
    if args.cmd == "eval":
        return evaluate()
    insts = instances()
    if args.cmd == "list":
        for i in insts:
            print(i["set"], i["row"], i["img"], i["target"], "|", i["defect_en"], "on", i["part_en"], "|", len(i["gold"]), "gold")
        print(len(insts), "instances")
        return
    peak = None
    if args.method in ("whole", "center"):
        preds = run_trivial(insts, args.method)
    elif args.method == "claude":
        preds = run_claude(insts)
    else:
        import torch
        torch.cuda.reset_peak_memory_stats()
        preds = {"dino": run_dino, "florence": run_florence, "qwen": run_qwen}[args.method](insts)
        peak = torch.cuda.max_memory_allocated() / 1e9
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.method}.json").write_text(json.dumps(
        {"meta": {"method": args.method, "n": len(insts), "peak_vram_gb": peak, "max_side": MAX_SIDE}, "preds": preds},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT / (args.method + '.json')} ({len(preds)} instances, peak VRAM {peak})")


if __name__ == "__main__":
    main()
