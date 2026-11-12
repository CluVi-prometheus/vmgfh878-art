"""Local VLM (4-bit) as a full labeler: main issue's part, defect, visibility and one box.

  python qwen_label.py run  --set gold            # the 118 gold reviews -> auto/gold_qwen_v1.3b.json
  python qwen_label.py run  --items gold90.json --out gold90   (any item file)
  python qwen_label.py run  --set gold --prompt v1.3c          # second-pass prompt variant
  python qwen_label.py run  --set gold --model internvl        # another model family -> auto/gold_internvl_v1.3b.json
  python qwen_label.py eval                        # local models vs Claude against gold, per source

Guide v1.3, condensed. Coordinates come back on a 0-1000 relative scale (Qwen3-VL's native scale; InternVL is told).
Qwen reads the stored pixels; InternVL reads photos turned upright by their EXIF tag, as the label tool shows them.
"""
import argparse
import json
import re
import time
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data" / "cesamard"
LAB = DS / "labeling"
AUTO = LAB / "auto"
MAX_SIDE = 768
MAX_IMAGES = 4
PROMPTS = ("v1.3b", "v1.3c")
MODELS = {
    "qwen": "Qwen/Qwen3-VL-8B-Instruct",
    # different model family as an independent voter; its own tiling is capped so 4 photos fit in 16 GB
    "internvl": "OpenGVLab/InternVL3_5-8B-HF",
}
INTERNVL_MAX_PATCHES = 2   # 2 tiles + thumbnail, ~770 tokens a photo (close to Qwen at 768 px); 4 overflows 16 GB with 4 photos
# v1.3: first prompt (defect list flattened; the model often answered a group name); v1.3b: group + defect asked separately;
# v1.3c: second-pass variant: quote the main complaint first, then group by an explicit decision list (v1.3b over-used
#        not_as_described and under-used non_visual on gold), group before part

PRODUCT_PARTS = {
    "Books": ["cover", "corner", "spine_binding", "pages", "print_surface"],
    "Fashion": ["body_fabric", "seam", "pocket", "sleeve", "collar", "fastener", "print_deco", "lining", "hem"],
    "Electronics": ["housing", "display", "camera", "battery", "port", "button", "audio", "internal", "software"],
    "Edible": ["contents", "container"],
    "Misc": [],
}
OTHER_PARTS = {"packaging": ["outer_box", "product_package", "seal"], "accessory": ["charger_cable", "other_accessory"],
               "label_doc": ["label", "manual"], "none": ["none"]}
DEFECTS = {
    "damage": ["tear_hole", "crush_break", "scratch", "opened_tampered"],
    "contamination": ["pest", "mold_spoil", "foreign_object", "stain_dirt"],
    "manufacturing": ["material", "stitching", "print", "color", "display", "assembly"],
    "not_as_described": ["counterfeit_wrong", "missing", "used", "label_expiry"],
    "function": ["malfunction", "performance", "fit_size", "design_usability"],
    "non_visual": ["taste_smell", "content", "service", "price", "vague"],
}
DEFECT_GROUP = {d: g for g, ds in DEFECTS.items() for d in ds}

RULES = """Rules:
- Label ONE issue: the main complaint. The complaint in the TITLE comes first; if the title has none, the one the body talks about most. Only complaints the review states.
- part = where the problem actually is (not where it is shown). Expired food -> contents. Size problems -> whole. Throwaway bag/box -> product_package; jar/bottle kept during use -> container. label/manual only if the label or manual itself is wrong or missing. Service/price -> none. No nameable part (overheating, slowness, color) -> whole.
- defect by SYMPTOM: visibly torn/broken/cracked -> damage; looks fine but works or displays wrong -> malfunction; made wrong from the start (material, stitching, print, color, glue, backlight bleed) -> manufacturing. Counterfeit, wrong size/color sent, missing, used, expiry -> not_as_described (takes priority). Metal piece, staple, hair -> foreign_object. Thin/cheap fabric, poor paper, pilling -> material. content is for book/media content only. Warranty/refund/delivery -> service.
- vis: "localized" if the defect (or evidence such as an error message, an expired date, a cracked screen) is visible in a specific region of a photo; "diffuse" if visible but over the whole product (color, fit, generally thin, a counterfeit that looks clearly different); "not_visible" otherwise (function, taste, content, missing items, unrelated photo, a counterfeit that looks identical). Size tags, price tags and numbers in a manual are not evidence.
- If localized, give the image index where it shows best and a tight box around the defect."""


RULES_C = """How to decide, in this order:
1. Find the MAIN complaint: the complaint in the TITLE comes first; if the title has none, the one the body talks about most. Only complaints the review states. Copy it as a short quote (at most 12 words) into "complaint".
2. Choose the group from that complaint:
   - non_visual: taste or smell; delivery, refund, return, warranty, seller or customer support; price or value for money; book/media content; or the review names no concrete problem ("bad product", "waste of money") and the photos show none.
   - not_as_described: ONLY when the item received differs from what was ordered or advertised: fake / duplicate / not original, a different product, size, colour or model than ordered, a piece or item missing, already used, expired. Poor quality, cheap material, "not as expected", "not as shown in the picture" because it looks cheap, or any defect is NOT not_as_described.
   - damage: visibly torn, broken, crushed, cracked (incl. a cracked screen), scratched, dented, or opened/tampered on arrival.
   - contamination: insects, mold or spoiled food, a foreign object (metal piece, staple, hair), stains or dirt.
   - manufacturing: made wrong from the start: thin or cheap material, pilling, poor paper, stitching, print, colour, display defect such as backlight bleed, assembly or glue.
   - function: looks fine but does not work or works badly; arrived as ordered but does not fit; awkward design.
3. part = where the problem actually is (not where it is shown). Expired food -> contents. Size problems -> whole. Throwaway bag/box -> product_package; jar/bottle kept during use -> container. label/manual only if the label or manual itself is wrong or missing. Service/price -> none. No nameable part (overheating, slowness, colour) -> whole.
4. vis: "localized" if the defect (or evidence such as an error message, an expired date, a cracked screen) is visible in a specific region of a photo; "diffuse" if visible but over the whole product (colour, fit, generally thin, a counterfeit that looks clearly different); "not_visible" otherwise (function, taste, service, content, missing items, unrelated photo, a counterfeit that looks identical). Size tags, price tags and numbers in a manual are not evidence.
5. If localized, give the image index where it shows best and a tight box around the defect itself."""


def parts_for(domain):
    return {"product": PRODUCT_PARTS.get(domain, []) + ["whole"], **OTHER_PARTS}


COORD_NOTE = ("\nbbox_2d uses coordinates from 0 to 1000 relative to that image: (0, 0) is the top-left corner "
              "and (1000, 1000) the bottom-right corner.")


def build_prompt(item, variant):
    parts = parts_for(item["domain"])
    part_list = "; ".join(f"{scope}: {', '.join(ps)}" for scope, ps in parts.items())
    defect_list = "\n".join(f"- group {g}: defects {', '.join(ds)}" for g, ds in DEFECTS.items())
    text = item["text"][:1500]
    if variant == "v1.3c":
        return (f"Amazon India customer complaint ({item['domain']}).\nTitle: {item['title']}\nReview: {text}\n\n"
                f"Defect groups and the defects inside each:\n{defect_list}\nAllowed part codes ({part_list}).\n{RULES_C}\n\n"
                "Answer with JSON only. \"group\" is one group name; \"defect\" must be one of the defects listed under that group:\n"
                '{"complaint": "<short quote>", "group": "<group>", "defect": "<defect>", "part": "<part code>", '
                '"vis": "localized|diffuse|not_visible", "img": <image index or null>, "bbox_2d": [x1, y1, x2, y2] or null}')
    return (f"Amazon India customer complaint ({item['domain']}).\nTitle: {item['title']}\nReview: {text}\n\n"
            f"Allowed part codes ({part_list}).\nDefect groups and the defects inside each:\n{defect_list}\n{RULES}\n\n"
            "Answer with JSON only. \"group\" is one group name; \"defect\" must be one of the defects listed under that group:\n"
            '{"part": "<part code>", "group": "<group>", "defect": "<defect>", "vis": "localized|diffuse|not_visible", '
            '"img": <image index or null>, "bbox_2d": [x1, y1, x2, y2] or null}')


def load_items(paths):
    items = {}
    for p in paths:
        for x in json.loads(p.read_text(encoding="utf-8")):
            items[str(x["row_id"])] = x
    return items


def parse(raw, item):
    m = re.search(r"\{.*\}", raw, re.S)
    out = {"valid": False, "raw": raw[:400]}
    if not m:
        return out
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return out
    parts = parts_for(item["domain"])
    scope = next((s for s, ps in parts.items() if d.get("part") in ps), None)
    defect = d.get("defect")
    # the model sometimes answers with the group name in the defect slot: keep the group, leave defect empty
    group = d.get("group") if d.get("group") in DEFECTS else (defect if defect in DEFECTS else DEFECT_GROUP.get(defect))
    vis = d.get("vis") if d.get("vis") in ("localized", "diffuse", "not_visible") else None
    issue = {"main": True, "scope": scope, "part": d.get("part") if scope else None, "part_text": "",
             "group": group, "defect": defect if defect in DEFECT_GROUP else None,
             "vis": vis, "boxes": []}
    n_img = min(len(item["images"]), MAX_IMAGES)
    box = d.get("bbox_2d")
    if vis == "localized" and isinstance(box, list) and len(box) == 4 and isinstance(d.get("img"), int) and 0 <= d["img"] < n_img:
        b = [max(0, min(1, float(v) / 1000)) for v in box]
        if b[0] < b[2] and b[1] < b[3]:
            issue["boxes"] = [{"img": d["img"], "bbox": [round(v, 4) for v in b], "target": "defect"}]
    out.update({"valid": all(issue[k] for k in ("part", "defect", "vis")), "issue": issue,
                "complaint": str(d.get("complaint") or "")[:200]})
    return out


def run(rows, items, out_name, tag, model_name="qwen"):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    mid = MODELS[model_name]
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForImageTextToText.from_pretrained(
        mid, device_map="cuda", torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    # transformers 5 only forwards processor options given as processor_kwargs (default tiling: 13 tiles, ~3.3k tokens a photo)
    extra = {"processor_kwargs": {"max_patches": INTERNVL_MAX_PATCHES}} if model_name == "internvl" else {}
    out_path = AUTO / f"{out_name}_{model_name}_{tag}.json"
    done = json.loads(out_path.read_text(encoding="utf-8"))["labels"] if out_path.exists() else {}
    labels = dict(done)
    t0 = time.time()
    for n, row in enumerate(rows, 1):
        if row in labels:
            continue
        item = items[row]
        content = []
        for im in item["images"][:MAX_IMAGES]:
            img = Image.open(DS / im["src"])
            img = (ImageOps.exif_transpose(img) if model_name == "internvl" else img).convert("RGB")
            img.thumbnail((MAX_SIDE, MAX_SIDE))
            content += [{"type": "text", "text": f"Image {im['idx']}:"}, {"type": "image", "image": img}]
        content.append({"type": "text", "text": build_prompt(item, tag) + (COORD_NOTE if model_name == "internvl" else "")})
        t = time.time()
        with torch.no_grad():
            enc = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                           add_generation_prompt=True, return_dict=True, return_tensors="pt",
                                           **extra).to("cuda")
            gen = model.generate(**enc, max_new_tokens=220 if tag == "v1.3c" else 160, do_sample=False)
        raw = proc.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        p = parse(raw, item)
        labels[row] = {"issues": [p["issue"]] if "issue" in p else [], "marks": {}, "unsure": False, "note": p.get("complaint", ""),
                       "valid": p["valid"], "raw": p["raw"], "sec": round(time.time() - t, 2)}
        if n % 10 == 0 or n == len(rows):
            out_path.write_text(json.dumps({"meta": {"labeler": f"{mid.split('/')[1].lower()}-nf4", "guide_version": "v1.3",
                                                     "prompt": tag, "max_side": MAX_SIDE}, "labels": labels},
                                           ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{n}/{len(rows)} reviews, {time.time() - t0:.0f}s, peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
    print(f"wrote {out_path}")


def evaluate():
    gold = json.loads((LAB / "gold_v1.3.json").read_text(encoding="utf-8"))["labels"]
    cands = {f"{m}-{t}": json.loads((AUTO / f"gold_{m}_{t}.json").read_text(encoding="utf-8"))["labels"]
             for m in MODELS for t in ("v1.3", *PROMPTS) if (AUTO / f"gold_{m}_{t}.json").exists()}
    claude = {}
    for f in ("round2_claude_cmm_v1.2.json", "gold90_claude_cmm_v1.3.json"):
        claude.update(json.loads((AUTO / f).read_text(encoding="utf-8"))["labels"])
    cands["claude"] = claude
    fields = ["part", "scope", "group", "defect", "vis"]
    print(f"{'labeler':14s} {'set':8s} {'n':>3s} " + " ".join(f"{f:>7s}" for f in fields) + f" {'box_pt':>7s} {'invalid':>7s}")
    for name, cand in cands.items():
        for subset in ("round2", "gold90"):
            rows = [r for r, v in gold.items() if v["source"] == subset]
            acc = {f: 0 for f in fields}
            pt = npt = invalid = 0
            for r in rows:
                g = gold[r]["issues"][0]
                c_issues = cand.get(r, {}).get("issues", [])
                c = ([i for i in c_issues if i.get("main")] or c_issues[:1] or [{}])[0]
                if name != "claude" and not cand.get(r, {}).get("valid", True):
                    invalid += 1
                for f in fields:
                    acc[f] += g.get(f) == c.get(f)
                if g["vis"] == "localized" and g["boxes"] and c.get("boxes"):
                    b = c["boxes"][0]
                    gb = [x["bbox"] for x in g["boxes"] if x["img"] == b["img"]]
                    cx, cy = (b["bbox"][0] + b["bbox"][2]) / 2, (b["bbox"][1] + b["bbox"][3]) / 2
                    npt += 1
                    pt += any(x[0] <= cx <= x[2] and x[1] <= cy <= x[3] for x in gb)
            n = len(rows)
            print(f"{name:14s} {subset:8s} {n:3d} " + " ".join(f"{acc[f] / n:7.0%}" for f in fields) +
                  f" {(f'{pt}/{npt}') if npt else '-':>7s} {invalid:7d}")
    print("note: on gold90 the human checked Claude's drafts, so Claude's gold90 numbers are inflated (anchoring).")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", default=None, help="'gold' = the gold_v1.3 reviews")
    r.add_argument("--items", default=None, help="item file name in data/cesamard/labeling")
    r.add_argument("--out", default=None)
    r.add_argument("--limit", type=int, default=None, help="label only the first N reviews (smoke test)")
    r.add_argument("--prompt", default=PROMPTS[0], choices=PROMPTS)
    r.add_argument("--model", default="qwen", choices=list(MODELS))
    sub.add_parser("eval")
    args = ap.parse_args()
    if args.cmd == "eval":
        return evaluate()
    if args.set == "gold":
        gold = json.loads((LAB / "gold_v1.3.json").read_text(encoding="utf-8"))["labels"]
        rows = sorted(gold, key=int)
        items = load_items([LAB / "round2.json", LAB / "gold90.json"])
        out = "gold"
    else:
        items = load_items([LAB / args.items])
        rows = sorted(items, key=int)
        out = args.out or Path(args.items).stem
    if args.limit:
        rows, out = rows[:args.limit], out + "_smoke"
    run(rows, items, out, args.prompt, args.model)


if __name__ == "__main__":
    main()
