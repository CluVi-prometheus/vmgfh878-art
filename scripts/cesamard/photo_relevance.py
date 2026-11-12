"""Photo-text relevance with SigLIP 2: does a photo go with the complaint, and what kind of photo is it?

A cheap, independent signal for "the photo is unrelated to the complaint" (calibrated later on the blind sample).
For every photo (turned upright by its EXIF tag) it scores
  title      the review title vs the photo                         (sigmoid match probability)
  complaint  the short complaint quote from Qwen pass 2 (note field), falling back to the title
  type       zero-shot photo type (softmax over TYPES)

  python photo_relevance.py      # main + gold items -> auto/siglip_relevance.json
"""
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data" / "cesamard"
LAB = DS / "labeling"
AUTO = LAB / "auto"
MODEL = "google/siglip2-so400m-patch14-384"
TYPES = {
    "product": "a photo of a product",
    "defect_closeup": "a close-up photo of a damaged or defective part of a product",
    "screen": "a screenshot of a phone or computer screen",
    "document": "a photo of a receipt, invoice, label or printed document",
    "box": "a photo of a cardboard delivery box or shipping package",
    "food": "a photo of food",
}


def load_items():
    items = {}
    for f in ("main.json", "round2.json", "gold90.json"):
        for x in json.loads((LAB / f).read_text(encoding="utf-8")):
            items[str(x["row_id"])] = x
    return items


def complaint_texts():
    out = {}
    for f in ("main_qwen_v1.3c.json", "gold_qwen_v1.3c.json"):
        p = AUTO / f
        if p.exists():
            out.update({r: v.get("note", "") for r, v in json.loads(p.read_text(encoding="utf-8"))["labels"].items()})
    return out


def unit(out):
    """transformers 5 returns an output object from get_*_features; the embedding is its pooler_output."""
    e = getattr(out, "pooler_output", out)
    return e / e.norm(dim=-1, keepdim=True)


@torch.no_grad()
def main():
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()

    def text_emb(texts):
        enc = proc(text=texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to("cuda")
        return unit(model.get_text_features(**enc))

    type_emb = text_emb(list(TYPES.values()))
    scale, bias = model.logit_scale.exp(), model.logit_bias
    items, quotes = load_items(), complaint_texts()
    out = {}
    for n, (row, it) in enumerate(sorted(items.items(), key=lambda x: int(x[0])), 1):
        texts = [it["title"] or it["text"][:200], quotes.get(row) or it["title"] or it["text"][:200]]
        t = text_emb(texts)
        imgs = [ImageOps.exif_transpose(Image.open(DS / im["src"])).convert("RGB") for im in it["images"]]
        enc = proc(images=imgs, return_tensors="pt").to("cuda")
        v = unit(model.get_image_features(pixel_values=enc["pixel_values"].half()))
        match = torch.sigmoid(v @ t.T * scale + bias).float().cpu()           # [n_img, 2]
        types = torch.softmax(v @ type_emb.T * scale, dim=-1).float().cpu()   # [n_img, n_types]
        per = [{"idx": im["idx"], "title": round(match[i, 0].item(), 4), "complaint": round(match[i, 1].item(), 4),
                "type": {k: round(types[i, j].item(), 3) for j, k in enumerate(TYPES)}}
               for i, im in enumerate(it["images"])]
        out[row] = {"imgs": per, "max_title": max(x["title"] for x in per),
                    "max_complaint": max(x["complaint"] for x in per), "complaint_text": texts[1][:200]}
        if n % 200 == 0:
            print(f"{n}/{len(items)}", flush=True)
    (AUTO / "siglip_relevance.json").write_text(
        json.dumps({"meta": {"model": MODEL, "types": TYPES}, "labels": out}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"wrote auto/siglip_relevance.json ({len(out)} reviews)")


if __name__ == "__main__":
    main()
