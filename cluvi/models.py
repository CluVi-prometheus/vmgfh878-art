"""Every model is an Analyzer: analyze(Sample) -> Card. Swap the model, keep the rest of the pipeline.

  text   Light Text  (TF-IDF + logistic regression, CPU)   reads the review only
  light  Light VLM   (Qwen2.5-VL-3B, bf16)                  review + photos
  heavy  Heavy VLM   (Qwen2.5-VL-7B, LLM 4-bit / ViT bf16)  review + photos
  mock   fake VLM for plumbing tests without a GPU

Token reduction = the same QwenVLM with hawk_keep (official HAWK code) or a lower max_pixels (resize).
Cost fields follow the token-reduction papers (see reports/초기실험_설계_v0.1.md, section 4):
  prefill_ms = time to the first generated token (VisionZip), incl. the vision encoder, after inputs are on the GPU
  e2e_ms     = photo file open -> last token (HAWK "E2E latency", per review)
  peak_vram_gb = torch.cuda.max_memory_allocated during this review
  visual_tokens / visual_tokens_full -> Token Retention Ratio
"""
import hashlib
import json
import re
import time

from PIL import Image, ImageOps

from .common import DEFECTS, PROMPT_VERSION, Card, Sample, build_prompt, group5_of, load_split, parse_vlm, scope_of

VLM_IDS = {"heavy": ("Qwen/Qwen2.5-VL-7B-Instruct", True), "light": ("Qwen/Qwen2.5-VL-3B-Instruct", False)}
MIN_PIXELS = 256 * 28 * 28     # 200,704 px   = 256 tokens   } example values from the official Qwen2.5-VL model card
MAX_PIXELS = 1280 * 28 * 28    # 1,003,520 px = 1,280 tokens } (28x28 px = 1 visual token); also the retention base
MAX_IMAGES = 4                 # 167 of the 171 gold reviews have <= 4 photos


def open_images(s: Sample, max_images=MAX_IMAGES):
    """Upright RGB photos. qwen-vl-utils does not apply EXIF rotation, and the gold boxes are on upright photos, so we
    rotate here and hand PIL images to the official pipeline (it accepts PIL images as they are)."""
    return [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in s.images[:max_images]]


def seen_sizes(sizes, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """(w, h) of each photo as Qwen2.5-VL sees it: smart_resize to multiples of 28 px inside the pixel range."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    return [smart_resize(h, w, factor=28, min_pixels=min_pixels, max_pixels=max_pixels)[::-1] for w, h in sizes]


def visual_tokens_at(sizes, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """Visual tokens Qwen2.5-VL makes for photos of these (w, h) sizes under a pixel range (28x28 px per token)."""
    return sum((w // 28) * (h // 28) for w, h in seen_sizes(sizes, min_pixels, max_pixels))


def encode_chat(proc, text, images=(), min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """Qwen's official input path (model card / README): photos first, then the text, chat template with numbered
    pictures. Returns (enc on CPU, prompt string). Shared by inference (QwenVLM.chat) and LoRA training
    (cluvi.lora_data), so the adapter is trained on exactly the tokens it later sees."""
    from qwen_vl_utils import process_vision_info
    content = [{"type": "image", "image": img, "min_pixels": min_pixels, "max_pixels": max_pixels} for img in images]
    content.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": content}]
    prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=True)
    if images and "Picture 1:" not in prompt:
        raise RuntimeError("this chat template ignored add_vision_id (no 'Picture 1:' before the photo)")
    image_inputs = process_vision_info(messages)[0]
    return proc(text=[prompt], images=image_inputs, padding=True, return_tensors="pt"), prompt


def review_prompt(s: Sample, imgs, prompt_version=PROMPT_VERSION, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """The prompt text for one review, with the photo sizes the model will actually see written into it (v0.3+)."""
    told = seen_sizes([im.size for im in imgs], min_pixels, max_pixels)
    return build_prompt(s, len(imgs), prompt_version, sizes=told), told


def adapter_id(path):
    """Short fingerprint of a LoRA adapter's weights (goes into the run config, so a resumed run can't mix adapters)."""
    from pathlib import Path
    f = Path(path) / "adapter_model.safetensors"
    return hashlib.sha1(f.read_bytes()).hexdigest()[:12] if f.exists() else None


def _pkg_version(name):
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _hawk_commit():
    """Git commit of the HAWK checkout the `hawk` package is imported from (HAWK is used from source, not pip)."""
    from pathlib import Path
    import hawk
    git = Path(hawk.__file__).resolve().parents[2] / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        return (git / head[5:]).read_text().strip() if head.startswith("ref: ") else head
    except OSError:
        return None


class QwenVLM:
    """Qwen2.5-VL run the way the official model card / README does it. Only the prompt text and the output JSON are ours.
      load      Qwen2_5_VLForConditionalGeneration.from_pretrained(torch_dtype="auto", device_map="auto")   (card)
      photos    qwen_vl_utils.process_vision_info, min/max pixels per photo and on the processor            (card, README)
      numbering apply_chat_template(..., add_vision_id=True) -> "Picture 1: <photo> Picture 2: ..."         (README)
      system    the template's default ("You are a helpful assistant.")                                    (template)
      decoding  greedy (do_sample=False, as in HAWK's infer.py); the rest of generation_config.json as shipped
                (repetition_penalty 1.05). The shipped temperature 1e-6 sampling is greedy in practice anyway.
    Forced by the 16 GB GPU: the LLM is loaded in NF4 4-bit and the vision encoder stays bf16 -- the same split as
    Qwen's own AWQ release (modules_to_not_convert = ["visual"]), which itself can't be loaded here (needs autoawq).
    hawk_keep: run the official HAWK code (github.com/peppery77/HAWK) with this keep ratio. Needs HAWK's patched
    transformers 4.52.0 runtime on PYTHONPATH (its scripts/run.sh). With hawk_keep=None the patched model is stock.
    adapter: folder of a LoRA adapter (E3, cluvi.train_lora). It is attached on top of the 4-bit model, NOT merged
    (merging into NF4 weights would add rounding error); everything else stays exactly as in E1. None = E1."""

    def __init__(self, name="heavy", min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, max_images=MAX_IMAGES,
                 max_new_tokens=640, prompt_version=PROMPT_VERSION, hawk_keep=None, repetition_penalty=None,
                 adapter=None):
        import torch
        import transformers
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        mid, quant4 = VLM_IDS[name]
        self.name, self.model_id, self.torch = name, mid, torch
        self.min_pixels, self.max_pixels, self.max_images = min_pixels, max_pixels, max_images
        self.max_new_tokens, self.prompt_version, self.hawk_keep = max_new_tokens, prompt_version, hawk_keep
        self.gen_override = {} if repetition_penalty is None else {"repetition_penalty": repetition_penalty}
        self.proc = AutoProcessor.from_pretrained(mid, min_pixels=min_pixels, max_pixels=max_pixels)
        # skip patterns cover both transformers 4.52 (substring match) and 5.x (prefix match)
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                               llm_int8_skip_modules=["visual", "model.visual", "lm_head"]) if quant4 else None
        self.quant = "nf4-llm / bf16-vision" if quant4 else "bf16"
        dtype_kw = {"dtype": "auto"} if int(transformers.__version__.split(".")[0]) >= 5 else {"torch_dtype": "auto"}
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            mid, device_map="auto", attn_implementation="sdpa", quantization_config=q, **dtype_kw).eval()
        off = {k: v for k, v in (getattr(self.model, "hf_device_map", None) or {}).items() if str(v) in ("cpu", "disk")}
        if off:
            raise RuntimeError(f"model parts were offloaded to CPU/disk (latency would be meaningless): {off}")
        self.img_tok = self.proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.visual = getattr(self.model.model, "visual", None) or self.model.visual
        self._vis_t = [0.0, 0.0]
        self.visual.register_forward_pre_hook(lambda *a: self._mark(0))
        self.visual.register_forward_hook(lambda *a: self._mark(1))
        self.vram_total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        self.hawk_stats, self.hawk_commit = None, None
        if hawk_keep is not None:
            if adapter:
                raise NotImplementedError("LoRA adapter + HAWK together is untested (E3-L runs with HAWK off)")
            from hawk import HawkConfig, configure_model, get_last_pruning_stats   # official HAWK code, unchanged
            configure_model(self.model, HawkConfig(keep_ratio=hawk_keep))
            self.hawk_stats, self.hawk_commit = get_last_pruning_stats, _hawk_commit()
        self.adapter, self.adapter_id, self.lora_modules = adapter, None, 0
        if adapter:
            from peft import PeftModel
            # the LoRA layers are inserted into the model in place; keep using the plain model object so that
            # generate / visual hooks / config work exactly as in E1
            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=False).get_base_model().eval()
            self.adapter_id = adapter_id(adapter)
            self.lora_modules = sum(hasattr(m, "lora_A") for m in self.model.modules())
        n4 = sum(type(m).__name__ == "Linear4bit" for m in self.visual.modules())
        print(f"[{name}] {mid} | transformers {transformers.__version__} | {self.quant} | 4-bit layers inside ViT: {n4} "
              f"(expect 0) | pixels {min_pixels}-{max_pixels} | HAWK keep {hawk_keep} | "
              f"weights {self.model.get_memory_footprint() / 2**30:.2f} GB"
              f"{f' | LoRA {adapter} ({self.lora_modules} modules)' if adapter else ''}", flush=True)

    def _mark(self, i):
        self.torch.cuda.synchronize()
        self._vis_t[i] = time.perf_counter()

    def chat(self, text, images=(), max_new_tokens=None, conf_key=None, t_start=None):
        """One VLM call. Returns (reply, cost, conf, sizes). sizes = (w, h) of each photo as the model saw it.
        conf = probability of the JSON value after "conf_key" in the reply (greedy decoding, no extra compute)."""
        from transformers import LogitsProcessor, LogitsProcessorList
        torch = self.torch
        t0 = t_start or time.perf_counter()
        enc = encode_chat(self.proc, text, images, self.min_pixels, self.max_pixels)[0].to(self.model.device)
        sizes = [(int(w) * 14, int(h) * 14) for _, h, w in enc["image_grid_thw"].tolist()] if images else []
        n_in = enc["input_ids"].shape[1]
        n_vis = int((enc["input_ids"] == self.img_tok).sum())
        first, step_p = [], []

        class StepRecorder(LogitsProcessor):
            """Runs after every forward pass (after the built-in processors such as repetition_penalty).
            First call = end of prefill. Keeps only the greedy token's probability per step (a 1-element GPU tensor),
            instead of output_scores, which would hold a full vocab-size float32 row per token and inflate peak VRAM."""
            def __call__(self, input_ids, scores):
                if not first:
                    torch.cuda.synchronize()
                    first.append(time.perf_counter())
                step_p.append(torch.softmax(scores[0].float(), -1).max().reshape(1))
                return scores

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self._vis_t = [0.0, 0.0]
        t_in = time.perf_counter()
        with torch.no_grad():
            seq = self.model.generate(**enc, max_new_tokens=max_new_tokens or self.max_new_tokens, do_sample=False,
                                      logits_processor=LogitsProcessorList([StepRecorder()]), **self.gen_override)
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        ids = seq[0, -len(step_p):]                    # generated tokens only (HAWK shortens the prompt part)
        probs = torch.cat(step_p).tolist() if step_p else []
        reply = self.proc.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        stats = self.hawk_stats(self.model) if self.hawk_stats else None
        pruned = bool(stats and stats.get("applied"))
        t_first = first[0] if first else t_end
        cost = {"vlm_calls": 1, "input_tokens": n_in, "output_tokens": int(len(ids)),
                "visual_tokens_input": n_vis,                                              # made by the ViT
                "visual_tokens": int(stats["kept_visual_tokens"]) if pruned else n_vis,    # reached the LLM
                "truncated": int(len(ids)) >= (max_new_tokens or self.max_new_tokens),
                "preprocess_ms": round((t_in - t0) * 1000, 1),
                "vision_ms": round((self._vis_t[1] - self._vis_t[0]) * 1000, 1) if images else 0.0,
                "prefill_ms": round((t_first - t_in) * 1000, 1),
                "decode_ms": round((t_end - t_first) * 1000, 1),
                "e2e_ms": round((t_end - t0) * 1000, 1),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3)}
        if cost["peak_reserved_gb"] > self.vram_total_gb - 0.5:
            cost["near_vram_limit"] = True     # Windows may spill to shared RAM instead of OOM -> latency outlier
        if stats:
            cost["hawk"] = {k: stats.get(k) for k in ("applied", "original_visual_tokens", "kept_visual_tokens",
                                                      "keep_ratio_actual", "query_tokens")}
        conf = self._value_prob(ids, probs, reply, conf_key) if conf_key else None
        return reply, cost, conf, sizes

    def _value_prob(self, ids, probs, reply, key):
        """Product of the greedy tokens' probabilities over the characters of the JSON value of `key`.
        Probabilities are after the shipped repetition_penalty (the distribution the token was actually picked from)."""
        m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', reply)
        if not m:
            return 0.0
        # v0.3.1+ writes "category": "group6/defect"; the group confidence is the part before the slash
        stop = m.start(1) + m.group(1).index("/") if key == "category" and "/" in m.group(1) else m.end(1)
        tok, p, start = self.proc.tokenizer, 1.0, 0
        for i in range(len(ids)):
            if start >= stop:
                break
            end = len(tok.decode(ids[:i + 1], skip_special_tokens=True, clean_up_tokenization_spaces=False))
            if end > m.start(1):
                p *= probs[i]
            start = end
        return round(p, 4)

    def analyze(self, s: Sample) -> Card:
        t0 = time.perf_counter()
        imgs = open_images(s, self.max_images)
        orig_sizes = [im.size for im in imgs]
        prompt, told = review_prompt(s, imgs, self.prompt_version, self.min_pixels, self.max_pixels)   # sizes (v0.3+)
        key = "category" if '"category"' in prompt else "group6"   # v0.3.1+ prompts answer group6 inside "category"
        reply, cost, conf, sizes = self.chat(prompt, imgs, conf_key=key, t_start=t0)
        card = parse_vlm(reply, s, Card(s.row_id, self.name), sizes)
        if [tuple(x) for x in told] != [tuple(x) for x in sizes]:
            card.issues.append("size_mismatch")           # the prompt told the model a size it did not actually see
        cost["visual_tokens_full"] = visual_tokens_at(orig_sizes)     # tokens at the E1 setting = retention base
        if (self.min_pixels, self.max_pixels) == (MIN_PIXELS, MAX_PIXELS) and \
                cost["visual_tokens_input"] != cost["visual_tokens_full"]:
            card.issues.append("token_count_mismatch")    # free check: at the E1 setting both counts must be equal
        if cost["truncated"]:
            card.issues.append("truncated")
        card.cost, card.conf = cost, {"group6": conf, "exit": conf}
        return card

    def run_config(self):
        import torch
        import transformers
        gc = self.model.generation_config
        return {"model": self.model_id, "quant": self.quant, "prompt_version": self.prompt_version,
                "max_images": self.max_images, "min_pixels": self.min_pixels, "max_pixels": self.max_pixels,
                "add_vision_id": True, "system_prompt": "chat template default",
                "decoding": {"do_sample": False, "max_new_tokens": self.max_new_tokens,
                             "repetition_penalty": self.gen_override.get("repetition_penalty",
                                                                         getattr(gc, "repetition_penalty", None))},
                "batch_size": 1, "attn_implementation": getattr(self.model.config, "_attn_implementation", None),
                "hawk_keep_ratio": self.hawk_keep, "hawk_commit": self.hawk_commit,
                "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "transformers": transformers.__version__,
                "qwen_vl_utils": _pkg_version("qwen-vl-utils"), "bitsandbytes": _pkg_version("bitsandbytes"),
                # keys only for adapter runs, so E1 / E2 configs (and resuming them) stay exactly as before
                **({"adapter": str(self.adapter), "adapter_sha1": self.adapter_id, "lora_modules": self.lora_modules,
                    "peft": _pkg_version("peft"), "adapter_merged": False} if self.adapter else {})}


class TextModel:
    """Light Text v0: one logistic regression per field on TF-IDF of title + review, trained on the silver train split.
    conf["exit"] = min(P(top group), P(not_visible)): leave the cascade only if the group is sure AND no photo is needed."""

    FIELDS = ("group6", "defect", "part", "visibility")

    def __init__(self, name="text"):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.name = name
        train = load_split("train")
        X = [self._text(s) for s in train]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(X)
        Xv = self.vec.transform(X)
        self.heads = {f: LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
                      .fit(Xv, [s.gold[f] or "none" for s in train]) for f in self.FIELDS}

    @staticmethod
    def _text(s):
        return f"{s.title}. {s.title}. {s.review}"     # title twice: the title's complaint wins (guide rule)

    def analyze(self, s: Sample) -> Card:
        t = time.perf_counter()
        x = self.vec.transform([self._text(s)])
        card = Card(s.row_id, self.name, valid=True)
        probs = {}
        for f, clf in self.heads.items():
            p = clf.predict_proba(x)[0]
            probs[f] = dict(zip(clf.classes_, p))
            setattr(card, f, clf.classes_[p.argmax()])
        card.part_scope, card.group5 = scope_of(card.part, s.domain), group5_of(card.group6)
        p_group, p_nv = max(probs["group6"].values()), probs["visibility"].get("not_visible", 0.0)
        card.conf = {"group6": round(p_group, 4), "text_sufficient": round(p_nv, 4), "exit": round(min(p_group, p_nv), 4)}
        card.cost = {"vlm_calls": 0, "visual_tokens": 0, "visual_tokens_full": 0,
                     "e2e_ms": round((time.perf_counter() - t) * 1000, 2)}
        return card


class MockVLM:
    """No GPU: returns a fixed-format answer (same JSON as the real prompt) so parsing, agent and evaluation can be
    tested anywhere. Pretends every photo was resized to 1000 x 1000."""

    def __init__(self, name="mock"):
        self.name = name

    def chat(self, text, images=(), max_new_tokens=256, conf_key=None, t_start=None):
        if '"tool"' in text:   # agent planner prompt: call the two lookup tools, then answer
            step = text.count("TOOL RESULT")
            reply = [json.dumps({"tool": "get_order", "args": {}}), json.dumps({"tool": "get_policy", "args": {}}),
                     json.dumps({"answer": "(mock) Replacement is possible under the policy; photo evidence matches the card."})][min(step, 2)]
        else:
            reply = "(mock) I can see the item in the photo."
        return reply, {"vlm_calls": 1, "visual_tokens": 256 * len(images)}, 0.5, [(1000, 1000)] * len(images)

    def analyze(self, s: Sample) -> Card:
        h = int(hashlib.md5(str(s.row_id).encode()).hexdigest(), 16)
        group = list(DEFECTS)[h % 6]
        defect = DEFECTS[group][h % len(DEFECTS[group])]
        n_img = len(s.images[:MAX_IMAGES])
        vis = "localized" if group in ("damage", "contamination") and n_img else "not_visible"
        boxes = [{"picture": 1, "bbox_2d": [250, 250, 750, 750], "label": "defect"}] if vis == "localized" else []
        reply = json.dumps({"part": "whole", "group6": group, "defect": defect, "visibility": vis, "boxes": boxes,
                            "evidence": {"customer_claim": "(mock) 고객은 제품에 문제가 있다고 주장함",
                                         "review_span": s.title[:60],
                                         "visual_findings": [{"picture": 1, "description": "(mock) 사진 가운데에 문제가 보임",
                                                              "box_ref": 1 if boxes else None}],
                                         "claim_support": "supported" if boxes else "not_verifiable",
                                         "limitations": ["(mock) 발생 시점은 확인할 수 없음"]}}, ensure_ascii=False)
        card = parse_vlm(reply, s, Card(s.row_id, self.name), [(1000, 1000)] * n_img)
        conf = round((h % 100) / 100, 2)
        card.conf = {"group6": conf, "exit": conf}
        card.cost = {"vlm_calls": 1, "visual_tokens": 256 * n_img, "visual_tokens_full": 1280 * n_img,
                     "prefill_ms": 0.0, "e2e_ms": 0.0, "peak_vram_gb": 0.0, "output_tokens": 0}
        return card


def load_model(name, **kw):
    if name == "mock":
        return MockVLM()
    if name == "text":
        return TextModel()
    return QwenVLM(name, **kw)
