"""Shared by every experiment: paths, label schema, data loading, the Card (= one prediction), prompt + parsing.

Everyone reads data and writes predictions through this file, so numbers from different people stay comparable.
"""
import html
import json
import re
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from string import Template

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "cesamard"
IMG_DIRS = [DS / "images_hr", DS / "images"]          # high-res first, old thumbnails as fallback
META = DS / "hf_release" / "metadata.parquet"
OUT = ROOT / "outputs"
PRED_DIR = OUT / "preds"
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

# ---- label schema (guide v1.3 + `other` part) ----
PRODUCT_PARTS = {
    "Books": ["cover", "corner", "spine_binding", "pages", "print_surface"],
    "Fashion": ["body_fabric", "seam", "pocket", "sleeve", "collar", "fastener", "print_deco", "lining", "hem"],
    "Electronics": ["housing", "display", "camera", "battery", "port", "button", "audio", "internal", "software"],
    "Edible": ["contents", "container"],
    "Misc": [],
}
OTHER_PARTS = {"packaging": ["outer_box", "product_package", "seal"], "accessory": ["charger_cable", "other_accessory"],
               "label_doc": ["label", "manual"], "none": ["none"], "other": ["other"]}
DEFECTS = {
    "damage": ["tear_hole", "crush_break", "scratch", "opened_tampered"],
    "contamination": ["pest", "mold_spoil", "foreign_object", "stain_dirt"],
    "manufacturing": ["material", "stitching", "print", "color", "display", "assembly"],
    "not_as_described": ["counterfeit_wrong", "missing", "used", "label_expiry"],
    "function": ["malfunction", "performance", "fit_size", "design_usability"],
    "non_visual": ["taste_smell", "content", "service", "price", "vague"],
}
DEFECT_GROUP = {d: g for g, ds in DEFECTS.items() for d in ds}
VIS = ("localized", "diffuse", "not_visible")
CLAIM_SUPPORT = ("supported", "partially_supported", "not_verifiable", "contradicted")
BOX_TARGETS = ("defect", "evidence")
MAX_BOXES = 3


def parts_for(domain):
    return {"product": PRODUCT_PARTS.get(domain, []) + ["whole"], **OTHER_PARTS}


def scope_of(part, domain):
    """part_scope is derived from the part code, never asked from the model (so the two can't contradict)."""
    return next((sc for sc, ps in parts_for(domain).items() if part in ps), None)


def group5_of(group6):
    """group5 merges the two groups people confuse most: damage + manufacturing -> physical_defect."""
    return "physical_defect" if group6 in ("damage", "manufacturing") else group6


# ---- data ----
@dataclass
class Sample:
    row_id: int
    domain: str
    title: str
    review: str
    images: list            # absolute paths, in the order boxes' image_index refers to
    gold: dict              # part_scope / part / group6 / group5 / defect / visibility / boxes (None where unknown)
    subset: str = ""        # original defect_split: test | test_blind | train | dev (test reports the two parts separately)
    dup_of: int = None      # row_id of the representative if this review's text is a duplicate


def _img_path(name):
    for d in IMG_DIRS:
        if (d / name).exists():
            return str(d / name)
    return None


def _boxes(raw):
    return [{"image_index": int(b["image_index"]), "bbox": [float(v) for v in b["bbox"]], "target": b.get("target", "defect")}
            for b in (raw if raw is not None else [])]


VAL_N, VAL_SEED = 100, 0       # team split 2026-11-11: train 881 / val 100 / test 185


@lru_cache(maxsize=1)
def val_ids(n=VAL_N, seed=VAL_SEED):
    """val = 100 of the 981 silver train reviews, stratified by group6 with a fixed seed (same 100 for everyone).
    Only reviews that are not part of a duplicate pair are eligible, so train and val never share a text."""
    import random
    df = pd.read_parquet(META)
    linked = set(df.duplicate_of.dropna().astype(int)) | set(df[df.duplicate_of.notna()].row_id.astype(int))
    tr = df[(df.defect_split == "train") & ~df.row_id.isin(linked)]
    groups = {g: sorted(int(i) for i in sub.row_id) for g, sub in tr.groupby("group6")}
    total = sum(map(len, groups.values()))
    share = {g: n * len(ids) / total for g, ids in groups.items()}
    take = {g: int(v) for g, v in share.items()}
    for g in sorted(share, key=lambda g: share[g] - take[g], reverse=True)[: n - sum(take.values())]:
        take[g] += 1                                   # largest remainders, so the total is exactly n
    rng, out = random.Random(seed), set()
    for g in sorted(groups):
        ids = groups[g][:]
        rng.shuffle(ids)
        out |= set(ids[: take[g]])
    return frozenset(out)


def load_split(split="test"):
    """Team split (2026-11-11):
      train  881  Claude silver labels (981 minus val)
      val    100  Claude silver labels, held out of train (choose checkpoints / settings, diagnose E1)
      test   185  human answers: test 122 (all fields) + test_blind 63 (group6 + visibility only; the 9 reviews
                  excluded during the human check have no answer and are left out of scoring, but still predicted)
      dev     88  no labels (HF v2 calls this split "validation"; it is NOT the team val) -> format checks only"""
    df = pd.read_parquet(META)
    if split == "test":
        df = df[df.defect_split.isin(["test", "test_blind"])]
    elif split in ("train", "val"):
        df = df[df.defect_split == "train"]
        df = df[df.row_id.isin(val_ids())] if split == "val" else df[~df.row_id.isin(val_ids())]
    elif split == "dev":
        df = df[df.defect_split == "dev"]
    else:
        raise ValueError(f"unknown split {split!r}: use train | val | test | dev")
    out = []
    for r in df.itertuples():
        imgs = [p for p in (_img_path(n) for n in r.image_files) if p]
        if r.defect_split == "test_blind":
            ok = r.human_check_excluded is not True
            gold = {"group6": r.human_check_group6 if ok else None,
                    "group5": group5_of(r.human_check_group6) if ok and r.human_check_group6 else None,
                    "visibility": r.human_check_visibility if ok else None}
        elif split == "dev":
            gold = {}
        else:
            gold = {"part_scope": r.part_scope, "part": r.part, "group6": r.group6, "group5": group5_of(r.group6),
                    "defect": r.defect, "visibility": r.visibility, "boxes": _boxes(r.boxes)}
        out.append(Sample(int(r.row_id), r.domain, r.title or "", r.review or "", imgs, gold, r.defect_split,
                          int(r.duplicate_of) if pd.notna(r.duplicate_of) else None))
    return out


# ---- the Card: one analysis result, the same shape for every model ----
@dataclass
class Card:
    row_id: int
    model: str
    valid: bool = False                           # JSON parsed and group6 / part / visibility are all allowed codes
    part_scope: str = None
    part: str = None
    group6: str = None
    group5: str = None
    defect: str = None
    visibility: str = None
    boxes: list = field(default_factory=list)     # [{image_index, bbox [x1,y1,x2,y2] 0-1 on the upright photo, target}]
    evidence: dict = field(default_factory=dict)  # customer_claim / review_span / visual_findings / claim_support / limitations
    issues: list = field(default_factory=list)    # format problems found while parsing (for error analysis, not scored)
    conf: dict = field(default_factory=dict)      # {"group6": p, "exit": p}; "exit" is what the cascade reads
    cost: dict = field(default_factory=dict)      # tokens, latency (ms), VRAM; see models.QwenVLM.analyze
    raw: str = ""

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False)


_CARD_FIELDS = {f.name for f in fields(Card)}


def save_cards(cards, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(c.to_json() for c in cards) + "\n", encoding="utf-8")


def load_cards(path):
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {c["row_id"]: Card(**{k: v for k, v in c.items() if k in _CARD_FIELDS}) for c in rows}


# ---- VLM prompt: versioned text files in cluvi/prompts/ ($name = slot) ----
PROMPT_VERSION = "vlm-v0.3.3"   # E0 (2026-11-11): output template + input reading fixed on dev; official decoding kept


def prompt_text(version=PROMPT_VERSION):
    return (PROMPT_DIR / f"{version}.txt").read_text(encoding="utf-8")


def clean_text(t):
    """The CSV keeps line breaks as a literal backslash-n (turn them into spaces) and some HTML entities (&amp; -> &)."""
    return re.sub(r"\s+", " ", html.unescape(str(t).replace("\\n", " "))).strip()


def build_prompt(s: Sample, n_images, version=PROMPT_VERSION, sizes=None):
    """sizes = (w, h) of each photo as the model sees it (after smart_resize). v0.3+ writes them into the prompt so the
    model knows each picture's pixel range for bbox_2d; older prompt files simply do not use $picture_sizes."""
    part_list = "; ".join(f"{k}: {', '.join(v)}" for k, v in parts_for(s.domain).items())
    defect_list = "\n".join(f"- group {g}: defects {', '.join(ds)}" for g, ds in DEFECTS.items())
    picture_sizes = "\n".join(f"- Picture {i}: {w} px wide x {h} px high" for i, (w, h) in enumerate(sizes or [], 1))
    # v0.3.2+: plain codes only. With "product: cover, ..." the model sometimes answered the group name "product".
    part_codes = ", ".join(p for ps in parts_for(s.domain).values() for p in ps)
    return Template(prompt_text(version)).substitute(
        domain=s.domain, title=clean_text(s.title), review=clean_text(s.review)[:1500], n_images=n_images, last_index=max(n_images - 1, 0),
        defect_list=defect_list, part_list=part_list, part_codes=part_codes, picture_sizes=picture_sizes).strip()


# ---- parsing the model's JSON into a Card ----
def _norm(t):
    return re.sub(r"\s+", " ", str(t)).strip().lower()


def _first_json(raw):
    """The first complete JSON object in the reply (ignores ```json fences and any text after the object)."""
    i = raw.find("{")
    while i != -1:
        try:
            return json.JSONDecoder().raw_decode(raw[i:])[0]
        except json.JSONDecodeError:
            i = raw.find("{", i + 1)
    return None


def _picture(item):
    """The model numbers photos like the chat template does ("Picture 1", "Picture 2", ...); we store 0-based."""
    if "picture" in item:
        return int(item["picture"]) - 1
    return int(item["image_index"])


def parse_vlm(raw, s: Sample, card: Card, sizes):
    """Fill card from the model's JSON.
    sizes = [(width, height)] of each photo AS THE MODEL SAW IT (after resizing). Qwen2.5-VL answers boxes in absolute
    pixels of that resized photo (tech report 2.1.2; the official grounding cookbook divides by image_grid_thw * 14),
    so dividing by it gives 0-1 coordinates on the upright photo. Picture numbers and box_ref are 1-based."""
    card.raw = raw[:1500]
    d = _first_json(raw)
    # a broken top-level object must not be replaced by an inner one (e.g. the first box) that happens to parse
    if not isinstance(d, dict) or not ({"category", "group6", "part", "visibility"} & set(d)):
        card.issues.append("json_parse_failed")
        return card
    cat = d.get("category")                  # v0.3.1+: one code "group6/defect" picked from a list of valid pairs
    if isinstance(cat, str) and cat.strip():
        g, _, df_ = cat.strip().partition("/")
        g, df_ = g.strip(), df_.strip()
        if df_ and df_ in DEFECT_GROUP and DEFECT_GROUP[df_] != g:
            # a pair that is not in the list (e.g. "manufacturing/price"): defect codes are unique across groups, so the
            # only list line with this defect is DEFECT_GROUP[defect]; keep that line and flag it
            card.issues.append("category_not_in_list")
            g = DEFECT_GROUP[df_]
        if df_:
            d.setdefault("group6", g)
            d.setdefault("defect", df_)
        else:                                # only one word: a defect code or a group name
            d.setdefault("defect" if g in DEFECT_GROUP else "group6", g)
    allowed = {p for ps in parts_for(s.domain).values() for p in ps}
    card.part = d.get("part") if d.get("part") in allowed else None
    card.group6 = d.get("group6") if d.get("group6") in DEFECTS else None
    defect = d.get("defect")
    if card.group6 is None and defect in DEFECT_GROUP:          # group missing but defect valid: keep the defect's group
        card.group6 = DEFECT_GROUP[defect]
        card.issues.append("group_from_defect")
    card.defect = defect if card.group6 and defect in DEFECTS[card.group6] else None
    if defect and card.defect is None:
        card.issues.append("defect_not_in_group")
    card.visibility = d.get("visibility") if d.get("visibility") in VIS else None
    card.part_scope = scope_of(card.part, s.domain) if card.part else None
    card.group5 = group5_of(card.group6) if card.group6 else None
    for f in ("part", "group6", "visibility"):
        if getattr(card, f) is None:
            card.issues.append(f"bad_{f}")
    card.valid = None not in (card.part, card.group6, card.visibility)

    boxes = d.get("boxes") if isinstance(d.get("boxes"), list) else []
    box_no = {}                                                    # model's box number (1-based) -> index in card.boxes
    pasted = []                                                    # (picture, raw numbers) of kept boxes
    for k, b in enumerate(boxes[:MAX_BOXES], 1):
        try:
            i, xy = _picture(b), [float(v) for v in b["bbox_2d"]]
            if i < 0:
                raise IndexError(i)
            w, h = sizes[i]
        except (KeyError, TypeError, ValueError, IndexError):
            card.issues.append("bad_box")
            continue
        if len(xy) != 4:
            card.issues.append("bad_box")
            continue
        if min(xy) < 0 or max(xy[0], xy[2]) > w * 1.02 or max(xy[1], xy[3]) > h * 1.02:
            card.issues.append("box_out_of_range")   # e.g. answered in original-photo pixels or 0-1000; clamped below
        bb = [min(1.0, max(0.0, v / sz)) for v, sz in zip(xy, (w, h, w, h))]
        if bb[0] >= bb[2] or bb[1] >= bb[3]:
            card.issues.append("bad_box")
            continue
        if (bb[2] - bb[0]) * (bb[3] - bb[1]) > 0.8:
            card.issues.append("box_whole_picture")      # diagnostic: covers >80% of the picture = not really a location
        # v0.3.1+ asks for "target" (defect|evidence) and keeps "label" free (Qwen's own grounding format: label = name)
        target = next((v for v in (b.get("target"), b.get("label")) if v in BOX_TARGETS), None)
        if target is None:
            card.issues.append("box_target_guessed")
        box_no[k] = len(card.boxes)
        card.boxes.append({"image_index": i, "bbox": [round(v, 4) for v in bb], "target": target or "defect"})
        pasted.append((i, tuple(xy)))
    if any(xy == xy2 and i != i2 for n, (i, xy) in enumerate(pasted) for i2, xy2 in pasted[n + 1:]):
        card.issues.append("box_copied")                 # diagnostic: the same numbers pasted onto another picture
    if card.visibility == "localized" and not card.boxes:
        card.issues.append("localized_without_box")
    if card.visibility in ("diffuse", "not_visible") and card.boxes:
        card.issues.append("box_but_not_localized")

    ev = d.get("evidence") if isinstance(d.get("evidence"), dict) else {}
    findings = []
    for v in ev.get("visual_findings") or []:
        if not isinstance(v, dict):
            continue
        try:
            pic = _picture(v)
        except (KeyError, TypeError, ValueError):
            pic = None
        ref = v.get("box_ref")
        findings.append({"image_index": pic if pic is not None and 0 <= pic < len(sizes) else None,
                         "description": str(v.get("description") or ""),
                         "box_ref": box_no.get(ref) if isinstance(ref, int) else None})
    span = str(ev.get("review_span") or "")
    card.evidence = {
        "customer_claim": str(ev.get("customer_claim") or ""),
        "review_span": span,
        "visual_findings": findings,
        "claim_support": ev.get("claim_support") if ev.get("claim_support") in CLAIM_SUPPORT else None,
        "limitations": [str(x) for x in ev.get("limitations") or [] if x][:2],
    }
    if not ev:
        card.issues.append("no_evidence")
    elif span and _norm(span) not in _norm(clean_text(f"{s.title} {s.review}")):
        card.issues.append("review_span_not_verbatim")    # quote not found word-for-word in the review = possible invention
    return card
