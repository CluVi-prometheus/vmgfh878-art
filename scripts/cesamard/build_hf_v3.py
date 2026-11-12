"""CESAMARD defect v3 = the team's baseline-experiment split (decided 2026-11-11), as a private HF dataset.

  train       881  Claude Opus silver labels (v2 train 981 minus validation)
  validation  100  Claude Opus silver labels, held out of train: group6-stratified, seed 0, no duplicate-linked texts
  test        185  human answers: v2 test 122 (all fields) + v2 test_blind 63 (group6 + visibility only;
                   9 of them were excluded during the human check -> no answer, not scored)
  dev          88  no labels (v2 called this split "validation"); used only for output-format checks (E0)

Rows and columns are v2's (same row_id, same labels, same text-sufficiency columns, read from the v2 repo), plus
  v3_split    the split above
  eval_group6 / eval_group5 / eval_visibility   the answer the experiments score against
  eval_fields  "all" (train / validation / test from v2 test) | "group6+visibility" (from v2 test_blind) | "none"
The split is exactly cluvi.common.load_split(), so the HF data and our experiment code can never disagree.

  python scripts/cesamard/build_hf_v3.py build                 # -> data/cesamard/hf_v3/ (parquet per split + README + splits.json)
  python scripts/cesamard/build_hf_v3.py push --repo CluVi-prometheus/cesamard-defect-v3
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cluvi.common import META, VAL_N, VAL_SEED, group5_of, load_split  # noqa: E402

OUT = ROOT / "data" / "cesamard" / "hf_v3"
V2 = "CluVi-prometheus/cesamard-defect-v2"
V2_FILES = {"train": "train", "validation": "dev", "test": "test", "test_blind": "test_blind"}
SUFF = ["text_sufficient", "part_sufficient", "defect_sufficient", "text_part", "text_defect", "text_part_scope"]
SPLITS = {"train": "train", "validation": "val", "test": "test", "dev": "dev"}     # HF name -> load_split name


def v2_sufficiency():
    """Text-sufficiency columns from v2 (teacher-LLM silver labels), read column-wise without the photo bytes."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    parts = []
    for f in fs.glob(f"datasets/{V2}/data/*.parquet"):
        with fs.open(f, "rb") as fh:
            parts.append(pq.read_table(fh, columns=["row_id"] + SUFF).to_pandas())
    df = pd.concat(parts, ignore_index=True)
    assert df.row_id.is_unique, "v2 has a row twice"
    return df


def build():
    meta = pd.read_parquet(META)
    suff = v2_sufficiency()
    print(f"v2 text-sufficiency rows: {len(suff)} (labeled {suff.text_sufficient.notna().sum()})")
    frames, ids = {}, {}
    for hf, ours in SPLITS.items():
        samples = load_split(ours)
        rid = [s.row_id for s in samples]
        df = meta.set_index("row_id").loc[rid].reset_index()
        df["v3_split"] = hf
        df["eval_group6"] = [s.gold.get("group6") for s in samples]
        df["eval_group5"] = [s.gold.get("group5") for s in samples]
        df["eval_visibility"] = [s.gold.get("visibility") for s in samples]
        df["eval_fields"] = ["none" if not s.gold.get("group6") else
                             "group6+visibility" if s.subset == "test_blind" else "all" for s in samples]
        df = df.merge(suff, on="row_id", how="left", validate="one_to_one")
        frames[hf], ids[hf] = df, rid
    allid = [i for v in ids.values() for i in v]
    assert len(allid) == len(set(allid)), "a review is in two splits"
    tr, va = set(ids["train"]), set(ids["validation"])
    dup = meta[meta.duplicate_of.notna()]
    for a, b in zip(dup.row_id.astype(int), dup.duplicate_of.astype(int)):     # same text never on both sides
        assert not ({a, b} & tr and {a, b} & va), f"duplicate pair {a}/{b} split across train/validation"
    OUT.mkdir(parents=True, exist_ok=True)
    for hf, df in frames.items():
        df.to_parquet(OUT / f"{hf}.parquet", index=False)
        print(f"{hf:10s} {len(df):4d} | eval_fields {df.eval_fields.value_counts().to_dict()} | "
              f"group6 {df.eval_group6.value_counts().to_dict()}")
    (OUT / "splits.json").write_text(json.dumps({
        "rule": f"validation = {VAL_N} of v2 train, stratified by group6 (largest remainder), random.Random({VAL_SEED}) "
                "shuffle per group, only reviews not linked to a duplicate; train = the rest of v2 train; "
                "test = v2 test + v2 test_blind; dev = v2 validation (no labels)",
        "code": "cluvi/common.py val_ids() / load_split()",
        **{k: sorted(v) for k, v in ids.items()}}, indent=1), encoding="utf-8")
    (OUT / "README.md").write_text(card(frames), encoding="utf-8")
    print(f"wrote {OUT}")


def card(frames):
    def counts(df, col):
        return ", ".join(f"{k} {v}" for k, v in df[col].value_counts().items())
    rows = "\n".join(f"| {hf} | {len(df)} | {src} | {counts(df, 'eval_fields')} |" for (hf, df), src in zip(
        frames.items(), ["Claude Opus (silver)", "Claude Opus (silver)", "human", "-"]))
    g6 = "\n".join(f"| {g} | " + " | ".join(str(int((frames[s].eval_group6 == g).sum())) for s in ("train", "validation", "test"))
                   + " |" for g in ["damage", "not_as_described", "function", "manufacturing", "non_visual", "contamination"])
    return f"""---
language:
  - en
  - ko
task_categories:
  - image-classification
  - text-classification
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*.parquet
      - split: validation
        path: data/validation-*.parquet
      - split: test
        path: data/test-*.parquet
      - split: dev
        path: data/dev-*.parquet
---

# CluVi — CESAMARD Defect v3 (baseline experiment split)

v2와 같은 리뷰·라벨·사진·텍스트 충분성 컬럼을 그대로 두고, **팀이 2026-11-11에 정한 실험 분할**만 새로 붙인 버전입니다.
실버(Claude Opus 라벨)로 학습하고, 실버 일부를 떼어 검증하고, 사람 정답으로 평가합니다.

| split | 리뷰 수 | 라벨 출처 | 채점 범위 (eval_fields) |
|---|---:|---|---|
{rows}

- **train 881** = v2 train 981에서 validation 100을 뺀 나머지. 학습(LoRA 등)에만 씁니다.
- **validation 100** = v2 train에서 group6 비율대로 뽑은 100건 (seed 0 고정, 누구나 같은 100건). 중복 리뷰(같은 글)와 묶인 리뷰는 뽑지 않아 train과 같은 글이 섞이지 않습니다. 설정·체크포인트 선택과 오류 분석에 씁니다.
- **test 185** = v2 test 122 (사람 정답, 모든 필드) + v2 test_blind 63 (사람이 group6·visibility만 확인). test_blind 중 9건은 사람 확인에서 제외되어 정답이 없으므로 채점에서 빠집니다(`eval_fields = none`). 최종 성능은 여기서 **한 번만** 잽니다.
- **dev 88** = v2의 "validation". 라벨이 없으며 출력 형식 점검(E0)에만 씁니다. 팀의 validation과 다릅니다.

## 불러오기

```python
from datasets import load_dataset
ds = load_dataset("CluVi-prometheus/cesamard-defect-v3", token=True)   # 비공개: `hf auth login` 먼저 (CluVi-prometheus 멤버)
row = ds["train"][0]
row["title"], row["review"], row["images"]      # images = PIL 사진 목록 (리뷰당 평균 1.9장)
row["group6"], row["defect"], row["part"], row["visibility"], row["boxes"]
# 사진 없이 표만: load_dataset(..., columns=[...]) 또는 data/*.parquet를 pandas로
```

## 컬럼 안내 (사진 포함 48개 = v2 컬럼 + v3 컬럼 5개)

| 묶음 | 컬럼 | 뜻 |
|---|---|---|
| 입력 | `title`, `review`, `images`, `image_files`, `domain` | 리뷰 제목 · 본문(영어), 고객 사진, 도메인 (Books · Fashion · Electronics · Edible · Misc) |
| 입력(번역) | `title_ko`, `text_ko` | 일부 리뷰의 한국어 번역 |
| 결함 라벨 | `group6`, `group5`, `defect`, `part_scope`, `part`, `part_text`, `visibility`, `boxes` | 6그룹 · 5그룹(damage+manufacturing 합침) · 세부 결함 · 부위 묶음 · 부위 · 보임 정도(localized / diffuse / not_visible) · 박스 (`image_index`, `bbox` 0~1 좌표 [x1, y1, x2, y2], `target` defect/evidence) |
| 라벨 출처 | `label_source`, `defect_split`, `claude_unsure`, `human_check_*`, `duplicate_of`, `boxes_dropped` | 누가 붙였나(claude_opus / human_round2 / human_gold90), v2 분할, Claude가 그룹을 헷갈려 한 표시, 사람 확인값(test_blind), 같은 글 중복 원본 |
| 텍스트 충분성 (v2) | `text_sufficient`, `part_sufficient`, `defect_sufficient`, `text_part`, `text_defect`, `text_part_scope` | **글만 보고** 부위 · 결함을 정할 수 있는지와, 글만 보고 고른 부위 · 결함 (teacher LLM 실버 라벨, 1,244건). 라우팅 · 사진 필요 여부 연구용 |
| 원본 (CESAMARD-Aspect) | `complaint`, `sentiment`, `emotion`, `severity`, `sarcastic`, `aspects`, `aspect_complaint`, `helpful_votes`, `review_date`, `page_url`, `image_urls` | 원 데이터셋의 라벨 · 메타데이터 |
| **v3 채점용** | `v3_split`, `eval_group6`, `eval_group5`, `eval_visibility`, `eval_fields` | 이 분할 이름, 채점 정답 (test_blind는 사람 확인값, 나머지는 원래 라벨), 채점 범위 `all` / `group6+visibility` / `none` |

- **사진 회전**: 휴대폰 사진이라 EXIF 회전 정보가 있음. 박스 좌표는 세운 사진 기준이므로 `PIL.ImageOps.exif_transpose(img)`를 먼저 적용
- 채점: group6 · visibility는 `eval_*` 컬럼으로, part · defect · boxes는 원래 컬럼으로 하되 `eval_fields = all`인 행만 씁니다.
- 지표 n (test): group6 · visibility 176건 (122 + 54), part · defect 122건, 박스 56건 (test 122 중 localized).

## group6 분포 (채점 대상)

| group6 | train | validation | test |
|---|---:|---:|---:|
{g6}

## 재현
분할 규칙과 row_id 목록은 `splits.json`에 있습니다. 코드: `cluvi/common.py`의 `val_ids()` / `load_split()`.

## 주의
- train·validation 라벨은 Claude Opus가 만든 실버 라벨이고, 텍스트 충분성 컬럼(`text_sufficient` 등)은 teacher LLM이 만든 실버 라벨입니다(v2와 동일).
- 원본 데이터셋 CESAMARD-Aspect의 라이선스를 따릅니다(`CluVi-prometheus/cesamard-defect`의 `original/` 참고).
"""


def push(repo):
    from datasets import Dataset, DatasetDict, Image, Sequence
    from huggingface_hub import HfApi
    sys.path.insert(0, str(ROOT))
    from cluvi.common import _img_path
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    dfs = {}
    for hf in SPLITS:
        df = pd.read_parquet(OUT / f"{hf}.parquet")
        df["images"] = [[str(p) for p in (_img_path(n) for n in files) if p] for files in df.image_files]
        missing = sum(len(a) != len(b) for a, b in zip(df.images, df.image_files))
        print(f"{hf}: {len(df)} rows, rows with a missing photo file: {missing}")
        dfs[hf] = df
    # column types from all splits together (a column that is empty in one split, e.g. human_check_* in train,
    # would otherwise get type null there and the splits would not match)
    feats = Dataset.from_pandas(pd.concat(dfs.values(), ignore_index=True), preserve_index=False).features
    dd = {hf: Dataset.from_pandas(df, features=feats, preserve_index=False).cast_column("images", Sequence(Image()))
          for hf, df in dfs.items()}
    DatasetDict(dd).push_to_hub(repo, private=True, max_shard_size="500MB")
    api.upload_file(path_or_fileobj=str(OUT / "splits.json"), path_in_repo="splits.json", repo_id=repo, repo_type="dataset")
    api.upload_file(path_or_fileobj=(OUT / "README.md").read_bytes(), path_in_repo="README.md", repo_id=repo,
                    repo_type="dataset")
    info = api.dataset_info(repo, files_metadata=True)
    gb = sum((s.size or 0) for s in info.siblings) / 2**30
    print(f"pushed https://huggingface.co/datasets/{repo} ({'private' if info.private else 'PUBLIC'}) | "
          f"{len(info.siblings)} files, {gb:.2f} GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "push"])
    ap.add_argument("--repo")
    a = ap.parse_args()
    build() if a.cmd == "build" else push(a.repo)
