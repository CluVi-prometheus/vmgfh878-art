---
language:
- en
- ko
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen2.5-VL-7B-Instruct
datasets:
- CluVi-prometheus/cesamard-defect-v3
metrics:
- f1
- mean_iou
tags:
- multimodal
- zero-shot
- complaint-analysis
- defect-classification
- visual-grounding
- qwen2.5-vl
- cluvi
model-index:
- name: cluvi-heavy-e1-qwen25vl7b-zeroshot
  results:
  - task:
      type: image-text-to-text
      name: Complaint defect analysis card
    dataset:
      type: CluVi-prometheus/cesamard-defect-v3
      name: CESAMARD Defect v3
      split: test
    metrics:
    - type: f1
      value: 0.675
      name: Group6 Macro-F1
    - type: f1
      value: 0.682
      name: Group5 Macro-F1
    - type: f1
      value: 0.630
      name: Fine defect Macro-F1
    - type: f1
      value: 0.507
      name: Visibility Macro-F1
    - type: f1
      value: 0.376
      name: Part scope Macro-F1
    - type: f1
      value: 0.340
      name: Part Macro-F1
    - type: mean_iou
      value: 0.238
      name: Box mean IoU
    - type: pointing_game
      value: 0.500
      name: Box Pointing
---

# Model Card for CluVi Heavy E1 — Qwen2.5-VL-7B 제로샷 (프롬프트 v0.3.3)

불만 리뷰 한 건(영어 제목·본문과 고객 사진 최대 4장)을 읽고 상담원용 **JSON 분석 카드**를 내는 분석기입니다. 카드에는 결함 그룹/세부 결함, 부위, 사진에 보이는 정도, 박스, 근거가 들어갑니다. Qwen2.5-VL-7B-Instruct를 **학습 없이** 공식 레시피 그대로 쓰고, 프롬프트와 출력 형식만 우리 것입니다. 이후 E2·E3는 이 카드의 모델과 비교합니다.

## Model Details

### Model Description

가중치는 바꾸지 않았습니다. 그래서 이 카드가 말하는 "모델"은 아래 네 가지의 묶음(시스템)입니다.

| 구성 | 내용 |
|---|---|
| 기반 가중치 | `Qwen/Qwen2.5-VL-7B-Instruct` (snapshot `cc594898137f`) |
| 로딩 | LLM은 NF4 4비트, ViT와 `lm_head`는 bf16, attention은 sdpa |
| 프롬프트 | `cluvi/prompts/vlm-v0.3.3.txt` (sha1 `9c00598ddb7f`). 출력 틀을 보여 주고 그대로 쓰게 하는 제로샷 프롬프트 |
| 파서 | `cluvi/common.py` `parse_vlm`. JSON을 카드로 바꾸고 형식 문제를 표시합니다 |

출력 카드의 칸:
- `category`: `그룹/세부결함` 한 쌍. 그룹은 6개(damage, contamination, manufacturing, not_as_described, function, non_visual)입니다.
- `part`: 도메인별 부위 코드
- `visibility`: `localized` / `diffuse` / `not_visible`
- `boxes`: 사진 번호, `bbox_2d` 픽셀 좌표, 이름(label), 대상(`defect`/`evidence`)
- `evidence`: 사진에서 본 것, 고객 주장 요약("고객은 … 주장함"), 주장 지지 여부, 사진으로 확인 못 하는 것 1~2개

- **Developed by:** 6팀 CluVi (프로메테우스)
- **Funded by [optional]:** [More Information Needed]
- **Shared by [optional]:** 6팀 CluVi (공개 레포 `CluVi-prometheus/vmgfh878-art`. 데이터셋은 비공개)
- **Model type:** 시각-언어 모델(VLM) 기반 다중모달 결함 분석기. image-text-to-text, 제로샷 프롬프팅
- **Language(s) (NLP):** 입력은 영어(리뷰), 출력은 JSON 코드값과 한국어 근거 문장
- **License:** 기반 모델 Apache-2.0. 평가 데이터는 원본 CESAMARD-Aspect 라이선스를 따릅니다
- **Finetuned from model [optional]:** 없음. `Qwen/Qwen2.5-VL-7B-Instruct`를 그대로 4비트로 불러 씁니다

### Model Sources [optional]

- **Repository:** 실험 코드 `cluvi/` (GitHub `CluVi-prometheus/vmgfh878-art`), 기반 모델 https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- **Paper [optional]:** Qwen2.5-VL Technical Report, https://arxiv.org/abs/2502.13923
- **Demo [optional]:** `python -m cluvi.demo --model heavy --row 3 --question "교환 가능해?"` (민원 1건을 카드에서 상담원 답변까지)

## Uses

### Direct Use

- 고객 불만 리뷰 1건을 분석 카드 1장으로 바꿉니다. 상담원은 카드로 "무엇이 어디가 어떻게 잘못됐다고 하는지, 사진으로 확인되는지"를 먼저 봅니다.
- 이 카드는 CluVi 실험에서 **기준선**(E1)입니다. E2(HAWK)와 E3(LoRA)는 같은 입력·같은 채점으로 이 모델과 짝 비교합니다.

### Downstream Use [optional]

- **상담원 에이전트** `cluvi/agent.py`: 카드가 Planner → Tool(주문·정책 조회) → Answer → Verifier로 들어갑니다. Verifier는 카드의 근거 칸으로 답변을 검증합니다.
- **cascade** `cluvi/cascade.py`: 싼 분석기(글만 보는 모델)가 못 정한 리뷰를 넘겨받는 Heavy 단계 후보입니다. 단, 이 실행의 확신도 값은 버그로 기록되지 않았습니다(아래 한계 참고).

### Out-of-Scope Use

- **사람 확인 없는 자동 결정**: 환불·교환 승인이나 거절, 고객 책임 판정. 카드는 상담원의 판단 근거이지 결정이 아닙니다.
- **박스를 증거로 쓰기**: 박스 위치는 참고용입니다. test Pointing이 0.50이라 둘 중 하나는 결함 위치를 벗어납니다.
- **사진 5장 이상**: 앞의 4장만 봅니다.
- **학습 범위 밖 입력**: 영상, 비영어 리뷰, CESAMARD 5개 도메인(Books, Fashion, Electronics, Edible, Misc) 밖의 상품
- **사진 속 개인정보 다루기**: 얼굴, 송장 주소 등을 식별하거나 옮겨 적는 일

## Bias, Risks, and Limitations

- **사진 과신**:
  - val에서 정답이 `not_visible`인 38건 중 29건을 "보인다"(localized/diffuse)고 답했습니다.
  - test Visibility F1(0.507)이 글만 보는 R0(0.577)보다 낮습니다.
  - 카드가 "사진으로 확인됨"이라고 잘못 말하면, 상담원이 고객 주장을 과하게 믿거나 반대로 과하게 의심할 수 있습니다.
- **그룹 쏠림**:
  - val에서 manufacturing을 정답보다 1.58배 많이 답했습니다.
  - function 재현율은 0.50, non_visual 재현율은 0.22입니다. "작동이 안 된다"나 "서비스·가격 불만"을 겉모양 결함으로 바꿔 읽는 경향입니다.
- **부위 `none` 누락**: 서비스·가격 불만(부위 없음) 7건을 모두 `product`로 답했습니다. 그래서 Scope Macro-F1이 낮습니다(test 0.376 < R0 0.594).
- **박스 품질** (test 185건 기준):
  - mIoU 0.238, Pointing 0.50
  - 같은 박스를 다른 사진에 복사: 37건
  - 박스가 사진의 80% 넘게 덮음: 19건
  - 박스가 사진 범위 밖: 10건
- **test 점수의 낙관 가능성**:
  - 프롬프트를 만들던 계보가 라벨링 단계에서 test 122건을 보았습니다. 그래서 test 122(G6 0.762)와 test_blind 54(G6 0.492)를 따로 보고합니다.
  - 일반화 성능은 test_blind 쪽에 가깝다고 봐야 합니다.
- **val은 실버 라벨**: val 100은 Claude Opus 라벨이라, val 점수는 "Claude 기준과의 일치도"입니다.
- **확신도 미기록**: 이 실행 파일들의 `conf.group6`은 모두 0입니다. 원인은 키 이름(`category`) 버그였고 2026-11-11에 고쳤습니다. 확신도 기반 라우팅에는 다시 실행한 결과가 필요합니다.
- **형식 실패**: test 정상 출력은 98.4%입니다(잘린 출력 0). 리뷰 원문 인용(`review_span`)이 원문과 다른 경우가 test 17건 있었습니다.
- **사회기술적 위험**:
  - 영어 아마존 리뷰로만 평가했습니다. 다른 언어·쇼핑몰의 고객에게 같은 성능을 기대할 수 없습니다.
  - 고객 사진에는 개인정보가 섞일 수 있습니다.

### Recommendations

- 카드는 상담원의 **판단 보조**로만 쓰고, 최종 결정은 사람이 합니다.
- `visibility`가 `localized`/`diffuse`여도 `evidence.visual_findings`가 구체적인지 상담원이 확인하게 하세요.
- 박스는 화면에 "참고 위치"로만 표시하세요.
- 성능은 test 122와 test_blind를 나눠 인용하세요.
- 확신도로 cascade를 짜려면 버그를 고친 코드로 다시 실행한 결과를 쓰세요.

## How to Get Started with the Model

Use the code below to get started with the model.

```bash
# 프로젝트 폴더에서. HAWK 런타임(패치된 transformers 4.52.0)으로 실행합니다. HAWK는 꺼진 상태입니다.
python -m cluvi.run_hawk --model heavy --split val      # -> outputs/preds/heavy_val.jsonl + _config.json
python -m cluvi.run_hawk --model heavy --split test     # -> outputs/preds/heavy_test.jsonl
python -m cluvi.evaluate outputs/preds/text_test.jsonl outputs/preds/heavy_test.jsonl --base heavy_test --format
```

로딩만 따로 보면 다음과 같습니다(`cluvi/models.py` 121~128행과 같은 설정).

```python
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                       llm_int8_skip_modules=["visual", "model.visual", "lm_head"])     # ViT·lm_head는 bf16 유지
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", device_map="auto", attn_implementation="sdpa",
    quantization_config=q, torch_dtype="auto").eval()                                   # transformers 4.52 기준
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
# 디코딩: do_sample=False, repetition_penalty=1.05 (공식 generation_config), max_new_tokens=640
```

## Training Details

### Training Data

**우리 쪽 학습은 없습니다(제로샷).** 기반 모델의 학습 데이터는 Qwen 팀의 기술 보고서(arXiv:2502.13923)를 따릅니다. 우리 데이터의 train 881건은 이 모델에 쓰지 않았습니다.

프롬프트 결정(E0)에는 라벨이 없는 dev 88건 중 60건만 썼습니다. 형식이 깨지는지와 입력을 잘못 읽는지만 보고 판을 고정했고, 정답은 보지 않았습니다.

### Training Procedure

#### Preprocessing [optional]

추론 전처리입니다(`cluvi/common.py`, `cluvi/models.py`).

1. **사진**: `PIL.ImageOps.exif_transpose`로 세운 뒤 리뷰당 최대 4장을 씁니다. `smart_resize`로 28px 배수로 맞추고, 사진당 256~1,280 시각 토큰(`min_pixels` 200,704, `max_pixels` 1,003,520)입니다.
2. **글**: 제목·본문의 HTML 엔티티를 풀고 공백을 정리합니다(`clean_text`).
3. **채팅 틀**: 기본 시스템 프롬프트를 쓰고, 사진마다 "Picture N:" 번호(`add_vision_id`)를 붙입니다. 이어서 `qwen_vl_utils.process_vision_info`를 거칩니다.
4. **프롬프트 v0.3.3**: 카테고리 목록 27줄, 도메인별 부위 코드, 출력 JSON 틀이 들어갑니다. `customer_claim`은 "고객은"으로 시작해 "주장함"으로 끝나게 하고, `limitations`는 1~2개입니다.

#### Training Hyperparameters

- **Training regime:** 해당 없음(학습 안 함). 추론은 bf16 연산 위에 LLM 가중치만 NF4 4비트(bitsandbytes 0.50.2)이고, greedy 디코딩, `repetition_penalty` 1.05, 최대 640토큰, batch 1입니다.

#### Speeds, Sizes, Times [optional]

리뷰당 값입니다. 중앙값이고, 괄호 안은 p90입니다.

| | 시각 토큰(평균) | 출력 토큰 | Prefill | 그중 ViT | 디코딩 | E2E | 최대 VRAM |
|---|---|---|---|---|---|---|---|
| test 185 | 2,082 | 222 (372) | 2.05 s (6.78) | 0.88 s | 7.14 s (12.32) | 10.57 s (18.70) | 9.08 GB |
| val 100 | 2,157 | 226 (408) | 1.99 s (6.98) | 0.91 s | 7.23 s (14.72) | 10.57 s (22.36) | 9.11 GB |

- 실행 시간: val 100은 약 23분, test 185는 약 38분입니다(2026-11-11).
- E2E의 약 3분의 2가 디코딩(근거 문장 생성)입니다.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

[`CluVi-prometheus/cesamard-defect-v3`](https://huggingface.co/datasets/CluVi-prometheus/cesamard-defect-v3) (비공개)

- **test 185**: 사람 정답 test 122(모든 칸)와 test_blind 63(사람이 group6·visibility만 확인)을 합친 것입니다. test_blind 중 9건은 정답이 없어 채점에서 뺍니다.
  - 채점 건수: G6·G5·Vis 176건, Scope·Part·Fine 122건, 박스 56건(test 122 중 localized)
- **val 100**: Claude Opus 실버 라벨입니다(group6 비율대로 뽑음, seed 0). 오류 분석용입니다.
- **비교 기준 R0**: 글만 보는 TF-IDF + 로지스틱 회귀(train 881로 학습). 사진이 실제로 도움이 되는지 보는 참고선입니다.

#### Factors

- **test 122 vs test_blind 54**: 라벨링 계보가 본 집합인지 아닌지로 나눴습니다.
- **group6 클래스별, visibility 클래스별**: 혼동행렬은 val에서만 봤습니다. test 혼동행렬은 E3를 고정한 뒤에 봅니다.
- **val의 Claude 확신(sure) 70건 vs 헷갈림(unsure) 30건**: G6 정확도 0.786 vs 0.533
- **도메인별 분해**(5개)는 아직 계산하지 않았습니다.

#### Metrics

| 지표 | 왜 보나 |
|---|---|
| **Group6 Macro-F1** (주 지표), Group5, Fine Macro-F1 | 클래스 불균형이 커서 정확도 대신 Macro-F1을 씁니다. 상담원이 먼저 보는 결함 종류입니다 |
| Visibility Macro-F1 | 카드가 "사진으로 확인됨"을 정직하게 말하는지 봅니다(서비스 신뢰의 핵심) |
| Scope·Part Macro-F1 | 어디가 문제인지(`none` 포함) 봅니다 |
| mIoU, IoU@0.5, Pointing | 박스를 상담원 화면에 쓸 수 있는지 봅니다. Pointing은 1순위 박스 중심이 같은 사진의 정답 박스 안에 들어가는 비율입니다 |
| 토큰 유지율, Prefill, E2E, 최대 VRAM | 서비스 비용(응답 시간·GPU)입니다 |

- 신뢰구간: bootstrap 95%
- 모델 간 비교: 같은 리뷰끼리 짝지은 bootstrap 2,000회와 답 바뀜 비율

### Results

| | Scope | Part | **G6** | G5 | Fine | Vis | mIoU | IoU@.5 | Pointing | 정상 출력 |
|---|---|---|---|---|---|---|---|---|---|---|
| **E1 test** | 0.376 | 0.340 | **0.675** [0.599, 0.743] | 0.682 | 0.630 | 0.507 | 0.238 | 0.179 | 0.500 | 98.4% |
| R0 test (글만) | 0.594 | 0.299 | 0.539 | 0.547 | 0.377 | 0.577 | – | – | – | 100% |
| **E1 val** | 0.280 | 0.365 | **0.657** [0.523, 0.763] | 0.651 | 0.406 | 0.561 | 0.071 | 0.022 | 0.178 | 100% |
| R0 val (글만) | 0.395 | 0.255 | 0.621 | 0.603 | 0.373 | 0.649 | – | – | – | 100% |

| test 나눠 보기 | G6 | Vis |
|---|---|---|
| test 122 | 0.762 | 0.517 |
| test_blind 54 | 0.492 | 0.486 |

- R0와 짝 비교(test, R0 − E1): G6 −0.136 [−0.233, −0.041] → E1이 높습니다. 답이 바뀐 비율은 36.8%입니다.

#### Summary

- 사진을 함께 보는 덕에 **결함 그룹(G6)은 글만 보는 R0보다 확실히 높습니다**(test +0.136).
- 반대로 **"사진에 보이는가"(Visibility)와 "어디가 문제인가"(Scope)는 R0보다 낮습니다.**
- 즉 이 모델은 결함 종류는 잘 읽지만, 사진에 안 보이는 것도 보인다고 말하는 경향이 있습니다.
- 박스는 둘 중 하나만 위치가 맞습니다.

## Model Examination [optional]

- **val group6 혼동**: function 22건 중 6건을 manufacturing, 4건을 not_as_described로 답했습니다. non_visual 9건 중 4건은 not_as_described로 답했습니다.
- **보임 정도 × 주장 지지**: 사진에 안 보인다(`not_visible`)면서 `claim_support = supported`라고 한 경우가 4건입니다.
- **판단 순서**: 현재 출력 틀은 `visibility`를 사진 묘사(`visual_findings`)보다 먼저 쓰게 합니다. 묘사 전에 보임 여부를 정하는 구조가 과신의 원인 후보입니다(E3 제안서의 예비안 E3-K).
- 자세한 사례: `reports/Qwen_베이스라인_실험결과.html`의 val 예시 8건(박스 겹침 그림 포함)

## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** NVIDIA GeForce RTX 5060 Ti 16GB (로컬 PC 1대)
- **Hours used:** 약 1.0시간(E1 val + test). 프롬프트 점검(E0, dev)까지 넣으면 약 2.6시간
- **Cloud Provider:** 없음(로컬)
- **Compute Region:** 대한민국(로컬)
- **Carbon Emitted:** 약 0.08 kgCO2eq(E1만), E0 포함 약 0.21 kgCO2eq. 추정치이며, GPU 보드 전력 180 W × 시간 × 전력 배출계수 약 0.45 kgCO2eq/kWh로 가정한 상한입니다. CPU 등 나머지 장치는 뺐습니다.

## Technical Specifications [optional]

### Model Architecture and Objective

Qwen2.5-VL-7B-Instruct의 구조이며, 전체 파라미터는 약 82.97억 개입니다.

- **ViT**: window attention과 2D RoPE를 쓰고, 패치는 14px입니다.
- **2×2 MLP merger**: 28px 칸 하나가 시각 토큰 1개가 됩니다.
- **LLM**: Qwen2.5-7B, 28층, M-RoPE

목적은 다음 토큰 생성입니다. 출력 JSON을 한 번의 생성으로 씁니다.

### Compute Infrastructure

로컬 PC 1대, GPU 1장, batch 1입니다.

#### Hardware

NVIDIA GeForce RTX 5060 Ti (16GB). 추론 최대 VRAM은 9.1GB입니다.

#### Software

HAWK 런타임(`HAWK/.runtime`, HAWK 저장소가 패치한 transformers 4.52.0)을 HAWK를 끈 채로 씁니다.

- torch 2.11.0+cu128
- qwen-vl-utils 0.0.11
- bitsandbytes 0.50.2
- accelerate 1.15

설정 전체는 `outputs/preds/heavy_test_config.json`에 기록돼 있습니다.

## Citation [optional]

**BibTeX:**

```bibtex
@misc{qwen2.5-VL,
    title = {Qwen2.5-VL},
    url = {https://qwenlm.github.io/blog/qwen2.5-vl/},
    author = {Qwen Team},
    month = {January},
    year = {2025}
}

@article{Qwen2VL,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng and Bai, Shuai and Tan, Sinan and Wang, Shijie and Fan, Zhihao and Bai, Jinze and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Fan, Yang and Dang, Kai and Du, Mengfei and Ren, Xuancheng and Men, Rui and Liu, Dayiheng and Zhou, Chang and Zhou, Jingren and Lin, Junyang},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}
```

(기반 모델 카드에 적힌 인용 그대로입니다. 기술 보고서는 arXiv:2502.13923)

**APA:**

Qwen Team. (2025). *Qwen2.5-VL*. https://qwenlm.github.io/blog/qwen2.5-vl/

## Glossary [optional]

| 용어 | 뜻 |
|---|---|
| G6 / G5 | 결함 6그룹 / damage와 manufacturing을 합친 5그룹 |
| Fine | 세부 결함(`category`의 `/` 뒤) |
| Scope / Part | 부위 묶음(product, packaging, accessory, label_doc, none, other) / 세부 부위 코드 |
| Vis | 보임 정도: localized(한 곳에 보임), diffuse(넓게 보임), not_visible(사진으로 확인 불가) |
| Pointing | 1순위 박스 중심이 같은 사진의 정답 박스 안에 드는 비율 |
| 실버 / 사람 정답 | Claude Opus가 붙인 라벨 / 사람이 붙이거나 확인한 라벨 |
| test_blind | 라벨링 때 프롬프트 계보가 보지 않은 test 부분 집합. 사람이 group6·visibility만 확인했습니다 |
| R0 | 글만 보는 TF-IDF 기준선 |
| E0 | 프롬프트·형식 점검 단계(dev, 정답 안 봄). 실험 수에 넣지 않습니다 |

## More Information [optional]

- 설계와 판정 기준: `reports/초기실험_설계_v0.1.md`
- 실행 단위 수치: `reports/중간실험_결과_기록.md`
- 보고서(예시 포함): `reports/Qwen_베이스라인_실험결과.html`
- 같은 묶음의 다른 카드: [E2 (+HAWK)](../e2-qwen25vl7b-hawk020/README.md), [E3-L (QLoRA)](../e3l-qwen25vl7b-qlora-head/README.md)

## Model Card Authors [optional]

6팀 CluVi — GitHub `vmgfh878-art` (초안 작성 보조: Claude)

## Model Card Contact

6팀 CluVi (GitHub org `CluVi-prometheus`)
