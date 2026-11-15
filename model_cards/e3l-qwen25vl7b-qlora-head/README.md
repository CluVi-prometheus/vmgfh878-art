---
language:
- en
- ko
license: apache-2.0
library_name: peft
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen2.5-VL-7B-Instruct
base_model_relation: adapter
datasets:
- CluVi-prometheus/cesamard-defect-v3
metrics:
- f1
- mean_iou
tags:
- multimodal
- lora
- qlora
- peft
- complaint-analysis
- defect-classification
- qwen2.5-vl
- cluvi
model-index:
- name: cluvi-heavy-e3l-qwen25vl7b-qlora-head
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
      value: 0.761
      name: Group6 Macro-F1
    - type: f1
      value: 0.773
      name: Group5 Macro-F1
    - type: f1
      value: 0.719
      name: Fine defect Macro-F1
    - type: f1
      value: 0.720
      name: Visibility Macro-F1
    - type: f1
      value: 0.656
      name: Part scope Macro-F1
    - type: f1
      value: 0.547
      name: Part Macro-F1
    - type: mean_iou
      value: 0.223
      name: Box mean IoU
    - type: pointing_game
      value: 0.518
      name: Box Pointing
---

# Model Card for CluVi Heavy E3-L — Qwen2.5-VL-7B + QLoRA 어댑터 (답의 앞머리만 학습)

> **상태: 고정(2026-11-15).**
> - 미리 정한 규칙으로 에폭 3을 골랐습니다.
> - val 성공 기준 9/9를 통과했습니다.
> - test 185는 한 번만 돌렸습니다.

E1([카드](../e1-qwen25vl7b-zeroshot/README.md))에 **LoRA 어댑터 하나만** 더한 모델입니다. 어댑터는 답의 앞머리 세 칸(`category`, `part`, `visibility`)을 train 881건으로 학습했습니다.

결과(test, 사람 정답):
- **Visibility 0.507 → 0.720**, **Scope 0.376 → 0.656**
- **Group6 0.675 → 0.761** (짝 차이 +0.086 [−0.000, +0.164])
- 박스는 그대로이고, 응답은 29% 느려졌습니다.

## Model Details

### Model Description

| 구성 | E1 | **E3-L** |
|---|---|---|
| 기반 가중치 · 로딩 | `Qwen/Qwen2.5-VL-7B-Instruct`, LLM NF4 4비트, ViT·`lm_head` bf16 | 같음(얼림) |
| **어댑터** | 없음 | **LoRA**: LLM 28층의 q·k·v·o_proj, 112개 모듈, r 8, alpha 16, 에폭 3. 병합하지 않고 추론 때 얹습니다 |
| 프롬프트 · 파서 · 디코딩 | v0.3.3, `parse_vlm`, greedy · rp 1.05 · 640토큰 | 같음 |
| HAWK | 끔 | 끔 |

**왜 LoRA인가**(E3 제안서 요약): 글만 보는 R0가 Visibility·Scope에서 E1을 이깁니다. 즉 E1의 오류는 "사진을 못 봐서"가 아니라 "무엇을 not_visible·function·none으로 칠지"라는 기준 문제입니다. 기준은 프롬프트 문장보다 예시 881건으로 가르치는 편이 맞다고 판단했습니다.

**왜 앞머리만인가**: loss는 답의 앞머리 토큰 28~35개에만 겁니다(` ```json\n{\n  "category": …, "part": …, "visibility": …, `). 그 뒤의 박스와 근거 문장은 학습하지 않아, 원래 모델이 쓰던 대로 남깁니다. 실버 라벨에는 사람이 확인한 박스·근거가 없기 때문입니다.

- **Developed by:** 6팀 CluVi (프로메테우스)
- **Funded by [optional]:** [More Information Needed]
- **Shared by [optional]:** 6팀 CluVi (공개 레포 `CluVi-prometheus/vmgfh878-art`. 데이터셋·어댑터 가중치는 비공개)
- **Model type:** VLM + PEFT LoRA 어댑터(QLoRA 방식 학습). image-text-to-text
- **Language(s) (NLP):** 입력은 영어, 출력은 JSON 코드값과 한국어 근거 문장
- **License:** 기반 모델 Apache-2.0. 학습 데이터는 원본 CESAMARD-Aspect 라이선스를 따릅니다(어댑터 배포 조건은 팀이 정해야 합니다)
- **Finetuned from model [optional]:** `Qwen/Qwen2.5-VL-7B-Instruct` (snapshot `cc594898137f`)

### Model Sources [optional]

- **Repository:**
  - 학습 `cluvi/train_lora.py`
  - 학습 데이터·loss 마스크 `cluvi/lora_data.py`
  - 실행 체인 `scripts/run_e3l_chain.py`
  - val 판정 `cluvi/judge_e3.py`
  - 학습 기록 `outputs/lora/e3l/`
  - 어댑터 가중치(20MB)는 레포에 넣지 않았습니다
- **Paper [optional]:**
  - LoRA, https://arxiv.org/abs/2106.09685
  - QLoRA, https://arxiv.org/abs/2305.14314
  - Qwen2.5-VL, https://arxiv.org/abs/2502.13923
- **Demo [optional]:** [More Information Needed]

## Uses

### Direct Use

- 리뷰 1건을 분석 카드 1장으로 바꾸는 일입니다(E1과 같은 출력 형식).
- E1보다 결함 그룹·보임 정도·부위 범위를 더 정확하게 씁니다. 특히 **사진으로 확인 안 되는 주장에 `not_visible`을 정직하게 씁니다**(test not_visible 재현율 0.125 → 0.734).

### Downstream Use [optional]

- **상담원 에이전트 `cluvi/agent.py`**: 기반 모델 하나에 어댑터를 켜고 끌 수 있습니다. 그래서 한 GPU에 7B 모델 하나만 올리고, 라벨 단계만 어댑터를 켜는 구성이 가능합니다.
- **나중에 얹을 수 있는 것**: E3 제안서의 예비안 E3-G(박스를 사진별로 따로 묻기)와 간단한 보임 규칙은 학습과 독립이라 이 어댑터 위에 더할 수 있습니다. 반면 키 순서 바꾸기(E3-K)는 학습 데이터의 답 순서와 묶여 있어 다시 학습해야 합니다.

### Out-of-Scope Use

- **박스로 결함 위치를 보여 주는 용도**: 박스는 학습하지 않아 E1 수준 그대로입니다(Pointing 0.518).
- **사람 정답이 확인한 수준으로 믿는 것**: 학습 라벨은 Claude Opus 실버입니다.
- **응답 시간이 빡빡한 화면**: 어댑터를 병합하지 않아 E2E가 E1보다 29% 깁니다.
- E1 카드의 금지 용도도 그대로 적용됩니다: 사람 확인 없는 자동 결정, 사진 5장 이상, 영상·비영어·도메인 밖 입력, 개인정보 식별.

## Bias, Risks, and Limitations

- **오른 폭의 절반은 실버 버릇일 수 있음**:
  - G6 짝 차이가 val(실버)에서 +0.173이었고, test(사람 정답)에서는 +0.086으로 절반 정도입니다. val에서 오른 몫의 일부는 Claude 라벨 방식을 따라 한 것으로 보입니다.
  - 반면 Visibility는 val +0.18, test +0.21로 사람 정답에서도 그대로 올랐습니다.
- **G6 개선은 경계선**: test 176건의 짝 차이 구간 하한이 −0.0003입니다. "구분 못 함"과 "더 높음"의 경계에 있습니다.
  - 안 본 test_blind 54건만 보면 +0.170 [+0.058, +0.288]로 0 위에 있습니다.
  - test 122는 +0.047 [−0.063, +0.160]입니다.
- **과교정**: not_visible을 더 잘 잡는 대신, 정답이 localized인 것을 not_visible로 답하는 실수가 새로 생겼습니다. localized 재현율은 test 0.962 → 0.861, val 0.867 → 0.711입니다.
- **manufacturing 재현율 하락**: test 0.82 → 0.70입니다. 44건 중 damage 5, function 3, not_as_described 3, 기타 2건으로 갔습니다.
- **박스 그대로**:
  - mIoU 0.238 → 0.223, IoU@0.5 0.179 → 0.161, Pointing 0.500 → 0.518입니다.
  - 박스 문제가 있는 리뷰는 57 → 28건으로 줄었습니다. 하지만 상당수는 not_visible로 바뀌어 박스를 아예 안 그린 경우입니다.
- **지연**:
  - E2E 중앙값이 10.6 → 13.6초(+29%)입니다. val에서 토큰당 디코딩이 32.0 → 47.8 ms(+49%)였습니다.
  - 원인은 4비트 모델 위에 병합하지 않은 어댑터를 얹어 쓰기 때문입니다.
  - 병합하려면 bf16으로 합친 뒤 다시 양자화해야 하고, 그러면 출력이 조금 달라집니다.
- **에폭을 val로 골랐음**: val 점수가 조금 낙관적일 수 있습니다. 다만 에폭 간 선택 점수 차이는 0.023 이내입니다.
- **라벨 손질 1건**: row 2654(Electronics, `body_fabric`)는 허용 부위 목록에 없어 `other`로 바꿨습니다.
- **E1에서 이어지는 한계**: 영어 아마존 리뷰로만 평가했고, test 122는 프롬프트 계보가 라벨링 때 본 집합입니다.

### Recommendations

- 성능은 **test 122와 test_blind를 나눠** 인용하세요. G6 개선은 "test_blind에서 유의, 전체로는 경계선"이라고 쓰는 것이 정확합니다.
- 박스는 참고용으로만 쓰세요. 박스가 필요하면 E3-G 같은 별도 단계를 더하세요.
- 서비스에 넣기 전에 어댑터 병합 + 재양자화가 출력을 얼마나 바꾸는지 확인하세요(지연 +29% 해소용).
- manufacturing 결함이 많은 도메인에서는 재현율 하락을 따로 감시하세요.

## How to Get Started with the Model

Use the code below to get started with the model.

```bash
# 학습 (HAWK 런타임, HAWK 끔). 중간에 멈추면 --resume
python -m cluvi.train_lora --probe 10     # GPU 점검: 가장 긴 입력 10건 + 무작위 5건 -> VRAM·속도·시간 추정
python -m cluvi.train_lora                # 3에폭, 에폭마다 val 앞머리 점검 -> outputs/lora/e3l/epoch{1,2,3}/
# 고른 에폭(3)으로 E1과 같은 형식의 전체 실행과 판정
python -m cluvi.run_hawk --model heavy --split val  --adapter outputs/lora/e3l/epoch3   # -> heavy_lora-e3l-epoch3_val
python -m cluvi.judge_e3 outputs/preds/heavy_lora-e3l-epoch3_val.jsonl                    # 성공 기준을 E1(heavy_val)과 비교
python -m cluvi.run_hawk --model heavy --split test --adapter outputs/lora/e3l/epoch3   # test는 한 번만
# 전 과정을 한 번에 (점검 -> 어댑터 로딩 확인 -> 학습 -> val -> 판정, test 전에서 멈춤)
.venv/Scripts/python.exe scripts/run_e3l_chain.py
```

어댑터만 따로 얹는 방법입니다(PEFT 0.21.0).

```python
from peft import PeftModel
# base_4bit_model = E1 카드의 로딩 코드로 불러온 모델 (cluvi/models.py 147~150행과 같은 방식)
model = PeftModel.from_pretrained(base_4bit_model, "outputs/lora/e3l/epoch3", is_trainable=False)  # 병합하지 않음
```

## Training Details

### Training Data

[`CluVi-prometheus/cesamard-defect-v3`](https://huggingface.co/datasets/CluVi-prometheus/cesamard-defect-v3)의 **train 881건**을 씁니다.

- 전부 Claude Opus 실버 라벨입니다(`label_source = claude_opus`).
- Claude가 그룹을 헷갈려 한 행(`claude_unsure`) 239건은 loss에 0.5를 곱합니다.
- 제외한 행은 없습니다. 가장 긴 입력까지 메모리에 들어가 `max_visual_tokens` 제한을 쓰지 않았습니다. seed는 0입니다.
- val 100은 train과 같은 글이 섞이지 않게 뽑았습니다(group6 비율, seed 0, 중복 글 제외).

### Training Procedure

#### Preprocessing [optional]

- **입력**: E1과 **같은 코드로** 만듭니다(EXIF 회전, 사진 최대 4장, 사진당 256~1,280 토큰, "Picture N:", 기본 시스템 프롬프트, 프롬프트 v0.3.3, sha1 `9c00598ddb7f`).
- **정답**: v0.3.3 답 형식의 앞머리입니다. E1 원문이 시작하는 토큰과 같게 맞췄습니다(20건 확인). 종료 토큰은 넣지 않습니다.
- **loss 마스크**: 앞머리 토큰 28~35개에만 겁니다. 로짓도 그 위치에서만 계산합니다(전체 로짓을 만들지 않아 VRAM 절약).
- **입력 길이**: 중앙값 2,934토큰, 상위 10% 5,360토큰, 최대 6,962토큰입니다.

#### Training Hyperparameters

- **Training regime:** QLoRA 방식입니다. 기반 LLM은 NF4 4비트로 얼리고(연산 bf16), ViT·`lm_head`·임베딩도 얼립니다. LoRA 어댑터 가중치는 fp32입니다. gradient checkpointing은 LLM에만 씁니다(non-reentrant).

| 항목 | 값 |
|---|---|
| LoRA 대상 | `model.language_model.layers.*.self_attn.(q_proj, k_proj, v_proj, o_proj)`, 28층 × 4 = 112개 모듈. ViT attention(`attn.qkv`, `attn.proj`)은 건드리지 않음 |
| r / alpha / dropout | 8 / 16 / 0.1 |
| 학습 변수 | 5,046,272개 (전체 약 82.97억의 0.061%) |
| 옵티마이저 | AdamW, lr 1e-4, weight decay 0 |
| 스케줄 | cosine, warmup 10%, 기울기 자르기 1.0 |
| 배치 | 1 × 누적 8 (에폭당 111번 갱신, 3에폭 총 333번) |
| loss | 앞머리 토큰 교차 엔트로피 평균 × 행 가중치(헷갈림 0.5) |
| **에폭 고르는 규칙 (학습 전에 정함)** | 에폭마다 val 100에서 "앞머리 점검"을 합니다(greedy로 visibility까지만 48토큰 생성. 전체 실행의 앞머리와 같은 답이며, 385건에서 불일치 0). **val G6·Vis·Scope Macro-F1 평균이 가장 높은 에폭**을 고르고, 같으면 앞 에폭을 고릅니다. **결과: 에폭 3** |

#### Speeds, Sizes, Times [optional]

- **학습**: 333번 갱신에 5시간 39분(2026-11-15). 스텝(리뷰 8건)당 약 45~65초입니다.
- **에폭별 평균 손실**: 0.096 → 0.059 → 0.042
- **학습 최대 VRAM**: 13.0GB (16GB 중)
- **어댑터 크기**: 20MB (`adapter_model.safetensors`)
- **에폭별 val 앞머리 점검**: 약 10분(리뷰당 6.3초)

추론(test 185, 리뷰당 중앙값, 괄호 안은 p90):

| | 출력 토큰 | Prefill | 디코딩 | E2E | 최대 VRAM |
|---|---|---|---|---|---|
| E1 | 222 (372) | 2.05 s (6.78) | 7.14 s (12.32) | 10.57 s (18.70) | 9.08 GB |
| **E3-L** | 214 (314) | 2.15 s (6.99) | 10.44 s (15.51) | **13.59 s (21.31), +29%** | 9.09 GB |

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

E1과 같습니다.

- **test 185**(사람 정답): G6·Vis 176건, Scope·Part·Fine 122건, 박스 56건. 에폭과 설정을 고정한 뒤 **한 번만** 돌렸습니다.
- **val 100**(실버): 에폭 선택과 성공 기준 판정에 썼습니다.

#### Factors

- test 122 vs test_blind 54
- val의 Claude 확신 70건 vs 헷갈림 30건
- group6·visibility 클래스별 재현율. 특히 function, non_visual, not_visible, 그리고 반대쪽 과잉을 볼 localized

#### Metrics

E1과 같은 13지표입니다([E1 카드](../e1-qwen25vl7b-zeroshot/README.md#metrics)). 여기에 **성공 기준(미리 정함, val 100에서 E1과 비교)**을 봅니다.

| # | 기준 |
|---|---|
| 1 | Group6 Macro-F1 +0.05 이상, function 재현율 0.65 이상, non_visual 재현율 0.45 이상 |
| 2 | Visibility Macro-F1 0.69 이상(간단한 규칙 참고선 0.688 초과), not_visible 재현율 0.60 이상 |
| 3 | Scope Macro-F1 0.395 이상 (R0) |
| 4 | 형식 유지: 정상 출력 95% 이상, `review_span` 원문 일치 90% 이상, 박스 문제 리뷰 수가 E1보다 늘지 않음 |

### Results

**에폭별 val 앞머리 점검** (같은 채점으로 E1은 G6 0.657, Vis 0.561, Scope 0.280, 평균 0.499)

| 에폭 | G6 | Vis | Scope | 평균 |
|---|---|---|---|---|
| 1 | 0.819 | 0.745 | 0.478 | 0.681 |
| 2 | 0.822 | 0.730 | 0.552 | 0.702 |
| **3 (선택)** | 0.830 | 0.740 | 0.543 | **0.704** |

**val 성공 기준 (실버 정답): 9/9 통과**

| 기준 | E1 | E3-L | 문턱 | 통과 |
|---|---|---|---|---|
| Group6 Macro-F1 | 0.657 | **0.830** | ≥ 0.707 | ✅ |
| function 재현율 | 0.500 | 0.955 | ≥ 0.65 | ✅ |
| non_visual 재현율 | 0.222 | 0.778 | ≥ 0.45 | ✅ |
| Visibility Macro-F1 | 0.561 | 0.740 | ≥ 0.69 | ✅ |
| not_visible 재현율 | 0.237 | 0.842 | ≥ 0.60 | ✅ |
| Scope Macro-F1 | 0.280 | 0.543 | ≥ 0.395 | ✅ |
| 정상 출력 | 1.000 | 1.000 | ≥ 0.95 | ✅ |
| review_span 원문 일치 | 0.920 | 0.910 | ≥ 0.90 | ✅ (간신히) |
| 박스 문제가 있는 리뷰 수 | 33 | 20 | ≤ 33 | ✅ |

- val 짝 차이: G6 +0.173 [+0.063, +0.297], 답 바뀜 28%
- Claude 확신/헷갈림 행의 G6 정확도: 확신 0.786 → 0.957, 헷갈림 0.533 → 0.600. 확신 행에서 더 많이 올라 "버릇만 배운" 신호는 아닙니다.

**test 185 (사람 정답, 한 번)**

| | Scope | Part | **G6** | G5 | Fine | Vis | mIoU | IoU@.5 | Pointing | 정상 출력 |
|---|---|---|---|---|---|---|---|---|---|---|
| R0 (글만) | 0.594 | 0.299 | 0.539 | 0.547 | 0.377 | 0.577 | – | – | – | 100% |
| E1 | 0.376 | 0.340 | 0.675 | 0.682 | 0.630 | 0.507 | 0.238 | 0.179 | 0.500 | 98.4% |
| E2 (+HAWK) | 0.368 | 0.372 | 0.678 | 0.676 | 0.639 | 0.483 | 0.142 | 0.071 | 0.268 | 97.8% |
| **E3-L** | **0.656** | **0.547** | **0.761** [0.674, 0.828] | 0.773 | 0.719 | **0.720** | 0.223 | 0.161 | 0.518 | **100%** |

| 짝 비교 (E3-L − E1, G6) | 차이 | 95% 구간 | 답 바뀜 |
|---|---|---|---|
| test 전체 176 | +0.086 | [−0.000, +0.164] | 20.5% |
| test 122 | +0.047 | [−0.063, +0.160] | 18.0% |
| **test_blind 54** | **+0.170** | **[+0.058, +0.288]** | 25.4% |

| test 나눠 보기 | G6 | Vis |
|---|---|---|
| test 122 | 0.809 (E1 0.762) | 0.714 (E1 0.517) |
| test_blind 54 | 0.662 (E1 0.492) | 0.730 (E1 0.486) |

| 재현율 (test) | E1 | E3-L |
|---|---|---|
| function | 0.53 | 0.78 |
| non_visual | 0.14 | 0.48 |
| not_visible | 0.125 | 0.734 |
| localized | 0.962 | 0.861 |
| manufacturing | 0.82 | 0.70 |

#### Summary

E3-L은 E1의 **판단 기준 문제를 사람 정답에서도 고쳤습니다.**
- Visibility 0.507 → 0.720
- Scope 0.376 → 0.656 (처음으로 R0를 넘음)
- function·non_visual 재현율 상승

G6는 점추정 +0.086이고, 안 본 test_blind에서는 구간이 0 위에 있습니다(+0.170). 대가는 세 가지입니다: E2E +29%, manufacturing 재현율 하락, localized 일부 과교정. 박스는 학습하지 않았으므로 그대로입니다.

## Model Examination [optional]

- **val 보임 정도 혼동**: E1의 가장 큰 실수였던 "정답 not_visible → localized"가 20건에서 2건으로 줄었습니다. 대신 "정답 localized → not_visible"이 0건에서 9건 새로 생겼습니다.
- **test 보임 정도**: E1의 "정답 not_visible → localized/diffuse"가 56건에서 17건으로 줄었습니다.
- **예측 분포 (val)**: function 13 → 25, non_visual 2 → 10건으로 늘고, manufacturing 30 → 16, not_as_described 29 → 21건으로 줄었습니다. E1의 그룹 쏠림이 풀렸습니다.
- **박스 (val)**: 전체 박스 수가 120 → 67개로 줄었습니다. 박스 문제가 고쳐진 19건 중 11건은 not_visible로 바뀌어 박스를 안 그린 경우입니다.
- 학습 기록: `outputs/lora/e3l/train_log.txt`, `steps.jsonl`, 에폭별 `val_head.json`. 평가 원자료: `outputs/eval/e3l_*.json`

## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** NVIDIA GeForce RTX 5060 Ti 16GB (로컬 PC 1대)
- **Hours used:** 약 7.4시간. 내역은 GPU 점검 약 0.3시간, 학습 5.65시간, 에폭별 val 점검 0.5시간, val 전체 약 0.4시간, test 약 0.7시간입니다
- **Cloud Provider:** 없음(로컬)
- **Compute Region:** 대한민국(로컬)
- **Carbon Emitted:** 약 0.6 kgCO2eq. 추정 상한이며, GPU 보드 전력 180 W × 7.4시간 × 약 0.45 kgCO2eq/kWh로 계산했습니다. 학습 변수가 전체의 0.061%뿐이라 GPU 1장으로 끝났습니다.

## Technical Specifications [optional]

### Model Architecture and Objective

- **구조**: Qwen2.5-VL-7B-Instruct(ViT → 2×2 merger → 28층 LLM)는 그대로 얼립니다. LLM의 attention 사영(q·k·v·o)마다 저랭크 행렬 쌍(A: d×8, B: 8×d)을 더합니다. 출력은 `W·x + (alpha/r)·B·A·x`입니다.
- **학습 목적**: 앞머리 토큰에 대한 다음 토큰 교차 엔트로피(가중)
- **추론 목적**: E1과 같이 JSON 전체를 한 번에 생성합니다.

### Compute Infrastructure

로컬 PC 1대, GPU 1장입니다.

#### Hardware

NVIDIA GeForce RTX 5060 Ti (16GB). 학습 최대 VRAM은 13.0GB, 추론 최대 VRAM은 9.1GB입니다.

#### Software

HAWK 런타임(패치된 transformers 4.52.0)을 HAWK를 끈 채로 씁니다.

- PEFT 0.21.0
- torch 2.11.0+cu128
- bitsandbytes 0.50.2
- qwen-vl-utils 0.0.11

설정 전체는 `outputs/lora/e3l/train_config.json`과 `outputs/preds/heavy_lora-e3l-epoch3_test_config.json`에 있습니다.

## Citation [optional]

**BibTeX:**

```bibtex
@article{hu2021lora,
  title   = {LoRA: Low-Rank Adaptation of Large Language Models},
  author  = {Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal = {arXiv preprint arXiv:2106.09685},
  year    = {2021}
}

@article{dettmers2023qlora,
  title   = {QLoRA: Efficient Finetuning of Quantized LLMs},
  author  = {Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal = {arXiv preprint arXiv:2305.14314},
  year    = {2023}
}

@misc{qwen2.5-VL,
    title = {Qwen2.5-VL},
    url = {https://qwenlm.github.io/blog/qwen2.5-vl/},
    author = {Qwen Team},
    month = {January},
    year = {2025}
}
```

**APA:**

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). *LoRA: Low-rank adaptation of large language models*. arXiv:2106.09685.

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient finetuning of quantized LLMs*. arXiv:2305.14314.

## Glossary [optional]

E1 카드의 용어표를 같이 보세요([E1 Glossary](../e1-qwen25vl7b-zeroshot/README.md#glossary-optional)). 이 카드에서 더 쓰는 말은 아래와 같습니다.

| 용어 | 뜻 |
|---|---|
| LoRA | 원래 가중치는 얼리고, 작은 행렬 두 개(A, B)만 학습해 더하는 방법 |
| QLoRA | 기반 모델을 4비트(NF4)로 얼린 채 LoRA를 학습하는 방법. 메모리가 적게 듭니다 |
| r / alpha | LoRA 행렬의 폭(랭크) / 더할 때의 배율(alpha/r = 2) |
| 어댑터 | 학습된 LoRA 가중치 묶음(20MB). 기반 모델에 얹었다 뗐다 할 수 있습니다 |
| 앞머리(head) | 답 JSON의 첫 세 칸 `category`·`part`·`visibility`. 여기에만 loss를 겁니다 |
| 앞머리 점검 | visibility까지만 생성해 세 칸을 빠르게 채점하는 val 검사. 에폭 선택용입니다 |
| 헷갈림(unsure) 행 | Claude가 라벨링 때 그룹을 헷갈려 한 표시가 있는 행. loss 가중치 0.5 |
| 과교정 | 고치려던 실수(not_visible을 보인다고 함)를 줄이다가, 반대 실수(보이는 것을 not_visible이라 함)가 생기는 것 |

## More Information [optional]

- E3 제안(근거·위험·예비안): `reports/E3_실험_제안.html`
- 방법론 지도(A~G): `reports/Qwen2.5-VL_방법론_지도.html`
- 설계 문서의 E3-L 절과 결과: `reports/초기실험_설계_v0.1.md` (v0.8)
- 코드 따라가기 노트: `reports/공부/E3/E3-L_코드로_따라가기.md`
- 같은 묶음의 다른 카드: [E1 (제로샷)](../e1-qwen25vl7b-zeroshot/README.md), [E2 (+HAWK)](../e2-qwen25vl7b-hawk020/README.md)

## Model Card Authors [optional]

6팀 CluVi — GitHub `vmgfh878-art` (초안 작성 보조: Claude)

## Model Card Contact

6팀 CluVi (GitHub org `CluVi-prometheus`)
