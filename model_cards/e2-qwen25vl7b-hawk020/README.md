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
- visual-token-pruning
- hawk
- complaint-analysis
- defect-classification
- qwen2.5-vl
- cluvi
model-index:
- name: cluvi-heavy-e2-qwen25vl7b-hawk020
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
      value: 0.678
      name: Group6 Macro-F1
    - type: f1
      value: 0.676
      name: Group5 Macro-F1
    - type: f1
      value: 0.639
      name: Fine defect Macro-F1
    - type: f1
      value: 0.483
      name: Visibility Macro-F1
    - type: f1
      value: 0.368
      name: Part scope Macro-F1
    - type: f1
      value: 0.372
      name: Part Macro-F1
    - type: mean_iou
      value: 0.142
      name: Box mean IoU
    - type: pointing_game
      value: 0.268
      name: Box Pointing
---

# Model Card for CluVi Heavy E2 — Qwen2.5-VL-7B 제로샷 + HAWK (시각 토큰 20% 유지)

E1과 똑같은 분석기에 **HAWK 원본 코드**만 더한 모델입니다. HAWK는 시각 토큰을 줄이는 방법입니다. LLM 첫 층에서 시각 토큰의 80%를 버리고 20%만 남겨 더 빨리 답합니다. 묻는 것은 하나입니다: **"분류 성능을 지키면서 얼마나 빨라지나?"**

## Model Details

### Model Description

E1([카드](../e1-qwen25vl7b-zeroshot/README.md))과 바뀐 것은 HAWK 하나뿐입니다. 평가 코드 `pair_check`로 두 실행의 설정을 대조해, 다른 값이 없음을 확인했습니다.

| 구성 | E1 | **E2** |
|---|---|---|
| 기반 가중치 · 로딩 | `Qwen/Qwen2.5-VL-7B-Instruct`, LLM NF4 4비트, ViT·`lm_head` bf16 | 같음 |
| 프롬프트 · 파서 · 디코딩 | v0.3.3, `parse_vlm`, greedy · rp 1.05 · 640토큰 | 같음 |
| 시각 토큰 | 전부 사용 | **HAWK로 20%만 남김** (`--hawk-keep 0.20`, 논문 설정의 pruning ratio 0.8) |

**HAWK가 하는 일**(Head Importance-Aware Visual Token Pruning, CVPR 2026):
1. LLM 0번 층에서 글 토큰이 각 시각 토큰에 주는 attention을 봅니다.
2. 이 attention을 head마다 미리 구한 중요도(HAWK 저자가 공개한 head-importance 벡터)로 가중합해 시각 토큰에 점수를 매깁니다.
3. 점수가 높은 20%만 남기고 나머지는 이후 층에서 버립니다.

학습은 없습니다. 우리 데이터로 head 중요도를 다시 구하지도 않았습니다.

- **Developed by:** 6팀 CluVi (프로메테우스). HAWK 방법과 코드는 Zhu et al.(2026)의 것입니다.
- **Funded by [optional]:** [More Information Needed]
- **Shared by [optional]:** 6팀 CluVi (공개 레포 `CluVi-prometheus/vmgfh878-art`. 데이터셋은 비공개)
- **Model type:** VLM 기반 다중모달 결함 분석기 + 학습 없는 시각 토큰 가지치기. image-text-to-text, 제로샷
- **Language(s) (NLP):** 입력은 영어, 출력은 JSON 코드값과 한국어 근거 문장
- **License:** 기반 모델 Apache-2.0, HAWK 코드 Apache-2.0
- **Finetuned from model [optional]:** 없음. `Qwen/Qwen2.5-VL-7B-Instruct`를 그대로 씁니다

### Model Sources [optional]

- **Repository:**
  - 실험 코드 `cluvi/` (`cluvi/run_hawk.py`)
  - HAWK 원본 https://github.com/peppery77/HAWK (로컬 `HAWK/`, 커밋 `3af2071`)
- **Paper [optional]:**
  - HAWK: Head Importance-Aware Visual Token Pruning in Multimodal Models, https://arxiv.org/abs/2604.07812
  - Qwen2.5-VL, https://arxiv.org/abs/2502.13923
- **Demo [optional]:** `python -m cluvi.demo --model heavy --row 3` (HAWK를 켜려면 `run_hawk` 경로에서 `--hawk-keep 0.20`)

## Uses

### Direct Use

- 리뷰 1건을 분석 카드 1장으로 바꾸는 일입니다(E1과 같은 출력).
- E1보다 응답이 18% 빠르고 GPU 메모리를 0.66GB 덜 씁니다.
- **결함 그룹·세부 결함만 필요한 화면**에 적합합니다.

### Downstream Use [optional]

- **상담원 에이전트 `cluvi/agent.py`의 분석 단계 후보**: 응답 시간이 중요한 경우에 씁니다. 단, 박스는 쓰지 않는 조건입니다(아래 한계).
- **E3-G와의 관계**: E3 제안서의 예비안 E3-G는 박스만 사진별로 따로 묻고, 그 호출에서는 HAWK를 끄는 구성입니다. E2에서 박스가 무너진 결과가 근거입니다.

### Out-of-Scope Use

- **박스·위치 정보가 필요한 용도**: test Pointing이 0.268로, 박스 4개 중 3개가 위치를 벗어납니다.
- E1 카드의 금지 용도도 모두 그대로 적용됩니다: 사람 확인 없는 자동 결정, 사진 5장 이상, 영상·비영어·도메인 밖 입력, 개인정보 식별.

## Bias, Risks, and Limitations

- **박스가 무너짐**: E1 대비 test 결과는 아래와 같습니다. 토큰을 고를 때 "글과 관련 있는 부분"을 남기다 보니, 결함의 정확한 경계 정보가 사라지는 것으로 보입니다(원인은 따로 실험하지 않았습니다).

  | 지표 | E1 | E2 | E1 대비 |
  |---|---|---|---|
  | Pointing | 0.500 | 0.268 | 54% |
  | IoU@0.5 | 0.179 | 0.071 | 40% |
  | mIoU | 0.238 | 0.142 | 60% |

- **사진 과신이 조금 심해짐**:
  - val에서 localized 예측이 63건에서 69건으로 늘었습니다.
  - test Visibility F1이 0.507에서 0.483으로 내려갔습니다. E1의 핵심 약점(사진에 안 보이는 것도 보인다고 함)이 그대로이거나 약간 나빠졌습니다.
- **test_blind에서 내림**: G6가 test 122에서는 0.762 → 0.785로 올랐지만, test_blind 54에서는 0.492 → 0.449로 내렸습니다. 54건이라 1건에도 크게 흔들리지만, 일반화 쪽이 나빠졌을 가능성을 배제할 수 없습니다.
- **잘린 출력**: test 2건이 640토큰에서 잘려 JSON이 깨졌습니다(E1은 0건). 정상 출력은 97.8%입니다.
- **val로는 비열등을 못 보임**: val G6 차이는 −0.040 [−0.122, +0.000]입니다. 100건이라 구간이 넓어 기준(하한 ≥ −0.05)을 통과하지 못했습니다.
- **head 중요도는 일반 벤치마크 기준**: HAWK가 공개한 head-importance 벡터를 그대로 썼습니다. 결함 사진에 맞는 head 순위인지는 확인하지 않았습니다.
- **E1에서 이어지는 한계**:
  - 부위 `none` 누락, function·non_visual 과소
  - test 122 낙관 가능성
  - val 실버 라벨
  - 확신도 미기록(이 실행도 `conf.group6` = 0)
  - 영어 아마존 리뷰로만 평가

### Recommendations

- E2를 쓴다면 카드에서 **박스를 숨기거나**, 박스만 HAWK 없이 따로 구하세요.
- 속도 이득(−18%)은 시각 토큰이 많은 리뷰일수록 큽니다(p90 E2E −21%). 사진이 많은 민원에서 쓰면 이득이 큽니다.
- 성능은 test 122와 test_blind를 나눠 인용하세요.

## How to Get Started with the Model

Use the code below to get started with the model.

```bash
# HAWK 런타임(패치된 transformers 4.52.0) + HAWK 켬
python -m cluvi.run_hawk --model heavy --split val  --hawk-keep 0.20   # -> outputs/preds/heavy_hawk200_val.jsonl
python -m cluvi.run_hawk --model heavy --split test --hawk-keep 0.20   # -> outputs/preds/heavy_hawk200_test.jsonl
python -m cluvi.evaluate outputs/preds/heavy_test.jsonl outputs/preds/heavy_hawk200_test.jsonl --base heavy_test --format
```

`run_hawk.py`는 HAWK 저장소의 `run.sh`를 파이썬으로 옮긴 것입니다. HAWK 런타임을 앞에 두고 `cluvi.predict`를 실행합니다. 런타임 위치는 `HAWK/.runtime`이고, `hawk-patch-manifest.json`이 없으면 멈춥니다.

## Training Details

### Training Data

**학습은 없습니다.** HAWK의 head 중요도 벡터는 저자가 공개한 값을 그대로 썼습니다. 그 벡터를 구한 데이터는 HAWK 논문을 따릅니다.

### Training Procedure

#### Preprocessing [optional]

E1과 같습니다: EXIF 회전, 사진 최대 4장, 사진당 256~1,280 토큰, "Picture N:" 번호, 프롬프트 v0.3.3. 자세한 내용은 [E1 카드](../e1-qwen25vl7b-zeroshot/README.md#preprocessing-optional)를 보세요.

여기에 HAWK가 LLM 0번 층에서 시각 토큰을 20%로 줄입니다. 185건 모두에 적용됐고, 실제 유지 비율은 0.200~0.201입니다.

#### Training Hyperparameters

- **Training regime:** 해당 없음(학습 안 함). 추론은 E1과 같습니다(NF4 4비트 LLM, bf16 ViT, greedy, rp 1.05, 640토큰, batch 1). HAWK 유지 비율은 0.20입니다.

#### Speeds, Sizes, Times [optional]

리뷰당 값입니다. 중앙값이고, 괄호 안은 p90입니다.

| | 시각 토큰(평균) | 출력 토큰 | Prefill | 그중 ViT | 디코딩 | E2E | 최대 VRAM |
|---|---|---|---|---|---|---|---|
| E1 test | 2,082 | 222 (372) | 2.05 s (6.78) | 0.88 s | 7.14 s (12.32) | 10.57 s (18.70) | 9.08 GB |
| **E2 test** | **417 (20.0%)** | 220 (326) | **1.69 s (5.57), −18%** | 0.89 s | 6.33 s (10.01), −11% | **8.69 s (14.69), −18%** | **8.41 GB, −0.66** |
| E2 val | 431 (20.0%) | 223 (331) | 1.61 s (5.66), −19% | 0.87 s | 6.33 s (9.87), −13% | 8.77 s (16.19), −17% | 8.43 GB |

- **ViT 시간은 그대로**입니다. HAWK는 ViT 뒤, LLM 안에서 토큰을 버리기 때문입니다.
- **디코딩도 빨라졌습니다.** 출력 길이는 같지만 문맥(시각 토큰)이 짧아져 토큰마다 계산이 줄었습니다.
- 실행 시간: val 100은 약 18분, test 185는 약 32분입니다(2026-11-11).

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

E1과 같습니다. [`CluVi-prometheus/cesamard-defect-v3`](https://huggingface.co/datasets/CluVi-prometheus/cesamard-defect-v3)를 씁니다.

- test 185: 채점 G6·Vis 176건, Scope·Part·Fine 122건, 박스 56건
- val 100: 실버

#### Factors

- test 122 vs test_blind 54
- val 클래스별 혼동
- E1과 같은 리뷰끼리 짝 비교

#### Metrics

E1과 같은 13지표입니다([E1 카드](../e1-qwen25vl7b-zeroshot/README.md#metrics)). 여기에 두 가지를 더 봅니다.

- **판정 기준(미리 정함)**: E2 − E1의 G6 짝 차이 95% 구간 하한이 −0.05 이상이면 **비열등**입니다.
- **HAWK 논문식 상대 성능**(E1 = 100%)과 답이 바뀐 비율

### Results

| | Scope | Part | **G6** | G5 | Fine | Vis | mIoU | IoU@.5 | Pointing | 정상 출력 |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 test | 0.376 | 0.340 | 0.675 | 0.682 | 0.630 | 0.507 | 0.238 | 0.179 | 0.500 | 98.4% |
| **E2 test** | 0.368 | 0.372 | **0.678** [0.588, 0.753] | 0.676 | 0.639 | 0.483 | 0.142 | 0.071 | 0.268 | 97.8% |
| E1 val | 0.280 | 0.365 | 0.657 | 0.651 | 0.406 | 0.561 | 0.071 | 0.022 | 0.178 | 100% |
| **E2 val** | 0.436 | 0.455 | **0.617** [0.481, 0.727] | 0.604 | 0.400 | 0.500 | 0.049 | 0.000 | 0.111 | 100% |

| 짝 비교 (E2 − E1, bootstrap 2,000회) | G6 차이 | 95% 구간 | 답 바뀜 | 비열등 |
|---|---|---|---|---|
| test | **+0.004** | [−0.048, +0.046] | 5.4% | **통과** |
| val | −0.040 | [−0.122, +0.000] | 4.0% | 통과 못 함 |

| test 나눠 보기 | G6 | Vis |
|---|---|---|
| test 122 | 0.785 (E1 0.762) | 0.506 (E1 0.517) |
| test_blind 54 | 0.449 (E1 0.492) | 0.430 (E1 0.486) |

| 상대 성능 (test, E1 = 100%) | Scope | Part | G6 | G5 | Fine | Vis | mIoU | IoU@.5 | Pointing |
|---|---|---|---|---|---|---|---|---|---|
| | 98% | 109% | **101%** | 99% | 101% | 95% | **60%** | **40%** | **54%** |

- val Scope 156%(표에는 없음)는 착시입니다. `accessory` 정답이 1건뿐이라, 그 1건을 맞힌 것만으로 클래스 F1이 0에서 1이 되었습니다.

#### Summary

- **분류(G6)는 test에서 E1과 같습니다**(+0.004, 비열등 통과).
- **응답은 18% 빨라지고** VRAM은 0.66GB 줄었습니다.
- 대신 **박스는 Pointing 0.50 → 0.27로 무너지고**, 보임 정도도 조금 나빠졌습니다(0.51 → 0.48).
- 결론: 박스가 필요 없는 화면에서만 쓸 만합니다.

## Model Examination [optional]

- **val group6 혼동표**는 E1과 거의 같습니다. contamination 맞힘이 3건에서 2건으로, function 맞힘이 11건에서 10건으로 줄었습니다. 쏠림은 manufacturing 1.58× → 1.47×, non_visual 0.22×(그대로)입니다.
- **박스 형식 문제**(test, 리뷰 수 기준):
  - 복사: 37 → 30건
  - 80% 넘게 덮음: 19 → 20건
  - 범위 밖: 10 → 11건
  - 따라서 박스 수치 하락은 형식 문제보다 위치 정확도 문제입니다.
- 사례 비교: `reports/Qwen_베이스라인_실험결과.html`의 "HAWK가 바꾼 답" 예시와 `reports/Qwen_베이스라인과_HAWK.html`(HAWK 해설)

## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** NVIDIA GeForce RTX 5060 Ti 16GB (로컬 PC 1대)
- **Hours used:** 약 0.8시간(val + test). HAWK 첫 점검(dev 5건)은 몇 분입니다
- **Cloud Provider:** 없음(로컬)
- **Compute Region:** 대한민국(로컬)
- **Carbon Emitted:** 약 0.07 kgCO2eq. 추정치이며, GPU 보드 전력 180 W × 시간 × 약 0.45 kgCO2eq/kWh로 가정한 상한입니다. 같은 일을 E1보다 18% 짧게 하므로 리뷰당 배출도 그만큼 줄어듭니다.

## Technical Specifications [optional]

### Model Architecture and Objective

Qwen2.5-VL-7B-Instruct 구조(ViT → 2×2 merger → 28층 LLM)는 그대로입니다. HAWK가 LLM 0번 층의 attention으로 시각 토큰 점수를 매겨, 상위 20%만 이후 층으로 넘깁니다. 목적은 E1과 같은 다음 토큰 생성입니다.

### Compute Infrastructure

로컬 PC 1대, GPU 1장, batch 1입니다.

#### Hardware

NVIDIA GeForce RTX 5060 Ti (16GB). 추론 최대 VRAM은 8.4GB입니다.

#### Software

HAWK 런타임(`HAWK/.runtime`, HAWK가 패치한 transformers 4.52.0)을 씁니다. HAWK 커밋은 `3af2071`입니다.

- torch 2.11.0+cu128
- qwen-vl-utils 0.0.11
- bitsandbytes 0.50.2

설정 전체는 `outputs/preds/heavy_hawk200_test_config.json`에 있습니다.

## Citation [optional]

**BibTeX:**

```bibtex
@inproceedings{zhu2026hawk,
  title     = {HAWK: Head Importance-Aware Visual Token Pruning in Multimodal Models},
  author    = {Zhu, Qihui and Zhang, Tao and Wang, Yuchen and Wen, Zijian and Zhang, Mengjie and Chen, Shuangwu and Tan, Xiaobin and Yang, Jian and Liu, Yang and Dong, Zhenhua and Yu, Xianzhi and Pan, Yinfei},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
  note      = {arXiv:2604.07812}
}

@misc{qwen2.5-VL,
    title = {Qwen2.5-VL},
    url = {https://qwenlm.github.io/blog/qwen2.5-vl/},
    author = {Qwen Team},
    month = {January},
    year = {2025}
}
```

(HAWK는 저장소의 `CITATION.cff`, Qwen은 기반 모델 카드의 인용을 옮겼습니다.)

**APA:**

Zhu, Q., Zhang, T., Wang, Y., Wen, Z., Zhang, M., Chen, S., Tan, X., Yang, J., Liu, Y., Dong, Z., Yu, X., & Pan, Y. (2026). HAWK: Head importance-aware visual token pruning in multimodal models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

## Glossary [optional]

E1 카드의 용어표를 같이 보세요([E1 Glossary](../e1-qwen25vl7b-zeroshot/README.md#glossary-optional)). 이 카드에서 더 쓰는 말은 아래와 같습니다.

| 용어 | 뜻 |
|---|---|
| 유지 비율(keep) / pruning ratio | 남기는 시각 토큰 비율 / 버리는 비율. 0.20 유지 = 0.8 가지치기 |
| head 중요도 | attention head마다 "시각 정보를 얼마나 잘 고르는지"를 매긴 값. HAWK가 미리 구해 공개했습니다 |
| 비열등 | 짝 차이 95% 구간 하한이 −0.05 이상. "5%p 넘게 떨어지지 않음" |
| 상대 성능 | E1 점수를 100%로 놓은 비율(HAWK 논문의 보고 방식) |

## More Information [optional]

- 설계와 판정 기준: `reports/초기실험_설계_v0.1.md`
- 실행 단위 수치: `reports/중간실험_결과_기록.md` 4절
- HAWK 해설: `reports/Qwen_베이스라인과_HAWK.html`
- 같은 묶음의 다른 카드: [E1 (제로샷)](../e1-qwen25vl7b-zeroshot/README.md), [E3-L (QLoRA)](../e3l-qwen25vl7b-qlora-head/README.md)

## Model Card Authors [optional]

6팀 CluVi — GitHub `vmgfh878-art` (초안 작성 보조: Claude)

## Model Card Contact

6팀 CluVi (GitHub org `CluVi-prometheus`)
