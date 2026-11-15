# E3-L 코드로 따라가기: QLoRA로 "답의 앞머리"만 가르치기

6팀 CluVi · CESAMARD · 공부 노트 (2026-11-15)

> **GitHub에서 볼 때**: `HAWK/` · `.venv/` 아래를 가리키는 링크는 로컬에만 있는 파일이라 레포에서는 열리지 않습니다. 로컬 파일은 HAWK 클론, HAWK가 패치한 transformers 4.52.0 런타임, 설치된 패키지입니다. HAWK 원본 코드는 https://github.com/peppery77/HAWK/tree/3af2071 에서 볼 수 있습니다. 설치 패키지는 transformers · PEFT 저장소의 같은 버전 태그에서 볼 수 있습니다.

이 노트는 세 부분으로 되어 있습니다.

| 부분 | 내용 | 언제 읽나 |
|---|---|---|
| **A. 코드 읽는 순서 (E0 → E3)** | 프로젝트 코드를 처음부터 어떤 순서로 읽으면 되는지. E1·E2 상세는 [베이스라인 노트](../베이스라인/E1_E2_코드로_따라가기.md)로 보냄 | 맨 처음 |
| **B. E3-L 코드로 따라가기** | LoRA·QLoRA 개념부터 학습 데이터 → 어댑터 → 손실 → 학습 → 에폭 고르기 → 추론 → 판정까지, 우리 코드 줄 단위로 | A를 읽은 뒤 |
| **C. Qwen 실험 전체 제안** | E1·E2·E3-L 결과를 놓고 다음에 무엇을 할지 | 마지막 |

---

## 읽기 전에

### 링크와 줄 번호

- 링크는 이 파일 위치(`reports/공부/E3/`) 기준 상대 경로이고, `#L123`은 123번째 줄입니다. VS Code에서 누르면 그 줄로 갑니다.
- 줄 번호는 **2026-11-15 저녁 파일 기준**입니다. `cluvi/`가 나중에 바뀌면 링크가 어긋날 수 있으니, 그때는 함수 이름으로 찾아 주세요.
- 라이브러리 코드는 **실제로 E3-L을 돌린 파일**을 링크합니다.
  - Qwen 모델: HAWK가 패치한 transformers 4.52.0 사본 [HAWK/.runtime/python/transformers/](../../../HAWK/.runtime/python/transformers/)
  - LoRA: 프로젝트 가상환경의 peft 0.21.0 [.venv/Lib/site-packages/peft/](../../../.venv/Lib/site-packages/peft/)

### 표시

| 표시 | 뜻 |
|---|---|
| **[우리]** | 우리가 쓴 코드 (`cluvi/`, `scripts/`) |
| **[Qwen]** | Qwen 모델 코드 (transformers) |
| **[peft]** | LoRA를 실제로 끼워 넣는 라이브러리 (Hugging Face peft) |

### 한 건을 끝까지 따라갈 예시

> **학습 리뷰 1건**: 사진 2장 + "사이즈가 안 맞아요" 류의 영어 리뷰
>
> - 입력은 E1과 똑같이 만듭니다. 사진 토큰 약 1,250개 + 글 약 1,600토큰 = **약 2,900토큰** (train 중앙값 2,934)
> - 정답은 답의 앞머리 **약 30토큰**뿐입니다: `` ```json\n{\n  "category": "function/fit_size",\n  "part": "whole",\n  "visibility": "diffuse", ``
> - 2,900토큰을 다 읽히고, 마지막 30토큰 자리에서만 채점합니다.

---

# A. 코드 읽는 순서: E0부터 E3까지

## A-0. 전체 지도

네 실험은 **같은 코드 한 줄기**를 씁니다. 실험마다 바뀐 곳은 한두 군데뿐입니다.

```
                      ┌──────────── 모든 실험이 같이 쓰는 줄기 ────────────┐
리뷰 + 사진 ─► load_split ─► build_prompt ─► encode_chat ─► generate ─► parse_vlm ─► evaluate
              common.py:112  common.py:196   models.py:47   models.py:191 common.py:232 evaluate.py:72
                                  ▲                              ▲
E0 (dev)   프롬프트 v0.2 → v0.3.3, 반복 벌점 1.05 vs 1.0 확인        │
E1         위 줄기 그대로 (공식 추론 + 프롬프트 v0.3.3)               │
E2         + HAWK 한 줄 (models.py:143) → LLM 0층에서 사진 토큰 20%만 남김
E3-L       + LoRA 어댑터 (models.py:150) → 학습은 train_lora.py, 판정은 judge_e3.py
```

| 실험 | 바꾼 것 | 읽을 파일 (순서대로) | 자세한 설명 |
|---|---|---|---|
| **E0** 사전 점검 (dev, 정답 안 봄) | 프롬프트 판, 반복 벌점 | 프롬프트 6판 → `predict.py` 옵션 → `parse_vlm`의 문제 기록 → `evaluate --format` | 이 노트 A-1, [중간실험 결과 기록 2절](../../중간실험_결과_기록.md#L60) |
| **E1** Qwen 제로샷 | (기준) | `run_hawk` → `predict` → `QwenVLM` → `parse_vlm` → `evaluate` | [베이스라인 노트 0~4단계](../베이스라인/E1_E2_코드로_따라가기.md#L48) |
| **E2** + HAWK | 사진 토큰 줄이기 | HAWK `pruning.py` 네 함수 | [베이스라인 노트 5단계](../베이스라인/E1_E2_코드로_따라가기.md#L635) |
| **E3-L** + QLoRA | 어댑터 하나 | `lora_data` → `train_lora` → `models`(어댑터 얹기) → `judge_e3` | 이 노트 B |

## A-1. E0: 형식만 고친 단계

E0는 정답을 보지 않고 dev 리뷰로 **"모델이 우리 틀대로 답을 쓰는가"**만 봤습니다. 그래서 읽을 코드는 "무엇을 바꿔 가며 돌렸나"와 "무엇을 셌나" 두 가지입니다.

1. **무엇을 바꿨나: 프롬프트 판** — [cluvi/prompts/](../../../cluvi/prompts/)에 6판이 있습니다.
   - `vlm-v0.2` → `v0.2.1` → `v0.3`(버림) → `v0.3.1` → `v0.3.2` → **`v0.3.3`**(최종, [61줄](../../../cluvi/prompts/vlm-v0.3.3.txt))
   - 두 판을 나란히 놓고 비교해 보면 E0에서 무엇을 고쳤는지 보입니다. 예를 들어 박스 안 `"label"` 키를 `"target"`으로 바꾼 이유는 Qwen이 `label`을 "물체 이름"으로 읽기 때문이었습니다.
   - 최종 판 이름은 코드 기본값 [common.py:184](../../../cluvi/common.py#L184) `PROMPT_VERSION`에 박혀 있습니다.
2. **어떻게 바꿔 가며 돌렸나: `predict.py` 옵션**
   - [predict.py:49-61](../../../cluvi/predict.py#L49): `--split dev --limit 30 --start 30 --prompt vlm-v0.3.1 --repetition-penalty 1.0` 같은 조합.
   - [predict.py:68-74](../../../cluvi/predict.py#L68): 바꾼 설정이 파일 이름에 들어갑니다. 그래서 `outputs/preds/heavy_rp100_vlm-v0.3.1_dev_smoke.jsonl` 같은 파일이 E0 기록입니다.
3. **무엇을 셌나: 형식 문제**
   - [common.py:232](../../../cluvi/common.py#L232) `parse_vlm`이 답을 카드로 바꾸면서 문제를 `issues`에 적습니다. 예: JSON이 안 읽힘 [241줄](../../../cluvi/common.py#L241), 박스 범위 밖 [291줄](../../../cluvi/common.py#L291).
   - [evaluate.py:163](../../../cluvi/evaluate.py#L163) `format_report`: 정답 없이 그 문제들을 세는 함수. E0의 판단 근거가 전부 여기서 나왔습니다.
4. **E0에서 얻은 결론** (설계 문서 6절 E0):
   - 반복 벌점 1.05가 프롬프트의 라벨 코드까지 벌줘서 형식이 깨지는 경우가 있었습니다.
   - 그렇다고 1.0으로 바꾸면 끝없이 반복하는 경우가 생겨서, 공식 값 1.05를 유지했습니다.
   - 박스·근거 문제는 프롬프트로 안 고쳐져서 E3 후보로 넘겼습니다.

## A-2. 읽는 순서 체크리스트 (전체)

하루에 다 읽을 필요는 없습니다. 굵은 줄만 읽어도 흐름은 잡힙니다.

- [ ] **[vlm-v0.3.3.txt](../../../cluvi/prompts/vlm-v0.3.3.txt) 61줄 전체** — 모델이 받는 글. 모든 실험이 이걸 씀
- [ ] [vlm-v0.2.txt](../../../cluvi/prompts/vlm-v0.2.txt)와 v0.3.3을 나란히 비교 — E0에서 바뀐 곳
- [ ] **[run_hawk.py:22](../../../cluvi/run_hawk.py#L22) `use_runtime` → [predict.py:46](../../../cluvi/predict.py#L46) `main`** — 실행 시작점
- [ ] **[models.py:238](../../../cluvi/models.py#L238) `analyze` → [163](../../../cluvi/models.py#L163) `chat` → [47](../../../cluvi/models.py#L47) `encode_chat`** — 리뷰 한 건 처리
- [ ] [common.py:232](../../../cluvi/common.py#L232) `parse_vlm` — 답 → 카드
- [ ] [evaluate.py:72](../../../cluvi/evaluate.py#L72) `score`, [108](../../../cluvi/evaluate.py#L108) `paired_diff` — 카드 → 지표, 같은 리뷰끼리 비교
- [ ] (E2) [베이스라인 노트 5단계](../베이스라인/E1_E2_코드로_따라가기.md#L635)
- [ ] **(E3) [lora_data.py:46](../../../cluvi/lora_data.py#L46) `head_tokens`** — 정답 토큰 만들기
- [ ] **(E3) [train_lora.py:49](../../../cluvi/train_lora.py#L49) `attach_lora` → [71](../../../cluvi/train_lora.py#L71) `loss_on` → [262](../../../cluvi/train_lora.py#L262) `train`** — 학습 전체
- [ ] (E3) [peft bnb.py:513](../../../.venv/Lib/site-packages/peft/tuners/lora/bnb.py#L513) `Linear4bit.forward` — LoRA가 실제로 더해지는 곳
- [ ] (E3) [models.py:145-152](../../../cluvi/models.py#L145) — 추론에 어댑터 얹기
- [ ] (E3) [judge_e3.py:52](../../../cluvi/judge_e3.py#L52) `judge` — 성공 기준 판정

---

# B. E3-L 코드로 따라가기

## B-0. 한 장 지도

```
                         ┌───────────── 학습 (train 881, 5시간 39분) ─────────────┐
리뷰 1건 ─► encode_example ─────────────► loss_on ──────────► backward ─► 8건마다 AdamW 한 걸음
            lora_data.py:72                train_lora.py:71                train_lora.py:343-346
            = E1과 같은 입력 (encode_chat)   = 28층 LLM (4bit, 얼림)
              + 정답 앞머리 약 30토큰         + LoRA 112개 (학습)
                                           → 앞머리 자리에서만 lm_head → 교차 엔트로피
                                                                          │
                         에폭이 끝날 때마다 ◄───────────────────────────────┘
                         epochN/ 저장 → val 100 앞머리 점검 (48토큰만 생성)   train_lora.py:366-371
                                                                          │
                         미리 정한 규칙으로 에폭 고르기 (→ 3)               train_lora.py:388
                                                                          │
                         ┌───────────── 추론 (E1과 같은 코드) ─────────────┐
리뷰 1건 ─► QwenVLM(adapter=epoch3) ─► generate (greedy, 640토큰) ─► parse_vlm ─► 카드
            models.py:150 (합치지 않고 얹음)                                         │
                                                                        judge_e3 (val 9/9)
                                                                        evaluate (test 1회)
```

| 단계 | 핵심 코드 | 우리 숫자 |
|---|---|---|
| 1. 학습 데이터 | [lora_data.py:72](../../../cluvi/lora_data.py#L72) | train 881 (헷갈림 239건 × 0.5), 입력 중앙 2,934 · 최대 6,935토큰, 정답 28~35토큰 |
| 2. 어댑터 붙이기 | [train_lora.py:49](../../../cluvi/train_lora.py#L49) | 112개 모듈, 5,046,272개 (전체의 0.061%) |
| 3. 손실 | [train_lora.py:71](../../../cluvi/train_lora.py#L71) | 에폭 평균 0.096 → 0.059 → 0.042 |
| 4. 학습 루프 | [train_lora.py:262](../../../cluvi/train_lora.py#L262) | 333번 갱신, 갱신 1번에 40~60초, 최대 VRAM 13.0 GB |
| 5. 에폭 고르기 | [train_lora.py:388](../../../cluvi/train_lora.py#L388) | 평균 0.681 / 0.702 / **0.704** → 에폭 3 |
| 6. 추론에 얹기 | [models.py:150](../../../cluvi/models.py#L150) | 어댑터 파일 20.2 MB, E2E 10.6 → 13.6초 |
| 7. 판정 | [judge_e3.py:52](../../../cluvi/judge_e3.py#L52) | val 9/9, test G6 0.675 → 0.761 |

---

## B-1. 개념: LoRA와 QLoRA

### 왜 전체 학습이 아니라 LoRA인가

알고 계신 "전체 학습"은 모든 가중치를 다 바꿉니다. 7B 모델에 그걸 하려면 이만큼 듭니다.

| 방식 | 가중치 | 기울기 + Adam 상태 | 합 (활성값 빼고) | 16 GB GPU |
|---|---|---|---|---|
| 전체 학습 (bf16 + fp32 Adam) | 82.9억 × 2바이트 = 16.6 GB | 82.9억 × 약 14바이트 ≈ 116 GB | **약 133 GB** | ✗ |
| LoRA (가중치 bf16) | 16.6 GB | 어댑터만 약 60 MB | 16.7 GB | ✗ (가중치만으로 넘침) |
| **QLoRA (LLM 4비트)** | **6.35 GB** (실측) | 어댑터만 약 60 MB | 6.4 GB | **✓** (남은 9 GB를 활성값에) |

- **LoRA**: 원래 가중치 W는 얼려 두고, 옆에 작은 행렬 두 개(A, B)를 붙여 그 둘만 학습합니다.
- **QLoRA**: 그 얼린 W를 4비트(NF4)로 줄여서 GPU에 올립니다. 얼린 가중치라서 거칠게 저장해도 학습에는 지장이 작습니다. 학습되는 A, B는 fp32로 정밀하게 둡니다.

### 그림: q_proj 하나

```
x (3584) ─┬─► W  (4bit, 얼림, 3584×3584 = 1,284만)  ──────────────────┐
          │                                                          (+) ─► 출력 (3584)
          └─► dropout 0.1 ─► A (3584→8, 학습) ─► B (8→3584, 학습) ─► ×2 ─┘
                              2.9만 개              2.9만 개          ↑
                                                              scaling = alpha / r = 16 / 8
```

- **r = 8**: A가 3584차원을 8차원으로 줄이고, B가 다시 늘립니다. "바꿀 수 있는 방향이 8개뿐"이라는 제약이라서 881건으로도 과적합이 덜합니다.
- **B는 0에서 시작합니다.** 그래서 학습 첫 순간의 모델은 E1과 정확히 같습니다. A는 무작위로 시작합니다.
- **scaling 2**: B·A 결과에 alpha/r = 2를 곱해서 더합니다.

이 구조가 실제로 코드로 있는 곳:

| 무엇 | 코드 |
|---|---|
| A, B 만들기 | [peft layer.py:263-264](../../../.venv/Lib/site-packages/peft/tuners/lora/layer.py#L263) `nn.Linear(in, r)`, `nn.Linear(r, out)` |
| scaling = alpha / r | [peft layer.py:281](../../../.venv/Lib/site-packages/peft/tuners/lora/layer.py#L281) |
| A 무작위, B는 0 | [peft layer.py:338](../../../.venv/Lib/site-packages/peft/tuners/lora/layer.py#L338) kaiming, [343](../../../.venv/Lib/site-packages/peft/tuners/lora/layer.py#L343) zeros |
| **forward: W 결과 + B(A(dropout(x))) × 2** | [peft bnb.py:527](../../../.venv/Lib/site-packages/peft/tuners/lora/bnb.py#L527) `result = self.base_layer(x)`, [549](../../../.venv/Lib/site-packages/peft/tuners/lora/bnb.py#L549) `output = lora_B(lora_A(dropout(x))) * scaling`, [552](../../../.venv/Lib/site-packages/peft/tuners/lora/bnb.py#L552) `result = result + output` |

`bnb.py`의 `Linear4bit`([313줄](../../../.venv/Lib/site-packages/peft/tuners/lora/bnb.py#L313))는 "4비트 W + LoRA"를 한 덩어리로 묶은 층입니다. peft가 원래 `Linear4bit` 자리를 이것으로 바꿔 끼웁니다.

### 숫자: 왜 5,046,272개인가

Qwen2.5-VL-7B의 LLM 한 층 어텐션은 q 28헤드, k·v 4헤드(GQA), 헤드당 128차원입니다 ([modeling:715-718](../../../HAWK/.runtime/python/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L715)). 그래서 k·v의 출력은 3584가 아니라 512(4 × 128)입니다.

| 모듈 | W 모양 | A | B | LoRA 개수 = 8 × (입력 + 출력) |
|---|---|---|---|---|
| q_proj | 3584 → 3584 | 3584×8 | 8×3584 | 57,344 |
| k_proj | 3584 → 512 | 3584×8 | 8×512 | 32,768 |
| v_proj | 3584 → 512 | 3584×8 | 8×512 | 32,768 |
| o_proj | 3584 → 3584 | 3584×8 | 8×3584 | 57,344 |
| 한 층 합 | | | | 180,224 |
| **× 28층** | | | | **5,046,272** |

fp32(4바이트)로 저장하니 어댑터 파일이 5,046,272 × 4 = **20.2 MB**입니다. 실제 파일 `outputs/lora/e3l/epoch3/adapter_model.safetensors`도 20,218,120바이트입니다.

---

## B-2. 학습 데이터: 정답을 "답의 앞머리"로 만들기 [우리]

파일: [cluvi/lora_data.py](../../../cluvi/lora_data.py)

### 왜 앞머리만인가

E1의 답은 항상 이렇게 시작합니다.

```
```json
{
  "category": "function/fit_size",     ← 우리가 가르칠 것 (그룹/결함)
  "part": "whole",                     ← 가르칠 것 (부위)
  "visibility": "diffuse",             ← 가르칠 것 (사진에 보이는 정도)
  "boxes": [ ... ],                    ← 손대지 않음 (모델이 원래대로 씀)
  "evidence": { ... }                  ← 손대지 않음
```

- train 881에는 박스·근거 문장의 정답이 없거나 품질이 확인되지 않았습니다. 그래서 **라벨 세 칸만** 가르칩니다.
- 라벨이 답의 맨 앞에 있어서 가능한 방법입니다. 앞머리를 우리 기준으로 고르면, 뒤는 모델이 그 앞머리를 보고 이어서 씁니다.
- **종료 토큰은 정답에 넣지 않습니다.** 넣으면 모델이 "visibility 다음엔 끝"이라고 배워서 박스·근거를 안 쓰게 됩니다.

### 코드

| 줄 | 하는 일 |
|---|---|
| [lora_data.py:30](../../../cluvi/lora_data.py#L30) | `UNSURE_WEIGHT = 0.5`: Claude가 헷갈린다고 표시한 행은 손실을 절반만 |
| [lora_data.py:32](../../../cluvi/lora_data.py#L32) | `TAIL = ',\n  "boxes": ['`: 정답 뒤에 실제로 오는 글. 마지막 토큰을 자연스럽게 자르는 데만 씀 |
| [lora_data.py:35-39](../../../cluvi/lora_data.py#L35) `head_values` | silver 라벨 → 세 칸 값. 허용 부위에 없는 1건(row 2654)은 프롬프트 규칙대로 `other`로 ([38줄](../../../cluvi/lora_data.py#L38)) |
| [lora_data.py:42-43](../../../cluvi/lora_data.py#L42) `head_text` | 세 칸을 E1 답과 똑같은 모양의 글로 |
| [lora_data.py:46-64](../../../cluvi/lora_data.py#L46) `head_tokens` | 글 → 토큰. 토큰마다 "어느 칸의 값인지(0~3)" 표시도 같이 |
| [lora_data.py:72-85](../../../cluvi/lora_data.py#L72) `encode_example` | **E1 추론과 같은 함수**로 입력을 만들고([77-78줄](../../../cluvi/lora_data.py#L77)), 뒤에 정답 토큰을 붙임([81-83줄](../../../cluvi/lora_data.py#L81)) |

### 꼭 짚을 곳 1: 마지막 토큰을 왜 이렇게 자르나

[52줄](../../../cluvi/lora_data.py#L52)에서 앞머리만이 아니라 `앞머리 + TAIL`을 같이 토큰으로 자릅니다. 그리고 [60-61줄](../../../cluvi/lora_data.py#L60)에서 앞머리가 끝나는 곳까지만 남깁니다.

```
앞머리만 자르면:      ... "diffuse  "        ← 따옴표가 혼자 토큰이 됨 (모델은 이렇게 안 씀)
앞머리+TAIL로 자르면: ... "diffuse  ",\n      ← E1이 실제로 쓰는 토큰
```

토크나이저는 붙어 있는 글자를 한 토큰으로 묶습니다. 그래서 앞머리만 따로 자르면 모델이 실제로 생성하지 않는 토큰이 정답이 됩니다.
처음엔 TAIL에 쉼표가 빠져 있어서 E1 답 20개와 0개가 일치했고, 고친 뒤 20/20이 일치했습니다. 이 확인은 [_check](../../../cluvi/lora_data.py#L115)의 [1]번([130줄](../../../cluvi/lora_data.py#L130))입니다.

### 꼭 짚을 곳 2: 학습 입력 = 추론 입력

[encode_example](../../../cluvi/lora_data.py#L72)은 `review_prompt`·`encode_chat`([models.py:47](../../../cluvi/models.py#L47), [62](../../../cluvi/models.py#L62))를 그대로 부릅니다. E3 준비 때 이 두 함수를 `QwenVLM` 안에서 밖으로 떼어 낸 이유가 이것입니다. 학습과 추론이 한 글자라도 다른 입력을 보면, 학습한 것이 추론에서 안 먹힙니다.
`_check`의 [2]번([148줄](../../../cluvi/lora_data.py#L148))은 "프롬프트와 정답을 따로 토큰으로 잘라 붙인 것 = 한 번에 자른 것"을 확인합니다. 이음매에서 토큰이 합쳐지지 않는다는 뜻입니다.

---

## B-3. 어댑터 붙이기: `attach_lora` [우리 → peft]

파일: [train_lora.py:49-68](../../../cluvi/train_lora.py#L49)

```
QwenVLM("heavy")  ← E1과 똑같이 로드 (NF4 LLM, bf16 ViT)                 train_lora.py:423
   │
get_peft_model(model, LoraConfig(r=8, alpha=16, dropout=0.1, target=TARGET))  train_lora.py:54-55
   │  TARGET = 정규식 "model.language_model.layers.<숫자>.self_attn.(q|k|v|o)_proj"   train_lora.py:36
   │  → 이름이 맞는 Linear4bit 112개를 LoRA 층으로 바꿔 끼움, 나머지는 전부 얼림
   ▼
확인: 112개인지, 전부 LLM 안인지, 학습 변수가 lora_ 뿐인지                   train_lora.py:58-61
   ▼
gradient checkpointing 켜기 (LLM만)                                       train_lora.py:62-63
```

| 줄 | 왜 |
|---|---|
| [36](../../../cluvi/train_lora.py#L36) `TARGET` | ViT의 어텐션 이름은 `attn.qkv`·`attn.proj`라서 이 정규식에 안 걸립니다. 비전 쪽은 안 건드린다는 결정(제안서 D축)을 이름으로 지킵니다 |
| [53](../../../cluvi/train_lora.py#L53) `manual_seed` | A의 무작위 시작값을 고정. 다시 돌려도 같은 결과 |
| [58-61](../../../cluvi/train_lora.py#L58) | 정규식을 잘못 쓰면 엉뚱한 곳에 붙거나 0개에 붙습니다. 숫자로 바로 멈추게 함 |
| [62](../../../cluvi/train_lora.py#L62) | **gradient checkpointing**: 아래 설명 |
| [65](../../../cluvi/train_lora.py#L65) | 4비트 가중치는 1바이트에 2개씩 들어 있어서 `numel()`이 실제의 절반으로 나옵니다. 비율(0.061%)을 바르게 세려고 2배 |

### gradient checkpointing과 제가 낸 버그

역전파를 하려면 각 층의 중간값(활성값)을 다 기억해야 합니다. 입력이 3,000~7,000토큰이고 28층이니 이게 VRAM을 가장 많이 먹습니다.
gradient checkpointing은 **층 경계 값만 남기고, 역전파 때 그 층을 다시 계산**하는 방법입니다. 계산은 약 30% 늘고, 메모리는 크게 줍니다.

그런데 Qwen 코드는 이 기능을 **학습 모드일 때만** 씁니다.

- [modeling:1178](../../../HAWK/.runtime/python/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1178) `if self.gradient_checkpointing and self.training:`

`QwenVLM`은 추론용이라 모델을 평가 모드(`.eval()`, [models.py:128](../../../cluvi/models.py#L128))로 로드합니다. 첫 점검 때 이걸 학습 모드로 안 바꿔서 checkpointing이 꺼져 있었고, 3,500토큰짜리도 메모리가 넘쳤습니다(14.75 GB).
[train_lora.py:198-200](../../../cluvi/train_lora.py#L198)에 `peft_model.train()` 한 줄을 넣어 고쳤고, 그 뒤로 가장 긴 6,962토큰도 12.2 GB에 들어갔습니다. 학습 모드는 LoRA dropout을 켜는 스위치이기도 합니다.

---

## B-4. 손실: `loss_on` [우리]

파일: [train_lora.py:71-82](../../../cluvi/train_lora.py#L71)

```
input_ids = [프롬프트 P개 ............................ | 정답 T개 (약 30)]
                                                    ↑ P-1 자리 출력이 정답 1번째 토큰을 맞혀야 함
model.model(...)  → 은닉상태 (P+T) × 3584                          train_lora.py:77
  └ 그중 [P-1, P-1+T) 구간만 꺼냄 (T × 3584)                         train_lora.py:80
lm_head  → T × 152,064 (어휘 확률)                                  train_lora.py:81
cross_entropy(정답 T개)  → 평균 손실 1개                             train_lora.py:82
```

- **한 칸 밀림**: 언어 모델은 t번째 자리에서 t+1번째 토큰을 맞힙니다. 그래서 `P-1`부터 꺼냅니다([80줄](../../../cluvi/train_lora.py#L80)).
- **왜 `model.model`을 직접 부르나**: 보통은 `model(...)`을 부르는데, 그러면 [modeling:1961](../../../HAWK/.runtime/python/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1961)에서 **모든 자리**의 로짓을 만듭니다.
  - 6,962토큰 × 152,064어휘 × 4바이트 ≈ **4.2 GB**가 되고, 그중 쓰는 건 30자리뿐입니다.
  - 그래서 몸통(`model.model`)만 부르고 정답 자리에서만 `lm_head`를 씁니다.
- **가중치와 누적** ([334줄](../../../cluvi/train_lora.py#L334)): `loss × weight / len(group)`
  - weight: 헷갈림 행 0.5, 나머지 1.0
  - `/ len(group)`: 8건의 기울기를 더해 평균을 내려고. 배치 8을 한 번에 못 올려서 1건씩 8번 역전파하고 기울기를 쌓습니다(gradient accumulation).
- **진행 줄의 "head acc"** ([85-87줄](../../../cluvi/train_lora.py#L85) `field_hits`): 칸의 토큰이 전부 맞았으면 1. 정답 토큰을 넣어 주면서(teacher forcing) 재므로, 실제 생성보다 약간 후합니다.

---

## B-5. 학습 루프: `train` [우리]

파일: [train_lora.py:262-385](../../../cluvi/train_lora.py#L262)

```
에폭마다:
  881건 섞기 (seed + epoch → 다시 시작해도 같은 순서)                    323
  8건씩 묶기 → 111묶음
  묶음마다:
    8건: encode_example (다음 건은 CPU 스레드가 미리)  → loss_on → backward   90, 331-334
    기울기 자르기 (norm 1.0) → AdamW 한 걸음 → 학습률 갱신                343-346
    진행 줄 + steps.jsonl                                                 353-362
    20걸음마다 last/ 저장 (끊겨도 이어서)                                 363-364
  에폭 끝: epochN/ 저장 → val 앞머리 점검 → last/ 저장                    366-381
```

| 설정 | 값 | 코드 | 왜 |
|---|---|---|---|
| 옵티마이저 | AdamW, lr 1e-4, weight decay 0 | [278](../../../cluvi/train_lora.py#L278) | LoRA의 흔한 출발값 |
| 학습률 일정 | cosine, warmup 10% (33걸음) | [279](../../../cluvi/train_lora.py#L279) | 처음엔 천천히 올리고, 끝으로 갈수록 줄임 |
| 걸음 수 | 111 × 3 = 333 | [275-276](../../../cluvi/train_lora.py#L275) | |
| 기울기 자르기 | 1.0 | [343](../../../cluvi/train_lora.py#L343) | 가끔 튀는 기울기 방지 |
| 이어하기 | 설정이 같아야만 | [296-303](../../../cluvi/train_lora.py#L296) | 다른 설정으로 이어 붙이면 결과를 믿을 수 없음 |
| 저장 | 임시 폴더에 쓰고 이름 바꾸기 | [161-171](../../../cluvi/train_lora.py#L161) | 저장 중 끊겨도 직전 저장본이 안 깨짐 |

**실제 진행** (`outputs/lora/e3l/train_log.txt`, `steps.jsonl`)

| 에폭 | 평균 손실 | 학습 중 정답률 (category / part / visibility) |
|---|---|---|
| 1 | 0.096 | 0.67 / 0.71 / 0.71 |
| 2 | 0.059 | 0.76 / 0.84 / 0.81 |
| 3 | 0.042 | 0.83 / 0.89 / 0.86 |

세 번째로 보는 데이터에서도 정답률이 1.0이 아닙니다. 881건을 외우지는 않았다는 뜻입니다.

---

## B-6. 에폭 고르기: val 앞머리 점검 [우리]

| 줄 | 하는 일 |
|---|---|
| [143-157](../../../cluvi/train_lora.py#L143) `val_head` | val 100건을 **48토큰까지만** 생성 ([150줄](../../../cluvi/train_lora.py#L150)). 평가 모드로 바꿨다가([145](../../../cluvi/train_lora.py#L145)) 다시 학습 모드로([154](../../../cluvi/train_lora.py#L154)) |
| [105-109](../../../cluvi/train_lora.py#L105) `head_card` | 48토큰 안의 세 칸을 정규식으로 읽어 카드로 |
| [112-131](../../../cluvi/train_lora.py#L112) `head_metrics` | G6·Vis·Scope F1, 재현율, 확실/헷갈림 따로. `select` = 세 F1의 평균 ([128줄](../../../cluvi/train_lora.py#L128)) |
| [388-394](../../../cluvi/train_lora.py#L388) `pick_epoch` | **평균이 가장 높은 에폭, 같으면 앞 에폭** ([394줄](../../../cluvi/train_lora.py#L394)) |

- **왜 48토큰만**: greedy 생성은 같은 입력이면 항상 같은 토큰을 고릅니다. 그래서 앞 48토큰은 640토큰 전체 실행의 앞 48토큰과 똑같습니다(385건 확인, 불일치 0). 시간은 리뷰당 약 6초로, 전체 실행(약 14초)의 절반 이하입니다.
- **왜 규칙을 학습 전에 정했나**: 결과를 본 뒤 고르는 기준을 정하면, 좋아 보이는 쪽으로 기준을 맞추게 됩니다. 설계 문서 v0.6에 먼저 적어 두었습니다.

| 에폭 | G6 | Vis | Scope | 평균 |
|---|---|---|---|---|
| E1 (참고) | 0.657 | 0.561 | 0.280 | 0.499 |
| 1 | 0.819 | 0.745 | 0.478 | 0.681 |
| 2 | 0.822 | 0.730 | 0.552 | 0.702 |
| **3** | 0.830 | 0.740 | 0.543 | **0.704** |

오른 폭의 대부분이 에폭 1에서 이미 나왔습니다. 새 능력을 익혔다기보다 **판단 기준을 옮긴** 것에 가깝습니다.

---

## B-7. 추론에 얹기: 합치지 않고 얹기 [우리 → peft]

| 줄 | 하는 일 |
|---|---|
| [predict.py:58](../../../cluvi/predict.py#L58) | `--adapter outputs/lora/e3l/epoch3` |
| [predict.py:68-70](../../../cluvi/predict.py#L68) | 파일 이름에 `_lora-e3l-epoch3`가 붙음 → `heavy_lora-e3l-epoch3_test.jsonl` |
| [predict.py:81-82](../../../cluvi/predict.py#L81) | 어댑터를 줄 때만 넘김 → E1·E2 실행은 한 글자도 안 바뀜 |
| [models.py:145-152](../../../cluvi/models.py#L145) | **`PeftModel.from_pretrained(...).get_base_model()`** ([150줄](../../../cluvi/models.py#L150)): LoRA 층을 모델 안에 끼워 넣고, 바깥 껍데기는 E1과 같은 모델 객체를 계속 씀 |
| [models.py:140-141](../../../cluvi/models.py#L140) | 어댑터 + HAWK는 아직 안 해 봐서 막아 둠 |
| [models.py:271-273](../../../cluvi/models.py#L271) | 설정 파일에 어댑터 경로·해시(`adapter_sha1`, [68줄](../../../cluvi/models.py#L68))·`adapter_merged: False` 기록 → `evaluate`가 "바뀐 것: adapter" 하나뿐인 깨끗한 짝으로 인정 |

### 왜 합치지 않았나, 그리고 그 대가

```
합치기:  W' = W + 2·B·A   → 층이 원래대로 하나. 속도 = E1
         그런데 W는 4비트 → W'를 다시 4비트로 반올림해야 함 → 학습한 작은 변화가 반올림에 묻힐 수 있음
얹기:    출력 = W(x) + 2·B(A(x))   → 정확. 대신 생성 토큰마다 112곳에서 작은 계산이 더 돎
```

측정한 대가는 이렇습니다.

| | E1 | E3-L | 변화 |
|---|---|---|---|
| 출력 길이 (중앙) | 226토큰 | 213토큰 | 비슷 |
| 토큰당 디코드 (val 중앙) | 32.0 ms | 47.8 ms | **+49%** |
| E2E (test 중앙) | 10.6초 | 13.6초 | +29% |
| Prefill (test 중앙) | 2.05초 | 2.16초 | +5% |
| VRAM | 9.08 GB | 9.09 GB | 같음 |

계산량은 W의 1%도 안 되는데 +49%나 느린 이유는 제 추정입니다.
- 토큰 하나 생성할 때 GPU는 계산보다 **작은 연산을 여러 번 띄우는 비용**에 묶입니다.
- 얹은 LoRA는 모듈마다 dropout·A·B·더하기에, fp32 ↔ bf16 변환([bnb.py:543-546](../../../.venv/Lib/site-packages/peft/tuners/lora/bnb.py#L543))까지 붙습니다.
- 그래서 토큰마다 112곳 × 몇 번씩 작은 연산이 늘어납니다.
- 이걸 줄이는 방법은 C부의 P2에 적었습니다.

---

## B-8. 판정: `judge_e3`와 체인 스크립트 [우리]

| 줄 | 하는 일 |
|---|---|
| [judge_e3.py:24](../../../cluvi/judge_e3.py#L24) `BOX_ISSUES` | "박스 문제"로 셀 6가지 |
| [judge_e3.py:33](../../../cluvi/judge_e3.py#L33) `measures` | F1, 재현율, 원문 일치율, 박스 문제 수, E2E 중앙 |
| [judge_e3.py:52](../../../cluvi/judge_e3.py#L52) `judge` | 제안서의 성공 기준 9개 |
| [judge_e3.py:97-98](../../../cluvi/judge_e3.py#L97) | **"Claude 버릇" 의심 규칙**: 헷갈림 행에서만 +0.02 넘게 오르고 확실한 행은 그대로면 의심 |
| [run_e3l_chain.py](../../../scripts/run_e3l_chain.py) | 점검([76](../../../scripts/run_e3l_chain.py#L76)) → 어댑터 로딩 확인([89](../../../scripts/run_e3l_chain.py#L89)) → 학습([103-109](../../../scripts/run_e3l_chain.py#L103)) → 에폭 고르기([113](../../../scripts/run_e3l_chain.py#L113)) → val 전체([121](../../../scripts/run_e3l_chain.py#L121)) → 판정([126](../../../scripts/run_e3l_chain.py#L126)). test는 일부러 안 넣음 |

---

## B-9. 결과를 코드로 풀기

| 결과 | val (silver) | test (사람) | 코드로 보면 |
|---|---|---|---|
| G6 Macro-F1 | 0.657 → 0.830 | 0.675 → **0.761** | 앞머리 학습이 그룹 기준을 옮김 |
| 같은 리뷰끼리 차이 | +0.173 [+0.063, +0.297] | +0.086 [−0.000, +0.164] | test에서 절반 → val 상승의 일부는 Claude 라벨 따라 하기 |
| test_blind 54 | – | 0.492 → 0.662, +0.170 [+0.058, +0.288] | 안 본 세트에서 확실히 오름 |
| function / non_visual 재현율 | 0.50 → 0.95 / 0.22 → 0.78 | 0.53 → 0.78 / 0.14 → 0.48 | E1이 거의 안 쓰던 그룹을 쓰게 됨 |
| Visibility F1 | 0.561 → 0.740 | 0.507 → **0.720** | 사람 정답에서도 그대로 오름 → 진짜 개선 |
| localized 재현율 | 0.867 → 0.711 | 0.962 → 0.861 | 반대 방향으로 조금 넘어감 (과교정) |
| manufacturing 재현율 | – | 0.82 → 0.70 | 대가 |
| Scope F1 | 0.280 → 0.543 | 0.376 → **0.656** | 글만 쓴 R0(0.594)를 처음 넘음 |
| Pointing (박스) | 0.178 → 0.178 | 0.500 → 0.518 | **안 가르쳤으니 그대로** |
| 박스 문제 리뷰 | 33 → 20 | 57 → 28 | 박스를 덜 그려서 (120개 → 67개, val) |
| E2E | 10.6 → 13.9초 | 10.6 → 13.6초 | 합치지 않은 어댑터 |

---

## B-10. 스스로 확인 문제

<details><summary>Q1. 정답에 종료 토큰을 넣으면 무슨 일이 생기나요?</summary>

모델이 "visibility 값 다음엔 답이 끝난다"고 배웁니다. 그러면 박스와 근거를 쓰지 않게 되어 상담원 카드가 비게 됩니다. 그래서 [lora_data.py:60-61](../../../cluvi/lora_data.py#L60)에서 visibility 값의 닫는 따옴표가 들어 있는 토큰에서 끊고, 그 뒤는 손실을 주지 않습니다.
</details>

<details><summary>Q2. k_proj에 붙은 B의 모양은? 왜 q_proj와 다르나요?</summary>

8 × 512입니다. k·v는 헤드가 4개(GQA)라 출력이 4 × 128 = 512차원이기 때문입니다 ([modeling:716](../../../HAWK/.runtime/python/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L716)). q는 28헤드라 3584입니다.
</details>

<details><summary>Q3. 학습 첫 걸음 직전의 E3-L 모델은 E1과 같은가요?</summary>

같습니다(dropout을 끈 평가 모드 기준). B가 0으로 시작해서([peft layer.py:343](../../../.venv/Lib/site-packages/peft/tuners/lora/layer.py#L343)) B·A = 0이니 `result + 0`입니다. 그래서 학습은 "E1에서 출발해 조금씩 옮기기"입니다.
</details>

<details><summary>Q4. 왜 `model(...)` 대신 `model.model(...)`을 부르나요?</summary>

`model(...)`은 [modeling:1961](../../../HAWK/.runtime/python/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1961)에서 모든 자리의 로짓을 만듭니다. 가장 긴 입력이면 6,962 × 152,064 × 4바이트 ≈ 4.2 GB입니다. 필요한 건 정답 30자리뿐이라 몸통만 부르고 그 자리에서만 `lm_head`를 씁니다([train_lora.py:80-81](../../../cluvi/train_lora.py#L80)).
</details>

<details><summary>Q5. 첫 점검이 왜 3,500토큰에서도 메모리가 넘쳤나요?</summary>

gradient checkpointing은 `self.training`일 때만 켜집니다([modeling:1178](../../../HAWK/.runtime/python/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1178)). 모델이 평가 모드로 로드된 상태라 꺼져 있었고, 28층의 활성값을 다 들고 있었습니다. [train_lora.py:200](../../../cluvi/train_lora.py#L200) `peft_model.train()`으로 고쳤습니다.
</details>

<details><summary>Q6. 손실을 8로 나누는 이유는? 안 나누면?</summary>

8건의 기울기를 더해서 한 걸음을 가므로, 나누지 않으면 기울기가 8배 커집니다. 그러면 학습률을 8배 올린 것과 비슷해지고, 기울기 자르기(1.0)에 자주 걸려 학습이 달라집니다. 나눠 두면 "배치 8의 평균 기울기"와 같아집니다([train_lora.py:334](../../../cluvi/train_lora.py#L334)).
</details>

<details><summary>Q7. val 상승(+0.17)이 test 상승(+0.09)의 두 배인 이유는?</summary>

train과 val은 둘 다 Claude가 붙인 silver 라벨입니다. 모델이 "Claude가 고르는 방식"을 배우면 val에서는 다 점수가 되지만, 사람 정답과 Claude는 대분류에서 약 73%만 일치하므로 test에서는 일부만 점수가 됩니다. 다만 visibility는 test에서도 그대로 올라서, 그 부분은 Claude 버릇이 아니라 진짜 개선으로 봅니다.
</details>

<details><summary>Q8. 박스 점수가 안 오른 건 실패인가요?</summary>

설계대로입니다. 손실은 앞머리 세 칸에만 걸려 있고 박스 토큰에는 기울기가 가지 않습니다. 박스 문제 리뷰 수가 줄어든 건, not_visible로 바꿔 답하면서 박스를 아예 안 그린 경우가 많아서입니다.
</details>

<details><summary>Q9. 어댑터를 W에 합치면 무엇이 좋아지고, 무엇이 위험한가요?</summary>

좋아지는 것: 생성 토큰마다 붙던 LoRA 계산이 사라져 E1 속도로 돌아갑니다(E2E 약 −3초 예상). 위험: W가 4비트라서 W + 2·B·A를 다시 4비트로 반올림해야 하고, 작은 변화가 반올림에 묻힐 수 있습니다. 그래서 합친 뒤에는 val에서 라벨이 얼마나 같은지 다시 재야 합니다(C부 P2).
</details>

<details><summary>Q10. 서비스 서버는 무엇을 받아서 어떻게 띄우나요?</summary>

원래 Qwen 가중치(HF 공식)와 어댑터 폴더(`adapter_model.safetensors` 20.2 MB + `adapter_config.json`)를 받습니다. 서버는 `QwenVLM("heavy", adapter=폴더)` 한 줄로 띄우고([models.py:150](../../../cluvi/models.py#L150)), 이후는 E1과 같은 `analyze` → 카드입니다. 아래 B-11 그림 참고.
</details>

---

## B-11. 서비스로 가는 길 (E3-L)

```
학습 (한 번)                    저장                              서버
train_lora.py ─► epoch3/ ─► HF 비공개 저장소 (어댑터 20.2 MB) ─┐
                                                               ├─► QwenVLM("heavy", adapter=…)   models.py:110, 150
Qwen 공식 가중치 (HF, 그대로) ──────────────────────────────────┘      │
                                                                      ▼
리뷰가 들어오면 ─► analyze ─► 카드 (category · part · visibility · boxes · evidence + 확신도)
                  models.py:238
```

- **E1 대비 서버에 더해지는 것**: 어댑터 파일 하나와 `adapter=` 인자 하나.
- **아직 안 한 것**: HF 비공개 저장소 업로드(올리기 전에 사용자 확인), 합치기로 속도 되찾기(C부 P2).

---

# C. Qwen 실험 전체 제안 (E3-L 다음)

> 이 부분은 **제안**입니다. 정해진 계획은 설계 문서에 있고, 여기서 고르신 것만 설계 문서에 옮깁니다.
> 원칙은 그대로입니다. 변형마다 하나만 바꾸고, val로 고르고, test는 마지막에 한 번만 봅니다. 실험은 적당히만 합니다.

## C-1. 지금까지 어디까지 왔나 (방법론 지도 A~G 기준)

| 축 | 무엇 | 한 일 | 결과 (test) |
|---|---|---|---|
| A 입력 | 프롬프트 · 해상도 | E0: 형식만 고침 | valid 0.98 |
| F 압축 | 사진 토큰 줄이기 | E2: HAWK 0.20 | G6 비열등, E2E −18%, **박스 무너짐** (Pointing 0.50 → 0.27) |
| D + E 학습 | QLoRA, 앞머리 SFT | E3-L | G6 +0.086 (경계선), **Vis · Scope 크게 오름**, E2E +29% |
| C 출력 구조 | 호출 나누기 | 안 함 (E3-G 예비안) | – |
| G 시스템 | 확신도 · 라우터 | 확신도 버그만 고침 | – |

## C-2. 남은 문제 (E3-L 기준, 숫자는 고정 뒤 계산)

1. **박스**: Pointing 0.52. 박스 복사 17건, 사진 전체 덮기 10건 (test). 앞머리만 학습했으니 손대지 않은 부분입니다.
2. **속도**: E2E 13.6초. 그중 디코딩이 대부분이고, 합치지 않은 어댑터가 토큰당 +49%를 더했습니다.
3. **silver 천장**: val 상승의 절반만 test로 넘어왔습니다. manufacturing 재현율은 0.82 → 0.70으로 떨어졌습니다.
4. **확신도는 쓸모가 있어 보입니다**: E3-L의 그룹 확신도(첫 토큰 확률)가 높은 절반과 낮은 절반의 정답률 차이가 큽니다.

| | 확신도 ≥ 0.95 | 나머지 |
|---|---|---|
| val 100 (silver) | 비율 57%, 정답률 **0.965** | 0.698 |
| test 176 (사람, 고정 뒤 확인만) | 비율 50%, 정답률 **0.920** | 0.636 |

## C-3. 제안 (추천 순서)

| 순서 | 이름 | 겨냥 | 바꾸는 것 하나 | GPU | 판정 (val에서, 미리 정함) |
|---|---|---|---|---|---|
| **P1** | 확신도로 "확인 필요" 표시 | 서비스 신뢰 (G축) | 학습 없음. 확신도 문턱 하나 | **0** (CPU, 기존 결과로) | 자동 통과 비율 ≥ 50%, 그 안 정답률 ≥ 0.95 |
| **P2** | 어댑터 속도 되찾기 | 속도 (F축) | ① 어댑터를 bf16으로 바꿔 얹기 → ② 안 되면 합친 뒤 다시 4비트 | 약 1시간 | E2E ≤ E1 + 5%, E3-L과 라벨 일치 ≥ 97% |
| **P3** | 박스만 사진별로 다시 묻기 (E3-G) | 박스 (C축) | localized일 때만 사진 한 장씩 공식 grounding 문장으로 | 약 1시간 (학습 없음) | 박스 복사 0, 전체 덮기 절반 이하, val Pointing 0.18 → 0.30 |
| P4 | 3B + 같은 LoRA (Light) | 속도 · 비용 (E축 증류) | 백본만 Qwen2.5-VL-3B (bf16 LoRA) | 약 4~5시간 | 3B+LoRA가 7B E1 이상이면서 E2E 절반 이하 |
| 보류 | HAWK + 어댑터 | 속도 | 라벨 호출에만 HAWK | 약 1시간 + 코드 | P3로 박스를 따로 받게 된 뒤에만 의미 있음 |

### P1. 확신도로 "확인 필요" 표시 — 가장 먼저 (GPU 안 씀)

- **무엇**: 카드에 이미 있는 확신도(`conf.group6`)에 문턱을 정합니다. 문턱 아래면 "상담원 확인 필요"로 표시합니다.
- **왜 먼저**: GPU가 안 들고, 이미 있는 val·test 결과로 바로 계산됩니다. 서비스에서 "모델이 틀릴 수 있는 리뷰를 골라 준다"는 기능이 됩니다.
- **어떻게**: 문턱은 **val로만** 정합니다. 예를 들어 "자동 통과 정답률 0.95 이상을 지키는 가장 낮은 문턱". 그 문턱으로 test를 한 번 봅니다.
- **더 나가면**: 확신도가 낮을 때만 R0(글만)나 다른 모델에 한 번 더 묻는 라우터. test에서 E3-L과 R0 중 하나라도 맞힌 리뷰는 147/176, E3-L 혼자는 137/176이라 여지가 있습니다. 다만 이건 P1 다음 일입니다.
- **서비스**: 카드에 `confidence`, `needs_review` 두 칸 추가.

### P2. 어댑터 속도 되찾기

- **① bf16 어댑터**: 지금 어댑터는 fp32라 모듈마다 형 변환이 붙습니다([bnb.py:543-546](../../../.venv/Lib/site-packages/peft/tuners/lora/bnb.py#L543)). 학습 없이 불러올 때 bf16으로 바꾸기만 하면 됩니다. 얼마나 빨라질지는 모릅니다(재 봐야 함).
- **② 합치기**: RAM(31.8 GB)에 bf16 7B(16.6 GB)를 올려 W + 2·B·A를 계산하고, 다시 NF4로 4비트화합니다. 속도는 E1과 같아지지만, 반올림 때문에 라벨이 조금 바뀔 수 있습니다.
- **판정**: E2E가 E1 + 5% 이내이고, val 100에서 E3-L과 라벨이 97% 이상 같으면 채택. 같은 리뷰끼리 비교로 G6가 떨어지지 않았는지도 봅니다.
- **왜 중요**: 이게 안 되면 E3-L은 E1보다 3초 느린 모델로 남습니다. 서비스로 가려면 풀어야 하는 문제입니다.

### P3. 박스만 사진별로 다시 묻기 (E3 제안서의 예비안 E3-G)

- **무엇**: 첫 호출(E3-L)이 localized라고 답했을 때만, 사진 한 장씩 Qwen 공식 grounding 문장("Outline the position of … and output all the coordinates in JSON format.")으로 박스를 다시 묻습니다. 이 호출에서는 HAWK를 끕니다.
- **왜**: 박스 복사는 사진 여러 장을 한 번에 넣을 때만 생깁니다. 한 장씩 물으면 구조상 사라집니다. E3-L은 localized를 덜 답하므로(test 79/185) 추가 호출도 그만큼 적습니다.
- **미리 정할 것**: 결함 이름을 무엇으로 물을지(뜻풀이 vs 리뷰 원문 문장), 여러 사진 박스 중 무엇을 1순위로 낼지.
- **한계**: E1에서 사진 1장짜리도 17건 중 3건만 맞았습니다. 복사가 없어져도 Pointing이 크게 오른다는 보장은 없습니다.

### P4. 3B + 같은 LoRA (원래 계획의 Light 경로)

- **무엇**: 같은 레시피(앞머리 SFT, r 8)를 Qwen2.5-VL-3B에 적용합니다. 3B는 bf16으로 올라가서 QLoRA가 필요 없습니다.
- **왜**: E3-L에서 "기준 옮기기"가 에폭 1에 대부분 끝났습니다. 그렇다면 작은 모델도 기준만 옮겨 주면 꽤 따라올 수 있습니다. "3B가 7B를 얼마나 따라잡나"는 프로젝트의 비용 결론으로 이어집니다.
- **이어지는 그림**: 3B가 먼저 답하고, 확신도가 낮을 때만 7B를 부르는 캐스케이드 (P1의 확신도를 그대로 씀).
- **비용**: 학습 약 3~4시간 + val · test. 가장 무거운 제안이라 P1~P3 다음입니다.

### 하지 말 것

- **val로 r · lr · 에폭을 더 파기**: val은 100건이고 silver라서, 거기서 오른 것은 test로 반만 넘어옵니다(이번에 확인).
- **test 결과를 보고 다음 변형을 고르기**: 이번 C-2의 test 숫자는 E3-L을 고정한 뒤 확인만 한 것입니다. 다음 변형의 문턱·설정은 val로만 정합니다.
- **JSON 전체 따라 쓰기 학습**: train에 근거 문장 정답이 없습니다.
- **ViT에 LoRA**: 글만 쓴 R0가 Vis · Scope에서 E1을 이겼습니다. 문제는 "못 봄"이 아니라 "판단 기준"이었습니다.
- **같은 test로 변형을 계속 비교하기**: 변형이 늘수록 우연히 좋아 보이는 것이 생깁니다. 다음 변형들은 묶어서 마지막에 한 번씩만 test를 봅니다.

### 사람이 할 일 (GPU 실험이 아님)

- **silver 천장을 올리는 가장 확실한 길은 사람 라벨**입니다. Claude가 헷갈린 train 239건, 또는 manufacturing 경계 리뷰만 사람이 다시 보면, 같은 레시피로 다시 학습할 수 있습니다. 팀과 정할 일입니다.

## C-4. 한 줄 정리

**P1(오늘, GPU 0) → P2(속도, 약 1시간) → P3(박스, 약 1시간) → 시간이 남으면 P4(3B, 반나절).** P1~P3를 다 하면 서비스 카드는 "E3-L 라벨 + 사진별 박스 + 확인 필요 표시"이고, 속도는 E1 수준이 됩니다.

---

### 출처

- **코드**: `cluvi/` · `scripts/run_e3l_chain.py` (2026-11-15 작업 폴더), peft 0.21.0 (`.venv`), HAWK가 패치한 transformers 4.52.0 (`HAWK/.runtime/python`)
- **모델 설정**: HF 캐시 `Qwen/Qwen2.5-VL-7B-Instruct`의 `config.json` (hidden 3584, 어텐션 28헤드, KV 4헤드, 28층, 어휘 152,064)
- **우리 숫자**: `outputs/lora/e3l/` (train_log, steps.jsonl, val_head.json), `outputs/eval/e3l_val_judge.json` · `e3l_val_eval.json` · `e3l_test_eval.json` · `e3l_test_measures.json`, [설계 문서 6절 E3-L](../../초기실험_설계_v0.1.md), [E3 실험 제안](../../E3_실험_제안.html), [Qwen2.5-VL 방법론 지도](../../Qwen2.5-VL_방법론_지도.html)
- **방법 원 논문**: LoRA (Hu et al., 2021, arXiv 2106.09685), QLoRA (Dettmers et al., 2023, arXiv 2305.14314)
