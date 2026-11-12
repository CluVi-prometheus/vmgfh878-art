# cluvi — 공용 실험 코드 (베이스라인 골격)

리뷰 글 + 사진 → **분석 카드**(부위·결함·보임 정도·박스) → **상담원 에이전트**.
모든 모델은 같은 입력(`Sample`)을 받아 같은 출력(`Card`)을 낸다. 그래서 모델만 바꿔 끼우면 나머지(평가·cascade·에이전트)는 그대로 쓴다.

```
Sample ──▶ [ 분석기 하나 ]  ──▶ Card ──▶ Agent (Planner → Tool → Answer → Verifier)
            text / light / heavy / heavy_ts…
            └ cascade = 이 칸을 "여러 분석기를 싼 순서로 시도"로 바꾸는 것뿐
```

| 파일 | 역할 | 주로 손댈 사람 |
|---|---|---|
| `common.py` | 경로, 라벨 체계, `load_split` (`gold` = test 118 + test_blind 53), `Card`, 프롬프트 채우기·파싱 | 공통 (리뷰 필수) |
| `prompts/vlm-v0.2.txt` | Heavy 제로샷 프롬프트 원문 (버전 = 파일 이름) | 공통 |
| `models.py` | `text` (TF-IDF+LR), `light` (Qwen2.5-VL-3B), `heavy` (Qwen2.5-VL-7B, LLM 4bit · ViT bf16), `mock` | 모델 담당 |
| `run_hawk.py` | HAWK 런타임(transformers 4.52.0 패치본)을 앞에 두고 predict 실행 (HAWK `run.sh`의 파이썬판) | — |
| `predict.py` | 모델 × split을 **한 번만** 돌려 `outputs/preds/{모델}_{split}.jsonl` + 설정 `_config.json` 저장 | — |
| `evaluate.py` | 지표 13개를 같은 표로 (Macro-F1 6종, Mean IoU · IoU@0.5 · Pointing, 토큰 유지율, Prefill · E2E 지연, Peak VRAM) + 신뢰구간 | 공통 |
| `cascade.py` | 저장된 예측으로 cascade를 **오프라인 시뮬레이션** (모델 재실행 없음) | cascade 담당 |
| `agent.py` | mock 주문·정책 도구 + 에이전트 루프 + 근거 검증 | 에이전트 담당 |
| `demo.py` | 민원 1건을 처음부터 끝까지 | — |

## 실행 순서

```bash
python -m cluvi.demo --model mock                              # GPU 없이 전체 흐름 확인
python -m cluvi.predict --model text  --split gold             # Light Text (CPU, 몇 초)
python -m cluvi.run_hawk --model heavy --split dev --limit 5      # E0 GPU 점검 (HAWK 런타임 = transformers 4.52.0)
python -m cluvi.run_hawk --model heavy --split gold                # E1 Qwen 제로샷 (171건)
python -m cluvi.run_hawk --model heavy --split train --ids diag200  # E1 진단용 silver 200건 (E3 결정용)
python -m cluvi.run_hawk --model heavy --split gold --hawk-keep 0.20   # E2 = E1 + HAWK 원본 코드
python -m cluvi.evaluate outputs/preds/heavy_gold.jsonl outputs/preds/heavy_hawk200_gold.jsonl --base heavy_gold --format
python -m cluvi.cascade --stages text heavy            # 2-stage
python -m cluvi.demo --model heavy --row 3 --question "교환 가능해?"
```

## 실험 = 예측 파일 하나 더 만들기

| 실험 | 만드는 것 |
|---|---|
| ① Light Text | `--model text` |
| ② Light VLM | `--model light` |
| ③ Heavy Full | `--model heavy` |
| 해상도(시각 토큰) | `--model heavy --max-pixels 200704 --tag _px256` (사진당 최대 256토큰) |
| ④ Token Selection | `models.py`에 `heavy_ts20` 추가 → `--model heavy_ts20` (유지율 20% = HAWK의 80% 제거) |
| LoRA 튜닝 | `models.py`에 어댑터 로드 옵션 추가 → `--tag _lora1` |
| Cascade | 위 파일들을 `cascade.py --stages …`로 조합 (GPU 안 씀) |

## Cascade 한 줄 정의

**싼 모델부터 답을 내 보고, 확신하면 거기서 끝(조기 종료), 아니면 다음 비싼 모델로 넘긴다.**
절감은 "앞 단계에서 끝난 샘플 수"에서 나온다. 라우터(싼 모델이 *어느 VLM을 쓸지만* 고르고 자기 답은 버림)와 다르다.

이 프로젝트에서 Light Text가 끝낼 수 있는 조건은 두 개 다:
1. 결함 대분류를 확신 (`conf.group6` 높음)
2. 사진이 필요 없는 민원 (`P(not_visible)` 높음 = Text Sufficiency) — 글 모델은 박스를 못 그리니까

→ `conf.exit = min(두 확률)`. VLM 단계의 `conf.exit`는 생성한 group 토큰의 확률.

주의: 임계값 τ는 test로 고르면 안 된다 (test에 맞춘 숫자가 됨). train(silver)이나 교차검증으로 고르고 test에는 한 번만 적용.

실험 설계 전체: `reports/초기실험_설계_v0.1.md`
