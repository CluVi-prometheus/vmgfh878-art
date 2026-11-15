# vmgfh878-art · CluVi 개인 실험

Heavy VLM 경로 베이스라인 실험 레포입니다. 고객 불만 리뷰(영어 글 + 사진)를 Qwen2.5-VL-7B가 읽고 상담원용 분석 카드를 냅니다. 팀 종합 레포는 [CluVi-prometheus/cluvi](https://github.com/CluVi-prometheus/cluvi), 팀이 맞춘 출력 카드 · 평가지표는 [`docs/agreements.md`](https://github.com/CluVi-prometheus/cluvi/blob/main/docs/agreements.md)에 있습니다.

## 실험과 결과 (test 185, 사람 정답)

| 실험 | 내용 | G6 F1 | Vis F1 | Scope F1 | Pointing | E2E 중앙값 | 모델 카드 |
|---|---|---|---|---|---|---|---|
| R0 | 글만 보는 TF-IDF (참고선) | 0.539 | 0.577 | 0.594 | – | – | – |
| E1 | Qwen2.5-VL-7B 제로샷 (공식 추론 방식 + 프롬프트 v0.3.3) | 0.675 | 0.507 | 0.376 | 0.500 | 10.6 s | [E1](model_cards/e1-qwen25vl7b-zeroshot/README.md) |
| E2 | E1 + HAWK 원본 코드 (시각 토큰 20%만 남김) | 0.678 | 0.483 | 0.368 | 0.268 | 8.7 s | [E2](model_cards/e2-qwen25vl7b-hawk020/README.md) |
| E3-L | E1 + QLoRA 어댑터 (답의 앞머리 3칸만 학습) | **0.761** | **0.720** | **0.656** | 0.518 | 13.6 s | [E3-L](model_cards/e3l-qwen25vl7b-qlora-head/README.md) |

- **E2**: G6는 E1과 비열등합니다(짝 차이 +0.004 [−0.048, +0.046]). 18% 빨라지지만 박스가 무너집니다.
- **E3-L**: Visibility·Scope를 사람 정답에서도 고쳤습니다.
  - G6 짝 차이는 +0.086 [−0.000, +0.164]로 경계선입니다.
  - 안 본 test_blind 54건에서는 +0.170 [+0.058, +0.288]입니다.
  - 대가는 E2E +29%입니다(병합하지 않은 어댑터).
- test 122는 프롬프트 계보가 라벨링 때 본 집합이라, 성능은 test 122와 test_blind를 나눠 봅니다.

설계와 근거는 [`reports/초기실험_설계_v0.1.md`](reports/초기실험_설계_v0.1.md)(문서 v0.8)에 있습니다. 실행 단위 수치는 [`reports/중간실험_결과_기록.md`](reports/중간실험_결과_기록.md), 실행법은 [`cluvi/README.md`](cluvi/README.md)에 있습니다.

## 폴더

| 폴더 | 내용 |
|---|---|
| [`cluvi/`](cluvi/) | 실험 코드: 데이터 로딩 · 모델 · 예측 · 평가 · LoRA 학습 · 판정 · 프롬프트 |
| [`model_cards/`](model_cards/) | E1 · E2 · E3-L 모델 카드 (Hugging Face 공식 템플릿 형식) |
| [`reports/`](reports/) | 설계 문서 · 결과 기록 · 해설 페이지 · E3 제안 |
| [`reports/공부/`](reports/공부/) | 코드 따라가기 노트 (E1 · E2, E3-L) |
| [`outputs/preds/`](outputs/preds/) | 예측 결과 (`{실행}.jsonl`)와 그 실행의 설정 (`{실행}_config.json`) |
| [`outputs/eval/`](outputs/eval/) | 지표 · 짝 비교 · E3-L 판정 결과 (json) |
| [`outputs/lora/`](outputs/lora/) | E3-L 학습 설정 · 로그 · 에폭별 val 점검 (어댑터 가중치는 없음) |
| [`scripts/`](scripts/) | E2 · E3-L 실행 체인, 데이터셋 구축 · 라벨링 스크립트 (`scripts/cesamard/`) |
| `CESAMARD_*.md` | 라벨링 가이드 · 결과 · 계획 |

## 환경

- GPU: RTX 5060 Ti 16GB (Windows). LLM만 4비트(NF4), 비전 인코더는 bf16
- VLM 실행과 LoRA 학습은 HAWK 런타임(transformers 4.52.0 패치본)으로 합니다. 준비 순서:
  1. [peppery77/HAWK](https://github.com/peppery77/HAWK) 커밋 `3af2071`을 `HAWK/`에 받습니다.
  2. 런타임을 만듭니다.
  3. `python -m cluvi.run_hawk ...`로 실행합니다(E1 · E3-L은 HAWK를 끈 채로 씀).
- 주요 버전: torch 2.11.0+cu128 · qwen-vl-utils 0.0.11 · bitsandbytes 0.50.2 · PEFT 0.21.0
- 데이터: 코드는 `data/cesamard/`(Hugging Face `CluVi-prometheus/cesamard-defect`, 비공개)를 읽습니다. 같은 분할(train 881 · val 100 · test 185 · dev 88)을 `CluVi-prometheus/cesamard-defect-v3`(비공개)로도 올려 두었습니다.
- 데이터 · 고객 사진 · 모델 가중치 · 어댑터는 이 레포에 없습니다.
