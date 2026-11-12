# vmgfh878-art · CluVi 개인 실험

Heavy VLM 경로 베이스라인 실험 레포입니다. 팀 종합 레포는 [CluVi-prometheus/cluvi](https://github.com/CluVi-prometheus/cluvi), 팀이 맞춘 출력 카드 · 평가지표는 [`docs/agreements.md`](https://github.com/CluVi-prometheus/cluvi/blob/main/docs/agreements.md)에 있습니다.

| 실험 | 내용 |
|---|---|
| E1 | Qwen2.5-VL-7B 제로샷 (모델 카드 공식 추론 방식) |
| E2 | E1 + HAWK 원본 코드 (시각 토큰 20%만 남김) |
| E3 | E1 · E2에서 보인 문제 기반 개선 (LoRA 등) |

설계와 근거는 [`reports/초기실험_설계_v0.1.md`](reports/초기실험_설계_v0.1.md), 실행법은 [`cluvi/README.md`](cluvi/README.md)에 있습니다.

## 폴더

| 폴더 | 내용 |
|---|---|
| [`cluvi/`](cluvi/) | 실험 코드: 데이터 로딩 · 모델 · 예측 · 평가 · LoRA 학습 · 프롬프트 |
| [`reports/`](reports/) | 설계 문서 · 해설 페이지 |
| [`outputs/preds/`](outputs/preds/) | 예측 결과 (`{실행}.jsonl`)와 그 실행의 설정 (`{실행}_config.json`) |
| [`scripts/cesamard/`](scripts/cesamard/) | 데이터셋 구축 · 라벨링 스크립트 |
| `CESAMARD_*.md` | 라벨링 가이드 · 결과 · 계획 |

## 환경

- GPU: RTX 5060 Ti 16GB (Windows). LLM만 4비트(NF4), 비전 인코더는 bf16
- VLM 실행은 HAWK 런타임(transformers 4.52.0 패치본)으로 함: [peppery77/HAWK](https://github.com/peppery77/HAWK) 커밋 `3af2071`을 `HAWK/`에 받고 런타임을 만든 뒤 `python -m cluvi.run_hawk ...`
- 데이터: Hugging Face `CluVi-prometheus/cesamard-defect` (비공개)를 `data/cesamard/`에 둠. 데이터 · 사진 · 체크포인트는 이 레포에 없습니다.
