# CluVi Heavy 모델 카드 (E1 · E2 · E3-L)

초기 실험 세 개의 모델 카드입니다. 폴더 하나가 Hugging Face 모델 저장소 하나에 해당하고, 각 폴더의 `README.md`가 그 저장소의 모델 카드입니다.

| 카드 | 모델 | 상태 | test G6 | test Vis | test Pointing | E2E (중앙값) |
|---|---|---|---|---|---|---|
| [E1](e1-qwen25vl7b-zeroshot/README.md) | Qwen2.5-VL-7B 제로샷 (프롬프트 v0.3.3) | 완료 | 0.675 | 0.507 | 0.500 | 10.6 s |
| [E2](e2-qwen25vl7b-hawk020/README.md) | E1 + HAWK (시각 토큰 20% 유지) | 완료 | 0.678 (비열등) | 0.483 | 0.268 | 8.7 s |
| [E3-L](e3l-qwen25vl7b-qlora-head/README.md) | E1 + QLoRA 어댑터 (앞머리 3칸 학습, 에폭 3) | 완료 · 고정 (2026-11-15) | **0.761** (짝 +0.086, test_blind +0.170) | **0.720** | 0.518 | 13.6 s |

## 따른 양식

- **Hugging Face 공식 모델 카드 템플릿**: `huggingface_hub` 1.31.0에 들어 있는 `templates/modelcard_template.md`의 절 구성과 순서를 그대로 썼습니다.
  - Model Details → Uses → Bias, Risks, and Limitations → How to Get Started → Training Details → Evaluation → Model Examination → Environmental Impact → Technical Specifications → Citation → Glossary → More Information → Authors → Contact
  - 절 제목은 영어 원문 그대로 두고, 본문만 한국어로 썼습니다.
- **메타데이터(YAML 머리말)**: Hub 명세(https://github.com/huggingface/hub-docs/blob/main/modelcard.md)를 따랐습니다. `base_model`, `pipeline_tag`, `datasets`, `model-index`(평가 결과) 등이 들어갑니다.
  - E3-L은 PEFT 어댑터라 `library_name: peft`, `base_model_relation: adapter`를 씁니다.
- **양식의 원래 근거**: Mitchell et al. (2019), *Model Cards for Model Reporting*, https://arxiv.org/abs/1810.03993. 쓰임새, 평가 요인, 지표, 윤리적 고려를 같이 적는 형식입니다.

## 쓸 때 주의

- E1·E2는 가중치를 바꾸지 않았습니다. 그래서 "모델" = 기반 가중치 + 4비트 로딩 + 프롬프트 + 파서(+ HAWK) 묶음입니다.
- test 성능은 **test 122와 test_blind 54를 나눠** 인용하세요. 프롬프트 계보가 라벨링 때 test 122를 보았습니다.
- 숫자 출처:
  - 원자료: `outputs/eval/e1_metrics.json`, `outputs/eval/e2_vs_e1_test.json`, `outputs/lora/e3l/`
  - 정리본: `reports/중간실험_결과_기록.md`
- 탄소 배출은 GPU 보드 전력 × 시간 × 배출계수(약 0.45 kgCO2eq/kWh)로 잡은 **추정 상한**입니다.
- 어댑터 가중치(`*.safetensors`)는 레포에 넣지 않았습니다. 학습 설정·기록만 `outputs/lora/`에 있습니다.
