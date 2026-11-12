# Heavy 경로 설계 초안 v0.1

- 작성: 2026-11-10 · 상태: **초안 (팀 확정 전)**
- 범위: 입력 → Heavy VLM → 효율화 → 멀티모달 분석 → 에이전트
- 보류: cascade (Text 모델 · 라우터 · Light VLM)는 나중에. 다만 Heavy 결과를 저장할 때 나중에 필요한 값(확신도)을 같이 남김
- 모델: Heavy = `Qwen/Qwen2.5-VL-7B-Instruct` (16GB GPU라 4비트)

```
① 입력 ──→ ② 실행 설정 ──→ ③ 효율화 ──→ Heavy VLM ──→ ④ 원출력 ──→ ⑤ 분석 결과 ──→ ⑥ 에이전트
                                                                        │
                                                                        └──→ ⑦ 평가 기록 (리뷰 1건 = 1줄)
```

---

## 0. 용어: silver와 gold

| | gold (정답) | silver (준정답) |
|---|---|---|
| 누가 만들었나 | 사람이 확정 | Claude Opus가 자동으로 (사진까지 보고) |
| 얼마나 맞나 | 기준 자체. 단 사람끼리도 대분류 일치 70% | 사람 정답 대비 대분류 약 73%, 보임 정도 약 86% (블라인드 53건) |
| 우리 데이터 | test 118건 (round2 30 블라인드 확정 + gold90 88 초안 검수) + test_blind 53건 (대분류 · 보임 정도만) = **171건** | train **981건** |
| 쓰는 곳 | **최종 평가**에만, 설정마다 한 번 | 학습, 설정끼리 비교, 프롬프트 시범 |

- gold90 88건은 사람이 Claude 초안을 보고 검수한 것이라 Claude 쪽으로 조금 기울 수 있음. round2 30건과 블라인드 53건이 가장 깨끗함
- 지민님의 텍스트 충분성 라벨도 silver (Claude teacher)

---

## 1. 각 칸 입출력 형식

### ① 입력 (Request)

```json
{
  "review_id": 3424,
  "domain": "Electronics",
  "title": "Below average but does the job",
  "review": "The product is below average... one hose fits loosely...",
  "images": ["3424_0.jpg", "3424_1.jpg"]
}
```

| 필드 | 형식 | 설명 |
|---|---|---|
| review_id | int | CESAMARD `row_id` |
| domain | Books · Fashion · Electronics · Edible · Misc | 부위 목록이 도메인별로 다름 |
| title, review | str | 영어 원문. review는 1,500자에서 자름 |
| images | list[str] | 고객이 올린 순서 그대로 |

주문 정보는 여기 넣지 않음. 에이전트 단계에서 mock 도구로 조회함.

### ② 실행 설정 (RunConfig) — 실험마다 바꾸는 손잡이와 고정값

```json
{
  "run_id": "heavy-full-v0.1",
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "quant": "nf4-llm / bf16-vision",
  "prompt_version": "vlm-v0.1",
  "input_mode": "text+image",
  "max_images": 4,
  "max_pixels_per_image": 802816,
  "efficiency": {"method": "none", "budget": null},
  "decoding": {"do_sample": false, "max_new_tokens": 200},
  "seed": 0
}
```

| 필드 | 값 | 비고 |
|---|---|---|
| input_mode | `text+image` / `text-only` | text-only = 같은 모델에서 사진 효과만 따로 재는 용도 |
| max_images | 4 | 리뷰의 96%가 4장 이하. 넘으면 앞 4장 |
| max_pixels_per_image | 802,816 (= 사진당 최대 1,024토큰) | Qwen2.5-VL은 토큰 1개 = 28×28px. 기본값(약 1,280만px)이면 16GB에서 터짐 |
| efficiency.method | `none` / `resize` / `token_select:<이름>` / `merge:<이름>` | 2단계 실험에서 이 칸만 바꿈 |
| efficiency.budget | 사진당 시각 토큰 수 (16 · 32 · 64 · 128 · 256 · …) | `none`이면 null |
| quant | LLM만 4비트, ViT는 bf16 (안) | 작은 결함을 보는 ViT는 정밀도를 지킴. 팀 확정 필요 |

`run_id`마다 설정 파일을 `results/<run_id>/config.json`으로 저장. **설정 파일이 없으면 결과로 인정하지 않음.**

### ③ 효율화 (Efficient Visual Compute)

- 입력: 사진 + `efficiency` 설정
- 출력: 처리된 시각 입력 + 통계

```json
{"n_images_used": 2, "visual_tokens_before": 1904, "visual_tokens_after": 512, "method": "resize", "budget": 256}
```

- `resize`는 `max_pixels`를 낮춰서 구현 (ViT 비용도 줄어듦)
- `token_select`, `merge`는 모델 안에서 토큰을 고름. Qwen2.5-VL은 DeepStack이 없어서, 남긴 토큰의 MRoPE 좌표만 원래 값으로 유지하면 됨
- 고른 토큰의 위치(`kept_index`)는 선택적으로 저장. 박스 덮음 비율을 잴 때 필요

### ④ VLM 원출력 (프롬프트가 요구하는 JSON)

```json
{"complaint": "one hose fits loosely", "group": "manufacturing", "defect": "assembly",
 "part": "other_accessory", "vis": "not_visible", "img": null, "bbox_2d": null}
```

### ⑤ 분석 결과 (Analysis) — 에이전트와 평가가 받는 형식

원출력을 파싱하고 검증한 결과. **에이전트와 평가는 이 형식만 봄.**

```json
{
  "review_id": 3424,
  "run_id": "heavy-full-v0.1",
  "valid": true,
  "analysis": {
    "complaint": "one hose fits loosely",
    "part_scope": "accessory",
    "part": "other_accessory",
    "group6": "manufacturing",
    "group5": "physical_defect",
    "defect": "assembly",
    "visibility": "not_visible",
    "evidence": {"image_index": null, "bbox": null}
  },
  "confidence": {"group6": 0.71},
  "visual": {"n_images_used": 2, "visual_tokens_before": 1904, "visual_tokens_after": 1904, "method": "none", "budget": null},
  "cost": {
    "latency_ms": {"vision": 410, "prefill": 620, "decode": 2300, "total": 3330},
    "input_tokens": 2310, "output_tokens": 58, "peak_vram_mb": 9800
  },
  "raw": "<모델 원문 앞 400자>"
}
```

| 필드 | 만드는 규칙 |
|---|---|
| part_scope | `part` 코드에서 자동으로 (코드표 기준) |
| group5 | damage · manufacturing → `physical_defect`, 나머지는 group6 그대로 |
| defect | 그 그룹 아래 목록에 없으면 null (그룹만 살림) |
| evidence.bbox | Qwen2.5-VL은 **리사이즈된 사진의 픽셀 좌표**로 답함 → 리사이즈 크기로 나눠 0~1로 바꿔 저장 |
| valid | JSON 파싱 성공 + group · defect · part · vis가 모두 목록 안. 실패하면 false, 분석 칸은 null (평가에서 오답 처리) |
| confidence.group6 | 생성된 group 값 첫 토큰의 확률 (`generate(..., output_scores=True)`). 추가 연산 없음. **지금은 기록만, cascade 때 씀** |
| cost.latency_ms | vision = ViT 시간 (forward hook), prefill = 첫 토큰까지 시간 − vision, decode = 전체 − 첫 토큰까지 시간. warm-up 3건 제외, batch 1 |

### ⑥ 에이전트 (가볍게)

입력: ⑤의 `analysis` + 주문 정보(mock) + 정책(mock)

```json
{
  "analysis": {"group6": "damage", "defect": "tear_hole", "part": "sleeve", "visibility": "localized", "...": "..."},
  "order": {"order_id": "A-1029", "product_name": "Men's blazer", "domain": "Fashion",
            "delivered_at": "2026-11-02", "price": 2499, "return_window_days": 10}
}
```

출력:

```json
{
  "review_id": 3424,
  "decision": "exchange",
  "needs_human": false,
  "reason": "배송 8일차, 반품 기간 안. 사진에 소매 찢어짐이 보임",
  "tool_calls": [{"tool": "get_order", "args": {"order_id": "A-1029"}},
                 {"tool": "get_policy", "args": {"domain": "Fashion", "group6": "damage"}}],
  "reply_draft_ko": "고객님, 보내주신 사진에서 소매 찢어짐을 확인했습니다..."
}
```

- decision: `exchange` · `refund` · `repair_guide` · `ask_more_info` · `escalate`
- 사람 이관(needs_human)은 **에이전트 쪽 규칙**으로 정함 (분석 결과에는 넣지 않음). 초안 규칙: `valid=false`이거나, `visibility=not_visible`이면서 `defect=vague`이거나, 정책상 판단 불가일 때

### ⑦ 평가 기록

- ⑤ 분석 결과를 그대로 **리뷰 1건 = JSONL 1줄**로 저장: `results/<run_id>/<split>.jsonl`
- 정답(gold · silver)은 기록에 넣지 않고, 평가할 때 HF 데이터에서 `review_id`로 붙임 (기록 파일을 봐도 정답이 안 보이게)
- 모든 split (train · validation · test · test_blind)에 다 돌려 둠. train 결과는 나중에 cascade와 설정 비교에 다시 씀

---

## 2. Heavy VLM 프롬프트 초안 `vlm-v0.1`

### 2.1 출발점과 주의

- 라벨링 때 Qwen3-VL-8B에 쓴 `v1.3c` (`scripts/cesamard/qwen_label.py`의 `RULES_C`)를 바탕으로 함
- **주의:** `v1.3c`는 gold 결과를 보고 고친 버전임. `vlm-v0.1`부터는 **dev · train에서만** 고치고, gold 171건은 고정 후 평가에만 씀
- 지난 라벨링 때 알려진 문제: 로컬 VLM이 `not_as_described`를 과하게 고름 (gold 118건에서 사람 28건, Qwen 40~46건). 그래서 규칙 2에 좁은 정의를 그대로 둠
- `v1.3c` 대비 바뀐 점: 사진 번호 안내 추가, `other` 부위 코드 추가 (gold에 7건 있음), Qwen2.5-VL 좌표 안내 추가

### 2.2 구조: 공통 부분 + 모델별 감싸기

| 층 | 내용 | 누가 |
|---|---|---|
| 공통 (모든 모델 같음) | A 리뷰 블록 · B 라벨 목록 · C 판단 규칙 · D 출력 형식 | 팀 공용, 버전 관리 |
| 모델별 | 채팅 템플릿, 사진 넣는 위치, 좌표 안내 한 줄, (Qwen3 텍스트 모델이면 thinking 끄기) | 모델 담당자 |

### 2.3 공통 부분 (영어, 그대로 씀)

`{...}`는 채워 넣는 자리. `{part_list}`는 도메인별로 만듦.

```text
Amazon India customer complaint ({domain}).
Title: {title}
Review: {review}

The customer attached {n_images} photo(s), shown above and numbered 0 to {last_index} in order.

Defect groups and the defects inside each:
- group damage: defects tear_hole, crush_break, scratch, opened_tampered
- group contamination: defects pest, mold_spoil, foreign_object, stain_dirt
- group manufacturing: defects material, stitching, print, color, display, assembly
- group not_as_described: defects counterfeit_wrong, missing, used, label_expiry
- group function: defects malfunction, performance, fit_size, design_usability
- group non_visual: defects taste_smell, content, service, price, vague
Allowed part codes ({part_list}; other: other).

How to decide, in this order:
1. Find the MAIN complaint: the complaint in the TITLE comes first; if the title has none, the one the body talks about most. Only complaints the review states. Copy it as a short quote (at most 12 words) into "complaint".
2. Choose the group from that complaint:
   - non_visual: taste or smell; delivery, refund, return, warranty, seller or customer support; price or value for money; book/media content; or the review names no concrete problem ("bad product", "waste of money") and the photos show none.
   - not_as_described: ONLY when the item received differs from what was ordered or advertised: fake / duplicate / not original, a different product, size, colour or model than ordered, a piece or item missing, already used, expired. Poor quality, cheap material, "not as expected", "not as shown in the picture" because it looks cheap, or any defect is NOT not_as_described.
   - damage: visibly torn, broken, crushed, cracked (incl. a cracked screen), scratched, dented, or opened/tampered on arrival.
   - contamination: insects, mold or spoiled food, a foreign object (metal piece, staple, hair), stains or dirt.
   - manufacturing: made wrong from the start: thin or cheap material, pilling, poor paper, stitching, print, colour, display defect such as backlight bleed, assembly or glue.
   - function: looks fine but does not work or works badly; arrived as ordered but does not fit; awkward design.
3. part = where the problem actually is (not where it is shown). Expired food -> contents. Size problems -> whole. Throwaway bag/box -> product_package; jar/bottle kept during use -> container. label/manual only if the label or manual itself is wrong or missing. Service/price -> none. No nameable part (overheating, slowness, colour) -> whole. A part not in the list -> other.
4. vis: "localized" if the defect (or evidence such as an error message, an expired date, a cracked screen) is visible in a specific region of a photo; "diffuse" if visible but over the whole product (colour, fit, generally thin, a counterfeit that looks clearly different); "not_visible" otherwise (function, taste, service, content, missing items, unrelated photo, a counterfeit that looks identical). Size tags, price tags and numbers in a manual are not evidence.
5. If localized, give the image index where it shows best and a tight box around the defect itself. Otherwise set "img" and "bbox_2d" to null.

Answer with JSON only, no other text. "group" is one group name; "defect" must be one of the defects listed under that group:
{"complaint": "<short quote>", "group": "<group>", "defect": "<defect>", "part": "<part code>", "vis": "localized|diffuse|not_visible", "img": <image index or null>, "bbox_2d": [x1, y1, x2, y2] or null}
```

`{part_list}` 예 (Fashion):
`product: body_fabric, seam, pocket, sleeve, collar, fastener, print_deco, lining, hem, whole; packaging: outer_box, product_package, seal; accessory: charger_cable, other_accessory; label_doc: label, manual; none: none`

### 2.4 Qwen2.5-VL용 감싸기

- 사진을 먼저, 글을 뒤에 넣음 (지난 라벨링과 같은 순서)
- 좌표 안내 한 줄을 공통 부분 끝에 붙임: `bbox_2d uses pixel coordinates of that photo as you see it.`
- 시스템 프롬프트는 모델 기본값 그대로

```python
content = [{"type": "image", "image": path} for path in images[:4]]
content.append({"type": "text", "text": common_prompt + "\nbbox_2d uses pixel coordinates of that photo as you see it."})
messages = [{"role": "user", "content": content}]
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                       return_dict=True, return_tensors="pt")
# processor는 min_pixels=3136, max_pixels=802816 으로 만듦 (RunConfig와 같게)
```

- `text-only` 변형: 사진 줄과 규칙 4 · 5, 출력의 `vis` · `img` · `bbox_2d`를 뺀 버전 (`vlm-v0.1-text`). 같은 모델에서 사진 효과를 잴 때만 씀

### 2.5 시범 검증 (dev에서 20~30건, 고정 전)

표본: 6개 그룹이 골고루, 사진 1장 · 여러 장 섞어서, 글만으로 충분한 건 · 불충분한 건 섞어서.

| 검증 항목 | 보는 법 | 기준 (안) |
|---|---|---|
| JSON 정상 출력률 | valid 비율 | 95% 이상 |
| 라벨 목록 준수 | group · defect · part가 목록 안 | 위와 같이 확인 |
| 모호 처리 | 구체적 불만 없는 리뷰가 `non_visual / vague`로 가는가 | 눈으로 확인 |
| 그룹 쏠림 | 30건의 그룹 분포를 silver 분포와 비교 (특히 not_as_described 과다) | 크게 쏠리지 않음 |
| 부위 출력 | `whole` · `other` 남발 여부 | 눈으로 확인 |
| 평균 입력 · 출력 토큰 | cost 필드 | 기록 |
| 평균 지연, 최대 VRAM | cost 필드 | 16GB 안 |

- 이 단계에서는 정답률을 보지 않음 (표본이 작아 의미 없음)
- 고치면 버전을 올리고 (`vlm-v0.2` …) 바뀐 점을 이 문서에 적음. 팀이 합의하면 `vlm-v1.0`으로 고정하고, 그 뒤에는 모델별로 성능을 올리려고 고치지 않음

---

## 3. VLM 평가지표 초안

### 3.1 평가셋

| 셋 | 건수 (중복 제외) | 있는 정답 | 용도 |
|---|---|---|---|
| gold-test | 118 | 부위 · 대분류 · 세부 · 보임 정도 · 박스 | 최종 평가 |
| gold-blind | 53 | 대분류 · 보임 정도 | 최종 평가 (대분류 · 보임 정도) |
| silver-train | 981 | Claude 라벨 전부 | 설정끼리 짝 비교, 답 유지율 |
| dev | 88 | (v2에 정답 없음) | 프롬프트 시범 (형식 확인만) |

- gold 171건은 **고정된 설정마다 한 번**만 돌림. 프롬프트 · 설정 고르기에는 쓰지 않음
- 같은 글의 중복 리뷰는 대표 1건만 셈

### 3.2 성능 지표

| 지표 | 정의 | 평가셋 | 구분 |
|---|---|---|---|
| **group6 정확도** | 예측 대분류 = 정답. `valid=false`는 오답 | gold 171 | **주** |
| group5 정확도 | 파손 · 제조를 합친 5개 기준 (사람끼리도 가장 헷갈리는 경계) | gold 171 | 보조 |
| group6 macro-F1 | 6개 그룹 F1의 평균. 오염이 6건뿐이라 흔들림 주의 | gold 171 | 보조 |
| 그룹별 재현율 + 혼동행렬 | 어느 그룹으로 틀리는지 (not_as_described 과다 확인) | gold 171 | 보조 |
| 세부 결함 정확도 | 27종 정확 일치 | gold 118 | 보조 |
| 부위 정확도 / 부위 묶음 정확도 | part 일치 / part_scope 일치 | gold 118 | 보조 |
| 보임 정도 정확도 | visibility 3단계 일치 | gold 171 | 보조 |
| 근거 포인팅 | 예측 박스 중심이 정답 박스 안 (정답이 localized이고 박스가 있는 건) | gold 53건 (+ silver 460건) | 선택 |
| 형식 성공률 | valid 비율 | 전체 | 운영 |
| 확신도 보정 오차 (ECE) | confidence.group6을 10구간으로 나눠 계산 | gold 171 | 기록만 (cascade용) |

### 3.3 비용 지표

| 지표 | 정의 | 구분 |
|---|---|---|
| **리뷰당 지연 중앙값** | `cost.latency_ms.total`. 기준 GPU 하나에서, batch 1, warm-up 3건 제외 | **주** |
| 지연 상위 10% (p90) | 느린 리뷰 쪽 | 보조 |
| 단계별 지연 | vision / prefill / decode | 보조 (어디서 시간이 드는지) |
| 시각 토큰 수 · 입력 토큰 수 · 출력 토큰 수 | GPU와 상관없는 비용. 다른 PC 결과끼리도 비교 가능 | 보조 |
| 최대 VRAM | `torch.cuda.max_memory_allocated` | 보조 |

### 3.4 효율화 비교 규칙 (2단계)

- 기준선: `heavy-full` (같은 프롬프트, 같은 사진 수, 효율화 없음). 바꾸는 것은 `efficiency` 하나뿐
- **답 유지율**: 같은 리뷰에서 효율화 설정의 group6이 `heavy-full`과 같은 비율. 정답이 필요 없어서 1,254건 전부 사용
- **정확도 차이**: silver-train 981건과 gold 171건에서 같은 리뷰끼리 짝지어 비교 (paired bootstrap 95% 신뢰구간)
- **통과 기준 (안)**: 답 유지율 95% 이상, 그리고 gold 정확도 차이의 신뢰구간 하한이 −3%p 이상. 3%p는 팀 확정 필요
- **결과 그림**: 가로축 리뷰당 시각 토큰 수(또는 지연), 세로축 group6 정확도. `resize` 선과 `token_select` 선을 겹쳐 그림
- **쪼개 보기**: 정답 visibility별 (한 곳에 보임 / 퍼짐 / 안 보임). 국소 결함에서 더 떨어지는지 확인
- **토큰 셀렉션 전용**: 박스 덮음 비율 = 남긴 토큰이 정답 박스 면적의 몇 %를 덮는가 (silver 박스 460건)

### 3.5 통계

- 신뢰구간: 리뷰를 복원추출로 다시 뽑는 bootstrap 10,000회, 95%
- 두 설정 비교: 같은 리뷰끼리 짝지은 bootstrap (또는 McNemar 검정)
- 표본 크기 감: 정확도 75% 근처에서 171건은 ±6.5%p, 981건은 ±2.7%p. **gold 171건에서 5%p 미만 차이는 결론 내지 않음**

### 3.6 해석 기준

- 사람끼리 대분류 일치 70%, Claude Opus 73% (블라인드 53건)
- 참고 (지난 라벨링, Qwen3-VL-8B 원본, 프롬프트 v1.3b · v1.3c): 블라인드 대분류 59% · 58%, round2 대분류 80% · 73%, round2 세부 70% · 63%, 부위 50%, 보임 정도 69%. Heavy 기준선의 대략적인 기대치

### 3.7 리포트 1차 표 양식

| run_id | 모델 | 입력 | 사진당 토큰 상한 | group6 정확도 [95% CI] | group5 | macro-F1 | 세부 | 부위 | 보임 정도 | 형식 성공률 | 지연 중앙값 (vision / prefill / decode) | 시각 토큰 | 출력 토큰 | VRAM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| heavy-full-v0.1 | 7B 4비트 | 글+사진 | 1,024 | | | | | | | | | | | |
| heavy-text-v0.1 | 7B 4비트 | 글만 | – | | | | | | | | | | | |
| heavy-resize256-v0.1 | 7B 4비트 | 글+사진 | 256 | | | | | | | | | | | |

---

## 4. 회의에서 정할 것

| 정할 것 | 초안 |
|---|---|
| 출력 필드 범위 | complaint 인용 · 박스까지 유지 (인용은 출력 토큰 약 15개 추가, 에이전트 근거로 씀) |
| 사진 최대 장수 · 해상도 상한 | 4장 · 사진당 1,024토큰 |
| 4비트 범위 | LLM만 4비트, ViT는 bf16 |
| 효율화 허용 하락폭 | 3%p |
| 기준 GPU | 비용 측정은 한 대에서만 |
| `other` 부위 코드 | 허용 (gold에 7건) |
| 시범 표본 | dev에서 30건, 그룹별 5건 |

## 5. 보류: cascade

- Text 모델 · 라우터 · Light VLM은 Heavy 경로가 끝난 뒤에 함
- 지금 해 둘 것은 하나: ⑤ 분석 결과에 `confidence.group6`를 남기고, 모든 split을 돌려 저장해 두기. 나중에 "Text가 틀리고 Heavy가 맞힌 리뷰"를 찾는 데 그대로 씀

## 변경 기록

| 버전 | 날짜 | 내용 |
|---|---|---|
| v0.1 | 2026-11-10 | 첫 초안 |
| → | 2026-11-10 | 출력 형식(evidence 추가) · 프롬프트(`vlm-v0.2`) · 평가지표(13개, Macro-F1 중심)는 `reports/초기실험_설계_v0.1.md`가 이 문서 2~3절을 대신함 |
