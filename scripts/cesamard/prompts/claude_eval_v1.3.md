You label 30 Amazon India complaint reviews (text + customer photos) for a dataset. Your labels are compared with other labelers. Work independently: do not open any `*_labels*.json`, `*_gold_*.json`, anything under `data/cesamard/labeling/snapshots/`, or any file in `data/cesamard/labeling/auto/` other than your own ID file and your own output file. Do not modify existing files.

Project root: `C:\Users\user\Documents\멀티모달 프로젝트`. Python: `.venv/Scripts/python.exe`.

## Input
Review IDs: `data/cesamard/labeling/auto/evalset_part{P}_ids.txt`. Full items are in `data/cesamard/labeling/evalset.json` (a list; use only items whose `row_id` is in your ID file): `row_id`, `domain`, `title`, `text` (English), `images` (up to 4; `idx`, `src` like `images_hr/XXXX.jpg`, relative to `data/cesamard/`). Read the English text and LOOK AT EVERY IMAGE with the Read tool (`data/cesamard/<src>`). Crop/zoom with PIL when a defect is small; write crops only to your scratchpad.

## Label (guide v1.3): exactly ONE issue per review, the main complaint
Pick the main complaint: the complaint stated in the TITLE comes first; if the title has none, the complaint the body spends the most words on. Only complaints the review explicitly states (a photo-only defect is used only when the text is vague, e.g. "damaged product"). Service/price can be the main complaint (part = "none").

**① Affected Part**: where the problem actually is (the thing that is wrong, not where it is shown). One `part` code; `scope` is its group:
- scope `product`: Books: cover, corner, spine_binding, pages, print_surface | Fashion: body_fabric, seam, pocket, sleeve, collar, fastener, print_deco, lining, hem | Electronics: housing, display, camera, battery, port, button, audio, internal, software | Edible: contents, container | any domain: whole (no specific part nameable: overheating, slowness, color, size)
- scope `packaging`: outer_box, product_package, seal
- scope `accessory`: charger_cable, other_accessory
- scope `label_doc`: label, manual. ONLY when the label/manual itself is wrong or missing
- scope `none`: none (service, price) | scope `other`: other (then `part_text` required)
Rules: expired → contents (the dated label is an evidence box in ③). Size problems → whole. Throwaway bag/box/pouch → packaging > product_package; jar/bottle/tub kept during use → container. Vague text → decide the part from the photo. `part_text` optional.

**② Issue / Defect**: `group` > `defect`:
- damage: tear_hole, crush_break, scratch, opened_tampered
- contamination: pest, mold_spoil, foreign_object (metal piece, staple pin, hair), stain_dirt
- manufacturing: material (thin / cheap fabric / poor paper / pilling / dirty material), stitching, print, color, display, assembly
- not_as_described: counterfeit_wrong (incl. a different size/color/product than ordered), missing, used, label_expiry
- function: malfunction, performance, fit_size (arrived as ordered but doesn't fit), design_usability (awkward design, e.g. must clean a filter every time)
- non_visual: taste_smell, content (book/media content ONLY), service (warranty, refund, delivery, support), price, vague
Rules: choose by SYMPTOM, not guessed cause (visibly broken/torn/cracked incl. cracked screen → damage; looks fine but works/displays wrongly → function > malfunction; made wrong from the start → manufacturing). not_as_described takes priority, but only when the item received differs from what was ordered/advertised (fake, wrong item/size/color, missing piece, used, expired); poor quality or "not as expected" alone is NOT not_as_described. Vague text + concrete defect in photo → that defect. non_visual > vague only if nothing concrete anywhere.

**③ Grounding**: `vis` on the image where it shows best:
- `localized`: defect or evidence visible in a specific region (< half the product) → boxes
- `diffuse`: visible over the whole product (color, fit, generally thin; a counterfeit that LOOKS clearly different) → no boxes
- `not_visible`: function, taste, content, missing items, unrelated photo, a counterfeit that looks identical to the real thing → no boxes
Boxes: REQUIRED on the single image where it shows best; other images optional. Tight around the defect; separate spots → separate boxes (≤5). Never box the part when the defect itself isn't visible. `target: "evidence"` only when the photo alone shows a problem (warning message, expired date, cracked screen); size tags, price tags, manual numbers, book pictures are NOT evidence → not_visible. Coordinates `[x1, y1, x2, y2]` normalized 0–1 of the ORIGINAL image, top-left origin, 3 decimals (if you measure on a crop or a resized copy, convert back).
`marks[idx] = true` if the customer drew a circle/arrow/box/text on that photo to point at the problem. Scribbles that only hide personal data (names, addresses, order numbers) are NOT marks.
`unsure: true` only when you would genuinely pick a different GROUP with a real chance; then `note` = the alternative in short Korean (e.g. "대안: function>malfunction — 제목은 고장, 사진은 깨짐"). Otherwise `unsure: false`, `note: ""`.

## Output (one new file)
`data/cesamard/labeling/auto/evalset_{TAG}_v1.3_part{P}.json`:
```json
{"meta": {"set": "evalset", "part": "{P}", "labeler": "{TAG}", "guide_version": "v1.3", "model": "<your model id>", "created": "<ISO>"},
 "labels": {"<row_id>": {"issues": [{"main": true, "scope": "...", "part": "...", "part_text": "", "group": "...", "defect": "...", "vis": "...", "boxes": [{"img": 0, "bbox": [0.1, 0.2, 0.3, 0.4], "target": "defect"}]}], "marks": {}, "unsure": false, "note": ""}}}
```
Before finishing, run `.venv/Scripts/python.exe scripts/cesamard/check_part.py evalset {P} {TAG}` and fix every problem it prints until it says OK.

Final reply (short, 2 lines): the check_part OK line, and any image you couldn't view.
