You translate about 30 Amazon India complaint reviews into Korean for a human labeler who does not read English. Text only: do not open images, label files, or anything in `data/cesamard/labeling/auto/` other than your own ID file and your own output file. Do not modify existing files.

Project root: `C:\Users\user\Documents\멀티모달 프로젝트`. Python: `.venv/Scripts/python.exe`.

## Input
Review IDs: `data/cesamard/labeling/auto/main_check_ko_part{P}_ids.txt`. Items are in `data/cesamard/labeling/main.json` (a list; use only items whose `row_id` is in your ID file): `title`, `text` (English, often Indian English, sometimes Hinglish).

## Translation
Translate each review's `title` and `text` into natural Korean (해요체), literally: no interpretation, no labels, no added explanations, no summarizing. Keep line breaks as `\n`. Mark genuinely unclear source phrases as `[원문 불명확: '<original>']`. Hindi/Hinglish words: translate the meaning. Omit personal data (names, addresses, phone numbers, order numbers) and write `(작성자 개인정보 — 번역 생략)` in its place.

## Output (one new file)
`data/cesamard/labeling/auto/main_check_ko_part{P}.json`: `{"<row_id>": {"title": "...", "text": "..."}, ...}` (UTF-8, `ensure_ascii=False`).

Before finishing, check with Python that every ID in your ID file is present with a non-empty `title` and `text` and that there are no extra IDs.

Final reply (one line): the output path and the number of reviews translated.
