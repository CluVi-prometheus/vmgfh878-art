"""Re-download CESAMARD review images at full resolution.

The dataset ships Amazon thumbnail URLs (e.g. `51RV5kC6QKL._SY88.jpg`, ~88px).
Dropping the resize suffix (`51RV5kC6QKL.jpg`) returns the original upload.

Every image of every review is fetched (the old copy kept only the first image
per review), deduplicated by Amazon image key. A low-resolution original in
`images/` is deleted only after its full-resolution replacement has downloaded,
decoded and cleared the size check, so a failed fetch never loses data.

Usage:
  python redownload_hr.py            # full run, then delete replaced originals
  python redownload_hr.py --limit 12 # smoke test, never deletes
"""
import argparse
import asyncio
import csv
import hashlib
import io
import json
import re
import ssl
import statistics
import sys
import time
from pathlib import Path

import aiohttp
import certifi
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data" / "cesamard"
SRC_CSV = DS / "CESAMARD-Aspect_with_local_images.csv"
OUT_DIR = DS / "images_hr"
OLD_DIR = DS / "images"
MANIFEST = DS / "images_hr_manifest.csv"
SUMMARY = DS / "images_hr_summary.json"

MIN_SIDE_OK = 300  # anything smaller is still a thumbnail
CONCURRENCY = 16
RETRIES = 3
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Referer": "https://www.amazon.in/",
}
FALLBACK_HOST = "https://m.media-amazon.com/images/I/"

csv.field_size_limit(10**9)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def split_url(url):
    """'.../I/51RV5kC6QKL._SY88.jpg' -> (prefix '.../I/', key '51RV5kC6QKL', ext 'jpg')."""
    prefix, _, base = url.rpartition("/")
    key = base.split(".", 1)[0]
    ext = base.rsplit(".", 1)[-1].lower() if "." in base else "jpg"
    if ext not in {"jpg", "jpeg", "png", "gif", "webp"}:
        ext = "jpg"
    return prefix + "/", key, ext


def safe_name(key, ext):
    return re.sub(r"[^A-Za-z0-9+_-]", "_", key) + "." + ext


def load_jobs(limit=None):
    rows = list(csv.DictReader(open(SRC_CSV, encoding="utf-8-sig", newline="")))
    per_image, unique = [], {}
    for r in rows:
        urls = [u.strip() for u in (r.get("Image_urls") or "").split(";") if u.strip()]
        for idx, url in enumerate(urls):
            prefix, key, ext = split_url(url)
            per_image.append({"row_id": r["row_id"], "img_idx": idx, "orig_url": url, "key": key})
            if key not in unique:
                unique[key] = {"key": key, "ext": ext,
                               "candidates": [f"{prefix}{key}.{ext}", f"{FALLBACK_HOST}{key}.{ext}"],
                               "file": OUT_DIR / safe_name(key, ext)}
    jobs = list(unique.values())
    if limit:
        jobs = jobs[:limit]
        keep = {j["key"] for j in jobs}
        per_image = [p for p in per_image if p["key"] in keep]
    return rows, per_image, jobs


def inspect_bytes(content):
    with Image.open(io.BytesIO(content)) as im:
        im.verify()
    with Image.open(io.BytesIO(content)) as im:
        return im.size


def inspect_file(path):
    try:
        return inspect_bytes(path.read_bytes())
    except Exception:
        return None


async def fetch(session, job, sem):
    if job["file"].exists():
        size = inspect_file(job["file"])
        if size and min(size) >= MIN_SIDE_OK:
            return {**job, "status": "ok_cached", "url_used": None, "w": size[0], "h": size[1],
                    "bytes": job["file"].stat().st_size}
    last = "unknown"
    async with sem:
        for url in job["candidates"]:
            for attempt in range(RETRIES):
                try:
                    async with session.get(url, headers=HEADERS,
                                           timeout=aiohttp.ClientTimeout(total=45)) as resp:
                        content = await resp.read()
                        if resp.status != 200:
                            last = f"http_{resp.status}"
                            if resp.status in (403, 404):
                                break  # retrying the same host will not help
                            await asyncio.sleep(1.5 * (attempt + 1))
                            continue
                    try:
                        w, h = inspect_bytes(content)
                    except Exception:
                        last = "decode_error"
                        break
                    job["file"].write_bytes(content)
                    status = "ok" if min(w, h) >= MIN_SIDE_OK else "still_small"
                    return {**job, "status": status, "url_used": url, "w": w, "h": h,
                            "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                except asyncio.TimeoutError:
                    last = "timeout"
                    await asyncio.sleep(1.5 * (attempt + 1))
                except Exception as e:
                    last = f"error:{type(e).__name__}"
                    await asyncio.sleep(1.5 * (attempt + 1))
    return {**job, "status": last, "url_used": None, "w": None, "h": None, "bytes": 0}


async def run(jobs):
    sem = asyncio.Semaphore(CONCURRENCY)
    ctx = ssl.create_default_context(cafile=certifi.where())
    conn = aiohttp.TCPConnector(limit=CONCURRENCY, ssl=ctx)
    results = []
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [fetch(session, j, sem) for j in jobs]
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            results.append(await coro)
            if i % 250 == 0 or i == len(tasks):
                ok = sum(r["status"] in ("ok", "ok_cached") for r in results)
                log(f"progress {i}/{len(tasks)}  ok={ok}")
    return results


def delete_replaced_originals(rows, by_key):
    """Old copies hold only each review's first image; delete one only if replaced."""
    deleted = kept = missing = 0
    kept_rows = []
    for r in rows:
        urls = [u.strip() for u in (r.get("Image_urls") or "").split(";") if u.strip()]
        old = OLD_DIR / f"{r['row_id']}.jpg"
        if not old.exists():
            missing += 1
            continue
        res = by_key.get(split_url(urls[0])[1]) if urls else None
        if res and res["status"] in ("ok", "ok_cached") and res["file"].exists():
            old.unlink()
            deleted += 1
        else:
            kept += 1
            kept_rows.append(r["row_id"])
    if OLD_DIR.exists() and not any(OLD_DIR.iterdir()):
        OLD_DIR.rmdir()
    return {"deleted": deleted, "kept_no_replacement": kept, "already_missing": missing,
            "kept_row_ids": kept_rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, per_image, jobs = load_jobs(args.limit)
    log(f"reviews={len(rows)} image_refs={len(per_image)} unique_images={len(jobs)}")

    t0 = time.time()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    results = asyncio.run(run(jobs))
    by_key = {r["key"]: r for r in results}

    with open(MANIFEST if not args.limit else DS / "images_hr_manifest_TEST.csv",
              "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "img_idx", "key", "orig_url", "url_used", "local_path",
                    "status", "width", "height", "bytes"])
        for p in per_image:
            r = by_key.get(p["key"], {})
            ok = r.get("status") in ("ok", "ok_cached", "still_small")
            w.writerow([p["row_id"], p["img_idx"], p["key"], p["orig_url"], r.get("url_used"),
                        f"images_hr/{r['file'].name}" if ok else "",
                        r.get("status"), r.get("w"), r.get("h"), r.get("bytes")])

    status_counts = {}
    for r in results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    good = [r for r in results if r["status"] in ("ok", "ok_cached")]
    summary = {
        "unique_images": len(results), "status_counts": status_counts,
        "median_w": statistics.median([r["w"] for r in good]) if good else None,
        "median_h": statistics.median([r["h"] for r in good]) if good else None,
        "min_side_min": min(min(r["w"], r["h"]) for r in good) if good else None,
        "total_gb": round(sum(r["bytes"] or 0 for r in results) / 1e9, 3),
        "seconds": round(time.time() - t0, 1),
    }
    if args.limit:
        summary["note"] = "smoke test - originals untouched"
    else:
        summary["deletion"] = delete_replaced_originals(rows, by_key)
    json.dump(summary, open(SUMMARY if not args.limit else DS / "images_hr_summary_TEST.json",
                            "w", encoding="utf-8"), indent=2, default=str)
    log("summary " + json.dumps({k: v for k, v in summary.items() if k != "deletion"}, default=str))
    if "deletion" in summary:
        d = summary["deletion"]
        log(f"deleted={d['deleted']} kept_no_replacement={d['kept_no_replacement']} "
            f"already_missing={d['already_missing']}")
    log("===== HR DOWNLOAD ALL DONE =====")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        log("===== HR DOWNLOAD FAILED =====")
        sys.exit(1)
