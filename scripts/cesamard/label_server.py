"""Local server for the defect labeling tool.

Serves the tool page, the review set, the full-res images, and saves labels
straight into data/cesamard/labeling/. Binds to 127.0.0.1 only.

Usage:
  python label_server.py --set round2          # http://127.0.0.1:8765
  python label_server.py                       # pilot30 (first batch, now the auto-test set)
  python label_server.py --port 8766 --labels <path>   # separate labels file (testing)
  python label_server.py --set gold90 --drafts data/cesamard/labeling/auto/gold90_claude_cmm_v1.3.json  # check-the-draft mode
  python label_server.py --set blind60 --page quick   # blind sample: no model answers, quick buttons
"""
import argparse
import json
import os
import shutil
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data" / "cesamard"
LAB = DS / "labeling"
IMG_DIR = DS / "images_hr"
TOOL_DIR = Path(__file__).resolve().parent / "label_tool"
MAX_BODY = 20 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    set_name: str
    items_path: Path
    labels_path: Path
    drafts_path: Path | None = None
    page: Path = TOOL_DIR / "index.html"

    def _send(self, code, body, ctype, cache=False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            return self._send(200, self.page.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/set":
            return self._json(200, {"set": self.set_name})
        if path == "/api/items":
            return self._send(200, self.items_path.read_bytes(), "application/json; charset=utf-8")
        if path == "/api/labels":
            body = self.labels_path.read_bytes() if self.labels_path.exists() else b"{}"
            return self._send(200, body, "application/json; charset=utf-8")
        if path == "/api/drafts":
            p = self.drafts_path
            body = p.read_bytes() if p and p.exists() else b"{}"
            return self._send(200, body, "application/json; charset=utf-8")
        if path.startswith("/images_hr/"):
            name = path[len("/images_hr/"):]
            f = IMG_DIR / name
            if "/" in name or "\\" in name or ".." in name or not f.is_file():
                return self._json(404, {"error": "not found"})
            return self._send(200, f.read_bytes(), "image/jpeg", cache=True)
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/labels":
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0 or n > MAX_BODY:
            return self._json(413, {"error": "bad size"})
        try:
            payload = json.loads(self.rfile.read(n))
            assert isinstance(payload, dict) and isinstance(payload.get("labels"), dict)
        except Exception:
            return self._json(400, {"error": "invalid json"})
        meta = payload.setdefault("meta", {})
        # a tab still open on another set must never overwrite this set's labels
        if meta.get("set") != self.set_name:
            return self._json(409, {"error": f"labels are for a different set (serving {self.set_name})"})
        # optimistic concurrency: a tab holding an older revision (e.g. a second window
        # left open) must not overwrite newer labels saved from another window
        current = 0
        if self.labels_path.exists():
            current = json.loads(self.labels_path.read_text(encoding="utf-8")).get("meta", {}).get("rev", 0)
        if meta.get("base_rev", 0) != current:
            return self._json(409, {"error": "stale", "rev": current})
        meta["rev"] = current + 1
        meta.pop("base_rev", None)
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        if self.labels_path.exists():
            shutil.copyfile(self.labels_path, self.labels_path.with_suffix(".bak.json"))
        tmp = self.labels_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.labels_path)
        return self._json(200, {"ok": True, "saved": time.strftime("%H:%M:%S"), "rev": meta["rev"]})

    def log_message(self, fmt, *args):
        if "/images_hr/" not in self.path:
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="pilot30")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--drafts", type=Path, default=None,
                    help="label file from the auto-labeler; its main issue pre-fills reviews the human hasn't saved yet")
    ap.add_argument("--page", choices=["full", "quick"], default="full",
                    help="quick = blind sample page (group + what the photos show, no model answers)")
    args = ap.parse_args()
    Handler.page = TOOL_DIR / ("quick.html" if args.page == "quick" else "index.html")
    Handler.set_name = args.set
    Handler.items_path = LAB / f"{args.set}.json"
    Handler.labels_path = args.labels or LAB / f"{args.set}_labels.json"
    Handler.drafts_path = args.drafts
    print(f"items  : {Handler.items_path}")
    print(f"labels : {Handler.labels_path}")
    print(f"drafts : {Handler.drafts_path or '-'}")
    print(f"open   : http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
