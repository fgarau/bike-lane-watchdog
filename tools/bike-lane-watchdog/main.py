"""Bike Lane Watchdog: finds bike-lane obstructions in NYC cyclist footage indexed in VSS,
classifies them with a W&B-hosted LLM, and drafts NYC 311 complaints."""

import gzip
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = int(os.environ.get("PORT", "8080"))
VSS_URL = os.environ["VSS_URL"].rstrip("/")
VSS_USERNAME = os.environ["VSS_USERNAME"]
VSS_PASSWORD = os.environ["VSS_PASSWORD"]
WANDB_API_KEY = os.environ["WANDB_API_KEY"]
WANDB_PROJECT = f"{os.environ['WANDB_TEAM']}/{os.environ['WANDB_PROJECT']}"
LLM_URL = "https://api.inference.wandb.ai/v1/chat/completions"
LLM_MODEL = os.environ.get("LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

CAMERA_ID = "nyc_bike_gopro-1"
MIN_SIMILARITY = 0.3
SCAN_QUERIES = [
    "vehicle blocking the bike lane",
    "delivery truck parked in the bike lane",
    "car double parked partially blocking the lane",
    "van stopped in the bike lane",
    "cyclist swerving around an obstacle into traffic",
    "construction cones or barriers blocking the bike lane",
    "taxi stopped picking up passengers in the bike lane",
    "bus blocking the bike lane at a stop",
]

HERE = Path(__file__).parent
CACHE_FILE = Path(os.environ.get("CACHE_FILE", "/tmp/bike-lane-cache.json"))
SEED_FILE = HERE / "seed.json.gz"

CLASSIFY_PROMPT = """You review captions written by a video model for 5-second clips from a cyclist's \
helmet/handlebar camera riding through New York City. You also get YOLO object counts for the clip.

Decide whether the clip shows the bike lane, or the cyclist's path of travel, being obstructed by \
something that should not be there (parked/stopped vehicles, delivery trucks, double parking, cones, \
construction, debris, pedestrians standing in the lane). A cyclist simply riding in a clear lane is NOT blocked.

Return only JSON with these keys:
{
  "blocked": true|false,
  "obstruction": "vehicle"|"construction"|"pedestrian"|"debris"|"other"|"none",
  "vehicle_type": "delivery truck"|"box truck"|"van"|"car"|"taxi"|"bus"|"pickup"|"other"|null,
  "company": brand/agency name only if the caption names one (e.g. "USPS", "UPS", "FedEx", "Amazon", "NYPD"), else null,
  "vehicle_description": short description with color and markings, or null,
  "severity": integer 0-5 (0 not blocked, 1 slight encroachment, 3 lane fully blocked, 5 cyclist forced into moving traffic or near-miss),
  "cyclist_forced_out": true|false,
  "headline": title of at most 8 words,
  "evidence": the exact sentence from the caption that supports your decision
}"""

REPORT_PROMPT = """You draft NYC 311 service requests of complaint type "Blocked Bike Lane" from \
evidence extracted from cyclist camera footage. Write a concise, factual, polite complaint that a \
resident could paste into the 311 portal. Use these sections in plain text:

Complaint Type: Blocked Bike Lane
Descriptor: (e.g. "Vehicle", "Construction", "Other")
Vehicle: (type, color, company/markings if known, plate "not legible" unless given)
Date/Time Observed:
Location: write "[add street / cross street]" since GPS is not in the footage metadata
Description: 2-4 sentences describing what the footage shows and the safety impact on cyclists.
Evidence: video clip reference

Do not invent details (plates, street names, injuries) that are not in the evidence."""

state_lock = threading.Lock()
state = {"incidents": {}, "candidates": 0, "scanned_at": None, "scanning": False,
         "camera_clips": None, "error": None}
token_lock = threading.Lock()
_token = {"value": None}


def http_json(url, payload=None, headers=None, timeout=60):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def vss_token(force=False):
    with token_lock:
        if force or not _token["value"]:
            resp = http_json(f"{VSS_URL}/api/v1/auth/login",
                             {"username": VSS_USERNAME, "password": VSS_PASSWORD})
            _token["value"] = resp["access_token"]
        return _token["value"]


def vss(path, payload=None, timeout=150):
    for attempt in range(2):
        try:
            return http_json(f"{VSS_URL}{path}", payload,
                             {"Authorization": f"Bearer {vss_token(force=attempt > 0)}"}, timeout)
        except urllib.error.HTTPError as e:
            if e.code != 401 or attempt:
                raise


def llm(messages, json_mode=False, max_tokens=600):
    payload = {"model": LLM_MODEL, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    resp = http_json(LLM_URL, payload, {"Authorization": f"Bearer {WANDB_API_KEY}",
                                        "OpenAI-Project": WANDB_PROJECT})
    return resp["choices"][0]["message"]["content"]


def search(query):
    resp = vss("/api/v1/search", {
        "query": query, "top_k": 40, "llm_top_n": 1, "min_similarity": MIN_SIMILARITY,
        "metadata_filters": {"camera_id": CAMERA_ID},
    })
    return resp.get("results", [])


def parse_counts(raw):
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}


def normalize_company(name):
    if not isinstance(name, str):
        return None
    name = re.sub(r"[\s,]+(co|inc|corp|company|llc)\.?$", "", name.strip(), flags=re.I).strip()
    if name.lower() in ("", "null", "none", "unknown", "n/a"):
        return None
    return name


def classify(row):
    counts = parse_counts(row.get("object_counts"))
    user = f"Caption:\n{row.get('reasoning_content', '')}\n\nYOLO object counts: {json.dumps(counts)}"
    for _ in range(2):
        try:
            out = json.loads(llm([{"role": "system", "content": CLASSIFY_PROMPT},
                                  {"role": "user", "content": user}], json_mode=True))
            break
        except (ValueError, urllib.error.URLError):
            out = None
    if not out:
        return None
    try:
        severity = max(0, min(5, int(out.get("severity") or 0)))
    except (TypeError, ValueError):
        severity = 0
    company = normalize_company(out.get("company"))
    return {
        "source": row["source"],
        "original_video": row.get("original_video"),
        "filename": row.get("filename"),
        "segment_start_sec": row.get("segment_start_sec"),
        "segment_end_sec": row.get("segment_end_sec"),
        "upload_timestamp": row.get("upload_timestamp"),
        "similarity": round(float(row.get("similarity_score") or 0), 3),
        "caption": row.get("reasoning_content", ""),
        "object_counts": counts,
        "matched_query": row.get("_query"),
        "blocked": bool(out.get("blocked")) and severity > 0,
        "obstruction": out.get("obstruction") or "none",
        "vehicle_type": out.get("vehicle_type"),
        "company": company,
        "vehicle_description": out.get("vehicle_description"),
        "severity": severity,
        "cyclist_forced_out": bool(out.get("cyclist_forced_out")),
        "headline": out.get("headline") or "Bike lane obstruction",
        "evidence": out.get("evidence") or "",
    }


def scan(queries):
    rows = {}
    errors = []
    # The VSS backend falls over under concurrent searches, so query one at a time.
    for q in queries:
        try:
            results = search(q)
        except Exception as e:
            errors.append(f"{q!r}: {type(e).__name__}")
            continue
        for r in results:
            if r["source"] not in rows or r["similarity_score"] > rows[r["source"]]["similarity_score"]:
                rows[r["source"]] = {**r, "_query": q}
    if errors and not rows:
        raise RuntimeError("all searches failed: " + "; ".join(errors))
    with state_lock:
        known = set(state["incidents"])
    todo = [r for s, r in rows.items() if s not in known]
    with ThreadPoolExecutor(max_workers=12) as pool:
        classified = [c for c in pool.map(classify, todo) if c]
    with state_lock:
        for c in classified:
            state["incidents"][c["source"]] = c
        state["candidates"] = len(state["incidents"])
    return [state["incidents"][s] for s in rows if s in state["incidents"]]


def camera_clip_count():
    try:
        stats = vss("/api/v1/dashboard/stats?scope=all")
        for item in stats.get("metadata", {}).get("camera_id", []):
            if item.get("label") == CAMERA_ID:
                return item.get("count")
    except Exception:
        pass
    return None


def save_cache():
    with state_lock:
        data = {k: state[k] for k in ("incidents", "candidates", "scanned_at", "camera_clips")}
    try:
        CACHE_FILE.write_text(json.dumps(data))
    except OSError:
        pass


def load_cache():
    for path in (CACHE_FILE, SEED_FILE):
        try:
            raw = path.read_bytes()
            data = json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)
        except (OSError, ValueError):
            continue
        with state_lock:
            state.update(data)
        return


def full_scan():
    with state_lock:
        if state["scanning"]:
            return
        state["scanning"] = True
        state["error"] = None
    try:
        scan(SCAN_QUERIES)
        clips = camera_clip_count()
        with state_lock:
            state["scanned_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if clips:
                state["camera_clips"] = clips
        save_cache()
    except Exception as e:
        with state_lock:
            state["error"] = f"{type(e).__name__}: {e}"
    finally:
        with state_lock:
            state["scanning"] = False


def draft_report(source):
    with state_lock:
        inc = state["incidents"].get(source)
    if not inc:
        raise KeyError(source)
    evidence = {k: inc[k] for k in ("headline", "obstruction", "vehicle_type", "company",
                                    "vehicle_description", "severity", "cyclist_forced_out",
                                    "caption", "upload_timestamp", "filename",
                                    "segment_start_sec", "segment_end_sec")}
    return llm([{"role": "system", "content": REPORT_PROMPT},
                {"role": "user", "content": json.dumps(evidence, indent=2)}], max_tokens=700)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if not self.path.startswith(("/health", "/video")):
            super().log_message(fmt, *args)

    def send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(url.query)
        if url.path == "/health":
            return self.send_json({"ok": True})
        if url.path in ("/", "/index.html"):
            body = (HERE / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)
        if url.path == "/api/incidents":
            with state_lock:
                return self.send_json({
                    "incidents": sorted(state["incidents"].values(),
                                        key=lambda i: (-i["severity"], -i["similarity"])),
                    "candidates": state["candidates"], "camera_clips": state["camera_clips"],
                    "scanned_at": state["scanned_at"], "scanning": state["scanning"],
                    "error": state["error"], "camera_id": CAMERA_ID, "model": LLM_MODEL,
                })
        if url.path == "/api/hunt":
            q = (qs.get("q") or [""])[0].strip()
            if not q:
                return self.send_json({"error": "missing q"}, 400)
            try:
                found = scan([q])
                save_cache()
            except Exception as e:
                return self.send_json({"error": str(e)}, 502)
            return self.send_json({"query": q, "results": sorted(
                found, key=lambda i: (-i["severity"], -i["similarity"]))})
        if url.path == "/video":
            return self.proxy_video((qs.get("source") or [""])[0])
        self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if url.path == "/api/report":
            try:
                return self.send_json({"report": draft_report(body.get("source", ""))})
            except KeyError:
                return self.send_json({"error": "unknown incident"}, 404)
            except Exception as e:
                return self.send_json({"error": str(e)}, 502)
        if url.path == "/api/rescan":
            threading.Thread(target=full_scan, daemon=True).start()
            return self.send_json({"started": True})
        self.send_json({"error": "not found"}, 404)

    def proxy_video(self, source):
        if not re.fullmatch(r"s3://[\w.\-]+/[\w./\-]+\.mp4", source):
            return self.send_json({"error": "bad source"}, 400)
        params = urllib.parse.urlencode({"source": source, "token": vss_token()})
        req = urllib.request.Request(f"{VSS_URL}/api/v1/videos/stream?{params}")
        if self.headers.get("Range"):
            req.add_header("Range", self.headers["Range"])
        try:
            upstream = urllib.request.urlopen(req, timeout=60)
        except urllib.error.HTTPError as e:
            if e.code == 401:
                vss_token(force=True)
            return self.send_json({"error": f"upstream {e.code}"}, 502)
        with upstream:
            self.send_response(upstream.status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "private, max-age=3600")
            for h in ("Content-Length", "Content-Range"):
                if upstream.headers.get(h):
                    self.send_header(h, upstream.headers[h])
            self.end_headers()
            try:
                while chunk := upstream.read(64 * 1024):
                    self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass


if __name__ == "__main__":
    load_cache()
    if not state["incidents"]:
        threading.Thread(target=full_scan, daemon=True).start()
    print(f"Bike Lane Watchdog on :{PORT} (model {LLM_MODEL})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
