"""Reviews every NYC bike clip in VastDB with the LLM and writes seed.json for the app.

Run on the VM (needs the vastdb SDK and the team config in the environment):
    VSS_URL=$INGRESS_URL VSS_USERNAME=$USERNAME VSS_PASSWORD=$PASSWORD python3 build_seed.py
"""

import gzip
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import vastdb
from ibis import _

import main

COLUMNS = ["source", "original_video", "filename", "reasoning_content", "object_counts",
           "segment_start_sec", "segment_end_sec", "upload_timestamp"]
REJECTED_CAPTION_CHARS = 400


def read_rows():
    session = vastdb.connect(endpoint=os.environ["S3_ENDPOINT"], access=os.environ["ACCESS_KEY"],
                             secret=os.environ["SECRET_KEY"], ssl_verify=False)
    with session.transaction() as tx:
        table = (tx.bucket(os.environ["VASTDB_BUCKET"]).schema(os.environ["VDB_SCHEMA"])
                 .table(os.environ["VDB_COLLECTION"]))
        rows = table.select(columns=COLUMNS, predicate=(_.camera_id == main.CAMERA_ID)).read_all().to_pylist()
    for r in rows:
        for k in ("segment_start_sec", "segment_end_sec"):
            try:
                r[k] = float(r[k])
            except (TypeError, ValueError):
                r[k] = None
        r["upload_timestamp"] = str(r.get("upload_timestamp") or "")
    return rows


def main_():
    rows = read_rows()
    print(f"read {len(rows)} clips for {main.CAMERA_ID}", flush=True)
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = [c for c in pool.map(main.classify, rows) if c]
    for c in results:
        c["matched_query"] = None
        if not c["blocked"]:
            c["caption"] = c["caption"][:REJECTED_CAPTION_CHARS]
    seed = {
        "incidents": {c["source"]: c for c in results},
        "candidates": len(results),
        "camera_clips": len(rows),
        "scanned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = main.SEED_FILE
    out.write_bytes(gzip.compress(json.dumps(seed, separators=(",", ":")).encode()))
    blocked = sum(c["blocked"] for c in results)
    print(f"classified {len(results)}/{len(rows)}, blocked {blocked}, wrote {out} ({out.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main_()
