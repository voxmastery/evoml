"""Push a compact live snapshot of the experiment to the public site.

Reads the local dashboard API plus the genome/hall-of-fame meta rows, bundles
them into one JSON document, and POSTs it to the site's ingest endpoint.
Runs forever, one push every INTERVAL seconds. The ingest key comes from the
environment, never from the repo.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
import urllib.request

API = os.environ.get("EVOML_LOCAL_API", "http://127.0.0.1:8765")
TARGET = os.environ.get("EVOML_INGEST_URL", "https://evoml-lab.higgsfield.app/api/ingest")
KEY = os.environ.get("EVOML_INGEST_KEY", "")
INTERVAL = float(os.environ.get("EVOML_PUSH_INTERVAL", "20"))
DB_PATH = os.environ.get("DB_PATH", "data/memescalp.db")
CURVE_POINTS = 240
ARM_FIELDS = ("accuracy", "resolved", "called", "skipped", "wins", "losses",
              "brier", "capital", "kelly_capital")
LEDGER_FIELDS = ("ts", "arm", "symbol", "direction", "confidence", "correct",
                 "status", "return_pct")


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{API}{path}", timeout=20) as r:
        return json.load(r)


def meta(key: str):
    con = sqlite3.connect(DB_PATH)
    try:
        row = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    finally:
        con.close()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except (TypeError, ValueError):
        return row[0]


def thin(points: list, n: int) -> list:
    if len(points) <= n:
        return points
    step = len(points) / n
    return [points[int(i * step)] for i in range(n - 1)] + [points[-1]]


def build() -> dict:
    summary = get("/api/predict/summary")
    curves = get("/api/predict/curves")
    ledger = get("/api/predict/ledger")["ledger"]
    evolution = get("/api/evolution")
    arms = {name: {k: v for k, v in arm.items() if k in ARM_FIELDS}
            for name, arm in summary["arms"].items()}
    return {
        "ts": time.time(),
        "summary": {"arms": arms, "pass_fail": summary["pass_fail"],
                    "subject": summary.get("subject")},
        "evolution": evolution,
        "genome": meta("ml_genome"),
        "hall": meta("ml_hall_of_fame"),
        "curves": {arm: thin(curves[arm]["accuracy"], CURVE_POINTS)
                   for arm in ("ml", "random", "hedge", "math") if arm in curves},
        "ledger": [{k: r.get(k) for k in LEDGER_FIELDS} for r in ledger[:12]],
    }


def push(doc: dict) -> str:
    body = json.dumps(doc).encode()
    req = urllib.request.Request(
        TARGET, data=body, method="POST",
        headers={"Content-Type": "application/json", "x-evoml-key": KEY,
                 "User-Agent": "evoml-push/1.0 (+https://github.com/voxmastery/evoml)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode()[:120]


def main() -> None:
    if not KEY:
        print("EVOML_INGEST_KEY missing", file=sys.stderr)
        sys.exit(2)
    once = "--once" in sys.argv
    while True:
        try:
            print(time.strftime("%H:%M:%S"), push(build()), flush=True)
        except Exception as exc:  # noqa: BLE001 - keep the pusher alive
            print(time.strftime("%H:%M:%S"), "push failed:", exc,
                  file=sys.stderr, flush=True)
        if once:
            return
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
