"""Append-only SQLite log. Rows are inserted, never updated or deleted."""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from .models import Decision, FeeBreakdown, Fill, Position

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    mint TEXT NOT NULL,
    symbol TEXT NOT NULL,
    price_usd REAL NOT NULL,
    source TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prices_mint_ts ON prices(mint, ts);

CREATE TABLE IF NOT EXISTS liquidity (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    mint TEXT NOT NULL,
    liquidity_usd REAL NOT NULL,
    volume_h24 REAL NOT NULL DEFAULT 0,
    price_change_h1 REAL NOT NULL DEFAULT 0,
    price_change_h24 REAL NOT NULL DEFAULT 0,
    symbol TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_liquidity_mint_ts ON liquidity(mint, ts);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    arm TEXT NOT NULL,
    window_start REAL NOT NULL,
    window_end REAL NOT NULL,
    mint TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    model TEXT NOT NULL,
    backend TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    arm TEXT NOT NULL,
    trade_id TEXT NOT NULL,
    side TEXT NOT NULL,
    mint TEXT NOT NULL,
    symbol TEXT NOT NULL,
    feed_price REAL NOT NULL,
    exec_price REAL NOT NULL,
    size_usd REAL NOT NULL,
    token_qty REAL NOT NULL,
    fee_lp REAL NOT NULL,
    fee_slippage REAL NOT NULL,
    fee_priority REAL NOT NULL,
    fee_tds REAL NOT NULL,
    realized_pnl REAL,
    balance_after REAL NOT NULL,
    stop_mode TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_fills_arm_ts ON fills(arm, ts);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS catalog (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    mint TEXT NOT NULL,
    symbol TEXT NOT NULL,
    price_usd REAL NOT NULL,
    liquidity_usd REAL NOT NULL,
    volume_h24 REAL NOT NULL,
    price_change_h1 REAL NOT NULL,
    price_change_h24 REAL NOT NULL,
    volatility_pct REAL NOT NULL,
    est_cost_pct REAL NOT NULL,
    score REAL NOT NULL,
    rank INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_catalog_ts ON catalog(ts);

CREATE TABLE IF NOT EXISTS flow (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    mint TEXT NOT NULL,
    buys_m5 INTEGER NOT NULL,
    sells_m5 INTEGER NOT NULL,
    vol_m5 REAL NOT NULL,
    chg_m5 REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_flow_mint_ts ON flow(mint, ts);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    arm TEXT NOT NULL,
    mint TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    confidence REAL NOT NULL,
    horizon_end REAL NOT NULL,
    price_at REAL NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    model TEXT NOT NULL,
    backend TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_arm ON predictions(arm, ts);

CREATE TABLE IF NOT EXISTS resolutions (
    id INTEGER PRIMARY KEY,
    prediction_id INTEGER NOT NULL UNIQUE,
    ts REAL NOT NULL,
    price_end REAL NOT NULL,
    return_pct REAL NOT NULL,
    correct INTEGER NOT NULL,
    status TEXT NOT NULL  -- resolved | void (no price data in time)
);

CREATE TABLE IF NOT EXISTS tsfm (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    mint TEXT NOT NULL,
    pred_ret REAL NOT NULL,
    spread REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tsfm_mint_ts ON tsfm(mint, ts);

CREATE TABLE IF NOT EXISTS evolution (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    generation INTEGER NOT NULL,
    champion TEXT NOT NULL,
    genome TEXT NOT NULL,
    scores TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS playbooks (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    arm TEXT NOT NULL,
    mint TEXT NOT NULL,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    confidence REAL NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    model TEXT NOT NULL,
    backend TEXT NOT NULL
);
"""


class Database:
    def __init__(self, path: Path | str):
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _execute(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def _query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # --- writes (append-only) ------------------------------------------------

    def insert_price(self, ts: float, mint: str, symbol: str, price_usd: float,
                     source: str) -> None:
        self._execute(
            "INSERT INTO prices (ts, mint, symbol, price_usd, source) VALUES (?,?,?,?,?)",
            (ts, mint, symbol, price_usd, source),
        )

    def insert_liquidity(self, ts: float, mint: str, symbol: str, liquidity_usd: float,
                         volume_h24: float, price_change_h1: float,
                         price_change_h24: float) -> None:
        self._execute(
            "INSERT INTO liquidity (ts, mint, symbol, liquidity_usd, volume_h24,"
            " price_change_h1, price_change_h24) VALUES (?,?,?,?,?,?,?)",
            (ts, mint, symbol, liquidity_usd, volume_h24, price_change_h1,
             price_change_h24),
        )

    def insert_decision(self, d: Decision) -> None:
        self._execute(
            "INSERT INTO decisions (ts, arm, window_start, window_end, mint, symbol,"
            " direction, prompt, response, model, backend)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (d.ts, d.arm, d.window_start, d.window_end, d.mint, d.symbol,
             d.direction, d.prompt, d.response, d.model, d.backend),
        )

    def insert_fill(self, f: Fill) -> None:
        self._execute(
            "INSERT INTO fills (ts, arm, trade_id, side, mint, symbol, feed_price,"
            " exec_price, size_usd, token_qty, fee_lp, fee_slippage, fee_priority,"
            " fee_tds, realized_pnl, balance_after, stop_mode, note)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f.ts, f.arm, f.trade_id, f.side, f.mint, f.symbol, f.feed_price,
             f.exec_price, f.size_usd, f.token_qty, f.fees.lp, f.fees.slippage,
             f.fees.priority, f.fees.tds, f.realized_pnl, f.balance_after,
             f.stop_mode, f.note),
        )

    def insert_flow(self, ts: float, mint: str, buys_m5: int, sells_m5: int,
                    vol_m5: float, chg_m5: float) -> None:
        self._execute(
            "INSERT INTO flow (ts, mint, buys_m5, sells_m5, vol_m5, chg_m5)"
            " VALUES (?,?,?,?,?,?)",
            (ts, mint, buys_m5, sells_m5, vol_m5, chg_m5),
        )

    def insert_catalog(self, ts: float, entries) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT INTO catalog (ts, mint, symbol, price_usd, liquidity_usd,"
                " volume_h24, price_change_h1, price_change_h24, volatility_pct,"
                " est_cost_pct, score, rank) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [(ts, e.mint, e.symbol, e.price_usd, e.liquidity_usd,
                  e.volume_h24, e.price_change_h1, e.price_change_h24,
                  e.volatility_pct, e.est_cost_pct, e.score, e.rank)
                 for e in entries],
            )
            self._conn.commit()

    def latest_catalog(self) -> list[dict]:
        rows = self._query(
            "SELECT * FROM catalog WHERE ts=(SELECT MAX(ts) FROM catalog)"
            " ORDER BY rank"
        )
        return [dict(r) for r in rows]

    def insert_analysis(self, a) -> None:
        self._execute(
            "INSERT INTO analyses (ts, arm, mint, symbol, action, confidence,"
            " prompt, response, model, backend) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (a.ts, a.arm, a.mint, a.symbol, a.action, a.confidence,
             a.prompt, a.response, a.model, a.backend),
        )

    def analyses(self, limit: int = 20) -> list[dict]:
        rows = self._query(
            "SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in rows]

    def insert_prediction(self, p) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO predictions (ts, arm, mint, symbol, direction,"
                " confidence, horizon_end, price_at, prompt, response, model,"
                " backend) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (p.ts, p.arm, p.mint, p.symbol, p.direction, p.confidence,
                 p.horizon_end, p.price_at, p.prompt, p.response, p.model,
                 p.backend),
            )
            self._conn.commit()
            return cur.lastrowid

    def insert_resolution(self, prediction_id: int, ts: float,
                          price_end: float, return_pct: float, correct: bool,
                          status: str) -> None:
        self._execute(
            "INSERT INTO resolutions (prediction_id, ts, price_end, return_pct,"
            " correct, status) VALUES (?,?,?,?,?,?)",
            (prediction_id, ts, price_end, return_pct, int(correct), status),
        )

    def due_unresolved(self, now: float) -> list[dict]:
        rows = self._query(
            "SELECT p.* FROM predictions p"
            " LEFT JOIN resolutions r ON r.prediction_id = p.id"
            " WHERE r.id IS NULL AND p.horizon_end <= ?"
            " AND p.direction IN ('UP','DOWN')"
            " ORDER BY p.id",
            (now,),
        )
        return [dict(r) for r in rows]

    def price_at_or_after(self, mint: str, ts: float) -> tuple[float, float] | None:
        rows = self._query(
            "SELECT ts, price_usd FROM prices WHERE mint=? AND ts>=?"
            " ORDER BY ts LIMIT 1",
            (mint, ts),
        )
        return (rows[0]["ts"], rows[0]["price_usd"]) if rows else None

    def resolved_predictions(self, arm: str) -> list[dict]:
        rows = self._query(
            "SELECT p.id AS prediction_id, p.ts, p.arm, p.mint, p.symbol,"
            " p.direction, p.confidence,"
            " p.price_at, r.ts AS resolved_ts, r.price_end, r.return_pct,"
            " r.correct, r.status"
            " FROM predictions p JOIN resolutions r ON r.prediction_id = p.id"
            " WHERE p.arm=? ORDER BY r.ts",
            (arm,),
        )
        return [dict(r) for r in rows]

    def prediction_ledger(self, limit: int = 60) -> list[dict]:
        rows = self._query(
            "SELECT p.id, p.ts, p.arm, p.symbol, p.mint, p.direction,"
            " p.confidence, p.horizon_end, p.price_at, r.price_end,"
            " r.return_pct, r.correct, r.status"
            " FROM predictions p LEFT JOIN resolutions r ON r.prediction_id=p.id"
            " ORDER BY p.id DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]

    def insert_tsfm(self, ts: float, mint: str, pred_ret: float,
                    spread: float) -> None:
        self._execute(
            "INSERT INTO tsfm (ts, mint, pred_ret, spread) VALUES (?,?,?,?)",
            (ts, mint, pred_ret, spread),
        )

    def tsfm_before(self, mint: str, ts: float,
                    max_age_s: float = 420.0) -> tuple[float, float] | None:
        rows = self._query(
            "SELECT pred_ret, spread FROM tsfm WHERE mint=? AND ts<=? AND ts>=?"
            " ORDER BY ts DESC LIMIT 1",
            (mint, ts, ts - max_age_s),
        )
        return (rows[0]["pred_ret"], rows[0]["spread"]) if rows else None

    def insert_evolution(self, ts: float, generation: int, champion: str,
                         genome: str, scores: str) -> None:
        self._execute(
            "INSERT INTO evolution (ts, generation, champion, genome, scores)"
            " VALUES (?,?,?,?,?)",
            (ts, generation, champion, genome, scores),
        )

    def evolution_log(self, limit: int = 30) -> list[dict]:
        rows = self._query(
            "SELECT * FROM evolution ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in rows]

    def insert_playbook(self, ts: float, prompt: str, response: str) -> None:
        self._execute(
            "INSERT INTO playbooks (ts, prompt, response) VALUES (?,?,?)",
            (ts, prompt, response),
        )

    def latest_playbook(self) -> dict | None:
        rows = self._query("SELECT * FROM playbooks ORDER BY id DESC LIMIT 1")
        return dict(rows[0]) if rows else None

    def catalog_history(self, before_ts: float, limit: int = 8000) -> list[dict]:
        rows = self._query(
            "SELECT * FROM catalog WHERE ts <= ? ORDER BY id DESC LIMIT ?",
            (before_ts, limit),
        )
        return [dict(r) for r in rows]

    def price_range(self, mint: str, t0: float, t1: float) -> list[tuple[float, float]]:
        rows = self._query(
            "SELECT ts, price_usd FROM prices WHERE mint=? AND ts>=? AND ts<=?"
            " ORDER BY ts",
            (mint, t0, t1),
        )
        return [(r["ts"], r["price_usd"]) for r in rows]

    def flow_before(self, mint: str, ts: float,
                    max_age_s: float = 180.0) -> tuple[int, int] | None:
        rows = self._query(
            "SELECT buys_m5, sells_m5 FROM flow WHERE mint=? AND ts<=? AND ts>=?"
            " ORDER BY ts DESC LIMIT 1",
            (mint, ts, ts - max_age_s),
        )
        return (rows[0]["buys_m5"], rows[0]["sells_m5"]) if rows else None

    def call_counts(self, arm: str) -> dict:
        rows = self._query(
            "SELECT direction, COUNT(*) AS n FROM predictions WHERE arm=?"
            " GROUP BY direction",
            (arm,),
        )
        calls = sum(r["n"] for r in rows if r["direction"] in ("UP", "DOWN"))
        abstains = sum(r["n"] for r in rows if r["direction"] == "SKIP")
        return {"calls": calls, "abstains": abstains}

    def recent_calls(self, arm: str, since_ts: float) -> int:
        rows = self._query(
            "SELECT COUNT(*) AS n FROM predictions WHERE arm=? AND ts>=?"
            " AND direction IN ('UP','DOWN')",
            (arm, since_ts),
        )
        return rows[0]["n"] if rows else 0

    def hedge_unprocessed(self, after_id: int) -> list[dict]:
        rows = self._query(
            "SELECT p.id AS prediction_id, p.prompt, r.return_pct, r.status"
            " FROM predictions p JOIN resolutions r ON r.prediction_id = p.id"
            " WHERE p.arm='hedge' AND p.id > ? ORDER BY p.id",
            (after_id,),
        )
        return [dict(r) for r in rows]

    def latest_llm_prediction(self) -> dict | None:
        rows = self._query(
            "SELECT * FROM predictions WHERE arm='llm' ORDER BY id DESC LIMIT 1"
        )
        return dict(rows[0]) if rows else None

    def first_prediction_ts(self) -> float | None:
        rows = self._query("SELECT MIN(ts) AS t FROM predictions")
        return rows[0]["t"] if rows and rows[0]["t"] is not None else None

    def set_meta(self, key: str, value: str) -> None:
        self._execute(
            "INSERT INTO meta (key, value) VALUES (?,?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    # --- reads ----------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        rows = self._query("SELECT value FROM meta WHERE key=?", (key,))
        return rows[0]["value"] if rows else None

    def latest_balance(self, arm: str, default: float) -> float:
        rows = self._query(
            "SELECT balance_after FROM fills WHERE arm=? ORDER BY id DESC LIMIT 1",
            (arm,),
        )
        return rows[0]["balance_after"] if rows else default

    def open_position(self, arm: str) -> Position | None:
        """Resume support: the last fill being a buy means the position is open."""
        rows = self._query(
            "SELECT * FROM fills WHERE arm=? ORDER BY id DESC LIMIT 1", (arm,)
        )
        if not rows or rows[0]["side"] != "buy":
            return None
        r = rows[0]
        return Position(
            trade_id=r["trade_id"],
            mint=r["mint"],
            symbol=r["symbol"],
            token_qty=r["token_qty"],
            entry_cost_usd=r["size_usd"],
            entry_ts=r["ts"],
            stop_mode=r["stop_mode"],
        )

    def fills(self, arm: str | None = None) -> list[dict]:
        if arm is None:
            rows = self._query("SELECT * FROM fills ORDER BY id")
        else:
            rows = self._query("SELECT * FROM fills WHERE arm=? ORDER BY id", (arm,))
        return [dict(r) for r in rows]

    def decisions(self, limit: int = 50) -> list[dict]:
        rows = self._query(
            "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in rows]

    def latest_price(self, mint: str) -> tuple[float, float] | None:
        """Return (ts, price_usd) of the newest cached price for a mint."""
        rows = self._query(
            "SELECT ts, price_usd FROM prices WHERE mint=? ORDER BY id DESC LIMIT 1",
            (mint,),
        )
        return (rows[0]["ts"], rows[0]["price_usd"]) if rows else None

    def latest_liquidity(self, mint: str) -> float | None:
        rows = self._query(
            "SELECT liquidity_usd FROM liquidity WHERE mint=? ORDER BY id DESC LIMIT 1",
            (mint,),
        )
        return rows[0]["liquidity_usd"] if rows else None

    def price_history(self, mint: str, limit: int = 300) -> list[tuple[float, float]]:
        rows = self._query(
            "SELECT ts, price_usd FROM prices WHERE mint=? ORDER BY id DESC LIMIT ?",
            (mint, limit),
        )
        return [(r["ts"], r["price_usd"]) for r in reversed(rows)]

    def first_fill_ts(self) -> float | None:
        rows = self._query("SELECT MIN(ts) AS t FROM fills")
        return rows[0]["t"] if rows and rows[0]["t"] is not None else None


def fill_from_row(row: dict) -> Fill:
    return Fill(
        ts=row["ts"], arm=row["arm"], trade_id=row["trade_id"], side=row["side"],
        mint=row["mint"], symbol=row["symbol"], feed_price=row["feed_price"],
        exec_price=row["exec_price"], size_usd=row["size_usd"],
        token_qty=row["token_qty"],
        fees=FeeBreakdown(lp=row["fee_lp"], slippage=row["fee_slippage"],
                          priority=row["fee_priority"], tds=row["fee_tds"]),
        realized_pnl=row["realized_pnl"], balance_after=row["balance_after"],
        stop_mode=row["stop_mode"], note=row.get("note", ""),
    )
