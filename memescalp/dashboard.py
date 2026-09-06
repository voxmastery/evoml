"""FastAPI app serving the single-page dashboard and its JSON endpoints."""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .config import Settings
from .control import Controller
from .db import Database
from .executor import sell_proceeds
from .feed import PriceFeed
from .metrics import (
    arm_stats, calibration, capital_progression, earn_stats, equity_curve,
    evaluate, evaluate_predictions, kelly_progression, prediction_curves,
    prediction_stats,
)
from .strategy import TradingArm

_HTML_PATH = Path(__file__).parent / "static" / "dashboard.html"
_PREDICT_HTML_PATH = Path(__file__).parent / "static" / "predict.html"


def create_app(settings: Settings, db: Database, feed: PriceFeed,
               arms: dict[str, TradingArm],
               controller: Controller | None = None) -> FastAPI:
    app = FastAPI(title="memescalp paper-trading dashboard")

    def _live_equity(arm: TradingArm) -> float:
        # Runs in Starlette's threadpool while the trading loop mutates the
        # arm: capture both attributes once so a concurrent exit/entry can
        # never yield an AttributeError mid-computation.
        equity = arm.balance
        pos = arm.position
        if pos is not None:
            snap = feed.snapshot(pos.mint)
            if snap is not None and snap.price_usd > 0:
                net, _ = sell_proceeds(settings, pos, snap)
                equity += net
            else:
                equity += pos.entry_cost_usd
        return equity

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        if settings.mode == "predict":
            return _PREDICT_HTML_PATH.read_text()
        return _HTML_PATH.read_text()

    @app.get("/api/predict/summary")
    def predict_summary() -> dict:
        llm_rows = db.resolved_predictions("llm")
        rnd_rows = db.resolved_predictions("random")
        math_rows = db.resolved_predictions("math")
        ml_rows = db.resolved_predictions("ml")
        hedge_rows = db.resolved_predictions("hedge")
        # With the Claude arm stopped, the experiment's subject is the ML arm.
        subject = "llm" if settings.llm_arm else "ml"
        subject_rows = llm_rows if subject == "llm" else ml_rows
        result = evaluate_predictions(
            subject_rows, rnd_rows, db.first_prediction_ts(),
            settings.min_predictions, settings.min_days, settings.edge_z_min,
        )
        pending = sum(1 for r in db.prediction_ledger(500)
                      if r["status"] is None and r["direction"])
        return {
            "pass_fail": {**asdict(result), "overall": result.overall},
            "subject": subject,
            "arms": {
                name: {
                    **asdict(prediction_stats(rows)),
                    **db.call_counts(name),
                    "capital": capital_progression(
                        rows, settings.start_balance_usd)["final"],
                    "kelly_capital": kelly_progression(
                        rows, settings.start_balance_usd)["final"],
                    "brier": calibration(rows)["brier"],
                    "earn": earn_stats(
                        rows, db.recent_calls(name, time.time() - 3 * 3600)
                        / 3.0),
                }
                for name, rows in (("llm", llm_rows), ("random", rnd_rows),
                                   ("math", math_rows), ("ml", ml_rows),
                                   ("hedge", hedge_rows))
            },
            "calibration": {
                "llm": calibration(llm_rows),
                "math": calibration(math_rows),
                "ml": calibration(ml_rows),
            },
            "playbook": db.latest_playbook(),
            "start_capital": settings.start_balance_usd,
            "pending": pending,
            "control": {
                "paused": controller.paused if controller else False,
                "experiment_start": (controller.experiment_start_ts()
                                     if controller else None),
                "now": time.time(),
            },
            "config": {
                "predict_minutes": settings.predict_minutes,
                "horizon_minutes": settings.horizon_minutes,
            },
        }

    @app.get("/api/predict/curves")
    def predict_curves() -> dict:
        out = {}
        for arm in ("llm", "random", "math", "ml", "hedge"):
            rows = db.resolved_predictions(arm)
            out[arm] = {
                **prediction_curves(rows),
                "capital": capital_progression(
                    rows, settings.start_balance_usd)["curve"],
                "kelly": kelly_progression(
                    rows, settings.start_balance_usd)["curve"],
            }
        out["start_capital"] = settings.start_balance_usd
        return out

    @app.get("/api/predict/ledger")
    def predict_ledger() -> dict:
        dollars: dict[int, dict] = {}
        for arm in ("llm", "random", "math", "ml", "hedge"):
            dollars.update(capital_progression(
                db.resolved_predictions(arm),
                settings.start_balance_usd)["by_id"])
        ledger = []
        for row in db.prediction_ledger(limit=40):
            extra = dollars.get(row["id"], {})
            ledger.append({**row, **extra})
        return {"ledger": ledger}

    @app.get("/api/evolution")
    def evolution() -> dict:
        import json as _json
        out = []
        for row in db.evolution_log(limit=25):
            try:
                genome = _json.loads(row["genome"])
                scores = _json.loads(row["scores"])
            except (ValueError, TypeError):
                genome, scores = {}, {}
            out.append({
                "ts": row["ts"], "generation": row["generation"],
                "champion": row["champion"],
                "holdout": scores.get(row["champion"]),
                "n_features": len(genome.get("features", [])),
                "synth": genome.get("synth", []),
                "thr": genome.get("thr"), "mut_rate": genome.get("mut_rate"),
                "population": len(scores),
            })
        gene_report = db.get_meta("ml_gene_report")
        growth = db.get_meta("ml_growth")
        return {"lineage": out,
                "gene_report": _json.loads(gene_report) if gene_report else {},
                "organism": _json.loads(growth) if growth else None}

    @app.get("/api/predict/latest")
    def predict_latest() -> dict:
        return {"prediction": db.latest_llm_prediction()}

    @app.get("/api/summary")
    def summary() -> dict:
        llm_fills = db.fills("llm")
        rnd_fills = db.fills("random")
        result = evaluate(
            llm_fills, rnd_fills, db.first_fill_ts(),
            settings.min_round_trips, settings.min_days, settings.edge_z_min,
        )
        arms_payload = {}
        for name, fills in (("llm", llm_fills), ("random", rnd_fills)):
            arm = arms[name]
            stats = arm_stats(fills)
            pos = arm.position
            equity = _live_equity(arm)
            arms_payload[name] = {
                **asdict(stats),
                "fees_total": stats.fees_total,
                "balance": arm.balance,
                "equity": equity,
                "busted": arm.busted,
                "position": None if pos is None else {
                    "symbol": pos.symbol, "mint": pos.mint,
                    "entry_cost_usd": pos.entry_cost_usd,
                    "entry_ts": pos.entry_ts, "stop_mode": pos.stop_mode,
                    "unrealized_pnl": equity - arm.balance - pos.entry_cost_usd,
                },
            }
        return {
            "pass_fail": {**asdict(result), "overall": result.overall},
            "arms": arms_payload,
            "control": {
                "paused": controller.paused if controller else False,
                "experiment_start": (controller.experiment_start_ts()
                                     if controller else None),
                "now": time.time(),
            },
            "config": {
                "start_balance_usd": settings.start_balance_usd,
                "start_balance_inr": settings.start_balance_inr,
                "target_profit_usd": settings.target_profit_usd,
                "stop_mode": settings.stop_mode,
                "pick_minutes": settings.pick_minutes,
            },
        }

    @app.get("/api/equity")
    def equity() -> dict:
        start = settings.start_balance_usd
        return {
            "start_balance_usd": start,
            "llm": equity_curve(db.fills("llm"), start),
            "random": equity_curve(db.fills("random"), start),
        }

    @app.get("/api/trades")
    def trades() -> dict:
        fills = db.fills()
        return {"fills": fills[-200:]}

    @app.get("/api/decisions")
    def decisions() -> dict:
        return {"decisions": db.decisions(limit=30)}

    @app.get("/api/prices/{mint}")
    def prices(mint: str) -> dict:
        return {"mint": mint, "prices": db.price_history(mint, limit=360)}

    @app.get("/api/live")
    def live() -> dict:
        from .micro import live_rank
        snapshots = feed.all_snapshots()
        ticks = {s.mint: db.price_history(s.mint, limit=150)
                 for s in snapshots}
        return {
            "ts": time.time(),
            "board": live_rank(settings, snapshots, ticks,
                               settings.start_balance_usd),
        }

    @app.get("/api/catalog")
    def catalog() -> dict:
        picks = {arms[a].pick_mint for a in arms if arms[a].pick_mint}
        return {"catalog": db.latest_catalog(), "picked": sorted(picks)}

    @app.get("/api/analyses")
    def analyses() -> dict:
        return {"analyses": db.analyses(limit=10)}

    @app.post("/api/control/start")
    def control_start() -> dict:
        if controller is None:
            return {"error": "no controller"}
        controller.start()
        return {"paused": controller.paused}

    @app.post("/api/control/pause")
    def control_pause() -> dict:
        if controller is None:
            return {"error": "no controller"}
        controller.pause()
        return {"paused": controller.paused}

    @app.post("/api/control/reset")
    def control_reset() -> dict:
        if controller is None:
            return {"error": "no controller"}
        archive = controller.reset()
        return {"archived_to": archive, "restarting": True}

    return app
