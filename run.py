"""Entry point: starts the feed, both trading arms, the picker scheduler,
and the dashboard. Simulation only — nothing here can touch a wallet.

Usage: python run.py
"""
from __future__ import annotations

import asyncio
import os
import logging
import time

import uvicorn

from memescalp.analyst import analyze_position
from memescalp.catalog import build_catalog
from memescalp.config import load_settings
from memescalp.control import Controller
from memescalp.csvlog import CsvMirror
from memescalp.executor import USDC_MINT, quote_sell_net, sell_proceeds
from memescalp.indicators import one_minute_closes
from memescalp.jupiter import QuoteError
from memescalp.dashboard import create_app
from memescalp.db import Database
from memescalp.feed import PriceFeed
from memescalp.jupiter import JupiterQuoter
from memescalp.llm import LlmError, make_backend
from memescalp.picker import llm_pick, random_pick
from memescalp.hedgepred import HedgeForecaster
from memescalp.mathpred import math_prediction
from memescalp.micro import micro_prompt_lines
from memescalp.mlpred import MLForecaster
from memescalp.predictor import (
    llm_prediction, random_prediction, score_resolution,
)
from memescalp.strategy import TradingArm
from memescalp.tsfm import ChronosFeatures

log = logging.getLogger("memescalp")

RESOLUTION_GRACE_SECONDS = 1800.0  # void a prediction with no price data by then
SOL_MINT = "So11111111111111111111111111111111111111112"


def _regime_line(feed: PriceFeed) -> str:
    s = feed.snapshot(SOL_MINT)
    if s is None or s.price_usd <= 0:
        return "n/a"
    return (f"SOL ${s.price_usd:,.2f} | 5m {s.chg_m5:+.2f}% | "
            f"1h {s.price_change_h1:+.2f}% | 24h {s.price_change_h24:+.2f}%")


def _track_record(db: Database, arm: str = "llm", last: int = 10) -> str:
    rows = [r for r in db.resolved_predictions(arm)
            if r["status"] == "resolved" and r["return_pct"] != 0][-last:]
    if not rows:
        return "no resolved calls yet"
    lines = [
        f"{r['symbol']} {r['direction']} (conf {r['confidence']:.2f}) -> "
        f"{'CORRECT' if r['correct'] else 'WRONG'} ({r['return_pct']:+.2f}%)"
        for r in rows
    ]
    correct = sum(1 for r in rows if r["correct"])
    lines.append(f"last {len(rows)}: {correct}/{len(rows)} correct")
    return "\n".join(lines)


async def tick_loop(arms: dict[str, TradingArm], interval: float,
                    controller: Controller) -> None:
    while True:
        if not controller.paused:
            for arm in arms.values():
                try:
                    await arm.tick()
                except Exception:
                    log.exception("[%s] tick failed", arm.arm)
        await asyncio.sleep(interval)


async def picker_loop(settings, db: Database, csv: CsvMirror, feed: PriceFeed,
                      backend, arms: dict[str, TradingArm],
                      controller: Controller) -> None:
    interval = settings.pick_minutes * 60.0
    while True:
        if controller.paused:
            await asyncio.sleep(3.0)
            continue
        window_start = time.time()
        window_end = window_start + interval
        # Nothing in a window may take down the process: the feed, tick loop,
        # and dashboard must outlive any picker failure.
        try:
            await _run_pick_window(settings, db, csv, feed, backend, arms,
                                   window_start, window_end)
        except Exception:
            log.exception("pick window failed; arms keep their previous picks")
        while time.time() < window_end:
            await asyncio.sleep(min(5.0, max(1.0, window_end - time.time())))


async def _run_pick_window(settings, db: Database, csv: CsvMirror,
                           feed: PriceFeed, backend,
                           arms: dict[str, TradingArm],
                           window_start: float, window_end: float) -> None:
    snapshots = await feed.trending_candidates()
    if not snapshots:
        log.warning("no tradable candidates this window")
        return

    history = {s.mint: db.price_history(s.mint, limit=720) for s in snapshots}
    size = max(arms["llm"].balance, arms["random"].balance,
               settings.start_balance_usd * 0.5)
    candidates = build_catalog(settings, snapshots, history, size)
    db.insert_catalog(window_start, candidates)

    try:
        decision = await llm_pick(backend, settings.claude_model,
                                  candidates, window_start, window_end)
        db.insert_decision(decision)
        csv.append_decision(decision)
        arms["llm"].apply_decision(decision)
        log.info("[llm] picked %s", decision.symbol)
    except LlmError as e:
        log.error("LLM pick failed, llm arm keeps previous pick: %s", e)

    control = random_pick(candidates, window_start, window_end)
    db.insert_decision(control)
    csv.append_decision(control)
    arms["random"].apply_decision(control)
    log.info("[random] picked %s", control.symbol)

    keep = {m for arm in arms.values()
            for m in (arm.pick_mint,
                      arm.position.mint if arm.position else None)
            if m}
    await feed.prune_watchlist(keep)


async def _unrealized_net(settings, feed, quoter, arm) -> float | None:
    """Net PnL if the arm's position were exited right now, or None if the
    data needed isn't available this cycle."""
    pos = arm.position
    snap = feed.snapshot(pos.mint)
    if snap is None or snap.price_usd <= 0:
        return None
    if settings.fill_model == "quote":
        if snap.decimals <= 0:
            return None
        units = int(pos.token_qty * 10**snap.decimals)
        try:
            quote = await quoter.quote(pos.mint, USDC_MINT, units)
        except QuoteError:
            return None
        return quote_sell_net(settings, quote) - pos.entry_cost_usd
    net, _ = sell_proceeds(settings, pos, snap)
    return net - pos.entry_cost_usd


async def analyst_loop(settings, db: Database, csv: CsvMirror, feed: PriceFeed,
                       quoter, backend, arms: dict[str, TradingArm],
                       controller: Controller) -> None:
    """Chart-analysis cadence: Claude predicts HOLD/EXIT for the llm arm's
    open position. The random control keeps mechanical rules only."""
    interval = settings.analysis_minutes * 60.0
    while True:
        await asyncio.sleep(interval)
        if controller.paused:
            continue
        arm = arms["llm"]
        pos = arm.position
        if pos is None or arm.pending_exit is not None:
            continue
        try:
            closes = one_minute_closes(db.price_history(pos.mint, limit=800))
            if len(closes) < 3:
                continue
            unrealized = await _unrealized_net(settings, feed, quoter, arm)
            if unrealized is None:
                continue
            analysis = await analyze_position(
                backend, settings.claude_model, pos, closes, unrealized,
                settings.target_profit_usd, settings.stop_mode,
            )
            db.insert_analysis(analysis)
            csv.append_analysis(analysis)
            log.info("[llm] analyst: %s %s (confidence %.2f)",
                     analysis.action, pos.symbol, analysis.confidence)
            if analysis.action == "EXIT" and arm.position is pos:
                arm.request_exit("predict")
        except LlmError as e:
            log.error("chart analysis failed, holding: %s", e)
        except Exception:
            log.exception("analyst loop error; position unaffected")


ML_RETRAIN_SECONDS = 1800.0
PLAYBOOK_SECONDS = 1800.0
PLAYBOOK_MIN_RESOLVED = 20

PLAYBOOK_PROMPT = """\
You are reviewing your own performance as a memecoin direction forecaster so
you can improve. Below are your recent resolved calls and calibration.

{track}

Calibration (stated confidence vs actual hit rate): {calibration}

Write an updated playbook for your future self: at most 8 short bullet
lines of concrete, testable lessons (what setups worked, what to avoid, when
to SKIP). No preamble, just the bullets. If the record shows no reliable
pattern, say so honestly and advise skipping more.
"""


async def ml_retrain_loop(settings, db: Database, ml: MLForecaster,
                          controller: Controller) -> None:
    """Retrain the open-source ML arm on the growing log, again and again."""
    while True:
        try:
            result = await asyncio.to_thread(
                ml.retrain, db, settings.horizon_minutes * 60.0)
            if result["trained"]:
                scores = ", ".join(f"{k}={v:.1%}"
                                   for k, v in result["scores"].items())
                log.info("[ml] retrained on %d rows | champion %s (%d params) | %s",
                         result["rows"], result["champion"],
                         result["n_params"], scores)
                log.info("[ml] gen %s | succession: %s | invented genes: %s | strongest base genes: %s",
                         result.get("generation"), result.get("succession"),
                         result.get("synth") or "none",
                         result.get("top_genes"))
            else:
                log.info("[ml] not trained yet (%d/%d rows)",
                         result["rows"], 300)
        except Exception:
            log.exception("ml retrain failed; keeping previous model")
        await asyncio.sleep(ML_RETRAIN_SECONDS)


async def playbook_loop(settings, db: Database, backend,
                        controller: Controller) -> None:
    """Claude periodically distills lessons from its own resolved calls."""
    await asyncio.sleep(60.0)  # first edition soon after boot, then periodic
    while True:
        if controller.paused:
            await asyncio.sleep(PLAYBOOK_SECONDS)
            continue
        try:
            rows = [r for r in db.resolved_predictions("llm")
                    if r["status"] == "resolved" and r["return_pct"] != 0]
            if len(rows) < PLAYBOOK_MIN_RESOLVED:
                continue
            from memescalp.metrics import calibration as _cal
            cal = _cal(rows)
            cal_txt = ", ".join(
                f"{b['label']}: said {b['avg_confidence']:.0%} was right "
                f"{b['accuracy']:.0%} (n={b['n']})" for b in cal["bins"]
            ) or "n/a"
            prompt = PLAYBOOK_PROMPT.format(
                track=_track_record(db, last=40), calibration=cal_txt)
            response = await backend.complete(prompt)
            db.insert_playbook(time.time(), prompt, response)
            log.info("[llm] playbook updated (%d chars)", len(response))
        except LlmError as e:
            log.error("playbook update failed: %s", e)
        except Exception:
            log.exception("playbook loop error")
        await asyncio.sleep(PLAYBOOK_SECONDS)


async def predict_loop(settings, db: Database, csv: CsvMirror, feed: PriceFeed,
                       backend, controller: Controller,
                       ml: MLForecaster | None = None,
                       hedge: HedgeForecaster | None = None,
                       tsfm: ChronosFeatures | None = None) -> None:
    """Prediction windows: catalog + charts -> Claude picks a coin and calls
    UP/DOWN over the horizon; the control arm flips a coin. Nothing trades."""
    interval = settings.predict_minutes * 60.0
    while True:
        if controller.paused:
            await asyncio.sleep(3.0)
            continue
        window_start = time.time()
        window_end = window_start + interval
        try:
            snapshots = await feed.trending_candidates()
            if snapshots:
                history = {s.mint: db.price_history(s.mint, limit=720)
                           for s in snapshots}
                catalog = build_catalog(settings, snapshots, history,
                                        settings.start_balance_usd)
                db.insert_catalog(window_start, catalog)
                closes = {m: one_minute_closes(h) for m, h in history.items()}

                tsfm_map: dict = {}
                if tsfm is not None and tsfm.available:
                    try:
                        tsfm_map = await asyncio.to_thread(
                            tsfm.forecast_batch, closes)
                        for mint, (ret, spread) in tsfm_map.items():
                            db.insert_tsfm(window_start, mint, ret, spread)
                    except Exception:
                        log.exception("tsfm forecast failed this window")

                micro = micro_prompt_lines(catalog, history)
                pb = db.latest_playbook()
                if not settings.llm_arm:
                    pass  # Claude arm stopped by config (LLM_ARM=off)
                else:
                  try:
                    pred = await llm_prediction(backend, settings.claude_model,
                                                catalog, closes,
                                                settings.horizon_minutes,
                                                micro=micro,
                                                regime=_regime_line(feed),
                                                track=_track_record(db),
                                                playbook=pb["response"] if pb
                                                else "none yet")
                    pid = db.insert_prediction(pred)
                    csv.append_prediction(pid, pred)
                    log.info("[llm] predicts %s %s (conf %.2f)",
                             pred.symbol, pred.direction or "PARSE-FAIL",
                             pred.confidence)
                  except LlmError as e:
                    log.error("LLM prediction failed this window: %s", e)

                control = random_prediction(catalog, settings.horizon_minutes)
                pid = db.insert_prediction(control)
                csv.append_prediction(pid, control)
                log.info("[random] predicts %s %s", control.symbol,
                         control.direction)

                classical = math_prediction(catalog, closes,
                                            settings.horizon_minutes)
                pid = db.insert_prediction(classical)
                csv.append_prediction(pid, classical)
                log.info("[math] predicts %s %s (%s)", classical.symbol,
                         classical.direction, classical.response)

                if ml is not None:
                    learned = ml.make_prediction(catalog, history,
                                                 settings.horizon_minutes, db)
                    pid = db.insert_prediction(learned)
                    csv.append_prediction(pid, learned)
                    log.info("[ml] predicts %s %s (%s)", learned.symbol,
                             learned.direction, learned.response)

                if hedge is not None:
                    committee = hedge.make_prediction(
                        catalog, history, settings.horizon_minutes,
                        tsfm_by_mint=tsfm_map)
                    pid = db.insert_prediction(committee)
                    csv.append_prediction(pid, committee)
                    log.info("[hedge] %s", committee.response)
            else:
                log.warning("no candidates this prediction window")
        except Exception:
            log.exception("prediction window failed; skipping")
        while time.time() < window_end:
            await asyncio.sleep(min(5.0, max(1.0, window_end - time.time())))


async def resolve_loop(settings, db: Database, csv: CsvMirror,
                       controller: Controller) -> None:
    """Score predictions whose horizon has passed, from the cached feed."""
    while True:
        await asyncio.sleep(settings.poll_seconds)
        # Resolution continues even while paused: an outstanding horizon is
        # already committed and its outcome is just data collection.
        try:
            now = time.time()
            for p in db.due_unresolved(now):
                hit = db.price_at_or_after(p["mint"], p["horizon_end"])
                if hit is not None:
                    _, price_end = hit
                    ret, correct = score_resolution(
                        p["direction"], p["price_at"], price_end)
                    # An exactly-flat price means a stale quote (inactive
                    # token): no market information, so void — never "wrong".
                    status = "resolved" if ret != 0.0 else "void"
                    db.insert_resolution(p["id"], now, price_end, ret,
                                         correct, status)
                    csv.append_resolution(p["id"], now, price_end, ret,
                                          correct, status)
                    log.info("[%s] %s %s resolved: %+.2f%% -> %s",
                             p["arm"], p["symbol"], p["direction"], ret,
                             "VOID(flat)" if status == "void"
                             else ("CORRECT" if correct else "WRONG"))
                elif now > p["horizon_end"] + RESOLUTION_GRACE_SECONDS:
                    db.insert_resolution(p["id"], now, 0.0, 0.0, False, "void")
                    csv.append_resolution(p["id"], now, 0.0, 0.0, False,
                                          "void")
                    log.warning("[%s] %s prediction voided (no price data)",
                                p["arm"], p["symbol"])
        except Exception:
            log.exception("resolver error; will retry next tick")


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = load_settings()
    db = Database(settings.db_path)
    csv = CsvMirror(settings.csv_dir)
    feed = PriceFeed(db, settings.watchlist, settings.jupiter_api_key)
    quoter = JupiterQuoter(settings.jupiter_api_key)
    backend = make_backend(settings.picker_backend, settings.claude_model,
                           settings.anthropic_api_key)
    arms = {
        "llm": TradingArm(settings, db, csv, feed, "llm", quoter=quoter),
        "random": TradingArm(settings, db, csv, feed, "random", quoter=quoter),
    }
    controller = Controller(settings, db)
    # Pre-existing experiments (started before the start button existed)
    # keep running; a freshly reset database comes up paused.
    if db.get_meta("experiment_start") is not None and db.get_meta("paused") is None:
        db.set_meta("paused", "0")
    for arm in arms.values():
        if arm.position is not None:
            await feed.add_to_watchlist(arm.position.mint)

    if settings.mode == "predict":
        log.info(
            "prediction experiment | window %.0f min | horizon %.0f min"
            " | backend %s | pass gate: %d predictions / %d days",
            settings.predict_minutes, settings.horizon_minutes,
            settings.picker_backend, settings.min_predictions,
            settings.min_days,
        )
    else:
        log.info(
            "paper account: ₹%.0f = $%.2f per arm | target +$%.2f | stop mode %s"
            " | backend %s | fills: %s",
            settings.start_balance_inr, settings.start_balance_usd,
            settings.target_profit_usd, settings.stop_mode,
            settings.picker_backend, settings.fill_model,
        )

    app = create_app(settings, db, feed, arms, controller)
    server = uvicorn.Server(uvicorn.Config(
        app, host=os.environ.get("DASH_HOST", "127.0.0.1"),
        port=settings.dash_port, log_level="warning"
    ))
    log.info("dashboard: http://127.0.0.1:%d", settings.dash_port)

    tasks = [feed.poll_forever(settings.poll_seconds), server.serve()]
    if settings.mode == "predict":
        ml = MLForecaster()
        hedge = HedgeForecaster(db)
        tsfm = ChronosFeatures(int(settings.horizon_minutes))
        tasks += [
            predict_loop(settings, db, csv, feed, backend, controller,
                         ml=ml, hedge=hedge, tsfm=tsfm),
            resolve_loop(settings, db, csv, controller),
            ml_retrain_loop(settings, db, ml, controller),
        ]
        if settings.llm_arm:
            tasks.append(playbook_loop(settings, db, backend, controller))
    else:
        tasks += [
            tick_loop(arms, settings.poll_seconds, controller),
            picker_loop(settings, db, csv, feed, backend, arms, controller),
            analyst_loop(settings, db, csv, feed, quoter, backend, arms,
                         controller),
        ]
    try:
        await asyncio.gather(*tasks)
    finally:
        await feed.close()
        await quoter.close()
        db.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
