"""Performance metrics and the experiment's pass/fail evaluation."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ArmStats:
    round_trips: int
    net_pnl: float          # realized, after every modeled cost
    win_rate: float
    avg_win: float
    avg_loss: float
    largest_loss: float
    fees_lp: float
    fees_slippage: float
    fees_priority: float
    fees_tds: float

    @property
    def fees_total(self) -> float:
        return self.fees_lp + self.fees_slippage + self.fees_priority + self.fees_tds


@dataclass(frozen=True)
class PassFail:
    profitable: bool
    net_pnl: float
    enough_trades: bool
    round_trips: int
    min_round_trips: int
    enough_days: bool
    days_elapsed: float
    min_days: int
    beats_control: bool
    edge_usd: float
    edge_z: float
    edge_z_min: float

    @property
    def overall(self) -> bool:
        return (self.profitable and self.enough_trades
                and self.enough_days and self.beats_control)


def arm_stats(fills: list[dict]) -> ArmStats:
    pnls = [f["realized_pnl"] for f in fills
            if f["side"] == "sell" and f["realized_pnl"] is not None]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    return ArmStats(
        round_trips=len(pnls),
        net_pnl=sum(pnls),
        win_rate=len(wins) / len(pnls) if pnls else 0.0,
        avg_win=sum(wins) / len(wins) if wins else 0.0,
        avg_loss=sum(losses) / len(losses) if losses else 0.0,
        largest_loss=min(pnls) if pnls else 0.0,
        fees_lp=sum(f["fee_lp"] for f in fills),
        fees_slippage=sum(f["fee_slippage"] for f in fills),
        fees_priority=sum(f["fee_priority"] for f in fills),
        fees_tds=sum(f["fee_tds"] for f in fills),
    )


def _trade_pnls(fills: list[dict]) -> list[float]:
    return [f["realized_pnl"] for f in fills
            if f["side"] == "sell" and f["realized_pnl"] is not None]


def edge_z_score(strategy_fills: list[dict], control_fills: list[dict]) -> float:
    """Welch z-statistic for mean per-trade PnL, strategy minus control."""
    a, b = _trade_pnls(strategy_fills), _trade_pnls(control_fills)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    var_a = sum((x - mean_a) ** 2 for x in a) / (len(a) - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt(var_a / len(a) + var_b / len(b))
    if se == 0:
        return 0.0
    return (mean_a - mean_b) / se


def evaluate(
    strategy_fills: list[dict],
    control_fills: list[dict],
    first_fill_ts: float | None,
    min_round_trips: int,
    min_days: int,
    edge_z_min: float,
    now: float | None = None,
) -> PassFail:
    now = time.time() if now is None else now
    s = arm_stats(strategy_fills)
    c = arm_stats(control_fills)
    days = (now - first_fill_ts) / 86400.0 if first_fill_ts is not None else 0.0
    z = edge_z_score(strategy_fills, control_fills)
    return PassFail(
        profitable=s.net_pnl > 0,
        net_pnl=s.net_pnl,
        enough_trades=s.round_trips >= min_round_trips,
        round_trips=s.round_trips,
        min_round_trips=min_round_trips,
        enough_days=days >= min_days,
        days_elapsed=days,
        min_days=min_days,
        beats_control=s.net_pnl > c.net_pnl and z >= edge_z_min,
        edge_usd=s.net_pnl - c.net_pnl,
        edge_z=z,
        edge_z_min=edge_z_min,
    )


def equity_curve(fills: list[dict], start_balance: float) -> list[tuple[float, float]]:
    """(ts, cash-equity-after-fill) points. Buys dip to residual cash; sells
    restore cash plus PnL — round-trip equity is exact at every sell."""
    points: list[tuple[float, float]] = []
    equity = start_balance
    for f in fills:
        if f["side"] == "sell":
            equity = f["balance_after"]
            points.append((f["ts"], equity))
        else:
            # Mark the buy at entry cost so the curve stays continuous.
            points.append((f["ts"], f["balance_after"] + f["size_usd"]))
    return points


# --- prediction-mode metrics ---------------------------------------------------

CAP_RETURN_PCT = 50.0  # winsorize: beyond this a 2-min print is a glitch,
                        # not an executable price (e.g. new-listing jumps)


def _signed_capped(r: dict) -> float:
    signed = r["return_pct"] if r["direction"] == "UP" else -r["return_pct"]
    return max(-CAP_RETURN_PCT, min(CAP_RETURN_PCT, signed))


def _scored(r: dict) -> bool:
    """A resolution carries information only if the price actually moved.
    An exactly-flat outcome means the feed served a stale quote (inactive
    token) — treated as void, never as a wrong call."""
    return r["status"] == "resolved" and r["return_pct"] != 0


@dataclass(frozen=True)
class PredictionStats:
    resolved: int
    correct: int
    accuracy: float
    avg_confidence: float
    cum_return_pct: float   # sum of signed per-prediction returns (frictionless)
    voids: int


@dataclass(frozen=True)
class PredictPassFail:
    skilled: bool           # accuracy significantly above coin-flip
    skill_z: float
    accuracy: float
    enough_predictions: bool
    resolved: int
    min_predictions: int
    enough_days: bool
    days_elapsed: float
    min_days: int
    beats_control: bool
    control_accuracy: float
    edge_z: float
    edge_z_min: float

    @property
    def overall(self) -> bool:
        return (self.skilled and self.enough_predictions
                and self.enough_days and self.beats_control)


def prediction_stats(rows: list[dict]) -> PredictionStats:
    ok = [r for r in rows if _scored(r)]
    voids = sum(1 for r in rows if r["status"] == "void"
                or (r["status"] == "resolved" and r["return_pct"] == 0))
    correct = sum(1 for r in ok if r["correct"])
    signed = [_signed_capped(r) for r in ok]
    return PredictionStats(
        resolved=len(ok),
        correct=correct,
        accuracy=correct / len(ok) if ok else 0.0,
        avg_confidence=(sum(r["confidence"] for r in ok) / len(ok)) if ok else 0.0,
        cum_return_pct=sum(signed),
        voids=voids,
    )


def evaluate_predictions(
    llm_rows: list[dict],
    control_rows: list[dict],
    first_ts: float | None,
    min_predictions: int,
    min_days: int,
    edge_z_min: float,
    now: float | None = None,
) -> PredictPassFail:
    now = time.time() if now is None else now
    a = prediction_stats(llm_rows)
    b = prediction_stats(control_rows)
    days = (now - first_ts) / 86400.0 if first_ts is not None else 0.0

    # One-sample z against the coin-flip null p = 0.5.
    skill_z = ((a.accuracy - 0.5) / math.sqrt(0.25 / a.resolved)
               if a.resolved >= 10 else 0.0)

    # Two-proportion z: llm accuracy vs control accuracy.
    edge_z = 0.0
    if a.resolved >= 10 and b.resolved >= 10:
        pooled = (a.correct + b.correct) / (a.resolved + b.resolved)
        se = math.sqrt(pooled * (1 - pooled)
                       * (1 / a.resolved + 1 / b.resolved))
        if se > 0:
            edge_z = (a.accuracy - b.accuracy) / se

    return PredictPassFail(
        skilled=skill_z >= edge_z_min,
        skill_z=skill_z,
        accuracy=a.accuracy,
        enough_predictions=a.resolved >= min_predictions,
        resolved=a.resolved,
        min_predictions=min_predictions,
        enough_days=days >= min_days,
        days_elapsed=days,
        min_days=min_days,
        beats_control=a.accuracy > b.accuracy and edge_z >= edge_z_min,
        control_accuracy=b.accuracy,
        edge_z=edge_z,
        edge_z_min=edge_z_min,
    )


def prediction_curves(rows: list[dict]) -> dict:
    """Cumulative signed return (%) and running accuracy over resolved rows."""
    cum_return, accuracy = [], []
    total = correct = 0
    running = 0.0
    for r in rows:
        if not _scored(r):
            continue
        total += 1
        correct += 1 if r["correct"] else 0
        running += _signed_capped(r)
        cum_return.append((r["resolved_ts"], running))
        accuracy.append((r["resolved_ts"], correct / total * 100.0))
    return {"cum_return": cum_return, "accuracy": accuracy}


def capital_progression(rows: list[dict], start_capital: float) -> dict:
    """Compound a paper capital through resolved predictions in time order.

    Frictionless: each resolved call stakes the full running capital on the
    predicted direction and books the signed move. Returns the curve, the
    final value, and per-prediction dollar deltas keyed by prediction id.
    """
    capital = start_capital
    curve: list[tuple[float, float]] = []
    by_id: dict[int, dict] = {}
    for r in rows:
        if not _scored(r):
            continue
        dollars = capital * _signed_capped(r) / 100.0
        capital = max(0.0, capital + dollars)
        curve.append((r["resolved_ts"], capital))
        pid = r.get("prediction_id")
        if pid is not None:
            by_id[pid] = {"usd_pnl": dollars, "capital_after": capital}
    return {"curve": curve, "final": capital, "by_id": by_id}


# --- Kelly sizing & calibration -------------------------------------------------

KELLY_WINDOW = 30       # trailing resolved calls used to estimate the edge
KELLY_MIN_N = 10        # no bets until this many outcomes exist (warmup)
KELLY_CAP = 0.25        # never stake more than 25% of capital
KELLY_HALF = 2.0        # half-Kelly: standard practice against estimation error


def kelly_progression(rows: list[dict], start_capital: float,
                      window: int = KELLY_WINDOW, min_n: int = KELLY_MIN_N,
                      cap: float = KELLY_CAP) -> dict:
    """Compound capital betting a Kelly fraction sized from TRAILING accuracy.

    The fraction for each prediction uses only outcomes known before it
    (no lookahead). Even-odds directional Kelly is f* = 2p - 1; we bet half
    of that, capped, and floored at zero — when the trailing record shows no
    edge, Kelly's answer is to not bet, and capital simply holds.
    """
    capital = start_capital
    curve: list[tuple[float, float]] = []
    by_id: dict[int, dict] = {}
    outcomes: list[int] = []
    for r in rows:
        if not _scored(r):
            continue
        if len(outcomes) >= min_n:
            recent = outcomes[-window:]
            p = sum(recent) / len(recent)
            fraction = min(max(0.0, (2.0 * p - 1.0) / KELLY_HALF), cap)
        else:
            fraction = 0.0
        dollars = capital * fraction * _signed_capped(r) / 100.0
        capital = max(0.0, capital + dollars)
        outcomes.append(1 if r["correct"] else 0)
        curve.append((r["resolved_ts"], capital))
        pid = r.get("prediction_id")
        if pid is not None:
            by_id[pid] = {"usd_pnl": dollars, "fraction": fraction,
                          "capital_after": capital}
    return {"curve": curve, "final": capital, "by_id": by_id}


_CAL_BINS = [(0.0, 0.55, "<=55%"), (0.55, 0.65, "55-65%"),
             (0.65, 0.75, "65-75%"), (0.75, 1.01, "75%+")]


def calibration(rows: list[dict]) -> dict:
    """Brier score (1950) plus reliability buckets: when the forecaster says
    X% confident, how often is it actually right? 0.25 = uninformative."""
    ok = [r for r in rows if _scored(r)]
    if not ok:
        return {"brier": None, "n": 0, "bins": []}
    brier = sum((r["confidence"] - (1 if r["correct"] else 0)) ** 2
                for r in ok) / len(ok)
    bins = []
    for lo, hi, label in _CAL_BINS:
        members = [r for r in ok if lo <= r["confidence"] < hi]
        if not members:
            continue
        bins.append({
            "label": label,
            "n": len(members),
            "avg_confidence": sum(r["confidence"] for r in members) / len(members),
            "accuracy": sum(1 for r in members if r["correct"]) / len(members),
        })
    return {"brier": brier, "n": len(ok), "bins": bins}


# --- measured earn-rate ---------------------------------------------------------

MARKET_FEE = 0.01  # ~1% per contract, typical of up/down prediction venues


def earn_stats(rows: list[dict], calls_per_hour: float,
               bankrolls: tuple[float, ...] = (23.0, 10_000.0),
               fee: float = MARKET_FEE) -> dict:
    """What following this arm would generate per hour in an even-odds
    up/down market charging `fee` per call, at half-Kelly sizing — measured
    from the last <=200 scored calls, with a binomial CI on accuracy."""
    ok = [r for r in rows if _scored(r)][-200:]
    n = len(ok)
    if n < 30:
        return {"n": n, "ready": False}
    p = sum(1 for r in ok if r["correct"]) / n
    se = math.sqrt(p * (1 - p) / n)
    edge = 2.0 * p - 1.0 - fee          # expected profit per $1 staked
    fraction = max(0.0, edge / 2.0)     # half-Kelly, floored at zero
    hourly_frac = fraction * edge * calls_per_hour
    return {
        "n": n, "ready": True, "accuracy": p, "ci95": 1.96 * se,
        "edge_per_call": edge, "kelly_fraction": fraction,
        "calls_per_hour": calls_per_hour,
        "usd_per_hour": {str(int(b)): b * hourly_frac for b in bankrolls},
        "bankroll_for_60_per_hour": (60.0 / hourly_frac
                                     if hourly_frac > 0 else None),
    }
