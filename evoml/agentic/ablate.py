"""Ablation on the simulator.

Arms:

- full: planner, meta-genome evolution, causal memory
- no_meta: planner and memory, fixed score weights
- no_memory: planner and meta-genome, no recall
- no_planner: meta-genome and memory, no hypotheses
- random_control: untouched pages, no evolution

The CSV and the markdown report contain the numbers this process just
computed. They are simulated. They are not a Search Console result.
"""
from __future__ import annotations

import csv
from pathlib import Path

from evoml.core.memory import NullMemory

from .config import AgentConfig
from .loop import AgentSession
from .memory import CausalMemory, InMemoryHook
from .planner import Hypothesis, Planner
from .sensor import booking_mean


ARMS = ("full", "no_meta", "no_memory", "no_planner", "random_control")

COLUMNS = (
    "arm",
    "booking_mean",
    "control_booking_mean",
    "promotions",
    "hypotheses",
    "recalled",
    "meta_booking_weight",
    "sections",
    "pages_changed",
    "simulated",
)


class _SilentPlanner:
    """The no-planner arm. The arena still mutates and still decides."""

    def propose(self, **kwargs) -> list[Hypothesis]:
        return []


def run_ablation(
    *,
    seed: int = 0,
    cycles: int = 2,
    cohort_size: int = 8,
    max_pages: int = 8,
    out_dir: str | Path,
) -> list[dict]:
    """Run every arm and write ``ablation.csv`` and ``ablation.md``."""
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = [
        _run_arm(
            arm,
            seed=seed,
            cycles=cycles,
            cohort_size=cohort_size,
            max_pages=max_pages,
            out_dir=destination / arm,
        )
        for arm in ARMS
    ]
    _write_csv(destination / "ablation.csv", rows)
    _write_markdown(destination / "ablation.md", rows, seed=seed, cycles=cycles)
    return rows


def _run_arm(
    arm: str,
    *,
    seed: int,
    cycles: int,
    cohort_size: int,
    max_pages: int,
    out_dir: Path,
) -> dict:
    if arm == "random_control":
        return _random_control(seed, cohort_size, cycles)
    config = AgentConfig(
        domain="seo_pages",
        seed=seed,
        dry_run=True,
        planner="stub",
        max_pages_changed=max_pages,
        output_dir=str(out_dir),
        memory="none",
        meta=arm != "no_meta",
        signals="simulator",
        cohort_size=cohort_size,
        generations=1,
        cycles=cycles,
    )
    planner: Planner | None = _SilentPlanner() if arm == "no_planner" else None
    # no_memory is the null hook. The other arms keep episodes in-process
    # so a later cycle can recall them. Nothing is written to a service.
    memory = (
        CausalMemory(NullMemory())
        if arm == "no_memory"
        else CausalMemory(InMemoryHook())
    )
    session = AgentSession(config, planner=planner, memory=memory)
    promotions = 0
    hypotheses = 0
    recalled = 0
    last = None
    for _ in range(cycles):
        last = session.cycle()
        promotions += int(last.promoted)
        hypotheses += len(last.hypotheses)
        recalled = last.recalled
    assert last is not None
    return {
        "arm": arm,
        "booking_mean": last.booking_mean,
        "control_booking_mean": last.control_booking_mean,
        "promotions": promotions,
        "hypotheses": hypotheses,
        "recalled": recalled,
        "meta_booking_weight": last.meta_booking_weight,
        "sections": " ".join(last.champion_sections),
        "pages_changed": last.manifest["pages_changed"],
        "simulated": True,
    }


def _random_control(seed: int, cohort_size: int, cycles: int) -> dict:
    """The untouched cohort, scored on the same simulator as the other arms."""
    from evoml.domains.seo_pages.evaluator import RecipeSimulator

    simulator = RecipeSimulator(seed=seed)
    # `cycles` shifts the day the way a treated arm's last cycle does,
    # so the control is read on the same clock.
    day = max(0, cycles - 1)
    rows = simulator.control_observations(cohort_size, day=day)
    return {
        "arm": "random_control",
        "booking_mean": booking_mean(rows, "control"),
        "control_booking_mean": booking_mean(rows, "control"),
        "promotions": 0,
        "hypotheses": 0,
        "recalled": 0,
        "meta_booking_weight": None,
        "sections": "control",
        "pages_changed": 0,
        "simulated": True,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in COLUMNS})


def _write_markdown(path: Path, rows: list[dict], *, seed: int, cycles: int) -> None:
    lines = [
        "# Agentic learning ablation",
        "",
        "Simulated. The table is the output of this run on the built-in",
        f"page simulator (seed {seed}, {cycles} cycle(s) per arm).",
        "It is not a Search Console measurement and it is not a benchmark",
        "to quote as a result for nammaambulance.in.",
        "",
        "| arm | booking mean | control booking mean | promotions | hypotheses | recalled | meta booking weight | sections |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        weight = row["meta_booking_weight"]
        weight_text = "" if weight is None else f"{weight:.4f}"
        lines.append(
            "| {arm} | {book:.4f} | {control:.4f} | {promotions} | {hypotheses} | "
            "{recalled} | {weight} | {sections} |".format(
                arm=row["arm"],
                book=row["booking_mean"],
                control=row["control_booking_mean"],
                promotions=row["promotions"],
                hypotheses=row["hypotheses"],
                recalled=row["recalled"],
                weight=weight_text,
                sections=row["sections"],
            )
        )
    lines.append("")
    lines.append(
        "booking mean is the simulator's mean `booking_starts` of the recipe "
        "the arm kept (the untouched cohort, for random_control)."
    )
    lines.append(
        "Arms differ by which pieces are switched off: meta-genome evolution, "
        "causal memory, the planner, or the whole loop."
    )
    lines.append("")
    path.write_text("\n".join(lines))
