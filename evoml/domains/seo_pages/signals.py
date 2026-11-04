"""Outcome signals for landing pages.

Loaders read CSV or JSON exports (Search Console, engagement, bookings).
They do not open network connections. The simulator builds the same row
shape so tests can run without exports.
"""
from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


METRIC_NAMES = (
    "ctr",
    "position_gain",
    "scroll_depth",
    "heatmap_cta",
    "time_on_page",
    "booking_starts",
)

# Fixed scales so a weight means the same thing on every run. They are
# units, not measured benchmarks.
SCALES = {
    "ctr": 0.25,
    "position_gain": 4.0,
    "scroll_depth": 1.0,
    "heatmap_cta": 1.0,
    "time_on_page": 180.0,
    "booking_starts": 5.0,
}

COLUMNS = (
    "page_id",
    "day",
    "group",
    "clicks",
    "impressions",
    "ctr",
    "position",
    "position_delta",
    "scroll_depth",
    "heatmap_cta",
    "time_on_page",
    "booking_starts",
    "http_404",
    "deindexed",
    "noindex",
)


@dataclass
class PageObservation:
    """One page over one day.

    `group` is the recipe key the page was assigned to, or ``control``
    when the page was left untouched. Search-console fields, engagement
    fields and booking starts sit on the same row so a window can be
    scored without joining sources in the caller.
    """

    page_id: str
    day: int
    group: str
    clicks: float
    impressions: float
    ctr: float
    position: float
    position_delta: float
    scroll_depth: float
    heatmap_cta: float
    time_on_page: float
    booking_starts: float
    http_404: float = 0.0
    deindexed: float = 0.0
    noindex: float = 0.0

    @property
    def position_gain(self) -> float:
        """Positive when average position improved (the delta went down)."""
        return -float(self.position_delta)

    def vector(self) -> dict[str, float]:
        return {
            "ctr": float(self.ctr),
            "position_gain": self.position_gain,
            "scroll_depth": float(self.scroll_depth),
            "heatmap_cta": float(self.heatmap_cta),
            "time_on_page": float(self.time_on_page),
            "booking_starts": float(self.booking_starts),
        }


def _row_to_observation(row: dict) -> PageObservation:
    def num(name: str, default: float = 0.0) -> float:
        raw = row.get(name, default)
        if raw is None or raw == "":
            return default
        return float(raw)

    return PageObservation(
        page_id=str(row["page_id"]),
        day=int(float(row["day"])),
        group=str(row.get("group") or "control"),
        clicks=num("clicks"),
        impressions=num("impressions"),
        ctr=num("ctr"),
        position=num("position"),
        position_delta=num("position_delta"),
        scroll_depth=num("scroll_depth"),
        heatmap_cta=num("heatmap_cta"),
        time_on_page=num("time_on_page"),
        booking_starts=num("booking_starts"),
        http_404=num("http_404"),
        deindexed=num("deindexed"),
        noindex=num("noindex"),
    )


def load_json(path: str | Path) -> list[PageObservation]:
    """Load a JSON list, or an object with an ``observations`` list."""
    data = json.loads(Path(path).read_text())
    rows = data["observations"] if isinstance(data, dict) else data
    return [_row_to_observation(row) for row in rows]


def load_csv(path: str | Path) -> list[PageObservation]:
    with Path(path).open(newline="") as handle:
        return [_row_to_observation(row) for row in csv.DictReader(handle)]


def dump_json(path: str | Path, rows: list[PageObservation]) -> None:
    payload = [asdict(row) for row in rows]
    Path(path).write_text(json.dumps(payload))


def dump_csv(path: str | Path, rows: list[PageObservation]) -> None:
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: getattr(row, column) for column in COLUMNS})


def assign_cohorts(
    page_ids: list[str],
    recipe_keys: list[str],
    cohort_size: int,
    rng: random.Random,
    *,
    control_fraction: float = 0.25,
) -> dict[str, str | None]:
    """Assign pages to recipes. A fraction is left untouched (``None``).

    Pages are the unit. Each recipe receives at most `cohort_size` pages.
    Pages that do not fit a cohort stay untouched, with the control group.
    """
    if cohort_size < 1:
        raise ValueError("cohort_size must be positive")
    if not page_ids:
        return {}
    ids = list(page_ids)
    rng.shuffle(ids)
    n_control = 0 if not recipe_keys else max(1, int(round(len(ids) * control_fraction)))
    n_control = min(n_control, len(ids) - 1) if len(ids) > 1 and recipe_keys else n_control
    assignment: dict[str, str | None] = {page_id: None for page_id in ids[:n_control]}
    counts = {key: 0 for key in recipe_keys}
    cursor = 0
    for page_id in ids[n_control:]:
        placed = False
        for _ in range(len(recipe_keys)):
            key = recipe_keys[cursor % len(recipe_keys)]
            cursor += 1
            if counts[key] < cohort_size:
                assignment[page_id] = key
                counts[key] += 1
                placed = True
                break
        if not placed:
            assignment[page_id] = None
    return assignment


def composite(vector: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted sum of scaled metrics. Weights are renormalised."""
    total_w = sum(max(0.0, w) for w in weights.values()) or 1.0
    value = 0.0
    for name, weight in weights.items():
        if weight <= 0:
            continue
        scale = SCALES.get(name, 1.0) or 1.0
        raw = float(vector.get(name, 0.0)) / scale
        raw = max(-1.0, min(2.0, raw))
        value += (weight / total_w) * raw
    return value


def mean_se(values: list[float]) -> tuple[float, float, int]:
    """Sample mean and standard error. A single point has SE 1 (unusable)."""
    n = len(values)
    if n == 0:
        return 0.0, 1.0, 0
    mean = sum(values) / n
    if n == 1:
        return mean, 1.0, 1
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, (var / n) ** 0.5, n


