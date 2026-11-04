"""Page-recipe genome for an ambulance-booking landing page.

A recipe is the ordered set of sections, a length target, FAQ count,
title and meta templates, an internal-link strategy, where the call to
action sits, and the schema.org types. Invented genes are new section
types or signal expressions, built from a report of which observed
signals move with bookings — the same role invented features play in
the forecaster.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

from evoml.core.genome import Genome

SECTION_TYPES = (
    "hero",
    "fare_table",
    "route_map",
    "faq",
    "hospital_info",
    "how_to_book",
    "trust_signals",
)

# Sections the simulator treats as useful on an ambulance booking page.
# This is the data-generating process for tests, not a measured lift.
HELPFUL_SECTIONS = {
    "hero": 0.2,
    "route_map": 0.5,
    "faq": 0.4,
    "hospital_info": 0.9,
    "fare_table": 1.6,
    "how_to_book": 1.5,
    "trust_signals": 1.1,
}

LINK_STRATEGIES = ("hub", "sibling", "hospital", "sparse")
CTA_PLACEMENTS = ("hero", "mid", "end", "sticky")
SCHEMA_TYPES = (
    "MedicalBusiness",
    "FAQPage",
    "Service",
    "BreadcrumbList",
    "Hospital",
)
TITLE_TEMPLATES = (
    "{keyword} | Namma Ambulance Bangalore",
    "Book {keyword} — ambulance in Bangalore",
    "{keyword}: fare, route and how to book",
)
META_TEMPLATES = (
    "Book an ambulance for {keyword} in Bangalore. Fares, the hospital route and a 24/7 number.",
    "{keyword} ambulance booking in Bangalore. Fare table, route and how to book.",
    "24/7 {keyword} in Bangalore. See the fare, the route and how to book.",
)

SIGNAL_NAMES = (
    "clicks",
    "impressions",
    "ctr",
    "position_delta",
    "scroll_depth",
    "heatmap_cta",
    "time_on_page",
    "booking_starts",
)
UNARY_OPS = ("neg", "abs")
BINARY_OPS = ("mul", "sub", "div", "max", "min")
MAX_INVENTED = 6
WORD_MIN = 400
WORD_MAX = 1800
WORD_STEP = 100
MUT_RATE_MIN = 0.5
MUT_RATE_MAX = 3.0


def quality(recipe: PageRecipe) -> float:
    """Latent page quality used only by the simulator."""
    q = 0.3
    seen: set[str] = set()
    for section in recipe.sections:
        if section in seen:
            continue
        seen.add(section)
        q += HELPFUL_SECTIONS.get(section, 0.0)
        if section.startswith("section_from_booking"):
            q += 2.2
    if recipe.cta_placement in ("hero", "sticky"):
        q += 0.5
    if recipe.link_strategy == "hospital":
        q += 0.35
    if "faq" in recipe.sections and recipe.faq_count >= 4:
        q += 0.25
    return q


def eval_invented(expr: dict, signals: dict[str, float]) -> float:
    """Evaluate one invented signal gene. Section genes are not numeric."""
    if expr.get("kind") != "signal":
        return 0.0
    op = expr.get("op")
    args = expr.get("args") or []
    a = float(signals.get(args[0], 0.0)) if args else 0.0
    if op == "neg":
        value = -a
    elif op == "abs":
        value = abs(a)
    else:
        b = float(signals.get(args[1], 0.0)) if len(args) > 1 else 0.0
        if op == "mul":
            value = a * b
        elif op == "sub":
            value = a - b
        elif op == "div":
            value = a / (abs(b) + 1e-6)
        elif op == "max":
            value = max(a, b)
        else:
            value = min(a, b)
    return max(-1e3, min(1e3, float(value)))


def _pick_signal(rng: random.Random, report: dict[str, float] | None) -> str:
    if report and rng.random() < 0.7:
        ranked = sorted(SIGNAL_NAMES, key=lambda name: -abs(report.get(name, 0.0)))
        return rng.choice(ranked[:3])
    return rng.choice(SIGNAL_NAMES)


def invent_gene(rng: random.Random, report: dict[str, float] | None = None) -> dict:
    """Invent a section type or a signal expression from the gene report.

    Half the time the gene is a new section whose name records the signal
    it was built from (``section_from_booking_starts``). Otherwise it is a
    small expression over observed signals, the same shape as the
    forecaster's synth genes.
    """
    if rng.random() < 0.5:
        signal = _pick_signal(rng, report)
        return {"kind": "section", "name": f"section_from_{signal}", "from": signal}
    if rng.random() < 0.35:
        return {"kind": "signal", "op": rng.choice(UNARY_OPS), "args": [_pick_signal(rng, report)]}
    return {
        "kind": "signal",
        "op": rng.choice(BINARY_OPS),
        "args": [_pick_signal(rng, report), _pick_signal(rng, report)],
    }


def _clamp_rate(rate: float) -> float:
    return min(MUT_RATE_MAX, max(MUT_RATE_MIN, rate))


@dataclass
class PageRecipe(Genome):
    sections: list[str]
    word_count: int
    faq_count: int
    title_template: str
    meta_template: str
    link_strategy: str
    cta_placement: str
    schema_types: list[str]
    invented: list[dict] = field(default_factory=list)
    mut_rate: float = 1.0
    cohort_id: str | None = None
    is_control: bool = False

    def causes_spike(self) -> bool:
        """Recipes marked with the unsafe schema are the guardrail fixture.

        Ordinary mutation never adds that schema. A loader-backed run
        blacklists from measured 404 / deindex / noindex rates instead.
        """
        return "UnsafeExperiment" in self.schema_types

    def mutate(
        self,
        rng: random.Random,
        report: dict[str, float] | None = None,
        rate: float | None = None,
    ) -> PageRecipe:
        child = self.clone()
        # A child has not been deployed, so it does not keep the parent's
        # cohort id. Measured rows stay attached to the recipe that earned them.
        child.cohort_id = None
        child.is_control = False
        edits = max(1, round(rate if rate is not None else child.mut_rate))
        for _ in range(edits):
            child._edit(rng, report)
        child._clamp()
        return child

    def _edit(self, rng: random.Random, report: dict[str, float] | None) -> None:
        can_invent = len(self.invented) < MAX_INVENTED
        ops = [
            "section_add",
            "section_drop",
            "reorder",
            "word",
            "faq",
            "title",
            "meta",
            "link",
            "cta",
            "schema",
            "invent" if can_invent else "section_add",
            "kill" if self.invented else "section_add",
            "rate",
        ]
        op = rng.choice(ops)
        if op == "section_add":
            choices = [s for s in SECTION_TYPES if s not in self.sections] or list(SECTION_TYPES)
            self.sections.append(rng.choice(choices))
        elif op == "section_drop" and len(self.sections) > 1:
            self.sections.pop(rng.randrange(len(self.sections)))
        elif op == "reorder" and len(self.sections) > 1:
            i, j = rng.sample(range(len(self.sections)), 2)
            self.sections[i], self.sections[j] = self.sections[j], self.sections[i]
        elif op == "word":
            self.word_count += rng.choice([-WORD_STEP, WORD_STEP])
        elif op == "faq":
            self.faq_count += rng.choice([-1, 1])
        elif op == "title":
            self.title_template = rng.choice(TITLE_TEMPLATES)
        elif op == "meta":
            self.meta_template = rng.choice(META_TEMPLATES)
        elif op == "link":
            self.link_strategy = rng.choice(LINK_STRATEGIES)
        elif op == "cta":
            self.cta_placement = rng.choice(CTA_PLACEMENTS)
        elif op == "schema":
            current = set(self.schema_types)
            pick = rng.choice(SCHEMA_TYPES)
            if pick in current and len(current) > 1:
                current.remove(pick)
            else:
                current.add(pick)
            self.schema_types = sorted(current)
        elif op == "invent":
            gene = invent_gene(rng, report)
            self.invented = self.invented + [gene]
            if gene["kind"] == "section" and gene["name"] not in self.sections:
                self.sections.append(gene["name"])
        elif op == "kill" and self.invented:
            index = rng.randrange(len(self.invented))
            dropped = self.invented[index]
            self.invented = [g for i, g in enumerate(self.invented) if i != index]
            if dropped.get("kind") == "section":
                self.sections = [s for s in self.sections if s != dropped.get("name")]
        else:
            self.mut_rate = self.mut_rate * rng.choice([1 / 1.3, 1.3])

    def _clamp(self) -> None:
        if not self.sections:
            self.sections = ["hero"]
        self.word_count = int(min(WORD_MAX, max(WORD_MIN, self.word_count)))
        self.faq_count = int(min(12, max(0, self.faq_count)))
        self.mut_rate = _clamp_rate(self.mut_rate)
        self.invented = list(self.invented)[:MAX_INVENTED]
        if self.link_strategy not in LINK_STRATEGIES:
            self.link_strategy = "hub"
        if self.cta_placement not in CTA_PLACEMENTS:
            self.cta_placement = "end"

    @classmethod
    def random(cls, rng: random.Random) -> PageRecipe:
        k = rng.randint(3, len(SECTION_TYPES))
        sections = rng.sample(list(SECTION_TYPES), k)
        schema = rng.sample(list(SCHEMA_TYPES), rng.randint(1, 3))
        return cls(
            sections=sections,
            word_count=rng.choice(list(range(WORD_MIN, WORD_MAX + 1, WORD_STEP))),
            faq_count=rng.randint(2, 8),
            title_template=rng.choice(TITLE_TEMPLATES),
            meta_template=rng.choice(META_TEMPLATES),
            link_strategy=rng.choice(LINK_STRATEGIES),
            cta_placement=rng.choice(CTA_PLACEMENTS),
            schema_types=sorted(schema),
            invented=[],
            mut_rate=1.0,
        )

    @classmethod
    def weak(cls) -> PageRecipe:
        """A thin recipe used as the starting champion in the simulator."""
        return cls(
            sections=["hero"],
            word_count=500,
            faq_count=1,
            title_template=TITLE_TEMPLATES[0],
            meta_template=META_TEMPLATES[0],
            link_strategy="sparse",
            cta_placement="end",
            schema_types=["Service"],
            invented=[],
            mut_rate=1.0,
        )

    @classmethod
    def control(cls) -> PageRecipe:
        recipe = cls.weak()
        recipe.is_control = True
        return recipe

    def crossover(self, other: Genome, rng: random.Random) -> PageRecipe:
        if not isinstance(other, PageRecipe):
            raise TypeError("crossover expects another PageRecipe")
        cut = rng.randint(1, max(1, len(self.sections)))
        sections = list(self.sections[:cut])
        for section in other.sections:
            if section not in sections:
                sections.append(section)
        child = self.clone()
        child.sections = sections[: len(SECTION_TYPES) + MAX_INVENTED]
        if rng.random() < 0.5:
            child.cta_placement = other.cta_placement
        if rng.random() < 0.5:
            child.link_strategy = other.link_strategy
        child.faq_count = (self.faq_count + other.faq_count) // 2
        child.cohort_id = None
        child.is_control = False
        child._clamp()
        return child

    def clone(self) -> PageRecipe:
        return PageRecipe.deserialize(self.serialize())

    def serialize(self) -> dict[str, Any]:
        return {
            "sections": list(self.sections),
            "word_count": int(self.word_count),
            "faq_count": int(self.faq_count),
            "title_template": self.title_template,
            "meta_template": self.meta_template,
            "link_strategy": self.link_strategy,
            "cta_placement": self.cta_placement,
            "schema_types": list(self.schema_types),
            "invented": json.loads(json.dumps(self.invented)),
            "mut_rate": round(float(self.mut_rate), 6),
            "cohort_id": self.cohort_id,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> PageRecipe:
        recipe = cls(
            sections=list(data["sections"]),
            word_count=int(data["word_count"]),
            faq_count=int(data["faq_count"]),
            title_template=str(data["title_template"]),
            meta_template=str(data["meta_template"]),
            link_strategy=str(data["link_strategy"]),
            cta_placement=str(data["cta_placement"]),
            schema_types=list(data.get("schema_types") or []),
            invented=json.loads(json.dumps(data.get("invented") or [])),
            mut_rate=float(data.get("mut_rate", 1.0)),
            cohort_id=data.get("cohort_id"),
        )
        recipe._clamp()
        return recipe

    def key(self) -> str:
        if self.is_control:
            return "control"
        if self.cohort_id:
            return str(self.cohort_id)
        return super().key()
