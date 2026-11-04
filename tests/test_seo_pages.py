"""SEO page recipes: promotion, control, guardrails, loaders, invented genes."""
import random

from evoml.core.arena import Arena, Contender
from evoml.core.memory import FluctlightMemory
from evoml.domains.seo_pages.evaluator import PageEvaluator, RecipeSimulator
from evoml.domains.seo_pages.experiment import SEOExperiment
from evoml.domains.seo_pages.genome import (
    MUT_RATE_MAX,
    MUT_RATE_MIN,
    WORD_MAX,
    WORD_MIN,
    PageRecipe,
    eval_invented,
    invent_gene,
    quality,
)
from evoml.domains.seo_pages.signals import (
    METRIC_NAMES,
    assign_cohorts,
    dump_csv,
    dump_json,
    load_csv,
    load_json,
)


def test_better_recipe_is_promoted_and_control_is_not(tmp_path):
    memory_rows = []

    class Brain:
        def experience(self, content, **kwargs):
            memory_rows.append((content, kwargs))
            return {"engram_id": "1"}

        def recall(self, cue, limit=None):
            return {"hits": [], "lanes_used": [], "mode": "lexical"}

        def checkpoint(self):
            return None

    experiment = SEOExperiment(
        seed=0,
        audit_path=tmp_path / "seo.db",
        memory=FluctlightMemory(brain=Brain()),
        verified=False,
    )
    start = quality(experiment.champion)
    journal = experiment.run(6)
    assert any(decision.promoted for decision in journal)
    assert quality(experiment.champion) > start
    assert all(decision.winner_key != "control" or not decision.promoted for decision in journal)
    assert experiment.audit.read()
    assert memory_rows
    assert memory_rows[0][1]["verified"] is False
    # the control arm is present on every generation and is not the winner
    for decision in journal:
        assert "control" in decision.scores
        assert decision.winner_key != "control"


def test_guardrail_blacklists_a_recipe_with_a_404_spike():
    toxic = PageRecipe.weak()
    toxic.schema_types = ["UnsafeExperiment"]
    clean = PageRecipe.random(random.Random(1))
    evaluator = PageEvaluator(
        weights={name: 1.0 for name in METRIC_NAMES},
        cohort_size=8,
        simulator=RecipeSimulator(seed=3, noise=0.01),
    )
    scores = evaluator.score([toxic, clean])
    assert scores[toxic.key()].disqualified
    assert scores[toxic.key()].detail["guardrail"] in ("http_404", "deindexed", "noindex")
    assert not scores[clean.key()].disqualified
    arena = Arena(name="seo_pages", require_beat_control=True, se_mode="difference")
    contenders = [
        Contender(clean.key(), "champion", clean),
        Contender(toxic.key(), "challenger", toxic),
        Contender("control", "control", PageRecipe.control()),
    ]
    decision = arena.decide(contenders, scores, generation=1, incumbent_key=clean.key())
    assert decision.winner_key == clean.key()
    assert toxic.key() in arena.blacklist
    assert not decision.promoted or decision.winner_key != toxic.key()


def test_invented_section_is_promoted_when_later_bookings_rise():
    weak = PageRecipe.weak()
    invented = weak.clone()
    invented.sections = ["hero", "section_from_booking_starts"]
    invented.invented = [{
        "kind": "section",
        "name": "section_from_booking_starts",
        "from": "booking_starts",
    }]
    weights = {name: 0.0 for name in METRIC_NAMES}
    weights["booking_starts"] = 1.0
    evaluator = PageEvaluator(
        weights=weights,
        cohort_size=8,
        simulator=RecipeSimulator(seed=5, noise=0.02),
    )
    scores = evaluator.score([weak, invented])
    assert scores[invented.key()].mean > scores[weak.key()].mean + scores[invented.key()].se
    arena = Arena(require_beat_control=True)
    contenders = [
        Contender(weak.key(), "champion", weak),
        Contender(invented.key(), "challenger", invented),
        Contender("control", "control", PageRecipe.control()),
    ]
    decision = arena.decide(contenders, scores, generation=1, incumbent_key=weak.key())
    assert decision.promoted
    assert decision.winner_key == invented.key()


def test_invent_gene_reads_the_report_and_mutation_stays_in_bounds():
    rng = random.Random(9)
    report = {name: 0.0 for name in (
        "clicks", "impressions", "ctr", "position_delta", "scroll_depth",
        "heatmap_cta", "time_on_page", "booking_starts",
    )}
    report["booking_starts"] = 0.9
    kinds = set()
    for _ in range(40):
        gene = invent_gene(rng, report)
        kinds.add(gene["kind"])
        if gene["kind"] == "section":
            assert gene["name"].startswith("section_from_")
            assert gene["from"]
        else:
            assert gene["op"] in ("neg", "abs", "mul", "sub", "div", "max", "min")
            value = eval_invented(gene, {"booking_starts": 2.0, "ctr": 0.1})
            assert -1e3 <= value <= 1e3
    assert kinds == {"section", "signal"}
    recipe = PageRecipe.weak()
    seen_invented = False
    for _ in range(200):
        recipe = recipe.mutate(rng, report, rate=2.0)
        assert WORD_MIN <= recipe.word_count <= WORD_MAX
        assert 0 <= recipe.faq_count <= 12
        assert MUT_RATE_MIN <= recipe.mut_rate <= MUT_RATE_MAX
        assert recipe.sections
        assert len(recipe.invented) <= 6
        seen_invented = seen_invented or any(g["kind"] == "section" for g in recipe.invented)
    assert seen_invented


def test_loaders_roundtrip_and_measured_rows_score_without_network(tmp_path):
    recipe = PageRecipe.weak()
    recipe.sections = ["hero", "fare_table", "how_to_book"]
    recipe.cohort_id = "fare-v1"
    assert recipe.key() == "fare-v1"
    rows = []
    for index in range(6):
        rows.append(_row("fare-v1", index, day=3, bookings=4.0, ctr=0.08))
        rows.append(_row("control", index, day=3, bookings=1.0, ctr=0.03))
        rows.append(_row("fare-v1", index, day=40, bookings=0.0, ctr=0.0))
    dump_json(tmp_path / "signals.json", rows)
    dump_csv(tmp_path / "signals.csv", rows)
    from_json = load_json(tmp_path / "signals.json")
    from_csv = load_csv(tmp_path / "signals.csv")
    assert len(from_json) == len(from_csv) == len(rows)
    assert from_csv[0].page_id == rows[0].page_id
    assert from_json[1].group == "control"
    evaluator = PageEvaluator(
        weights={name: (1.0 if name == "booking_starts" else 0.0) for name in METRIC_NAMES},
        cohort_size=6,
        observations=from_csv,
        window=(0, 14),
    )
    scores = evaluator.score([recipe])
    assert recipe.key() in scores
    assert scores[recipe.key()].n == 6
    assert scores[recipe.key()].mean > scores["control"].mean
    # the day-40 rows are outside the window and do not pull the mean down
    assert scores[recipe.key()].mean > 0.5


def test_measured_404_spike_disqualifies():
    recipe = PageRecipe.weak()
    recipe.cohort_id = "toxic-v1"
    rows = []
    for index in range(4):
        rows.append(_row("toxic-v1", index, day=2, bookings=3.0, ctr=0.1, http_404=0.2))
        rows.append(_row("control", index, day=2, bookings=1.0, ctr=0.04, http_404=0.0))
    evaluator = PageEvaluator(
        weights={name: 1.0 for name in METRIC_NAMES},
        cohort_size=4,
        observations=rows,
    )
    scores = evaluator.score([recipe])
    assert scores[recipe.key()].disqualified
    assert scores[recipe.key()].detail["guardrail"] == "http_404"


def test_cohort_assignment_leaves_a_control_group_and_caps_size():
    pages = [f"NA-{i:05d}" for i in range(10)]
    assignment = assign_cohorts(pages, ["fare", "route"], cohort_size=3, rng=random.Random(1))
    assert set(assignment) == set(pages)
    untouched = [page for page, recipe in assignment.items() if recipe is None]
    assert untouched
    assert sum(1 for recipe in assignment.values() if recipe == "fare") <= 3
    assert sum(1 for recipe in assignment.values() if recipe == "route") <= 3


def _row(group, index, *, day, bookings, ctr, http_404=0.0):
    from evoml.domains.seo_pages.signals import PageObservation

    return PageObservation(
        page_id=f"{group[:12]}-{index}",
        day=day,
        group=group,
        clicks=ctr * 100,
        impressions=100,
        ctr=ctr,
        position=5.0,
        position_delta=-0.4,
        scroll_depth=0.6,
        heatmap_cta=0.3,
        time_on_page=80.0,
        booking_starts=bookings,
        http_404=http_404,
    )
