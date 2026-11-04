"""Meta-genome: the SEO policy is evolved by the same arena."""
import random

from evoml.domains.seo_pages.genome import PageRecipe, quality
from evoml.domains.seo_pages.meta import (
    MetaEvaluator,
    MetaExperiment,
    MetaGenome,
    simulate_bangalore_history,
)
from evoml.domains.seo_pages.signals import METRIC_NAMES, PageObservation
from evoml.domains.seo_pages.experiment import SEOExperiment


def test_meta_tuning_promotes_the_policy_that_predicted_later_bookings(tmp_path):
    history = simulate_bangalore_history(seed=0, noise=0.05)
    experiment = MetaExperiment(history=history, seed=0, audit_path=tmp_path / "meta.db")
    start = MetaGenome.ctr_start()
    assert experiment.evaluator.replay(start).spiked
    journal = experiment.run(4)
    final = experiment.champion
    replay = experiment.evaluator.replay(final)
    assert any(decision.promoted for decision in journal)
    assert not replay.spiked
    assert final.weights["booking_starts"] > final.weights["ctr"]
    assert replay.later
    control = experiment.evaluator.score([MetaGenome.control()])["control"]
    final_score = experiment.evaluator.score([final])[final.key()]
    assert final_score.mean > control.mean + final_score.se
    assert all(decision.winner_key != "control" for decision in journal)
    assert experiment.audit.read()[0]["arena"] == "seo_meta"
    # the promoted policy then drives the page arena
    pages = SEOExperiment(
        seed=1,
        weights=final.weights,
        cohort_size=final.cohort_size,
        mut_rate=final.mut_rate,
        se_mult=final.se_mult,
    )
    before = quality(PageRecipe.weak())
    pages.run(5)
    assert quality(pages.champion) > before


def test_window_length_changes_which_recipe_is_chosen():
    # A 6-day CTR spike sits at the end of a 14-day window and is diluted
    # across 28 days. Bookings stay with the steady cohort the whole time.
    rows = []
    for day in range(42):
        spike = day >= 36
        rows.extend(_flat("steady", day, bookings=4.0, ctr=0.08, pages=6))
        rows.extend(
            _flat(
                "decoy",
                day,
                bookings=1.0,
                ctr=0.20 if spike else 0.04,
                pages=6,
            )
        )
        rows.extend(_flat("control", day, bookings=1.1, ctr=0.04, pages=6))
    evaluator = MetaEvaluator(rows, groups=("steady", "decoy"), decision_days=(42,))
    ctr_only = {name: 0.0 for name in METRIC_NAMES}
    ctr_only["ctr"] = 1.0
    short = MetaGenome(weights=ctr_only, window_days=14, se_mult=1.0, cohort_size=6)
    long = MetaGenome(weights=ctr_only, window_days=28, se_mult=1.0, cohort_size=6)
    short_choice, _ = evaluator.choose(short, 42, incumbent="steady")
    long_choice, _ = evaluator.choose(long, 42, incumbent="steady")
    assert short_choice == "decoy"
    assert long_choice == "steady"


def test_cohort_size_ignores_a_leading_outlier_only_when_the_cohort_is_wide():
    rows = []
    for index in range(8):
        bookings = 0.0 if index < 2 else 5.0
        rows.append(_page("good", day=0, index=index, bookings=bookings, ctr=0.05))
        rows.append(_page("weak", day=0, index=index, bookings=2.0, ctr=0.05))
        rows.append(_page("control", day=0, index=index, bookings=1.2, ctr=0.04))
    weights = {name: 0.0 for name in METRIC_NAMES}
    weights["booking_starts"] = 1.0
    evaluator = MetaEvaluator(rows, groups=("good", "weak"), decision_days=(14,))
    narrow = MetaGenome(weights=weights, window_days=14, cohort_size=2, se_mult=1.0)
    wide = MetaGenome(weights=weights, window_days=14, cohort_size=8, se_mult=1.0)
    narrow_choice, _ = evaluator.choose(narrow, 14, incumbent="weak")
    wide_choice, _ = evaluator.choose(wide, 14, incumbent="weak")
    assert narrow_choice == "weak"
    assert wide_choice == "good"


def test_promotion_threshold_blocks_a_thin_margin():
    rows = []
    for index in range(8):
        spread = (index - 3.5) * 0.35
        rows.append(_page("good", day=0, index=index, bookings=2.0 + spread, ctr=0.05))
        rows.append(_page("weak", day=0, index=index, bookings=1.55 + spread, ctr=0.05))
        rows.append(_page("control", day=0, index=index, bookings=1.2, ctr=0.04))
    weights = {name: 0.0 for name in METRIC_NAMES}
    weights["booking_starts"] = 1.0
    evaluator = MetaEvaluator(rows, groups=("good", "weak"), decision_days=(14,))
    loose = MetaGenome(weights=weights, window_days=14, cohort_size=8, se_mult=0.5)
    strict = MetaGenome(weights=weights, window_days=14, cohort_size=8, se_mult=3.0)
    assert evaluator.choose(loose, 14, incumbent="weak")[0] == "good"
    assert evaluator.choose(strict, 14, incumbent="weak")[0] == "weak"


def test_mut_rate_scores_that_many_page_variants():
    history = simulate_bangalore_history(seed=1, days=30, pages=4, noise=0.0)
    seen = []

    def source(recipe, start, end, cohort_size):
        seen.append((recipe.key(), start, end, cohort_size))
        return []

    genome = MetaGenome.ctr_start()
    genome.mut_rate = 3.0
    evaluator = MetaEvaluator(
        history, decision_days=(14, 28), proposal_source=source,
    )
    replay = evaluator.replay(genome, rng=random.Random(0))
    assert replay.proposal_count == 2 * 3
    assert len(seen) == 2 * 3


def _flat(group, day, *, bookings, ctr, pages):
    return [
        _page(group, day=day, index=index, bookings=bookings, ctr=ctr)
        for index in range(pages)
    ]


def _page(group, *, day, index, bookings, ctr):
    return PageObservation(
        page_id=f"{group}-p{index:02d}",
        day=day,
        group=group,
        clicks=ctr * 100,
        impressions=100,
        ctr=ctr,
        position=5.0,
        position_delta=-0.2,
        scroll_depth=0.5,
        heatmap_cta=0.2,
        time_on_page=60.0,
        booking_starts=bookings,
    )
