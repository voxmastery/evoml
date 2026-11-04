"""Dry run of the SEO adapter on simulated page data.

Prints the generation journal. The numbers are whatever this seed just
produced on the simulator. They are not Search Console measurements and
they are not a benchmark.
"""
from __future__ import annotations

import argparse

from evoml.domains.seo_pages.experiment import SEOExperiment
from evoml.domains.seo_pages.meta import MetaExperiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate SEO recipe evolution")
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--meta-generations", type=int, default=4)
    args = parser.parse_args()

    meta = MetaExperiment(seed=args.seed)
    meta_journal = meta.run(args.meta_generations)
    print("meta policy (simulated replay, not a measured result)")
    for decision in meta_journal:
        score = decision.scores.get(decision.winner_key)
        mean = f"{score.mean:.3f}" if score else "n/a"
        print(
            f"  gen {decision.generation}: {decision.succession} "
            f"mean_later_bookings={mean} promoted={decision.promoted}"
        )
    print(
        "  weights:",
        {k: round(v, 3) for k, v in meta.champion.weights.items()},
        "window", meta.champion.window_days,
        "se_mult", meta.champion.se_mult,
        "cohort", meta.champion.cohort_size,
        "mut_rate", round(meta.champion.mut_rate, 3),
    )

    pages = SEOExperiment(
        seed=args.seed,
        weights=meta.champion.weights,
        cohort_size=meta.champion.cohort_size,
        mut_rate=meta.champion.mut_rate,
        se_mult=meta.champion.se_mult,
    )
    print("page recipes (simulator, not a measured result)")
    for decision in pages.run(args.generations):
        score = decision.scores.get(decision.winner_key)
        mean = f"{score.mean:.3f}" if score else "n/a"
        print(
            f"  gen {decision.generation}: {decision.succession} "
            f"composite={mean} promoted={decision.promoted}"
        )
    print("  sections:", pages.champion.sections)
    print("  cta:", pages.champion.cta_placement, "links:", pages.champion.link_strategy)


if __name__ == "__main__":
    main()
