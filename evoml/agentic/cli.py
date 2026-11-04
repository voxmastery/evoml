"""`evoml-agent` command line. One cycle per `run`, for cron."""
from __future__ import annotations

import argparse
import json
import sys

from .ablate import run_ablation
from .config import load_config
from .loop import AgentSession


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evoml-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="one cycle: sense, recall, plan, evolve, act, log")
    run_parser.add_argument("--domain", required=True)
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--planner", choices=("stub", "groq"), default=None)

    ablate_parser = sub.add_parser("ablate", help="simulator ablation; writes csv and markdown")
    ablate_parser.add_argument("--out", required=True)
    ablate_parser.add_argument("--seed", type=int, default=0)
    ablate_parser.add_argument("--cycles", type=int, default=2)

    args = parser.parse_args(argv)
    if args.command == "ablate":
        rows = run_ablation(seed=args.seed, cycles=args.cycles, out_dir=args.out)
        print(f"wrote {args.out}/ablation.csv and ablation.md ({len(rows)} arms, simulated)")
        return 0

    config = load_config(args.config)
    if args.domain != config.domain:
        print(
            f"config domain is {config.domain!r}; --domain is {args.domain!r}",
            file=sys.stderr,
        )
        return 2
    if args.planner:
        config.planner = args.planner
    # Cron runs one cycle per invocation. `cycles` in the file is ignored here
    # so a second schedule tick does not replay the whole horizon in-process.
    session = AgentSession(config)
    report = session.cycle()
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
