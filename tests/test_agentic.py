"""Agentic learning loop. Offline: the Groq client is not opened."""
import csv
import json
import socket

import pytest

from evoml.agentic.ablate import run_ablation
from evoml.agentic.config import AgentConfig, load_config
from evoml.agentic.goal import Goal
from evoml.agentic.loop import AgentSession
from evoml.agentic.memory import CausalMemory
from evoml.agentic.planner import (
    GroqPlanner,
    Hypothesis,
    StubPlanner,
    apply_mutations,
    parse_hypotheses,
)
from evoml.domains.seo_pages.genome import PageRecipe
from evoml.domains.seo_pages.signals import PageObservation, dump_csv


def test_goal_decomposes_and_budget_is_a_constraint():
    from evoml.agentic.goal import Budget

    goal = Goal(
        metric="booking_starts",
        budget=Budget(max_pages_changed=3, cooldown_generations=2, blacklist=("fare_table",)),
    )
    names = [item.name for item in goal.decompose()]
    assert names == ["measure", "recall", "propose", "evolve", "act"]
    assert "3" in goal.decompose()[-1].detail
    with pytest.raises(ValueError):
        Budget(max_pages_changed=0)


def test_stub_planner_is_deterministic_and_respects_recalled_blocks():
    planner = StubPlanner()
    goal = Goal(metric="booking_starts")
    champion = PageRecipe.weak()
    first = planner.propose(goal=goal, memory_hits=[], audit_rows=[], champion=champion)
    second = planner.propose(goal=goal, memory_hits=[], audit_rows=[], champion=champion)
    assert first == second
    assert first[0].mutations[0]["section"] == "fare_table"
    assert "arena decides" in first[0].rationale
    blocked = [{"episode": {"content": "blocked_section:fare_table\nverdict: held"}}]
    nxt = planner.propose(goal=goal, memory_hits=blocked, audit_rows=[], champion=champion)
    assert nxt[0].mutations[0]["section"] == "how_to_book"
    child = apply_mutations(champion, first[0].mutations)
    assert "fare_table" in child.sections
    # A model is not allowed to smuggle a verdict in as an op.
    parsed = parse_hypotheses(
        '{"hypotheses":[{"rationale":"try the fare table","mutations":'
        '[{"op":"promote","winner":true},{"op":"section_add","section":"fare_table"}]}]}'
    )
    assert parsed[0].mutations == ({"op": "section_add", "section": "fare_table"},)


def test_groq_planner_parses_a_response_without_opening_a_socket(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("socket should not open")

    monkeypatch.setattr(socket, "create_connection", explode)
    monkeypatch.setattr(socket, "socket", explode)
    seen = {}

    def transport(url, headers, body):
        seen["url"] = url
        seen["headers"] = headers
        seen["body"] = body
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"hypotheses":[{"rationale":"Fare table matches booking intent.",'
                            '"mutations":[{"op":"section_add","section":"fare_table"}]}]}'
                        )
                    }
                }
            ]
        }

    planner = GroqPlanner(api_key="test-key", transport=transport)
    found = planner.propose(
        goal=Goal(metric="booking_starts"),
        memory_hits=[],
        audit_rows=[],
        champion=PageRecipe.weak(),
    )
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert seen["body"]["temperature"] == 0
    assert found[0].mutations[0]["section"] == "fare_table"
    offline = GroqPlanner(api_key="")
    with pytest.raises(RuntimeError):
        offline.propose(
            goal=Goal(metric="booking_starts"),
            memory_hits=[],
            audit_rows=[],
            champion=PageRecipe.weak(),
        )


def test_cycle_recalls_before_planning_and_does_not_deploy(tmp_path):
    order = []

    class Hook:
        def __init__(self):
            self.rows = []

        def recall(self, cue, limit=None):
            order.append("recall")
            return []

        def experience(self, content, **kwargs):
            order.append("experience")
            self.rows.append(content)
            return {"engram_id": "1"}

    class Planner:
        def propose(self, **kwargs):
            order.append("plan")
            assert kwargs["memory_hits"] == []
            return [
                Hypothesis(
                    "add the fare table and let the arena decide",
                    ({"op": "section_add", "section": "fare_table"},),
                )
            ]

    config = AgentConfig(
        seed=0,
        dry_run=True,
        planner="stub",
        meta=False,
        max_pages_changed=2,
        pages=["NA-00001", "NA-00002", "NA-00003", "NA-00004"],
        output_dir=str(tmp_path / "render"),
        audit_path=str(tmp_path / "audit.db"),
        cohort_size=8,
        generations=1,
    )
    session = AgentSession(config, planner=Planner(), memory=CausalMemory(Hook()))
    before = _bookings(session, session.champion)
    report = session.cycle()
    assert order[:3] == ["recall", "plan", "experience"]
    assert report.manifest["dry_run"] is True
    assert report.manifest["deployed"] is False
    assert report.manifest["pages_changed"] == 2
    assert list((tmp_path / "render").glob("*.html")) == []
    assert (tmp_path / "render" / "manifest.json").exists()
    assert report.promoted
    assert report.booking_mean > before
    assert "hypothesis:" in session.memory.hook.rows[0]
    assert "verdict:" in session.memory.hook.rows[0]


def test_blacklist_vetoes_a_proposal_the_planner_insists_on(tmp_path):
    class Blind:
        def propose(self, **kwargs):
            return [
                Hypothesis(
                    "add fare_table even though it is blacklisted",
                    ({"op": "section_add", "section": "fare_table"},),
                )
            ]

    config = AgentConfig(
        seed=1,
        dry_run=True,
        meta=False,
        blacklist=["fare_table"],
        output_dir=str(tmp_path / "render"),
        cohort_size=6,
        generations=1,
    )
    session = AgentSession(config, planner=Blind())
    report = session.cycle()
    assert report.proposal_keys
    assert report.proposal_keys[0] in report.rejected_keys
    assert report.proposal_keys[0] != session.champion.key()
    assert "fare_table" not in report.manifest["recipe"]["sections"]


def test_spiked_recipe_is_not_the_rendered_winner(tmp_path):
    config = AgentConfig(
        seed=2,
        dry_run=False,
        meta=False,
        output_dir=str(tmp_path / "html"),
        max_pages_changed=1,
        pages=["NA-00001", "NA-00002"],
        cohort_size=6,
    )
    session = AgentSession(config, planner=StubPlanner())
    toxic = PageRecipe.weak()
    toxic.schema_types = ["UnsafeExperiment"]
    journal = session.experiment.run(1, [toxic])
    assert toxic.key() in journal[-1].blacklisted or journal[-1].scores[toxic.key()].disqualified
    assert session.experiment.champion.causes_spike() is False
    report = session.actuator.apply(session.experiment.champion, session.pages(), dry_run=False, max_pages=1)
    assert report["deployed"] is False
    assert report["pages_changed"] == 1
    html = list((tmp_path / "html").glob("*.html"))
    assert len(html) == 1
    assert "UnsafeExperiment" not in html[0].read_text()


def test_file_sensor_does_not_invent_a_promotion(tmp_path):
    rows = []
    for index in range(4):
        rows.append(_observation("control", index, bookings=1.2))
    path = tmp_path / "signals.csv"
    dump_csv(path, rows)
    config = AgentConfig(
        signals=str(path),
        dry_run=True,
        meta=False,
        output_dir=str(tmp_path / "out"),
        pages=["NA-00001"],
        max_pages_changed=1,
    )
    report = AgentSession(config).cycle()
    assert report.promoted is False
    assert "export" in report.succession
    assert report.manifest["dry_run"] is True
    assert report.hypotheses


def test_ablation_report_is_simulated_and_reproducible(tmp_path):
    first = run_ablation(seed=0, cycles=2, cohort_size=6, max_pages=3, out_dir=tmp_path / "a")
    second = run_ablation(seed=0, cycles=2, cohort_size=6, max_pages=3, out_dir=tmp_path / "b")
    assert [row["arm"] for row in first] == [
        "full",
        "no_meta",
        "no_memory",
        "no_planner",
        "random_control",
    ]
    assert [row["booking_mean"] for row in first] == [row["booking_mean"] for row in second]
    by_arm = {row["arm"]: row for row in first}
    assert by_arm["no_planner"]["hypotheses"] == 0
    assert by_arm["full"]["hypotheses"] > 0
    assert by_arm["no_memory"]["recalled"] == 0
    assert by_arm["full"]["recalled"] > by_arm["no_memory"]["recalled"]
    assert by_arm["random_control"]["promotions"] == 0
    assert by_arm["random_control"]["pages_changed"] == 0
    assert by_arm["full"]["pages_changed"] <= 3
    assert by_arm["full"]["meta_booking_weight"] != by_arm["no_meta"]["meta_booking_weight"]
    text = (tmp_path / "a" / "ablation.md").read_text()
    assert "Simulated" in text
    assert "not a Search Console" in text
    with (tmp_path / "a" / "ablation.csv").open() as handle:
        loaded = list(csv.DictReader(handle))
    assert loaded[0]["simulated"] == "True"
    assert loaded[0]["arm"] == "full"
    assert pytest.approx(float(loaded[0]["booking_mean"])) == first[0]["booking_mean"]


def test_cli_one_cycle_on_the_example_shape(tmp_path, monkeypatch):
    config = {
        "domain": "seo_pages",
        "seed": 0,
        "dry_run": True,
        "planner": "stub",
        "meta": False,
        "signals": "simulator",
        "output_dir": str(tmp_path / "out"),
        "audit_path": str(tmp_path / "audit.db"),
        "max_pages_changed": 2,
        "pages": ["NA-00001", "NA-00002", "NA-00003"],
        "cohort_size": 4,
        "generations": 1,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    from evoml.agentic.cli import main

    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    code = main(["run", "--domain", "seo_pages", "--config", str(path)])
    assert code == 0
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["dry_run"] is True and manifest["deployed"] is False
    assert manifest["pages_changed"] == 2
    loaded = load_config(path)
    assert loaded.planner == "stub"


def _bookings(session: AgentSession, recipe: PageRecipe) -> float:
    rows = session.simulator.observations(recipe, cohort_size=session.config.cohort_size, day=0)
    return sum(row.booking_starts for row in rows) / len(rows)


def _observation(group: str, index: int, bookings: float) -> PageObservation:
    return PageObservation(
        page_id=f"{group}-{index}",
        day=1,
        group=group,
        clicks=1,
        impressions=10,
        ctr=0.1,
        position=5,
        position_delta=0,
        scroll_depth=0.4,
        heatmap_cta=0.2,
        time_on_page=40,
        booking_starts=bookings,
    )
