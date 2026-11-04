"""Planners propose hypotheses. They do not promote anything.

The default remote provider speaks Groq's OpenAI-compatible chat API.
Tests use `StubPlanner` or a Groq planner with an injected transport, so
the suite never opens a socket.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from evoml.domains.seo_pages.genome import (
    CTA_PLACEMENTS,
    LINK_STRATEGIES,
    SECTION_TYPES,
    SIGNAL_NAMES,
    PageRecipe,
)

from .goal import Goal

# Highest latent weight first. The stub walks this list; it is the
# simulator's own section table, used so offline proposals are reproducible.
_STUB_ORDER = (
    "fare_table",
    "how_to_book",
    "trust_signals",
    "hospital_info",
    "route_map",
    "faq",
    "hero",
)

_SECTION_FROM = re.compile(r"^section_from_([a-z0-9_]+)$")


@dataclass(frozen=True)
class Hypothesis:
    """A proposed change and the reason it was proposed.

    There is no score and no promotion flag. The arena assigns both.
    """

    rationale: str
    mutations: tuple[dict[str, Any], ...]


class Planner(Protocol):
    def propose(
        self,
        *,
        goal: Goal,
        memory_hits: Sequence[Any],
        audit_rows: Sequence[dict],
        champion: PageRecipe,
    ) -> list[Hypothesis]:
        """Read memory and the audit, then suggest mutations of `champion`."""


class StubPlanner:
    """Deterministic offline planner.

    It proposes the next helpful section that is not already on the
    champion and not named in a recalled blacklist. When every standard
    section is present it proposes one invented section gene. The same
    inputs always yield the same hypothesis.
    """

    def propose(
        self,
        *,
        goal: Goal,
        memory_hits: Sequence[Any],
        audit_rows: Sequence[dict],
        champion: PageRecipe,
    ) -> list[Hypothesis]:
        blocked = blocked_sections(memory_hits) | set(goal.budget.blacklist)
        present = set(champion.sections)
        recalled = len(list(memory_hits))
        audited = len(list(audit_rows))
        for section in _STUB_ORDER:
            if section in present or section in blocked:
                continue
            return [
                Hypothesis(
                    rationale=(
                        f"Recall saw {recalled} episode(s) and the audit has {audited} row(s). "
                        f"{section} is missing from the champion and is not blacklisted. "
                        f"Proposing it as a hypothesis for {goal.metric}; the arena decides."
                    ),
                    mutations=({"op": "section_add", "section": section},),
                )
            ]
        invented = "section_from_booking_starts"
        if invented not in present and invented not in blocked:
            return [
                Hypothesis(
                    rationale=(
                        f"Recall saw {recalled} episode(s). Standard sections are already "
                        f"on the page or blocked. Proposing an invented section built from "
                        f"booking_starts. The arena decides whether it is promoted."
                    ),
                    mutations=(
                        {
                            "op": "invent_section",
                            "name": invented,
                            "from": "booking_starts",
                        },
                    ),
                )
            ]
        return [
            Hypothesis(
                rationale=(
                    f"Recall saw {recalled} episode(s). No further section is eligible. "
                    "Proposing a call-to-action move to the hero. The arena decides."
                ),
                mutations=({"op": "set_cta", "value": "hero"},),
            )
        ]


class GroqPlanner:
    """Groq chat completions (OpenAI-compatible).

    `GROQ_API_KEY` is required. `GROQ_BASE_URL` defaults to
    ``https://api.groq.com/openai/v1`` and `GROQ_MODEL` defaults to
    ``llama-3.3-70b-versatile``. `transport`, when set, is called instead
    of the network: ``transport(url, headers, body) -> parsed JSON``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        base_url: str | None = None,
        transport: Callable[[str, dict[str, str], dict], dict] | None = None,
        timeout: float = 30.0,
    ):
        self.api_key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY", "")
        self.model = model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.base_url = (
            base_url or os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        ).rstrip("/")
        self.transport = transport
        self.timeout = timeout

    def propose(
        self,
        *,
        goal: Goal,
        memory_hits: Sequence[Any],
        audit_rows: Sequence[dict],
        champion: PageRecipe,
    ) -> list[Hypothesis]:
        if not self.api_key and self.transport is None:
            raise RuntimeError(
                "GROQ_API_KEY is not set. The Groq planner does not run without it. "
                "Use planner 'stub' for an offline cycle."
            )
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You propose landing-page experiments. You do not decide "
                        "whether a change wins. Return JSON only: "
                        '{"hypotheses":[{"rationale":"...","mutations":[{"op":"section_add","section":"fare_table"}]}]}. '
                        "Allowed ops: section_add, section_drop, set_cta, set_link, set_faq, "
                        "invent_section, invent_signal. Do not include a score or a verdict."
                    ),
                },
                {
                    "role": "user",
                    "content": _prompt(goal, memory_hits, audit_rows, champion),
                },
            ],
        }
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.transport is not None:
            body = self.transport(url, headers, payload)
        else:
            body = _post_json(url, headers, payload, self.timeout)
        content = body["choices"][0]["message"]["content"]
        return parse_hypotheses(content)


def make_planner(name: str, **kwargs: Any) -> Planner:
    """Build a planner. ``groq`` is the remote default; ``stub`` is offline."""
    if name == "stub":
        return StubPlanner()
    if name == "groq":
        return GroqPlanner(**kwargs)
    raise ValueError(f"unknown planner {name!r}; expected 'groq' or 'stub'")


def parse_hypotheses(text: str) -> list[Hypothesis]:
    """Pull a hypotheses list out of a model message. Invalid ops are dropped."""
    raw = _extract_json(text)
    items = raw.get("hypotheses") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        raise ValueError("planner response has no hypotheses list")
    hypotheses: list[Hypothesis] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        mutations = tuple(
            mutation
            for mutation in item.get("mutations") or []
            if isinstance(mutation, dict) and _mutation_ok(mutation)
        )
        rationale = str(item.get("rationale") or "").strip()
        if not rationale or not mutations:
            continue
        hypotheses.append(Hypothesis(rationale=rationale, mutations=mutations))
    if not hypotheses:
        raise ValueError("planner response contained no usable hypotheses")
    return hypotheses


def apply_mutations(recipe: PageRecipe, mutations: Sequence[dict[str, Any]]) -> PageRecipe:
    """Return a child recipe. Unknown or forbidden edits are skipped."""
    child = recipe.clone()
    child.cohort_id = None
    child.is_control = False
    for mutation in mutations:
        if not _mutation_ok(mutation):
            continue
        op = mutation["op"]
        if op == "section_add":
            section = mutation["section"]
            if section not in child.sections:
                child.sections.append(section)
        elif op == "section_drop" and len(child.sections) > 1:
            child.sections = [s for s in child.sections if s != mutation["section"]]
        elif op == "set_cta":
            child.cta_placement = mutation["value"]
        elif op == "set_link":
            child.link_strategy = mutation["value"]
        elif op == "set_faq":
            child.faq_count = int(mutation["value"])
        elif op == "invent_section":
            gene = {
                "kind": "section",
                "name": mutation["name"],
                "from": mutation["from"],
            }
            child.invented = list(child.invented) + [gene]
            if gene["name"] not in child.sections:
                child.sections.append(gene["name"])
        elif op == "invent_signal":
            gene = {
                "kind": "signal",
                "op": mutation["signal_op"],
                "args": list(mutation["args"]),
            }
            child.invented = list(child.invented) + [gene]
    child._clamp()
    return child


def blocked_sections(memory_hits: Sequence[Any]) -> set[str]:
    """Sections a prior episode marked blacklisted."""
    blocked: set[str] = set()
    for hit in memory_hits:
        text = _hit_text(hit)
        for match in re.finditer(r"blocked_section:([A-Za-z0-9_]+)", text):
            blocked.add(match.group(1))
    return blocked


def _mutation_ok(mutation: dict[str, Any]) -> bool:
    op = mutation.get("op")
    if op == "section_add":
        return _section_ok(mutation.get("section"))
    if op == "section_drop":
        return _section_ok(mutation.get("section"))
    if op == "set_cta":
        return mutation.get("value") in CTA_PLACEMENTS
    if op == "set_link":
        return mutation.get("value") in LINK_STRATEGIES
    if op == "set_faq":
        try:
            value = int(mutation.get("value"))
        except (TypeError, ValueError):
            return False
        return 0 <= value <= 12
    if op == "invent_section":
        name = str(mutation.get("name") or "")
        match = _SECTION_FROM.match(name)
        return bool(match) and match.group(1) in SIGNAL_NAMES
    if op == "invent_signal":
        args = mutation.get("args")
        return (
            mutation.get("signal_op") in ("neg", "abs", "mul", "sub", "div", "max", "min")
            and isinstance(args, (list, tuple))
            and 1 <= len(args) <= 2
            and all(arg in SIGNAL_NAMES for arg in args)
        )
    return False


def _section_ok(name: object) -> bool:
    if not isinstance(name, str):
        return False
    if name in SECTION_TYPES:
        return True
    match = _SECTION_FROM.match(name)
    return bool(match) and match.group(1) in SIGNAL_NAMES


def _extract_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    blob = fenced.group(1) if fenced else text
    start = blob.find("{")
    end = blob.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("planner response did not contain JSON")
    parsed = json.loads(blob[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("planner JSON was not an object")
    return parsed


def _prompt(goal: Goal, memory_hits: Sequence[Any], audit_rows: Sequence[dict], champion: PageRecipe) -> str:
    subgoals = "\n".join(f"- {item.name}: {item.detail}" for item in goal.decompose())
    memories = "\n".join(_hit_text(hit) for hit in list(memory_hits)[:8]) or "(none)"
    audit = "\n".join(
        f"gen {row.get('generation')}: {row.get('succession')} champion={row.get('champion')}"
        for row in list(audit_rows)[:8]
    ) or "(none)"
    return (
        f"Goal metric: {goal.metric} ({goal.direction})\n"
        f"Subgoals:\n{subgoals}\n"
        f"Champion sections: {champion.sections}\n"
        f"CTA: {champion.cta_placement} links: {champion.link_strategy} faq: {champion.faq_count}\n"
        f"Blacklist: {list(goal.budget.blacklist)}\n"
        f"Recalled episodes:\n{memories}\n"
        f"Audit:\n{audit}\n"
        "Propose at most two hypotheses."
    )


def _hit_text(hit: Any) -> str:
    if isinstance(hit, str):
        return hit
    if isinstance(hit, dict):
        episode = hit.get("episode")
        if isinstance(episode, dict) and episode.get("content"):
            return str(episode["content"])
        if hit.get("content"):
            return str(hit["content"])
        return json.dumps(hit, sort_keys=True, default=str)
    return str(hit)


def _post_json(url: str, headers: dict[str, str], payload: dict, timeout: float) -> dict:
    import urllib.request

    data = json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())
