"""Agentic learning on top of the evolution engine.

A cycle sets a goal, recalls what was tried, asks a planner for hypotheses,
and lets the arena accept or reject them. The actuator can render the
survivor. It does not deploy.
"""

from .goal import Goal
from .loop import AgentSession, run_cycle

__all__ = ["AgentSession", "Goal", "run_cycle"]
