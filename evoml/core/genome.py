"""Genome interface. Mutation and randomness belong to the domain."""
from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from typing import Any


class Genome(ABC):
    """Something the arena can vary, score, and store.

    `crossover` is optional. Domains that do not recombine raise
    `NotImplementedError` from the default.
    """

    @abstractmethod
    def mutate(self, rng: random.Random) -> Genome:
        """Return a child. Must not modify `self`."""

    @classmethod
    @abstractmethod
    def random(cls, rng: random.Random) -> Genome:
        """A fresh challenger, not a child of the champion."""

    def crossover(self, other: Genome, rng: random.Random) -> Genome:
        raise NotImplementedError(f"{type(self).__name__} has no crossover")

    @abstractmethod
    def serialize(self) -> dict[str, Any]:
        """JSON-ready mapping. Stable for a given genome."""

    @classmethod
    @abstractmethod
    def deserialize(cls, data: dict[str, Any]) -> Genome:
        ...

    def key(self) -> str:
        return json.dumps(self.serialize(), sort_keys=True, default=str)
