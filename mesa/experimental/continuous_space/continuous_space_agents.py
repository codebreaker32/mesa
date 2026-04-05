"""Continuous space agents."""

from __future__ import annotations

from itertools import compress
from typing import Protocol

import numpy as np

from mesa.agent import Agent
from mesa.experimental.continuous_space import ContinuousSpace


class HasPositionProtocol(Protocol):
    position: np.ndarray

class ContinuousSpaceAgent(Agent):
    __slots__ = ["_mesa_index", "space"]

    @property
    def position(self) -> np.ndarray:
        return self.space._agent_positions[self._mesa_index]

    @position.setter
    def position(self, value: np.ndarray) -> None:
        if not self.space.in_bounds(value):
            if self.space.torus:
                value = self.space.torus_correct(value)
            else:
                raise ValueError(f"point {value} is outside the bounds of the space")

        self.space._agent_positions[self._mesa_index] = value
        
        if hasattr(self.space, "_mark_dirty"):
            self.space._mark_dirty()

    def __init__(self, space: ContinuousSpace, model):
        super().__init__(model)
        self.space: ContinuousSpace = space
        self.space._add_agent(self)

    def remove(self) -> None:
        super().remove()
        self.space._remove_agent(self)
        self._mesa_index = None
        self.space = None

    def get_neighbors_in_radius(
        self, radius: float | int = 1
    ) -> tuple[list, np.ndarray]:
        agents, dists = self.space.get_agents_in_radius(self.position, radius=radius)
        logical = np.asarray([agent is not self for agent in agents], dtype=bool)
        agents = list(compress(agents, logical))
        return agents, dists[logical]