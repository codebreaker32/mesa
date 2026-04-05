"""A Continuous Space class."""

import warnings
from collections.abc import Iterable
from itertools import compress
from random import Random

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from mesa.agent import Agent, AgentSet


class ContinuousSpace:
    """Continuous space where each agent can have an arbitrary position."""

    @property
    def x_min(self):
        return self.dimensions[0, 0]

    @property
    def x_max(self):
        return self.dimensions[0, 1]

    @property
    def y_min(self):
        return self.dimensions[1, 0]

    @property
    def y_max(self):
        return self.dimensions[1, 1]

    @property
    def width(self):
        return self.size[0]

    @property
    def height(self):
        return self.size[1]

    def __init__(
        self,
        dimensions: ArrayLike,
        torus: bool = False,
        random: Random | None = None,
        n_agents: int = 100,
    ) -> None:
        if random is None:
            random = Random()
        self.random = random

        self.dimensions: np.array = np.asanyarray(dimensions)
        self.ndims: int = self.dimensions.shape[0]

        self.size: np.array = self.dimensions[:, 1] - self.dimensions[:, 0]
        self.center: np.array = np.sum(self.dimensions, axis=1) / 2
        self.torus: bool = torus

        self._agent_positions: np.array = np.empty(
            (n_agents, self.dimensions.shape[0]), dtype=float
        )
        self.active_agents = []
        self._n_agents = 0
        self._index_to_agent: dict[int, Agent] = {}

        # The C++ Spatial Index
        self._kdtree = None
        self._tree_dirty = True

    def build_index(self):
        """Build the C++ spatial index. Call once per step for massive performance."""
        if self._n_agents > 0:
            if self.torus:
                self._kdtree = cKDTree(
                    self._agent_positions[:self._n_agents], 
                    boxsize=self.dimensions[:, 1]
                )
            else:
                self._kdtree = cKDTree(self._agent_positions[:self._n_agents])
        self._tree_dirty = False

    def _mark_dirty(self):
        """Flag that agents have moved. We DO NOT delete the tree here to allow caching."""
        self._tree_dirty = True

    def _add_agent(self, agent: Agent) -> int:
        index = self._n_agents
        self._n_agents += 1
        if index >= self._agent_positions.shape[0]:
            new_positions = np.empty(
                (self._agent_positions.shape[0] * 2, self.dimensions.shape[0]),
                dtype=float,
            )
            new_positions[:index] = self._agent_positions
            self._agent_positions = new_positions
            
        agent._mesa_index = index
        self._index_to_agent[index] = agent
        self.active_agents.append(agent)
        self._mark_dirty()
        return index

    def _remove_agent(self, agent: Agent) -> None:
        index = agent._mesa_index
        last_index = self._n_agents - 1
        
        if index != last_index:
            last_agent = self._index_to_agent[last_index]
            self._agent_positions[index] = self._agent_positions[last_index]
            self._index_to_agent[index] = last_agent
            last_agent._mesa_index = index

        self._n_agents -= 1
        del self._index_to_agent[last_index]
        self.active_agents.remove(agent)
        self._mark_dirty()

    def calculate_distances(self, point: ArrayLike, agents=None, **kwargs) -> tuple:
        if agents is None:
            agents = self.active_agents
            positions = self._agent_positions[: self._n_agents]
        else:
            indices = [agent._mesa_index for agent in agents]
            positions = self._agent_positions[indices]

        if self.torus:
            delta = np.abs(np.asanyarray(point) - positions)
            delta = np.minimum(delta, self.size - delta)
            return np.linalg.norm(delta, axis=1), agents
        return cdist(np.asanyarray(point)[np.newaxis, :], positions, **kwargs)[0, :], agents

    def calculate_difference_vector(self, point: ArrayLike, agents=None) -> np.ndarray:
        if agents is None:
            positions = self._agent_positions[: self._n_agents]
        else:
            indices = [agent._mesa_index for agent in agents]
            positions = self._agent_positions[indices]

        point = np.asanyarray(point)
        delta = positions - point
        if self.torus:
            half_size = self.size / 2.0
            delta = (delta + half_size) % self.size - half_size
        return delta

    def get_agents_in_radius(
        self, point: ArrayLike, radius: float | int = 1
    ) -> tuple[list, np.ndarray]:
        point = np.asarray(point)

        # Use compiled C++ Tree if available
        if self._kdtree is not None:
            indices = self._kdtree.query_ball_point(point, r=radius)
            if not indices:
                return [], np.array([], dtype=float)

            idx_array = np.array(indices, dtype=int)
            positions = self._agent_positions[idx_array]

            # Inlined C-speed math for exact distances
            if self.torus:
                delta = np.abs(point - positions)
                delta = np.minimum(delta, self.size - delta)
                if self.ndims == 2:
                    dists = np.sqrt(delta[:, 0]**2 + delta[:, 1]**2)
                else:
                    dists = np.linalg.norm(delta, axis=1)
            else:
                if self.ndims == 2:
                    delta = point - positions
                    dists = np.sqrt(delta[:, 0]**2 + delta[:, 1]**2)
                else:
                    dists = np.linalg.norm(point - positions, axis=1)

            agents = [self._index_to_agent[i] for i in indices]
            return agents, dists

        # Fallback to pure numpy
        distances, agents = self.calculate_distances(point)
        logical = distances <= radius
        agents = list(compress(agents, logical))
        return agents, distances[logical]

    def in_bounds(self, point: ArrayLike) -> bool:
        return bool(
            (
                (np.asanyarray(point) >= self.dimensions[:, 0])
                & (point <= self.dimensions[:, 1])
            ).all()
        )

    def torus_correct(self, point: ArrayLike) -> np.ndarray:
        return self.dimensions[:, 0] + np.mod(
            np.asanyarray(point) - self.dimensions[:, 0], self.size
        )