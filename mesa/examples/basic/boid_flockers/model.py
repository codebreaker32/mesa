import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

import numpy as np
from mesa import Model
from mesa.examples.basic.boid_flockers.agents import Boid
from mesa.experimental.continuous_space import ContinuousSpace
from mesa.experimental.scenarios import Scenario


class BoidsScenario(Scenario):
    population_size: int = 100
    width: int = 100
    height: int = 100
    speed: float = 1.0
    vision: float = 10.0
    separation: float = 2.0
    cohere: float = 0.03
    separate: float = 0.015
    match: float = 0.05


class BoidFlockers(Model):
    def __init__(self, scenario: BoidsScenario = BoidsScenario):
        super().__init__(scenario=scenario)
        self.agent_angles = np.zeros(scenario.population_size)

        self.space = ContinuousSpace(
            [[0, scenario.width], [0, scenario.height]],
            torus=True,
            random=self.random,
            n_agents=scenario.population_size,
        )

        positions = self.rng.random(size=(scenario.population_size, 2)) * self.space.size
        directions = self.rng.uniform(-1, 1, size=(scenario.population_size, 2))
        
        for i in range(scenario.population_size):
            Boid(
                self,
                self.space,
                position=positions[i],
                direction=directions[i],
                cohere=scenario.cohere,
                separate=scenario.separate,
                match=scenario.match,
                speed=scenario.speed,
                vision=scenario.vision,
                separation=scenario.separation,
            )

        self.average_heading = None
        self.update_average_heading()

    def calculate_angles(self):
        d1 = np.array([agent.direction[0] for agent in self.agents])
        d2 = np.array([agent.direction[1] for agent in self.agents])
        self.agent_angles = np.degrees(np.arctan2(d1, d2))
        for agent, angle in zip(self.agents, self.agent_angles):
            agent.angle = angle

    def update_average_heading(self):
        if not self.agents:
            self.average_heading = 0
            return
        headings = np.array([agent.direction for agent in self.agents])
        mean_heading = np.mean(headings, axis=0)
        self.average_heading = np.arctan2(mean_heading[1], mean_heading[0])

    def step(self):
        # ---------------------------------------------------------
        # THE PERFORMANCE CACHE
        # Builds the C++ tree ONCE per frame, turning O(N^2) loops into O(N log N)
        # ---------------------------------------------------------
        if hasattr(self.space, "build_index"):
            self.space.build_index()

        self.agents.shuffle_do("step")
        self.update_average_heading()
        self.calculate_angles()