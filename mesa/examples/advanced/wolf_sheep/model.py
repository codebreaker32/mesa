"""
Wolf-Sheep Predation Model
================================

Replication of the model found in NetLogo:
    Wilensky, U. (1997). NetLogo Wolf Sheep Predation model.
    http://ccl.northwestern.edu/netlogo/models/WolfSheepPredation.
    Center for Connected Learning and Computer-Based Modeling,
    Northwestern University, Evanston, IL.
"""

import math

from mesa import Model
from mesa.discrete_space import OrthogonalVonNeumannGrid
from mesa.examples.advanced.wolf_sheep.agents import GrassPatch, Sheep, Wolf
from mesa.experimental.data_collection import DataRecorder, DatasetConfig
from mesa.experimental.devs import ABMSimulator


class WolfSheep(Model):
    """Wolf-Sheep Predation Model.

    A model for simulating wolf and sheep (predator-prey) ecosystem modelling.
    """

    description = (
        "A model for simulating wolf and sheep (predator-prey) ecosystem modelling."
    )

    def __init__(
        self,
        width=20,
        height=20,
        initial_sheep=100,
        initial_wolves=50,
        sheep_reproduce=0.04,
        wolf_reproduce=0.05,
        wolf_gain_from_food=20,
        grass=True,
        grass_regrowth_time=30,
        sheep_gain_from_food=4,
        rng=None,
        simulator: ABMSimulator = None,
    ):
        """Create a new Wolf-Sheep model with the given parameters."""
        super().__init__(rng=rng)

        # Handle simulator if provided (DEVS support)
        if simulator:
            self.simulator = simulator
            self.simulator.setup(self)

        # Initialize model parameters
        self.height = height
        self.width = width
        self.grass = grass

        # Create grid using experimental cell space
        self.grid = OrthogonalVonNeumannGrid(
            [self.height, self.width],
            torus=True,
            capacity=math.inf,
            random=self.random,
        )

        model_fields = ["num_wolves", "num_sheep"]
        if grass:
            model_fields.append("num_grass")

        self.data_registry.track_model(self, "model_data", fields=model_fields)

        # Create sheep:
        Sheep.create_agents(
            self,
            initial_sheep,
            energy=self.rng.random((initial_sheep,)) * 2 * sheep_gain_from_food,
            p_reproduce=sheep_reproduce,
            energy_from_food=sheep_gain_from_food,
            cell=self.random.choices(self.grid.all_cells.cells, k=initial_sheep),
        )
        # Create Wolves:
        Wolf.create_agents(
            self,
            initial_wolves,
            energy=self.rng.random((initial_wolves,)) * 2 * wolf_gain_from_food,
            p_reproduce=wolf_reproduce,
            energy_from_food=wolf_gain_from_food,
            cell=self.random.choices(self.grid.all_cells.cells, k=initial_wolves),
        )

        # Create grass patches if enabled
        if grass:
            possibly_fully_grown = [True, False]
            for cell in self.grid:
                fully_grown = self.random.choice(possibly_fully_grown)
                countdown = (
                    0 if fully_grown else self.random.randrange(0, grass_regrowth_time)
                )
                GrassPatch(self, countdown, grass_regrowth_time, cell)

        self.running = True
        self.recorder = DataRecorder(
            self, config={"model_data": DatasetConfig(interval=1, start_time=0)}
        )

    @property
    def num_wolves(self):
        """Wolves Count."""
        return len(self.agents_by_type[Wolf])

    @property
    def num_sheep(self):
        """Sheep Count."""
        return len(self.agents_by_type[Sheep])

    @property
    def num_grass(self):
        """Count fully grown grass."""
        return len(self.agents_by_type[GrassPatch].select(lambda a: a.fully_grown))

    def step(self):
        """Execute one step of the model."""
        self.agents_by_type[Sheep].shuffle_do("step")
        self.agents_by_type[Wolf].shuffle_do("step")
