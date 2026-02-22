"""
Boltzmann Wealth Model
=====================

A simple model of wealth distribution based on the Boltzmann-Gibbs distribution.
Agents move randomly on a grid, giving one unit of wealth to a random neighbor
when they occupy the same cell.
"""

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.examples.basic.boltzmann_wealth_model.agents import MoneyAgent
from mesa.experimental.data_collection import DataRecorder, DatasetConfig
from mesa.experimental.scenarios import Scenario


class BoltzmannScenario(Scenario):
    """Scenario parameters for the Boltzmann Wealth model."""

    n: int = 100
    width: int = 10
    height: int = 10


class BoltzmannWealth(Model):
    """A simple model of an economy where agents exchange currency at random.

    All agents begin with one unit of currency, and each time step agents can give
    a unit of currency to another agent in the same cell. Over time, this produces
    a highly skewed distribution of wealth.

    Attributes:
        num_agents (int): Number of agents in the model
        grid (MultiGrid): The space in which agents move
        running (bool): Whether the model should continue running
        datacollector (DataCollector): Collects and stores model data
    """

    def __init__(self, scenario=None):
        """Initialize the model.

        Args:
            scenario: BoltzmannScenario object containing model parameters.
        """
        if scenario is None:
            scenario = BoltzmannScenario()

        super().__init__(scenario=scenario)

        self.num_agents = scenario.n
        self.grid = OrthogonalMooreGrid(
            (scenario.width, scenario.height), random=self.random
        )

        self.recorder = DataRecorder(self)
        (
            self.data_registry.track_agents(self.agents, "agent_data", "wealth").record(
                self.recorder
            )
        )
        (
            self.data_registry.track_model(self, "model_data", "gini").record(
                self.recorder, configuration=DatasetConfig(start_time=0, interval=1)
            )
        )

        # Set up data collection
        self.datacollector = DataCollector(
            model_reporters={"Gini": "gini"},
            agent_reporters={"Wealth": "wealth"},
        )
        MoneyAgent.create_agents(
            self,
            self.num_agents,
            self.random.choices(self.grid.all_cells.cells, k=self.num_agents),
        )

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        self.agents.shuffle_do("step")  # Activate all agents in random order
        self.datacollector.collect(self)  # Collect data

    @property
    def gini(self):
        """Calculate the Gini coefficient for the model's current wealth distribution.

        The Gini coefficient is a measure of inequality in distributions.
        - A Gini of 0 represents complete equality, where all agents have equal wealth.
        - A Gini of 1 represents maximal inequality, where one agent has all wealth.
        """
        agent_wealths = [agent.wealth for agent in self.agents]
        x = sorted(agent_wealths)
        n = self.num_agents
        # Calculate using the standard formula for Gini coefficient
        if sum(x) == 0:
            return 0
        b = sum(xi * (n - i) for i, xi in enumerate(x)) / (n * sum(x))
        return 1 + (1 / n) - 2 * b


if __name__ == "__main__":
    model = BoltzmannWealth()

    model.run_for(5)
    df1 = model.recorder.get_table_dataframe("model_data")
    print(f"Current Dataframe:\n{df1.to_string()}")
    print(
        f"\nCurrent Gini at time 5.0: {df1.loc[df1['time'] == 5.0, 'gini'].values[0]:.6f}\n"
    )

    def trade():
        for agent in model.agents:
            agent.wealth += 10

    model.schedule_event(trade, at=5.0)
    model.run_until(10)

    df2 = model.recorder.get_table_dataframe("model_data")
    print(f"\nFinal Dataframe:\n {df2.to_string()}")
    print(
        f"\nUpdated Gini at time 5.0 after second run: {df2.loc[df2['time'] == 5.0, 'gini'].values[0]:.6f}\n"
    )
