from mesa.discrete_space import CellAgent, FixedAgent
from mesa.experimental.behaviorals.decision import DecisionSystem, RulePriority
from mesa.experimental.behaviorals.state import BehavioralState
from mesa.experimental.behaviorals.task import Task, TaskManager
from mesa.experimental.mesa_signals.core import HasEmitters
from mesa.time import Priority, Schedule


class Animal(CellAgent, HasEmitters):
    """The base animal class utilizing the Behavioral Framework."""

    # STATE & CONSTRAINTS - Energy decays automatically by 1 per time unit
    energy = BehavioralState(decay_rate=-1.0)

    def __init__(
        self, model, energy=8, p_reproduce=0.04, energy_from_food=4, cell=None
    ):
        super().__init__(model)
        self.p_reproduce = p_reproduce
        self.energy_from_food = energy_from_food
        self.cell = cell

        # QUERY INTERFACE & DECISION LOGIC
        self.task_manager = TaskManager(self)
        self.decision_system = DecisionSystem(self, self.task_manager)

        # Locks in the initial value and lazy decay timer
        self.energy = energy

        self._setup_rules()
        self.model.schedule_recurring(
            self.decision_system.evaluate, schedule=Schedule()
        )

    def _setup_rules(self):
        """Declarative Rules Engine."""

        # CRITICAL: Die if starved
        self.decision_system.add_rule(
            name="die_of_starvation",
            condition=lambda: self.energy <= 0,
            action=lambda: Task(
                agent=self, duration=0.0, action=self.remove, priority=Priority.HIGH
            ),
            priority=RulePriority.CRITICAL,
        )

        # URGENT: Reproduce if well-fed and lucky (Biological Constraint)
        self.decision_system.add_rule(
            name="reproduce",
            condition=lambda: self.energy > 15
            and self.random.random() < self.p_reproduce,
            action=lambda: Task(
                agent=self,
                duration=2.0,
                action=self.spawn_offspring,
                priority=Priority.DEFAULT,
            ),
            priority=RulePriority.URGENT,
            cooldown=3.0,
        )

        # HIGH: Feed if food is available
        self.decision_system.add_rule(
            name="feed",
            condition=self.can_feed,
            action=lambda: Task(
                agent=self, duration=1.0, action=self.feed, priority=Priority.DEFAULT
            ),
            priority=RulePriority.HIGH,
        )

        # DEFAULT: Wander blindly if nothing else to do
        self.decision_system.add_rule(
            name="wander",
            condition=lambda: True,
            action=lambda: Task(
                agent=self, duration=1.0, action=self.move, priority=Priority.LOW
            ),
            priority=RulePriority.DEFAULT,
        )

    def spawn_offspring(self):
        """Create offspring by splitting energy and creating new instance."""
        self.energy /= 2
        self.__class__(
            self.model,
            self.energy,
            self.p_reproduce,
            self.energy_from_food,
            self.cell,
        )

    def can_feed(self) -> bool:
        """Fast boolean check for capability query."""
        return False

    def feed(self):
        """Action execution logic."""

    def move(self):
        """Blind movement: Move to any random neighboring cell."""
        self.cell = self.cell.neighborhood.select_random_cell()


class Sheep(Animal):
    def can_feed(self):
        grass_patch = next(
            (obj for obj in self.cell.agents if isinstance(obj, GrassPatch)), None
        )
        return grass_patch is not None and grass_patch.fully_grown

    def feed(self):
        grass_patch = next(
            (obj for obj in self.cell.agents if isinstance(obj, GrassPatch)), None
        )
        if grass_patch and grass_patch.fully_grown:
            self.energy += self.energy_from_food
            grass_patch.get_eaten()


class Wolf(Animal):
    def can_feed(self):
        return any(isinstance(obj, Sheep) for obj in self.cell.agents)

    def feed(self):
        sheep = [obj for obj in self.cell.agents if isinstance(obj, Sheep)]
        if sheep:
            sheep_to_eat = self.random.choice(sheep)
            self.energy += self.energy_from_food
            sheep_to_eat.remove()


class GrassPatch(FixedAgent):
    def __init__(self, model, countdown, grass_regrowth_time, cell):
        super().__init__(model)
        self.fully_grown = countdown == 0
        self.grass_regrowth_time = grass_regrowth_time
        self.cell = cell

        if not self.fully_grown:
            self._schedule_regrowth(countdown)

    def _schedule_regrowth(self, delay: int):
        self.model.schedule_event(
            self.regrow, at=self.model.time + delay, priority=Priority.DEFAULT
        )

    def regrow(self):
        self.fully_grown = True

    def get_eaten(self):
        self.fully_grown = False
        self._schedule_regrowth(self.grass_regrowth_time)
