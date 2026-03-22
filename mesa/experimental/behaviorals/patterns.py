"""Concrete behavioral patterns built on BehavioralAgent.

Provides three ready-to-subclass agent archetypes:

- NeedsAgent   — homeostatic needs-satisfaction (hunger, thirst, energy, …)
- BDIAgent     — Belief–Desire–Intention deliberative architecture
- RLAgent      — Epsilon-greedy Q-learning with pluggable state/action spaces

Each is a drop-in BehavioralAgent subclass. Override the documented hooks
to plug in your model-specific logic.

Quick reference
---------------

NeedsAgent::

    class Villager(NeedsAgent):
        hunger = BehavioralState(decay_rate=0.04, min_value=0, max_value=1)
        thirst = BehavioralState(decay_rate=0.06, min_value=0, max_value=1)

        def satisfier_for_hunger(self): return Task(self, 3.0, action=self._eat)
        def satisfier_for_thirst(self): return Task(self, 1.0, action=self._drink)

BDIAgent::

    class Scout(BDIAgent):
        def generate_desires(self) -> list[Desire]:
            desires = [Desire("explore", priority=1)]
            if self.beliefs.get("enemy_nearby"):
                desires.append(Desire("flee", priority=10))
            return desires

        def plan(self, desire: Desire) -> Task | None:
            if desire.name == "flee":
                return Task(self, 2.0, action=self._run_away)
            return Task(self, 5.0, action=self._move_random)

RLAgent::

    class LearnerAgent(RLAgent):
        def get_state(self):
            return (round(self.pos[0] / 10), round(self.pos[1] / 10))

        def get_actions(self) -> list[str]:
            return ["move_north", "move_south", "move_east", "move_west"]

        def execute_action(self, action: str) -> Task:
            return Task(self, 1.0, action=getattr(self, f"_{action}"))

        def compute_reward(self) -> float:
            return self.food_collected - self.energy_spent
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from mesa.experimental.behaviorals.behavioral_agent import BehavioralAgent
from mesa.experimental.behaviorals.decision import RulePriority
from mesa.experimental.behaviorals.state import BehavioralState
from mesa.experimental.behaviorals.task import Task


# ===========================================================================
# NeedsAgent — homeostatic needs satisfaction
# ===========================================================================

class NeedsAgent(BehavioralAgent):
    """Agent driven by homeostatic needs (hunger, thirst, fatigue, …).

    Define needs as ``BehavioralState`` class attributes with a ``decay_rate``.
    For each need, declare a ``satisfier_for_<need>(self) -> Task`` method.
    NeedsAgent automatically creates rules that fire when the need exceeds its
    critical threshold.

    Thresholds:
      - ``"critical"`` (default 0.75 of max_value) → urgent priority rule
      - ``"warning"`` (default 0.50 of max_value) → default priority rule

    Override ``critical_threshold`` / ``warning_threshold`` class attributes
    to change the defaults, or pass explicit thresholds to ``BehavioralState``.

    Example::

        class Villager(NeedsAgent):
            hunger = BehavioralState(
                decay_rate=0.04,
                min_value=0.0, max_value=1.0,
                thresholds={"warning": 0.5, "critical": 0.8},
            )
            thirst = BehavioralState(
                decay_rate=0.06,
                min_value=0.0, max_value=1.0,
            )

            def satisfier_for_hunger(self) -> Task:
                return Task(self, duration=3.0, action=self._eat)

            def satisfier_for_thirst(self) -> Task:
                return Task(self, duration=1.0, action=self._drink)

            def _eat(self):
                self.hunger = max(0, self.hunger - 0.6)

            def _drink(self):
                self.thirst = max(0, self.thirst - 0.8)
    """

    #: Fraction of max_value above which the URGENT rule fires (if no explicit threshold).
    critical_fraction: float = 0.75
    #: Fraction of max_value above which the DEFAULT-priority rule fires.
    warning_fraction: float = 0.50

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self._register_need_rules()

    # ------------------------------------------------------------------
    # Automatic rule registration
    # ------------------------------------------------------------------

    def _register_need_rules(self) -> None:
        """Scan for BehavioralState descriptors and auto-register need rules."""
        for attr_name in dir(type(self)):
            descriptor = getattr(type(self), attr_name, None)
            if not isinstance(descriptor, BehavioralState):
                continue
            if descriptor.decay_rate == 0.0:
                continue  # stationary — not a need
            satisfier_name = f"satisfier_for_{attr_name}"
            if not hasattr(self, satisfier_name):
                continue  # no satisfier defined, skip

            satisfier = getattr(self, satisfier_name)
            max_val = descriptor.max_value if descriptor.max_value is not None else 1.0
            thresholds = descriptor.thresholds

            if "critical" in thresholds:
                critical_level = thresholds["critical"]
            else:
                critical_level = max_val * self.critical_fraction

            if "warning" in thresholds:
                warning_level = thresholds["warning"]
            else:
                warning_level = max_val * self.warning_fraction

            # Critical rule (urgent)
            crit_name = f"satisfy_{attr_name}_critical"
            if not any(r.name == crit_name for r in self.decision_system.rules):
                level = critical_level

                def make_crit_condition(a, lv):
                    return lambda: getattr(a, a._need_attr) > lv

                self.decision_system.add_rule(
                    name=crit_name,
                    condition=_need_condition(self, attr_name, critical_level),
                    action=satisfier,
                    priority=RulePriority.URGENT,
                )

            # Warning rule (default priority)
            warn_name = f"satisfy_{attr_name}_warning"
            if not any(r.name == warn_name for r in self.decision_system.rules):
                self.decision_system.add_rule(
                    name=warn_name,
                    condition=_need_condition(self, attr_name, warning_level),
                    action=satisfier,
                    priority=RulePriority.DEFAULT,
                )

    # ------------------------------------------------------------------
    # Override step to sync states (so decay-triggered thresholds fire)
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Sync all need states (triggering threshold callbacks) then evaluate rules."""
        self.sync_states()
        super().step()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def needs_summary(self) -> dict[str, float]:
        """Return current level of every BehavioralState with decay."""
        return {
            name: value
            for name, value in self.states.items()
            if isinstance(getattr(type(self), name, None), BehavioralState)
            and getattr(type(self), name).decay_rate != 0.0
        }

    def most_urgent_need(self) -> str | None:
        """Return the name of the most critical need (highest % of max), or None."""
        worst_name, worst_ratio = None, -1.0
        for attr_name in dir(type(self)):
            descriptor = getattr(type(self), attr_name, None)
            if not isinstance(descriptor, BehavioralState):
                continue
            if descriptor.decay_rate == 0.0:
                continue
            max_val = descriptor.max_value if descriptor.max_value is not None else 1.0
            ratio = getattr(self, attr_name) / max_val
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_name = attr_name
        return worst_name


def _need_condition(agent: NeedsAgent, attr_name: str, threshold: float) -> Callable:
    """Build a zero-arg lambda that checks a need threshold on the agent."""
    return lambda a=agent, n=attr_name, t=threshold: getattr(a, n) > t


# ===========================================================================
# BDIAgent — Belief–Desire–Intention
# ===========================================================================

@dataclass(order=True)
class Desire:
    """A goal the agent wants to achieve.

    Args:
        name: Identifies the goal (used to look up a plan).
        priority: Higher value = more important. Desires are sorted descending.
        data: Optional payload (target position, target agent, etc.).
        deadline: Optional model-time by which this desire should be achieved.
    """

    priority: float  # first field → used for comparison / sorting
    name: str = field(compare=False)
    data: Any = field(default=None, compare=False)
    deadline: float | None = field(default=None, compare=False)


class BDIAgent(BehavioralAgent):
    """Belief–Desire–Intention agent architecture.

    Lifecycle each step:
      1. **Perceive** → update ``self.beliefs``
      2. **Deliberate** → call ``generate_desires()`` to produce a ranked desire list
      3. **Plan** → call ``plan(desire)`` on the top desire to get a Task
      4. **Execute** → schedule the Task via TaskManager

    Override these three hooks in your subclass:

    - ``perceive()``: update ``self.beliefs`` from the model
    - ``generate_desires() -> list[Desire]``: return current desires, sorted by priority
    - ``plan(desire: Desire) -> Task | None``: return the Task for a given desire

    Example::

        class Firefighter(BDIAgent):
            def perceive(self):
                self.beliefs["fires"] = [a for a in self.model.agents
                                         if isinstance(a, Fire)]

            def generate_desires(self) -> list[Desire]:
                desires = [Desire(priority=1, name="patrol")]
                if self.beliefs.get("fires"):
                    nearest = min(self.beliefs["fires"],
                                  key=lambda f: dist(self.pos, f.pos))
                    desires.append(Desire(priority=10, name="extinguish", data=nearest))
                return desires

            def plan(self, desire: Desire) -> Task | None:
                if desire.name == "extinguish":
                    return Task(self, duration=5.0, action=lambda: self._fight(desire.data))
                return Task(self, duration=3.0, action=self._patrol)
    """

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.beliefs: dict[str, Any] = {}
        self.desires: list[Desire] = []
        self.current_intention: Desire | None = None

    # ------------------------------------------------------------------
    # Hooks to override
    # ------------------------------------------------------------------

    def perceive(self) -> None:
        """Update self.beliefs from the environment. Override in subclass."""
        pass

    def generate_desires(self) -> list[Desire]:
        """Return the agent's current desires sorted by priority (highest first).

        Override in subclass to implement domain logic.
        """
        return []

    def plan(self, desire: Desire) -> Task | None:
        """Return a Task that pursues *desire*, or None if no plan is possible.

        Override in subclass to map desires to tasks.
        """
        return None

    # ------------------------------------------------------------------
    # BDI deliberation loop
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Run the full BDI sense-deliberate-act cycle."""
        # 1. Perceive
        self.perceive()

        # 2. Deliberate — skip if already executing an intention
        if self.is_busy:
            return

        # 3. Generate and rank desires
        self.desires = sorted(self.generate_desires(), reverse=True)  # highest priority first
        if not self.desires:
            return

        # 4. Pick the top desire and plan for it
        for desire in self.desires:
            task = self.plan(desire)
            if task is not None:
                self.current_intention = desire
                self.task_manager.schedule(task)
                return

    # ------------------------------------------------------------------
    # Belief helpers
    # ------------------------------------------------------------------

    def update_belief(self, key: str, value: Any) -> None:
        """Set a belief value."""
        self.beliefs[key] = value

    def drop_belief(self, key: str) -> None:
        """Remove a belief."""
        self.beliefs.pop(key, None)

    def has_belief(self, key: str, value: Any | None = None) -> bool:
        """Check if a belief exists (and optionally equals *value*)."""
        if key not in self.beliefs:
            return False
        return True if value is None else self.beliefs[key] == value


# ===========================================================================
# RLAgent — tabular Q-learning
# ===========================================================================

class RLAgent(BehavioralAgent):
    """Agent that learns action values via tabular epsilon-greedy Q-learning.

    Override three hooks:

    - ``get_state() -> Hashable``: discretised world state (used as Q-table key)
    - ``get_actions() -> list[str]``: available actions from the current state
    - ``execute_action(action: str) -> Task``: return the Task for an action
    - ``compute_reward() -> float``: reward signal after an action completes

    The Q-table is updated automatically after each task completes.

    Args:
        alpha: Learning rate (0 < α ≤ 1).
        gamma: Discount factor (0 ≤ γ < 1).
        epsilon: Initial exploration rate.
        epsilon_decay: Multiplied by epsilon after each episode.
        epsilon_min: Minimum exploration rate.

    Example::

        class Forager(RLAgent):
            def get_state(self):
                x, y = self.pos
                return (x // 5, y // 5, int(self.energy > 0.5))

            def get_actions(self):
                return ["north", "south", "east", "west", "eat"]

            def execute_action(self, action):
                if action == "eat":
                    return Task(self, 1.0, action=self._eat)
                return Task(self, 1.0, action=lambda: self._move(action))

            def compute_reward(self):
                return self.food_eaten - 0.1  # food reward minus step cost
    """

    def __init__(
        self,
        model,
        *,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: {state: {action: q_value}}
        self.q_table: dict[Any, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        self._prev_state: Any = None
        self._prev_action: str | None = None
        self._total_reward: float = 0.0
        self._episode_count: int = 0

    # ------------------------------------------------------------------
    # Hooks to override
    # ------------------------------------------------------------------

    def get_state(self) -> Any:
        """Return a hashable representation of the current world state."""
        raise NotImplementedError("Override get_state() to return a hashable state.")

    def get_actions(self) -> list[str]:
        """Return the list of action names available in the current state."""
        raise NotImplementedError("Override get_actions() to return available actions.")

    def execute_action(self, action: str) -> Task:
        """Return a Task that executes *action*."""
        raise NotImplementedError("Override execute_action() to map action names to Tasks.")

    def compute_reward(self) -> float:
        """Return the immediate reward after the last action completed."""
        return 0.0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: Any, actions: list[str]) -> str:
        """Epsilon-greedy action selection."""
        if not actions:
            raise ValueError("get_actions() returned an empty list.")
        if random.random() < self.epsilon:
            return random.choice(actions)
        q_vals = self.q_table[state]
        return max(actions, key=lambda a: q_vals.get(a, 0.0))

    def best_action(self, state: Any | None = None) -> str | None:
        """Return the greedy best action for *state* (or current state if None)."""
        s = state if state is not None else self.get_state()
        actions = self.get_actions()
        if not actions:
            return None
        return max(actions, key=lambda a: self.q_table[s].get(a, 0.0))

    # ------------------------------------------------------------------
    # Q-update
    # ------------------------------------------------------------------

    def _update_q(self, reward: float, new_state: Any, actions: list[str]) -> None:
        if self._prev_state is None or self._prev_action is None:
            return
        old_q = self.q_table[self._prev_state][self._prev_action]
        future_q = max((self.q_table[new_state].get(a, 0.0) for a in actions), default=0.0)
        new_q = old_q + self.alpha * (reward + self.gamma * future_q - old_q)
        self.q_table[self._prev_state][self._prev_action] = new_q

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Sense-decide-act: update Q-table, then select and execute the next action."""
        if self.is_busy:
            return

        current_state = self.get_state()
        actions = self.get_actions()

        # Update Q from the previous step's reward
        if self._prev_state is not None:
            reward = self.compute_reward()
            self._total_reward += reward
            self._update_q(reward, current_state, actions)

        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Select and execute action
        action = self.select_action(current_state, actions)
        self._prev_state = current_state
        self._prev_action = action

        try:
            task = self.execute_action(action)
        except Exception as exc:
            print(f"[RLAgent] execute_action('{action}') raised: {exc}")
            return

        self.task_manager.schedule(task)
        self._episode_count += 1

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def total_reward(self) -> float:
        """Cumulative reward across all episodes."""
        return self._total_reward

    @property
    def episode_count(self) -> int:
        """Number of actions taken."""
        return self._episode_count

    def q_values(self, state: Any | None = None) -> dict[str, float]:
        """Return the Q-values for *state* (default: current state)."""
        s = state if state is not None else self.get_state()
        return dict(self.q_table[s])

    def policy_summary(self, n_states: int = 10) -> list[dict]:
        """Return a summary of the n most-visited states and their best actions."""
        states = sorted(self.q_table.keys(),
                        key=lambda s: max(self.q_table[s].values(), default=0),
                        reverse=True)[:n_states]
        result = []
        for s in states:
            q_vals = self.q_table[s]
            best = max(q_vals, key=q_vals.get) if q_vals else None
            result.append({"state": s, "best_action": best, "q_values": dict(q_vals)})
        return result