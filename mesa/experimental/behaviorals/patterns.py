"""Concrete behavioral patterns built on BehavioralAgent.

Three ready-to-subclass archetypes, each clearly mapped to a usage pattern:

NeedsAgent  — Pattern A (step-based, duration=0 tasks)
              Homeostatic needs (hunger, thirst, fatigue).
              Auto-generates rules from BehavioralState descriptors + satisfier methods.

BDIAgent    — Pattern B (continuous-time, real durations)
              Belief-Desire-Intention deliberative architecture.
              Tasks represent intentions that take real simulation time.

RLAgent     — Pattern A or B depending on execute_action duration.
              Tabular epsilon-greedy Q-learning.

Quick reference
---------------

NeedsAgent::

    class Villager(NeedsAgent):
        hunger = BehavioralState(decay_rate=0.04, min_value=0, max_value=1)
        thirst = BehavioralState(decay_rate=0.06, min_value=0, max_value=1)

        def satisfier_for_hunger(self):
            self.hunger = max(0, self.hunger - 0.6)

        def satisfier_for_thirst(self):
            self.thirst = max(0, self.thirst - 0.8)

BDIAgent::

    class Scout(BDIAgent):
        def perceive(self):
            self.beliefs["enemy_nearby"] = self._sense_enemy()

        def generate_desires(self) -> list[Desire]:
            if self.beliefs.get("enemy_nearby"):
                return [Desire(priority=10, name="flee")]
            return [Desire(priority=1, name="explore")]

        def plan(self, desire: Desire) -> Task | None:
            if desire.name == "flee":
                return Task(self, duration=3.0, action=self._run_away,
                            reschedule_on_interrupt=False)
            return Task(self, duration=5.0, action=self._move_random,
                        reschedule_on_interrupt="remainder")

RLAgent::

    class Forager(RLAgent):
        def get_state(self):
            return (int(self.energy * 5),)

        def get_actions(self) -> list[str]:
            return ["search", "rest", "eat"]

        def execute_action(self, action: str) -> Task:
            return Task(self, duration=0, action=getattr(self, f"_{action}"),
                        reschedule_on_interrupt=False)

        def compute_reward(self) -> float:
            return self.food_collected - 0.05
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from mesa.experimental.behaviorals.behavioral_agent import BehavioralAgent
from mesa.experimental.behaviorals.decision import RulePriority
from mesa.experimental.behaviorals.state import BehavioralState
from mesa.experimental.behaviorals.task import Task


# ===========================================================================
# NeedsAgent — homeostatic needs satisfaction  (Pattern A: duration=0)
# ===========================================================================

class NeedsAgent(BehavioralAgent):
    """Agent driven by homeostatic needs (hunger, thirst, fatigue, …).

    **Pattern A — step-based (duration=0 tasks)**

    Define needs as ``BehavioralState`` class attributes with a non-zero
    ``decay_rate``.  For each need, declare a ``satisfier_for_<need>(self)``
    method that returns a Task with ``duration=0``.  NeedsAgent auto-generates
    two decision rules per need:

    - ``satisfy_<need>_critical`` (URGENT priority) — fires above ``critical_fraction``
    - ``satisfy_<need>_warning``  (DEFAULT priority) — fires above ``warning_fraction``

    Threshold levels default to fractions of ``max_value``; override them by
    naming thresholds ``"critical"`` and ``"warning"`` in the BehavioralState.

    ``step()`` calls ``sync_states()`` first to materialise lazy decay and
    fire threshold callbacks, then evaluates rules.

    Example::

        class Villager(NeedsAgent):
            hunger = BehavioralState(
                decay_rate=0.04, min_value=0.0, max_value=1.0,
                thresholds={"warning": 0.5, "critical": 0.8},
            )
            thirst = BehavioralState(
                decay_rate=0.06, min_value=0.0, max_value=1.0,
            )

            # satisfier_for_<need> is a plain method — no Task, no boilerplate.
            def satisfier_for_hunger(self):
                self.hunger = max(0.0, self.hunger - 0.6)

            def satisfier_for_thirst(self):
                self.thirst = max(0.0, self.thirst - 0.8)
    """

    #: Fraction of max_value above which the URGENT rule fires.
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
        """Scan class for BehavioralState descriptors and auto-register rules."""
        for attr_name in dir(type(self)):
            descriptor = getattr(type(self), attr_name, None)
            if not isinstance(descriptor, BehavioralState):
                continue
            if descriptor.decay_rate == 0.0:
                continue  # stationary — not a need
            satisfier_name = f"satisfier_for_{attr_name}"
            if not hasattr(self, satisfier_name):
                continue  # no satisfier defined — skip

            satisfier = getattr(self, satisfier_name)
            max_val = descriptor.max_value if descriptor.max_value is not None else 1.0

            # Resolve threshold levels: explicit name wins, else fraction of max
            critical_level = descriptor.thresholds.get(
                "critical", max_val * self.critical_fraction
            )
            warning_level = descriptor.thresholds.get(
                "warning", max_val * self.warning_fraction
            )

            crit_name = f"satisfy_{attr_name}_critical"
            if not any(r.name == crit_name for r in self.decision_system.rules):
                self.decision_system.add_rule(
                    name=crit_name,
                    condition=_need_condition(self, attr_name, critical_level),
                    action=satisfier,
                    priority=RulePriority.URGENT,
                )

            warn_name = f"satisfy_{attr_name}_warning"
            if not any(r.name == warn_name for r in self.decision_system.rules):
                self.decision_system.add_rule(
                    name=warn_name,
                    condition=_need_condition(self, attr_name, warning_level),
                    action=satisfier,
                    priority=RulePriority.DEFAULT,
                )

    # ------------------------------------------------------------------
    # step — sync states first so threshold callbacks fire during decay
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Materialise lazy decay, fire threshold callbacks, then evaluate rules."""
        self.sync_states()
        super().step()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def needs_summary(self) -> dict[str, float]:
        """Return current level of every decaying BehavioralState."""
        return {
            name: getattr(self, name)
            for name in dir(type(self))
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


def _need_condition(agent: "NeedsAgent", attr_name: str, threshold: float) -> Callable:
    """Return a zero-arg callable that checks a need threshold on the agent."""
    return lambda a=agent, n=attr_name, t=threshold: getattr(a, n) > t


# ===========================================================================
# BDIAgent — Belief-Desire-Intention  (Pattern B: real durations)
# ===========================================================================

@dataclass(order=True)
class Desire:
    """A goal the agent wants to achieve.

    Args:
        priority: Higher value = more important. Desires are sorted descending.
        name: Identifies the goal (used to look up a plan).
        data: Optional payload (target position, target agent, etc.).
        deadline: Optional model-time by which this desire should be satisfied.
    """

    priority: float         # first field → used for comparison / sorting
    name: str = field(compare=False)
    data: Any = field(default=None, compare=False)
    deadline: float | None = field(default=None, compare=False)


class BDIAgent(BehavioralAgent):
    """Belief-Desire-Intention agent architecture.

    **Pattern B — continuous-time (real durations)**

    Each step runs a four-phase deliberation cycle.  If a task is already
    running the agent is committed to it and steps 2-4 are skipped.

    1. ``perceive()``         — update ``self.beliefs`` from the environment
    2. ``generate_desires()`` — return ranked list of current goals
    3. ``plan(desire)``       — return a Task for the top achievable desire
    4. Schedule the task via TaskManager

    Example (continuous-time firefighter)::

        class Firefighter(BDIAgent):
            def perceive(self):
                self.beliefs["fires"] = [a for a in self.model.agents
                                         if isinstance(a, Fire)]

            def generate_desires(self) -> list[Desire]:
                desires = [Desire(priority=1, name="patrol")]
                if self.beliefs.get("fires"):
                    target = min(self.beliefs["fires"],
                                 key=lambda f: self._dist(f))
                    desires.append(Desire(priority=10, name="extinguish",
                                         data=target))
                return desires

            def plan(self, desire: Desire) -> Task | None:
                if desire.name == "extinguish":
                    return Task(self, duration=30.0,
                                action=lambda: self._fight(desire.data),
                                reschedule_on_interrupt="remainder")
                return Task(self, duration=60.0, action=self._patrol,
                            reschedule_on_interrupt="remainder")
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

    def generate_desires(self) -> list[Desire]:
        """Return current desires, highest priority first. Override in subclass."""
        return []

    def plan(self, desire: Desire) -> Task | None:
        """Return a Task for *desire*, or None if no plan is possible. Override in subclass."""
        return None

    # ------------------------------------------------------------------
    # BDI deliberation loop
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Run perceive each tick; deliberate only when idle.

        For pure continuous-time models that have no step loop, pair this with
        ``on_action_complete`` (see below) so deliberation triggers immediately
        when a task finishes rather than waiting for the next tick.
        """
        self.perceive()
        if not self.is_busy:
            self._deliberate()

    def on_action_complete(self, task) -> None:
        """Re-deliberate immediately when a task finishes (Pattern B hook).

        Fires via TaskManager when the agent goes idle. Allows continuous-time
        models to react without waiting for the next model step::

            class Scout(BDIAgent):
                # Nothing to override — on_action_complete calls _deliberate()
                # automatically so the Scout picks its next intention immediately.
                pass
        """
        self.perceive()
        self._deliberate()

    def _deliberate(self) -> None:
        """Select and schedule the highest-priority achievable desire."""
        self.desires = sorted(self.generate_desires(), reverse=True)
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
# RLAgent — tabular Q-learning  (Pattern A or B)
# ===========================================================================

class RLAgent(BehavioralAgent):
    """Agent that learns action values via tabular epsilon-greedy Q-learning.

    **Pattern A or B depending on execute_action duration.**

    Use ``duration=0`` for step-based models (each action = one step).
    Use real durations for continuous-time models.

    Override four hooks:

    - ``get_state() -> Hashable``        — discretised world state (Q-table key)
    - ``get_actions() -> list[str]``     — available actions from current state
    - ``execute_action(action) -> Task`` — Task for a given action name
    - ``compute_reward() -> float``      — immediate reward after last action

    The Q-table is updated automatically at the start of each step using the
    reward from the previous action.

    Args:
        alpha: Learning rate (0 < α ≤ 1).
        gamma: Discount factor (0 ≤ γ < 1).
        epsilon: Initial exploration rate.
        epsilon_decay: Multiplier applied to epsilon after each action.
        epsilon_min: Floor for epsilon.
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
        self.q_table: dict[Any, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        self._prev_state: Any = None
        self._prev_action: str | None = None
        self._total_reward: float = 0.0
        self._episode_count: int = 0

    # ------------------------------------------------------------------
    # Hooks to override
    # ------------------------------------------------------------------

    def get_state(self) -> Any:
        """Return a hashable representation of the current world state."""
        raise NotImplementedError("Override get_state().")

    def get_actions(self) -> list[str]:
        """Return available action names from the current state."""
        raise NotImplementedError("Override get_actions().")

    def execute_action(self, action: str) -> Task:
        """Return a Task that executes *action*."""
        raise NotImplementedError("Override execute_action().")

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
        """Return the greedy best action for *state* (default: current state)."""
        s = state if state is not None else self.get_state()
        actions = self.get_actions()
        if not actions:
            return None
        return max(actions, key=lambda a: self.q_table[s].get(a, 0.0))

    # ------------------------------------------------------------------
    # Q-update (Bellman equation)
    # ------------------------------------------------------------------

    def _update_q(self, reward: float, new_state: Any, actions: list[str]) -> None:
        if self._prev_state is None or self._prev_action is None:
            return
        old_q = self.q_table[self._prev_state][self._prev_action]
        future_q = max(
            (self.q_table[new_state].get(a, 0.0) for a in actions), default=0.0
        )
        new_q = old_q + self.alpha * (reward + self.gamma * future_q - old_q)
        self.q_table[self._prev_state][self._prev_action] = new_q

    # ------------------------------------------------------------------
    # step / continuous-time hook
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Update Q-table then select next action (step-based Pattern A).

        For continuous-time (Pattern B) models, override ``on_action_complete``
        instead — it fires immediately when the previous task finishes.
        """
        if self.is_busy:
            return
        self._sense_and_act()

    def on_action_complete(self, task) -> None:
        """Re-evaluate and act immediately when a task finishes (Pattern B hook).

        Fires via TaskManager when the agent goes idle, enabling continuous-time
        Q-learning without a step loop.
        """
        self._sense_and_act()

    def _sense_and_act(self) -> None:
        """Update Q-table from previous reward, select next action, schedule task."""
        current_state = self.get_state()
        actions = self.get_actions()

        if self._prev_state is not None:
            reward = self.compute_reward()
            self._total_reward += reward
            self._update_q(reward, current_state, actions)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

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
        """Return Q-values for *state* (default: current state)."""
        s = state if state is not None else self.get_state()
        return dict(self.q_table[s])

    def policy_summary(self, n_states: int = 10) -> list[dict]:
        """Return a summary of the n highest-value states and their best actions."""
        states = sorted(
            self.q_table.keys(),
            key=lambda s: max(self.q_table[s].values(), default=0),
            reverse=True,
        )[:n_states]
        return [
            {
                "state": s,
                "best_action": max(self.q_table[s], key=self.q_table[s].get)
                if self.q_table[s] else None,
                "q_values": dict(self.q_table[s]),
            }
            for s in states
        ]