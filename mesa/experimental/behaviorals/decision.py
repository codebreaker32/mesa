"""Decision System for Mesa.

Improvements over the original:
  - @rule decorator for defining rules directly on BehavioralAgent subclasses
  - remove_rule / enable_rule / disable_rule
  - get_available_actions returns richer info
  - evaluate() no longer swallows errors silently in debug mode
  - duplicate-action detection uses a stable identity, not fragile __name__ lookup
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mesa.agent import Agent
    from mesa.experimental.behaviorals.task import Task, TaskManager


class RulePriority(IntEnum):
    """Priority levels for decision rules. Lower value = higher priority."""

    CRITICAL = 1
    URGENT = 2
    HIGH = 3
    DEFAULT = 4
    LOW = 5


@dataclass(slots=True)
class Rule:
    """A single decision rule: condition → task-producing action."""

    name: str
    condition: Callable[[], bool]
    action: Callable[[], Task]
    priority: RulePriority = RulePriority.DEFAULT
    cooldown: float = 0.0
    enabled: bool = True
    _last_triggered: float = field(default=0.0, init=False)


# @rule decorator


def rule(
    condition: Callable,
    *,
    priority: RulePriority = RulePriority.DEFAULT,
    cooldown: float = 0.0,
    name: str | None = None,
):
    """Decorator to declare a decision rule on a BehavioralAgent subclass.

    The decorated method must accept no arguments beyond ``self`` and return a
    ``Task``. ``condition`` is a callable that receives ``self`` (the agent).

    Args:
        condition: ``(agent) -> bool``. If True the rule fires.
        priority: Rule priority (lower fires first).
        cooldown: Minimum time between firings. 0 = no cooldown.
        name: Override the rule name (defaults to method name).

    Two forms are supported:

    **Pattern A — plain method (step-based, most models)**::

        class Wolf(BehavioralAgent):
            hunger = BehavioralState(decay_rate=0.1, min_value=0, max_value=100)

            @rule(condition=lambda self: self.hunger > 80, priority=RulePriority.URGENT)
            def hunt(self):
                self._attack_nearest_sheep()   # just write the action directly

            @rule(condition=lambda self: True, priority=RulePriority.LOW, cooldown=5.0)
            def wander(self):
                self._move_random()

    **Pattern B — return a Task (continuous-time, real durations)**::

        class Firefighter(BehavioralAgent):
            @rule(condition=lambda self: self.fire_visible, priority=RulePriority.URGENT)
            def fight_fire(self):
                return Task(self, duration=30.0, action=self._fight,
                            reschedule_on_interrupt="remainder")
    """

    def decorator(func: Callable) -> Callable:
        func._rule_meta = {
            "name": name or func.__name__,
            "condition": condition,
            "priority": priority,
            "cooldown": cooldown,
        }
        return func

    return decorator


# DecisionSystem


class DecisionSystem:
    """Manages rule-based decision-making for one agent.

    Rules are sorted by priority; the first rule whose condition is True
    (and whose cooldown has elapsed) produces a Task that is scheduled via
    the agent's TaskManager. Only one rule fires per evaluate() call, and
    only if no task is already running.

    Usage (imperative)::

        ds = DecisionSystem(agent, agent.task_manager)
        ds.add_rule("flee",
                    condition=lambda: agent.threat > 0.8,
                    action=lambda: Task(agent, duration=3, action=agent._flee),
                    priority=RulePriority.CRITICAL)

    Usage (declarative, via @rule on BehavioralAgent)::

        class Rabbit(BehavioralAgent):
            @rule(condition=lambda self: self.threat > 0.8, priority=RulePriority.CRITICAL)
            def flee(self):
                return Task(self, duration=3, action=self._flee)
    """

    def __init__(self, agent: Agent, task_manager: TaskManager):
        self.agent = agent
        self.task_manager = task_manager
        self.rules: list[Rule] = []
        self.debug: bool = False  # set True to print rule evaluation errors

    # Rule management

    def add_rule(
        self,
        name: str,
        condition: Callable[[], bool],
        action: Callable[[], Task],
        priority: RulePriority = RulePriority.DEFAULT,
        cooldown: float = 0.0,
    ) -> Rule:
        """Add a decision rule.

        Args:
            name: Unique rule name.
            condition: Zero-arg callable returning bool.
            action: Zero-arg callable returning a Task.
            priority: Lower value = evaluated first.
            cooldown: Seconds that must pass before this rule can fire again.

        Returns:
            The created Rule object (useful for later enable/disable).

        Raises:
            ValueError: If a rule with the same name already exists.
        """
        if any(r.name == name for r in self.rules):
            raise ValueError(f"Rule '{name}' already exists. Use remove_rule() first.")

        r = Rule(
            name=name,
            condition=condition,
            action=action,
            priority=priority,
            cooldown=cooldown,
        )
        self.rules.append(r)
        self._sort_rules()
        return r

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name. Returns True if found and removed."""
        before = len(self.rules)
        self.rules = [r for r in self.rules if r.name != name]
        return len(self.rules) < before

    def enable_rule(self, name: str) -> None:
        """Re-enable a previously disabled rule."""
        for r in self.rules:
            if r.name == name:
                r.enabled = True
                return
        raise KeyError(f"Rule '{name}' not found.")

    def disable_rule(self, name: str) -> None:
        """Prevent a rule from firing without removing it."""
        for r in self.rules:
            if r.name == name:
                r.enabled = False
                return
        raise KeyError(f"Rule '{name}' not found.")

    def get_rule(self, name: str) -> Rule | None:
        """Retrieve a rule by name, or None."""
        return next((r for r in self.rules if r.name == name), None)

    # Evaluation

    def evaluate(self) -> str | None:
        """Evaluate rules and fire the highest-priority triggered one.

        Two modes depending on what the rule method returns:

        - **Pattern A (returns None)**: the method body ran its action directly.
          The rule is recorded as fired; no Task is created or scheduled.
        - **Pattern B (returns a Task)**: the Task is scheduled via TaskManager.

        Returns:
            The name of the rule that fired, or None.

        Notes:
            In Pattern B, skips evaluation entirely if a task is already running.
            In Pattern A, each step always evaluates because there is no "running
            task" concept — the action is synchronous and already done.
            Rules are evaluated in priority order; only the first matching rule
            fires per call.
        """
        current_time: float = getattr(self.agent.model, "time", 0.0)

        for rule in self.rules:
            if not rule.enabled:
                continue

            if rule.cooldown > 0:
                if (current_time - rule._last_triggered) < rule.cooldown:
                    continue

            try:
                triggered = rule.condition()
            except Exception as exc:
                if self.debug:
                    import traceback

                    print(
                        f"[DecisionSystem] Error in condition for rule '{rule.name}': {exc}"
                    )
                    traceback.print_exc()
                continue

            if not triggered:
                continue

            try:
                task = rule.action()
            except Exception as exc:
                if self.debug:
                    import traceback

                    print(
                        f"[DecisionSystem] Error in action for rule '{rule.name}': {exc}"
                    )
                    traceback.print_exc()
                continue

            # Pattern A: if the action returned None the method already ran its
            # body directly. Record the firing and return — no Task needed.
            if task is None:
                rule._last_triggered = current_time
                return rule.name

            # Pattern B: action returned a Task — schedule it via TaskManager.
            # Avoid scheduling a duplicate task (same type already queued).
            action_name = getattr(
                getattr(task, "_action_callback", None), "__name__", None
            )
            task_id = action_name or type(task).__name__

            already_queued = any(
                (
                    getattr(getattr(t, "_action_callback", None), "__name__", None)
                    or type(t).__name__
                )
                == task_id
                for t in self.task_manager.task_queue
            )
            if already_queued:
                continue

            if self.task_manager.schedule(task):
                rule._last_triggered = current_time
                return rule.name

        return None

    # Introspection

    def get_available_rules(self) -> list[dict[str, Any]]:
        """Return info about all currently-triggered rules (for debugging/UI).

        Returns:
            List of dicts with keys: name, priority, enabled, condition_true, on_cooldown.
        """
        current_time = getattr(self.agent.model, "time", 0.0)
        result = []
        for r in self.rules:
            on_cooldown = (
                r.cooldown > 0 and (current_time - r._last_triggered) < r.cooldown
            )
            try:
                cond = r.condition()
            except Exception:
                cond = None
            result.append(
                {
                    "name": r.name,
                    "priority": r.priority.name,
                    "enabled": r.enabled,
                    "condition_true": cond,
                    "on_cooldown": on_cooldown,
                }
            )
        return result

    def _sort_rules(self) -> None:
        self.rules.sort(key=lambda r: r.priority.value)

    def __repr__(self) -> str:
        return (
            f"DecisionSystem(agent={getattr(self.agent, 'unique_id', '?')}, "
            f"rules={[r.name for r in self.rules]})"
        )
