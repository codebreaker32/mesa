"""Decision System - rule-based decision making connecting BehavioralStates to Tasks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from .task import Task, TaskManager


class RulePriority(IntEnum):
    """Evaluation order for rules (lower = evaluated first, independent of Task.priority)."""

    CRITICAL = 0
    URGENT = 1
    HIGH = 2
    DEFAULT = 5
    LOW = 8
    IDLE = 10


@dataclass(slots=True)
class DecisionRule:
    """A condition→action pair that schedules a Task when the condition is true.

    Example:
        >>> rule = DecisionRule(
        ...     name="eat_when_hungry",
        ...     condition=lambda: self.energy < 30,
        ...     action=lambda: Task(self, self.eat, duration=5, priority=Priority.HIGH),
        ...     priority=RulePriority.HIGH,
        ...     cooldown=10,
        ... )
    """

    name: str
    condition: Callable[[], bool]
    action: Callable[[], Task]
    priority: RulePriority = RulePriority.DEFAULT
    cooldown: float = 0.0
    one_shot: bool = False
    enabled: bool = True

    _last_triggered: float = field(default=-float("inf"), init=False)
    _trigger_count: int = field(default=0, init=False)

    def matches(self, current_time: float) -> bool:
        if not self.enabled:
            return False
        if current_time - self._last_triggered < self.cooldown:
            return False
        try:
            return self.condition()
        except Exception:
            return False

    def trigger(self, current_time: float) -> Task:
        self._last_triggered = current_time
        self._trigger_count += 1
        if self.one_shot:
            self.enabled = False
        return self.action()


class DecisionSystem:
    """Evaluates rules in priority order and schedules matching tasks.

    Prevents duplicate tasks (same action already active or queued).

    Usage:
        self.task_manager = TaskManager(self)
        self.decision_system = DecisionSystem(self, self.task_manager)
        self.decision_system.add_rule(
            name="eat_when_hungry",
            condition=lambda: self.energy < 30,
            action=lambda: Task(self, self.eat, duration=5),
        )
        # In agent.step():
        self.decision_system.evaluate()
    """

    def __init__(self, agent, task_manager: TaskManager | None = None):
        self.agent = agent
        self._task_manager = task_manager
        self.rules: list[DecisionRule] = []
        self._evaluations = 0
        self._triggers = 0
        self._prevented_duplicates = 0

    @property
    def task_manager(self) -> TaskManager:
        if self._task_manager is None:
            if not hasattr(self.agent, "task_manager"):
                raise AttributeError("Pass task_manager to DecisionSystem or set agent.task_manager.")
            return self.agent.task_manager
        return self._task_manager

    def add_rule(
        self,
        name: str | None = None,
        condition: Callable[[], bool] | None = None,
        action: Callable[[], Task] | None = None,
        priority: RulePriority = RulePriority.DEFAULT,
        cooldown: float = 0.0,
        one_shot: bool = False,
        rule: DecisionRule | None = None,
    ) -> DecisionRule:
        if rule is None:
            if condition is None or action is None:
                raise ValueError("Provide 'rule' or both 'condition' and 'action'.")
            if name is None:
                name = f"rule_{len(self.rules)}"
            rule = DecisionRule(name, condition, action, priority, cooldown, one_shot)
        self.rules.append(rule)
        self.rules.sort(key=lambda r: (r.priority.value, r.name))
        return rule

    def remove_rule(self, name: str) -> bool:
        for i, rule in enumerate(self.rules):
            if rule.name == name:
                self.rules.pop(i)
                return True
        return False

    def get_rule(self, name: str) -> DecisionRule | None:
        return next((r for r in self.rules if r.name == name), None)

    def enable_rule(self, name: str) -> bool:
        rule = self.get_rule(name)
        if rule:
            rule.enabled = True
            return True
        return False

    def disable_rule(self, name: str) -> bool:
        rule = self.get_rule(name)
        if rule:
            rule.enabled = False
            return True
        return False

    def evaluate(self) -> Task | None:
        """Evaluate rules in priority order; schedule first matching task."""
        self._evaluations += 1
        current_time = self.agent.model.time

        for rule in self.rules:
            if rule.matches(current_time):
                task = rule.trigger(current_time)
                if self._is_duplicate(task):
                    self._prevented_duplicates += 1
                    return None
                if self.task_manager.schedule(task):
                    self._triggers += 1
                    return task

        return None

    def _get_task_id(self, task: Task) -> str:
        """Get unique identifier for a task's action."""
        if task.action is not None:
            return getattr(task.action, "__name__", str(task.action))
        # For subclassed tasks without action, use class name
        return task.__class__.__name__

    def _is_duplicate(self, task: Task) -> bool:
        """Prevent scheduling same action twice."""
        task_id = self._get_task_id(task)
        
        current = self.task_manager.current_task
        if current and self._get_task_id(current) == task_id:
            return True
        
        return any(self._get_task_id(q) == task_id for q in self.task_manager.task_queue)

    def get_stats(self) -> dict[str, Any]:
        return {
            "evaluations": self._evaluations,
            "triggers": self._triggers,
            "prevented_duplicates": self._prevented_duplicates,
            "active_rules": sum(1 for r in self.rules if r.enabled),
            "total_rules": len(self.rules),
        }


# Helpers for common rule patterns

def threshold_rule(
    agent, state_name: str, threshold: float, comparison: str,
    action: Callable[[], Task], priority: RulePriority = RulePriority.DEFAULT, **kwargs,
) -> DecisionRule:
    """Create a rule triggered by a state comparison (e.g., energy < 30)."""
    ops = {"<": float.__lt__, ">": float.__gt__, "<=": float.__le__,
           ">=": float.__ge__, "==": float.__eq__, "!=": float.__ne__}
    if comparison not in ops:
        raise ValueError(f"Invalid comparison: {comparison}")
    compare_fn = ops[comparison]
    return DecisionRule(
        name=kwargs.pop("name", f"{state_name}_{comparison}_{threshold}"),
        condition=lambda: compare_fn(getattr(agent, state_name), threshold),
        action=action, priority=priority, **kwargs,
    )


def combined_rule(
    agent, conditions: list[Callable[[], bool]], operator: str,
    action: Callable[[], Task], priority: RulePriority = RulePriority.DEFAULT, **kwargs,
) -> DecisionRule:
    """Create a rule with multiple conditions combined via 'and' / 'or'."""
    if operator == "and":
        condition = lambda: all(c() for c in conditions)
    elif operator == "or":
        condition = lambda: any(c() for c in conditions)
    else:
        raise ValueError(f"Invalid operator: {operator}. Use 'and' or 'or'.")
    return DecisionRule(
        name=kwargs.pop("name", f"combined_{operator}"),
        condition=condition, action=action, priority=priority, **kwargs,
    )