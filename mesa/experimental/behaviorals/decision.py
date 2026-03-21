"""Decision System for Mesa"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mesa.agent import Agent
    from mesa.experimental.behaviorals.task import Task, TaskManager


class RulePriority(IntEnum):
    """Priority levels for decision rules."""

    CRITICAL = auto()
    URGENT = auto()
    HIGH = auto()
    DEFAULT = auto()
    LOW = auto()


@dataclass(slots=True)
class Rule:
    """A decision rule that triggers actions when conditions are met."""

    name: str
    condition: Callable[[], bool]
    action: Callable[[], Task]
    priority: RulePriority = RulePriority.DEFAULT
    cooldown: float = 0.0
    _last_triggered: float = 0.0


class DecisionSystem:
    """Manages rule-based decision-making for an agent."""

    def __init__(self, agent: Agent, task_manager: TaskManager):
        self.agent = agent
        self.task_manager = task_manager
        self.rules: list[Rule] = []

    def add_rule(
        self,
        name: str,
        condition: Callable[[], bool],
        action: Callable[[], Task],
        priority: RulePriority = RulePriority.DEFAULT,
        cooldown: float = 0.0,
    ):
        """Add a decision rule."""
        # Check for duplicate rule names
        if any(rule.name == name for rule in self.rules):
            raise ValueError(f"Rule '{name}' already exists")

        rule = Rule(
            name=name,
            condition=condition,
            action=action,
            priority=priority,
            cooldown=cooldown,
        )
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority.value)

    def evaluate(self):
        """Evaluate all rules and schedule the highest-priority triggered task."""
        if self.task_manager.current_task is not None:
            return

        for rule in self.rules:
            if rule.cooldown > 0:
                time_since_trigger = self.agent.model.time - rule._last_triggered
                if time_since_trigger < rule.cooldown:
                    continue

            try:
                if rule.condition():
                    task = rule.action()

                    # Duplicate detection: check if same task already in queue
                    task_id = self._get_task_id(task)
                    if any(
                        self._get_task_id(t) == task_id
                        for t in self.task_manager.task_queue
                    ):
                        continue

                    if self.task_manager.schedule(task):
                        rule._last_triggered = self.agent.model.time
                        return
            except Exception as e:
                print(f"Error evaluating rule '{rule.name}': {e}")

    def _get_task_id(self, task: Task) -> str:
        """Get a unique identifier for a task to detect duplicates."""
        if task.action is not None:
            # Use action name if available (handles lambdas and regular functions)
            return getattr(task.action, "__name__", str(task.action))
        # Use task class name for subclassed tasks
        return task.__class__.__name__

    def get_available_actions(self) -> list[str]:
        """Query which actions are currently available."""
        available = []
        for rule in self.rules:
            try:
                if rule.condition():
                    available.append(rule.name)
            except Exception:
                pass
        return available
