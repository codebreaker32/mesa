"""BehavioralAgent: Mesa Agent with integrated task management and rule-based decisions.

Drop-in replacement for mesa.Agent that wires together:
  - TaskManager  (durative tasks with priority and interruption)
  - DecisionSystem (rule-based decision making with @rule decorator support)
  - BehavioralState (lazy-decay state variables)

Usage::

    from mesa.experimental.behaviorals.behavioral_agent import BehavioralAgent
    from mesa.experimental.behaviorals.state import BehavioralState
    from mesa.experimental.behaviorals.decision import rule, RulePriority
    from mesa.experimental.behaviorals.task import Task

    class Sheep(BehavioralAgent):
        hunger = BehavioralState(decay_rate=0.05, min_value=0, max_value=100)
        energy = BehavioralState(decay_rate=-0.02, min_value=0, max_value=1, initial=1.0)

        def __init__(self, model):
            super().__init__(model)
            self.hunger = 0.0
            self.energy = 1.0

        @rule(condition=lambda self: self.hunger > 70, priority=RulePriority.HIGH)
        def graze(self):
            return Task(self, duration=3.0, action=self._eat)

        @rule(condition=lambda self: self.energy < 0.2, priority=RulePriority.URGENT)
        def rest(self):
            return Task(self, duration=5.0, action=self._sleep)

        def _eat(self):
            self.hunger = max(0, self.hunger - 50)

        def _sleep(self):
            self.energy = min(1.0, self.energy + 0.5)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mesa.agent import Agent
from mesa.experimental.behaviorals.decision import DecisionSystem, RulePriority
from mesa.experimental.behaviorals.state import BehavioralState
from mesa.experimental.behaviorals.task import Task, TaskManager

if TYPE_CHECKING:
    from mesa.model import Model


class BehavioralAgent(Agent):
    """Mesa Agent with built-in TaskManager, DecisionSystem, and BehavioralState support.

    Subclass this instead of Agent when your agent needs:
      - Durative tasks (actions that take simulation time)
      - Rule-based decisions (if condition → schedule task)
      - Time-decaying state variables (hunger, energy, fear, …)

    Rules can be defined in two ways:

    1. **@rule decorator** (declarative, preferred)::

           @rule(condition=lambda self: self.hunger > 70, priority=RulePriority.HIGH)
           def graze(self):
               return Task(self, duration=3.0, action=self._eat)

    2. **add_rule()** (imperative, good for dynamic rules)::

           self.add_rule("graze",
                         condition=lambda: self.hunger > 70,
                         action=lambda: Task(self, duration=3.0, action=self._eat),
                         priority=RulePriority.HIGH)

    Override step() to add custom logic. Always call super().step() to ensure
    decision evaluation and state synchronisation happen::

        def step(self):
            self.do_perception()
            super().step()  # evaluates rules + syncs states

    Attributes:
        task_manager: The agent's TaskManager instance.
        decision_system: The agent's DecisionSystem instance.
    """

    def __init__(self, model: Model, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.task_manager: TaskManager = TaskManager(self)
        self.decision_system: DecisionSystem = DecisionSystem(self, self.task_manager)
        self._register_decorated_rules()

    # Auto-register @rule decorated methods

    def _register_decorated_rules(self) -> None:
        """Scan the MRO for methods tagged with @rule and register them."""
        seen: set[str] = set()
        # Walk MRO so subclass rules shadow parent rules
        for klass in type(self).__mro__:
            for attr_name, obj in vars(klass).items():
                if attr_name in seen:
                    continue
                if not callable(obj) or not hasattr(obj, "_rule_meta"):
                    continue
                seen.add(attr_name)
                meta = obj._rule_meta
                rule_name = meta["name"]
                condition_spec = meta["condition"]
                priority = meta.get("priority", RulePriority.DEFAULT)
                cooldown = meta.get("cooldown", 0.0)

                # Bind condition to self so "lambda self: ..." works
                bound_method = getattr(self, attr_name)
                condition = lambda s=self, c=condition_spec: c(s)

                # Don't re-register if add_rule was already called with same name
                if not any(r.name == rule_name for r in self.decision_system.rules):
                    self.decision_system.add_rule(
                        name=rule_name,
                        condition=condition,
                        action=bound_method,
                        priority=priority,
                        cooldown=cooldown,
                    )

    # Convenience API that delegates to sub-systems

    def add_rule(
        self,
        name: str,
        condition,
        action,
        priority: RulePriority = RulePriority.DEFAULT,
        cooldown: float = 0.0,
    ):
        """Shortcut for self.decision_system.add_rule(...).

        condition is a zero-arg callable: ``lambda: self.hunger > 70``
        action is a zero-arg callable that returns a Task.
        """
        self.decision_system.add_rule(name, condition, action, priority, cooldown)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name."""
        return self.decision_system.remove_rule(name)

    def enable_rule(self, name: str) -> None:
        """Re-enable a disabled rule."""
        self.decision_system.enable_rule(name)

    def disable_rule(self, name: str) -> None:
        """Temporarily disable a rule without removing it."""
        self.decision_system.disable_rule(name)

    def schedule_task(self, task: Task) -> bool:
        """Directly schedule a task, bypassing the decision system."""
        return self.task_manager.schedule(task)

    def cancel_task(self) -> bool:
        """Cancel the currently running task."""
        return self.task_manager.cancel_current()

    def sync_states(self) -> dict[str, float]:
        """Materialise all lazy-decay BehavioralState values and fire threshold callbacks.

        Call this if you need decay-triggered threshold events (e.g. in NeedsAgent).
        """
        return BehavioralState.sync_all(self)

    # Properties

    @property
    def current_task(self) -> Task | None:
        """The currently executing Task, or None."""
        return self.task_manager.current_task

    @property
    def is_busy(self) -> bool:
        """True if a task is currently executing."""
        return self.task_manager.current_task is not None

    @property
    def task_progress(self) -> float:
        """Progress of the current task (0.0–1.0), or 0.0 if idle."""
        t = self.task_manager.current_task
        return t.progress if t is not None else 0.0

    @property
    def states(self) -> dict[str, float]:
        """Snapshot of all BehavioralState values (post-decay) on this agent."""
        return BehavioralState.get_all(self)

    # step() hook (Pattern A)

    def step(self) -> None:
        """Evaluate decision rules each step.

        Override to add custom logic — call super().step() to keep rule
        evaluation and state syncing active::

            def step(self):
                self.perceive_environment()
                super().step()
        """
        self.decision_system.evaluate()

    # Continuous-time hook (Pattern B)

    def on_action_complete(self, task: Task) -> None:
        """Called when the agent goes idle after a task completes.

        For step-based (Pattern A) models the default no-op is correct:
        step() re-evaluates rules each tick regardless.

        For continuous-time (Pattern B) models, override to re-evaluate
        immediately after a task finishes without waiting for the next step::

            class Scout(BDIAgent):
                def on_action_complete(self, task):
                    self.decision_system.evaluate()
        """

    # Lifecycle

    def remove(self) -> None:
        """Clean up tasks before removing the agent from the model."""
        if self.task_manager.current_task is not None:
            self.task_manager.cancel_current()

        # Clear the queue to prevent queued tasks from holding references
        # to this agent, which causes memory leaks.
        self.task_manager.clear_queue()

        super().remove()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(id={getattr(self, 'unique_id', '?')}, "
            f"task={self.current_task!r}, "
            f"queued={len(self.task_manager.task_queue)})"
        )
