"""mesa.experimental.behaviorals — Behavioral framework for Mesa agents.

Provides:
  BehavioralAgent   — Agent subclass with TaskManager + DecisionSystem built in
  BehavioralState   — Lazy-decay numeric state descriptor
  rule              — Decorator for declaring decision rules on agent classes
  RulePriority      — Priority enum for rules (CRITICAL > URGENT > HIGH > DEFAULT > LOW)
  Task              — Durative action that properly inherits from Action
  TaskManager       — Manages task scheduling, interruption, and queuing
  TaskSignals       — Signal enum for task lifecycle events
  TaskState         — State enum for tasks (PENDING/ACTIVE/PAUSED/COMPLETED/INTERRUPTED/FAILED)
  TaskRequirementsError — Raised when task requirements are not met

Patterns (concrete subclasses of BehavioralAgent):
  NeedsAgent        — Homeostatic needs-satisfaction (hunger, thirst, …)
  BDIAgent          — Belief-Desire-Intention deliberative architecture
  Desire            — Dataclass for BDIAgent goals
  RLAgent           — Tabular epsilon-greedy Q-learning

Drop all files in this directory into ``mesa/experimental/behaviorals/``.
"""

from mesa.experimental.behaviorals.behavioral_agent import BehavioralAgent
from mesa.experimental.behaviorals.decision import RulePriority, rule
from mesa.experimental.behaviorals.patterns import BDIAgent, Desire, NeedsAgent, RLAgent
from mesa.experimental.behaviorals.state import BehavioralState, StateSignal
from mesa.experimental.behaviorals.task import (
    Resource,
    Task,
    TaskManager,
    TaskRequirementsError,
    TaskSignals,
    TaskState,
    exponential_reward,
    linear_reward,
    quadratic_reward,
    threshold_reward,
)

__all__ = [
    # Core
    "BehavioralAgent",
    "BehavioralState",
    "StateSignal",
    # Decision
    "rule",
    "RulePriority",
    # Task
    "Task",
    "TaskManager",
    "TaskSignals",
    "TaskState",
    "TaskRequirementsError",
    "Resource",
    # Reward helpers
    "linear_reward",
    "quadratic_reward",
    "threshold_reward",
    "exponential_reward",
    # Patterns
    "NeedsAgent",
    "BDIAgent",
    "Desire",
    "RLAgent",
]