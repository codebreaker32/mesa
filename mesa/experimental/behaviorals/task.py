"""Task System for Mesa - Extends mesa.experimental.actions.Action"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any

from mesa.experimental.actions import Action
from mesa.experimental.mesa_signals.signals_util import SignalType
from mesa.time import Event, Priority

if TYPE_CHECKING:
    pass


class TaskSignals(SignalType):
    """Signals emitted by Tasks during their lifecycle."""

    STARTED = "started"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    RESUMED = "resumed"
    FAILED = "failed"


class TaskState(IntEnum):
    """Current execution status of a task."""

    PENDING = auto()
    ACTIVE = auto()
    PAUSED = auto()
    COMPLETED = auto()
    INTERRUPTED = auto()
    FAILED = auto()


@dataclass(slots=True)
class Task(Action):
    """A durative action that takes time to complete.

    Extends Mesa's Action class with requirements, rewards, and rescheduling policies.
    Tasks integrate with Mesa's scheduler to schedule completion events.
    They can be interrupted by higher-priority tasks and re-scheduled based on
    their `reschedule_on_interrupt` policy.

    Attributes:
        agent: Agent performing the task
        duration: Time required (in model time units) or callable(agent) -> float
        action: Optional Callable to execute on completion (if not subclassing)
        priority: Mesa Priority enum or callable(agent) -> Priority
        interruptible: Whether task can be interrupted
        reschedule_on_interrupt: False (discard), "remainder" (resume), or "full" (restart)
        requirements: Dict of {name: condition} that must be True
        reward: Callable or constant reward value
    """

    agent: Any
    duration: float | Callable[[Any], float]
    action: Callable[[], Any] | None = None
    priority: Priority | Callable[[Any], Priority] = Priority.DEFAULT
    interruptible: bool = True
    reschedule_on_interrupt: bool | str = "remainder"
    requirements: dict[str, Callable[[], bool]] | None = None
    reward: Callable[[float], float] | float | None = None

    # Internal state
    _state: TaskState = field(default=TaskState.PENDING, init=False)
    _progress: float = field(default=0.0, init=False)
    _start_time: float = field(default=0.0, init=False)
    _event_ref: weakref.ref | None = field(default=None, init=False)
    _pause_time: float = field(default=0.0, init=False)
    _resolved_duration: float = field(default=0.0, init=False)
    _resolved_priority: Priority = field(default=Priority.DEFAULT, init=False)

    def on_start(self):
        """Lifecycle hook: Called when the task begins execution. Override in subclasses."""

    def on_resume(self):
        """Lifecycle hook: Called when the task resumes after interruption. Override in subclasses."""
        self.on_start()

    def on_complete(self) -> Any:
        """Lifecycle hook: Called when the task finishes normally. Override in subclasses."""
        if self.action:
            return self.action()
        return None

    def on_interrupt(self, progress: float):
        """Lifecycle hook: Called when the task is interrupted. Override in subclasses.

        Args:
            progress: Fraction of duration completed (0.0 to 1.0)
        """

    @property
    def state(self) -> TaskState:
        return self._state

    @property
    def progress(self) -> float:
        if (
            self._state == TaskState.ACTIVE
            and self._resolved_duration > 0
            and self._start_time >= 0
        ):
            elapsed = self.agent.model.time - self._start_time
            return min(1.0, self._progress + elapsed / self._resolved_duration)
        return self._progress

    @property
    def elapsed_time(self) -> float:
        if self._state == TaskState.ACTIVE:
            return self.agent.model.time - self._start_time
        elif self._state == TaskState.PAUSED:
            return self._pause_time - self._start_time
        return 0.0

    @property
    def remaining_time(self) -> float:
        return max(0.0, self._resolved_duration * (1.0 - self.progress))

    @property
    def is_resumable(self) -> bool:
        return self._state == TaskState.PAUSED and self._progress < 1.0

    def check_requirements(self) -> tuple[bool, list[str]]:
        """Validate all requirements are met."""
        if not self.requirements:
            return True, []

        failed = []
        for name, condition in self.requirements.items():
            try:
                if not condition():
                    failed.append(name)
            except Exception as e:
                failed.append(f"{name} (error: {e})")

        return len(failed) == 0, failed

    def calculate_reward(self) -> float:
        """Calculate reward based on progress."""
        if self.reward is None:
            return 0.0
        return (
            self.reward(self._progress)
            if callable(self.reward)
            else self.reward * self._progress
        )


class Resource:
    """A shared object with limited capacity and a waiting queue.

    Tasks can request access to this resource. If capacity is full,
    they are queued. When a task releases the resource, the next
    task in the queue is automatically granted access and scheduled.
    """

    def __init__(self, model, capacity: int = 1):
        self.model = model
        self.capacity = capacity
        self.available = capacity
        self.queue: list[Task] = []
        self.active: set[Task] = set()

    def request(self, task: Task):
        """Request access to the resource. If full, adds task to queue."""
        if self.available > 0:
            self._grant(task)
        else:
            self.queue.append(task)

    def _grant(self, task: Task):
        self.available -= 1
        self.active.add(task)
        if hasattr(task.agent, "task_manager"):
            task.agent.task_manager.schedule(task)

    def release(self, task: Task):
        """Release the resource and serve the next task in the queue."""
        if task in self.active:
            self.active.discard(task)
            self.available += 1
            self._serve_next()

    def remove(self, task: Task):
        """Remove a task from the waiting queue."""
        if task in self.queue:
            self.queue.remove(task)

    def _serve_next(self):
        while self.queue and self.available > 0:
            next_task = self.queue.pop(0)
            if (
                hasattr(next_task.agent, "model")
                and next_task.agent.model is not None
                and next_task.agent not in next_task.agent.model.agents
            ):
                continue
            self._grant(next_task)


class TaskManager:
    """Manages tasks for an agent, integrating with Mesa's event system."""

    def __init__(self, agent):
        self.agent = agent
        self.current_task: Task | None = None
        self.task_queue: list[Task] = []
        self._completed_tasks: list[Task] = []
        self._interrupted_tasks: list[Task] = []

    def schedule(self, task: Task) -> bool:
        """Schedule a task for execution."""
        can_run, failed = task.check_requirements()
        if not can_run:
            task._state = TaskState.FAILED
            self._emit_signal(task, TaskSignals.FAILED, failed_requirements=failed)
            return False

        if self.current_task is None:
            self._start_task(task)
            return True

        task_priority = (
            task.priority(task.agent) if callable(task.priority) else task.priority
        )
        current_priority = self.current_task._resolved_priority

        if task_priority.value < current_priority.value:
            if self.current_task.interruptible:
                self._interrupt_current_task()
                self._start_task(task)
                return True

        self._enqueue_task(task)
        return True

    def cancel_current(self) -> bool:
        """Cancel currently executing task."""
        if self.current_task is None:
            return False
        self._interrupt_current_task()
        self._process_queue()
        return True

    def get_current(self) -> Task | None:
        return self.current_task

    def get_queue(self) -> list[Task]:
        return list(self.task_queue)

    def clear_queue(self) -> int:
        count = len(self.task_queue)
        self.task_queue.clear()
        return count

    def _start_task(self, task: Task):
        """Begin executing a task using Mesa's Event."""
        resuming = task._state == TaskState.PAUSED

        if not resuming:
            task._resolved_duration = (
                task.duration(task.agent) if callable(task.duration) else task.duration
            )
            task._resolved_priority = (
                task.priority(task.agent) if callable(task.priority) else task.priority
            )

            if task._resolved_duration < 0:
                raise ValueError(
                    f"Task duration must be >= 0, got {task._resolved_duration}"
                )

        task._state = TaskState.ACTIVE
        task._start_time = self.agent.model.time
        self.current_task = task

        self._emit_signal(
            task, TaskSignals.RESUMED if resuming else TaskSignals.STARTED
        )

        if resuming:
            task.on_resume()
        else:
            task.on_start()

        remaining = task._resolved_duration * (1.0 - task._progress)

        if remaining <= 0:
            self._complete_task(task)
        else:
            completion_time = self.agent.model.time + remaining
            event = Event(
                time=completion_time,
                function=self._complete_task,
                priority=task._resolved_priority,
                function_args=[task],
            )
            self.agent.model._event_list.add_event(event)
            task._event_ref = weakref.ref(event)

    def _complete_task(self, task: Task):
        """Called by Mesa's event scheduler when task duration elapses."""
        if self.current_task is not task:
            return

        can_complete, failed = task.check_requirements()
        if not can_complete:
            task._state = TaskState.FAILED
            self._emit_signal(task, TaskSignals.FAILED, failed_requirements=failed)
            self.current_task = None
            self._process_queue()
            return

        task._state = TaskState.COMPLETED
        task._progress = 1.0

        try:
            result = task.on_complete()
            reward = task.calculate_reward()
            self._emit_signal(task, TaskSignals.COMPLETED, result=result, reward=reward)
            self._completed_tasks.append(task)
        except Exception as e:
            task._state = TaskState.FAILED
            self._emit_signal(task, TaskSignals.FAILED, error=str(e))

        self.current_task = None
        self._process_queue()

    def _interrupt_current_task(self):
        """Interrupt current task and apply reschedule policy."""
        if self.current_task is None:
            return

        task = self.current_task

        if task._event_ref is not None:
            event = task._event_ref()
            if event is not None:
                event.cancel()

        if task._resolved_duration > 0 and task._start_time >= 0:
            elapsed = self.agent.model.time - task._start_time
            task._progress = min(
                1.0, task._progress + elapsed / task._resolved_duration
            )
        else:
            task._progress = 1.0

        task._pause_time = self.agent.model.time

        task.on_interrupt(task._progress)

        if task.reschedule_on_interrupt == "remainder":
            task._state = TaskState.PAUSED
            self._enqueue_task(task)
        elif task.reschedule_on_interrupt == "full":
            task._state = TaskState.PENDING
            task._progress = 0.0
            task._pause_time = 0.0
            task._start_time = 0.0
            self._enqueue_task(task)
        else:
            task._state = TaskState.INTERRUPTED
            self._interrupted_tasks.append(task)

        self._emit_signal(task, TaskSignals.INTERRUPTED, progress=task._progress)
        self.current_task = None

    def _enqueue_task(self, task: Task):
        """Add task to queue sorted by priority."""
        if task._state not in (TaskState.PENDING, TaskState.PAUSED):
            task._state = TaskState.PENDING

        self.task_queue.append(task)
        self.task_queue.sort(
            key=lambda t: (
                t._resolved_priority.value if hasattr(t, "_resolved_priority") else 5,
                t._start_time,
            )
        )

    def _process_queue(self):
        """Start next task from queue if available."""
        if not self.task_queue:
            return
        next_task = self.task_queue.pop(0)
        self._start_task(next_task)

    def _emit_signal(self, task: Task, signal_type: TaskSignals, **kwargs):
        """Emit task lifecycle signal."""
        if hasattr(self, "_has_subscribers") and not self._has_subscribers(
            "tasks", signal_type
        ):
            return

        if hasattr(self.agent, "notify"):
            self.agent.notify("task_manager", signal_type, task=task, **kwargs)

    def get_stats(self) -> dict[str, Any]:
        """Get task execution statistics."""
        return {
            "completed": len(self._completed_tasks),
            "interrupted": len(self._interrupted_tasks),
            "current": (
                getattr(
                    self.current_task.action, "__name__", str(self.current_task.action)
                )
                if self.current_task and self.current_task.action
                else None
            ),
            "queued": len(self.task_queue),
            "current_progress": self.current_task.progress
            if self.current_task
            else 0.0,
        }


# Reward Functions


def linear_reward(base_value: float) -> Callable[[float], float]:
    return lambda progress: base_value * progress


def quadratic_reward(base_value: float) -> Callable[[float], float]:
    return lambda progress: base_value * (progress**2)


def threshold_reward(
    base_value: float, threshold: float = 0.8
) -> Callable[[float], float]:
    return lambda progress: base_value if progress >= threshold else 0.0


def exponential_reward(
    base_value: float, rate: float = 2.0
) -> Callable[[float], float]:
    import math

    return lambda progress: base_value * (math.exp(rate * progress - rate))
