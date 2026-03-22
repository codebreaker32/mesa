"""Task System for Mesa — Task properly inherits from Action.

Key design decisions
--------------------

Task(Action) is a real subclass, not a dataclass-that-happens-to-name-Action:

1. ``Task.__init__`` calls ``super().__init__()`` so every Action attribute
   (``self.state``, ``self._progress``, ``self._event``, ``self._start_time``,
   ``self.duration``, ``self.priority``, …) is correctly initialised.

2. Priority uses ``mesa.time.Priority`` (IntEnum), not a raw float.
   ``Task.start()`` overrides Action's to pass this to ``schedule_event``.

3. ``Task._do_complete()`` overrides Action's to route completion through
   TaskManager (so the queue is processed) rather than clearing
   ``agent.current_action`` which TaskManager doesn't use.

4. ``Task.interrupt()`` overrides Action's to honour ``reschedule_on_interrupt``
   policy and re-queue via TaskManager when appropriate.

5. TaskManager no longer creates Event objects directly — it just calls
   ``task.start()`` and lets Task/Action handle scheduling.

Typical usage
-------------

Inline (quick one-liners)::

    task = Task(agent, duration=3.0, action=lambda: agent.eat())
    agent.task_manager.schedule(task)

Subclassed (for complex lifecycle logic)::

    class HuntTask(Task):
        def __init__(self, wolf, target):
            super().__init__(wolf, duration=2.0, priority=Priority.HIGH)
            self.target = target

        def on_start(self):
            print(f"Wolf {self.agent.unique_id} starts hunting")

        def on_complete(self):
            if self.target in self.agent.model.agents:
                self.target.remove()

        def on_interrupt(self, progress):
            print(f"Hunt interrupted at {progress:.0%}")
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from enum import IntEnum, auto
from typing import Any, TYPE_CHECKING

from mesa.experimental.actions import Action, ActionState
from mesa.time import Priority
from mesa.experimental.mesa_signals.signals_util import SignalType

if TYPE_CHECKING:
    from mesa.agent import Agent
    from mesa.time import Event


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

class TaskSignals(SignalType):
    """Signals emitted by Tasks during their lifecycle."""

    STARTED = "started"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    RESUMED = "resumed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# TaskState — extends ActionState with PAUSED and FAILED
# ---------------------------------------------------------------------------

class TaskState(IntEnum):
    """Richer lifecycle state for Tasks.

    PAUSED is INTERRUPTED + will_resume (reschedule_on_interrupt != False).
    FAILED means requirements were not met.
    """

    PENDING = auto()
    ACTIVE = auto()
    PAUSED = auto()       # interrupted but queued for resumption
    COMPLETED = auto()
    INTERRUPTED = auto()  # interrupted and discarded
    FAILED = auto()


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class Task(Action):
    """A durative action managed by TaskManager.

    Extends Action with:
    - Mesa ``Priority`` enum (not raw float) for event scheduling
    - Requirements checking before start and before completion
    - ``reschedule_on_interrupt`` policy: "remainder", "full", or False
    - Reward calculation (linear, quadratic, threshold-based helpers below)
    - Optional ``action`` callback for one-liner task construction

    The ``task_state`` property gives the richer 6-way state that maps
    INTERRUPTED → PAUSED or INTERRUPTED depending on reschedule policy,
    and adds FAILED for requirements failures.

    Args:
        agent: The agent performing the task.
        duration: Seconds to complete, or ``callable(agent) -> float``.
        action: Optional zero-arg callable executed in ``on_complete``.
                Ignored if you override ``on_complete`` in a subclass.
        name: Human-readable name (defaults to class name).
        priority: ``Priority`` enum or ``callable(agent) -> Priority``.
        interruptible: Whether the task can be interrupted mid-execution.
        reschedule_on_interrupt:
            - ``"remainder"`` (default): pause and resume from where it left off.
            - ``"full"``: discard progress and restart from the beginning.
            - ``False``: discard entirely on interruption.
        requirements: ``{name: callable() -> bool}`` checked before start and
                      optionally before completion.
        reward: ``float`` constant, or ``callable(progress: float) -> float``.
    """

    def __init__(
        self,
        agent: "Agent",
        duration: float | Callable[["Agent"], float] = 1.0,
        *,
        action: Callable[[], Any] | None = None,
        name: str | None = None,
        priority: Priority | Callable[["Agent"], Priority] = Priority.DEFAULT,
        interruptible: bool = True,
        reschedule_on_interrupt: bool | str = "remainder",
        requirements: dict[str, Callable[[], bool]] | None = None,
        reward: Callable[[float], float] | float | None = None,
    ) -> None:
        # Action.__init__ sets: agent, model, interruptible, _name,
        # _duration_spec, _priority_spec, duration=0.0, priority=0.0,
        # state=PENDING, _progress=0.0, _start_time=-1.0, _event=None
        #
        # We pass priority=0.0 to Action (it uses float) but store the real
        # Priority enum spec ourselves; Task.start() uses it directly.
        super().__init__(
            agent,
            duration,
            name=name,
            priority=0.0,      # placeholder; Task overrides priority resolution
            interruptible=interruptible,
        )

        # Override Action's _priority_spec with the Priority enum version
        self._priority_spec: Priority | Callable[["Agent"], Priority] = priority

        # Task-specific attributes
        self._action_callback: Callable[[], Any] | None = action
        self.reschedule_on_interrupt: bool | str = reschedule_on_interrupt
        self.requirements: dict[str, Callable[[], bool]] = requirements or {}
        self.reward: Callable[[float], float] | float | None = reward

        # Resolved Priority enum, set in start() and available for queue sorting
        self._resolved_priority: Priority = (
            priority if isinstance(priority, Priority) else Priority.DEFAULT
        )
        self._failed: bool = False

    # ------------------------------------------------------------------
    # task_state — richer 6-way state view
    # ------------------------------------------------------------------

    @property
    def task_state(self) -> TaskState:
        """Richer state that includes PAUSED and FAILED.

        Mapping from Action's 4-state to Task's 6-state:

        - FAILED overrides everything when requirements have not been met.
        - PENDING  → TaskState.PENDING
        - ACTIVE   → TaskState.ACTIVE
        - COMPLETED → TaskState.COMPLETED
        - INTERRUPTED + reschedule != False → TaskState.PAUSED
        - INTERRUPTED + reschedule == False → TaskState.INTERRUPTED
        """
        if self._failed:
            return TaskState.FAILED
        mapping = {
            ActionState.PENDING: TaskState.PENDING,
            ActionState.ACTIVE: TaskState.ACTIVE,
            ActionState.COMPLETED: TaskState.COMPLETED,
            ActionState.INTERRUPTED: (
                TaskState.PAUSED
                if self.reschedule_on_interrupt and self._progress < 1.0
                else TaskState.INTERRUPTED
            ),
        }
        return mapping.get(self.state, TaskState.PENDING)

    @property
    def is_resumable(self) -> bool:
        """True if the task was paused and can continue from current progress."""
        return (
            self.state is ActionState.INTERRUPTED
            and self._progress < 1.0
            and self.reschedule_on_interrupt == "remainder"
        )

    # ------------------------------------------------------------------
    # Lifecycle hooks (override in subclasses)
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Called when the task first begins. Override in subclasses."""

    def on_resume(self) -> None:
        """Called when the task resumes after interruption. Defaults to on_start."""
        self.on_start()

    def on_complete(self) -> Any:
        """Called when the task finishes normally.

        If ``action`` was passed to ``__init__``, it is called here.
        Override in subclasses for richer logic.
        """
        if self._action_callback is not None:
            return self._action_callback()
        return None

    def on_interrupt(self, progress: float) -> None:
        """Called when interrupted. ``progress`` is the fraction completed (0–1)."""

    # ------------------------------------------------------------------
    # Requirements
    # ------------------------------------------------------------------

    def check_requirements(self) -> tuple[bool, list[str]]:
        """Evaluate all requirements. Returns ``(all_passed, [failed_names])``."""
        if not self.requirements:
            return True, []
        failed = []
        for req_name, condition in self.requirements.items():
            try:
                if not condition():
                    failed.append(req_name)
            except Exception as exc:
                failed.append(f"{req_name} (error: {exc})")
        return len(failed) == 0, failed

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def calculate_reward(self) -> float:
        """Return the reward for the task's current progress."""
        if self.reward is None:
            return 0.0
        return self.reward(self._progress) if callable(self.reward) else float(self.reward) * self._progress

    # ------------------------------------------------------------------
    # start() — full override to use Priority enum in schedule_event
    # ------------------------------------------------------------------

    def start(self) -> Task:
        """Start (or resume) this task.

        Differences from ``Action.start()``:

        - Checks requirements before first start; marks task FAILED and raises
          ``TaskRequirementsError`` on failure.
        - Resolves priority to a ``Priority`` enum, not a float.
        - Passes ``priority=`` to ``model.schedule_event`` so Mesa's event queue
          orders tasks correctly.
        - On completion, ``_do_complete`` notifies TaskManager instead of
          clearing ``agent.current_action``.
        """
        resuming = self.state is ActionState.INTERRUPTED

        if self.state not in (ActionState.PENDING, ActionState.INTERRUPTED):
            raise ValueError(
                f"Cannot start Task in {self.state.name} state. "
                f"Only PENDING or INTERRUPTED tasks can be started."
            )

        if not resuming:
            # --- Requirements check ---
            ok, failed_reqs = self.check_requirements()
            if not ok:
                self._failed = True
                self.state = ActionState.INTERRUPTED  # closest Action state
                raise TaskRequirementsError(self, failed_reqs)

            # --- Resolve duration ---
            self.duration = (
                self._duration_spec(self.agent)
                if callable(self._duration_spec)
                else self._duration_spec
            )
            if self.duration < 0:
                raise ValueError(f"Task duration must be >= 0, got {self.duration}")

            # --- Resolve priority ---
            self._resolved_priority = (
                self._priority_spec(self.agent)
                if callable(self._priority_spec)
                else self._priority_spec
            )
            # Keep Action's float priority attribute in sync (for __repr__ etc.)
            self.priority = self._resolved_priority

        self._start_time = self.agent.model.time
        self.state = ActionState.ACTIVE

        if resuming:
            self.on_resume()
        else:
            self.on_start()

        remaining = self.duration * (1.0 - self._progress)

        if remaining <= 0:
            self._do_complete()
            return self

        # Schedule with the Priority enum so Mesa's heap orders correctly
        self._event = self.agent.model.schedule_event(
            self._do_complete,
            after=remaining,
            priority=self._resolved_priority,
        )
        return self

    # ------------------------------------------------------------------
    # _do_complete() — notifies TaskManager instead of agent.current_action
    # ------------------------------------------------------------------

    def _do_complete(self) -> None:
        """Called by Mesa's event scheduler when the task duration elapses.

        Overrides Action._do_complete() to route through TaskManager so the
        task queue is processed after completion.
        """
        if self.state is not ActionState.ACTIVE:
            return

        self._progress = 1.0
        self._event = None
        self.state = ActionState.COMPLETED

        # Route through TaskManager if the agent has one (BehavioralAgent)
        if hasattr(self.agent, "task_manager"):
            self.agent.task_manager._on_task_completed(self)
        else:
            # Standalone usage (no TaskManager): just call on_complete directly
            self.on_complete()

    # ------------------------------------------------------------------
    # interrupt() — applies reschedule_on_interrupt policy
    # ------------------------------------------------------------------

    def interrupt(self) -> bool:
        """Interrupt this task, applying the reschedule_on_interrupt policy.

        - ``"remainder"``: freezes progress → TaskState.PAUSED, re-queued by TaskManager
        - ``"full"``: resets progress → TaskState.PENDING, re-queued by TaskManager
        - ``False``: discards task → TaskState.INTERRUPTED

        Returns:
            True if the interrupt succeeded, False if non-interruptible or not active.
        """
        if self.state is not ActionState.ACTIVE:
            return False
        if not self.interruptible:
            return False

        # Freeze the live progress into _progress
        self._freeze_progress()
        self._cancel_event()
        self.state = ActionState.INTERRUPTED

        # Apply reschedule policy
        if self.reschedule_on_interrupt == "full":
            # Reset completely — will be re-queued as a fresh run
            self._progress = 0.0

        # Fire the user hook
        self.on_interrupt(self._progress)
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _priority_value(self) -> int:
        """Return an int suitable for queue sorting (lower = higher priority)."""
        return self._resolved_priority.value

    def __repr__(self) -> str:
        return (
            f"{self.name}("
            f"state={self.task_state.name}, "
            f"progress={self.progress:.0%}, "
            f"duration={self.duration}, "
            f"priority={self._resolved_priority.name})"
        )


# ---------------------------------------------------------------------------
# TaskRequirementsError
# ---------------------------------------------------------------------------

class TaskRequirementsError(RuntimeError):
    """Raised when a Task cannot start because requirements are not met."""

    def __init__(self, task: Task, failed: list[str]):
        self.task = task
        self.failed = failed
        super().__init__(
            f"Task '{task.name}' requirements not met: {failed}"
        )


# ---------------------------------------------------------------------------
# Resource — shared object with limited capacity and waiting queue
# ---------------------------------------------------------------------------

class Resource:
    """A shared object with limited capacity and a waiting queue.

    Tasks can request access. If capacity is full they are queued. When
    a task releases the resource, the next queued task is granted access
    and scheduled automatically.
    """

    def __init__(self, model, capacity: int = 1):
        self.model = model
        self.capacity = capacity
        self.available = capacity
        self.queue: list[Task] = []
        self.active: set[Task] = set()

    def request(self, task: Task) -> None:
        """Request access; queues the task if capacity is full."""
        if self.available > 0:
            self._grant(task)
        else:
            self.queue.append(task)

    def release(self, task: Task) -> None:
        """Release the resource and serve the next waiting task."""
        if task in self.active:
            self.active.discard(task)
            self.available += 1
            self._serve_next()

    def remove(self, task: Task) -> None:
        """Remove a task from the waiting queue without granting access."""
        if task in self.queue:
            self.queue.remove(task)

    def _grant(self, task: Task) -> None:
        self.available -= 1
        self.active.add(task)
        if hasattr(task.agent, "task_manager"):
            task.agent.task_manager.schedule(task)

    def _serve_next(self) -> None:
        while self.queue and self.available > 0:
            next_task = self.queue.pop(0)
            # Skip tasks whose agent has been removed from the model
            agent = next_task.agent
            if (
                hasattr(agent, "model")
                and agent.model is not None
                and agent in agent.model.agents
            ):
                self._grant(next_task)


# ---------------------------------------------------------------------------
# TaskManager
# ---------------------------------------------------------------------------

class TaskManager:
    """Manages tasks for a single BehavioralAgent.

    Responsibilities
    ----------------
    - Schedule incoming tasks: start immediately if idle, otherwise enqueue
    - Handle priority-based interruption of the current task
    - Process the queue when the current task finishes or is cancelled
    - Emit TaskSignals via agent.notify() when present

    The actual event scheduling is done by ``Task.start()``; TaskManager
    no longer creates ``Event`` objects directly.
    """

    def __init__(self, agent: "Agent"):
        self.agent = agent
        self.current_task: Task | None = None
        self.task_queue: list[Task] = []
        self._completed_tasks: list[Task] = []
        self._interrupted_tasks: list[Task] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule(self, task: Task) -> bool:
        """Schedule a task for execution.

        If no task is running, starts immediately.
        If a lower-priority task is running and the new task is interruptible,
        interrupts it and starts the new one.
        Otherwise queues the task sorted by priority.

        Returns:
            True if the task was accepted (started or queued).
            False if the task failed requirements at scheduling time.
        """
        # Fast-fail for tasks whose requirements are already broken
        ok, failed = task.check_requirements()
        if not ok:
            task._failed = True
            self._emit(task, TaskSignals.FAILED, failed_requirements=failed)
            return False

        if self.current_task is None:
            self._start_task(task)
            return True

        incoming_prio = task._resolved_priority
        current_prio = self.current_task._resolved_priority

        # Lower .value means higher priority in Mesa's Priority enum
        if incoming_prio.value < current_prio.value and self.current_task.interruptible:
            self._interrupt_current_task()
            self._start_task(task)
            return True

        self._enqueue(task)
        return True

    def cancel_current(self) -> bool:
        """Cancel the currently executing task and start the next queued one.

        Returns:
            True if there was a task to cancel, False if idle.
        """
        if self.current_task is None:
            return False
        self._interrupt_current_task()
        self._process_queue()
        return True

    def clear_queue(self) -> int:
        """Discard all queued (not-yet-started) tasks. Returns the count removed."""
        count = len(self.task_queue)
        self.task_queue.clear()
        return count

    def get_stats(self) -> dict[str, Any]:
        """Summary statistics for monitoring / data collection."""
        current = self.current_task
        return {
            "completed": len(self._completed_tasks),
            "interrupted": len(self._interrupted_tasks),
            "queued": len(self.task_queue),
            "current": current.name if current else None,
            "current_state": current.task_state.name if current else None,
            "current_progress": current.progress if current else 0.0,
        }

    # ------------------------------------------------------------------
    # Callbacks from Task (not called directly by users)
    # ------------------------------------------------------------------

    def _on_task_completed(self, task: Task) -> None:
        """Called by Task._do_complete() when a task finishes normally."""
        if self.current_task is not task:
            # Stale event from a cancelled task — ignore
            return

        # Final requirements check (post-completion guard)
        ok, failed = task.check_requirements()
        if not ok:
            task._failed = True
            task.state = ActionState.INTERRUPTED
            self._emit(task, TaskSignals.FAILED, failed_requirements=failed)
            self.current_task = None
            self._process_queue()
            return

        # MUST clear current_task BEFORE calling on_complete().
        # If on_complete() calls agent.remove(), BehavioralAgent.remove() checks
        # current_task. If it's still set, it calls cancel_current() on an already-
        # completed task, corrupting state. Clearing first makes remove() a no-op
        # for the task-manager path.
        self.current_task = None

        try:
            result = task.on_complete()
            reward = task.calculate_reward()
            self._emit(task, TaskSignals.COMPLETED, result=result, reward=reward)
            self._completed_tasks.append(task)
        except Exception as exc:
            task._failed = True
            self._emit(task, TaskSignals.FAILED, error=str(exc))

        self._process_queue()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _start_task(self, task: Task) -> None:
        """Start a task immediately via Task.start()."""
        resuming = task.state is ActionState.INTERRUPTED

        self.current_task = task
        self._emit(task, TaskSignals.RESUMED if resuming else TaskSignals.STARTED)

        try:
            task.start()  # Task.start() handles scheduling, hooks, event creation
        except TaskRequirementsError as exc:
            task._failed = True
            self._emit(task, TaskSignals.FAILED, failed_requirements=exc.failed)
            self.current_task = None
            self._process_queue()
        except Exception as exc:
            task._failed = True
            self._emit(task, TaskSignals.FAILED, error=str(exc))
            self.current_task = None
            self._process_queue()

    def _interrupt_current_task(self) -> None:
        """Interrupt the current task and apply its reschedule policy."""
        if self.current_task is None:
            return

        task = self.current_task
        self.current_task = None  # clear before task.interrupt() so _do_complete ignores it

        interrupted = task.interrupt()  # freezes progress, cancels event, calls on_interrupt hook

        # task.interrupt() returns False if the task was already completed or non-interruptible.
        # In that case there is nothing to re-queue or discard — just bail out.
        if not interrupted:
            return

        self._emit(task, TaskSignals.INTERRUPTED, progress=task._progress)

        # Apply re-queue policy
        if task.reschedule_on_interrupt == "remainder":
            # task.state is INTERRUPTED + progress preserved → PAUSED in task_state
            self._enqueue(task)
        elif task.reschedule_on_interrupt == "full":
            # task._progress was reset to 0.0 in Task.interrupt()
            task.state = ActionState.PENDING
            self._enqueue(task)
        else:
            # Discard
            self._interrupted_tasks.append(task)

    def _enqueue(self, task: Task) -> None:
        """Add task to the queue, sorted by priority then by start time."""
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: (t._priority_value(), t._start_time))

    def _process_queue(self) -> None:
        """Pop and start the next task from the queue, if any."""
        while self.task_queue:
            next_task = self.task_queue.pop(0)
            # Skip tasks that have been externally failed or completed
            if next_task.task_state in (TaskState.COMPLETED, TaskState.FAILED):
                continue
            self._start_task(next_task)
            return

    def _emit(self, task: Task, signal: TaskSignals, **kwargs: Any) -> None:
        """Emit a task lifecycle signal via the agent if it supports notify()."""
        if hasattr(self.agent, "notify"):
            self.agent.notify("task_manager", signal, task=task, **kwargs)


# ---------------------------------------------------------------------------
# Reward helper functions
# ---------------------------------------------------------------------------

def linear_reward(base_value: float) -> Callable[[float], float]:
    """Reward proportional to progress: ``base_value * progress``."""
    return lambda p: base_value * p


def quadratic_reward(base_value: float) -> Callable[[float], float]:
    """Reward scaling as progress²."""
    return lambda p: base_value * (p ** 2)


def threshold_reward(base_value: float, threshold: float = 0.8) -> Callable[[float], float]:
    """Full reward only when progress reaches *threshold*, zero otherwise."""
    return lambda p: base_value if p >= threshold else 0.0


def exponential_reward(base_value: float, rate: float = 2.0) -> Callable[[float], float]:
    """Exponentially increasing reward (near-zero until close to completion)."""
    import math
    return lambda p: base_value * math.exp(rate * p - rate)