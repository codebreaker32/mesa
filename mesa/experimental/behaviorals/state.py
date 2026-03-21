"""Behavioral State System for Mesa"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any

from mesa.experimental.mesa_signals.core import BaseObservable, HasEmitters
from mesa.experimental.mesa_signals.signals_util import SignalType
from mesa.time import Priority

if TYPE_CHECKING:
    pass


class BehavioralSignals(SignalType):
    """Signals emitted by BehavioralState."""

    DECAYED = "decayed"
    GREW = "grew"
    THRESHOLD_CROSSED = "threshold_crossed"


class BehavioralState(BaseObservable):
    """A descriptor for agent state variables with automatic decay/growth.

    Automatically schedules decay/growth events when set, emits signals
    when thresholds are crossed, and supports lazy evaluation for performance.

    Usage:
        class Agent(CellAgent, HasEmitters):
            energy = BehavioralState(decay_rate=-1.0)

            def __init__(self, model):
                super().__init__(model)
                self.energy = 100  # Locks in value, schedules decay
    """

    def __init__(
        self,
        decay_rate: float = 0.0,
        min_value: float | None = None,
        max_value: float | None = None,
        thresholds: dict[str, float] | None = None,
    ):
        super().__init__()
        self.decay_rate = decay_rate
        self.min_value = min_value
        self.max_value = max_value
        self.thresholds = thresholds or {}

        # Track which thresholds have been crossed per agent instance
        self._crossed_thresholds: dict[int, set[str]] = {}

    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = f"_{name}_value"
        self.event_ref_name = f"_{name}_event_ref"

    def __get__(self, obj: Any, objtype=None) -> float | BehavioralState:
        if obj is None:
            return self
        return getattr(obj, self.private_name, 0.0)

    def __set__(self, obj: Any, value: float):
        # Cancel existing decay event if any
        if hasattr(obj, self.event_ref_name):
            event_ref = getattr(obj, self.event_ref_name)
            if event_ref is not None:
                event = event_ref()
                if event is not None:
                    event.cancel()

        # Clamp value to min/max
        if self.min_value is not None:
            value = max(self.min_value, value)
        if self.max_value is not None:
            value = min(self.max_value, value)

        old_value = getattr(obj, self.private_name, value)
        setattr(obj, self.private_name, value)

        # Check thresholds
        self._check_thresholds(obj, old_value, value)

        # Schedule decay if rate is non-zero
        if self.decay_rate != 0.0 and hasattr(obj, "model"):
            event = obj.model.schedule_recurring(
                lambda: self._apply_decay(obj),
                schedule=obj.model._default_schedule._schedule,
                priority=Priority.DEFAULT,
            )
            # Store weak reference to the EventGenerator
            setattr(obj, self.event_ref_name, weakref.ref(event))

        # Emit signal
        if isinstance(obj, HasEmitters):
            signal = (
                BehavioralSignals.GREW
                if value > old_value
                else BehavioralSignals.DECAYED
            )
            obj.notify(self.name, signal, old_value=old_value, new_value=value)

    def _apply_decay(self, obj: Any):
        """Apply decay to the state variable."""
        current = getattr(obj, self.private_name, 0.0)
        new_value = current + self.decay_rate

        # Clamp to bounds
        if self.min_value is not None:
            new_value = max(self.min_value, new_value)
        if self.max_value is not None:
            new_value = min(self.max_value, new_value)

        # Check thresholds
        self._check_thresholds(obj, current, new_value)

        setattr(obj, self.private_name, new_value)

        # Emit decay signal
        if isinstance(obj, HasEmitters):
            signal = (
                BehavioralSignals.GREW
                if new_value > current
                else BehavioralSignals.DECAYED
            )
            obj.notify(self.name, signal, old_value=current, new_value=new_value)

    def _check_thresholds(self, obj: Any, old_value: float, new_value: float):
        """Check if any thresholds were crossed and emit signals."""
        if not self.thresholds or not isinstance(obj, HasEmitters):
            return

        agent_id = id(obj)
        if agent_id not in self._crossed_thresholds:
            self._crossed_thresholds[agent_id] = set()

        for threshold_name, threshold_value in self.thresholds.items():
            # Detect crossing (works for both directions)
            crossed_down = old_value >= threshold_value > new_value
            crossed_up = old_value <= threshold_value < new_value

            if crossed_down or crossed_up:
                direction = "down" if crossed_down else "up"
                obj.notify(
                    self.name,
                    BehavioralSignals.THRESHOLD_CROSSED,
                    threshold=threshold_name,
                    value=threshold_value,
                    direction=direction,
                    old_value=old_value,
                    new_value=new_value,
                )
                self._crossed_thresholds[agent_id].add(threshold_name)
