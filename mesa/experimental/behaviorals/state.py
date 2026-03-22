"""Behavioral State System for Mesa.

BehavioralState is a plain Python descriptor — it does NOT inherit from
BaseObservable or depend on HasEmitters. The original version broke because:

  1. Agent is not a HasEmitters, so obj.notify() raised AttributeError.
  2. schedule_recurring() returns a generator with no strong reference —
     it was garbage-collected immediately, so decay never ran.
  3. BaseObservable.__get__ has no fallback: accessing an unset attribute
     raised AttributeError on first read.

Fix: lazy decay (computed on __get__), no scheduler, lightweight callback
system that works on any object.

Usage::

    class Sheep(BehavioralAgent):
        # Hunger rises 0.05 per time unit, clamped to [0, 100]
        hunger = BehavioralState(
            decay_rate=0.05,
            min_value=0.0,
            max_value=100.0,
            thresholds={"hungry": 60.0, "starving": 85.0},
        )
        energy = BehavioralState(
            decay_rate=-0.03,   # negative = grows over time (recovery)
            min_value=0.0,
            max_value=1.0,
            initial=1.0,
        )

        def __init__(self, model):
            super().__init__(model)
            self.hunger = 0.0
            self.energy = 1.0

            # React to threshold crossings
            BehavioralState.on_threshold(self, "hunger", "hungry",
                                         lambda old, new: print(f"Agent {self.unique_id} is hungry!"))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Signals (lightweight, no Mesa dependency)
# ---------------------------------------------------------------------------

class StateSignal:
    """Emitted by BehavioralState callbacks. Mirrors Mesa Message shape."""

    __slots__ = ("attr", "owner", "old_value", "new_value", "threshold", "direction")

    def __init__(
        self,
        attr: str,
        owner: Any,
        old_value: float,
        new_value: float,
        threshold: str | None = None,
        direction: str | None = None,
    ):
        self.attr = attr
        self.owner = owner
        self.old_value = old_value
        self.new_value = new_value
        self.threshold = threshold
        self.direction = direction

    def __repr__(self) -> str:
        return (
            f"StateSignal({self.attr!r}, "
            f"old={self.old_value:.3f}, new={self.new_value:.3f}"
            + (f", threshold={self.threshold!r}({self.direction})" if self.threshold else "")
            + ")"
        )


# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------

class BehavioralState:
    """A descriptor for a numeric agent state variable with optional time-based decay.

    Decay is computed **lazily on read** — no event scheduling, no weak
    references, no hidden timers. The value stored is always ``(raw, timestamp)``;
    reading applies ``raw + decay_rate * elapsed`` on the fly.

    Args:
        decay_rate: Change per model-time-unit (positive = grows, negative = shrinks).
                    0.0 means no decay.
        min_value: Lower clamp. None means unbounded.
        max_value: Upper clamp. None means unbounded.
        thresholds: Named crossing points, e.g. ``{"hungry": 60.0, "starving": 85.0}``.
                    Threshold callbacks fire on explicit ``__set__`` only (not during
                    lazy decay). Call ``agent.sync_state("attr")`` to materialise lazy
                    decay and trigger any crossed thresholds.
        initial: Value returned before the attribute has ever been set.

    Threshold callbacks (fire on explicit set and on sync_state):
        Use ``BehavioralState.on_threshold(agent, attr, name, fn)`` or
        ``BehavioralState.on_change(agent, attr, fn)`` after ``__init__``.
    """

    def __init__(
        self,
        decay_rate: float = 0.0,
        min_value: float | None = None,
        max_value: float | None = None,
        thresholds: dict[str, float] | None = None,
        initial: float = 0.0,
    ):
        self.decay_rate = decay_rate
        self.min_value = min_value
        self.max_value = max_value
        self.thresholds: dict[str, float] = thresholds or {}
        self.initial = initial

        # Set by __set_name__
        self.attr_name: str = ""
        self._data_key: str = ""       # obj.__dict__ key for (value, timestamp)
        self._change_key: str = ""     # obj.__dict__ key for on-change callbacks
        self._threshold_key: str = ""  # obj.__dict__ key for threshold callbacks

    # ------------------------------------------------------------------
    # Descriptor protocol
    # ------------------------------------------------------------------

    def __set_name__(self, owner: type, name: str) -> None:
        self.attr_name = name
        self._data_key = f"__bs_{name}__"
        self._change_key = f"__bs_{name}_on_change__"
        self._threshold_key = f"__bs_{name}_on_threshold__"

    def __get__(self, obj: Any, objtype: type | None = None) -> float | "BehavioralState":
        if obj is None:
            return self  # class-level access returns the descriptor

        data: tuple[float, float] | None = obj.__dict__.get(self._data_key)
        if data is None:
            return self.initial

        value, timestamp = data

        if self.decay_rate == 0.0:
            return value

        model = getattr(obj, "model", None)
        if model is None:
            return value

        elapsed = model.time - timestamp
        return self._clamp(value + self.decay_rate * elapsed)

    def __set__(self, obj: Any, value: float | int) -> None:
        value = self._clamp(float(value))
        old_value = self.__get__(obj, type(obj))

        # Store (raw_value, current_model_time) for lazy-decay computation
        model = getattr(obj, "model", None)
        timestamp = model.time if model is not None else 0.0
        obj.__dict__[self._data_key] = (value, timestamp)

        # Fire threshold callbacks
        self._check_thresholds(obj, old_value, value)

        # Fire on-change callbacks
        if value != old_value:
            sig = StateSignal(self.attr_name, obj, old_value, value)
            for cb in obj.__dict__.get(self._change_key, []):
                try:
                    cb(sig)
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    print(f"[BehavioralState] on_change callback raised: {exc}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clamp(self, value: float) -> float:
        if self.min_value is not None:
            value = max(self.min_value, value)
        if self.max_value is not None:
            value = min(self.max_value, value)
        return value

    def _check_thresholds(self, obj: Any, old: float, new: float) -> None:
        """Fire threshold callbacks for any thresholds crossed between old → new."""
        threshold_callbacks: dict[str, list] = obj.__dict__.get(self._threshold_key, {})
        if not self.thresholds or not threshold_callbacks:
            return

        for name, level in self.thresholds.items():
            crossed_up = old < level <= new
            crossed_down = old >= level > new
            if not (crossed_up or crossed_down):
                continue

            direction = "up" if crossed_up else "down"
            sig = StateSignal(self.attr_name, obj, old, new, threshold=name, direction=direction)

            for cb in threshold_callbacks.get(name, []):
                try:
                    cb(sig)
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    print(f"[BehavioralState] threshold callback raised: {exc}")

    # ------------------------------------------------------------------
    # Materialise lazy decay (call from step to trigger threshold signals)
    # ------------------------------------------------------------------

    def sync(self, obj: Any) -> float:
        """Materialise the current lazy-decay value and fire any threshold callbacks.

        Call this from an agent's step if you want threshold signals to fire
        during decay (not just on explicit ``__set__``).

        Returns:
            The current (post-decay) value.
        """
        current = self.__get__(obj, type(obj))
        data = obj.__dict__.get(self._data_key)
        if data is None:
            return current

        _, stored_timestamp = data
        model = getattr(obj, "model", None)
        if model is None or model.time == stored_timestamp:
            return current

        old_raw, _ = data
        # what was the value at the last materialised timestamp?
        old_materialised = old_raw  # raw value (already clamped at set time)
        self._check_thresholds(obj, old_materialised, current)

        # Write back to reset the decay clock
        obj.__dict__[self._data_key] = (current, model.time)

        # Fire on-change callbacks
        if current != old_materialised:
            sig = StateSignal(self.attr_name, obj, old_materialised, current)
            for cb in obj.__dict__.get(self._change_key, []):
                try:
                    cb(sig)
                except Exception as exc:
                    print(f"[BehavioralState] on_change callback raised: {exc}")

        return current

    # ------------------------------------------------------------------
    # Public callback registration helpers (static methods)
    # ------------------------------------------------------------------

    @staticmethod
    def on_change(obj: Any, attr: str, callback: Callable) -> None:
        """Register a callback fired whenever *attr* changes via explicit set or sync.

        Args:
            obj: The agent instance.
            attr: The BehavioralState attribute name.
            callback: Called with a ``StateSignal``.

        Example::

            BehavioralState.on_change(self, "hunger",
                                      lambda sig: print(f"hunger: {sig.new_value:.1f}"))
        """
        descriptor: BehavioralState = getattr(type(obj), attr)
        key = descriptor._change_key
        if key not in obj.__dict__:
            obj.__dict__[key] = []
        obj.__dict__[key].append(callback)

    @staticmethod
    def on_threshold(obj: Any, attr: str, threshold_name: str, callback: Callable) -> None:
        """Register a callback fired when a named threshold is crossed.

        Args:
            obj: The agent instance.
            attr: The BehavioralState attribute name.
            threshold_name: One of the threshold names declared in the descriptor.
            callback: Called with a ``StateSignal`` (has ``.threshold`` and ``.direction``).

        Example::

            BehavioralState.on_threshold(self, "hunger", "starving",
                lambda sig: print(f"STARVING! dir={sig.direction}"))
        """
        descriptor: BehavioralState = getattr(type(obj), attr)
        key = descriptor._threshold_key
        if key not in obj.__dict__:
            obj.__dict__[key] = {}
        obj.__dict__[key].setdefault(threshold_name, []).append(callback)

    @staticmethod
    def get_all(obj: Any) -> dict[str, float]:
        """Return a snapshot of all BehavioralState values on *obj* (post-decay)."""
        result = {}
        for name in dir(type(obj)):
            descriptor = getattr(type(obj), name, None)
            if isinstance(descriptor, BehavioralState):
                result[name] = getattr(obj, name)
        return result

    @staticmethod
    def sync_all(obj: Any) -> dict[str, float]:
        """Materialise all BehavioralState values, firing threshold callbacks as needed.

        Useful to call once per step from NeedsAgent or any agent that wants
        decay-triggered threshold events.

        Returns:
            Dict of ``{attr_name: current_value}``
        """
        result = {}
        for name in dir(type(obj)):
            descriptor = getattr(type(obj), name, None)
            if isinstance(descriptor, BehavioralState):
                result[name] = descriptor.sync(obj)
        return result

    def __repr__(self) -> str:
        return (
            f"BehavioralState(decay_rate={self.decay_rate}, "
            f"range=[{self.min_value}, {self.max_value}], "
            f"thresholds={list(self.thresholds)})"
        )