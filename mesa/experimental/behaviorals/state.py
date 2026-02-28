"""Behavioral State - Observable with automatic decay and threshold detection."""

from __future__ import annotations

import math
from collections.abc import Callable
from weakref import WeakKeyDictionary

from mesa.experimental.mesa_signals.core import BaseObservable, HasObservables
from mesa.experimental.mesa_signals.signals_util import SignalType


class BehavioralSignals(SignalType):
    CHANGED = "changed"
    THRESHOLD_CROSSED = "threshold_crossed"
    DECAYED = "decayed"


class BehavioralState(BaseObservable):
    """An Observable that decays/grows over time and emits signals at thresholds.

    Uses lazy evaluation: value only updates when accessed.
    decay_rate can be a float (linear: value += rate * elapsed)
    or callable(current_value, elapsed) -> delta for custom curves.

    Example:
        >>> class Animal(Agent, HasObservables):
        ...     energy = BehavioralState(
        ...         initial_value=100,
        ...         decay_rate=-1.5,
        ...         thresholds={30: "hungry", 0: "starved"}
        ...     )
        ...     def __init__(self, model):
        ...         super().__init__(model)
        ...         self.energy = 100
        ...         self.observe("energy", BehavioralSignals.THRESHOLD_CROSSED, self.on_threshold)
    """

    signal_types: type[SignalType] = BehavioralSignals

    def __init__(
        self,
        initial_value: float = 0.0,
        decay_rate: float | Callable[[float, float], float] = 0.0,
        thresholds: dict[float, str] | None = None,
        track_history: bool = False,
    ):
        super().__init__(fallback_value=initial_value)
        self.decay_rate = decay_rate
        self.thresholds = thresholds or {}
        self.track_history = track_history
        self._instance_values = (
            WeakKeyDictionary()
        )  # avoids memory leaks on agent removal

    def _get_state(self, instance: HasObservables) -> _StateInstance:
        if instance not in self._instance_values:
            current_time = self._get_current_time(instance)
            self._instance_values[instance] = _StateInstance(
                value=self.fallback_value,
                last_update_time=current_time,
                history=[] if self.track_history else None,
            )
        return self._instance_values[instance]

    def __get__(
        self, instance: HasObservables | None, owner
    ) -> float | BehavioralState:
        if instance is None:
            return self

        state = self._get_state(instance)
        current_time = self._get_current_time(instance)
        elapsed = current_time - state.last_update_time

        if elapsed > 0:
            old_value = state.value
            new_value = self._compute_new_value(state.value, elapsed)
            crossed = self._check_thresholds(old_value, new_value)

            state.value = new_value
            state.last_update_time = current_time

            if self.track_history:
                state.history.append((current_time, new_value))

            for threshold, label, direction in crossed:
                instance.notify(
                    self.public_name,
                    BehavioralSignals.THRESHOLD_CROSSED,
                    old=old_value,
                    new=new_value,
                    threshold=threshold,
                    label=label,
                    direction=direction,
                    time=current_time,
                )

            if abs(new_value - old_value) > 1e-10 and self.decay_rate != 0:
                instance.notify(
                    self.public_name,
                    BehavioralSignals.DECAYED,
                    old=old_value,
                    new=new_value,
                    elapsed=elapsed,
                    time=current_time,
                )

        return state.value

    def __set__(self, instance: HasObservables, value: float) -> None:
        state = self._get_state(instance)
        current_time = self._get_current_time(instance)
        old_value = state.value
        crossed = self._check_thresholds(old_value, value)

        state.value = value
        state.last_update_time = current_time

        if self.track_history:
            state.history.append((current_time, value))

        for threshold, label, direction in crossed:
            instance.notify(
                self.public_name,
                BehavioralSignals.THRESHOLD_CROSSED,
                old=old_value,
                new=value,
                threshold=threshold,
                label=label,
                direction=direction,
                time=current_time,
            )

        instance.notify(
            self.public_name,
            BehavioralSignals.CHANGED,
            old=old_value,
            new=value,
            time=current_time,
        )

    def _compute_new_value(self, current_value: float, elapsed: float) -> float:
        if self.decay_rate == 0:
            return current_value
        if callable(self.decay_rate):
            return current_value + self.decay_rate(current_value, elapsed)
        return current_value + (self.decay_rate * elapsed)

    def _check_thresholds(
        self, old_val: float, new_val: float
    ) -> list[tuple[float, str, str]]:
        crossed = []
        for threshold, label in self.thresholds.items():
            if old_val > threshold >= new_val:
                crossed.append((threshold, label, "down"))
            elif old_val < threshold <= new_val:
                crossed.append((threshold, label, "up"))
        return crossed

    def _get_current_time(self, instance: HasObservables) -> float:
        try:
            return float(instance.model.time)
        except (AttributeError, TypeError):
            return 0.0

    def get_history(self, instance: HasObservables) -> list[tuple[float, float]] | None:
        if not self.track_history:
            return None
        state = self._get_state(instance)
        return list(state.history) if state.history else []


class _StateInstance:
    __slots__ = ["history", "last_update_time", "value"]

    def __init__(self, value: float, last_update_time: float, history: list | None):
        self.value = value
        self.last_update_time = last_update_time
        self.history = history


# Decay function helpers


def exponential_decay(rate: float) -> Callable[[float, float], float]:
    """N(t) = N0 * e^(-rt)"""
    return lambda v, elapsed: v * (math.exp(-rate * elapsed) - 1)


def threshold_decay(
    base_rate: float,
    threshold: float,
    multiplier_below: float = 1.0,
    multiplier_above: float = 1.0,
) -> Callable[[float, float], float]:
    """Faster decay above or below a threshold."""
    return (
        lambda v, elapsed: base_rate
        * (multiplier_below if v < threshold else multiplier_above)
        * elapsed
    )


def logistic_decay(
    max_rate: float, midpoint: float, steepness: float = 1.0
) -> Callable[[float, float], float]:
    """S-curve decay using logistic function."""
    return (
        lambda v, elapsed: (max_rate / (1 + math.exp(-steepness * (v - midpoint))))
        * elapsed
    )
