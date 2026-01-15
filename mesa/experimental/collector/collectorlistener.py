"""High-performance Listener replacement for DataCollector.

This module provides a faster alternative to Mesa's standard DataCollector by
leveraging:
1. Columnar Storage (Dictionary of Lists) for O(1) appends.
2. Pre-compiled C-level attribute accessors (operator.attrgetter).
3. Zero-Copy integration with Polars (if installed).
"""

from __future__ import annotations

import contextlib
import operator
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from mesa.model import Model

# Attempt to import Polars for performance, but stay optional
with contextlib.suppress(ImportError):
    import polars as pl


class BaseListener:
    """Base class for listeners that subscribe to model signals.

    This class handles the basic setup of observing standard model signals
    ('step', 'end', 'reset'). Subclasses should override the specific
    event handlers (`on_step`, `on_run_end`, `on_reset`).
    """

    def __init__(self, model: Model):
        """Initialize the listener and subscribe to model signals.

        Args:
            model: The Mesa model instance to observe.
        """
        self.model = model
        # Subscribe to standard signals
        self.model.observe("step", "step", self.on_step)
        self.model.observe("end", "end", self.on_run_end)
        self.model.observe("reset", "reset", self.on_reset)

    def on_step(self, signal):
        """Handler for the 'step' signal."""

    def on_run_end(self, signal):
        """Handler for the 'end' signal (simulation complete)."""

    def on_reset(self, signal):
        """Handler for the 'reset' signal."""


class CollectorListener(BaseListener):
    """The direct replacement for DataCollector with C-speed optimizations.

    Architecture:
    - **Columnar Storage:** Uses `Dict[str, List]` instead of `List[Dict]`. This ensures
      contiguity in memory and allows for O(1) appends.
    - **Pre-compiled Accessors:** When reporters are strings (e.g., "wealth"),
      this class compiles them into `operator.attrgetter` callables, which run at
      C-speed, bypassing Python's `getattr` loop overhead.
    - **Sparse Handling:** Automatically handles agents dying or being removed;
      data is only collected for agents present in `model.agents` at the current step.

    Attributes:
        model_vars (dict): Dictionary storing model-level data.
        agent_reporters (dict): Configuration for agent-level data collection.
        tables (dict): Storage for custom tabular data.
    """

    def __init__(
        self,
        model: Model,
        model_reporters: dict[str, Callable | str] | None = None,
        agent_reporters: dict[str, Callable | str] | None = None,
        tables: dict[str, list[str]] | None = None,
    ):
        """Initialize the CollectorListener.

        Args:
            model: The Mesa model to observe.
            model_reporters: Dictionary mapping column names to attributes/functions
                             for model-level data.
            agent_reporters: Dictionary mapping column names to attributes/functions
                             for agent-level data.
            tables: Dictionary defining schemas for custom tables.
                    Format: {'table_name': ['col1', 'col2']}
        """
        super().__init__(model)

        # --- 1. SETUP STORAGE ---
        raw_model_reporters = model_reporters or {}
        self.agent_reporters = agent_reporters or {}
        self.tables_config = tables or {}

        # Columnar Storage (Dict of Lists)
        self.model_vars: dict[str, list[Any]] = {k: [] for k in raw_model_reporters}
        self._agent_data: dict[str, list[Any]] = {k: [] for k in self.agent_reporters}
        self._agent_data["Step"] = []
        self._agent_data["AgentID"] = []

        self.tables: dict[str, dict[str, list[Any]]] = {
            name: {col: [] for col in cols} for name, cols in self.tables_config.items()
        }

        # --- 2. COMPILE MODEL COLLECTOR ---
        # Handle string reporters by converting to attrgetter
        self._model_reporters = {}
        for name, reporter in raw_model_reporters.items():
            if isinstance(reporter, str):
                self._model_reporters[name] = operator.attrgetter(reporter)
            else:
                self._model_reporters[name] = reporter

        # --- 3. COMPILE AGENT COLLECTOR ---
        # Optimization: Bundle 'unique_id' into the getter to fetch everything in one pass
        if not self.agent_reporters:
            self._collect_agents = self._collect_agents_noop
        elif all(isinstance(v, str) for v in self.agent_reporters.values()):
            # Create a single getter for [unique_id, attr1, attr2, ...]
            # This allows fetching ALL data for an agent in ONE C-call.
            # FIX: RUF005 - Use iterable unpacking instead of concatenation
            all_attrs = ["unique_id", *self.agent_reporters.values()]
            self._agent_getter = operator.attrgetter(*all_attrs)

            # Cache the destination lists: [AgentID List, Attr1 List, Attr2 List...]
            # Order matches the attrgetter keys
            self._agent_targets = [self._agent_data["AgentID"]] + [
                self._agent_data[k] for k in self.agent_reporters
            ]

            self._collect_agents = self._collect_agents_fast_zip
        else:
            self._collect_agents = self._collect_agents_slow

    # --- MODEL COLLECTION STRATEGIES ---

    def _collect_model(self, model):
        """Collects data for all model reporters."""
        for name, reporter in self._model_reporters.items():
            self.model_vars[name].append(reporter(model))

    # --- AGENT COLLECTION STRATEGIES ---

    def _collect_agents_noop(self, model, step):
        """No-op strategy when no agent reporters are defined."""

    def _collect_agents_fast_zip(self, model, step):
        """The 'Zero-Overhead' Strategy for agent collection.

        Architecture:
        1. Accesses the raw iterator of agents.
        2. Uses `map(getter, agents)` to fetch all attributes in C-speed.
        3. Uses `zip(*...)` to transpose the rows into columns (also C-speed).
        4. Uses `list.extend` to bulk append data.

        Args:
            model: The model instance.
            step: The current step number.
        """
        # Rely on model.agents. Standard AgentSet iteration yields agents (Keys).
        # We do NOT try to guess .values() anymore, as that breaks AgentSet structure.
        agents = model.agents

        # If it's an AgentSet, access the internal dict to skip the __iter__
        # method overhead, but iterate it directly (Keys).
        if hasattr(agents, "_agents"):
            agents = agents._agents
        elif isinstance(agents, dict):
            # Only if it's a raw dict (user custom), assume ID->Agent and take values.
            agents = agents.values()

        # zip(*map(...)) is the standard idiom for fast transposition
        try:
            cols = zip(*map(self._agent_getter, agents))

            # Bulk Extend
            for target, data in zip(self._agent_targets, cols):
                target.extend(data)

            # Handle Step Column (Efficient Fill)
            count = len(self._agent_targets[0]) - len(self._agent_data["Step"])
            self._agent_data["Step"].extend([step] * count)

        except ValueError:
            # Handles empty agent set case (zip(*[]) raises ValueError or returns empty)
            pass

    def _collect_agents_slow(self, model, step):
        """Fallback strategy for mixed or complex callable reporters.

        Iterates in Python, which is slower but supports full flexibility.
        """
        agents = model.agents

        ids = []
        rows = []

        reporters = [self.agent_reporters[k] for k in self.agent_reporters]

        for agent in agents:
            ids.append(agent.unique_id)
            row = []
            for r in reporters:
                if isinstance(r, str):
                    row.append(getattr(agent, r))
                else:
                    row.append(r(agent))
            rows.append(row)

        count = len(ids)
        if count > 0:
            self._agent_data["Step"].extend([step] * count)
            self._agent_data["AgentID"].extend(ids)

            cols = zip(*rows)
            keys = list(self.agent_reporters.keys())
            for key, col_data in zip(keys, cols):
                self._agent_data[key].extend(col_data)

    def on_step(self, signal):
        """Triggered automatically by the model's 'step' signal."""
        model = signal.new
        self._collect_model(model)
        self._collect_agents(model, model.time)

    def add_table_row(self, table_name: str, row: dict[str, Any]):
        """Add a row to a custom table.

        Args:
            table_name: The name of the table to append to.
            row: A dictionary matching the table's schema.
        """
        for col, val in row.items():
            self.tables[table_name][col].append(val)

    def get_model_vars_dataframe(self):
        """Returns model variables as a DataFrame.

        Returns:
            pd.DataFrame: A pandas DataFrame of model variables.
        """
        if "pl" in globals() and pl is not None:
            return pl.DataFrame(self.model_vars).to_pandas()
        return pd.DataFrame(self.model_vars)

    def get_agent_vars_dataframe(self):
        """Returns agent variables as a DataFrame.

        Optimization:
            If Polars is installed, this method uses `pl.DataFrame` to ingest
            the dictionary of lists. This is often 'Zero-Copy' or significantly
            faster than Pandas' list-of-dicts ingestion.

        Returns:
            pd.DataFrame: A pandas DataFrame of agent variables.
        """
        if "pl" in globals() and pl is not None:
            return pl.DataFrame(self._agent_data).to_pandas()
        return pd.DataFrame(self._agent_data)
