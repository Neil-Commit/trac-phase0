"""Command output seam.

Phase 0 only SIMULATES commands. ``CommandSink`` is where a future ``UartCommandSink``
(Raspberry Pi -> ESP32) would plug in. No hardware code exists in this project.
"""

from __future__ import annotations

from typing import Optional, Protocol

from .navigation import NavDecision, NavState


class CommandSink(Protocol):
    def send(self, decision: NavDecision) -> None:
        """Deliver the decision's command (simulation: just record/log it)."""


class SimulatedCommandSink:
    """Records the latest decision and optionally logs state changes to the console."""

    def __init__(self, log: bool = False):
        self._log = log
        self.last: Optional[NavDecision] = None
        self._last_state: Optional[NavState] = None

    def send(self, decision: NavDecision) -> None:
        self.last = decision
        if self._log and decision.state != self._last_state:
            c = decision.command
            print(f"[SIM] {decision.state.value}: linear={c.linear_mps:.2f} m/s "
                  f"angular={c.angular_rps:+.2f} rad/s | {decision.reason}")
        self._last_state = decision.state
