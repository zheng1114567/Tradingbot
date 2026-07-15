"""Roundtable debate adapters."""

from .contradiction_detector import ContradictionDetector
from .debate_engine import DebateEngine
from .harness import (
    DataAgentBrief,
    RoundtableAgentContext,
    RoundtableContext,
    RoundtableHarness,
)
from .schemas import (
    AgentStance,
    ContradictionRecord,
    DebateTurn,
    EvidenceItem,
    ModeratorOutput,
)

__all__ = [
    "AgentStance",
    "AutoGenRoundtable",
    "ContradictionDetector",
    "ContradictionRecord",
    "DataAgentBrief",
    "DebateEngine",
    "DebateTurn",
    "EvidenceItem",
    "ModeratorOutput",
    "RoundtableAgentContext",
    "RoundtableContext",
    "RoundtableHarness",
    "RoundtableResult",
]


def __getattr__(name: str):
    """Lazy-import AutoGen types to break circular import chain.

    The cycle: roundtable.__init__ → autogen_roundtable → tool_nodes.registry
    → backtest_tools → agents → tool_nodes.registry.
    By deferring the autogen import until first use, importing ``harness`` or
    ``schemas`` no longer triggers the cycle.
    """
    if name in ("AutoGenRoundtable", "RoundtableResult"):
        from .autogen_roundtable import AutoGenRoundtable, RoundtableResult

        return locals()[name]
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
