"""Roundtable debate adapters."""

from .autogen_roundtable import AutoGenRoundtable, RoundtableResult
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
