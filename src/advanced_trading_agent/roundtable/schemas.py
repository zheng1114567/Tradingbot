"""Debate-specific Pydantic models for Round 2 roundtable.

All models are designed to be used with LLMClient.chat(response_format=...)
for structured output during debate turns and moderator rulings.
They serialize to JSON via .model_dump(mode="json") for LangGraph state storage.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """一条可追溯的证据，标明来源 Agent 和 DataAgent 字段路径。"""

    id: str = Field(description="证据唯一 ID, 如 ev_market_001")
    agent: str = Field(description="来源 Agent 名称, 如 Market/Event/Analysis/Backtest")
    field_path: str = Field(description="DataAgent 字段路径, 如 tier1_data.capital.confirmation")
    value: str = Field(description="证据值摘要")
    report_excerpt: str = Field(description="Agent 报告中引用该证据的片段", default="")
    contradiction_tag: str | None = Field(
        description="关联的矛盾 ID (ct_xxx), 属于某个矛盾的关键证据", default=None
    )


class AgentStance(BaseModel):
    """Agent 对当前矛盾/推荐的立场。"""

    pressure: Literal["upgrade", "neutral", "downgrade"] = Field(
        description="对最终推荐的压力方向"
    )
    confidence: float = Field(
        description="置信度 0-1", ge=0.0, le=1.0
    )
    evidence_ids: list[str] = Field(
        description="引用证据的 ID 列表 [ev_xxxxx]", default_factory=list
    )
    reasoning: str = Field(description="立场理由，引用具体数据字段")
    changed_from_previous: bool = Field(
        description="是否与上一轮立场不同", default=False
    )


class DebateTurn(BaseModel):
    """一个 Agent 在一轮辩论中的完整发言。"""

    agent_name: str = Field(description="发言 Agent 名称")
    stance: AgentStance = Field(description="本轮立场")
    rebuts: list[str] = Field(
        description="反驳了哪些 Agent 的观点", default_factory=list
    )
    new_evidence: list[EvidenceItem] = Field(
        description="本轮新增的证据引用", default_factory=list
    )


class ModeratorOutput(BaseModel):
    """Moderator 在每轮辩论结束后的综合裁定。"""

    round_number: int = Field(description="当前辩论轮次")
    final_pressure: Literal["upgrade", "neutral", "downgrade"] = Field(
        description="综合后的最终压力方向"
    )
    unresolved_contradiction_ids: list[str] = Field(
        description="仍未解决的矛盾 ID 列表", default_factory=list
    )
    consensus_items: list[str] = Field(
        description="各方达成一致的观点", default_factory=list
    )
    dissent_items: list[str] = Field(
        description="仍然存在分歧的观点", default_factory=list
    )
    converged: bool = Field(description="是否已达成收敛")
    reasoning: str = Field(description="裁定推理过程")
    risk_focus: list[str] = Field(
        description="风控应关注的领域", default_factory=list
    )


class ContradictionRecord(BaseModel):
    """一条检测到的矛盾元数据。"""

    id: str = Field(description="矛盾唯一 ID, 如 ct_001")
    description: str = Field(description="人类可读的矛盾描述")
    agents_involved: list[str] = Field(
        description="涉及哪些 Agent, 如 [Market, Event]"
    )
    detection_method: Literal["pattern", "llm"] = Field(
        description="检测方式: 确定性模式匹配 / LLM 语义检测"
    )
    severity: Literal["high", "medium", "low"] = Field(
        description="严重程度: high/medium/low"
    )
    evidence_pair: tuple[str, str] = Field(
        description="矛盾双方证据 ID, 如 (ev_market_001, ev_event_003)"
    )
