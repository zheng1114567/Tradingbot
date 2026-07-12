"""Double-layer contradiction detector for Round 2 roundtable.

Layer 1 — Pattern-based (8 deterministic rules, zero LLM cost):
  Always runs. Checks agent report objects for known contradiction patterns.

Layer 2 — LLM semantic (optional, token-gated):
  Only runs when Layer 1 finds < 2 contradictions. Asks LLM to read all
  agent reports and identify subtle inconsistencies patterns cannot catch.

Design:
  - Pure function interface: detect(...) → list[ContradictionRecord]
  - All pattern checks are null-safe (None report → skip)
  - ContradictionRecord carries metadata for downstream use
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from ..llm.client import LLMClient
from .schemas import ContradictionRecord

logger = logging.getLogger(__name__)


class LLMContradictions(BaseModel):
    """Wrapper model for LLM semantic contradiction detection output."""
    contradictions: list[ContradictionRecord]

# Pattern definitions for traceability
_PATTERNS: list[dict[str, Any]] = [
    {
        "id": "ct_capital_event",
        "description": "Market 资金背离/不足 但 Event 判定利好",
        "agents": ["Market", "Event"],
        "severity": "high",
    },
    {
        "id": "ct_backtest_analysis",
        "description": "Backtest 样本不足 (<30) 但 Analysis 评分 >7",
        "agents": ["Backtest", "Analysis"],
        "severity": "medium",
    },
    {
        "id": "ct_overheat_event",
        "description": "Market 高潮 但 Event 判定利好 (过热叠加利好, 趋势见顶风险)",
        "agents": ["Market", "Event"],
        "severity": "high",
    },
    {
        "id": "ct_weak_event_strong_backtest",
        "description": "Event 链条弱(weak) 但 Backtest 回测胜率 >=60%",
        "agents": ["Event", "Backtest"],
        "severity": "medium",
    },
    {
        "id": "ct_crowding_capital",
        "description": "Analysis 发出拥挤警告 且 Market 资金背离",
        "agents": ["Analysis", "Market"],
        "severity": "medium",
    },
    {
        "id": "ct_freeze_event",
        "description": "Market 冰点 但 Event 利空 (冰点遇利空, 系统性风险)",
        "agents": ["Market", "Event"],
        "severity": "high",
    },
    {
        "id": "ct_backtest_zero_excess",
        "description": "Backtest 超额收益 <=0 但 Analysis 评分 >7",
        "agents": ["Backtest", "Analysis"],
        "severity": "medium",
    },
    {
        "id": "ct_memory_low_accuracy",
        "description": "Memory 显示某 Agent 历史准确率 <40% 且该 Agent 观点与主流相悖",
        "agents": ["Memory"],
        "severity": "medium",
    },
]


class ContradictionDetector:
    """Detect contradictions between agent reports.

    Usage:
        detector = ContradictionDetector()
        records = detector.detect(
            market_rpt=market_report_obj,
            event_rpt=event_report_obj,
            analysis_rpt=analysis_report_obj,
            backtest_rpt=backtest_report_obj,
            memory_recall=memory_recall_obj,
            llm=llm_client,  # optional
        )
    """

    def detect(
        self,
        market_rpt: Any | None = None,
        event_rpt: Any | None = None,
        analysis_rpt: Any | None = None,
        backtest_rpt: Any | None = None,
        memory_recall: Any | None = None,
        llm: LLMClient | None = None,
    ) -> list[ContradictionRecord]:
        """Run pattern-based detection always, LLM semantic if < 2 found."""
        records: list[ContradictionRecord] = []

        # Layer 1: pattern-based
        self._check_capital_event(market_rpt, event_rpt, records)
        self._check_backtest_analysis(backtest_rpt, analysis_rpt, records)
        self._check_overheat_event(market_rpt, event_rpt, records)
        self._check_weak_event_strong_backtest(event_rpt, backtest_rpt, records)
        self._check_crowding_capital(analysis_rpt, market_rpt, records)
        self._check_freeze_event(market_rpt, event_rpt, records)
        self._check_backtest_zero_excess(backtest_rpt, analysis_rpt, records)
        self._check_memory_low_accuracy(memory_recall, records)

        # Layer 2: LLM semantic (only if < 2 found by patterns)
        if llm is not None and len(records) < 2:
            llm_records = self._llm_semantic_detect(
                llm=llm,
                market_rpt=market_rpt,
                event_rpt=event_rpt,
                analysis_rpt=analysis_rpt,
                backtest_rpt=backtest_rpt,
                existing_count=len(records),
            )
            records.extend(llm_records)

        return records

    # ------------------------------------------------------------------
    # Layer 1 — Pattern checks
    # ------------------------------------------------------------------

    def _check_capital_event(
        self,
        market_rpt: Any | None,
        event_rpt: Any | None,
        records: list[ContradictionRecord],
    ) -> None:
        if not market_rpt or not event_rpt:
            return
        capital = getattr(market_rpt, "capital_confirmation", "")
        direction = getattr(event_rpt, "direction", "")
        if capital in ("资金背离", "资金不足") and direction == "利好":
            records.append(ContradictionRecord(
                id="ct_capital_event",
                description=(
                    f"Market 资金{capital} ↔ Event 方向{direction}。"
                    "资金面不支持事件驱动的做多逻辑。"
                ),
                agents_involved=["Market", "Event"],
                detection_method="pattern",
                severity="high",
                evidence_pair=("ev_market_capital", "ev_event_direction"),
            ))

    def _check_backtest_analysis(
        self,
        backtest_rpt: Any | None,
        analysis_rpt: Any | None,
        records: list[ContradictionRecord],
    ) -> None:
        if not backtest_rpt or not analysis_rpt:
            return
        sample_size = getattr(backtest_rpt, "sample_size", 0) or 0
        rankings = getattr(analysis_rpt, "stock_rankings", []) or []
        top_score = rankings[0].composite_score if rankings else 0
        if sample_size < 30 and top_score > 7:
            records.append(ContradictionRecord(
                id="ct_backtest_analysis",
                description=(
                    f"Backtest 样本不足({sample_size}) ↔ Analysis 高分({top_score:.1f})。"
                    "回测样本不足以支撑高评分结论。"
                ),
                agents_involved=["Backtest", "Analysis"],
                detection_method="pattern",
                severity="medium",
                evidence_pair=("ev_backtest_samples", "ev_analysis_score"),
            ))

    def _check_overheat_event(
        self,
        market_rpt: Any | None,
        event_rpt: Any | None,
        records: list[ContradictionRecord],
    ) -> None:
        if not market_rpt or not event_rpt:
            return
        market_state = getattr(market_rpt, "market_state", "")
        direction = getattr(event_rpt, "direction", "")
        if market_state == "高潮" and direction == "利好":
            records.append(ContradictionRecord(
                id="ct_overheat_event",
                description=(
                    f"Market 状态{market_state} ↔ Event 方向{direction}。"
                    "市场过热叠加利好, 趋势见顶风险高, 警惕利好出货。"
                ),
                agents_involved=["Market", "Event"],
                detection_method="pattern",
                severity="high",
                evidence_pair=("ev_market_state", "ev_event_direction"),
            ))

    def _check_weak_event_strong_backtest(
        self,
        event_rpt: Any | None,
        backtest_rpt: Any | None,
        records: list[ContradictionRecord],
    ) -> None:
        if not event_rpt or not backtest_rpt:
            return
        chain = getattr(event_rpt, "chain_quality", "")
        win_rate = getattr(backtest_rpt, "win_rate", 0) or 0
        if chain == "weak" and win_rate >= 0.6:
            records.append(ContradictionRecord(
                id="ct_weak_event_strong_backtest",
                description=(
                    f"Event 链条{chain} ↔ Backtest 胜率{win_rate:.0%}。"
                    "事件链条弱但回测胜率高, 可能回测依赖的并非当前事件驱动逻辑。"
                ),
                agents_involved=["Event", "Backtest"],
                detection_method="pattern",
                severity="medium",
                evidence_pair=("ev_event_chain", "ev_backtest_winrate"),
            ))

    def _check_crowding_capital(
        self,
        analysis_rpt: Any | None,
        market_rpt: Any | None,
        records: list[ContradictionRecord],
    ) -> None:
        if not analysis_rpt or not market_rpt:
            return
        warnings = getattr(analysis_rpt, "factor_warnings", []) or []
        capital = getattr(market_rpt, "capital_confirmation", "")
        has_crowding = any("拥挤" in (w or "") for w in warnings)
        if has_crowding and capital == "资金背离":
            records.append(ContradictionRecord(
                id="ct_crowding_capital",
                description=(
                    f"Analysis 拥挤警告 ↔ Market 资金{capital}。"
                    "因子拥挤叠加资金背离, 板块回调风险高。"
                ),
                agents_involved=["Analysis", "Market"],
                detection_method="pattern",
                severity="medium",
                evidence_pair=("ev_analysis_crowding", "ev_market_capital"),
            ))

    def _check_freeze_event(
        self,
        market_rpt: Any | None,
        event_rpt: Any | None,
        records: list[ContradictionRecord],
    ) -> None:
        if not market_rpt or not event_rpt:
            return
        market_state = getattr(market_rpt, "market_state", "")
        direction = getattr(event_rpt, "direction", "")
        if market_state == "冰点" and direction == "利空":
            records.append(ContradictionRecord(
                id="ct_freeze_event",
                description=(
                    f"Market 状态{market_state} ↔ Event 方向{direction}。"
                    "冰点遇利空, 系统性风险加剧, 应回避。"
                ),
                agents_involved=["Market", "Event"],
                detection_method="pattern",
                severity="high",
                evidence_pair=("ev_market_state", "ev_event_direction"),
            ))

    def _check_backtest_zero_excess(
        self,
        backtest_rpt: Any | None,
        analysis_rpt: Any | None,
        records: list[ContradictionRecord],
    ) -> None:
        if not backtest_rpt or not analysis_rpt:
            return
        excess = getattr(backtest_rpt, "avg_excess_return", 1) or 1
        rankings = getattr(analysis_rpt, "stock_rankings", []) or []
        top_score = rankings[0].composite_score if rankings else 0
        if excess <= 0 and top_score > 7:
            records.append(ContradictionRecord(
                id="ct_backtest_zero_excess",
                description=(
                    f"Backtest 超额{excess:+.2%} ↔ Analysis 高分({top_score:.1f})。"
                    "回测超额非正, 历史证据不支持分析的高评分。"
                ),
                agents_involved=["Backtest", "Analysis"],
                detection_method="pattern",
                severity="medium",
                evidence_pair=("ev_backtest_excess", "ev_analysis_score"),
            ))

    def _check_memory_low_accuracy(
        self,
        memory_recall: Any | None,
        records: list[ContradictionRecord],
    ) -> None:
        if not memory_recall:
            return
        # Handle both dict (from model_dump) and object (MemoryRecall instance)
        accuracy = (
            memory_recall.get("agent_accuracy", {})
            if isinstance(memory_recall, dict)
            else getattr(memory_recall, "agent_accuracy", {})
        ) or {}
        for agent, acc in accuracy.items():
            if isinstance(acc, (int, float)) and acc < 0.4:
                records.append(ContradictionRecord(
                    id="ct_memory_low_accuracy",
                    description=(
                        f"Memory 显示 {agent} 历史准确率仅 {acc:.0%}。"
                        f"该 Agent 当前观点可信度应折价。"
                    ),
                    agents_involved=["Memory", agent],
                    detection_method="pattern",
                    severity="medium",
                    evidence_pair=(f"ev_memory_{agent.lower()}_accuracy", ""),
                ))

    # ------------------------------------------------------------------
    # Layer 2 — LLM semantic detection
    # ------------------------------------------------------------------

    def _llm_semantic_detect(
        self,
        llm: LLMClient,
        market_rpt: Any | None,
        event_rpt: Any | None,
        analysis_rpt: Any | None,
        backtest_rpt: Any | None,
        existing_count: int,
    ) -> list[ContradictionRecord]:
        summaries = []
        if market_rpt:
            summaries.append(
                f"Market: 市场状态={getattr(market_rpt, 'market_state', 'N/A')}, "
                f"资金确认={getattr(market_rpt, 'capital_confirmation', 'N/A')}"
            )
        if event_rpt:
            summaries.append(
                f"Event: 方向={getattr(event_rpt, 'direction', 'N/A')}, "
                f"链条={getattr(event_rpt, 'chain_quality', 'N/A')}, "
                f"置信度={getattr(event_rpt, 'confidence', 'N/A')}"
            )
        if analysis_rpt:
            rankings = getattr(analysis_rpt, "stock_rankings", []) or []
            score = rankings[0].composite_score if rankings else "N/A"
            warnings = getattr(analysis_rpt, "factor_warnings", []) or []
            summaries.append(
                f"Analysis: Top评分={score}, "
                f"拥挤警告={warnings}"
            )
        if backtest_rpt:
            summaries.append(
                f"Backtest: 样本={getattr(backtest_rpt, 'sample_size', 'N/A')}, "
                f"胜率={getattr(backtest_rpt, 'win_rate', 'N/A')}, "
                f"超额={getattr(backtest_rpt, 'avg_excess_return', 'N/A')}"
            )

        prompt = (
            "你是矛盾检测专家。以下是一些 Agent 对同一标的的分析摘要。\n\n"
            + "\n".join(summaries)
            + "\n\n"
            + f"已有 {existing_count} 条确定性模式匹配到的矛盾, 请找出其它潜在的矛盾或不一致之处。\n"
            "注意:\n"
            "1. 忽略模式匹配已覆盖的矛盾\n"
            "2. 关注逻辑冲突而非表述差异\n"
            "3. 如果没有发现额外矛盾, response 输出空列表\n"
            "4. 每条矛盾需要描述、涉及哪些 Agent、严重程度(high/medium/low)"
        )

        try:
            result = llm.chat(
                messages=[
                    ("system", "你严格根据数据判断矛盾, 不编造不存在的冲突。"),
                    ("human", prompt),
                ],
                response_format=LLMContradictions,
            )
            if result and result.contradictions:
                for rec in result.contradictions:
                    rec.detection_method = "llm"
                    if not rec.id:
                        rec.id = f"ct_llm_{hash(rec.description) % 10000:04d}"
                logger.info(
                    "LLM contradiction detection found %d additional contradictions",
                    len(result.contradictions),
                )
                return result.contradictions
        except Exception as e:
            logger.warning("LLM contradiction detection failed: %s", e)

        return []
