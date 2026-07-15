"""JSON-first sector ETF watchlist contracts and deterministic decision rules."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


WatchlistStatus = Literal["active", "monitor", "excluded"]
Confidence = Literal["high", "medium", "low"]
ExcludedReason = Literal[
    "no_tradable_etf",
    "low_etf_liquidity",
    "etf_suspended",
    "mapping_uncertain",
]


class ETFWatchlistLimits(BaseModel):
    """Portfolio and report limits for the ETF observation pool."""

    max_roundtable_sectors: int = 8
    max_final_decisions: int = 3
    max_final_etfs_per_sector: int = 3
    max_active_sectors: int = 3
    max_total_active_weight: float = 0.60
    max_single_sector_weight: float = 0.15
    default_active_weight: float = 0.10
    high_confidence_active_weight: float = 0.15
    low_confidence_active_weight: float = 0.05
    monitor_weight: float = 0.0
    excluded_weight: float = 0.0


class WatchlistETFCandidate(BaseModel):
    """ETF candidate shown to and returned by the sector ETF roundtable."""

    code: str
    name: str
    match_score: float = 0.0
    liquidity_score: float = 0.0
    tracking_purity_score: float = 0.0
    total_score: float = 0.0
    reason: str = ""
    tradable: bool = True
    blocked_reasons: list[str] = Field(default_factory=list)
    pre_rank: int | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class SectorCandidatePayload(BaseModel):
    """Sector candidate passed into the batch roundtable."""

    sector_name: str
    pre_score: float
    momentum_score: float = 0.0
    breadth_score: float = 0.0
    event_score: float = 0.0
    evidence: dict[str, list[str]] = Field(default_factory=dict)
    support_evidence: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    raw_etf_candidates: list[WatchlistETFCandidate] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class ExcludedSectorCandidate(BaseModel):
    """Sector removed before roundtable because no executable ETF exists."""

    sector: str
    excluded_stage: Literal["pre_roundtable"] = "pre_roundtable"
    excluded_reason: ExcludedReason
    brief_evidence: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class SectorWatchlistDecision(BaseModel):
    """Final JSON contract for one sector in the ETF watchlist."""

    sector: str
    status: WatchlistStatus
    primary_etf: WatchlistETFCandidate
    backup_etfs: list[WatchlistETFCandidate] = Field(default_factory=list, max_length=2)
    watchlist_weight_hint: float = 0.0
    support_reasons: list[str] = Field(default_factory=list)
    objections: list[str] = Field(default_factory=list)
    risk_details: list[str] = Field(default_factory=list)
    why_primary_etf: list[str] = Field(default_factory=list)
    why_not_backups: list[str] = Field(default_factory=list)
    invalid_conditions: list[str] = Field(default_factory=list)
    confidence: Confidence = "low"
    review_horizon_days: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    execution_requires_approval: bool = True
    execution_allowed: bool = False
    roundtable_score: float = 0.0


class DailyETFWatchlistReport(BaseModel):
    """Top-level JSON-first daily ETF observation pool report."""

    trade_date: str
    run_id: str
    scope: Literal["a_share_sector_etf_watchlist"] = "a_share_sector_etf_watchlist"
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    limits: ETFWatchlistLimits = Field(default_factory=ETFWatchlistLimits)
    decisions: list[SectorWatchlistDecision] = Field(default_factory=list)
    excluded_sector_candidates: list[ExcludedSectorCandidate] = Field(default_factory=list)
    data_quality: dict[str, Any] = Field(default_factory=dict)
    roundtable_summary: dict[str, Any] = Field(default_factory=dict)
    approval: dict[str, Any] = Field(default_factory=lambda: {
        "status": "pending",
        "execution_allowed": False,
    })


class RoundtableAgentOutput(BaseModel):
    """One fast roundtable participant's structured opinion."""

    agent: Literal["Market", "Event", "Analysis", "Risk"]
    sector: str
    stance: Literal["support", "caution", "block"]
    summary: str
    evidence: list[str] = Field(default_factory=list)
    objections: list[str] = Field(default_factory=list)


class RoundtableDialogueTurn(BaseModel):
    """One auditable dialogue turn in the fast JSON roundtable."""

    round: int
    sector: str
    speaker: Literal["Moderator", "Market", "Event", "Analysis", "Risk"]
    message: str
    references: list[str] = Field(default_factory=list)


def build_watchlist_report(
    *,
    trade_date: str,
    candidates: list[SectorCandidatePayload],
    excluded: list[ExcludedSectorCandidate],
    limits: ETFWatchlistLimits | None = None,
    provider: str = "deterministic_batch_roundtable",
) -> DailyETFWatchlistReport:
    """Build deterministic JSON decisions from sector and ETF candidates.

    This is the fallback roundtable adapter: it preserves the contract that the
    System Agent must output JSON with a primary ETF, support reasons, objections,
    and weight hints, without requiring live LLM/AutoGen calls in tests.
    """
    limits = limits or ETFWatchlistLimits()
    roundtable_outputs = [
        output
        for candidate in candidates[: limits.max_roundtable_sectors]
        for output in _fast_roundtable_outputs(candidate)
    ]
    dialogue_records = _fast_roundtable_dialogue(candidates[: limits.max_roundtable_sectors], roundtable_outputs)
    decisions = [_decision_from_candidate(c, limits) for c in candidates[: limits.max_roundtable_sectors]]
    decisions.sort(key=lambda item: item.roundtable_score, reverse=True)
    decisions = _enforce_active_limits(decisions[: limits.max_final_decisions], limits)
    return DailyETFWatchlistReport(
        trade_date=trade_date,
        run_id=f"etf_watchlist_{trade_date}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        limits=limits,
        decisions=decisions,
        excluded_sector_candidates=excluded,
        data_quality={
            "candidate_count": len(candidates),
            "excluded_count": len(excluded),
            "final_decision_count": len(decisions),
            "all_final_decisions_have_primary_etf": all(bool(d.primary_etf.code) for d in decisions),
        },
        roundtable_summary={
            "provider": provider,
            "mode": "fast_json_roundtable",
            "backtest_used": False,
            "agent_outputs": [output.model_dump(mode="json") for output in roundtable_outputs],
            "dialogue_records": [turn.model_dump(mode="json") for turn in dialogue_records],
            "round_history": _round_history_from_dialogue(dialogue_records),
            "input_sector_count": len(candidates),
            "roundtable_candidate_count": min(len(candidates), limits.max_roundtable_sectors),
            "decision_count": len(decisions),
            "max_final_decisions": limits.max_final_decisions,
            "max_active_sectors": limits.max_active_sectors,
            "note": "Cache-first fast roundtable; no backtest and no long-running LLM calls on the default path.",
        },
    )


def render_watchlist_markdown(report: DailyETFWatchlistReport) -> str:
    """Render a readable Markdown report from the JSON contract."""
    lines = [
        "# A股板块 ETF 观察池",
        "",
        f"**交易日期**: {report.trade_date}",
        f"**运行ID**: {report.run_id}",
        f"**生成时间**: {report.generated_at}",
        "",
        "## 观察池结论",
        "",
    ]
    if not report.decisions:
        lines.append("未发现具备可交易 ETF 的板块候选。")
    for idx, decision in enumerate(report.decisions, start=1):
        primary = decision.primary_etf
        backups = "；".join(f"{e.code} {e.name}" for e in decision.backup_etfs) or "无"
        lines.extend([
            f"### {idx}. {decision.sector} - {decision.status}",
            "",
            f"- **首选 ETF**: {primary.code} {primary.name}",
            f"- **备选 ETF**: {backups}",
            f"- **观察池权重**: {decision.watchlist_weight_hint:.0%}",
            f"- **置信度**: {decision.confidence}",
            "",
            "**支持理由**:",
            *[f"- {reason}" for reason in decision.support_reasons],
            "",
            "**圆桌输出**:",
            *[
                f"- {item.get('agent')}: {item.get('stance')} - {item.get('summary')}"
                for item in report.roundtable_summary.get("agent_outputs", [])
                if item.get("sector") == decision.sector
            ],
            "",
            "**圆桌对话记录**:",
            *[
                f"- R{turn.get('round')} {turn.get('speaker')}: {turn.get('message')}"
                for turn in report.roundtable_summary.get("dialogue_records", [])
                if turn.get("sector") == decision.sector
            ],
            "",
            "**反对理由**:",
            *[f"- {reason}" for reason in (decision.objections or ["暂无重大反对意见"])],
            "",
            "**为什么是首选 ETF**:",
            *[f"- {reason}" for reason in decision.why_primary_etf],
            "",
            "**失效条件**:",
            *[f"- {item}" for item in decision.invalid_conditions],
            "",
        ])
    if report.excluded_sector_candidates:
        lines.extend(["## 剔除清单", ""])
        for item in report.excluded_sector_candidates:
            evidence = "；".join(item.brief_evidence[:4])
            lines.append(f"- **{item.sector}**: {item.excluded_reason}。{evidence}")
    lines.extend([
        "",
        "## 审批",
        "",
        f"- 状态: {report.approval.get('status', 'pending')}",
        f"- 可执行: {report.approval.get('execution_allowed', False)}",
    ])
    return "\n".join(lines)


def _decision_from_candidate(
    candidate: SectorCandidatePayload,
    limits: ETFWatchlistLimits,
) -> SectorWatchlistDecision:
    etfs = sorted(candidate.raw_etf_candidates, key=lambda item: item.total_score, reverse=True)
    final_etfs = etfs[: limits.max_final_etfs_per_sector]
    if not final_etfs:
        raise ValueError(f"Sector {candidate.sector_name} has no ETF candidates")
    primary = final_etfs[0]
    backups = final_etfs[1:3]
    objections = list(candidate.risk_flags)
    score = candidate.pre_score + min(primary.total_score / 4.0, 3.0)
    confidence: Confidence = "high" if score >= 9 and len(objections) <= 1 else "medium" if score >= 7 else "low"
    status: WatchlistStatus
    if score >= 8 and primary.liquidity_score >= 1.0:
        status = "active"
    elif score >= 5:
        status = "monitor"
    else:
        status = "excluded"
    if primary.blocked_reasons:
        status = "excluded"
        objections.extend(primary.blocked_reasons)
    weight = _weight_for_status(status, confidence, limits)
    why_not_backups = [
        f"{etf.code} {etf.name}: 备选，综合分 {etf.total_score:.1f} 低于首选 {primary.total_score:.1f}"
        for etf in backups
    ]
    return SectorWatchlistDecision(
        sector=candidate.sector_name,
        status=status,
        primary_etf=primary,
        backup_etfs=backups,
        watchlist_weight_hint=weight,
        support_reasons=_support_reasons(candidate, primary),
        objections=objections,
        risk_details=objections,
        why_primary_etf=[
            primary.reason,
            f"ETF 综合分 {primary.total_score:.1f}，流动性评分 {primary.liquidity_score:.1f}",
            f"主题跟踪纯度评分 {primary.tracking_purity_score:.1f}",
        ],
        why_not_backups=why_not_backups,
        invalid_conditions=[
            "板块动量和宽度明显回落",
            "首选 ETF 成交额跌破流动性阈值",
            "事件催化证伪或新闻半衰期结束",
            "ETF 停牌、涨跌停或出现异常溢价",
        ],
        confidence=confidence,
        roundtable_score=round(score, 2),
    )


def _fast_roundtable_outputs(candidate: SectorCandidatePayload) -> list[RoundtableAgentOutput]:
    """Produce fast deterministic agent opinions from collected/processed data."""
    best_etf = max(candidate.raw_etf_candidates, key=lambda item: item.total_score)
    market_stance: Literal["support", "caution", "block"] = (
        "support" if candidate.momentum_score >= 5 and candidate.breadth_score >= 1 else "caution"
    )
    event_stance: Literal["support", "caution", "block"] = (
        "support" if candidate.event_score >= 2 else "caution"
    )
    analysis_stance: Literal["support", "caution", "block"] = (
        "support" if best_etf.total_score >= 7 and best_etf.tradable else "block"
    )
    risk_stance: Literal["support", "caution", "block"] = (
        "block" if candidate.risk_flags and any("未匹配" in item or "流动性" in item for item in candidate.risk_flags)
        else "caution" if candidate.risk_flags
        else "support"
    )
    return [
        RoundtableAgentOutput(
            agent="Market",
            sector=candidate.sector_name,
            stance=market_stance,
            summary=f"板块动量 {candidate.momentum_score:.1f}，宽度 {candidate.breadth_score:.1f}",
            evidence=candidate.evidence.get("momentum", []) + candidate.evidence.get("breadth", []),
        ),
        RoundtableAgentOutput(
            agent="Event",
            sector=candidate.sector_name,
            stance=event_stance,
            summary=f"事件/新闻评分 {candidate.event_score:.1f}",
            evidence=candidate.evidence.get("events", []),
            objections=[] if event_stance == "support" else ["事件催化不足，状态不应仅靠动量上调"],
        ),
        RoundtableAgentOutput(
            agent="Analysis",
            sector=candidate.sector_name,
            stance=analysis_stance,
            summary=f"ETF 候选 {len(candidate.raw_etf_candidates)} 个，首选候选综合分 {best_etf.total_score:.1f}",
            evidence=[best_etf.reason],
            objections=best_etf.blocked_reasons,
        ),
        RoundtableAgentOutput(
            agent="Risk",
            sector=candidate.sector_name,
            stance=risk_stance,
            summary="默认不使用回测；只检查 ETF 可交易性、流动性和组合上限",
            evidence=[f"primary_candidate={best_etf.code}", f"liquidity_score={best_etf.liquidity_score:.1f}"],
            objections=candidate.risk_flags,
        ),
    ]


def _fast_roundtable_dialogue(
    candidates: list[SectorCandidatePayload],
    outputs: list[RoundtableAgentOutput],
) -> list[RoundtableDialogueTurn]:
    """Build deterministic dialogue records without slow LLM calls."""
    outputs_by_sector: dict[str, list[RoundtableAgentOutput]] = {}
    for output in outputs:
        outputs_by_sector.setdefault(output.sector, []).append(output)

    turns: list[RoundtableDialogueTurn] = []
    for round_idx, candidate in enumerate(candidates, start=1):
        sector = candidate.sector_name
        best_etf = max(candidate.raw_etf_candidates, key=lambda item: item.total_score)
        turns.append(
            RoundtableDialogueTurn(
                round=round_idx,
                sector=sector,
                speaker="Moderator",
                message=(
                    f"讨论 {sector}：先按板块强度、事件、ETF 可交易性和风险约束判断，"
                    f"必须落到首选 ETF {best_etf.code}。"
                ),
                references=["sector_score", "etf_candidates"],
            )
        )
        for output in outputs_by_sector.get(sector, []):
            refs = [f"{output.agent.lower()}_evidence"]
            if output.evidence:
                refs.extend(output.evidence[:2])
            turns.append(
                RoundtableDialogueTurn(
                    round=round_idx,
                    sector=sector,
                    speaker=output.agent,
                    message=f"{output.stance}: {output.summary}",
                    references=refs,
                )
            )
            for objection in output.objections[:2]:
                turns.append(
                    RoundtableDialogueTurn(
                        round=round_idx,
                        sector=sector,
                        speaker=output.agent,
                        message=f"反对意见：{objection}",
                        references=[f"{output.agent.lower()}_objection"],
                    )
                )
        turns.append(
            RoundtableDialogueTurn(
                round=round_idx,
                sector=sector,
                speaker="Moderator",
                message=(
                    f"小结：保留 {sector} 入最终排序，首选 ETF 为 {best_etf.code} {best_etf.name}；"
                    "执行仍需人工审批。"
                ),
                references=["primary_etf", "approval_required"],
            )
        )
    return turns


def _round_history_from_dialogue(dialogue_records: list[RoundtableDialogueTurn]) -> list[dict[str, Any]]:
    """Group dialogue turns into a compact round_history for audit consumers."""
    grouped: dict[int, list[RoundtableDialogueTurn]] = {}
    for turn in dialogue_records:
        grouped.setdefault(turn.round, []).append(turn)
    return [
        {
            "round": round_number,
            "sector": turns[0].sector if turns else "",
            "turn_count": len(turns),
            "turns": [turn.model_dump(mode="json") for turn in turns],
        }
        for round_number, turns in sorted(grouped.items())
    ]


def _support_reasons(candidate: SectorCandidatePayload, primary: WatchlistETFCandidate) -> list[str]:
    reasons = [
        f"板块预评分 {candidate.pre_score:.1f}",
        *candidate.support_evidence[:5],
        f"首选 ETF {primary.code} {primary.name} 匹配该板块且具备可交易性",
    ]
    return [reason for reason in reasons if reason]


def _weight_for_status(
    status: WatchlistStatus,
    confidence: Confidence,
    limits: ETFWatchlistLimits,
) -> float:
    if status != "active":
        return 0.0
    if confidence == "high":
        return limits.high_confidence_active_weight
    if confidence == "low":
        return limits.low_confidence_active_weight
    return limits.default_active_weight


def _enforce_active_limits(
    decisions: list[SectorWatchlistDecision],
    limits: ETFWatchlistLimits,
) -> list[SectorWatchlistDecision]:
    active_count = 0
    total_weight = 0.0
    adjusted: list[SectorWatchlistDecision] = []
    for decision in decisions:
        if decision.status != "active":
            adjusted.append(decision)
            continue
        would_exceed_count = active_count >= limits.max_active_sectors
        would_exceed_weight = total_weight + decision.watchlist_weight_hint > limits.max_total_active_weight
        if would_exceed_count or would_exceed_weight:
            downgraded = decision.model_copy(deep=True)
            downgraded.status = "monitor"
            downgraded.watchlist_weight_hint = 0.0
            downgraded.objections = [
                "active 名额或总权重约束降级为 monitor",
                *downgraded.objections,
            ]
            downgraded.risk_details = downgraded.objections
            adjusted.append(downgraded)
            continue
        decision.watchlist_weight_hint = min(decision.watchlist_weight_hint, limits.max_single_sector_weight)
        total_weight += decision.watchlist_weight_hint
        active_count += 1
        adjusted.append(decision)
    return adjusted
