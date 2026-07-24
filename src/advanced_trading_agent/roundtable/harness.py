"""Roundtable harness — builds structured debate context from DataAgent tier1/tier2.

This module is stateless and deterministic.  It reads DataAgent's structured
output and produces:
  1. Agent contexts (system messages + evidence snippets) per specialist
  2. An evidence board (field-level traceability)
  3. Agent speaking order

Outputs are immutable context objects that can feed the local DebateEngine.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..agents.schemas import BacktestReport, MarketReport


@dataclass
class DataAgentBrief:
    """Compact summary of DataAgent field values for one agent."""
    system_message: str = ""
    evidence_text: str = ""
    agent_order: tuple[str, ...] = ("Market", "Event", "Analysis", "Backtest", "Risk")


@dataclass
class RoundtableAgentContext:
    """Context for one agent in the roundtable."""
    system_message: str = ""
    evidence_text: str = ""


@dataclass
class RoundtableContext:
    """Immutable context built by RoundtableHarness."""
    agent_contexts: dict[str, RoundtableAgentContext] = field(default_factory=dict)
    agent_order: tuple[str, ...] = ("Market", "Event", "Analysis", "Backtest", "Risk")
    evidence_board: list[Any] = field(default_factory=list)
    contradiction_strings: list[str] = field(default_factory=list)


class RoundtableHarness:
    """Build structured debate context from DataAgent tier1/tier2.

    Usage:
        harness = RoundtableHarness()
        context = harness.build_context(state, contradiction_strings)
        for agent_name in context.agent_order:
            ctx = context.agent_contexts[agent_name]
            # ctx.system_message, ctx.evidence_text
    """

    def build_context(
        self,
        state: dict[str, Any],
        contradiction_strings: list[str] | None = None,
    ) -> RoundtableContext:
        tier1 = state.get("tier1_data", {}) or {}
        tier2 = state.get("tier2_data", {}) or {}
        market = tier1.get("market", {}) or {}
        capital = tier1.get("capital", {}) or {}
        risk_data = tier1.get("risk", {}) or {}
        sector = tier1.get("sector", {}) or {}
        sentiment = tier1.get("sentiment", {}) or {}
        events_data = tier2.get("events", []) or []
        factors_data = tier2.get("factors", []) or []
        price_data = tier2.get("price_data", []) or []
        a_share = tier2.get("a_share_signals", {}) or {}
        data_quality = tier2.get("data_quality") or tier1.get("data_quality", {})

        evidence_board = self._build_evidence_board(
            market=market, capital=capital, risk=risk_data,
            sector=sector, sentiment=sentiment,
            events=events_data, factors=factors_data,
            price=price_data, a_share=a_share,
            data_quality=data_quality,
        )

        market_ctx = RoundtableAgentContext(
            system_message="你是 Market Agent，分析大盘情绪、资金流向和市场阶段。",
            evidence_text=self._format_market_evidence(market, capital, sentiment, sector, a_share),
        )
        event_ctx = RoundtableAgentContext(
            system_message="你是 Event Agent，分析事件驱动逻辑、传导链条和定价状态。",
            evidence_text=self._format_event_evidence(events_data, a_share),
        )
        analysis_ctx = RoundtableAgentContext(
            system_message="你是 Analysis Agent，分析因子数据、技术面和板块拥挤度。",
            evidence_text=self._format_analysis_evidence(factors_data, price_data, a_share),
        )
        backtest_ctx = RoundtableAgentContext(
            system_message="你是 Backtest Agent，评估回测证据质量和历史胜率。",
            evidence_text=self._format_backtest_evidence(tier2, price_data),
        )
        risk_ctx = RoundtableAgentContext(
            system_message="你是 Risk Agent，评估硬风控、流动性和数据质量风险。",
            evidence_text=self._format_risk_evidence(risk_data, data_quality, events_data),
        )

        return RoundtableContext(
            agent_contexts={
                "Market": market_ctx,
                "Event": event_ctx,
                "Analysis": analysis_ctx,
                "Backtest": backtest_ctx,
                "Risk": risk_ctx,
            },
            evidence_board=evidence_board,
            contradiction_strings=contradiction_strings or [],
        )

    # ------------------------------------------------------------------
    # Evidence board builders
    # ------------------------------------------------------------------

    def _build_evidence_board(self, **kwargs: Any) -> list[dict[str, Any]]:
        board: list[dict[str, Any]] = []
        field_map: dict[str, tuple[str, str]] = {
            "ev_market_state": ("market", "index_change_pct"),
            "ev_market_capital": ("capital", "confirmation"),
            "ev_risk_st_status": ("risk", "st_status"),
            "ev_sector_name": ("sector", "matched_sector"),
            "ev_sentiment": ("sentiment", "sentiment"),
        }
        for ev_id, (section, field) in field_map.items():
            section_data = kwargs.get(section, {}) or {}
            value = section_data.get(field)
            if value is not None:
                board.append({"id": ev_id, "field_path": f"tier1.{section}.{field}", "value": str(value)[:200], "agent": ""})
        return board

    def _format_market_evidence(self, market: dict, capital: dict, sentiment: dict, sector: dict, a_share: dict) -> str:
        return json.dumps({
            "index_change_pct": market.get("index_change_pct"),
            "index_volume_change_pct": market.get("index_volume_change_pct"),
            "capital_confirmation": capital.get("confirmation"),
            "capital_flow_summary": capital.get("flow_summary"),
            "sentiment": sentiment.get("sentiment"),
            "matched_sector": sector.get("matched_sector"),
            "sector_match_confidence": sector.get("match_confidence"),
            "limit_up_count": market.get("limit_up_summary", {}).get("limit_up_count"),
            "hot_sectors": [s.get("sector_name") for s in (sector.get("top_sectors") or [])[:5]],
        }, ensure_ascii=False, indent=2)

    def _format_event_evidence(self, events: list[dict], a_share: dict) -> str:
        event_lines = []
        for e in (events or [])[:8]:
            event_lines.append({
                "title": e.get("title", "")[:80],
                "direction": e.get("direction"),
                "chain_quality": e.get("chain_quality"),
                "confidence": e.get("confidence"),
                "evidence_level": e.get("evidence_level"),
                "pricing_status": e.get("pricing_status"),
            })
        result = {"events": event_lines}
        policy = (a_share or {}).get("policy", {})
        if policy.get("data_status") in {"available", "partial"}:
            result["policy_signal"] = policy.get("signal")
        return json.dumps(result, ensure_ascii=False, indent=2)

    def _format_analysis_evidence(self, factors: list[dict], price_data: list[dict], a_share: dict) -> str:
        latest = factors[-1] if factors else {}
        result = {
            "latest_composite_score": latest.get("composite_score"),
            "latest_factor_warnings": latest.get("warnings", [])[:5],
            "latest_close": price_data[-1].get("close") if price_data else None,
            "price_change_pct_5d": price_data[-1].get("pct_chg") if price_data and len(price_data) > 1 else None,
        }
        hm = (a_share or {}).get("hot_money", {})
        if hm.get("data_status") in {"available", "partial"}:
            result["hot_money_signal"] = hm.get("signal")
            result["hot_money_score"] = hm.get("score")
        return json.dumps(result, ensure_ascii=False, indent=2)

    def _format_backtest_evidence(self, tier2: dict, price_data: list[dict]) -> str:
        backtest_samples = tier2.get("backtest_samples", []) or []
        sample = backtest_samples[0] if backtest_samples else {}
        return json.dumps({
            "sample_size": sample.get("sample_size", 0),
            "win_rate": sample.get("win_rate"),
            "avg_excess_return": sample.get("avg_excess_return"),
            "price_records_count": len(price_data or []),
        }, ensure_ascii=False, indent=2)

    def _format_risk_evidence(self, risk: dict, data_quality: dict, events: list[dict]) -> str:
        return json.dumps({
            "risk_data_available": risk.get("risk_data_available", True),
            "st_status": risk.get("st_status"),
            "suspended_status": risk.get("suspended_status"),
            "daily_volume": risk.get("daily_volume"),
            "estimated_impact_bps": risk.get("estimated_impact_bps"),
            "data_quality_consistency": (data_quality or {}).get("daily_consistency"),
            "event_risk": any(e.get("direction") == "利空" for e in (events or [])[:5]),
        }, ensure_ascii=False, indent=2)

    @staticmethod
    def _signal_meets_criteria(signal_data: dict, rules: dict) -> bool:
        """Check a single A-share signal against its criteria thresholds."""
        for field, threshold in rules.items():
            value = signal_data.get(field)
            if value is None:
                return False
            if isinstance(threshold, dict):
                op = threshold.get("op", "ge")
                target = threshold.get("value")
                if op == "ge" and not (value >= target):
                    return False
                elif op == "le" and not (value <= target):
                    return False
                elif op == "eq" and not (value == target):
                    return False
        return True
