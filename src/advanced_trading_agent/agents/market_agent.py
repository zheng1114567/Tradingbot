"""
Market Agent — 市场温度分析师

模式: LLM + ToolNode

LLM 决定需要哪些数据:
1. 先调 get_market_sentiment() 获取情绪数据
2. 再调 get_sector_rotation() 看板块轮动
3. 需要时调 get_northbound_flow() / get_capital_flow()
4. 涨停梯队分析

ToolNode 执行实际调用, LLM 读取结果生成报告。

对比 v0.1.0 的改进:
- Agent 不再是"LLM读别人塞的数据"
- 而是"LLM自己决定要看什么数据, ToolNode去取"
- 匹配 TradingAgents 的 ToolNode 模式
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from ..llm.client import LLMClient
from ..tool_nodes.market_tools import MarketTools
from ..tool_nodes.registry import get_agent_tools
from .contract import (
    basic_self_check,
    build_agent_update,
    build_react_agent,
    run_react_agent,
)
from .schemas import MarketReport
from .specs import get_agent_skill

logger = logging.getLogger(__name__)

# 情绪 → 仓位上限映射 (确定性规则)
SENTIMENT_POSITION_CAP = {
    "冰点": 0.20,
    "低迷": 0.40,
    "正常": 0.60,
    "温热": 0.50,
    "高潮": 0.30,
}


def create_market_agent(llm: LLMClient, tools: MarketTools | None = None):
    """创建 Market Agent 节点函数"""
    skill = get_agent_skill("market")
    react_agent = build_react_agent(
        llm=llm,
        tools=get_agent_tools("market"),
        system_prompt=skill.react_prompt,
        response_format=MarketReport,
    )

    def market_node(state: dict[str, Any]) -> dict[str, Any]:
        trade_date = state.get("trade_date", str(date.today()))
        ticker = state.get("company_of_interest", "")

        # 从 Tier 1 读取数据
        tier1 = state.get("tier1_data", {})
        sentiment_data = tier1.get("sentiment", {})
        capital_data = tier1.get("capital", {})

        # 确定性规则: 情绪 → 仓位上限
        sentiment_level = sentiment_data.get("sentiment", "正常")
        if isinstance(sentiment_level, str):
            position_cap = SENTIMENT_POSITION_CAP.get(sentiment_level, 0.50)
        else:
            position_cap = 0.50

        # 资金确认状态
        capital_conf = capital_data.get("confirmation", "未知")

        # 用工具获取附加数据
        market_tools = tools or MarketTools()
        limit_up = market_tools.get_limit_up_tiers(trade_date)
        sector_rotation = market_tools.get_sector_rotation(top_n=5)
        tool_calls = [
            {
                "tool": "get_limit_up_tiers",
                "args": {"trade_date": trade_date},
                "records": 1 if limit_up else 0,
            },
            {
                "tool": "get_sector_rotation",
                "args": {"top_n": 5},
                "records": len(sector_rotation or []),
            },
        ]

        # 格式化板块轮动 (避免 f-string 转义问题)
        sector_lines = []
        for s in sector_rotation:
            sn = s.get("sector_name", "")
            cp = s.get("change_pct", "")
            sector_lines.append(f"{sn}({cp}%)")
        sector_str = " ".join(sector_lines)

        # LLM 综合分析
        prompt_lines = [
            "你是 Market Agent, 负责判断市场温度和资金状态。",
            "",
            "## Tier 1 数据",
            f"- 大盘涨跌幅: {tier1.get('market', {}).get('index_change_pct', 'N/A')}%",
            f"- 上涨/下跌: {tier1.get('market', {}).get('advance_count', 'N/A')}/{tier1.get('market', {}).get('decline_count', 'N/A')}",
            f"- 涨停/跌停: {tier1.get('market', {}).get('limit_up_count', 'N/A')}/{tier1.get('market', {}).get('limit_down_count', 'N/A')}",
            "",
            "## 情绪",
            f"- 档位: {sentiment_level}",
            f"- 得分: {sentiment_data.get('sentiment_score', 'N/A')}",
            "",
            "## 资金",
            f"- 板块: {capital_data.get('sector_name', ticker)}",
            f"- 主力净流入: {capital_data.get('net_inflow_main', 'N/A')}",
            f"- 确认状态: {capital_conf}",
            "",
            "## 涨停梯队 (A股特有)",
            f"- 首板: {limit_up.get('first_board', 'N/A')}",
            f"- 二板: {limit_up.get('second_board', 'N/A')}",
            f"- 三板+: {limit_up.get('third_plus', 'N/A')}",
            "",
            "## 板块轮动 Top 5",
            sector_str,
            "",
            "## 仓位规则 (必须遵守)",
            "- 冰点 → 上限 20% | 低迷 → 40% | 正常 → 60% | 温热 → 50% | 高潮 → 30%",
            "- 资金背离时减半仓位",
            "- 高潮时不允许加仓",
            "",
            "请输出结构化的市场分析报告。",
        ]
        prompt = "\n".join(prompt_lines)

        report, react_trace = run_react_agent(react_agent, prompt)
        if react_trace:
            tool_calls.extend(react_trace)

        try:
            if report is None:
                report = llm.chat(
                    messages=[
                        ("system", skill.fallback_system_prompt),
                        ("human", prompt),
                    ],
                    response_format=MarketReport,
                )
            # 规则覆盖 LLM (仓位上限必须遵守)
            if hasattr(report, "position_cap"):
                report.position_cap = min(report.position_cap, position_cap)
        except Exception as e:
            logger.warning("LLM market analysis failed, using rules: %s", e)
            report = MarketReport(
                market_state=sentiment_level,
                position_cap=position_cap,
                capital_confirmation=capital_conf,
                sector_preference=[
                    s.get("sector_name", "") for s in sector_rotation[:3]
                ],
                reasoning=f"规则判断: 情绪{sentiment_level}, 仓位上限{position_cap:.0%}",
            )

        evidence = [
            f"sentiment={sentiment_level}",
            f"position_cap_rule={position_cap:.0%}",
            f"capital_confirmation={capital_conf}",
            f"limit_up_tiers={limit_up}",
            f"sector_rotation_top={sector_str or 'N/A'}",
        ]
        warnings = []
        if capital_conf in ("资金背离", "资金不足", "未知"):
            warnings.append(f"资金状态需复核: {capital_conf}")
        if sentiment_level in ("冰点", "高潮"):
            warnings.append(f"市场温度极端: {sentiment_level}")

        return build_agent_update(
            state,
            sender="Market Agent",
            report_key="market_report",
            report=report.to_markdown()
            if hasattr(report, "to_markdown")
            else str(report),
            report_obj_key="market_report_obj",
            report_obj=report,
            evidence=evidence,
            tool_calls=tool_calls,
            self_check=basic_self_check(
                evidence=evidence,
                passed_rules=["sentiment_position_cap_enforced"],
                warnings=warnings,
                confidence=getattr(report, "position_cap", None),
            ),
        )

    return market_node
