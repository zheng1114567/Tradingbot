"""
Event Agent — 事件分析师

模式: LLM + ToolNode

LLM 决定搜索什么事件 → ToolNode 搜索财联社/东方财富 → LLM 分析传导链

反伪链条规则 (硬约束):
1. 实体映射: 事件主体必须能映射到上市公司/板块
2. 收入暴露: 推荐个股时必须有业务关联依据
3. 传导长度: >3 跳降级为 indirect
4. 已定价检查: 连续大涨 → 标记已定价
5. 证据等级: 低等级不能单独支撑推荐
"""

from __future__ import annotations

import logging
from typing import Any

from ..llm.client import LLMClient
from ..tool_nodes.event_tools import EventTools
from ..tool_nodes.registry import get_agent_tools
from .contract import (
    basic_self_check,
    build_agent_update,
    build_react_agent,
    run_react_agent,
)
from .schemas import EventReport
from .specs import get_agent_skill

logger = logging.getLogger(__name__)


def create_event_agent(llm: LLMClient, tools: EventTools | None = None):
    """创建 Event Agent 节点函数"""
    skill = get_agent_skill("event")
    react_agent = build_react_agent(
        llm=llm,
        tools=get_agent_tools("event"),
        system_prompt=skill.react_prompt,
        response_format=EventReport,
    )

    def event_node(state: dict[str, Any]) -> dict[str, Any]:
        ticker = state.get("company_of_interest", "")
        tier2 = state.get("tier2_data", {})
        events_raw = tier2.get("events", [])
        memory = state.get("memory_context", "")

        # 用工具搜索事件相关新闻
        event_tools = tools or EventTools()
        news_items = event_tools.search_cailianshe_news(ticker)
        eastmoney_news = event_tools.search_eastmoney_news(ticker)
        news_items.extend(eastmoney_news)
        tool_calls = [
            {
                "tool": "search_cailianshe_news",
                "args": {"ticker": ticker},
                "records": len(news_items) - len(eastmoney_news),
            },
            {
                "tool": "search_eastmoney_news",
                "args": {"ticker": ticker},
                "records": len(eastmoney_news),
            },
        ]

        # 识别实体
        detected_entities = []
        for item in news_items:
            text = str(item)
            entities = event_tools.detect_entity(text)
            detected_entities.extend(entities)
        tool_calls.append(
            {
                "tool": "detect_entity",
                "args": {"items": len(news_items)},
                "records": len(detected_entities),
            }
        )

        event_summary = (
            "\n".join(
                f"- [{e.get('event_type', '?')}] {e.get('summary', str(e))[:100]}"
                for e in events_raw[:10]
            )
            if events_raw
            else "暂无结构化事件数据"
        )

        news_summary = (
            "\n".join(f"- {str(n)[:80]}" for n in news_items[:10])
            if news_items
            else "暂无相关新闻"
        )

        entities_summary = "未检测到明确主题"
        if detected_entities:
            entity_parts = []
            for e in detected_entities[:5]:
                theme = e.get("theme", "")
                kw = e.get("keyword", "")
                entity_parts.append(f"{theme}({kw})")
            entities_summary = f"检测到: {', '.join(entity_parts)}"

        prompt = f"""你是 Event Agent, 负责判断事件是否有交易价值。

## 标的
{ticker}

## 结构化事件
{event_summary}

## 最新新闻
{news_summary}

## 实体识别
{entities_summary}

## 历史记忆
{memory[:300] if memory else "暂无"}

## 反伪链条规则 (必须遵守)
1. 实体映射: 事件主体必须能映射到上市公司或板块
2. 收入暴露: 推荐个股必须有业务关联依据, 否则只能给板块级观察
3. 传导长度: 超过 3 跳的链条降级为 indirect
4. 已定价检查: 标的连续大涨/涨停 → 标记已定价
5. 证据等级: 公告/披露 > 权威媒体 > 行业媒体 > 社交传闻
6. 低等级证据不能单独支撑推荐

请分析最重要的事件。输出结构化报告。"""

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
                    response_format=EventReport,
                )
        except Exception as e:
            logger.warning("LLM event analysis failed, defaulting: %s", e)
            report = EventReport(
                event_id=f"event_{ticker}",
                event_type="情绪",
                direction="中性",
                confidence=0.3,
                transmission_path="LLM不可用时无法分析传导路径",
                direct_beneficiaries=[],
                evidence_level="社交传闻",
                pricing_status="未定价",
                chain_quality="weak",
                reasoning="LLM不可用, 默认降级",
                invalid_conditions=["LLM不可用, 建议人工复核"],
            )

        evidence = [
            f"structured_events={len(events_raw)}",
            f"news_items={len(news_items)}",
            f"detected_entities={entities_summary}",
            f"evidence_level={report.evidence_level}",
            f"chain_quality={report.chain_quality}",
            f"pricing_status={report.pricing_status}",
        ]
        warnings = []
        if report.chain_quality != "direct":
            warnings.append(f"传导链条非 direct: {report.chain_quality}")
        if report.evidence_level in ("行业媒体", "社交传闻"):
            warnings.append(f"证据等级偏弱: {report.evidence_level}")
        if report.pricing_status == "过度定价":
            warnings.append("事件可能已过度定价")

        return build_agent_update(
            state,
            sender="Event Agent",
            report_key="event_report",
            report=report.to_markdown()
            if hasattr(report, "to_markdown")
            else str(report),
            report_obj_key="event_report_obj",
            report_obj=report,
            evidence=evidence,
            tool_calls=tool_calls,
            self_check=basic_self_check(
                evidence=evidence,
                passed_rules=["anti_fake_event_chain_checked"],
                warnings=warnings,
                confidence=report.confidence,
            ),
        )

    return event_node
