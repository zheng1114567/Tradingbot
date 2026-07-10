"""
Event Agent — 事件分析师

职责:
- 识别事件类型和方向
- 判断传导路径
- 估计半衰期和失效条件
- 检查证据等级和定价状态

反伪链条规则 (硬约束在 schema 中):
- 实体映射: 必须能映射到上市公司
- 收入暴露: 必须说明业务关联
- 传导长度: >3 跳降级为观察
- 已定价检查: 连续大涨标记已定价
"""
from __future__ import annotations

import logging
from typing import Any

from ..llm.client import LLMClient
from .schemas import EventReport

logger = logging.getLogger(__name__)


def create_event_agent(llm: LLMClient):
    """创建 Event Agent 节点函数"""

    def event_node(state: dict[str, Any]) -> dict[str, Any]:
        ticker = state.get("company_of_interest", "")
        market_report = state.get("market_report_obj", None)
        tier2_data = state.get("tier2_data", {})
        events = tier2_data.get("events", [])
        memory = state.get("memory_context", "")

        event_summary = "\n".join(
            f"- [{e.get('event_type', '未知')}] {e.get('summary', '')} "
            f"(证据: {e.get('evidence_level', 'N/A')})"
            for e in events[:10]
        ) if events else "暂无事件数据"

        prompt = f"""你是 Event Agent, 负责判断事件是否有交易价值。

## 当前标的
{ticker}

## 市场背景
{market_report}

## 事件列表
{event_summary}

## 历史记忆
{memory[:500] if memory else "暂无"}

请分析最重要的事件:
1. 事件是否可映射到实体/板块?
2. 传导路径是否直接? 超过 3 跳必须降级
3. 是否已被市场定价?(连续大涨 = 已定价)
4. 证据等级够不够? (低等级不能单独支撑推荐)
5. 半衰期多长? 失效条件是什么?

输出结构化事件分析报告。"""

        try:
            report = llm.chat(
                messages=[
                    ("system", "你是 A 股事件分析师。严格遵守反伪链条规则。"),
                    ("human", prompt),
                ],
                response_format=EventReport,
            )
        except Exception as e:
            logger.warning("LLM event analysis failed: %s", e)
            report = EventReport(
                event_id="default",
                event_type="情绪",
                direction="中性",
                confidence=0.3,
                transmission_path="无明确传导路径",
                direct_beneficiaries=[],
                evidence_level="社交传闻",
                pricing_status="未定价",
                chain_quality="weak",
                reasoning="LLM 不可用时的默认降级分析",
                invalid_conditions=["无法确认事件影响"],
            )

        return {
            "event_report": report.model_dump() if hasattr(report, "model_dump") else str(report),
            "event_report_obj": report,
        }

    return event_node
