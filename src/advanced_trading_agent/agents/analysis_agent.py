"""
Analysis Agent — 量化分析师

职责:
- 板块六因子评分
- 个股排序
- 识别估值拥挤/流动性不足/动量衰竭
- 设置择时过滤条件

因子数据由 data_service/factors.py 确定性计算,
Analysis Agent 负责解释和排序。
"""
from __future__ import annotations

import logging
from typing import Any

from ..llm.client import LLMClient
from .schemas import AnalysisReport, StockRanking

logger = logging.getLogger(__name__)


def create_analysis_agent(llm: LLMClient):
    """创建 Analysis Agent 节点函数"""

    def analysis_node(state: dict[str, Any]) -> dict[str, Any]:
        ticker = state.get("company_of_interest", "")
        tier2_data = state.get("tier2_data", {})
        factors = tier2_data.get("factors", [])
        market_report = state.get("market_report_obj", None)

        factor_summary = "\n".join(
            f"- {f.get('name', f.get('code', ''))}: "
            f"综合分 {f.get('composite_score', 'N/A')}, "
            f"质量 {f.get('quality_score', 'N/A')}, "
            f"成长 {f.get('growth_score', 'N/A')}, "
            f"估值 {f.get('valuation_score', 'N/A')}, "
            f"动量 {f.get('momentum_score', 'N/A')}, "
            f"波动 {f.get('volatility_score', 'N/A')}, "
            f"流动 {f.get('liquidity_score', 'N/A')}"
            for f in factors[:20]
        ) if factors else "暂无因子数据"

        prompt = f"""你是 Analysis Agent, 量化分析师。

## 标的
{ticker}

## 市场背景
{market_report}

## 因子数据
{factor_summary}

请基于因子数据进行评估:
1. 板块因子评分如何? (质量/成长/估值/动量/波动/流动)
2. 个股排序如何? 哪些因子驱动?
3. 是否有估值拥挤、流动性不足或动量衰竭?
4. 择时过滤条件是什么?

输出结构化因子分析报告。"""

        try:
            report = llm.chat(
                messages=[
                    ("system", "你是 A 股量化分析师。基于因子数据做评估, 不凭感觉。"),
                    ("human", prompt),
                ],
                response_format=AnalysisReport,
            )
        except Exception as e:
            logger.warning("LLM analysis failed, using factor-only: %s", e)
            rankings = []
            for i, f in enumerate(factors[:10]):
                rankings.append(StockRanking(
                    code=f.get("code", ""),
                    name=f.get("name", ""),
                    composite_score=f.get("composite_score", 5) or 5,
                    main_driver=f"质量:{f.get('quality_score', 'N/A')} "
                                f"成长:{f.get('growth_score', 'N/A')}",
                ))
            report = AnalysisReport(
                sector_score=None,
                stock_rankings=rankings,
                factor_explanation="规则模式(LLM不可用)",
                factor_warnings=[],
                reasoning="基于因子数据的排序结果",
            )

        return {
            "analysis_report": report.model_dump() if hasattr(report, "model_dump") else str(report),
            "analysis_report_obj": report,
        }

    return analysis_node
