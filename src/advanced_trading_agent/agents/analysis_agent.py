"""
Analysis Agent — 量化因子分析师

因子数据由 DataService 的 FactorCalculator 确定性计算 (pandas/numpy, 无 LLM)。
Analysis Agent 读取预计算因子, 做解释、排序、识别风险。

LLM 职责: 解释因子、识别异常、给出择时建议
非 LLM 职责: factor 计算 (DataService)、排序 (工具函数)
"""
from __future__ import annotations

import logging
from typing import Any

from ..llm.client import LLMClient
from ..tool_nodes.analysis_tools import AnalysisTools
from .schemas import AnalysisReport, StockRanking

logger = logging.getLogger(__name__)


def create_analysis_agent(llm: LLMClient):
    """创建 Analysis Agent 节点函数"""

    def analysis_node(state: dict[str, Any]) -> dict[str, Any]:
        ticker = state.get("company_of_interest", "")
        tier2 = state.get("tier2_data", {})
        factors_raw = tier2.get("factors", [])

        # 用工具获取更多因子数据
        tools = AnalysisTools()
        factors = tools.get_factor_data(code=ticker, top_n=20)
        if not factors:
            factors = factors_raw

        # 确定性排序 (不用 LLM)
        sorted_factors = sorted(
            factors,
            key=lambda x: x.get("composite_score", 0) or 0,
            reverse=True,
        )

        # 生成 StockRanking (确定性代码, 不用 LLM)
        rankings = []
        for i, f in enumerate(sorted_factors[:10]):
            warnings = []
            if f.get("factor_warning"):
                warnings.append(str(f.get("factor_warning", "")))
            if f.get("liquidity_score", 1) and f["liquidity_score"] < 3:
                warnings.append("流动性不足")
            if f.get("valuation_score", 5) and f["valuation_score"] > 8:
                warnings.append("估值偏高")

            rankings.append(StockRanking(
                code=f.get("code", ""),
                name=f.get("name", ""),
                composite_score=f.get("composite_score", 5) or 5,
                main_driver=f"质量:{f.get('quality_score', 'N/A')} "
                            f"成长:{f.get('growth_score', 'N/A')}",
                warnings=warnings,
            ))

        # LLM 分析因子模式
        factor_details = "\n".join(
            f"{r.name}({r.code}): {r.composite_score:.1f}分 {'⚠'+'|'.join(r.warnings) if r.warnings else ''}"
            for r in rankings[:5]
        )

        prompt = f"""你是 Analysis Agent, 量化因子分析师。

## 标的
{ticker}

## 因子排序 Top 5
{factor_details}

请分析:
1. 主要驱动因子是什么? (质量/成长/估值/动量/波动/流动中的哪些在驱动)
2. 是否有因子拥挤或失效的迹象?
3. 择时条件: 什么情况下应该买入/避免?
4. 综合板块评分如何?

输出结构化分析报告。"""

        try:
            report = llm.chat(
                messages=[
                    ("system", "你是 A 股因子分析师。基于因子数据分析, 不凭感觉。"),
                    ("human", prompt),
                ],
                response_format=AnalysisReport,
            )
            # 用确定性排序结果覆盖 LLM 的排序 (LLM 排序不可靠)
            if rankings:
                report.stock_rankings = rankings
        except Exception as e:
            logger.warning("LLM analysis failed, using deterministic: %s", e)
            report = AnalysisReport(
                sector_score=sum(
                    r.composite_score for r in rankings
                ) / len(rankings) if rankings else None,
                stock_rankings=rankings,
                factor_explanation="基于因子数据的确定性排序 (LLM不可用)",
                factor_warnings=[],
                reasoning="根据预计算因子的规则分析",
            )

        return {
            "analysis_report": report.to_markdown() if hasattr(report, 'to_markdown') else str(report),
            "analysis_report_obj": report,
            "sender": "Analysis Agent",
        }

    return analysis_node
