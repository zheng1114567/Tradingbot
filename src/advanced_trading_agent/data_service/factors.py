"""
因子计算 — 质量和风险因子

确定性计算, 不使用 LLM。
借鉴 TradingAgents 的 stockstats_utils 技术指标计算模式。
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FactorCalculator:
    """因子计算器 — 用于 Analysis Agent 的因子框架"""

    # ============================================================
    # 质量因子
    # ============================================================

    @staticmethod
    def roe(df: pd.DataFrame) -> pd.DataFrame:
        """ROE (净资产收益率)"""
        if "roe" in df.columns:
            return df
        if "net_profit" in df.columns and "equity" in df.columns:
            df["roe"] = df["net_profit"] / df["equity"].replace(0, np.nan)
        return df

    @staticmethod
    def gross_margin(df: pd.DataFrame) -> pd.DataFrame:
        """毛利率"""
        if "gross_margin" in df.columns:
            return df
        if "revenue" in df.columns and "cost" in df.columns:
            df["gross_margin"] = (df["revenue"] - df["cost"]) / df["revenue"].replace(0, np.nan)
        return df

    # ============================================================
    # 成长因子
    # ============================================================

    @staticmethod
    def revenue_growth(df: pd.DataFrame, periods: int = 4) -> pd.DataFrame:
        """营收增速 (同比)"""
        if "revenue" in df.columns and len(df) > periods:
            df["revenue_growth"] = df["revenue"].pct_change(periods=periods)
        return df

    @staticmethod
    def profit_growth(df: pd.DataFrame, periods: int = 4) -> pd.DataFrame:
        """利润增速 (同比)"""
        if "net_profit" in df.columns and len(df) > periods:
            df["profit_growth"] = df["net_profit"].pct_change(periods=periods)
        return df

    # ============================================================
    # 估值因子
    # ============================================================

    @staticmethod
    def pe_quantile(df: pd.DataFrame) -> pd.DataFrame:
        """PE 历史分位数"""
        if "pe" in df.columns and len(df) > 20:
            df["pe_quantile"] = df["pe"].rank(pct=True)
        return df

    @staticmethod
    def pb_quantile(df: pd.DataFrame) -> pd.DataFrame:
        """PB 历史分位数"""
        if "pb" in df.columns and len(df) > 20:
            df["pb_quantile"] = df["pb"].rank(pct=True)
        return df

    # ============================================================
    # 动量因子
    # ============================================================

    @staticmethod
    def momentum(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
        """N 日动量"""
        if periods is None:
            periods = [5, 10, 20, 60]
        if "close" not in df.columns:
            return df
        for p in periods:
            if len(df) > p:
                df[f"momentum_{p}d"] = df["close"].pct_change(p)
        return df

    @staticmethod
    def ma_trend(df: pd.DataFrame) -> pd.DataFrame:
        """均线趋势 (MA5, MA10, MA20 位置关系)"""
        if "close" not in df.columns:
            return df
        if len(df) >= 5:
            df["ma5"] = df["close"].rolling(5).mean()
        if len(df) >= 10:
            df["ma10"] = df["close"].rolling(10).mean()
        if len(df) >= 20:
            df["ma20"] = df["close"].rolling(20).mean()
        # 趋势强度: 短均线在长均线之上为正
        if all(c in df.columns for c in ["ma5", "ma20"]):
            df["ma_trend"] = (df["ma5"] - df["ma20"]) / df["ma20"].replace(0, np.nan)
        return df

    # ============================================================
    # 波动率因子
    # ============================================================

    @staticmethod
    def volatility(df: pd.DataFrame, periods: int = 20) -> pd.DataFrame:
        """N 日年化波动率"""
        if "pct_chg" in df.columns and len(df) > periods:
            df["volatility"] = df["pct_chg"].rolling(periods).std() * np.sqrt(252)
        return df

    @staticmethod
    def max_drawdown(df: pd.DataFrame, periods: int = 60) -> pd.DataFrame:
        """N 日最大回撤"""
        if "close" in df.columns and len(df) > periods:
            rolling_max = df["close"].rolling(periods).max()
            df["max_drawdown"] = (df["close"] - rolling_max) / rolling_max.replace(0, np.nan)
        return df

    # ============================================================
    # 流动性因子
    # ============================================================

    @staticmethod
    def turnover_rate(df: pd.DataFrame) -> pd.DataFrame:
        """换手率"""
        if "turnover" in df.columns:
            return df
        if "volume" in df.columns and "close" in df.columns:
            # 简单估算: 成交额/流通市值 近似
            df["turnover"] = df["volume"] * df["close"]
        return df

    @staticmethod
    def amihud_illiquidity(df: pd.DataFrame, periods: int = 20) -> pd.DataFrame:
        """Amihud 非流动性指标 (越大越不流动)"""
        if "pct_chg" in df.columns and "amount" in df.columns and len(df) > periods:
            df["amihud"] = (df["pct_chg"].abs() / df["amount"].replace(0, np.nan))
            df["amihud"] = df["amihud"].replace([np.inf, -np.inf], np.nan)
        return df

    # ============================================================
    # 情绪/资金指标
    # ============================================================

    @staticmethod
    def capital_trend(df: pd.DataFrame) -> pd.DataFrame:
        """资金流向趋势"""
        if "net_inflow_main" in df.columns:
            df["net_inflow_ma5"] = df["net_inflow_main"].rolling(5).mean()
            df["capital_strength"] = (df["net_inflow_main"] > 0).astype(int)
        return df

    # ============================================================
    # 综合评分
    # ============================================================

    @staticmethod
    def composite_score(df: pd.DataFrame,
                         weights: dict[str, float] | None = None) -> pd.DataFrame:
        """因子综合评分 (Z-Score 加权)"""
        if weights is None:
            weights = {
                "roe": 0.15,
                "revenue_growth": 0.15,
                "profit_growth": 0.10,
                "pe_quantile": -0.10,  # 低PE得分高
                "momentum_20d": 0.15,
                "volatility": -0.10,
                "turnover": 0.10,
                "ma_trend": 0.15,
            }
        score = pd.Series(0.0, index=df.index)
        for factor, weight in weights.items():
            if factor not in df.columns:
                continue
            valid = df[factor].notna() & (~np.isinf(df[factor].replace([np.inf, -np.inf], np.nan)))
            if valid.sum() < 10:
                continue
            z = (df[factor] - df[factor].mean()) / df[factor].std().replace(0, np.nan)
            score += z.fillna(0) * weight
        df["composite_score"] = score
        return df

    @staticmethod
    def run_all(df: pd.DataFrame) -> pd.DataFrame:
        """运行所有因子计算"""
        if df.empty:
            return df
        df = FactorCalculator.roe(df)
        df = FactorCalculator.revenue_growth(df)
        df = FactorCalculator.profit_growth(df)
        df = FactorCalculator.pe_quantile(df)
        df = FactorCalculator.pb_quantile(df)
        df = FactorCalculator.momentum(df)
        df = FactorCalculator.ma_trend(df)
        df = FactorCalculator.volatility(df)
        df = FactorCalculator.max_drawdown(df)
        df = FactorCalculator.amihud_illiquidity(df)
        df = FactorCalculator.composite_score(df)
        return df
