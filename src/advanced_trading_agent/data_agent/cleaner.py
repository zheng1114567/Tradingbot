"""
数据清洗 — 缺失值、异常值、停牌、ST、复权、时间对齐
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """数据清洗器 — 负责所有数据清洗逻辑"""

    @staticmethod
    def clean_daily(data: list[dict[str, Any]]) -> pd.DataFrame:
        """清洗日K数据"""
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        # 列名标准化 (akshare / baostock 字段不同)
        df = DataCleaner._standardize_columns(df)
        if "code" not in df.columns:
            df["code"] = ""
        df["code"] = df["code"].astype(str)
        # 时间索引
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
            sort_cols = ["code", "trade_date"] if "code" in df.columns else ["trade_date"]
            df = df.sort_values(sort_cols)
        for col in [
            "open", "high", "low", "close", "pre_close", "change", "pct_chg",
            "volume", "amount", "turnover_rate",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        grouped = df.groupby("code", dropna=False, sort=False) if "code" in df.columns else None
        if "close" in df.columns:
            prev_close = (
                grouped["close"].shift(1)
                if grouped is not None
                else df["close"].shift(1)
            )
            if "pre_close" not in df.columns:
                df["pre_close"] = prev_close
            else:
                df["pre_close"] = df["pre_close"].fillna(prev_close)
        if "pct_chg" not in df.columns and {"close", "pre_close"}.issubset(df.columns):
            denominator = df["pre_close"].replace(0, np.nan)
            df["pct_chg"] = (df["close"] - df["pre_close"]) / denominator * 100
        elif "pct_chg" in df.columns and {"close", "pre_close"}.issubset(df.columns):
            denominator = df["pre_close"].replace(0, np.nan)
            derived_pct = (df["close"] - df["pre_close"]) / denominator * 100
            df["pct_chg"] = df["pct_chg"].fillna(derived_pct)
        # 缺失值: 回测数据禁止用未来值回填过去；多标的批量数据也不能跨代码串值。
        if grouped is not None:
            preserved_code = df["code"].copy()
            df = df.groupby("code", dropna=False, sort=False).ffill()
            df.insert(0, "code", preserved_code)
        else:
            df = df.ffill()
        # 异常值
        if "pct_chg" in df.columns:
            df = df[df["pct_chg"].isna() | (df["pct_chg"].abs() <= 100)]  # 过滤极端涨跌幅
        return df

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名为后续 Agent 统一消费的字段."""
        rename_map = {
            # common legacy/normalized fields
            "ts_code": "code",
            "vol": "volume",
            # akshare -> 标准
            "代码": "code",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "涨跌幅": "pct_chg",
            "涨跌额": "change",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover_rate",
            "日期": "trade_date",
            # baostock -> 标准
            "date": "trade_date",
            "datetime": "trade_date",
            "code": "code",
            "preclose": "pre_close",
            "pctChg": "pct_chg",
            "volume": "volume",
            "turn": "turnover_rate",
            "amount": "amount",
            # 兼容英文 OHLCV 字段
            "Date": "trade_date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            # already-standard fields
            "trade_date": "trade_date",
            "pre_close": "pre_close",
            "pct_chg": "pct_chg",
            "turnover_rate": "turnover_rate",
        }
        # 只重命名存在的列
        existing = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=existing)
        # Some vendors provide both a source field (e.g. vol) and an already
        # normalized field (volume). After renaming, keep the first non-empty
        # version so downstream scalar operations do not receive duplicate columns.
        if df.columns.duplicated().any():
            merged = pd.DataFrame(index=df.index)
            for col in dict.fromkeys(df.columns):
                cols = df.loc[:, df.columns == col]
                series = cols.iloc[:, 0].copy()
                for idx in range(1, cols.shape[1]):
                    other = cols.iloc[:, idx]
                    series = series.where(series.notna(), other)
                merged[col] = series
            df = merged
        return df

    @staticmethod
    def detect_limit_up_down(df: pd.DataFrame) -> pd.DataFrame:
        """检测涨跌停状态 (A股: 普通股票 ±10%, ST ±5%, 科创/创业板 ±20%)"""
        if "close" not in df.columns or "pre_close" not in df.columns:
            return df
        denominator = pd.to_numeric(df["pre_close"], errors="coerce").replace(0, np.nan)
        df["pct_chg"] = (pd.to_numeric(df["close"], errors="coerce") - denominator) / denominator * 100
        df["is_limit_up"] = False
        df["is_limit_down"] = False
        for idx, row in df.iterrows():
            pct = row.get("pct_chg", 0)
            code = str(row.get("code", ""))
            if "ST" in code:  # 同时匹配 ST 和 *ST
                limit = 5.0
            elif code.startswith(("68", "30", "300", "301")):  # 科创/创业板
                limit = 20.0
            else:
                limit = 10.0
            df.at[idx, "is_limit_up"] = pct >= limit - 0.02  # 容差
            df.at[idx, "is_limit_down"] = pct <= -limit + 0.02
        return df

    @staticmethod
    def filter_suspended_st(df: pd.DataFrame,
                            suspended_list: list[str] | None = None,
                            st_list: list[str] | None = None) -> pd.DataFrame:
        """过滤停牌和ST股票"""
        if suspended_list:
            df = df[~df["code"].isin(suspended_list)]
        if st_list:
            # ST 是通过 namechange 判断, 这里筛选代码前缀
            st_codes = [c for c in st_list]
            df = df[~df["code"].isin(st_codes)]
        return df

    @staticmethod
    def align_time(data: pd.DataFrame, as_of_date: date) -> pd.DataFrame:
        """时间对齐: 只保留 as_of_date 之前的数据"""
        if data.empty or "trade_date" not in data.columns:
            return data
        data = data.copy()
        data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
        return data[data["trade_date"] <= pd.Timestamp(as_of_date)]
