"""
数据清洗 — 缺失值、异常值、停牌、ST、复权、时间对齐
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

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
        # 列名标准化 (tushare / akshare 字段不同)
        df = DataCleaner._standardize_columns(df)
        # 时间索引
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values("trade_date")
        # 缺失值
        df = df.ffill().bfill()
        # 异常值
        if "pct_chg" in df.columns:
            df = df[df["pct_chg"].abs() <= 100]  # 过滤极端涨跌幅
        return df

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名 (tushare -> 统一格式, akshare -> 统一格式)"""
        rename_map = {
            # tushare -> 标准
            "ts_code": "code",
            "trade_date": "trade_date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "pre_close": "pre_close",
            "change": "change",
            "pct_chg": "pct_chg",
            "vol": "volume",
            "amount": "amount",
            # akshare -> 标准
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "涨跌幅": "pct_chg",
            "涨跌额": "change",
            "成交量": "volume",
            "成交额": "amount",
            "日期": "trade_date",
        }
        # 只重命名存在的列
        existing = {k: v for k, v in rename_map.items() if k in df.columns}
        return df.rename(columns=existing)

    @staticmethod
    def detect_limit_up_down(df: pd.DataFrame) -> pd.DataFrame:
        """检测涨跌停状态 (A股: 普通股票 ±10%, ST ±5%, 科创/创业板 ±20%)"""
        if "close" not in df.columns or "pre_close" not in df.columns:
            return df
        df["pct_chg"] = (df["close"] - df["pre_close"]) / df["pre_close"] * 100
        df["is_limit_up"] = False
        df["is_limit_down"] = False
        for idx, row in df.iterrows():
            pct = row.get("pct_chg", 0)
            code = str(row.get("code", ""))
            if "ST" in code or "*ST" in code:
                limit = 5.0
            elif code.startswith("68") or code.startswith("30"):  # 科创/创业板
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
        return data[data["trade_date"] <= pd.Timestamp(as_of_date)]
