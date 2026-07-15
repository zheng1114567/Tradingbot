"""A-share specialist signal builder.

DataAgent layer: rule-based scoring only, no LLM inference.
Outputs auditable structured signals consumed by Roundtable conditional participants.

Current implementation (Phase 0.5):
  - hot_money: fully implemented
  - policy / unlock / multifactor: placeholders returning data_status=missing
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# HotMoney signal thresholds (configurable at module level for now)
_HOT_MONEY_THRESHOLDS = {
    "board_high_risk": 3,
    "board_speculative": 2,
    "dragon_tiger_active_limit": 2,
    "market_limit_up_high": 30,
    "market_limit_up_extreme": 50,
}


class AShareSignalBuilder:
    """Build structured A-share specialist signals from tier2_data.

    Usage:
        signals = AShareSignalBuilder.build(tier2_data)
        tier2_data["a_share_signals"] = signals
    """

    # ── Public API ──────────────────────────────────────────────

    @classmethod
    def build(cls, tier2: dict[str, Any]) -> dict[str, Any]:
        """Build all A-share signals from tier2 data."""
        signals: dict[str, Any] = {}

        hot_money = cls._build_hot_money(tier2)
        signals["hot_money"] = hot_money if hot_money is not None else _insufficient("hot_money", "游资数据源未接入")

        signals["policy"] = cls._build_policy(tier2)
        signals["unlock"] = cls._build_unlock(tier2)
        signals["multifactor"] = cls._build_multifactor(tier2)

        return signals

    # ── HotMoney ────────────────────────────────────────────────

    @classmethod
    def _build_hot_money(cls, tier2: dict[str, Any]) -> dict[str, Any] | None:
        """Assess speculative-capital / short-term sentiment signals.

        Reads from:
          - tier2_data.limit_up_summary
          - tier2_data.dragon_tiger
          - tier2_data.data_summary (for the target ticker)

        Returns None when the data structure is absent entirely
        (no limit_up or dragon_tiger keys in tier2).
        """
        limit_up = tier2.get("limit_up_summary", None)
        dragon_tiger = tier2.get("dragon_tiger", None)

        # If neither data source exists, this builder cannot run
        if limit_up is None and dragon_tiger is None:
            return None

        limit_up = limit_up if isinstance(limit_up, dict) else {}
        dragon_tiger_list = (
            list(dragon_tiger) if isinstance(dragon_tiger, list) else []
        )

        # ── Extract raw data ──
        first_board = int(limit_up.get("first_board", 0) or 0)
        second_board = int(limit_up.get("second_board", 0) or 0)
        third_plus = int(limit_up.get("third_plus", 0) or 0)
        limit_up_stocks = limit_up.get("stocks", []) or []
        total_limit_up = first_board + second_board + third_plus

        dt_active = (
            len(dragon_tiger_list)
            >= _HOT_MONEY_THRESHOLDS["dragon_tiger_active_limit"]
        )

        # ── Target ticker board count ──
        ticker = (
            tier2.get("data_summary", {}).get("ticker", "")
        )
        target_board_count = 0
        for s in limit_up_stocks:
            if isinstance(s, dict) and s.get("code") == ticker:
                target_board_count = int(s.get("board_count", 0) or 0)
                break

        # ── Evidence trail ──
        evidence: list[str] = [
            f"limit_up_summary: first_board={first_board}, second_board={second_board}, "
            f"third_plus={third_plus}, total={total_limit_up}",
            f"dragon_tiger: records={len(dragon_tiger_list)}, active={dt_active}",
        ]
        if target_board_count:
            evidence.append(f"target_board_count={target_board_count}")

        # ── Signal determination ──
        warnings: list[str] = []
        signal: str = "absent"
        score: float = 0.0

        if not limit_up_stocks and not dragon_tiger_list:
            # Both sources present but empty → no activity detected
            if tier2.get("limit_up_summary") is not None and tier2.get("dragon_tiger") is not None:
                signal = "absent"
                score = 10.0
                data_status = "available"
            else:
                signal = "insufficient"
                data_status = "missing"
        else:
            data_status = "available"

            # 1) Target ticker board-chain analysis
            if target_board_count >= _HOT_MONEY_THRESHOLDS["board_high_risk"]:
                signal = "overheated"
                score = 80.0 + min(target_board_count * 5.0, 20.0)
                warnings.append(
                    f"标的 {target_board_count} 连板, 短线兑现风险上升"
                )
                if dt_active:
                    warnings.append("龙虎榜活跃，游资博弈激烈")
            elif target_board_count >= _HOT_MONEY_THRESHOLDS["board_speculative"]:
                signal = "speculative"
                score = 60.0
                warnings.append("标的 2 连板，关注分歧转一致机会")
            elif target_board_count >= 1:
                signal = "confirmed"
                score = 40.0

            # 2) Dragon-tiger net-buy boost
            dt_net_buy = cls._dragon_tiger_net_buy(dragon_tiger_list)
            if dt_net_buy is not None and dt_net_buy > 0:
                if signal in ("absent",):
                    signal = "confirmed"
                score = max(score, 50.0)
                evidence.append(f"dragon_tiger_net_buy={dt_net_buy:.0f}")

        # 3) Broader market overheat — runs regardless of individual ticker data
        if total_limit_up > _HOT_MONEY_THRESHOLDS["market_limit_up_extreme"]:
            warnings.append("全市场涨停 > 50，警惕情绪过热回落")
            if signal == "absent":
                signal = "speculative"
                score = max(score, 35.0)
        elif total_limit_up > _HOT_MONEY_THRESHOLDS["market_limit_up_high"]:
            if signal == "absent":
                warnings.append("全市场涨停家数偏多，短线情绪亢奋")

        return {
            "signal": signal,
            "score": round(score, 1),
            "limit_up_count": total_limit_up,
            "board_count": target_board_count,
            "dragon_tiger_active": dt_active,
            "warnings": warnings,
            "evidence": evidence,
            "data_status": data_status,
        }

    @staticmethod
    def _dragon_tiger_net_buy(
        records: list[dict[str, Any]],
    ) -> float | None:
        """Sum net_buy across dragon-tiger records, or None if absent."""
        total = 0.0
        found = False
        for r in records:
            net = r.get("net_buy")
            if net is not None:
                try:
                    total += float(net)
                    found = True
                except (TypeError, ValueError):
                    continue
        return total if found else None

    # ── Policy ────────────────────────────────────────────────────

    _POLICY_KEYWORDS = [
        # 国务院/中央级别
        "国务院", "发改委", "央行", "人民银行", "证监会", "银保监", "财政部",
        "国常会", "政治局", "中央经济工作",
        # 行业政策
        "集采", "带量采购", "医保目录", "医保谈判", "DRG", "两票制",
        "补贴", "税收优惠", "减税", "出口退税",
        # 监管
        "反垄断", "立案调查", "问询函", "监管函", "警示函", "停牌核查",
        "立案", "处罚", "整顿",
        # 利好信号
        "重大利好", "产业规划", "十四五", "示范项目", "试点",
        "国产替代", "自主可控", "新基建",
    ]

    @classmethod
    def _build_policy(cls, tier2: dict[str, Any]) -> dict[str, Any]:
        """Scan news events for policy-related keywords."""
        events = tier2.get("events", [])
        if not isinstance(events, list):
            events = []

        matched: list[dict[str, Any]] = []
        for ev in events:
            if not isinstance(ev, dict):
                continue
            text = f"{ev.get('summary', '')} {ev.get('evidence_text', '')}"
            for kw in cls._POLICY_KEYWORDS:
                if kw in text:
                    matched.append({
                        "keyword": kw,
                        "summary": str(ev.get("summary", ""))[:120],
                        "direction": ev.get("direction", "neutral"),
                    })
                    break

        if not matched:
            return {
                "signal": "absent",
                "strength": 0.0,
                "policy_level": "none",
                "matched_events": [],
                "evidence": ["未发现明确的政策事件信号"],
                "data_status": "available",
            }

        # Score: each match adds 20, capped at 80
        strength = min(len(matched) * 20.0, 80.0)
        level = "high" if strength >= 60 else "medium" if strength >= 30 else "low"

        return {
            "signal": "confirmed" if strength >= 30 else "weak",
            "strength": strength,
            "policy_level": level,
            "matched_events": matched[:5],
            "evidence": [f"匹配 {len(matched)} 条政策关键词: {', '.join(m['keyword'] for m in matched[:5])}"],
            "data_status": "available",
        }

    # ── Unlock (限售解禁) ──────────────────────────────────────────

    @classmethod
    def _build_unlock(cls, tier2: dict[str, Any]) -> dict[str, Any]:
        """Check for upcoming lock-up expiry events (requires akshare)."""
        # baostock has no share_unlock function; akshare is required for this signal
        return {
            "signal": "insufficient",
            "risk_level": "unavailable",
            "unlock_date": None,
            "unlock_ratio_float": None,
            "warnings": ["baostock 无解禁数据接口，需 akshare stock_restricted_release_detail_em"],
            "evidence": [],
            "data_status": "missing",
        }

    # ── Multifactor ───────────────────────────────────────────────

    @classmethod
    def _build_multifactor(cls, tier2: dict[str, Any]) -> dict[str, Any]:
        """Build composite factor signal from computed factors."""
        factors_list = tier2.get("factors", [])
        if not isinstance(factors_list, list) or not factors_list:
            return _insufficient("multifactor", "无因子数据")

        latest = factors_list[-1] if factors_list else {}
        if not isinstance(latest, dict):
            return _insufficient("multifactor", "因子数据格式异常")

        momentum = latest.get("momentum_20d")
        ma_trend = latest.get("ma_trend")
        volatility = latest.get("volatility")
        composite = latest.get("composite_score")

        top_factors: list[dict[str, Any]] = []
        evidence: list[str] = []

        if isinstance(momentum, (int, float)):
            top_factors.append({"factor": "momentum_20d", "value": round(momentum, 4), "direction": "bullish" if momentum > 0 else "bearish"})
            evidence.append(f"momentum_20d={momentum:.3f}")
        if isinstance(ma_trend, (int, float)):
            top_factors.append({"factor": "ma_trend", "value": round(ma_trend, 4), "direction": "bullish" if ma_trend > 0 else "bearish"})
            evidence.append(f"ma_trend={ma_trend:.3f}")
        if isinstance(volatility, (int, float)):
            top_factors.append({"factor": "volatility", "value": round(volatility, 4), "direction": "neutral"})

        if not top_factors:
            return _insufficient("multifactor", "因子值为空")

        # Score from composite or average of available factors
        if isinstance(composite, (int, float)):
            score = round(max(0, min(100, (composite + 0.5) * 100)), 1)
        else:
            bull = sum(1 for f in top_factors if f["direction"] == "bullish")
            score = round(bull / max(len(top_factors), 1) * 60 + 20, 1)

        signal = "bullish" if score >= 60 else "bearish" if score <= 30 else "neutral"

        return {
            "signal": signal,
            "score": score,
            "top_factors": top_factors,
            "crowding_warnings": [],
            "evidence": evidence,
            "data_status": "available",
        }


# ── Helpers ────────────────────────────────────────────────────

def _insufficient(category: str, reason: str) -> dict[str, Any]:
    return {
        "signal": "insufficient",
        "score": 0.0,
        "warnings": [reason],
        "evidence": [],
        "data_status": "missing",
    }
