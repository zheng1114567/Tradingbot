"""
Token 用量追踪 — 控制每次运行的成本

目标: 每次运行 ≤ ¥5

借鉴 TradingAgents: TradingAgents 不做 token 追踪,
但用户要求省 token 省钱, 所以需要这个模块。

预估成本:
- DeepSeek: ¥1 / 1M input tokens
- 每次 Agent 调用: ~500 input + ~200 output tokens
- 完整运行(4 Agents + 1 System): ~3500 tokens ≈ ¥0.0035
- Round 2(8轮): ~8000 tokens ≈ ¥0.008
- 总计: ~¥0.01-0.02/次 (远低于 ¥5 上限)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CostRecord:
    """单次 LLM 调用的成本记录"""
    agent: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    duration_sec: float = 0
    cost_cny: float = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class CostTracker:
    """Token 成本追踪器

    价格 (DeepSeek):
    - input: ¥1 / 1M tokens
    - output: ¥2 / 1M tokens
    """

    PRICES = {
        "deepseek-chat": {"input_per_m": 1.0, "output_per_m": 2.0},
        "deepseek-reasoner": {"input_per_m": 4.0, "output_per_m": 16.0},
        "gpt-4o": {"input_per_m": 15.0, "output_per_m": 60.0},
    }

    def __init__(self, model: str = "deepseek-chat"):
        self.model = model
        self.records: list[CostRecord] = []
        self.warning_threshold = 5.0  # ¥5 警告

    def record(self, agent: str, input_tokens: int, output_tokens: int,
               duration_sec: float = 0) -> CostRecord:
        """记录一次 LLM 调用"""
        prices = self.PRICES.get(self.model, {"input_per_m": 1.0, "output_per_m": 2.0})
        cost = (input_tokens / 1_000_000 * prices["input_per_m"]
                + output_tokens / 1_000_000 * prices["output_per_m"])

        record = CostRecord(
            agent=agent,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_sec=duration_sec,
            cost_cny=cost,
        )
        self.records.append(record)

        total = self.total_cost
        if total > self.warning_threshold:
            logger.warning("Cost warning: ¥%.4f (threshold: ¥%.2f)", total, self.warning_threshold)

        return record

    @property
    def total_cost(self) -> float:
        return sum(r.cost_cny for r in self.records)

    @property
    def total_tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self.records)

    def summary(self) -> str:
        """成本摘要"""
        if not self.records:
            return "No LLM calls recorded."

        lines = [
            "## Token 成本报告",
            f"总调用: {len(self.records)} 次",
            f"总 Token: {self.total_tokens:,}",
            f"总成本: ¥{self.total_cost:.4f}",
            "",
            "### 每次调用:",
        ]
        for r in self.records:
            lines.append(
                f"  {r.agent}: {r.input_tokens}in + {r.output_tokens}out "
                f"= ¥{r.cost_cny:.6f} ({r.duration_sec:.1f}s)"
            )
        return "\n".join(lines)
