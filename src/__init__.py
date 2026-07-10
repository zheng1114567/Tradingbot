"""
advanced_trading_agent — 多智能体量化交易分析系统

基于 TradingAgents (TauricResearch) 的 LangGraph 架构，
适配 A 股市场，聚焦盘后观察池生成。

关键设计决策:
- 数据源抽象层: 多供应商路由 + 降级 (借鉴 TradingAgents interface.py)
- Agent 输出结构化: Pydantic BaseModel (借鉴 TradingAgents schemas.py)
- 延迟反思: 先存 pending, 下次运行时拉取真实收益再反思 (借鉴 TradingAgents memory.py)
- 硬风控独立: 代码执行, LLM 不可覆盖
- Point-in-time 数据: 回测和实盘走同一套字段定义
"""

__version__ = "0.1.0"
