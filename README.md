# 基础测试 — 验证核心模块

## 安装

```bash
cd advanced_trading_agent
pip install -e .
```

## 配置

复制 `.env.example` 为 `.env`, 填入:
- LLM API Key (DeepSeek)
- tushare token

### DataAgent 需要接入的 API

- `TUSHARE_TOKEN`: 推荐配置。用于 A 股日 K、资金流、财务、ST/停牌、北向资金、龙虎榜、融资融券等数据。
  - 注册/Token: `https://tushare.pro/register`
  - 文档: `https://tushare.pro/document/1?doc_id=40`
- `akshare`: 无需 token，但需要安装 `pip install -e ".[akshare]"` 或 `pip install akshare`。主要作为行情、新闻、板块、涨停梯队的降级数据源。
  - 文档: `https://akshare.akfamily.xyz/`
- `DEEPSEEK_API_KEY`: 完整多 Agent 分析默认 LLM key；单独测试 DataAgent 不需要。
  - API Key: `https://platform.deepseek.com/api_keys`
  - 文档: `https://api-docs.deepseek.com/`
- 可选 LLM 供应商:
  - `OPENAI_API_KEY`: `https://platform.openai.com/api-keys`
  - `ANTHROPIC_API_KEY`: `https://console.anthropic.com/settings/keys`

### 单独测试 DataAgent

```bash
python -m advanced_trading_agent.main --data-agent --ticker 000001.SZ --date 2026-07-10 --start-date 20260101 --end-date 20260710 --output-dir ./data/results
```

如需先做 ReAct-style 数据规划，再执行确定性数据流水线:

```bash
python -m advanced_trading_agent.main --data-agent --react-planner --ticker 000001.SZ --date 2026-07-10 --start-date 20260101 --output-dir ./data/results
```

DataAgent 会按步骤留痕并分层保存:

```text
data/results/data_agent_runs/<date>_<ticker>_<timestamp>/
├── 00_planner/plan.json           # ReAct Planner 计划和 trace；仅 --react-planner 开启
├── 01_input/request.json          # 输入参数、供应商链
├── 02_raw/raw_data.json           # 原始供应商数据
├── 03_cleaned/cleaned_data.json   # 标准化、清洗后的数据
├── 04_analysis/analysis_data.json # 因子、摘要和分析数据
└── 05_final/response.json         # 汇总返回: input/raw/cleaned/analysis/manifest
```

## 运行

```bash
# 单标的分
python -m advanced_trading_agent.main --ticker 000001.SZ

# 批量分析
python -m advanced_trading_agent.main --batch stocks.txt

# 调试模式
python -m advanced_trading_agent.main --ticker 000001.SZ --debug
```

## 项目结构

```
src/advanced_trading_agent/
├── config.py              # 配置管理 (env覆盖)
├── data_agent/            # DataAgent 数据层
│   ├── schema.py          # Tier 1/Tier 2 Schema (Pydantic)
│   ├── vendor_router.py   # 多供应商路由 + 降级
│   ├── collector.py       # 数据采集 (tushare/akshare)
│   ├── data_agent.py      # 独立 DataAgent: 输入→原始→清洗→分析→最终返回
│   ├── cleaner.py         # 数据清洗
│   └── factors.py         # 因子计算
├── agents/                # Agent 层
│   ├── schemas.py         # Agent 输出 Schema (Pydantic)
│   ├── market_agent.py    # 市场温度分析师
│   ├── event_agent.py     # 事件分析师
│   ├── analysis_agent.py  # 因子分析师
│   ├── backtest_agent.py  # 回验证审查员
│   ├── system_agent.py    # 系统组长 + 裁定
│   ├── memory_agent.py    # 经验库 + 复盘
│   └── report_agent.py    # 报告输出
├── graph/                 # LangGraph 工作流
│   ├── state.py           # Agent 状态定义
│   └── workflow.py        # 工作流图
├── risk/                  # 风控
│   ├── hard_risk.py       # 硬风控 (代码执行)
│   └── soft_risk.py       # 软风控 (规则+LLM)
├── backtest/              # 回测
│   ├── engine.py          # 回测引擎 (A股约束)
│   ├── metrics.py         # 绩效指标
│   └── comparison.py      # 对比实验
├── llm/
│   └── client.py          # LLM 客户端 (DeepSeek)
└── main.py                # 入口
```

## 借鉴来源

本项目的 LangGraph 工作流架构和 Agent 模式借鉴了:
- **TradingAgents** (TauricResearch) — ICML 2025, LangGraph 多 Agent 框架
- **TradingAgents-CN-AI** (zhouke2020) — A 股适配版

具体借鉴点:
1. 多供应商数据路由 (TradingAgents' interface.py)
2. Pydantic 结构化输出 (TradingAgents' schemas.py)
3. 延迟反思 Memory (TradingAgents' memory.py + reflection.py)
4. 图状态的定义方式 (TradingAgents' agent_states.py)
5. 风控辩论模式 (TradingAgents' risk_mgmt)
