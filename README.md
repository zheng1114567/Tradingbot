# 基础测试 — 验证核心模块

## 安装

```bash
cd advanced_trading_agent
pip install -e .
```

## 配置

复制 `.env.example` 为 `.env`, 填入:
- LLM API Key (DeepSeek)
- 如只测试 DataAgent，不需要行情 API Key；默认走 AkShare/BaoStock/Sina/Eastmoney 这些 A 股免费数据源

### DataAgent 需要接入的 API

- `akshare`: 默认免费主数据源，无需 API Key。主要用于 A 股行情、新闻、板块、涨停梯队和部分资金流数据。
  - 文档: `https://akshare.akfamily.xyz/`
- `eastmoney`: 免费东方财富板块榜单兜底，无需 API Key。主要用于 `sector_context`。
- `sina`: 免费新浪个股新闻兜底，无需 API Key。主要用于 AkShare 新闻为空时补 `news.events`。
- `baostock`: 免费 A 股历史行情和停牌状态兜底，无需 API Key。
  - 文档: `http://baostock.com/baostock/index.php/Python_API%E6%96%87%E6%A1%A3`
- `DEEPSEEK_API_KEY`: 完整多 Agent 分析默认 LLM key；单独测试 DataAgent 不需要。
  - API Key: `https://platform.deepseek.com/api_keys`
  - 文档: `https://api-docs.deepseek.com/`
- 可选 LLM 供应商:
  - `OPENAI_API_KEY`: `https://platform.openai.com/api-keys`
  - `ANTHROPIC_API_KEY`: `https://console.anthropic.com/settings/keys`

### 单独测试 DataAgent

```bash
dataagent --ticker 000001.SZ --date 2026-07-10 --start-date 20260101 --end-date 20260710 --output-dir ./data/results
```

也可以直接用模块方式运行:

```bash
python -m advanced_trading_agent.data_agent.cli --ticker 000001.SZ --date 2026-07-10 --start-date 20260101 --end-date 20260710 --output-dir ./data/results
```

如需先做 ReAct-style 数据规划，再执行确定性数据流水线:

```bash
dataagent --react-planner --ticker 000001.SZ --date 2026-07-10 --start-date 20260101 --output-dir ./data/results
```

也可以给新闻采集加关键词过滤:

```bash
dataagent --react-planner --ticker 000001.SZ --date 2026-07-10 --start-date 20260101 --output-dir ./data/results --news-keyword 平安银行
```

默认还会采集市场板块榜单，生成 `sector_context` 给 Market / Analysis / System Agent 使用。如果你已经知道标的所属板块，可以显式传入板块关键词，帮助 DataAgent 匹配:

```bash
dataagent --ticker 000001.SZ --date 2026-07-10 --sector-keyword 银行 --output-dir ./data/results
```

如果只想测单标的行情，不需要板块上下文:

```bash
dataagent --ticker 000001.SZ --date 2026-07-10 --no-sector-context
```

默认会使用 LLM 对新闻相关性、方向、置信度做筛选；如果没有 LLM Key 或调用失败，会自动降级为规则筛选并在 `analysis.events.filter` 里留痕。调试时也可以关闭 LLM 新闻筛选:

```bash
dataagent --ticker 000001.SZ --date 2026-07-10 --no-llm-news-filter
```

默认还会尝试访问新闻 URL 抽取正文，把正文片段保存为 `evidence_text`，供后续 Event / Analysis / System Agent 直接读取。如果正文抓取失败，会保留摘要并标记 `content_status=summary_only`；调试时也可以关闭正文抓取:

```bash
dataagent --ticker 000001.SZ --date 2026-07-10 --no-news-full-text
```

DataAgent 会按步骤留痕并分层保存:

```text
data/results/data_agent_runs/<date>_<ticker>_<timestamp>/
├── 00_planner/plan.json           # ReAct Planner 计划和 trace；仅 --react-planner 开启
├── 01_input/request.json          # 输入参数、供应商链
├── 02_raw/raw_data.json           # 原始供应商数据，新闻含 full_text/content_status/evidence_text
├── 03_cleaned/cleaned_data.json   # 标准化、清洗后的数据
├── 04_analysis/analysis_data.json # 因子、摘要和分析数据
├── 04_analysis/news_events.json   # 新闻筛选、LLM 判断、事件化结果的独立留痕
├── 05_agent_payload/agent_payload.json # 给后续 Agent 消费的 Tier 1 / Tier 2 数据
└── 06_final/response.json         # 汇总返回: input/raw/cleaned/analysis/agent_payload/manifest
```

Agent 推荐读取 `05_agent_payload/agent_payload.json`:

```text
tier1_data.market / sentiment / sector / capital / risk
tier2_data.price_data / factors / events / sector_context / data_quality
tier2_data.events[*].evidence_text / content_status / llm_reason
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
│   ├── collector.py       # 数据采集 (akshare/baostock/sina/eastmoney)
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
