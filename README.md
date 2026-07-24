# A 股板块 ETF 智能分析系统

<div align="center">

**多智能体量化交易分析系统 — 多因子板块轮动 + AutoGen 圆桌决策 + LangGraph 工作流**

</div>

---

## 概览

基于 LangGraph 多智能体框架的 A 股板块轮动分析系统。先选板块，再买 ETF，个股只作为板块宽度、热度和事件传导的证据来源。

```text
板块排名 / 新闻事件 / 热点成分股 / ETF 流动性
        ↓
  SectorETFSelector  →  AutoGen 圆桌会议  →  观察池报告
        ↓
  TradingSystem Workflow (LangGraph)
   ├─ Market / Event / Analysis / Backtest Agent
   ├─ 三层硬风控 + 软风控
   └─ 最终裁定
```

---

## 核心功能

| 模块 | 说明 |
|------|------|
| **板块轮动** | 多因子评分排序，自动映射可交易 ETF |
| **AutoGen 圆桌** | 多智能体辩论决策，Market/Event/Analysis/Risk 四角色 + Moderator |
| **LangGraph 工作流** | 7 个 LLM Agent 顺序执行 + 矛盾检测，最终 System Agent 裁定 |
| **三层硬风控** | ST/停牌/退市 → 流动性/涨跌停 → 冲击成本/仓位，代码执行不可覆盖 |
| **回测引擎** | A 股专属约束 (T+1、涨跌停、停牌、成本模型) |
| **复盘记忆** | 延迟反思机制，历史命中率评估，策略校准 |
| **多数据源** | 本地缓存优先，免费数据源兜底 (akshare/baostock/东方财富/财联社/新浪) |

---

## 快速开始

### 安装

```bash
git clone <your-repo-url>
cd Tradingbot
pip install -e .
pip install -e ".[data,anthropic]"  # 数据 + LLM 依赖
```

### 配置

```bash
cp .env.example .env
# 编辑 .env，填入 LLM API Key（至少一个）
```

支持多种 LLM 供应商：DeepSeek、Qwen、OpenAI、Anthropic、Kimi、GLM

### 运行

```bash
# 生成每日板块 ETF 观察池（默认输出 Markdown）
python -m advanced_trading_agent.main --date 2026-07-15

# JSON 输出（便于程序消费）
python -m advanced_trading_agent.main --sector-etf-analyze --date 2026-07-15 --json

# 扫描全市场板块
python -m advanced_trading_agent.main --sector-etf-scan --date 2026-07-15 --scan-top-n 8

# 分析指定板块
python -m advanced_trading_agent.main --sector-etf-analyze --sector 半导体 --date 2026-07-15

# 自然语言问答
python -m advanced_trading_agent.main --ask "半导体板块为什么不好" --date 2026-07-15
```

---

## 架构

### 工作流拓扑

```
START
  → Risk Check 1      [代码]  ST/停牌/退市 → HARD_VETO → END
  → System Init        [代码]  数据质量检查
  → Memory Agent       [LLM]  历史记忆召回
  → Market Agent       [LLM]  市场情绪/资金面
    → 冰点 → 跳过深度分析
    → 正常 → Event[LLM] → Analysis[LLM] → Backtest[LLM]
  → Risk Check 2       [代码]  流动性/涨跌停检查
  → 矛盾检测           [模式8+LLM]  检测 Agent 间矛盾
  → Risk Check 3       [代码]  冲击成本/仓位 → HARD_VETO → END
  → System Agent       [LLM]  综合裁定 (硬风控不可覆盖)
  → Approval → Report → END
```

### 数据流

```text
data/results/data_agent_runs/<date>_<ticker>_<timestamp>/
├── 01_input/request.json            # 输入参数
├── 02_raw/raw_data.json             # 原始供应商数据
├── 03_cleaned/cleaned_data.json     # 标准化清洗后数据
├── 04_analysis/analysis_data.json   # 因子计算/事件分析
├── 05_agent_payload/agent_payload.json  # Tier1/Tier2 → Agent 消费
└── 06_final/response.json           # 汇总返回
```

---

## 项目结构

```
src/advanced_trading_agent/
├── main.py                    入口 CLI
├── config.py                  全局配置 (env 覆盖)
├── pipeline.py                数据采集 + 工作流编排
├── graph/                     LangGraph 工作流
│   ├── workflow.py            核心工作流图
│   ├── sector_etf_workflow.py 板块ETF流水线
│   ├── state.py               AgentState 定义
│   ├── conditional.py         条件路由逻辑
│   └── risk_nodes.py          硬风控节点
├── agents/                    LLM Agent
│   ├── schemas.py             结构化输出 Schema
│   ├── system_agent.py        组长 + 裁定
│   ├── market_agent.py        市场温度分析
│   ├── event_agent.py         事件驱动分析
│   ├── analysis_agent.py      因子分析
│   ├── backtest_agent.py      回测审查
│   ├── memory_agent.py        经验库复盘
│   ├── report_agent.py        报告生成
│   └── approval_agent.py      审批记录
├── roundtable/
│   ├── etf_watchlist_autogen.py  AutoGen 圆桌
│   ├── contradiction_detector.py  矛盾检测
│   └── sector_qa.py               板块问答
├── risk/
│   ├── hard_risk.py           硬风控 (代码执行)
│   └── soft_risk.py           软风控 (规则+LLM)
├── backtest/
│   ├── engine.py              回测引擎 (A股约束)
│   ├── portfolio.py           组合回测
│   ├── review.py              复盘引擎
│   ├── scheduler.py           定时复盘
│   └── sector_backtest.py     板块回测
├── data_agent/                数据采集/处理
│   ├── data_agent.py          DataAgent 主入口
│   ├── scanner.py             扫描器
│   ├── collector.py           供应商适配器
│   ├── cleaner.py             数据清洗
│   ├── factors.py             因子计算
│   ├── sector_etf.py          板块ETF评分
│   ├── etf_watchlist.py       观察池规则
│   └── ...                    供应商路由/缓存/过滤
├── llm/
│   ├── client.py              LLM 客户端
│   └── provider_registry.py   供应商注册
└── tool_nodes/                工具节点
    ├── analysis_tools.py
    ├── market_tools.py
    ├── event_tools.py
    └── backtest_tools.py
```

---

## 风控体系

| 层次 | 检查点 | 触发条件 | 后果 |
|------|--------|----------|------|
| 硬风控 1 | 分析前 | ST/停牌/退市 | HARD_VETO → 终止 |
| 硬风控 2 | Round 1 后 | 日成交额 < 1000万 / 涨跌停 | SOFT_VETO → 带入讨论 |
| 硬风控 3 | 裁定前 | 冲击成本 > 30% 收益 / 超仓位 | HARD_VETO → 不可覆盖 |
| 软风控 | 全程 | 事件半衰期/止损/止盈/熔断 | LLM 可辩论 |

---

## 配置参考

关键环境变量：

| 变量 | 说明 | 默认 |
|------|------|------|
| `ATA_LLM_PROVIDER` | LLM 供应商 | `qwen` |
| `ATA_LLM_MODEL` | 模型名称 | `qwen3.6-flash` |
| `DASHSCOPE_API_KEY` | 通义千问 Key | — |
| `DEEPSEEK_API_KEY` | DeepSeek Key | — |
| `ATA_DATA_CACHE_DIR` | 数据缓存目录 | `./data/cache` |
| `ATA_RESULTS_DIR` | 结果输出目录 | `./data/results` |

---

## 借鉴来源

本项目的 LangGraph 工作流架构和 Agent 模式借鉴了：

- **[TradingAgents](https://github.com/TauricResearch/TradingAgents)** (TauricResearch) — ICML 2025, LangGraph 多 Agent 框架
- **TradingAgents-CN-AI** (zhouke2020) — A 股适配版

具体借鉴点：多供应商数据路由、Pydantic 结构化输出、延迟反思 Memory、图状态定义、风控辩论模式。
