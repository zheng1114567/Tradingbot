# 基础测试 — 验证核心模块

## 安装

```bash
cd advanced_trading_agent
pip install -e .
```

## 配置

复制 `.env.example` 为 `.env`, 填入:
- LLM API Key (DeepSeek)
- 如只测试 Scan/DataAgent，不需要行情 API Key；默认走本地缓存 + mootdx/BaoStock/东方财富/财联社/新浪这些 A 股免费数据源

### Scan 需要接入的 API

- `local_cache`: 默认第一数据源。每日通过 `build-cache` / `--refresh-cache` 从免费直连源刷新，再供 Scan/DataAgent 离线优先读取。
- `mootdx`: 通达信 TCP 日线行情，无需 API Key，优先用于补日线缓存。
- `cls`: 财联社快讯，无需 API Key。新闻预缓存优先走这个快接口，再降级新浪个股新闻页。
- `eastmoney`: 免费东方财富板块榜单兜底，无需 API Key。主要用于 `sector_context`。
- `sina`: 免费新浪个股新闻兜底，无需 API Key。主要用于财联社快讯为空时补 `news.events`。
- `baostock`: 免费 A 股历史行情和停牌状态兜底，无需 API Key。
  - 文档: `http://baostock.com/baostock/index.php/Python_API%E6%96%87%E6%A1%A3`
- `DEEPSEEK_API_KEY`: 完整多 Agent 分析默认 LLM key；单独测试 Scan/DataAgent 数据流水线不需要。
  - API Key: `https://platform.deepseek.com/api_keys`
  - 文档: `https://api-docs.deepseek.com/`
- 可选 LLM 供应商:
  - `OPENAI_API_KEY`: `https://platform.openai.com/api-keys`
  - `ANTHROPIC_API_KEY`: `https://console.anthropic.com/settings/keys`

### 单独测试 DataAgent

建议把行情缓存和新闻缓存分开刷新。默认 `scan` 只读取已有本地行情缓存，缺数据时跳过对应信号，不因新闻缺失而阻塞；使用 `--refresh-scan-cache` 可在扫描前补行情，使用 `--refresh-cache` 可单独批量预缓存新闻。`--force-news` 会重拉当天已有新闻文件。

如需手动预热缓存:

```bash
build-cache --date 2026-07-10 --force-news
# 或通过主入口只刷新缓存
python -m advanced_trading_agent.main --refresh-cache --date 2026-07-10 --force-news
```

### 板块 → ETF 策略主线

当前策略改造方向是“先选板块，再买相关 ETF”。个股不再作为最终买入对象，只作为板块宽度、热度和事件传导的证据来源。核心 seam 是 `SectorETFSelector`:

```text
板块排名 / 新闻事件 / 热点成分股宽度 / ETF 现货流动性
        ↓
SectorETFSelector
        ↓
SectorCandidate + ETFCandidate
        ↓
LangGraph / AutoGen 圆桌 / 对话问答
```

手动刷新 ETF 缓存:

```bash
python -m advanced_trading_agent.main --refresh-etf-cache --date 2026-07-15
```

扫描全市场板块并映射 ETF:

```bash
python -m advanced_trading_agent.main --sector-etf-scan --date 2026-07-15 --scan-top-n 8
```

默认 ETF 扫描/观察池路径只读取已保存缓存，适合盘后快速圆桌；如果需要先补行情缓存，再显式加 `--refresh-scan-cache`:

```bash
python -m advanced_trading_agent.main --sector-etf-analyze --date 2026-07-15 --json --refresh-scan-cache
```

只分析一个指定板块:

```bash
python -m advanced_trading_agent.main --sector-etf-scan --sector 半导体 --date 2026-07-15
```

生成每日板块 ETF 观察池（最多 8 个板块进入 JSON 圆桌；最终报告只保留 Top 3 个板块决策，每个板块落到 1 个首选 ETF + 最多 2 个备选 ETF）:

```bash
python -m advanced_trading_agent.main --sector-etf-analyze --date 2026-07-15
```

输出 JSON-first 报告，便于复盘、审批和后续 paper trading:

```bash
python -m advanced_trading_agent.main --sector-etf-analyze --date 2026-07-15 --json
```

只分析一个指定板块的对话问答路径仍保留:

```bash
python -m advanced_trading_agent.main --sector-etf-analyze --sector 半导体 --date 2026-07-15 --no-autogen --no-conversation-memory
```

对话问答会优先把“xx 板块为什么不好/能不能买”路由到同一条板块 ETF LangGraph 流水线。圆桌优先使用 AutoGen；如果 AutoGen 依赖或 LLM Key 不可用，会返回确定性 fallback，并明确说明是 fallback。对话记忆只保存问答上下文，不参与回测复盘:

```bash
python -m advanced_trading_agent.main --ask "半导体板块为什么不好" --date 2026-07-15
```

问答也支持同样的调试开关:

```bash
python -m advanced_trading_agent.main --ask "半导体板块为什么不好" --date 2026-07-15 --no-autogen --no-conversation-memory
```

消息队列/中间件暂不实现到运行时，但设计 seam 已明确：后续只需要把 LangGraph 节点之间的状态转移事件发布出去，不改变节点内部业务逻辑。

```text
SectorETFTradingSystem
  Select Sector ETF
    -> publish sector.selected
  Recall Conversation Memory
    -> publish memory.recalled
  AutoGen Roundtable
    -> publish roundtable.completed
  Store Conversation Memory
    -> publish memory.stored
  Report
    -> publish report.created
```

推荐先用本地进程内 event bus 做开发版中间件，接口保持 `publish(topic, payload)` / `subscribe(topic, handler)`；如果后面需要多进程或任务队列，再把 adapter 换成 Redis Streams、RabbitMQ 或 Kafka。核心约束是：消息队列只承载观测、异步通知和后续扩展，不参与交易裁定，裁定仍由 LangGraph state 和报告留痕负责。

热点扫描默认只读 `local_cache`。新闻仍然有必要，但只作为候选增强信号：优先读取本地批量缓存，Top 候选无缓存时才按需走财联社/新浪兜底；正文抓取默认关闭，需要时再显式开启。

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

默认情况下，DataAgent 会先根据 `ticker` 自动解析股票画像，再把公司名传给 Scan 做新闻采集，把行业关键词用于板块匹配。比如 `000001.SZ` 会自动补成 `news_keyword=平安银行`、`sector_keyword=银行`，这些结果会写入 `01_input/request.json` 的 `stock_profile` 留痕。

如果你要覆盖自动识别结果，也可以显式给新闻采集加关键词过滤:

```bash
dataagent --react-planner --ticker 000001.SZ --date 2026-07-10 --start-date 20260101 --output-dir ./data/results --news-keyword 平安银行
```

默认还会采集市场板块榜单，生成 `sector_context` 给 Market / Analysis / System Agent 使用。如果你已经知道标的所属板块，也可以显式传入板块关键词，覆盖自动识别结果:

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

默认由 Scan 访问新闻 URL 抽取正文，去除广告/编辑/版权等噪声并去重段落，把清洗后的正文片段保存为 `evidence_text`，供后续 Event / Analysis / System Agent 直接读取。如果正文抓取失败，会保留摘要并标记 `content_status=summary_only`；正文清洗过程会记录在 `content_cleaning` 里。`MarketScanner.scan_and_collect()` 和 DataAgent 单独运行时都会通过 Scan 层完成正文抽取，并打上 `full_text_source=scanner` / `full_text_attempted=true`；DataAgent 只接收 Scan 产出的 raw/cleaned payload，不重复访问同一 URL。调试时也可以关闭正文抓取:

```bash
dataagent --ticker 000001.SZ --date 2026-07-10 --no-news-full-text
```

DataAgent 会按步骤留痕并分层保存。职责边界是：Scan 负责数据抓取和清洗，DataAgent 负责结构化处理和 Agent Payload，圆桌会议负责交易分析和辩论:

```text
data/results/data_agent_runs/<date>_<ticker>_<timestamp>/
├── 00_planner/plan.json           # ReAct Planner 计划和 trace；仅 --react-planner 开启
├── 01_input/request.json          # 输入参数、供应商链
├── 02_raw/raw_data.json           # Scan 抓取的原始供应商数据，新闻含 raw_full_text/full_text/content_cleaning/evidence_text
├── 03_cleaned/cleaned_data.json   # Scan 标准化、清洗后的数据
├── 04_analysis/analysis_data.json # DataAgent 结构化处理结果：因子、摘要、事件、Payload 输入
├── 04_analysis/news_events.json   # 新闻筛选、LLM 判断、事件化结果的独立留痕
├── 05_agent_payload/agent_payload.json # 给后续 Agent 消费的 Tier 1 / Tier 2 数据
└── 06_final/response.json         # 汇总返回: input/raw/cleaned/analysis/agent_payload/manifest
```

Agent 推荐读取 `05_agent_payload/agent_payload.json`:

```text
tier1_data.market / sentiment / sector / capital / risk
tier2_data.price_data / factors / events / sector_context / data_quality
tier2_data.events[*].evidence_text / content_status / content_cleaning / llm_reason
```

## 运行

```bash
# 默认：生成 A 股板块 ETF 观察池
python -m advanced_trading_agent.main --date 2026-07-15

# JSON-first 输出
python -m advanced_trading_agent.main --sector-etf-analyze --date 2026-07-15 --json

# 需要刷新扫描缓存时才打开；默认使用已保存缓存，避免圆桌耗时过长
python -m advanced_trading_agent.main --sector-etf-analyze --date 2026-07-15 --json --refresh-scan-cache

# 只刷新 ETF 数据缓存
python -m advanced_trading_agent.main --refresh-etf-cache --date 2026-07-15
```

旧 `--ticker` / `--batch` / `--scan` 个股买入入口已下线；个股数据仍作为板块宽度、事件传导和风险证据源保留。

## 项目结构

```
src/advanced_trading_agent/
├── config.py              # 配置管理 (env覆盖)
├── data_agent/            # DataAgent 数据层
│   ├── schema.py          # Tier 1/Tier 2 Schema (Pydantic)
│   ├── vendor_router.py   # 多供应商路由 + 降级
│   ├── scanner.py         # Scan: 扫描、数据抓取、正文抽取、清洗入口
│   ├── collector.py       # Scan 使用的数据采集适配器 (akshare/baostock/sina/eastmoney)
│   ├── data_agent.py      # DataAgent: 输入→接收 Scan raw/cleaned→结构化处理→最终返回
│   ├── cleaner.py         # Scan 使用的确定性数据清洗规则
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
