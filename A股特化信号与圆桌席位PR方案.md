# A 股特化信号与圆桌席位 PR 方案

## 一、PR 标题

`Add A-share specialist signals and conditional roundtable participants`

## 二、背景

当前项目已经具备：

- DataAgent：负责数据采集、清洗、因子计算、事件结构化和 Agent Payload 生成。
- Market / Event / Analysis / Backtest：负责第一轮分析。
- Roundtable：负责在观点冲突时进行圆桌辩论。
- System Agent：负责最终裁定。

TradingAgents-astock 的政策分析师、游资追踪师、解禁监控师值得借鉴，但不应直接照搬成常驻 workflow 节点。本项目已有更重的 DataAgent 层，适合先把 A 股特化信息沉淀为可审计的结构化信号，再由 Roundtable 按条件引入专门发言席位。

核心原则：

- DataAgent 负责证据和信号计算。
- Roundtable 负责冲突讨论和角色发言。
- System Agent 负责裁定。
- 不让 LLM 在缺数据时编造判断。

## 三、目标

本 PR 目标：

- 新增 `tier2_data.a_share_signals`。
- 支持政策、游资、解禁、多因子四类 A 股特化信号。
- Roundtable 支持条件参与者：
  - `Policy`
  - `HotMoney`
  - `Unlock`
- 默认圆桌仍保持四席：
  - `Market`
  - `Event`
  - `Analysis`
  - `Backtest`
- 不新增常驻 LangGraph workflow 节点。
- 不增加无证据 LLM 发言。

## 四、总体设计

推荐结构：

```text
DataAgent
  -> a_share_signals
      -> policy
      -> hot_money
      -> unlock
      -> multifactor

Roundtable
  -> default participants
      Market / Event / Analysis / Backtest
  -> conditional participants
      Policy / HotMoney / Unlock

System Agent
  -> reads tier2_data.a_share_signals
  -> reads Roundtable moderator output
  -> applies downgrade / veto rules
```

## 五、DataAgent 改动

新增模块：

```text
src/advanced_trading_agent/data_agent/a_share_signals.py
```

### 输出结构

```python
tier2_data["a_share_signals"] = {
    "policy": {
        "signal": "positive | neutral | negative | insufficient",
        "strength": 0.0,
        "policy_level": "national | ministry | local | unknown",
        "matched_events": [],
        "evidence": [],
        "data_status": "available | partial | missing",
    },
    "hot_money": {
        "signal": "confirmed | speculative | overheated | absent | insufficient",
        "score": 0.0,
        "limit_up_count": 0,
        "board_count": 0,
        "dragon_tiger_active": False,
        "warnings": [],
        "evidence": [],
        "data_status": "available | partial | missing",
    },
    "unlock": {
        "risk_level": "high | medium | low | unknown",
        "unlock_date": None,
        "unlock_ratio_float": None,
        "warnings": [],
        "evidence": [],
        "data_status": "available | missing",
    },
    "multifactor": {
        "signal": "strong | neutral | weak | insufficient",
        "score": 0.0,
        "top_factors": [],
        "crowding_warnings": [],
        "evidence": [],
        "data_status": "available | partial | missing",
    },
}
```

### Hook 位置

```text
DataAgent._analyze()
  -> daily summary
  -> factor records
  -> event records
  -> AShareSignalBuilder
  -> _build_agent_payload()
```

### DataAgent 职责边界

DataAgent 只做：

- 数据归集。
- 规则打标。
- 评分。
- 风险标记。
- 证据字段记录。

DataAgent 不做：

- 最终买卖判断。
- 仓位判断。
- LLM 主观解释。
- 无数据时的推断。

## 六、四类信号定义

### 1. Policy Signal

目的：识别政策事件是否具备交易含义。

输入：

- `tier2_data.events`
- 新闻标题、摘要、`evidence_text`
- 事件类型、方向、证据等级
- 公司名、板块关键词

判断维度：

- 政策级别：国家级、部委级、地方级、未知。
- 方向：利好、利空、中性。
- 传导强度：直接受益、间接受益、概念映射。
- 半衰期：短期情绪、中期产业、长期制度。
- 是否已定价。

#### LLM 辅助策略（可选增强）

纯规则打标在政策传导路径识别上存在局限。建议在 DataAgent 规则打标后，增加一次轻量 LLM 调用：

- **输入**：政策事件原文 + 规则初步分类结果 + 股票/板块上下文
- **LLM 职责**（严格限定）：
  - 归类政策传导路径：直接受益 / 间接受益 / 概念映射
  - 评估半衰期类型：短期情绪 / 中期产业 / 长期制度
  - 判断是否已被市场定价
- **Guardrail**：
  - LLM 必须引用 `evidence` 原文，不能编造数据
  - 上下文不足时输出 `insufficient`
  - LLM 不做买卖建议、不做仓位判断、不做无数据推断

输出示例：

```python
{
    "signal": "positive",
    "strength": 0.72,
    "policy_level": "ministry",
    "matched_events": ["event_0001"],
    "evidence": [
        "tier2_data.events[0].event_type=政策",
        "tier2_data.events[0].direction=利好"
    ],
    "data_status": "available"
}
```

### 2. Hot Money Signal

目的：识别游资和短线情绪资金是否形成有效驱动，或是否已经过热。

输入：

- `limit_up_summary`
- `dragon_tiger`
- `market_breadth`
- `short_term_signals`
- 成交额、涨停、连板、炸板信息

判断维度：

- 是否在涨停池。
- 连板高度。
- 是否上龙虎榜。
- 净买入强度。
- 是否高位接力。
- 是否一字板、缩量板、放量分歧板。

输出示例：

```python
{
    "signal": "overheated",
    "score": 83.0,
    "limit_up_count": 1,
    "board_count": 3,
    "dragon_tiger_active": True,
    "warnings": ["三连板后龙虎榜活跃，短线兑现风险上升"],
    "evidence": [
        "tier2_data.limit_up_summary.stocks",
        "tier2_data.dragon_tiger"
    ],
    "data_status": "available"
}
```

### 3. Unlock Risk

目的：识别限售股解禁、减持、供给冲击风险。

输入：

- 解禁日历。
- 公告。
- 风险数据。
- 未来 3-6 个月解禁计划。

第一版可以先做占位：

- 有稳定数据时输出风险。
- 无数据时明确 `data_status=missing`。
- 不编造解禁日期或比例。

输出示例：

```python
{
    "risk_level": "unknown",
    "unlock_date": None,
    "unlock_ratio_float": None,
    "warnings": ["解禁数据源未接入，本次未评估限售解禁风险"],
    "evidence": [],
    "data_status": "missing"
}
```

### 4. Multifactor Signal

目的：增强现有 Analysis Agent 的因子证据，而不是新增重复 Agent。

输入：

- `factor_records`
- 动量、波动、流动性、估值、成长、质量
- 因子拥挤度
- 行业上下文

判断维度：

- 综合因子评分。
- Top factor drivers。
- 因子是否和短线情绪共振。
- 是否存在拥挤或失效风险。

#### 与 Analysis Agent 的职责边界

现有 Analysis Agent 已有因子分析。DataAgent Multifactor 与其分工如下：

| 维度 | DataAgent Multifactor | Analysis Agent |
|------|----------------------|----------------|
| 方式 | 规则化打分，可审计 | LLM 综合语境解读 |
| 输出 | score + top_factors + crowding_warnings | 因子语境的自然语言分析 |
| 证据 | 明确引用 factor_records | 综合多源信息 |
| 目的 | 提供可复现的定量基础 | 提供更深度的语境解读 |

两套输出互为补充，不构成重复。如果后续实践中发现明显重叠，Multifactor 可作为独立模块从 DataAgent 剥离。

输出示例：

```python
{
    "signal": "strong",
    "score": 76.5,
    "top_factors": ["momentum", "liquidity", "quality"],
    "crowding_warnings": [],
    "evidence": [
        "tier2_data.factors[0].momentum",
        "tier2_data.factors[0].liquidity"
    ],
    "data_status": "available"
}
```

## 七、Roundtable 改动

修改模块：

```text
src/advanced_trading_agent/roundtable/harness.py
src/advanced_trading_agent/roundtable/schemas.py
src/advanced_trading_agent/agents/specs.py
```

### 默认参与者

```python
DEFAULT_PARTICIPANTS = ["Market", "Event", "Analysis", "Backtest"]
```

### 条件参与者

```python
SPECIALIST_PARTICIPANTS = {
    "Policy": {
        "focus": "政策级别、政策传导、政策半衰期、是否已定价",
    },
    "HotMoney": {
        "focus": "龙虎榜、涨停梯队、连板、高位接力、短线资金拥挤",
    },
    "Unlock": {
        "focus": "限售解禁、减持压力、供给冲击、是否需要降级或 veto",
    },
}
```

### 触发规则（可配置）

```python
# 阈值从配置读取，可在运行时调整，避免硬编码
TRIGGER_RULES = {
    "Policy": {
        "min_strength": 0.6,
        "required_signals": ["positive", "negative"],
        "require_data_status": ["available", "partial"],
    },
    "HotMoney": {
        "required_signals": ["confirmed", "speculative", "overheated"],
        "require_data_status": ["available", "partial"],
    },
    "Unlock": {
        "required_risk_levels": ["high", "medium"],
        "require_data_status": ["available"],
    },
}

if _meets_criteria(policy, TRIGGER_RULES["Policy"]):
    participants.append("Policy")

if _meets_criteria(hot_money, TRIGGER_RULES["HotMoney"]):
    participants.append("HotMoney")

if _meets_criteria(unlock, TRIGGER_RULES["Unlock"]):
    participants.append("Unlock")
```

### 条件加入后的顺序

```python
["Market", "Event", "Policy", "HotMoney", "Analysis", "Unlock", "Backtest"]
```

### 圆桌规则

- `Policy` 只讨论政策级别、传导路径、半衰期和是否已定价。
- `HotMoney` 只讨论龙虎榜、涨停、连板、短线筹码和情绪资金。
- `Unlock` 只提出供给冲击和降级/veto 风险，不提出买入理由。
- 所有 Specialist 必须引用 `tier2_data.a_share_signals.*`。
- 缺少对应数据时不能加入圆桌。
- Specialist 不替代默认四席，只在必要时补充。

### 冲突场景下的辩论流

当多个 Specialist 观点冲突时，发言顺序和辩论逻辑如下：

**发言顺序（含 Specialist）：**

```text
Market → Event → Policy → HotMoney → Analysis → Unlock → Backtest
```

每轮按此顺序发言，Moderator 在每轮结束后识别冲突点，在下一轮开头定向追问。

**冲突场景处理：**

| 冲突 | 处理方式 |
|------|---------|
| Policy 利好 + HotMoney 过热 | Moderator 追问："政策利好兑现是否充分？市场是否已提前定价？" |
| Unlock 高风险 + 其他席位强烈推荐 | Moderator 要求各席引用 unlock 数据回应，不能忽略供给冲击风险 |
| HotMoney confirmed + Multifactor weak | Moderator 追问"短线资金驱动能否改变基本面判断？" |
| Policy 利好 + Unlock 高解禁 | Moderator 要求讨论"利好是否足以抵消供给冲击" |

**辩论轮次：**

- 无 Specialist 加入：保持原有 N 轮。
- 有 1-2 个 Specialist 加入：保持 N 轮，在现有轮次中按序插入。
- 有 3 个 Specialist 加入：N+1 轮，确保每个 Specialist 至少有一次完整发言和被追问的机会。

## 八、System Agent 改动

修改模块：

```text
src/advanced_trading_agent/agents/system_agent.py
```

System Agent 读取：

```python
tier2_data["a_share_signals"]
round2_state["moderator_output"]
round2_state["unresolved_conflicts"]
```

### 裁定规则

- `hot_money.signal == overheated`：至少降一级，除非 Backtest 和 Market 都强支持。
- `unlock.risk_level == high`：强制进入观察或拒绝。
- `policy.signal == positive`：不能单独支撑推荐，必须同时有资金确认或因子支持。
- `multifactor.signal == weak` 且 `hot_money.signal == speculative`：偏观察或拒绝。
- `policy.signal == positive` 但 `hot_money.signal == overheated`：提示利好兑现风险。
- `unlock.data_status == missing`：不 veto，但写入数据缺口。

## 九、不做的事

本 PR 不做：

- 不新增三个常驻 LangGraph workflow 节点。
- 不把政策、游资、解禁做成无条件 LLM Agent。
- 不让 Roundtable 在没有对应数据时发言。
- 不引入付费数据源。
- 不重构现有 Market / Event / Analysis / Backtest。
- 不改变默认圆桌行为，除非 A 股特化信号触发条件参与者。

## 十、测试计划

新增测试：

```text
tests/test_a_share_signals.py
tests/test_roundtable_specialists.py
tests/test_system_a_share_rules.py
```

### DataAgent 测试

- 政策事件能生成 `policy.signal`。
- 龙虎榜 + 涨停数据能生成 `hot_money.signal`。
- 缺少解禁数据时 `unlock.data_status=missing`。
- 因子记录能生成 `multifactor.signal`。
- 无事件/无龙虎榜/无因子时输出 `insufficient` 或 `missing`，不报错。

### Roundtable 测试

- 无 A 股特化信号时圆桌保持原四席。
- `policy.strength >= 0.6` 触发 `Policy` 席位。
- `hot_money.signal == overheated` 触发 `HotMoney` 席位。
- `unlock.risk_level == high` 触发 `Unlock` 席位。
- `unlock.data_status == missing` 不触发 `Unlock` 席位。
- Specialist prompt 只包含对应证据字段。

### System 测试

- 高解禁风险导致降级。
- 游资过热导致降级。
- 政策利好不能单独支撑推荐。
- 多因子弱 + 游资投机导致偏观察/拒绝。

## 十一、建议分阶段

### Phase 0.5：HotMoney 先行接入

内容：

- Roundtable 条件参与者通用机制（动态 participants 路由、prompt 注入）。
- HotMoney 信号计算（基于已有 short_term_signals + limit_up_summary + dragon_tiger）。
- Roundtable HotMoney 条件席位。
- HotMoney 相关测试。

特点：

- 数据源已就位（short_term_signals、龙虎榜、涨停数据），技术风险最低。
- 与现有 Event / Analysis 最易产生有效观点冲突，快速验证价值。
- 用小范围实现验证条件参与者模式可行性，为后续 Phase 积累经验。

### Phase 1：DataAgent 信号层（Policy + Multifactor + Unlock）

内容：

- 新增 `a_share_signals.py`（不含 HotMoney，已在 Phase 0.5 完成）。
- 输出 `tier2_data.a_share_signals`（policy / multifactor / unlock）。
- 增加 DataAgent 测试。

特点：

- 不改变圆桌行为。
- 不改变 System 裁定。
- 风险最低。

### Phase 2：Policy + Unlock 条件席位

内容：

- Roundtable 支持动态 participants（通用机制已在 Phase 0.5 完成）。
- 新增 `Policy`、`Unlock` prompt spec。
- Specialist 只在数据触发时加入。
- 增加 Roundtable 测试（含 HotMoney 回归测试）。

特点：

- 开始影响辩论内容。
- 不影响第一轮分析顺序。

### Phase 3：System 裁定规则

内容：

- System Agent 读取 A 股特化信号。
- 增加降级/veto 规则。
- 增加 System 测试。

特点：

- 开始影响最终结论。
- 需要重点回归测试。

## 十二、推荐实施顺序

建议按 Phase 0.5 → 1 → 2 → 3 严格依次推进。

**Phase 0.5 优先的原因：**

- HotMoney 数据源最成熟（short_term_signals、龙虎榜、涨停数据已就位），技术风险最低。
- 条件参与者模式需要先在小范围内验证有效性，避免一次性引入三个 Specialist 后的不可控交互。
- Phase 0.5 完成后即有短线信号增强，可以快速验证价值，为后续 Phase 争取信心。

**进入下一阶段的前提条件：**

- Phase 0.5 → Phase 1：HotMoney 信号质量通过验证（测试覆盖 + 信号有效性回测）。
- Phase 1 → Phase 2：Policy 和 Unlock 信号层稳定，data_status 处理正确。
- Phase 2 → Phase 3：所有条件席位在圆桌中运行稳定，冲突模式可预测，Moderator 能有效管理多 Specialist 辩论。
