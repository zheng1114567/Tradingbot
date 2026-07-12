"""Agent role specs and reusable prompt rules.

The specs are intentionally small: they centralize each agent's role contract
without hiding the data assembly logic that still belongs in the agent node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..tool_nodes.registry import get_allowed_tool_names

@dataclass(frozen=True)
class AgentSkillSpec:
    """Prompt contract for one project agent."""

    key: str
    display_name: str
    fallback_system_prompt: str
    react_prompt: str
    roundtable_focus: str = ""


AGENT_SKILLS: dict[str, AgentSkillSpec] = {
    "market": AgentSkillSpec(
        key="market",
        display_name="Market Agent",
        fallback_system_prompt="你是 A 股市场分析师。评估当前市场温度和资金状态。",
        react_prompt=(
            "你是 A 股市场分析师。必须用可用工具检查市场情绪、资金、板块轮动和涨停梯队，"
            "再输出结构化市场分析。仓位上限必须服从硬规则。"
        ),
        roundtable_focus="市场温度、资金确认、仓位约束、行业环境",
    ),
    "event": AgentSkillSpec(
        key="event",
        display_name="Event Agent",
        fallback_system_prompt=(
            "你是 A 股事件分析师。严格遵守反伪链条规则。"
            "没有明确实体映射的事件，只能给 indirect。"
        ),
        react_prompt=(
            "你是 A 股事件分析师。必须用工具搜索新闻、公告或日历事件，"
            "再按反伪链条规则判断事件是否有交易价值。"
        ),
        roundtable_focus="事件传导、证据等级、定价状态、证伪条件",
    ),
    "analysis": AgentSkillSpec(
        key="analysis",
        display_name="Analysis Agent",
        fallback_system_prompt="你是 A 股因子分析师。基于因子数据分析，不凭感觉。",
        react_prompt=(
            "你是 A 股因子分析师。必须先用工具检查因子、排序和拥挤度，"
            "再输出结构化因子分析；个股排序最终必须以确定性排序结果为准。"
        ),
        roundtable_focus="因子排序、拥挤风险、择时过滤、数据质量",
    ),
    "backtest": AgentSkillSpec(
        key="backtest",
        display_name="Backtest Agent",
        fallback_system_prompt=(
            "你是 A 股历史证据审查员。样本不足时不能支撑买入。"
            "必须同时考虑成功样本和失败样本。"
        ),
        react_prompt=(
            "你是 A 股历史证据审查员。必须用工具运行回测并查找相似历史样本，"
            "再输出结构化报告。样本不足、胜率不足和负超额收益必须降级。"
        ),
        roundtable_focus="样本量、胜率、超额收益、统计可靠性",
    ),
}


ROUNDTABLE_RULES = (
    "只能引用自己的 AgentContext、DATA_AGENT_BRIEF 和你已有的报告。",
    "只能引用自己的 AgentContext 中的数据字段；不要替其他 Agent 解释其私有证据。",
    "如果证据缺失，必须明确说“数据不足”，不得补造外部数据。",
    "必须回应矛盾点，并说明对最终裁定的影响: upgrade/neutral/downgrade。",
    "引用共享证据板时使用 [ev_xxxxx] ID，以方便 Moderator 追溯。",
)


def get_agent_skill(agent_key: str) -> AgentSkillSpec:
    """Return the prompt contract for an agent key."""
    normalized = agent_key.lower()
    if normalized not in AGENT_SKILLS:
        raise KeyError(f"Unknown agent skill: {agent_key}")
    return AGENT_SKILLS[normalized]


def build_roundtable_system_message(
    *,
    agent: str,
    report: str,
    evidence: str,
    shared_evidence: str,
    evidence_board: list[Any],
    char_limit: int,
) -> str:
    """Build the Round 2 scoped system message from the shared skill rules."""
    skill = get_agent_skill(agent)
    allowed_tools = ", ".join(get_allowed_tool_names(agent.lower()))
    board_text = ""
    if evidence_board:
        formatted = "\n".join(
            f"  [{e.id}] {e.agent}: {e.field_path} = {e.value}" for e in evidence_board
        )
        board_text = f"\n\n共享证据板 (引用时请使用 [ev_xxxxx] ID):\n{formatted[:800]}"

    rules = "\n".join(f"{idx}. {rule}" for idx, rule in enumerate(ROUNDTABLE_RULES, 1))
    return f"""你是 {skill.display_name}，参加 Round 2 圆桌会议。

边界规则:
{rules}
6. 如果需要调用工具，只能使用当前角色白名单内已注册工具；禁止调用任何未注册工具。{board_text}

关注范围: {skill.roundtable_focus}

允许使用的工具:
{allowed_tools}

DATA_AGENT_BRIEF:
{shared_evidence[:char_limit]}

AgentContext:
{evidence[:char_limit]}

既有报告:
{report[:char_limit]}
"""
