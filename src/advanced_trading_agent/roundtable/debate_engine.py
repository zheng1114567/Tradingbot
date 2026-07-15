"""Multi-round debate engine for Round 2 roundtable.

Runs one debate round per invocation (matching LangGraph's per-node-call pattern).
The conditional loop in workflow.py handles multi-round orchestration.

One round:
  1. Build evidence board via RoundtableHarness
  2. Market → Event → Analysis → Backtest speak in order
     Each agent sees: evidence board, other agents' latest positions, own prior position
  3. Moderator synthesizes all speeches into a structured ruling
  4. Convergence check → sets completed flag for the conditional router

Design principles:
  - All structured output uses Pydantic models via LLMClient.chat(response_format=...)
  - Every LLM call has a deterministic fallback
  - Backward compatible: populates old round2_state fields + new structured fields
"""
from __future__ import annotations

import logging
from typing import Any

from ..llm.client import LLMClient
from .harness import RoundtableHarness
from .schemas import (
    AgentStance,
    DebateTurn,
    EvidenceItem,
    ModeratorOutput,
)

logger = logging.getLogger(__name__)

_AGENT_ORDER = ("Market", "Event", "Analysis", "Backtest")


class DebateEngine:
    """Multi-round roundtable debate engine.

    Usage (from workflow node):
        engine = DebateEngine(llm=llm, harness=harness)
        result = engine.run_round(state, round_number=0, contradictions=[...])
        return {"round2_state": {**existing_round2, **result}}
    """

    def __init__(
        self,
        llm: LLMClient,
        harness: RoundtableHarness | None = None,
        max_rounds: int = 5,
    ):
        self.llm = llm
        self.harness = harness or RoundtableHarness()
        self.max_rounds = max_rounds

    def run_round(
        self,
        state: dict[str, Any],
        round_number: int,
        contradictions: list[dict[str, Any]],
        round_history: list[dict[str, Any]],
        previous_moderator_output: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Execute one debate round.

        Args:
            state: Full workflow state.
            round_number: 0-indexed round number.
            contradictions: List of ContradictionRecord dicts.
            round_history: Serialized previous rounds (list of round dicts).
            previous_moderator_output: Previous ModeratorOutput dict or None.

        Returns:
            Dict matching round2_state fields (old + new).
        """
        # Build context from harness
        contradiction_strings = [
            c.get("description", str(c)) for c in contradictions
        ]
        context = self.harness.build_context(state, contradiction_strings)
        evidence_board = context.evidence_board or []

        # Collect previous positions from last round (if any)
        prev_positions: dict[str, dict[str, Any]] = {}
        if round_history:
            last_round = round_history[-1]
            for turn_data in last_round.get("turns", []):
                stance = turn_data.get("stance", {})
                if stance:
                    prev_positions[turn_data.get("agent_name", "")] = stance

        # Current round turns
        turns: list[DebateTurn] = []
        agent_positions: dict[str, dict[str, Any]] = {}

        # Get latest positions from previous round as initial (will overwrite)
        agent_positions.update(prev_positions)

        # Each agent speaks (order from harness context)
        for agent_name in context.agent_order:
            if agent_name not in context.agent_contexts:
                continue  # skip agents with no context
            turn = self._agent_speak(
                agent_name=agent_name,
                round_number=round_number,
                context=context,
                evidence_board=evidence_board,
                prev_stance=prev_positions.get(agent_name),
                other_positions=agent_positions,
                contradiction_strings=contradiction_strings,
            )
            turns.append(turn)
            agent_positions[agent_name] = turn.stance.model_dump(mode="json")

        # Moderator speaks
        moderator_output = self._moderate(
            round_number=round_number,
            turns=turns,
            contradictions=contradictions,
            evidence_board=evidence_board,
            previous_moderator_output=previous_moderator_output,
        )

        # Convergence check
        converged = self._check_convergence(
            round_number=round_number,
            turns=turns,
            moderator_output=moderator_output,
        )

        # Build round record for history
        round_record = {
            "round_number": round_number,
            "turns": [t.model_dump(mode="json") for t in turns],
            "moderator_output": moderator_output.model_dump(mode="json"),
        }

        # Build summary
        summary_lines = [f"Round {round_number + 1} 圆桌会议:"]
        for turn in turns:
            summary_lines.append(
                f"  {turn.agent_name}: {turn.stance.pressure} "
                f"(conf={turn.stance.confidence:.1f})"
            )
            if turn.rebuts:
                summary_lines.append(f"    反驳: {', '.join(turn.rebuts)}")
            if turn.stance.changed_from_previous:
                summary_lines.append("    ⚠ 改变立场")
        summary_lines.append(
            f"  Moderator: {moderator_output.final_pressure} "
            f"(converged={moderator_output.converged})"
        )
        summary = "\n".join(summary_lines)

        # Determine final_pressure string (backward compat)
        final_pressure_str = moderator_output.final_pressure

        # Build results
        return {
            # Old fields (backward compatible)
            "round_count": round_number + 1,
            "current_speaker": "Moderator" if converged else (context.agent_order[-1] if context.agent_order else ""),
            "completed": converged,
            "summary": summary,
            "final_pressure": final_pressure_str,
            "unresolved_conflicts": [
                c.get("description", str(c)) for c in contradictions
                if c.get("id") in moderator_output.unresolved_contradiction_ids
            ] or [c.get("description", str(c)) for c in contradictions],
            "provider": "debate_engine",
            "fallback_reason": "",

            # New structured fields
            "contradiction_records": contradictions,
            "evidence_board": [e.model_dump(mode="json") for e in evidence_board],
            "round_history": [*round_history, round_record],
            "moderator_output": moderator_output.model_dump(mode="json"),
        }

    # ------------------------------------------------------------------
    # Agent speaking
    # ------------------------------------------------------------------

    def _agent_speak(
        self,
        agent_name: str,
        round_number: int,
        context: Any,
        evidence_board: list[EvidenceItem],
        prev_stance: dict[str, Any] | None,
        other_positions: dict[str, dict[str, Any]],
        contradiction_strings: list[str],
    ) -> DebateTurn:
        """Generate one agent's turn in the debate."""
        ctx = context.agent_contexts.get(agent_name)
        if not ctx:
            return self._default_turn(agent_name)

        # Format evidence board
        board_text = "\n".join(
            f"  [{e.id}] {e.agent}: {e.field_path} = {e.value[:120]}"
            for e in evidence_board
        ) if evidence_board else "  (暂无证据)"

        # Format other agents' positions (what they said this round)
        others_text = self._format_other_positions(
            agent_name, other_positions, round_number
        )

        # Format own previous stance
        prev_text = ""
        if prev_stance:
            prev_text = (
                f"你上一轮的立场:\n"
                f"  pressure={prev_stance.get('pressure', 'N/A')}, "
                f"confidence={prev_stance.get('confidence', 'N/A')}\n"
                f"  理由: {prev_stance.get('reasoning', 'N/A')[:300]}\n"
                f"你可以在本轮改变立场, 但必须说明理由。"
            )

        prompt = f"""Round {round_number + 1} — 请 {agent_name} Agent 发言。

## 待处理的矛盾
{chr(10).join(f"- {c}" for c in contradiction_strings) if contradiction_strings else "无明确矛盾, 请基于数据给出综合判断。"}

## 共享证据板
{board_text[:1200]}

## 其他 Agent 最新立场
{others_text[:1200]}

## 上一轮立场
{prev_text or "本轮是首次发言。"}

## 输出要求
1. 对每项矛盾表明立场 (upgrade/neutral/downgrade)
2. 引用证据时使用 [ev_xxxxx] ID, 必须指明具体的 DataAgent 字段
3. 如果你反驳其他 Agent, 在 rebuts 字段列出其名称
4. 置信度 confidence 在 0-1 之间
5. 如果本轮立场与上轮不同, 设置 changed_from_previous=true
6. 如果证据不足, 必须说"数据不足", 不得编造
7. 回复使用结构化格式"""

        try:
            result = self.llm.chat(
                messages=[
                    ("system", ctx.system_message),
                    ("human", prompt),
                ],
                response_format=DebateTurn,
            )
            if result:
                result.agent_name = agent_name
                return result
        except Exception as e:
            logger.warning(
                "%s Agent LLM call failed in round %d: %s",
                agent_name, round_number, e,
            )

        # Fallback
        return self._fallback_turn(agent_name, prev_stance, contradiction_strings)

    def _format_other_positions(
        self,
        agent_name: str,
        positions: dict[str, dict[str, Any]],
        round_number: int,
    ) -> str:
        """Format other agents' positions for the current agent's prompt."""
        others = {k: v for k, v in positions.items() if k != agent_name}
        if not others:
            return "  (其他 Agent 尚未发言)"
        lines = []
        for name, stance in others.items():
            lines.append(
                f"  {name}: pressure={stance.get('pressure', 'N/A')}, "
                f"confidence={stance.get('confidence', 'N/A')}"
            )
            reasoning = stance.get("reasoning", "")
            if reasoning:
                lines.append(f"    理由: {reasoning[:200]}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Moderator
    # ------------------------------------------------------------------

    def _moderate(
        self,
        round_number: int,
        turns: list[DebateTurn],
        contradictions: list[dict[str, Any]],
        evidence_board: list[EvidenceItem],
        previous_moderator_output: dict[str, Any] | None,
    ) -> ModeratorOutput:
        """Run moderator synthesis after all agents speak."""
        turns_text = "\n\n".join(
            self._format_turn_for_moderator(t) for t in turns
        )

        contradiction_descriptions = "\n".join(
            f"- [{c.get('id', '?')}] {c.get('description', str(c))}"
            for c in contradictions
        ) if contradictions else "无"

        prompt = f"""Round {round_number + 1} 辩论结束, 请 Moderator 综合裁定。

## 本轮发言记录
{turns_text[:3000]}

## 待处理矛盾
{contradiction_descriptions[:1000]}

## 输出要求
1. final_pressure: upgrade / neutral / downgrade
2. converged: 是否已达成收敛 (各 Agent 立场一致且无 stance 变化趋势)
3. unresolved_contradiction_ids: 仍未被解决的矛盾 ID
4. consensus_items: 各方达成的共识
5. dissent_items: 仍存在的分歧
6. risk_focus: 风控应关注的风险点
7. reasoning: 裁定逻辑

注意: 如果本轮是第 1 轮, 通常尚未收敛。"""

        system_msg = (
            "你是圆桌会议 Moderator, 根据所有 Agent 发言进行综合裁定。"
            "保持客观中立, 基于证据而非个人判断。"
        )

        try:
            result = self.llm.chat(
                messages=[("system", system_msg), ("human", prompt)],
                response_format=ModeratorOutput,
            )
            if result:
                result.round_number = round_number
                return result
        except Exception as e:
            logger.warning("Moderator LLM call failed in round %d: %s", round_number, e)

        # Deterministic fallback
        return self._fallback_moderator(round_number, turns, contradictions)

    def _format_turn_for_moderator(self, turn: DebateTurn) -> str:
        """Format one turn for moderator consumption."""
        evidence_ids = ", ".join(turn.stance.evidence_ids) if turn.stance.evidence_ids else "无"
        rebuts = ", ".join(turn.rebuts) if turn.rebuts else "无"
        changed = " (立场改变)" if turn.stance.changed_from_previous else ""
        return (
            f"--- {turn.agent_name} ---{changed}\n"
            f"立场: {turn.stance.pressure} (置信度: {turn.stance.confidence:.1f})\n"
            f"引用证据: [{evidence_ids}]\n"
            f"反驳: {rebuts}\n"
            f"理由: {turn.stance.reasoning[:500]}"
        )

    # ------------------------------------------------------------------
    # Convergence
    # ------------------------------------------------------------------

    def _check_convergence(
        self,
        round_number: int,
        turns: list[DebateTurn],
        moderator_output: ModeratorOutput,
    ) -> bool:
        """Check if debate should stop."""
        # Hard limit
        if round_number >= self.max_rounds - 1:
            return True
        # Minimum 2 rounds
        if round_number < 1:
            return False
        # Moderator says converged
        if moderator_output.converged:
            return True
        # No agent changed position
        if not any(t.stance.changed_from_previous for t in turns):
            return True
        return False

    # ------------------------------------------------------------------
    # Fallbacks
    # ------------------------------------------------------------------

    def _default_turn(self, agent_name: str) -> DebateTurn:
        """Return a neutral default turn when agent is unavailable."""
        return DebateTurn(
            agent_name=agent_name,
            stance=AgentStance(
                pressure="neutral",
                confidence=0.3,
                evidence_ids=[],
                reasoning="数据不足, 默认中立。",
                changed_from_previous=False,
            ),
            rebuts=[],
            new_evidence=[],
        )

    def _fallback_turn(
        self,
        agent_name: str,
        prev_stance: dict[str, Any] | None,
        contradictions: list[str],
    ) -> DebateTurn:
        """Return a fallback turn when LLM call fails."""
        if prev_stance:
            return DebateTurn(
                agent_name=agent_name,
                stance=AgentStance(
                    pressure=prev_stance.get("pressure", "neutral"),
                    confidence=prev_stance.get("confidence", 0.3) * 0.8,
                    evidence_ids=prev_stance.get("evidence_ids", []),
                    reasoning=f"LLM 不可用, 沿用上一轮立场: {prev_stance.get('reasoning', '')[:200]}",
                    changed_from_previous=False,
                ),
            )
        return DebateTurn(
            agent_name=agent_name,
            stance=AgentStance(
                pressure="neutral",
                confidence=0.3,
                evidence_ids=[],
                reasoning="LLM 不可用, 默认中立立场。矛盾仍需在最终裁定中关注。",
                changed_from_previous=False,
            ),
        )

    def _fallback_moderator(
        self,
        round_number: int,
        turns: list[DebateTurn],
        contradictions: list[dict[str, Any]],
    ) -> ModeratorOutput:
        """Deterministic moderator fallback when LLM is unavailable."""
        pressure_counts = {"upgrade": 0, "neutral": 0, "downgrade": 0}
        total_conf = 0.0

        for turn in turns:
            p = turn.stance.pressure
            pressure_counts[p] = pressure_counts.get(p, 0) + 1
            total_conf += turn.stance.confidence

        # Majority pressure
        if pressure_counts["downgrade"] > pressure_counts["upgrade"]:
            final_pressure = "downgrade"
        elif pressure_counts["upgrade"] > pressure_counts["downgrade"]:
            final_pressure = "upgrade"
        else:
            final_pressure = "neutral"

        # Conservative convergence when LLM unavailable
        converged = round_number >= 1

        return ModeratorOutput(
            round_number=round_number,
            final_pressure=final_pressure,
            unresolved_contradiction_ids=[c.get("id", "") for c in contradictions],
            consensus_items=["LLM 不可用, 无法提取共识"],
            dissent_items=[c.get("description", str(c)) for c in contradictions],
            converged=converged,
            reasoning=f"LLM 不可用, 使用确定性回退: "
                      f"upgrade={pressure_counts['upgrade']}, "
                      f"neutral={pressure_counts['neutral']}, "
                      f"downgrade={pressure_counts['downgrade']}",
            risk_focus=["LLM不可用, 所有矛盾需人工关注"],
        )
