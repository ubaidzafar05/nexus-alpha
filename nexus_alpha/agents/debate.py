"""
Multi-Agent Debate Protocol — For high-stakes trade decisions.

Before any position exceeding 5% NAV, agents formally debate the trade.
Three rounds:
1. PROPOSAL: Proposing agent presents its thesis
2. CHALLENGE: Devil's advocate agent stress-tests the thesis
3. SYNTHESIS: Meta-agent weighs debate and makes final recommendation
"""

from __future__ import annotations

from dataclasses import dataclass, field

import orjson

from nexus_alpha.config import LLMConfig
from nexus_alpha.log_config import get_logger
from nexus_alpha.schema_types import DebateVerdict, Signal

logger = get_logger(__name__)

DEBATE_TRIGGER_THRESHOLD_NAV = 0.05  # 5% NAV

PROPOSAL_PROMPT = """\
<role>You are a senior quantitative analyst proposing a trade.</role>

<trade>
Symbol: {symbol}
Direction: {direction}
Confidence: {confidence:.2f}
Source: {source}
Lineage Depth: {lineage_depth}
Ancestor: {ancestor_id}
</trade>

<market_context>
Regime: {regime}
Volatility: {volatility:.4f}
Trend Strength: {trend_strength:.2f}
Recent Returns (5-period): {recent_returns}
</market_context>

<historical_precedents>
{memories_text}
</historical_precedents>

<instructions>
Present a concise investment thesis for this trade. Include:
1. Primary catalyst driving this signal
2. Expected holding period and target
3. Key risk factors you've already considered
4. How your lineage (depth {lineage_depth}) justifies your conviction vs transient variations
5. What would invalidate this thesis

Be specific. Use numbers. No vague language.
</instructions>
"""

CHALLENGE_PROMPT = """\
<role>You are a risk-focused devil's advocate. \
Your job is to stress-test this trade proposal.</role>

<proposal>
{proposal_text}
</proposal>

<market_context>
Regime: {regime}
Volatility: {volatility:.4f}
Current Drawdown: {drawdown:.2f}%
Correlation to BTC: {btc_correlation:.2f}
</market_context>

<instructions>
Challenge this proposal rigorously:
1. What confirmation bias might the proposer have?
2. What regime shift could destroy this trade?
3. What historical analogues ended badly?
4. If this signal is right, why hasn't the market already priced it in?
5. What is the tail-risk scenario?

Be adversarial but intellectually honest. Cite specific risks.
</instructions>
"""

SYNTHESIS_PROMPT = """\
<role>You are the Chief Risk Officer (Meta-Juror) making a final trading decision.</role>

<personality_context>
The juror panel for this debate includes:
1. SNIPER JUROR: Highest precision requirements, handles 20% NAV. Vetoes any misalignment with the Macro trend.
2. TACTICAL JUROR: Generalist, handles 50% NAV. Focuses on risk-reward efficiency.
3. SCOUT JUROR: Exploration-focused, handles 30% NAV. Open to micro-signal discovery.
</personality_context>

<proposal>
{proposal_text}
</proposal>

<challenge>
{challenge_text}
</challenge>

<portfolio_context>
Current NAV: ${nav:,.2f}
Proposed Size: {size_pct:.1f}% NAV
Current Drawdown: {drawdown:.2f}%
Open Positions: {n_positions}
Portfolio Heat: {portfolio_heat:.1f}%
</portfolio_context>

<instructions>
Make a final recommendation based on the Juror consensus. If the SNIPER JUROR logic contradicts the entry, you must REJECT or REDUCE_SIZE.
Output EXACTLY this JSON:
{{
    "recommendation": "proceed" | "reduce_size" | "reject",
    "adjusted_confidence": <0.0 to 1.0>,
    "suggested_size_pct": <0.0 to 20.0>,
    "juror_consensus": {
        "sniper": "approve" | "reject",
        "tactical": "approve" | "reject",
        "scout": "approve" | "reject"
    },
    "reasoning": "<2-3 sentence justification reflecting the juror debate>",
    "key_risk": "<single most important risk noted by the Sniper>"
}}
</instructions>
"""


@dataclass
class DebateContext:
    """All context needed for a debate."""
    signal: Signal
    regime: str
    volatility: float
    trend_strength: float
    recent_returns: str
    drawdown: float
    btc_correlation: float
    nav: float
    size_pct: float
    n_positions: int
    portfolio_heat: float
    historical_memories: list[dict[str, Any]] = field(default_factory=list)

    @property
    def memories_text(self) -> str:
        if not self.historical_memories:
            return "No close historical precedents found."
        
        import json
        lines = []
        for i, m in enumerate(self.historical_memories):
            lines.append(f"Precedent {i+1} (Importance: {m.get('importance', 0):.2f}, PnL: {m.get('pnl', 0):.2f}%):")
            lines.append(f"  Reasoning: {m.get('reasoning', 'N/A')}")
            lines.append(f"  Outcome: {json.dumps(m.get('outcome', {}))}")
        return "\n".join(lines)


class AgentDebateProtocol:
    """
    Multi-agent debate protocol for high-conviction trades.
    Uses LLM agents in structured adversarial roles.
    """

    def __init__(self, llm_config: LLMConfig | None = None):
        self.config = llm_config or LLMConfig()

    async def should_debate(self, signal: Signal, nav: float, position_value: float) -> bool:
        """Check if position size warrants a debate."""
        if nav <= 0:
            return False
        size_pct = position_value / nav
        return size_pct >= DEBATE_TRIGGER_THRESHOLD_NAV

    async def conduct_debate(self, context: DebateContext) -> DebateVerdict:
        """
        Execute the 3-round debate protocol.
        Falls back to conservative default if LLM is unavailable.
        """
        try:
            # Round 1: Proposal
            proposal_text = await self._llm_call(
                PROPOSAL_PROMPT.format(
                    symbol=context.signal.symbol,
                    direction="LONG" if context.signal.direction > 0 else "SHORT",
                    confidence=context.signal.confidence,
                    source=context.signal.source,
                    lineage_depth=context.signal.metadata.get("lineage_depth", 0),
                    ancestor_id=context.signal.metadata.get("ancestor_id", "v6-root"),
                    regime=context.regime,
                    volatility=context.volatility,
                    trend_strength=context.trend_strength,
                    recent_returns=context.recent_returns,
                    memories_text=context.memories_text,
                )
            )

            # Round 2: Challenge
            challenge_text = await self._llm_call(
                CHALLENGE_PROMPT.format(
                    proposal_text=proposal_text,
                    regime=context.regime,
                    volatility=context.volatility,
                    drawdown=context.drawdown * 100,
                    btc_correlation=context.btc_correlation,
                )
            )

            # Round 3: Synthesis
            synthesis_raw = await self._llm_call(
                SYNTHESIS_PROMPT.format(
                    proposal_text=proposal_text,
                    challenge_text=challenge_text,
                    nav=context.nav,
                    size_pct=context.size_pct,
                    drawdown=context.drawdown * 100,
                    n_positions=context.n_positions,
                    portfolio_heat=context.portfolio_heat,
                )
            )

            verdict = self._parse_synthesis(synthesis_raw, context.signal)

            try:
                from nexus_alpha.learning.rft_vault import ARTVault
                market_ctx = (
                    f"Regime: {context.regime}\n"
                    f"Volatility: {context.volatility:.4f}\n"
                    f"Drawdown: {context.drawdown * 100:.2f}%\n"
                    f"NAV: ${context.nav:,.2f}"
                )
                system_prompt = "You are a Chief Risk Officer synthesizing a debate between a Sniper, Tactical, and Scout juror."
                vault = ARTVault()
                traj_id = vault.record_debate(
                    symbol=context.signal.symbol,
                    proposal_text=proposal_text,
                    challenge_text=challenge_text,
                    synthesis_text=synthesis_raw,
                    system_prompt=system_prompt,
                    market_context_prompt=market_ctx
                )
                verdict.trajectory_id = traj_id
            except Exception as e:
                logger.error("failed_to_record_rft_trajectory", error=str(e))

            logger.info(
                "debate_complete",
                symbol=context.signal.symbol,
                recommendation=verdict.synthesis_recommendation,
                adjusted_confidence=f"{verdict.adjusted_confidence:.2f}",
            )

            return verdict

        except Exception:
            logger.exception("debate_failed_using_conservative_default")
            return DebateVerdict(
                proposed_trade=context.signal,
                proposal_strength=0.5,
                challenge_strength=0.5,
                synthesis_recommendation="reduce_size",
                adjusted_confidence=context.signal.confidence * 0.5,
                reasoning="Debate failed — defaulting to conservative position sizing.",
            )

    async def _llm_call(self, prompt: str) -> str:
        """Call free LLM (Ollama → Groq fallback). No paid API key needed."""
        from nexus_alpha.intelligence.free_llm import FreeLLMClient
        try:
            client = FreeLLMClient.from_config(self.config)
            # Use reasoning model (DeepSeek-R1) for debate — structured adversarial reasoning
            return await client.complete_reasoning(
                prompt,
                system="You are a senior quantitative analyst. "
                "Be analytical, precise, and concise.",
            )
        except Exception:
            logger.exception("debate_llm_failed")
            return "(LLM unavailable — using conservative default)"

    def _parse_synthesis(self, raw: str, signal: Signal) -> DebateVerdict:
        """Parse the synthesis JSON from the LLM."""
        try:
            # Extract JSON from response (may have surrounding text)
            start = raw.index("{")
            end = raw.rindex("}") + 1
            data = orjson.loads(raw[start:end])

            verdict = DebateVerdict(
                proposed_trade=signal,
                proposal_strength=0.7,
                challenge_strength=0.6,
                synthesis_recommendation=data.get("recommendation", "reduce_size"),
                adjusted_confidence=float(data.get("adjusted_confidence", signal.confidence * 0.7)),
                reasoning=data.get("reasoning", "No reasoning provided."),
            )

            # Sniper Veto Enforcement: If sniper juror explicitly rejected, enforce rejection
            jurors = data.get("juror_consensus", {})
            if jurors.get("sniper") == "reject" and verdict.synthesis_recommendation == "proceed":
                verdict.synthesis_recommendation = "reduce_size"
                verdict.reasoning = f"(Sniper Veto Applied) {verdict.reasoning}"

            return verdict
        except (ValueError, KeyError, orjson.JSONDecodeError):
            return DebateVerdict(
                proposed_trade=signal,
                proposal_strength=0.5,
                challenge_strength=0.5,
                synthesis_recommendation="reduce_size",
                adjusted_confidence=signal.confidence * 0.5,
                reasoning="Failed to parse synthesis — defaulting to conservative.",
            )

    async def close(self) -> None:
        pass
