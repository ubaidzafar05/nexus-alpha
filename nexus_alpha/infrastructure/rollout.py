"""Phase 7 staged live-rollout controller."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from nexus_alpha.logging import get_logger

logger = get_logger(__name__)


class DeploymentStage(str, Enum):
    PAPER = "paper"
    MICRO = "micro_live"
    SMALL = "small_live"
    PRODUCTION = "production"


@dataclass(frozen=True)
class PromotionCriteria:
    min_days: int
    min_sharpe: float
    max_drawdown: float
    max_pnl_divergence: float | None = None
    require_adversarial_pass: bool = False
    require_human_signoff: bool = False


@dataclass(frozen=True)
class StageMetrics:
    elapsed_days: int
    sharpe: float
    max_drawdown: float
    pnl_divergence: float
    adversarial_passed: bool
    human_signoff: bool


@dataclass(frozen=True)
class PromotionRecord:
    from_stage: DeploymentStage
    to_stage: DeploymentStage
    approved: bool
    reason_codes: list[str]
    timestamp: datetime


class LiveRolloutController:
    """Controls promotion through paper -> micro -> small -> production."""

    CRITERIA = {
        DeploymentStage.PAPER: PromotionCriteria(
            min_days=60,
            min_sharpe=1.0,
            max_drawdown=0.20,
        ),
        DeploymentStage.MICRO: PromotionCriteria(
            min_days=30,
            min_sharpe=0.2,
            max_drawdown=0.20,
            max_pnl_divergence=0.20,
        ),
        DeploymentStage.SMALL: PromotionCriteria(
            min_days=60,
            min_sharpe=0.8,
            max_drawdown=0.12,
        ),
        DeploymentStage.PRODUCTION: PromotionCriteria(
            min_days=90,
            min_sharpe=0.8,
            max_drawdown=0.12,
            require_adversarial_pass=True,
            require_human_signoff=True,
        ),
    }

    _NEXT = {
        DeploymentStage.PAPER: DeploymentStage.MICRO,
        DeploymentStage.MICRO: DeploymentStage.SMALL,
        DeploymentStage.SMALL: DeploymentStage.PRODUCTION,
        DeploymentStage.PRODUCTION: DeploymentStage.PRODUCTION,
    }

    def __init__(self) -> None:
        self.current_stage = DeploymentStage.PAPER
        self.history: list[PromotionRecord] = []

    def evaluate_promotion(self, metrics: StageMetrics) -> PromotionRecord:
        current = self.current_stage
        target = self._NEXT[current]
        if current == DeploymentStage.PRODUCTION:
            record = PromotionRecord(
                from_stage=current,
                to_stage=current,
                approved=False,
                reason_codes=["already_at_production"],
                timestamp=datetime.utcnow(),
            )
            self.history.append(record)
            return record

        criteria = self.CRITERIA[current]
        reasons = self._evaluate_against_criteria(criteria, metrics)
        approved = len(reasons) == 0
        if approved:
            reasons = ["promotion_approved"]
            self.current_stage = target
            logger.info("rollout_promoted", from_stage=current.value, to_stage=target.value)

        record = PromotionRecord(
            from_stage=current,
            to_stage=target,
            approved=approved,
            reason_codes=reasons,
            timestamp=datetime.utcnow(),
        )
        self.history.append(record)
        return record

    def _evaluate_against_criteria(
        self,
        criteria: PromotionCriteria,
        metrics: StageMetrics,
    ) -> list[str]:
        reasons: list[str] = []
        if metrics.elapsed_days < criteria.min_days:
            reasons.append("insufficient_runtime_days")
        if metrics.sharpe < criteria.min_sharpe:
            reasons.append("sharpe_below_threshold")
        if metrics.max_drawdown > criteria.max_drawdown:
            reasons.append("drawdown_above_threshold")
        if (
            criteria.max_pnl_divergence is not None
            and metrics.pnl_divergence > criteria.max_pnl_divergence
        ):
            reasons.append("paper_live_divergence_too_high")
        if criteria.require_adversarial_pass and not metrics.adversarial_passed:
            reasons.append("adversarial_gate_not_passed")
        if criteria.require_human_signoff and not metrics.human_signoff:
            reasons.append("human_signoff_missing")
        return reasons
