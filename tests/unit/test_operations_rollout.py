from __future__ import annotations

from nexus_alpha.infrastructure.operations import ChaosDrillOrchestrator, DisasterRecoveryRunbook
from nexus_alpha.infrastructure.rollout import DeploymentStage, LiveRolloutController, StageMetrics


def test_chaos_drill_and_dr_runbook_execute() -> None:
    chaos = ChaosDrillOrchestrator()
    result = chaos.run(base_price=65000.0)
    assert result.total_scenarios >= 1

    runbook = DisasterRecoveryRunbook()
    records = runbook.execute()
    assert len(records) >= 1
    assert all(record.success for record in records)


def test_rollout_controller_promotes_through_stages() -> None:
    controller = LiveRolloutController()

    paper_metrics = StageMetrics(
        elapsed_days=61,
        sharpe=1.2,
        max_drawdown=0.10,
        pnl_divergence=0.0,
        adversarial_passed=True,
        human_signoff=True,
    )
    paper_record = controller.evaluate_promotion(paper_metrics)
    assert paper_record.approved is True
    assert controller.current_stage == DeploymentStage.MICRO

    micro_metrics = StageMetrics(
        elapsed_days=35,
        sharpe=0.4,
        max_drawdown=0.10,
        pnl_divergence=0.15,
        adversarial_passed=True,
        human_signoff=True,
    )
    micro_record = controller.evaluate_promotion(micro_metrics)
    assert micro_record.approved is True
    assert controller.current_stage == DeploymentStage.SMALL
