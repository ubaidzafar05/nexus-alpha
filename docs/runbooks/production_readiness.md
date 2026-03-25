# NEXUS-ALPHA Production Readiness Runbook

## Objective
Get the system from scaffold to production-ready using the original MD phase gates.

## Required Inputs (You Provide)
- Exchange credentials and enabled venues.
- Infrastructure access (Kafka, TimescaleDB, Redis, K8s/cloud, monitoring).
- Production SLO thresholds and risk policy approvals.
- Compliance/audit requirements and retention rules.
- Human signoff owners for stage promotion.

Use: `config/production_inputs.example.yaml` as the template.

## Preflight
1. Fill `.env` from `.env.example`.
2. Create `config/production_inputs.yaml` from the example template.
3. Run:
   - `python scripts/validate_production_readiness.py`
   - `python scripts/validate_production_readiness.py --check-network`

## Phase Gates (MD-aligned)
- Phase 0 gate: sustained ingestion + data quality + observability SLOs.
- Phase 1 gate: reproducible intelligence artifacts + explainability per decision.
- Phase 2/3 gate: end-to-end signal lifecycle + OpenClaw quarantine/signalization.
- Phase 4 gate: execution benchmark report against TWAP/VWAP.
- Phase 5 gate: risk firewall + adversarial deployment gate pass.
- Phase 6 gate: chaos drill pass + DR runbook execution records.
- Phase 7 gate: staged rollout approvals with criteria evidence.

## Evidence Required Per Gate
- Test report (unit/integration/system/adversarial).
- Metrics snapshot (latency/freshness/errors/slippage/drawdown).
- Artifact bundle (model, report, promotion record, runbook log).

## Stop Conditions
- Missing credentials/access.
- Unknown/unsigned risk policy.
- Failed adversarial gate.
- Failed chaos or DR checks.
