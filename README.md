# NEXUS-ALPHA

Self-evolving autonomous trading intelligence scaffold for crypto research and execution.

## Architecture
- CLI entrypoint drives run, paper, backtest, health, and adversarial modes.
- Signal, regime, causal, world-model, risk, execution, and portfolio modules are separated into distinct layers.
- FastAPI, websockets, Kafka, and Redis support service-style execution and streaming data flows.
- Reinforcement learning and evolutionary components sit alongside classical ML and causal validation.
- Infrastructure manifests support Docker and Kubernetes-based deployment.

## Problem + Solution
### Problem
Trading systems become brittle when signals, risk, execution, and research logic are tangled together or hidden inside opaque scripts.

### Solution
Built a modular scaffold that separates decision layers, supports offline validation through backtests and health checks, and makes it easier to reason about trading behavior before capital is exposed.

## Tech Stack
Python, FastAPI, PyTorch, scikit-learn, Stable Baselines3, Gymnasium, DoWhy, DEAP, CCXT, Kafka, Redis, SQLAlchemy, NumPy, Pandas, SciPy, Statsmodels, XGBoost, LightGBM, Optuna, MLflow, Prometheus, OpenTelemetry, Pydantic.

## Status
Research scaffold. Several subsystems are intentionally incomplete and should be validated before any production use.

## Local Run
1. Copy `.env.example` to `.env`.
2. Install dependencies from `pyproject.toml`.
3. Run `nexus --help` to inspect the available modes.
4. Use `python scripts/validate_production_readiness.py` before any live rollout.
