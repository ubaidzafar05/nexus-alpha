# NEXUS-ALPHA

NEXUS-ALPHA is a Python trading research and execution scaffold for multi-agent crypto trading.

## Current State

This repository contains:

- configuration models for exchanges, infrastructure, and risk
- a CLI entrypoint for run, paper, backtest, health, and adversarial modes
- execution, risk, signal, portfolio, and intelligence modules
- Docker and infrastructure manifests for local services

This codebase is not production-ready as-is. Several subsystems are scaffolds or partially wired implementations intended for further integration and validation.

## Quick Start

1. Copy `.env.example` to `.env`.
2. Fill in required credentials and service URLs.
3. Activate the managed environment.
4. Run the CLI with `nexus --help`.

## Verification

The package source under `nexus_alpha/` compiles with Python 3.11 in the managed environment.
