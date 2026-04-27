# NEXUS-ULTRA v9 Production Handover Checklist

This document details the final operational requirements for running NEXUS-ULTRA in a live, real-capital trading environment.

## 1. API Audit & Security
- [ ] **Exchange**: Binance (Live)
- [ ] **Permissions**: Spot/Futures enabled.
- [ ] **Security**: "Enable Withdrawals" MUST be **OFF**.
- [ ] **IP Whitelist**: Configure your static server IP in the Binance API management page.

## 2. Infrastructure Health
- [ ] **Redis**: Low latency (< 1ms). `redis-cli ping` should return `PONG`.
- [ ] **Kafka**: `market.ticks` stream must be receiving data before starting.
- [ ] **Ollama**: Verify `qwen3:8b` and `mistral:7b` models are pre-loaded.

## 3. Environment Switchover (.env)
Update these values in your `.env` file to go live:
```bash
ENVIRONMENT=production
TRADING_MODE=production
BINANCE_TESTNET=false
# Replace with real keys
BINANCE_API_KEY=YOUR_REAL_KEY
BINANCE_API_SECRET=YOUR_REAL_SECRET
```

## 4. Execution Strategy
To ensure the "God-Loop" (autonomous trade cycle) runs indefinitely without terminal attachment, use the provided background runner:
```bash
bash cicd/background_runner.sh
```

## 5. Emergency Protocol
- **Kill Switch**: If `circuit_breaker` is active, the bot will stop itself.
- **Manual Halt**: Press `Ctrl+C` in the attached terminal OR use `pkill -f "nexus run"` to stop all background cycles immediately.
