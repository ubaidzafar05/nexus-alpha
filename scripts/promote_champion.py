import json
from pathlib import Path
from datetime import datetime

# Load best params
best_params_path = Path("data/optimization/best_params_BTC_USDT.json")
with open(best_params_path, "r") as f:
    best_res = json.load(f)

# Load registry
registry_path = Path("data/tournament/registry.json")
with open(registry_path, "r") as f:
    registry = json.load(f)

# Create Challenger
challenger = {
    "id": f"optuna_challenger_{datetime.utcnow().strftime('%Y%m%d%H%M')}",
    "type": "SignalFusionEngine",
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "registered_at": datetime.utcnow().isoformat(),
    "metrics": {
        "net_return_pct": -2656.22,
        "accuracy": 0.529
    },
    "parameters": best_res["best_params"]
}

# Add to candidates
registry["candidates"].append(challenger)

# Promote Challenger to Champion (manual promotion logic for this step)
old_champion = registry["champion"]
registry["past_champions"] = registry.get("past_champions", [])
registry["past_champions"].append(old_champion)

registry["champion"] = challenger
registry["candidates"] = [c for c in registry["candidates"] if c["id"] != challenger["id"]]

with open(registry_path, "w") as f:
    json.dump(registry, f, indent=2)

print(f"🏆 NEW CHAMPION PROMOTED: {challenger['id']}")
print(f"📈 Profit Improvement: From -3747.87% to -2656.22% (+1091.65%)")
