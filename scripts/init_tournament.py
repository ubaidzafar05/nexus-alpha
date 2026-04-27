import json
import uuid
from pathlib import Path
from datetime import datetime

# Initialize the tournament registry with a baseline model
registry_path = Path("data/tournament/registry.json")
registry_path.parent.mkdir(parents=True, exist_ok=True)

baseline = {
    "champion": {
        "id": f"baseline_{uuid.uuid4().hex[:8]}",
        "type": "SignalFusionEngine",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "registered_at": datetime.utcnow().isoformat(),
        "metrics": {
            "net_return_pct": 0.0,
            "sharpe": 0.0,
            "accuracy": 0.50
        },
        "parameters": {
            "weights": {
                "technical": 0.4,
                "sentiment": 0.3,
                "on_chain": 0.3
            },
            "min_confidence": 0.25
        }
    },
    "candidates": []
}

with open(registry_path, "w") as f:
    json.dump(baseline, f, indent=2)

print(f"✅ Tournament Registry Initialized with Baseline Champion: {baseline['champion']['id']}")
