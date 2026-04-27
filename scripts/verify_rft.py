"""
Verification script for Nexus-Alpha RFT Loop.
Simulates a debate and a trade closure to verify ARTVault recording.
"""
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from nexus_alpha.learning.rft_vault import ARTVault
from nexus_alpha.schema_types import Signal, DebateVerdict, ExchangeName

async def verify_rft_loop():
    print("🧪 Starting RFT Loop Verification...")
    
    vault = ARTVault()
    symbol = "BTC/USDT"
    
    # 1. Simulate a Debate Recording
    print("1. Recording mock debate...")
    traj_id = vault.record_debate(
        symbol=symbol,
        proposal_text="SNIPER: RSI is oversold at 28. Strong buy signal cluster.",
        challenge_text="TACTICAL: Volatility is spiked. Potential liquidity grab below $60k. Caution.",
        synthesis_text="<thought>The RSI is indeed low, but the tactical juror's concern about liquidity is valid. I will proceed with a reduced position size to balance risk.</thought> PROCEED with 0.5x size.",
        system_prompt="Verification System Template",
        market_context_prompt="Mock Context: Vol=0.4, RSI=30, Trend=Bullish"
    )
    print(f"   Stored Trajectory ID: {traj_id}")
    
    # 2. Check if file exists and contains the ID
    with open("data/rft/trajectories.jsonl", "r") as f:
        content = f.read()
        if traj_id in content:
            print("   ✅ Trajectory found in trajectories.jsonl")
        else:
            print("   ❌ Trajectory NOT found!")
            return

    # 3. Simulate a Trade Reward update
    print("2. Simulating trade closure (PnL=5.0%)...")
    # In a real environment, this reward comes from TradeLogger._compute_reward
    mock_reward = 0.85 
    success = vault.update_reward(traj_id, mock_reward)
    
    if success:
        print("   ✅ Reward successfully updated in vault.")
    else:
        print("   ❌ Failed to update reward!")
        return

    # 4. Final Verification
    with open("data/rft/trajectories.jsonl", "r") as f:
        for line in f:
            if traj_id in line:
                if '"status": "rewarded"' in line and f'"reward": {mock_reward}' in line:
                    print("   ✅ Record marked as 'rewarded' with correct value.")
                    print("\n🎉 RFT PIPELINE VERIFIED SUCCESSFULLY!")
                    return

    print("   ❌ Record status or reward was not updated correctly.")

if __name__ == "__main__":
    asyncio.run(verify_rft_loop())
