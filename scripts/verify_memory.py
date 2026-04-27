import asyncio
import json
from nexus_alpha.config import load_config
from nexus_alpha.intelligence.free_llm import FreeLLMClient
from nexus_alpha.learning.memory import MemoryManager

async def test_memory_flow():
    print("🚀 Starting Memory Verification...")
    config = load_config()
    llm = FreeLLMClient.from_config(config.llm)
    
    # Overrides for host-based verification
    config.qdrant.url = "http://localhost:6333"
    config.llm.ollama_base_url = "http://localhost:11434"
    
    # Increase timeout for local reasoning (can be slow on 16GB Mac)
    llm = FreeLLMClient(
        ollama_base_url="http://localhost:11434",
        primary_model=config.llm.ollama_primary_model,
        fast_model=config.llm.ollama_fast_model,
        reasoning_model=config.llm.ollama_reasoning_model,
        embed_model=config.llm.ollama_embed_model,
        timeout=300.0 
    )
    memory = MemoryManager(config, llm)
    
    if memory.client is None:
        print("❌ Qdrant Client failed to initialize. Is the service running?")
        return

    # Simulation data
    symbol = "BTCUSDT"
    context = {"regime": "strong_trend", "volatility": 0.02, "symbol": symbol}
    reasoning = "Entering long because of clear breakout on 4h and positive sentiment."
    outcome = {"realized_pnl_pct": 2.5, "exit_reason": "take_profit"}
    pnl_pct = 2.5

    print("🧠 Step 1: Storing a 'High Importance' memory...")
    # This will trigger DeepSeek-R1 for scoring
    success = await memory.store_memory(symbol, context, reasoning, outcome, pnl_pct)
    if success:
        print("✅ Memory stored successfully.")
    else:
        print("❌ Failed to store memory.")
        return

    print("🔍 Step 2: Retrieving relevant memories for same context...")
    results = await memory.retrieve_relevant_memories(context, top_k=1)
    
    if results:
        print(f"✅ Retrieved {len(results)} memory(ies).")
        print(f"   Top Match Importance: {results[0]['importance']}")
        print(f"   Top Match Reasoning: {results[0]['reasoning'][:50]}...")
    else:
        print("❌ No memories retrieved.")

if __name__ == "__main__":
    asyncio.run(test_memory_flow())
