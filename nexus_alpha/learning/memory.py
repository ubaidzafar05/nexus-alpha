"""
Agentic Memory Layer for Nexus-Alpha.
Uses Qdrant for vector storage and DeepSeek-R1 for importance-weighted filtering.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
except ImportError:
    # Fallback to avoid breaking boot if client is missing
    QdrantClient = None 

from nexus_alpha.config import NexusConfig
from nexus_alpha.intelligence.free_llm import FreeLLMClient
from nexus_alpha.log_config import get_logger

logger = get_logger(__name__)

IMPORTANCE_PROMPT = """
You are a Senior Memory Architect for a quantitative trading bot.
Evaluate the "IMPORTANCE" of the following trading experience.

MARKET CONTEXT:
{context}

REASONING:
{reasoning}

OUTCOME:
{outcome}

Score the importance from 0.0 to 1.0.
- 0.0: Boring, redundant, or random noise.
- 0.5: Standard profitable trade or textbook loss.
- 1.0: Critical insight, black-swan event, or highly unique regime-shift pattern.

Output ONLY a JSON object: {{"importance_score": float, "reasoning": str}}
"""

class MemoryManager:
    """
    Manages long-term agent memory using Qdrant.
    Implements Importance-Weighted RAG.
    """
    
    def __init__(self, config: NexusConfig, llm_client: FreeLLMClient):
        self.config = config
        self.llm = llm_client
        self.qdrant_cfg = config.qdrant
        
        if QdrantClient is None:
            self.client = None
            logger.error("qdrant_client_not_installed")
            return

        try:
            self.client = QdrantClient(url=self.qdrant_cfg.url)
            self._ensure_collection()
        except Exception as e:
            logger.error("qdrant_connection_failed", error=str(e))
            self.client = None

    def _ensure_collection(self) -> None:
        """Create the memories collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        if not any(c.name == self.qdrant_cfg.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.qdrant_cfg.collection_name,
                vectors_config=rest.VectorParams(
                    size=self.qdrant_cfg.vector_size,
                    distance=rest.Distance.COSINE
                )
            )
            logger.info("qdrant_collection_created", name=self.qdrant_cfg.collection_name)

    async def store_memory(
        self,
        symbol: str,
        context: Dict[str, Any],
        reasoning: str,
        outcome: Dict[str, Any],
        pnl_pct: float
    ) -> bool:
        """
        Evaluate importance and store a new memory in Qdrant.
        """
        if self.client is None: return False

        try:
            # 1. Evaluate Importance via LLM (Reasoning Model)
            importance_resp = await self.llm.complete_json(
                IMPORTANCE_PROMPT.format(
                    context=json.dumps(context),
                    reasoning=reasoning,
                    outcome=json.dumps(outcome)
                ),
                model=self.config.llm.ollama_reasoning_model
            )
            importance_score = importance_resp.get("importance_score", 0.5)
            importance_rationale = importance_resp.get("reasoning", "")

            # 2. Generate Embedding for the context
            vector = await self.llm.embed(json.dumps(context))

            # 3. Store in Qdrant
            point_id = int(datetime.utcnow().timestamp() * 1000)
            self.client.upsert(
                collection_name=self.qdrant_cfg.collection_name,
                points=[
                    rest.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "symbol": symbol,
                            "timestamp": datetime.utcnow().isoformat(),
                            "importance_score": importance_score,
                            "importance_rationale": importance_rationale,
                            "pnl_pct": pnl_pct,
                            "reasoning": reasoning,
                            "context": context,
                            "outcome": outcome
                        }
                    )
                ]
            )
            logger.info("memory_stored", symbol=symbol, importance=importance_score, pnl=round(pnl_pct, 4))
            return True

        except Exception as e:
            logger.error("memory_storage_failed", error=str(e), symbol=symbol)
            return False

    async def retrieve_relevant_memories(
        self, 
        context: Dict[str, Any], 
        top_k: int = 3,
        min_importance: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve past high-value memories similar to the current context.
        """
        if self.client is None: return []

        try:
            vector = await self.llm.embed(json.dumps(context))
            
            # Search with Importance Filter
            results = self.client.search(
                collection_name=self.qdrant_cfg.collection_name,
                query_vector=vector,
                query_filter=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="importance_score",
                            range=rest.Range(gte=min_importance)
                        )
                    ]
                ),
                limit=top_k
            )
            
            memories = []
            for hit in results:
                memories.append({
                    "score": hit.score,
                    "importance": hit.payload.get("importance_score"),
                    "pnl": hit.payload.get("pnl_pct"),
                    "reasoning": hit.payload.get("reasoning"),
                    "outcome": hit.payload.get("outcome")
                })
            
            return memories

        except Exception as e:
            logger.error("memory_retrieval_failed", error=str(e))
            return []
