"""
Evolutionary Kernel — Biological adaptation for agent swarms.
"""

import copy
import random
import uuid
import numpy as np
from typing import Dict, List, Any, Optional
from nexus_alpha.log_config import get_logger

logger = get_logger(__name__)

class EvolutionaryKernel:
    """
    Biological Engine for Agent Autonomy.
    Facilitates soft-mutations and genetic refinement.
    """
    
    def __init__(self, mutation_rate: float = 0.05, stability_bounds: dict = None):
        self.mutation_rate = mutation_rate
        self.stability_bounds = stability_bounds or {
            "threshold": (0.01, 0.99),
            "window": (5, 1000),
            "z_threshold": (1.5, 6.0)
        }
        logger.info("evolutionary_kernel_initialized", mutation_rate=mutation_rate)

    def mutate_agent(self, agent: Any) -> bool:
        """
        Perform a Gaussian drift on an agent's genome.
        Returns True if mutation was successful.
        """
        genome = agent.get_genome()
        if not genome:
            return False
            
        new_genome = copy.deepcopy(genome)
        mutated_keys = []
        
        for key, value in new_genome.items():
            if isinstance(value, (int, float)):
                # Apply Gaussian mutation
                drift = np.random.normal(0, self.mutation_rate * abs(value))
                mutated_value = value + drift
                
                # Check safe bounds if defined for this key-type
                if "threshold" in key:
                    mutated_value = np.clip(mutated_value, *self.stability_bounds["threshold"])
                elif "window" in key or "lookback" in key:
                    mutated_value = int(np.clip(mutated_value, *self.stability_bounds["window"]))
                
                new_genome[key] = mutated_value
                mutated_keys.append(key)
        
        if mutated_keys:
            agent.set_genome(new_genome)
            logger.info(
                "agent_mutated", 
                agent_id=agent.agent_id, 
                keys=mutated_keys, 
                mutation_magnitude=f"{self.mutation_rate:.3f}"
            )
            return True
            
        return False

    def evolve_swarm(self, agents: List[Any], performance: Dict[str, Any]) -> int:
        """
        Identify candidates for soft-evolution.
        Top 20% agents are protected.
        Bottom 20% are candidates for cull (handled by Tournament).
        Middle 60% are candidates for soft-mutation.
        """
        if not agents or not performance:
            return 0
            
        # Rank by sharpe ratio or total return
        # We target the 'Middle Class' of agents for refinement
        agent_list = [a for a in agents if a.agent_id in performance]
        ranked = sorted(agent_list, key=lambda a: performance[a.agent_id].sharpe_ratio)
        
        # Mutation target: 30% to 70% percentile
        start_idx = int(len(ranked) * 0.3)
        end_idx = int(len(ranked) * 0.7)
        mutation_targets = ranked[start_idx:end_idx]
        
        count = 0
        for agent in mutation_targets:
            if random.random() < 0.5: # 50% chance of mutation per cycle
                if self.mutate_agent(agent):
                    count += 1
                    
        return count

class RecursiveSpawner:
    """
    Proliferation Engine for Elastic Swarms.
    Creates new lineage variations from top-performing champions.
    """
    
    def __init__(self, kernel: EvolutionaryKernel, max_agents: int = 20):
        self.kernel = kernel
        self.max_agents = max_agents
        logger.info("recursive_spawner_active", max_agents=max_agents)

    def bootstrap_variations(self, champion: Any, count: int = 2) -> List[Any]:
        """
        Produce transient clones of a champion with Gaussian genome noise.
        """
        new_agents = []
        for i in range(count):
            # We assume agent has a standard constructor that can be inferred or uses clone()
            # For FusionEnsembleAgent, we just instantiate a new one with same base params
            try:
                # Deep copy would miss complex engine state, better to re-init with same genome
                current_genome = champion.get_genome()
                
                # Create a variation
                new_id = f"bootstrap-{champion.agent_id[:8]}-{uuid.uuid4().hex[:4]}"
                
                # We need the class to instantiate
                agent_class = champion.__class__
                
                # Basic re-init (StrategyAgentLifecycle.bootstrap pattern)
                variation = agent_class(
                    agent_id=new_id,
                    symbol=getattr(champion, "symbol", "BTCUSDT"),
                    leader_id=getattr(champion, "leader_id", None)
                )
                
                # V8 Phase 17: Propagate Lineage
                variation.lineage_depth = champion.lineage_depth + 1
                variation.ancestor_id = getattr(champion, "ancestor_id", champion.agent_id)
                
                # Apply champion genome
                variation.set_genome(current_genome)
                
                # Mutate slightly (higher noise for bootstrapping)
                self.kernel.mutate_agent(variation)
                
                # Tag as transient for tournament pruning
                if hasattr(variation, "metadata"):
                    variation.metadata["transient"] = True
                    variation.metadata["lineage"] = champion.agent_id
                
                new_agents.append(variation)
                logger.info("recursive_bootstrap_spawned", parent=champion.agent_id, child=new_id)
            except Exception:
                logger.exception("bootstrap_spawn_failed", champion_id=champion.agent_id)
                
        return new_agents

    def spawn_hedge_agent(self, champion: Any, cluster_id: str) -> Any:
        """
        Create a dedicated Symmetrization Variant (Hedge).
        Flips the directional thesis of the champion to provide cluster-neutrality.
        """
        try:
            current_genome = champion.get_genome()
            new_id = f"hedge-{cluster_id}-{uuid.uuid4().hex[:4]}"
            
            agent_class = champion.__class__
            hedge = agent_class(
                agent_id=new_id,
                symbol=getattr(champion, "symbol", "BTCUSDT"),
                leader_id=getattr(champion, "leader_id", None),
                cluster_id=cluster_id
            )
            
            # Invert the genome weights (simple hedge)
            # Higher-order v9 would use LLM for more nuance
            hedge_genome = {
                key: -val if isinstance(val, (int, float)) and "weight" in key else val 
                for key, val in current_genome.items()
            }
            hedge.set_genome(hedge_genome)
            
            # Register metadata
            if hasattr(hedge, "metadata"):
                hedge.metadata["hedge"] = True
                hedge.metadata["cluster"] = cluster_id
                
            logger.info("symmetry_hedge_spawned", cluster=cluster_id, agent_id=new_id)
            return hedge
        except Exception:
            logger.exception("hedge_spawn_failed", cluster=cluster_id)
            return None
