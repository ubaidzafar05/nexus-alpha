"""
Multi-Modal Data Fusion — Cross-modal attention network.

Different data modalities have different temporal resolutions,
noise levels, and predictive horizons. The fusion layer handles
this heterogeneity via cross-attention.

Modalities:
- Market (tick-level, continuous)
- On-chain (block-level, 10-15min resolution)
- Sentiment (15min-1H resolution)
- Macro (daily resolution)
- Options (daily resolution)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nexus_alpha.logging import get_logger

logger = get_logger(__name__)


class ModalityEncoder(nn.Module):
    """Encodes a single modality into the shared latent space."""

    def __init__(self, input_dim: int, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention: one modality 'queries' another.
    Allows market data to ask questions of sentiment data, etc.
    """

    def __init__(self, latent_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            latent_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        query attends to key_value.
        Returns (attended output, attention weights).
        """
        attn_out, attn_weights = self.attention(query, key_value, key_value)
        output = self.layer_norm(query + self.dropout(attn_out))
        return output, attn_weights


class MultiModalFusionNetwork(nn.Module):
    """
    Cross-modal attention fusion of heterogeneous data streams.

    Each modality is embedded into a common latent space,
    then cross-attention allows modalities to 'ask questions'
    of each other. The final output is a fused representation
    that captures cross-modal dependencies.
    """

    # Default modality dimensions (overridable)
    DEFAULT_MODALITY_DIMS = {
        "market": 64,       # Price, volume, order book features
        "sentiment": 32,    # NLP sentiment scores
        "onchain": 32,      # Blockchain metrics
        "macro": 16,        # Macro indicators
        "options": 24,      # Options Greeks and flow
    }

    def __init__(
        self,
        modality_dims: dict[str, int] | None = None,
        latent_dim: int = 256,
        n_heads: int = 4,
        n_cross_attention_layers: int = 3,
        dropout: float = 0.1,
        output_dim: int = 128,
    ):
        super().__init__()
        self.modality_dims = modality_dims or self.DEFAULT_MODALITY_DIMS
        self.latent_dim = latent_dim

        # Per-modality encoders
        self.encoders = nn.ModuleDict({
            name: ModalityEncoder(dim, latent_dim, dropout)
            for name, dim in self.modality_dims.items()
        })

        # Regime conditioning
        self.regime_encoder = nn.Embedding(6, latent_dim)  # 6 regime types

        # Cross-attention layers (each modality attends to all others)
        self.cross_attention_layers = nn.ModuleList([
            nn.ModuleDict({
                f"{m1}_to_{m2}": CrossModalAttention(latent_dim, n_heads, dropout)
                for m1 in self.modality_dims
                for m2 in self.modality_dims
                if m1 != m2
            })
            for _ in range(n_cross_attention_layers)
        ])

        # Fusion head
        n_modalities = len(self.modality_dims)
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * n_modalities, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, output_dim),
        )

        logger.info(
            "multimodal_fusion_initialized",
            modalities=list(self.modality_dims.keys()),
            latent_dim=latent_dim,
            params=sum(p.numel() for p in self.parameters()),
        )

    def forward(
        self,
        modality_inputs: dict[str, torch.Tensor],
        regime_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass: encode each modality, cross-attend, fuse.

        Args:
            modality_inputs: Dict mapping modality name → tensor (batch, features)
            regime_idx: Optional regime index tensor (batch,) for conditioning

        Returns:
            (fused_output, attention_weights_dict)
        """
        # 1. Encode each modality into latent space
        encoded = {}
        for name, encoder in self.encoders.items():
            if name in modality_inputs:
                x = modality_inputs[name]
                if x.dim() == 2:
                    x = x.unsqueeze(1)  # Add sequence dim: (batch, 1, features)
                encoded[name] = encoder(x)
            else:
                # Zero-fill missing modalities
                batch_size = next(iter(modality_inputs.values())).shape[0]
                encoded[name] = torch.zeros(batch_size, 1, self.latent_dim, device=next(self.parameters()).device)

        # 2. Add regime conditioning if provided
        if regime_idx is not None:
            regime_emb = self.regime_encoder(regime_idx).unsqueeze(1)  # (batch, 1, latent)
            for name in encoded:
                encoded[name] = encoded[name] + regime_emb

        # 3. Cross-modal attention layers
        all_attention_weights = {}
        for layer in self.cross_attention_layers:
            updated = {}
            for name in encoded:
                # Each modality attends to all others
                attended_sum = encoded[name]
                for other_name in encoded:
                    if other_name != name:
                        key = f"{name}_to_{other_name}"
                        if key in layer:
                            attn_out, attn_w = layer[key](encoded[name], encoded[other_name])
                            attended_sum = attended_sum + attn_out
                            all_attention_weights[key] = attn_w
                updated[name] = attended_sum
            encoded = updated

        # 4. Concatenate all modality representations and fuse
        # Take last token from each modality
        representations = []
        for name in sorted(self.modality_dims.keys()):
            rep = encoded[name][:, -1, :]  # (batch, latent_dim)
            representations.append(rep)

        concat = torch.cat(representations, dim=-1)  # (batch, latent_dim * n_modalities)
        output = self.fusion(concat)  # (batch, output_dim)

        return output, all_attention_weights
