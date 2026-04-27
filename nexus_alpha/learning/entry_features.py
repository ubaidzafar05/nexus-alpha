from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

META_SIGNAL_KEYS = {
    "ml_agreement",
    "ml_reward_overlay",
    "ml_confidence",
    "mtf_alignment",
    "pair_quality",
    "regime_mult",
    "symbol_learning",
}

CONTEXT_FEATURE_NAMES = [
    "ctx_signal_confidence",
    "ctx_pair_quality",
    "ctx_mtf_alignment",
    "ctx_regime_multiplier",
    "ctx_ml_confidence",
    "ctx_ml_alignment",
    "ctx_signal_agreement_ratio",
    "ctx_signal_dispersion",
    "ctx_signal_consensus_strength",
    "ctx_signal_opposition_strength",
    "ctx_directional_persistence_24",
    "ctx_volatility_compression",
    "ctx_stop_distance_pct",
    "ctx_breakeven_to_stop_ratio",
    "ctx_trailing_to_stop_ratio",
]


def _float_value(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def signal_context_features(
    contributing_signals: Mapping[str, object] | None,
    trade_direction: float,
    ml_signal: float = 0.0,
) -> dict[str, float]:
    usable = []
    for key, value in (contributing_signals or {}).items():
        if key in META_SIGNAL_KEYS:
            continue
        if isinstance(value, bool):
            continue
        numeric = _float_value(value, default=np.nan)
        if np.isnan(numeric):
            continue
        usable.append(numeric)

    if usable:
        arr = np.asarray(usable, dtype=np.float32)
        direction_sign = float(np.sign(trade_direction) or 0.0)
        aligned = np.abs(arr[np.sign(arr) == direction_sign]) if direction_sign != 0 else np.array([], dtype=np.float32)
        opposing = np.abs(arr[np.sign(arr) == -direction_sign]) if direction_sign != 0 else np.array([], dtype=np.float32)
        agreement_ratio = float(len(aligned) / len(arr))
        dispersion = float(np.std(arr))
        consensus_strength = float(np.mean(aligned)) if len(aligned) else 0.0
        opposition_strength = float(np.mean(opposing)) if len(opposing) else 0.0
    else:
        agreement_ratio = 0.5
        dispersion = 0.0
        consensus_strength = 0.0
        opposition_strength = 0.0

    direction_sign = float(np.sign(trade_direction) or 0.0)
    ml_alignment = 0.0
    if direction_sign != 0 and abs(ml_signal) > 1e-9:
        ml_alignment = 1.0 if np.sign(ml_signal) == direction_sign else -1.0

    return {
        "ctx_signal_agreement_ratio": round(float(np.clip(agreement_ratio, 0.0, 1.0)), 4),
        "ctx_signal_dispersion": round(float(np.clip(dispersion, 0.0, 1.5)), 4),
        "ctx_signal_consensus_strength": round(float(np.clip(consensus_strength, 0.0, 1.0)), 4),
        "ctx_signal_opposition_strength": round(float(np.clip(opposition_strength, 0.0, 1.0)), 4),
        "ctx_ml_alignment": ml_alignment,
    }


def projected_risk_features(
    entry_price: float,
    atr: float,
    sl_atr_mult: float,
    sl_floor_pct: float,
    sl_cap_pct: float,
    breakeven_trigger_pct: float,
    trailing_trigger_pct: float,
) -> dict[str, float]:
    if entry_price > 0 and atr > 0:
        atr_pct = atr / entry_price
        stop_distance_pct = min(max(sl_atr_mult * atr_pct, sl_floor_pct), sl_cap_pct)
    else:
        stop_distance_pct = sl_floor_pct

    breakeven_to_stop_ratio = breakeven_trigger_pct / stop_distance_pct if stop_distance_pct > 0 else 0.0
    trailing_to_stop_ratio = trailing_trigger_pct / stop_distance_pct if stop_distance_pct > 0 else 0.0
    return {
        "ctx_stop_distance_pct": round(float(np.clip(stop_distance_pct, 0.0, 0.25)), 4),
        "ctx_breakeven_to_stop_ratio": round(float(np.clip(breakeven_to_stop_ratio, 0.0, 10.0)), 4),
        "ctx_trailing_to_stop_ratio": round(float(np.clip(trailing_to_stop_ratio, 0.0, 10.0)), 4),
    }


def build_augmented_feature_vector(
    base_vector: Sequence[float],
    *,
    signal_confidence: float,
    pair_quality: float,
    mtf_alignment: float,
    regime_multiplier: float,
    ml_confidence: float,
    ml_signal: float,
    contributing_signals: Mapping[str, object] | None,
    directional_persistence_24: float,
    volatility_compression: float,
    entry_price: float,
    atr: float,
    sl_atr_mult: float,
    sl_floor_pct: float,
    sl_cap_pct: float,
    breakeven_trigger_pct: float,
    trailing_trigger_pct: float,
    trade_direction: float,
) -> tuple[list[float], dict[str, float]]:
    context = {
        "ctx_signal_confidence": round(float(np.clip(signal_confidence, 0.0, 1.0)), 4),
        "ctx_pair_quality": round(float(np.clip(pair_quality, 0.0, 1.0)), 4),
        "ctx_mtf_alignment": round(float(np.clip(mtf_alignment, 0.0, 1.0)), 4),
        "ctx_regime_multiplier": round(float(np.clip(regime_multiplier, 0.0, 2.0)), 4),
        "ctx_ml_confidence": round(float(np.clip(ml_confidence, 0.0, 1.0)), 4),
        "ctx_directional_persistence_24": round(float(np.clip(directional_persistence_24, -1.0, 1.0)), 4),
        "ctx_volatility_compression": round(float(np.clip(volatility_compression, -1.0, 1.0)), 4),
    }
    context.update(
        signal_context_features(
            contributing_signals=contributing_signals,
            trade_direction=trade_direction,
            ml_signal=ml_signal,
        )
    )
    context.update(
        projected_risk_features(
            entry_price=entry_price,
            atr=atr,
            sl_atr_mult=sl_atr_mult,
            sl_floor_pct=sl_floor_pct,
            sl_cap_pct=sl_cap_pct,
            breakeven_trigger_pct=breakeven_trigger_pct,
            trailing_trigger_pct=trailing_trigger_pct,
        )
    )

    augmented = [_float_value(value, default=0.0) for value in base_vector]
    augmented.extend(context[name] for name in CONTEXT_FEATURE_NAMES)
    return augmented, context


def augmented_feature_names(base_feature_names: Sequence[str]) -> list[str]:
    return [*base_feature_names, *CONTEXT_FEATURE_NAMES]
