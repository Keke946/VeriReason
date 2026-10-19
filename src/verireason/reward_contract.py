"""Frozen M2 reward math; the SQL feature extractor remains M6 work."""

from __future__ import annotations

import math
from collections.abc import Collection, Hashable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardComponents:
    """Frozen per-completion inputs shared by E2, E3, and E4."""

    r_exec: int
    r_valid: int
    r_struct: float
    g_conf: int


def _set_f1(gold: Collection[Hashable], prediction: Collection[Hashable]) -> float:
    gold_set = frozenset(gold)
    prediction_set = frozenset(prediction)
    if not gold_set and not prediction_set:
        raise ValueError("inactive empty/empty dimensions do not have an F1 score")
    if not gold_set or not prediction_set:
        return 0.0
    return 2.0 * len(gold_set & prediction_set) / (len(gold_set) + len(prediction_set))


def active_dimension_structural_reward(
    gold_features: Mapping[str, Collection[Hashable]],
    prediction_features: Mapping[str, Collection[Hashable]],
    weights: Mapping[str, float],
) -> float:
    """Normalize F1 only across dimensions active in gold or prediction.

    Empty/empty dimensions contribute neither numerator nor denominator. If every dimension is
    empty, the conservative score is zero. Feature collections use set semantics here; M6 must
    declare any truly multiset-valued extractor dimensions before implementing them.
    """
    unknown = {
        key
        for key, values in (*gold_features.items(), *prediction_features.items())
        if values and key not in weights
    }
    if unknown:
        raise ValueError(f"unknown non-empty structural dimensions: {sorted(unknown)}")
    active = [
        key for key in weights if gold_features.get(key, ()) or prediction_features.get(key, ())
    ]
    if not active:
        return 0.0
    denominator = sum(weights[key] for key in active)
    if denominator <= 0:
        raise ValueError("active structural dimensions must have positive total weight")
    numerator = sum(
        weights[key] * _set_f1(gold_features.get(key, ()), prediction_features.get(key, ()))
        for key in active
    )
    return numerator / denominator


def raw_utility(
    *,
    r_exec: int,
    r_valid: int,
    r_struct: float,
    g_conf: int,
    w_valid: float = 0.05,
    w_struct: float = 0.20,
) -> float:
    if r_exec not in {0, 1} or r_valid not in {0, 1} or g_conf not in {0, 1}:
        raise ValueError("binary reward components must be zero or one")
    if not 0.0 <= r_struct <= 1.0:
        raise ValueError("r_struct must be in [0, 1]")
    return r_exec + g_conf * (w_valid * r_valid + w_struct * r_struct)


def population_zscores(values: Collection[float], *, epsilon: float = 1e-6) -> tuple[float, ...]:
    """Population-standardize one completion group; constant groups become all-zero."""
    observed = tuple(float(value) for value in values)
    if not observed:
        raise ValueError("reward group must not be empty")
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be positive and finite")
    if any(not math.isfinite(value) for value in observed):
        raise ValueError("reward values must be finite")
    mean = sum(observed) / len(observed)
    variance = sum((value - mean) ** 2 for value in observed) / len(observed)
    standard_deviation = math.sqrt(variance)
    if standard_deviation < epsilon:
        return tuple(0.0 for _ in observed)
    return tuple((value - mean) / standard_deviation for value in observed)


def group_rewards(
    components: Collection[RewardComponents],
    *,
    arm: str,
    w_valid: float = 0.05,
    w_struct: float = 0.20,
    epsilon: float = 1e-6,
) -> tuple[float, ...]:
    """Compute the frozen E2/E3/E4 group reward without training-framework coupling."""
    group = tuple(components)
    if not group:
        raise ValueError("reward group must not be empty")
    utilities = tuple(
        raw_utility(
            r_exec=item.r_exec,
            r_valid=item.r_valid,
            r_struct=item.r_struct,
            g_conf=item.g_conf,
            w_valid=w_valid,
            w_struct=w_struct,
        )
        for item in group
    )
    if arm == "E2":
        return tuple(float(item.r_exec) for item in group)
    if arm == "E3":
        return population_zscores(utilities, epsilon=epsilon)
    if arm != "E4":
        raise ValueError(f"unknown reward arm: {arm!r}")

    exec_z = population_zscores((item.r_exec for item in group), epsilon=epsilon)
    valid_z = population_zscores((item.g_conf * item.r_valid for item in group), epsilon=epsilon)
    struct_z = population_zscores((item.g_conf * item.r_struct for item in group), epsilon=epsilon)
    return tuple(
        exec_value + w_valid * valid_value + w_struct * struct_value
        for exec_value, valid_value, struct_value in zip(exec_z, valid_z, struct_z, strict=True)
    )
