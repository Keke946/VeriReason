"""CPU prototype of the Qwen-offset -> TRL (B,T) advantage loss seam.

The production trainer remains E2/E3/E4-only. The loss below is an analytical CPU mock
of the audited TRL 1.9.0 token-importance, beta=0 DAPO branch, not a model trainer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from verireason.e5_credit import ClauseSpan, ExecutionCredit, TokenCredit, token_credit
from verireason.reward_contract import RewardComponents, group_rewards


def verified_offsets(
    tokenizer: Any,
    completion: str,
    completion_ids: tuple[int, ...],
    valid_mask: tuple[int, ...],
    *,
    eos_id: int,
    truncated: bool = False,
) -> tuple[tuple[tuple[int, int], ...] | None, str]:
    """Bind original generated IDs to exact decoded text before trusting character offsets.

    Qwen backend tokenizers provide Python-character offsets. Re-encoding alone is insufficient:
    actual generation may use a different segmentation of the same string. Such IDs abstain.
    Only a contiguous valid prefix ending in one EOS is admitted; padding is never decoded.
    """
    if truncated or len(completion_ids) != len(valid_mask) or not completion_ids:
        return None, "truncated_or_shape"
    if any(m not in (0, 1) for m in valid_mask):
        return None, "invalid_completion_mask"
    count = sum(valid_mask)
    if valid_mask != (1,) * count + (0,) * (len(valid_mask) - count):
        return None, "noncontiguous_completion_mask"
    ids = completion_ids[:count]
    if not ids or ids[-1] != eos_id or eos_id in ids[:-1]:
        return None, "missing_or_interior_eos"
    text_ids = ids[:-1]
    try:
        if tokenizer.decode(list(text_ids), skip_special_tokens=False) != completion:
            return None, "decoded_text_mismatch"
        encoded = tokenizer.encode(completion, add_special_tokens=False)
        if tuple(encoded.ids) != text_ids:
            return None, "generated_segmentation_mismatch"
        offsets = tuple(tuple(pair) for pair in encoded.offsets)
    except (AttributeError, TypeError, ValueError):
        return None, "offset_api_unavailable"
    if len(offsets) != len(text_ids):
        return None, "offset_count_mismatch"
    return offsets + ((0, 0),) * (len(completion_ids) - len(text_ids)), "verified_ids_and_text"


def qwen_token_credit(
    credit: ExecutionCredit,
    completion: str,
    completion_ids: tuple[int, ...],
    valid_mask: tuple[int, ...],
    tokenizer: Any,
    *,
    eos_id: int,
    terminal_advantage: float,
    truncated: bool = False,
) -> TokenCredit:
    fallback = TokenCredit(
        (terminal_advantage,) * len(completion_ids),
        (1,) * len(completion_ids),
        "e2_terminal_only",
    )
    if not math.isfinite(terminal_advantage):
        raise ValueError("terminal advantage must be finite")
    if credit.span is None or credit.terminal_reward or terminal_advantage >= 0:
        return fallback
    offsets, reason = verified_offsets(
        tokenizer, completion, completion_ids, valid_mask, eos_id=eos_id, truncated=truncated
    )
    if offsets is None:
        return TokenCredit(fallback.advantages, fallback.mask, reason)
    # Clause spans include separator whitespace. Qwen may attach it to the following
    # keyword token (e.g. ' FROM'); do not mistake that separator for target SQL text.
    start, end = credit.span.start, credit.span.end
    if not 0 <= start < end <= len(completion):
        return fallback
    while start < end and completion[start].isspace():
        start += 1
    while end > start and completion[end - 1].isspace():
        end -= 1
    credit = replace(credit, span=ClauseSpan(credit.span.clause, start, end))
    return token_credit(credit, completion, offsets, terminal_advantage=terminal_advantage)


def terminal_advantages(rewards: tuple[int, ...], group_size: int = 4) -> tuple[float, ...]:
    """Exactly E2 binary rewards followed by TRL scale_rewards=none group centering."""
    if group_size < 2 or not rewards or len(rewards) % group_size:
        raise ValueError("complete terminal reward groups required")
    raw = group_rewards([RewardComponents(r, 1, 0.0, 0) for r in rewards], arm="E2")
    result = []
    for start in range(0, len(raw), group_size):
        group = raw[start : start + group_size]
        mean = sum(group) / group_size
        result.extend(r - mean for r in group)
    return tuple(result)


def advantage_matrix(
    terminal: tuple[float, ...],
    credits: tuple[TokenCredit, ...],
    valid_masks: tuple[tuple[int, ...], ...],
    *,
    mode: str = "local_unscaled",
) -> tuple[tuple[float, ...], ...]:
    """Keep the DAPO denominator unchanged; rescaling is an explicit diagnostic arm.

    local_unscaled: A*m. local_mass_matched: A*m*N/K for negative localized rows only.
    N/K preserves pre-clipping advantage L1 mass, NOT an arbitrary model gradient norm.
    It can amplify a tiny span; do not silently promote this control to a training default.
    """
    if mode not in {"terminal", "local_unscaled", "local_mass_matched"}:
        raise ValueError("unknown credit control")
    if not len(terminal) == len(credits) == len(valid_masks):
        raise ValueError("advantage rows are misaligned")
    output = []
    for a, credit, valid in zip(terminal, credits, valid_masks, strict=True):
        if not math.isfinite(a) or len(credit.advantages) != len(valid):
            raise ValueError("invalid advantage shape or value")
        if any(m not in (0, 1) for m in valid) or any(
            not math.isfinite(x) for x in credit.advantages
        ):
            raise ValueError("invalid completion mask or advantage")
        values = (a,) * len(valid) if mode == "terminal" else credit.advantages
        if mode == "local_mass_matched" and a < 0 and credit.reason == "localized":
            n = sum(valid)
            k = sum(m * v for m, v in zip(credit.mask, valid, strict=True))
            if not k:
                raise ValueError("localized credit has no valid target tokens")
            values = tuple(x * n / k for x in values)
        output.append(tuple(x * v for x, v in zip(values, valid, strict=True)))
    return tuple(output)


def attach_token_advantages(
    scored_inputs: dict[str, Any],
    credits: tuple[TokenCredit, ...],
    *,
    mode: str = "local_unscaled",
) -> dict[str, Any]:
    """CPU mock seam: replace only the already-centered local-row advantages.

    A future pinned-TRL subclass would convert this matrix with the original tensor's
    device/dtype after generation/scoring and before buffering. Do not recenter token rows,
    alter completion masks, or recompute num_items_in_batch from localized tokens.
    """
    matrix = advantage_matrix(
        tuple(scored_inputs["advantages"]),
        credits,
        tuple(scored_inputs["completion_mask"]),
        mode=mode,
    )
    return {**scored_inputs, "advantages": matrix}


@dataclass(frozen=True)
class MockLoss:
    loss: float
    logits_gradient: tuple[tuple[tuple[float, ...], ...], ...]
    negative_credit_mass: float


def mock_dapo_loss(
    logits: tuple[tuple[tuple[float, ...], ...], ...],
    labels: tuple[tuple[int, ...], ...],
    old_logps: tuple[tuple[float, ...], ...],
    advantages: tuple[tuple[float, ...], ...],
    completion_mask: tuple[tuple[int, ...], ...],
    *,
    num_items_in_batch: int,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.28,
) -> MockLoss:
    """One CPU process; denominator is original valid tokens over the accumulation window.

    No extra gradient-accumulation division: that would double-scale TRL's DAPO branch.
    Token logit gradients are analytical derivatives; tests compare numerical differentiation.
    """
    if num_items_in_batch <= 0 or not len(logits) == len(labels) == len(old_logps) == len(
        advantages
    ) == len(completion_mask):
        raise ValueError("invalid DAPO batch or denominator")
    loss = mass = 0.0
    gradients = []
    for row, targets, old, adv, mask in zip(
        logits, labels, old_logps, advantages, completion_mask, strict=True
    ):
        if not len(row) == len(targets) == len(old) == len(adv) == len(mask):
            raise ValueError("token rows are misaligned")
        row_gradient = []
        for z, target, old_logp, a, valid in zip(row, targets, old, adv, mask, strict=True):
            if (
                not z
                or not 0 <= target < len(z)
                or valid not in (0, 1)
                or any(not math.isfinite(x) for x in (*z, old_logp, a))
            ):
                raise ValueError("nonfinite or invalid loss input")
            peak = max(z)
            weights = tuple(math.exp(x - peak) for x in z)
            total = sum(weights)
            probabilities = tuple(x / total for x in weights)
            logp = z[target] - peak - math.log(total)
            try:
                ratio = math.exp(logp - old_logp)
            except OverflowError as error:
                raise ValueError("nonfinite importance ratio") from error
            clipped = min(max(ratio, 1 - epsilon_low), 1 + epsilon_high)
            loss -= min(ratio * a, clipped * a) * valid / num_items_in_batch
            active = not (
                (a > 0 and ratio > 1 + epsilon_high) or (a < 0 and ratio < 1 - epsilon_low)
            )
            coefficient = -a * ratio * valid / num_items_in_batch if active else 0.0
            row_gradient.append(
                tuple(coefficient * (int(i == target) - p) for i, p in enumerate(probabilities))
            )
            mass += max(-a, 0.0) * valid
        gradients.append(tuple(row_gradient))
    if not math.isfinite(loss) or any(
        not math.isfinite(x) for row in gradients for token in row for x in token
    ):
        raise ValueError("nonfinite loss or gradient")
    return MockLoss(loss, tuple(gradients), mass)
