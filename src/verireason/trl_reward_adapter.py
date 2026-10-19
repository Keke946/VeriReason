"""Thin TRL callback boundary for the framework-independent M6 reward contract."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from verireason.reward_contract import RewardComponents, group_rewards


@dataclass(frozen=True)
class RewardInput:
    """One completion plus the aligned prompt row supplied by TRL."""

    prompt: str
    completion: str
    completion_ids: tuple[int, ...]
    metadata: Mapping[str, Any]


ComponentScorer = Callable[[RewardInput], RewardComponents]


class TRLGroupRewardAdapter:
    """Validate TRL row alignment and return frozen E2/E3/E4 group rewards.

    TRL 1.9 repeats each prompt row ``num_generations`` times in contiguous order before calling a
    custom reward function. ``group_id_field`` is an explicit dataset identity; prompt text is not
    used as identity because two examples may legitimately share a prompt.
    """

    def __init__(
        self,
        *,
        arm: str,
        num_generations: int,
        component_scorer: ComponentScorer,
        metadata_fields: Sequence[str],
        group_id_field: str = "reward_group_id",
    ) -> None:
        if arm not in {"E2", "E3", "E4"}:
            raise ValueError(f"unknown reward arm: {arm!r}")
        if num_generations < 2:
            raise ValueError("num_generations must be at least two")
        if isinstance(metadata_fields, str):
            raise TypeError("metadata_fields must be a sequence of field names")
        fields = tuple(metadata_fields)
        if not fields or len(set(fields)) != len(fields):
            raise ValueError("metadata_fields must be non-empty and unique")
        if group_id_field in fields:
            raise ValueError("group identity must not be duplicated in metadata_fields")
        self.arm = arm
        self.num_generations = num_generations
        self.component_scorer = component_scorer
        self.metadata_fields = fields
        self.group_id_field = group_id_field

    def __call__(
        self,
        *,
        prompts: Sequence[str],
        completions: Sequence[str],
        completion_ids: Sequence[Sequence[int]],
        **kwargs: Any,
    ) -> list[float]:
        size = len(completions)
        if size == 0 or size % self.num_generations:
            raise ValueError("completion count must contain complete generation groups")
        if len(prompts) != size or len(completion_ids) != size:
            raise ValueError("TRL prompt/completion/token rows are misaligned")

        required = (self.group_id_field, *self.metadata_fields)
        columns: dict[str, Sequence[Any]] = {}
        for field in required:
            values = kwargs.get(field)
            if not isinstance(values, Sequence) or isinstance(values, str) or len(values) != size:
                raise ValueError(f"TRL metadata column is missing or misaligned: {field}")
            columns[field] = values

        identities = columns[self.group_id_field]
        seen: set[Hashable] = set()
        output: list[float] = []
        for start in range(0, size, self.num_generations):
            stop = start + self.num_generations
            group_ids = identities[start:stop]
            identity = group_ids[0]
            if not isinstance(identity, Hashable) or any(item != identity for item in group_ids):
                raise ValueError("TRL generation group identity is not contiguous and constant")
            if identity in seen:
                raise ValueError("TRL generation group identity appears in multiple blocks")
            seen.add(identity)

            components: list[RewardComponents] = []
            for index in range(start, stop):
                prompt = prompts[index]
                completion = completions[index]
                ids = completion_ids[index]
                if not isinstance(prompt, str) or not isinstance(completion, str):
                    raise TypeError("M6 supports TRL standard string prompts and completions only")
                if not isinstance(ids, Sequence) or isinstance(ids, str):
                    raise TypeError("completion_ids must be token sequences")
                if any(not isinstance(token, int) or isinstance(token, bool) for token in ids):
                    raise TypeError("completion_ids must contain integer token IDs")
                metadata = {field: columns[field][index] for field in self.metadata_fields}
                component = self.component_scorer(
                    RewardInput(
                        prompt=prompt,
                        completion=completion,
                        completion_ids=tuple(ids),
                        metadata=metadata,
                    )
                )
                if not isinstance(component, RewardComponents):
                    raise TypeError("component_scorer must return RewardComponents")
                components.append(component)
            output.extend(group_rewards(components, arm=self.arm))
        return output
