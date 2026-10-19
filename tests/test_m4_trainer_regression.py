"""Exact-runtime regression for the Transformers 5.14.1 custom-loss/GAS boundary."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest


def _exact_runtime():
    torch = pytest.importorskip(
        "torch",
        reason="M4 actual Trainer regression is unavailable: install the pinned cloud torch wheel",
    )
    transformers = pytest.importorskip(
        "transformers",
        reason="M4 actual Trainer regression is unavailable: install transformers==5.14.1",
    )
    if transformers.__version__ != "5.14.1":
        pytest.fail(
            "M4 actual Trainer regression requires transformers==5.14.1; "
            f"found {transformers.__version__}"
        )
    return torch, transformers


def _collate(torch, features):
    return {
        "input_ids": torch.tensor([[feature["input_id"]] for feature in features]),
        "labels": torch.tensor([[feature["target"]] for feature in features]),
        "m4_loss_scale": torch.tensor([feature["loss_scale"] for feature in features]),
    }


def test_regression_fixture_uses_the_formal_outer_checkpoint_layout(tmp_path):
    from verireason.m4_runtime import checkpoint_directory, train_formal_adapter

    outer_output = tmp_path / "m4-session"
    assert checkpoint_directory(outer_output) == outer_output / "checkpoints"
    assert "checkpoint_directory(output)" in inspect.getsource(train_formal_adapter)


@pytest.mark.parametrize(
    ("targets", "scales", "expected_weight"),
    [
        ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1.0] * 8, 0.9),
        ([1.0, 3.0, 5.0, 7.0, 1.0, 3.0, 5.0, 7.0], [2.0] * 4 + [0.0] * 4, 0.8),
    ],
    ids=("normal-eight-real-mean", "final-four-real-plus-four-zero-pad-mean"),
)
def test_transformers_514_training_step_matches_m4_gradient_means(
    tmp_path, targets, scales, expected_weight
):
    torch, transformers = _exact_runtime()
    from transformers import TrainingArguments

    from verireason.m4_runtime import (
        RequiredUpdateEvidenceCallback,
        build_remainder_aware_trainer,
        checkpoint_directory,
        required_update_observations,
    )

    class KwargsLossModel(torch.nn.Module):
        """A real torch module whose **kwargs triggers the 5.14.1 inference under audit."""

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))
            self.forward_calls = 0

        def forward(self, input_ids=None, labels=None, **kwargs):
            self.forward_calls += 1
            prediction = self.weight * input_ids.float()
            return SimpleNamespace(loss=(prediction - labels.float()).square().mean())

    source = inspect.getsource(transformers.Trainer.training_step)
    assert "model_accepts_loss_kwargs" in source
    assert "current_gradient_accumulation_steps" in source
    assert "not self.model_accepts_loss_kwargs or num_items_in_batch is None" in source
    model = KwargsLossModel()
    outer_output = tmp_path / "m4-session"
    arguments = TrainingArguments(
        output_dir=str(checkpoint_directory(outer_output)),
        use_cpu=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=1,
        learning_rate=0.1,
        max_grad_norm=10_000.0,
        save_strategy="steps",
        save_steps=999,
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=999,
        report_to="none",
        remove_unused_columns=False,
    )
    contract = {"test": "m4-transformers-5.14.1-loss-gas"}
    callback = RequiredUpdateEvidenceCallback({1}, contract)
    trainer = build_remainder_aware_trainer(
        model=model,
        args=arguments,
        train_dataset=[
            {"input_id": 1.0, "target": target, "loss_scale": scale}
            for target, scale in zip(targets, scales, strict=True)
        ],
        data_collator=lambda features: _collate(torch, features),
        callbacks=[callback.as_transformers_callback()],
    )
    assert trainer._m4_original_model_accepts_loss_kwargs is True
    assert trainer.model_accepts_loss_kwargs is False
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer.train()
    assert trainer.current_gradient_accumulation_steps == 8
    assert model.forward_calls == 8
    assert model.weight.item() == pytest.approx(expected_weight, abs=1e-6)
    observations = required_update_observations(
        trainer=trainer,
        output_dir=outer_output,
        required_updates={1},
        elapsed_seconds=callback.elapsed_seconds,
    )
    assert observations[0]["training_loss_window"] is not None
    assert observations[0]["gradient_norm"] is not None
    assert observations[0]["learning_rate"] is not None
    assert observations[0]["checkpoint_manifest_sha256"] is not None
