"""Authored CPU fixtures only; no private replay, model weights or benchmark inputs."""

import copy
import json
import os
from pathlib import Path

import pytest

from verireason.e5_credit import ClauseSpan, ExecutionCredit, TokenCredit
from verireason.e5_interface import qwen_token_credit, terminal_advantages, verified_offsets
from verireason.e5_smoke import PrivateJournal, SmokeSession, validate_record

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("trl")
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import WhitespaceSplit  # noqa: E402
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from verireason.e5_backend import HFReplayBackend  # noqa: E402
from verireason.e5_trl import (  # noqa: E402
    E5ReplayTrainer,
    attach_tensor_advantages,
    replay_batches,
)


@pytest.fixture(scope="module")
def hf_qwen():
    path = Path("artifacts/e5-interface/qwen3-tokenizer/tokenizer.json")
    if not path.exists():
        pytest.skip("fixed-revision real Qwen tokenizer asset unavailable")
    return PreTrainedTokenizerFast(tokenizer_file=str(path), eos_token="<|endoftext|>",
                                   pad_token="<|endoftext|>")


@pytest.mark.parametrize("text", [
    "SELECT id FROM sales WHERE amount > 20",
    "select id\nfrom sales\nwhere amount > 20",
    "SELECT identifier_with_subwords FROM sales WHERE amount > 200",
    "SELECT 'WHERE' FROM sales WHERE amount > 20",
    "SELECT x.id FROM sales x WHERE x.amount > 20",
    "SELECT id FROM (SELECT id FROM sales) WHERE id > 20",
])
def test_actual_hf_wrapper_offsets(text, hf_qwen):
    ids = tuple(hf_qwen.encode(text, add_special_tokens=False)) + (hf_qwen.eos_token_id,)
    offsets, reason = verified_offsets(hf_qwen, text, ids, (1,) * len(ids),
                                       eos_id=hf_qwen.eos_token_id)
    assert reason == "verified_ids_and_text"
    assert len(offsets) == len(ids)
    assert offsets[-1] == (0, 0)


@pytest.mark.parametrize("kind", ["text", "ids", "eos", "truncated", "padding", "boundary"])
def test_hf_uncertainty_is_exact_e2_fallback(hf_qwen, kind):
    text = "SELECT id FROM sales WHERE amount > 20"
    ids = tuple(hf_qwen.encode(text, add_special_tokens=False)) + (hf_qwen.eos_token_id,)
    valid = (1,) * len(ids)
    end = len(text)
    span = ClauseSpan("WHERE", text.index("WHERE"), end)
    credit = ExecutionCredit(0, "synthetic", span, text)
    truncated = False
    if kind == "text":
        text += " "
    elif kind == "ids":
        ids = (0,) + ids[1:]
    elif kind == "eos":
        ids = ids[:-1] + (0,)
    elif kind == "truncated":
        truncated = True
    elif kind == "padding":
        valid = (0,) + valid[1:]
    elif kind == "boundary":
        credit = ExecutionCredit(0, "synthetic", ClauseSpan("SELECT", 1, 5), text)
    result = qwen_token_credit(credit, text, ids, valid, hf_qwen,
                              eos_id=hf_qwen.eos_token_id, terminal_advantage=-.25,
                              truncated=truncated)
    assert result.advantages == (-.25,) * len(ids)
    assert result.reason != "localized"


@pytest.fixture
def tokenizer():
    vocab = {x: i for i, x in enumerate([
        "<pad>", "<eos>", "<unk>", "SELECT", "id", "FROM", "sales", "WHERE", "x", "y",
    ])}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>",
                                   pad_token="<pad>", unk_token="<unk>")


def rows():
    return [dict(example_id=f"authored-{i}", split="train", operational_eligible=True,
                 source="synthetic", prompt="SELECT", gold_sql="SELECT id FROM sales WHERE y",
                 database_path="synthetic.sqlite") for i in range(8)]


CONTEXT = {"parent": "e1_m4_formal_sft_001", "fixture": "authored_cpu_only"}


class Backend:
    """Deterministic authored records for state-machine tests, not a generation claim."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.optimizer_steps = 0
        self.calls = []
        self.numerics = {"synthetic_fixture": True}

    def preflight(self, context):
        assert context == CONTEXT

    def capture_group(self, row, group):
        self.calls.append(group)
        result = []
        for j in range(4):
            text = "SELECT id FROM sales WHERE " + ("y" if j < 2 else "x")
            ids = self.tokenizer.encode(text, add_special_tokens=False) + [1]
            result.append(dict(example_id=row["example_id"], group=group, rollout=j,
                               prompt_ids=[3], completion_ids=ids, completion_mask=[1] * len(ids),
                               old_logps=[-2.3] * len(ids), old_logps_dtype="float32",
                               text=text, reward=int(j < 2),
                               truncated=False, eos_id=1, pad_id=0, terminal_status="ok"))
        return result

    def localize(self, record, row):
        if record["reward"]:
            return ExecutionCredit(1, "terminal_correct")
        text = record["text"]
        return ExecutionCredit(0, "synthetic_repair", ClauseSpan("WHERE", 26, len(text)), text)

    def check_numerics(self, records, credits):
        assert len(records) == len(credits) == 32

    def update(self, records, credits, session):
        for i in range(8):
            session.before_update(i)
            self.optimizer_steps += 1
            session.after_update(i, {"loss": .1, "gradient_norm": .2})

    def finish(self, directory):
        return {"adapter_delta_l2": .1, "independent_reload": True}


def review(session):
    return dict(reviewer="authored-test-reviewer",
                session_id=session.data["session_id"],
                candidate_rows=[x["row"] for x in session.data["audit"] if x["semantic"]],
                wrong_rows=[], unresolved_rows=[])


def captured(tmp_path, tokenizer):
    s = SmokeSession(tmp_path / "private", rows(), CONTEXT)
    b = Backend(tokenizer)
    assert s.run(b)["phase"] == "awaiting_review"
    return s, b


def test_gate_pass_updates_only_after_all_capture_and_review(tmp_path, tokenizer):
    s, b = captured(tmp_path, tokenizer)
    assert b.calls == list(range(8))
    assert b.optimizer_steps == s.data["optimizer_steps"] == 0
    result = s.run(b, review(s))
    assert result["phase"] == "complete" and result["optimizer_steps"] == 8
    assert s.data["gate"]["coverage"] == 1
    assert len(b.calls) == 8
    with pytest.raises(RuntimeError):
        s.before_update(8)
    s.close()


def test_optimizer_before_gate_rejected(tmp_path, tokenizer):
    s = SmokeSession(tmp_path / "p", rows(), CONTEXT)
    with pytest.raises(RuntimeError, match="before gate"):
        s.before_update(0)
    s.close()


@pytest.mark.parametrize("bad", ["review", "coverage", "signal", "mapping", "useful"])
def test_failed_frozen_gate_means_zero_updates(tmp_path, tokenizer, bad):
    s, b = captured(tmp_path, tokenizer)
    r = review(s)
    if bad == "review":
        r["wrong_rows"] = [2]
    elif bad == "coverage":
        for a in s.data["audit"]:
            a["mapped"] = False
    elif bad == "signal":
        for record in s.data["records"]:
            record["reward"] = 0
    elif bad == "mapping":
        for a in s.data["audit"][8:]:
            a["mapped"] = False
    elif bad == "useful":
        for a in s.data["audit"]:
            a["useful"] = False
    s.decide(r)
    assert s.data["phase"] == "stopped" and s.data["optimizer_steps"] == 0
    assert b.optimizer_steps == 0
    s.close()


def test_capture_resume_does_not_regenerate(tmp_path, tokenizer):
    s, b = captured(tmp_path, tokenizer)
    s.close()
    s = SmokeSession(tmp_path / "private", rows(), CONTEXT)
    b2 = Backend(tokenizer)
    s.run(b2)
    assert b2.calls == []
    assert s.data["optimizer_steps"] == 0
    s.close()


@pytest.mark.parametrize("pending", [{"generation": 0}, {"update": 0}])
def test_uncertain_intent_never_replayed(tmp_path, pending):
    s = SmokeSession(tmp_path / "p", rows(), CONTEXT)
    s.data["pending"] = pending
    s.save()
    s.close()
    with pytest.raises(RuntimeError, match="uncertain"):
        SmokeSession(tmp_path / "p", rows(), CONTEXT)


def test_private_atomic_permissions_and_no_public_payload(tmp_path, tokenizer):
    s, b = captured(tmp_path, tokenizer)
    assert s.journal.path.stat().st_mode & 0o777 == 0o600
    assert s.journal.directory.stat().st_mode & 0o777 == 0o700
    text = json.dumps(s.summary())
    assert "SELECT" not in text and "authored-" not in text and "completion_ids" not in text
    s.close()
    with pytest.raises(ValueError, match="ignored"):
        PrivateJournal(Path.cwd() / "tests" / "private-replay-forbidden")


@pytest.mark.parametrize("split", ["validation", "heldout", "Mini-Dev"])
def test_nontraining_input_rejected_before_write(tmp_path, split):
    data = rows()
    data[0]["split"] = split
    with pytest.raises(ValueError, match="forbidden"):
        SmokeSession(tmp_path / "p", data, CONTEXT)
    assert not (tmp_path / "p").exists()


@pytest.mark.parametrize("error", [ValueError("nan"), torch.OutOfMemoryError("private SQL"),
                                   KeyboardInterrupt()])
def test_capture_exception_persists_safe_stop(tmp_path, tokenizer, error):
    s = SmokeSession(tmp_path / "p", rows(), CONTEXT)
    b = Backend(tokenizer)
    b.capture_group = lambda *a: (_ for _ in ()).throw(error)
    with pytest.raises(type(error)):
        s.run(b)
    saved = s.journal.read()
    assert saved["pending"] == {"generation": 0} and saved["optimizer_steps"] == 0
    assert "private SQL" not in json.dumps(s.summary())
    s.close()


def test_evidence_write_failure_prevents_generation(tmp_path, tokenizer, monkeypatch):
    s = SmokeSession(tmp_path / "p", rows(), CONTEXT)
    b = Backend(tokenizer)
    monkeypatch.setattr(s.journal, "write", lambda d: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        s.run(b)
    assert b.calls == [] and b.optimizer_steps == 0
    s.close()


@pytest.mark.parametrize("field,value", [("old_logps", [float("nan")] * 7),
                                         ("old_logps", [float("inf")] * 7),
                                         ("group", 3), ("completion_mask", [1, 0])])
def test_misaligned_or_nonfinite_capture_rejected(tokenizer, field, value):
    b = Backend(tokenizer)
    r = b.capture_group(rows()[0], 0)[0]
    r[field] = value
    with pytest.raises(ValueError):
        validate_record(r, rows()[0], 0, 0)


def make_trainer(tmp_path, tokenizer, *, cls=E5ReplayTrainer):
    torch.manual_seed(23)
    config = Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         head_dim=8, max_position_embeddings=512,
                         eos_token_id=1, pad_token_id=0, attn_implementation="eager")
    model = get_peft_model(Qwen3ForCausalLM(config), LoraConfig(
        r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
    ))
    args = GRPOConfig(output_dir=str(tmp_path / "trainer"), use_cpu=True, bf16=False,
                      max_steps=8, per_device_train_batch_size=1, gradient_accumulation_steps=4,
                      generation_batch_size=4, num_generations=4, beta=0, loss_type="dapo",
                      scale_rewards="none", epsilon=.2, epsilon_high=.28, num_iterations=1,
                      learning_rate=5e-6, optim="adamw_torch", lr_scheduler_type="constant",
                      save_strategy="no", report_to="none", disable_tqdm=True,
                      gradient_checkpointing=False, mask_truncated_completions=True,
                      max_completion_length=384, shuffle_dataset=False, logging_steps=999,
                      temperature=1., top_p=1., top_k=0)
    return cls(model=model, processing_class=tokenizer, args=args,
               reward_funcs=lambda completions, **kw: [0.] * len(completions),
               train_dataset=Dataset.from_list([{"prompt": "SELECT"}] * 8))


def test_tensor_seam_preserves_every_other_object(tmp_path, tokenizer):
    s, b = captured(tmp_path, tokenizer)
    credit = tuple(TokenCredit(tuple(a["advantages"]), tuple(a["mask"]), a["token_reason"])
                   for a in s.data["audit"])
    batch = replay_batches(s.data["records"], credit, "cpu")[0]
    raw = {**batch, "advantages": torch.tensor([.5, .5, -.5, -.5])}
    mapped = attach_tensor_advantages(raw, credit[:4])
    for key in raw.keys() - {"advantages"}:
        assert mapped[key] is raw[key]
    assert mapped["advantages"].shape == batch["completion_ids"].shape
    s.close()


@pytest.mark.parametrize("ratio", [.6, 1., 1.5])
def test_real_trl_fallback_loss_and_autograd_exactly_e2(tmp_path, tokenizer, ratio):
    trainer = make_trainer(tmp_path, tokenizer)
    records = [r for i in range(8) for r in Backend(tokenizer).capture_group(rows()[i], i)]
    adv = terminal_advantages(tuple(r["reward"] for r in records))
    credits = tuple(TokenCredit((a,) * len(r["completion_ids"]),
                                (1,) * len(r["completion_ids"]), "fallback")
                    for r, a in zip(records, adv, strict=True))
    batch = replay_batches(records, credits, "cpu")[0]
    model = trainer.model
    model.eval()
    with torch.no_grad():
        current, _, _ = trainer._get_per_token_logps_and_entropies(
            model, torch.cat([batch["prompt_ids"], batch["completion_ids"]], 1),
            torch.cat([batch["prompt_mask"], batch["completion_mask"]], 1),
            batch["completion_ids"].shape[1])
    batch["old_per_token_logps"] = current - torch.log(torch.tensor(ratio))
    scalar = {**batch, "advantages": torch.tensor(adv[:4])}
    parameters = [p for p in model.parameters() if p.requires_grad]
    e2 = GRPOTrainer._compute_loss(trainer, model, scalar)
    e2_grad = torch.autograd.grad(e2, parameters)
    e5 = trainer._compute_loss(model, batch)
    e5_grad = torch.autograd.grad(e5, parameters)
    assert torch.equal(e2, e5)
    assert all(torch.equal(a, b) for a, b in zip(e2_grad, e5_grad, strict=True))
    assert torch.isfinite(e5) and all(torch.isfinite(g).all() for g in e5_grad)


def test_actual_trainer_optimizer_gate_and_full_cpu_update_reload(tmp_path, tokenizer):
    s, authored = captured(tmp_path, tokenizer)
    trainer = make_trainer(tmp_path, tokenizer)
    with pytest.raises(RuntimeError, match="before gate"):
        trainer.create_optimizer()
    backend = HFReplayBackend(trainer, tmp_path)
    backend.preflight(CONTEXT)
    credits = tuple(TokenCredit(tuple(a["advantages"]), tuple(a["mask"]), a["token_reason"])
                    for a in s.data["audit"])
    backend.check_numerics(s.data["records"], credits)
    assert all(all(torch.isfinite(torch.tensor(v)) for v in m.values())
               for m in backend.numerics.values())
    s.decide(review(s))
    backend.update(s.data["records"], credits, s)
    assert s.data["optimizer_steps"] == trainer.state.global_step == 8
    finish = backend.finish(s.journal.directory)
    assert finish["adapter_delta_l2"] > 0 and finish["independent_reload"]
    assert len(authored.calls) == 8
    s.close()


def test_real_hf_generate_ids_are_retained_before_decode(tmp_path, tokenizer, monkeypatch):
    import sqlite3

    database = tmp_path / "synthetic.sqlite"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE sales (id INTEGER, y INTEGER)")
    trainer = make_trainer(tmp_path, tokenizer)
    original = trainer.model.generate
    captured_ids = []

    def short_cpu_generate(**kwargs):
        kwargs["max_new_tokens"] = 3
        output = original(**kwargs)
        captured_ids.extend(output[:, kwargs["input_ids"].shape[1]:].tolist())
        return output

    monkeypatch.setattr(trainer.model, "generate", short_cpu_generate)
    backend = HFReplayBackend(trainer, tmp_path)
    generated = backend.capture_group(rows()[0], 0)
    assert [r["completion_ids"] for r in generated] == captured_ids
    assert len(generated) == 4 and trainer.state.global_step == 0 and trainer.optimizer is None


def test_partial_file_blocks_resume(tmp_path):
    s = SmokeSession(tmp_path / "p", rows(), CONTEXT)
    (s.journal.directory / "replay.partial").write_text("partial")
    s.close()
    with pytest.raises(RuntimeError, match="partial"):
        SmokeSession(tmp_path / "p", rows(), CONTEXT)


def test_completed_capture_group_resumes_at_next_group(tmp_path, tokenizer):
    s = SmokeSession(tmp_path / "p", rows(), CONTEXT)
    b = Backend(tokenizer)
    s.data["records"] = b.capture_group(rows()[0], 0)
    s.data["phase"] = "capturing"
    s.save()
    s.close()
    s = SmokeSession(tmp_path / "p", rows(), CONTEXT)
    b = Backend(tokenizer)
    s.run(b)
    assert b.calls == list(range(1, 8))
    assert s.data["optimizer_steps"] == 0
    s.close()


def test_update_intent_and_write_failure_never_repeat_step(tmp_path, tokenizer, monkeypatch):
    s, b = captured(tmp_path, tokenizer)
    s.decide(review(s))
    s.before_update(0)
    s.close()
    with pytest.raises(RuntimeError, match="uncertain"):
        SmokeSession(tmp_path / "private", rows(), CONTEXT)


def test_hf_eos_padding_and_repeated_text_target(hf_qwen):
    text = "SELECT 'x' FROM sales WHERE name = 'x'"
    ids = tuple(hf_qwen.encode(text, add_special_tokens=False)) + (hf_qwen.eos_token_id,)
    ids += (hf_qwen.pad_token_id,) * 2
    mask = (1,) * (len(ids) - 2) + (0, 0)
    c = ExecutionCredit(0, "synthetic", ClauseSpan("WHERE", text.index("WHERE"), len(text)), text)
    result = qwen_token_credit(c, text, ids, mask, hf_qwen,
                              eos_id=hf_qwen.eos_token_id, terminal_advantage=-.5)
    assert result.reason == "localized"
    assert result.mask[-3:] == (0, 0, 0)
    offsets, _ = verified_offsets(hf_qwen, text, ids, mask, eos_id=hf_qwen.eos_token_id)
    assert all(end > text.index("WHERE") for (start, end), m in
               zip(offsets, result.mask, strict=True) if m)


def test_real_trl_loss_local_gradient_targets_and_accumulation(tmp_path, tokenizer, monkeypatch):
    trainer = make_trainer(tmp_path, tokenizer)
    r = [record for i in range(8) for record in Backend(tokenizer).capture_group(rows()[i], i)]
    adv = terminal_advantages(tuple(x["reward"] for x in r))
    credits = tuple(TokenCredit(
        tuple(a if j in (4, 5) or a >= 0 else 0. for j in range(7)),
        tuple(int(j in (4, 5)) if a < 0 else 1 for j in range(7)),
        "localized" if a < 0 else "fallback",
    ) for a in adv)
    batch = replay_batches(r, credits, "cpu")[0]
    logits = torch.randn(4, 7, 16, dtype=torch.float64, requires_grad=True)
    current = logits.log_softmax(-1).gather(-1, batch["completion_ids"].unsqueeze(-1)).squeeze(-1)
    batch["old_per_token_logps"] = current.detach()
    trainer.model.train()

    def compute(model, inputs):
        trainer._e5_current_logps = inputs["test_logps"]
        return inputs["test_logps"], torch.zeros_like(inputs["test_logps"]), None

    monkeypatch.setattr(trainer, "_get_per_token_logps_and_entropies",
                        lambda *a, **k: compute(None, active))
    active = {**batch, "test_logps": current}
    e5 = trainer._compute_loss(trainer.model, batch)
    e5_grad = torch.autograd.grad(e5, logits, retain_graph=True)[0]
    active = {**batch, "test_logps": current}
    e2 = GRPOTrainer._compute_loss(trainer, trainer.model,
                                   {**batch, "advantages": torch.tensor(adv[:4])})
    e2_grad = torch.autograd.grad(e2, logits, retain_graph=True)[0]
    assert torch.equal(e5_grad[:2], e2_grad[:2])
    assert e5_grad[2:, [0, 1, 2, 3, 6]].count_nonzero() == 0
    assert torch.equal(e5_grad[2:, 4:6], e2_grad[2:, 4:6])
    parts = []
    for i in range(4):
        active = {**batch, "test_logps": current[i:i + 1]}
        micro = {k: v if v.ndim == 0 else v[i:i + 1] for k, v in batch.items()}
        parts.append(trainer._compute_loss(trainer.model, micro))
    accumulated = sum(parts)
    assert torch.allclose(e5, accumulated, atol=1e-12)
    assert torch.allclose(torch.autograd.grad(accumulated, logits)[0], e5_grad, atol=1e-12)


def test_actual_trainer_gas_parameter_updates_match_full_windows(tmp_path, tokenizer):
    s, b = captured(tmp_path, tokenizer)
    trainer = make_trainer(tmp_path, tokenizer)
    trainer.args.max_grad_norm = 1000.
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=.05)
    expected = copy.deepcopy(trainer.model)
    optimizer = torch.optim.SGD(expected.parameters(), lr=.05)
    credits = tuple(TokenCredit(tuple(a["advantages"]), tuple(a["mask"]), a["token_reason"])
                    for a in s.data["audit"])
    expected.train()
    trainer.model.train()
    for batch in replay_batches(s.data["records"], credits, "cpu"):
        optimizer.zero_grad()
        GRPOTrainer._compute_loss(trainer, expected, batch).backward()
        optimizer.step()
    s.decide(review(s))
    trainer.prepare_replay(s, s.data["records"], credits)
    trainer.train()
    diffs = [float((p.detach() - q.detach()).abs().max()) for p, q in
             zip(trainer.model.parameters(), expected.parameters(), strict=True)]
    assert max(diffs) < 1e-7
    assert s.data["optimizer_steps"] == 8
    s.close()


def test_private_schema_accepts_real_journal_rejects_nontrain(tmp_path, tokenizer):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(Path("results/schemas/e5-private-replay.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    s, b = captured(tmp_path, tokenizer)
    jsonschema.validate(s.journal.read(), schema)
    s.data["selection"][0]["split"] = "validation"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(s.data, schema)
    s.close()


def test_hf_alternative_generated_segmentation_abstains(hf_qwen):
    text = "SELECT"
    alt = tuple(t for char in text for t in hf_qwen.encode(char, add_special_tokens=False))
    assert hf_qwen.decode(alt, clean_up_tokenization_spaces=False) == text
    assert alt != tuple(hf_qwen.encode(text, add_special_tokens=False))
    ids = alt + (hf_qwen.eos_token_id,)
    offsets, reason = verified_offsets(hf_qwen, text, ids, (1,) * len(ids),
                                       eos_id=hf_qwen.eos_token_id)
    assert offsets is None and reason == "generated_segmentation_mismatch"


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_replay_retains_original_logp_dtype(tokenizer, dtype):
    records = [r for i in range(8) for r in Backend(tokenizer).capture_group(rows()[i], i)]
    for r in records:
        r["old_logps_dtype"] = dtype
    a = terminal_advantages(tuple(r["reward"] for r in records))
    credits = tuple(TokenCredit((v,) * 7, (1,) * 7, "fallback") for v in a)
    assert replay_batches(records, credits, "cpu")[0]["old_per_token_logps"].dtype == getattr(
        torch, dtype
    )


def test_wrong_session_review_stops_without_updates(tmp_path, tokenizer):
    s, b = captured(tmp_path, tokenizer)
    r = review(s)
    r["session_id"] = "a-different-session"
    s.run(b, r)
    assert s.data["phase"] == "stopped" and b.optimizer_steps == 0
    s.close()


def test_write_failure_before_optimizer_intent_prevents_update(tmp_path, tokenizer, monkeypatch):
    s, b = captured(tmp_path, tokenizer)
    s.decide(review(s))
    monkeypatch.setattr(s.journal, "write", lambda d: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        b.update(s.data["records"], (), s)
    assert b.optimizer_steps == 0
    s.close()


def test_nonfinite_model_logits_stop_before_loss_update(tmp_path, tokenizer, monkeypatch):
    trainer = make_trainer(tmp_path, tokenizer)
    r = [record for i in range(8) for record in Backend(tokenizer).capture_group(rows()[i], i)]
    a = terminal_advantages(tuple(x["reward"] for x in r))
    credits = tuple(TokenCredit((v,) * 7, (1,) * 7, "fallback") for v in a)
    batch = replay_batches(r, credits, "cpu")[0]
    batch["old_per_token_logps"][0, 0] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        trainer._compute_loss(trainer.model, batch)
    assert trainer.state.global_step == 0 and trainer.optimizer is None


def test_full_real_sqlite_hf_trl_cpu_state_machine(tmp_path, tokenizer, monkeypatch):
    """Synthetic SQL/token encodings drive real evaluator/audit/TRL, never real coverage."""
    import sqlite3

    with sqlite3.connect(tmp_path / "synthetic.sqlite") as db:
        db.execute("CREATE TABLE sales(id INTEGER, x INTEGER, y INTEGER)")
        db.executemany("INSERT INTO sales VALUES (?, ?, ?)", [(1, 0, 1), (2, 0, 1)])
    s = SmokeSession(tmp_path / "p", rows(), CONTEXT)
    trainer = make_trainer(tmp_path, tokenizer)
    backend = HFReplayBackend(trainer, tmp_path)
    authored = Backend(tokenizer)
    monkeypatch.setattr(backend, "capture_group", authored.capture_group)
    assert s.run(backend)["phase"] == "awaiting_review"
    assert s.data["optimizer_steps"] == 0
    s.close()
    s = SmokeSession(tmp_path / "p", rows(), CONTEXT)
    backend = HFReplayBackend(make_trainer(tmp_path / "resume", tokenizer), tmp_path)
    monkeypatch.setattr(backend, "capture_group", lambda *a: pytest.fail("duplicate generation"))
    assert s.run(backend, review(s))["phase"] == "complete"
    assert s.summary()["optimizer_steps"] == 8
    assert s.data["gate"]["W"] == s.data["gate"]["S"] == s.data["gate"]["M"] == 16
    assert authored.calls == list(range(8))
    receipt = os.environ.get("E5_CPU_RECEIPT")
    if receipt:
        import trl

        from verireason.m3_contract import atomic_write_json

        public = {
            "experiment_id": "e5_cpu_runner_001",
            "scope": "authored synthetic SQLite/IDs; random tiny Qwen3/LoRA; CPU only",
            "torch": torch.__version__, "transformers": transformers.__version__,
            "trl": trl.__version__, "device": "cpu", "base_model_weights_downloaded": 0,
            "capture": "authored synthetic record encodings, not real E1 generations",
            "optimizer_steps_before_gate": s.data["gate"]["optimizer_steps_before_gate"],
            "optimizer_steps": s.data["optimizer_steps"],
            "synthetic_gate_counts": {k: s.data["gate"][k] for k in ("W", "S", "M", "U")},
            "synthetic_gate_passed": s.data["gate"]["passed"],
            "adapter_delta_l2": s.data["finish"]["adapter_delta_l2"],
            "independent_adapter_hydration": s.data["finish"]["independent_reload"],
            "reload_scope": (
                "separate CPU process; LoRA hydrated, base architecture meta; no forward"
            ),
            "actual_trl_autograd_controls": s.data["finish"]["numerics"],
            "real_training_coverage": None, "validation_payloads_read": 0,
            "heldout_payloads_read": 0, "gpu_runs": 0, "cloud_cost_cny": 0,
            "decision": "CPU interface passes; real E1 capture/audit and launch approval pending",
        }
        atomic_write_json(receipt, public)
    s.close()
