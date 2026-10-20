"""SQL-free receipts for the E5 CPU interface and allowlisted training evidence."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from verireason.e5_credit import ExecutionCredit, clause_spans
from verireason.e5_diagnostics import CASES, create_database
from verireason.e5_interface import (
    advantage_matrix,
    mock_dapo_loss,
    qwen_token_credit,
    terminal_advantages,
)

MAPPING_SQL = (
    "SELECT id FROM sales WHERE amount > 20",
    "  SELECT  id\nFROM sales\n WHERE amount > 20  ",
    "select id from sales where amount>20",
    "SELECT extraordinarily_long_customer_identifier FROM sales WHERE amount > 20",
    "SELECT 'amount > 20', amount FROM sales WHERE amount > 20",
    "SELECT id FROM (SELECT id, amount FROM sales WHERE amount>1) WHERE amount>20",
    "SELECT s.id FROM sales AS s WHERE s.amount>20",
    "SELECT '猫' FROM sales WHERE amount>20",
)
TRAINING_SUMMARIES = (
    "results/m7/m7-e2-formal-summary.json",
    "results/m8/m8-e3-formal-summary.json",
    "results/m8/m8-e4-formal-summary.json",
)


def training_source_audit(root: Path) -> dict:
    """Read ONLY these three SQL-free training aggregates, never validation/heldout files.

    No dataset discovery, manifest parsing, database opening, or artifact recursion. Missing
    row-level inputs are blockers, not grounds to substitute aggregate counts for coverage.
    """
    sources = []
    for relative in TRAINING_SUMMARIES:
        summary = json.loads((root / relative).read_text())
        if (
            not summary["scope"].endswith("training_not_validation_accuracy")
            or summary.get("heldout_examples_loaded") != 0
            or summary.get("heldout_examples_evaluated") != 0
            or summary.get("validation_examples_evaluated", 0) != 0
        ):
            raise ValueError("allowlisted evidence is not training-only")
        sources.append(
            {
                "path": relative,
                "experiment_id": summary["experiment_id"],
                "training_prompts": summary["training_prompts"],
                "rollouts": summary["rollouts"],
                "row_level_sql_present": False,
            }
        )
    # Check only known archive locations from STATE/registry, without opening their contents.
    locations = ("data/processed", "artifacts/cloud-backup", "artifacts/m7-e2-formal-001")
    return {
        "sources_read": sources,
        "known_local_training_locations_exist": {p: (root / p).exists() for p in locations},
        "real_training_rows_examined": 0,
        "real_rollout_sql_examined": 0,
        "real_localization_coverage": None,
        "real_token_mapping_coverage": None,
        "validation_or_heldout_payloads_read": 0,
        "missing_inputs": [
            "frozen training-manifest.json with split=train membership",
            "selected training-only SQLite databases for strict execution and repair",
            "existing generated training SQL, original completion IDs, EOS/truncation metadata",
            "prepared training prompt/row identity needed to join those outputs to the manifest",
        ],
        "blocker": "Only training aggregates are available in this checkout. They establish "
        "historical runs, not clause attribution or real rollout/token coverage.",
    }


def trl_source_audit(wheel: Path) -> dict:
    """Inspect the pinned wheel as text/AST without importing its GPU/model dependency tree."""
    with ZipFile(wheel) as archive:
        metadata = archive.read("trl-1.9.0.dist-info/METADATA").decode()
        source = archive.read("trl/trainer/grpo_trainer.py").decode()
    if "\nVersion: 1.9.0\n" not in metadata:
        raise ValueError("requires the exact TRL 1.9.0 wheel")
    tree = ast.parse(source)
    method = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_compute_loss"
    )
    body = ast.get_source_segment(source, method)
    checks = {
        "two_dimensional_advantages_supported": "if advantages.dim() == 1:" in body,
        "scalar_advantages_only_unsqueezed": "advantages = advantages.unsqueeze(1)" in body,
        "clipped_token_objective": "per_token_loss = -torch.min(per_token_loss1, per_token_loss2)"
        in body,
        "dapo_original_denominator": 'inputs["num_items_in_batch"].clamp(min=1.0)' in body,
        "dapo_masked_sum": "loss = (per_token_loss * mask).sum() / normalizer" in body,
        "e2_center_without_extra_normalization": "advantages = rewards - mean_grouped_rewards"
        in source
        and 'if self.scale_rewards != "none":' in source,
    }
    if not all(checks.values()):
        raise ValueError("pinned TRL loss seam no longer matches the audited CPU mock")
    return {
        "version": "1.9.0",
        "wheel": wheel.name,
        "checks": checks,
        "method_line": method.lineno,
        "execution": "source_inspection_only; no TRL import or real GRPOTrainer execution",
        "seam": "after _generate_and_score_completions local row slicing, before buffered "
        "inputs enter _compute_loss; replace only advantages with (B,T), keep masks/denominator",
    }


def rollout_export_audit(root: Path) -> dict:
    """Inspect export keys, not production outputs or dataset rows."""
    runner = ast.parse((root / "scripts/run_m7_smoke_train.py").read_text())
    assignment = next(
        n
        for n in ast.walk(runner)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "evidence" for t in n.targets)
    )
    keys = [k.value for k in assignment.value.keys if isinstance(k, ast.Constant)]
    runtime = (root / "src/verireason/m7_runtime.py").read_text()
    return {
        "source_files": ["scripts/run_m7_smoke_train.py", "src/verireason/m7_runtime.py"],
        "exported_evidence_keys": keys,
        "original_completion_ids_exported": "completion_ids" in keys,
        "raw_generated_sql_exported": "completions" in keys,
        "completion_logging_explicitly_enabled": '"log_completions": True' in runtime,
        "interpretation": "Current runner exports adapter and aggregates, not a row-level "
        "SQL/token-ID replay bundle. Do not assume moving its archived logs will recover IDs; "
        "no separate private capture has been demonstrated here.",
    }


def mapping_audit(tokenizer) -> dict:
    from verireason.e5_credit import ClauseSpan

    eos = tokenizer.token_to_id("<|endoftext|>")
    observations = []
    for index, sql in enumerate(MAPPING_SQL):
        encoded = tokenizer.encode(sql, add_special_tokens=False)
        ids = tuple(encoded.ids) + (eos, eos)
        valid = (1,) * (len(ids) - 1) + (0,)
        span = next(s for s in clause_spans(sql) if s.clause == "WHERE")
        output = qwen_token_credit(
            ExecutionCredit(0, "geometry_only", span, sql),
            sql,
            ids,
            valid,
            tokenizer,
            eos_id=eos,
            terminal_advantage=-0.25,
        )
        observations.append(
            {
                "case_index": index,
                "token_count": len(encoded.ids),
                "target_token_count": sum(output.mask),
                "outcome": output.reason,
            }
        )
    sql = MAPPING_SQL[0]
    encoded = tokenizer.encode(sql, add_special_tokens=False)
    ids = tuple(encoded.ids) + (eos,)
    valid = (1,) * len(ids)
    span = next(s for s in clause_spans(sql) if s.clause == "WHERE")
    failures = []
    for kind in ("stale_text", "missing_eos", "truncated", "bpe_boundary", "noncanonical_ids"):
        text, row_ids, row_valid, target, truncated = sql, ids, valid, span, False
        if kind == "stale_text":
            text = sql.replace("20", "21")
        elif kind == "missing_eos":
            row_ids, row_valid = ids[:-1], valid[:-1]
        elif kind == "truncated":
            truncated = True
        elif kind == "bpe_boundary":
            target = ClauseSpan("SELECT", 2, 4)
        elif kind == "noncanonical_ids":
            text = "SELECT"
            row_ids = tuple(tokenizer.token_to_id(c) for c in text) + (eos,)
            row_valid, target = (1,) * len(row_ids), ClauseSpan("SELECT", 0, len(text))
        output = qwen_token_credit(
            ExecutionCredit(0, "geometry_only", target, text),
            text,
            row_ids,
            row_valid,
            tokenizer,
            eos_id=eos,
            terminal_advantage=-0.25,
            truncated=truncated,
        )
        failures.append(
            {
                "case": kind,
                "reason": output.reason,
                "exact_e2_fallback": output.advantages == (-0.25,) * len(row_ids),
            }
        )
    return {
        "geometry_cases": observations,
        "mapped_geometry_cases": sum(o["outcome"] == "localized" for o in observations),
        "uncertain_cases": failures,
        "uncertain_exact_fallback_count": sum(o["exact_e2_fallback"] for o in failures),
        "nested_alias_scope": "geometry only; semantic localizer still abstains",
        "real_generated_completion_ids_used": 0,
    }


def loss_audit() -> dict:
    from verireason.e5_credit import TokenCredit

    a = terminal_advantages((1, 0, 0, 0))
    masks = ((1,) * 6,) * 4
    credits = (TokenCredit((a[0],) * 6, (1,) * 6, "e2_terminal_only"),) + tuple(
        TokenCredit((0.0,) * 4 + (x, x), (0,) * 4 + (1, 1), "localized") for x in a[1:]
    )
    z = (0.2, -0.1, 0.4)
    logits, labels = ((z,) * 6,) * 4, ((0,) * 6,) * 4
    logp = z[0] - math.log(sum(math.exp(x) for x in z))
    old = ((logp,) * 6,) * 4
    modes = {}
    for mode in ("terminal", "local_unscaled", "local_mass_matched"):
        matrix = advantage_matrix(a, credits, masks, mode=mode)
        result = mock_dapo_loss(logits, labels, old, matrix, masks, num_items_in_batch=24)
        modes[mode] = {
            "loss": result.loss,
            "negative_credit_mass": result.negative_credit_mass,
            "logits_gradient_l1": sum(abs(x) for r in result.logits_gradient for t in r for x in t),
            "negative_logits_gradient_l1": sum(
                abs(x) for r in result.logits_gradient[1:] for t in r for x in t
            ),
        }
    errors = []
    for ratio in (0.6, 1.0, 1.5):
        for adv in (-0.25, 0.0, 0.75):

            def evaluate(values, ratio=ratio, adv=adv):
                return mock_dapo_loss(
                    ((values,),),
                    ((0,),),
                    ((logp - math.log(ratio),),),
                    ((adv,),),
                    ((1,),),
                    num_items_in_batch=7,
                )

            result = evaluate(z)
            for i in range(3):
                up, down = list(z), list(z)
                up[i] += 1e-6
                down[i] -= 1e-6
                numeric = (evaluate(tuple(up)).loss - evaluate(tuple(down)).loss) / 2e-6
                errors.append(abs(result.logits_gradient[0][0][i] - numeric))
    return {
        "kind": "analytical CPU logits mock; no model/autograd/optimizer run",
        "terminal_rewards": [1, 0, 0, 0],
        "terminal_advantages": list(a),
        "valid_tokens_per_row": 6,
        "localized_negative_tokens_per_row": 2,
        "unchanged_dapo_denominator": 24,
        "modes": modes,
        "finite_difference_coordinates": len(errors),
        "maximum_gradient_absolute_error": max(errors),
        "all_values_finite": all(math.isfinite(v) for m in modes.values() for v in m.values()),
    }


def synthetic_execution_mapping_audit(tokenizer) -> dict:
    """Bridge actual execution localization to real tokenizer IDs on authored SQL only."""
    from verireason.e5_credit import localize_execution

    eos = tokenizer.token_to_id("<|endoftext|>")
    candidates = mapped = forbidden = 0
    with TemporaryDirectory(prefix="verireason-e5-interface-") as directory:
        database = Path(directory) / "synthetic.sqlite"
        create_database(database)
        for case in CASES:
            credit = localize_execution(database, case.gold, case.prediction, split="train")
            ids = tuple(tokenizer.encode(case.prediction, add_special_tokens=False).ids) + (eos,)
            result = qwen_token_credit(
                credit,
                case.prediction,
                ids,
                (1,) * len(ids),
                tokenizer,
                eos_id=eos,
                terminal_advantage=-0.25,
            )
            candidates += int(credit.span is not None)
            mapped += int(result.reason == "localized")
            forbidden += int(credit.span is None and result.reason == "localized")
    return {
        "dataset": "hand_authored_synthetic_train_v1",
        "cases": len(CASES),
        "semantic_localized_candidates": candidates,
        "token_mapped_candidates": mapped,
        "abstention_overrides": forbidden,
        "scope": "synthetic only; generated IDs here are tokenizer encodings, not model rollouts",
    }
