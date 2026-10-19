import copy
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from verireason.data_pipeline import Example
from verireason.m2_diagnostics import (
    DiagnosticItem,
    build_public_report,
    load_diagnostic_config,
    require_expected_selection,
    require_private_output_directory,
    run_diagnostics,
    select_failed_queries,
    verify_input_evidence,
)
from verireason.sql_evaluator import EvaluatorLimits, QueryResult


def _database(root: Path, db_id: str) -> None:
    directory = root / db_id
    directory.mkdir(parents=True)
    connection = sqlite3.connect(directory / f"{db_id}.sqlite")
    connection.execute("CREATE TABLE values_table(value INTEGER)")
    connection.close()


def _private_preflight() -> dict[str, object]:
    return {
        "schema_version": 1,
        "training": {
            "failures": [
                {
                    "db_id": "train_db",
                    "example_ids": ["train-cpu"],
                    "status": "execution_error",
                    "error": "worker IPC failed (EOFError); exit code -24",
                },
                {
                    "db_id": "train_db",
                    "example_ids": ["train-row"],
                    "status": "resource_limit",
                    "error": "result row limit exceeded",
                },
            ]
        },
        "heldout": {
            "failures": [
                {
                    "db_id": "secret_db",
                    "example_ids": ["secret-id"],
                    "status": "execution_error",
                    "error": "worker IPC failed (EOFError); exit code None",
                }
            ]
        },
    }


def _examples() -> dict[str, list[Example]]:
    return {
        "training": [
            Example("train-cpu", "train_db", "cpu", "SELECT 1"),
            Example("train-row", "train_db", "rows", "SELECT 2"),
            Example("not-failed", "train_db", "ignored", "SELECT 3"),
        ],
        "heldout": [Example("secret-id", "secret_db", "heldout", "SELECT 4")],
    }


def _source_identity() -> dict[str, str]:
    return dict(load_diagnostic_config("configs/m2_diagnostic.toml")["evidence"])


def test_diagnostic_profiles_are_separate_from_frozen_evaluator_config() -> None:
    historical_formal = Path("configs/m2_data_v1_3s.toml").read_bytes()
    amended_formal = Path("configs/m2_data.toml").read_bytes()
    config = load_diagnostic_config("configs/m2_diagnostic.toml")

    assert hashlib.sha256(historical_formal).hexdigest() == (
        "d468ca938964597832ce5fb4c23b452fa362e8b6770c9d73a0de80d6c669598b"
    )
    assert hashlib.sha256(amended_formal).hexdigest() == (
        "1fa045ebf72a29d9209f5b37f1042126a81e66e3f7511b1c1fc8a8b21d553181"
    )
    assert config["diagnostic_only"] is True
    assert config["maximum_session_wall_seconds"] == 5400
    assert config["profile_a"]["maximum_cpu_hard_seconds"] == 13
    assert config["profile_b"]["maximum_cpu_hard_seconds"] == 33


def test_diagnostic_selection_count_gate_rejects_unexpected_input() -> None:
    items = select_failed_queries(_private_preflight(), _examples())
    config = load_diagnostic_config("configs/m2_diagnostic.toml")

    with pytest.raises(ValueError, match="selection count changed"):
        require_expected_selection(items, config)


def test_diagnostic_selection_checkpoint_resume_and_public_redaction(tmp_path: Path) -> None:
    items = select_failed_queries(_private_preflight(), _examples())
    assert len(items) == 3
    assert {item.profile for item in items} == {"profile_a", "profile_b"}
    assert all(item.example_ids != ("not-failed",) for item in items)

    training_root = tmp_path / "training"
    heldout_root = tmp_path / "heldout"
    _database(training_root, "train_db")
    _database(heldout_root, "secret_db")
    calls: list[str] = []

    def fake_executor(
        _database_path: str | Path,
        sql: str,
        limits: EvaluatorLimits,
        *,
        collect_rows: bool,
    ) -> QueryResult:
        assert not collect_rows
        calls.append(sql)
        if limits.maximum_result_rows == 1_000_000:
            return QueryResult("ok", row_count=300_000, duration_seconds=6, worker_state="exited")
        return QueryResult("ok", row_count=1, duration_seconds=4, worker_state="exited")

    checkpoint_path = tmp_path / "diagnostic-checkpoint-private.json"
    config = load_diagnostic_config("configs/m2_diagnostic.toml")
    checkpoint = run_diagnostics(
        items,
        {"training": training_root, "heldout": heldout_root},
        config,
        checkpoint_path,
        source_identity=_source_identity(),
        executor=fake_executor,
    )
    assert len(calls) == 3
    assert checkpoint_path.stat().st_mode & 0o777 == 0o600

    resumed = run_diagnostics(
        items,
        {"training": training_root, "heldout": heldout_root},
        config,
        checkpoint_path,
        source_identity=_source_identity(),
        executor=fake_executor,
    )
    assert resumed["results"].keys() == checkpoint["results"].keys()
    assert len(calls) == 3

    public = build_public_report(checkpoint, items)
    serialized = json.dumps(public, sort_keys=True)
    assert "SELECT" not in serialized
    assert "secret-id" not in serialized
    assert "secret_db" not in serialized
    assert public["plan_sha256"] == checkpoint["plan_sha256"]
    assert public["complete"] is True
    assert public["completed_count"] == public["expected_count"] == 3
    assert public["input_evidence_sha256"] == _source_identity()
    assert public["splits"]["training"]["cpu_ipc"]["original_failure_count"] == 1
    assert public["splits"]["training"]["cpu_ipc"]["duration_buckets"]["gt_3_le_5_seconds"] == 1
    assert public["splits"]["training"]["large_result"]["250k_to_500k"] == 1
    assert public["splits"]["heldout"]["cpu_ipc"]["unstable_exitcode"] == 0


@pytest.mark.parametrize(
    ("identity_name", "path_name"),
    [
        ("private_preflight_sha256", "private_preflight"),
        ("training_labels_sha256", "training_labels"),
        ("heldout_labels_sha256", "heldout_labels"),
        ("formal_config_sha256", "formal_config"),
        ("source_receipt_sha256", "source_receipt"),
    ],
)
def test_each_input_evidence_hash_mismatch_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_name: str,
    path_name: str,
) -> None:
    config = load_diagnostic_config("configs/m2_diagnostic.toml")
    paths = {
        name: tmp_path / name
        for name in (
            "private_preflight",
            "training_labels",
            "heldout_labels",
            "formal_config",
            "source_receipt",
        )
    }
    expected_by_path = {
        str(paths[name]): config["evidence"][identity]
        for identity, name in (
            ("private_preflight_sha256", "private_preflight"),
            ("training_labels_sha256", "training_labels"),
            ("heldout_labels_sha256", "heldout_labels"),
            ("formal_config_sha256", "formal_config"),
            ("source_receipt_sha256", "source_receipt"),
        )
    }
    monkeypatch.setattr(
        "verireason.m2_diagnostics.sha256_file",
        lambda path: "0" * 64
        if str(path) == str(paths[path_name])
        else expected_by_path[str(path)],
    )

    with pytest.raises(ValueError, match=path_name):
        verify_input_evidence(config, paths)
    assert config["evidence"][identity_name] != "0" * 64


def test_correct_selection_count_does_not_bypass_evidence_hash_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    items = (
        tuple(
            DiagnosticItem(str(index), "training", "profile_a", "db", (str(index),), "SELECT 1")
            for index in range(220)
        )
        + tuple(
            DiagnosticItem(f"b{index}", "training", "profile_b", "db", (str(index),), "SELECT 1")
            for index in range(22)
        )
        + tuple(
            DiagnosticItem(f"h{index}", "heldout", "profile_a", "db", (str(index),), "SELECT 1")
            for index in range(2)
        )
    )
    config = load_diagnostic_config("configs/m2_diagnostic.toml")
    require_expected_selection(items, config)
    paths = {
        name: tmp_path / name
        for name in (
            "private_preflight",
            "training_labels",
            "heldout_labels",
            "formal_config",
            "source_receipt",
        )
    }
    monkeypatch.setattr("verireason.m2_diagnostics.sha256_file", lambda _path: "0" * 64)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_input_evidence(config, paths)


def test_output_directory_rejects_git_tree_and_symlink_back_into_it(tmp_path: Path) -> None:
    git_root = tmp_path / "repo"
    partial = tmp_path / "data" / "m2-v1.partial"
    outside = tmp_path / "diagnostics"
    git_root.mkdir()
    partial.mkdir(parents=True)

    with pytest.raises(ValueError, match="Git worktree"):
        require_private_output_directory(
            git_root / "private", git_worktree=git_root, materialization_partial=partial
        )
    link = outside
    link.symlink_to(git_root, target_is_directory=True)
    with pytest.raises(ValueError, match="Git worktree"):
        require_private_output_directory(
            link / "private", git_worktree=git_root, materialization_partial=partial
        )
    with pytest.raises(ValueError, match="materialization"):
        require_private_output_directory(
            partial / "diagnostics", git_worktree=git_root, materialization_partial=partial
        )


def test_checkpoint_resume_rejects_non_private_mode_and_changed_source_identity(
    tmp_path: Path,
) -> None:
    training_root = tmp_path / "training"
    _database(training_root, "train_db")
    item = DiagnosticItem("one", "training", "profile_a", "train_db", ("1",), "SELECT 1")
    config = load_diagnostic_config("configs/m2_diagnostic.toml")
    checkpoint_path = tmp_path / "checkpoint.json"
    result = QueryResult("ok", worker_state="exited")

    run_diagnostics(
        [item],
        {"training": training_root},
        config,
        checkpoint_path,
        source_identity=_source_identity(),
        executor=lambda *_args, **_kwargs: result,
    )
    checkpoint_path.chmod(0o644)
    with pytest.raises(PermissionError, match="0600"):
        run_diagnostics(
            [item],
            {"training": training_root},
            config,
            checkpoint_path,
            source_identity=_source_identity(),
            executor=lambda *_args, **_kwargs: result,
        )
    checkpoint_path.chmod(0o600)
    changed_identity = _source_identity()
    changed_identity["source_receipt_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="plan differs"):
        run_diagnostics(
            [item],
            {"training": training_root},
            config,
            checkpoint_path,
            source_identity=changed_identity,
            executor=lambda *_args, **_kwargs: result,
        )


def test_session_wall_gate_checkpoints_and_public_report_stays_incomplete(tmp_path: Path) -> None:
    training_root = tmp_path / "training"
    _database(training_root, "train_db")
    items = (
        DiagnosticItem("one", "training", "profile_a", "train_db", ("1",), "SELECT 1"),
        DiagnosticItem("two", "training", "profile_a", "train_db", ("2",), "SELECT 2"),
    )
    clock = iter((0.0, 0.0, 5400.0))
    calls = 0

    def executor(*_args: object, **_kwargs: object) -> QueryResult:
        nonlocal calls
        calls += 1
        return QueryResult("ok", duration_seconds=1, worker_state="exited")

    checkpoint = run_diagnostics(
        items,
        {"training": training_root},
        load_diagnostic_config("configs/m2_diagnostic.toml"),
        tmp_path / "checkpoint.json",
        source_identity=_source_identity(),
        executor=executor,
        monotonic=lambda: next(clock),
    )
    assert calls == 1
    assert checkpoint["complete"] is False
    assert checkpoint["stop_reason"] == "maximum_session_wall_seconds"
    report = build_public_report(checkpoint, items)
    assert report["complete"] is False
    assert report["completed_count"] == 1
    assert report["expected_count"] == 2

    falsified = copy.deepcopy(checkpoint)
    falsified["complete"] = True
    assert build_public_report(falsified, items)["complete"] is False
