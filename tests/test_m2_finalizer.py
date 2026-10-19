import hashlib
import json
import os
import sqlite3
import stat
from pathlib import Path

import pytest

from verireason.data_pipeline import DuplicateCounts, Example, LeakageReport
from verireason.m2_diagnostics import select_failed_queries
from verireason.m2_finalizer import (
    _hardlink_tree,
    _post_exclusion_report,
    derive_operational_eligibility,
    finalize_m2,
    load_finalizer_config,
    load_operational_training_examples,
    score_heldout_scopes,
    verify_finalizer_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def _key(split: str, example: Example) -> str:
    return hashlib.sha256(f"{split}\0{example.db_id}\0{example.sql}".encode()).hexdigest()


def _evidence_fixture() -> tuple[
    list[Example], list[Example], dict[str, object], dict[str, object]
]:
    training = [
        Example(str(index), "training_db", f"question {index}", f"SELECT {index}")
        for index in range(6524)
    ]
    heldout = [
        Example(str(index), "heldout_db", f"heldout {index}", f"SELECT {index}", "heldout")
        for index in range(500)
    ]
    training_failures = []
    for example in training[:220]:
        training_failures.append(
            {
                "db_id": example.db_id,
                "example_ids": [example.row_id],
                "status": "resource_limit",
                "signal_name": "SIGXCPU",
                "limit_kind": "cpu",
                "error": "worker exited with signal",
            }
        )
    for example in training[220:242]:
        training_failures.append(
            {
                "db_id": example.db_id,
                "example_ids": [example.row_id],
                "status": "resource_limit",
                "limit_kind": "rows",
                "error": "result row limit exceeded",
            }
        )
    heldout_failures = [
        {
            "db_id": example.db_id,
            "example_ids": [example.row_id],
            "status": "execution_error",
            "error": "worker exit code None",
        }
        for example in heldout[:2]
    ]
    private = {
        "training": {
            "unique_sql_count": 6524,
            "status_counts": {"ok": 6282},
            "failures": training_failures,
        },
        "heldout": {
            "unique_sql_count": 500,
            "status_counts": {"ok": 498},
            "failures": heldout_failures,
        },
    }
    from verireason.m2_diagnostics import select_failed_queries

    items = select_failed_queries(private, {"training": training, "heldout": heldout})
    profile_a_training = [
        item for item in items if item.split == "training" and item.profile == "profile_a"
    ]
    results: dict[str, object] = {}
    for index, item in enumerate(profile_a_training):
        if index < 214:
            result = {"status": "ok", "limit_kind": None, "exit_code": 0}
        elif index < 219:
            result = {"status": "timeout", "limit_kind": "wall", "exit_code": -15}
        else:
            result = {"status": "worker_error", "limit_kind": None, "exit_code": 1}
        results[item.key] = {"result": result}
    for item in items:
        if item.key in results:
            continue
        if item.profile == "profile_b":
            result = {"status": "resource_limit", "limit_kind": "rows", "exit_code": 0}
        else:
            result = {"status": "timeout", "limit_kind": "wall", "exit_code": -15}
        results[item.key] = {"result": result}
    checkpoint = {
        "complete": True,
        "plan_sha256": "3ded2e3718c9d2d397c07dd6c2792651dbf3992113d500733fbc562dd699cd6b",
        "results": results,
    }
    return training, heldout, private, checkpoint


def test_operational_eligibility_freezes_6496_28_and_498_2() -> None:
    training, heldout, private, checkpoint = _evidence_fixture()
    config = load_finalizer_config(ROOT / "configs/m2_finalizer.toml")

    eligibility, audit = derive_operational_eligibility(
        private, checkpoint, training, heldout, config
    )

    assert audit["training_eligible_unique_sql"] == 6496
    assert audit["training_ineligible_unique_sql"] == 28
    assert audit["training_original_ok_unique_sql"] + audit["training_recovered_unique_sql"] == 6496
    assert audit["heldout_eligible_examples"] == 498
    assert audit["heldout_ineligible_examples"] == 2
    assert sum(not allowed for allowed, _reason in eligibility.values()) == 30


@pytest.mark.parametrize("raw_status", ["execution_error", "resource_limit"])
def test_profile_a_raw_fallback_matches_public_worker_error_aggregate(raw_status: str) -> None:
    training, heldout, private, checkpoint = _evidence_fixture()
    config = load_finalizer_config(ROOT / "configs/m2_finalizer.toml")
    items = select_failed_queries(private, {"training": training, "heldout": heldout})
    worker_item = next(
        item
        for item in items
        if item.split == "training"
        and item.profile == "profile_a"
        and checkpoint["results"][item.key]["result"]["status"] == "worker_error"
    )
    checkpoint["results"][worker_item.key]["result"] = {
        "status": raw_status,
        "limit_kind": None,
        "exit_code": 1,
        "worker_state": "exited_without_payload",
    }

    eligibility, audit = derive_operational_eligibility(
        private, checkpoint, training, heldout, config
    )

    assert eligibility[worker_item.key] == (False, "worker_error")
    assert audit["exclusion_reasons"]["training:worker_error"] == 1


def test_profile_a_cpu_limit_remains_a_hard_failure() -> None:
    training, heldout, private, checkpoint = _evidence_fixture()
    config = load_finalizer_config(ROOT / "configs/m2_finalizer.toml")
    items = select_failed_queries(private, {"training": training, "heldout": heldout})
    worker_item = next(
        item
        for item in items
        if item.split == "training"
        and item.profile == "profile_a"
        and checkpoint["results"][item.key]["result"]["status"] == "worker_error"
    )
    checkpoint["results"][worker_item.key]["result"] = {
        "status": "resource_limit",
        "limit_kind": "cpu",
        "exit_code": -24,
    }

    with pytest.raises(ValueError, match="unexpected Profile-A CPU limit"):
        derive_operational_eligibility(private, checkpoint, training, heldout, config)


def test_training_exclusion_count_mismatch_hard_fails() -> None:
    training, heldout, private, checkpoint = _evidence_fixture()
    config = load_finalizer_config(ROOT / "configs/m2_finalizer.toml")
    recovered = [
        key for key, entry in checkpoint["results"].items() if entry["result"]["status"] == "ok"
    ]
    checkpoint["results"][recovered[0]]["result"] = {
        "status": "timeout",
        "limit_kind": "wall",
        "exit_code": -15,
    }

    with pytest.raises(ValueError, match="recovered training unique SQL count"):
        derive_operational_eligibility(private, checkpoint, training, heldout, config)


def test_wrong_plan_or_completed_count_blocks_eligibility() -> None:
    training, heldout, private, checkpoint = _evidence_fixture()
    config = load_finalizer_config(ROOT / "configs/m2_finalizer.toml")
    checkpoint["plan_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="plan SHA-256"):
        derive_operational_eligibility(private, checkpoint, training, heldout, config)

    checkpoint["plan_sha256"] = config["diagnostic_plan_sha256"]
    checkpoint["results"].pop(next(iter(checkpoint["results"])))
    with pytest.raises(ValueError, match="completed count"):
        derive_operational_eligibility(private, checkpoint, training, heldout, config)


def test_evidence_hash_and_checkpoint_mode_are_hard_gates(tmp_path: Path) -> None:
    names = (
        "private_preflight",
        "diagnostic_checkpoint",
        "source_receipt",
        "training_labels",
        "heldout_labels",
        "historical_formal_config",
        "amended_formal_config",
    )
    paths = {name: tmp_path / name for name in names}
    for name, path in paths.items():
        path.write_text(name, encoding="utf-8")
    os.chmod(paths["private_preflight"], 0o600)
    os.chmod(paths["diagnostic_checkpoint"], 0o600)
    config = {
        "evidence": {
            f"{name}_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in paths.items()
        }
    }

    verify_finalizer_evidence(config, **paths)
    config["evidence"]["historical_formal_config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="historical_formal_config SHA-256"):
        verify_finalizer_evidence(config, **paths)

    config["evidence"]["historical_formal_config_sha256"] = hashlib.sha256(
        paths["historical_formal_config"].read_bytes()
    ).hexdigest()
    os.chmod(paths["diagnostic_checkpoint"], 0o644)
    with pytest.raises(PermissionError, match="mode 0600"):
        verify_finalizer_evidence(config, **paths)


@pytest.mark.parametrize(
    ("private_mode", "checkpoint_mode", "message"),
    [
        (0o600, 0o600, None),
        (0o644, 0o600, "private preflight must have mode 0600"),
        (0o600, 0o644, "diagnostic checkpoint must have mode 0600"),
    ],
)
def test_both_private_evidence_files_require_exact_0600(
    tmp_path: Path, private_mode: int, checkpoint_mode: int, message: str | None
) -> None:
    names = (
        "private_preflight",
        "diagnostic_checkpoint",
        "source_receipt",
        "training_labels",
        "heldout_labels",
        "historical_formal_config",
        "amended_formal_config",
    )
    paths = {name: tmp_path / name for name in names}
    for name, path in paths.items():
        path.write_text(name, encoding="utf-8")
    os.chmod(paths["private_preflight"], private_mode)
    os.chmod(paths["diagnostic_checkpoint"], checkpoint_mode)
    config = {
        "evidence": {
            f"{name}_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in paths.items()
        }
    }

    if message is None:
        verify_finalizer_evidence(config, **paths)
    else:
        with pytest.raises(PermissionError, match=message):
            verify_finalizer_evidence(config, **paths)


@pytest.mark.parametrize(
    ("train_databases", "validation_databases", "passes"),
    [(59, 10, True), (58, 10, False), (59, 9, False)],
)
def test_post_exclusion_database_coverage_is_exactly_59_10(
    train_databases: int, validation_databases: int, passes: bool
) -> None:
    rows: list[tuple[str, Example, bool]] = []
    for index in range(train_databases):
        rows.append(("train", Example(f"t{index}", f"train_{index}", "q", "SELECT 1"), True))
    for index in range(validation_databases):
        rows.append(
            (
                "validation",
                Example(f"v{index}", f"validation_{index}", "q", "SELECT 1"),
                True,
            )
        )
    config = {
        "training_source": {"expected_database_count": 69},
        "split": {
            "validation_database_count": 10,
            "minimum_validation_example_ratio": 0.10,
            "maximum_validation_example_ratio": 0.25,
            "critical_sql_features": [],
        },
    }

    report = _post_exclusion_report(rows, config)

    assert report["passed"] is passes
    assert report["database_counts"] == {
        "train": train_databases,
        "validation": validation_databases,
    }
    if not passes:
        assert any("database count changed" in error for error in report["gate_errors"])


def test_training_loader_filters_ineligible_and_rejects_heldout(tmp_path: Path) -> None:
    manifest = tmp_path / "training.json"
    manifest.write_text(
        json.dumps(
            {
                "examples": [
                    {
                        "row_id": "a",
                        "source": "training",
                        "split": "train",
                        "operational_eligible": True,
                    },
                    {
                        "row_id": "b",
                        "source": "training",
                        "split": "train",
                        "operational_eligible": False,
                    },
                    {
                        "row_id": "c",
                        "source": "training",
                        "split": "validation",
                        "operational_eligible": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    loaded = load_operational_training_examples(manifest, split="train")
    assert [row["row_id"] for row in loaded] == ["a"]

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["examples"].append(
        {"row_id": "secret", "source": "heldout", "split": "train", "operational_eligible": True}
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="heldout data"):
        load_operational_training_examples(manifest, split="train")


def test_heldout_metrics_fix_full_500_and_eligible_498_denominators() -> None:
    rows = [
        {
            "operational_eligible": index < 498,
            "bird_ex_behavior_compatible": True,
            "strict_execution_accuracy": True,
        }
        for index in range(500)
    ]
    scored = score_heldout_scopes(rows)
    assert scored["full_denominator"] == 500
    assert scored["eligible_denominator"] == 498
    assert scored["strict_execution_accuracy"]["full_correct"] == 498
    assert (
        scored["strict_execution_accuracy"]["full_heldout_conservative_execution_accuracy"]
        == 498 / 500
    )
    assert scored["strict_execution_accuracy"]["eligible_heldout_execution_accuracy"] == 1.0


def _database(path: Path) -> None:
    path.parent.mkdir(parents=True)
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE t (x INTEGER)")
    connection.commit()
    connection.close()


def test_hardlink_tree_ignores_sql_symlink_without_following_it(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _database(source / "db" / "db.sqlite")
    external = tmp_path / "external.sql"
    external.write_text("secret SQL outside database tree", encoding="utf-8")
    (source / "unreferenced.sql").symlink_to(external)

    ignored = _hardlink_tree(source, destination)

    assert ignored == 1
    assert (destination / "db/db.sqlite").is_file()
    assert not (destination / "unreferenced.sql").exists()
    assert external.read_text(encoding="utf-8") == "secret SQL outside database tree"


@pytest.mark.parametrize("link_name", ["database.sqlite", "database.db", "other.txt"])
def test_hardlink_tree_rejects_every_other_symlink(tmp_path: Path, link_name: str) -> None:
    source = tmp_path / "source"
    source.mkdir()
    external = tmp_path / "external"
    external.write_text("outside", encoding="utf-8")
    (source / link_name).symlink_to(external)

    with pytest.raises(ValueError, match="contains a symbolic link"):
        _hardlink_tree(source, tmp_path / "destination")


def test_finalizer_never_executes_gold_and_atomic_failure_leaves_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "m2-v1.partial"
    _database(source / "databases/training/train_a/train_a.sqlite")
    _database(source / "databases/training/train_b/train_b.sqlite")
    _database(source / "databases/heldout/hold_a/hold_a.sqlite")
    _database(source / "databases/heldout/hold_b/hold_b.sqlite")
    source_databases = sorted((source / "databases").rglob("*.sqlite"))
    source_identity_before = {
        path: (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_size,
            path.read_bytes(),
        )
        for path in source_databases
    }
    (source / "source-receipt.json").write_text("{}\n", encoding="utf-8")
    (source / "gold-preflight-private.json").write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}\n", encoding="utf-8")
    os.chmod(checkpoint, 0o600)
    labels = tmp_path / "labels"
    labels.write_text("[]\n", encoding="utf-8")
    final = tmp_path / "m2-v1"
    training = [
        Example("a", "train_a", "q", "SELECT 1"),
        Example("b", "train_b", "q", "SELECT 2"),
    ]
    heldout = [
        Example("h1", "hold_a", "secret q1", "SELECT 3", "heldout"),
        Example("h2", "hold_b", "secret q2", "SELECT 4", "heldout"),
    ]
    amended = {
        "training_source": {"expected_rows": 2, "expected_database_count": 2},
        "heldout_source": {"expected_rows": 2, "expected_database_count": 2},
        "split": {
            "seed": "seed",
            "validation_database_count": 1,
            "critical_sql_features": [],
            "minimum_validation_example_ratio": 0.1,
            "maximum_validation_example_ratio": 0.9,
        },
    }
    identities = {
        "private_preflight_sha256": "a" * 64,
        "diagnostic_checkpoint_sha256": "b" * 64,
        "source_receipt_sha256": hashlib.sha256(b"{}\n").hexdigest(),
        "training_labels_sha256": "c" * 64,
        "heldout_labels_sha256": "d" * 64,
        "historical_formal_config_sha256": "e" * 64,
        "amended_formal_config_sha256": "f" * 64,
    }
    finalizer_config = {
        "publication": {"staging_suffix": ".finalization.partial"},
        "diagnostic_plan_sha256": "plan",
    }
    eligibility = {
        _key("heldout", heldout[0]): (False, "formal_wall_timeout"),
        _key("heldout", heldout[1]): (False, "formal_wall_timeout"),
    }
    audit_counts = {
        "training_eligible_unique_sql": 6496,
        "training_ineligible_unique_sql": 28,
        "training_original_ok_unique_sql": 6282,
        "training_recovered_unique_sql": 214,
        "heldout_eligible_examples": 498,
        "heldout_ineligible_examples": 2,
        "exclusion_reasons": {},
    }
    clean_leakage = LeakageReport(
        hard_failures={"db_id": DuplicateCounts(0, 0)},
        review_findings={},
        pending_checks=(),
    )
    monkeypatch.setattr(
        "verireason.m2_finalizer.load_finalizer_config", lambda _path: finalizer_config
    )
    monkeypatch.setattr("verireason.m2_finalizer.load_m2_config", lambda _path: amended)
    monkeypatch.setattr(
        "verireason.m2_finalizer.verify_finalizer_evidence", lambda *args, **kwargs: identities
    )
    monkeypatch.setattr(
        "verireason.m2_finalizer.load_examples",
        lambda _path, source="training": heldout if source == "heldout" else training,
    )
    monkeypatch.setattr(
        "verireason.m2_finalizer.derive_operational_eligibility",
        lambda *args, **kwargs: (eligibility, audit_counts),
    )
    monkeypatch.setattr("verireason.m2_finalizer.audit_leakage", lambda _rows: clean_leakage)
    monkeypatch.setattr(
        "verireason.m2_finalizer.atomic_publish_directory",
        lambda *_args: (_ for _ in ()).throw(OSError("simulated atomic publish failure")),
    )
    monkeypatch.setattr(
        "verireason.sql_evaluator.execute_read_only",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not execute SQL")),
    )

    with pytest.raises(OSError, match="simulated atomic publish failure"):
        finalize_m2(
            materialization_partial=source,
            diagnostic_checkpoint=checkpoint,
            training_labels=labels,
            heldout_labels=labels,
            historical_formal_config=ROOT / "configs/m2_data_v1_3s.toml",
            amended_formal_config=ROOT / "configs/m2_data.toml",
            finalizer_config=ROOT / "configs/m2_finalizer.toml",
            final_dir=final,
        )

    staging = tmp_path / "m2-v1.finalization.partial"
    assert source.is_dir()
    assert not final.exists()
    assert staging.is_dir()
    source_identity_after = {
        path: (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_size,
            path.read_bytes(),
        )
        for path in source_databases
    }
    assert source_identity_after == source_identity_before
    failure = staging / "finalization-failure.json"
    assert stat.S_IMODE(failure.stat().st_mode) == 0o600
    public = "".join(
        path.read_text(encoding="utf-8")
        for path in staging.glob("*.json")
        if "private" not in path.name and path.name != "finalization-failure.json"
    )
    assert "secret q" not in public
    assert "hold_a" not in public
    assert '"h1"' not in public
    assert "SELECT 3" not in public
