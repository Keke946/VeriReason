import io
import json
import stat
import zipfile
from pathlib import Path

import pytest

from verireason.data_pipeline import sha256_file
from verireason.materialization import (
    ArchiveValidation,
    ExtractionCapacityGate,
    FilesystemCapacity,
    atomic_publish_directory,
    build_source_receipt,
    discover_database_root,
    evaluate_extraction_capacity,
    read_http_headers,
    require_extraction_capacity,
    require_free_space,
    require_outside_git_worktree,
    safe_extract_zip,
    safe_extract_zip_after_capacity_gate,
    safely_observe_and_extract_zip,
    validate_archive_observation,
    write_json,
)


def _zip(path: Path, members: list[tuple[str, bytes, int | None]]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload, mode in members:
            info = zipfile.ZipInfo(name)
            if mode is not None:
                info.create_system = 3
                info.external_attr = mode << 16
            archive.writestr(info, payload)


def test_safe_extract_validates_counts_sizes_and_publishes_atomically(tmp_path: Path) -> None:
    archive = tmp_path / "safe.zip"
    _zip(archive, [("db/data.sqlite", b"sqlite", stat.S_IFREG | 0o644)])
    partial = tmp_path / "dataset.partial"
    final = tmp_path / "dataset"

    result = safe_extract_zip(
        archive,
        partial,
        expected_member_count=1,
        expected_uncompressed_bytes=6,
    )
    atomic_publish_directory(partial, final)

    assert result.member_count == 1
    assert (final / "db/data.sqlite").read_bytes() == b"sqlite"
    assert not partial.exists()


@pytest.mark.parametrize("name", ["/absolute", "../escape", "a/../../escape", "bad\\path"])
def test_safe_extract_rejects_path_escape(tmp_path: Path, name: str) -> None:
    archive = tmp_path / "bad.zip"
    _zip(archive, [(name, b"bad", stat.S_IFREG | 0o644)])

    with pytest.raises(ValueError, match="ZIP"):
        safe_extract_zip(
            archive,
            tmp_path / "partial",
            expected_member_count=1,
            expected_uncompressed_bytes=3,
        )


def test_safe_extract_rejects_symlink(tmp_path: Path) -> None:
    archive = tmp_path / "link.zip"
    _zip(archive, [("link", b"target", stat.S_IFLNK | 0o777)])

    with pytest.raises(ValueError, match="symbolic links"):
        safe_extract_zip(
            archive,
            tmp_path / "partial",
            expected_member_count=1,
            expected_uncompressed_bytes=6,
        )


def test_safe_extract_rejects_duplicate_and_prefix_conflict(tmp_path: Path) -> None:
    archive = tmp_path / "conflict.zip"
    _zip(
        archive,
        [
            ("path", b"file", stat.S_IFREG | 0o644),
            ("path/child", b"child", stat.S_IFREG | 0o644),
        ],
    )

    with pytest.raises(ValueError, match="conflict"):
        safe_extract_zip(
            archive,
            tmp_path / "partial",
            expected_member_count=2,
            expected_uncompressed_bytes=9,
        )


def test_safe_extract_rejects_duplicate_member(tmp_path: Path) -> None:
    archive = tmp_path / "duplicate.zip"
    with pytest.warns(UserWarning):
        _zip(
            archive,
            [
                ("same", b"one", stat.S_IFREG | 0o644),
                ("same", b"two", stat.S_IFREG | 0o644),
            ],
        )

    with pytest.raises(ValueError, match="duplicate"):
        safe_extract_zip(
            archive,
            tmp_path / "partial",
            expected_member_count=2,
            expected_uncompressed_bytes=6,
        )


def test_bounded_first_observation_uses_full_safe_extractor(tmp_path: Path) -> None:
    archive = tmp_path / "nested.zip"
    _zip(archive, [("db/db.sqlite", b"sqlite", stat.S_IFREG | 0o644)])

    result = safely_observe_and_extract_zip(
        archive,
        tmp_path / "extracted",
        maximum_member_count=2,
        maximum_uncompressed_bytes=10,
    )

    assert result.member_count == 1
    assert result.uncompressed_bytes == 6
    assert (tmp_path / "extracted/db/db.sqlite").read_bytes() == b"sqlite"


def test_about_40_gib_synthetic_zip_metadata_passes_50_gib_ceiling() -> None:
    observed = validate_archive_observation(
        member_count=700,
        uncompressed_bytes=40 * 1024**3,
        maximum_member_count=10_000,
        maximum_uncompressed_bytes=50 * 1024**3,
    )

    assert observed.uncompressed_bytes == 40 * 1024**3


def test_synthetic_zip_metadata_above_50_gib_ceiling_fails() -> None:
    with pytest.raises(ValueError, match="uncompressed bytes exceed cap"):
        validate_archive_observation(
            member_count=700,
            uncompressed_bytes=50 * 1024**3 + 1,
            maximum_member_count=10_000,
            maximum_uncompressed_bytes=50 * 1024**3,
        )


@pytest.mark.parametrize(
    ("member_cap", "byte_cap", "message"),
    [(0, 10, "member count exceeds cap"), (2, 5, "uncompressed bytes exceed cap")],
)
def test_bounded_first_observation_rejects_cap_overrun_before_extracting(
    tmp_path: Path, member_cap: int, byte_cap: int, message: str
) -> None:
    archive = tmp_path / "nested.zip"
    destination = tmp_path / "extracted"
    _zip(archive, [("db/db.sqlite", b"sqlite", stat.S_IFREG | 0o644)])

    with pytest.raises(ValueError, match=message):
        safely_observe_and_extract_zip(
            archive,
            destination,
            maximum_member_count=member_cap,
            maximum_uncompressed_bytes=byte_cap,
        )

    assert not destination.exists()


def test_bounded_first_observation_still_rejects_unsafe_members(tmp_path: Path) -> None:
    archive = tmp_path / "nested.zip"
    _zip(archive, [("../escape", b"bad", stat.S_IFREG | 0o644)])

    with pytest.raises(ValueError, match="ZIP path traversal"):
        safely_observe_and_extract_zip(
            archive,
            tmp_path / "extracted",
            maximum_member_count=2,
            maximum_uncompressed_bytes=10,
        )


def test_database_root_is_discovered_from_actual_sqlite_layout(tmp_path: Path) -> None:
    root = tmp_path / "extracted"
    expected = root / "observed" / "train_databases"
    for db_id in ("alpha", "beta"):
        database_dir = expected / db_id
        database_dir.mkdir(parents=True)
        (database_dir / f"{db_id}.sqlite").write_bytes(b"sqlite")

    assert discover_database_root(root, {"alpha", "beta"}) == expected


def test_database_root_discovery_rejects_ambiguous_layout(tmp_path: Path) -> None:
    root = tmp_path / "extracted"
    for prefix in ("one", "two"):
        database_dir = root / prefix / "alpha"
        database_dir.mkdir(parents=True)
        (database_dir / "alpha.sqlite").write_bytes(b"sqlite")

    with pytest.raises(ValueError, match="exactly once"):
        discover_database_root(root, {"alpha"})


def test_two_level_archive_is_safely_extracted_and_discovered(tmp_path: Path) -> None:
    inner_buffer = io.BytesIO()
    with zipfile.ZipFile(inner_buffer, "w") as inner:
        inner.writestr("train_databases/alpha/alpha.sqlite", b"sqlite")
    outer = tmp_path / "train.zip"
    _zip(
        outer,
        [("train/train_databases.zip", inner_buffer.getvalue(), stat.S_IFREG | 0o644)],
    )
    outer_target = tmp_path / "outer"
    database_target = tmp_path / "databases"

    safe_extract_zip(
        outer,
        outer_target,
        expected_member_count=1,
        expected_uncompressed_bytes=len(inner_buffer.getvalue()),
    )
    result = safely_observe_and_extract_zip(
        outer_target / "train/train_databases.zip",
        database_target,
        maximum_member_count=2,
        maximum_uncompressed_bytes=10,
    )

    assert result.member_count == 1
    assert discover_database_root(database_target, {"alpha"}) == (
        database_target / "train_databases"
    )


def test_capacity_gate_uses_target_filesystem(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="insufficient free space"):
        require_free_space(tmp_path / "future" / "dataset", 10**30)


@pytest.mark.parametrize(
    ("free_bytes", "passes"),
    [(70 * 1024**3, True), (70 * 1024**3 - 1, False)],
)
def test_initial_70_gib_capacity_floor_is_enforced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, free_bytes: int, passes: bool
) -> None:
    capacity = FilesystemCapacity(
        existing_path=str(tmp_path),
        device=7,
        total_bytes=100 * 1024**3,
        used_bytes=100 * 1024**3 - free_bytes,
        free_bytes=free_bytes,
    )
    monkeypatch.setattr("verireason.materialization.filesystem_capacity", lambda target: capacity)

    if passes:
        assert require_free_space(tmp_path / "actual-m2-root", 70 * 1024**3) == capacity
    else:
        with pytest.raises(ValueError, match="insufficient free space"):
            require_free_space(tmp_path / "actual-m2-root", 70 * 1024**3)


@pytest.mark.parametrize(
    ("free_bytes", "passed"),
    [(55 * 1024**3, True), (52 * 1024**3, False)],
)
def test_dynamic_capacity_gate_rechecks_observed_requirement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, free_bytes: int, passed: bool
) -> None:
    capacity = FilesystemCapacity(
        existing_path=str(tmp_path),
        device=7,
        total_bytes=100 * 1024**3,
        used_bytes=100 * 1024**3 - free_bytes,
        free_bytes=free_bytes,
    )
    monkeypatch.setattr("verireason.materialization.filesystem_capacity", lambda target: capacity)

    gate = evaluate_extraction_capacity(
        tmp_path / "actual-m2-data-root",
        observed_inner_uncompressed_bytes=40 * 1024**3,
        heldout_expected_uncompressed_bytes=3 * 1024**3,
        reserve_bytes=10 * 1024**3,
    )

    assert gate.required_bytes == 53 * 1024**3
    assert gate.passed is passed
    if passed:
        require_extraction_capacity(gate)
    else:
        with pytest.raises(ValueError, match="before nested extraction"):
            require_extraction_capacity(gate)


def test_insufficient_dynamic_capacity_stops_before_extractor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called = False

    def forbidden_extractor(*args: object, **kwargs: object) -> ArchiveValidation:
        nonlocal called
        called = True
        raise AssertionError("extractor must not run")

    monkeypatch.setattr("verireason.materialization.safe_extract_zip", forbidden_extractor)
    gate = ExtractionCapacityGate(
        filesystem=FilesystemCapacity(
            existing_path=str(tmp_path),
            device=7,
            total_bytes=100,
            used_bytes=60,
            free_bytes=40,
        ),
        observed_inner_uncompressed_bytes=30,
        heldout_expected_uncompressed_bytes=5,
        reserve_bytes=10,
        required_bytes=45,
        passed=False,
    )

    with pytest.raises(ValueError, match="before nested extraction"):
        safe_extract_zip_after_capacity_gate(
            tmp_path / "nested.zip",
            tmp_path / "databases",
            observation=ArchiveValidation(1, 30),
            capacity_gate=gate,
        )

    assert not called


def test_sufficient_dynamic_capacity_continues_to_safe_extraction(tmp_path: Path) -> None:
    archive = tmp_path / "nested.zip"
    _zip(archive, [("db/db.sqlite", b"sqlite", stat.S_IFREG | 0o644)])
    gate = ExtractionCapacityGate(
        filesystem=FilesystemCapacity(
            existing_path=str(tmp_path),
            device=7,
            total_bytes=100,
            used_bytes=40,
            free_bytes=60,
        ),
        observed_inner_uncompressed_bytes=6,
        heldout_expected_uncompressed_bytes=3,
        reserve_bytes=10,
        required_bytes=19,
        passed=True,
    )

    result = safe_extract_zip_after_capacity_gate(
        archive,
        tmp_path / "databases",
        observation=ArchiveValidation(1, 6),
        capacity_gate=gate,
    )

    assert result == ArchiveValidation(1, 6)
    assert (tmp_path / "databases/db/db.sqlite").read_bytes() == b"sqlite"


def test_materialization_target_must_be_outside_git_worktree(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()

    with pytest.raises(ValueError, match="outside Git"):
        require_outside_git_worktree(tmp_path / "data" / "processed")


def test_http_206_resume_uses_content_range_total(tmp_path: Path) -> None:
    headers = tmp_path / "resume.headers"
    headers.write_text(
        "HTTP/2 206\nContent-Length: 400\nContent-Range: bytes 600-999/1000\n\n",
        encoding="ascii",
    )

    observed = read_http_headers(headers)

    assert observed["http_status"] == 206
    assert observed["response_content_length"] == 400
    assert observed["content_length"] == 1000
    assert observed["content_range"] == "bytes 600-999/1000"


@pytest.mark.parametrize(
    "content_range",
    [None, "bytes 600-998/1000", "bytes 600-1000/1000", "garbage"],
)
def test_http_206_resume_rejects_invalid_content_range(
    tmp_path: Path, content_range: str | None
) -> None:
    headers = tmp_path / "resume.headers"
    range_line = f"Content-Range: {content_range}\n" if content_range is not None else ""
    headers.write_text(
        f"HTTP/2 206\nContent-Length: 400\n{range_line}\n",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="HTTP 206"):
        read_http_headers(headers)


def test_source_receipt_records_first_archive_hash_and_is_itself_hashed(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("schema_version = 1\n", encoding="utf-8")
    paths = {}
    for name, payload in {
        "training_labels": b"train labels",
        "heldout_labels": b"heldout labels",
        "training_archive": b"train archive",
        "heldout_archive": b"heldout archive",
    }.items():
        path = tmp_path / name
        path.write_bytes(payload)
        paths[name] = path
    header_paths = {}
    for name, path in paths.items():
        headers = tmp_path / f"{name}.headers"
        headers.write_text(
            f"HTTP/2 200\nContent-Length: {path.stat().st_size}\n\n",
            encoding="ascii",
        )
        header_key = (
            f"{name.removesuffix('s')}_headers" if name.endswith("labels") else f"{name}_headers"
        )
        header_paths[header_key] = headers
    config = {
        "training_source": {
            "labels_observed_sha256": sha256_file(paths["training_labels"]),
            "labels_content_length": paths["training_labels"].stat().st_size,
            "database_archive_content_length": paths["training_archive"].stat().st_size,
            "database_archive_observed_sha256": sha256_file(paths["training_archive"]),
            "labels_url": "https://example.invalid/train-labels",
            "revision": "train-revision",
            "labels_etag": "labels-etag",
            "labels_last_modified": "",
            "database_url": "https://example.invalid/train.zip",
            "database_archive_etag": "archive-etag",
            "database_archive_last_modified": "date",
        },
        "heldout_source": {
            "labels_observed_sha256": sha256_file(paths["heldout_labels"]),
            "labels_content_length": paths["heldout_labels"].stat().st_size,
            "archive_content_length": paths["heldout_archive"].stat().st_size,
            "archive_observed_sha256": sha256_file(paths["heldout_archive"]),
            "labels_url": "https://example.invalid/heldout-labels",
            "revision": "heldout-revision",
            "labels_etag": "labels-etag",
            "labels_last_modified": "",
            "archive_url": "https://example.invalid/heldout.zip",
            "archive_etag": "",
            "archive_last_modified": "date",
        },
    }
    receipt = build_source_receipt(
        config_path,
        config,
        **paths,
        training_label_headers=header_paths["training_label_headers"],
        heldout_label_headers=header_paths["heldout_label_headers"],
        training_archive_headers=header_paths["training_archive_headers"],
        heldout_archive_headers=header_paths["heldout_archive_headers"],
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_sha = write_json(receipt_path, receipt)
    private_path = tmp_path / "heldout-private.json"
    write_json(private_path, {"private": True}, private=True)

    assert receipt["artifacts"]["training_archive"]["sha256"] == sha256_file(
        paths["training_archive"]
    )
    assert receipt_sha == sha256_file(receipt_path)
    assert str(tmp_path) not in json.dumps(receipt)
    assert stat.S_IMODE(private_path.stat().st_mode) == 0o600
