"""Safe archive extraction, source receipts, and atomic M2 publication helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from verireason.data_pipeline import resolve_database_path, sha256_file, validate_observed_identity


@dataclass(frozen=True)
class ArchiveValidation:
    member_count: int
    uncompressed_bytes: int


@dataclass(frozen=True)
class FilesystemCapacity:
    existing_path: str
    device: int
    total_bytes: int
    used_bytes: int
    free_bytes: int


@dataclass(frozen=True)
class ExtractionCapacityGate:
    filesystem: FilesystemCapacity
    observed_inner_uncompressed_bytes: int
    heldout_expected_uncompressed_bytes: int
    reserve_bytes: int
    required_bytes: int
    passed: bool


def _logical_member_path(name: str) -> PurePosixPath:
    if not name or "\x00" in name or "\\" in name:
        raise ValueError(f"unsafe ZIP member name: {name!r}")
    normalized = unicodedata.normalize("NFKC", name)
    path = PurePosixPath(normalized)
    if path.is_absolute() or normalized.startswith("/") or ".." in path.parts:
        raise ValueError(f"ZIP path traversal or absolute member: {name!r}")
    if not path.parts or path == PurePosixPath("."):
        raise ValueError(f"empty ZIP member path: {name!r}")
    return path


def validate_zip_archive(
    archive_path: str | Path,
    *,
    expected_member_count: int,
    expected_uncompressed_bytes: int,
) -> ArchiveValidation:
    seen: set[str] = set()
    files: set[PurePosixPath] = set()
    directories: set[PurePosixPath] = set()
    uncompressed = 0
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) != expected_member_count:
            raise ValueError(f"ZIP member count changed: {len(members)} != {expected_member_count}")
        for member in members:
            path = _logical_member_path(member.filename)
            collision_key = path.as_posix().casefold()
            if collision_key in seen:
                raise ValueError(f"duplicate or case-conflicting ZIP member: {member.filename!r}")
            seen.add(collision_key)
            mode_type = stat.S_IFMT(member.external_attr >> 16)
            if mode_type == stat.S_IFLNK:
                raise ValueError(f"symbolic links are forbidden in ZIP: {member.filename!r}")
            if mode_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise ValueError(f"special ZIP member is forbidden: {member.filename!r}")
            parents = set(path.parents) - {PurePosixPath(".")}
            if parents & files:
                raise ValueError(f"ZIP file/directory prefix conflict: {member.filename!r}")
            is_directory = member.is_dir() or mode_type == stat.S_IFDIR
            if is_directory:
                if path in files:
                    raise ValueError(f"ZIP file/directory conflict: {member.filename!r}")
                directories.add(path)
            else:
                if path in directories:
                    raise ValueError(f"ZIP directory/file conflict: {member.filename!r}")
                files.add(path)
                uncompressed += member.file_size
        if uncompressed != expected_uncompressed_bytes:
            raise ValueError(
                f"ZIP uncompressed size changed: {uncompressed} != {expected_uncompressed_bytes}"
            )
    return ArchiveValidation(len(members), uncompressed)


def safe_extract_zip(
    archive_path: str | Path,
    destination: str | Path,
    *,
    expected_member_count: int,
    expected_uncompressed_bytes: int,
) -> ArchiveValidation:
    validation = validate_zip_archive(
        archive_path,
        expected_member_count=expected_member_count,
        expected_uncompressed_bytes=expected_uncompressed_bytes,
    )
    target = Path(destination)
    if target.exists():
        raise ValueError(f"extraction destination already exists: {target}")
    target.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            logical = _logical_member_path(member.filename)
            output = target.joinpath(*logical.parts)
            if member.is_dir() or stat.S_IFMT(member.external_attr >> 16) == stat.S_IFDIR:
                output.mkdir(parents=True, exist_ok=True)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with archive.open(member) as source, output.open("xb") as destination_handle:
                while chunk := source.read(1024 * 1024):
                    destination_handle.write(chunk)
                    written += len(chunk)
            if written != member.file_size:
                raise ValueError(
                    f"ZIP member size mismatch for {member.filename!r}: "
                    f"{written} != {member.file_size}"
                )
    return validation


def observe_zip_under_caps(
    archive_path: str | Path,
    *,
    maximum_member_count: int,
    maximum_uncompressed_bytes: int,
) -> ArchiveValidation:
    """Read central-directory metadata and reject observations above hard ceilings."""
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        member_count = len(members)
        uncompressed_bytes = sum(member.file_size for member in members if not member.is_dir())
    return validate_archive_observation(
        member_count=member_count,
        uncompressed_bytes=uncompressed_bytes,
        maximum_member_count=maximum_member_count,
        maximum_uncompressed_bytes=maximum_uncompressed_bytes,
    )


def validate_archive_observation(
    *,
    member_count: int,
    uncompressed_bytes: int,
    maximum_member_count: int,
    maximum_uncompressed_bytes: int,
) -> ArchiveValidation:
    """Validate synthetic or observed ZIP central-directory totals against hard ceilings."""
    if member_count > maximum_member_count:
        raise ValueError(
            f"nested ZIP member count exceeds cap: {member_count} > {maximum_member_count}"
        )
    if uncompressed_bytes > maximum_uncompressed_bytes:
        raise ValueError(
            "nested ZIP uncompressed bytes exceed cap: "
            f"{uncompressed_bytes} > {maximum_uncompressed_bytes}"
        )
    return ArchiveValidation(member_count, uncompressed_bytes)


def safely_observe_and_extract_zip(
    archive_path: str | Path,
    destination: str | Path,
    *,
    maximum_member_count: int,
    maximum_uncompressed_bytes: int,
) -> ArchiveValidation:
    """Freeze first-observed metadata under hard caps, then use exact safe extraction."""
    observation = observe_zip_under_caps(
        archive_path,
        maximum_member_count=maximum_member_count,
        maximum_uncompressed_bytes=maximum_uncompressed_bytes,
    )
    return safe_extract_zip(
        archive_path,
        destination,
        expected_member_count=observation.member_count,
        expected_uncompressed_bytes=observation.uncompressed_bytes,
    )


def discover_database_root(extraction_root: str | Path, database_ids: set[str]) -> Path:
    """Find the one logical root that resolves every expected BIRD database."""
    root = Path(extraction_root)
    candidates: set[Path] = set()
    for path in root.rglob("*"):
        if (
            path.is_file()
            and path.suffix.casefold() in {".sqlite", ".db"}
            and path.stem in database_ids
        ):
            candidates.add(path.parent)
            if path.stem == path.parent.name:
                candidates.add(path.parent.parent)
    matches: list[Path] = []
    for candidate in sorted(candidates):
        try:
            if all(resolve_database_path(candidate, db_id).is_file() for db_id in database_ids):
                matches.append(candidate)
        except ValueError:
            continue
    matches = [
        candidate
        for candidate in matches
        if not any(parent != candidate and parent in candidate.parents for parent in matches)
    ]
    if len(matches) != 1:
        relative = [path.relative_to(root).as_posix() for path in matches]
        raise ValueError(f"database root must be discovered exactly once: {relative}")
    return matches[0]


def filesystem_capacity(target: str | Path) -> FilesystemCapacity:
    existing = Path(target).resolve()
    while not existing.exists():
        if existing.parent == existing:
            raise ValueError(f"no existing parent for capacity target: {target}")
        existing = existing.parent
    usage = shutil.disk_usage(existing)
    return FilesystemCapacity(
        existing_path=str(existing),
        device=existing.stat().st_dev,
        total_bytes=usage.total,
        used_bytes=usage.used,
        free_bytes=usage.free,
    )


def require_free_space(target: str | Path, minimum_free_bytes: int) -> FilesystemCapacity:
    capacity = filesystem_capacity(target)
    if capacity.free_bytes < minimum_free_bytes:
        raise ValueError(
            f"insufficient free space on target filesystem: "
            f"{capacity.free_bytes} < {minimum_free_bytes}"
        )
    return capacity


def evaluate_extraction_capacity(
    target: str | Path,
    *,
    observed_inner_uncompressed_bytes: int,
    heldout_expected_uncompressed_bytes: int,
    reserve_bytes: int,
) -> ExtractionCapacityGate:
    """Re-read target filesystem capacity after observing the nested ZIP."""
    required_bytes = (
        observed_inner_uncompressed_bytes + heldout_expected_uncompressed_bytes + reserve_bytes
    )
    capacity = filesystem_capacity(target)
    return ExtractionCapacityGate(
        filesystem=capacity,
        observed_inner_uncompressed_bytes=observed_inner_uncompressed_bytes,
        heldout_expected_uncompressed_bytes=heldout_expected_uncompressed_bytes,
        reserve_bytes=reserve_bytes,
        required_bytes=required_bytes,
        passed=capacity.free_bytes >= required_bytes,
    )


def require_extraction_capacity(gate: ExtractionCapacityGate) -> None:
    if not gate.passed:
        raise ValueError(
            "insufficient free space before nested extraction: "
            f"{gate.filesystem.free_bytes} < {gate.required_bytes}"
        )


def safe_extract_zip_after_capacity_gate(
    archive_path: str | Path,
    destination: str | Path,
    *,
    observation: ArchiveValidation,
    capacity_gate: ExtractionCapacityGate,
) -> ArchiveValidation:
    """Guarantee that the dynamic capacity gate precedes nested extraction."""
    require_extraction_capacity(capacity_gate)
    return safe_extract_zip(
        archive_path,
        destination,
        expected_member_count=observation.member_count,
        expected_uncompressed_bytes=observation.uncompressed_bytes,
    )


def require_outside_git_worktree(target: str | Path) -> None:
    candidate = Path(target).resolve()
    for parent in (candidate, *candidate.parents):
        if (parent / ".git").exists():
            raise ValueError(
                f"materialized data and private heldout evidence must stay outside Git: {target}"
            )


def atomic_publish_directory(partial: str | Path, final: str | Path) -> None:
    partial_path = Path(partial)
    final_path = Path(final)
    if not partial_path.is_dir():
        raise ValueError(f"partial directory does not exist: {partial_path}")
    if final_path.exists():
        raise ValueError(f"final directory already exists: {final_path}")
    if partial_path.stat().st_dev != final_path.parent.stat().st_dev:
        raise ValueError("partial and final directories must share a filesystem for atomic publish")
    os.replace(partial_path, final_path)


def write_json(path: str | Path, payload: object, *, private: bool = False) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    mode = 0o600 if private else 0o644
    descriptor = os.open(target, flags, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    os.chmod(target, mode)
    return hashlib.sha256(encoded).hexdigest()


def read_http_headers(path: str | Path) -> dict[str, str | int | None]:
    current: dict[str, str] = {}
    status: int | None = None
    for raw_line in Path(path).read_text(encoding="iso-8859-1").splitlines():
        line = raw_line.strip()
        if line.upper().startswith("HTTP/"):
            current = {}
            parts = line.split()
            status = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else None
        elif ":" in line:
            name, value = line.split(":", 1)
            current[name.strip().casefold()] = value.strip()
    if status not in {200, 206}:
        raise ValueError(f"final HTTP response must be 200 or 206, observed {status}: {path}")
    length = current.get("content-length")
    if length is None:
        raise ValueError(f"final HTTP response has no Content-Length: {path}")
    response_length = int(length)
    content_range = current.get("content-range")
    representation_length = response_length
    if status == 206:
        match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+)", content_range or "", re.IGNORECASE)
        if match is None:
            raise ValueError(f"HTTP 206 response has invalid Content-Range: {path}")
        start, end, total = (int(value) for value in match.groups())
        if end < start or end - start + 1 != response_length or end >= total:
            raise ValueError(f"HTTP 206 range and Content-Length disagree: {path}")
        representation_length = total
    return {
        "http_status": status,
        "response_content_length": response_length,
        "content_range": content_range,
        "content_length": representation_length,
        "etag": current.get("etag"),
        "last_modified": current.get("last-modified"),
    }


def build_source_receipt(
    config_path: str | Path,
    config: dict[str, Any],
    *,
    training_labels: str | Path,
    heldout_labels: str | Path,
    training_archive: str | Path,
    heldout_archive: str | Path,
    training_label_headers: str | Path,
    heldout_label_headers: str | Path,
    training_archive_headers: str | Path,
    heldout_archive_headers: str | Path,
) -> dict[str, object]:
    training_label_sha = sha256_file(training_labels)
    heldout_label_sha = sha256_file(heldout_labels)
    validate_observed_identity(
        expected_sha256=config["training_source"]["labels_observed_sha256"],
        observed_sha256=training_label_sha,
    )
    validate_observed_identity(
        expected_sha256=config["heldout_source"]["labels_observed_sha256"],
        observed_sha256=heldout_label_sha,
    )
    paths = {
        "training_labels": Path(training_labels),
        "heldout_labels": Path(heldout_labels),
        "training_archive": Path(training_archive),
        "heldout_archive": Path(heldout_archive),
    }
    observed_headers = {
        "training_labels": read_http_headers(training_label_headers),
        "heldout_labels": read_http_headers(heldout_label_headers),
        "training_archive": read_http_headers(training_archive_headers),
        "heldout_archive": read_http_headers(heldout_archive_headers),
    }
    expected_sizes = {
        "training_labels": config["training_source"]["labels_content_length"],
        "heldout_labels": config["heldout_source"]["labels_content_length"],
        "training_archive": config["training_source"]["database_archive_content_length"],
        "heldout_archive": config["heldout_source"]["archive_content_length"],
    }
    for key, path in paths.items():
        if path.stat().st_size != expected_sizes[key]:
            raise ValueError(
                f"source size changed for {key}: {path.stat().st_size} != {expected_sizes[key]}"
            )
        if observed_headers[key]["content_length"] != path.stat().st_size:
            raise ValueError(
                f"HTTP Content-Length does not match downloaded bytes for {key}: "
                f"{observed_headers[key]['content_length']} != {path.stat().st_size}"
            )
    previous_headers = {
        "training_labels": {
            "content_length": config["training_source"]["labels_content_length"],
            "etag": config["training_source"]["labels_etag"] or None,
            "last_modified": config["training_source"]["labels_last_modified"] or None,
        },
        "heldout_labels": {
            "content_length": config["heldout_source"]["labels_content_length"],
            "etag": config["heldout_source"]["labels_etag"] or None,
            "last_modified": config["heldout_source"]["labels_last_modified"] or None,
        },
        "training_archive": {
            "content_length": config["training_source"]["database_archive_content_length"],
            "etag": config["training_source"]["database_archive_etag"] or None,
            "last_modified": config["training_source"]["database_archive_last_modified"] or None,
        },
        "heldout_archive": {
            "content_length": config["heldout_source"]["archive_content_length"],
            "etag": config["heldout_source"]["archive_etag"] or None,
            "last_modified": config["heldout_source"]["archive_last_modified"] or None,
        },
    }
    observed_shas = {
        "training_labels": training_label_sha,
        "heldout_labels": heldout_label_sha,
        "training_archive": sha256_file(paths["training_archive"]),
        "heldout_archive": sha256_file(paths["heldout_archive"]),
    }
    warnings: list[str] = []
    for key in paths:
        expected_sha = (
            config["training_source"]["labels_observed_sha256"]
            if key == "training_labels"
            else config["heldout_source"]["labels_observed_sha256"]
            if key == "heldout_labels"
            else config["training_source"]["database_archive_observed_sha256"]
            if key == "training_archive"
            else config["heldout_source"]["archive_observed_sha256"]
            if key == "heldout_archive"
            else None
        )
        if expected_sha is not None:
            warnings.extend(
                f"{key}: {warning}"
                for warning in validate_observed_identity(
                    expected_sha256=expected_sha,
                    observed_sha256=observed_shas[key],
                    previous_headers=previous_headers[key],
                    observed_headers=observed_headers[key],
                )
            )
        elif any(
            previous_headers[key].get(name) != observed_headers[key].get(name)
            for name in ("content_length", "etag", "last_modified")
        ):
            warnings.append(
                f"{key}: observed HTTP headers differ from Day 3A metadata; "
                "the first downloaded archive SHA-256 is authoritative"
            )
    return {
        "schema_version": 1,
        "config_sha256": sha256_file(config_path),
        "provenance_warnings": warnings,
        "artifacts": {
            "training_labels": {
                "logical_name": "training-labels.jsonl",
                "url": config["training_source"]["labels_url"],
                "revision": config["training_source"]["revision"],
                **observed_headers["training_labels"],
                "sha256": training_label_sha,
            },
            "heldout_labels": {
                "logical_name": "heldout-labels.json",
                "url": config["heldout_source"]["labels_url"],
                "revision": config["heldout_source"]["revision"],
                **observed_headers["heldout_labels"],
                "sha256": heldout_label_sha,
            },
            "training_archive": {
                "logical_name": "train.zip",
                "url": config["training_source"]["database_url"],
                **observed_headers["training_archive"],
                "sha256": observed_shas["training_archive"],
            },
            "heldout_archive": {
                "logical_name": "minidev.zip",
                "url": config["heldout_source"]["archive_url"],
                **observed_headers["heldout_archive"],
                "sha256": observed_shas["heldout_archive"],
            },
        },
    }
