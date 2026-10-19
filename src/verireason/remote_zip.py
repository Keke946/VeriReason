"""Read a remote ZIP central directory without downloading the archive body."""

from __future__ import annotations

import json
import struct
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

_EOCD = struct.Struct("<4s4H2LH")
_ZIP64_LOCATOR = struct.Struct("<4sLQL")
_ZIP64_EOCD = struct.Struct("<4sQ2H2L4Q")
_CENTRAL = struct.Struct("<4s6H3L5H2L")
_MAX_TAIL_BYTES = 65_557
_MAX_CENTRAL_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class ZipDirectorySummary:
    final_url: str
    content_length: int
    etag: str | None
    last_modified: str | None
    accept_ranges: str | None
    entry_count: int
    member_compressed_bytes: int
    member_uncompressed_bytes: int
    central_directory_bytes: int
    metadata_bytes_fetched: int


def _request(url: str, method: str = "GET", byte_range: tuple[int, int] | None = None):
    headers = {"User-Agent": "VeriReason-M2/1.0"}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    return urllib.request.urlopen(  # noqa: S310 -- URL is an explicit operator input.
        urllib.request.Request(url, method=method, headers=headers), timeout=30
    )


def _zip64_sizes(extra: bytes, compressed: int, uncompressed: int) -> tuple[int, int]:
    cursor = 0
    while cursor + 4 <= len(extra):
        field_id, size = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        payload = extra[cursor : cursor + size]
        cursor += size
        if field_id != 0x0001:
            continue
        offset = 0
        if uncompressed == 0xFFFFFFFF:
            uncompressed = struct.unpack_from("<Q", payload, offset)[0]
            offset += 8
        if compressed == 0xFFFFFFFF:
            compressed = struct.unpack_from("<Q", payload, offset)[0]
        return compressed, uncompressed
    if compressed == 0xFFFFFFFF or uncompressed == 0xFFFFFFFF:
        raise ValueError("ZIP64 sizes are required but the ZIP64 extra field is missing")
    return compressed, uncompressed


def parse_central_directory(data: bytes, expected_entries: int) -> tuple[int, int, int]:
    cursor = 0
    entries = 0
    compressed_total = 0
    uncompressed_total = 0
    while cursor < len(data):
        if cursor + _CENTRAL.size > len(data):
            raise ValueError("truncated central-directory entry")
        fields = _CENTRAL.unpack_from(data, cursor)
        if fields[0] != b"PK\x01\x02":
            raise ValueError(f"invalid central-directory signature at byte {cursor}")
        compressed = fields[8]
        uncompressed = fields[9]
        name_length, extra_length, comment_length = fields[10:13]
        entry_size = _CENTRAL.size + name_length + extra_length + comment_length
        if cursor + entry_size > len(data):
            raise ValueError("central-directory variable fields are truncated")
        extra_start = cursor + _CENTRAL.size + name_length
        extra = data[extra_start : extra_start + extra_length]
        compressed, uncompressed = _zip64_sizes(extra, compressed, uncompressed)
        compressed_total += compressed
        uncompressed_total += uncompressed
        entries += 1
        cursor += entry_size
    if cursor != len(data) or entries != expected_entries:
        raise ValueError(
            f"central-directory count mismatch: parsed {entries}, expected {expected_entries}"
        )
    return entries, compressed_total, uncompressed_total


def _range_get(url: str, start: int, end: int) -> tuple[bytes, str]:
    with _request(url, byte_range=(start, end)) as response:
        status = getattr(response, "status", response.getcode())
        if status != 206:
            raise RuntimeError(f"server did not honor HTTP Range request: status {status}")
        data = response.read(end - start + 2)
        if len(data) != end - start + 1:
            raise RuntimeError(
                f"range length mismatch: received {len(data)}, expected {end - start + 1}"
            )
        return data, response.geturl()


def parse_zip64_eocd(data: bytes) -> tuple[int, int, int]:
    if len(data) < _ZIP64_EOCD.size:
        raise ValueError("truncated ZIP64 end-of-central-directory record")
    fields = _ZIP64_EOCD.unpack_from(data)
    if fields[0] != b"PK\x06\x06" or fields[1] < 44:
        raise ValueError("invalid ZIP64 end-of-central-directory record")
    disk_number, central_disk = fields[4], fields[5]
    disk_entries, total_entries, central_size, central_offset = fields[6:10]
    if disk_number != 0 or central_disk != 0 or disk_entries != total_entries:
        raise ValueError("multi-disk ZIP64 archives are unsupported")
    return total_entries, central_size, central_offset


def inspect_remote_zip(url: str) -> ZipDirectorySummary:
    with _request(url, method="HEAD") as response:
        headers = response.headers
        final_url = response.geturl()
        length_raw = headers.get("Content-Length")
        if not length_raw:
            raise RuntimeError("remote archive has no Content-Length")
        content_length = int(length_raw)
        provenance: dict[str, Any] = {
            "etag": headers.get("ETag"),
            "last_modified": headers.get("Last-Modified"),
            "accept_ranges": headers.get("Accept-Ranges"),
        }

    tail_size = min(content_length, _MAX_TAIL_BYTES)
    tail, final_url = _range_get(final_url, content_length - tail_size, content_length - 1)
    marker = tail.rfind(b"PK\x05\x06")
    if marker < 0 or marker + _EOCD.size > len(tail):
        raise ValueError("ZIP end-of-central-directory record not found")
    (
        signature,
        disk_number,
        central_disk,
        disk_entries,
        total_entries,
        central_size,
        central_offset,
        comment_length,
    ) = _EOCD.unpack_from(tail, marker)
    if signature != b"PK\x05\x06" or disk_number != 0 or central_disk != 0:
        raise ValueError("multi-disk or invalid ZIP archives are unsupported")
    if disk_entries != total_entries:
        raise ValueError("multi-disk ZIP entry counts are unsupported")
    if marker + _EOCD.size + comment_length > len(tail):
        raise ValueError("ZIP comment is truncated")
    metadata_bytes = tail_size
    if total_entries == 0xFFFF or central_size == 0xFFFFFFFF or central_offset == 0xFFFFFFFF:
        locator_offset = marker - _ZIP64_LOCATOR.size
        if locator_offset < 0:
            raise ValueError("ZIP64 locator is missing from the archive tail")
        locator = _ZIP64_LOCATOR.unpack_from(tail, locator_offset)
        if locator[0] != b"PK\x06\x07" or locator[1] != 0 or locator[3] != 1:
            raise ValueError("invalid or multi-disk ZIP64 locator")
        zip64_record, final_url = _range_get(
            final_url, locator[2], locator[2] + _ZIP64_EOCD.size - 1
        )
        metadata_bytes += len(zip64_record)
        total_entries, central_size, central_offset = parse_zip64_eocd(zip64_record)

    if central_size > _MAX_CENTRAL_BYTES:
        raise ValueError(f"central directory exceeds {_MAX_CENTRAL_BYTES} byte safety limit")
    if central_offset + central_size > content_length:
        raise ValueError("central directory lies outside the advertised archive length")

    central, final_url = _range_get(final_url, central_offset, central_offset + central_size - 1)
    entries, compressed, uncompressed = parse_central_directory(central, total_entries)
    return ZipDirectorySummary(
        final_url=final_url,
        content_length=content_length,
        etag=provenance["etag"],
        last_modified=provenance["last_modified"],
        accept_ranges=provenance["accept_ranges"],
        entry_count=entries,
        member_compressed_bytes=compressed,
        member_uncompressed_bytes=uncompressed,
        central_directory_bytes=central_size,
        metadata_bytes_fetched=metadata_bytes + central_size,
    )


def summary_json(summary: ZipDirectorySummary) -> str:
    return json.dumps(asdict(summary), indent=2, sort_keys=True)
