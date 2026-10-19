import io
import struct
import zipfile

from verireason.remote_zip import parse_central_directory, parse_zip64_eocd


def test_central_directory_parser_sums_member_sizes() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("one.txt", "hello" * 20)
        archive.writestr("nested/two.txt", "world" * 30)
    payload = buffer.getvalue()
    end = payload.rfind(b"PK\x05\x06")
    fields = struct.unpack_from("<4s4H2LH", payload, end)
    entries, central_size, central_offset = fields[4], fields[5], fields[6]

    parsed_entries, compressed, uncompressed = parse_central_directory(
        payload[central_offset : central_offset + central_size], entries
    )

    assert parsed_entries == 2
    assert uncompressed == 250
    assert 0 < compressed < uncompressed


def test_zip64_eocd_parser_reads_large_offsets() -> None:
    record = struct.pack(
        "<4sQ2H2L4Q",
        b"PK\x06\x06",
        44,
        45,
        45,
        0,
        0,
        246,
        246,
        36_934,
        8_919_000_000,
    )

    assert parse_zip64_eocd(record) == (246, 36_934, 8_919_000_000)
