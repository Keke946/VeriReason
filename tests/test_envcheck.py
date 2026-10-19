from verireason.envcheck import _base_distribution_version, validate_environment


def test_cuda_local_version_matches_base_distribution() -> None:
    assert _base_distribution_version("2.11.0+cu130") == "2.11.0"


def test_local_environment_accepts_matching_python() -> None:
    report = {
        "profile": "local",
        "python": "3.11.15",
        "system": "Darwin",
        "machine": "arm64",
        "packages": {},
        "cuda": {"available": False},
        "expected": {"python_major_minor": "3.11"},
    }

    assert validate_environment(report) == []


def test_gpu_environment_rejects_old_driver_and_missing_package() -> None:
    report = {
        "profile": "gpu",
        "python": "3.11.15",
        "system": "Linux",
        "machine": "x86_64",
        "packages": {"torch": None},
        "cuda": {
            "available": True,
            "runtime": "13.0",
            "driver": "570.00",
            "device_count": 1,
            "devices": [{"index": 0, "total_memory_gib": 47.5}],
        },
        "expected": {
            "python_major_minor": "3.11",
            "system": "Linux",
            "machine": "x86_64",
            "packages": {"torch": "2.11.0"},
            "cuda_runtime_major_minor": "13.0",
            "minimum_nvidia_driver": "580.65.06",
            "minimum_cuda_devices": 1,
            "minimum_device_memory_gib": 45.0,
        },
    }

    errors = validate_environment(report)

    assert "torch missing does not match 2.11.0" in errors
    assert any("driver" in error for error in errors)
