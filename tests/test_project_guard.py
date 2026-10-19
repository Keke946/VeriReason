import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_guard_module():
    path = ROOT / "scripts/check_project_state.py"
    spec = importlib.util.spec_from_file_location("check_project_state", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_files_exclude_ignored_virtual_environment() -> None:
    module = _load_guard_module()

    relative_paths = {path.relative_to(ROOT).as_posix() for path in module.repository_files()}

    assert not any(path.startswith(".venv/") for path in relative_paths)
    assert "scripts/check_project_state.py" in relative_paths
