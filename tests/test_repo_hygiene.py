"""Guards against committing local state or secrets."""

import subprocess
from pathlib import Path

import pytest

FORBIDDEN_PREFIXES = ("data/", "uploads/")
FORBIDDEN_SUFFIXES = (".db", ".sqlite", ".sqlite3", ".pth")


def tracked_files() -> list[str]:
    root = Path(__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("git is not available or this is not a git repository")
    return result.stdout.splitlines()


def test_no_local_state_or_secrets_are_tracked():
    bad = [
        path
        for path in tracked_files()
        if path.startswith(FORBIDDEN_PREFIXES)
        or path.endswith(FORBIDDEN_SUFFIXES)
        or Path(path).name == ".env"
    ]
    assert bad == [], f"These files must not be tracked by git: {bad}"