from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("ufb") / "data"
    script = Path(__file__).parents[1] / "scripts" / "make_synthetic_urbanfloodbench.py"
    subprocess.run([sys.executable, str(script), "--output", str(root), "--models", "3",
                    "--events", "3", "--steps", "20", "--seed", "7"], check=True)
    return root
