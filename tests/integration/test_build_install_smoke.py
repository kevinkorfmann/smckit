"""Build/install smoke tests for packaged native quickstarts."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_uv_built_wheel_installs_and_runs_packaged_native_quickstarts(tmp_path: Path) -> None:
    if shutil.which("uv") is None:
        pytest.skip("uv is not installed")

    env = dict(os.environ)
    env["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")

    dist_dir = tmp_path / "dist"
    subprocess.run(
        [
            "uv",
            "build",
            "--offline",
            "--no-build-isolation",
            "--wheel",
            "--sdist",
            "--out-dir",
            str(dist_dir),
        ],
        check=True,
        cwd=ROOT,
        env=env,
    )
    wheel = next(dist_dir.glob("smckit-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        packaged = set(archive.namelist())
    assert "smckit/__init__.py" in packaged
    assert "smckit/tl/_smcpp.py" in packaged
    assert "smckit/data/examples/smcpp/example.smc.gz" in packaged

    venv_dir = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True, cwd=ROOT)
    python = venv_dir / "bin" / "python"
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), str(wheel)],
        check=True,
        cwd=ROOT,
        env=env,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    script = """
import os
import warnings

import smckit
from smckit.tl._implementation import NativeTrustWarning

os.chdir(%r)
example = smckit.io.example_path("psmc/NA12878_chr22.psmcfa")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    data = smckit.io.read_psmcfa(example)
    data = smckit.tl.psmc(data, implementation="native", n_iterations=2, seed=1)
    assert data.results["psmc"]["implementation"] == "native"
    assert not any(isinstance(item.message, NativeTrustWarning) for item in caught)

smcpp_example = smckit.io.example_path("smcpp/example.smc.gz")
smcpp_data = smckit.io.read_smcpp_input(smcpp_example)
smcpp_data = smckit.tl.smcpp(
    smcpp_data,
    implementation="native",
    n_intervals=2,
    max_iterations=0,
    regularization=10.0,
)
assert smcpp_data.results["smcpp"]["implementation"] == "native"
assert smcpp_data.results["smcpp"]["n_distinguished"] == 2
""" % str(run_dir)
    subprocess.run([str(python), "-c", script], check=True, cwd=ROOT, env=env)
