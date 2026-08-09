"""Exact, shell-free execution of preserved upstream command-line tools."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from smckit._provenance import sha256_file
from smckit.upstream._registry import get_tool


@dataclass(frozen=True)
class UpstreamRunResult:
    """Captured result of one original-tool execution."""

    tool: str
    command: list[str]
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    runtime_seconds: float
    artifacts: list[dict[str, Any]]
    compatibility_patches: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _existing_executable(path: Path, *, tool: str) -> str:
    if not path.exists():
        raise RuntimeError(
            f"Upstream {tool} executable is not ready at {path}. "
            f"Run smckit.upstream.bootstrap('{tool}') first."
        )
    return str(path)


def command_prefix(tool: str, entrypoint: str | None = None) -> list[str]:
    """Resolve the preserved command prefix for one upstream tool."""
    spec = get_tool(tool)
    status = spec.status()
    cache_path = Path(status["cache_path"])
    vendor_path = None if status["vendor_path"] is None else Path(status["vendor_path"])
    runtime_path = status["runtime"]["path"]

    if tool == "psmc":
        entrypoint = entrypoint or "psmc"
        allowed = {
            "avg.pl",
            "calD",
            "cntcpg",
            "ctime_plot.pl",
            "dec2ctime.pl",
            "decode2bed.pl",
            "fq2psmcfa",
            "history2ms.pl",
            "ms2psmcfa.pl",
            "mutDiff",
            "pcnt_bezier.lua",
            "psmc",
            "psmc2history.pl",
            "psmc_plot.pl",
            "psmc_trunc.pl",
            "split-time.js",
            "splitfa",
            "vcf2snp.pl",
        }
        if entrypoint not in allowed:
            choices = ", ".join(sorted(allowed))
            raise ValueError(f"Unknown PSMC entry point {entrypoint!r}; choose from: {choices}.")
        cached = cache_path / "bin/psmc"
        if entrypoint != "psmc":
            cached = cache_path / "bin" / entrypoint
        vendored = (
            None
            if vendor_path is None
            else vendor_path / ("psmc" if entrypoint == "psmc" else f"utils/{entrypoint}")
        )
        candidate = cached if cached.exists() or vendored is None else vendored
        executable = _existing_executable(candidate, tool=f"{tool}/{entrypoint}")
        if entrypoint.endswith(".js"):
            runtime = "k8" if entrypoint == "split-time.js" else "node"
            resolved = shutil.which(runtime)
            if resolved is None:
                raise RuntimeError(
                    f"Original PSMC utility {entrypoint} requires the {runtime} runtime."
                )
            return [resolved, executable]
        if entrypoint.endswith(".lua"):
            resolved = shutil.which("luajit")
            if resolved is None:
                raise RuntimeError(
                    f"Original PSMC utility {entrypoint} requires the luajit runtime."
                )
            return [resolved, executable]
        return [executable]
    if tool == "msmc2":
        candidates = [
            cache_path / "bin/msmc2",
            *(
                ()
                if vendor_path is None
                else (vendor_path / "build/release/msmc2", vendor_path / "build/msmc2")
            ),
        ]
        candidate = next((path for path in candidates if path.exists()), candidates[0])
        return [_existing_executable(candidate, tool=tool)]
    if tool == "msmc_im":
        if vendor_path is None:
            raise RuntimeError("Vendored MSMC-IM source is unavailable.")
        return [sys.executable, str(vendor_path / "MSMC_IM.py")]
    if tool == "smcpp":
        if runtime_path is None:
            raise RuntimeError(
                "The preserved SMC++ Python environment is unavailable. "
                "Set SMCKIT_SMCPP_PYTHON to its interpreter."
            )
        return [str(runtime_path), "-m", "smcpp"]
    if tool == "esmc2":
        if runtime_path is None:
            raise RuntimeError(
                "Rscript is unavailable. Supply an R script as the first raw argument "
                "after installing/bootstraping eSMC2."
            )
        return [str(runtime_path)]
    if tool == "asmc":
        binary = cache_path / "bin/ASMC_exe"
        return [_existing_executable(binary, tool=tool)]
    if tool == "dical2":
        if runtime_path is None or vendor_path is None:
            raise RuntimeError("Java or the vendored diCal2 jar is unavailable.")
        return [str(runtime_path), "-jar", str(vendor_path / "diCal2.jar")]
    if tool == "psmcplus":
        entrypoint = entrypoint or "PSMCplus.py"
        allowed = {"PSMCplus.py", "simulate_HMM.py"}
        if entrypoint not in allowed:
            choices = ", ".join(sorted(allowed))
            raise ValueError(f"Unknown PSMC+ entry point {entrypoint!r}; choose from: {choices}.")
        if runtime_path is None or vendor_path is None:
            raise RuntimeError(
                "The preserved PSMC+ source or Python dependency stack is unavailable. "
                "Install smckit[psmcplus] or set SMCKIT_PSMCPLUS_PYTHON."
            )
        runner = Path(__file__).with_name("_psmcplus_runner.py")
        return [str(runtime_path), str(runner), str(vendor_path / entrypoint)]
    if tool == "phlash":
        raise RuntimeError(
            "PHLASH 1.0.6 intentionally has no command-line interface. Use "
            "smckit.tl.phlash(...) for the normalized external execution path or "
            "import phlash directly for its complete original Python API."
        )
    raise KeyError(f"No raw upstream command is registered for {tool}.")


def _artifact_manifest(directory: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        artifacts.append(
            {
                "path": str(path.relative_to(directory)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return artifacts


def _compatibility_patches(tool: str) -> list[str]:
    if tool == "psmcplus":
        return [
            "Restore numpy.math from the Python math module when absent; "
            "vendored source unchanged."
        ]
    return []


def run(
    tool: str,
    args: Sequence[str],
    *,
    output_dir: str | Path | None = None,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
    entrypoint: str | None = None,
) -> UpstreamRunResult:
    """Execute an original tool without a shell and capture its artifacts.

    Relative input paths should be resolved by the caller before execution,
    because the command runs in an isolated output directory.
    """
    prefix = command_prefix(tool) if entrypoint is None else command_prefix(tool, entrypoint)
    caller_directory = Path.cwd()
    resolved_args = [
        str((caller_directory / value).resolve())
        if not Path(value).is_absolute() and (caller_directory / value).exists()
        else value
        for value in map(str, args)
    ]
    command = [*prefix, *resolved_args]
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if output_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix=f"smckit-{tool}-")
        workdir = Path(temporary.name)
    else:
        workdir = Path(output_dir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)

    process_env = os.environ.copy()
    if tool == "esmc2":
        r_library = get_tool("esmc2").cache_path
        if r_library.exists():
            process_env.setdefault("R_LIBS_USER", str(r_library))
    if env:
        process_env.update({str(key): str(value) for key, value in env.items()})

    started = time.perf_counter()
    try:
        try:
            completed = subprocess.run(
                command,
                cwd=workdir,
                env=process_env,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            runtime_seconds = time.perf_counter() - started
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return UpstreamRunResult(
                tool=tool,
                command=command,
                cwd=str(workdir),
                returncode=124,
                stdout=stdout,
                stderr=f"{stderr}\nExecution exceeded timeout of {timeout} seconds.".lstrip(),
                runtime_seconds=runtime_seconds,
                artifacts=_artifact_manifest(workdir),
                compatibility_patches=_compatibility_patches(tool),
            )
        runtime_seconds = time.perf_counter() - started
        return UpstreamRunResult(
            tool=tool,
            command=command,
            cwd=str(workdir),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            runtime_seconds=runtime_seconds,
            artifacts=_artifact_manifest(workdir),
            compatibility_patches=_compatibility_patches(tool),
        )
    finally:
        if temporary is not None:
            temporary.cleanup()


__all__ = ["UpstreamRunResult", "command_prefix", "run"]
