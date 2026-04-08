"""Shared upstream tool registry and local bootstrap helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smckit.settings import settings
from smckit.upstream._install import install_help


def repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[3]


def _cache_root() -> Path:
    cache = Path(settings.upstream_cache_dir)
    if not cache.is_absolute():
        cache = repo_root() / cache
    return cache


def _runtime_override(tool_name: str, env_var: str | None = None) -> str | None:
    if tool_name in settings.upstream_runtime_overrides:
        return settings.upstream_runtime_overrides[tool_name]
    if env_var:
        override = os.environ.get(env_var)
        if override:
            return override
    return None


def _find_executable(name: str, *, env_var: str | None = None, tool_name: str) -> str | None:
    override = _runtime_override(tool_name, env_var=env_var)
    if override:
        return override
    return shutil.which(name)


def _find_java_executable(*, env_var: str | None = None, tool_name: str) -> str | None:
    override = _runtime_override(tool_name, env_var=env_var)
    if override:
        return override
    candidates: list[str] = []
    found = shutil.which("java")
    if found:
        candidates.append(found)
    for candidate in (
        Path("/opt/homebrew/opt/openjdk/bin/java"),
        Path("/usr/local/opt/openjdk/bin/java"),
    ):
        if candidate.exists():
            candidates.append(str(candidate))
    for candidate in candidates:
        probe = subprocess.run(
            [candidate, "-version"],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            return candidate
    return None


@dataclass(frozen=True)
class UpstreamToolSpec:
    """Description of a vendored upstream tool and its bootstrap contract."""

    name: str
    method_name: str
    vendor_subpath: str | None
    runtime_name: str
    runtime_executable: str | None = None
    runtime_env_var: str | None = None
    bootstrap_summary: str = ""
    cache_outputs: tuple[str, ...] = ()
    bootstrap_commands: tuple[tuple[str, ...], ...] = ()
    notes: str = ""
    version: str = "vendored"
    requires_vendor_source: bool = True
    public_upstream: bool = False
    adapter_ready: bool = False

    @property
    def vendor_path(self) -> Path | None:
        if self.vendor_subpath is None:
            return None
        return repo_root() / self.vendor_subpath

    @property
    def cache_path(self) -> Path:
        if self.name == "esmc2":
            return repo_root() / ".r-lib"
        return _cache_root() / self.name

    def runtime_path(self) -> str | None:
        if self.name == "dical2":
            return _find_java_executable(
                env_var=self.runtime_env_var,
                tool_name=self.name,
            )
        if self.name == "asmc":
            if importlib.util.find_spec("asmc") is not None:
                return sys.executable
            return _find_executable(
                self.runtime_executable,
                env_var=self.runtime_env_var,
                tool_name=self.name,
            )
        if self.name == "msmc_im":
            return sys.executable
        if self.name == "smcpp":
            override = _runtime_override(self.name, env_var=self.runtime_env_var)
            if override:
                return override
            default_python = repo_root() / "vendor/smcpp/.venv/bin/python"
            if default_python.exists():
                return str(default_python)
            legacy_python = Path("/tmp/smcpp39/bin/python")
            if legacy_python.exists():
                return str(legacy_python)
            return None
        if self.runtime_executable is None:
            return None
        return _find_executable(
            self.runtime_executable,
            env_var=self.runtime_env_var,
            tool_name=self.name,
        )

    def source_present(self) -> bool:
        if not self.requires_vendor_source:
            return True
        return self.vendor_path is not None and self.vendor_path.exists()

    def cache_ready(self) -> bool:
        if self.name == "esmc2":
            return (repo_root() / ".r-lib/eSMC2").exists()
        if self.name == "asmc" and importlib.util.find_spec("asmc") is not None:
            return True
        if self.name == "dical2":
            return self.vendor_path is not None and (self.vendor_path / "diCal2.jar").exists()
        if not self.cache_outputs:
            return True
        return all((self.cache_path / rel).exists() for rel in self.cache_outputs)

    def runtime_ready(self) -> bool:
        if self.name == "dical2":
            java = self.runtime_path()
            if java is None:
                return False
            probe = subprocess.run(
                [java, "-version"],
                check=False,
                capture_output=True,
                text=True,
            )
            return probe.returncode == 0
        if self.runtime_executable is None:
            return True
        return self.runtime_path() is not None

    def ready(self) -> bool:
        if not self.public_upstream or not self.adapter_ready:
            return False
        return self.source_present() and self.runtime_ready() and self.cache_ready()

    def status(self) -> dict[str, Any]:
        missing: list[str] = []
        if not self.public_upstream:
            missing.append("public upstream bridge not implemented")
        elif not self.adapter_ready:
            missing.append("upstream adapter not ready")
        if self.requires_vendor_source and not self.source_present():
            missing.append("vendored source missing")
        if not self.runtime_ready():
            missing.append(f"{self.runtime_name} runtime missing")
        if self.cache_outputs and not self.cache_ready():
            missing.append("bootstrap artifacts missing")

        return {
            "tool": self.name,
            "method": self.method_name,
            "vendor_path": None if self.vendor_path is None else str(self.vendor_path),
            "cache_path": str(self.cache_path),
            "runtime": {
                "name": self.runtime_name,
                "path": self.runtime_path(),
                "env_var": self.runtime_env_var,
            },
            "version": self.version,
            "source_present": self.source_present(),
            "runtime_ready": self.runtime_ready(),
            "cache_ready": self.cache_ready(),
            "public_upstream": self.public_upstream,
            "adapter_ready": self.adapter_ready,
            "ready": self.ready(),
            "bootstrap_summary": self.bootstrap_summary,
            "bootstrap_commands": [list(cmd) for cmd in self.bootstrap_commands],
            "missing": missing,
            "install_help": install_help(self.name, source_present=self.source_present()),
            "notes": self.notes,
        }


REGISTRY: dict[str, UpstreamToolSpec] = {
    "psmc": UpstreamToolSpec(
        name="psmc",
        method_name="psmc",
        vendor_subpath="vendor/psmc",
        runtime_name="C compiler and make",
        runtime_executable="make",
        bootstrap_summary="Build vendored psmc and copy the binary into the upstream cache.",
        cache_outputs=("bin/psmc",),
        bootstrap_commands=(("make",),),
        notes="Build currently happens from the vendored source tree; the cache stores the executable used by smckit.",
        public_upstream=True,
        adapter_ready=True,
    ),
    "msmc2": UpstreamToolSpec(
        name="msmc2",
        method_name="msmc2",
        vendor_subpath="vendor/msmc2",
        runtime_name="D compiler and make",
        runtime_executable="make",
        bootstrap_summary="Build vendored MSMC2 and copy the executable into the upstream cache.",
        cache_outputs=("bin/msmc2",),
        bootstrap_commands=(("make",),),
        notes="Requires the upstream D toolchain; smckit runs the vendored CLI when the binary is available.",
        public_upstream=True,
        adapter_ready=True,
    ),
    "msmc_im": UpstreamToolSpec(
        name="msmc_im",
        method_name="msmc_im",
        vendor_subpath="vendor/MSMC-IM",
        runtime_name="current Python",
        runtime_executable=None,
        bootstrap_summary="No build step; uses the vendored MSMC_IM.py script directly.",
        notes="Vendored Python script already acts as the oracle entrypoint.",
        public_upstream=True,
        adapter_ready=True,
    ),
    "smcpp": UpstreamToolSpec(
        name="smcpp",
        method_name="smcpp",
        vendor_subpath="vendor/smcpp",
        runtime_name="Python side environment",
        runtime_executable="python",
        runtime_env_var="SMCKIT_SMCPP_PYTHON",
        bootstrap_summary="Use the vendored upstream source tree with a controlled side Python environment for the compiled extension/runtime.",
        notes="Vendored source is tracked in-repo; the current upstream bridge still relies on a dedicated Python environment to execute the original package.",
        version="6779faec78f1db2d84b3cd8176cd99731c71d584",
        public_upstream=True,
        adapter_ready=True,
    ),
    "esmc2": UpstreamToolSpec(
        name="esmc2",
        method_name="esmc2",
        vendor_subpath="vendor/eSMC2",
        runtime_name="Rscript",
        runtime_executable="Rscript",
        runtime_env_var="SMCKIT_ESMC2_RSCRIPT",
        bootstrap_summary="Install the vendored eSMC2 R package into the repo-local .r-lib cache.",
        cache_outputs=("eSMC2/DESCRIPTION",),
        bootstrap_commands=(
            ("R", "CMD", "INSTALL", "--library", ".r-lib", "vendor/eSMC2/eSMC2"),
        ),
        notes="The upstream bridge expects a local R library containing the vendored eSMC2 package.",
        public_upstream=True,
        adapter_ready=True,
    ),
    "asmc": UpstreamToolSpec(
        name="asmc",
        method_name="asmc",
        vendor_subpath="vendor/ASMC",
        runtime_name="PyPI package or CMake/C++ toolchain",
        runtime_executable="cmake",
        bootstrap_summary="Prefer the official PyPI package `asmc-asmc`; fall back to building vendored ASMC_exe.",
        cache_outputs=("bin/ASMC_exe",),
        bootstrap_commands=(("cmake", "-S", "vendor/ASMC", "-B", ".smckit-cache/upstream/asmc/build"), ("cmake", "--build", ".smckit-cache/upstream/asmc/build", "--target", "ASMC_exe")),
        notes="smckit prefers the official PyPI `asmc-asmc` module and falls back to the documented ASMC_exe CLI path.",
        public_upstream=True,
        adapter_ready=True,
    ),
    "dical2": UpstreamToolSpec(
        name="dical2",
        method_name="dical2",
        vendor_subpath="vendor/diCal2",
        runtime_name="Java",
        runtime_executable="java",
        runtime_env_var="SMCKIT_DICAL2_JAVA",
        bootstrap_summary="No build step; uses the vendored diCal2.jar directly.",
        notes="smckit runs the vendored diCal2.jar and parses the EM-path stdout into structured results.",
        public_upstream=True,
        adapter_ready=True,
    ),
}


def tool_names() -> list[str]:
    return sorted(REGISTRY)


def get_tool(name: str) -> UpstreamToolSpec:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown upstream tool: {name}") from exc


def get_tool_for_method(method_name: str) -> UpstreamToolSpec | None:
    for spec in REGISTRY.values():
        if spec.method_name == method_name:
            return spec
    return None


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def bootstrap_tool(name: str) -> dict[str, Any]:
    spec = get_tool(name)
    status = spec.status()
    if not spec.source_present():
        raise RuntimeError(
            "Cannot bootstrap upstream "
            f"{name}: vendored source is missing from {status['vendor_path']}.\n"
            f"{status['install_help']}"
        )
    if not spec.runtime_ready():
        raise RuntimeError(
            "Cannot bootstrap upstream "
            f"{name}: {spec.runtime_name} runtime is unavailable.\n"
            f"{status['install_help']}"
        )

    cache_path = spec.cache_path
    cache_path.mkdir(parents=True, exist_ok=True)

    if name == "esmc2":
        vendor_pkg = repo_root() / "vendor/eSMC2/eSMC2"
        local_r_lib = repo_root() / ".r-lib"
        local_r_lib.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["R", "CMD", "INSTALL", "--library", str(local_r_lib), str(vendor_pkg)],
            check=True,
            cwd=repo_root(),
        )
    elif name == "psmc":
        subprocess.run(["make"], check=True, cwd=spec.vendor_path)
        built = spec.vendor_path / "psmc"
        if not built.exists():
            raise RuntimeError("Bootstrapping psmc did not produce a psmc binary.")
        _copy_file(built, cache_path / "bin/psmc")
    elif name == "msmc2":
        subprocess.run(["make"], check=True, cwd=spec.vendor_path)
        candidates = [
            spec.vendor_path / "build/msmc2",
            spec.vendor_path / "build/release/msmc2",
            spec.vendor_path / "msmc2",
        ]
        built = next((path for path in candidates if path.exists()), None)
        if built is None:
            raise RuntimeError("Bootstrapping msmc2 did not produce an msmc2 executable.")
        _copy_file(built, cache_path / "bin/msmc2")
    elif name == "asmc":
        build_dir = cache_path / "build"
        subprocess.run(
            ["cmake", "-S", str(spec.vendor_path), "-B", str(build_dir)],
            check=True,
            cwd=repo_root(),
        )
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--target", "ASMC_exe"],
            check=True,
            cwd=repo_root(),
        )
        candidates = [build_dir / "ASMC_exe"]
        built = next((path for path in candidates if path.exists()), None)
        if built is None:
            raise RuntimeError("Bootstrapping asmc did not produce an ASMC_exe executable.")
        _copy_file(built, cache_path / "bin/ASMC_exe")
    elif name in {"msmc_im", "dical2"}:
        # No build required; presence of vendored script/jar is the bootstrap contract.
        pass
    else:
        raise RuntimeError(
            f"Upstream bootstrap for {name} is not implemented yet."
        )

    return get_tool(name).status()
