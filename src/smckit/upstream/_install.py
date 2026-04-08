"""Platform-aware install guidance for optional upstream runtimes."""

from __future__ import annotations

import platform


def _platform_key() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    return "linux"


def _join_lines(lines: list[str]) -> str:
    return "\n".join(lines)


def install_help(tool: str, *, source_present: bool) -> str:
    """Return actionable install guidance for an upstream tool."""
    os_name = _platform_key()
    sections: list[str] = []
    extra_hint = f'pip install "smckit[{tool}]"'

    if not source_present and tool in {"psmc", "msmc2", "msmc_im", "esmc2", "dical2"}:
        sections.append(
            "This install does not include vendored upstream sources. "
            "Use a source checkout for upstream mode:\n"
            "  git clone https://github.com/kevinkorfmann/smckit.git\n"
            "  cd smckit\n"
            "  pip install -e \".[dev]\""
        )

    if tool == "asmc":
        if os_name in {"macos", "linux"}:
            sections.append(
                "Install the packaged ASMC runtime:\n"
                "  pip install \"smckit[asmc]\""
            )
        else:
            sections.append(
                "ASMC wheels are not expected on native Windows. "
                "Use WSL2 or run the native smckit implementation."
            )
        return _join_lines(sections)

    if tool == "esmc2":
        if os_name == "macos":
            cmd = "brew install --cask r"
        elif os_name == "windows":
            cmd = "winget install RProject.R"
        else:
            cmd = "sudo apt-get install r-base"
        sections.append(
            "Install the package extras and R runtime for eSMC2, then bootstrap it:\n"
            f"  {extra_hint}\n"
            f"  {cmd}\n"
            "  python -c \"import smckit; smckit.upstream.bootstrap('esmc2')\""
        )
        return _join_lines(sections)

    if tool == "dical2":
        if os_name == "macos":
            cmd = "brew install --cask temurin"
        elif os_name == "windows":
            cmd = "winget install EclipseAdoptium.Temurin.21.JRE"
        else:
            cmd = "sudo apt-get install openjdk-21-jre"
        sections.append(
            "Install the package extras and Java runtime for upstream diCal2:\n"
            f"  {extra_hint}\n"
            f"  {cmd}\n"
            "  java -version"
        )
        return _join_lines(sections)

    if tool == "psmc":
        if os_name == "macos":
            cmd = "brew install make gcc"
        elif os_name == "windows":
            cmd = "wsl --install"
        else:
            cmd = "sudo apt-get install build-essential make"
        sections.append(
            "Install the package extras and build toolchain for upstream PSMC:\n"
            f"  {extra_hint}\n"
            f"  {cmd}\n"
            "  python -c \"import smckit; smckit.upstream.bootstrap('psmc')\""
        )
        return _join_lines(sections)

    if tool == "msmc2":
        if os_name == "macos":
            cmd = "brew install make llvm ldc"
        elif os_name == "windows":
            cmd = "wsl --install"
        else:
            cmd = "sudo apt-get install build-essential make ldc"
        sections.append(
            "Install the package extras and D toolchain for upstream MSMC2:\n"
            f"  {extra_hint}\n"
            f"  {cmd}\n"
            "  python -c \"import smckit; smckit.upstream.bootstrap('msmc2')\""
        )
        return _join_lines(sections)

    if tool == "smcpp":
        sections.append(
            f"Install the semantic extras first:\n  {extra_hint}\n"
            "Upstream SMC++ still expects a separate Python environment. "
            "Set SMCKIT_SMCPP_PYTHON to that interpreter, or use the native "
            "smckit implementation for the packaged quickstart."
        )
        return _join_lines(sections)

    if tool == "msmc_im":
        sections.append(
            f"Install the semantic extras first:\n  {extra_hint}\n"
            "The upstream MSMC-IM script runs directly from the vendored source tree. "
            "Use a source checkout if you need implementation=\"upstream\"."
        )
        return _join_lines(sections)

    return _join_lines(sections)
