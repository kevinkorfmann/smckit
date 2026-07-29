"""Command-line interface for native and preserved smckit workflows."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import smckit
from smckit._provenance import sha256_file


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _load_params(raw: str | None) -> dict[str, Any]:
    if raw is None:
        return {}
    candidate = Path(raw)
    text = candidate.read_text(encoding="utf-8") if candidate.is_file() else raw
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Run parameters must decode to a JSON object.")
    return payload


def _read_single_input(method: str, path: str):
    if method in {"psmc", "esmc2"}:
        return smckit.io.read_psmcfa(path)
    if method == "msmc2":
        return smckit.io.read_multihetsep(path)
    if method == "smcpp":
        return smckit.io.read_smcpp_input(path)
    raise ValueError(
        f"smckit run {method} needs a method-specific multi-input command that "
        "is not part of the 0.1 single-input CLI yet; use the Python API."
    )


def _run_native_command(namespace: argparse.Namespace) -> int:
    params = _load_params(namespace.params)
    params["implementation"] = namespace.implementation
    data = _read_single_input(namespace.method, namespace.input)
    method = getattr(smckit.tl, namespace.method)
    started = time.perf_counter()
    result_data = method(data, **params)
    result = result_data.results[namespace.method]
    result["provenance"]["runtime_seconds"] = time.perf_counter() - started
    result["provenance"]["input_sha256"] = {namespace.input: sha256_file(namespace.input)}
    if namespace.output:
        Path(namespace.output).write_text(
            json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
    else:
        _print_json(result)
    return 0


def _upstream_command(namespace: argparse.Namespace) -> int:
    raw_args = list(namespace.raw_args)
    option_tokens: list[str] = []
    if "--" in raw_args:
        delimiter = raw_args.index("--")
        option_tokens = raw_args[:delimiter]
        raw_args = raw_args[delimiter + 1 :]

    output_dir = namespace.output_dir
    timeout = namespace.timeout
    option_index = 0
    while option_index < len(option_tokens):
        option = option_tokens[option_index]
        if option == "--output-dir" and option_index + 1 < len(option_tokens):
            output_dir = option_tokens[option_index + 1]
            option_index += 2
            continue
        if option == "--timeout" and option_index + 1 < len(option_tokens):
            timeout = float(option_tokens[option_index + 1])
            option_index += 2
            continue
        raise ValueError(
            f"Unknown smckit upstream option {option!r}; put original-tool "
            "arguments after an explicit '--' delimiter."
        )

    if output_dir is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = f".smckit-runs/{namespace.tool}-{stamp}"
    result = smckit.upstream.run(
        namespace.tool,
        raw_args,
        output_dir=output_dir,
        timeout=timeout,
    )
    _print_json(result.to_dict())
    return result.returncode


def build_parser() -> argparse.ArgumentParser:
    """Build the public argument parser."""
    parser = argparse.ArgumentParser(prog="smckit")
    parser.add_argument("--version", action="version", version=smckit.__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    methods_parser = subparsers.add_parser("methods", help="List method capabilities.")
    methods_parser.add_argument("method", nargs="?")
    methods_parser.set_defaults(
        handler=lambda ns: _print_json(smckit.capabilities(ns.method)) or 0
    )

    status_parser = subparsers.add_parser("status", help="Inspect upstream readiness.")
    status_parser.add_argument("tool", nargs="?")
    status_parser.set_defaults(
        handler=lambda ns: _print_json(smckit.upstream.status(ns.tool)) or 0
    )

    run_parser = subparsers.add_parser("run", help="Run a typed single-input workflow.")
    run_parser.add_argument("method", choices=["psmc", "msmc2", "esmc2", "smcpp"])
    run_parser.add_argument("input")
    run_parser.add_argument(
        "--implementation",
        choices=["auto", "native", "upstream"],
        default="auto",
    )
    run_parser.add_argument(
        "--params",
        help="JSON object or path to a JSON file containing method parameters.",
    )
    run_parser.add_argument("--output", help="Write normalized result JSON to this path.")
    run_parser.set_defaults(handler=_run_native_command)

    upstream_parser = subparsers.add_parser(
        "upstream",
        help="Execute an original tool with unmodified raw arguments.",
    )
    upstream_parser.add_argument("tool", choices=sorted(smckit.upstream.status()))
    upstream_parser.add_argument("--output-dir")
    upstream_parser.add_argument("--timeout", type=float)
    upstream_parser.add_argument("raw_args", nargs=argparse.REMAINDER)
    upstream_parser.set_defaults(handler=_upstream_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the smckit command line."""
    parser = build_parser()
    try:
        namespace = parser.parse_args(argv)
        return int(namespace.handler(namespace))
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, f"smckit: error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
