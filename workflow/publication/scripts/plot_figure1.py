"""Generate the primary four-panel evidence figure from frozen JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

# Publication rendering must never depend on a desktop display or an
# environment-selected interactive backend. This must precede pyplot import.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#6B6B6B"


def _canonical_hash(payload: dict[str, Any]) -> str:
    unhashed = {key: value for key, value in payload.items() if key != "aggregate_sha256"}
    canonical = json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _load_aggregate(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Figure input must use aggregate schema version 1.")
    if payload.get("aggregate_sha256") != _canonical_hash(payload):
        raise ValueError("Figure input failed its aggregate_sha256 integrity check.")
    performance = payload.get("performance_comparisons", [])
    accuracy = payload.get("accuracy_records", [])
    parity = [item for item in accuracy if item.get("evaluation_kind") == "parity"]
    validation = [
        item for item in accuracy if item.get("evaluation_kind") in {"simulation", "empirical"}
    ]
    if not performance:
        raise ValueError("Figure 1 requires native/upstream performance comparisons.")
    if not parity:
        raise ValueError("Figure 1 requires parity accuracy records.")
    if not validation:
        raise ValueError("Figure 1 requires simulation or empirical validation records.")
    return payload


def _panel_label(axis: matplotlib.axes.Axes, label: str) -> None:
    axis.text(
        -0.13,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )


def _architecture_panel(axis: matplotlib.axes.Axes) -> None:
    axis.set_axis_off()
    boxes = {
        "typed": (0.04, 0.67, 0.25, 0.18, "Typed API / CLI", BLUE),
        "dispatch": (0.38, 0.67, 0.25, 0.18, "Capability registry", PURPLE),
        "native": (0.04, 0.20, 0.25, 0.18, "Native smckit", GREEN),
        "upstream": (0.38, 0.20, 0.25, 0.18, "Pinned upstream", ORANGE),
        "external": (0.71, 0.20, 0.25, 0.18, "Maintained external", GRAY),
    }
    for x, y, width, height, label, color in boxes.values():
        axis.add_patch(
            matplotlib.patches.FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.015",
                facecolor="white",
                edgecolor=color,
                linewidth=1.6,
            )
        )
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=7)
    arrow = dict(arrowstyle="-|>", mutation_scale=9, color="#333333", linewidth=1)
    axis.annotate("", xy=(0.38, 0.76), xytext=(0.29, 0.76), arrowprops=arrow)
    for destination in (0.165, 0.505, 0.835):
        axis.annotate("", xy=(destination, 0.39), xytext=(0.505, 0.67), arrowprops=arrow)
    axis.text(0.50, 0.49, "auto selects only promoted capabilities", ha="center", fontsize=6.5)
    axis.text(0.165, 0.12, "optimized", ha="center", color=GREEN, fontsize=6.5)
    axis.text(0.505, 0.12, "exact oracle", ha="center", color=ORANGE, fontsize=6.5)
    axis.text(0.835, 0.12, "PHLASH", ha="center", color=GRAY, fontsize=6.5)


def _parity_panel(axis: matplotlib.axes.Axes, records: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[float]] = {}
    for record in records:
        metric = record["metrics"].get("parity_error")
        if metric is None:
            raise ValueError("Every parity record must contain metrics.parity_error.")
        grouped.setdefault(record["method"], []).append(float(metric))
    methods = sorted(grouped)
    positions = np.arange(len(methods))
    values = [max(float(np.median(grouped[method])), 1e-12) for method in methods]
    axis.scatter(positions, values, color=BLUE, marker="o", s=28, zorder=3)
    for index, method in enumerate(methods):
        observations = np.asarray(grouped[method], dtype=float)
        if observations.size > 1:
            axis.vlines(
                index,
                max(float(np.min(observations)), 1e-12),
                max(float(np.max(observations)), 1e-12),
                color=BLUE,
                linewidth=1,
                alpha=0.7,
            )
    axis.axhline(1e-3, color=VERMILLION, linestyle="--", linewidth=1, label="0.1% threshold")
    axis.set_yscale("log")
    axis.set_xticks(positions, methods, rotation=35, ha="right")
    axis.set_ylabel("Native–upstream parity error")
    axis.legend(frameon=False, fontsize=6.5, loc="best")


def _performance_panel(axis: matplotlib.axes.Axes, records: list[dict[str, Any]]) -> None:
    labels = [f"{item['method']} · {item['dataset']}" for item in records]
    positions = np.arange(len(records))
    values = np.asarray([item["speedup"] for item in records], dtype=float)
    intervals = np.asarray([item["speedup_confidence_interval"] for item in records], dtype=float)
    colors = [GREEN if item["promotable"] else ORANGE for item in records]
    axis.errorbar(
        values,
        positions,
        xerr=np.vstack((values - intervals[:, 0], intervals[:, 1] - values)),
        fmt="none",
        ecolor=GRAY,
        elinewidth=1,
        capsize=2,
        zorder=1,
    )
    axis.scatter(values, positions, c=colors, marker="s", s=28, zorder=2)
    axis.axvline(1.0, color=VERMILLION, linestyle="--", linewidth=1)
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Warmed speedup (upstream/native)")
    axis.invert_yaxis()
    for position, item in zip(positions, records, strict=True):
        axis.annotate(
            f"M={item['memory_ratio']:.2f}",
            (item["speedup_confidence_interval"][1], position),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=5.8,
            color=GRAY,
        )


def _validation_panel(axis: matplotlib.axes.Axes, records: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[float]] = {}
    for record in records:
        metric = record["metrics"].get("trajectory_error")
        if metric is None:
            raise ValueError("Validation records must contain metrics.trajectory_error.")
        grouped.setdefault((record["method"], record["evaluation_kind"]), []).append(float(metric))
    groups = sorted(grouped)
    positions = np.arange(len(groups))
    for index, group in enumerate(groups):
        values = np.asarray(grouped[group], dtype=float)
        kind = group[1]
        color = BLUE if kind == "simulation" else PURPLE
        marker = "o" if kind == "simulation" else "^"
        jitter = np.linspace(-0.10, 0.10, values.size) if values.size > 1 else np.asarray([0.0])
        axis.scatter(
            np.full(values.size, index) + jitter,
            values,
            color=color,
            marker=marker,
            s=16,
            alpha=0.65,
        )
        axis.hlines(
            float(np.median(values)),
            index - 0.18,
            index + 0.18,
            color="#111111",
            linewidth=1.5,
        )
    axis.set_xticks(
        positions,
        [f"{method}\n{kind}" for method, kind in groups],
        rotation=35,
        ha="right",
    )
    axis.set_ylabel("Log-integrated trajectory error")


def plot_figure1(
    aggregate_path: Path,
    output_prefix: Path,
    *,
    raster_dpi: int = 350,
) -> list[Path]:
    """Render vector working files and a journal-resolution TIFF."""
    if raster_dpi < 350:
        raise ValueError("Color submission figures require at least 350 dpi.")
    payload = _load_aggregate(aggregate_path)
    accuracy = payload["accuracy_records"]
    parity = [item for item in accuracy if item["evaluation_kind"] == "parity"]
    validation = [
        item for item in accuracy if item["evaluation_kind"] in {"simulation", "empirical"}
    ]

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
        _architecture_panel(axes[0, 0])
        _parity_panel(axes[0, 1], parity)
        _performance_panel(axes[1, 0], payload["performance_comparisons"])
        _validation_panel(axes[1, 1], validation)
        for axis, label in zip(axes.flat, "abcd", strict=True):
            _panel_label(axis, label)
        figure.suptitle(
            "Preserved execution, validated parity, and reproducible performance",
            fontsize=9,
            fontweight="bold",
        )

        output_prefix = output_prefix.expanduser().resolve()
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        outputs = [
            output_prefix.with_suffix(".pdf"),
            output_prefix.with_suffix(".svg"),
            output_prefix.with_suffix(".tiff"),
        ]
        for output in outputs:
            options: dict[str, Any] = {
                "bbox_inches": "tight",
                "facecolor": "white",
            }
            if output.suffix == ".tiff":
                options.update(
                    {
                        "dpi": raster_dpi,
                        "pil_kwargs": {"compression": "tiff_lzw"},
                    }
                )
            figure.savefig(output, **options)
        plt.close(figure)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--raster-dpi", type=int, default=350)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plot_figure1(args.aggregate, args.output_prefix, raster_dpi=args.raster_dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
