"""Render the preservation, validation, and promotion architecture schematic."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon  # noqa: E402

BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#666666"
LIGHT = "#F7F7F7"
DARK = "#202124"


def _box(
    axis: matplotlib.axes.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    detail: str,
    color: str,
    fill: str = "white",
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor=fill,
            edgecolor=color,
            linewidth=1.8,
        )
    )
    axis.text(
        x + width / 2,
        y + height * 0.64,
        title,
        ha="center",
        va="center",
        color=DARK,
        fontsize=8.2,
        fontweight="bold",
    )
    axis.text(
        x + width / 2,
        y + height * 0.30,
        detail,
        ha="center",
        va="center",
        color=GRAY,
        fontsize=6.5,
        linespacing=1.15,
    )


def _arrow(
    axis: matplotlib.axes.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = GRAY,
    dashed: bool = False,
) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": 1.35,
            "linestyle": "--" if dashed else "-",
            "mutation_scale": 10,
            "shrinkA": 1,
            "shrinkB": 1,
        },
    )


def _gate(
    axis: matplotlib.axes.Axes,
    *,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
) -> None:
    points = [
        (center_x, center_y + height / 2),
        (center_x + width / 2, center_y),
        (center_x, center_y - height / 2),
        (center_x - width / 2, center_y),
    ]
    axis.add_patch(
        Polygon(points, closed=True, facecolor="#FFF7E6", edgecolor=VERMILLION, linewidth=1.8)
    )
    axis.text(
        center_x,
        center_y + 0.012,
        "All gates\npass?",
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color=DARK,
        linespacing=1.0,
    )


def plot_architecture(output_prefix: Path, *, raster_dpi: int = 600) -> list[Path]:
    """Render the conceptual workflow as vector files and a line-art TIFF."""
    if raster_dpi < 300:
        raise ValueError("Architecture raster export requires at least 300 dpi.")

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        figure, axis = plt.subplots(figsize=(10.0, 5.8))
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_axis_off()
        figure.patch.set_facecolor("white")

        axis.text(
            0.5,
            0.965,
            "Preserve the original. Validate the native implementation. "
            "Promote only the evidence.",
            ha="center",
            va="top",
            fontsize=11.5,
            fontweight="bold",
            color=DARK,
        )
        axis.text(
            0.02,
            0.865,
            "UPSTREAM PRESERVATION LANE",
            color=ORANGE,
            fontsize=7.4,
            fontweight="bold",
        )
        axis.text(
            0.02,
            0.535,
            "INDEPENDENT NATIVE LANE",
            color=GREEN,
            fontsize=7.4,
            fontweight="bold",
        )

        upper_y = 0.665
        lower_y = 0.335
        width = 0.135
        height = 0.145
        positions = [0.02, 0.19, 0.36, 0.53]
        upper = [
            ("Original surface", "manual · help\nexamples · artifacts"),
            ("Immutable identity", "commit · archive hash\nlicense · citation"),
            ("Isolated execution", "pinned runtime\nstdout · stderr · exit"),
            ("Oracle evidence", "intermediates · outputs\nfailure behavior"),
        ]
        lower = [
            ("Native\nimplementation", "independent code\nstructured options"),
            ("Kernel agreement", "transitions · emissions\nlikelihood terms"),
            ("Workflow parity", "inputs · fits · artifacts\nedge cases · platforms"),
            ("Performance gate", "same-process warm CI\npeak RSS ≤ 1.25×"),
        ]
        for x, (title, detail) in zip(positions, upper, strict=True):
            _box(
                axis,
                x=x,
                y=upper_y,
                width=width,
                height=height,
                title=title,
                detail=detail,
                color=ORANGE,
            )
        for x, (title, detail) in zip(positions, lower, strict=True):
            _box(
                axis,
                x=x,
                y=lower_y,
                width=width,
                height=height,
                title=title,
                detail=detail,
                color=GREEN,
            )
        for first, second in zip(positions[:-1], positions[1:], strict=True):
            _arrow(
                axis,
                (first + width, upper_y + height / 2),
                (second, upper_y + height / 2),
                color=ORANGE,
            )
            _arrow(
                axis,
                (first + width, lower_y + height / 2),
                (second, lower_y + height / 2),
                color=GREEN,
            )

        _gate(axis, center_x=0.755, center_y=0.50, width=0.105, height=0.19)
        _arrow(axis, (positions[-1] + width, upper_y + height / 2), (0.72, 0.565), color=ORANGE)
        _arrow(axis, (positions[-1] + width, lower_y + height / 2), (0.72, 0.435), color=GREEN)

        _box(
            axis,
            x=0.84,
            y=0.64,
            width=0.135,
            height=0.145,
            title="auto → native",
            detail="capability promoted\nprovenance retained",
            color=GREEN,
            fill="#F0FAF6",
        )
        _box(
            axis,
            x=0.84,
            y=0.305,
            width=0.135,
            height=0.145,
            title="auto → upstream",
            detail="reason recorded\nno silent substitution",
            color=ORANGE,
            fill="#FFF8EC",
        )
        _arrow(axis, (0.807, 0.55), (0.84, 0.70), color=GREEN)
        _arrow(axis, (0.807, 0.45), (0.84, 0.38), color=ORANGE)
        axis.text(0.814, 0.625, "YES", color=GREEN, fontsize=6.5, fontweight="bold")
        axis.text(0.814, 0.395, "NO", color=ORANGE, fontsize=6.5, fontweight="bold")

        _box(
            axis,
            x=0.77,
            y=0.835,
            width=0.205,
            height=0.065,
            title="explicit upstream — permanent",
            detail="",
            color=ORANGE,
            fill="#FFF8EC",
        )
        _arrow(
            axis,
            (positions[-1] + width, upper_y + height),
            (0.77, 0.86),
            color=ORANGE,
            dashed=True,
        )

        axis.add_patch(
            FancyBboxPatch(
                (0.02, 0.06),
                0.955,
                0.16,
                boxstyle="round,pad=0.012,rounding_size=0.012",
                facecolor=LIGHT,
                edgecolor="#B8B8B8",
                linewidth=1.0,
            )
        )
        axis.text(
            0.045,
            0.17,
            "MODERN EXTENSIONS",
            color=PURPLE,
            fontsize=7.2,
            fontweight="bold",
            va="center",
        )
        axis.text(
            0.245,
            0.17,
            "PHLASH",
            color=PURPLE,
            fontsize=8,
            fontweight="bold",
            va="center",
        )
        axis.text(
            0.245,
            0.112,
            "Maintained external Bayesian integration\nposterior uncertainty · optional GPU",
            color=GRAY,
            fontsize=6.6,
            va="center",
            ha="left",
            linespacing=1.2,
        )
        axis.plot([0.59, 0.59], [0.085, 0.195], color="#C8C8C8", linewidth=1)
        axis.text(
            0.62,
            0.17,
            "PSMC+",
            color=BLUE,
            fontsize=8,
            fontweight="bold",
            va="center",
        )
        axis.text(
            0.62,
            0.112,
            "Next preservation + native target\n"
            "mutation/recombination maps · genomic heterogeneity",
            color=GRAY,
            fontsize=6.6,
            va="center",
            ha="left",
            linespacing=1.2,
        )

        output_prefix = Path(output_prefix).expanduser().resolve()
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        outputs = [
            output_prefix.with_suffix(".pdf"),
            output_prefix.with_suffix(".svg"),
            output_prefix.with_suffix(".tiff"),
        ]
        for output in outputs:
            options: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
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
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--raster-dpi", type=int, default=600)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plot_architecture(args.output_prefix, raster_dpi=args.raster_dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
