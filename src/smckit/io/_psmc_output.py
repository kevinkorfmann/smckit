"""Read/write PSMC output format (.psmc files)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def read_psmc_output(path: str | Path) -> list[dict[str, Any]]:
    """Read a PSMC output file and extract results per iteration round.

    Parameters
    ----------
    path : str or Path
        Path to ``.psmc`` output file.

    Returns
    -------
    list of dict
        One dict per EM round, each containing:
        - ``"round"``: iteration number
        - ``"log_likelihood"``: log P(data)
        - ``"theta"``: θ₀
        - ``"rho"``: ρ₀
        - ``"time"``: array of t_k values
        - ``"lambda"``: array of λ_k values
        - ``"pi"``: array of π_k values
        - ``"sigma"``: array of σ_k values
        - ``"post_sigma"``: array of posterior σ_k values
    """
    path = Path(path)
    rounds: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    rs_rows: list[list[float]] = []
    metadata: list[str] = []

    def _flush_round():
        nonlocal current, rs_rows
        if current is not None and rs_rows:
            arr = np.array(rs_rows)
            current["time"] = arr[:, 1]
            current["lambda"] = arr[:, 2]
            current["pi"] = arr[:, 3]
            current["sigma"] = arr[:, 4]
            current["post_sigma"] = arr[:, 5]
            rounds.append(current)
        current = None
        rs_rows = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("CC"):
                continue
            parts = line.split("\t")
            tag = parts[0]

            if tag == "RD":
                _flush_round()
                current = {"round": int(parts[1]), "metadata": metadata.copy()}
                rs_rows = []
            elif tag == "MM":
                message = "\t".join(parts[1:])
                metadata.append(message)
                if current is not None:
                    current.setdefault("messages", []).append(message)
            elif tag == "LK" and current is not None:
                current["log_likelihood"] = float(parts[1])
            elif tag == "QD" and current is not None:
                values = " ".join(parts[1:]).replace("->", " ").split()
                current["Q0"], current["Q1"] = (float(values[0]), float(values[1]))
            elif tag == "RI" and current is not None:
                current["relative_information"] = float(parts[1])
            elif tag == "TR" and current is not None:
                current["theta"] = float(parts[1])
                current["rho"] = float(parts[2])
            elif tag == "MT" and current is not None:
                current["max_t"] = float(parts[1])
            elif tag == "DT" and current is not None:
                current["divergence_time"] = float(parts[1])
            elif tag == "RS" and current is not None:
                rs_rows.append([float(x) for x in parts[1:]])
            elif tag == "PA" and current is not None:
                pa_parts = parts[1].split()
                current["pattern"] = pa_parts[0]
                current["params"] = np.array([float(x) for x in pa_parts[1:]])
            elif tag == "TC" and current is not None:
                current.setdefault("time_centres", []).append(
                    {
                        "state": int(parts[1]),
                        "left": float(parts[2]),
                        "centre": float(parts[3]),
                        "right": float(parts[4]),
                    }
                )
            elif tag == "DC" and current is not None:
                current.setdefault("decoded_segments", []).append(
                    {
                        "name": parts[1],
                        "start": int(parts[2]),
                        "end": int(parts[3]),
                        "state": int(parts[4]),
                        "scaled_time": float(parts[5]),
                        "max_probability": float(parts[6]),
                    }
                )
            elif tag == "DF" and current is not None:
                current.setdefault("decoded_full", []).append(
                    {
                        "position": int(parts[1]),
                        "recombination_probability": float(parts[2]),
                        "posterior": np.array(
                            [float(value) for value in parts[3:]],
                            dtype=np.float64,
                        ),
                    }
                )
            elif tag == "PR" and current is not None:
                current.setdefault("sequence_probabilities", []).append(
                    {
                        "name": parts[1],
                        "length": int(parts[2]),
                        "scale": np.array(
                            [float(value) for value in parts[3:]],
                            dtype=np.float64,
                        ),
                    }
                )
            elif tag == "FA" and current is not None:
                content = "\t".join(parts[1:])
                if content.startswith(">"):
                    current.setdefault("simulated_records", []).append(
                        {"name": content[1:], "sequence": ""}
                    )
                elif current.get("simulated_records"):
                    current["simulated_records"][-1]["sequence"] += content

    _flush_round()
    return rounds


def write_psmc_output(
    path: str | Path,
    rounds: list[dict[str, Any]],
    pattern: str = "",
    metadata: dict[str, str] | None = None,
    decoded: dict[str, Any] | None = None,
    sequence_probabilities: list[dict[str, Any]] | None = None,
    simulated_records: list[dict[str, Any]] | None = None,
) -> None:
    """Write PSMC results in the standard .psmc output format.

    Parameters
    ----------
    path : str or Path
        Output file path.
    rounds : list of dict
        Results per round (same format as ``read_psmc_output`` returns).
    pattern : str
        Parameter pattern string.
    metadata : dict, optional
        Additional metadata lines (key-value pairs written as MM lines).
    """
    path = Path(path)
    with open(path, "w") as f:
        f.write("CC\n")
        f.write("CC\tGenerated by smckit\n")
        f.write("CC\n")
        if metadata:
            for k, v in metadata.items():
                f.write(f"MM\t{k}: {v}\n")

        for rd in rounds:
            f.write(f"RD\t{rd['round']}\n")
            if "log_likelihood" in rd:
                f.write(f"LK\t{rd['log_likelihood']:.6f}\n")
            if "Q0" in rd and "Q1" in rd:
                f.write(f"QD\t{rd['Q0']:.6f} -> {rd['Q1']:.6f}\n")
            if "relative_information" in rd:
                f.write(f"RI\t{rd['relative_information']:.10f}\n")
            if "theta" in rd and "rho" in rd:
                f.write(f"TR\t{rd['theta']:.6f}\t{rd['rho']:.6f}\n")
            if "max_t" in rd:
                f.write(f"MT\t{rd['max_t']:.6f}\n")
            if rd.get("divergence_time") is not None:
                f.write(f"DT\t{rd['divergence_time']:.6f}\n")

            if "time" in rd and "lambda" in rd:
                n_states = len(rd["time"])
                pi = rd.get("pi", np.zeros(n_states))
                sigma = rd.get("sigma", np.zeros(n_states))
                post_sigma = rd.get("post_sigma", np.zeros(n_states))
                for k in range(n_states):
                    f.write(
                        f"RS\t{k}\t{rd['time'][k]:.6f}\t{rd['lambda'][k]:.6f}"
                        f"\t{pi[k]:.6f}\t{sigma[k]:.6f}\t{post_sigma[k]:.6f}\n"
                    )

            if "params" in rd:
                params_str = " ".join(f"{p:.9f}" for p in rd["params"])
                f.write(f"PA\t{pattern} {params_str}\n")
            f.write("//\n")

        if decoded is not None:
            if decoded["mode"] == "posterior":
                for record in decoded["records"]:
                    for segment in record["segments"]:
                        f.write(
                            "DC"
                            f"\t{record['name']}"
                            f"\t{segment['start']}"
                            f"\t{segment['end']}"
                            f"\t{segment['state']}"
                            f"\t{segment['scaled_time']:.6f}"
                            f"\t{segment['max_probability']:.6f}\n"
                        )
            elif decoded["mode"] == "full":
                for record in decoded["records"]:
                    for position, recombination, posterior in zip(
                        record["position"],
                        record["recombination_probability"],
                        record["posterior"],
                        strict=True,
                    ):
                        probabilities = "".join(f"\t{value:.6f}" for value in posterior)
                        f.write(
                            f"DF\t{int(position)}\t{float(recombination):.6f}{probabilities}\n"
                        )
        if sequence_probabilities is not None:
            for record in sequence_probabilities:
                scales = "".join(f"\t{float(value):.6f}" for value in record["scale"])
                f.write(f"PR\t{record['name']}\t{record['length']}{scales}\n")
        if simulated_records is not None:
            decode_codes = np.array(["T", "K", "N"], dtype=object)
            for record in simulated_records:
                sequence = "".join(
                    decode_codes[int(value)] for value in np.asarray(record["codes"])
                )
                f.write(f"FA\t>{record['name']}\n")
                for start in range(0, len(sequence), 60):
                    f.write(f"FA\t{sequence[start : start + 60]}\n")
