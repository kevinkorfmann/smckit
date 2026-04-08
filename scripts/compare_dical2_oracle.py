"""Compare smckit's diCal2 implementation against the upstream diCal2 jar.

This is a parity aid for the bundled vendor examples, not a public API.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smckit.io import read_dical2  # noqa: E402
from smckit.tl import dical2  # noqa: E402

JAVA = Path("/opt/homebrew/opt/openjdk/bin/java")
DICAL2_JAR = ROOT / "vendor" / "diCal2" / "diCal2.jar"

EXAMPLES = {
    "exp": {
        "cwd": ROOT / "vendor" / "diCal2" / "examples" / "fromReadme",
        "oracle_args": [
            "--paramFile",
            "test.param",
            "--demoFile",
            "exp.demo",
            "--ratesFile",
            "exp.rates",
            "--vcfFile",
            "test.vcf",
            "--vcfFilterPassString",
            ".",
            "--vcfReferenceFile",
            "test.fa",
            "--configFile",
            "exp.config",
            "--metaStartFile",
            "exp.rand",
            "--seed",
            "541816302422",
            "--lociPerHmmStep",
            "3",
            "--compositeLikelihood",
            "lol",
            "--metaNumIterations",
            "2",
            "--metaKeepBest",
            "1",
            "--metaNumPoints",
            "3",
            "--numberIterationsEM",
            "2",
            "--numberIterationsMstep",
            "2",
            "--disableCoordinateWiseMStep",
            "--intervalType",
            "logUniform",
            "--intervalParams",
            "11,0.01,4",
            "--bounds",
            "1.00001,1000;0.01,0.06;0.01,0.23;0.02,2;0.5,4",
        ],
        "param_file": "test.param",
        "demo_file": "exp.demo",
        "rates_file": "exp.rates",
        "config_file": "exp.config",
        "meta_start_file": "exp.rand",
        "bounds": "1.00001,1000;0.01,0.06;0.01,0.23;0.02,2;0.5,4",
        "composite_mode": "lol",
        "loci_per_hmm_step": 3,
        "n_intervals": 11,
        "max_t": 4.0,
        "n_em_iterations": 2,
        "meta_num_iterations": 2,
        "meta_keep_best": 1,
        "meta_num_points": 3,
        "seed": 541816302422,
        "native_options": {
            "interval_type": "logUniform",
            "interval_params": "11,0.01,4",
            "number_iterations_mstep": 2,
            "disableCoordinateWiseMStep": True,
        },
    },
    "IM": {
        "cwd": ROOT / "vendor" / "diCal2" / "examples" / "fromReadme",
        "oracle_args": [
            "--paramFile",
            "test.param",
            "--demoFile",
            "IM.demo",
            "--vcfFile",
            "test.vcf",
            "--vcfFilterPassString",
            ".",
            "--vcfReferenceFile",
            "test.fa",
            "--configFile",
            "IM.config",
            "--metaStartFile",
            "IM.rand",
            "--seed",
            "60643714832",
            "--lociPerHmmStep",
            "4",
            "--compositeLikelihood",
            "pcl",
            "--metaNumIterations",
            "2",
            "--metaKeepBest",
            "1",
            "--metaNumPoints",
            "3",
            "--numberIterationsEM",
            "2",
            "--numberIterationsMstep",
            "2",
            "--intervalType",
            "logUniform",
            "--intervalParams",
            "11,0.01,4",
            "--bounds",
            "0.01,0.32;0.05,1.0001;0.05,5;0.05,5;0.02,2;0.9,5;0.1,500",
        ],
        "param_file": "test.param",
        "demo_file": "IM.demo",
        "config_file": "IM.config",
        "meta_start_file": "IM.rand",
        "bounds": "0.01,0.32;0.05,1.0001;0.05,5;0.05,5;0.02,2;0.9,5;0.1,500",
        "composite_mode": "pcl",
        "loci_per_hmm_step": 4,
        "n_intervals": 11,
        "max_t": 4.0,
        "n_em_iterations": 2,
        "meta_num_iterations": 2,
        "meta_keep_best": 1,
        "meta_num_points": 3,
        "seed": 60643714832,
        "native_options": {
            "interval_type": "logUniform",
            "interval_params": "11,0.01,4",
            "number_iterations_mstep": 2,
        },
    },
    "three": {
        "cwd": ROOT / "vendor" / "diCal2" / "examples" / "fromReadme",
        "oracle_args": [
            "--paramFile",
            "test.param",
            "--demoFile",
            "three.demo",
            "--vcfFile",
            "test.vcf",
            "--vcfFilterPassString",
            ".",
            "--vcfReferenceFile",
            "test.fa",
            "--configFile",
            "three.config",
            "--seed",
            "241438375231",
            "--lociPerHmmStep",
            "3",
            "--compositeLikelihood",
            "lol",
            "--numberIterationsEM",
            "5",
            "--numberIterationsMstep",
            "4",
            "--startPoint",
            "0.1,0.2,0.1,0.1,0.1,0.1,0.2",
            "--intervalType",
            "logUniform",
            "--intervalParams",
            "11,0.01,4",
            "--bounds",
            "0.005,0.2;0.05,1.0001;0.05,5;0.05,5;0.05,5;0.03,3;0.1,5",
        ],
        "param_file": "test.param",
        "demo_file": "three.demo",
        "config_file": "three.config",
        "start_point": [0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2],
        "bounds": "0.005,0.2;0.05,1.0001;0.05,5;0.05,5;0.05,5;0.03,3;0.1,5",
        "composite_mode": "lol",
        "loci_per_hmm_step": 3,
        "n_intervals": 11,
        "max_t": 4.0,
        "n_em_iterations": 5,
        "seed": 241438375231,
        "native_options": {
            "interval_type": "logUniform",
            "interval_params": "11,0.01,4",
            "number_iterations_mstep": 4,
            "coordinatewise_mstep": True,
            "nm_fraction": 0.2,
        },
    },
}
def _run_oracle(example: dict) -> tuple[float | None, list[float] | None]:
    cmd = [str(JAVA), "-jar", str(DICAL2_JAR), *example["oracle_args"]]
    proc = subprocess.run(
        cmd,
        cwd=example["cwd"],
        capture_output=True,
        text=True,
        check=True,
    )
    log_likelihood = None
    params = None
    for line in proc.stdout.splitlines():
        if not line or line.startswith("# ["):
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            try:
                log_likelihood = float(parts[0])
                params = [float(x) for x in parts[2:-1]]
            except ValueError:
                continue
    return log_likelihood, params


def _run_smckit(example: dict, implementation: str) -> dict:
    cwd = example["cwd"]
    data = read_dical2(
        sequences=cwd / "test.vcf",
        param_file=cwd / example["param_file"],
        demo_file=cwd / example["demo_file"],
        rates_file=cwd / example.get("rates_file") if example.get("rates_file") else None,
        config_file=cwd / example["config_file"],
        reference_file=cwd / "test.fa",
        filter_pass_string=".",
    )
    out = dical2(
        data,
        implementation=implementation,
        n_intervals=example["n_intervals"],
        max_t=example["max_t"],
        n_em_iterations=example["n_em_iterations"],
        composite_mode=example["composite_mode"],
        loci_per_hmm_step=example.get("loci_per_hmm_step", 1),
        meta_start_file=str(cwd / example["meta_start_file"]) if example.get("meta_start_file") else None,
        meta_num_iterations=example.get("meta_num_iterations", 1),
        meta_keep_best=example.get("meta_keep_best", 1),
        meta_num_points=example.get("meta_num_points"),
        start_point=example.get("start_point"),
        bounds=example.get("bounds"),
        seed=example.get("seed"),
        native_options=example.get("native_options"),
        upstream_options=example.get("native_options"),
    )
    return out.results["dical2"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("example", choices=sorted(EXAMPLES), nargs="+")
    parser.add_argument(
        "--implementation",
        choices=("native", "upstream"),
        default="native",
        help="smckit diCal2 backend to compare against the oracle",
    )
    args = parser.parse_args()

    for example_name in args.example:
        example = EXAMPLES[example_name]
        oracle_ll, oracle_params = _run_oracle(example)
        smc = _run_smckit(example, implementation=args.implementation)

        print(f"example: {example_name}")
        print(f"implementation: {args.implementation}")
        print(f"oracle_log_likelihood: {oracle_ll}")
        print(f"oracle_params: {oracle_params}")
        print(f"smckit_log_likelihood: {smc['log_likelihood']}")
        print(f"smckit_best_params: {smc.get('best_params')}")
        print(f"smckit_ordered_params: {smc.get('ordered_params')}")
        if "pop_sizes" in smc:
            print(f"smckit_pop_sizes: {smc['pop_sizes'].tolist()}")
        if "growth_rates" in smc:
            print(f"smckit_growth_rates: {smc['growth_rates'].tolist()}")
        if "structured_ne" in smc:
            print(f"smckit_structured_ne: {smc['structured_ne']}")
        if "interval_boundaries" in smc:
            print(f"smckit_interval_boundaries: {smc['interval_boundaries'].tolist()}")
        print(f"smckit_rounds: {smc['rounds']}")
        print()


if __name__ == "__main__":
    main()
