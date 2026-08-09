# Runtime and resource planning

Runtime is part of smckit's validation contract, but a number without its
dataset, hardware, thread count, warm-up state, and measured component is not a
portable estimate. This page therefore separates three evidence classes:

- **Frozen smckit evidence**: raw JSON, checksums, hardware metadata, repeated
  timings, peak memory, and (for promotion) a paired-bootstrap confidence
  interval.
- **Published upstream guidance**: useful for planning, but measured by another
  project on different software and hardware.
- **Benchmark pending**: the method works, but smckit does not yet have evidence
  broad enough to quote a production runtime.

Tiny fixture timings are shown to make implementation overhead and relative
speed concrete. They must not be multiplied by genome length to predict a
whole-genome run: convergence, sample count, pair count, state count, masks,
I/O, and parallelism can change the scaling regime.

## At-a-glance status for every method

| Method | Implementations | Best current timing evidence | Practical interpretation |
|---|---|---|---|
| **PSMC** | native + upstream | Historical NA12878 chr22 observation: 10 EM iterations took 5.4 s native and 6.3 s upstream. No retained repeated raw record or confidence interval. | Useful as a smoke-test expectation only; a new frozen whole-genome benchmark is pending. |
| **PSMC+** | native + upstream | **Frozen on Linux x86-64 and macOS ARM64.** Native fit is 2.16–2.32× faster and decode is 1.10–1.11× faster; native peak memory is 0.363–0.387× upstream. | The speed claim is valid for the warmed frozen core fixture. Whole-genome background-selection analyses still need retained empirical timing. |
| **MSMC2** | native + upstream | No frozen smckit comparison yet. Published upstream examples cover whole human autosomes; see below. | Runtime grows with sequence length, selected haplotype pairs, and approximately the square of the time-state count. |
| **MSMC-IM** | native + upstream | Benchmark pending. | Its input is a compact combined MSMC2 trajectory, but optimizer starts and tolerances can dominate. Measure the exact fit rather than budgeting from file size. |
| **eSMC2** | native + upstream R | Benchmark pending. | Budget separately for data preparation and repeated optimization. Dormancy/selfing models and multi-pair composites can materially change convergence time. |
| **SMC++** | native + upstream | **Frozen one-thread Linux split control:** 0.247 s native versus about 2.74 s upstream when warm, an 11.01× speedup (95% CI 9.47–11.11×); native peak memory is 0.512× upstream. | This promotes the tested split-inference capability, not an arbitrary whole-genome, sample-size, or cross-validation workload. |
| **ASMC** | native + upstream | Benchmark pending in smckit. | Cost is close to linear in decoded positions for one pair, then scales with the requested pair count. All pairs among *h* haplotypes means *h(h−1)/2* decodes and potentially very large output. |
| **diCal2** | native + upstream Java | **Frozen macOS ARM64 one-step fits:** native is 1.138× faster for pulse introgression and 1.305× faster for exponential growth; native peak memory is 0.491× and 0.596× upstream, respectively. | These are capability-specific results. Model complexity, CSD construction, particles, meta-starts, EM iterations, and M-step iterations can move a real analysis from seconds to long batch jobs. |
| **PHLASH** | maintained external package through smckit | No retained smckit timing yet. The 2025 PHLASH paper reports about 20–30 CPU min Gbp⁻¹ for one diploid sample across the compared methods and favorable scaling for PHLASH at larger sample sizes. | Prefer a GPU. Particle count, iterations, early stopping, sample count, and held-out data determine the final runtime. The frozen 80-fit smckit validation run is still pending. |
| **SSM** | experimental native framework | No production benchmark is defined. | Treat each model as a research workload. State count, transition structure, optimizer, differentiation backend, and JIT compilation determine cost. |

## Frozen smckit comparisons

### PSMC+

The promotion benchmark uses one thread, five counterbalanced warmed pairs, and
20,000 paired bootstrap samples. Input preparation is excluded from the warmed
core timing; cold typed execution and process-tree memory are recorded
separately.

| Platform | Capability | Native median | Upstream median | Speedup (95% CI) | Native/upstream memory |
|---|---:|---:|---:|---:|---:|
| Linux x86-64 | fit | 23.20 ms | 50.05 ms | 2.161× (2.068–2.188×) | 0.387× |
| Linux x86-64 | decode | 8.37 ms | 9.31 ms | 1.096× (1.059–1.137×) | 0.385× |
| macOS ARM64 | fit | 8.03 ms | 18.65 ms | 2.321× (2.288–2.346×) | 0.363× |
| macOS ARM64 | decode | 4.01 ms | 4.47 ms | 1.112× (1.108–1.123×) | 0.366× |

Authoritative evidence:

- [`workflow/publication/evidence/psmcplus-performance-paired/`](https://github.com/kevinkorfmann/smckit/tree/main/workflow/publication/evidence/psmcplus-performance-paired)
- [`docs/parity/oracle-specifications.json`](../parity/oracle-specifications.json)

### SMC++ split inference

The frozen `split-control-v1` record measures the in-process inference call,
excluding input preparation, in a persistent process on one Linux x86-64 CPU
thread. Five warm repetitions follow the cold call.

| Native warm median | Upstream warm median | Speedup (95% CI) | Native/upstream memory | Native cold | Upstream cold |
|---:|---:|---:|---:|---:|---:|
| about 0.247 s | about 2.74 s | 11.01× (9.47–11.11×) | 0.512× | 0.250 s | 2.360 s |

Authoritative evidence:

- [`workflow/publication/evidence/smcpp-split/`](https://github.com/kevinkorfmann/smckit/tree/main/workflow/publication/evidence/smcpp-split)

### diCal2 structured fits

Both records use one macOS ARM64 thread and ten warmed pairs. Input preparation
is measured separately. The absolute times below are the cold inference calls;
the promotion comparison is based on paired warmed calls.

| Capability | Native cold | Upstream cold | Warm speedup (95% CI) | Native/upstream memory |
|---|---:|---:|---:|---:|
| pulse introgression, one EM/M-step | 1.295 s | 1.509 s | 1.138× (1.134–1.143×) | 0.491× |
| exponential growth, one EM/M-step | 0.615 s | 0.792 s | 1.305× (1.283–1.354×) | 0.596× |

Authoritative evidence:

- [`workflow/publication/evidence/dical2-introgression/`](https://github.com/kevinkorfmann/smckit/tree/main/workflow/publication/evidence/dical2-introgression)
- [`workflow/publication/evidence/dical2-growth/`](https://github.com/kevinkorfmann/smckit/tree/main/workflow/publication/evidence/dical2-growth)

## Published whole-genome anchors

These values describe the original/external methods, not smckit performance.
They are included because they are more useful for scheduler planning than a
millisecond-scale test fixture.

The MSMC/MSMC2 protocol chapter reports the following MSMC2 examples for 22
human autosomes, 11 CPUs, and default time patterning:

| Samples | Analysis | Published wall time | Published memory |
|---|---|---:|---:|
| one diploid | within-population | 18 min | 7 GB |
| two diploids | same population | 2 h | 36 GB |
| two diploids | two populations | 90 min | 21 GB |
| four diploids | two populations | 8 h | 100 GB |

The authors emphasize that CPU and memory depend on sequence count, time
segments, and CPU count; haplotype and time-segment effects are approximately
quadratic. See the [MSMC/MSMC2 protocol and resource
requirements](https://link.springer.com/protocol/10.1007/978-1-0716-0199-0_7).

The 2025 PHLASH study reports roughly 20–30 CPU min Gbp⁻¹ for one diploid
sample across its compared methods. At larger sample sizes, PHLASH used less
CPU time and memory than the HMM-based SMC++ and MSMC2 comparisons and was the
only compared method to complete the 1,000-sample benchmark within the study's
24 h/256 GB limit. Those are paper-level CPU-resource comparisons, not smckit
wall-clock guarantees. See the [PHLASH runtime and memory
study](https://doi.org/10.1038/s41588-025-02323-x).

## How to estimate your own run safely

1. Prepare one representative chromosome or 1–5% of the loci using the exact
   masks, sample count, pair selection, state pattern, and model options planned
   for production.
2. Run one cold call, then at least five warmed calls in a persistent process.
   Record preparation, inference, serialization, and plotting separately.
3. Repeat with native and upstream in alternating order, one numeric thread,
   and identical inputs. Do not compare an in-process native kernel with a
   fresh upstream subprocess.
4. Scale only the dimension known to be close to linear. Pairwise methods grow
   with the number of selected pairs; optimizer iterations and convergence are
   not safely extrapolated from sequence length alone.
5. Add bootstrap, cross-validation, posterior particles, meta-starts, and
   chromosomes as explicit multipliers. Report accelerator compilation and
   data transfer separately.

The publication workflow's benchmark runner records wall time, process-tree
peak memory, platform, threads, command, input hashes, and output hashes:

```console
python workflow/publication/scripts/run_benchmark.py --help
```

## What is still missing

A complete 1.0 runtime table requires the same frozen protocol for every
scientifically meaningful capability on Linux x86-64 and macOS ARM64, plus the
documented NVIDIA/JAX environment where applicable. PSMC, MSMC2, MSMC-IM,
eSMC2, ASMC, PHLASH, and SSM still need retained smckit production-scale timing
records. The published and historical numbers above are planning anchors, not
substitutes for that evidence.
