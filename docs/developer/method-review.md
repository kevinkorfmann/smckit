# Quarterly method review

Review candidates in January, April, July, and October. Score each from 0–2
for biological relevance, peer review, license compatibility, maintenance
health, reproducibility, input overlap, and scientific distinctness.

Current order:

1. PHLASH — integrated externally; posterior uncertainty and large-sample scaling.
2. PSMC+ — accepted for preservation and native implementation; genomic
   heterogeneity and background-selection-aware demographic inference.
3. cobraa — next investigation for ancestral structure.
4. SMCSMC and CHIMP — secondary candidates.
5. SINGER and other ARG samplers — deferred until smckit has a tree-sequence/ARG
   result architecture.

An intake decision records the score, package/repository version, license,
maintainer activity, reproducible example, result-schema impact, and whether
an external adapter or a distinct native contribution is justified.

## 2026-08 review: PSMC+

[PSMC+](https://github.com/trevorcousins/PSMCplus) is an MIT-licensed Python
extension of PSMC that accepts multihetsep inputs and optional local-rate maps.
It supports demographic fitting, posterior decoding, binning, fixed or fitted
recombination, grouped time intervals, and genomic coalescence-rate
heterogeneity. Its accompanying [bioRxiv preprint](https://doi.org/10.1101/2024.01.18.576291)
uses the local-rate model to address background-selection bias; as of this
review, the work remains a preprint rather than a peer-reviewed article.

| Criterion | Score | Evidence |
|---|---:|---|
| Biological relevance | 2 | Directly addresses selection-induced bias in demographic inference. |
| Peer review | 0 | The method paper remains a bioRxiv preprint. |
| License compatibility | 2 | The software repository declares the MIT license. |
| Maintenance health | 1 | Public source and examples exist, but there is no tagged release. |
| Reproducibility | 2 | A bundled simulation, tutorials, and a documented smoke command are available. |
| Input overlap | 2 | Multihetsep is already a first-class smckit input family. |
| Scientific distinctness | 2 | Local coalescence/mutation-rate maps extend the existing neutral methods. |
| **Total** | **11/14** | Accepted with preprint-status limitations stated explicitly. |

The implementation decision is preservation first, followed by a native port.
The upstream layer must pin an immutable commit and retain every original
argument and artifact. Native promotion requires parity for ordinary fitting,
rate-map fitting, decoding, binning, interval grouping, initialization,
convergence controls, and fixed/fitted recombination, followed by the same
runtime and memory gates as the existing methods. The paper must not describe
PSMC+ as peer reviewed unless that status changes and is reverified.

The repository audit on 2026-08-08 resolved commit
`032168f2ceed3c0e46b7f214f890faf83dff41ae` as the candidate immutable pin.
The GitHub source archive for that commit has SHA-256
`91bcc572ad59e1c9f14d01552bb915e13a051f1c827a2f6493e89b13d038685b`.
There is still no tagged release, so the commit—not a moving branch name—must
be the preservation identity. The source tree is small (the CLI plus
`BaumWelch.py`, `transition_matrix.py`, `utils.py`, `simulate_HMM.py`, one
bundled constant-size fixture, tutorials, and processing workflow) and declares
the MIT license in `LICENSE.txt`.

The complete CLI ledger at that pin contains:

- one or more multihetsep inputs and matched per-input mutation (`-in_M`) and
  recombination (`-in_R`) maps;
- time-window count/spread, final-boundary, bin-size, theta, rho,
  mutation-to-recombination ratio, fixed-rho, and rate-map-downsampling
  controls;
- lambda initialization, grouped/free/fixed interval patterns, lower/upper
  bounds, optimizer choice, and Powell `xtol`/`ftol` controls;
- iteration and likelihood stopping criteria, optional iteration artifacts,
  midpoint transition/emission variants, and the alternative recombination
  probability approximation;
- posterior decoding/downsampling, optional marginal recombination-probability
  output, final parameter files, and parallel-core selection.

Preservation must cover all 31 arguments and their short/long option forms plus every generated
artifact, including `params_iterationN.txt`, `final_parameters.txt`, decoded
posterior matrices, marginal recombination probabilities, stdout, stderr, and
exit status. Vendoring is deliberately deferred until the SMC++ promotion
increment is committed so the two method changes remain independently
reviewable.
