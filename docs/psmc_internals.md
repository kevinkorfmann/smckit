# PSMC Internals — Annotated Reference

Technical reference for the PSMC (Pairwise Sequentially Markovian Coalescent)
implementation by Heng Li. This document maps the C source code to its underlying
mathematics, intended as the blueprint for the smckit reimplementation.

**Reference paper:** Li & Durbin (2011) "Inference of human population history from
individual whole-genome sequences." *Nature* 475:493-496.

---

## 1. Source File Map

| File     | Purpose |
|----------|---------|
| `main.c` | Entry point: parse CLI → init → EM loop → decode |
| `cli.c`  | CLI parsing, I/O, pattern parsing, sequence reading, decoding output |
| `core.c` | Coalescent-to-HMM parameter mapping (`psmc_update_hmm`), time interval setup |
| `em.c`   | EM algorithm: E-step (forward-backward + expected counts), M-step (Hooke-Jeeves optimization) |
| `khmm.c` | Generic HMM engine: forward, backward, Viterbi, expected counts, Q-function, simulation |
| `kmin.c` | Hooke-Jeeves direct search optimizer |
| `psmc.h` | Data structures: `psmc_par_t` (config), `psmc_data_t` (model state) |
| `khmm.h` | HMM data structures: `hmm_par_t` (HMM params), `hmm_data_t` (per-sequence), `hmm_exp_t` (expected counts) |
| `aux.c`  | Bootstrap resampling, output formatting, parameter I/O, post-EM decoding |

---

## 2. Input Format (PSMCFA)

PSMC reads FASTA-like files where each character represents a window (default 100 bp)
of the diploid consensus sequence:

| Character          | Code | Meaning |
|--------------------|------|---------|
| `T`, `A`, `C`, `G`, `0` | 0    | Homozygous — no heterozygosity in window |
| `K`, `M`, `R`, `S`, `W`, `Y`, `1` | 1    | Heterozygous — at least one het site in window |
| Everything else (`N`, etc.)  | 2    | Missing data |

**Source:** `cli.c:15-32` — `conv_table[]` defines the full mapping.

The sequences are stored in `psmc_seq_t`:
- `L` — total number of windows (including missing)
- `L_e` — number of callable windows (code 0 or 1)
- `n_e` — number of heterozygous windows (code 1)

---

## 3. HMM Structure

### 3.1 States

The HMM has **n+1 states** (indexed k = 0, 1, ..., n), where each state represents
coalescence occurring in the time interval [t_k, t_{k+1}).

Default: with pattern `"4+5*3+4"`, n = 22, giving 23 states.

### 3.2 Observation Symbols

- m = 2 observation symbols: 0 (homozygous) and 1 (heterozygous)
- Symbol 2 (missing) is handled specially: `e[2][k] = 1.0` for all k
  (missing data is equally likely under all states)

**Source:** `khmm.c:21` — `hp->e[m][i] = 1.0` sets the missing-data emission to 1.

### 3.3 Data Structures

```
hmm_par_t {
    m = 2           // number of symbols
    n = n+1         // number of states (psmc.h n + 1)
    a[k][l]         // transition matrix: P(next=l | current=k), size (n+1)×(n+1)
    e[b][k]         // emission matrix: P(obs=b | state=k), size 3×(n+1)
    ae[b*n+k][l]    // precomputed a[k][l]*e[b][l] for backward algorithm efficiency
    a0[k]           // initial state distribution (σ_k)
}
```

---

## 4. Time Discretization

**Source:** `core.c:6-19` — `psmc_update_intv()`

Time intervals are exponentially spaced to provide finer resolution in the recent
past and coarser resolution in the distant past:

```
β = ln(1 + max_t / α) / n
t_k = α · (exp(β·k) - 1)    for k = 0, ..., n-1
t_n = max_t                   (default 15.0, in units of 2N₀)
t_{n+1} = 1000.0              (effective infinity)
```

Default: α = 0.1, max_t = 15.0.

**Interval widths:** τ_k = t_{k+1} - t_k

---

## 5. Parameter Pattern

**Source:** `cli.c:66-99` — `psmc_parse_pattern()`

The pattern string (e.g., `"4+5*3+4"`) defines how coalescent time intervals are
grouped to share the same λ (population size) parameter:

```
"4+5*3+4" → stack = [4, 3, 3, 3, 3, 3, 4]
                      |  |           |  |
                      |  5 groups    |  |
                      |  of 3 each  |  |
                      4 at start    4 at end
```

- **n** = (sum of stack) - 1 = 23 - 1 = 22
- **n_free** = len(stack) = 7 free λ parameters
- **par_map[k]** maps state k to its free parameter index (0..n_free-1)

Example mapping for `"4+5*3+4"`:
- States 0-3 → free param 0 (4 intervals share one λ)
- States 4-6 → free param 1 (3 intervals share one λ)
- States 7-9 → free param 2
- ...
- States 19-22 → free param 6

---

## 6. Free Parameters

**Source:** `core.c:32-48`, `psmc.h:61`

The parameter vector `pd->params[]` contains:

| Index | Symbol | Meaning |
|-------|--------|---------|
| 0     | θ₀     | Scaled mutation rate: 4N₀μ × window_size |
| 1     | ρ₀     | Scaled recombination rate: 4N₀r × window_size |
| 2     | max_t  | Maximum coalescent time (in units of 2N₀) |
| 3..3+n_free-1 | λ_k | Relative population sizes: N_e(t_k)/N₀ |

**Initial values:**
- θ₀ = -ln(1 - n_e/L_e) — estimated from observed heterozygosity
- ρ₀ = θ₀ / tr_ratio — scaled by theta/rho ratio (default 4.0)
- λ_k = 1.0 + uniform_noise — small random perturbation around 1

---

## 7. Coalescent-to-HMM Mapping (The Core Math)

**Source:** `core.c:61-133` — `psmc_update_hmm()`

This is the most mathematically dense function. It maps population parameters
(θ₀, ρ₀, λ_k, time boundaries) to HMM parameters (transition matrix a[k][l],
emission matrix e[b][k], initial distribution a0[k]).

### 7.1 Survival Probabilities (α_k)

α_k = probability that two lineages have NOT coalesced before reaching interval k:

```
α_0 = 1.0
α_k = α_{k-1} · exp(-τ_{k-1} / λ_{k-1})    for k = 1, ..., n
α_{n+1} = 0.0
```

The coalescence rate in interval k is 1/λ_k (in units of 2N₀). The probability
of surviving through interval k without coalescing is exp(-τ_k / λ_k).

### 7.2 Weighted Accumulated Time (β_k)

```
β_0 = 0.0
β_k = β_{k-1} + λ_{k-1} · (1/α_k - 1/α_{k-1})
```

### 7.3 Auxiliary Array (q_aux)

```
q_aux_l = (α_l - α_{l+1}) · (β_l - λ_l/α_l) + τ_l
```

### 7.4 Normalization Constants

```
C_π = Σ_l λ_l · (α_l - α_{l+1})       // total expected coalescence time
C_σ = 1/(C_π · ρ₀) + 0.5              // normalization for initial distribution
```

### 7.5 Per-State Quantities

For each state k, define:
- ak1 = α_k - α_{k+1} (probability of coalescing IN interval k)
- cpik = ak1 · (sum_t + λ_k) - α_{k+1} · τ_k

**Stationary distribution π_k:**
```
π_k = cpik / C_π
```

**Initial state distribution σ_k (= a0[k]):**
```
σ_k = (ak1/(C_π·ρ₀) + π_k/2) / C_σ
```

**Average coalescence time within interval k (for emission):**
```
avg_t = -ln(1 - π_k/(C_σ·σ_k)) / ρ₀
```
With fallback: if NaN or out of bounds → `avg_t = sum_t + (λ_k - τ_k·α_{k+1}/ak1)`

### 7.6 Transition Matrix

The transition probabilities involve an intermediate quantity q[k][l] — the probability
of coalescing in interval l given a recombination event from state k:

```
For l < k:   q[k][l] = (ak1 / cpik) · q_aux[l]
For l = k:   q[k][k] = (ak1² · (β_k - λ_k/α_k) + 2·λ_k·ak1 - 2·α_{k+1}·τ_k) / cpik
For l > k:   q[k][l] = (α_l - α_{l+1}) · q_aux[k] / cpik
```

Then the transition matrix combines "no recombination" (stay) and "recombination" (jump):
```
tmp = π_k / (C_σ · σ_k)               // probability of recombination

a[k][l] = tmp · q[k][l]               for l ≠ k
a[k][k] = tmp · q[k][k] + (1 - tmp)   for l = k
```

**Physical interpretation:** With probability (1-tmp), no recombination occurs and
the coalescent state stays at k. With probability tmp, recombination occurs, and
the new coalescent time is drawn from q[k][·].

### 7.7 Emission Matrix

```
e[0][k] = exp(-θ₀ · (avg_t + dt))     // P(homozygous | state k)
e[1][k] = 1 - e[0][k]                 // P(heterozygous | state k)
e[2][k] = 1.0                         // P(missing | any state)
```

Where dt is the divergence time parameter (0 unless divergence model is enabled).

**Physical interpretation:** The expected number of mutations between two lineages
that coalesced at time avg_t is 2 · avg_t · μ. The probability of observing zero
mutations in a window decays exponentially with coalescence time.

---

## 8. EM Algorithm

**Source:** `em.c`

### 8.1 E-Step (lines 32-55)

For each input sequence:
1. **Forward algorithm** (`hmm_forward`) — compute f[u][k] and scaling factors s[u]
2. **Backward algorithm** (`hmm_backward`) — compute b[u][k]
3. **Log-likelihood** (`hmm_lk`) — LL = Σ_u log(s[u])
4. **Expected counts** (`hmm_expect`) — compute:
   - A[k][l] = Σ_u f[u][k] · a[k][l] · e[seq[u+1]][l] · b[u+1][l]  (transition counts)
   - E[b][k] = Σ_{u: seq[u]=b} f[u][k] · b[u][k] · s[u]  (emission counts)
5. **Accumulate** counts across sequences (`hmm_add_expect`)

### 8.2 M-Step (lines 56-68)

The M-step does NOT use closed-form updates (unlike standard Baum-Welch).
Instead, it uses **Hooke-Jeeves direct search** (`kmin_hj`) to maximize the
Q-function:

```
Q(params) = Σ_{b,k} E[b][k]·log(e[b][k]) + Σ_{k,l} A[k][l]·log(a[k][l]) - Q₀
```

Where Q₀ normalizes so that Q(params_old) = 0.

**Why not closed-form?** Because the HMM parameters (a, e) are not free — they are
nonlinear functions of the population parameters (θ₀, ρ₀, λ_k) via `psmc_update_hmm`.
The M-step optimizes over the population parameters directly.

The optimization function (`func` in em.c:15-25):
1. Takes absolute values of all parameters (ensures positivity)
2. Calls `psmc_update_hmm` to recompute HMM parameters
3. Returns `-hmm_Q()` (negated because kmin_hj minimizes)

### 8.3 Post-EM Update (lines 69-74)

After EM, update posterior sigma (the posterior state distribution):
```
post_sigma[k] = (E[0][k] + E[1][k]) / Σ_k(E[0][k] + E[1][k])
```

---

## 9. HMM Engine Details

**Source:** `khmm.c`

### 9.1 Forward Algorithm (lines 145-190)

Standard scaled forward algorithm:

```
f[1][k] = a0[k] · e[seq[1]][k]        (normalized by s[1] = Σ_k f[1][k])
f[u][k] = e[seq[u]][k] · Σ_l f[u-1][l] · a[l][k]   (normalized by s[u])
```

**Implementation detail:** Uses transposed `at[k][l] = a[l][k]` for cache-friendly
inner loop. The inner loop computes `Σ_l f[u-1][l] · at[k][l]` which accesses
both arrays sequentially.

**Complexity:** O(L · n²) per sequence, where L = sequence length, n = number of states.

### 9.2 Backward Algorithm (lines 210-241)

Standard scaled backward:

```
b[L][k] = 1 / s[L]
b[u][k] = (Σ_l a[k][l] · e[seq[u+1]][l] · b[u+1][l]) / s[u]
```

**Optimization:** Uses precomputed `ae[b·n + k][l] = e[b][l] · a[k][l]` so the
inner loop is just `Σ_l ae[b·n+k][l] · b[u+1][l]`.

**Underflow check:** After backward, verifies that
`Σ_l a0[l] · b[1][l] · e[seq[1]][l] ≈ 1.0` (within 1e-6).

### 9.3 Log-Likelihood (lines 245-260)

```
log P(seq) = Σ_u log(s[u])
```

Accumulated with periodic resets (`prod *= s[u]`, flush to sum when prod
approaches overflow/underflow boundaries).

### 9.4 Expected Counts (lines 297-324)

```
A[k][l] += f[u][k] · ae[seq[u+1]·n + k][l] · b[u+1][l]    for u = 1..L-1
E[b][k] += f[u][k] · b[u][k] · s[u]                         for u = 1..L-1
A0[l]   += a0[l] · e[seq[1]][l] · b[1][l]
```

Initialized with `HMM_TINY = 1e-25` (not zero) to avoid log(0) in Q-function.

### 9.5 Q-Function (lines 363-382)

```
Q(params) = Σ_{b,k} E[b][k] · log(e[b][k]) + Σ_{k,l} A[k][l] · log(a[k][l]) - Q₀
```

Where Q₀ (`hmm_Q0`, lines 326-342) computes the entropy-like baseline from
expected counts alone:
```
Q₀ = Σ_k Σ_b E[b][k] · log(E[b][k] / Σ_b' E[b'][k])
   + Σ_k Σ_l A[k][l] · log(A[k][l] / Σ_l' A[k][l'])
```

### 9.6 Viterbi Algorithm (lines 83-141)

Standard Viterbi in log-space. Not used in the main EM loop —
only for explicit decoding (`-d` flag).

### 9.7 Posterior Decoding (lines 264-282)

```
decoded[u] = argmax_k  f[u][k] · b[u][k] · s[u]
```

---

## 10. Hooke-Jeeves Optimizer

**Source:** `kmin.c`

A derivative-free direct search method. At each iteration:

1. **Explore:** For each dimension k, try stepping ±dx[k]. Keep the step if it
   reduces the objective.
2. **Pattern move:** If exploration improved, extrapolate in the same direction
   and repeat.
3. **Shrink:** If no improvement, reduce step sizes by factor r and retry.
4. **Converge:** Stop when step radius < eps or max_calls reached.

**Parameters:**
- `KMIN_RADIUS = 0.5` — initial step size relative to |x[k]|
- `KMIN_EPS = 1e-7` — convergence threshold
- `KMIN_MAXCALL = 50000` — maximum function evaluations

---

## 11. Decoding & Output

**Source:** `aux.c:129-232`

After EM converges, PSMC can:

1. **Posterior decoding** (`-d`): For each window, find the most likely coalescent
   time state. Output consecutive runs as segments:
   ```
   DC  seq_name  start  end  state_k  t_k·θ  max_posterior_prob
   ```

2. **Full decoding** (`-D`): For each window, output the full posterior distribution
   over all states plus the recombination probability:
   ```
   DF  position  P(recomb)  P(state_0)  ...  P(state_n)
   ```

3. **Simulation** (`-S`): Generate synthetic sequences from the fitted HMM.

### Output Format

The main output (`.psmc` file) is a structured text format:

```
CC  comments
MM  metadata (version, pattern, parameters)
RD  round_number                      // EM iteration marker
LK  log_likelihood
QD  Q_before → Q_after               // Q-function before and after M-step
RI  relative_information              // KL(sigma || post_sigma)
TR  theta_0  rho_0
MT  max_t
RS  k  t_k  lambda_k  pi_k  sigma_k  post_sigma_k   // per-state results
PA  pattern  param_0  param_1  ...    // raw parameter vector
//                                     // end of round marker
```

---

## 12. Converting to Physical Units

PSMC outputs are in coalescent units. To convert:

```
N₀ = θ₀ / (4μ · window_size)
N_e(t_k) = λ_k · N₀
T_years(t_k) = t_k · 2 · N₀ · g
```

Where:
- μ = per-base per-generation mutation rate (e.g., 1.25e-8 for humans)
- g = generation time in years (e.g., 25 for humans)
- window_size = 100 (default)

---

## 13. Computational Profile & GPU Opportunities

### Bottlenecks (in order of cost)

1. **Forward/Backward algorithms** — O(L · n²) per sequence, per EM iteration.
   For a human genome: L ≈ 3×10⁷ windows, n ≈ 23 states → ~1.6×10¹⁰ FLOPs per pass.
   **GPU target: matrix-vector products along the sequence.**

2. **Expected counts** — O(L · n²) per sequence. Same structure as forward/backward.
   **GPU target: parallel reduction.**

3. **Hooke-Jeeves M-step** — Each function evaluation calls `psmc_update_hmm` (O(n²))
   then `hmm_Q` (O(n²)). With ~50000 max calls and ~30 params, this is relatively
   cheap compared to the E-step. **Moderate GPU benefit.**

4. **psmc_update_hmm** — O(n²) matrix construction. Called once per M-step function
   evaluation. **Low GPU priority** unless n is very large.

### Memory Profile

- Forward/backward matrices: 2 × L × n doubles = ~1 GB for human genome with n=23
- Transition matrix: n² = tiny
- Expected counts: n² + m×n = tiny

### Parallelism Opportunities

- **Across chromosomes:** Each sequence is independent in the E-step → trivially parallel.
- **Within forward/backward:** Each position u depends on u-1 (sequential along genome),
  but the n-state vector at each position is a matrix-vector product (parallel).
- **Across bootstrap replicates:** Fully independent.

### Numerical Considerations

- Scaling factors s[u] prevent underflow in forward/backward
- HMM_TINY = 1e-25 added to expected counts to avoid log(0)
- Hooke-Jeeves uses |params| to ensure positivity (not log-transform)
- Potential issue: for very large n (many states), the transition matrix can become
  ill-conditioned near α_k ≈ 0

---

## 14. C Function → smckit Method Mapping

| C Function | smckit Target | Module |
|------------|-------------|--------|
| `psmc_read_seq` / `conv_table` | `smckit.io.read_psmcfa()` | `io/` |
| `psmc_parse_pattern` | `smckit.tl._psmc.parse_pattern()` | `tl/` |
| `psmc_update_intv` | `smckit.tl._psmc.compute_time_intervals()` | `tl/` |
| `psmc_update_hmm` | `smckit.tl._psmc.compute_hmm_params()` | `tl/` |
| `hmm_forward` | `smckit.backends._numpy.forward()` | `backends/` |
| `hmm_backward` | `smckit.backends._numpy.backward()` | `backends/` |
| `hmm_expect` | `smckit.backends._numpy.expected_counts()` | `backends/` |
| `hmm_Q` | `smckit.backends._numpy.q_function()` | `backends/` |
| `hmm_Viterbi` | `smckit.backends._numpy.viterbi()` | `backends/` |
| `hmm_post_decode` | `smckit.backends._numpy.posterior_decode()` | `backends/` |
| `kmin_hj` | `scipy.optimize.minimize` (or custom) | `tl/` |
| `psmc_em` | `smckit.tl._psmc.em_step()` | `tl/` |
| `psmc_decode` | `smckit.tl._psmc.decode()` | `tl/` |
| `psmc_print_data` | `smckit.io.write_psmc()` | `io/` |
| `psmc_resamp` | `smckit.tl._psmc.bootstrap_resample()` | `tl/` |
| `psmc_simulate` | `smckit.tl._psmc.simulate()` | `tl/` |

---

## 15. SSM Interpretation

PSMC maps naturally to a state-space model:

```
State-Space Model:
    x_u ∈ {0, 1, ..., n}               hidden state (coalescent interval)
    y_u ∈ {0, 1, 2}                     observation (homo/het/missing)

    P(x_1 = k) = σ_k                    initial distribution
    P(x_u = l | x_{u-1} = k) = a[k][l]  transition (coalescent + recombination)
    P(y_u = b | x_u = k) = e[b][k]      emission (mutation process)
```

The transition model encodes:
- **No recombination (diagonal):** coalescent state stays the same
- **Recombination (off-diagonal):** new coalescent drawn from q[k][·], weighted by
  the probability of recombination occurring

The emission model encodes:
- **Infinite-sites mutation:** probability of seeing a heterozygous site increases
  with coalescence time (deeper coalescence → more mutations → more heterozygosity)
