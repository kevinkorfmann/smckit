"""eSMC2: ecological Sequentially Markovian Coalescent 2.

Reimplementation of Sellinger et al. (2020) for inference of demographic
history, dormancy (seed banking), and self-fertilization rates from
pairwise whole-genome sequence data.

Reference: Sellinger et al., "Inference of past demography, dormancy and
self-fertilization rates from whole genome sequence data", PLoS Genetics, 2020.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from smckit._core import SmcData
from smckit import upstream as upstream_registry
from smckit.backends._numba_esmc2 import (
    HMM_TINY,
    esmc2_build_hmm,
    esmc2_forward_loglik,
)
from smckit.tl._implementation import (
    annotate_result,
    choose_implementation,
    method_upstream_available,
    normalize_implementation,
    standard_upstream_metadata,
)

logger = logging.getLogger(__name__)


def _resolve_upstream_esmc2_rscript() -> str | None:
    """Locate an Rscript executable that can run the vendored eSMC2 package."""
    env_rscript = os.environ.get("SMCKIT_ESMC2_RSCRIPT")
    if env_rscript:
        return env_rscript
    status = upstream_registry.method_status("esmc2")
    runtime = status.get("runtime", {}) if status is not None else {}
    runtime_path = runtime.get("path")
    if runtime_path:
        return str(runtime_path)
    return None


def _sequence_to_upstream_matrix(seq: np.ndarray) -> np.ndarray:
    """Convert one dense 0/1 pairwise sequence to upstream eSMC2's O matrix.

    The current upstream bridge intentionally supports only fully callable
    sequences with observations in {0, 1}. This is enough for the parity
    fixtures we use to track native-vs-upstream behavior.
    """
    arr = np.asarray(seq, dtype=np.int8)
    if arr.ndim != 1:
        raise ValueError("Upstream eSMC2 bridge expects a one-dimensional sequence.")
    if len(arr) == 0:
        raise ValueError("Upstream eSMC2 bridge received an empty sequence.")
    if np.any((arr != 0) & (arr != 1)):
        raise ValueError(
            "Upstream eSMC2 bridge currently supports only clean 0/1 sequences "
            "without missing values."
        )

    het_positions = np.flatnonzero(arr == 1)
    rows: list[tuple[int, int, int, int]] = []
    prev_pos = 0
    L = int(len(arr))
    if len(het_positions) == 0:
        rows.append((0, 0, L, L))
    else:
        for idx in het_positions:
            pos = int(idx) + 1  # upstream uses 1-based positions
            rows.append((0, 1, pos - prev_pos, pos))
            prev_pos = pos
        if prev_pos < L:
            rows.append((0, 0, L - prev_pos, L))

    return np.asarray(rows, dtype=np.int64).T


def _build_default_zip_symbols() -> tuple[
    tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]],
    tuple[tuple[int, tuple[int, int]], ...],
    dict[int, int],
    dict[str, int],
]:
    """Recreate the default upstream zip symbol tables numerically."""
    hom_defaults = ("00", "aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh", "ii", "jj", "kk", "ll")
    missing_defaults = ("22", "nn", "oo", "pp", "qq", "rr", "ss", "tt", "uu")
    replacements: list[tuple[tuple[str, str], ...]] = []
    numeric_rows: list[tuple[int, tuple[int, int]]] = []
    lengths: dict[int, int] = {0: 1, 1: 1, 2: 1}
    letter_to_code: dict[str, int] = {}
    letter_offset = 0

    for group_idx, defaults in enumerate((hom_defaults, missing_defaults), start=1):
        group_replacements: list[tuple[str, str]] = []
        for count, rep_symbol in enumerate(defaults, start=1):
            letter = chr(ord("a") + letter_offset + count - 1)
            code = 10 + 20 * (group_idx - 1) + count
            letter_to_code[letter] = code
            group_replacements.append((letter, rep_symbol))
            pair = tuple(
                letter_to_code[ch] if ch in letter_to_code else int(ch) for ch in rep_symbol
            )
            numeric_rows.append((code, pair))
            lengths[code] = lengths[pair[0]] + lengths[pair[1]]
        replacements.append(tuple(group_replacements))
        letter_offset += len(defaults)

    return (
        tuple(replacements),
        tuple(numeric_rows),
        lengths,
        letter_to_code,
    )


(
    _ESMC2_ZIP_REPLACEMENTS,
    _ESMC2_ZIP_NUMERIC_ROWS,
    _ESMC2_ZIP_LENGTHS,
    _ESMC2_ZIP_LETTER_TO_CODE,
) = _build_default_zip_symbols()


def _coerce_real(array: np.ndarray, *, name: str) -> np.ndarray:
    """Reject meaningful imaginary drift while tolerating eigensolver noise."""
    arr = np.asarray(array)
    if np.iscomplexobj(arr):
        imag_max = float(np.max(np.abs(arr.imag)))
        if imag_max > 1e-8:
            raise RuntimeError(
                f"Unexpected complex component in {name}: max |imag|={imag_max:.3e}"
            )
        arr = arr.real
    return np.asarray(arr, dtype=np.float64)


def _zip_sequence_to_upstream_numeric(seq: np.ndarray) -> np.ndarray:
    """Mirror upstream Zip_seq() + symbol2Num() for dense {0,1,2} sequences."""
    tokens = [str(int(x)) for x in np.asarray(seq, dtype=np.int8)]
    if not tokens:
        raise ValueError("Cannot zip an empty sequence.")

    ini = tokens[0]
    zipped = ".".join(tokens[1:])
    for group_replacements in _ESMC2_ZIP_REPLACEMENTS:
        for letter, rep_symbol in group_replacements:
            pattern = f".{rep_symbol[0]}.{rep_symbol[1]}"
            zipped = zipped.replace(pattern, f".{letter}")

    pieces: list[str] = [ini, ini]
    if zipped:
        pieces.extend(zipped.split("."))
    numeric = [
        _ESMC2_ZIP_LETTER_TO_CODE[piece] if piece in _ESMC2_ZIP_LETTER_TO_CODE else int(piece)
        for piece in pieces
    ]
    return np.asarray(numeric, dtype=np.int64)


def _build_zip_cache(Q: np.ndarray, e: np.ndarray) -> dict[str, object]:
    """Build the zipped forward/backward matrices used by upstream eSMC2."""
    g = np.asarray(e, dtype=np.float64).T
    n = Q.shape[0]
    identity = np.eye(n, dtype=np.complex128)

    c_mats: dict[int, np.ndarray] = {
        sym: np.diag(g[:, sym]) @ Q for sym in (0, 1, 2)
    }
    to_mats: dict[int, np.ndarray] = {
        sym: Q.T @ np.diag(g[:, sym]) for sym in (0, 1, 2)
    }
    q_prime: dict[int, np.ndarray] = {
        sym: identity.copy() for sym in (0, 1, 2)
    }
    q_power: dict[int, np.ndarray] = {
        sym: identity.copy() for sym in (0, 1, 2)
    }
    eig_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for base_symbol in (0, 2):
        eigvals, w = np.linalg.eig(to_mats[base_symbol])
        eig_cache[base_symbol] = (eigvals, w, np.linalg.inv(w))

    for code, pair in _ESMC2_ZIP_NUMERIC_ROWS:
        c_mats[code] = c_mats[pair[1]] @ c_mats[pair[0]]
        to_mats[code] = to_mats[pair[0]] @ to_mats[pair[1]]

        base_symbol = 0 if code < 30 else 2
        eigvals, _, _ = eig_cache[base_symbol]
        run_length = _ESMC2_ZIP_LENGTHS[code]
        q_matrix = np.zeros((n, n), dtype=np.complex128)
        q_prime_matrix = np.zeros((n, n), dtype=np.complex128)

        for row in range(n):
            lam_row = eigvals[row]
            for col in range(n):
                lam_col = eigvals[col]
                if np.isclose(lam_row, lam_col, rtol=1e-12, atol=1e-15):
                    q_val = run_length * (lam_col ** (run_length - 1))
                else:
                    q_val = ((lam_row**run_length) - (lam_col**run_length)) / (lam_row - lam_col)
                q_matrix[row, col] = q_val
                q_prime_matrix[row, col] = lam_row * q_val

        q_power[code] = q_matrix
        q_prime[code] = q_prime_matrix

    return {
        "g": g,
        "C": c_mats,
        "TO": to_mats,
        "Q_prime": q_prime,
        "Q_power": q_power,
        "eig": eig_cache,
    }


def _zip_forward(
    symbols: np.ndarray,
    g: np.ndarray,
    q: np.ndarray,
    c_mats: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the upstream zipped forward algorithm."""
    n = len(q)
    t_prime = len(symbols)
    alpha = np.zeros((n, t_prime), dtype=np.float64)
    scales = np.ones(t_prime, dtype=np.float64)

    first_symbol = int(symbols[0])
    alpha[:, 0] = g[:, first_symbol] * q
    total = float(np.sum(alpha[:, 0]))
    scales[0] = total
    if total > 0.0:
        alpha[:, 0] /= total

    for idx in range(1, t_prime):
        sym = int(symbols[idx])
        alpha[:, idx] = c_mats[sym] @ alpha[:, idx - 1]
        total = float(np.sum(alpha[:, idx]))
        scales[idx] = total
        if total > 0.0:
            alpha[:, idx] /= total

    log_likelihood = float(np.sum(np.log(np.maximum(scales, HMM_TINY))))
    return alpha, scales, log_likelihood


def _zip_backward(
    symbols: np.ndarray,
    to_mats: dict[int, np.ndarray],
    scales: np.ndarray,
) -> np.ndarray:
    """Run the upstream zipped backward algorithm."""
    n = next(iter(to_mats.values())).shape[0]
    t_prime = len(symbols)
    beta = np.ones((n, t_prime), dtype=np.float64)

    for idx in range(t_prime - 2, -1, -1):
        sym = int(symbols[idx + 1])
        beta[:, idx] = (to_mats[sym] @ beta[:, idx + 1]) / scales[idx + 1]

    return beta


def _zip_expected_counts(
    *,
    symbols: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    scales: np.ndarray,
    Q: np.ndarray,
    zip_cache: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce upstream Baum-Welch sufficient-stat accumulation exactly."""
    g = np.asarray(zip_cache["g"], dtype=np.float64)
    c_mats = zip_cache["C"]
    q_prime = zip_cache["Q_prime"]
    q_power = zip_cache["Q_power"]
    eig_cache = zip_cache["eig"]

    n = Q.shape[0]
    n_counts = np.zeros((n, n), dtype=np.complex128)
    m_counts = np.zeros((n, 3), dtype=np.complex128)
    q_post = np.zeros(n, dtype=np.float64)

    gamma0 = alpha[:, 0] * beta[:, 0]
    gamma0_sum = float(np.sum(gamma0))
    if gamma0_sum > 0.0:
        q_post += gamma0 / gamma0_sum

    if len(symbols) > 1:
        sym = int(symbols[1])
        if sym <= 2:
            beta_scaled = beta[:, 1] / scales[1]
            n_counts += np.outer(alpha[:, 0], beta_scaled * g[:, sym])
            m_counts[:, sym] += alpha[:, 0] * (q_prime[sym] @ beta_scaled)
        else:
            base_symbol = 0 if sym < 30 else 2
            _, w, w_inv = eig_cache[base_symbol]
            beta_scaled = beta[:, 1] / scales[1]
            outer = np.outer(alpha[:, 0], beta_scaled)
            transformed = w.T @ outer @ w_inv.T
            n_counts += (
                w_inv.T
                @ (transformed * q_power[sym])
                @ w.T
                @ np.diag(g[:, base_symbol])
            )
            m_counts[:, base_symbol] += np.diag(
                w_inv.T @ (transformed * q_prime[sym]) @ w.T
            )

    middle_symbols = np.unique(symbols[1:-1]) if len(symbols) > 2 else np.array([], dtype=np.int64)
    for sym in middle_symbols:
        pos = np.where(symbols[1:-1] == sym)[0] + 1
        if len(pos) == 0:
            continue

        if sym <= 2:
            beta_scaled = beta[:, pos] / scales[pos]
            truc = np.sum(alpha[:, pos - 1] * (c_mats[int(sym)] @ beta_scaled), axis=1)
            truc_sum = float(np.sum(truc))
            if truc_sum > 0.0:
                truc = (truc / truc_sum) * len(pos)
            m_counts[:, int(sym)] += truc
            n_counts += alpha[:, pos - 1] @ (np.diag(g[:, int(sym)]) @ beta_scaled).T
            continue

        base_symbol = 0 if sym < 30 else 2
        _, w, w_inv = eig_cache[base_symbol]
        beta_scaled = beta[:, pos] / scales[pos]
        transformed = w.T @ alpha[:, pos - 1] @ beta_scaled.T @ w_inv.T
        m_counts[:, base_symbol] += np.diag(
            w_inv.T @ (transformed * q_prime[int(sym)]) @ w.T
        )
        n_counts += (
            w_inv.T @ (transformed * q_power[int(sym)]) @ w.T @ np.diag(g[:, base_symbol])
        )

    last_symbol = int(symbols[-1])
    gamma_last = alpha[:, -1] * beta[:, -1]
    if last_symbol <= 2:
        m_counts[:, last_symbol] += gamma_last
    else:
        gamma_last_sum = float(np.sum(gamma_last))
        if gamma_last_sum > 0.0:
            base_symbol = 0 if last_symbol < 30 else 2
            m_counts[:, base_symbol] += gamma_last / gamma_last_sum

    n_real = _coerce_real(n_counts, name="zipped transition counts")
    m_real = _coerce_real(m_counts, name="zipped emission counts")
    return n_real, m_real.T, q_post


def _run_upstream_esmc2(
    *,
    data: SmcData,
    n_states: int,
    rho_over_theta: float,
    n_iterations: int,
    estimate_beta: bool,
    estimate_sigma: bool,
    estimate_rho: bool,
    beta: float,
    sigma: float,
    mu_b: float,
    mu: float,
    generation_time: float,
    pop_vect: list[int] | None,
    box_b: tuple[float, float],
    box_s: tuple[float, float],
    box_p: tuple[float, float],
    box_r: tuple[float, float],
    rp: tuple[float, float],
    capture_sufficient_statistics: bool = False,
    capture_hmm_builder: bool = False,
    capture_final_sufficient_statistics: bool = False,
) -> dict:
    """Run vendored upstream eSMC2 through Rscript.

    This path is intentionally wired like the SMC++ upstream backend, but it
    still requires a working R installation plus the vendored eSMC2 package.
    """
    rscript = _resolve_upstream_esmc2_rscript()
    if rscript is None:
        raise RuntimeError(
            "Upstream eSMC2 backend is unavailable. Set SMCKIT_ESMC2_RSCRIPT "
            "to an Rscript executable with the vendored eSMC2 package installed."
        )
    sequences = _sequences_from_smcdata(data)
    if len(sequences) != 1:
        raise RuntimeError(
            "Upstream eSMC2 backend currently supports exactly one clean pairwise sequence."
        )
    o_matrix = _sequence_to_upstream_matrix(sequences[0])

    repo_root = Path(__file__).resolve().parents[3]
    local_r_lib = repo_root / ".r-lib"
    r_expr = """
args <- commandArgs(trailingOnly = TRUE)
o_path <- args[[1]]
out_dir <- args[[2]]
n_states <- as.integer(args[[3]])
rho_over_theta <- as.numeric(args[[4]])
maxit <- as.integer(args[[5]])
estimate_beta <- as.logical(as.integer(args[[6]]))
estimate_sigma <- as.logical(as.integer(args[[7]]))
estimate_rho <- as.logical(as.integer(args[[8]]))
beta <- as.numeric(args[[9]])
sigma <- as.numeric(args[[10]])
mu_b <- as.numeric(args[[11]])
box_b <- c(as.numeric(args[[12]]), as.numeric(args[[13]]))
box_s <- c(as.numeric(args[[14]]), as.numeric(args[[15]]))
box_p <- c(as.numeric(args[[16]]), as.numeric(args[[17]]))
box_r <- c(as.numeric(args[[18]]), as.numeric(args[[19]]))
rp <- c(as.numeric(args[[20]]), as.numeric(args[[21]]))
pop_vect_raw <- args[[22]]
r_lib <- args[[23]]
capture_final_sufficient_statistics <- as.logical(as.integer(args[[24]]))
if (nchar(r_lib) > 0) {
  .libPaths(c(r_lib, .libPaths()))
}
suppressPackageStartupMessages(library(eSMC2))
O <- as.matrix(read.table(o_path, header = FALSE))
if (pop_vect_raw == "") {
  pop_vect <- NA
} else {
  pop_vect <- as.integer(strsplit(pop_vect_raw, ",", fixed = TRUE)[[1]])
}
compute_final_sufficient_statistics <- function(
  O,
  n_states,
  rho_internal,
  beta_final,
  sigma_final,
  Xi_final,
  mu_scaled,
  mu_b
) {
  M_rows <- dim(O)[1] - 2
  L <- as.numeric(O[(M_rows + 2), dim(O)[2]])
  seq_obj <- eSMC2:::seqSNP2_MH(O[c(1, 2, (M_rows + 1), (M_rows + 2)), ], L, NA)
  Mat_symbol <- eSMC2:::Mat_symbol_ID()
  zipped <- eSMC2:::Zip_seq(seq_obj$seq, Mat_symbol)
  zipped <- eSMC2:::symbol2Num(zipped[[1]], zipped[[2]])

  builder <- eSMC2:::build_HMM_matrix(
    n = n_states,
    rho = rho_internal,
    beta = beta_final,
    L = L,
    Pop = FALSE,
    Xi = Xi_final,
    Beta = beta_final,
    scale = c(1, 0),
    sigma = sigma_final,
    Sigma = sigma_final,
    Big_Window = FALSE,
    npair = 2
  )
  Q <- builder[[1]]
  nu <- builder[[2]]
  t_vals <- builder[[3]]
  g <- matrix(0, nrow = length(t_vals), ncol = 3)
  mu_eff <- beta_final + ((1 - beta_final) * mu_b)
  g[, 2] <- 1 - exp(-mu_scaled * mu_eff * 2 * t_vals)
  g[, 1] <- exp(-mu_scaled * mu_eff * 2 * t_vals)
  g[, 3] <- 1

  test <- eSMC2:::Build_zip_Matrix_mailund(Q, g, zipped[[2]], nu)
  fo <- eSMC2:::forward_zip_mailund(as.numeric(zipped[[1]]), g, nu, test[[1]])
  c <- exp(fo[[2]])
  ba <- eSMC2:::Backward_zip_mailund(as.numeric(zipped[[1]]), test[[3]], length(t_vals), c)

  W_P <- list()
  W_P_ <- list()
  count_oo <- 0
  for (oo in c(1, 3)) {
    count_oo <- count_oo + 1
    int <- eigen(t(Q) %*% diag(g[, oo]))
    W_P[[count_oo]] <- int$vectors
    W_P_[[count_oo]] <- solve(W_P[[count_oo]])
  }

  N <- matrix(0, length(t_vals), length(t_vals))
  M_counts <- matrix(0, nrow = length(t_vals), ncol = 3)
  q_ <- rep(0, length(t_vals))
  symbol <- c(0:2, 10:40)
  ob <- 1
  truc_M <- matrix(0, nrow = length(t_vals), ncol = 3)
  if (as.numeric(zipped[[1]][(ob + 1)]) <= 2) {
    truc_N <- fo[[1]][, ob] %*% t(
      ba[, (ob + 1)] * g[, (as.numeric(zipped[[1]][(ob + 1)]) + 1)] / c[(ob + 1)]
    )
    truc_M[, (as.numeric(zipped[[1]][(ob + 1)]) + 1)] <- fo[[1]][, ob] * (
      test[[2]][[(as.numeric(zipped[[1]][(ob + 1)]) + 1)]] %*% (ba[, (ob + 1)] / c[(ob + 1)])
    )
  } else if (as.numeric(zipped[[1]][(ob + 1)]) < 30) {
    truc_N <- t(W_P_[[1]]) %*% (
      t(W_P[[1]]) %*% fo[[1]][, ob] %*% t(ba[, (ob + 1)] / c[(ob + 1)]) %*% t(W_P_[[1]]) *
      test[[4]][[(as.numeric(zipped[[1]][(ob + 1)]))]]
    ) %*% t(W_P[[1]]) %*% diag(g[, 1])
    truc_M[, 1] <- diag(
      t(W_P_[[1]]) %*% (
        t(W_P[[1]]) %*% fo[[1]][, ob] %*% t(ba[, (ob + 1)] / c[(ob + 1)]) %*% t(W_P_[[1]])
      ) * test[[2]][[(as.numeric(zipped[[1]][(ob + 1)]))]] %*% t(W_P[[1]])
    )
  } else {
    truc_N <- t(W_P_[[2]]) %*% (
      t(W_P[[2]]) %*% fo[[1]][, ob] %*% t(ba[, (ob + 1)] / c[(ob + 1)]) %*% t(W_P_[[2]]) *
      test[[4]][[(as.numeric(zipped[[1]][(ob + 1)]))]]
    ) %*% t(W_P[[2]]) %*% diag(g[, 3])
    truc_M[, 3] <- diag(
      t(W_P_[[2]]) %*% (
        t(W_P[[2]]) %*% fo[[1]][, ob] %*% t(ba[, (ob + 1)] / c[(ob + 1)]) %*% t(W_P_[[2]])
      ) * test[[2]][[(as.numeric(zipped[[1]][(ob + 1)]))]] %*% t(W_P[[2]])
    )
  }
  N <- N + truc_N
  M_counts <- M_counts + truc_M

  for (sym in sort(symbol)) {
    ob <- as.numeric(sym)
    pos <- which(as.numeric(zipped[[1]][-c(1, length(zipped[[1]]))]) == ob)
    if (length(pos) == 0) {
      next
    }
    pos <- pos + 1
    if (ob < 3) {
      ba_t <- t(t(ba[, pos]) / c[pos])
      truc <- c(rowSums(fo[[1]][, (pos - 1)] * (test[[1]][[(ob + 1)]] %*% ba_t)))
      truc <- (truc / sum(truc)) * length(pos)
      M_counts[, (ob + 1)] <- M_counts[, (ob + 1)] + truc
      N <- N + (fo[[1]][, (pos - 1)] %*% t(diag(g[, (ob + 1)]) %*% ba_t))
    } else if (ob < 30) {
      A <- t(W_P[[1]]) %*% fo[[1]][, (pos - 1)] %*% t(t(t(ba[, pos]) / c[pos])) %*% t(W_P_[[1]])
      A_ <- A * test[[2]][[(ob)]]
      A_ <- t(W_P_[[1]]) %*% A_ %*% t(W_P[[1]])
      M_counts[, 1] <- M_counts[, 1] + diag(A_)
      A_ <- A * test[[4]][[(ob)]]
      A_ <- t(W_P_[[1]]) %*% A_ %*% t(W_P[[1]])
      N <- N + (A_ %*% diag(g[, 1]))
    } else {
      A <- t(W_P[[2]]) %*% fo[[1]][, (pos - 1)] %*% t(t(t(ba[, pos]) / c[pos])) %*% t(W_P_[[2]])
      A_ <- A * test[[2]][[(ob)]]
      A_ <- t(W_P_[[2]]) %*% A_ %*% t(W_P[[2]])
      M_counts[, 3] <- M_counts[, 3] + diag(A_)
      A_ <- A * test[[4]][[(ob)]]
      A_ <- t(W_P_[[2]]) %*% A_ %*% t(W_P[[2]])
      N <- N + (A_ %*% diag(g[, 3]))
    }
  }

  last_symbol <- as.numeric(zipped[[1]][length(zipped[[1]])])
  if (last_symbol <= 3) {
    M_counts[, (last_symbol + 1)] <- M_counts[, (last_symbol + 1)] +
      (fo[[1]][, length(zipped[[1]])] * ba[, length(zipped[[1]])])
  } else if (last_symbol < 30) {
    M_counts[, 1] <- M_counts[, 1] + (
      (fo[[1]][, length(zipped[[1]])] * ba[, length(zipped[[1]])]) /
      sum(fo[[1]][, length(zipped[[1]])] * ba[, length(zipped[[1]])])
    )
  } else {
    M_counts[, 3] <- M_counts[, 3] + (
      (fo[[1]][, length(zipped[[1]])] * ba[, length(zipped[[1]])]) /
      sum(fo[[1]][, length(zipped[[1]])] * ba[, length(zipped[[1]])])
    )
  }
  q_ <- q_ + ((fo[[1]][, 1] * ba[, 1]) / sum(fo[[1]][, 1] * ba[, 1]))
  N <- N * t(Q)
  if (is.complex(N)) {
    N <- Re(N)
  }
  if (is.complex(M_counts)) {
    M_counts <- Re(M_counts)
  }
  N <- N * ((L - 1) / sum(N))
  M_counts <- M_counts * (L / sum(M_counts))
  q_ <- q_ / sum(q_)
  list(N = N, M = t(M_counts), q = q_)
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
sink(file.path(out_dir, "stdout.log"))
res <- eSMC2::eSMC2(
  n = n_states,
  rho = rho_over_theta,
  O = O,
  maxit = maxit,
  BoxB = box_b,
  BoxP = box_p,
  Boxr = box_r,
  Boxs = box_s,
  pop = FALSE,
  SB = estimate_beta,
  SF = estimate_sigma,
  Rho = estimate_rho,
  BW = FALSE,
  NC = 1,
  pop_vect = pop_vect,
  mu_b = mu_b,
  sigma = sigma,
  beta = beta,
  rp = rp
)
sink()
write.table(res$Tc, file=file.path(out_dir, "Tc.txt"), row.names=FALSE, col.names=FALSE)
write.table(res$Xi, file=file.path(out_dir, "Xi.txt"), row.names=FALSE, col.names=FALSE)
write.table(res$rho, file=file.path(out_dir, "rho.txt"), row.names=FALSE, col.names=FALSE)
write.table(res$beta, file=file.path(out_dir, "beta.txt"), row.names=FALSE, col.names=FALSE)
write.table(res$sigma, file=file.path(out_dir, "sigma.txt"), row.names=FALSE, col.names=FALSE)
write.table(res$mu, file=file.path(out_dir, "mu.txt"), row.names=FALSE, col.names=FALSE)
write.table(res$LH, file=file.path(out_dir, "LH.txt"), row.names=FALSE, col.names=FALSE)
builder <- eSMC2:::build_HMM_matrix(
  n = n_states,
  rho = as.numeric(res$rho) * 2 * as.numeric(res$L),
  beta = as.numeric(res$beta),
  L = as.numeric(res$L),
  Pop = FALSE,
  Xi = as.numeric(res$Xi),
  Beta = as.numeric(res$beta),
  scale = c(1, 0),
  sigma = as.numeric(res$sigma),
  Sigma = as.numeric(res$sigma),
  Big_Window = FALSE,
  npair = 2
)
Q_builder <- builder[[1]]
nu_builder <- builder[[2]]
t_builder <- builder[[3]]
Tc_builder <- builder[[4]]
g_builder <- matrix(0, nrow=length(t_builder), ncol=3)
mu_eff <- as.numeric(res$beta) + ((1 - as.numeric(res$beta)) * mu_b)
g_builder[,2] <- 1 - exp(-as.numeric(res$mu) * mu_eff * 2 * t_builder)
g_builder[,1] <- exp(-as.numeric(res$mu) * mu_eff * 2 * t_builder)
g_builder[,3] <- 1
write.table(
  Q_builder,
  file=file.path(out_dir, "Q_builder.txt"),
  row.names=FALSE,
  col.names=FALSE
)
write.table(
  nu_builder,
  file=file.path(out_dir, "nu_builder.txt"),
  row.names=FALSE,
  col.names=FALSE
)
write.table(
  t_builder,
  file=file.path(out_dir, "t_builder.txt"),
  row.names=FALSE,
  col.names=FALSE
)
write.table(
  Tc_builder,
  file=file.path(out_dir, "Tc_builder.txt"),
  row.names=FALSE,
  col.names=FALSE
)
write.table(
  g_builder,
  file=file.path(out_dir, "g_builder.txt"),
  row.names=FALSE,
  col.names=FALSE
)
if (!is.null(res$N)) {
  write.table(as.matrix(res$N), file=file.path(out_dir, "N.txt"), row.names=FALSE, col.names=FALSE)
}
if (!is.null(res$M)) {
  write.table(as.matrix(res$M), file=file.path(out_dir, "M.txt"), row.names=FALSE, col.names=FALSE)
}
if (!is.null(res$q_)) {
  write.table(res$q_, file=file.path(out_dir, "q.txt"), row.names=FALSE, col.names=FALSE)
}
if (capture_final_sufficient_statistics) {
  final_stats <- compute_final_sufficient_statistics(
    O = O,
    n_states = n_states,
    rho_internal = as.numeric(res$rho) * 2 * as.numeric(res$L),
    beta_final = as.numeric(res$beta),
    sigma_final = as.numeric(res$sigma),
    Xi_final = as.numeric(res$Xi),
    mu_scaled = as.numeric(res$mu),
    mu_b = mu_b
  )
  write.table(
    as.matrix(final_stats$N),
    file=file.path(out_dir, "final_N.txt"),
    row.names=FALSE,
    col.names=FALSE
  )
  write.table(
    as.matrix(final_stats$M),
    file=file.path(out_dir, "final_M.txt"),
    row.names=FALSE,
    col.names=FALSE
  )
  write.table(
    final_stats$q,
    file=file.path(out_dir, "final_q.txt"),
    row.names=FALSE,
    col.names=FALSE
  )
}
"""

    with tempfile.TemporaryDirectory(prefix="smckit-esmc2-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        o_path = tmpdir_path / "O.txt"
        script_path = tmpdir_path / "run_esmc2.R"
        np.savetxt(o_path, o_matrix, fmt="%d")
        script_path.write_text(r_expr, encoding="utf-8")
        out_dir = tmpdir_path / "out"

        proc = subprocess.run(
            [
                rscript,
                str(script_path),
                str(o_path),
                str(out_dir),
                str(int(n_states)),
                repr(float(rho_over_theta)),
                str(int(n_iterations)),
                "1" if estimate_beta else "0",
                "1" if estimate_sigma else "0",
                "1" if estimate_rho else "0",
                repr(float(beta)),
                repr(float(sigma)),
                repr(float(mu_b)),
                repr(float(box_b[0])),
                repr(float(box_b[1])),
                repr(float(box_s[0])),
                repr(float(box_s[1])),
                repr(float(box_p[0])),
                repr(float(box_p[1])),
                repr(float(box_r[0])),
                repr(float(box_r[1])),
                repr(float(rp[0])),
                repr(float(rp[1])),
                "" if pop_vect is None else ",".join(str(int(v)) for v in pop_vect),
                str(local_r_lib if local_r_lib.exists() else ""),
                "1" if capture_final_sufficient_statistics else "0",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Upstream eSMC2 backend failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

        Tc_raw = np.loadtxt(out_dir / "Tc.txt", dtype=np.float64)
        Xi = np.loadtxt(out_dir / "Xi.txt", dtype=np.float64)
        rho_public = float(np.loadtxt(out_dir / "rho.txt", dtype=np.float64))
        beta_final = float(np.loadtxt(out_dir / "beta.txt", dtype=np.float64))
        sigma_final = float(np.loadtxt(out_dir / "sigma.txt", dtype=np.float64))
        mu_scaled = float(np.loadtxt(out_dir / "mu.txt", dtype=np.float64))
        log_likelihood = float(np.loadtxt(out_dir / "LH.txt", dtype=np.float64))
        n = int(np.size(Tc_raw))
        Xi = np.asarray(Xi, dtype=np.float64).reshape(n)
        Tc_raw = np.asarray(Tc_raw, dtype=np.float64).reshape(n)
        q_builder = (
            np.asarray(np.loadtxt(out_dir / "nu_builder.txt", dtype=np.float64), dtype=np.float64)
            .reshape(n)
        )
        t_builder = (
            np.asarray(np.loadtxt(out_dir / "t_builder.txt", dtype=np.float64), dtype=np.float64)
            .reshape(n)
        )
        Tc_builder = (
            np.asarray(np.loadtxt(out_dir / "Tc_builder.txt", dtype=np.float64), dtype=np.float64)
            .reshape(n)
        )
        g_builder = np.asarray(
            np.loadtxt(out_dir / "g_builder.txt", dtype=np.float64),
            dtype=np.float64,
        ).reshape(n, 3)
        rho_per_sequence = _sequence_rho_from_public_rho(rho_public, len(sequences[0]))
        _, _, t, Tc_check, _ = esmc2_build_hmm(
            n,
            Xi,
            beta_final,
            sigma_final,
            rho_per_sequence,
            mu_scaled,
            mu_b,
            len(sequences[0]),
        )
        if not np.allclose(Tc_check, Tc_builder, rtol=1e-10, atol=1e-12):
            raise RuntimeError(
                "Upstream eSMC2 returned time boundaries inconsistent with local HMM builder."
            )
        theta = mu_scaled * 2.0 * len(sequences[0])
        ne0 = theta / (4.0 * mu)
        ne = Xi * ne0
        time_years = t * 2.0 * ne0 * generation_time
        stdout_log = out_dir / "stdout.log"
        payload = {
            "Tc": Tc_builder.tolist(),
            "t": t_builder.tolist(),
            "Xi": Xi.tolist(),
            "ne": np.asarray(ne, dtype=np.float64).tolist(),
            "time_years": np.asarray(time_years, dtype=np.float64).tolist(),
            "beta": float(beta_final),
            "sigma": float(sigma_final),
            "mu": float(mu_scaled),
            "rho": float(rho_public),
            "rho_per_sequence": float(rho_per_sequence),
            "theta": float(theta),
            "log_likelihood": float(log_likelihood),
            "n_iterations": int(n_iterations),
            "rounds": [],
            "upstream": {
                "sequence_length": int(len(sequences[0])),
                "rscript": rscript,
                "Tc_returned": Tc_raw.tolist(),
                "stdout_log": (
                    stdout_log.read_text(encoding="utf-8") if stdout_log.exists() else ""
                ),
            },
        }
        if capture_sufficient_statistics:
            stats: dict[str, list] = {}
            n_path = out_dir / "N.txt"
            if n_path.exists():
                stats["N"] = np.loadtxt(n_path, dtype=np.float64).reshape(n, n).tolist()
            m_path = out_dir / "M.txt"
            if m_path.exists():
                m = np.asarray(np.loadtxt(m_path, dtype=np.float64), dtype=np.float64)
                if m.ndim == 1:
                    m = m.reshape(n, 3)
                if m.shape == (n, 3):
                    m = m.T
                elif m.shape != (3, n):
                    raise RuntimeError(
                        "Upstream eSMC2 returned emission counts with unexpected shape "
                        f"{m.shape}; expected {(n, 3)} or {(3, n)}."
                    )
                stats["M"] = m.tolist()
            q_path = out_dir / "q.txt"
            if q_path.exists():
                stats["q"] = (
                    np.asarray(np.loadtxt(q_path, dtype=np.float64), dtype=np.float64)
                    .reshape(n)
                    .tolist()
                )
            payload["sufficient_statistics"] = stats
        if capture_final_sufficient_statistics:
            final_stats: dict[str, list] = {
                "N": (
                    np.asarray(
                        np.loadtxt(out_dir / "final_N.txt", dtype=np.float64),
                        dtype=np.float64,
                    )
                    .reshape(n, n)
                    .tolist()
                ),
                "M": (
                    np.asarray(
                        np.loadtxt(out_dir / "final_M.txt", dtype=np.float64),
                        dtype=np.float64,
                    )
                    .reshape(3, n)
                    .tolist()
                ),
                "q": (
                    np.asarray(
                        np.loadtxt(out_dir / "final_q.txt", dtype=np.float64),
                        dtype=np.float64,
                    )
                    .reshape(n)
                    .tolist()
                ),
            }
            payload["final_sufficient_statistics"] = final_stats
        if capture_hmm_builder:
            builder_payload = {
                "Q": np.asarray(np.loadtxt(out_dir / "Q_builder.txt", dtype=np.float64))
                .reshape(n, n)
                .tolist(),
                "q": q_builder.tolist(),
                "t": t_builder.tolist(),
                "Tc": Tc_builder.tolist(),
                "g": g_builder.T.tolist(),
            }
            payload["hmm_builder"] = builder_payload
        return payload


def _sequences_from_smcdata(data: SmcData) -> list[np.ndarray]:
    """Extract dense pairwise sequences from supported SmcData payloads."""
    if "segments" in data.uns:
        pairs = data.uns.get("pairs", [(0, 1)])
        sequences = _segments_to_sequences(data.uns["segments"], pairs)
    elif "records" in data.uns:
        sequences = _psmcfa_to_sequences(data.uns["records"])
    else:
        raise ValueError("No input data found. Use smckit.io.read_psmcfa or read_multihetsep.")

    if not sequences:
        raise ValueError("No valid sequences found in input data.")
    return sequences


def _store_esmc2_result(
    data: SmcData,
    *,
    result_obj: Esmc2Result,
    implementation_used: str,
    implementation_requested: str,
    mu: float,
    generation_time: float,
) -> SmcData:
    """Write a normalized eSMC2 result payload back into SmcData."""
    data.results["esmc2"] = annotate_result({
        "Tc": result_obj.Tc,
        "t": result_obj.t,
        "Xi": result_obj.Xi,
        "ne": result_obj.ne,
        "time_years": result_obj.time_years,
        "beta": result_obj.beta,
        "sigma": result_obj.sigma,
        "mu": result_obj.mu,
        "rho": result_obj.rho,
        "rho_per_sequence": result_obj.rho_per_sequence,
        "theta": result_obj.theta,
        "log_likelihood": result_obj.log_likelihood,
        "rounds": result_obj.rounds,
        "backend": implementation_used,
    }, implementation_requested=implementation_requested, implementation_used=implementation_used)
    data.params["mu"] = mu
    data.params["generation_time"] = generation_time
    return data


# ---------------------------------------------------------------------------
# Data preparation: multihetsep segments → per-base pairwise sequences
# ---------------------------------------------------------------------------


def _segments_to_sequences(
    segments: list[dict],
    pairs: list[tuple[int, int]],
) -> list[np.ndarray]:
    """Convert multihetsep segments to per-base observation sequences.

    Each segment has positions, n_called counts, and per-pair observations.
    We expand these into dense per-base sequences of {0=hom, 1=het, 2=missing}.

    Parameters
    ----------
    segments : list of dicts
        From ``smckit.io.read_multihetsep``.
    pairs : list of (int, int)
        Haplotype pairs to extract.

    Returns
    -------
    sequences : list of int8 arrays
        One sequence per (segment, pair) combination.
    """
    sequences = []
    for seg in segments:
        positions = seg["positions"]
        n_called = seg["n_called"]
        obs_dict = seg["obs"]

        # Total length = last position
        L = int(positions[-1]) if len(positions) > 0 else 0
        if L == 0:
            continue

        for pair in pairs:
            if pair not in obs_dict:
                continue
            obs = obs_dict[pair]
            seq = np.full(L, 2, dtype=np.int8)  # default = missing

            for idx in range(len(positions)):
                pos = int(positions[idx])
                nc = int(n_called[idx])
                o = int(obs[idx])

                # The segment covers nc called sites ending at pos
                start = pos - nc
                if start < 0:
                    start = 0

                # Fill called sites as homozygous (0)
                for bp in range(start, pos):
                    if bp < L:
                        seq[bp] = 0

                # The observation at the position itself
                if o == 2:  # heterozygous
                    if pos - 1 < L:
                        seq[pos - 1] = 1
                elif o == 0:  # missing
                    if pos - 1 < L:
                        seq[pos - 1] = 2

            sequences.append(seq)

    return sequences


def _psmcfa_to_sequences(records: list[dict]) -> list[np.ndarray]:
    """Convert PSMC-style records to per-base observation sequences.

    Parameters
    ----------
    records : list of dicts
        From ``smckit.io.read_psmcfa``, each with a ``"codes"`` array.

    Returns
    -------
    sequences : list of int8 arrays
    """
    return [rec["codes"].astype(np.int8) for rec in records]


# ---------------------------------------------------------------------------
# Parameter mapping utilities
# ---------------------------------------------------------------------------


def _build_pop_vector(n: int, pop_vect: np.ndarray | None = None) -> np.ndarray:
    """Build population grouping vector.

    Parameters
    ----------
    n : int
        Number of hidden states.
    pop_vect : array-like, optional
        Group sizes. Must sum to n. Default: groups of 2.

    Returns
    -------
    pop_vect : (n_groups,) int array
    """
    if pop_vect is None:
        n_groups = n // 2
        pop_vect = np.full(n_groups, 2, dtype=np.int32)
        if n % 2 != 0:
            pop_vect = np.append(pop_vect, n - n_groups * 2)
    pop_vect = np.asarray(pop_vect, dtype=np.int32)
    if pop_vect.sum() != n:
        raise ValueError(f"pop_vect must sum to n={n}, got {pop_vect.sum()}")
    return pop_vect


def _xi_from_free_params(xi_free: np.ndarray, pop_vect: np.ndarray, n: int) -> np.ndarray:
    """Expand free Xi parameters to full n-length vector.

    Parameters
    ----------
    xi_free : (n_groups,) float64
        Free population size parameters (one per group).
    pop_vect : (n_groups,) int
        Number of states per group.
    n : int
        Total number of states.

    Returns
    -------
    Xi : (n,) float64
        Population sizes for each state.
    """
    Xi = np.empty(n, dtype=np.float64)
    idx = 0
    for g in range(len(pop_vect)):
        for _ in range(pop_vect[g]):
            Xi[idx] = xi_free[g]
            idx += 1
    return Xi


def _xi_from_unit_params(
    xi_params: np.ndarray, pop_vect: np.ndarray, box_p: tuple[float, float]
) -> np.ndarray:
    """Convert unit Xi parameters (0–1) into group-level Xi scalings."""
    scaled = xi_params * sum(box_p) - box_p[0]
    return 10.0 ** scaled


def _rho_from_unit(unit: float, rho_base: float, box_r: tuple[float, float]) -> float:
    """Convert a [0,1] unit rho parameter into the per-sequence recombination rate."""
    scaled = unit * sum(box_r) - box_r[0]
    return rho_base * 10.0 ** scaled


def _beta_from_unit(unit: float, box_b: tuple[float, float]) -> float:
    """Convert a [0,1] unit beta parameter into the germination rate."""
    val = unit * (box_b[1] - box_b[0]) + box_b[0]
    return val * val


def _sigma_from_unit(unit: float, box_s: tuple[float, float]) -> float:
    """Convert a [0,1] unit sigma parameter into the selfing rate."""
    return unit * (box_s[1] - box_s[0]) + box_s[0]


def _theta_from_beta_sigma(beta: float, sigma: float, mu_b: float, theta_W: float) -> float:
    """Compute theta_W-derived theta for current beta/sigma."""
    return theta_W * (beta**2) * 2.0 / ((2.0 - sigma) * (beta + (1.0 - beta) * mu_b))


def _theta_w_from_sequences(sequences: list[np.ndarray]) -> float:
    """Reproduce upstream theta_W aggregation on dense sequences."""
    theta_terms: list[float] = []
    for seq in sequences:
        seq_arr = np.asarray(seq, dtype=np.int8)
        callable_sites = int(np.sum(seq_arr < 2))
        if callable_sites == 0:
            continue
        het_sites = int(np.sum(seq_arr == 1))
        theta_terms.append(het_sites / (callable_sites / float(len(seq_arr))))

    if not theta_terms:
        raise ValueError("No callable sites in input data.")
    return float(np.mean(theta_terms))


def _public_rho_from_sequence_rho(rho_per_sequence: float, L: int) -> float:
    """Convert internal per-sequence rho to upstream/public per-bp units."""
    if L <= 0:
        raise ValueError("Sequence length must be positive when converting rho.")
    return float(rho_per_sequence) / (2.0 * float(L))


def _sequence_rho_from_public_rho(rho_public: float, L: int) -> float:
    """Convert upstream/public per-bp rho to the internal per-sequence scale."""
    if L <= 0:
        raise ValueError("Sequence length must be positive when converting rho.")
    return float(rho_public) * (2.0 * float(L))


def _stage_order(
    estimate_beta: bool, estimate_sigma: bool, estimate_rho: bool
) -> list[tuple[str, ...]]:
    """Reproduce the R Baum-Welch stage order for the requested parameters."""
    stages: list[tuple[str, ...]] = []

    def _append(stage: tuple[str, ...]) -> None:
        if stage not in stages:
            stages.append(stage)

    if estimate_rho:
        _append(("rho",))
    if estimate_beta:
        _append(("beta",))
    if estimate_sigma:
        _append(("sigma",))

    if estimate_rho and estimate_beta:
        _append(("rho", "beta"))
    if estimate_rho and estimate_sigma:
        _append(("rho", "sigma"))
    if estimate_beta and estimate_sigma:
        _append(("beta", "sigma"))

    if estimate_rho:
        _append(("rho", "Xi"))
    if estimate_beta:
        _append(("beta", "Xi"))
    if estimate_sigma:
        _append(("sigma", "Xi"))
    if estimate_rho and estimate_beta:
        _append(("rho", "beta", "Xi"))
    if estimate_rho and estimate_sigma:
        _append(("rho", "sigma", "Xi"))
    if estimate_beta and estimate_sigma:
        _append(("beta", "sigma", "Xi"))
    if estimate_rho and estimate_beta and estimate_sigma:
        _append(("rho", "beta", "sigma", "Xi"))

    _append(("Xi",))
    return stages


def _xi_penalty(xi_free: np.ndarray, rp: tuple[float, float]) -> float:
    """Reproduce upstream L1/L2 penalties on absolute population scalings."""
    if rp[0] <= 0.0 and rp[1] <= 0.0:
        return 0.0
    xi_rp = 10.0 ** np.abs(np.log10(np.asarray(xi_free, dtype=np.float64)))
    return float(rp[0] * np.sum(xi_rp) + rp[1] * np.sum(xi_rp**2))


def _rho_unit_from_value(rho: float, rho_base: float, box_r: tuple[float, float]) -> float:
    """Map a concrete per-sequence rho back into the upstream unit interval."""
    if rho <= 0.0 or rho_base <= 0.0:
        return 0.5
    return max(0.0, min(1.0, (np.log10(rho / rho_base) + box_r[0]) / sum(box_r)))


def _xi_units_from_values(xi_free: np.ndarray, box_p: tuple[float, float]) -> np.ndarray:
    """Map concrete Xi group scalings back into the upstream unit interval."""
    xi_free = np.asarray(xi_free, dtype=np.float64)
    units = (np.log10(xi_free) + box_p[0]) / sum(box_p)
    return np.clip(units, 0.0, 1.0)


def _baum_welch_transition_objective(Q: np.ndarray, n_counts: np.ndarray) -> float:
    """Evaluate the upstream default BW=FALSE transition objective."""
    q_row = np.asarray(Q, dtype=np.float64).T
    counts = np.asarray(n_counts, dtype=np.float64)
    keep = (q_row > 0.0) & (counts > 0.0)
    if not np.any(keep):
        return 0.0
    return float(-np.sum(np.log(q_row[keep]) * counts[keep]))


def _project_box(
    param: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Project parameters into a box, mirroring BB::spg's default projector."""
    return np.minimum(np.maximum(param, lower), upper)


def _spg_box(
    *,
    initial: np.ndarray,
    objective,
    lower: np.ndarray,
    upper: np.ndarray,
    method: int = 2,
    memory: int = 20,
    maxit: int = 100,
    ftol: float = 1e-10,
    gtol: float = 1e-5,
    maxfeval: int = 10000,
    eps: float = 1e-7,
) -> tuple[np.ndarray, float, bool]:
    """Port BB::spg() for box constraints with finite-difference gradients."""

    def _safe_objective(param: np.ndarray) -> float:
        value = float(objective(np.asarray(param, dtype=np.float64)))
        if not np.isfinite(value):
            return float("inf")
        return value

    def _gradient(param: np.ndarray, f_current: float) -> np.ndarray:
        grad = np.empty_like(param)
        for idx in range(len(param)):
            shifted = param.copy()
            shifted[idx] += eps
            grad[idx] = (_safe_objective(shifted) - f_current) / eps
        return grad

    def _nmls(
        param: np.ndarray,
        f_current: float,
        direction: np.ndarray,
        gtd: float,
        last_values: np.ndarray,
        feval: int,
    ) -> tuple[np.ndarray | None, float | None, int, int]:
        gamma = 1e-4
        fmax = float(np.max(last_values))
        alpha = 1.0
        while True:
            candidate = param + alpha * direction
            f_candidate = _safe_objective(candidate)
            feval += 1
            if f_candidate <= fmax + gamma * alpha * gtd:
                return candidate, f_candidate, feval, 0
            if alpha <= 0.1:
                alpha /= 2.0
            else:
                denom = 2.0 * (f_candidate - f_current - alpha * gtd)
                if denom == 0.0:
                    alpha /= 2.0
                else:
                    atemp = -(gtd * alpha * alpha) / denom
                    if atemp < 0.1 or atemp > 0.9 * alpha:
                        alpha /= 2.0
                    else:
                        alpha = atemp
            if feval > maxfeval:
                return None, None, feval, 2

    param = _project_box(np.asarray(initial, dtype=np.float64), lower, upper)
    f_value = _safe_objective(param)
    feval = 1
    if not np.isfinite(f_value):
        return param.copy(), float("inf"), False
    grad = _gradient(param, f_value)

    lmin = 1e-30
    lmax = 1e30
    projected_grad = _project_box(param - grad, lower, upper) - param
    pginf = float(np.max(np.abs(projected_grad)))
    lam = lmax if pginf == 0.0 else min(lmax, max(lmin, 1.0 / pginf))

    last_values = np.full(int(memory), -1e99, dtype=np.float64)
    last_values[0] = f_value
    best_param = param.copy()
    best_value = f_value
    f_change = float("inf")
    iteration = 0
    lsflag = 0

    while pginf > gtol and iteration <= maxit and f_change > ftol:
        iteration += 1
        direction = _project_box(param - lam * grad, lower, upper) - param
        gtd = float(np.sum(grad * direction))
        candidate, f_candidate, feval, lsflag = _nmls(
            param,
            f_value,
            direction,
            gtd,
            last_values,
            feval,
        )
        if lsflag != 0 or candidate is None or f_candidate is None:
            break
        f_change = abs(f_value - f_candidate)
        f_value = f_candidate
        last_values[iteration % int(memory)] = f_value
        grad_new = _gradient(candidate, f_value)
        s = candidate - param
        y = grad_new - grad
        sts = float(np.sum(s * s))
        yty = float(np.sum(y * y))
        sty = float(np.sum(s * y))

        if method == 1:
            lam = lmax if (sts == 0.0 or sty < 0.0) else min(lmax, max(lmin, sts / sty))
        elif method == 2:
            lam = lmax if (sty < 0.0 or yty == 0.0) else min(lmax, max(lmin, sty / yty))
        else:
            lam = (
                lmax
                if (sts == 0.0 or yty == 0.0)
                else min(lmax, max(lmin, np.sqrt(sts / yty)))
            )

        param = candidate
        grad = grad_new
        projected_grad = _project_box(param - grad, lower, upper) - param
        pginf = float(np.max(np.abs(projected_grad)))
    if f_value < best_value:
        best_value = f_value
        best_param = param.copy()

    success = lsflag == 0 and (pginf <= gtol or f_change <= ftol or iteration >= maxit)
    if not success and np.isfinite(best_value):
        return best_param, best_value, False
    return best_param, best_value, True


def _optimize_baum_welch(
    *,
    stats: dict,
    n: int,
    L: int,
    mu: float,
    mu_b: float,
    beta: float,
    sigma: float,
    rho: float,
    xi_free: np.ndarray,
    pop_vect: np.ndarray,
    box_b: tuple[float, float],
    box_s: tuple[float, float],
    box_r: tuple[float, float],
    box_p: tuple[float, float],
    rp: tuple[float, float],
    rho_base: float,
    rho_penalty: float,
    estimate_beta: bool,
    estimate_sigma: bool,
    estimate_rho: bool,
    estimate_pop: bool = True,
) -> tuple[float, float, float, np.ndarray, float]:
    """Optimize one upstream-style M-step with fixed sufficient statistics."""
    n_groups = len(pop_vect)
    units: list[str] = []
    if estimate_rho:
        units.append("rho")
    if estimate_beta:
        units.append("beta")
    if estimate_sigma:
        units.append("sigma")
    if estimate_pop:
        units.extend(["Xi"] * n_groups)

    bounds = [(0.0, 1.0)] * len(units)
    transition_counts = np.asarray(stats["N"], dtype=np.float64)

    def _unpack(unit_values: np.ndarray) -> tuple[float, float, float, np.ndarray]:
        idx = 0
        rho_val = rho
        beta_val = beta
        sigma_val = sigma
        xi_val = xi_free.copy()

        if estimate_rho:
            rho_val = _rho_from_unit(float(unit_values[idx]), rho_base, box_r)
            idx += 1
        if estimate_beta:
            beta_val = _beta_from_unit(float(unit_values[idx]), box_b)
            idx += 1
        if estimate_sigma:
            sigma_val = _sigma_from_unit(float(unit_values[idx]), box_s)
            idx += 1

        if estimate_pop:
            xi_units = np.asarray(unit_values[idx : idx + n_groups], dtype=np.float64)
            xi_val = _xi_from_unit_params(xi_units, pop_vect, box_p)
        return beta_val, sigma_val, rho_val, xi_val

    def _objective(unit_values: np.ndarray) -> float:
        beta_val, sigma_val, rho_val, xi_val = _unpack(unit_values)
        Xi = _xi_from_free_params(xi_val, pop_vect, n)
        Q, q, t, Tc, e = esmc2_build_hmm(n, Xi, beta_val, sigma_val, rho_val, mu, mu_b, L)
        objective = _baum_welch_transition_objective(Q, transition_counts)
        objective += _xi_penalty(xi_val, rp)
        if rho_penalty > 0.0 and rho_base > 0.0 and rho_val > 0.0:
            objective += rho_penalty * (np.log10(rho_val / rho_base)) ** 2
        return float(objective)

    initial: list[float] = []
    if estimate_rho:
        initial.append(_rho_unit_from_value(rho, rho_base, box_r))
    if estimate_beta:
        initial.append(_to_unit(np.sqrt(beta), np.sqrt(box_b[0]), np.sqrt(box_b[1])))
    if estimate_sigma:
        initial.append(_to_unit(sigma, box_s[0], box_s[1]))
    if estimate_pop:
        initial.extend(_xi_units_from_values(xi_free, box_p).tolist())
    initial_array = np.clip(np.asarray(initial, dtype=np.float64), 0.0, 1.0)
    lower = np.zeros_like(initial_array)
    upper = np.ones_like(initial_array)
    best_x, best_fun, success = _spg_box(
        initial=initial_array,
        objective=_objective,
        lower=lower,
        upper=upper,
        method=2,
        memory=20,
        maxit=15 + len(initial_array),
    )
    if not success:
        polish = minimize(
            _objective,
            np.clip(best_x, 0.0, 1.0),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 50, "ftol": 1e-12},
        )
        if polish.success and float(polish.fun) < float(best_fun):
            best_x = np.asarray(polish.x, dtype=np.float64)
            best_fun = float(polish.fun)
    best_x = np.clip(np.asarray(best_x, dtype=np.float64), 0.0, 1.0)
    best_beta, best_sigma, best_rho, best_xi = _unpack(best_x)
    return best_beta, best_sigma, best_rho, best_xi, float(best_fun)


def _optimize_stage(
    *,
    sequences: list[np.ndarray],
    n: int,
    L: int,
    mu: float,
    mu_b: float,
    beta: float,
    sigma: float,
    rho: float,
    theta: float,
    xi_free: np.ndarray,
    pop_vect: np.ndarray,
    box_b: tuple[float, float],
    box_s: tuple[float, float],
    box_r: tuple[float, float],
    box_p: tuple[float, float],
    rp: tuple[float, float],
    rho_base: float,
    rho_penalty: float,
    stage: tuple[str, ...],
) -> tuple[float, float, float, np.ndarray, float]:
    """Optimize a single subset of parameters (rho/beta/sigma/Xi)."""

    n_groups = len(pop_vect)
    units = []
    if "rho" in stage:
        units.append("rho")
    if "beta" in stage:
        units.append("beta")
    if "sigma" in stage:
        units.append("sigma")
    if "Xi" in stage:
        units.extend(["Xi"] * n_groups)

    bounds = [(0.001, 0.999)] * len(units)

    def _objective(unit_values: np.ndarray) -> float:
        idx = 0
        rho_val = rho
        beta_val = beta
        sigma_val = sigma
        xi_val = xi_free.copy()

        if "rho" in stage:
            rho_val = _rho_from_unit(unit_values[idx], rho_base, box_r)
            idx += 1
        if "beta" in stage:
            beta_val = _beta_from_unit(unit_values[idx], box_b)
            idx += 1
        if "sigma" in stage:
            sigma_val = _sigma_from_unit(unit_values[idx], box_s)
            idx += 1
        if "Xi" in stage:
            xi_units = unit_values[idx : idx + n_groups]
            xi_val = _xi_from_unit_params(xi_units, pop_vect, box_p)

        Xi = _xi_from_free_params(xi_val, pop_vect, n)
        _, total_ll, _, _ = _expectation_step(
            sequences=sequences,
            n=n,
            Xi=Xi,
            beta=beta_val,
            sigma=sigma_val,
            rho=rho_val,
            mu=mu,
            mu_b=mu_b,
            L=L,
        )
        penalty = _xi_penalty(xi_val, rp)
        if rho_penalty > 0.0 and rho_base > 0.0 and rho_val > 0.0:
            ratio = rho_val / rho_base
            if ratio > 0.0:
                penalty += rho_penalty * (np.log10(ratio)) ** 2
        return -(total_ll - penalty)

    if not bounds:
        return beta, sigma, rho, xi_free, 0.0

    initial = []
    if "rho" in stage:
        # convert current rho to unit by reversing _rho_from_unit (approx)
        log_rho = np.log10(rho / rho_base) if rho_base > 0 else 0.0
        initial.append((log_rho + box_r[1]) / sum(box_r))
    if "beta" in stage:
        initial.append(_from_unit(np.sqrt(beta), np.sqrt(box_b[0]), np.sqrt(box_b[1])))
    if "sigma" in stage:
        initial.append(_to_unit(sigma, box_s[0], box_s[1]))
    if "Xi" in stage:
        xi_unit = []
        for val in xi_free:
            log_val = np.log10(val)
            xi_unit.append(_to_unit(log_val + box_p[0], 0.0, sum(box_p)))
        initial.extend(xi_unit)

    result = minimize(
        _objective,
        np.array(initial, dtype=np.float64),
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 25, "ftol": 1e-8},
    )

    log_likelihood = -result.fun
    updated_beta = beta
    updated_sigma = sigma
    updated_rho = rho
    updated_xi_free = xi_free.copy()
    idx = 0

    if "rho" in stage:
        updated_rho = _rho_from_unit(result.x[idx], rho_base, box_r)
        idx += 1
    if "beta" in stage:
        updated_beta = _beta_from_unit(result.x[idx], box_b)
        idx += 1
    if "sigma" in stage:
        updated_sigma = _sigma_from_unit(result.x[idx], box_s)
        idx += 1
    if "Xi" in stage:
        xi_units = result.x[idx : idx + len(pop_vect)]
        updated_xi_free = _xi_from_unit_params(xi_units, pop_vect, box_p)

    return updated_beta, updated_sigma, updated_rho, updated_xi_free, log_likelihood


def _expectation_step(
    sequences: list[np.ndarray],
    n: int,
    Xi: np.ndarray,
    beta: float,
    sigma: float,
    rho: float,
    mu: float,
    mu_b: float,
    L: int,
) -> tuple[dict, float, np.ndarray, np.ndarray]:
    """Run the upstream-equivalent zipped E-step for a parameter set."""
    Q, q, t, Tc, e = esmc2_build_hmm(n, Xi, beta, sigma, rho, mu, mu_b, L)
    zip_cache = _build_zip_cache(Q, e)
    accum_N = np.zeros((n, n), dtype=np.float64)
    accum_M = np.zeros((3, n), dtype=np.float64)
    accum_q_post = np.zeros(n, dtype=np.float64)
    total_ll = 0.0
    for seq in sequences:
        zipped = _zip_sequence_to_upstream_numeric(seq)
        alpha, scales, ll = _zip_forward(
            zipped,
            np.asarray(zip_cache["g"], dtype=np.float64),
            q,
            zip_cache["C"],
        )
        beta_post = _zip_backward(zipped, zip_cache["TO"], scales)
        N_counts, M_counts, q_post = _zip_expected_counts(
            symbols=zipped,
            alpha=alpha,
            beta=beta_post,
            scales=scales,
            Q=Q,
            zip_cache=zip_cache,
        )
        accum_N += N_counts
        accum_M += M_counts
        accum_q_post += q_post
        total_ll += ll

    accum_N *= np.asarray(Q, dtype=np.float64).T

    n_total = float(np.sum(accum_N))
    if n_total > 0.0:
        accum_N *= (L - 1) / n_total
    m_total = float(np.sum(accum_M))
    if m_total > 0.0:
        accum_M *= L / m_total
    q_total = float(np.sum(accum_q_post))
    if q_total > 0.0:
        accum_q_post /= q_total

    stats = {
        "Q": Q,
        "q": q,
        "t": t,
        "Tc": Tc,
        "e": e,
        "N": np.maximum(accum_N, 0.0),
        "M": np.maximum(accum_M, 0.0),
        "q_post": np.maximum(accum_q_post, 0.0),
    }
    return (
        stats,
        total_ll,
        t,
        Tc,
    )


# ---------------------------------------------------------------------------
# Bounded parameter transform utilities
# ---------------------------------------------------------------------------


def _to_unit(val: float, lo: float, hi: float) -> float:
    """Map value from [lo, hi] to [0, 1]."""
    if hi <= lo:
        return 0.5
    return (val - lo) / (hi - lo)


def _from_unit(u: float, lo: float, hi: float) -> float:
    """Map value from [0, 1] to [lo, hi]."""
    u = max(0.0, min(1.0, u))
    return u * (hi - lo) + lo


def _rho_transform(u: float, rho_base: float, box_r: tuple[float, float]) -> float:
    """Transform unit parameter to recombination rate."""
    log_scale = u * sum(box_r) - box_r[0]
    return rho_base * 10.0**log_scale


def _xi_transform(u: float, box_p: tuple[float, float]) -> float:
    """Transform unit parameter to population size scaling."""
    log_scale = u * sum(box_p) - box_p[0]
    return 10.0**log_scale


# ---------------------------------------------------------------------------
# Negative log-likelihood for optimization
# ---------------------------------------------------------------------------


def _neg_loglik(
    param: np.ndarray,
    sequences: list[np.ndarray],
    n: int,
    mu: float,
    mu_b: float,
    L: int,
    rho_base: float,
    beta_fixed: float | None,
    sigma_fixed: float | None,
    pop_vect: np.ndarray,
    box_r: tuple[float, float],
    box_p: tuple[float, float],
    box_b: tuple[float, float],
    box_s: tuple[float, float],
    estimate_rho: bool,
    estimate_beta: bool,
    estimate_sigma: bool,
    estimate_pop: bool,
    rp: tuple[float, float],
) -> float:
    """Compute negative log-likelihood for parameter optimization.

    Parameters are packed into a single vector in order:
    [rho?, sigma?, beta?, Xi_0, ..., Xi_{n_groups-1}]
    """
    idx = 0
    n_groups = len(pop_vect)

    # Unpack recombination rate
    if estimate_rho:
        rho = _rho_transform(param[idx], rho_base, box_r)
        idx += 1
    else:
        rho = rho_base

    # Unpack selfing rate
    if estimate_sigma:
        sigma = _from_unit(param[idx], box_s[0], box_s[1])
        idx += 1
    else:
        sigma = sigma_fixed if sigma_fixed is not None else 0.0

    # Unpack germination rate
    if estimate_beta:
        sqrt_beta = _from_unit(param[idx], np.sqrt(box_b[0]), np.sqrt(box_b[1]))
        beta = sqrt_beta**2
        idx += 1
    else:
        beta = beta_fixed if beta_fixed is not None else 1.0

    # Unpack population sizes
    if estimate_pop:
        xi_free = np.empty(n_groups, dtype=np.float64)
        for g in range(n_groups):
            xi_free[g] = _xi_transform(param[idx], box_p)
            idx += 1
        Xi = _xi_from_free_params(xi_free, pop_vect, n)
    else:
        Xi = np.ones(n, dtype=np.float64)

    # Build HMM
    Q, q, t, Tc, e = esmc2_build_hmm(n, Xi, beta, sigma, rho, mu, mu_b, L)

    # Check for invalid matrices
    if np.any(np.isnan(Q)) or np.any(np.isnan(e)):
        return 1e20

    # Compute total log-likelihood
    total_ll = 0.0
    for seq in sequences:
        ll = esmc2_forward_loglik(Q, e, q, seq)
        total_ll += ll

    # Regularization penalty on Xi
    if estimate_pop and (rp[0] > 0 or rp[1] > 0):
        penalty = 0.0
        for g in range(n_groups):
            penalty += rp[0] * abs(np.log10(xi_free[g]))  # L1
            penalty += rp[1] * np.log10(xi_free[g]) ** 2  # L2
        total_ll -= penalty

    return -total_ll


# ---------------------------------------------------------------------------
# eSMC2 result
# ---------------------------------------------------------------------------


@dataclass
class Esmc2Result:
    """Results from an eSMC2 run."""

    Tc: np.ndarray  # (n,) time boundaries
    t: np.ndarray  # (n,) expected coalescent times
    Xi: np.ndarray  # (n,) relative population sizes
    ne: np.ndarray  # (n,) Ne(t) in absolute units
    time_years: np.ndarray  # (n,) time in years
    beta: float = 1.0  # estimated germination rate
    sigma: float = 0.0  # estimated selfing rate
    mu: float = 0.0  # mutation rate per bp per 2Ne
    rho: float = 0.0  # recombination rate per bp per 2Ne (public/oracle units)
    rho_per_sequence: float = 0.0  # recombination rate on the internal HMM scale
    theta: float = 0.0  # population-scaled mutation rate
    log_likelihood: float = 0.0
    n_iterations: int = 0
    rounds: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _esmc2_native(
    data: SmcData,
    n_states: int = 20,
    rho_over_theta: float = 1.0,
    n_iterations: int = 20,
    estimate_beta: bool = False,
    estimate_sigma: bool = False,
    estimate_rho: bool = True,
    beta: float = 1.0,
    sigma: float = 0.0,
    mu_b: float = 1.0,
    mu: float = 1.25e-8,
    generation_time: float = 1.0,
    pop_vect: np.ndarray | list[int] | None = None,
    box_b: tuple[float, float] = (0.05, 1.0),
    box_s: tuple[float, float] = (0.0, 0.99),
    box_p: tuple[float, float] = (2.0, 2.0),
    box_r: tuple[float, float] = (0.0, 3.0),
    rp: tuple[float, float] = (0.0, 0.0),
    rho_penalty: float = 0.0,
    implementation_requested: str = "native",
) -> SmcData:
    """Run eSMC2 demographic inference with dormancy and selfing.

    Parameters
    ----------
    data : SmcData
        Input data from ``smckit.io.read_psmcfa()`` or
        ``smckit.io.read_multihetsep()``.
    n_states : int
        Number of hidden states (time intervals).
    rho_over_theta : float
        Initial ratio of recombination over mutation rate.
    n_iterations : int
        Maximum number of EM iterations.
    estimate_beta : bool
        If True, estimate the germination rate (dormancy).
    estimate_sigma : bool
        If True, estimate the self-fertilization rate.
    estimate_rho : bool
        If True, estimate the recombination rate.
    beta : float
        Initial/fixed germination rate (0 < β ≤ 1). β = 1 means no dormancy.
    sigma : float
        Initial/fixed self-fertilization rate (0 ≤ σ < 1).
    mu_b : float
        Ratio of mutation rate in seed bank over active mutation rate.
    mu : float
        Per-base per-generation mutation rate.
    generation_time : float
        Generation time in years.
    pop_vect : array-like, optional
        Population grouping vector. Default: groups of 2 states.
    box_b : (float, float)
        Bounds for germination rate.
    box_s : (float, float)
        Bounds for selfing rate.
    box_p : (float, float)
        Log10 bounds for population size variation.
    box_r : (float, float)
        Log10 bounds for recombination rate variation.
    rp : (float, float)
        L1 and L2 regularization penalties for population size.
    rho_penalty : float
        Weight applied to ``log10(rho / rho_base)**2`` during rho estimation.

    Returns
    -------
    SmcData
        Input data with results in ``data.results["esmc2"]``.
    """
    n = n_states

    # --- Prepare sequences ---
    sequences = _sequences_from_smcdata(data)

    # --- Compute initial theta from data ---
    theta_W = _theta_w_from_sequences(sequences)
    # Per-sequence Watterson's theta
    L_total = max(len(seq) for seq in sequences)

    # theta_W corrected for β and σ: θ = θ_W * β² * 2 / ((2-σ) * (β + (1-β)*μ_b))
    theta = theta_W * (beta**2) * 2.0 / ((2.0 - sigma) * (beta + (1.0 - beta) * mu_b))
    rho_scaled = rho_over_theta * theta  # recombination rate per sequence

    logger.info(
        "eSMC2: n=%d, L=%d, theta=%.4f, rho=%.4f, beta=%.4f, sigma=%.4f",
        n,
        L_total,
        theta,
        rho_scaled,
        beta,
        sigma,
    )

    # --- Population grouping ---
    pv = _build_pop_vector(n, np.array(pop_vect, dtype=np.int32) if pop_vect is not None else None)
    n_groups = len(pv)

    # --- EM loop ---
    rounds: list[dict] = []
    current_beta = beta
    current_sigma = sigma
    current_rho = rho_scaled
    current_xi_free = np.ones(n_groups, dtype=np.float64)
    gamma = float(rho_over_theta)
    max_bits = 1
    bit = 0
    theta_run = _theta_from_beta_sigma(current_beta, current_sigma, mu_b, theta_W)
    mu_run = theta_run / (2.0 * L_total)

    while bit < max_bits:
        bit += 1
        inner_maxit = int(n_iterations)
        theta_run = _theta_from_beta_sigma(current_beta, current_sigma, mu_b, theta_W)
        mu_run = theta_run / (2.0 * L_total)
        rho_base = gamma * theta_run if estimate_rho else current_rho

        # Upstream restarts rho/Xi from the new prior when redo_R triggers.
        current_rho = rho_base
        current_xi_free = np.ones(n_groups, dtype=np.float64)

        previous_beta = None
        previous_sigma = None
        previous_gamma = None
        old_likelihood = None
        saved_state: tuple[float, float, float, np.ndarray] | None = None
        inner_it = 0

        while inner_it < inner_maxit:
            inner_it += 1
            logger.info("eSMC2 outer %d inner %d", bit, inner_it)

            gamma_current = current_rho / theta_run if theta_run > 0.0 else 0.0
            if inner_it > 1:
                diff_o = max(
                    abs(float(previous_sigma) - current_sigma),
                    abs(float(previous_beta) - current_beta),
                    abs(float(previous_gamma) - gamma_current),
                )
                if diff_o >= 0.01 and inner_it == inner_maxit:
                    inner_maxit += 1

            previous_beta = current_beta
            previous_sigma = current_sigma
            previous_gamma = gamma_current

            Xi_current = _xi_from_free_params(current_xi_free, pv, n)
            stats, current_ll, _, _ = _expectation_step(
                sequences=sequences,
                n=n,
                Xi=Xi_current,
                beta=current_beta,
                sigma=current_sigma,
                rho=current_rho,
                mu=mu_run,
                mu_b=mu_b,
                L=L_total,
            )

            if inner_it > 1 and (
                (old_likelihood is not None and old_likelihood > current_ll)
                or not np.isfinite(current_ll)
            ):
                if inner_it > 2 and saved_state is not None:
                    current_beta = saved_state[0]
                    current_sigma = saved_state[1]
                    current_rho = saved_state[2]
                    current_xi_free = saved_state[3].copy()
                break

            old_likelihood = current_ll
            saved_state = (
                current_beta,
                current_sigma,
                current_rho,
                current_xi_free.copy(),
            )

            for stage in _stage_order(estimate_beta, estimate_sigma, estimate_rho):
                (
                    current_beta,
                    current_sigma,
                    current_rho,
                    current_xi_free,
                    _,
                ) = _optimize_baum_welch(
                    stats=stats,
                    n=n,
                    L=L_total,
                    mu=mu_run,
                    mu_b=mu_b,
                    beta=current_beta,
                    sigma=current_sigma,
                    rho=current_rho,
                    xi_free=current_xi_free,
                    pop_vect=pv,
                    box_b=box_b,
                    box_s=box_s,
                    box_r=box_r,
                    box_p=box_p,
                    rp=rp,
                    rho_base=rho_base,
                    rho_penalty=rho_penalty,
                    estimate_beta=("beta" in stage),
                    estimate_sigma=("sigma" in stage),
                    estimate_rho=("rho" in stage),
                    estimate_pop=("Xi" in stage),
                )

            Xi_round = _xi_from_free_params(current_xi_free, pv, n)
            _, ll_round, t_round, Tc_round = _expectation_step(
                sequences=sequences,
                n=n,
                Xi=Xi_round,
                beta=current_beta,
                sigma=current_sigma,
                rho=current_rho,
                mu=mu_run,
                mu_b=mu_b,
                L=L_total,
            )

            rounds.append({
                "round": len(rounds) + 1,
                "outer_round": bit,
                "inner_round": inner_it,
                "log_likelihood": ll_round,
                "beta": current_beta,
                "sigma": current_sigma,
                "rho": _public_rho_from_sequence_rho(current_rho, L_total),
                "rho_per_sequence": current_rho,
                "mu": mu_run,
                "theta": mu_run * 2.0 * L_total,
                "Xi": Xi_round.copy(),
                "Tc": Tc_round.copy(),
                "t": t_round.copy(),
            })

            logger.info(
                "  LL=%.2f beta=%.4f sigma=%.4f rho=%.4f",
                ll_round,
                current_beta,
                current_sigma,
                current_rho,
            )

        gamma_next = current_rho / theta_run if theta_run > 0.0 else 0.0
        if estimate_rho and (gamma_next <= 0.5 * gamma or gamma_next >= 2.0 * gamma):
            max_bits += 1
            gamma = gamma_next

    # --- Final results ---
    theta_final = mu_run * 2.0 * L_total
    mu_final = mu_run

    Xi_final = _xi_from_free_params(current_xi_free, pv, n)
    _, _, t_final, Tc_final, _ = esmc2_build_hmm(
        n,
        Xi_final,
        current_beta,
        current_sigma,
        current_rho,
        mu_final,
        mu_b,
        L_total,
    )

    # Ne = Xi * Ne0, where Ne0 = theta / (4 * mu_per_gen)
    # theta = mu_scaled * 2 * L = 4 * Ne0 * mu_per_gen * L / L (per bp)
    # Actually: Ne0 = theta_final / (4 * mu) where mu is per-base per-gen
    ne0 = theta_final / (4.0 * mu)
    ne = Xi_final * ne0

    # Time in years: t_final is in units of 2*Ne0 generations
    # (from the coalescent scaling: time boundaries are in 2Ne units)
    time_years = t_final * 2.0 * ne0 * generation_time
    rho_public = _public_rho_from_sequence_rho(current_rho, L_total)

    result_obj = Esmc2Result(
        Tc=Tc_final,
        t=t_final,
        Xi=Xi_final,
        ne=ne,
        time_years=time_years,
        beta=current_beta,
        sigma=current_sigma,
        mu=mu_final,
        rho=rho_public,
        rho_per_sequence=current_rho,
        theta=theta_final,
        log_likelihood=float(
            _expectation_step(
                sequences=sequences,
                n=n,
                Xi=Xi_final,
                beta=current_beta,
                sigma=current_sigma,
                rho=current_rho,
                mu=mu_final,
                mu_b=mu_b,
                L=L_total,
            )[1]
        ),
        n_iterations=n_iterations,
        rounds=rounds,
    )

    return _store_esmc2_result(
        data,
        result_obj=result_obj,
        implementation_used="native",
        implementation_requested=implementation_requested,
        mu=mu,
        generation_time=generation_time,
    )


def _esmc2_upstream(
    data: SmcData,
    n_states: int = 20,
    rho_over_theta: float = 1.0,
    n_iterations: int = 20,
    estimate_beta: bool = False,
    estimate_sigma: bool = False,
    estimate_rho: bool = True,
    beta: float = 1.0,
    sigma: float = 0.0,
    mu_b: float = 1.0,
    mu: float = 1.25e-8,
    generation_time: float = 1.0,
    pop_vect: np.ndarray | list[int] | None = None,
    box_b: tuple[float, float] = (0.05, 1.0),
    box_s: tuple[float, float] = (0.0, 0.99),
    box_p: tuple[float, float] = (2.0, 2.0),
    box_r: tuple[float, float] = (0.0, 3.0),
    rp: tuple[float, float] = (0.0, 0.0),
    rho_penalty: float = 0.0,
    implementation_requested: str = "upstream",
    upstream_options: dict | None = None,
) -> SmcData:
    """Run the upstream eSMC2 backend and map results into SmcData."""
    capture_sufficient_statistics = bool(
        upstream_options.get("capture_sufficient_statistics", False) if upstream_options else False
    )
    capture_hmm_builder = bool(
        upstream_options.get("capture_hmm_builder", False) if upstream_options else False
    )
    capture_final_sufficient_statistics = bool(
        upstream_options.get("capture_final_sufficient_statistics", False)
        if upstream_options
        else False
    )
    effective_args = {
        "n_states": int(n_states),
        "rho_over_theta": float(rho_over_theta),
        "n_iterations": int(n_iterations),
        "estimate_beta": bool(estimate_beta),
        "estimate_sigma": bool(estimate_sigma),
        "estimate_rho": bool(estimate_rho),
        "beta": float(beta),
        "sigma": float(sigma),
        "mu_b": float(mu_b),
        "mu": float(mu),
        "generation_time": float(generation_time),
        "pop_vect": None if pop_vect is None else np.asarray(pop_vect, dtype=np.int32).tolist(),
        "box_b": tuple(float(x) for x in box_b),
        "box_s": tuple(float(x) for x in box_s),
        "box_p": tuple(float(x) for x in box_p),
        "box_r": tuple(float(x) for x in box_r),
        "rp": tuple(float(x) for x in rp),
    }
    if upstream_options:
        effective_args.update(upstream_options)
    payload = _run_upstream_esmc2(
        data=data,
        n_states=n_states,
        rho_over_theta=rho_over_theta,
        n_iterations=n_iterations,
        estimate_beta=estimate_beta,
        estimate_sigma=estimate_sigma,
        estimate_rho=estimate_rho,
        beta=beta,
        sigma=sigma,
        mu_b=mu_b,
        mu=mu,
        generation_time=generation_time,
        pop_vect=None if pop_vect is None else np.asarray(pop_vect, dtype=np.int32).tolist(),
        box_b=box_b,
        box_s=box_s,
        box_p=box_p,
        box_r=box_r,
        rp=rp,
        capture_sufficient_statistics=capture_sufficient_statistics,
        capture_hmm_builder=capture_hmm_builder,
        capture_final_sufficient_statistics=capture_final_sufficient_statistics,
    )
    result_obj = Esmc2Result(
        Tc=np.asarray(payload["Tc"], dtype=np.float64),
        t=np.asarray(payload["t"], dtype=np.float64),
        Xi=np.asarray(payload["Xi"], dtype=np.float64),
        ne=np.asarray(payload["ne"], dtype=np.float64),
        time_years=np.asarray(payload["time_years"], dtype=np.float64),
        beta=float(payload["beta"]),
        sigma=float(payload["sigma"]),
        mu=float(payload["mu"]),
        rho=float(payload["rho"]),
        rho_per_sequence=float(payload["rho_per_sequence"]),
        theta=float(payload["theta"]),
        log_likelihood=float(payload["log_likelihood"]),
        n_iterations=int(payload.get("n_iterations", n_iterations)),
        rounds=list(payload.get("rounds", [])),
    )
    stored = _store_esmc2_result(
        data,
        result_obj=result_obj,
        implementation_used="upstream",
        implementation_requested=implementation_requested,
        mu=mu,
        generation_time=generation_time,
    )
    stored.results["esmc2"]["upstream"] = standard_upstream_metadata(
        "esmc2",
        effective_args=effective_args,
        extra=dict(payload.get("upstream", {})),
    )
    if "sufficient_statistics" in payload:
        stored.results["esmc2"]["upstream"]["sufficient_statistics"] = payload[
            "sufficient_statistics"
        ]
    if "hmm_builder" in payload:
        stored.results["esmc2"]["upstream"]["hmm_builder"] = payload["hmm_builder"]
    if "final_sufficient_statistics" in payload:
        stored.results["esmc2"]["upstream"]["final_sufficient_statistics"] = payload[
            "final_sufficient_statistics"
        ]
    return stored


def esmc2(
    data: SmcData,
    n_states: int = 20,
    rho_over_theta: float = 1.0,
    n_iterations: int = 20,
    estimate_beta: bool = False,
    estimate_sigma: bool = False,
    estimate_rho: bool = True,
    beta: float = 1.0,
    sigma: float = 0.0,
    mu_b: float = 1.0,
    mu: float = 1.25e-8,
    generation_time: float = 1.0,
    pop_vect: np.ndarray | list[int] | None = None,
    box_b: tuple[float, float] = (0.05, 1.0),
    box_s: tuple[float, float] = (0.0, 0.99),
    box_p: tuple[float, float] = (2.0, 2.0),
    box_r: tuple[float, float] = (0.0, 3.0),
    rp: tuple[float, float] = (0.0, 0.0),
    rho_penalty: float = 0.0,
    implementation: str = "auto",
    backend: str | None = None,
    upstream_options: dict | None = None,
    native_options: dict | None = None,
) -> SmcData:
    """Run eSMC2 demographic inference.

    Parameters are the same as the native implementation. ``implementation``
    may be ``"native"``, ``"upstream"``, or ``"auto"``. ``"auto"`` currently
    prefers upstream when an executable R environment is available. ``backend``
    is a deprecated compatibility alias.
    """
    implementation = normalize_implementation(
        implementation,
        backend=backend,
    )
    implementation_used = choose_implementation(
        implementation,
        upstream_available=method_upstream_available("esmc2"),
    )

    if implementation_used == "upstream":
        return _esmc2_upstream(
            data,
            n_states=n_states,
            rho_over_theta=rho_over_theta,
            n_iterations=n_iterations,
            estimate_beta=estimate_beta,
            estimate_sigma=estimate_sigma,
            estimate_rho=estimate_rho,
            beta=beta,
            sigma=sigma,
            mu_b=mu_b,
            mu=mu,
            generation_time=generation_time,
            pop_vect=pop_vect,
            box_b=box_b,
            box_s=box_s,
            box_p=box_p,
            box_r=box_r,
            rp=rp,
            rho_penalty=rho_penalty,
            implementation_requested=implementation,
            upstream_options=upstream_options,
        )

    if native_options:
        unsupported = ", ".join(sorted(native_options))
        raise TypeError(f"Unsupported esmc2 native_options keys: {unsupported}")

    return _esmc2_native(
        data,
        n_states=n_states,
        rho_over_theta=rho_over_theta,
        n_iterations=n_iterations,
        estimate_beta=estimate_beta,
        estimate_sigma=estimate_sigma,
        estimate_rho=estimate_rho,
        beta=beta,
        sigma=sigma,
        mu_b=mu_b,
        mu=mu,
        generation_time=generation_time,
        pop_vect=pop_vect,
        box_b=box_b,
        box_s=box_s,
        box_p=box_p,
        box_r=box_r,
        rp=rp,
        rho_penalty=rho_penalty,
        implementation_requested=implementation,
    )
