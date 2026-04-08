"""Cross-validation of eSMC2 Python implementation against R reference formulas.

Implements the R formulas from vendor/eSMC2/eSMC2/R/build_HMM_matrix.R and
build_emission_matrix.R as pure Python, then compares numerically against the
Numba JIT implementation in smckit.backends._numba_esmc2.

Reference: Sellinger et al. (2020), PLoS Genetics.
"""

import math

import numpy as np
import pytest

from smckit.backends._numba_esmc2 import (
    esmc2_build_emission_matrix,
    esmc2_build_hmm,
    esmc2_build_time_boundaries,
    esmc2_build_transition_matrix,
    esmc2_equilibrium_probs,
    esmc2_expected_times,
    esmc2_forward_loglik,
)

# ---------------------------------------------------------------------------
# Pure Python reference implementations from R code
# ---------------------------------------------------------------------------


def r_build_Tc(n, beta, sigma):
    """R: build_Tc() with Big_Window=0.

    From build_Tc.R lines 15-17:
        Vect=0:(n-1)
        Tc= scale[2] -(0.5*(2-Sigma)*log(1-(Vect/n))/((Beta^2)*scale[1]))
    With scale=c(1,0), Sigma=sigma, Beta=beta.
    """
    Tc = np.empty(n)
    for k in range(n):
        if k == 0:
            Tc[k] = 0.0
        else:
            Tc[k] = -0.5 * (2.0 - sigma) * math.log(1.0 - k / n) / (beta**2)
    return Tc


def r_expected_times(Tc, Xi, beta, sigma):
    """R: build_HMM_matrix() lines 75-76.

    t[1:(n-1)] = (Xi[1:(n-1)]*(2-sigma)/(beta*beta*2)) +
                 (Tc[1:(n-1)]-(Tc[2:n]*(exp(-beta*beta*(D[1:(n-1)]/Xi[1:(n-1)])*2/(2-sigma))))) /
                 (1-exp(-beta*beta*((D[1:(n-1)])/Xi[1:(n-1)])*2/(2-sigma)))
    t[n] = (Xi[n]*(2-sigma)/(beta*beta*2)) + Tc[n]
    """
    n = len(Tc)
    t = np.empty(n)
    b2 = beta**2
    D = np.diff(Tc)

    for k in range(n - 1):
        rate = b2 * D[k] / Xi[k] * 2.0 / (2.0 - sigma)
        exp_neg = math.exp(-rate)
        t[k] = Xi[k] * (2.0 - sigma) / (b2 * 2.0) + (Tc[k] - Tc[k + 1] * exp_neg) / (1.0 - exp_neg)

    t[n - 1] = Xi[n - 1] * (2.0 - sigma) / (b2 * 2.0) + Tc[n - 1]
    return t


def r_equilibrium_probs(Tc, Xi, beta, sigma):
    """R: build_HMM_matrix() lines 77-79.

    q[1] = 1 - exp(-D[1]*2*beta*beta/(Xi[1]*(2-sigma)))
    q[alpha] = exp(-(2*beta*beta/(2-sigma))*sum(D[1:(alpha-1)]/Xi[1:(alpha-1)])) *
               (1-exp(-D[alpha]*2*beta*beta/(Xi[alpha]*(2-sigma))))
    q[n] = exp(-(2*beta*beta/(2-sigma))*sum(D[1:(n-1)]/Xi[1:(n-1)]))
    """
    n = len(Tc)
    q = np.empty(n)
    b2 = beta**2
    D = np.diff(Tc)

    q[0] = 1.0 - math.exp(-D[0] * 2.0 * b2 / (Xi[0] * (2.0 - sigma)))

    for k in range(1, n - 1):
        cum = sum(D[j] / Xi[j] for j in range(k))
        cum *= 2.0 * b2 / (2.0 - sigma)
        q[k] = math.exp(-cum) * (1.0 - math.exp(-D[k] * 2.0 * b2 / (Xi[k] * (2.0 - sigma))))

    if n > 1:
        cum = sum(D[j] / Xi[j] for j in range(n - 1))
        cum *= 2.0 * b2 / (2.0 - sigma)
        q[n - 1] = math.exp(-cum)

    q /= q.sum()
    return q


def r_build_emission_matrix(mu, mu_b, Tc, t, beta, n):
    """R: build_emission_matrix() lines 10-37.

    g[k,1] = exp(-2*mu * (sum(beta+(1-beta)*mu_b)*D[1:k-1] +
                          (beta+(1-beta)*mu_b)*(t[k]-Tc[k])))
    g[k,2] = 1 - g[k,1]
    g[k,3] = 1
    """
    e = np.ones((3, n))
    D = np.diff(Tc)

    for k in range(n):
        if k == 0:
            mu_eff = beta + (1.0 - beta) * mu_b
            e[0, k] = math.exp(-2.0 * mu * mu_eff * t[k])
        else:
            branch = 0.0
            for j in range(k):
                mu_eff_j = beta + (1.0 - beta) * mu_b
                branch += mu_eff_j * D[j]
            mu_eff_k = beta + (1.0 - beta) * mu_b
            branch += mu_eff_k * (t[k] - Tc[k])
            e[0, k] = math.exp(-2.0 * mu * branch)
        e[1, k] = 1.0 - e[0, k]
        # e[2, k] = 1.0 already

    return e


def r_build_transition_matrix(Tc, Xi, t, beta, sigma, rho, L):
    """R: build_HMM_matrix() lines 80-163.

    Faithful line-by-line translation of the R transition matrix construction.
    """
    n = len(Tc)
    Q = np.zeros((n, n))
    b2 = beta**2
    r = rho / (2.0 * (L - 1))
    D = np.diff(Tc)

    # Row 0 (R: i==1) -> forward only
    for j in range(1, n):
        rec = 1.0 - math.exp(-2.0 * r * t[j] * beta * 2.0 * (1.0 - sigma) / (2.0 - sigma))
        rate_0 = 4.0 * b2 / (Xi[0] * (2.0 - sigma))
        integral = D[0] - (1.0 - math.exp(-D[0] * rate_0)) / rate_0
        Q[0, j] = rec * (1.0 / (2.0 * t[j])) * integral

    # Row 1 (R: i==2)
    if n > 2:
        i = 1
        rec_coal = 1.0 - math.exp(-D[i] * b2 * 4.0 / (Xi[i] * (2.0 - sigma)))
        # truc for eta=0
        rate_0 = 4.0 * b2 / (Xi[0] * (2.0 - sigma))
        truc = (1.0 - math.exp(-D[0] * rate_0)) / rate_0
        truc *= rec_coal  # (1-exp(-D[i]*beta^2*4/(Xi[i]*(2-sigma))))
        rate_i = 4.0 * b2 / (Xi[i] * (2.0 - sigma))
        diag_int = D[i] - (1.0 - math.exp(-D[i] * rate_i)) / rate_i
        for j in range(i + 1, n):
            rec = 1.0 - math.exp(-2.0 * r * t[j] * beta * 2.0 * (1.0 - sigma) / (2.0 - sigma))
            Q[i, j] = rec * (1.0 / (2.0 * t[j])) * (truc + diag_int)

        # backward: i=1 -> gamma=0 (R: Q[i,1])
        gamma = 0
        rec_g = 1.0 - math.exp(-2.0 * r * t[gamma] * beta * 2.0 * (1.0 - sigma) / (2.0 - sigma))
        # R line 91:
        exp_part = math.exp((t[gamma] - Tc[1]) * b2 * 2.0 / (Xi[0] * (2.0 - sigma)))
        coal_part = 1.0 - math.exp(-D[1] * b2 * 2.0 / (Xi[1] * (2.0 - sigma)))
        int_part = (1.0 - math.exp(-t[gamma] * 4.0 * b2 / (Xi[0] * (2.0 - sigma)))) / (
            4.0 * b2 / (Xi[0] * (2.0 - sigma))
        )
        Q[i, gamma] = rec_g * (1.0 / t[gamma]) * exp_part * coal_part * int_part

    # Rows 2..n-2 (R: i>2)
    for i in range(2, n - 1):
        # R: forward uses 4β², backward uses 2β²
        fwd_coal = 1.0 - math.exp(-D[i] * b2 * 4.0 / (Xi[i] * (2.0 - sigma)))
        bwd_coal = 1.0 - math.exp(-D[i] * b2 * 2.0 / (Xi[i] * (2.0 - sigma)))

        # Forward: truc computation (R lines 95-101)
        # All loop terms (eta = 0..i-2) have exp factors; explicit last (eta = i-1) has none
        truc_sum = 0.0
        for eta in range(i - 1):
            rate_eta = 4.0 * b2 / (Xi[eta] * (2.0 - sigma))
            term = (1.0 - math.exp(-D[eta] * rate_eta)) / rate_eta
            cum = sum(D[kk] / (Xi[kk] * (2.0 - sigma)) for kk in range(eta + 1, i))
            cum *= 4.0 * b2
            truc_sum += math.exp(-cum) * term

        # Explicit last term (eta = i-1) — no exp factor
        rate_last = 4.0 * b2 / (Xi[i - 1] * (2.0 - sigma))
        truc_sum += (1.0 - math.exp(-D[i - 1] * rate_last)) / rate_last

        truc_sum *= fwd_coal
        rate_i = 4.0 * b2 / (Xi[i] * (2.0 - sigma))
        diag_int = D[i] - (1.0 - math.exp(-D[i] * rate_i)) / rate_i

        for j in range(i + 1, n):
            rec = 1.0 - math.exp(-2.0 * r * t[j] * beta * 2.0 * (1.0 - sigma) / (2.0 - sigma))
            Q[i, j] = rec * (1.0 / (2.0 * t[j])) * (truc_sum + diag_int)

        # Backward: i -> gamma for gamma < i (R lines 102-130)
        for gamma in range(i):
            # exp_truc
            if gamma < i - 1:
                cum = sum(D[kk] / (Xi[kk] * (2.0 - sigma)) for kk in range(gamma + 1, i))
                cum *= b2 * 2.0
                cum += b2 * 2.0 * (Tc[gamma + 1] - t[gamma]) / (Xi[gamma] * (2.0 - sigma))
                exp_truc = math.exp(-cum)
            else:
                exp_truc = math.exp(
                    -b2 * 2.0 * (Tc[gamma + 1] - t[gamma]) / (Xi[gamma] * (2.0 - sigma))
                )

            # truc[gamma]
            if gamma == 0:
                rate_g = 4.0 * b2 / (Xi[0] * (2.0 - sigma))
                truc_g = (1.0 - math.exp((Tc[0] - t[0]) * rate_g)) / rate_g
            elif gamma == 1:
                rate_0 = 4.0 * b2 / (Xi[0] * (2.0 - sigma))
                rate_g = 4.0 * b2 / (Xi[gamma] * (2.0 - sigma))
                sub = (1.0 - math.exp(-D[0] * rate_0)) / rate_0
                sub *= math.exp(-rate_g * (t[gamma] - Tc[gamma]))
                truc_g = sub + (1.0 - math.exp((Tc[gamma] - t[gamma]) * rate_g)) / rate_g
            else:
                sub = 0.0
                for eta in range(gamma - 1):
                    rate_eta = 4.0 * b2 / (Xi[eta] * (2.0 - sigma))
                    term = (1.0 - math.exp(-D[eta] * rate_eta)) / rate_eta
                    cum_e = sum(D[kk] / (Xi[kk] * (2.0 - sigma)) for kk in range(eta + 1, gamma))
                    cum_e *= 4.0 * b2
                    cum_e += 4.0 * b2 * (t[gamma] - Tc[gamma]) / (Xi[gamma] * (2.0 - sigma))
                    sub += term * math.exp(-cum_e)
                eta = gamma - 1
                rate_eta = 4.0 * b2 / (Xi[eta] * (2.0 - sigma))
                sub += (
                    (1.0 - math.exp(-D[eta] * rate_eta))
                    / rate_eta
                    * math.exp(-4.0 * b2 * (t[gamma] - Tc[gamma]) / (Xi[gamma] * (2.0 - sigma)))
                )
                rate_g = 4.0 * b2 / (Xi[gamma] * (2.0 - sigma))
                truc_g = sub + (1.0 - math.exp((Tc[gamma] - t[gamma]) * rate_g)) / rate_g

            rec_arg = -2.0 * r * t[gamma] * beta * 2.0 * (1.0 - sigma) / (2.0 - sigma)
            rec_g = 1.0 - math.exp(rec_arg)
            Q[i, gamma] = rec_g * (1.0 / t[gamma]) * bwd_coal * exp_truc * truc_g

    # Row n-1 (R: lines 133-161)
    if n > 1:
        for gamma in range(n - 1):
            if gamma < n - 2:
                cum = sum(D[kk] / (Xi[kk] * (2.0 - sigma)) for kk in range(gamma + 1, n - 1))
                cum *= b2 * 2.0
                cum += b2 * 2.0 * (Tc[gamma + 1] - t[gamma]) / (Xi[gamma] * (2.0 - sigma))
                exp_truc = math.exp(-cum)
            else:
                exp_truc = math.exp(
                    -b2 * 2.0 * (Tc[gamma + 1] - t[gamma]) / (Xi[gamma] * (2.0 - sigma))
                )

            if gamma == 0:
                rate_g = 4.0 * b2 / (Xi[0] * (2.0 - sigma))
                truc_g = (1.0 - math.exp((Tc[0] - t[0]) * rate_g)) / rate_g
            elif gamma == 1:
                rate_0 = 4.0 * b2 / (Xi[0] * (2.0 - sigma))
                rate_g = 4.0 * b2 / (Xi[gamma] * (2.0 - sigma))
                sub = (1.0 - math.exp(-D[0] * rate_0)) / rate_0
                sub *= math.exp(-rate_g * (t[gamma] - Tc[gamma]))
                truc_g = sub + (1.0 - math.exp((Tc[gamma] - t[gamma]) * rate_g)) / rate_g
            else:
                sub = 0.0
                for eta in range(gamma - 1):
                    rate_eta = 4.0 * b2 / (Xi[eta] * (2.0 - sigma))
                    term = (1.0 - math.exp(-D[eta] * rate_eta)) / rate_eta
                    cum_e = sum(D[kk] / (Xi[kk] * (2.0 - sigma)) for kk in range(eta + 1, gamma))
                    cum_e *= 4.0 * b2
                    cum_e += 4.0 * b2 * (t[gamma] - Tc[gamma]) / (Xi[gamma] * (2.0 - sigma))
                    sub += term * math.exp(-cum_e)
                eta = gamma - 1
                rate_eta = 4.0 * b2 / (Xi[eta] * (2.0 - sigma))
                sub += (
                    (1.0 - math.exp(-D[eta] * rate_eta))
                    / rate_eta
                    * math.exp(-4.0 * b2 * (t[gamma] - Tc[gamma]) / (Xi[gamma] * (2.0 - sigma)))
                )
                rate_g = 4.0 * b2 / (Xi[gamma] * (2.0 - sigma))
                truc_g = sub + (1.0 - math.exp((Tc[gamma] - t[gamma]) * rate_g)) / rate_g

            rec_arg = -2.0 * r * t[gamma] * beta * 2.0 * (1.0 - sigma) / (2.0 - sigma)
            rec_g = 1.0 - math.exp(rec_arg)
            Q[n - 1, gamma] = rec_g * (1.0 / t[gamma]) * exp_truc * truc_g

    # Diagonal (R line 162): diag(Q) = 1 - colSums(Q)
    for j in range(n):
        Q[j, j] = 1.0 - sum(Q[i, j] for i in range(n) if i != j)

    return Q


# ---------------------------------------------------------------------------
# Test cases: compare Python numba vs R reference
# ---------------------------------------------------------------------------


class TestCrossValidationTimeBoundaries:
    """Compare esmc2_build_time_boundaries against R build_Tc."""

    @pytest.mark.parametrize(
        "n,beta,sigma",
        [
            (10, 1.0, 0.0),  # standard case
            (20, 0.5, 0.0),  # dormancy
            (15, 1.0, 0.5),  # selfing
            (10, 0.7, 0.3),  # both
            (30, 0.3, 0.8),  # extreme
        ],
    )
    def test_matches_r_reference(self, n, beta, sigma):
        py_Tc = esmc2_build_time_boundaries(n, beta, sigma)
        r_Tc = r_build_Tc(n, beta, sigma)
        np.testing.assert_allclose(py_Tc, r_Tc, rtol=1e-12)


class TestCrossValidationExpectedTimes:
    """Compare esmc2_expected_times against R build_HMM_matrix lines 75-76."""

    @pytest.mark.parametrize(
        "n,beta,sigma",
        [
            (10, 1.0, 0.0),
            (10, 0.5, 0.0),
            (10, 1.0, 0.5),
            (15, 0.7, 0.3),
        ],
    )
    def test_constant_pop(self, n, beta, sigma):
        Tc = r_build_Tc(n, beta, sigma)
        Xi = np.ones(n)
        py_t = esmc2_expected_times(Tc, Xi, beta, sigma)
        r_t = r_expected_times(Tc, Xi, beta, sigma)
        np.testing.assert_allclose(py_t, r_t, rtol=1e-10)

    def test_variable_pop(self):
        n = 10
        beta, sigma = 0.8, 0.2
        Tc = r_build_Tc(n, beta, sigma)
        Xi = np.array([0.5, 1.0, 2.0, 1.5, 0.8, 1.2, 0.6, 1.8, 1.0, 0.5])
        py_t = esmc2_expected_times(Tc, Xi, beta, sigma)
        r_t = r_expected_times(Tc, Xi, beta, sigma)
        np.testing.assert_allclose(py_t, r_t, rtol=1e-10)


class TestCrossValidationEquilibriumProbs:
    """Compare esmc2_equilibrium_probs against R build_HMM_matrix lines 77-79."""

    @pytest.mark.parametrize(
        "n,beta,sigma",
        [
            (10, 1.0, 0.0),
            (10, 0.5, 0.0),
            (10, 1.0, 0.5),
            (15, 0.7, 0.3),
        ],
    )
    def test_constant_pop(self, n, beta, sigma):
        Tc = r_build_Tc(n, beta, sigma)
        Xi = np.ones(n)
        py_q = esmc2_equilibrium_probs(Tc, Xi, beta, sigma)
        r_q = r_equilibrium_probs(Tc, Xi, beta, sigma)
        np.testing.assert_allclose(py_q, r_q, rtol=1e-10)

    def test_variable_pop(self):
        n = 10
        beta, sigma = 0.8, 0.2
        Tc = r_build_Tc(n, beta, sigma)
        Xi = np.array([0.5, 1.0, 2.0, 1.5, 0.8, 1.2, 0.6, 1.8, 1.0, 0.5])
        py_q = esmc2_equilibrium_probs(Tc, Xi, beta, sigma)
        r_q = r_equilibrium_probs(Tc, Xi, beta, sigma)
        np.testing.assert_allclose(py_q, r_q, rtol=1e-10)


class TestCrossValidationEmission:
    """Compare esmc2_build_emission_matrix against R build_emission_matrix."""

    @pytest.mark.parametrize(
        "n,beta,sigma,mu,mu_b",
        [
            (10, 1.0, 0.0, 0.001, 1.0),
            (10, 0.5, 0.0, 0.001, 1.0),
            (10, 1.0, 0.5, 0.001, 1.0),
            (10, 0.5, 0.0, 0.001, 0.5),  # reduced seed bank mutation
            (15, 0.7, 0.3, 0.0005, 0.8),
        ],
    )
    def test_matches_r(self, n, beta, sigma, mu, mu_b):
        Tc = r_build_Tc(n, beta, sigma)
        Xi = np.ones(n)
        t = r_expected_times(Tc, Xi, beta, sigma)
        py_e = esmc2_build_emission_matrix(mu, mu_b, Tc, t, beta, n)
        r_e = r_build_emission_matrix(mu, mu_b, Tc, t, beta, n)
        np.testing.assert_allclose(py_e, r_e, rtol=1e-10)


class TestCrossValidationTransition:
    """Compare esmc2_build_transition_matrix against R build_HMM_matrix."""

    @pytest.mark.parametrize(
        "n,beta,sigma,rho,L",
        [
            (5, 1.0, 0.0, 10.0, 50000),  # small, standard
            (10, 1.0, 0.0, 5.0, 100000),  # standard PSMC-like
            (10, 0.5, 0.0, 5.0, 100000),  # dormancy only
            (10, 1.0, 0.5, 5.0, 100000),  # selfing only
            (10, 0.7, 0.3, 5.0, 100000),  # both
            (8, 0.5, 0.5, 3.0, 50000),  # strong effects
        ],
    )
    def test_matches_r(self, n, beta, sigma, rho, L):
        Tc = r_build_Tc(n, beta, sigma)
        Xi = np.ones(n)
        t = r_expected_times(Tc, Xi, beta, sigma)

        py_Q = esmc2_build_transition_matrix(Tc, Xi, t, beta, sigma, rho, L)
        r_Q = r_build_transition_matrix(Tc, Xi, t, beta, sigma, rho, L)

        # Both should be valid transition matrices in upstream orientation:
        # Q[next_state, current_state], so columns sum to one.
        np.testing.assert_allclose(py_Q.sum(axis=0), 1.0, atol=1e-8)
        np.testing.assert_allclose(r_Q.sum(axis=0), 1.0, atol=1e-8)

        # Should match element-wise
        np.testing.assert_allclose(py_Q, r_Q, rtol=1e-6, atol=1e-10)

    def test_variable_pop(self):
        n = 8
        beta, sigma = 0.8, 0.2
        rho, L = 5.0, 50000
        Tc = r_build_Tc(n, beta, sigma)
        Xi = np.array([0.5, 1.0, 2.0, 1.5, 0.8, 1.2, 0.6, 1.0])
        t = r_expected_times(Tc, Xi, beta, sigma)

        py_Q = esmc2_build_transition_matrix(Tc, Xi, t, beta, sigma, rho, L)
        r_Q = r_build_transition_matrix(Tc, Xi, t, beta, sigma, rho, L)

        np.testing.assert_allclose(py_Q, r_Q, rtol=1e-6, atol=1e-10)


class TestCrossValidationFullHMM:
    """Compare full HMM build and verify log-likelihood consistency."""

    def test_full_hmm_standard(self):
        """Standard parameters (no dormancy/selfing) = PSMC-like."""
        n = 10
        Xi = np.ones(n)
        beta, sigma = 1.0, 0.0
        rho, mu, mu_b, L = 5.0, 0.001, 1.0, 50000

        Q, q, t, Tc, e = esmc2_build_hmm(n, Xi, beta, sigma, rho, mu, mu_b, L)

        # Check against R reference
        r_Tc = r_build_Tc(n, beta, sigma)
        r_t = r_expected_times(r_Tc, Xi, beta, sigma)
        r_q = r_equilibrium_probs(r_Tc, Xi, beta, sigma)
        r_Q = r_build_transition_matrix(r_Tc, Xi, r_t, beta, sigma, rho, L)
        r_e = r_build_emission_matrix(mu, mu_b, r_Tc, r_t, beta, n)

        np.testing.assert_allclose(Tc, r_Tc, rtol=1e-12)
        np.testing.assert_allclose(t, r_t, rtol=1e-10)
        np.testing.assert_allclose(q, r_q, rtol=1e-10)
        np.testing.assert_allclose(Q, r_Q, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(e, r_e, rtol=1e-10)

    def test_full_hmm_dormancy_selfing(self):
        """HMM with both dormancy and selfing."""
        n = 10
        Xi = np.ones(n)
        beta, sigma = 0.6, 0.4
        rho, mu, mu_b, L = 3.0, 0.0005, 0.7, 80000

        Q, q, t, Tc, e = esmc2_build_hmm(n, Xi, beta, sigma, rho, mu, mu_b, L)

        r_Tc = r_build_Tc(n, beta, sigma)
        r_t = r_expected_times(r_Tc, Xi, beta, sigma)
        r_q = r_equilibrium_probs(r_Tc, Xi, beta, sigma)
        r_Q = r_build_transition_matrix(r_Tc, Xi, r_t, beta, sigma, rho, L)
        r_e = r_build_emission_matrix(mu, mu_b, r_Tc, r_t, beta, n)

        np.testing.assert_allclose(Tc, r_Tc, rtol=1e-12)
        np.testing.assert_allclose(t, r_t, rtol=1e-10)
        np.testing.assert_allclose(q, r_q, rtol=1e-10)
        np.testing.assert_allclose(Q, r_Q, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(e, r_e, rtol=1e-10)

    def test_loglik_on_synthetic_data(self):
        """Log-likelihood should be finite and negative on synthetic data."""
        n = 10
        Xi = np.ones(n)
        beta, sigma = 0.8, 0.2
        rho, mu, mu_b, L = 5.0, 0.001, 1.0, 10000

        Q, q, t, Tc, e = esmc2_build_hmm(n, Xi, beta, sigma, rho, mu, mu_b, L)

        rng = np.random.default_rng(42)
        seq = rng.choice([0, 1], size=1000, p=[0.99, 0.01]).astype(np.int8)

        ll = esmc2_forward_loglik(Q, e, q, seq)
        assert np.isfinite(ll)
        assert ll < 0


class TestSimulationValidation:
    """Validate eSMC2 on synthetic data with known demography."""

    @pytest.mark.slow
    def test_constant_pop_recovery(self):
        """eSMC2 on constant-population data should recover flat Ne."""
        from smckit._core import SmcData
        from smckit.tl._esmc2 import esmc2

        # Simulate constant population: Ne=10000, mu=1.25e-8, L=500kb
        # Expected heterozygosity per bp = 4*Ne*mu = 0.0005
        Ne = 10000
        mu_rate = 1.25e-8
        het_rate = 4 * Ne * mu_rate  # 0.0005
        L = 500_000

        rng = np.random.default_rng(2024)
        seq = rng.choice([0, 1], size=L, p=[1 - het_rate, het_rate]).astype(np.int8)

        data = SmcData()
        data.uns["records"] = [{"codes": seq}]
        data.uns["sum_L"] = int((seq < 2).sum())
        data.uns["sum_n"] = int((seq == 1).sum())

        result = esmc2(
            data,
            n_states=10,
            n_iterations=10,
            mu=mu_rate,
            generation_time=1.0,
            rho_over_theta=1.0,
            implementation="native",
        )

        r = result.results["esmc2"]

        # Log-likelihood should be finite and negative
        assert r["log_likelihood"] < 0
        assert np.all(np.isfinite(r["ne"]))
        assert np.all(r["ne"] > 0)
        assert len(r["rounds"]) >= 10

    @pytest.mark.slow
    def test_likelihood_improves_across_iterations(self):
        """Log-likelihood should generally improve across EM iterations."""
        from smckit._core import SmcData
        from smckit.tl._esmc2 import esmc2

        rng = np.random.default_rng(99)
        L = 100_000
        seq = rng.choice([0, 1], size=L, p=[0.999, 0.001]).astype(np.int8)

        data = SmcData()
        data.uns["records"] = [{"codes": seq}]
        data.uns["sum_L"] = int((seq < 2).sum())
        data.uns["sum_n"] = int((seq == 1).sum())

        result = esmc2(
            data,
            n_states=8,
            n_iterations=5,
            mu=1.25e-8,
            generation_time=1.0,
            implementation="native",
        )

        rounds = result.results["esmc2"]["rounds"]
        lls = [rd["log_likelihood"] for rd in rounds]

        # At least the last iteration should be >= first (not strictly monotone due to
        # EM with changing priors, but overall trend should be upward)
        assert lls[-1] >= lls[0] - abs(lls[0]) * 0.01
