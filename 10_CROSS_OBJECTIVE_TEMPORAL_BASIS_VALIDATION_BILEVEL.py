#!/usr/bin/env python3
"""
10_CROSS_OBJECTIVE_TEMPORAL_BASIS_VALIDATION_BILEVEL.py

Cross-objective validation of adaptive temporal memory.

Primary question
----------------
Is temporal-basis adaptation objective-dependent rather than universally beneficial?

Design
------
Phase A1 Learn a prediction-adaptive short basis from self-prediction only.
Phase A2 Learn a regulation-adaptive short basis by a bilevel closed-loop objective.
          For each candidate theta, an inner linear probe is fitted to the common
          disturbance-cancellation teacher on fixed exploratory trajectories, but theta
          is selected only by actual closed-loop homeostatic error J_reg on independent
          validation episodes. Only theta is retained from either acquisition.
Phase B  Freeze both bases; fit identical downstream behavioral probes on a new,
         independent trajectory.
Phase C  Cross-test the frozen bases on an independent self-prediction trajectory and
         an independent closed-loop regulation trajectory.
Phase D  Test pulse-disturbance generalization.

No acquisition readout is transferred between objectives. Prediction-adaptive and
regulation-adaptive bases start from the same short bank. The regulation learner uses
the privileged disturbance-cancellation command only for the inner policy fit; latent
disturbance is never a representation feature and teacher-command decoding error is
never the outer theta-selection objective. Regulation theta is selected solely by
held-out closed-loop J_reg.

Actuator authority is fixed analytically rather than tuned against memory results.
With kappa_u=5, the maximum steady control contribution to external-state drift is
chi*kappa_u/lambda_A = 0.4167, comparable to the stationary SD 0.3536 of the
primary OU disturbance. The privileged training command is the steady-state
disturbance-cancellation command u* = -lambda_A D/(chi*kappa_u), clipped to the
common action bound.

Full mode implements the fixed validation specification. Smoke mode is an engineering
test only and is not scientifically valid.

Dependencies: numpy, scipy, pandas, numba (preferred; required for practical full runs).
"""

from __future__ import annotations

import argparse
import json
import math
import os

# The full validation parallelizes across independent seeds.  Force numerical
# libraries to one thread per worker to prevent BLAS/OpenMP oversubscription
# (e.g. 8 worker processes x 8 BLAS threads).  Advanced users can override
# this intentionally with CLOSED_LOOP_BLAS_THREADS.
_BLAS_THREADS = os.environ.get("CLOSED_LOOP_BLAS_THREADS", "1")
for _env_name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_env_name] = _BLAS_THREADS

import sys
import time
import traceback
import pickle
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

try:
    from numba import njit
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "numba is required for this validation because the full design contains long "
        "stochastic trajectories. Install it with: python -m pip install numba"
    ) from exc


VERSION = "2026-09-03-cross-objective-v2.0-bilevel-regulation"
MASTER_SEED = 20260902

# Independent random-stream phases. Keep these fixed so no acquisition stage can
# accidentally access a final cross-test or held-out regulation trajectory.
PHASE_PRED_ACQ = 1
PHASE_REG_FIT_1 = 10
PHASE_REG_VAL_1 = 11
PHASE_REG_FIT_2 = 12
PHASE_REG_VAL_2 = 13
PHASE_FINAL_PROBE = 20
PHASE_FINAL_TEST = 21
PHASE_PULSE_POS = 22
PHASE_PULSE_NEG = 23
PHASE_SELF_CROSS = 30
PHASE_SPSA = 70

COND_BASELINE = "baseline_loop"
COND_INSTANT = "instantaneous"
COND_FIXED_SHORT = "fixed_short"
COND_ADAPTIVE_SHORT = "prediction_adaptive"
COND_REG_ADAPTIVE = "regulation_adaptive"
COND_FIXED_BROAD = "fixed_broad"
COND_ORACLE = "oracle_delay"
COND_TEACHER = "privileged_teacher"

PROBE_CONDITIONS = (
    COND_INSTANT,
    COND_FIXED_SHORT,
    COND_ADAPTIVE_SHORT,
    COND_REG_ADAPTIVE,
    COND_FIXED_BROAD,
    COND_ORACLE,
)
ALL_CONDITIONS = (COND_BASELINE, COND_TEACHER) + PROBE_CONDITIONS

SHORT_THETA = np.array([0.05, 0.08, 0.12, 0.18, 0.27, 0.40], dtype=np.float64)
BROAD_THETA = np.array([0.03, 0.07, 0.16, 0.37, 0.86, 2.00], dtype=np.float64)


@dataclass(frozen=True)
class Config:
    # Core generative system
    dt: float = 0.01
    mu: float = 0.20
    lambda_s: float = 0.60
    gamma: float = 0.40
    lambda_a: float = 0.60
    kappa: float = 1.00
    alpha: float = 1.00
    rho: float = 0.80
    chi: float = 0.05
    sigma_e: float = 0.50
    sigma_s: float = 0.10
    sigma_i: float = 0.10
    sigma_a: float = 0.10

    # Full fixed-delay design
    delays: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 1.60, 2.00)
    n_seeds: int = 100

    # Phase A: predictive acquisition
    T_acq: float = 2000.0
    burn_frac: float = 0.20
    self_horizon: float = 0.20
    self_alpha: float = 1e-3
    theta_min: float = 0.01
    theta_max: float = 4.00
    adaptive_epochs: int = 24
    adaptive_lrs: Tuple[float, ...] = (0.03, 0.10)
    fit_cap: int = 30000
    val_cap: int = 15000
    test_cap: int = 30000

    # Pre-registered accessibility rescue from the previous validated mechanism
    nonexp_threshold: float = 0.40
    nonexp_tol: float = 1e-12
    multistart_screen_epochs: int = 12
    multistart_refine_epochs: int = 48
    multistart_check_every: int = 8
    multistart_patience_checks: int = 3

    # Phase A2: true regulation-objective temporal-basis acquisition.
    # Bilevel design: each candidate theta receives the same inner policy learner,
    # then is scored only by actual held-out closed-loop J_reg. Two independent
    # fit/validation folds reduce trajectory-specific policy overfitting.
    T_reg_fit: float = 800.0
    reg_fit_settle: float = 100.0
    T_reg_validation: float = 800.0
    reg_validation_settle: float = 100.0
    reg_validation_folds: int = 2
    reg_spsa_iterations: int = 24
    reg_spsa_a: float = 1.00
    reg_spsa_c: float = 1.00
    reg_spsa_alpha: float = 0.602
    reg_spsa_gamma: float = 0.101
    reg_spsa_A_fraction: float = 0.10
    reg_spsa_check_every: int = 4
    reg_spsa_grad_clip: float = 10.0
    reg_invalid_penalty: float = 1000.0

    # Independent cross-objective self-prediction assay after both bases are frozen.
    T_self_cross: float = 2000.0

    # Phase B: common behavioral-probe training
    # Scale-matched actuator authority: chi*kappa_u/lambda_a ~= 0.4167,
    # comparable to OU stationary SD sigma/sqrt(2*lambda) ~= 0.3536.
    kappa_u: float = 5.00
    u_bound: float = 1.00
    control_dt: float = 0.05
    T_probe: float = 800.0
    probe_settle: float = 100.0
    explore_amp: float = 0.50
    explore_block: float = 0.50
    probe_ridges: Tuple[float, ...] = (1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0)
    probe_ou_lambda: float = 1.00
    probe_ou_sigma: float = 0.50

    # Phase C: held-out OU disturbance
    T_test: float = 2000.0
    test_settle: float = 200.0
    ou_lambda: float = 1.00
    ou_sigma: float = 0.50

    # Phase D: pulse generalization
    pulse_amp: float = 1.00
    pulse_duration: float = 0.50
    pulse_onset: float = 200.0
    pulse_eval: float = 20.0

    # Numerical stability
    stability_maxabs: float = 1e6
    stability_growthratio: float = 1e4

    # Statistics
    bootstrap_reps: int = 2000

    # Randomization. True intentionally pairs exogenous innovations across delays within seed.
    pair_noise_across_delays: bool = True

    # Engineering
    mode: str = "full"

    @property
    def pulse_T(self) -> float:
        return self.pulse_onset + self.pulse_eval

    @property
    def control_stride(self) -> int:
        return int(round(self.control_dt / self.dt))

    @property
    def teacher_gain(self) -> float:
        # For a constant disturbance D, the control-dependent steady active state is
        # A_u = kappa_u*u/lambda_a and its contribution to dE/dt is chi*A_u.
        # Setting chi*kappa_u*u/lambda_a = -D gives this analytic command gain.
        denom = self.chi * self.kappa_u
        if abs(denom) < 1e-15:
            raise ValueError("chi*kappa_u must be non-zero for disturbance-cancellation teacher")
        return self.lambda_a / denom


# -----------------------------
# Numba numerical kernels
# -----------------------------


@njit(cache=True)
def _simulate_open_loop(
    z_e, z_s, z_i, z_a, dt, delay_steps,
    mu, lambda_s, gamma, lambda_a, kappa, alpha, rho, chi,
    sigma_e, sigma_s, sigma_i, sigma_a,
):
    n = z_e.shape[0]
    E = np.zeros(n, dtype=np.float64)
    S = np.zeros(n, dtype=np.float64)
    I = np.zeros(n, dtype=np.float64)
    A = np.zeros(n, dtype=np.float64)
    sqdt = math.sqrt(dt)
    for t in range(n - 1):
        sd = S[t - delay_steps] if t >= delay_steps else 0.0
        e0 = E[t]
        s0 = S[t]
        i0 = I[t]
        a0 = A[t]
        E[t + 1] = e0 + dt * (-mu * e0 + chi * a0) + sigma_e * sqdt * z_e[t]
        S[t + 1] = s0 + dt * (-lambda_s * s0 + kappa * e0) + sigma_s * sqdt * z_s[t]
        I[t + 1] = i0 + dt * (-gamma * i0 + alpha * math.tanh(sd)) + sigma_i * sqdt * z_i[t]
        A[t + 1] = a0 + dt * (-lambda_a * a0 + rho * math.tanh(i0)) + sigma_a * sqdt * z_a[t]
    return E, S, I, A


@njit(cache=True)
def _simulate_forced_control(
    z_e, z_s, z_i, z_a, u_series, dt, delay_steps,
    mu, lambda_s, gamma, lambda_a, kappa, alpha, rho, chi,
    sigma_e, sigma_s, sigma_i, sigma_a, kappa_u,
):
    n = z_e.shape[0]
    E = np.zeros(n, dtype=np.float64)
    S = np.zeros(n, dtype=np.float64)
    I = np.zeros(n, dtype=np.float64)
    A = np.zeros(n, dtype=np.float64)
    sqdt = math.sqrt(dt)
    for t in range(n - 1):
        sd = S[t - delay_steps] if t >= delay_steps else 0.0
        e0 = E[t]
        s0 = S[t]
        i0 = I[t]
        a0 = A[t]
        u0 = u_series[t]
        E[t + 1] = e0 + dt * (-mu * e0 + chi * a0) + sigma_e * sqdt * z_e[t]
        S[t + 1] = s0 + dt * (-lambda_s * s0 + kappa * e0) + sigma_s * sqdt * z_s[t]
        I[t + 1] = i0 + dt * (-gamma * i0 + alpha * math.tanh(sd)) + sigma_i * sqdt * z_i[t]
        A[t + 1] = a0 + dt * (-lambda_a * a0 + rho * math.tanh(i0) + kappa_u * u0) + sigma_a * sqdt * z_a[t]
    return E, S, I, A


@njit(cache=True)
def _simulate_forced_control_disturbance(
    z_e, z_s, z_i, z_a, u_series, disturbance, dt, delay_steps,
    mu, lambda_s, gamma, lambda_a, kappa, alpha, rho, chi,
    sigma_e, sigma_s, sigma_i, sigma_a, kappa_u,
):
    """Common Phase-B trajectory with PRBS action and exogenous disturbance.

    The physical trajectory is generated once per seed/delay and reused by every
    representation. Therefore behavioral-probe comparisons cannot be confounded by
    representation-dependent training trajectories.
    """
    n = z_e.shape[0]
    E = np.zeros(n, dtype=np.float64)
    S = np.zeros(n, dtype=np.float64)
    I = np.zeros(n, dtype=np.float64)
    A = np.zeros(n, dtype=np.float64)
    sqdt = math.sqrt(dt)
    for t in range(n - 1):
        sd = S[t - delay_steps] if t >= delay_steps else 0.0
        e0 = E[t]
        s0 = S[t]
        i0 = I[t]
        a0 = A[t]
        u0 = u_series[t]
        E[t + 1] = e0 + dt * (-mu * e0 + chi * a0 + disturbance[t]) + sigma_e * sqdt * z_e[t]
        S[t + 1] = s0 + dt * (-lambda_s * s0 + kappa * e0) + sigma_s * sqdt * z_s[t]
        I[t + 1] = i0 + dt * (-gamma * i0 + alpha * math.tanh(sd)) + sigma_i * sqdt * z_i[t]
        A[t + 1] = a0 + dt * (-lambda_a * a0 + rho * math.tanh(i0) + kappa_u * u0) + sigma_a * sqdt * z_a[t]
    return E, S, I, A


@njit(cache=True)
def _traces_only(S, theta, dt):
    n = S.shape[0]
    k = theta.shape[0]
    M = np.zeros((n, k), dtype=np.float64)
    a = np.exp(-dt / theta)
    for t in range(n - 1):
        for j in range(k):
            M[t + 1, j] = a[j] * M[t, j] + (1.0 - a[j]) * S[t]
    return M


@njit(cache=True)
def _traces_and_phi_derivative(S, theta, theta_min, theta_max, dt):
    n = S.shape[0]
    k = theta.shape[0]
    M = np.zeros((n, k), dtype=np.float64)
    D = np.zeros((n, k), dtype=np.float64)  # dm/dphi
    q = np.zeros(k, dtype=np.float64)       # dm/da
    a = np.exp(-dt / theta)
    # dtheta/dphi from bounded logistic parameterization, expressed via theta directly
    z = (theta - theta_min) / (theta_max - theta_min)
    dtheta_dphi = (theta_max - theta_min) * z * (1.0 - z)
    da_dtheta = a * dt / (theta * theta)
    da_dphi = da_dtheta * dtheta_dphi
    for t in range(n - 1):
        for j in range(k):
            q_new = a[j] * q[j] + M[t, j] - S[t]
            M[t + 1, j] = a[j] * M[t, j] + (1.0 - a[j]) * S[t]
            q[j] = q_new
            D[t + 1, j] = q[j] * da_dphi[j]
    return M, D


@njit(cache=True)
def _traces_at_indices(S, theta, indices, dt):
    """Compute recurrent traces but retain only requested sorted sample indices.

    This preserves the exact recurrence used by _traces_only while avoiding an
    O(T x K) retained matrix when only capped fitting/validation/test samples or
    control-time samples are required.
    """
    m = indices.shape[0]
    k = theta.shape[0]
    out = np.zeros((m, k), dtype=np.float64)
    if m == 0:
        return out
    a = np.exp(-dt / theta)
    state = np.zeros(k, dtype=np.float64)
    pos = 0
    # Robustly support index zero even though scientific partitions start later.
    while pos < m and indices[pos] == 0:
        for j in range(k):
            out[pos, j] = state[j]
        pos += 1
    max_idx = indices[m - 1]
    for t in range(max_idx):
        s0 = S[t]
        for j in range(k):
            state[j] = a[j] * state[j] + (1.0 - a[j]) * s0
        current_idx = t + 1
        while pos < m and indices[pos] == current_idx:
            for j in range(k):
                out[pos, j] = state[j]
            pos += 1
    return out


@njit(cache=True)
def _traces_and_phi_derivative_at_indices(S, theta, theta_min, theta_max, indices, dt):
    """Trace states and dm/dphi sampled only at requested sorted indices."""
    m = indices.shape[0]
    k = theta.shape[0]
    Mout = np.zeros((m, k), dtype=np.float64)
    Dout = np.zeros((m, k), dtype=np.float64)
    if m == 0:
        return Mout, Dout
    state = np.zeros(k, dtype=np.float64)
    q = np.zeros(k, dtype=np.float64)
    a = np.exp(-dt / theta)
    z = (theta - theta_min) / (theta_max - theta_min)
    dtheta_dphi = (theta_max - theta_min) * z * (1.0 - z)
    da_dtheta = a * dt / (theta * theta)
    da_dphi = da_dtheta * dtheta_dphi
    pos = 0
    while pos < m and indices[pos] == 0:
        pos += 1
    max_idx = indices[m - 1]
    for t in range(max_idx):
        s0 = S[t]
        for j in range(k):
            q_new = a[j] * q[j] + state[j] - s0
            state[j] = a[j] * state[j] + (1.0 - a[j]) * s0
            q[j] = q_new
        current_idx = t + 1
        while pos < m and indices[pos] == current_idx:
            for j in range(k):
                Mout[pos, j] = state[j]
                Dout[pos, j] = q[j] * da_dphi[j]
            pos += 1
    return Mout, Dout


@njit(cache=True)
def _stability_from_arrays_fast(E, S, I, A, stability_maxabs, stability_growthratio):
    n = E.shape[0]
    q = max(1, n // 4)
    q4 = n - q
    maxabs = 0.0
    ss1 = 0.0
    ss4 = 0.0
    for t in range(n):
        e = E[t]
        s = S[t]
        i = I[t]
        a = A[t]
        if not (math.isfinite(e) and math.isfinite(s) and math.isfinite(i) and math.isfinite(a)):
            return False, math.inf, math.inf
        ae = abs(e)
        ass = abs(s)
        ai = abs(i)
        aa = abs(a)
        if ae > maxabs:
            maxabs = ae
        if ass > maxabs:
            maxabs = ass
        if ai > maxabs:
            maxabs = ai
        if aa > maxabs:
            maxabs = aa
        sq = e * e + s * s + i * i + a * a
        if t < q:
            ss1 += sq
        if t >= q4:
            ss4 += sq
    rms1 = math.sqrt(ss1 / (4.0 * q))
    rms4 = math.sqrt(ss4 / (4.0 * q))
    growth = rms4 / max(rms1, 1e-15)
    stable = (maxabs <= stability_maxabs) and (growth <= stability_growthratio)
    return stable, maxabs, growth


@njit(cache=True)
def _generate_ou(z_d, dt, onset_idx, lam, sigma):
    n = z_d.shape[0]
    D = np.zeros(n, dtype=np.float64)
    sqdt = math.sqrt(dt)
    for t in range(n - 1):
        if t < onset_idx:
            D[t + 1] = 0.0
        else:
            D[t + 1] = D[t] + (-lam * D[t]) * dt + sigma * sqdt * z_d[t]
    return D


@njit(cache=True)
def _simulate_closed_loop_summary(
    z_e, z_s, z_i, z_a, disturbance, dt, delay_steps,
    mu, lambda_s, gamma, lambda_a, kappa, alpha, rho, chi,
    sigma_e, sigma_s, sigma_i, sigma_a, kappa_u,
    condition_code, theta, coef, scales, control_stride, u_bound, teacher_gain,
    eval_start_idx, stability_maxabs, stability_growthratio,
):
    """Run one held-out closed-loop assay and retain summary metrics only.

    condition_code:
      0 = no-control baseline
      1 = instantaneous behavioral probe [I,S,A]
      2 = recurrent behavioral probe [I,S,A,m_1..m_K]
      3 = single-lag oracle probe [I,S,A,S(t-tau)]
      4 = privileged disturbance-cancellation teacher

    Probe coefficients directly predict the signed control command learned in Phase B.
    No representation-specific dynamical model or LQR is used.
    """
    n = z_e.shape[0]
    sqdt = math.sqrt(dt)
    E = 0.0
    S = 0.0
    I = 0.0
    A = 0.0
    Kmem = theta.shape[0]
    mem = np.zeros(Kmem, dtype=np.float64)
    a_mem = np.exp(-dt / theta) if Kmem > 0 else np.empty(0, dtype=np.float64)
    hist_len = max(1, delay_steps + 1)
    s_hist = np.zeros(hist_len, dtype=np.float64)
    u = 0.0

    sum_i2 = 0.0
    sum_u2 = 0.0
    max_abs_i_eval = 0.0
    sat_count = 0
    eval_count = 0

    max_abs_state = 0.0
    q1_end = max(1, n // 4)
    q4_start = max(0, 3 * n // 4)
    q1_ss = 0.0
    q4_ss = 0.0
    q1_n = 0
    q4_n = 0

    for t in range(n - 1):
        s_hist[t % hist_len] = S
        if delay_steps == 0:
            sd_obs_now = S
        elif t >= delay_steps:
            sd_obs_now = s_hist[(t - delay_steps) % hist_len]
        else:
            sd_obs_now = 0.0

        if t % control_stride == 0:
            if condition_code == 0:
                u = 0.0
            elif condition_code == 4:
                # Privileged reference used only to validate assay controllability.
                u = -teacher_gain * disturbance[t]
            elif condition_code == 1:
                u = (
                    coef[0] * (I / scales[0])
                    + coef[1] * (S / scales[1])
                    + coef[2] * (A / scales[2])
                )
            elif condition_code == 2:
                val = (
                    coef[0] * (I / scales[0])
                    + coef[1] * (S / scales[1])
                    + coef[2] * (A / scales[2])
                )
                for j in range(Kmem):
                    val += coef[3 + j] * (mem[j] / scales[3 + j])
                u = val
            else:
                u = (
                    coef[0] * (I / scales[0])
                    + coef[1] * (S / scales[1])
                    + coef[2] * (A / scales[2])
                    + coef[3] * (sd_obs_now / scales[3])
                )
            if u > u_bound:
                u = u_bound
            elif u < -u_bound:
                u = -u_bound

        if t >= eval_start_idx:
            sum_i2 += I * I
            sum_u2 += u * u
            ai = abs(I)
            if ai > max_abs_i_eval:
                max_abs_i_eval = ai
            if abs(u) >= u_bound - 1e-12:
                sat_count += 1
            eval_count += 1

        mabs = max(abs(E), abs(S), abs(I), abs(A))
        if mabs > max_abs_state:
            max_abs_state = mabs
        ss = E * E + S * S + I * I + A * A
        if t < q1_end:
            q1_ss += ss
            q1_n += 1
        if t >= q4_start:
            q4_ss += ss
            q4_n += 1

        sd = sd_obs_now
        e0 = E
        s0 = S
        i0 = I
        a0 = A
        E = e0 + dt * (-mu * e0 + chi * a0 + disturbance[t]) + sigma_e * sqdt * z_e[t]
        S = s0 + dt * (-lambda_s * s0 + kappa * e0) + sigma_s * sqdt * z_s[t]
        I = i0 + dt * (-gamma * i0 + alpha * math.tanh(sd)) + sigma_i * sqdt * z_i[t]
        A = a0 + dt * (-lambda_a * a0 + rho * math.tanh(i0) + kappa_u * u) + sigma_a * sqdt * z_a[t]

        for j in range(Kmem):
            mem[j] = a_mem[j] * mem[j] + (1.0 - a_mem[j]) * s0

        if not (math.isfinite(E) and math.isfinite(S) and math.isfinite(I) and math.isfinite(A)):
            return math.nan, math.nan, math.nan, math.nan, False, math.inf, math.inf

    mabs = max(abs(E), abs(S), abs(I), abs(A))
    if mabs > max_abs_state:
        max_abs_state = mabs
    if eval_count == 0:
        return math.nan, math.nan, math.nan, math.nan, False, max_abs_state, math.inf

    rms1 = math.sqrt(q1_ss / max(1, q1_n) / 4.0)
    rms4 = math.sqrt(q4_ss / max(1, q4_n) / 4.0)
    growth = rms4 / max(rms1, 1e-15)
    stable = (max_abs_state <= stability_maxabs) and (growth <= stability_growthratio)
    return (
        sum_i2 / eval_count,
        sum_u2 / eval_count,
        max_abs_i_eval,
        sat_count / eval_count,
        stable,
        max_abs_state,
        growth,
    )


# -----------------------------
# Utility functions
# -----------------------------


def _phi_to_theta(phi: np.ndarray, cfg: Config) -> np.ndarray:
    pc = np.clip(phi, -12.0, 12.0)
    sig = 1.0 / (1.0 + np.exp(-pc))
    return cfg.theta_min + (cfg.theta_max - cfg.theta_min) * sig


def _theta_to_phi(theta: np.ndarray, cfg: Config) -> np.ndarray:
    z = (theta - cfg.theta_min) / (cfg.theta_max - cfg.theta_min)
    z = np.clip(z, 1e-6, 1.0 - 1e-6)
    return np.log(z / (1.0 - z))


def _even_cap(indices: np.ndarray, cap: int) -> np.ndarray:
    if len(indices) <= cap:
        return indices
    pos = np.linspace(0, len(indices) - 1, cap, dtype=np.int64)
    return indices[pos]


def _nmse(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    v = float(np.var(y))
    if not np.isfinite(v) or v <= 1e-15:
        return float("nan")
    return float(np.mean((y - pred) ** 2) / v)


def _ridge_self_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    # X includes intercept in column 0. Intercept unpenalized.
    p = X.shape[1]
    gram = X.T @ X
    rhs = X.T @ y
    penalty = np.eye(p, dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    try:
        return np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram + penalty, rhs, rcond=None)[0]


def _self_design(I, S, A, M, idx):
    X = np.empty((len(idx), 4 + M.shape[1]), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1] = I[idx]
    X[:, 2] = S[idx]
    X[:, 3] = A[idx]
    X[:, 4:] = M[idx]
    return X


def _self_design_sampled(I, S, A, M_sampled, idx):
    """Build the self-prediction design from already sampled trace rows."""
    X = np.empty((len(idx), 4 + M_sampled.shape[1]), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1] = I[idx]
    X[:, 2] = S[idx]
    X[:, 3] = A[idx]
    X[:, 4:] = M_sampled
    return X


def _acq_partitions(cfg: Config, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    burn = int(round(cfg.burn_frac * (n - 1)))
    start = burn + int(round(max(cfg.delays) / cfg.dt))
    end = n - int(round(cfg.self_horizon / cfg.dt)) - 1
    if end <= start + 10:
        raise RuntimeError(f"Acquisition trajectory too short: start={start}, end={end}, n={n}")
    base = np.arange(start, end, dtype=np.int64)
    n0 = len(base)
    a = int(math.floor(0.60 * n0))
    b = int(math.floor(0.80 * n0))
    fit = _even_cap(base[:a], cfg.fit_cap)
    val = _even_cap(base[a:b], cfg.val_cap)
    test = _even_cap(base[b:], cfg.test_cap)
    return fit, val, test


@dataclass
class AdamState:
    phi: np.ndarray
    m: np.ndarray
    v: np.ndarray
    epoch: int = 0


def _theta_epoch(
    S: np.ndarray,
    I: np.ndarray,
    A: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    state: AdamState,
    lr: float,
    cfg: Config,
) -> Tuple[AdamState, np.ndarray, np.ndarray]:
    theta = _phi_to_theta(state.phi, cfg)
    # Only capped fit samples are retained. The recurrence is still advanced
    # sequentially from t=0, preserving the scientific model exactly.
    Mfit, Dfit = _traces_and_phi_derivative_at_indices(
        S, theta, cfg.theta_min, cfg.theta_max, fit_idx, cfg.dt
    )
    Xf = _self_design_sampled(I, S, A, Mfit, fit_idx)
    beta = _ridge_self_fit(Xf, y[fit_idx], cfg.self_alpha)
    err = Xf @ beta - y[fit_idx]
    w = beta[4:]
    grad = 2.0 * np.mean(err[:, None] * Dfit * w[None, :], axis=0)
    grad = np.clip(grad, -100.0, 100.0)

    t = state.epoch + 1
    m = 0.9 * state.m + 0.1 * grad
    v = 0.999 * state.v + 0.001 * (grad * grad)
    mhat = m / (1.0 - 0.9 ** t)
    vhat = v / (1.0 - 0.999 ** t)
    phi = state.phi - lr * mhat / (np.sqrt(vhat) + 1e-8)
    phi = np.clip(phi, -8.0, 8.0)
    return AdamState(phi=phi, m=m, v=v, epoch=t), beta, theta


def _eval_theta(
    S: np.ndarray,
    I: np.ndarray,
    A: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    theta: np.ndarray,
    cfg: Config,
) -> Dict[str, float]:
    all_idx = np.concatenate((fit_idx, val_idx, test_idx)).astype(np.int64, copy=False)
    Mall = _traces_at_indices(S, theta, all_idx, cfg.dt)
    nf = len(fit_idx)
    nv = len(val_idx)
    Mfit = Mall[:nf]
    Mval = Mall[nf:nf + nv]
    Mtest = Mall[nf + nv:]
    Xf = _self_design_sampled(I, S, A, Mfit, fit_idx)
    beta = _ridge_self_fit(Xf, y[fit_idx], cfg.self_alpha)
    out = {
        "val_nmse": _nmse(y[val_idx], _self_design_sampled(I, S, A, Mval, val_idx) @ beta),
        "test_nmse": _nmse(y[test_idx], _self_design_sampled(I, S, A, Mtest, test_idx) @ beta),
    }
    return out


def _optimize_fixed_epochs(
    S: np.ndarray,
    I: np.ndarray,
    A: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    init_theta: np.ndarray,
    lr: float,
    epochs: int,
    cfg: Config,
    start_state: Optional[AdamState] = None,
) -> Tuple[AdamState, np.ndarray, Dict[str, float]]:
    if start_state is None:
        state = AdamState(
            phi=np.clip(_theta_to_phi(init_theta, cfg), -8.0, 8.0),
            m=np.zeros_like(init_theta),
            v=np.zeros_like(init_theta),
            epoch=0,
        )
    else:
        state = AdamState(
            phi=start_state.phi.copy(), m=start_state.m.copy(), v=start_state.v.copy(), epoch=start_state.epoch
        )
    for _ in range(epochs):
        state, _, _ = _theta_epoch(S, I, A, y, fit_idx, state, lr, cfg)
    theta = _phi_to_theta(state.phi, cfg)
    metrics = _eval_theta(S, I, A, y, fit_idx, val_idx, test_idx, theta, cfg)
    return state, theta, metrics


def _refine_with_early_stop(
    S, I, A, y, fit_idx, val_idx, test_idx,
    initial_state: AdamState, lr: float, cfg: Config
) -> Tuple[AdamState, np.ndarray, Dict[str, float]]:
    # The screened state itself remains admissible.
    best_state = AdamState(initial_state.phi.copy(), initial_state.m.copy(), initial_state.v.copy(), initial_state.epoch)
    best_theta = _phi_to_theta(best_state.phi, cfg)
    best_metrics = _eval_theta(S, I, A, y, fit_idx, val_idx, test_idx, best_theta, cfg)
    best_val = best_metrics["val_nmse"]
    state = AdamState(initial_state.phi.copy(), initial_state.m.copy(), initial_state.v.copy(), initial_state.epoch)
    checks_without = 0
    done = 0
    while done < cfg.multistart_refine_epochs:
        chunk = min(cfg.multistart_check_every, cfg.multistart_refine_epochs - done)
        for _ in range(chunk):
            state, _, _ = _theta_epoch(S, I, A, y, fit_idx, state, lr, cfg)
        done += chunk
        theta = _phi_to_theta(state.phi, cfg)
        metrics = _eval_theta(S, I, A, y, fit_idx, val_idx, test_idx, theta, cfg)
        if np.isfinite(metrics["val_nmse"]) and (not np.isfinite(best_val) or metrics["val_nmse"] < best_val):
            best_val = metrics["val_nmse"]
            best_state = AdamState(state.phi.copy(), state.m.copy(), state.v.copy(), state.epoch)
            best_theta = theta.copy()
            best_metrics = metrics
            checks_without = 0
        else:
            checks_without += 1
            if checks_without >= cfg.multistart_patience_checks:
                break
    return best_state, best_theta, best_metrics


def _learn_adaptive_short(S, I, A, cfg: Config) -> Dict[str, object]:
    n = len(S)
    hsteps = int(round(cfg.self_horizon / cfg.dt))
    y = np.zeros(n, dtype=np.float64)
    y[:-hsteps] = I[hsteps:] - I[:-hsteps]
    fit_idx, val_idx, test_idx = _acq_partitions(cfg, n)

    baseline_candidates = []
    for lr in cfg.adaptive_lrs:
        st, th, met = _optimize_fixed_epochs(
            S, I, A, y, fit_idx, val_idx, test_idx,
            SHORT_THETA, float(lr), cfg.adaptive_epochs, cfg
        )
        baseline_candidates.append((met["val_nmse"], float(lr), st, th, met))
    baseline_candidates.sort(key=lambda x: (np.inf if not np.isfinite(x[0]) else x[0]))
    best_val, eta0, best_state, best_theta, best_metrics = baseline_candidates[0]
    used_multistart = False

    nonexp = float(np.max(best_theta)) <= cfg.nonexp_threshold + cfg.nonexp_tol
    if nonexp:
        local_lrs = sorted(set(float(np.clip(v, 0.001, 0.10)) for v in (eta0 / 10.0, eta0 / 3.0, eta0)))
        starts = [
            SHORT_THETA.copy(),
            SHORT_THETA * 0.80,
            np.minimum(SHORT_THETA * 1.25, 0.40),
            np.minimum(SHORT_THETA * np.array([0.85, 1.15, 0.85, 1.15, 0.85, 1.00]), 0.40),
        ]
        screened = []
        for si, stheta in enumerate(starts):
            for lr in local_lrs:
                st, th, met = _optimize_fixed_epochs(
                    S, I, A, y, fit_idx, val_idx, test_idx,
                    stheta, lr, cfg.multistart_screen_epochs, cfg
                )
                screened.append((met["val_nmse"], si, lr, st, th, met))
        screened.sort(key=lambda x: (np.inf if not np.isfinite(x[0]) else x[0]))
        refined = []
        for item in screened[:2]:
            _, si, lr, st, _, _ = item
            rst, rth, rmet = _refine_with_early_stop(
                S, I, A, y, fit_idx, val_idx, test_idx, st, lr, cfg
            )
            refined.append((rmet["val_nmse"], si, lr, rst, rth, rmet))
        candidates = screened + refined
        candidates.sort(key=lambda x: (np.inf if not np.isfinite(x[0]) else x[0]))
        mval, _, mlr, mst, mth, mmet = candidates[0]
        if np.isfinite(mval) and (not np.isfinite(best_val) or mval < best_val):
            best_val, eta0, best_state, best_theta, best_metrics = mval, mlr, mst, mth, mmet
            used_multistart = True

    return {
        "theta": best_theta.astype(np.float64),
        "eta": float(eta0),
        "val_nmse": float(best_metrics["val_nmse"]),
        "test_nmse": float(best_metrics["test_nmse"]),
        "baseline_nonexp": bool(nonexp),
        "used_multistart": bool(used_multistart),
        "max_theta": float(np.max(best_theta)),
        "count_gt_040": int(np.sum(best_theta > 0.40 + 1e-12)),
    }




def _regulation_candidate_key(phi: np.ndarray) -> Tuple[float, ...]:
    """Stable cache key for an evaluated SPSA candidate."""
    return tuple(np.round(np.asarray(phi, dtype=np.float64), 12).tolist())


def _evaluate_regulation_candidate(
    phi: np.ndarray,
    folds: Sequence[Dict[str, object]],
    delay_steps: int,
    cfg: Config,
) -> Dict[str, object]:
    """Evaluate one temporal basis by its actual closed-loop regulation objective.

    This is the outer objective of Phase A2. For each pre-generated fold, a new
    linear behavioral probe is fitted on that fold's fixed exploratory trajectory
    using the privileged teacher command only as the inner policy-learning target.
    The probe is then frozen and evaluated on a separate validation disturbance
    episode. Candidate theta is scored by mean held-out J_reg across folds.

    The latent disturbance is never supplied as a representation feature. Probe
    teacher-NMSE is diagnostic only and never enters candidate selection.
    """
    theta = _phi_to_theta(np.asarray(phi, dtype=np.float64), cfg)
    jregs: List[float] = []
    jacts: List[float] = []
    sats: List[float] = []
    probe_nmses: List[float] = []
    invalid = 0
    eval_start = int(round(cfg.reg_validation_settle / cfg.dt))

    for fold in folds:
        try:
            probe = _build_probe_for_basis(
                COND_REG_ADAPTIVE,
                theta,
                np.asarray(fold["I_fit"]),
                np.asarray(fold["S_fit"]),
                np.asarray(fold["A_fit"]),
                np.asarray(fold["D_fit"]),
                delay_steps,
                cfg,
                theta_override=theta,
                settle_time=cfg.reg_fit_settle,
            )
            met = _closed_loop_metrics(
                COND_REG_ADAPTIVE,
                theta,
                probe,
                delay_steps,
                fold["val_noise"],
                np.asarray(fold["D_val"]),
                eval_start,
                cfg,
                theta_override=theta,
            )
            if (not met["stable"]) or (not np.isfinite(met["J_reg"])):
                invalid += 1
                jregs.append(float(cfg.reg_invalid_penalty))
            else:
                jregs.append(float(met["J_reg"]))
                jacts.append(float(met["J_act"]))
                sats.append(float(met["sat_fraction"]))
            probe_nmses.append(float(probe["test_nmse"]))
        except Exception:
            invalid += 1
            jregs.append(float(cfg.reg_invalid_penalty))

    objective = float(np.mean(jregs)) if jregs else float(cfg.reg_invalid_penalty)
    return {
        "theta": theta.astype(np.float64),
        "objective_Jreg": objective,
        "mean_J_act": float(np.mean(jacts)) if jacts else np.nan,
        "mean_sat_fraction": float(np.mean(sats)) if sats else np.nan,
        "mean_probe_teacher_nmse": float(np.mean(probe_nmses)) if probe_nmses else np.nan,
        "invalid_folds": int(invalid),
    }


def _learn_regulation_adaptive_short_bilevel(
    folds: Sequence[Dict[str, object]],
    delay_steps: int,
    seed_id: int,
    delay_index: int,
    cfg: Config,
) -> Dict[str, object]:
    """Optimize theta with SPSA using held-out closed-loop J_reg as the outer loss.

    Predictable failure modes are handled explicitly:
    - all theta candidates use identical pre-generated fit/validation trajectories;
    - plus/minus SPSA candidates use common random numbers;
    - the initial short basis is an admissible candidate, so noisy optimization cannot
      force a validation degradation;
    - invalid/unstable candidates receive a fixed pre-specified penalty and are never
      silently discarded;
    - evaluated candidates are cached to avoid repeated expensive simulations;
    - the final center and periodic centers are evaluated in addition to SPSA pairs;
    - no Phase-B or Phase-C trajectory is available to this optimizer.
    """
    if len(folds) != cfg.reg_validation_folds:
        raise ValueError(
            f"Expected {cfg.reg_validation_folds} regulation folds, received {len(folds)}"
        )

    phi = np.clip(_theta_to_phi(SHORT_THETA, cfg), -8.0, 8.0)
    cache: Dict[Tuple[float, ...], Dict[str, object]] = {}

    def evaluate(p: np.ndarray) -> Dict[str, object]:
        pc = np.clip(np.asarray(p, dtype=np.float64), -8.0, 8.0)
        key = _regulation_candidate_key(pc)
        if key not in cache:
            cache[key] = _evaluate_regulation_candidate(pc, folds, delay_steps, cfg)
        return cache[key]

    initial = evaluate(phi)
    best_phi = phi.copy()
    best = dict(initial)
    invalid_evaluations = int(initial["invalid_folds"] > 0)

    # When delays are paired, use the same perturbation sequence across delays in a
    # seed. This isolates the generative delay rather than changing optimizer noise.
    rng = _seedseq_rng(
        MASTER_SEED, PHASE_SPSA, seed_id, 1, delay_index, cfg.pair_noise_across_delays
    )
    n_iter = int(cfg.reg_spsa_iterations)
    Aoff = float(cfg.reg_spsa_A_fraction) * max(1, n_iter)

    for k in range(1, n_iter + 1):
        ak = float(cfg.reg_spsa_a) / ((k + Aoff) ** float(cfg.reg_spsa_alpha))
        ck = float(cfg.reg_spsa_c) / (k ** float(cfg.reg_spsa_gamma))
        delta = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=phi.shape[0])
        p_plus = np.clip(phi + ck * delta, -8.0, 8.0)
        p_minus = np.clip(phi - ck * delta, -8.0, 8.0)
        ev_plus = evaluate(p_plus)
        ev_minus = evaluate(p_minus)
        invalid_evaluations += int(ev_plus["invalid_folds"] > 0)
        invalid_evaluations += int(ev_minus["invalid_folds"] > 0)

        for p_cand, ev in ((p_plus, ev_plus), (p_minus, ev_minus)):
            if np.isfinite(ev["objective_Jreg"]) and ev["objective_Jreg"] < best["objective_Jreg"]:
                best_phi = p_cand.copy()
                best = dict(ev)

        grad = ((ev_plus["objective_Jreg"] - ev_minus["objective_Jreg"]) / (2.0 * ck)) * delta
        grad = np.clip(grad, -float(cfg.reg_spsa_grad_clip), float(cfg.reg_spsa_grad_clip))
        phi = np.clip(phi - ak * grad, -8.0, 8.0)

        if cfg.reg_spsa_check_every > 0 and (k % cfg.reg_spsa_check_every == 0):
            ev_center = evaluate(phi)
            invalid_evaluations += int(ev_center["invalid_folds"] > 0)
            if np.isfinite(ev_center["objective_Jreg"]) and ev_center["objective_Jreg"] < best["objective_Jreg"]:
                best_phi = phi.copy()
                best = dict(ev_center)

    final_center = evaluate(phi)
    invalid_evaluations += int(final_center["invalid_folds"] > 0)
    if np.isfinite(final_center["objective_Jreg"]) and final_center["objective_Jreg"] < best["objective_Jreg"]:
        best_phi = phi.copy()
        best = dict(final_center)

    theta = np.asarray(best["theta"], dtype=np.float64)
    return {
        "theta": theta,
        "initial_val_Jreg": float(initial["objective_Jreg"]),
        "best_val_Jreg": float(best["objective_Jreg"]),
        "val_Jreg_improvement": float(initial["objective_Jreg"] - best["objective_Jreg"]),
        "best_val_J_act": float(best["mean_J_act"]),
        "best_val_sat_fraction": float(best["mean_sat_fraction"]),
        "best_probe_teacher_nmse": float(best["mean_probe_teacher_nmse"]),
        "max_theta": float(np.max(theta)),
        "count_gt_040": int(np.sum(theta > 0.40 + 1e-12)),
        "within_initial_range": bool(float(np.max(theta)) <= cfg.nonexp_threshold + cfg.nonexp_tol),
        "spsa_iterations": n_iter,
        "unique_candidates_evaluated": int(len(cache)),
        "candidate_evaluations_with_invalid_fold": int(invalid_evaluations),
    }

def _evaluate_self_basis(S, I, A, theta, cfg: Config) -> Dict[str, float]:
    n = len(S)
    hsteps = int(round(cfg.self_horizon / cfg.dt))
    y = np.zeros(n, dtype=np.float64)
    y[:-hsteps] = I[hsteps:] - I[:-hsteps]
    fit_idx, val_idx, test_idx = _acq_partitions(cfg, n)
    return _eval_theta(S, I, A, y, fit_idx, val_idx, test_idx, np.asarray(theta, dtype=np.float64), cfg)

def _seedseq_rng(master: int, phase: int, seed_id: int, stream: int, delay_index: Optional[int], paired: bool) -> np.random.Generator:
    key = [int(master), int(phase), int(seed_id), int(stream)]
    if (not paired) and delay_index is not None:
        key.append(int(delay_index))
    return np.random.default_rng(np.random.SeedSequence(key))


def _noise_arrays(cfg: Config, phase: int, seed_id: int, n: int, delay_index: Optional[int] = None):
    return tuple(
        _seedseq_rng(MASTER_SEED, phase, seed_id, stream, delay_index, cfg.pair_noise_across_delays).standard_normal(n)
        for stream in (1, 2, 3, 4)
    )


def _physical_args(cfg: Config):
    return (
        cfg.mu, cfg.lambda_s, cfg.gamma, cfg.lambda_a,
        cfg.kappa, cfg.alpha, cfg.rho, cfg.chi,
        cfg.sigma_e, cfg.sigma_s, cfg.sigma_i, cfg.sigma_a,
    )


def _stability_from_arrays(E, S, I, A, cfg: Config) -> Tuple[bool, float, float]:
    vals = _stability_from_arrays_fast(
        E, S, I, A, cfg.stability_maxabs, cfg.stability_growthratio
    )
    return bool(vals[0]), float(vals[1]), float(vals[2])


def _make_prbs(cfg: Config, seed_id: int, n: int, delay_index: Optional[int] = None, phase: int = 2) -> np.ndarray:
    block_steps = int(round(cfg.explore_block / cfg.dt))
    nblocks = int(math.ceil(n / block_steps))
    rng = _seedseq_rng(MASTER_SEED, phase, seed_id, 10, delay_index, cfg.pair_noise_across_delays)
    signs = rng.choice(np.array([-1.0, 1.0]), size=nblocks)
    u = np.repeat(signs * cfg.explore_amp, block_steps)[:n]
    return u.astype(np.float64)


def _build_representation(
    condition: str,
    I: np.ndarray,
    S: np.ndarray,
    A: np.ndarray,
    delay_steps: int,
    theta_adaptive: np.ndarray,
    cfg: Config,
    theta_override: Optional[np.ndarray] = None,
) -> np.ndarray:
    if condition == COND_INSTANT:
        return np.column_stack([I, S, A])
    if condition == COND_ORACLE:
        sd = np.zeros_like(S)
        if delay_steps > 0:
            sd[delay_steps:] = S[:-delay_steps]
        else:
            sd[:] = S
        return np.column_stack([I, S, A, sd])
    if condition == COND_FIXED_SHORT:
        theta = SHORT_THETA
    elif condition == COND_FIXED_BROAD:
        theta = BROAD_THETA
    elif condition == COND_ADAPTIVE_SHORT:
        theta = theta_adaptive if theta_override is None else theta_override
    elif condition == COND_REG_ADAPTIVE:
        if theta_override is None:
            raise ValueError("regulation_adaptive requires theta_override")
        theta = theta_override
    else:
        raise ValueError(f"Unknown representation condition: {condition}")
    M = _traces_only(S, np.asarray(theta, dtype=np.float64), cfg.dt)
    return np.column_stack([I, S, A, M])


def _build_representation_at_indices(
    condition: str,
    I: np.ndarray,
    S: np.ndarray,
    A: np.ndarray,
    delay_steps: int,
    theta_adaptive: np.ndarray,
    indices: np.ndarray,
    cfg: Config,
    theta_override: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build only the representation rows used by the downstream behavioral probe."""
    if condition == COND_INSTANT:
        return np.column_stack([I[indices], S[indices], A[indices]])
    if condition == COND_ORACLE:
        sd = np.zeros(len(indices), dtype=np.float64)
        if delay_steps == 0:
            sd[:] = S[indices]
        else:
            valid = indices >= delay_steps
            sd[valid] = S[indices[valid] - delay_steps]
        return np.column_stack([I[indices], S[indices], A[indices], sd])
    if condition == COND_FIXED_SHORT:
        theta = SHORT_THETA
    elif condition == COND_FIXED_BROAD:
        theta = BROAD_THETA
    elif condition == COND_ADAPTIVE_SHORT:
        theta = theta_adaptive if theta_override is None else theta_override
    elif condition == COND_REG_ADAPTIVE:
        if theta_override is None:
            raise ValueError("regulation_adaptive requires theta_override")
        theta = theta_override
    else:
        raise ValueError(f"Unknown representation condition: {condition}")
    Ms = _traces_at_indices(S, np.asarray(theta, dtype=np.float64), indices, cfg.dt)
    return np.column_stack([I[indices], S[indices], A[indices], Ms])


def _fit_behavioral_probe(X_control: np.ndarray, teacher_u: np.ndarray, cfg: Config) -> Dict[str, object]:
    """Fit a zero-intercept linear probe to the common signed teacher command.

    Hyperparameter selection uses a contiguous validation segment. The held-out test
    segment is diagnostic only. Predictors are scale-normalized without mean
    subtraction so the physical zero state remains mapped to zero command.
    """
    X = np.asarray(X_control, dtype=np.float64)
    y = np.asarray(teacher_u, dtype=np.float64)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("Behavioral-probe X/y shape mismatch")
    n = len(y)
    if n < 30:
        raise RuntimeError(f"Too few behavioral-probe samples: {n}")

    a = int(math.floor(0.60 * n))
    b = int(math.floor(0.80 * n))
    fit = np.arange(0, a)
    val = np.arange(a, b)
    test = np.arange(b, n)

    scales = np.std(X[fit], axis=0, ddof=0)
    scales = np.maximum(scales, 1e-8)
    Z = X / scales

    def fit_ridge(rows, ridge):
        Zr = Z[rows]
        yr = y[rows]
        gram = Zr.T @ Zr + ridge * np.eye(Zr.shape[1])
        rhs = Zr.T @ yr
        try:
            return np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(gram, rhs, rcond=None)[0]

    best = None
    for ridge in cfg.probe_ridges:
        coef = fit_ridge(fit, float(ridge))
        pred = Z[val] @ coef
        mse = float(np.mean((y[val] - pred) ** 2))
        item = (mse, float(ridge), coef)
        if best is None or item[0] < best[0]:
            best = item
    assert best is not None
    val_mse, ridge, _ = best

    train = np.arange(0, b)
    coef = fit_ridge(train, ridge)
    pred_test = Z[test] @ coef
    test_mse = float(np.mean((y[test] - pred_test) ** 2))
    target_var = float(np.var(y[test]))
    test_nmse = test_mse / target_var if target_var > 1e-15 else float("nan")
    if np.std(y[test]) > 1e-12 and np.std(pred_test) > 1e-12:
        test_corr = float(np.corrcoef(y[test], pred_test)[0, 1])
    else:
        test_corr = float("nan")

    return {
        "coef": coef.astype(np.float64),
        "scales": scales.astype(np.float64),
        "ridge": float(ridge),
        "val_mse": float(val_mse),
        "test_mse": float(test_mse),
        "test_nmse": float(test_nmse),
        "test_corr": float(test_corr),
        "n_samples": int(n),
        "target_rms": float(np.sqrt(np.mean(y * y))),
        "target_sat_fraction": float(np.mean(np.abs(y) >= cfg.u_bound - 1e-12)),
    }


def _teacher_command(disturbance: np.ndarray, cfg: Config) -> np.ndarray:
    """Privileged Phase-B target; never supplied as a probe feature."""
    return np.clip(
        -cfg.teacher_gain * np.asarray(disturbance, dtype=np.float64),
        -cfg.u_bound,
        cfg.u_bound,
    )


def _condition_code_and_theta(condition: str, theta_adaptive: np.ndarray, theta_override: Optional[np.ndarray] = None):
    if condition == COND_BASELINE:
        return 0, np.empty(0, dtype=np.float64)
    if condition == COND_TEACHER:
        return 4, np.empty(0, dtype=np.float64)
    if condition == COND_INSTANT:
        return 1, np.empty(0, dtype=np.float64)
    if condition == COND_ORACLE:
        return 3, np.empty(0, dtype=np.float64)
    if condition == COND_FIXED_SHORT:
        return 2, SHORT_THETA.copy()
    if condition == COND_FIXED_BROAD:
        return 2, BROAD_THETA.copy()
    if condition == COND_ADAPTIVE_SHORT:
        return 2, (theta_adaptive if theta_override is None else theta_override).copy()
    if condition == COND_REG_ADAPTIVE:
        if theta_override is None:
            raise ValueError("regulation_adaptive requires theta_override")
        return 2, np.asarray(theta_override, dtype=np.float64).copy()
    raise ValueError(condition)


def _closed_loop_metrics(
    condition: str,
    theta_adaptive: np.ndarray,
    controller: Optional[Dict[str, object]],
    delay_steps: int,
    z_noise: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    disturbance: np.ndarray,
    eval_start_idx: int,
    cfg: Config,
    theta_override: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    code, theta = _condition_code_and_theta(condition, theta_adaptive, theta_override)
    if condition in (COND_BASELINE, COND_TEACHER):
        coef = np.zeros(1, dtype=np.float64)
        scales = np.ones(1, dtype=np.float64)
    else:
        assert controller is not None
        coef = np.asarray(controller["coef"], dtype=np.float64)
        scales = np.asarray(controller["scales"], dtype=np.float64)
    vals = _simulate_closed_loop_summary(
        z_noise[0], z_noise[1], z_noise[2], z_noise[3], disturbance,
        cfg.dt, delay_steps,
        *_physical_args(cfg), cfg.kappa_u,
        code, theta, coef, scales, cfg.control_stride, cfg.u_bound, cfg.teacher_gain,
        eval_start_idx, cfg.stability_maxabs, cfg.stability_growthratio,
    )
    return {
        "J_reg": float(vals[0]),
        "J_act": float(vals[1]),
        "max_abs_I": float(vals[2]),
        "sat_fraction": float(vals[3]),
        "stable": bool(vals[4]),
        "max_abs_state": float(vals[5]),
        "growth_ratio": float(vals[6]),
    }


def _build_probe_for_basis(
    condition: str,
    theta_adaptive: np.ndarray,
    I_probe: np.ndarray,
    S_probe: np.ndarray,
    A_probe: np.ndarray,
    disturbance_probe: np.ndarray,
    delay_steps: int,
    cfg: Config,
    theta_override: Optional[np.ndarray] = None,
    settle_time: Optional[float] = None,
) -> Dict[str, object]:
    if settle_time is None:
        settle_time = cfg.probe_settle
    start = int(round(float(settle_time) / cfg.dt))
    idx = np.arange(start, len(I_probe), cfg.control_stride, dtype=np.int64)
    Xc = _build_representation_at_indices(
        condition, I_probe, S_probe, A_probe, delay_steps,
        theta_adaptive, idx, cfg, theta_override,
    )
    target = _teacher_command(disturbance_probe[idx], cfg)
    probe = _fit_behavioral_probe(Xc, target, cfg)
    return {
        **probe,
        "state_dim": int(Xc.shape[1]),
    }


def _bootstrap_ci(values: np.ndarray, reps: int, seed: int) -> Tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=np.float64)
    n = len(v)
    for r in range(reps):
        means[r] = np.mean(v[rng.integers(0, n, size=n)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _safe_wilcoxon(x: np.ndarray, y: Optional[np.ndarray] = None) -> float:
    try:
        if y is None:
            arr = np.asarray(x, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0 or np.allclose(arr, 0.0):
                return 1.0
            return float(wilcoxon(arr, alternative="two-sided", zero_method="wilcox").pvalue)
        a = np.asarray(x, dtype=float)
        b = np.asarray(y, dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        if np.sum(mask) == 0 or np.allclose(a[mask] - b[mask], 0.0):
            return 1.0
        return float(wilcoxon(a[mask], b[mask], alternative="two-sided", zero_method="wilcox").pvalue)
    except Exception:
        return float("nan")


def _pulse_disturbance(cfg: Config, sign: float, n: int) -> np.ndarray:
    D = np.zeros(n, dtype=np.float64)
    onset = int(round(cfg.pulse_onset / cfg.dt))
    duration = int(round(cfg.pulse_duration / cfg.dt))
    D[onset:min(n, onset + duration)] = sign * cfg.pulse_amp
    return D


# -----------------------------
# Seed-level execution
# -----------------------------



def _run_seed(seed_id: int, cfg: Config) -> Dict[str, List[dict]]:
    t0 = time.time()
    out: Dict[str, List[dict]] = {
        "status": [], "acquisition": [], "self_cross": [], "probe": [],
        "primary": [], "sanity": [], "pulse": [],
    }

    n_acq = int(round(cfg.T_acq / cfg.dt)) + 1
    n_reg_fit = int(round(cfg.T_reg_fit / cfg.dt)) + 1
    n_reg_val = int(round(cfg.T_reg_validation / cfg.dt)) + 1
    n_probe = int(round(cfg.T_probe / cfg.dt)) + 1
    n_test = int(round(cfg.T_test / cfg.dt)) + 1
    n_self_cross = int(round(cfg.T_self_cross / cfg.dt)) + 1
    n_pulse = int(round(cfg.pulse_T / cfg.dt)) + 1

    # Default scientific design pairs exogenous innovations across delays within a
    # seed. Generate them once per seed to avoid repeated RNG and allocation costs.
    pred_noise = _noise_arrays(cfg, PHASE_PRED_ACQ, seed_id, n_acq)
    self_cross_noise = _noise_arrays(cfg, PHASE_SELF_CROSS, seed_id, n_self_cross)

    reg_fit_specs = []
    for fit_phase, val_phase in (
        (PHASE_REG_FIT_1, PHASE_REG_VAL_1),
        (PHASE_REG_FIT_2, PHASE_REG_VAL_2),
    ):
        fit_noise = _noise_arrays(cfg, fit_phase, seed_id, n_reg_fit)
        u_fit = _make_prbs(cfg, seed_id, n_reg_fit, phase=fit_phase)
        zD_fit = _seedseq_rng(
            MASTER_SEED, fit_phase, seed_id, 5, None, cfg.pair_noise_across_delays
        ).standard_normal(n_reg_fit)
        D_fit = _generate_ou(
            zD_fit, cfg.dt, int(round(cfg.reg_fit_settle / cfg.dt)),
            cfg.probe_ou_lambda, cfg.probe_ou_sigma,
        )
        val_noise = _noise_arrays(cfg, val_phase, seed_id, n_reg_val)
        zD_val = _seedseq_rng(
            MASTER_SEED, val_phase, seed_id, 5, None, cfg.pair_noise_across_delays
        ).standard_normal(n_reg_val)
        D_val = _generate_ou(
            zD_val, cfg.dt, int(round(cfg.reg_validation_settle / cfg.dt)),
            cfg.ou_lambda, cfg.ou_sigma,
        )
        reg_fit_specs.append((fit_noise, u_fit, D_fit, val_noise, D_val))

    final_probe_noise = _noise_arrays(cfg, PHASE_FINAL_PROBE, seed_id, n_probe)
    u_probe = _make_prbs(cfg, seed_id, n_probe, phase=PHASE_FINAL_PROBE)
    zD_probe = _seedseq_rng(
        MASTER_SEED, PHASE_FINAL_PROBE, seed_id, 5, None, cfg.pair_noise_across_delays
    ).standard_normal(n_probe)
    probeD = _generate_ou(
        zD_probe, cfg.dt, int(round(cfg.probe_settle / cfg.dt)),
        cfg.probe_ou_lambda, cfg.probe_ou_sigma,
    )

    final_test_noise = _noise_arrays(cfg, PHASE_FINAL_TEST, seed_id, n_test)
    zD_test = _seedseq_rng(
        MASTER_SEED, PHASE_FINAL_TEST, seed_id, 5, None, cfg.pair_noise_across_delays
    ).standard_normal(n_test)
    ouD = _generate_ou(
        zD_test, cfg.dt, int(round(cfg.test_settle / cfg.dt)),
        cfg.ou_lambda, cfg.ou_sigma,
    )

    pulse_noise_pos = _noise_arrays(cfg, PHASE_PULSE_POS, seed_id, n_pulse)
    pulse_noise_neg = _noise_arrays(cfg, PHASE_PULSE_NEG, seed_id, n_pulse)
    pulseD_pos = _pulse_disturbance(cfg, +1.0, n_pulse)
    pulseD_neg = _pulse_disturbance(cfg, -1.0, n_pulse)

    for di, tau in enumerate(cfg.delays):
        delay_steps = int(round(tau / cfg.dt))
        try:
            # Phase A1: prediction-specific temporal-basis acquisition.
            E, S, I, A = _simulate_open_loop(
                *pred_noise, cfg.dt, delay_steps, *_physical_args(cfg)
            )
            stable, maxabs, growth = _stability_from_arrays(E, S, I, A, cfg)
            if not stable:
                raise RuntimeError(
                    f"Prediction-acquisition plant unstable at tau={tau}: "
                    f"maxabs={maxabs}, growth={growth}"
                )
            pred_learned = _learn_adaptive_short(S, I, A, cfg)
            theta_pred = np.asarray(pred_learned["theta"], dtype=np.float64)

            # Phase A2: regulation-specific bilevel acquisition. Physical fit
            # trajectories are independent of theta because actions are fixed PRBS.
            folds: List[Dict[str, object]] = []
            for fold_id, (fit_noise, u_fit, D_fit, val_noise, D_val) in enumerate(reg_fit_specs):
                Er, Sr, Ir, Ar = _simulate_forced_control_disturbance(
                    *fit_noise, u_fit, D_fit, cfg.dt, delay_steps,
                    *_physical_args(cfg), cfg.kappa_u,
                )
                stable_reg, maxabs_reg, growth_reg = _stability_from_arrays(
                    Er, Sr, Ir, Ar, cfg
                )
                if not stable_reg:
                    raise RuntimeError(
                        f"Regulation inner-fit plant unstable at tau={tau}, fold={fold_id}: "
                        f"maxabs={maxabs_reg}, growth={growth_reg}"
                    )
                folds.append({
                    "I_fit": Ir, "S_fit": Sr, "A_fit": Ar, "D_fit": D_fit,
                    "val_noise": val_noise, "D_val": D_val,
                })

            reg_learned = _learn_regulation_adaptive_short_bilevel(
                folds, delay_steps, seed_id, di, cfg
            )
            theta_reg = np.asarray(reg_learned["theta"], dtype=np.float64)

            out["acquisition"].append({
                "seed": seed_id, "tau": tau, "objective": "prediction",
                "optimizer": "Adam coordinate gradient",
                "max_theta": pred_learned["max_theta"],
                "count_theta_gt_040": pred_learned["count_gt_040"],
                "within_initial_range": pred_learned["baseline_nonexp"],
                "used_multistart": pred_learned["used_multistart"],
                "prediction_eta": pred_learned["eta"],
                "prediction_val_nmse": pred_learned["val_nmse"],
                "prediction_test_nmse": pred_learned["test_nmse"],
                **{f"theta_{j+1}": float(theta_pred[j]) for j in range(6)},
            })
            out["acquisition"].append({
                "seed": seed_id, "tau": tau, "objective": "regulation",
                "optimizer": "SPSA bilevel closed-loop J_reg",
                "max_theta": reg_learned["max_theta"],
                "count_theta_gt_040": reg_learned["count_gt_040"],
                "within_initial_range": reg_learned["within_initial_range"],
                "used_multistart": False,
                "reg_initial_val_Jreg": reg_learned["initial_val_Jreg"],
                "reg_best_val_Jreg": reg_learned["best_val_Jreg"],
                "reg_val_Jreg_improvement": reg_learned["val_Jreg_improvement"],
                "reg_best_val_J_act": reg_learned["best_val_J_act"],
                "reg_best_val_sat_fraction": reg_learned["best_val_sat_fraction"],
                "reg_best_probe_teacher_nmse": reg_learned["best_probe_teacher_nmse"],
                "reg_spsa_iterations": reg_learned["spsa_iterations"],
                "reg_unique_candidates_evaluated": reg_learned["unique_candidates_evaluated"],
                "reg_candidate_evaluations_with_invalid_fold": reg_learned["candidate_evaluations_with_invalid_fold"],
                **{f"theta_{j+1}": float(theta_reg[j]) for j in range(6)},
            })

            # Independent cross-objective self-prediction assay. This trajectory was
            # used by neither acquisition objective, preventing the prediction learner
            # from receiving a same-realization advantage at cross-test.
            Ex, Sx, Ix, Ax = _simulate_open_loop(
                *self_cross_noise, cfg.dt, delay_steps, *_physical_args(cfg)
            )
            stable_self, maxabs_self, growth_self = _stability_from_arrays(
                Ex, Sx, Ix, Ax, cfg
            )
            if not stable_self:
                raise RuntimeError(
                    f"Self-cross plant unstable at tau={tau}: "
                    f"maxabs={maxabs_self}, growth={growth_self}"
                )
            self_bases = {
                COND_FIXED_SHORT: SHORT_THETA,
                COND_ADAPTIVE_SHORT: theta_pred,
                COND_REG_ADAPTIVE: theta_reg,
                COND_FIXED_BROAD: BROAD_THETA,
            }
            for condition, theta in self_bases.items():
                met_self = _evaluate_self_basis(Sx, Ix, Ax, theta, cfg)
                out["self_cross"].append({
                    "seed": seed_id, "tau": tau, "condition": condition,
                    "val_self_nmse": met_self["val_nmse"],
                    "test_self_nmse": met_self["test_nmse"],
                })

            # Phase B: new common downstream-probe training trajectory. Neither
            # acquisition readout is reused.
            Ep, Sp, Ip, Ap = _simulate_forced_control_disturbance(
                *final_probe_noise, u_probe, probeD, cfg.dt, delay_steps,
                *_physical_args(cfg), cfg.kappa_u,
            )
            stable_probe, maxabs_probe, growth_probe = _stability_from_arrays(
                Ep, Sp, Ip, Ap, cfg
            )
            if not stable_probe:
                raise RuntimeError(
                    f"Final probe-training plant unstable at tau={tau}: "
                    f"maxabs={maxabs_probe}, growth={growth_probe}"
                )

            probes = {}
            for condition in PROBE_CONDITIONS:
                theta_override = theta_reg if condition == COND_REG_ADAPTIVE else None
                probe = _build_probe_for_basis(
                    condition, theta_pred, Ip, Sp, Ap, probeD, delay_steps, cfg,
                    theta_override=theta_override,
                )
                probes[condition] = probe
                out["probe"].append({
                    "seed": seed_id, "tau": tau, "condition": condition,
                    "state_dim": probe["state_dim"], "ridge": probe["ridge"],
                    "val_teacher_mse": probe["val_mse"],
                    "test_teacher_mse": probe["test_mse"],
                    "test_teacher_nmse": probe["test_nmse"],
                    "test_teacher_corr": probe["test_corr"],
                    "n_samples": probe["n_samples"],
                    "coef_norm": float(np.linalg.norm(probe["coef"])),
                    "teacher_target_rms": probe["target_rms"],
                    "teacher_target_sat_fraction": probe["target_sat_fraction"],
                })

            # Phase C: completely held-out closed-loop regulation.
            eval_start = int(round(cfg.test_settle / cfg.dt))
            metrics = {}
            for condition in ALL_CONDITIONS:
                controller = None if condition in (COND_BASELINE, COND_TEACHER) else probes[condition]
                theta_override = theta_reg if condition == COND_REG_ADAPTIVE else None
                met = _closed_loop_metrics(
                    condition, theta_pred, controller, delay_steps,
                    final_test_noise, ouD, eval_start, cfg,
                    theta_override=theta_override,
                )
                metrics[condition] = met
                out["primary"].append({
                    "seed": seed_id, "tau": tau, "condition": condition,
                    "teacher_gain": cfg.teacher_gain, "kappa_u": cfg.kappa_u, **met,
                })

            out["sanity"].append({
                "seed": seed_id, "tau": tau,
                "teacher_regulation_benefit": metrics[COND_BASELINE]["J_reg"] - metrics[COND_TEACHER]["J_reg"],
                "teacher_J_reg": metrics[COND_TEACHER]["J_reg"],
                "baseline_J_reg": metrics[COND_BASELINE]["J_reg"],
                "teacher_sat_fraction": metrics[COND_TEACHER]["sat_fraction"],
                "prediction_probe_nmse": probes[COND_ADAPTIVE_SHORT]["test_nmse"],
                "regulation_probe_nmse": probes[COND_REG_ADAPTIVE]["test_nmse"],
                "regulation_probe_advantage": probes[COND_ADAPTIVE_SHORT]["test_nmse"] - probes[COND_REG_ADAPTIVE]["test_nmse"],
                "reg_acquisition_validation_improvement": reg_learned["val_Jreg_improvement"],
                "reg_acquisition_validation_sat_fraction": reg_learned["best_val_sat_fraction"],
            })

            # Phase D: pulse generalization with the same frozen Phase-B probes.
            pulse_eval_start = int(round(cfg.pulse_onset / cfg.dt))
            for sign, pnoise, pD in (
                (+1.0, pulse_noise_pos, pulseD_pos),
                (-1.0, pulse_noise_neg, pulseD_neg),
            ):
                for condition in ALL_CONDITIONS:
                    controller = None if condition in (COND_BASELINE, COND_TEACHER) else probes[condition]
                    theta_override = theta_reg if condition == COND_REG_ADAPTIVE else None
                    met = _closed_loop_metrics(
                        condition, theta_pred, controller, delay_steps,
                        pnoise, pD, pulse_eval_start, cfg,
                        theta_override=theta_override,
                    )
                    out["pulse"].append({
                        "seed": seed_id, "tau": tau, "condition": condition,
                        "pulse_sign": int(sign),
                        "J_pulse_integrated": float(met["J_reg"] * cfg.pulse_eval),
                        "peak_abs_I": met["max_abs_I"], "J_act": met["J_act"],
                        "sat_fraction": met["sat_fraction"], "stable": met["stable"],
                    })

        except Exception as exc:
            out["status"].append({
                "seed": seed_id, "tau": tau, "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=3),
            })
            continue
        else:
            out["status"].append({
                "seed": seed_id, "tau": tau, "ok": True,
                "error": "", "traceback": "",
            })

    out["status"].append({
        "seed": seed_id, "tau": np.nan, "ok": True,
        "error": f"seed_complete_seconds={time.time()-t0:.3f}", "traceback": "",
    })
    return out


# -----------------------------
# Aggregate analysis and output
# -----------------------------


def _flatten(results: List[Dict[str, List[dict]]], key: str) -> pd.DataFrame:
    rows: List[dict] = []
    for r in results:
        rows.extend(r[key])
    return pd.DataFrame(rows)


def _cross_objective_statistics(self_cross: pd.DataFrame, primary: pd.DataFrame, cfg: Config):
    if self_cross.empty or primary.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    sp = self_cross.pivot_table(index=["seed", "tau"], columns="condition", values="test_self_nmse", aggfunc="first").reset_index()
    rp = primary.pivot_table(index=["seed", "tau"], columns="condition", values="J_reg", aggfunc="first").reset_index()
    merged = sp.merge(rp, on=["seed", "tau"], suffixes=("_self", "_reg"))

    pred_self = f"{COND_ADAPTIVE_SHORT}_self"
    reg_self = f"{COND_REG_ADAPTIVE}_self"
    fixed_self = f"{COND_FIXED_SHORT}_self"
    pred_reg = f"{COND_ADAPTIVE_SHORT}_reg"
    reg_reg = f"{COND_REG_ADAPTIVE}_reg"
    fixed_reg = f"{COND_FIXED_SHORT}_reg"
    required = [pred_self, reg_self, fixed_self, pred_reg, reg_reg, fixed_reg]
    if any(c not in merged.columns for c in required):
        return pd.DataFrame(), pd.DataFrame(), merged

    # Positive values mean the objective-matched basis performs better.
    merged["S_prediction"] = merged[reg_self] - merged[pred_self]
    merged["S_regulation"] = merged[pred_reg] - merged[reg_reg]
    merged["B_prediction_vs_fixed"] = merged[fixed_self] - merged[pred_self]
    merged["B_regulation_vs_fixed"] = merged[fixed_reg] - merged[reg_reg]

    seed_rows = []
    for seed, g in merged.groupby("seed"):
        g = g.sort_values("tau")
        s_pred = g["S_prediction"].to_numpy(float)
        s_reg = g["S_regulation"].to_numpy(float)
        taus = g["tau"].to_numpy(float)
        mp = np.isfinite(s_pred) & np.isfinite(taus)
        mr = np.isfinite(s_reg) & np.isfinite(taus)
        bp = g["B_prediction_vs_fixed"].to_numpy(float)
        br = g["B_regulation_vs_fixed"].to_numpy(float)
        mbp = np.isfinite(bp)
        mbr = np.isfinite(br)
        seed_rows.append({
            "seed": int(seed),
            "mean_S_prediction": float(np.mean(s_pred[mp])) if np.any(mp) else np.nan,
            "mean_S_regulation": float(np.mean(s_reg[mr])) if np.any(mr) else np.nan,
            "mean_B_prediction_vs_fixed": float(np.mean(bp[mbp])) if np.any(mbp) else np.nan,
            "mean_B_regulation_vs_fixed": float(np.mean(br[mbr])) if np.any(mbr) else np.nan,
            "rs_tau_S_prediction": float(spearmanr(taus[mp], s_pred[mp]).statistic) if np.sum(mp) >= 3 else np.nan,
            "rs_tau_S_regulation": float(spearmanr(taus[mr], s_reg[mr]).statistic) if np.sum(mr) >= 3 else np.nan,
            "matched_both": bool(np.any(mp) and np.any(mr) and np.mean(s_pred[mp]) > 0 and np.mean(s_reg[mr]) > 0),
        })
    sdf = pd.DataFrame(seed_rows)

    rows = []
    def add_summary(label, vals, definition, seedoff):
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        ci = _bootstrap_ci(v, cfg.bootstrap_reps, MASTER_SEED + seedoff)
        rows.append({
            "analysis": label, "n": len(v),
            "effect": float(np.mean(v)) if len(v) else np.nan,
            "median": float(np.median(v)) if len(v) else np.nan,
            "ci_low": ci[0], "ci_high": ci[1],
            "proportion_predicted": float(np.mean(v > 0)) if len(v) else np.nan,
            "p_value": _safe_wilcoxon(v), "definition": definition,
        })

    add_summary(
        "Prediction-objective specialization",
        sdf["mean_S_prediction"],
        "mean_tau(NMSE_regulation_adaptive - NMSE_prediction_adaptive); positive favors prediction-adaptive",
        7101,
    )
    add_summary(
        "Regulation-objective specialization",
        sdf["mean_S_regulation"],
        "mean_tau(Jreg_prediction_adaptive - Jreg_regulation_adaptive); positive favors regulation-adaptive",
        7102,
    )
    add_summary(
        "Prediction-adaptive benefit over fixed short",
        sdf["mean_B_prediction_vs_fixed"],
        "mean_tau(NMSE_fixed_short - NMSE_prediction_adaptive); positive favors prediction-adaptive",
        7105,
    )
    add_summary(
        "Regulation-adaptive benefit over fixed short",
        sdf["mean_B_regulation_vs_fixed"],
        "mean_tau(Jreg_fixed_short - Jreg_regulation_adaptive); positive favors regulation-adaptive",
        7106,
    )
    add_summary(
        "Delay dependence of prediction specialization",
        sdf["rs_tau_S_prediction"],
        "seed-level Spearman r_s(tau, S_prediction)",
        7103,
    )
    add_summary(
        "Delay dependence of regulation specialization",
        sdf["rs_tau_S_regulation"],
        "seed-level Spearman r_s(tau, S_regulation)",
        7104,
    )
    both = sdf["matched_both"].to_numpy(bool)
    rows.append({
        "analysis": "Both objective-matched advantages within seed",
        "n": len(both), "effect": float(np.mean(both)) if len(both) else np.nan,
        "median": np.nan, "ci_low": np.nan, "ci_high": np.nan,
        "proportion_predicted": float(np.mean(both)) if len(both) else np.nan,
        "p_value": np.nan,
        "definition": "mean S_prediction > 0 and mean S_regulation > 0 in the same seed",
    })

    for tau, g in merged.groupby("tau"):
        add_summary(
            f"Prediction specialization, tau={tau:g}", g["S_prediction"],
            "NMSE_regulation_adaptive - NMSE_prediction_adaptive", 8000 + int(round(tau*100)),
        )
        add_summary(
            f"Regulation specialization, tau={tau:g}", g["S_regulation"],
            "Jreg_prediction_adaptive - Jreg_regulation_adaptive", 9000 + int(round(tau*100)),
        )

    return pd.DataFrame(rows), sdf, merged


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small DataFrame as Markdown without optional dependencies.

    Pandas Markdown export requires the external ``tabulate`` package.
    The validation report should not depend on an otherwise-unused package, so
    this intentionally small formatter handles the scalar summary tables used
    by this script.
    """
    if df.empty:
        return ""

    def _cell(value) -> str:
        if value is None:
            text = ""
        elif isinstance(value, (float, np.floating)) and np.isnan(value):
            text = "NaN"
        else:
            text = str(value)
        return text.replace("|", r"\|").replace("\n", "<br>")

    headers = [_cell(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    return "\n".join(lines)


def _write_report(outdir: Path, cfg: Config, frames: Dict[str, pd.DataFrame], elapsed: float):
    stats = frames.get("stats", pd.DataFrame())
    status = frames.get("status", pd.DataFrame())
    primary = frames.get("primary", pd.DataFrame())
    sanity = frames.get("sanity", pd.DataFrame())
    acq = frames.get("acquisition", pd.DataFrame())
    n_fail = int(np.sum(status["ok"] == False)) if (not status.empty and "ok" in status) else 0  # noqa: E712
    n_unstable = int(np.sum(primary["stable"] == False)) if (not primary.empty and "stable" in primary) else 0  # noqa: E712
    teacher_benefit = float(np.nanmean(sanity["teacher_regulation_benefit"])) if not sanity.empty else np.nan
    pred_range = 0
    reg_range = 0
    reg_val_improvement = float("nan")
    reg_invalid = 0
    if not acq.empty:
        pred_rows = acq[acq["objective"] == "prediction"]
        reg_rows = acq[acq["objective"] == "regulation"]
        if "within_initial_range" in pred_rows:
            pred_range = int(np.sum(pred_rows["within_initial_range"].fillna(False).astype(bool)))
        if "within_initial_range" in reg_rows:
            reg_range = int(np.sum(reg_rows["within_initial_range"].fillna(False).astype(bool)))
        if "reg_val_Jreg_improvement" in reg_rows:
            reg_val_improvement = float(np.nanmean(reg_rows["reg_val_Jreg_improvement"].to_numpy(float)))
        if "reg_candidate_evaluations_with_invalid_fold" in reg_rows:
            reg_invalid = int(np.nansum(reg_rows["reg_candidate_evaluations_with_invalid_fold"].to_numpy(float)))
    lines = [
        "# Cross-objective temporal-basis validation report", "",
        f"- Version: `{VERSION}`", f"- Mode: `{cfg.mode}`", f"- Master seed: `{MASTER_SEED}`",
        f"- Runtime: {elapsed:.1f} s", f"- Seed-level execution failures: {n_fail}",
        f"- Unstable closed-loop condition rows: {n_unstable}",
        f"- Prediction-basis rows remaining within the initial short range: {pred_range}",
        f"- Regulation-basis rows remaining within the initial short range: {reg_range}",
        f"- Mean Phase-A2 validation J_reg improvement over the initial short basis: {reg_val_improvement:.6g}",
        f"- Regulation candidate evaluations containing an invalid validation fold: {reg_invalid}",
        f"- Mean privileged-teacher regulation benefit over no control: {teacher_benefit:.6g}", "",
        "## Primary question", "",
        "Is temporal-basis adaptation objective-dependent rather than universally beneficial?", "",
        "## Primary statistics", "",
        _dataframe_to_markdown(stats) if not stats.empty else "Statistics unavailable.", "",
        "## Prespecified separation", "",
        "Prediction-adaptive theta is learned only from self-prediction on Phase A1.", "",
        "Regulation-adaptive theta uses a bilevel Phase A2 objective. The privileged disturbance-cancellation command is used only to fit the inner linear policy for each candidate theta. Candidate theta is selected only by mean closed-loop J_reg on independent validation episodes; teacher-command decoding error is diagnostic and is not the outer objective.", "",
        "The initial short basis is retained as an admissible Phase-A2 candidate, plus/minus SPSA evaluations use common random numbers, and invalid candidates receive a fixed pre-specified penalty rather than being silently excluded.", "",
        "Both learned theta vectors are frozen before final testing. Final behavioral probes are refitted on a new trajectory, closed-loop regulation is evaluated on another independent trajectory, and self-prediction cross-testing uses a separate open-loop trajectory not used by either acquisition phase.", "",
        "The latent disturbance is never a representation feature. No representation-specific system identification or LQR is used.",
    ]
    (outdir / "10_RUN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _smoke_config() -> Config:
    return Config(
        delays=(0.20, 1.20), n_seeds=2,
        T_acq=80.0, burn_frac=0.10, adaptive_epochs=2, adaptive_lrs=(0.03,),
        fit_cap=3000, val_cap=1500, test_cap=3000,
        multistart_screen_epochs=1, multistart_refine_epochs=2,
        multistart_check_every=1, multistart_patience_checks=1,
        T_reg_fit=40.0, reg_fit_settle=5.0,
        T_reg_validation=40.0, reg_validation_settle=5.0,
        reg_validation_folds=2, reg_spsa_iterations=2,
        reg_spsa_check_every=1,
        T_self_cross=80.0,
        T_probe=40.0, probe_settle=5.0, probe_ridges=(1e-6, 1e-3),
        T_test=60.0, test_settle=10.0,
        pulse_onset=10.0, pulse_eval=5.0, bootstrap_reps=200, mode="smoke",
    )


def _warm_numba():
    # Small deterministic calls compile the heavy kernels before work starts.
    z = np.zeros(8, dtype=np.float64)
    _simulate_open_loop(
        z, z, z, z, 0.01, 1,
        0.2, 0.6, 0.4, 0.6, 1.0, 1.0, 0.8, 0.05,
        0.5, 0.1, 0.1, 0.1,
    )
    _simulate_forced_control_disturbance(
        z, z, z, z, z, z, 0.01, 1,
        0.2, 0.6, 0.4, 0.6, 1.0, 1.0, 0.8, 0.05,
        0.5, 0.1, 0.1, 0.1, 5.0,
    )
    _traces_only(z, SHORT_THETA, 0.01)
    _traces_and_phi_derivative(z, SHORT_THETA, 0.01, 4.0, 0.01)
    idx = np.array([0, 2, 5], dtype=np.int64)
    _traces_at_indices(z, SHORT_THETA, idx, 0.01)
    _traces_and_phi_derivative_at_indices(z, SHORT_THETA, 0.01, 4.0, idx, 0.01)
    _stability_from_arrays_fast(z, z, z, z, 1e6, 1e4)
    _generate_ou(z, 0.01, 2, 1.0, 0.5)
    _simulate_closed_loop_summary(
        z, z, z, z, z, 0.01, 1,
        0.2, 0.6, 0.4, 0.6, 1.0, 1.0, 0.8, 0.05,
        0.5, 0.1, 0.1, 0.1, 5.0,
        0, np.empty(0), np.zeros(1), np.ones(1), 5, 1.0, 2.4,
        2, 1e6, 1e4,
    )


def _effective_cpu_count() -> int:
    """Conservative CPU count for process-level parallelism.

    On Apple Silicon, prefer performance-core count; on Linux, respect cgroup
    CPU quotas when present. Falling back to os.cpu_count keeps the script
    portable.
    """
    if sys.platform == "darwin":
        try:
            val = subprocess.check_output(
                ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                text=True, stderr=subprocess.DEVNULL, timeout=2.0,
            ).strip()
            n = int(val)
            if n > 0:
                return n
        except Exception:
            pass
    if sys.platform.startswith("linux"):
        try:
            txt = Path("/sys/fs/cgroup/cpu.max").read_text().strip().split()
            if len(txt) >= 2 and txt[0] != "max":
                quota = float(txt[0])
                period = float(txt[1])
                if quota > 0 and period > 0:
                    n = max(1, int(math.floor(quota / period + 1e-12)))
                    return min(n, os.cpu_count() or n)
        except Exception:
            pass
    return max(1, os.cpu_count() or 1)


def _recommended_workers(mode: str) -> int:
    if mode == "smoke":
        return 1
    cores = _effective_cpu_count()
    # Each seed repeatedly scans long trajectories and is memory-bandwidth
    # intensive. Using roughly half the available performance/physical cores
    # is intentionally conservative and avoids thermal/memory contention,
    # especially on fanless Apple Silicon laptops. Users may override manually.
    if cores <= 2:
        return 1
    return max(1, min(6, cores // 2))


def _validate_config(cfg: Config) -> None:
    phases = (
        PHASE_PRED_ACQ, PHASE_REG_FIT_1, PHASE_REG_VAL_1,
        PHASE_REG_FIT_2, PHASE_REG_VAL_2, PHASE_FINAL_PROBE,
        PHASE_FINAL_TEST, PHASE_PULSE_POS, PHASE_PULSE_NEG,
        PHASE_SELF_CROSS, PHASE_SPSA,
    )
    if len(set(phases)) != len(phases):
        raise ValueError("Random-stream phase IDs must be unique")
    if cfg.reg_validation_folds != 2:
        raise ValueError("This fixed validation design requires exactly two regulation folds")
    if cfg.control_stride < 1:
        raise ValueError("control_dt must be at least dt")
    if cfg.T_reg_fit <= cfg.reg_fit_settle + 20 * cfg.control_dt:
        raise ValueError("T_reg_fit is too short relative to reg_fit_settle")
    if cfg.T_reg_validation <= cfg.reg_validation_settle + 20 * cfg.control_dt:
        raise ValueError("T_reg_validation is too short relative to reg_validation_settle")
    if cfg.reg_spsa_iterations < 1:
        raise ValueError("reg_spsa_iterations must be positive")
    if cfg.reg_spsa_a <= 0 or cfg.reg_spsa_c <= 0:
        raise ValueError("SPSA gain parameters must be positive")
    if cfg.reg_spsa_grad_clip <= 0 or cfg.reg_invalid_penalty <= 0:
        raise ValueError("SPSA safeguards must be positive")
    if not np.isfinite(cfg.teacher_gain):
        raise ValueError("teacher_gain must be finite")


def _parse_args():
    p = argparse.ArgumentParser(description="Cross-objective temporal-basis validation")
    p.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    p.add_argument("--workers", type=int, default=0, help="Seed-level worker processes; 0=auto (full: conservative physical-core use, smoke: 1).")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seeds", type=int, default=None, help="Optional override for engineering runs.")
    return p.parse_args()


def main():
    args = _parse_args()
    cfg = _smoke_config() if args.mode == "smoke" else Config(mode="full")
    if args.seeds is not None:
        cfg = replace(cfg, n_seeds=int(args.seeds))
    _validate_config(cfg)

    if args.output_dir:
        outdir = Path(args.output_dir).expanduser().resolve()
    else:
        outdir = Path.home() / "Desktop" / "10_CROSS_OBJECTIVE_TEMPORAL_BASIS_VALIDATION_BILEVEL"
    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / "00_CONFIG.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    workers = args.workers
    if workers <= 0:
        workers = _recommended_workers(cfg.mode)
    print(
        f"[{VERSION}] mode={cfg.mode} seeds={cfg.n_seeds} delays={len(cfg.delays)} "
        f"workers={workers} effective_cpus={_effective_cpu_count()} BLAS_threads={_BLAS_THREADS}"
    )
    print(f"Output: {outdir}")
    print("Compiling numerical kernels...")
    _warm_numba()
    print("Compilation ready.")

    t0 = time.time()
    seed_ids = list(range(1, cfg.n_seeds + 1))
    results: List[Dict[str, List[dict]]] = []
    checkpoint_dir = outdir / ("_checkpoints_" + VERSION.replace("/", "_").replace(" ", "_"))
    checkpoint_dir.mkdir(exist_ok=True)
    progress_log = outdir / "RUN_PROGRESS.log"

    pending = []
    for seed in seed_ids:
        cp = checkpoint_dir / f"seed_{seed:03d}.pkl"
        if cp.exists():
            try:
                with cp.open("rb") as fh:
                    results.append(pickle.load(fh))
                msg = f"seed {seed:03d} loaded from checkpoint"
                print(msg, flush=True)
                with progress_log.open("a", encoding="utf-8") as lg:
                    lg.write(msg + "\n")
                continue
            except Exception:
                pass
        pending.append(seed)

    def save_checkpoint(seed, res):
        cp = checkpoint_dir / f"seed_{seed:03d}.pkl"
        tmp = cp.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cp)

    if workers <= 1:
        for k, seed in enumerate(pending, 1):
            st = time.time()
            res = _run_seed(seed, cfg)
            results.append(res)
            save_checkpoint(seed, res)
            msg = f"seed {seed:03d}/{cfg.n_seeds:03d} complete in {time.time()-st:.1f}s"
            print(msg, flush=True)
            with progress_log.open("a", encoding="utf-8") as lg:
                lg.write(msg + "\n")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futures = {ex.submit(_run_seed, seed, cfg): seed for seed in pending}
            done = 0
            for fut in as_completed(futures):
                seed = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    res = {k: [] for k in ("status", "acquisition", "self_cross", "probe", "primary", "sanity", "pulse")}
                    res["status"] = [{"seed": seed, "tau": np.nan, "ok": False, "error": f"worker failure: {exc}", "traceback": traceback.format_exc()}]
                results.append(res)
                save_checkpoint(seed, res)
                done += 1
                msg = f"seed {seed:03d} complete ({done}/{len(pending)} pending; {len(results)}/{cfg.n_seeds} total)"
                print(msg, flush=True)
                with progress_log.open("a", encoding="utf-8") as lg:
                    lg.write(msg + "\n")

    frames = {
        "status": _flatten(results, "status"),
        "acquisition": _flatten(results, "acquisition"),
        "self_cross": _flatten(results, "self_cross"),
        "probe": _flatten(results, "probe"),
        "primary": _flatten(results, "primary"),
        "sanity": _flatten(results, "sanity"),
        "pulse": _flatten(results, "pulse"),
    }
    stats, seed_stats, cross_table = _cross_objective_statistics(frames["self_cross"], frames["primary"], cfg)
    frames["stats"] = stats
    frames["seed_stats"] = seed_stats
    frames["cross_table"] = cross_table

    filenames = {
        "status": "01_RUN_STATUS.csv",
        "acquisition": "02_BASIS_ACQUISITION.csv",
        "self_cross": "03_SELF_PREDICTION_CROSS_TEST.csv",
        "probe": "04_BEHAVIORAL_PROBE_FIT.csv",
        "primary": "05_CLOSED_LOOP_REGULATION.csv",
        "stats": "06_CROSS_OBJECTIVE_STATISTICS.csv",
        "seed_stats": "07_SEED_LEVEL_OBJECTIVE_EFFECTS.csv",
        "sanity": "08_CONTROL_SANITY.csv",
        "pulse": "09_PULSE_GENERALIZATION.csv",
    }
    for key, fn in filenames.items():
        frames[key].to_csv(outdir / fn, index=False)

    elapsed = time.time() - t0
    _write_report(outdir, cfg, frames, elapsed)
    print(f"Complete in {elapsed:.1f}s")
    print(f"Report: {outdir / '10_RUN_REPORT.md'}")


if __name__ == "__main__":
    main()
