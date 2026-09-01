"""Reseau de neurones a impulsions (SNN) multi-regions, couple a
l'horloge quantique thalamo-corticale.

Architecture (doc kiki.docx, RATISS-SNN-WHOLEBRAIN) :
  - Unite AdEx (Adaptive Exponential Integrate-and-Fire) : bifurcation
    spike/burst, adaptation de frequence, initiation aigue du spike.
  - Micro-circuit par region : 80% excitateurs / 20% inhibiteurs,
    connectivite locale p_conn.
  - Couplage inter-regional via le connectome structural W_ij avec
    delais de conduction.
  - Courant quantique I_quantum injecte par region (p_n du collapse
    module l'excitabilite) + horloge thalamique theta (Kuramoto).

La dynamique est *forward causale* : les micro-etats EEG emergent des
spikes, ce qui cible ISO -> 1.0 (au-dela du surrogate statistique).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AdExParams:
    """Parametres AdEx calibres cortex (Brette & Gerstner 2005)."""
    c_m: float = 200.0        # pF
    g_l: float = 10.0         # nS
    e_l: float = -70.0        # mV
    v_t: float = -50.0        # mV (seuil exponentiel)
    delta_t: float = 2.0      # mV (pente d'initiation aigue)
    a: float = 2.0            # nS (couplage adaptation sous-seuil)
    tau_w: float = 100.0      # ms
    b: float = 60.0           # pA (saut d'adaptation post-spike)
    v_reset: float = -58.0    # mV
    v_spike: float = 0.0      # mV (detection)
    tau_syn_e: float = 5.0    # ms (AMPA)
    tau_syn_i: float = 10.0   # ms (GABA)


@dataclass
class SNNResult:
    spikes: np.ndarray        # (n_steps, n_neurons) bool
    v_trace: np.ndarray       # (n_steps, n_neurons) mV
    lfp: np.ndarray           # (n_steps,) champ moyen par region somme
    region_rates: np.ndarray  # (n_regions, n_steps) taux de feu par region
    fs: float


def build_microcircuit(n_regions: int, n_per_region: int = 40,
                       frac_exc: float = 0.8, p_local: float = 0.15,
                       w_connectome: np.ndarray | None = None,
                       p_inter: float = 0.05, seed: int = 7
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Matrice de poids synaptiques (n_total, n_total) creuse + types E/I.

    w_connectome : (n_regions, n_regions) poids structuraux inter-regions.
    Retourne (W, is_exc) avec W en unites de conductance normalisee.
    """
    rng = np.random.default_rng(seed)
    n_total = n_regions * n_per_region
    n_exc = int(n_per_region * frac_exc)
    is_exc = np.zeros(n_total, dtype=bool)
    for r in range(n_regions):
        is_exc[r * n_per_region: r * n_per_region + n_exc] = True

    W = np.zeros((n_total, n_total), dtype=np.float32)

    # connectivite locale intra-region
    for r in range(n_regions):
        idx = np.arange(r * n_per_region, (r + 1) * n_per_region)
        mask = rng.random((n_per_region, n_per_region)) < p_local
        np.fill_diagonal(mask, False)
        w_local = np.where(np.isin(np.arange(n_per_region),
                                   np.arange(n_exc)), 0.5, -1.0)
        W[np.ix_(idx, idx)] = mask * w_local[:, None] * rng.uniform(
            0.5, 1.5, (n_per_region, n_per_region))

    # connectivite inter-regionale via le connectome structural
    if w_connectome is not None:
        for i in range(n_regions):
            for j in range(n_regions):
                w_ij = w_connectome[i, j]
                if w_ij <= 0 or i == j:
                    continue
                idx_i = np.arange(i * n_per_region, i * n_per_region + n_exc)
                idx_j = np.arange(j * n_per_region, (j + 1) * n_per_region)
                mask = rng.random((n_exc, n_per_region)) < p_inter
                W[np.ix_(idx_i, idx_j)] += mask * float(w_ij) * 0.3

    return W, is_exc


def simulate_snn(w: np.ndarray, is_exc: np.ndarray,
                 params: AdExParams, duration_s: float, dt_ms: float = 0.1,
                 i_quantum: np.ndarray | None = None,
                 theta_clock: np.ndarray | None = None,
                 i_ext_pa: float = 300.0, seed: int = 11,
                 ) -> SNNResult:
    """Simulation AdEx vectorisee (Euler).

    i_quantum : (n_regions,) courant quantique par region (pA), constant
        sur la simulation — module l'excitabilite regionale.
    theta_clock : (n_steps,) coherence theta de l'horloge thalamique,
        module l'input externe (pacemaker).
    """
    n_total = w.shape[0]
    n_regions = i_quantum.size if i_quantum is not None else 1
    n_per_region = n_total // n_regions
    n_steps = int(duration_s * 1000.0 / dt_ms)
    dt = dt_ms

    v = np.full(n_total, params.e_l, dtype=np.float64)
    w_adapt = np.zeros(n_total, dtype=np.float64)
    g_e = np.zeros(n_total, dtype=np.float64)
    g_i = np.zeros(n_total, dtype=np.float64)

    spikes = np.zeros((n_steps, n_total), dtype=bool)
    v_trace = np.zeros((n_steps, n_total), dtype=np.float32)

    rng = np.random.default_rng(seed)
    w_sign = np.where(is_exc, 1.0, 0.0)  # courant entrant deja signe dans W

    for step in range(n_steps):
        # courant synaptique : conductances exponentielles
        g_e *= np.exp(-dt / params.tau_syn_e)
        g_i *= np.exp(-dt / params.tau_syn_i)

        # input externe module par l'horloge thalamique
        pacemaker = 1.0
        if theta_clock is not None:
            pacemaker = 0.5 + 0.5 * theta_clock[step % theta_clock.size]
        i_ext = i_ext_pa * pacemaker + 50.0 * rng.standard_normal(n_total)

        # courant quantique par region
        if i_quantum is not None:
            i_q = np.repeat(i_quantum, n_per_region)
        else:
            i_q = 0.0

        # courant synaptique total (W deja signe : E>0, I<0)
        i_syn = w @ (np.where(is_exc, g_e, 0.0) + np.where(~is_exc, g_i, 0.0))

        # AdEx Euler
        exp_term = params.delta_t * np.exp((v - params.v_t) / params.delta_t)
        dv = (-params.g_l * (v - params.e_l) + params.g_l * exp_term
              - w_adapt + i_ext + i_q + i_syn) / params.c_m
        dw = (params.a * (v - params.e_l) - w_adapt) / params.tau_w
        v += dv * dt
        w_adapt += dw * dt

        # detection spike + reset
        fired = v >= params.v_spike
        spikes[step] = fired
        v[fired] = params.v_reset
        w_adapt[fired] += params.b

        # transmission synaptique
        g_e += fired * is_exc * 1.0
        g_i += fired * (~is_exc) * 1.0

        v_trace[step] = v

    # LFP : somme des taux de feu par region (proxy du champ EEG)
    region_rates = np.zeros((n_regions, n_steps))
    for r in range(n_regions):
        region_rates[r] = spikes[:, r * n_per_region:
                                 (r + 1) * n_per_region].sum(axis=1)
    lfp = region_rates.sum(axis=0)

    return SNNResult(spikes=spikes, v_trace=v_trace, lfp=lfp,
                     region_rates=region_rates, fs=1000.0 / dt_ms)


def snn_to_eeg(result: SNNResult, target_fs: float,
               kern_ms: float = 15.0) -> np.ndarray:
    """Convertit le champ SNN (potentiel membranaire moyen) en EEG.

    Le potentiel moyen est le bon proxy du LFP/EEG : il preserve la
    dynamique lente du drive (lineaire sous-seuil) la ou le taux de
    spikes est filtre par la refractarite. Convolution exponentielle
    (PSP) puis sous-echantillonnage.
    """
    fs_snn = result.fs
    v_mean = result.v_trace.mean(axis=1)
    kern_t = np.arange(0, 100, 1.0 / fs_snn * 1000.0) / 1000.0
    kern = np.exp(-kern_t / (kern_ms / 1000.0))
    kern /= kern.sum()
    smooth = np.convolve(v_mean, kern, mode="same")
    factor = max(1, int(round(fs_snn / target_fs)))
    return smooth[::factor].astype(np.float64)


def theta_drive_from_eeg(ref: np.ndarray, fs: float, n_steps: int,
                         f_lo: float = 1.0, f_hi: float = 16.0,
                         depth: float = 0.5) -> np.ndarray:
    """Enveloppe lente reelle extraite de l'EEG pour driver le SNN.

    C'est la condition aux limites de Dirichlet du doc : les phases
    < 16 Hz (pacemaker thalamo-cortical) pilotent le cortex. Retourne
    un drive multiplicatif 1 + depth * slow, upsample a n_steps.
    """
    from scipy.signal import butter, sosfiltfilt
    n = min(800, ref.size)
    sos = butter(4, [f_lo, min(f_hi, fs / 2 - 1)], btype="band", fs=fs,
                 output="sos")
    slow = sosfiltfilt(sos, ref[:n])
    slow_up = np.repeat(slow, int(np.ceil(n_steps / n)))[:n_steps]
    slow_up /= slow_up.std() + 1e-12
    # clamp : un drive multiplicatif doit rester positif
    return np.clip(1.0 + depth * slow_up, 0.05, None)
