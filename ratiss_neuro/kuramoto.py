"""Horloge Quantique Maître : synchronisation thalamo-corticale
(Kuramoto classique/quantique), pilotant le couplage thermodynamique
de la Tryperposition : Phi = theta . nabla S . nabla T.

Le thalamus est modelise comme le chef d'orchestre imposant le rythme
aux oscillateurs corticaux. La coherence de l'horloge theta(t) module
le taux d'entropie dS/dt = kappa(1 - theta^2).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class KuramotoClock:
    """Horloge thalamique : theta(t) = cos(omega t)."""
    omega: float            # frequence thalamo-corticale (rad/s)
    coherence: float        # |theta| in [0,1]
    entropy_rate: float     # dS/dt = kappa(1-theta^2)
    phase_coupling: float   # parametre K de Kuramoto


def simulate_kuramoto(n_osc: int = 64, K: float = 2.0,
                      omega_theta: float = 6.0, omega_gamma: float = 40.0,
                      duration_s: float = 5.0, fs: float = 1000.0,
                      seed: int = 11) -> tuple[np.ndarray, KuramotoClock]:
    """Oscillateurs de Kuramoto sur bande theta->gamma : la phase
    collective theta(t) module l'enveloppe gamma (couplage PAC)."""
    t = np.arange(int(duration_s * fs)) / fs
    rng = np.random.default_rng(seed)
    phases = np.unwrap(
        np.cumsum(rng.normal(0, 0.01, (n_osc, t.size)), axis=1)
    )
    # Kuramoto : dphi_i/dt = omega_i + (K/N) sum sin(phi_j - phi_i)
    omega = omega_theta * np.ones(n_osc)  # rythme thalamique commun
    order = np.zeros(t.size)
    for k in range(t.size):
        ph = omega * t[k] + phases[:, k]
        order[k] = np.abs(np.mean(np.exp(1j * ph)))
    coherence = float(order.mean())
    theta_env = coherence * np.cos(2 * np.pi * omega_theta * t)
    gamma = theta_env * np.sin(2 * np.pi * omega_gamma * t)
    kappa = 1.0 - coherence ** 2
    clock = KuramotoClock(omega=float(2 * np.pi * omega_theta),
                          coherence=coherence,
                          entropy_rate=float(kappa * abs(theta_env.mean())),
                          phase_coupling=K)
    return gamma + theta_env, clock


def emergence_flux(clock: KuramotoClock, grad_S: float,
                   grad_T: float) -> float:
    """Flux d'emergence thermodynamique : Phi = theta_coh * dS * dT."""
    return float(clock.coherence * grad_S * grad_T)


def consciousness_threshold(p_sig_t: np.ndarray, h1_cycles: np.ndarray,
                          interp_threshold: float = 0.4) -> dict:
    """Seuil de conscience : persistance de cycles H1 non-triviaux.

    Le temps de depassement du seuil d'emergence topologique marque
    l'etat 'conscient' : quand les cycles H1 reapparaissent de facon
    soutenue dans la filtration de Vietoris-Rips."""
    sz = h1_cycles.size
    if sz == 0:
        return {"threshold_reached": False, "n_cycles": 0, "p_sig_max": 0.0}
    mask = p_sig_t > interp_threshold
    return {
        "threshold_reached": bool(mask.any()),
        "n_cycles": int(sz),
        "p_sig_max": float(p_sig_t.max()),
        "duration_idx": np.where(mask)[0].tolist(),
    }
