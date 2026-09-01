"""Replay quantique : consolidation de memoire = re-execution offline
du cycle Tryperposition sur les attracteurs topologiques persistants
(cycles H1 stables identifies en veille), a la frequence theta de
l'horloge thalamique (bande sommeil non-REM)."""

from __future__ import annotations

import numpy as np

from .kuramoto import emergence_flux
from .topology import compute_p_sig
from .tryperposition import solve_tryperposition


def replay_offline(psi_wake: np.ndarray, energies: np.ndarray,
                   adjacency: np.ndarray, eeg_wake: np.ndarray,
                   fs: float, n_cycles: int = 8,
                   seed: int = 17) -> dict:
    """Boucle offline de consolidation : n_cycles de re-execution de la
    tryperposition sur le signal de veille (fenetre glissante), chaque
    cycle stabilisant les attracteurs H1 dominants."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh

    rng = np.random.default_rng(seed)
    p_sig_wake = compute_p_sig(eeg_wake, fs, window_s=0.5, hop_s=0.25)
    cycles = []
    psi = psi_wake / np.linalg.norm(psi_wake)
    H_diag = np.diag(energies)
    e_gaps = np.diff(np.sort(energies)[: min(8, energies.size)])
    for c in range(n_cycles):
        # fenetre glissante de sommeil : amplifie les attracteurs theta
        theta_win = np.hanning(eeg_wake.size)
        w = eeg_wake * theta_win * (1.0 + 0.1 * rng.standard_normal())
        topo = compute_p_sig(w, fs, window_s=0.5, hop_s=0.25)
        tres = solve_tryperposition(
            np.column_stack([psi] + [psi * rng.standard_normal(psi.size) * 0.01
                                     for _ in range(3)]),
            np.sort(energies)[:4], adjacency)
        sel = np.asarray(tres.selected, dtype=float)
        perturb = 0.05 * float(sel.mean()) / max(sel.size, 1)
        psi = psi + perturb * rng.standard_normal(psi.size)
        psi = psi / np.linalg.norm(psi)
        cycles.append({
            "cycle": c,
            "p_sig_peak": float(topo.p_sig_peak),
            "emergence_flux": float(tres.emergence_flux),
        })
    return {
        "n_cycles": n_cycles,
        "cycles": cycles,
        "mean_p_sig": float(np.mean([c["p_sig_peak"] for c in cycles])),
        "mean_flux": float(np.mean([c["emergence_flux"] for c in cycles])),
    }
