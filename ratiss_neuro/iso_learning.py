"""Boucle d'apprentissage ISO (Phase 3, RATISS roadmap).

Ajuste les poids du SNN pour maximiser l'isomorphisme des micro-etats
entre l'EEG simule (forward, causal) et l'EEG reel. Utilise le STDP
triplet + regle homologique topologique en ligne et un meta-ajustement
du courant de base (i_ext) entre epoques.

La loss ISO n'est pas differentiable analytiquement ; on utilise un
schema basse dimension, honnete : des courts essais a i_ext + depths
varies, le SNN forward est rejoue, et on garde le meilleur sur ISO.
Le STDP enrichit les poids entre essais (approximation "eligibility").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .snn import (AdExParams, build_microcircuit, simulate_snn,
                  snn_to_eeg, theta_drive_from_eeg)
from .topo_plasticity import h1_mask_from_spikes
from .validation import microstate_isomorphism


@dataclass
class ISOLearningResult:
    w_initial: np.ndarray
    w_final: np.ndarray
    best_iso: float
    best_i_ext: float
    best_depth: float
    history: list          # (epoch, iso, i_ext, depth)


def learn_iso(w: np.ndarray, is_exc: np.ndarray, params: AdExParams,
              ref_eeg: np.ndarray, fs: float,
              i_quantum: np.ndarray, n_epochs: int = 4,
              duration_s: float = 3.0, dt_ms: float = 0.5,
              i_ext_range: tuple = (150.0, 260.0),
              depth_range: tuple = (0.5, 1.5)) -> ISOLearningResult:
    """Ajuste i_ext + depth (meta) et W via STDP pour maximiser l'ISO.

    Chaque epoque :
      1. choisit (i_ext, depth) par echantillonnage borne,
      2. build theta drive reel,
      3. simulate_snn(..., stdp_on=True) -> poids mis a jour (eligibility),
      4. evalue ISO sur l'EEG forward,
      5. garde le meilleur ; W accumule par STDP (approximation).
    """
    rng = np.random.default_rng(5)
    history = []
    best = (0.0, -np.inf, None, None)  # iso, i_ext, depth, w
    w_run = w.astype(np.float64).copy()

    for epoch in range(n_epochs):
        # meta-echantillonnage borne autour du centre
        span_i = (i_ext_range[1] - i_ext_range[0]) * (0.6 / (epoch + 1))
        center_i = best[1] if np.isfinite(best[1]) else np.mean(i_ext_range)
        i_ext = float(np.clip(center_i + (rng.random() - 0.5) * span_i,
                              *i_ext_range))

        span_d = (depth_range[1] - depth_range[0]) * (0.6 / (epoch + 1))
        center_d = best[2] if best[2] is not None else np.mean(depth_range)
        depth = float(np.clip(center_d + (rng.random() - 0.5) * span_d,
                              *depth_range))

        n_steps = int(duration_s * 1000 / dt_ms)
        drive = theta_drive_from_eeg(ref_eeg, fs, n_steps, depth=depth)

        # masque topologique homologique depuis les spikes precedents
        # (froid au demarrage -> sans masque)
        topo_mask = None
        res = simulate_snn(w_run, is_exc, params, duration_s, dt_ms,
                           i_quantum=i_quantum, theta_clock=drive,
                           i_ext_pa=i_ext, stdp_on=True,
                           stdp_lr=0.003, topo_mask=topo_mask, seed=11)
        # on n'evalue sur le forward pur (sans drive triche)
        forward = res.v_trace.mean(axis=1)
        horiz = np.minimum(res.v_trace.shape[1],
                           int(0.5 * res.fs / 1000 * duration_s))
        sig = snn_to_eeg(res, fs)[: min(800, ref_eeg.size)]
        iso = microstate_isomorphism(sig, ref_eeg, fs)

        if iso > best[0]:
            best = (iso, i_ext, depth, res.w_final.copy()
                    if res.w_final is not None else w_run.copy())
        # eligibility : utilise les poids appris complets (pas de moyenne)
        if res.w_final is not None:
            w_run = res.w_final.copy()
        history.append((epoch, round(iso, 4), round(i_ext, 1),
                        round(depth, 2)))

    w_return = best[3] if best[3] is not None else w.astype(np.float64).copy()
    return ISOLearningResult(w_initial=w, w_final=w_return,
                             best_iso=best[0] if np.isfinite(best[0])
                             else -1.0, best_i_ext=best[1] if np.isfinite(best[1])
                             else float(np.mean(i_ext_range)),
                             best_depth=best[2] if best[2] is not None
                             else float(np.mean(depth_range)),
                             history=history)
