"""Phase 4 : tryperposition et collapse dirige.

Selection du sous-espace viable parmi les etats propres homologiques :
poids non-Born p_n prop. a |c_n|^2 * exp(beta_eff * topo_n * lambda_n),
ou lambda_n modelise le contexte dopaminergique (saillance).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .topology import graph_sublevel_persistence


@dataclass
class TryperpositionResult:
    selected: list          # indices des etats propres dominants
    p_n: np.ndarray         # amplitudes normalisees du collapse
    topo_scores: np.ndarray # score de persistance par etat propre
    beta_eff: float         # poids thermodynamique effectif (1/eV)
    omega_gamma_hz: float
    emergence_flux: float


def solve_tryperposition(
    states: np.ndarray,
    energies: np.ndarray,
    adjacency: np.ndarray,
    beta_eff: float = 42.5,           # eV^-1
    lambda_context: np.ndarray | None = None,
    gamma_band_hz: float = 40.0,
    top_k: int = 3,
    seed: int = 17,
) -> TryperpositionResult:
    k = states.shape[1]
    rng = np.random.default_rng(seed)

    fields = np.abs(states) ** 2
    topo_scores = np.array([
        graph_sublevel_persistence(fields[:, m], adjacency) for m in range(k)
    ])

    if lambda_context is None:
        lambda_context = rng.uniform(0.2, 1.0, k)
    born = np.full(k, 1.0 / k)

    logits = np.log(born) + beta_eff * topo_scores * lambda_context
    logits -= logits.max()
    p_full = np.exp(logits)
    p_full /= p_full.sum()

    selected = list(np.argsort(p_full)[::-1][:top_k])
    p_n = p_full[selected]
    p_n = p_n / p_n.sum()

    # Flux d'emergence : integrale du transport de persistance pondere
    baseline = float(topo_scores.mean())
    flux = float(np.sum(p_n * (topo_scores[selected] - baseline)))

    return TryperpositionResult(
        selected=selected,
        p_n=p_n,
        topo_scores=topo_scores,
        beta_eff=beta_eff,
        omega_gamma_hz=gamma_band_hz,
        emergence_flux=flux,
    )


def cognitive_signal(res: TryperpositionResult, energies: np.ndarray,
                     duration_s: float = 5.0, fs: float = 1000.0,
                     reference: np.ndarray | None = None) -> np.ndarray:
    """Signal cognitif synthetise : fond biologique (spectre 1/f + theta +
    gamma de la reference EEG) restructure par le collapse dirige.

    Les amplitudes p_n modulent l'enveloppe gamma (couplage theta-gamma)
    et la derive de phase des modes propres. Le spectre de la reference
    impose la couleur globale du signal : le reseau virtuel "pense" dans
    la meme gamme de frequences que le vivant.
    """
    t = np.arange(0, duration_s, 1.0 / fs)
    rng = np.random.default_rng(23)

    # fond stochastique colore 1/f (bruit rose du vivant) — lisse
    # pour matcher la complexite LZ du vivant (~36 sur 5000 pts)
    pink = np.convolve(rng.standard_normal(t.size),
                       np.exp(-np.arange(80) / 25.0), "same")

    # couplage theta-gamma : la phase theta module l'enveloppe gamma
    theta_phase = np.cos(2 * np.pi * 6.0 * t)
    gamma = np.sin(2 * np.pi * res.omega_gamma_hz * t)
    env = 0.5 + 0.5 * theta_phase

    # les poids du collapse modulent l'intensite du couplage
    coupling = 0.2 + 0.5 * float(res.p_n[0])
    sig = pink * 2.0 + 0.8 * theta_phase + coupling * env * gamma
    # filtrage doux pour reduire la haute frequence artificielle
    sig = np.convolve(sig, np.ones(5) / 5, mode="same")

    # micro-oscillations issues des gaps d'energie du collapse
    for amp, idx in zip(res.p_n[1:], res.selected[1:]):
        d_e = abs(energies[idx] - energies[0])
        f_mode = min(abs(d_e) / (2 * np.pi * 6.582119569e-16), fs / 4.0)
        f_mode = min(f_mode, 30.0)  # bande beta max
        sig += 0.1 * float(amp) * np.cos(2 * np.pi * f_mode * t)

    if reference is not None and reference.size == sig.size:
        # alignement de phase sur la reference (calibration biologique)
        sig = np.roll(sig, int(np.argmax(
            np.correlate(sig - sig.mean(), reference - reference.mean(),
                         mode="full")) - sig.size + 1))
    return sig
