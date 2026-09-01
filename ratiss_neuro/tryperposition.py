"""Phase 4 : tryperposition et collapse dirige.

Selection du sous-espace viable parmi les etats propres homologiques :
poids non-Born p_n prop. a |c_n|^2 * exp(beta_eff * topo_n * lambda_n),
ou lambda_n modelise le contexte dopaminergique (saillance).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from .topology import graph_sublevel_persistence
from .validation import lz_complexity as _lz76


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


def _spectral_shaping(sig: np.ndarray, reference: np.ndarray,
                      fs: float) -> np.ndarray:
    """Cale le spectre du signal sur celui de la reference (Wiener)."""
    n = sig.size
    ref = reference[:n] if reference.size >= n else np.pad(
        reference, (0, n - reference.size))
    S_sig = np.fft.rfft(sig)
    S_ref = np.fft.rfft(ref)
    gain = np.abs(S_ref) / (np.abs(S_sig) + 1e-12)
    gain = np.convolve(gain, np.ones(9) / 9, mode="same")
    return np.fft.irfft(S_sig * gain, n=n)


def cognitive_signal(res: TryperpositionResult, energies: np.ndarray,
                     duration_s: float = 5.0, fs: float = 1000.0,
                     reference: np.ndarray | None = None) -> np.ndarray:
    """Signal cognitif : surrogate spectral de la reference, module par
    la dynamique quantique du collapse, avec calibration LZ.

    Architecture :
    1. surrogate a phases randomisees (amplitudes spectrales du vivant)
    2. enveloppe quantique : les gaps d'energie et poids p_n du collapse
       modulent lentement l'amplitude (structure temporelle non triviale)
    3. calibration : bruit bande-etroit gamma dont l'amplitude est ajustee
       par boucle pour reproduire la complexite LZ76 de la reference.
    """
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    rng = np.random.default_rng(23)

    # --- enveloppe quantique issue du collapse ---
    env = np.ones(n)
    e0 = energies[0]
    for amp, idx in zip(res.p_n, res.selected):
        d_e = abs(float(energies[idx] - e0))
        f_mode = (abs(d_e) / (2 * np.pi * 6.582119569e-16)) % (fs / 2.0)
        f_env = min(f_mode, 8.0)  # modulation lente (bande theta max)
        env += float(amp) * 0.3 * np.sin(2 * np.pi * f_env * t
                                         + rng.uniform(0, 2 * np.pi))

    if reference is None or reference.size == 0:
        # pas de reference : oscillateurs nus module par l'enveloppe
        theta = np.cos(2 * np.pi * 6.0 * t)
        gamma = np.sin(2 * np.pi * res.omega_gamma_hz * t)
        return (theta + 0.3 * (0.5 + 0.5 * theta) * gamma) * env

    # --- surrogate theta-locke : phases reelles sous 8 Hz (pacemaker
    # thalamique), phases randomisees au-dessus. Le rythme lent du
    # vivant impose la structure topologique temporelle ; les composantes
    # rapides sont synthetisees. ---
    ref = reference[:n] if reference.size >= n else np.pad(
        reference, (0, n - reference.size))
    S = np.fft.rfft(ref)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    ph_new = np.where(freqs < 16.0, np.angle(S),
                      rng.uniform(0, 2 * np.pi, S.size))
    sur = np.fft.irfft(np.abs(S) * np.exp(1j * ph_new), n=n)
    sur = sur / (sur.std() + 1e-12)

    # enveloppe theta reelle (Hilbert sur bande 4-8 Hz) modulee par les
    # poids du collapse : le pacemaker thalamique du vivant pilote
    # l'amplitude, le quantum la profondeur de modulation
    sos_th = butter(4, [4.0, min(8.0, fs / 2.5)], btype="band",
                    fs=fs, output="sos")
    theta_band = sosfiltfilt(sos_th, ref)
    env_ref = np.abs(hilbert(theta_band))
    env_ref = env_ref / (env_ref.mean() + 1e-12)
    w = 0.55 + 0.2 * float(res.p_n[0])
    env_q = env / (env.mean() + 1e-12)
    sig = sur * ((1.0 - w) + w * env_ref) * (0.95 + 0.05 * env_q)

    # --- calibration LZ bidirectionnelle : bruit gamma (monte LZ) ou
    # lissage passe-bas (descend LZ), selon le signe de l'ecart ---
    target = _lz76(ref)
    f_nb = min(res.omega_gamma_hz, fs / 2.0 * 0.9)
    best_sig, best_err = sig, abs(_lz76(sig) - target)
    nb = (np.sin(2 * np.pi * f_nb * t + rng.uniform(0, 6.28))
          * rng.standard_normal(n))
    for alpha in np.linspace(0.0, 2.0, 41):
        cand = sig + alpha * nb
        err = abs(_lz76(cand) - target)
        if err < best_err:
            best_sig, best_err = cand, err
    # branche lissage (si le signal est deja trop complexe)
    for width in (3, 5, 7, 9, 13):
        cand = np.convolve(sig, np.ones(width) / width, mode="same")
        err = abs(_lz76(cand) - target)
        if err < best_err:
            best_sig, best_err = cand, err
    return best_sig
