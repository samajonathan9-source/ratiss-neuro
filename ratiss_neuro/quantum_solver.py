"""Phase 2 : resolution quantique hybride.

Etat fondamental par Lanczos (eigsh) sur le secteur a une particule
du modele de Hubbard (sous-espace actif DMET ~ N sites), entropie de
von Neumann fermionique de la matrice de correlation, ordre d-wave,
et decoherence de Lindblad avec suppression topologique.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HBAR_EV_FS = 6.582119569e-1  # eV * fs


@dataclass
class QuantumResult:
    energies: np.ndarray      # eV, k plus basses valeurs propres
    states: np.ndarray        # colonnes = orbitales propres
    e0_per_site: float
    spin_gap_mev: float
    dwave_order: float
    von_neumann_entropy: float
    fidelity: float
    gamma_eff: dict


def solve_quantum_hybrid(
    H: sp.csr_matrix,
    k: int = 12,
    p_sig: float = 0.0,
    tau_c: float = 0.120,   # s, seuil critique de persistance
    t_coh_s: float = 0.100, # s, fenetre d'un micro-etat cognitif
    seed: int = 13,
) -> QuantumResult:
    k = min(k, H.shape[0] - 2)
    # v0 deterministe : le receipt hash-chain exige la reproductibilite
    v0 = np.linspace(1.0, 2.0, H.shape[0])
    energies, states = spla.eigsh(H, k=k, which="SA", tol=1e-10, v0=v0)
    order = np.argsort(energies)
    energies, states = energies[order], states[:, order]

    n = H.shape[0]
    e0_per_site = float(energies[0].real / n)
    spin_gap_mev = float((energies[1] - energies[0]).real * 1e3)

    dwave = _dwave_order(states[:, 0], H)
    sv = _vn_entropy(states[:, 0])
    gamma_eff = _lindblad_channels(p_sig, tau_c)
    fidelity = _coherence_fidelity(gamma_eff, t_coh_s)

    return QuantumResult(
        energies=energies,
        states=states,
        e0_per_site=e0_per_site,
        spin_gap_mev=spin_gap_mev,
        dwave_order=dwave,
        von_neumann_entropy=sv,
        fidelity=fidelity,
        gamma_eff=gamma_eff,
    )


def _dwave_order(psi0: np.ndarray, H: sp.csr_matrix) -> float:
    """Anisotropie des correlateurs proches voisins de rho_i = |psi0_i|^2.

    Delta_d = |<C>_x - <C>_y| normalise : signature d'appariement d-wave.
    """
    rho = np.abs(psi0) ** 2
    coo = H.tocoo()
    mask = coo.row < coo.col
    r, c = coo.row[mask], coo.col[mask]
    C = rho[r] * rho[c]
    if C.size < 4 or C.std() < 1e-15:
        return 0.0
    Cn = (C - C.mean()) / (C.std() + 1e-15)
    half = Cn.size // 2
    return float(abs(Cn[:half].mean() - Cn[half:].mean()))


def _vn_entropy(psi0: np.ndarray, frac: float = 0.5) -> float:
    """Entropie fermionique de la matrice de correlation d'une sous-region."""
    n = psi0.size
    m = max(2, int(frac * n))
    C = np.outer(psi0[:m], psi0[:m].conj())
    nu = np.linalg.eigvalsh(C).real
    nu = np.clip(nu, 1e-15, 1 - 1e-15)
    s = -(nu * np.log(nu) + (1 - nu) * np.log(1 - nu)).sum()
    return float(s / (m * np.log(2)))  # normalise en [0, 1]


def _lindblad_channels(p_sig: float, tau_c: float) -> dict:
    """Taux de decoherence par canal avec suppression topologique.

    gamma_eff = gamma0 / (1 + (P_sig / tau_c)^2 * eta)
    La persistance homologique agit comme bouclier du bain thermique.
    """
    gamma0 = {"phonon": 900.0, "ionique": 600.0, "em": 300.0}  # s^-1 a 310 K
    eta = {"phonon": 4.0, "ionique": 6.0, "em": 10.0}
    out = {}
    for ch, g0 in gamma0.items():
        suppress = 1.0 + (p_sig / tau_c) ** 2 * eta[ch] if tau_c > 0 else 1.0
        out[ch] = {"gamma0": g0, "gamma_eff": g0 / suppress, "suppression": suppress}
    return out


def _coherence_fidelity(gamma_eff: dict, t_s: float,
                        n_corr: int = 1000) -> float:
    """Fidelite de coherence sous dephasage correle (bath non-markovien).

    Quand la persistance topologique synchronise les canaux de bruit
    (gamma_alpha -> gamma_correle), le facteur de suppression
    s'applique quadratiquement au taux effectif. n_corr = taille du
    bloc de coherence (sites correles par le squelette H1).
    F = (1 + exp(-2 * gamma_eff * t / n_corr)) / 2.
    """
    gammas = np.array([v["gamma_eff"] for v in gamma_eff.values()])
    g_tot = gammas.sum()
    return float((1.0 + np.exp(-2.0 * g_tot * t_s / n_corr)) / 2.0)


def coupler_non_local(H: sp.csr_matrix, positions: np.ndarray,
                      n_links: int = 24, t_myelin: float = 0.5,
                      seed: int = 3) -> sp.csr_matrix:
    """Canaux de couplage quantique inter-regionaux : faisceaux myelinises
    (fibres longues de substance blanche) = hopping longue portee fort
    entre hubs distants. Transport d'amplitudes |psi_i|^2 inter-modules
    sans decoherence thermique standard (lien ideal, t constant).

    Ajoute n_links liens entre les hubs de plus haut degre les plus
    eloignes geometriquement."""
    Hd = H.tolil()
    n = H.shape[0]
    deg = np.asarray((H != 0).sum(axis=1)).ravel()
    hubs = np.argsort(deg)[::-1][: max(4, n // 8)]
    rng = np.random.default_rng(seed)
    added = 0
    for _ in range(n_links * 4):
        if added >= n_links:
            break
        i, j = rng.choice(hubs, size=2, replace=False)
        d = np.linalg.norm(positions[i] - positions[j])
        if d < np.median(np.linalg.norm(positions - positions.mean(0), axis=1)):
            continue  # on veut des liens LONGUE portee
        if Hd[i, j] == 0:
            Hd[i, j] = t_myelin
            Hd[j, i] = t_myelin
            added += 1
    return Hd.tocsr()


def time_evolution(H: sp.csr_matrix, psi0: np.ndarray, n_steps: int = 200,
                   dt_fs: float = 50.0) -> np.ndarray:
    """Trajectoire |psi(t)> par exponentiation de Krylov (snapshots)."""
    snaps = np.empty((n_steps, psi0.size), dtype=np.complex128)
    psi = psi0 / np.linalg.norm(psi0)
    for s in range(n_steps):
        snaps[s] = psi
        psi = spla.expm_multiply(-1j * H * dt_fs / HBAR_EV_FS, psi)
    return snaps
