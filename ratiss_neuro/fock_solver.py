"""Solveur espace de Fock complet pour H_cog (tous les secteurs de
particule), via quspin (diagonalisation brute en spin-pLiens /
fermions sur le graphe connectomique).

Complementaire a quantum_solver.solve_quantum_hybrid (secteur 1
particule). Ici, le Hamiltonien hopping est resolu sur l'espace de
Fock complet 2^N via la representation des spins-1/2 (chaines
XY/Ising sur le graphe), ou Hubbard U devient une interaction ZZ.

Pour N > ~18 sites, la dimension 2^N explose : on reste en solveur
Lanczos sur secteur ou en approximation DMET. Le module fonctionne
reellement pour les blocs tractables et sert de reference d'exactitude.
"""

from __future__ import annotations

import numpy as np

try:
    from quspin.operators import hamiltonian
    from quspin.basis import spin_basis_1d
    HAS_QUSPIN = True
except ImportError:
    HAS_QUSPIN = False


def solve_fock_exact(H_single, k: int = 6, u_onsite: float = 0.0) -> dict:
    """Diagonalise le Hamiltonien dans l'espace de Fock complet des
    spins-1/2 associes au reseau (Jordan-Wigner du modele hopping).

    H_single : matrice hermitienne (N, N) de hopping (secteur 1
    particule).
    Retourne les k energies les plus basses E_n/(site) et le gap.
    """
    if not HAS_QUSPIN:
        raise ImportError("quspin requis : pip install quspin")

    Hs = np.asarray(H_single.toarray() if hasattr(H_single, "toarray") else H_single)
    N = Hs.shape[0]
    if N > 18:
        raise ValueError(f"Fock complet demande 2^N dim ; N={N} > 18 trop grand — "
                         "utiliser solve_quantum_hybrid (secteur)")

    # hopping XY entre sites : (J/2)(c+ c- + c- c+) -> termes xx + yy
    # + potentiel onsite -> termes z  + Hubbard U -> interaction zz
    J_xy, J_z, h_fields = [], [], []
    for i in range(N):
        for j in range(i + 1, N):
            c = float(Hs[i, j].real)
            if abs(c) > 1e-12:
                J_xy.append([c, i, j])            # xx et yy (conservent Sz tot)
        h_fields.append([float(Hs[i, i].real), i])

    # couple xx et yy avec meme amplitude (modele XX isotrope)
    static = [["xx", J_xy], ["yy", J_xy], ["z", h_fields]]
    if u_onsite != 0.0:
        static.append(["zz", [[u_onsite, i, j] for i in range(N)
                              for j in range(i + 1, N) if abs(Hs[i, j]) > 1e-12]])

    basis = spin_basis_1d(L=N)
    Hop = hamiltonian(static, [], basis=basis, dtype=np.float64)
    Ediag = Hop.eigsh(k=k, which="SA")
    E = Ediag[0]

    return {
        "n_states": int(basis.Ns),
        "eigenvalues_per_site": E / N,
        "ground_state_e0_per_site": float(E[0] / N),
        "gap": float(E[1] - E[0]) if k > 1 else np.nan,
        "fock_complete": True,
    }


def compare_hybrid_vs_fock(H_single, n_block: int = 12) -> dict:
    """Compare hybride (secteur 1 particule) et Fock complet (2^n) sur le
    MEME bloc principal de n_block sites, pour quantifier l'erreur
    du sous-espace tractable."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh

    Hs = csr_matrix(H_single)
    block = Hs[:n_block, :n_block].toarray()

    # solveur hybride : spectre du bloc en secteur 1 particule (eigsh complet)
    e, v = eigsh(csr_matrix(block), k=min(6, block.shape[0] - 1), which="SA")
    hybrid_e0 = float(e[0] / block.shape[0])

    fock = solve_fock_exact(block, k=min(6, block.shape[0]))
    fock_e0 = fock["eigenvalues_per_site"][0]
    err = abs(hybrid_e0 - fock_e0)
    return {
        "hybrid_e0_per_site": hybrid_e0,
        "fock_e0_per_site": float(fock_e0),
        "absolute_error_per_site": float(err),
        "fock_hilbert_dim": fock["n_states"],
    }
