"""Phase 1 : construction de l'Hamiltonien fermionique H_cog.

H = -sum_ij t_ij e^{i phi_ij} c_i^+ c_j  +  sum_i U_i n_i

  - hopping  t_ij = W_ij * exp(-d_ij / lambda)
  - phase de Peierls phi_ij : approximation U(1) de l'holonomie SU(2)
    du champ EM endogene (derive MEG) sur le graphe G_bio
  - Hubbard U_i : saturation synaptique (courbe STDP) en potentiel onsite
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .bioloader import Connectome


def build_hamiltonian(
    con: Connectome,
    t0: float = 1.0,          # eV, echelle d'energie du hopping
    lam_decay: float = 25.0,  # mm, longueur de decroissance spatiale
    berry_flux: float = 0.35, # flux effectif par unite d'aire (rad / mm^2 x 1e-3)
    u0: float = 0.6,          # eV, amplitude de saturation synaptique
) -> sp.csr_matrix:
    W, D, pos = con.weights, con.distances, con.positions
    n = con.n_nodes
    ii, jj = np.nonzero(W)

    t_hop = t0 * W[ii, jj] * np.exp(-D[ii, jj] / lam_decay)

    # phase de Peierls : circulation du potentiel vecteur sur le triangle (0, i, j)
    # antisymetrique en i <-> j pour garantir l'hermiticite
    cross = pos[ii, 0] * pos[jj, 1] - pos[ii, 1] * pos[jj, 0]
    phi = berry_flux * 1e-3 * 0.5 * cross
    phase = np.exp(1j * phi)

    degree = np.asarray((W > 0).sum(axis=1)).ravel()
    onsite = u0 * 1.0 / (1.0 + np.exp(-(degree - degree.mean()) / (degree.std() + 1e-9)))

    H = sp.csr_matrix((-t_hop * phase, (ii, jj)), shape=(n, n), dtype=np.complex128)
    H = H + sp.diags(onsite.astype(np.complex128))
    return H.tocsr()
