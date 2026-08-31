"""Phase 3 : signature topologique P_sig^neuro(t) en temps reel.

Embedding de Takens du signal projete, filtration de Vietoris-Rips
(ripser), score de persistance H1/H2 par fenetre glissante.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ripser import ripser


@dataclass
class TopoResult:
    p_sig_t: np.ndarray        # score par fenetre, dans [0, 1]
    p_sig_peak: float
    h1_lifetime_ms: float
    n_h2_cavities: int
    barcodes: list


def takens_embed(x: np.ndarray, dim: int = 3, delay: int = 8) -> np.ndarray:
    n = x.size - (dim - 1) * delay
    if n < 8:
        raise ValueError("signal trop court pour l'embedding")
    return np.column_stack([x[i * delay: i * delay + n] for i in range(dim)])


def window_persistence(cloud: np.ndarray, maxdim: int = 1) -> dict:
    return ripser(cloud, maxdim=maxdim)


def persistence_score(dgm: np.ndarray) -> float:
    if dgm.size == 0:
        return 0.0
    pers = dgm[:, 1] - dgm[:, 0]
    pers = pers[np.isfinite(pers)]
    if pers.size == 0:
        return 0.0
    return float(np.clip(pers.sum() / (pers.size * pers.max() + 1e-12) - 1.0 / pers.size, 0.0, 1.0)
                 if pers.max() > 0 else 0.0)


def compute_p_sig(
    signal: np.ndarray,
    fs: float,
    window_s: float = 0.5,
    hop_s: float = 0.05,
    max_points: int = 120,
) -> TopoResult:
    win, hop = int(window_s * fs), int(hop_s * fs)
    starts = list(range(0, signal.size - win + 1, hop))
    scores, barcodes, lifetimes = [], [], []
    n_h2 = 0
    for s in starts:
        seg = signal[s: s + win]
        seg = (seg - seg.mean()) / (seg.std() + 1e-12)
        cloud = takens_embed(seg)
        if cloud.shape[0] > max_points:
            idx = np.linspace(0, cloud.shape[0] - 1, max_points).astype(int)
            cloud = cloud[idx]
        res = window_persistence(cloud, maxdim=1)
        dgm1 = res["dgms"][1]
        barcodes.append(dgm1)
        scores.append(persistence_score(dgm1))
        finite = dgm1[np.isfinite(dgm1[:, 1])]
        if finite.size:
            scale_ms = window_s * 1000.0 / (np.ptp(cloud[:, 0]) + 1e-12)
            lifetimes.append(float(np.mean(finite[:, 1] - finite[:, 0]) * scale_ms))
        if cloud.shape[0] <= 60:
            dgm2 = window_persistence(cloud, maxdim=2)["dgms"]
            if len(dgm2) > 2 and dgm2[2].size:
                n_h2 = max(n_h2, int(np.sum(np.isfinite(dgm2[2][:, 1]))))

    p_sig_t = np.asarray(scores)
    return TopoResult(
        p_sig_t=p_sig_t,
        p_sig_peak=float(p_sig_t.max()) if p_sig_t.size else 0.0,
        h1_lifetime_ms=float(np.mean(lifetimes)) if lifetimes else 0.0,
        n_h2_cavities=n_h2,
        barcodes=barcodes,
    )


def graph_sublevel_persistence(field: np.ndarray, adjacency: np.ndarray) -> float:
    """Score H0 de la filtration sous-niveau de |psi|^2 sur le graphe
    (union-find) : longevite de la composante dominante d'un etat propre."""
    n = field.size
    order = np.argsort(field)
    parent = np.arange(n)
    birth = np.full(n, np.inf)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    max_pers = 0.0
    for idx in order:
        birth[idx] = field[idx]
        for j in np.nonzero(adjacency[idx])[0]:
            if field[j] <= field[idx]:
                ri, rj = find(idx), find(j)
                if ri != rj:
                    elder = ri if birth[ri] <= birth[rj] else rj
                    younger = rj if elder == ri else ri
                    max_pers = max(max_pers, field[idx] - birth[younger])
                    parent[younger] = elder
    rng = field.max() - field.min()
    return float(np.clip(max_pers / (rng + 1e-12), 0.0, 1.0))
