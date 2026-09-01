"""TopologyCompressor : Betti H1 temps reel guide la plasticite.

Regle homologique RATISS : naissance d'un cycle H1 non-trivial
(assemblee cellulaire emergente) dans les spikes -> potentialiation
structurelle des synapses participantes. Mort d'un cycle -> depression.

La persistence H1 sur graphe est calculee de facon rapide (tractable
temps reel) : seuillage du graphe de correlation des neurones
sentinelles, puis cycles hors-arbre = E - (V - 1) via Kruskal /
union-find. Chaque arc non-arbre ferme un cycle H1 independant.
"""

from __future__ import annotations

import numpy as np


def h1_mask_from_spikes(spikes_window: np.ndarray,
                        n_sentinel: int = 24,
                        seed: int | None = None) -> np.ndarray:
    """Detecte les cycles H1 dans la fenetre et retourne le masque de
    synapses (i, j) : 1 = synapse potentialisable (siege sur un cycle
    H1 naissant).

    spikes_window : (n_neurons, n_steps) bool ou float.
    Retourne un masque binaire (n_neurons, n_neurons).
    """
    n_neurons = spikes_window.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n_neurons, size=min(n_sentinel, n_neurons),
                             replace=False))
    sub = spikes_window[idx].astype(np.float64)

    # correlation cosine entre neurones sentinelles
    norm = np.linalg.norm(sub, axis=1, keepdims=True) + 1e-12
    corr = (sub @ sub.T) / (norm * norm.T)
    np.fill_diagonal(corr, 0.0)

    # seuillage (median des correlations positives)
    pos = corr[corr > 0]
    thr = np.median(pos) if pos.size else 0.0
    edges = [(corr[i, j], i, j)
             for i in range(len(idx)) for j in range(i + 1, len(idx))
             if corr[i, j] >= thr]
    edges.sort(key=lambda e: -e[0])

    # Kruskal : arcs hors-arbre = cycles H1
    parent = list(range(len(idx)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    mask = np.zeros((n_neurons, n_neurons))
    for _, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
        else:
            mask[idx[i], idx[j]] = 1.0
    return mask


def n_h1_cycles(spikes_window: np.ndarray, n_sentinel: int = 24,
                seed: int | None = None) -> int:
    """Nombre de cycles H1 detectes dans la fenetre (proxy rapide)."""
    return int(h1_mask_from_spikes(spikes_window, n_sentinel, seed).sum())
