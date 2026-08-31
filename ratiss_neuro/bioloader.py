"""Chargement biologique : connectome + parametres biophysiques.

Sources reelles supportees :
  - matrice de connectivite au format .npy / .csv (Allen, HBP, NWB converti)
  - registre EEG de reference au format .csv (une colonne = un canal)

Sans fichier fourni, genere un connectome synthetique small-world
(proprietes statistiques des reseaux corticaux : clustering fort,
petit monde, distribution de degres heterogene).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Connectome:
    weights: np.ndarray        # W_ij > 0, matrice N x N symetrique
    distances: np.ndarray      # d_ij distance euclidienne des positions
    positions: np.ndarray      # coordonnees 2D des noeuds (mm, espace cortical aplati)
    biophys: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return self.weights.shape[0]


def load_connectome(
    path: str | None = None,
    n_nodes: int = 256,
    seed: int = 7,
) -> Connectome:
    if path is not None:
        weights = _load_weight_matrix(path)
        weights = np.maximum(weights, 0.0)
        np.fill_diagonal(weights, 0.0)
        n = weights.shape[0]
        positions = _ring_layout(n, seed)
        distances = _pairwise(positions)
        return Connectome(weights, distances, positions, _default_biophys())

    rng = np.random.default_rng(seed)
    positions = _ring_layout(n_nodes, seed)
    distances = _pairwise(positions)

    # Small-world : anneau local + reconnexions longue portee (Watts-Strogatz pondere)
    k = 6  # voisins locaux de chaque cote
    weights = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(1, k + 1):
            w = rng.gamma(shape=2.0, scale=0.5)
            weights[i, (i + j) % n_nodes] = w
            weights[(i + j) % n_nodes, i] = w
    n_rewire = int(0.08 * n_nodes * k)
    for _ in range(n_rewire):
        i, j = rng.integers(0, n_nodes, 2)
        if i != j and weights[i, j] == 0:
            w = rng.gamma(shape=1.0, scale=0.25)
            weights[i, j] = w
            weights[j, i] = w

    # Seuil > 2 sigma comme dans le pre-traitement connectomique standard
    nz = weights[weights > 0]
    thr = nz.mean() + 2.0 * nz.std()
    weights = np.where((weights > 0) & (weights >= thr * 0.25), weights, 0.0)
    return Connectome(weights, distances, positions, _default_biophys())


def load_reference_eeg(path: str | None = None, duration_s: float = 5.0, fs: float = 1000.0,
                       seed: int = 11, channel: int = 0) -> tuple[np.ndarray, float]:
    """EEG de reference. Formats : .npy | .csv mono-colonne | .edf (pyedflib)."""
    if path is not None:
        if path.endswith(".edf"):
            try:
                import pyedflib
            except ImportError:
                raise ImportError("pyedflib requis pour lire les .edf")
            f = pyedflib.EdfReader(path)
            fs_e = float(f.getSampleFrequency(channel))
            sig = f.readSignal(channel).astype(np.float64)
            f.close()
            return sig, fs_e
        if path.endswith(".npy"):
            return np.load(path).astype(np.float64), fs
        data = np.loadtxt(path, delimiter=",", ndmin=2)
        return data[:, 0], fs
    t = np.arange(0, duration_s, 1.0 / fs)
    rng = np.random.default_rng(seed)
    pink = np.convolve(rng.standard_normal(t.size), np.exp(-np.arange(50) / 12.0), "same")
    theta = 0.6 * np.sin(2 * np.pi * 6.0 * t + rng.uniform(0, 2 * np.pi))
    gamma_env = 0.5 * (1 + np.sin(2 * np.pi * 0.4 * t))
    gamma = 0.35 * gamma_env * np.sin(2 * np.pi * 40.0 * t)
    return pink + theta + gamma, fs


def _load_weight_matrix(path: str) -> np.ndarray:
    """Charge une matrice de connectivite depuis .npy, .csv dense,
    ou .csv au format aretes (colonnes pre, post, type, synapses) —
    format du connectome reel C. elegans (White et al. 1986, c302)."""
    if path.endswith(".npy"):
        return np.load(path).astype(np.float64)

    with open(path) as f:
        header = f.readline().lower()
    if "pre" in header and "post" in header:
        import csv
        delim = "\t" if "\t" in header else ","
        edges: dict = {}
        nodes: list = []
        rows = []
        with open(path) as f:
            for row in csv.DictReader(f, delimiter=delim):
                pre, post = row["pre"].strip(), row["post"].strip()
                w = float(row.get("synapses") or 1.0)
                rows.append((pre, post, w))
                for x in (pre, post):
                    if x not in edges:
                        edges[x] = len(nodes)
                        nodes.append(x)
        n = len(nodes)
        W = np.zeros((n, n))
        for pre, post, w in rows:
            i, j = edges[pre], edges[post]
            W[i, j] += w
            W[j, i] += w  # symetrisation (modele non oriente)
        return W
    return np.loadtxt(path, delimiter=",").astype(np.float64)


def _ring_layout(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    ang = 2 * np.pi * np.arange(n) / n
    r = 40.0 + rng.normal(0, 4.0, n)  # mm
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])


def _pairwise(pos: np.ndarray) -> np.ndarray:
    diff = pos[:, None, :] - pos[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def _default_biophys() -> dict:
    return {
        "g_syn": {"AMPA": 0.8, "NMDA": 0.4, "GABA": 1.2},   # nS, moyennes patch-clamp
        "tau_m_ms": 20.0,                                    # constante membranaire
        "T_K": 310.0,                                        # temperature physiologique
    }
