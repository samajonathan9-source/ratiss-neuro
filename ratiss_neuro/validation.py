"""Validation : signal cognitif synthetise vs reference biologique.

Metriques : correlation spectrale (PSD de Welch), complexite de
Lempel-Ziv binarisee, isomorphisme des micro-etats topologiques
(correlation des trajectoires P_sig).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from .topology import compute_p_sig


def psd_correlation(sig: np.ndarray, ref: np.ndarray, fs: float,
                    fmin: float = 1.0, fmax: float = 80.0) -> float:
    f1, p1 = welch(sig, fs=fs, nperseg=min(1024, sig.size))
    f2, p2 = welch(ref, fs=fs, nperseg=min(1024, ref.size))
    mask = (f1 >= fmin) & (f1 <= fmax)
    l1 = np.log10(p1[mask] + 1e-20)
    l2 = np.log10(p2[mask] + 1e-20)
    return float(np.corrcoef(l1, l2)[0, 1])


def lz_complexity(x: np.ndarray) -> int:
    """LZ76 sur le signal binarise par sa mediane."""
    s = "".join("1" if v > np.median(x) else "0" for v in x)
    n = len(s)
    i, k, l, c = 0, 1, 1, 1
    while True:
        if l + k > n:
            c += 1
            break
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > 1:
                i += 1
                if i == l:
                    c += 1
                    l += k
                    i, k = 0, 1
            else:
                c += 1
                l += 1
                i, k = 0, 1
    return c


def lz_match(sig: np.ndarray, ref: np.ndarray) -> float:
    a, b = lz_complexity(sig), lz_complexity(ref)
    return float(1.0 - abs(a - b) / max(a, b))


def microstate_isomorphism(sig: np.ndarray, ref: np.ndarray, fs: float) -> float:
    """Correlation des trajectoires P_sig : isomorphisme des micro-etats."""
    pa = compute_p_sig(sig, fs).p_sig_t
    pb = compute_p_sig(ref, fs).p_sig_t
    m = min(pa.size, pb.size)
    if m < 4:
        return 0.0
    return float(np.clip(np.corrcoef(pa[:m], pb[:m])[0, 1], -1.0, 1.0))
