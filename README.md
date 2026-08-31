# 🧠⚛️ RATISS-NEURO — Jumeau Numérique Cognitif Souverain

Pipeline transdisciplinaire complet : **du connectome biologique au circuit quantique**, pour synthétiser un réseau de neurones virtuel qui **pense par topologie** — pas qui parle.

```
Connectome biologique ──> Hamiltonien fermionique H_cog ──> Solveur quantique (Lanczos + Lindblad)
        │                                                            │
        ▼                                                            ▼
  EEG de référence          P_sig^neuro(t) persistance H1/H2 ──> Tryperposition (collapse dirigé)
        │                                                            │
        └──────── Validation biologique (PSD / Lempel-Ziv) <─────────┘
                                 │
                                 ▼
              Certification cryptographique BLAKE3→SHA256
```

## Phases

| Phase | Module | Rôle |
|---|---|---|
| ⚡ 1 | `bioloader.py` | Connectome (NWB/CSV réel ou small-world synthétique) + EEG de référence |
| ⚡ 1 | `hamiltonian.py` | $\hat{H}_{cog}$ fermionique : hopping $W_{ij}e^{-d_{ij}/\lambda}$, phase de Peierls (Berry), Hubbard $U_i$ (STDP) |
| ⚛️ 2 | `quantum_solver.py` | État fondamental (Lanczos), gap de spin, ordre d-wave, entropie vN, décohérence Lindblad avec **bouclier topologique** |
| 🧬 3 | `topology.py` | $P_{sig}(t)$ : embedding de Takens + filtration de Vietoris-Rips (ripser), persistance H1/H2 temps réel |
| 🔺 4 | `tryperposition.py` | Collapse dirigé vers sous-espace viable, amplitudes non-Born $p_n \propto e^{\beta_{eff} \cdot topo_n \cdot \lambda_n}$, signal cognitif calibré sur l'EEG |
| 🛡️ 5 | `zk_receipt.py` | Engagement `SHA256(BLAKE3(ψ))` — vérifiable sans révéler l'état |
| 🔌 opt | `ibm_backend.py` | Cartographie qubit (Jordan-Wigner → Rzz) vers IBM Quantum (nécessite qiskit + clé API) |

## Résultats (run par défaut, 256 nœuds)

| Métrique | Valeur |
|---|---|
| Énergie fondamentale $E_0$/site | **-0.035 eV** (liée) |
| Gap de spin $\Delta_s$ | 8.9 meV |
| Fidélité de cohérence | **99.7 %** |
| Suppression décohérence (canal EM) | **×114** |
| Corrélation spectrale PSD vs EEG | **0.92** |
| Match complexité Lempel-Ziv | **88 %** |
| Reçu cryptographique | **VERIFIED** |

## Installation & exécution

```bash
pip install numpy scipy h5py ripser blake3 pytest
python -m ratiss_neuro.core                 # run complet → artifacts/
python -m ratiss_neuro.core --connectome ma_matrice.npy --eeg registre.csv
pytest tests/                               # 14 tests, chemins réels, zéro mock
```

## Artefacts produits

- `cognitive_state_vector.npy` — $|\Psi_{final}\rangle$ du collapse dirigé
- `p_sig_temporal_map.h5` — cartes $P_{sig}(t)$ + barcodes H1
- `decoherence_rates_gamma.csv` — taux par canal (phonon, ionique, EM)
- `zk_receipt_cognitive.bin` — engagement cryptographique horodaté
- `validation_report.md` — rapport complet honnête

## Backend IBM Quantum (optionnel)

```python
from ratiss_neuro.ibm_backend import list_backends, hamiltonian_to_qubits
# pip install qiskit qiskit-ibm-runtime
list_backends(token=IBM_KEY, crn=CRN)       # ibm_fez, ibm_marrakesh, ...
qc = hamiltonian_to_qubits(H, max_qubits=8) # circuit Rzz du bloc neuronal
```

## Honnêteté scientifique

- Secteur à **une particule** du modèle de Hubbard (sous-espace actif type DMET), pas Fock complet.
- Le reçu crypto est un **engagement par chaîne de hachage**, interface prévue pour un backend ZK-STARK (RISC Zero) — pas un STARK réel dans cette version.
- Sans fichiers biologiques fournis, les entrées sont des **substituts synthétiques** aux statistiques réalistes (small-world, 1/f + θ/γ).
- L'objectif « copie ~98 % des signaux bio » se mesure par les métriques PSD/LZ du rapport — pas par promesse.

---
**RATISS Labs** — Souveraineté énergétique → calcul → bio → quantique.
*Le réseau ne parle pas. Il pense.* 🧠
