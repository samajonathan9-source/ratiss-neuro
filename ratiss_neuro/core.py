"""RATISS-NEURO core : pipeline complet du Jumeau Numerique Cognitif.

Phase 1 : chargement biologique + construction H_cog (fermionique)
Phase 2 : resolution quantique hybride (Lanczos + Lindblad)
Phase 3 : P_sig^neuro(t) par persistance homologique temps reel
Phase 4 : tryperposition et collapse dirige
Phase 5 : certification cryptographique + artefacts + rapport

Usage :  python -m ratiss_neuro.core [--connectome PATH] [--eeg PATH]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

from .bioloader import load_connectome, load_reference_eeg
from .hamiltonian import build_hamiltonian
from .kuramoto import consciousness_threshold, simulate_kuramoto
from .quantum_solver import coupler_non_local, solve_quantum_hybrid
from .replay import replay_offline
from .topology import compute_p_sig
from .iso_learning import learn_iso
from .snn import (AdExParams, build_microcircuit, simulate_snn,
                  snn_to_eeg, theta_drive_from_eeg)
from .tryperposition import cognitive_signal, solve_tryperposition
from .validation import lz_match, microstate_isomorphism, psd_correlation
from .zk_receipt import check_invariants, generate_receipt, verify_receipt


def run_pipeline(
    connectome_path: str | None = None,
    eeg_path: str | None = None,
    out_dir: str = "artifacts",
    n_nodes: int = 256,
    verbose: bool = True,
) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    # ---------- PHASE 1 : chargement biologique + H_cog ----------
    log("\n=== PHASE 1 : CHARGEMENT BIOLOGIQUE & CONSTRUCTION H_cog ===")
    con = load_connectome(connectome_path, n_nodes=n_nodes)
    n_edges = int((con.weights > 0).sum() // 2)
    log(f"[BIO_LOADER] Connectome : {con.n_nodes} noeuds, {n_edges} aretes")
    log(f"[BIO_LOADER] Biophysique : {con.biophys}")
    H = build_hamiltonian(con)
    log(f"[H_BUILDER] H_cog assemble : {H.shape}, nnz={H.nnz}, hermitien={bool(np.allclose((H - H.getH()).data, 0, atol=1e-10))}")

    # ---------- PHASE 3a : P_sig prealable (pilote la decoherence) ----------
    ref_eeg, fs = load_reference_eeg(eeg_path)
    topo_pre = compute_p_sig(ref_eeg, fs)

    # ---------- PHASE 2 : resolution quantique hybride ----------
    log("\n=== PHASE 2 : RESOLUTION QUANTIQUE HYBRIDE ===")
    qres = solve_quantum_hybrid(H, k=12, p_sig=topo_pre.p_sig_peak)
    log(f"[SOLVER] E0/site        = {qres.e0_per_site:.6f} eV")
    log(f"[SOLVER] Gap de spin    = {qres.spin_gap_mev:.3f} meV")
    log(f"[SOLVER] Ordre d-wave   = {qres.dwave_order:.4f}")
    log(f"[SOLVER] Entropie vN    = {qres.von_neumann_entropy:.4f}")
    for ch, g in qres.gamma_eff.items():
        log(f"[LINDBLAD] {ch:8s} : {g['gamma0']:.0f} -> {g['gamma_eff']:.3f} s^-1 (suppression x{g['suppression']:.1f})")
    log(f"[SOLVER] Fidelite coherence = {qres.fidelity * 100:.2f} %")

    # ---------- PHASE 3 : P_sig^neuro(t) temps reel ----------
    log("\n=== PHASE 3 : SIGNATURE TOPOLOGIQUE P_sig^neuro(t) ===")
    log(f"[TOPO] P_sig^peak          = {topo_pre.p_sig_peak:.3f}")
    log(f"[TOPO] Duree de vie H1     = {topo_pre.h1_lifetime_ms:.1f} ms")
    log(f"[TOPO] Cavites H2 max      = {topo_pre.n_h2_cavities}")

    # Horloge thalamique + seuil de conscience
    sig_k, clock = simulate_kuramoto(duration_s=len(ref_eeg) / fs, fs=fs)
    h1_cycles = np.array([d.size for d in topo_pre.barcodes])
    consc = consciousness_threshold(topo_pre.p_sig_t, h1_cycles)
    log(f"[KURAMOTO] coherence theta = {clock.coherence:.3f} | seuil conscience = {consc['threshold_reached']} ({consc['n_cycles']} cycles H1)")

    # ---------- PHASE 4 : tryperposition ----------
    log("\n=== PHASE 4 : TRYPERSITION & COLLAPSE DIRIGE ===")
    adj = (con.weights > 0).astype(np.int8)
    tres = solve_tryperposition(qres.states, qres.energies, adj)
    log(f"[TRYP] Etats selectionnes : {[int(i) for i in tres.selected]}")
    log(f"[TRYP] Amplitudes p_n     = {[round(float(p), 3) for p in tres.p_n]}")
    log(f"[TRYP] Flux d'emergence   = {tres.emergence_flux:+.4f}")
    sig = cognitive_signal(tres, qres.energies, duration_s=5.0, fs=fs,
                           reference=ref_eeg)

    # ---------- PHASE 4b : SNN quantique-couple (dynamique forward) ----------
    # Le surrogate ci-dessus matche la STATISTIQUE (PSD, LZ). Le SNN AdEx
    # tente la dynamique CAUSALE (ISO) : les spikes generent les
    # micro-etats. Couplage : I_quantum = p_n du collapse module
    # l'excitabilite regionale + drive thalamique = phases < 16 Hz.
    # ---------- PHASE 4b : SNN + ISO-LEARNING (Phase 3 roadmap) ----------
    log("\n=== PHASE 4b : SNN ADEX + ISO LEARNING (forward causal) ===")
    n_snn_reg = 8
    n_per = 30
    w_sub = con.weights[:n_snn_reg, :n_snn_reg]
    W_snn, is_exc = build_microcircuit(n_snn_reg, n_per_region=n_per,
                                       w_connectome=w_sub)
    pn = np.array(tres.p_n, dtype=float)
    span = float(pn.max() - pn.min())
    i_quantum = np.clip(pn - pn.mean(), -0.5, 0.5) * 120.0 if span > 0 \
        else np.zeros(n_snn_reg)
    if i_quantum.size < n_snn_reg:
        i_quantum = np.resize(i_quantum, n_snn_reg)
    # Phase 3 : boucle d'apprentissage ISO (STDP + ajustement meta)
    iso_learn = learn_iso(W_snn, is_exc, AdExParams(), ref_eeg, fs,
                          i_quantum, n_epochs=6, duration_s=2.5)
    W_snn = iso_learn.w_final
    log(f"[ISO-LEARN] 6 epoques | meilleur ISO forward = "
        f"{iso_learn.best_iso:.3f} @ i_ext={iso_learn.best_i_ext:.0f}, "
        f"depth={iso_learn.best_depth:.2f}")
    for e, iso_v, ie, dep in iso_learn.history:
        log(f"   epoque {e}: ISO={iso_v:.3f} i_ext={ie} depth={dep}")
    # forward final avec les poids appris (sans drive : dynamique propre)
    n_steps = int(2.5 * 1000 / 0.5)
    snn_res = simulate_snn(W_snn, is_exc, AdExParams(), duration_s=2.5,
                           dt_ms=0.5, i_ext_pa=iso_learn.best_i_ext,
                           stdp_on=False)
    rate = float(snn_res.spikes.sum() /
                 (snn_res.spikes.shape[1] * 2.5))
    sig_snn = snn_to_eeg(snn_res, fs)[:min(800, ref_eeg.size)]
    iso_snn = microstate_isomorphism(sig_snn, ref_eeg, fs)
    log(f"[SNN] {W_snn.shape[0]} neurones, {n_snn_reg} regions | "
        f"taux moyen = {rate:.1f} Hz | ISO forward (post-apprentissage) "
        f"= {iso_snn:.3f}")

    # Couplage non-local (myeline/microtubules) : le Hamiltonien acquiert
    # des canaux inter-regionaux sans decoherence
    H_coupled = coupler_non_local(H, con.positions)
    qres_coupled = solve_quantum_hybrid(H_coupled, k=6, p_sig=topo_pre.p_sig_peak)
    log(f"[COUPLAGE] E0/site couple = {qres_coupled.e0_per_site:.5f} eV "
        f"(vs {qres.e0_per_site:.5f} decouple)")

    # ---------- PHASE 5 : certification + artefacts ----------
    log("\n=== PHASE 5 : CERTIFICATION CRYPTOGRAPHIQUE ===")
    psi_final = np.zeros(qres.states.shape[0], dtype=np.complex128)
    for amp, idx in zip(tres.p_n, tres.selected):
        psi_final += np.sqrt(amp) * qres.states[:, idx]
    psi_bytes = psi_final.astype(np.complex64).tobytes()

    suppression_tot = float(np.prod([g["suppression"] for g in qres.gamma_eff.values()]))
    receipt = generate_receipt(psi_bytes, {
        "final_fidelity": round(qres.fidelity, 6),
        "p_sig_peak": round(topo_pre.p_sig_peak, 4),
        "decoherence_suppression_factor": round(suppression_tot, 2),
    })
    verif = verify_receipt(receipt, psi_bytes)
    invariants = check_invariants(qres.e0_per_site, qres.von_neumann_entropy,
                                  con.n_nodes, verif["verified"])
    log(f"[ZK] Commitment = {receipt['zk_commitment'][:18]}...")
    log(f"[ZK] Verification = {verif['status']} en {verif['verification_time_ms']} ms")
    log(f"[ZK] Invariants = {invariants}")

    # ---------- REPLAY QUANTIQUE (consolidation offline) ----------
    log("\n=== REPLAY QUANTIQUE (sommeil / consolidation) ===")
    psi_wake = qres.states[:, 0].copy()
    replay = replay_offline(psi_wake, qres.energies, adj, ref_eeg, fs, n_cycles=6)
    log(f"[REPLAY] {replay['n_cycles']} cycles | P_sig moyen = {replay['mean_p_sig']:.3f} | flux = {replay['mean_flux']:+.3f}")

    # Artefacts
    np.save(out / "cognitive_state_vector.npy", psi_final.astype(np.complex64))
    with h5py.File(out / "p_sig_temporal_map.h5", "w") as f:
        f.create_dataset("p_sig_t", data=topo_pre.p_sig_t)
        for i, dgm in enumerate(topo_pre.barcodes):
            f.create_dataset(f"barcode_h1/{i:04d}", data=dgm)
        f.attrs["p_sig_peak"] = topo_pre.p_sig_peak
        f.attrs["h1_lifetime_ms"] = topo_pre.h1_lifetime_ms
    with open(out / "decoherence_rates_gamma.csv", "w") as f:
        f.write("canal,gamma0_s-1,gamma_eff_s-1,suppression\n")
        for ch, g in qres.gamma_eff.items():
            f.write(f"{ch},{g['gamma0']},{g['gamma_eff']:.6f},{g['suppression']:.3f}\n")
    with open(out / "zk_receipt_cognitive.bin", "wb") as f:
        f.write(json.dumps(receipt, indent=2).encode())

    # Validation biologique
    psd_corr = psd_correlation(sig, ref_eeg, fs)
    lz = lz_match(sig, ref_eeg)
    iso = microstate_isomorphism(sig, ref_eeg, fs)
    report = f"""# Validation Report — RATISS-NEURO Jumeau Numerique Cognitif

Reference biologique : {'fichier ' + str(eeg_path) if eeg_path else 'substitut synthetique 1/f + theta(6Hz) + gamma(40Hz)'}
Connectome : {'fichier ' + str(connectome_path) if connectome_path else 'small-world synthetique (256 noeuds)'}

## Metriques quantiques
| Observable | Valeur |
|---|---|
| E0 / site | {qres.e0_per_site:.6f} eV |
| Gap de spin | {qres.spin_gap_mev:.3f} meV |
| Ordre d-wave | {qres.dwave_order:.4f} |
| Entropie vN (normalisee) | {qres.von_neumann_entropy:.4f} |
| Fidelite de coherence | {qres.fidelity * 100:.2f} % |

## Metriques topologiques
| Observable | Valeur |
|---|---|
| P_sig peak | {topo_pre.p_sig_peak:.3f} |
| Duree de vie H1 | {topo_pre.h1_lifetime_ms:.1f} ms |
| Suppression decoherence (produit canaux) | x{suppression_tot:.1f} |
| Seuil conscience (cycles H1 persistants) | {consc['threshold_reached']} ({consc['n_cycles']} cycles) |
| Couplage non-local (E0/site) | {qres_coupled.e0_per_site:.5f} eV (delta {abs(qres_coupled.e0_per_site - qres.e0_per_site):.6f} eV vs decouple) |
| Replay quantique (P_sig moyen) | {replay['mean_p_sig']:.3f} |

## Correspondance biologique (objectif : copie ~98 % des signaux)
| Metrique | Valeur |
|---|---|
| Correlation spectrale PSD | {psd_corr:.4f} |
| Match complexite Lempel-Ziv | {lz * 100:.2f} % |
| Isomorphisme micro-etats (corr P_sig) | {iso:.4f} |
| **SNN forward (dynamique causale)** | **ISO = {iso_snn:.3f}** ({W_snn.shape[0]} neurones, {n_snn_reg} regions, {rate:.1f} Hz) |

## Certification
- Commitment : `{receipt['zk_commitment']}`
- Statut : {verif['status']} ({verif['verification_time_ms']} ms)
- Invariants : {json.dumps(invariants)}

Note d'honnetete : le receipt est un engagement par chaine de hachage
BLAKE3->SHA256 (bind + timestamp). H_cog est resolu en deux regimes :
solveur hybride (secteur 1 particule, scalable a N=1024) et solveur
Fock complet (2^n dim, bench d'exactitude, fock_solver.py).

Methode du signal cognitif : surrogate "theta-locke" de la reference
(amplitudes spectrales reelles, phases reelles < 16 Hz = pacemaker
thalamo-cortical, phases randomisees au-dessus), module par les poids
p_n du collapse, avec calibration LZ bidirectionnelle (bruit gamma /
lissage). La PSD et LZ sont donc matchees par construction ; l'ISO
(correlation des trajectoires P_sig) mesure la correspondance
dynamique residuelle — la frontiere ouverte du jumeau.

SNN AdEx quantique-couple + apprentissage ISO (Phase 3 roadmap) :
reseau forward causal de neurones AdEx (80/20 E/I, connectome
structural, I_quantum = p_n du collapse, drive thalamique = phases
< 16 Hz), avec STDP triplet (Pfister-Gerstner), regle homologique
topologique (Betti H1 guide la plasticite via topo_plasticity) et
boucle meta-d'apprentissage sur (i_ext, depth) — iso_learning.py.
Resultat honnete : l'ISO post-apprentissage = {iso_snn:.3f} ; la
refractarite AdEx filtre la dynamique lente du drive, donc le
surrogate statistique (ISO = {iso:.2f}) reste superieur. L'ecart
est documente ici : c'est la frontiere scientifique reelle.
"""
    (out / "validation_report.md").write_text(report)

    elapsed = time.perf_counter() - t_start
    log(f"\n=== MISSION TERMINEE en {elapsed:.1f} s ===")
    log(f"[ARTEFACTS] {sorted(p.name for p in out.iterdir())}")
    log(f"[VALIDATION] PSD={psd_corr:.3f}  LZ={lz * 100:.1f}%  ISO={iso:.3f}")

    return {
        "e0_per_site": qres.e0_per_site,
        "spin_gap_mev": qres.spin_gap_mev,
        "fidelity": qres.fidelity,
        "p_sig_peak": topo_pre.p_sig_peak,
        "h1_lifetime_ms": topo_pre.h1_lifetime_ms,
        "emergence_flux": tres.emergence_flux,
        "psd_corr": psd_corr,
        "lz_match": lz,
        "iso": iso,
        "verified": verif["verified"],
        "invariants": invariants,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="RATISS-NEURO cognitive twin pipeline")
    ap.add_argument("--connectome", default=None, help="matrice de connectivite (.npy/.csv)")
    ap.add_argument("--eeg", default=None, help="EEG de reference (.csv)")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--nodes", type=int, default=256)
    args = ap.parse_args()
    run_pipeline(args.connectome, args.eeg, args.out, args.nodes)


if __name__ == "__main__":
    main()
