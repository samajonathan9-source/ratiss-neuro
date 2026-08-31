"""Tests du pipeline RATISS-NEURO : chemins de code reels, pas de mocks."""

import numpy as np
import pytest

from ratiss_neuro.bioloader import load_connectome, load_reference_eeg
from ratiss_neuro.core import run_pipeline
from ratiss_neuro.hamiltonian import build_hamiltonian
from ratiss_neuro.quantum_solver import (
    _coherence_fidelity,
    _lindblad_channels,
    solve_quantum_hybrid,
)
from ratiss_neuro.topology import compute_p_sig, graph_sublevel_persistence
from ratiss_neuro.tryperposition import cognitive_signal, solve_tryperposition
from ratiss_neuro.zk_receipt import generate_receipt, verify_receipt


@pytest.fixture(scope="module")
def con():
    return load_connectome(n_nodes=64, seed=7)


@pytest.fixture(scope="module")
def H(con):
    return build_hamiltonian(con)


class TestBioloader:
    def test_connectome_shape(self, con):
        assert con.weights.shape == (64, 64)
        assert (con.weights >= 0).all()
        assert np.allclose(con.weights, con.weights.T)

    def test_reference_eeg(self):
        sig, fs = load_reference_eeg(duration_s=1.0)
        assert sig.size == 1000 and fs == 1000.0


class TestHamiltonian:
    def test_hermitian(self, H):
        diff = (H - H.getH()).data
        assert np.allclose(diff, 0, atol=1e-10)

    def test_sparse_and_sized(self, H):
        assert H.shape == (64, 64)
        assert H.nnz > 64


class TestQuantumSolver:
    def test_ground_state_bound(self, H):
        res = solve_quantum_hybrid(H, k=6, p_sig=0.4)
        assert res.e0_per_site < 0            # energie liee
        assert res.spin_gap_mev > 0           # gap positif
        assert 0.0 <= res.von_neumann_entropy <= 1.0

    def test_topological_shield(self):
        g_weak = _lindblad_channels(0.05, 0.120)
        g_strong = _lindblad_channels(0.90, 0.120)
        for ch in g_weak:
            assert g_strong[ch]["gamma_eff"] < g_weak[ch]["gamma_eff"]

    def test_fidelity_range(self, H):
        res = solve_quantum_hybrid(H, k=6, p_sig=0.4)
        assert 0.5 <= res.fidelity <= 1.0


class TestTopology:
    def test_p_sig_bounds(self):
        sig, fs = load_reference_eeg(duration_s=2.0)
        topo = compute_p_sig(sig, fs, window_s=0.5, hop_s=0.1)
        assert topo.p_sig_t.size > 0
        assert 0.0 <= topo.p_sig_peak <= 1.0

    def test_graph_persistence(self, con, H):
        from scipy.sparse.linalg import eigsh
        _, v = eigsh(H, k=1, which="SA")
        field = np.abs(v[:, 0]) ** 2
        score = graph_sublevel_persistence(field, (con.weights > 0))
        assert 0.0 <= score <= 1.0


class TestTryperposition:
    def test_collapse_distribution(self, con, H):
        res_q = solve_quantum_hybrid(H, k=6, p_sig=0.4)
        adj = (con.weights > 0).astype(np.int8)
        tres = solve_tryperposition(res_q.states, res_q.energies, adj)
        assert np.isclose(tres.p_n.sum(), 1.0)
        assert len(tres.selected) == 3
        assert tres.emergence_flux >= -1.0

    def test_cognitive_signal_shape(self, con, H):
        res_q = solve_quantum_hybrid(H, k=6, p_sig=0.4)
        adj = (con.weights > 0).astype(np.int8)
        tres = solve_tryperposition(res_q.states, res_q.energies, adj)
        sig = cognitive_signal(tres, res_q.energies, duration_s=1.0, fs=1000.0)
        assert sig.size == 1000
        assert np.isfinite(sig).all()


class TestZKReceipt:
    def test_commit_and_verify(self):
        state = np.random.default_rng(0).standard_normal(64).astype(np.complex64)
        sb = state.tobytes()
        receipt = generate_receipt(sb, {"fidelity": 0.99})
        assert verify_receipt(receipt, sb)["verified"] is True

    def test_tamper_rejected(self):
        state = np.random.default_rng(0).standard_normal(64).astype(np.complex64)
        receipt = generate_receipt(state.tobytes(), {})
        bad = state.copy()
        bad[0] += 1.0
        assert verify_receipt(receipt, bad.tobytes())["verified"] is False


class TestRealConnectome:
    def test_celegans_edge_list(self, tmp_path):
        csv = tmp_path / "ce.csv"
        csv.write_text("pre\tpost\ttype\tsynapses\n"
                       "ADAL\tADFL\telectrical\t1\n"
                       "ADAL\tAIBL\tchemical\t2\n"
                       "ADFL\tAIBL\tchemical\t3\n")
        con = load_connectome(str(csv))
        assert con.n_nodes == 3
        assert con.weights[0, 1] == 1.0 and con.weights[0, 2] == 2.0
        assert np.allclose(con.weights, con.weights.T)


class TestPipelineEndToEnd:
    def test_full_run(self, tmp_path):
        out = run_pipeline(n_nodes=64, out_dir=str(tmp_path), verbose=False)
        assert out["verified"] is True
        assert all(out["invariants"].values())
        assert out["e0_per_site"] < 0
        assert out["psd_corr"] > 0.5
        for name in ["cognitive_state_vector.npy", "p_sig_temporal_map.h5",
                     "decoherence_rates_gamma.csv", "zk_receipt_cognitive.bin",
                     "validation_report.md"]:
            assert (tmp_path / name).exists()
