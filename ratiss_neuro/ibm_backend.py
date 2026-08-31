"""Backend IBM Quantum optionnel : cartographie atomique du Hamiltonien
neuronal sur circuit qubit via Qiskit (Jordan-Wigner -> VQE).

Necessite :
    pip install qiskit qiskit-ibm-runtime
    export IBM_QUANTUM_TOKEN=...  (cle API IBM Quantum)

Sans cle ni qiskit, le module reste importable et le pipeline utilise
le solveur local Lanczos (quantum_solver.py).
"""

from __future__ import annotations

import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit_ibm_runtime import QiskitRuntimeService
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


DEFAULT_CRN = None  # a fournir via le parametre crn= ou instance IBM


def list_backends(token: str, crn: str, limit: int = 8) -> list:
    """Liste les backends IBM Quantum disponibles pour l'instance."""
    import urllib.request
    import json as _json

    req = urllib.request.Request(
        "https://quantum.cloud.ibm.com/api/v1/backends?limit=%d" % limit,
        headers={
            "Service-CRN": crn,
            "Authorization": f"apikey {token}",
            "IBM-API-Version": "2025-05-01",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = _json.loads(resp.read())
    return [d["name"] for d in data.get("devices", [])]


def hamiltonian_to_qubits(H, max_qubits: int = 8) -> "QuantumCircuit":
    """Cartographie le bloc principal de H_cog sur un circuit QAOA-like.

    Chaque site fermionique -> un qubit (Jordan-Wigner), hopping t_ij ->
    portes d'intrication paramétrées Rzz(theta_ij).
    """
    if not HAS_QISKIT:
        raise ImportError("qiskit requis : pip install qiskit qiskit-ibm-runtime")
    n = min(max_qubits, H.shape[0])
    Hb = H[:n, :n].toarray().real
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(Hb[i, j]) > 1e-9:
                qc.rzz(2 * float(Hb[i, j]), i, j)
    qc.measure_all()
    return qc


def run_circuit_remote(token: str, crn: str, H, backend_name: str = "ibm_fez",
                       max_qubits: int = 8, shots: int = 1024) -> dict:
    """Execute le circuit neuronal (cartographie du bloc H_cog) sur
    hardware IBM via Sampler, et retourne les comptages mesures."""
    if not HAS_QISKIT:
        raise ImportError("qiskit requis : pip install qiskit qiskit-ibm-runtime")
    from qiskit import transpile
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform", token=token, instance=crn
    )
    backend = service.backend(backend_name)
    qc = hamiltonian_to_qubits(H, max_qubits=max_qubits)
    tqc = transpile(qc, backend=backend, optimization_level=1)
    sampler = Sampler(mode=backend)
    job = sampler.run([tqc], shots=shots)
    result = job.result()
    counts = result[0].data.meas.get_counts()
    return {
        "job_id": job.job_id(),
        "backend": backend_name,
        "shots": shots,
        "counts": counts,
        "status": "DONE",
    }
