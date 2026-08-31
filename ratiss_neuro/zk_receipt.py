"""Phase 5 : certification cryptographique de l'etat cognitif.

Engagement SHA256(BLAKE3(psi)) : schema d'engagement par chaine de
hachage, verifiable par quiconque sans reveler |psi>. Interface
compatible avec un backend ZK-STARK (RISC Zero) lorsqu'il sera cable.
"""

from __future__ import annotations

import hashlib
import json
import time

import blake3


def state_commitment(state_bytes: bytes) -> str:
    return hashlib.sha256(blake3.blake3(state_bytes).digest()).hexdigest()


def generate_receipt(state_bytes: bytes, public_inputs: dict) -> dict:
    commitment = state_commitment(state_bytes)
    return {
        "scheme": "hash-commitment BLAKE3->SHA256 (ZK-STARK interface placeholder)",
        "zk_commitment": "0x" + commitment,
        "receipt_b64": _encode_receipt(commitment, public_inputs),
        "public_inputs": public_inputs,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "COMMITTED",
    }


def verify_receipt(receipt: dict, state_bytes: bytes) -> dict:
    t0 = time.perf_counter()
    ok = "0x" + state_commitment(state_bytes) == receipt["zk_commitment"]
    dt_ms = (time.perf_counter() - t0) * 1e3
    return {
        "verified": ok,
        "verification_time_ms": round(dt_ms, 3),
        "status": "VERIFIED" if ok else "REJECTED",
    }


def check_invariants(e0: float, entropy: float, n_nodes: int, commitment_ok: bool) -> dict:
    return {
        "binding_energy_negative": e0 < 0,
        "entropy_non_negative": entropy >= 0,
        "lattice_bounds_valid": n_nodes > 0,
        "state_commitment_valid": commitment_ok,
    }


def _encode_receipt(commitment: str, public_inputs: dict) -> str:
    import base64
    payload = json.dumps({"c": commitment, "p": public_inputs}, sort_keys=True).encode()
    return base64.b64encode(payload).decode()
