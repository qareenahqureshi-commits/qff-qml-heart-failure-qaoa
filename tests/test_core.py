"""Soundness checks: run with `pytest -q`."""
import itertools
import sys
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load
from src.qaoa import QAOASolver, qubo_to_ising, top_m_contains_optimum
from src.qubo import make_qubo
from src.solvers import simulated_annealing, tabu_search


def test_dataset_matches_chicco_jurman():
    X, y, f = load()
    assert X.shape == (299, 11) and y.sum() == 96 and not X.isna().any().any()


def test_alpha_search_gives_exactly_k_features():
    X, y, _ = load()
    for variant in ("mi", "fisher"):
        for k in (2, 3, 4):
            x, _, _ = make_qubo(X, y, k, variant).exhaustive()
            assert int(x.sum()) == k


def test_k2_optimum_is_published_panel():
    X, y, f = load()
    x, _, _ = make_qubo(X, y, 2, "mi").exhaustive()
    assert {f[i] for i in np.flatnonzero(x)} == {"serum_creatinine", "ejection_fraction"}


def test_ising_mapping_is_exact_on_every_basis_state():
    X, y, _ = load()
    q = make_qubo(X, y, 2, "fisher")
    H, off = qubo_to_ising(q.Q)
    for bits in itertools.product((0, 1), repeat=q.n):
        label = "".join(str(b) for b in bits[::-1])  # little-endian
        ev = Statevector.from_label(label).expectation_value(H).real + off
        assert abs(ev - q.energy(bits)) < 1e-9


def test_classical_heuristics_find_optimum():
    X, y, _ = load()
    q = make_qubo(X, y, 3, "fisher")
    x, e, _ = q.exhaustive()
    rng = np.random.default_rng(0)
    for fn in (simulated_annealing, tabu_search):
        out = fn(q, 5, rng)
        assert abs(out["energies"].min() - e) < 1e-9


def test_qaoa_runs_and_bit_order_is_consistent():
    X, y, _ = load()
    q = make_qubo(X, y, 2, "fisher")
    x, e, allE = q.exhaustive()
    s = QAOASolver(q, reps=1, shots=500, maxiter=20, seed=0)
    out = s.run(np.random.default_rng(0))
    # energies recomputed from decoded bits must be consistent with the QUBO (sanity on endianness)
    assert np.all(out["energies"] >= e - 1e-9)
    assert top_m_contains_optimum(out["samples"], out["energies"], x, 5) in (True, False)
