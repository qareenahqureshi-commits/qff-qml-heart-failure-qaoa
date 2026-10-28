"""Soundness checks: run with `pytest -q`. Tests that need Qiskit are skipped when it is not installed."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.costs import cost_matrix, panel_cost
from src.data import load
from src.qaoa import QAOASolver, cvar_from_counts, interp_params, top_m_contains_optimum
from src.qubo import make_qubo
from src.solvers import random_feasible, simulated_annealing, tabu_search
from src.statevector import StatevectorQAOA, apply_gates, basis_bits, dicke_gate_list, dicke_state
from src.stats import paired_test, split_sizes


# ----------------------------------------------------------------------------- data and QUBO
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


def test_classical_heuristics_find_optimum():
    X, y, _ = load()
    q = make_qubo(X, y, 3, "fisher")
    x, e, _ = q.exhaustive()
    rng = np.random.default_rng(0)
    for fn in (simulated_annealing, tabu_search):
        out = fn(q, 5, rng)
        assert abs(out["energies"].min() - e) < 1e-9


def test_random_feasible_only_samples_k_panels():
    X, y, _ = load()
    q = make_qubo(X, y, 3, "fisher")
    out = random_feasible(q, 500, np.random.default_rng(0))
    assert np.all(out["samples"].sum(1) == 3)


# ----------------------------------------------------------------------------- statistics
def test_split_sizes_match_sklearn():
    from sklearn.model_selection import StratifiedShuffleSplit
    X, y, _ = load()
    tr, te = next(StratifiedShuffleSplit(1, test_size=0.2, random_state=0).split(X, y))
    assert split_sizes(299, 0.2) == (len(tr), len(te)) == (239, 60)


def test_corrected_test_is_more_conservative_by_the_right_factor():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0.43, 0.1, 100), rng.normal(0.40, 0.1, 100)
    r = paired_test(a, b, 239, 60)
    assert r["p_corrected"] > r["p_naive"]
    assert abs(r["t_naive"] / r["t_corrected"] - np.sqrt((1 / 100 + 60 / 239) / (1 / 100))) < 1e-9


# ----------------------------------------------------------------------------- QAOA helpers
def test_interp_params():
    # p = 1 -> 2 duplicates each angle; p = 2 -> 3 interpolates linearly (Zhou et al. 2020)
    assert np.allclose(interp_params([0.3, 0.7], 1), [0.3, 0.3, 0.7, 0.7])
    new = interp_params([1.0, 2.0, 3.0, 5.0], 2)
    assert np.allclose(new, [1.0, 1.5, 2.0, 3.0, 4.0, 5.0])


def test_cvar_from_counts():
    energies = np.array([3.0, 1.0, 2.0, 0.0])
    counts = np.array([5, 3, 2, 1])
    assert cvar_from_counts(energies, counts, 0.1) == pytest.approx(0.5)      # best 2 shots: 0, 1
    assert cvar_from_counts(energies, counts, 1.0) == pytest.approx((15 + 3 + 4 + 0) / 11)


def test_dicke_circuit_prepares_dicke_state():
    for n in range(2, 12):
        for k in range(1, n):
            psi = np.zeros(2 ** n, complex)
            psi[0] = 1
            out = apply_gates(psi, dicke_gate_list(n, k), n)
            assert abs(np.vdot(dicke_state(n, k), out)) ** 2 > 1 - 1e-9


def test_xy_mixer_keeps_panel_size():
    X, y, _ = load()
    q = make_qubo(X, y, 3, "fisher")
    sv = StatevectorQAOA(q.Q, 3, "xy", k=3)
    psi = sv.state(np.random.default_rng(1).uniform(0, np.pi, 6))
    w = basis_bits(11).sum(1)
    assert abs(np.linalg.norm(psi) - 1) < 1e-9 and np.abs(psi[w != 3]).max() < 1e-12


def test_statevector_backend_runs_without_qiskit():
    X, y, _ = load()
    q = make_qubo(X, y, 3, "fisher")
    x, e, _ = q.exhaustive()
    for mixer, objective in [("x", "mean"), ("xy", "cvar")]:
        s = QAOASolver(q, reps=1, shots=500, maxiter=20, seed=0, mixer=mixer, objective=objective, backend="statevector")
        out = s.run(np.random.default_rng(0))
        assert np.all(out["energies"] >= e - 1e-9)
        if mixer == "xy":
            assert np.all(out["samples"].sum(1) == 3)


# ----------------------------------------------------------------------------- cost model
def test_cost_qubo_is_exact_on_every_subset():
    _, _, f = load()
    C = cost_matrix(f)
    for bits in itertools.product((0, 1), repeat=11):
        x = np.array(bits)
        assert abs(x @ C @ x - panel_cost([f[i] for i in range(11) if x[i]])) < 1e-9


# ----------------------------------------------------------------------------- Qiskit cross checks
def test_ising_mapping_is_exact_on_every_basis_state():
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector
    from src.qaoa import qubo_to_ising
    X, y, _ = load()
    q = make_qubo(X, y, 2, "fisher")
    H, off = qubo_to_ising(q.Q)
    for bits in itertools.product((0, 1), repeat=q.n):
        label = "".join(str(b) for b in bits[::-1])  # little-endian
        ev = Statevector.from_label(label).expectation_value(H).real + off
        assert abs(ev - q.energy(bits)) < 1e-9


@pytest.mark.parametrize("mixer", ["x", "xy"])
def test_numpy_statevector_matches_qiskit_circuit(mixer):
    """Same probabilities for the same parameter vector: checks parameter order, signs, mixer and Dicke circuit."""
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector
    from src.qaoa import build_ansatz
    X, y, _ = load()
    q = make_qubo(X, y, 3, "fisher")
    params = np.random.default_rng(7).uniform(0, np.pi, 4)
    circ = build_ansatz(q, 2, mixer)
    pr_qiskit = Statevector(circ.assign_parameters(params)).probabilities()
    pr_numpy = StatevectorQAOA(q.Q, 2, mixer, k=q.k).probabilities(params)
    assert np.allclose(pr_qiskit, pr_numpy, atol=1e-8)


def test_qaoa_runs_and_bit_order_is_consistent():
    pytest.importorskip("qiskit_aer")
    X, y, _ = load()
    q = make_qubo(X, y, 2, "fisher")
    x, e, allE = q.exhaustive()
    s = QAOASolver(q, reps=1, shots=500, maxiter=20, seed=0)
    out = s.run(np.random.default_rng(0))
    # energies recomputed from decoded bits must be consistent with the QUBO (sanity on endianness)
    assert np.all(out["energies"] >= e - 1e-9)
    assert np.allclose(out["energies"], [q.energy(b) for b in out["samples"]])
    assert top_m_contains_optimum(out["samples"], out["energies"], x, 5) in (True, False)
