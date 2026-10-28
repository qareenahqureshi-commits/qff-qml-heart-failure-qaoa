"""Run the panel selection QAOA on a real IBM quantum computer (or a fake backend for a local dry run).

Protocol A of the noise study (parameter transfer), at depth p = 1, the depth the noise study recommends:
  1. for k = 2, 3, 4 (Fisher QUBO), optimise the p = 1 angles on the ideal Aer simulator (best of a few seeds),
     once with the mean energy objective and once with the CVaR objective (best 10% of shots);
  2. compile each circuit for the device (optimization_level 3) and sample all of them in ONE Sampler job
     (6 circuits x 2000 shots: a few seconds of QPU time);
  3. score the shots exactly like the simulations (approximation ratio, P(optimum), enrichment, found).
On real hardware the Sampler uses dynamical decoupling and gate twirling (cheap error suppression, no extra shots).

One time setup (IBM Quantum Platform, https://quantum.cloud.ibm.com, free plan is enough for this job):
  python -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; S.save_account(channel='ibm_quantum_platform', token='<API key>', instance='<instance CRN or name>', set_as_default=True)"

Usage:
  python scripts/10_hardware.py --backend fake_torino      # local dry run with the Heron noise model, no account
  python scripts/10_hardware.py                            # least busy real device your account can use
  python scripts/10_hardware.py --backend ibm_torino       # a specific device
  options: --shots 2000  --seeds 5  --objectives mean cvar  --no-mitigation

Outputs: results/hardware.csv (one row per k x objective), results/hardware_counts.json (raw counts, job id)
Not run by `make all`: it needs an IBM account and spends QPU time.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load
from src.qaoa import QAOASolver, _unroute, counts_to_vector
from src.qubo import make_qubo
from src.solvers import optimiser_metrics
from src.statevector import basis_bits

RES = Path(__file__).resolve().parents[1] / "results"

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--backend", default=None, help="device name (e.g. ibm_torino) or a fake backend (e.g. fake_torino); "
                                                 "default: least busy real device")
ap.add_argument("--shots", type=int, default=2000)
ap.add_argument("--seeds", type=int, default=5, help="ideal-simulator restarts per circuit; the best is sent")
ap.add_argument("--objectives", nargs="+", default=["mean", "cvar"], choices=["mean", "cvar"])
ap.add_argument("--ks", nargs="+", type=int, default=[2, 3, 4])
ap.add_argument("--no-mitigation", action="store_true", help="turn off dynamical decoupling and gate twirling")
args = ap.parse_args()

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # noqa: E402
from qiskit_ibm_runtime import SamplerV2 as Sampler  # noqa: E402

# ---------------------------------------------------------------- device
fake = args.backend is not None and args.backend.startswith("fake_")
if fake:
    from qiskit_ibm_runtime import fake_provider
    backend = getattr(fake_provider, "".join(w.capitalize() for w in args.backend.split("_")))()
else:
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    backend = (service.backend(args.backend) if args.backend
               else service.least_busy(operational=True, simulator=False, min_num_qubits=11))
print(f"backend: {backend.name}  ({'local fake backend' if fake else 'real device'})", flush=True)
pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=0)

# ---------------------------------------------------------------- 1. ideal angles, 2. compile
X, y, feats = load()
jobs_meta, pubs = [], []
for k in args.ks:
    q = make_qubo(X, y, k, "fisher")
    x_opt, e_opt, allE = q.exhaustive()
    e_max = max(allE.values())
    for objective in args.objectives:
        runs = []
        for seed in range(args.seeds):
            s = QAOASolver(q, reps=1, shots=args.shots, maxiter=150, seed=seed, objective=objective, backend="aer")
            runs.append((s.run(np.random.default_rng(seed)), s))
        out, solver = min(runs, key=lambda r: r[0]["final_expectation"])
        ideal = optimiser_metrics(q, out["samples"], out["energies"], x_opt, e_opt, e_max)
        isa = pm.run(solver.ansatz)
        two_q = sum(v for g, v in isa.count_ops().items() if g in ("cz", "ecr", "cx"))
        pubs.append((isa, out["opt_params"]))
        jobs_meta.append(dict(k=k, objective=objective, q=q, x_opt=x_opt, e_opt=e_opt, e_max=e_max, ideal=ideal,
                              params=out["opt_params"].tolist(), two_qubit_gates=int(two_q), depth=isa.depth()))
        print(f"k={k} {objective:4s}: ideal P(opt) {ideal['p_optimum']:.3f}  AR {ideal['approx_ratio_mean']:.3f}  "
              f"compiled: {two_q} two qubit gates, depth {isa.depth()}", flush=True)

# ---------------------------------------------------------------- 3. one Sampler job for everything
sampler = Sampler(mode=backend)
sampler.options.default_shots = args.shots
if not fake and not args.no_mitigation:
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.num_randomizations = "auto"
t = time.time()
job = sampler.run(pubs)
job_id = job.job_id() if callable(getattr(job, "job_id", None)) else str(getattr(job, "job_id", "local"))
print(f"submitted job {job_id}; waiting for results ...", flush=True)
result = job.result()
print(f"done in {time.time() - t:.0f}s", flush=True)

rows, raw = [], {"backend": backend.name, "job_id": job_id, "shots": args.shots, "runs": []}
B = basis_bits(len(feats)).astype(int)
for meta, res in zip(jobs_meta, result):
    q = meta["q"]
    counts = _unroute(res.data.meas.get_counts(), None, q.n)
    vec = counts_to_vector(counts, q.n)
    idx = np.repeat(np.arange(vec.size), vec)
    samples = B[idx]
    energies = np.array([q.energy(b) for b in B])[idx]
    m = optimiser_metrics(q, samples, energies, meta["x_opt"], meta["e_opt"], meta["e_max"])
    rows.append(dict(backend=backend.name, job_id=job_id, k=meta["k"], objective=meta["objective"],
                     two_qubit_gates=meta["two_qubit_gates"], depth=meta["depth"],
                     mitigation=not fake and not args.no_mitigation,
                     **{f"hw_{c}": v for c, v in m.items()},
                     **{f"ideal_{c}": meta["ideal"][c] for c in ("approx_ratio_mean", "p_optimum", "enrichment_feasible")}))
    raw["runs"].append(dict(k=meta["k"], objective=meta["objective"], params=meta["params"], counts=counts))

df = pd.DataFrame(rows)
name = "hardware_dryrun" if fake else "hardware"
df.to_csv(RES / f"{name}.csv", index=False)
(RES / f"{name}_counts.json").write_text(json.dumps(raw))
pd.set_option("display.width", 220)
print(df[["k", "objective", "two_qubit_gates", "hw_approx_ratio_mean", "ideal_approx_ratio_mean", "hw_p_optimum",
          "ideal_p_optimum", "hw_enrichment_feasible", "hw_jaccard_best_vs_opt", "hw_best_features"]].round(4).to_string(index=False))
print(f"\nwrote results/{name}.csv and results/{name}_counts.json")
