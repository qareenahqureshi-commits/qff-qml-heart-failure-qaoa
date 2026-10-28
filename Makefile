.PHONY: all baseline optimisers downstream noise select_k stats variants cost figures validate \
        statevector hardware hardware-dryrun test clean

PY ?= python

# Full pipeline with Qiskit Aer (roughly 2 hours on 2 CPU cores; the fake_torino noise study is about 40 min
# and the nested selection about 40 min).
all: baseline optimisers downstream noise select_k stats variants cost figures

# Everything that does not need hardware noise, on the exact numpy statevector backend (no Qiskit needed,
# about 45 min on 2 cores; most of it is the random forest refits in the nested selection).
statevector:
	QFF_BACKEND=statevector $(MAKE) select_k stats variants cost figures

baseline:
	$(PY) -W ignore scripts/01_baseline.py

optimisers:
	$(PY) -W ignore scripts/02_optimisers.py

downstream:
	$(PY) -W ignore scripts/03_downstream.py

figures:
	$(PY) -W ignore scripts/04_figures.py

noise:
	$(PY) -W ignore scripts/05_noise.py

select_k:
	$(PY) -W ignore scripts/06_select_k.py

stats:
	$(PY) -W ignore scripts/07_stats.py

variants:
	$(PY) -W ignore scripts/08_qaoa_variants.py

cost:
	$(PY) -W ignore scripts/09_cost_aware.py

# statevector backend vs the committed Aer results (same QUBOs, same initial angles)
validate:
	$(PY) -W ignore scripts/validate_statevector.py

# real IBM device: needs a saved IBM Quantum Platform account and spends a few seconds of QPU time
hardware:
	$(PY) -W ignore scripts/10_hardware.py

hardware-dryrun:
	$(PY) -W ignore scripts/10_hardware.py --backend fake_torino

test:
	pytest -q tests

clean:
	rm -rf results/*.csv results/*.json results/figures
