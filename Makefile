.PHONY: all baseline optimisers downstream noise select_k figures test clean

all: baseline optimisers downstream noise select_k figures

noise:
	python -W ignore scripts/05_noise.py

select_k:
	python -W ignore scripts/06_select_k.py

test:
	pytest -q tests

baseline:
	python -W ignore scripts/01_baseline.py

optimisers:
	python -W ignore scripts/02_optimisers.py

downstream:
	python -W ignore scripts/03_downstream.py

figures:
	python -W ignore scripts/04_figures.py

clean:
	rm -rf results/*.csv results/figures
