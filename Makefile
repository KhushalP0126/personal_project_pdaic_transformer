PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
PIP := $(VENV_PYTHON) -m pip
SETUP_STAMP := $(VENV)/.setup-stamp

CPU_ARGS ?= --device cpu --p-list 3 5 --r-list 8 16 --samples 4096 --classes 16 --tokens-per-class 256 --triplets 20000 --distance-pairs 200000
GPU_ARGS ?= --device cuda --p-list 3 5 --r-list 8 16 24 32 --samples 16384 --classes 32 --tokens-per-class 128 --triplets 100000 --distance-pairs 1000000
INT8_ARGS ?= --r 8

.DEFAULT_GOAL := all

.PHONY: all setup test run benchmark cpu gpu int8 hardware clean help

all: setup test run

$(VENV_PYTHON):
	$(PYTHON) -m venv $(VENV)

$(SETUP_STAMP): pyproject.toml $(VENV_PYTHON)
	$(PIP) install -e .
	touch $(SETUP_STAMP)

setup: $(SETUP_STAMP)

test: setup
	$(VENV_PYTHON) -m unittest discover -s tests

run: cpu

benchmark: cpu

cpu: setup
	$(VENV_PYTHON) scripts/run_padic_benchmark.py $(CPU_ARGS)

gpu: setup
	$(VENV_PYTHON) scripts/run_padic_benchmark.py $(GPU_ARGS)

hardware: int8

int8: setup
	$(VENV_PYTHON) scripts/verify_int8_2adic.py $(INT8_ARGS)

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

help:
	@echo "Targets:"
	@echo "  make        Setup, test, and run the CPU reference benchmark"
	@echo "  make setup  Create .venv and install the editable package"
	@echo "  make test   Run unit tests"
	@echo "  make cpu    Run the local CPU benchmark"
	@echo "  make gpu    Run the cloud CUDA benchmark"
	@echo "  make int8   Verify unsigned INT8 against 2-adic arithmetic"
	@echo "  make clean  Remove local Python caches"
