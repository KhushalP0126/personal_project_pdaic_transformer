PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
PIP := $(VENV_PYTHON) -m pip
SETUP_STAMP := $(VENV)/.setup-stamp

CPU_ARGS ?= --device cpu --p-list 3 5 --r-list 8 16 --samples 4096 --classes 16 --tokens-per-class 256 --triplets 20000 --distance-pairs 200000
GPU_ARGS ?= --device cuda --p-list 3 5 --r-list 8 16 24 32 --samples 16384 --classes 32 --tokens-per-class 128 --triplets 100000 --distance-pairs 1000000
INT8_ARGS ?= --r 8
TRAIN_GPU_ARGS ?= --device cuda --p 3 --r 16 --d-model 256 --n-heads 8 --n-layers 4 --ffn-dim 1024 --head-hidden 128 --dropout 0.1 --window-size 32 --attack-fraction 0.3 --attack-min-len 2 --attack-max-len 8 --n-train 65536 --n-val 8192 --samples 16384 --classes 32 --tokens-per-class 128 --epochs 20 --batch-size 512 --grad-accum 1 --lr 3e-4 --weight-decay 1e-2 --warmup-epochs 2 --num-workers 4 --alpha 0.5 --pos-weight 1.0 --margin-pos 0.1 --margin-neg 0.5 --save-every 5
TRAIN_CPU_ARGS ?= --device cpu --p 3 --r 8 --d-model 64 --n-heads 4 --n-layers 2 --ffn-dim 256 --head-hidden 32 --dropout 0.1 --window-size 16 --attack-fraction 0.3 --attack-min-len 2 --attack-max-len 4 --n-train 1024 --n-val 256 --samples 4096 --classes 16 --tokens-per-class 64 --epochs 3 --batch-size 64 --lr 3e-4 --num-workers 0 --save-every 999

.DEFAULT_GOAL := all

.PHONY: all setup test run benchmark cpu gpu int8 hardware train train-cpu train-gpu clean help

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

train: train-gpu

train-gpu: setup
	$(VENV_PYTHON) scripts/train_anomaly_detector.py $(TRAIN_GPU_ARGS)

train-cpu: setup
	$(VENV_PYTHON) scripts/train_anomaly_detector.py $(TRAIN_CPU_ARGS)

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
	@echo "  make train  Run the GPU training pipeline"
	@echo "  make train-cpu  Run the CPU training smoke test"
	@echo "  make clean  Remove local Python caches"
