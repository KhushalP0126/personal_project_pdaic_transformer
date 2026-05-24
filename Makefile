PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
PIP := $(VENV_PYTHON) -m pip
SETUP_STAMP := $(VENV)/.setup-stamp

# ---------------------------------------------------------------------------
# Benchmark defaults
# ---------------------------------------------------------------------------
CPU_ARGS ?= --device cpu --p-list 3 5 --r-list 8 16 --samples 4096 --classes 16 --tokens-per-class 256 --triplets 20000 --distance-pairs 200000
GPU_ARGS ?= --device cuda --p-list 3 5 --r-list 8 16 24 32 --samples 16384 --classes 32 --tokens-per-class 128 --triplets 100000 --distance-pairs 1000000

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
TRAIN_GPU_ARGS ?= --device cuda --p 3 --r 16 --d-model 384 --n-heads 8 --n-layers 6 --ffn-dim 1536 --head-hidden 192 --dropout 0.1 --window-size 48 --attack-fraction 0.35 --attack-min-len 2 --attack-max-len 10 --n-train 131072 --n-val 16384 --samples 32768 --classes 32 --tokens-per-class 256 --epochs 30 --batch-size 768 --grad-accum 2 --lr 2e-4 --weight-decay 1e-2 --warmup-epochs 3 --num-workers 4 --alpha 0.5 --pos-weight 1.0 --margin-pos 0.1 --margin-neg 0.5 --save-every 5
TRAIN_CPU_ARGS ?= --device cpu --p 3 --r 8 --d-model 64 --n-heads 4 --n-layers 2 --ffn-dim 256 --head-hidden 32 --dropout 0.1 --window-size 16 --attack-fraction 0.3 --attack-min-len 2 --attack-max-len 4 --n-train 1024 --n-val 256 --samples 4096 --classes 16 --tokens-per-class 64 --epochs 3 --batch-size 64 --lr 3e-4 --num-workers 0 --save-every 999
TRAIN_ATTENTION_CPU_ARGS ?= --attention --device cpu --p 3 --r 8 --d-model 64 --n-heads 4 --n-layers 2 --ffn-dim 256 --head-hidden 32 --dropout 0.1 --d-digit 8 --window-size 16 --attack-fraction 0.3 --attack-min-len 2 --attack-max-len 4 --n-train 4096 --n-val 512 --samples 4096 --classes 16 --tokens-per-class 64 --epochs 15 --batch-size 64 --lr 3e-4 --warmup-epochs 2 --num-workers 0 --alpha 0.0 --save-every 999
TRAIN_ATTENTION_GPU_ARGS ?= --attention --device cuda --p 3 --r 16 --d-model 384 --n-heads 8 --n-layers 6 --ffn-dim 1536 --head-hidden 192 --dropout 0.1 --d-digit 16 --window-size 48 --attack-fraction 0.35 --attack-min-len 2 --attack-max-len 10 --n-train 131072 --n-val 16384 --samples 32768 --classes 32 --tokens-per-class 256 --epochs 30 --batch-size 768 --grad-accum 2 --lr 2e-4 --weight-decay 1e-2 --warmup-epochs 3 --num-workers 4 --alpha 0.5 --pos-weight 1.0 --margin-pos 0.1 --margin-neg 0.5 --save-every 5
TRAIN_ATTENTION_BCE_GPU_ARGS ?= --attention --device cuda --p 3 --r 16 --d-model 384 --n-heads 8 --n-layers 6 --ffn-dim 1536 --head-hidden 192 --dropout 0.1 --d-digit 16 --window-size 48 --attack-fraction 0.30 --attack-min-len 2 --attack-max-len 10 --n-train 131072 --n-val 16384 --samples 32768 --classes 32 --tokens-per-class 256 --epochs 30 --batch-size 768 --grad-accum 2 --lr 1e-4 --weight-decay 1e-2 --warmup-epochs 3 --num-workers 4 --alpha 0.0 --margin-pos 0.1 --margin-neg 0.5 --save-every 5
TRAIN_ATTENTION_RULE_GPU_ARGS ?= --attention --device cuda --p 3 --r 16 --d-model 384 --n-heads 8 --n-layers 6 --ffn-dim 1536 --head-hidden 192 --dropout 0.1 --d-digit 16 --window-size 48 --hierarchy-rule-dataset --rule-subtree-depth 2 --rule-stay-steps 4 --rule-attack-tokens 1 --attack-fraction 0.30 --n-train 131072 --n-val 16384 --samples 32768 --classes 32 --tokens-per-class 256 --epochs 30 --batch-size 768 --grad-accum 2 --lr 1e-4 --weight-decay 1e-2 --warmup-epochs 3 --num-workers 4 --alpha 0.0 --margin-pos 0.1 --margin-neg 0.5 --save-every 5
TRAIN_ATTENTION_REALISTIC_GPU_ARGS ?= --attention --device cuda --p 3 --r 16 --d-model 384 --n-heads 8 --n-layers 6 --ffn-dim 1536 --head-hidden 192 --dropout 0.1 --d-digit 16 --window-size 48 --realistic-dataset --realistic-attack-fraction 0.005 --idle-fraction 0.70 --n-train 131072 --n-val 16384 --samples 32768 --classes 32 --tokens-per-class 256 --epochs 30 --batch-size 768 --grad-accum 2 --lr 1e-4 --weight-decay 1e-2 --warmup-epochs 3 --num-workers 4 --alpha 0.0 --margin-pos 0.1 --margin-neg 0.5 --save-every 5
COMPARE_TRAIN_BASE_ARGS ?= --device cuda --r 16 --d-model 384 --n-heads 8 --n-layers 6 --ffn-dim 1536 --head-hidden 192 --dropout 0.1 --window-size 48 --attack-fraction 0.35 --attack-min-len 2 --attack-max-len 10 --n-train 131072 --n-val 16384 --samples 32768 --classes 32 --tokens-per-class 256 --epochs 10 --batch-size 768 --grad-accum 2 --lr 2e-4 --weight-decay 1e-2 --warmup-epochs 3 --num-workers 4 --alpha 0.5 --pos-weight 1.0 --margin-pos 0.1 --margin-neg 0.5 --save-every 5
SWEEP_P_ARGS ?= --sweep-p-bases --device cpu --p-list 3 5 7 --sweep-r 8 --sweep-samples 512 --sweep-classes 16 --sweep-tokens-per-class 64 --sweep-window-size 32 --sweep-attack-fraction 0.30 --sweep-attack-min-len 2 --sweep-attack-max-len 8 --sweep-batch-size 64 --sweep-batches 4
BASELINE_RULE_ARGS ?= --device cpu --hierarchy-rule-dataset --p 3 --r 8 --samples 4096 --classes 16 --tokens-per-class 64 --window-size 32 --attack-fraction 0.30 --rule-subtree-depth 2 --rule-stay-steps 4 --rule-attack-tokens 1 --train-samples 2048 --val-samples 512 --epochs 5 --output-json results/baseline_report.json
TRAINED_EVAL_ARGS ?= --device cpu --trained-eval-checkpoint results/checkpoints/best.pt --trained-eval-dataset hierarchy_rules --trained-eval-samples 512 --trained-eval-window-size 32 --trained-eval-attack-fraction 0.30 --trained-eval-batch-size 64
OPEN_DATASET_ADFA_ARGS ?= --dataset adfa --data-dir ./data/adfa --p 3 --r 8 --window-size 32 --stride 4 --d-model 128 --n-heads 4 --n-layers 2 --epochs 5 --batch-size 256 --device cpu
OPEN_DATASET_BETH_ARGS ?= --dataset beth --data-dir ./data/beth --p 3 --r 8 --window-size 32 --stride 4 --d-model 128 --n-heads 4 --n-layers 2 --epochs 5 --batch-size 256 --device cpu
OPEN_DATASET_STATS_ARGS ?= --dataset adfa --data-dir ./data/adfa --stats-only --no-download --p 3 --r 8 --window-size 32 --stride 4 --device cpu

# ---------------------------------------------------------------------------
# Analysis defaults
# ---------------------------------------------------------------------------
ANALYSIS_ARGS ?= --device cuda --p 3 --r 8 --d-model 128 --n-heads 4 --n-layers 2 --ffn-dim 512 --head-hidden 64 --dropout 0.1 --window-size 16 --attack-fraction 0.3 --attack-min-len 2 --attack-max-len 4 --n-train 4096 --n-val 1024 --samples 4096 --classes 16 --tokens-per-class 64 --epochs 8 --batch-size 128 --grad-accum 1 --lr 3e-4 --num-workers 0 --alpha 0.5 --pos-weight 1.0 --margin-pos 0.1 --margin-neg 0.5 --save-every 999

# ---------------------------------------------------------------------------
# 2-adic / INT8 defaults
# ---------------------------------------------------------------------------
INT8_ARGS ?= --r 8

.DEFAULT_GOAL := help

.PHONY: all setup test cpu gpu benchmark run \
        int8 hardware \
        smoke train hierarchy realistic primes sweep baselines eval threshold diagnose ablate \
        adfa beth adfa-stats \
        train-attention-cpu train-attention-bce-gpu train-attention-hierarchy-gpu train-attention-realistic-gpu \
        compare-primes sweep-p-bases run-baselines eval-trained-attention tune-threshold over-underfit \
        open-adfa open-beth open-adfa-stats \
        ablate-no-contrastive ablate-small-model ablate-r8 ablate-p3 ablate-p5 ablate-p7 \
        clean help clean-results clean-caches clean-checkpoints

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
all: setup test run

$(VENV_PYTHON):
	$(PYTHON) -m venv $(VENV)

$(SETUP_STAMP): pyproject.toml $(VENV_PYTHON)
	$(VENV_PYTHON) -c "import numpy, torch" || $(PIP) install numpy torch
	touch $(SETUP_STAMP)

setup: $(SETUP_STAMP)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
test: setup
	$(VENV_PYTHON) -m unittest discover -s tests

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
run: cpu

benchmark: cpu

cpu: setup
	$(VENV_PYTHON) scripts/run_padic_benchmark.py $(CPU_ARGS)

gpu: setup
	$(VENV_PYTHON) scripts/run_padic_benchmark.py $(GPU_ARGS)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
smoke: setup
	$(VENV_PYTHON) scripts/train_anomaly_detector.py $(TRAIN_ATTENTION_CPU_ARGS)

train: setup
	$(VENV_PYTHON) scripts/train_anomaly_detector.py $(TRAIN_ATTENTION_BCE_GPU_ARGS)

hierarchy: setup
	$(VENV_PYTHON) scripts/train_anomaly_detector.py $(TRAIN_ATTENTION_RULE_GPU_ARGS)

realistic: setup
	$(VENV_PYTHON) scripts/train_anomaly_detector.py $(TRAIN_ATTENTION_REALISTIC_GPU_ARGS)

primes: setup
	@set -e; \
	for p in 3 5 7; do \
		$(VENV_PYTHON) scripts/train_anomaly_detector.py $(COMPARE_TRAIN_BASE_ARGS) --p $$p --log-json results/compare_p$$p.json --log-md results/compare_p$$p.md; \
	done

sweep: setup
	$(VENV_PYTHON) scripts/run_padic_benchmark.py $(SWEEP_P_ARGS)

baselines: setup
	$(VENV_PYTHON) scripts/run_baselines.py $(BASELINE_RULE_ARGS)

eval: setup
	$(VENV_PYTHON) scripts/run_padic_benchmark.py $(TRAINED_EVAL_ARGS)

threshold: setup
	$(VENV_PYTHON) scripts/train_anomaly_detector.py $(TRAIN_GPU_ARGS) --log-json results/tune_threshold.json --log-md results/tune_threshold.md

diagnose: setup
	$(VENV_PYTHON) scripts/over_underfit.py --device cuda --log-json results/over_underfit.json --log-md results/over_underfit.md

train-attention-cpu: smoke
train-attention-bce-gpu: train
train-attention-hierarchy-gpu: hierarchy
train-attention-realistic-gpu: realistic
compare-primes: primes
sweep-p-bases: sweep
run-baselines: baselines
eval-trained-attention: eval
tune-threshold: threshold
over-underfit: diagnose

ablate-no-contrastive: setup
	$(VENV_PYTHON) scripts/train_anomaly_detector.py $(TRAIN_GPU_ARGS) --alpha 0.0 --log-json results/ablate_no_contrastive.json --log-md results/ablate_no_contrastive.md

ablate-small-model: setup
	$(VENV_PYTHON) scripts/train_anomaly_detector.py --device cuda --p 7 --r 16 --d-model 128 --n-heads 4 --n-layers 2 --ffn-dim 512 --head-hidden 64 --epochs 10 --n-train 131072 --n-val 16384 --batch-size 768 --num-workers 4 --log-json results/ablate_small_model.json --log-md results/ablate_small_model.md

ablate-r8: setup
	$(VENV_PYTHON) scripts/train_anomaly_detector.py --device cuda --p 7 --r 8 --d-model 384 --n-heads 8 --n-layers 6 --ffn-dim 1536 --head-hidden 192 --epochs 10 --n-train 131072 --n-val 16384 --batch-size 768 --num-workers 4 --log-json results/ablate_r8.json --log-md results/ablate_r8.md

ablate-p3: setup
	$(VENV_PYTHON) scripts/train_anomaly_detector.py --device cuda --p 3 --r 16 --d-model 384 --n-heads 8 --n-layers 6 --ffn-dim 1536 --head-hidden 192 --epochs 10 --n-train 131072 --n-val 16384 --batch-size 768 --num-workers 4 --log-json results/ablate_p3.json --log-md results/ablate_p3.md

ablate-p5: setup
	$(VENV_PYTHON) scripts/train_anomaly_detector.py --device cuda --p 5 --r 16 --d-model 384 --n-heads 8 --n-layers 6 --ffn-dim 1536 --head-hidden 192 --epochs 10 --n-train 131072 --n-val 16384 --batch-size 768 --num-workers 4 --log-json results/ablate_p5.json --log-md results/ablate_p5.md

ablate-p7: setup
	$(VENV_PYTHON) scripts/train_anomaly_detector.py --device cuda --p 7 --r 16 --d-model 384 --n-heads 8 --n-layers 6 --ffn-dim 1536 --head-hidden 192 --epochs 10 --n-train 131072 --n-val 16384 --batch-size 768 --num-workers 4 --log-json results/ablate_p7.json --log-md results/ablate_p7.md

ablate: ablate-no-contrastive ablate-small-model ablate-r8 ablate-p3 ablate-p5 ablate-p7

# ---------------------------------------------------------------------------
# Open datasets
# ---------------------------------------------------------------------------
adfa: setup
	$(VENV_PYTHON) scripts/run_open_dataset.py $(OPEN_DATASET_ADFA_ARGS)

beth: setup
	$(VENV_PYTHON) scripts/run_open_dataset.py $(OPEN_DATASET_BETH_ARGS)

adfa-stats: setup
	$(VENV_PYTHON) scripts/run_open_dataset.py $(OPEN_DATASET_STATS_ARGS)

open-adfa: adfa
open-beth: beth
open-adfa-stats: adfa-stats

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
analysis: threshold diagnose primes

# ---------------------------------------------------------------------------
# 2-adic / INT8
# ---------------------------------------------------------------------------
hardware: int8

int8: setup
	$(VENV_PYTHON) scripts/verify_int8_2adic.py $(INT8_ARGS)

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean: clean-caches clean-results clean-checkpoints

clean-caches:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-results:
	rm -rf results/*.json results/*.md results/*.pt results/checkpoints results/over_underfit_checkpoints

clean-checkpoints:
	rm -rf results/checkpoints results/over_underfit_checkpoints

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:
	@echo "Targets:"
	@echo "  make setup           Create .venv and install the editable package"
	@echo "  make test            Run unit tests"
	@echo "  make cpu             Run the local CPU benchmark"
	@echo "  make gpu             Run the CUDA benchmark"
	@echo "  make smoke           Run the local CPU sanity-training path"
	@echo "  make train           Run the recommended BCE-first GPU training path"
	@echo "  make hierarchy       Run the hierarchy-rule dataset training path"
	@echo "  make realistic       Run the realistic idle-heavy training path"
	@echo "  make primes          Run p=3,5,7 training comparisons"
	@echo "  make sweep           Run the untrained hierarchy/sparsity benchmark sweep"
	@echo "  make baselines       Run majority/logreg/MLP/transformer/PDAIC baselines"
	@echo "  make eval            Evaluate a trained checkpoint on true/shuffled/random hierarchy"
	@echo "  make threshold       Run training with validation threshold search"
	@echo "  make diagnose        Run the learning vs generalization diagnostic"
	@echo "  make adfa            Download ADFA-LD and run the open dataset benchmark"
	@echo "  make beth            Download BETH and run the open dataset benchmark"
	@echo "  make adfa-stats      Show ADFA-LD stats only, no training"
	@echo "  make analysis        Run the analysis workflows"
	@echo "  make int8            Verify unsigned INT8 against 2-adic arithmetic"
	@echo "  make ablate          Run the full ablation suite"
	@echo "  make clean           Remove caches, checkpoints, and generated outputs"
