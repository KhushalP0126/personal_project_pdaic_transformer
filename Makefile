PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
PIP := $(VENV_PYTHON) -m pip
SETUP_STAMP := $(VENV)/.setup-stamp
ACCEL_DEVICE ?= auto

# ---------------------------------------------------------------------------
# Benchmark defaults
# ---------------------------------------------------------------------------
CPU_ARGS ?= --device cpu --p-list 3 5 --r-list 8 16 --samples 4096 --classes 16 --tokens-per-class 256 --triplets 20000 --distance-pairs 200000
GPU_ARGS ?= --device $(ACCEL_DEVICE) --p-list 3 5 --r-list 8 16 24 32 --samples 16384 --classes 32 --tokens-per-class 128 --triplets 100000 --distance-pairs 1000000

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
TRAIN_GPU_ARGS ?= --device $(ACCEL_DEVICE) --p 3 --r 16 --d-model 192 --n-heads 4 --n-layers 3 --ffn-dim 768 --head-hidden 96 --dropout 0.1 --window-size 32 --attack-fraction 0.35 --attack-min-len 2 --attack-max-len 8 --n-train 32768 --n-val 4096 --samples 16384 --classes 32 --tokens-per-class 128 --epochs 20 --batch-size 128 --grad-accum 4 --lr 2e-4 --weight-decay 1e-2 --warmup-epochs 2 --num-workers 4 --alpha 0.5 --pos-weight 1.0 --margin-pos 0.1 --margin-neg 0.5 --save-every 5
TRAIN_CPU_ARGS ?= --device cpu --p 3 --r 8 --d-model 64 --n-heads 4 --n-layers 2 --ffn-dim 256 --head-hidden 32 --dropout 0.1 --window-size 16 --attack-fraction 0.3 --attack-min-len 2 --attack-max-len 4 --n-train 1024 --n-val 256 --samples 4096 --classes 16 --tokens-per-class 64 --epochs 3 --batch-size 64 --lr 3e-4 --num-workers 0 --save-every 999
TRAIN_ATTENTION_CPU_ARGS ?= --attention --device cpu --p 3 --r 8 --d-model 64 --n-heads 4 --n-layers 2 --ffn-dim 256 --head-hidden 32 --dropout 0.1 --d-digit 8 --window-size 16 --attack-fraction 0.3 --attack-min-len 2 --attack-max-len 4 --n-train 4096 --n-val 512 --samples 4096 --classes 16 --tokens-per-class 64 --epochs 15 --batch-size 64 --lr 3e-4 --warmup-epochs 2 --num-workers 0 --alpha 0.0 --save-every 999
TRAIN_ATTENTION_GPU_ARGS ?= --attention --device $(ACCEL_DEVICE) --p 3 --r 16 --d-model 192 --n-heads 4 --n-layers 3 --ffn-dim 768 --head-hidden 96 --dropout 0.1 --d-digit 8 --window-size 32 --attack-fraction 0.35 --attack-min-len 2 --attack-max-len 8 --n-train 32768 --n-val 4096 --samples 16384 --classes 32 --tokens-per-class 128 --epochs 20 --batch-size 128 --grad-accum 4 --lr 2e-4 --weight-decay 1e-2 --warmup-epochs 2 --num-workers 4 --alpha 0.5 --pos-weight 1.0 --margin-pos 0.1 --margin-neg 0.5 --save-every 5
TRAIN_ATTENTION_BCE_GPU_ARGS ?= --attention --device $(ACCEL_DEVICE) --p 3 --r 16 --d-model 192 --n-heads 4 --n-layers 3 --ffn-dim 768 --head-hidden 96 --dropout 0.1 --d-digit 8 --window-size 32 --attack-fraction 0.30 --attack-min-len 2 --attack-max-len 8 --n-train 32768 --n-val 4096 --samples 16384 --classes 32 --tokens-per-class 128 --epochs 20 --batch-size 128 --grad-accum 4 --lr 1e-4 --weight-decay 1e-2 --warmup-epochs 2 --num-workers 4 --alpha 0.0 --margin-pos 0.1 --margin-neg 0.5 --save-every 5
TRAIN_ATTENTION_RULE_GPU_ARGS ?= --attention --device $(ACCEL_DEVICE) --p 3 --r 16 --d-model 192 --n-heads 4 --n-layers 3 --ffn-dim 768 --head-hidden 96 --dropout 0.1 --d-digit 8 --window-size 32 --hierarchy-rule-dataset --rule-subtree-depth 2 --rule-stay-steps 4 --rule-attack-tokens 1 --attack-fraction 0.30 --n-train 32768 --n-val 4096 --samples 16384 --classes 32 --tokens-per-class 128 --epochs 20 --batch-size 128 --grad-accum 4 --lr 1e-4 --weight-decay 1e-2 --warmup-epochs 2 --num-workers 4 --alpha 0.0 --margin-pos 0.1 --margin-neg 0.5 --save-every 5
TRAIN_ATTENTION_REALISTIC_GPU_ARGS ?= --attention --device $(ACCEL_DEVICE) --p 3 --r 16 --d-model 192 --n-heads 4 --n-layers 3 --ffn-dim 768 --head-hidden 96 --dropout 0.1 --d-digit 8 --window-size 32 --realistic-dataset --realistic-attack-fraction 0.005 --idle-fraction 0.70 --n-train 32768 --n-val 4096 --samples 16384 --classes 32 --tokens-per-class 128 --epochs 20 --batch-size 128 --grad-accum 4 --lr 1e-4 --weight-decay 1e-2 --warmup-epochs 2 --num-workers 4 --alpha 0.0 --margin-pos 0.1 --margin-neg 0.5 --save-every 5
COMPARE_TRAIN_BASE_ARGS ?= --device $(ACCEL_DEVICE) --r 16 --d-model 192 --n-heads 4 --n-layers 3 --ffn-dim 768 --head-hidden 96 --dropout 0.1 --window-size 32 --attack-fraction 0.35 --attack-min-len 2 --attack-max-len 8 --n-train 32768 --n-val 4096 --samples 16384 --classes 32 --tokens-per-class 128 --epochs 8 --batch-size 128 --grad-accum 4 --lr 2e-4 --weight-decay 1e-2 --warmup-epochs 2 --num-workers 4 --alpha 0.5 --pos-weight 1.0 --margin-pos 0.1 --margin-neg 0.5 --save-every 5
COMPARE_PDAIC_TRAIN_BASE_ARGS ?= --attention --device $(ACCEL_DEVICE) --r 16 --d-model 192 --n-heads 4 --n-layers 3 --ffn-dim 768 --head-hidden 96 --dropout 0.1 --d-digit 8 --window-size 32 --attack-fraction 0.35 --attack-min-len 2 --attack-max-len 8 --n-train 32768 --n-val 4096 --samples 16384 --classes 32 --tokens-per-class 128 --epochs 8 --batch-size 128 --grad-accum 4 --lr 2e-4 --weight-decay 1e-2 --warmup-epochs 2 --num-workers 4 --alpha 0.0 --margin-pos 0.1 --margin-neg 0.5 --save-every 5
SWEEP_P_ARGS ?= --sweep-p-bases --device cpu --p-list 3 5 7 --sweep-r 8 --sweep-samples 512 --sweep-classes 16 --sweep-tokens-per-class 64 --sweep-window-size 32 --sweep-attack-fraction 0.30 --sweep-attack-min-len 2 --sweep-attack-max-len 8 --sweep-batch-size 64 --sweep-batches 4
BASELINE_RULE_ARGS ?= --device $(ACCEL_DEVICE) --hierarchy-rule-dataset --p 3 --r 16 --samples 16384 --classes 32 --tokens-per-class 128 --window-size 32 --attack-fraction 0.30 --rule-subtree-depth 2 --rule-stay-steps 4 --rule-attack-tokens 1 --train-samples 32768 --val-samples 4096 --epochs 20 --batch-size 128 --lr 2e-4 --d-model 192 --n-heads 4 --n-layers 3 --d-digit 8 --output-json results/baseline_report.json
TRAINED_EVAL_ARGS ?= --device $(ACCEL_DEVICE) --trained-eval-checkpoint results/checkpoints/best.pt --trained-eval-dataset hierarchy_rules --trained-eval-samples 512 --trained-eval-window-size 32 --trained-eval-attack-fraction 0.30 --trained-eval-batch-size 64
COMPARE_ANALYSIS_ARGS ?= --p-list 3 5 7 --output-json results/prime_comparison.json --output-md results/prime_comparison.md
OPEN_DATASET_BETH_ARGS ?= --dataset beth --data-dir ./data/beth --p 3 --r 8 --window-size 32 --stride 4 --d-model 128 --n-heads 4 --n-layers 2 --epochs 5 --batch-size 256 --device $(ACCEL_DEVICE)
CPU_ONE_EPOCH_TRAIN_ARGS ?= --device cpu --p 3 --r 8 --d-model 32 --n-heads 4 --n-layers 1 --ffn-dim 64 --head-hidden 16 --dropout 0.1 --window-size 16 --attack-fraction 0.30 --attack-min-len 2 --attack-max-len 4 --n-train 256 --n-val 64 --samples 512 --classes 8 --tokens-per-class 32 --epochs 1 --batch-size 32 --lr 3e-4 --num-workers 0 --alpha 0.0 --save-every 999 --max-seq-len 32
CPU_ONE_EPOCH_BASELINE_ARGS ?= --device cpu --hierarchy-rule-dataset --p 3 --r 8 --samples 512 --classes 8 --tokens-per-class 32 --window-size 16 --attack-fraction 0.30 --rule-subtree-depth 2 --rule-stay-steps 4 --rule-attack-tokens 1 --train-samples 256 --val-samples 64 --epochs 1 --batch-size 32 --lr 2e-4 --d-model 32 --n-heads 4 --n-layers 1 --d-digit 8 --output-json results/cpu_1epoch_baselines.json
IP_CPU_ARGS ?= --device cpu --train-samples 512 --val-samples 128 --window-size 16 --prefix-len 24 --num-prefixes 16 --attack-fraction 0.30 --attack-min-len 1 --attack-max-len 4 --epochs 1 --batch-size 128 --lr 3e-4 --d-model 64 --n-heads 4 --n-layers 1 --d-digit 8 --output-json results/ip_synthetic.json --output-md results/ip_synthetic.md
IP_DAY4_ARGS ?= --device cpu --train-samples 2048 --val-samples 512 --prefix-len 24 --num-prefixes 32 --attack-fraction 0.30 --attack-min-len 1 --attack-max-len 4 --batch-size 256 --lr 3e-4 --d-digit 8 --output-json results/ip_day4_tuning.json --output-md results/ip_day4_tuning.md
IP_DAY5_ARGS ?= --device cpu --seeds 20260504 20260505 20260506 --train-samples 2048 --val-samples 512 --window-size 16 --prefix-len 24 --num-prefixes 32 --attack-fraction 0.30 --attack-min-len 1 --attack-max-len 4 --epochs 3 --batch-size 256 --lr 3e-4 --d-model 64 --n-heads 4 --n-layers 1 --d-digit 8 --dropout 0.1 --output-json results/ip_day5_multiseed.json --output-md results/ip_day5_multiseed.md
IP_DAY5_FAST_ARGS ?= --device cpu --seeds 20260504 20260505 20260506 --train-samples 1024 --val-samples 256 --window-size 16 --prefix-len 24 --num-prefixes 32 --attack-fraction 0.30 --attack-min-len 1 --attack-max-len 4 --epochs 3 --batch-size 256 --lr 3e-4 --d-model 64 --n-heads 4 --n-layers 1 --d-digit 8 --dropout 0.1 --output-json results/ip_day5_multiseed.json --output-md results/ip_day5_multiseed.md

# ---------------------------------------------------------------------------
# Analysis defaults
# ---------------------------------------------------------------------------
ANALYSIS_ARGS ?= --device $(ACCEL_DEVICE) --p 3 --r 8 --d-model 128 --n-heads 4 --n-layers 2 --ffn-dim 512 --head-hidden 64 --dropout 0.1 --window-size 16 --attack-fraction 0.3 --attack-min-len 2 --attack-max-len 4 --n-train 4096 --n-val 1024 --samples 4096 --classes 16 --tokens-per-class 64 --epochs 8 --batch-size 128 --grad-accum 1 --lr 3e-4 --num-workers 0 --alpha 0.5 --pos-weight 1.0 --margin-pos 0.1 --margin-neg 0.5 --save-every 999

# ---------------------------------------------------------------------------
# 2-adic / INT8 defaults
# ---------------------------------------------------------------------------
INT8_ARGS ?= --r 8

.DEFAULT_GOAL := help

.PHONY: all setup test cpu gpu \
        int8 hardware \
        smoke train vanilla hierarchy realistic primes pdaic-primes compare-analysis sweep baselines eval threshold diagnose ablate \
        beth audit cpu-all-1epoch ip-cpu ip-day4 ip-day5 ip-day5-fast \
        ablate-no-contrastive ablate-small-model ablate-r8 ablate-p3 ablate-p5 ablate-p7 \
        clean help clean-results clean-caches clean-checkpoints

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
all: setup test cpu

$(VENV_PYTHON):
	$(PYTHON) -m venv $(VENV)

$(SETUP_STAMP): pyproject.toml $(VENV_PYTHON)
	$(PIP) install -e .
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

vanilla: setup
	$(VENV_PYTHON) scripts/train_anomaly_detector.py $(TRAIN_GPU_ARGS)

hierarchy: setup
	$(VENV_PYTHON) scripts/train_anomaly_detector.py $(TRAIN_ATTENTION_RULE_GPU_ARGS)

realistic: setup
	$(VENV_PYTHON) scripts/train_anomaly_detector.py $(TRAIN_ATTENTION_REALISTIC_GPU_ARGS)

primes: setup
	@set -e; \
	for p in 3 5 7; do \
		$(VENV_PYTHON) scripts/train_anomaly_detector.py $(COMPARE_TRAIN_BASE_ARGS) --p $$p --log-json results/compare_p$$p.json --log-md results/compare_p$$p.md; \
	done

pdaic-primes: setup
	@set -e; \
	for p in 3 5 7; do \
		$(VENV_PYTHON) scripts/train_anomaly_detector.py $(COMPARE_PDAIC_TRAIN_BASE_ARGS) --p $$p --log-json results/compare_pdaic_p$$p.json --log-md results/compare_pdaic_p$$p.md; \
	done

compare-analysis: setup
	$(VENV_PYTHON) scripts/compare_prime_runs.py $(COMPARE_ANALYSIS_ARGS)

sweep: setup
	$(VENV_PYTHON) scripts/run_padic_benchmark.py $(SWEEP_P_ARGS)

baselines: setup
	$(VENV_PYTHON) scripts/run_baselines.py $(BASELINE_RULE_ARGS)

eval: setup
	$(VENV_PYTHON) scripts/run_padic_benchmark.py $(TRAINED_EVAL_ARGS)

threshold: setup
	$(VENV_PYTHON) scripts/train_anomaly_detector.py $(TRAIN_GPU_ARGS) --log-json results/tune_threshold.json --log-md results/tune_threshold.md

diagnose: setup
	$(VENV_PYTHON) scripts/over_underfit.py --device $(ACCEL_DEVICE) --log-json results/over_underfit.json --log-md results/over_underfit.md

ablate-no-contrastive: setup
	$(VENV_PYTHON) scripts/train_anomaly_detector.py $(TRAIN_GPU_ARGS) --alpha 0.0 --log-json results/ablate_no_contrastive.json --log-md results/ablate_no_contrastive.md

ablate-small-model: setup
	$(VENV_PYTHON) scripts/train_anomaly_detector.py --device $(ACCEL_DEVICE) --p 7 --r 16 --d-model 128 --n-heads 4 --n-layers 2 --ffn-dim 512 --head-hidden 64 --epochs 8 --n-train 32768 --n-val 4096 --batch-size 128 --grad-accum 4 --num-workers 4 --log-json results/ablate_small_model.json --log-md results/ablate_small_model.md

ablate-r8: setup
	$(VENV_PYTHON) scripts/train_anomaly_detector.py --device $(ACCEL_DEVICE) --p 7 --r 8 --d-model 192 --n-heads 4 --n-layers 3 --ffn-dim 768 --head-hidden 96 --epochs 8 --n-train 32768 --n-val 4096 --batch-size 128 --grad-accum 4 --num-workers 4 --log-json results/ablate_r8.json --log-md results/ablate_r8.md

ablate-p3: setup
	$(VENV_PYTHON) scripts/train_anomaly_detector.py --device $(ACCEL_DEVICE) --p 3 --r 16 --d-model 192 --n-heads 4 --n-layers 3 --ffn-dim 768 --head-hidden 96 --epochs 8 --n-train 32768 --n-val 4096 --batch-size 128 --grad-accum 4 --num-workers 4 --log-json results/ablate_p3.json --log-md results/ablate_p3.md

ablate-p5: setup
	$(VENV_PYTHON) scripts/train_anomaly_detector.py --device $(ACCEL_DEVICE) --p 5 --r 16 --d-model 192 --n-heads 4 --n-layers 3 --ffn-dim 768 --head-hidden 96 --epochs 8 --n-train 32768 --n-val 4096 --batch-size 128 --grad-accum 4 --num-workers 4 --log-json results/ablate_p5.json --log-md results/ablate_p5.md

ablate-p7: setup
	$(VENV_PYTHON) scripts/train_anomaly_detector.py --device $(ACCEL_DEVICE) --p 7 --r 16 --d-model 192 --n-heads 4 --n-layers 3 --ffn-dim 768 --head-hidden 96 --epochs 8 --n-train 32768 --n-val 4096 --batch-size 128 --grad-accum 4 --num-workers 4 --log-json results/ablate_p7.json --log-md results/ablate_p7.md

ablate: ablate-no-contrastive ablate-small-model ablate-r8 ablate-p3 ablate-p5 ablate-p7

# ---------------------------------------------------------------------------
# Open datasets
# ---------------------------------------------------------------------------
beth: setup
	$(VENV_PYTHON) scripts/run_open_dataset.py $(OPEN_DATASET_BETH_ARGS)

audit: setup
	$(VENV_PYTHON) scripts/audit_datasets.py

cpu-all-1epoch: setup
	@printf "\n[####----------------] 1/5 vanilla synthetic\n"
	$(VENV_PYTHON) scripts/train_anomaly_detector.py $(CPU_ONE_EPOCH_TRAIN_ARGS) --checkpoint-dir results/checkpoints/cpu_1epoch_vanilla --log-json results/cpu_1epoch_vanilla.json --log-md results/cpu_1epoch_vanilla.md
	@printf "\n[########------------] 2/5 PDAIC synthetic\n"
	$(VENV_PYTHON) scripts/train_anomaly_detector.py --attention --d-digit 8 $(CPU_ONE_EPOCH_TRAIN_ARGS) --checkpoint-dir results/checkpoints/cpu_1epoch_pdaic --log-json results/cpu_1epoch_pdaic.json --log-md results/cpu_1epoch_pdaic.md
	@printf "\n[############--------] 3/5 PDAIC hierarchy-rule\n"
	$(VENV_PYTHON) scripts/train_anomaly_detector.py --attention --d-digit 8 $(CPU_ONE_EPOCH_TRAIN_ARGS) --hierarchy-rule-dataset --rule-subtree-depth 2 --rule-stay-steps 4 --rule-attack-tokens 1 --checkpoint-dir results/checkpoints/cpu_1epoch_hierarchy --log-json results/cpu_1epoch_hierarchy.json --log-md results/cpu_1epoch_hierarchy.md
	@printf "\n[################----] 4/5 PDAIC realistic\n"
	$(VENV_PYTHON) scripts/train_anomaly_detector.py --attention --d-digit 8 $(CPU_ONE_EPOCH_TRAIN_ARGS) --realistic-dataset --realistic-attack-fraction 0.05 --idle-fraction 0.70 --checkpoint-dir results/checkpoints/cpu_1epoch_realistic --log-json results/cpu_1epoch_realistic.json --log-md results/cpu_1epoch_realistic.md
	@printf "\n[####################] 5/5 baseline suite\n"
	$(VENV_PYTHON) scripts/run_baselines.py $(CPU_ONE_EPOCH_BASELINE_ARGS)
	@printf "\n[####################] done CPU one-epoch sweep\n"

ip-cpu: setup
	$(VENV_PYTHON) scripts/run_ip_experiment.py $(IP_CPU_ARGS)

ip-day4: setup
	$(VENV_PYTHON) scripts/tune_ip_day4.py $(IP_DAY4_ARGS)

ip-day5: setup
	$(VENV_PYTHON) scripts/run_ip_day5_multiseed.py $(IP_DAY5_ARGS)

ip-day5-fast: setup
	$(VENV_PYTHON) scripts/run_ip_day5_multiseed.py $(IP_DAY5_FAST_ARGS)

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
analysis: threshold diagnose primes pdaic-primes compare-analysis

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
	@echo "  make vanilla         Run the standard transformer GPU training path"
	@echo "  make hierarchy       Run the hierarchy-rule dataset training path"
	@echo "  make realistic       Run the realistic idle-heavy training path"
	@echo "  make primes          Run vanilla p=3,5,7 training comparisons"
	@echo "  make pdaic-primes    Run PDAIC p=3,5,7 training comparisons"
	@echo "  make compare-analysis  Compare vanilla vs PDAIC prime sweep logs"
	@echo "  make sweep           Run the untrained hierarchy/sparsity benchmark sweep"
	@echo "  make baselines       Run majority/logreg/MLP/transformer/PDAIC baselines"
	@echo "  make eval            Evaluate a trained checkpoint on true/shuffled/random hierarchy"
	@echo "  make threshold       Run training with validation threshold search"
	@echo "  make diagnose        Run the learning vs generalization diagnostic"
	@echo "  make beth            Download BETH with Kaggle CLI if needed and run the benchmark"
	@echo "  make audit           Audit synthetic datasets for imbalance, leakage, and artifacts"
	@echo "  make cpu-all-1epoch  Run one CPU epoch across vanilla, PDAIC, hierarchy, realistic, and baselines"
	@echo "  make ip-cpu          Run the CPU IP-prefix synthetic experiment"
	@echo "  make ip-day4         Run the small CPU IP-prefix tuning pass"
	@echo "  make ip-day5         Run the Day 5 multi-seed IP-prefix validation"
	@echo "  make ip-day5-fast    Run the smaller Day 5 multi-seed smoke validation"
	@echo "  make analysis        Run the analysis workflows"
	@echo "  make int8            Verify unsigned INT8 against 2-adic arithmetic"
	@echo "  make ablate          Run the full ablation suite"
	@echo "  make clean           Remove caches, checkpoints, and generated outputs"
