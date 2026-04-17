#!/usr/bin/env bash
# Phase 2c A100 sweep — 5 seeds, data-scaled, CUDA, parallel FxLMS residual.
# Diff vs Mac sweep: 50 epochs (vs 30), 800 train (vs 600), 8 parallel workers
# (vs 0), save-dir suffix ieee_a100_seed* (vs ieee_m4c_seed*), config override
# to use larger batch size appropriate for A100 80 GB.
#
# Override data path if needed:
#   export EXTRA_DATA_ROOT=/path/on/a100/to/data_large
set -euo pipefail
cd "$(dirname "$0")"
EXTRA_ROOT="${EXTRA_DATA_ROOT:-$HOME/data/data_large}"

if [ ! -d "$EXTRA_ROOT" ]; then
  echo "ERROR: EXTRA_DATA_ROOT does not exist: $EXTRA_ROOT"
  echo "       set it with: export EXTRA_DATA_ROOT=/correct/path/to/data_large"
  exit 1
fi

for seed in 42 123 456 789 2024; do
  python3 train_real_data.py \
    --config configs/filterbank_attention_a100.yaml \
    --model-type filterbank_attention \
    --epochs 50 --num-train 800 --num-test 5 \
    --holdout-per-category 1 \
    --filterbank-K 16 \
    --pretrain-filterbank --fb-init-topk --trainable-filters \
    --filterbank-pretrain-scenarios 100 \
    --temperature-anneal \
    --hybrid-residual --hybrid-adaptive-scale \
    --hybrid-scale-mu 0.05 --hybrid-scale-init 0.2 \
    --hybrid-energy-cap 1.0 --hybrid-scale-max 1.0 \
    --run-fxlms --run-mm-fxlms \
    --chunk-stride 512 \
    --fxlms-parallel-workers 8 \
    --extra-data-root "$EXTRA_ROOT" \
    --extra-scenarios 400 \
    --seed "$seed" \
    --save-dir "outputs/ieee_a100_seed${seed}"
done

python3 src/experiments/run_paper_evaluation.py \
  --seeds 42 123 456 789 2024 \
  --model-type filterbank_attention \
  --run-dir-contains ieee_a100_seed \
  --output outputs/ieee_a100_aggregated
