#!/usr/bin/env bash
# Phase 2c: M4 data-scaled 5-seed sweep (~overnight). Override data root if needed:
#   export EXTRA_DATA_ROOT="$HOME/Desktop/PhD_Desktop/data_large"
set -euo pipefail
cd "$(dirname "$0")"
EXTRA_ROOT="${EXTRA_DATA_ROOT:-$HOME/Desktop/PhD_Desktop/data_large}"

for seed in 42 123 456 789 2024; do
  python3 train_real_data.py \
    --config configs/filterbank_attention.yaml \
    --model-type filterbank_attention \
    --epochs 30 --num-train 600 --num-test 5 \
    --holdout-per-category 1 \
    --filterbank-K 16 \
    --pretrain-filterbank --fb-init-topk --trainable-filters \
    --filterbank-pretrain-scenarios 80 \
    --temperature-anneal \
    --hybrid-residual --hybrid-adaptive-scale \
    --hybrid-scale-mu 0.05 --hybrid-scale-init 0.2 \
    --hybrid-energy-cap 1.0 --hybrid-scale-max 1.0 \
    --run-fxlms --run-mm-fxlms \
    --chunk-stride 512 \
    --fxlms-parallel-workers 0 \
    --extra-data-root "$EXTRA_ROOT" \
    --extra-scenarios 400 \
    --seed "$seed" \
    --save-dir "outputs/ieee_m4c_seed${seed}"
done

python3 src/experiments/run_paper_evaluation.py \
  --seeds 42 123 456 789 2024 \
  --model-type filterbank_attention \
  --run-dir-contains ieee_m4c_seed \
  --output outputs/ieee_m4c_aggregated
