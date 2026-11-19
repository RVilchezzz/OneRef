#!/usr/bin/env bash
# The single test run, unattended (about 40 h on the RTX 3060 Laptop): S with the frozen config, then B3.
# Run from the repo root, with `OLLAMA_CONTEXT_LENGTH=8192 ollama serve` running:
#   nohup bash scripts/run_test.sh > outputs/logs/run_test.log 2>&1 &
# Resumable: every pass caches per image, so re-running the script after a crash continues where it stopped.
#
# 0. Waits for the dev B3 run (dev_B3_k3) if it is still on the GPU, and scores it against the dev runs.
# 1. S on the 2,000-image stratified test sample (configs/2026-11-19_frozen.env, never edited after this).
# 2. CPU re-aggregations of the test cache: B1 (reader A alone), B2 (reader B alone), AB vote without
#    the verifier, S without the judge. Same answers as in S, no extra model calls.
# 3. B3 on the same images: reader A sampled k = 3 times (reuses S's expert pass).
# 4. Comparison table with McNemar and bootstrap CIs.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:scripts PYDANTIC_AI_NO_BANNER=1
PY=.venv/bin/python
N=2000
S=outputs/test_S
B3=outputs/test_B3_k3
R=outputs/reagg/test_S
OUT=results/test/frozen_2026-11-19
mkdir -p outputs/logs "$OUT"

# 0. Dev B3 (GPU) must finish first; then score it (CPU).
while pgrep -f "run-name dev_B3_k3$" >/dev/null; do sleep 60; done
if [ "$(ls outputs/dev_B3_k3/7_aggregator 2>/dev/null | wc -l)" -ge 835 ]; then
  $PY scripts/compare_runs.py S_judge=outputs/reagg/dev_full_S_judge/override_C2 \
    B1_qwen=outputs/reagg/dev_full_S_pre/B1_qwen B3_k3=outputs/dev_B3_k3 \
    --out results/compare/2026-11-18_dev_B3 > /dev/null
  echo "$(date '+%F %T') dev B3 scored: results/compare/2026-11-18_dev_B3"
fi

# 1. S, frozen config, once.
echo "$(date '+%F %T') test S: start"
( set -a; source configs/2026-11-19_frozen.env; set +a
  $PY -m inspector.main --split test --sample $N --frozen-config --run-name test_S ) > outputs/logs/test_S.log 2>&1
echo "$(date '+%F %T') test S: done"

# 2. Baselines and ablations from the cache (CPU).
( set -a; source configs/2026-11-19_frozen.env; set +a
  $PY scripts/reaggregate.py $S --from-pass 3_reader_b --out $R/pre \
    --variant 'B1_qwen={"aggregate_readers": ["reader_a"], "expert_weight_discrimination": 0}' \
    --variant 'B2_internvl={"aggregate_readers": ["reader_b"], "expert_weight_discrimination": 0}' \
    --variant 'AB_noverif={"judge_mode": "off"}'
  $PY scripts/reaggregate.py $S --out $R/post \
    --variant 'S={}' \
    --variant 'S_nojudge={"judge_mode": "off"}' \
    --variant 'S_noexpertvote={"expert_weight_discrimination": 0, "expert_weight_by_source": {}}' )

# 3. B3, same test images.
echo "$(date '+%F %T') test B3: start"
mkdir -p $B3
[ -d $B3/1_expert ] || cp -r $S/1_expert $B3/
( set -a; source configs/2026-11-17_small_models.env; source configs/2026-11-18_dev_B3_k3.env; set +a
  $PY -m inspector.main --split test --sample $N --frozen-config --steps expert,reader_a,aggregator \
    --run-name test_B3_k3 ) > outputs/logs/test_B3_k3.log 2>&1
echo "$(date '+%F %T') test B3: done"

# 4. Final table: S first, so every McNemar row "S vs X" is the success test.
$PY scripts/compare_runs.py S=$R/post/S B1_qwen=$R/pre/B1_qwen B3_k3=$B3 B2_internvl=$R/pre/B2_internvl \
  AB_noverif=$R/pre/AB_noverif S_nojudge=$R/post/S_nojudge S_noexpertvote=$R/post/S_noexpertvote \
  --out "$OUT"
echo "$(date '+%F %T') done: $OUT"
