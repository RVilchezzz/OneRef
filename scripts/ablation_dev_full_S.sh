#!/usr/bin/env bash
# Ablations of the dev run dev_full_S (dev only, never test). Run from the repo root:
#   bash scripts/ablation_dev_full_S.sh            # waits for dev_full_S if it is still running
# CPU parts (re-aggregation, sweep) reuse the cache; the only GPU part is the rules-only verifier run,
# which reuses passes 1-3 (expert, reader A, reader B) and recomputes only the critic and the reopens.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source configs/2026-11-17_small_models.env; set +a
export PYTHONPATH=src:scripts
PY=.venv/bin/python
RUN=outputs/dev_full_S
RULES=outputs/dev_full_S_rules
OUT=results/compare/2026-11-17_dev_full_S
mkdir -p "$OUT" outputs/logs

# 0. Wait for the main run (its last pass is 7_aggregator with 835 states).
while pgrep -f "run-name dev_full_S$" >/dev/null; do sleep 60; done
n=$(ls "$RUN/7_aggregator" 2>/dev/null | wc -l)
[ "$n" -ge 835 ] || { echo "dev_full_S is not finished ($n/835 in 7_aggregator)"; exit 1; }

# 1. Main result of S.
$PY scripts/compare_runs.py S=$RUN --out "$OUT/S"

# 2. Re-aggregation from the cache (CPU). Before the verifier = pass 3_reader_b; after = last pass.
$PY scripts/reaggregate.py $RUN --from-pass 3_reader_b --out outputs/reagg/dev_full_S_pre \
  --variant 'B1_qwen={"aggregate_readers": ["reader_a"], "expert_weight_discrimination": 0}' \
  --variant 'B2_internvl={"aggregate_readers": ["reader_b"], "expert_weight_discrimination": 0}' \
  --variant 'AB_noverif={}' \
  --variant 'AB_noverif_noexpertvote={"expert_weight_discrimination": 0}'
$PY scripts/reaggregate.py $RUN --out outputs/reagg/dev_full_S_post \
  --variant 'S={}' \
  --variant 'S_noexpertvote={"expert_weight_discrimination": 0}' \
  --variant 'S_A_only={"aggregate_readers": ["reader_a"]}'
P=outputs/reagg
$PY scripts/compare_runs.py S=$P/dev_full_S_post/S B1_qwen=$P/dev_full_S_pre/B1_qwen \
  B2_internvl=$P/dev_full_S_pre/B2_internvl AB_noverif=$P/dev_full_S_pre/AB_noverif \
  AB_noverif_noexpertvote=$P/dev_full_S_pre/AB_noverif_noexpertvote \
  S_noexpertvote=$P/dev_full_S_post/S_noexpertvote S_A_only=$P/dev_full_S_post/S_A_only \
  --out "$OUT/ablations"

# 3. Calibration of the expert vote on discrimination (dev only), after the verifier.
$PY scripts/sweep_discrimination.py $RUN --out "$OUT/discrimination"

# 4. Verifier: Gemma vs rules only (GPU: critic + reopens only).
mkdir -p "$RULES"
for p in 1_expert 2_reader_a 3_reader_b; do [ -d "$RULES/$p" ] || cp -r "$RUN/$p" "$RULES/"; done
ONEREF_USE_LLM_CRITIC=false $PY -m inspector.main --split dev --run-name dev_full_S_rules \
  > outputs/logs/dev_full_S_rules.log 2>&1
$PY scripts/compare_runs.py gemma=$RUN rules=$RULES noverif=$P/dev_full_S_pre/AB_noverif --out "$OUT/verifier"
$PY scripts/reopen_stats.py gemma=$RUN rules=$RULES --out "$OUT/verifier"

echo "done: $OUT"
