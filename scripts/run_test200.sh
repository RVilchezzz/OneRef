#!/usr/bin/env bash
# Presentation test (Nov 25): 200 stratified test images, frozen config. See
# docs/reports/2026-11-25_test_protocol_change.md. Resumable and self-restarting: if a step fails (laptop suspended,
# GPU lost, Ollama down) it restarts Ollama and runs the same command again; finished images come from the cache.
#   setsid nohup bash scripts/run_test200.sh > outputs/logs/run_test200.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:scripts PYDANTIC_AI_NO_BANNER=1
PY=.venv/bin/python
N=200
S=outputs/test200_S; NH=outputs/test200_qwen_alone; B3=outputs/test200_B3_k3
R=outputs/reagg/test200; OUT=results/test/test200_2026-11-25
mkdir -p outputs/logs "$OUT"
log() { echo "$(date '+%F %T') $*"; }

ensure_ollama() {
  curl -sf -m 5 localhost:11434/api/version >/dev/null && return
  log "ollama down: restarting"
  OLLAMA_CONTEXT_LENGTH=8192 setsid nohup ollama serve >> outputs/logs/ollama_serve_1125.log 2>&1 < /dev/null &
  sleep 8
}
# run <run name> <steps or ''> <env file...>
run() {
  local name=$1 steps=$2; shift 2
  local tries=0
  until [ -f "outputs/$name/.done" ]; do
    tries=$((tries + 1)); [ $tries -gt 20 ] && { log "$name: gave up after 20 tries"; return 1; }
    ensure_ollama
    log "$name: try $tries"
    if ( set -a; for e in "$@"; do source "$e"; done; set +a
         $PY -m inspector.main --split test --sample $N --frozen-config ${steps:+--steps $steps} --run-name "$name" ) \
         >> "outputs/logs/$name.log" 2>&1; then touch "outputs/$name/.done"
    else log "$name: failed, restarting Ollama"; pkill -x ollama; sleep 15; fi
  done
  log "$name: done"
}
score() {  # tables with whatever is finished
  local f=configs/2026-11-19_frozen.env
  ( set -a; source $f; set +a
    $PY scripts/reaggregate.py $S --from-pass 3_reader_b --out $R/pre \
      --variant 'qwen_expert={"aggregate_readers": ["reader_a"], "expert_weight_discrimination": 0, "expert_weight_by_source": {}}' \
      --variant 'internvl_expert={"aggregate_readers": ["reader_b"], "expert_weight_discrimination": 0, "expert_weight_by_source": {}}'
    $PY scripts/reaggregate.py $S --from-pass 6b_judge --out $R/post --variant 'S_nojudge={"judge_mode": "off"}' ) > /dev/null
  local runs="S=$S qwen_alone=$NH qwen_expert=$R/pre/qwen_expert S_nojudge=$R/post/S_nojudge internvl_expert=$R/pre/internvl_expert"
  [ -f $NH/.done ] || runs=${runs/qwen_alone=$NH /}
  [ -f $B3/.done ] && runs="$runs B3_k3=$B3"
  $PY scripts/compare_runs.py $runs --out "$OUT" > /dev/null && log "scored: $OUT"
}

run test200_S "" configs/2026-11-19_frozen.env || exit 1
score
mkdir -p $NH $B3
[ -d $NH/1_expert ] || cp -r $S/1_expert $NH/
[ -d $B3/1_expert ] || cp -r $S/1_expert $B3/
run test200_qwen_alone expert,reader_a,aggregator configs/2026-11-17_small_models.env configs/2026-11-25_single_model_nohint.env && score
run test200_B3_k3 expert,reader_a,aggregator configs/2026-11-17_small_models.env configs/2026-11-18_dev_B3_k3.env && score
log "all done"
