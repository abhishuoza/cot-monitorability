#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

# Format: <dataset substring>:<output dir name>
DATASETS=(
  "gpqa.jsonl:gpqa-low-N5"
  "scruples_first_person:scruples_fp-low-N5"
  "scruples_suggest_right:scruples_sr-low-N5"
  "scruples_suggest_wrong:scruples_sw-low-N5"
  "wmdp_sandbagging:wmdp-low-N5"
)

for entry in "${DATASETS[@]}"; do
  dataset="${entry%%:*}"
  outname="${entry##*:}"
  outdir="results/$outname"
  mkdir -p "$outdir"
  echo "[chain] START dataset=$dataset out=$outdir at $(date -Iseconds)"
  if "$PY" scripts/run_eval.py \
      --dataset "$dataset" \
      --max-rows-per-dataset 999 \
      --samples-per-row 5 \
      --reasoning-effort low \
      --concurrency 8 \
      --output-dir "$(pwd)/$outdir" > "$outdir/run.log" 2>&1; then
    echo "[chain] DONE dataset=$dataset at $(date -Iseconds)"
  else
    rc=$?
    echo "[chain] FAILED dataset=$dataset rc=$rc at $(date -Iseconds)"
  fi
done
echo "[chain] ALL_DONE at $(date -Iseconds)"
