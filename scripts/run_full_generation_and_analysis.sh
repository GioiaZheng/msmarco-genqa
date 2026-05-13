#!/usr/bin/env bash
# Run BM25 + reranked generation on the full eligible dev/small pool, then
# the comparative analysis. Assumes:
# - outputs/week02_bm25/run.tsv exists (W2 baseline)
# - outputs/week05_reranker_full/run.tsv covers all 6,980 dev/small queries
#
# Designed to be the "second half" of the full dev/small experiment: kick
# off once the long reranker run finishes. Idempotent — re-running just
# overwrites the same output dirs.

set -euo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

BM25_OUT="outputs/week03_generation_bm25_full"
RERANK_OUT="outputs/week03_generation_reranked_full"
ANALYSIS_OUT="outputs/week06_analysis"

echo "=== 0/3 Pre-flight: validate full reranked run.tsv ==="
# Validates: 6,980 qids present, max rank = 100, no duplicate (qid, rank)
# pairs, manifest records resume/chunking info. Exits non-zero on failure;
# `set -e` then prevents the generation step from running on a bad input.
python3 scripts/validate_full_rerank.py \
    --run-tsv outputs/week05_reranker_full/run.tsv \
    --manifest outputs/week05_reranker_full/manifest.json \
    --expected-qids 6980 \
    --rerank-top-k 100

echo "=== 1/3 BM25 generation on full eligible pool ==="
# Restrict eligibility to queries present in the full reranked run so both
# generation sides evaluate on exactly the same eligible set. --num-eval-queries
# 99999 collapses the cap so we use ALL eligible (the runner already clamps to
# min(n_eval, len(eligible))).
python3 experiments/run_generation_baseline.py \
    --input-run outputs/week02_bm25/run.tsv \
    --output-dir "${BM25_OUT}" \
    --retrieval-source bm25 \
    --restrict-to-run outputs/week05_reranker_full/run.tsv \
    --num-eval-queries 99999 \
    >> "${LOG_DIR}/gen_bm25_full.log" 2>&1

echo "=== 2/3 Reranked generation on full eligible pool ==="
python3 experiments/run_generation_baseline.py \
    --input-run outputs/week05_reranker_full/run.tsv \
    --output-dir "${RERANK_OUT}" \
    --retrieval-source reranked \
    --num-eval-queries 99999 \
    >> "${LOG_DIR}/gen_reranked_full.log" 2>&1

echo "=== 3/3 Comparative analysis ==="
python3 scripts/analyze_generation_rerank.py \
    --bm25-dir "${BM25_OUT}" \
    --reranked-dir "${RERANK_OUT}" \
    --output-dir "${ANALYSIS_OUT}" \
    >> "${LOG_DIR}/analysis.log" 2>&1

echo "All done. Outputs:"
echo "  ${BM25_OUT}/"
echo "  ${RERANK_OUT}/"
echo "  ${ANALYSIS_OUT}/  (report.md, summary.json, qualitative_examples.json)"
