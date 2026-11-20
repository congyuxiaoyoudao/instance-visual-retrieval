#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PYTHON="${PYTHON:-.venv/bin/python}"
mkdir -p logs/coursework
# Fail early on unfinished exercises, before downloading or launching long experiments.
"$PYTHON" -m pytest -q 2>&1 | tee logs/coursework/tests.log
"$PYTHON" scripts/prepare.py 2>&1 | tee logs/coursework/prepare.log
"$PYTHON" -m instance_retrieval.cli deep 2>&1 | tee logs/coursework/deep.log
"$PYTHON" -m instance_retrieval.cli local 2>&1 | tee logs/coursework/local.log
"$PYTHON" -m instance_retrieval.cli tables
"$PYTHON" -m instance_retrieval.cli failures
