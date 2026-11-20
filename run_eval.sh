#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PYTHON="${PYTHON:-.venv/bin/python}"
mkdir -p logs/coursework
"$PYTHON" -m instance_retrieval.cli eval 2>&1 | tee logs/coursework/reevaluate.log
