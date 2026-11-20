#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-.venv/bin/python}"
mkdir -p logs/coursework
"$PYTHON" -m pytest -q 2>&1 | tee logs/coursework/tests.log
"$PYTHON" scripts/make_submission.py
