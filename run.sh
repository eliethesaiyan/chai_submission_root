#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
COMMAND="${1:-demo}"
if [[ $# -gt 0 ]]; then shift; fi
case "$COMMAND" in
    demo) exec "$PYTHON_BIN" -m src.extraction.demo "$@" ;;
    annotate) exec "$PYTHON_BIN" -m src.preprocessing.run "$@" ;;
    tokenize) exec "$PYTHON_BIN" -m src.training.tokenization "$@" ;;
    train) exec "$PYTHON_BIN" -u -m src.training.run hydra.run.dir=outputs/training hydra.output_subdir=null "$@" ;;
    predict) exec "$PYTHON_BIN" -u -m src.extraction.run hydra.run.dir=outputs/inference hydra.output_subdir=null "$@" ;;
    guideline) exec "$PYTHON_BIN" -m src.guideline_engine.run hydra.run.dir=outputs/guideline hydra.output_subdir=null "$@" ;;
    explain) exec "$PYTHON_BIN" -m src.explainability.explainer "$@" ;;
    evaluate) exec "$PYTHON_BIN" -m src.evaluation.run "$@" ;;
    test) exec "$PYTHON_BIN" -m pytest -q -p no:cacheprovider src/evaluation/test_system.py "$@" ;;
    api) exec "$PYTHON_BIN" -m uvicorn src.api.app:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" "$@" ;;
    package) exec "$PYTHON_BIN" -m src.evaluation.package "$@" ;;
    all)
        "$PYTHON_BIN" -m src.preprocessing.run
        "$PYTHON_BIN" -m src.evaluation.run
        "$PYTHON_BIN" -m src.explainability.explainer
        "$PYTHON_BIN" -m pytest -q -p no:cacheprovider src/evaluation/test_system.py
        ;;
    *)
        echo "Usage: ./run.sh [demo|annotate|tokenize|train|predict|guideline|explain|evaluate|test|api|package|all]"
        echo "No command runs the three-note demo using saved model weights."
        echo "Set PYTHON_BIN to choose an environment. Training overrides: ./run.sh train trainer.num_epochs=15"
        exit 1 ;;
esac
