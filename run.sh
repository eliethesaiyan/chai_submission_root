#!/usr/bin/env bash

set -e

# Always move to the project root.
cd "$(dirname "$0")"

COMMAND="${1:-}"

case "$COMMAND" in

    annotate)
        echo "Generating annotations..."
        python -m src.preprocessing.run
        ;;

    tokenize)
        echo "Testing tokenization..."
        python -m src.training.tokenization
        ;;

    train)
        echo "Training entity extraction model..."
        python -m src.training.train
        ;;

    predict)
        echo "Running entity extraction inference..."
        python -m src.extraction.predictor
        ;;

    guideline)
        echo "Running guideline evaluator..."
        python -m src.guideline_engine.evaluator
        ;;

    explain)
        echo "Running explainability module..."
        python -m src.explainability.explainer
        ;;

    evaluate)
        echo "Running model evaluation..."
        python -m src.evaluation.evaluate
        ;;

    api)
        HOST="${HOST:-0.0.0.0}"
        PORT="${PORT:-8000}"

        echo "Starting API at ${HOST}:${PORT}..."

        exec python -m uvicorn src.api.app:app \
            --host "$HOST" \
            --port "$PORT"
        ;;

    *)
        echo "Usage: ./run.sh <command>"
        echo ""
        echo "Available commands:"
        echo "  annotate    Generate training annotations"
        echo "  tokenize    Test tokenization/BIO alignment"
        echo "  train       Train the entity extraction model"
        echo "  predict     Run entity extraction inference"
        echo "  guideline   Run the guideline evaluator"
        echo "  explain     Run the explainability module"
        echo "  evaluate    Evaluate the entity extraction model"
        echo "  api         Start the FastAPI server"
        exit 1
        ;;
esac