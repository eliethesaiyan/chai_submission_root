# Clinical Note AI

[![GitHub stars](https://img.shields.io/github/stars/eliethesaiyan/chai_submission_root?style=flat&logo=github)](https://github.com/eliethesaiyan/chai_submission_root/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/eliethesaiyan/chai_submission_root?style=flat&logo=github)](https://github.com/eliethesaiyan/chai_submission_root/forks)
[![GitHub issues](https://img.shields.io/github/issues/eliethesaiyan/chai_submission_root?style=flat&logo=github)](https://github.com/eliethesaiyan/chai_submission_root/issues)
[![Last commit](https://img.shields.io/github/last-commit/eliethesaiyan/chai_submission_root/main?style=flat&logo=github)](https://github.com/eliethesaiyan/chai_submission_root/commits/main/)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](requirements.txt)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white)](src/training/train.py)
[![DistilBERT](https://img.shields.io/badge/Model-DistilBERT-FFD21E)](src/extraction/predictor.py)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](src/api/app.py)
[![Dockerfile](https://img.shields.io/badge/Deployment-Dockerfile-2496ED?logo=docker&logoColor=white)](Dockerfile)

A local end-to-end prototype: a fine-tuned DistilBERT token classifier extracts age, sex, symptoms, diagnosis, and medications; deterministic rules assess the supplied guideline JSON; FastAPI returns structured output, evidence, and gradient-based token attribution.

The project is organized into models/, src/, and outputs/. Rules come exclusively from src/data/guidelines.json. This is a research prototype, not a validated clinical decision-support system.

## Quick start

Use Python 3.11. The submission ZIP includes weights; inference needs no training or Hugging Face access.

    python -m venv /tmp/clinical-ai-venv
    source /tmp/clinical-ai-venv/bin/activate
    python -m pip install -r requirements.txt
    ./run.sh

Running ./run.sh without arguments launches a terminal demo using the saved model. It processes three supplied notes: a recommended medication with a missing diagnostic investigation, a forbidden medication, and recommended medications with a documented ECG. For each note it prints extracted fields, guideline decisions, reasoning, the most influential diagnosis tokens, review warnings, and processing time. Full results are saved to outputs/demo_results.json.

The explicit command ./run.sh demo runs the same demo. It uses the existing inference, guideline, and explanation modules without retraining or starting a server. To start the API separately:

    ./run.sh api

For a smaller CPU installation, install PyTorch first:

    python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
    python -m pip install -r requirements.txt

Select another environment with PYTHON_BIN=/path/to/python. The runner defaults to four CPU threads; the service defaults to CPU. MODEL_DEVICE=cuda enables an available CUDA device. Model loading is strictly local.

    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/analyze_note \
      -H 'Content-Type: application/json' \
      -d '{"text":"45 year old male with fever and productive cough for five days. Diagnosed with pneumonia. Started on amoxicillin."}'

Interactive documentation: http://localhost:8000/docs.

Input accepts the clinical note, an optional note identifier, and optional documented diagnostic investigations (for example, a chest X-ray). Caller-reported investigations are identified separately in the explanation. Extra keys, blank text, non-string input, and excessive payloads are rejected. Notes over 512 WordPiece tokens return HTTP 422; text is never silently truncated. The character limit is 12,000.

Responses include structured_entities, guideline_assessment, explanation, extraction warnings, model/guideline versions, a generated request ID, and latency. Age is an integer or null; absent scalar fields are null. Evidence offsets refer to original input text.

## Commands and deliverables

| Command | Purpose |
| --- | --- |
| ./run.sh or ./run.sh demo | Run the three-note terminal demo and save outputs/demo_results.json |
| ./run.sh annotate | Validate notes, generate weak annotations and normalized analysis text, save outputs/data_summary.json |
| ./run.sh tokenize | Check WordPiece/BIO alignment for all annotations |
| ./run.sh train | Train and save the best validation checkpoint under models/entity_extractor/ |
| ./run.sh predict | Save a configurable example prediction as a list |
| ./run.sh guideline | Assess the currently saved predictions |
| ./run.sh explain | Save outputs/explanation_example.json |
| ./run.sh evaluate | Regenerate all 50 predictions, split-specific metrics, and ten observed error cases |
| ./run.sh all | Regenerate annotations, evaluation, and explanation; run regression checks with existing weights |
| ./run.sh api | Serve POST /analyze_note and GET /health |
| ./run.sh package | Build and verify outputs/submission.zip, including weights |

The all command does not retrain. After changing annotations, run annotate, train, then all. Training accepts Hydra overrides:

    ./run.sh train trainer.num_epochs=15 trainer.seed=42

Hydra outputs stay under outputs/. Canonical training code is in src/training/. Pre-existing model-side Python scripts are legacy: they remain in the workspace, are not used, and are excluded from the ZIP.

## Data and preprocessing

The supplied dataset has 50 notes, 19 guidelines, ages 5–72, 26 male and 24 female notes. Ten guideline conditions occur in notes; nine have no examples. Most notes share a demographic/symptom/diagnosis/medication template. Diagnostic investigations, doses, duration, negation, history, and multiple diagnoses are poorly represented or absent.

No labeled entities were supplied. The annotator creates **weak labels** from demographic patterns, symptom clauses, and guideline diagnosis/medication vocabularies. They are not independent expert ground truth. Symptom durations are excluded from symptom spans while original text is retained. Note IDs and ordered, non-overlapping offsets are validated.

Normalized analysis text is lowercase with normalized whitespace. Training and inference retain raw text and rely on the uncased tokenizer's normalization, preserving offsets. Stopwords, punctuation, negation, and qualifiers are not removed from model input.

## Model and reproducibility

DistilBERT offers contextual extraction with manageable local CPU cost. It is fine-tuned for 11 BIO/O labels across five entity types using cross-entropy. Every nonspecial WordPiece is aligned to character annotations; special/padding labels use -100.

Settings: seed 42, 15 epochs, learning rate 2e-5, AdamW weight decay 0.01, batch size 8, training length 128. The longest supplied note has 33 tokens. Oversized training notes fail explicitly. Validation F1 selects the checkpoint; held-out evaluation notes are not used in training or selection.

Seeded group splitting keeps exact text duplicates after removal of age/sex together. The saved run has **28 training, 11 validation, and 11 held-out evaluation notes**. It is not stratified: validation covers six conditions, held-out evaluation covers eight. Neither supports claims across all ten conditions. Near-duplicate templates can still cross partitions, and results on this tiny dataset are optimistic relative to real notes.

The model directory contains weights, tokenizer and label configuration, split_manifest.json with IDs and coverage, training_config.json, validation metrics, and metadata.json. Metadata records model SHA-256/version, annotation SHA-256, base revision, and runtime provenance. Keep these files together.

The base model is distilbert-base-uncased, pinned to revision 12040accade4e8a0f71eabdb258fecc2e7e948be. First-time training downloads it; cached training supports HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1. Two independent CPU training runs produced identical model SHA-256 hashes (outputs/reproducibility.json). Fixed packages, a fixed base revision, seeded splits, and deterministic PyTorch settings support reproduction; hardware/kernel differences can still affect floating-point output.

Postprocessing reconstructs BIO spans, deduplicates lists, normalizes casing, and converts ages. A transparent sentence-level assertion heuristic excludes obvious negated/historical entities and flags uncertain diagnoses. It is separate from the learned model and is not a general assertion classifier. Multiple diagnoses retain full evidence, use the first for the single-diagnosis schema, and trigger a warning.

## Guideline reasoning

The decision module has no neural-model dependency. It performs exact normalized matching with recommended_drugs, avoid_drugs, and required_tests. Forbidden drugs take precedence. Unlisted drugs trigger review and are not presumed forbidden or safe. Missing diagnoses or unavailable guidelines yield unknown (null) compliance.

The compliant field indicates **medication-list compliance**: at least one recommended drug, no forbidden drug, no unlisted drug. Missing diagnostic investigations can coexist with compliant=true. The additional fully_compliant field also requires all diagnostic investigations to be documented. Flat guideline lists do not distinguish alternatives from mandatory combinations: the engine cannot certify regimen completeness, doses, duration, or patient-specific suitability.

Documented diagnostic investigations come from explicit caller input and affirmative note mentions. Aliases cover ECG/EKG, CXR/chest X-ray, and configured investigation names. Sentence exclusions reject negative, planned, ordered, or pending mentions. Exact evidence spans are returned. Mixed assertions and complex sentences can be missed. “Not documented” does not mean “not performed.”

Every decision includes medication statuses, missing investigations, warnings, explanation, and a guideline content version. Uncertain diagnoses receive conditional-assessment warnings. The supplied fixture is not asserted to be current national medical guidance.

## Explainability

The explainer computes **gradient × input embeddings** for mean log-probability of the selected diagnosis's BIO tags. Absolute token scores sum to one; signed scores and raw offsets are also returned. This uses the actual trained classifier. If no diagnosis is detected, attribution is empty and other evidence remains available.

In the saved example, “pneumonia” has the largest attribution (22.2%), followed by “with” (16.1%) and a period (15.2%). This shows both entity-word sensitivity and reliance on template context. Attribution measures local sensitivity, not causation or clinical correctness. Softmax entity scores are uncalibrated. Medication acceptance/flags are explained by exact guideline comparisons.

## Evaluation and errors

Evaluation joins by note ID and rejects duplicates, missing IDs, mismatched text, and invalid spans. Strict character-span precision, recall, F1, per-label results, and exact-note accuracy are split-specific. Boundary errors count as both false positive and false negative. Raw learned spans are scored; assertion filtering is not hidden inside span metrics.

| Split | Notes | Precision | Recall | Entity F1 |
| --- | ---: | ---: | ---: | ---: |
| Training | 28 | 0.9249 | 0.9524 | 0.9384 |
| Validation | 11 | 0.8750 | 0.9180 | 0.8960 |
| Held-out evaluation | 11 | 0.9000 | 0.9403 | 0.9197 |

Structured-field exact accuracy on the held-out notes is 100% for age and sex, 72.73% for symptoms and diagnosis, and 90.91% for medications. Only 63.64% of held-out notes have entirely correct span sets. This makes the practical limitations clearer than aggregate F1 alone.

These measure agreement with **weak labels, not clinically validated accuracy**. Detailed counts appear in outputs/evaluation_metrics.json. Ten actual errors in outputs/error_analysis.md include original notes, split, missing/spurious spans, impact, and proposed improvements: four held-out cases, five validation cases, and one training case.

Examples: a spurious “left” diagnosis in a chest-pain description; “nebulization” merged into salbutamol; “Asthma” omitted from the diagnosis; pneumonia mislabeled as a symptom. These affect guideline matching despite high aggregate F1. Do not repeatedly tune on the frozen evaluation set. Next steps: clinician-reviewed labels, diverse note styles, coverage of absent conditions, robust assertions, and a new independent external evaluation set.

## API, deployment, and MLOps

FastAPI verifies and loads weights once at startup. A lock serializes inference and attribution to limit concurrent gradient work and memory use. Readiness is available after model initialization. This single-process prototype has throughput bounded by CPU inference plus attribution's backward pass.

Saved API responses and audit events are in outputs/api_examples.json and outputs/prediction_log_examples.json. The regression suite verifies these against the real model.

JSON stdout logs contain generated request IDs, model/guideline versions, predicted entity counts, decision status, warning count, and end-to-end latency. They exclude note text, caller note IDs, entity values, and attribution text. Evidence is returned to the caller; logs hold prediction summaries.

Weight SHA-256 identifies deployments; mismatches fail loading. Guideline changes alter their content version. Production follow-up should archive artifacts/config together, compare frozen evaluations before rollout, canary changes, retain rollback versions, and monitor latency, errors, distributions, and reviewed accuracy. Versioning and per-request logs are implemented; a live drift detector and registry are not.

    docker build -t clinical-note-ai .
    docker run --rm -p 8000:8000 clinical-note-ai

The Dockerfile installs CPU dependencies, bundles weights, runs as a non-root user, and includes a health check. Build context excludes historical checkpoints, caches, and output archives. Inference downloads no model. The final local suite passed 37 automated checks, including real-model API and saved evidence checks. Container build/run verification was blocked by Docker daemon permissions in this environment; local model and API checks are reported separately.

## Packaging and development credits

The package command explicitly includes model.safetensors despite the original repository's ignore rule. Submit the ZIP to avoid omitting weights in a plain Git push. It preserves the required root directories/files (plus .dockerignore), excludes old checkpoints and legacy scripts, and verifies archive integrity.

I developed the original data preprocessing and model-training implementation. The remaining work—entity inference, guideline evaluation, explainability, API integration, evaluation and error analysis, deployment, logging, the demo, and documentation—was developed collaboratively with OpenAI Codex. Codex also assisted with integration fixes and reproducibility checks for the existing pipeline. Weak labels and analysis are automated, not clinician-authored.
