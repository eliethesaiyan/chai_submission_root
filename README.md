# chai_submission_root
# Clinical-AI
### About
This projects consists of processing clinical notes from text into structured entities as data annoations.
The following steps were followed:
#### 1. Preprocessing.
1. Using regex to extract age and sex entities.
2. Using regex to infers diagnosis using keywords like `with` and `diagnosed`.
3. Drop the concepts of the duration of the symptoms.
4. Building a vocabulary f the diagonses terms from clincal notes.
5. Building a  medications from guidelines.

This step can be reproduced as follows:

```console 
$ ./run.sh annotate
```
It creates model data annotation provides a data analsysis summary as follow:
```console
[2026-09-08 00:10:15,464][__main__][INFO] - Dataset Analysis Summary
[2026-09-08 00:10:15,464][__main__][INFO] - ==================================================
[2026-09-08 00:10:15,464][__main__][INFO] - Total Clinical Notes:50
[2026-09-08 00:10:15,464][__main__][INFO] - Total Guidelines: 19
[2026-09-08 00:10:15,464][__main__][INFO] - Demographics Summary:
[2026-09-08 00:10:15,464][__main__][INFO] -   Max Age: 72
[2026-09-08 00:10:15,464][__main__][INFO] -   Minimum Age: 5
[2026-09-08 00:10:15,464][__main__][INFO] -   Male Count: 26
[2026-09-08 00:10:15,464][__main__][INFO] -   Female Count: 24
[2026-09-08 00:10:15,464][__main__][INFO] - Diagnosis in clinical Counts Summary: 3 diags examples
[2026-09-08 00:10:15,464][__main__][INFO] -   pneumonia: 7 recorded in clinical notes.
[2026-09-08 00:10:15,464][__main__][INFO] -   meningitis: 5 recorded in clinical notes.
[2026-09-08 00:10:15,464][__main__][INFO] -   myocardial infarction: 5 recorded in clinical notes.
[2026-09-08 00:10:15,464][__main__][INFO] - Medications in clinical note Counts Summary: 3 meds examples
[2026-09-08 00:10:15,464][__main__][INFO] -   nitrofurantoin: 2 recorded in clinical notes .
[2026-09-08 00:10:15,464][__main__][INFO] -   ciprofloxacin: 3 recorded in clinical notes .
[2026-09-08 00:10:15,464][__main__][INFO] -   azithromycin: 7 recorded in clinical notes .
```

#### 2. Training.

1. Tokenizing each clinical note with DistilBERT's fast tokenizer.
2. Aligning character-level entity annotations with BIO token labels.
3. Splitting the annotated notes into training and validation sets.
4. Fine-tuning DistilBERT for token classification.
5. Evaluating each epoch using precision, recall, and F1 score.
6. Saving the best model, tokenizer, training configuration, and validation metrics in `models/checkpoints/`.

This step can be reproduced as follows:

```console
$ ./run.sh train
```

It trains the entity extraction model and reports validation metrics as follows:

```console
[__main__][INFO] - Validation metrics:
{
  "eval_loss": 0.21514029800891876,
  "eval_precision": 0.9661016949152542,
  "eval_recall": 0.9827586206896551,
  "eval_f1": 0.9743589743589743,
  "epoch": 15.0
}
[__main__][INFO] - Saved model artifacts to models/checkpoints
```

The main training outputs are:

```text
models/checkpoints/
├── config.json
├── evaluation_metrics.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── training_args.bin
├── training_config.json
└── vocab.txt
```

#### 3. Evaluation.

1. Loading the extracted diagnosis and medications from `outputs/sample_predictions.json`.
2. Loading the corresponding guideline from `src/data/guidelines.json`.
3. Comparing each extracted medication with `recommended_drugs` and `avoid_drugs`.
4. Classifying each medication as `recommended`, `avoid`, or `not_listed`.
5. Checking observed tests against the guideline's `required_tests`.
6. Saving the deterministic guideline decision to `outputs/guideline_evaluation.json`.

This step can be reproduced as follows:

```console
$ ./run.sh guideline
```

It creates a guideline evaluation report as follows:

```json
{
  "status": "evaluated",
  "diagnosis": "pneumonia",
  "medications": [
    {
      "medication": "amoxicillin",
      "status": "recommended"
    }
  ],
  "recommended_medications_given": [
    "amoxicillin"
  ],
  "forbidden_medications": [],
  "recommended_drugs": [
    "amoxicillin",
    "azithromycin"
  ],
  "required_tests": [
    {
      "test": "chest_xray",
      "present": false
    }
  ],
  "missing_tests": [
    "chest_xray"
  ]
}
```

The decision contains only recommendations and required tests defined in `src/data/guidelines.json`.
