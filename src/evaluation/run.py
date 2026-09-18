"""Generate fresh split-specific predictions, metrics, and five-case error review."""
import json
import hashlib
from pathlib import Path
from time import perf_counter
from statistics import median

from src.evaluation.evaluate import EntityEvaluator
from src.extraction.predictor import EntityPredictor
from src.guideline_engine.evaluator import evaluate_guideline, extract_observed_tests

ROOT = Path(__file__).resolve().parents[2]


def main():
    output = ROOT / "outputs"
    predictor = EntityPredictor()
    predictor.load()
    notes = json.loads((ROOT / "src/data/annotations.json").read_text())
    metadata = json.loads((predictor.model_dir / "metadata.json").read_text())
    if hashlib.sha256((ROOT / "src/data/annotations.json").read_bytes()).hexdigest() != metadata["annotation_sha256"]:
        raise ValueError("Annotations changed since training; retrain before evaluating")
    manifest = json.loads((predictor.model_dir / "split_manifest.json").read_text())
    splits = [set(manifest[s]) for s in ["train", "validation", "test"]]
    if any(splits[i] & splits[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Split leakage: note IDs overlap")
    if set.union(*splits) != {n["note_id"] for n in notes}:
        raise ValueError("Split manifest does not match current annotations")
    predictions, timings = [], []
    for note in notes:
        start = perf_counter()
        p = predictor.predict(note)
        timings.append((perf_counter() - start) * 1000)
        tests = extract_observed_tests(note["text"])
        p["observed_tests"] = [e["test"] for e in tests]
        p["guideline_assessment"] = evaluate_guideline(p["structured"]["diagnosis"], p["structured"]["medications"], p["observed_tests"])
        predictions.append(p)
    (output / "sample_predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")
    report = {"model_version": predictor.model_version,
              "reference_source": "Automatic weak labels, not independent clinician annotations",
              "limitations": "Grouped held-out notes remain templated; metrics measure agreement with weak labels, not clinical validity. Validation selects checkpoint; test is not used in fitting.",
              "latency_ms": {"median": median(timings), "max": max(timings), "samples": len(timings), "includes_explanations": False},
              "splits": {}}
    for split in ["train", "validation", "test"]:
        ids = set(manifest[split])
        report["splits"][split] = EntityEvaluator.score_notes([n for n in notes if n["note_id"] in ids],
                                                              [p for p in predictions if p["note_id"] in ids])
        references = {n["note_id"]: EntityPredictor.to_structured_output(n["entities"]) for n in notes if n["note_id"] in ids}
        guesses = [p for p in predictions if p["note_id"] in ids]
        def comparable(value):
            return sorted(value) if isinstance(value, list) else value
        report["splits"][split]["structured_field_exact_accuracy"] = {
            field: sum(comparable(p["structured"][field]) == comparable(references[p["note_id"]][field]) for p in guesses) / len(guesses)
            for field in ["age", "sex", "symptoms", "diagnosis", "medications"]}
    (output / "evaluation_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    cases = []
    for split in ["test", "validation", "train"]:
        cases.extend([{**e, "split": split} for e in report["splits"][split]["errors"]])
    selected = cases[:max(5, min(10, len(cases)))]
    lines = ["# Extraction error analysis", "", "These are observed errors against weak labels. No clinician review is claimed.",
             "", f"Model: {predictor.model_version}", ""]
    for case in selected:
        lines += [f"## {case['note_id']} ({case['split']})", "", case["text"], "",
                  "Missing: " + json.dumps(case["missing"]), "", "Spurious: " + json.dumps(case["spurious"]), ""]
        overlaps = any(a["label"] == b["label"] and a["start"] < b["end"] and b["start"] < a["end"]
                       for a in case["missing"] for b in case["spurious"])
        lines += ["Interpretation: " + ("Overlapping spans indicate a boundary or BIO fragmentation error; strict span scoring counts both a false positive and a false negative."
                                       if overlaps else "The model omitted or mislabeled a field. Sparse training examples and unseen context can explain this, but the cause is not proven."), "",
                  "Impact: missing/fragmented diagnosis can prevent guideline matching; medication errors can change the decision. Review original evidence.",
                  "", "Improvement: obtain independent span labels and more varied training contexts, then validate on a new frozen test set.", ""]
    lines += ["## Common failure modes", "", "Subword fragmentation; symptom-boundary ambiguity; unseen diagnosis/medication names; negation and historical mentions; multiple diagnoses. The rule-based assertion filter is separate from learned span detection and does not establish general negation accuracy.",
              "", "Tests inferred from mentions use conservative sentence exclusions. 'Not documented' is not proof that a test was not performed.",
              "", "Grouped splitting removes exact demographic-only duplicates, but near-duplicate templates and weak-label bias remain."]
    (output / "error_analysis.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({s: report["splits"][s]["overall"] for s in report["splits"]}, indent=2))
    print(f"Saved {len(selected)} observed error cases.")


if __name__ == "__main__":
    main()
