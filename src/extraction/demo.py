"""A terminal demo of learned extraction, guideline checks, and explanations."""
import json
from pathlib import Path
from time import perf_counter

from src.extraction.predictor import EntityPredictor
from src.explainability.explainer import explain_prediction
from src.guideline_engine.evaluator import evaluate_guideline, extract_observed_tests

ROOT = Path(__file__).resolve().parents[2]
CASES = [
    ("N001", "Recommended medication; required test not documented"),
    ("N004", "Forbidden medication detected"),
    ("N021", "Recommended medications with a documented ECG"),
]


def main():
    print("Clinical note AI demo", flush=True)
    print("Loading the saved model for three example notes...", flush=True)
    predictor = EntityPredictor()
    predictor.load()
    notes = {n["note_id"]: n for n in json.loads((ROOT / "src/data/clinical_notes.json").read_text())}
    results = []
    for index, (note_id, title) in enumerate(CASES, start=1):
        note = notes[note_id]
        start = perf_counter()
        prediction = predictor.predict(note)
        observed = [e["test"] for e in extract_observed_tests(note["text"])]
        fields = prediction["structured"]
        assessment = evaluate_guideline(fields["diagnosis"], fields["medications"], observed)
        explanation = explain_prediction(predictor, prediction, assessment)
        elapsed = round((perf_counter() - start) * 1000, 2)
        results.append({"note_id": note_id, "text": note["text"], "structured_entities": fields,
                        "guideline_assessment": assessment, "explanation": explanation,
                        "warnings": prediction["warnings"], "model_version": predictor.model_version,
                        "latency_ms": elapsed})

        print(f"\n{index}. {title} ({note_id})")
        print(f"Note: {note['text']}")
        print("Extracted fields:")
        print(json.dumps(fields, indent=2))
        print(f"Medication compliant: {assessment['compliant']}; fully compliant: {assessment['fully_compliant']}")
        for medication in assessment["medications"]:
            print(f"  {medication['medication']}: {medication['status']}")
        print("Tests not documented: " + (", ".join(assessment["missing_tests"]) or "none"))
        print("Reason: " + assessment["explanation"])
        top = sorted(explanation["token_importance"], key=lambda t: -t["importance"])[:3]
        print("Top diagnosis attribution: " + (", ".join(f"{t['text']!r} ({t['importance']:.1%})" for t in top) or "no diagnosis detected"))
        for warning in prediction["warnings"]:
            print("Review: " + warning)
        print(f"Processing time: {elapsed:.2f} ms", flush=True)

    output = ROOT / "outputs/demo_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print("\nAttribution shows model sensitivity, not clinical correctness. This is an assessment prototype.")
    print(f"Full results saved to {output}")


if __name__ == "__main__":
    main()
