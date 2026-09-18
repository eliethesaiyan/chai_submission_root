"""Validate source data, generate weak labels, and report dataset coverage."""
import json
from collections import Counter
from pathlib import Path

from src.preprocessing.annotator import Annotator
from src.preprocessing.utils import normalize_text

ROOT = Path(__file__).resolve().parents[2]


def main():
    data = ROOT / "src/data"
    annotator = Annotator(data, data / "clinical_notes.json", data / "guidelines.json", data / "annotations.json")
    notes = annotator.clinical_notes
    ids = [note["note_id"] for note in notes]
    if len(ids) != len(set(ids)) or any(not n["text"].strip() for n in notes):
        raise ValueError("Notes must have unique IDs and nonempty text")
    diagnoses = annotator.build_diagnostic_vocabulary()
    medications = annotator.build_medication_vocabulary()
    annotations = [annotator.annotate_entities_in_text(n, diagnoses, medications) for n in notes]
    for note in annotations:
        end = 0
        for entity in note["entities"]:
            assert end <= entity["start"] < entity["end"] <= len(note["text"])
            assert entity["text"] == note["text"][entity["start"]:entity["end"]]
            end = entity["end"]
    (data / "annotations.json").write_text(json.dumps(annotations, indent=2) + "\n")
    normalized = [{**n, "normalized_text": normalize_text(n["text"])} for n in notes]
    (data / "normalized_notes.json").write_text(json.dumps(normalized, indent=2) + "\n")
    counts = {label: Counter(e["text"].lower() for n in annotations for e in n["entities"] if e["label"] == label)
              for label in ["sex", "diagnosis", "medications", "symptoms"]}
    ages = [int(e["text"]) for n in annotations for e in n["entities"] if e["label"] == "age"]
    report = {"notes": len(notes), "guidelines": len(annotator.guidelines), "age_min": min(ages),
              "age_max": max(ages), "counts": counts,
              "guidelines_without_training_examples": sorted(set(diagnoses) - set(counts["diagnosis"])),
              "annotation_source": "Deterministic weak supervision; not clinician-validated ground truth",
              "normalization": "Lowercase/whitespace normalization for analysis; raw text retained for character offsets. DistilBERT uncased WordPiece normalizes model input.",
              "challenges": ["Only 50 highly templated notes", "No expert entity labels supplied",
                             "Nine guidelines have no corresponding notes", "Rare or absent negation, tests, doses, and multiple diagnoses",
                             "Suspected diagnoses must not be interpreted as confirmed disease"]}
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs/data_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
