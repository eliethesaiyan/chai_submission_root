"""Strict character-span extraction scoring joined by stable note IDs."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path


class EntityEvaluator:
    def __init__(self, annotations_path, predictions_path, output_path):
        self.annotations_path = Path(annotations_path)
        self.predictions_path = Path(predictions_path)
        self.output_path = Path(output_path)

    @staticmethod
    def _load_json(path):
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return data

    @staticmethod
    def _indexed(notes):
        result = {}
        for note in notes:
            key = note.get("note_id", note.get("id"))
            if not key or key in result:
                raise ValueError(f"Missing or duplicate note_id: {key!r}")
            result[key] = note
        return result

    @staticmethod
    def _entity_set(note):
        result = set()
        for e in note.get("entities", []):
            try:
                label, start, end = str(e["label"]), int(e["start"]), int(e["end"])
                if start < 0 or end <= start or end > len(note["text"]):
                    raise ValueError("Invalid offsets")
                if e.get("text", note["text"][start:end]) != note["text"][start:end]:
                    raise ValueError("Span text mismatch")
                result.add((label, start, end))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid entity in {note.get('note_id')}: {e}") from error
        return result

    @staticmethod
    def _scores(tp, fp, fn):
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return {"precision": precision, "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "true_positives": tp, "false_positives": fp, "false_negatives": fn}

    @classmethod
    def score_notes(cls, annotations, predictions):
        references, predicted = cls._indexed(annotations), cls._indexed(predictions)
        if set(references) != set(predicted):
            raise ValueError("Reference/prediction note IDs differ; incomplete or unrelated evaluation is not allowed")
        counts = defaultdict(lambda: [0, 0, 0])
        exact = 0
        errors = []
        for key, note in references.items():
            if predicted[key]["text"] != note["text"]:
                raise ValueError(f"Text differs for note {key}")
            expected, actual = cls._entity_set(note), cls._entity_set(predicted[key])
            exact += expected == actual
            for label in {e[0] for e in expected | actual}:
                gold, guess = {e for e in expected if e[0] == label}, {e for e in actual if e[0] == label}
                counts[label][0] += len(gold & guess)
                counts[label][1] += len(guess - gold)
                counts[label][2] += len(gold - guess)
            if expected != actual:
                def show(items):
                    return [{"label": label, "start": s, "end": e, "text": note["text"][s:e]} for label, s, e in sorted(items)]
                errors.append({"note_id": key, "text": note["text"], "missing": show(expected - actual),
                               "spurious": show(actual - expected)})
        totals = [sum(v[p] for v in counts.values()) for p in range(3)]
        return {"notes": len(references), "overall": cls._scores(*totals),
                "exact_note_accuracy": exact / len(references) if references else 0.0,
                "by_label": {k: cls._scores(*v) for k, v in sorted(counts.items())},
                "errors": errors}

    def evaluate(self):
        return self.score_notes(self._load_json(self.annotations_path), self._load_json(self.predictions_path))

    def save(self, metrics):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(metrics, indent=2) + "\n")
