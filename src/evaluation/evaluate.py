"""Evaluate extracted clinical entities against reference annotations."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


class EntityEvaluator:
    """Compute strict, micro-averaged entity extraction metrics."""

    def __init__(
        self,
        annotations_path: str | Path,
        predictions_path: str | Path,
        output_path: str | Path,
    ) -> None:
        self.annotations_path = Path(annotations_path)
        self.predictions_path = Path(predictions_path)
        self.output_path = Path(output_path)

    @staticmethod
    def _load_json(path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return data

    @staticmethod
    def _entity_set(note: dict[str, Any]) -> set[tuple[str, int, int]]:
        entities = set()
        for entity in note.get("entities", []):
            try:
                entities.add(
                    (
                        str(entity["label"]),
                        int(entity["start"]),
                        int(entity["end"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid entity in note {note.get('note_id')!r}: {entity}") from error
        return entities

    @staticmethod
    def _scores(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float | int]:
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positive,
            "false_positives": false_positive,
            "false_negatives": false_negative,
        }

    def evaluate(self) -> dict[str, Any]:
        annotations = self._load_json(self.annotations_path)
        predictions = self._load_json(self.predictions_path)
        counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

        note_count = max(len(annotations), len(predictions))
        for index in range(note_count):
            expected = self._entity_set(annotations[index]) if index < len(annotations) else set()
            predicted = self._entity_set(predictions[index]) if index < len(predictions) else set()

            labels = {entity[0] for entity in expected | predicted}
            for label in labels:
                expected_for_label = {entity for entity in expected if entity[0] == label}
                predicted_for_label = {entity for entity in predicted if entity[0] == label}
                counts[label][0] += len(expected_for_label & predicted_for_label)
                counts[label][1] += len(predicted_for_label - expected_for_label)
                counts[label][2] += len(expected_for_label - predicted_for_label)

        totals = [sum(values[position] for values in counts.values()) for position in range(3)]
        return {
            "notes": {
                "annotations": len(annotations),
                "predictions": len(predictions),
            },
            "overall": self._scores(*totals),
            "by_label": {
                label: self._scores(*values)
                for label, values in sorted(counts.items())
            },
        }

    def save(self, metrics: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)
