"""Configurable inference for the trained clinical entity extractor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


class EntityPredictor:
    """Load a local token-classification model and extract clinical entities."""

    def __init__(
        self,
        model_dir: str | Path,
        output_path: str | Path,
        clinical_note: str,
        max_length: int = 128,
        device: str = "auto",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.output_path = Path(output_path)
        self.clinical_note = clinical_note
        self.max_length = max_length
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "cpu" if device == "auto" else device
        )
        self.tokenizer: Any = None
        self.model: Any = None
        self.id_to_label: dict[int, str] = {}

    def load(self) -> None:
        """Load model artifacts strictly from ``model_dir``."""
        required = ("config.json", "model.safetensors", "tokenizer.json")
        missing = [name for name in required if not (self.model_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing model artifacts in {self.model_dir}: {', '.join(missing)}"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
            use_fast=True,
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.id_to_label = {
            int(index): label for index, label in self.model.config.id2label.items()
        }

    def predict_bio(self, text: str) -> list[dict[str, Any]]:
        if self.model is None or self.tokenizer is None:
            self.load()
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
        )
        offsets = encoding.pop("offset_mapping")[0]
        encoding = {name: value.to(self.device) for name, value in encoding.items()}

        with torch.no_grad():
            predicted_ids = torch.argmax(self.model(**encoding).logits, dim=-1)[0]

        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
        predictions = []
        for token, predicted_id, offset in zip(tokens, predicted_ids, offsets):
            start, end = int(offset[0]), int(offset[1])
            if start == end:
                continue
            predictions.append(
                {
                    "token": token,
                    "label": self.id_to_label[int(predicted_id)],
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                }
            )
        return predictions

    @staticmethod
    def reconstruct_entities(text: str, predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        entities: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None

        def close_entity() -> None:
            nonlocal current
            if current is not None:
                current["text"] = text[current["start"]:current["end"]]
                entities.append(current)
                current = None

        for prediction in predictions:
            label = prediction["label"]
            if label == "O":
                close_entity()
                continue
            prefix, entity_type = label.split("-", maxsplit=1)
            if prefix == "B" or current is None or current["label"] != entity_type:
                close_entity()
                current = {
                    "label": entity_type,
                    "start": prediction["start"],
                    "end": prediction["end"],
                }
            else:
                current["end"] = prediction["end"]
        close_entity()
        return entities

    @staticmethod
    def to_structured_output(entities: list[dict[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {
            "age": None,
            "sex": None,
            "symptoms": [],
            "diagnosis": None,
            "medications": [],
        }
        for entity in entities:
            label, value = entity["label"], entity["text"]
            if label in {"symptoms", "medications"}:
                output[label].append(value)
            elif label in output:
                output[label] = value
        return output

    def predict(self, note: dict[str, Any]) -> dict[str, Any]:
        text = str(note.get("text", ""))
        entities = self.reconstruct_entities(text, self.predict_bio(text))
        return {
            "note_id": note.get("id", note.get("note_id", "")),
            "text": text,
            "entities": entities,
            "structured": self.to_structured_output(entities),
        }

    def run(self) -> dict[str, Any]:
        """Run inference for the configured note and save the structured result."""
        prediction = self.predict({"text": self.clinical_note})
        structured = prediction["structured"]
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as file:
            json.dump(structured, file, indent=2, ensure_ascii=False)
        return structured
