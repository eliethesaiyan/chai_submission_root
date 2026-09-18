"""Local learned BIO extraction with original-text offsets and explicit limits."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]


class EntityPredictor:
    def __init__(self, model_dir=ROOT / "models/entity_extractor",
                 output_path=ROOT / "outputs/sample_predictions.json", clinical_note="",
                 max_length=512, device="auto"):
        self.model_dir = Path(model_dir)
        self.output_path = Path(output_path)
        self.clinical_note = clinical_note
        self.max_length = max_length
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                   else "cpu" if device == "auto" else device)
        self.tokenizer: Any = None
        self.model: Any = None
        self.id_to_label = {}
        self.model_version = "unloaded"

    def load(self):
        required = ("config.json", "model.safetensors", "tokenizer.json")
        missing = [name for name in required if not (self.model_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing model artifacts in {self.model_dir}: {', '.join(missing)}. Run ./run.sh train.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_dir, local_files_only=True)
        self.model.to(self.device).eval()
        self.id_to_label = {int(i): label for i, label in self.model.config.id2label.items()}
        self.max_length = min(self.max_length, self.model.config.max_position_embeddings)
        digest = hashlib.sha256((self.model_dir / "model.safetensors").read_bytes()).hexdigest()
        metadata_path = self.model_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        if metadata.get("sha256", digest) != digest:
            raise ValueError("Model weights do not match metadata checksum")
        self.model_version = metadata.get("model_version", "distilbert-" + digest[:12])

    def encode(self, text):
        if self.model is None:
            self.load()
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonempty string")
        encoding = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=False)
        if encoding["input_ids"].shape[1] > self.max_length:
            raise ValueError(f"Note exceeds the {self.max_length}-token limit; split it into shorter notes. No text was silently truncated.")
        offsets = encoding.pop("offset_mapping")[0].tolist()
        return {name: value.to(self.device) for name, value in encoding.items()}, offsets

    def predict_bio(self, text):
        encoding, offsets = self.encode(text)
        with torch.no_grad():
            probabilities = self.model(**encoding).logits.softmax(dim=-1)[0]
            scores, predicted_ids = probabilities.max(dim=-1)
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0].tolist())
        return [{"token": token, "label": self.id_to_label[int(label)], "score": float(score),
                 "start": start, "end": end, "text": text[start:end]}
                for token, label, score, (start, end) in zip(tokens, predicted_ids, scores, offsets) if start != end]

    @staticmethod
    def reconstruct_entities(text, predictions):
        entities, current = [], None
        def close():
            nonlocal current
            if current is not None:
                current["text"] = text[current["start"]:current["end"]]
                current["score"] = sum(current.pop("scores")) / current.pop("count")
                entities.append(current)
                current = None
        for prediction in predictions:
            if prediction["label"] == "O":
                close()
                continue
            prefix, label = prediction["label"].split("-", 1)
            # Subword BIO fragments of a single word are combined.
            contiguous_subword = (current is not None and prediction.get("token", "").startswith("##")
                                  and current["end"] == prediction["start"] and current["label"] == label)
            if current is None or current["label"] != label or (prefix == "B" and not contiguous_subword):
                close()
                current = {"label": label, "start": prediction["start"], "end": prediction["end"],
                           "scores": [], "count": 0}
            current["end"] = prediction["end"]
            current["scores"].append(prediction.get("score", 1.0))
            current["count"] += 1
        close()
        return entities

    @staticmethod
    def assertion(text, entity):
        """Small transparent postprocessor; learned extraction remains unchanged."""
        before = re.split(r"[.!?;\n]", text[:entity["start"]])[-1].lower()
        after = re.split(r"[.!?;\n]", text[entity["end"]:])[0].lower()
        if re.search(r"\b(no|denies|denied|without|negative for|ruled out|not)\b", before) or re.match(r"\s+(?:was |is )?(?:ruled out|not given|withheld)\b", after):
            return "negated"
        if re.search(r"\b(history of|previous|previously|past|stopped|discontinued|avoid|allerg\w*)\b", before):
            return "historical_or_excluded"
        if re.search(r"\b(suspected|possible|probable|suggest\w*|impression|rule out)\b", before):
            return "uncertain"
        return "affirmed"

    @staticmethod
    def to_structured_output(entities):
        output = {"age": None, "sex": None, "symptoms": [], "diagnosis": None, "medications": []}
        for entity in entities:
            if entity.get("assertion") in {"negated", "historical_or_excluded"}:
                continue
            label, value = entity["label"], entity["text"].strip().lower()
            if label in {"symptoms", "medications"}:
                if value not in output[label]:
                    output[label].append(value)
            elif label == "age":
                output[label] = int(value) if value.isdigit() and 0 <= int(value) <= 120 else None
            elif label == "sex":
                output[label] = value if value in {"male", "female"} else None
            elif label == "diagnosis" and output[label] is None:
                output[label] = value
        return output

    def predict(self, note):
        text = note.get("text", "")
        tokens = self.predict_bio(text)
        entities = self.reconstruct_entities(text, tokens)
        for e in entities:
            e["assertion"] = self.assertion(text, e) if e["label"] in {"diagnosis", "medications", "symptoms"} else "affirmed"
        diagnoses = [e for e in entities if e["label"] == "diagnosis" and e["assertion"] not in {"negated", "historical_or_excluded"}]
        warnings = []
        if len(diagnoses) > 1:
            warnings.append("Multiple diagnoses detected; the single-diagnosis schema uses the first. Review required.")
        if any(e["assertion"] == "uncertain" for e in diagnoses):
            warnings.append("Diagnosis is uncertain in the note; the guideline assessment is conditional.")
        if any(e["score"] < 0.8 for e in entities):
            warnings.append("At least one entity has a model score below 0.8; scores are not calibrated probabilities of correctness.")
        return {"note_id": note.get("note_id", note.get("id", "")), "text": text, "entities": entities,
                "structured": self.to_structured_output(entities), "tokens": tokens,
                "warnings": warnings, "model_version": self.model_version}

    def run(self):
        notes = ([{"note_id": "example", "text": self.clinical_note}] if self.clinical_note
                 else json.loads((ROOT / "src/data/clinical_notes.json").read_text()))
        predictions = [self.predict(n) for n in notes]
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(predictions, indent=2) + "\n")
        return predictions
