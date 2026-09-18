"""Configurable token-classification training logic."""

from __future__ import annotations

import json
import hashlib
import re
import platform
import importlib.metadata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.training.tokenization import ClinicalEntityDataset


class EntityTrainer:
    """Train and save a transformer model for clinical entity extraction."""

    def __init__(
        self,
        annotations_path: str | Path,
        output_dir: str | Path,
        model_name: str,
        labels: list[str],
        seed: int = 42,
        validation_size: float = 0.2,
        max_length: int = 128,
        learning_rate: float = 2e-5,
        batch_size: int = 8,
        num_epochs: float = 10,
        weight_decay: float = 0.01,
        save_total_limit: int = 2,
        model_revision: str = "12040accade4e8a0f71eabdb258fecc2e7e948be",
    ) -> None:
        self.annotations_path = Path(annotations_path)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.model_revision = model_revision
        self.labels = list(labels)
        self.seed = seed
        self.validation_size = validation_size
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.save_total_limit = save_total_limit
        self.label_to_id = {label: index for index, label in enumerate(self.labels)}
        self.id_to_label = {index: label for label, index in self.label_to_id.items()}

    def _load_annotations(self) -> list[dict[str, Any]]:
        with self.annotations_path.open("r", encoding="utf-8") as file:
            annotations = json.load(file)
        if not isinstance(annotations, list) or len(annotations) < 2:
            raise ValueError("Training requires a JSON list containing at least two annotations")
        return annotations

    @staticmethod
    def _diagnosis(note: dict[str, Any]) -> str | None:
        for entity in note.get("entities", []):
            if entity.get("label") == "diagnosis":
                return str(entity.get("text", "")).lower()
        return None

    @staticmethod
    def template_group(note):
        text = re.sub(r"^\d+ year old (?:male|female)\s*", "", note["text"].lower())
        return re.sub(r"\s+", " ", text).strip()

    def _split(self, annotations):
        groups = [self.template_group(n) for n in annotations]
        pool_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=self.seed).split(annotations, groups=groups))
        pool = [annotations[i] for i in pool_idx]
        train_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=self.validation_size / 0.8,
                                                    random_state=self.seed).split(pool, groups=[self.template_group(n) for n in pool]))
        train = [pool[i] for i in train_idx]
        validation = [pool[i] for i in val_idx]
        test = [annotations[i] for i in test_idx]
        self.split_manifest = {"seed": self.seed, "method": "Group split by exact text after removing age/sex; test groups 20%, validation groups 20%",
                               "train": [n["note_id"] for n in train], "validation": [n["note_id"] for n in validation],
                               "test": [n["note_id"] for n in test]}
        for name, notes in [("train", train), ("validation", validation), ("test", test)]:
            self.split_manifest[name + "_diagnoses"] = dict(Counter(self._diagnosis(n) for n in notes))
        return train, validation

    def _compute_metrics(self, evaluation: Any) -> dict[str, float]:
        predictions, labels = evaluation
        predicted_ids = np.argmax(predictions, axis=2)
        true_predictions: list[list[str]] = []
        true_labels: list[list[str]] = []

        for prediction, label in zip(predicted_ids, labels):
            included = [(predicted, expected) for predicted, expected in zip(prediction, label) if expected != -100]
            true_predictions.append([self.id_to_label[int(predicted)] for predicted, _ in included])
            true_labels.append([self.id_to_label[int(expected)] for _, expected in included])

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

    def train(self) -> dict[str, float]:
        """Train, evaluate, and persist all artifacts beneath ``output_dir``."""
        set_seed(self.seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        annotations = self._load_annotations()
        train_notes, validation_notes = self._split(annotations)
        (self.output_dir / "split_manifest.json").write_text(json.dumps(self.split_manifest, indent=2) + "\n")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.model_revision, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            revision=self.model_revision,
            num_labels=len(self.labels),
            id2label=self.id_to_label,
            label2id=self.label_to_id,
        )
        train_dataset = ClinicalEntityDataset(
            train_notes, tokenizer, self.label_to_id, self.max_length
        )
        validation_dataset = ClinicalEntityDataset(
            validation_notes, tokenizer, self.label_to_id, self.max_length
        )
        arguments = TrainingArguments(
            output_dir=str(self.output_dir),
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=self.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=self.save_total_limit,
            seed=self.seed,
            report_to="none",
            full_determinism=True,
            save_safetensors=True,
            disable_tqdm=True,
        )
        trainer = Trainer(
            model=model,
            args=arguments,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
            processing_class=tokenizer,
            compute_metrics=self._compute_metrics,
        )
        trainer.train()
        metrics = trainer.evaluate()
        trainer.save_model(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))

        digest = hashlib.sha256((self.output_dir / "model.safetensors").read_bytes()).hexdigest()
        metadata = {"model_version": "distilbert-clinical-v1-" + digest[:12], "sha256": digest,
                    "base_model": self.model_name, "base_revision": getattr(model.config, "_commit_hash", None),
                    "annotation_sha256": hashlib.sha256(self.annotations_path.read_bytes()).hexdigest(),
                    "python": platform.python_version(), "label_source": "weak supervision; not clinician validated",
                    "packages": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "accelerate", "numpy", "scikit-learn", "seqeval"]},
                    "seed": self.seed, "max_length": self.max_length}
        (self.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        with (self.output_dir / "evaluation_metrics.json").open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)
        return metrics
