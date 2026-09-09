"""Tokenization and BIO-label alignment for clinical entities."""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset


def find_entity_for_token(
    token_start: int,
    token_end: int,
    entities: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the entity overlapping a non-special token, if one exists."""
    for entity in entities:
        if token_start < int(entity["end"]) and token_end > int(entity["start"]):
            return entity
    return None


def tokenize_and_align_labels(
    note: dict[str, Any],
    tokenizer: Any,
    label_to_id: dict[str, int],
    max_length: int,
) -> dict[str, Any]:
    """Tokenize one note and align character-span entities to BIO labels."""
    encoding = tokenizer(
        note["text"],
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    labels: list[int] = []
    previous_entity: dict[str, Any] | None = None

    for token_start, token_end in encoding["offset_mapping"]:
        if token_start == token_end:
            labels.append(-100)
            previous_entity = None
            continue

        entity = find_entity_for_token(token_start, token_end, note.get("entities", []))
        if entity is None:
            labels.append(label_to_id["O"])
            previous_entity = None
            continue

        prefix = "I" if entity is previous_entity else "B"
        bio_label = f"{prefix}-{entity['label']}"
        if bio_label not in label_to_id:
            raise ValueError(f"No configured training label for {bio_label!r}")
        labels.append(label_to_id[bio_label])
        previous_entity = entity

    encoding["labels"] = labels
    encoding.pop("offset_mapping")
    return encoding


class ClinicalEntityDataset(Dataset):
    """Lazily tokenize an annotated-note collection."""

    def __init__(
        self,
        notes: list[dict[str, Any]],
        tokenizer: Any,
        label_to_id: dict[str, int],
        max_length: int,
    ) -> None:
        self.notes = notes
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.notes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return tokenize_and_align_labels(
            self.notes[index],
            self.tokenizer,
            self.label_to_id,
            self.max_length,
        )
