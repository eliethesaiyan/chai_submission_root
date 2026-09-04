import json
from pathlib import Path
from transformers import AutoTokenizer

from src.preprocessing.preprocess import load_json_file


MAX_LENGTH = 128

LABELS = [
    "O",
    "B-age",
    "I-age",
    "B-sex",
    "I-sex",
    "B-symptoms",
    "I-symptoms",
    "B-diagnosis",
    "I-diagnosis",
    "B-medications",
    "I-medications",
]

LABELS2ID = {
    label:idx for idx, label in enumerate(LABELS)
}

ID2LABEL = {
    idx:label for idx, label in enumerate(LABELS)
}

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)

def load_annnotations(annotation_path: str|Path)-> list:
    data = load_json_file(annotation_path)
    return data

def find_entity_for_token(token_start: int, token_end:int, entities: list)-> str:

    for entity in entities:
        entity_start = entity["start"]
        entity_end = entity["end"]

        if (token_start < entity_end and token_end> entity_start): # return entity that contains the token
            return entity
    return None

def tokenize_and_align_labels(note:dict):
    text = note["text"]
    entities = note["entities"]

    encoding = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
        )
    offsets = encoding["offset_mapping"]
    labels =[]

    previous_entity = None

    for token_start, token_end in offsets:

        if token_start == token_end:
            labels.append(-100)
            previous_entity = None
            continue
        
        entity = find_entity_for_token(
            token_start=token_start, 
            token_end=token_end,
            entities=entities
        )

        if entity is None:
            labels.append(LABELS2ID["O"])
            previous_entity = None
            continue

        entity_type = entity["label"]

        if previous_entity is not entity:
            bio_label =(f"B-{entity_type}")
        else:
            bio_label = (f"I-{entity_type}")
        
        labels.append(LABELS2ID[bio_label])

        previous_entity = entity
    encoding["labels"] = labels
    
    return encoding
    
def print_tokenized_example(note):

    encoding = tokenize_and_align_labels(
       note 
    )

    tokens = tokenizer.convert_ids_to_tokens(
        encoding["input_ids"]
    )

    labels = encoding["labels"]

    offsets = encoding[
        "offset_mapping"
    ]

    print()
    print("=" * 75)

    print(
        f"NOTE: {note['note_id']}"
    )

    print("=" * 75)

    print(note["text"])

    print()

    print(
        f"{'TOKEN':<20}"
        f"{'LABEL':<20}"
        f"{'OFFSET'}"
    )

    print("-" * 75)

    for token, label_id, offset in zip(
        tokens,
        labels,
        offsets,
    ):

        if label_id == -100:

            label = "IGNORE"

        else:

            label = ID2LABEL[
                label_id
            ]

        print(
            f"{token:<20}"
            f"{label:<20}"
            f"{offset}"
        )



if __name__ == "__main__":
  ROOT_DIR = Path(__file__).resolve().parent.parent.parent
  entity_annotation_output_path = ROOT_DIR / "data/processed/entity_annotation_output.json"
  data = load_annnotations(annotation_path=entity_annotation_output_path)
  print(len(data))
  for label, label_id in LABELS2ID.items():
      print(f"{label_id:>2}-> {label}")
  
  for note in data:
      print_tokenized_example(note)
  