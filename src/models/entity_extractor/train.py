import json
import random
from pathlib import Path

import numpy as np
import torch

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from seqeval.metrics import(
    precision_score,
    recall_score,
    f1_score
)


from src.models.entity_extractor.tokenization import(
    MODEL_NAME,
    LABELS,
    LABELS2ID,
    ID2LABEL,
    tokenizer,
    tokenize_and_align_labels,
)
from src.preprocessing.preprocess import(
    set_seed, 
    load_json_file
)

#configs
SEED = 42
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ANNOTATION_PATH = ROOT_DIR / "data/processed/entity_annotation_output.json"
MODEL_CHECKPOINTS_PATH = ROOT_DIR / "models/checkpoints"
TRAINING_CONFIG_PATH = MODEL_CHECKPOINTS_PATH / "training_config.json"

lr = 2e-5
BATCH_SIZE = 8
NUM_EPOCHS = 10
WEIGHT_DECAY = 0.01
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")

# all diagnosis are not created equal, use them for stratified sampling criteria

def get_diagnosis(annotated_note:dict):
    for entity in annotated_note["entities"]:
        if entity["label"] == "diagnosis":
            return entity["text"].lower()
## Dataset class

class ClinicalAIDataset(Dataset):

    def __init__(self, data):
        self.data = data 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        _item = self.data[idx]
        encoding = (
            tokenize_and_align_labels(
               _item 
                )
        )
        encoding.pop("offset_mapping", None)# DistillBERT requirement

        return encoding


def create_splits(annotations):
    diagnoses = [
        get_diagnosis( annotated_note=note) for note in annotations

    ]
    train_split, val_split = (
        train_test_split(
            annotations,
            test_size=0.20,
            random_state=SEED,
            stratify=diagnoses,

        )
    )
    return train_split, val_split

def print_split_info(
        train_notes,
        val_notes,
    ):
    print("="* 50)
    print("Dataset split")
    print("="* 50)
    print(f"training notes", len(train_notes))
    print(f"val notes", len(val_notes))
    for note in val_notes:
        print(f"notes {note['note_id']}", get_diagnosis(note))

def compute_metrics(eval_prediction):
    predictions, labels = eval_prediction

    predictions = np.argmax(
        predictions,
        axis=2,
    )

    true_predictions = []
    true_labels = []

    for prediction, label in zip(
        predictions,
        labels,
    ):

        predicted_sequence = []
        label_sequence = []

        for predicted_id, label_id in zip(
            prediction,
            label,
        ):

            # Ignore special tokens and padding
            if label_id == -100:
                continue

            predicted_sequence.append(
                ID2LABEL[int(predicted_id)]
            )

            label_sequence.append(
                ID2LABEL[int(label_id)]
            )

        true_predictions.append(
            predicted_sequence
        )

        true_labels.append(
            label_sequence
        )

    return {
        "precision": precision_score(
            true_labels,
            true_predictions,
        ),
        "recall": recall_score(
            true_labels,
            true_predictions,
        ),
        "f1": f1_score(
            true_labels,
            true_predictions,
        ),
    }

def save_training_config(config_file_path:str)->None:
    config = {
        "model_name": MODEL_NAME,
        "seed": SEED, 
        "lr": lr,
        "device":DEVICE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "labels": LABELS,
        "labels2id": LABELS2ID
    }
    with open(config_file_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    

if __name__ == "__main__":
    set_seed(seed=SEED)
    # dataset prep
    annotated_notes = load_json_file(ANNOTATION_PATH)
    print(f"annotations {len(annotated_notes)}")
    train_notes, val_notes = create_splits(annotated_notes)

    train_dataset = (
        ClinicalAIDataset(train_notes)
    )
    val_dataset = (
        ClinicalAIDataset(val_notes)
    )

model = (
    AutoModelForTokenClassification
    .from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABELS2ID,
        )
)
#padding and data collation

data_collator = (
    DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )
)

# train args

training_args = TrainingArguments(
    output_dir=str(MODEL_CHECKPOINTS_PATH),
    learning_rate=lr,
    per_device_train_batch_size=(
        BATCH_SIZE
    ),
    per_device_eval_batch_size=(
        BATCH_SIZE
    ),
    num_train_epochs=(
        NUM_EPOCHS
    ),
    weight_decay=(
        WEIGHT_DECAY
        ),
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    save_total_limit=2,
    seed=SEED,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=(data_collator),
    processing_class=(tokenizer),
    compute_metrics=compute_metrics,
)   


print()
print("=" * 60)
print("TRAINING")
print("=" * 60)

trainer.train()

# -----------------------------------------------------
# Final validation
# -----------------------------------------------------

print()
print("=" * 60)
print("VALIDATION")
print("=" * 60)

results = trainer.evaluate()

for key, value in results.items():
    print(
         f"{key}: {value}"
    )

# -----------------------------------------------------
    # Save final model
    # -----------------------------------------------------

MODEL_CHECKPOINTS_PATH.mkdir(
parents=True,
exist_ok=True,
)

trainer.save_model(
    str(MODEL_CHECKPOINTS_PATH)
    )

tokenizer.save_pretrained(
        str(MODEL_CHECKPOINTS_PATH)
    )

save_training_config(config_file_path=MODEL_CHECKPOINTS_PATH/"train_config.json")

print()
print("=" * 60)

print(
    "Model saved to:"
)
    

