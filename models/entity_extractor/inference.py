from pathlib import Path
import torch

from transformers import(
    AutoTokenizer,
    AutoModelForTokenClassification
)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODEL_CHECKPOINT_DIR = ROOT_DIR /"src/models/checkpoints"
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_CHECKPOINT_DIR
)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT_DIR
)

model.to(DEVICE)
model.eval()

ID2LABEL = {
    int(k):v
    for k, v in model.config.id2label.items()
}

def predict_bio(text):
    encoding = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
    )
    offsets = encoding.pop("offset_mapping")[0]

    encoding = {
        key: value.to(DEVICE)
        for key, value in encoding.items()
    }

    with torch.no_grad():
        ouputs = model(**encoding)

    predictions = torch.argmax(ouputs.logits,dim=-1,)[0]

    tokens = tokenizer.convert_ids_to_tokens(
        encoding["input_ids"][0]
    )

    results = []
    for token, prediction, offset in zip(
        tokens,
        predictions,
        offsets
    ):
        start = int(offset[0])
        end = int(offset[1])

        if start == end:
            continue
        label = ID2LABEL[
            int(prediction)
        ]

        results.append(
            {
                "token": token,
                "label": label,
                "start": start,
                "end": end,
                "text": text[start:end],
                }
            )
    return results

def reconstruct_entities(
    text, 
    predictions
):
    entities = []
    current_entity = None
    for prediction in predictions:
        label = prediction["label"]
        if label == "O":
            if current_entity is not None:
                entities.append(
                    current_entity
                )
                current_entity = None
            continue

        prefix, entity_type = (
            label.split(
                "-",
                maxsplit=1,
            )
        )
        start = prediction["start"] 
        end = prediction["end"]
        
        if prefix == "B":
            if current_entity is not None:
                entities.append(
                    current_entity
                )
            current_entity = {
                "label": entity_type,
                "start": start,
                "end": end
            }
        elif prefix == "I":
            if (
                current_entity is not None
                and current_entity["label"] == entity_type
            ):
                current_entity["end"] = end
            else:
                if current_entity is not None:
                    entities.append(
                        current_entity
                    )
                current_entity = {
                    "label": entity_type,
                    "start": start,
                    "end": end
                }
    if current_entity is not None:
        entities.append(current_entity)
    
    for entity in entities:
        entity["text"] = text[
            entity["start"]:entity["end"]
        ]
    return entities

def predict_bio(text):

    encoding = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
    )

    offsets = encoding.pop(
        "offset_mapping"
    )[0]

    encoding = {
        key: value.to(DEVICE)
        for key, value in encoding.items()
    }

    with torch.no_grad():

        outputs = model(
            **encoding
        )

    predictions = torch.argmax(
        outputs.logits,
        dim=-1,
    )[0]

    tokens = tokenizer.convert_ids_to_tokens(
        encoding["input_ids"][0]
    )

    results = []

    for token, prediction, offset in zip(
        tokens,
        predictions,
        offsets,
    ):

        start = int(
            offset[0]
        )

        end = int(
            offset[1]
        )

        # Ignore special tokens
        if start == end:
            continue

        label = ID2LABEL[
            int(prediction)
        ]

        results.append(
            {
                "token": token,
                "label": label,
                "start": start,
                "end": end,
                "text": text[start:end],
            }
        )

    return results

def reconstruct_entities(
    text,
    predictions,
):
    entities = []

    current_type = None
    current_start = None
    current_end = None

    def close_entity():
        nonlocal current_type
        nonlocal current_start
        nonlocal current_end

        if current_type is not None:
            entities.append(
                {
                    "label": current_type,
                    "start": current_start,
                    "end": current_end,
                    "text": text[
                        current_start:
                        current_end
                    ],
                }
            )

        current_type = None
        current_start = None
        current_end = None

    for prediction in predictions:

        label = prediction["label"]

        # ---------------------------------
        # Outside an entity
        # ---------------------------------

        if label == "O":
            close_entity()
            continue

        prefix, entity_type = label.split(
            "-",
            maxsplit=1,
        )

        start = prediction["start"]
        end = prediction["end"]

        # ---------------------------------
        # Beginning of a new entity
        # ---------------------------------

        if prefix == "B":

            close_entity()

            current_type = entity_type
            current_start = start
            current_end = end

        # ---------------------------------
        # Continuation of an entity
        # ---------------------------------

        elif prefix == "I":

            if (
                current_type
                == entity_type
            ):
                current_end = end

            else:
                # Invalid I without matching B.
                # Treat it as a new entity.
                close_entity()

                current_type = entity_type
                current_start = start
                current_end = end

    # Close last entity
    close_entity()

    return entities


def to_structured_output(
    entities,
):

    output = {
        "age": None,
        "sex": None,
        "symptoms": [],
        "diagnosis": None,
        "medications": [],
    }

    for entity in entities:

        label = entity["label"]
        value = entity["text"]

        if label == "age":

            output["age"] = value

        elif label == "sex":

            output["sex"] = value

        elif label == "symptoms":

            output["symptoms"].append(
                value
            )

        elif label == "diagnosis":

            output["diagnosis"] = value

        elif label == "medications":

            output["medications"].append(
                value
            )

    return output

def extract_entities(text):

    predictions = predict_bio(
        text
    )

    entities = reconstruct_entities(
        text,
        predictions,
    )

    structured = to_structured_output(
        entities
    )

    return structured

if __name__ == "__main__":
    '''text = (
        "42 year old female with cough and fever. "
        "Diagnosed pneumonia. "
        "Started amoxicillin."
    )
    
    result = extract_entities( 
        text
    )

    print()
    print("INPUT")
    print("=" * 60)

    print(text)

    print()
    print("EXTRACTED ENTITIES")
    print("=" * 60)

    print(result)
    '''
    text1 = (
        "42 year old female with cough and fever. "
        "Diagnosed pneumonia. "
        "Started amoxicillin."
    )
    text2 = (
    "45 year old male with fever and productive cough for five days. "
    "Diagnosed with pneumonia. "
    "Started on amoxicillin."
    )

    predictions = predict_bio(text2)

    print("\nRAW MODEL OUTPUT")
    print("=" * 70)

    for pred in predictions:
        print(
            f"{pred['text']:<15}"
            f"{pred['label']:<20}"
            f"{pred['start']}:{pred['end']}"
        )

    print("\nRECONSTRUCTED")
    print("=" * 70)

    entities = reconstruct_entities(
        text2,
        predictions,
    )

    for entity in entities:
        print(entity)

    print("\nFINAL")
    print("=" * 70)

    print(
        to_structured_output(
            entities
        )
    )
    

    