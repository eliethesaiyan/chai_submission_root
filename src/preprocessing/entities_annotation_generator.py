import re, json

from pathlib import Path
from dataset_analysis import (
    load_json_file,
    extract_age_from_text,
    extract_sex_from_text,
)

def make_entity_from_text(text: str, start: int, end: int, label: str) -> dict:
    """
    uses position of the entity in the text to create an entity annotation. 

    Args:
        text (str): The text to annotate.
        start (int): The start position of the entity.
        end (int): The end position of the entity.
        label (str): The label for the entity.

    Returns:
        dict: An entity annotation.
    """
    return {
        "text": text[start:end],
        "start": start,
        "end": end,
        "label": label
    }

def extract_symptoms_from_text(text: str) -> list:
    """
    Extract symptoms from note from the first phrase marked by.  given text.

    Args:
        text (str): Input text.

    Returns:
        List[str]: A list of extracted symptoms.
    """
    # Use regex to find symptom patterns (e.g., "symptoms: fever, cough, headache")
    symptom_entities = []
    symptom_sentence_end = text.find(".")
    raw_symptom_text = text[:symptom_sentence_end] if symptom_sentence_end != -1 else text
    match = re.search(
        r"(?:presenting\s+)?with\s+(.+)$",
        raw_symptom_text,
        flags=re.IGNORECASE,
    )

    if not match:
        return []
    symptom_text = match.group(1)
    # remove days from the end of the symptom sentence if present
    symptom_text = re.sub(
        r"\s+for\s+\w+\s+days\s*$",
        "",
        symptom_text,
        flags=re.IGNORECASE,
    )

    valid_symptoms = [symptom.strip() for symptom in re.split(r",|and", symptom_text) if symptom.strip()]
    symptom_section_start = match.start(1)
    search_position = symptom_section_start
    # find symptom entities offset in the original text
    for symptom in valid_symptoms:
        symptom_match = re.search(
            re.escape(symptom), 
            text[search_position:], 
            flags=re.IGNORECASE
            )
        if not symptom_match:
            continue
        symptom_start = search_position + symptom_match.start()
        symptom_end = search_position + symptom_match.end()
        entity_annotation = make_entity_from_text(text, symptom_start, symptom_end, "symptoms")
        symptom_entities.append(entity_annotation)
        search_position = symptom_end  # Update search position to avoid overlapping matches
    return symptom_entities

def build_diagnostic_vocabulary(guidelines  : dict) -> list:
    """
    Build diagnosis conditions from guidelines, longest term first.

    Args:
        clinical_notes (dict): List of diagnosable conditions.
   """
    diagnostic_conditions = list(guidelines.keys())

    diagnostic_conditions.sort(key=len, reverse=True)
    return diagnostic_conditions

def build_medication_vocabulary(guidelines  : dict) -> list:
    """
    Build medication vocabulary from guidelines, longest term first.

    Args:
        medications (set): List of all available medications.
   """
    medications = set() 
    for rules in guidelines.values():
        for rule in rules["recommended_drugs"]:
            medications.add(rule)
        for rule in rules["avoid_drugs"]:
            medications.add(rule)
    #medications = [item for sublist in medications for item in sublist]  # Flatten

    medications = sorted(medications, key=len, reverse=True)

    return medications

def find_terms_in_notes(text: str, terms: list, label) -> list:
    """
    Find vocabularyterms in text and return their positions.

    Args:
        text (str): Input text.
        terms (list): List of terms to find.

    Returns:
        List[dict]: A list of found terms with their positions.
    """
    entities = []

    for term in terms:
        pattern = ( r"(?<!\w)" + re.escape(term) + r"(?!\w)")
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            entities.append(make_entity_from_text(text, match.start(), match.end(), label))
    return entities

def extract_diagnosis_from_text(text: str, diagnostic_conditions: list) -> list:
    """
    Extract diagnosis from note from the first phrase marked by.  given text.

    Args:
        text (str): Input text.
        diagnostic_conditions (list): List of diagnosable conditions.

    Returns:
        List[dict]: A list of extracted diagnosis entities.
    """
    return find_terms_in_notes(text, diagnostic_conditions, "diagnosis")

def extract_medications_from_text(text: str, medications: list) -> list:
    """
    Extract medications from note from the first phrase marked by.  given text.

    Args:
        text (str): Input text.
        medications (list): List of medications.

    Returns:
        List[dict]: A list of extracted medication entities.
    """
    return find_terms_in_notes(text, medications, "medications")

def annotate_entities_in_text(note: dict, diagnostic_conditions: list, medications: list) -> dict:
    """
    Annotate entities in the given text.

    Args:
        text (str): Input text.
        diagnostic_conditions (list): List of diagnosable conditions.
        medications (list): List of medications.

    Returns:
        dict: A dictionary containing extracted entities.
    """
    entities = []
    text = note.get("original_text", "")
    note_id = note.get("id", "")
    age, age_start, age_end = extract_age_from_text(text)
    age_entity = make_entity_from_text(text, age_start, age_end, "age") if age is not None else None
    sex, sex_start, sex_end = extract_sex_from_text(text)
    sex_entity = make_entity_from_text(text, sex_start, sex_end, "sex") if sex is not None else None

    entities.extend([age_entity,sex_entity])
    entities.extend(extract_symptoms_from_text(text))
    entities.extend(extract_diagnosis_from_text(text, diagnostic_conditions))
    entities.extend(extract_medications_from_text(text, medications))
    entities.sort(
        key=lambda entity: (
            entity["start"],
            entity["end"]
        )
    )

    return {
        "note_id": note_id,
        "text": text,
        "entities": entities
    }

def build_clinical_fields_from_annotated_note(annotated_note: dict) -> dict:
    """
    Builds structured field from an annotated note.

    Args:
        annotated_note (dict): The annotated note.

    Returns:
        dict: The structured clinical fields.
    """
    clinical_fields = {
        "note_id": annotated_note.get("note_id"),
        "age": None,
        "sex": None,
        "symptoms": [],
        "diagnosis": [],
        "medications": []
    }

    for entity in annotated_note.get("entities", []):
        if entity.get("label") == "age":
            clinical_fields["age"] = entity.get("text")
        elif entity.get("label") == "sex":
            clinical_fields["sex"] = entity.get("text")
        elif entity.get("label") == "symptoms":
            clinical_fields["symptoms"].append(entity.get("text"))
        elif entity.get("label") == "diagnosis":
            clinical_fields["diagnosis"].append(entity.get("text"))
        elif entity.get("label") == "medications":
            clinical_fields["medications"].append(entity.get("text"))

    return clinical_fields  

if __name__ == "__main__":
    # Example usage
    ROOT_DIR = Path(__file__).resolve().parent.parent
    processed_clinical_note_path = ROOT_DIR / "data/processed/processed_clinical_notes.json"
    processed_guideline_path = ROOT_DIR / "data/processed/processed_guidelines.json"
    entity_annotation_output_path = ROOT_DIR / "data/processed/entity_annotation_output.json"
    clinical_notes = load_json_file(processed_clinical_note_path)
    guidelines_notes = load_json_file(processed_guideline_path)
    diagnostic_conditions = build_diagnostic_vocabulary(guidelines_notes)
    medications = build_medication_vocabulary(guidelines_notes)
    print(f"Loaded {clinical_notes} clinical notes.")
    annotations = []
    for note in clinical_notes:
        annotated_note = annotate_entities_in_text(note, diagnostic_conditions, medications)
        annotations.append(annotated_note)
        print(len(annotated_note))
        clinical_fields =  build_clinical_fields_from_annotated_note(annotated_note=annotated_note)
        print(f"Extracted symptoms from note {clinical_fields}")

    with open(entity_annotation_output_path, "w", encoding="utf-8") as f:
        json.dump(
            annotations, 
            f,
            indent=2,
            ensure_ascii=False,
            )