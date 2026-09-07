# This file contains the preprocessing functions for the data.
#Silencing warnings from IDE
from __future__ import annotations
import os

import json, re
from pathlib import Path

from typing import List, Dict, Any
import numpy as np

import logging

def get_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)

    return logger

def load_json_data(file_path: Path| str) -> List[Dict[str, Any]]:
    """
    Load clinical  JSON file.

    Args:
        file_path (str): Path to the JSON file 
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    with file_path.open("r") as f:
        data = json.load(f)

    return data

def normalize_text(text: str) -> str:
    """
    Normalize text by :
    whitespace and short words removal, converting to lowercase and stripping whitespace.
    stemming and lemmatization,.

    Args:
        text (str): Input text.

    Returns:
        str: Normalized text.
    """
    text = text.lower().strip()
    # Apply stemming and lemmatization here if needed
    text = re.sub(r"\s+", " ", text)  # Remove extra whitespace
    text = re.sub(r"\s+([.,;:!?])", r"\1", text) # Remove whitespace before punctuation
    return text

def process_clinical_notes(clinical_notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process clinical notes to extract relevant information and normalize the text.

    Args:
        clinical_notes (List[Dict[str, Any]]): List of clinical notes.

    Returns:
        List[Dict[str, Any]]: Processed clinical notes.
    """
    processed_notes = []
    for note in clinical_notes:
        # Extracting only the 'text' and 'id' fields and origin text
        processed_note = {
            "id": note.get("note_id"),
            "original_text": note.get("text"),
            "normalized_text": normalize_text(note.get("text", "")),
        }
        processed_notes.append(processed_note)
    
    return processed_notes  

def process_guidelines(guidelines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process guidelines to extract relevant information and normalize the text.

    Args:
        guidelines (List[Dict[str, Any]]): List of guidelines.

    Returns:
        List[Dict[str, Any]]: Processed guidelines.
    """
    processed_guidelines = {} 
    for condition, rules in guidelines.items():
        processed_guideline = {
            "recommended_drugs": rules.get("recommended_drugs", []),
            "avoid_drugs": rules.get("avoid_drugs", []),
            "required_tests": rules.get("required_tests", []),
        }
        print(condition, processed_guideline)
        processed_guidelines[condition] = processed_guideline
    return processed_guidelines

def make_entity_from_text( text: str, start: int, end: int, label: str) -> dict:
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

def save_json_file(data: List[Dict[str, Any]], file_path: Path| str):
    """
    Save data to a JSON file.

    Args:
        data (List[Dict[str, Any]]): Data to be saved.
        file_path (str): Path to the output JSON file.
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    with file_path.open("w", encoding="utf-8") as f:
        try:
            json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving JSON file {file_path}: {e}")

#test functions
if __name__ == "__main__":
    # Example usage
    ROOT_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = Path(ROOT_DIR, "src", "data")
    clinical_notes = load_json_file(Path(DATA_DIR, "clinical_notes.json"))
    guidelines = load_json_file(Path(DATA_DIR, "guidelines.json"))
    print(f"Loaded {len(clinical_notes)} clinical notes.")
    print(f"Loaded {len(guidelines)} guidelines.")

    processed_clinical_notes = process_clinical_notes(clinical_notes)
    print(f"Processed {processed_clinical_notes} ")
    print(f"Processed {len(processed_clinical_notes)} clinical notes.")

    processed_guidelines = process_guidelines(guidelines)
    print(f"Processed {processed_guidelines} ")
    print(f"Processed {len(processed_guidelines)} guidelines.")

    save_json_file(processed_clinical_notes, Path(DATA_DIR, "processed", "processed_clinical_notes.json"))
    save_json_file(processed_guidelines, Path(DATA_DIR, "processed", "processed_guidelines.json"))