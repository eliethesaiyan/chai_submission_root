from __future__ import annotations# silencing the warning for forward references in type hints

import json
import re

from collections import Counter
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

def load_json_file(file_path: Path | str) -> List[Dict[str, Any]]:
    """
    Load clinical/guidelines JSON files.

    Args:
        file_path (str): Path to the JSON file 
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    with file_path.open("r") as f:
        data = json.load(f)

    return data

def word_frequencies_counter(text: str) -> Counter:
    """
    Compute word frequencies in a given text.

    Args:
        text (str): Input text.

    Returns:
        Counter: A Counter object with word frequencies.
    """
    # Normalize and tokenize the text
    words = re.findall(r"\b\w+(?:[-']\w+)*\b", text.lower().strip())
    return Counter(words)

def sentence_count(texts: List[str]) -> int:
    """
    Compute the distribution of sentence lengths in a list of texts.

    Args:
        texts (List[str]): List of input texts.

    Returns:
        int: Total number of sentences in the input texts.
    """
    total_sentences = 0
    for text in texts:
        # Count sentences using a simple heuristic (split by '.', '!', '?')
        sentences = re.split(r"[.!?]", text)
        total_sentences += len(sentences)
    return total_sentences

def extract_age_from_text(text: str) -> int:
    """
    Extract age information from a given text.

    Args:
        text (str): Input text.

    Returns:
        List[int]: A list of extracted ages.
    """
    # Use regex to find age patterns (e.g., "45 years old as in clincal notes")
    match_patterns = re.search(r"\b(\d+)\s+year old\b", text.lower())
    age = int(match_patterns.group(1)) if match_patterns else None
    start, end  = match_patterns.span(1) if match_patterns else (0, 0)
    return age, start, end 

def extract_sex_from_text(text: str) -> str:
    """
    Extract sex information from a given text.

    Args:
        text (str): Input text.

    Returns:
        str: Extracted sex concept or None if not found.
    """
    # Use regex to find sex patterns (e.g., "male" or "female")
    match_patterns = re.search(r"\b year old\s+(male|female)\b", text.lower())
    sex = match_patterns.group(1) if match_patterns else None
    start, end  = match_patterns.span(1) if match_patterns else (0,0)

    return sex, start, end 

if __name__ == "__main__":
    # Example usage
    ROOT_DIR = Path(__file__).resolve().parent.parent
    clinical_notes_path = Path(ROOT_DIR, "data", "processed", "processed_clinical_notes.json")
    guidelines_path = Path(ROOT_DIR, "data", "processed", "processed_guidelines.json")

    clinical_notes = load_json_file(clinical_notes_path)
    guidelines = load_json_file(guidelines_path)

    # Compute word frequencies for clinical notes
    all_texts = [note.get("normalized_text", "") for note in clinical_notes]
    combined_text = " ".join(all_texts)
    word_frequencies_counter = word_frequencies_counter(combined_text)
    print("Word Frequencies in Clinical Notes:", word_frequencies_counter.most_common(10))

    # Compute sentence count for clinical notes
    total_sentences = sentence_count(all_texts)
    print("Total Sentences in Clinical Notes:", total_sentences)

    # Extract age information from clinical notes
    ages = [extract_age_from_text(note.get("normalized_text", "")) for note in clinical_notes]
    ages = [age[0] for age in ages if age is not None]
    print("Extracted Ages from Clinical Notes:", ages)

    # Extract sex information from clinical notes
    sexes = [extract_sex_from_text(note.get("normalized_text", "")) for note in clinical_notes]
    sexes = [sex[0] for sex in sexes if sex is not None]
    male_count = sexes.count("male")
    female_count = sexes.count("female")
    print("Extracted Sexes from Clinical Notes:", sexes)

    diagnosis_counts = {}

    for diagnosis in guidelines:
        count = sum(diagnosis.lower() in note.get("normalized_text", "").lower() for note in clinical_notes)
        diagnosis_counts[diagnosis] = count

    print("Diagnosis Counts in Clinical Notes:", diagnosis_counts)
    print("=" * 50)
    print("Dataset Analysis Summary")
    print("=" * 50)
    print("Total Clinical Notes:", len(clinical_notes),"\n")
    print("Total Guidelines:", len(guidelines),"\n")
    print("Demographics Summary:")
    population_size = len(ages)
    print("  Max Age:", max(ages))
    print("  Minimum Age:", min(ages))
    print("  Male Count:", male_count)
    print("  Female Count:", female_count)
    print("Diagnosis Counts Summary:")
    for diagnosis, count in diagnosis_counts.items():
        print(f"  {diagnosis}: {count} recorded in clinical notes.\n")