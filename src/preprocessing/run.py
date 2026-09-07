import os
import hydra
import pyrootutils
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.preprocessing.utils import (
    get_logger, 
    save_json_file,
    extract_age_from_text,
    extract_sex_from_text,
    word_frequencies_counter,
    )
from collections import defaultdict
from src.preprocessing import annotator



root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
OmegaConf.register_new_resolver(
    "root",
    lambda: str(root),
)

logger = get_logger(__name__)

@hydra.main(version_base=None, config_path=str(root / "src/preprocessing"), config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg))

    annotator = instantiate(cfg.annotator)

    diagnostic_conditions = annotator.build_diagnostic_vocabulary()
    medications = annotator.build_medication_vocabulary()
    logger.info(f"Loaded {annotator.clinical_notes} clinical notes.")
    annotations = []
    for note in annotator.clinical_notes:
        annotated_note = annotator.annotate_entities_in_text(note, diagnostic_conditions, medications)
        annotations.append(annotated_note)
        logger.info(len(annotated_note))
        clinical_fields =  annotator.build_clinical_fields_from_annotated_note(annotated_note=annotated_note)
        logger.info(f"Extracted entities from note {annotated_note['note_id']}: {clinical_fields}")
    logger.info(f"saving data annotation at {annotator.output_path}")
    save_json_file(data=annotations, file_path=annotator.output_path)


    all_texts = [note.get("text", "") for note in annotator.clinical_notes]
    combined_text = " ".join(all_texts)


    # Extract age information from clinical notes
    ages = [extract_age_from_text(note.get("text", "")) for note in annotator.clinical_notes]
    ages = [age[0] for age in ages if age is not None]

    # Extract sex information from clinical notes
    sexes = [extract_sex_from_text(note.get("text", "")) for note in annotator.clinical_notes]
    sexes = [sex[0] for sex in sexes if sex is not None]
    male_count = sexes.count("male")
    female_count = sexes.count("female")

    diagnosis_counts = {}
    medication_counts = defaultdict(int)

    for diagnosis in annotator.guidelines:
        diag_count = sum(diagnosis.lower() in note.get("text", "").lower() for note in annotator.clinical_notes)
        diagnosis_counts[diagnosis] = diag_count 
    for med in medications:
        for note in annotator.clinical_notes:
            if med.lower() in note['text'].lower():
                medication_counts[med] =  medication_counts[med] + 1
        
    logger.info("Dataset Analysis Summary")
    logger.info("=" * 50)
    logger.info(f"Total Clinical Notes:{len(annotator.clinical_notes)}")
    logger.info(f"Total Guidelines: {len(annotator.guidelines)}")
    logger.info("Demographics Summary:")
    population_size = len(ages)
    logger.info(f"  Max Age: {max(ages)}")
    logger.info(f"  Minimum Age: {min(ages)}")
    logger.info(f"  Male Count: {male_count}")
    logger.info(f"  Female Count: {female_count}")
    logger.info("Diagnosis in clinical Counts Summary: 3 diags examples")

    for diagnosis, count in list(diagnosis_counts.items())[:3]:
        logger.info(f"  {diagnosis}: {count} recorded in clinical notes.")
        
    logger.info("Medications in clinical note Counts Summary: 3 meds examples")
    for med, count in list(medication_counts.items())[:3]:
        logger.info(f"  {med}: {count} recorded in clinical notes .")

if __name__ == "__main__":
    main()