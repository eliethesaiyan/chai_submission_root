import os
import hydra
import pyrootutils
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from src.preprocessing.utils import get_logger, save_json_file
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
        logger.info(f"Extracted symptoms from note {clinical_fields}")
    logger.info(f"saving data annotation at {annotator.output_path}")
    save_json_file(data=annotations, file_path=annotator.output_path)


if __name__ == "__main__":
    main()