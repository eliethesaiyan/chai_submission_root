"""Hydra entry point for clinical entity extraction."""

import logging

import hydra
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.extraction.predictor import EntityPredictor


root = Path(__file__).resolve().parents[2]
OmegaConf.register_new_resolver("root", lambda: str(root), replace=True)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Inference configuration:\n%s", OmegaConf.to_yaml(cfg))
    predictor: EntityPredictor = instantiate(cfg.predictor)
    prediction = predictor.run()
    logger.info("Inference result:\n%s", OmegaConf.to_yaml(prediction))
    logger.info("Saved prediction to %s", predictor.output_path)


if __name__ == "__main__":
    main()
