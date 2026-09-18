"""Hydra entry point for deterministic guideline evaluation."""

import json
import logging

import hydra
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.guideline_engine.evaluator import GuidelineEvaluator


root = Path(__file__).resolve().parents[2]
OmegaConf.register_new_resolver("root", lambda: str(root), replace=True)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Guideline configuration:\n%s", OmegaConf.to_yaml(cfg))
    evaluator: GuidelineEvaluator = instantiate(cfg.evaluator)
    result = evaluator.run()
    logger.info("Guideline result:\n%s", json.dumps(result, indent=2))
    logger.info("Saved guideline result to %s", evaluator.output_path)


if __name__ == "__main__":
    main()
