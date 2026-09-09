"""Hydra entry point for entity extraction evaluation."""

import json
import logging

import hydra
import pyrootutils
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.evaluation.evaluate import EntityEvaluator


root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
OmegaConf.register_new_resolver("root", lambda: str(root), replace=True)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Evaluation configuration:\n%s", OmegaConf.to_yaml(cfg))
    evaluator: EntityEvaluator = instantiate(cfg.evaluator)
    metrics = evaluator.evaluate()
    evaluator.save(metrics)
    logger.info("Evaluation metrics:\n%s", json.dumps(metrics, indent=2))
    logger.info("Saved evaluation report to %s", evaluator.output_path)


if __name__ == "__main__":
    main()
