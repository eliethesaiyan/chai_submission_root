"""Hydra entry point for entity extraction model training."""

import json
import logging

import hydra
import pyrootutils
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.training.train import EntityTrainer


root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
OmegaConf.register_new_resolver("root", lambda: str(root), replace=True)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Training configuration:\n%s", OmegaConf.to_yaml(cfg))
    trainer: EntityTrainer = instantiate(cfg.trainer)
    metrics = trainer.train()

    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    config_path = trainer.output_dir / "training_config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(resolved_config, file, indent=2)

    logger.info("Validation metrics:\n%s", json.dumps(metrics, indent=2))
    logger.info("Saved model artifacts to %s", trainer.output_dir)


if __name__ == "__main__":
    main()
