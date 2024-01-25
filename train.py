import logging
from pathlib import Path

import dotenv
import hydra
import lightning as lt
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.utils.exptool import (
    prepare_trainer_config,
    print_config,
    register_omegaconf_resolver,
    try_resume,
)

register_omegaconf_resolver()
logger = logging.getLogger(__name__)

lt._logger.handlers = []
lt._logger.propagate = True

dotenv.load_dotenv(override=True)


# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
if torch.__version__ >= "1.12":
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True


@hydra.main(version_base="1.3", config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:

    # Assign hydra.run.dir=<previous_log_dir> to resume training.
    # If previous checkpoints are detected, `cfg` will be replaced with the
    # previous config
    cfg = try_resume(cfg)

    # Print & save config to logdir
    output_dir = Path(cfg.paths.output_dir)
    print_config(cfg, save_path=output_dir / "config.yaml")

    # Set random seed
    if cfg.seed is not None:
        lt.seed_everything(cfg.seed)

    # Initialize datamodule
    datamodule = instantiate(cfg.dataset)

    # Initialize pipeline
    pipeline = instantiate(cfg.pipeline, cfg=cfg, _recursive_=False)

    # Initialize trainer
    cfg_trainer = prepare_trainer_config(cfg)
    trainer = instantiate(cfg_trainer)

    # Training
    if cfg.resume_ckpt is not None:
        logger.info(f"resume from {cfg.resume_ckpt}")
    trainer.fit(pipeline, datamodule, ckpt_path=cfg.resume_ckpt)

    # Testing
    if cfg.run_test:
        trainer.test(pipeline, datamodule, ckpt_path="best")

    # print logdir for conveniently copy
    logger.info(f"Logdir: {Path('.').resolve()}")


if __name__ == "__main__":
    main()
