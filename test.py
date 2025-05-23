import argparse
import copy
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, List, Union

import lightning as lt
import wandb
from hydra import compose, initialize, initialize_config_dir
from hydra.utils import instantiate, to_absolute_path
from omegaconf import OmegaConf, open_dict

from src.utils.exptool import (
    Experiment,
    prepare_trainer_config,
    print_config,
    register_omegaconf_resolver,
)

register_omegaconf_resolver()

logging.basicConfig(
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

main_dir = Path(__file__).resolve().parent


# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
if torch.__version__ >= "1.12":
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True


# ======================================================
# testing override functions
# ======================================================


def default_override(config):
    # adjust values for devices
    config.trainer.num_nodes = 1
    config.trainer.devices = 1

    # larger batch size for testing
    config.dataset.batch_size = config.dataset.batch_size * 2

    return config


def test_original(config):
    return config


def test_example(config):
    config_dir = main_dir / "conf"
    with initialize_config_dir(config_dir=str(config_dir)):
        cfg = compose(config_name="train", overrides=["experiment=mnist_lenet"])
    # For example, test the model on a different dataset.
    # (Just for example, actually they share the same dataset here.)
    config.dataset = cfg.dataset
    return config


# ======================================================
# end of testing override functions
# ======================================================

# ======================================================
# testing pipeline
# ======================================================


def test(
    logdir: Union[str, Path],
    ckpt: Union[str, Path] = "best",
    update_config_func: Union[Callable, List[Callable]] = test_original,
    update_wandb: bool = False,
    wandb_entity: str = None,
    metrics_prefix: Union[str, List[str]] = "",
):
    logdir = Path(logdir).expanduser()
    os.chdir(logdir)

    # load experiment record from logdir
    experiment = Experiment(logdir, wandb_entity=wandb_entity)

    # deal with update_config_func & metrics_prefix
    if not isinstance(update_config_func, Iterable):
        update_config_func = [update_config_func]
    if isinstance(metrics_prefix, str):
        metrics_prefix = [metrics_prefix]
    if len(metrics_prefix) == 1 and len(update_config_func) > 1:
        metrics_prefix = [metrics_prefix[0]] * len(update_config_func)

    assert len(update_config_func) == len(
        metrics_prefix
    ), "update_config_func and metrics_prefix must have the same length"

    for func, prefix in zip(update_config_func, metrics_prefix):

        # override experiment config with default_override & update_config_func
        config = copy.deepcopy(experiment.config)
        OmegaConf.set_struct(config, True)
        with open_dict(config):
            config = default_override(config)
            if func is not None:
                logger.info(
                    f"\n===== Override experiment config with {func.__name__} ====="
                )
                config = func(config)

        # show experiment config
        print_config(config)

        # seed everything
        lt.seed_everything(config.seed)

        # initialize datamodule
        datamodule = instantiate(config.dataset)

        # initialize model
        pipeline = experiment.get_pipeline_model_loaded(ckpt, config=config)

        # initialize trainer
        cfg_trainer = prepare_trainer_config(config, logging=False)
        trainer = instantiate(cfg_trainer)

        # testing
        results = trainer.test(pipeline, datamodule=datamodule)

        if trainer.global_rank == 0:
            # log results
            prefix_link = (
                "" if len(prefix) == 0 or prefix.endswith("_") else "_"
            )
            results = [
                {
                    f"{prefix}{prefix_link}{key}": val
                    for key, val in result.items()
                }
                for result in results
            ]
            logger.info(f"{results}")

            # save results to file
            with open(logdir / "results.jsonl", "a") as f:
                record = {
                    "results": results,
                    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "func": func.__name__ if func is not None else "original",
                    "prefix": prefix,
                }
                f.write(f"{record}\n")

            # update wandb record
            if update_wandb:
                logger.info("update wandb.")
                api = wandb.Api()
                run = api.run(experiment.wandb_run_path)
                for result in results:
                    run.summary.update(result)


# ======================================================
# end of testing pipeline
# ======================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir")
    parser.add_argument("--ckpt", default="best")
    parser.add_argument(
        "--update_func",
        nargs="+",
        default=["test_original"],
        help="config update function",
    )

    parser.add_argument("--update_wandb", action="store_true")
    parser.add_argument("--entity", default=None)
    parser.add_argument(
        "--prefix", nargs="+", default="", help="wandb metrics prefix"
    )
    args = parser.parse_args()

    # name to funcs
    if args.update_func is None:
        args.update_func = [None]
    else:
        mod = sys.modules[__name__]
        update_config_func = [getattr(mod, func) for func in args.update_func]

    test(
        args.logdir,
        ckpt=args.ckpt,
        update_config_func=update_config_func,
        update_wandb=args.update_wandb,
        wandb_entity=args.entity,
        metrics_prefix=args.prefix,
    )
