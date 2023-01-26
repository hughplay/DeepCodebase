import logging
from typing import Any, Dict, List

from hydra.utils import instantiate
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.memory import get_model_size_mb

logger = logging.getLogger(__name__)


class ClassificationLitModule(LightningModule):
    """Example of LightningModule for MNIST classification. A LightningModule
    organizes your PyTorch code into 5 sections:

        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(cfg)

        # initialize the model from configuration
        self.model = instantiate(self.hparams.model)

        # initialize the criterion from configuration
        self.criterion = instantiate(self.hparams.criterion)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your
        optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = instantiate(self.hparams.optim, self.parameters())
        scheduler = instantiate(
            self._set_num_training_steps(self.hparams.scheduler), optimizer
        )
        # torch's scheduler is epoch-based, but transformers' is step-based
        interval = (
            "step"
            if self.hparams.scheduler._target_.startswith("transformers")
            else "epoch"
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": interval,
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def _set_num_training_steps(self, scheduler_cfg):
        if "num_training_steps" in scheduler_cfg:
            scheduler_cfg = dict(scheduler_cfg)
            logger.info("Computing number of training steps...")
            scheduler_cfg[
                "num_training_steps"
            ] = self.trainer.estimated_stepping_batches

            if self.global_rank == 0:
                logger.info(
                    f"Training steps: {scheduler_cfg['num_training_steps']}"
                )
        return scheduler_cfg

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        # self.val_acc_best.reset()
        self.criterion.train_start()

        self.log(
            "model_size/total",
            get_model_size_mb(self.model),
            rank_zero_only=True,
            logger=True,
        )

    def step(self, batch: Any, stage="train"):
        features, targets = batch
        logits = self.model(features)

        # Compute loss and metrics, input keys and values are reserved
        outputs = self.criterion(
            {"logits": logits, "targets": targets},
            stage=stage,
        )
        return outputs

    def training_step(self, batch: Any, batch_idx: int):
        outputs = self.step(batch, stage="train")

        self.log(
            "train/loss",
            outputs["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train/acc",
            outputs["acc"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()`
        # or else backpropagation will fail!
        # but also remeber not to return much redundant tensors, or they may
        # accumulate and excced your GPU memory
        return {
            "loss": outputs["loss"],
            "preds": outputs["preds"],
            "targets": outputs["targets"],
        }

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.criterion.epoch_end(stage="train")

    def validation_step(self, batch: Any, batch_idx: int):
        outputs = self.step(batch, stage="val")

        self.log(
            "val/loss",
            outputs["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "val/acc",
            outputs["acc"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {
            "loss": outputs["loss"],
            "preds": outputs["preds"],
            "targets": outputs["targets"],
        }

    def validation_epoch_end(self, outputs: List[Any]):
        self.log(
            "val/acc_best",
            self.criterion.val_acc_best,
            on_epoch=True,
            prog_bar=True,
        )
        self.criterion.epoch_end(stage="val")

    def test_step(self, batch: Any, batch_idx: int):
        outputs = self.step(batch, stage="test")
        self.log("test/loss", outputs["loss"], on_step=False, on_epoch=True)
        self.log("test/acc", outputs["acc"], on_step=False, on_epoch=True)

        return {
            "loss": outputs["loss"],
            "preds": outputs["preds"],
            "targets": outputs["targets"],
        }

    def test_epoch_end(self, outputs: List[Any]):
        self.criterion.epoch_end(stage="test")
