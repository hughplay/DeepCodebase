from typing import Dict

import torch
from torch import nn
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy


class ClassificationCriterion(nn.Module):
    """Example of Criterion for classification tasks.

    The duty of a criterion class: model outputs and targets in, loss and
    metrics out.
    """

    def __init__(self):
        super().__init__()

        self.loss = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.acc_train = Accuracy()
        self.acc_val = Accuracy()
        self.acc_test = Accuracy()

        self.acc = {
            "train": self.acc_train,
            "val": self.acc_val,
            "test": self.acc_test,
        }

        # for logging best so far validation accuracy
        self.acc_best_val = MaxMetric()
        self.acc_best = {
            "val": self.acc_best_val,
        }

    def forward(
        self, outputs: Dict[str, torch.Tensor], stage="train"
    ) -> Dict[str, torch.Tensor]:
        # compute loss
        loss = self.loss(outputs["logits"], outputs["targets"])
        # compute predictions from logits
        preds = torch.argmax(outputs["logits"], dim=1)
        # compute metrics
        acc = self.acc[stage](preds, outputs["targets"])

        outputs.update({"preds": preds, "loss": loss, "acc": acc})
        return outputs

    def epoch_end(self, stage="train"):
        self.acc[stage].reset()

    def train_start(self):
        for key in self.acc:
            self.acc[key].reset()
        for key in self.acc_best:
            self.acc[key].reset()

    @property
    def val_acc_best(self):
        acc = self.acc["val"].compute()
        self.acc_best["val"].update(acc)
        return self.acc_best["val"].compute()
