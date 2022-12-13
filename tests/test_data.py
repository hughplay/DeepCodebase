import logging
from pathlib import Path

import pytest
import torch

from src.dataset.mnist import MNISTDataModule

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("batch_size", [32, 128])
def test_datamodule(batch_size):
    datamodule = MNISTDataModule(batch_size=batch_size)
    datamodule.prepare_data()

    assert (
        not datamodule.data_train
        and not datamodule.data_val
        and not datamodule.data_test
    )

    assert (Path("data") / "MNIST").exists()
    assert (Path("data") / "MNIST" / "raw").exists()

    datamodule.setup()

    assert (
        datamodule.data_train and datamodule.data_val and datamodule.data_test
    )
    assert (
        len(datamodule.data_train)
        + len(datamodule.data_val)
        + len(datamodule.data_test)
        == 70_000
    )

    logger.info(f"Training samples: {len(datamodule.data_train)}")
    logger.info(f"Validation samples: {len(datamodule.data_val)}")
    logger.info(f"Test samples: {len(datamodule.data_test)}")

    assert datamodule.train_dataloader()
    assert datamodule.val_dataloader()
    assert datamodule.test_dataloader()

    batch = next(iter(datamodule.train_dataloader()))
    x, y = batch

    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
