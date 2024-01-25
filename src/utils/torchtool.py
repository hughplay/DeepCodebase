from io import BytesIO

import torch
import torch.distributed as dist
from torch import nn


def is_rank_zero():
    """Check if the current process is in rank zero (master process).

    Returns:
        bool: True if the current process is in rank zero, False otherwise.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # Check in PyTorch distributed training
        return dist.get_rank() == 0
    else:
        # Default behavior (assume rank zero if not in distributed setting)
        return True


def get_model_size_mb(model: nn.Module) -> float:
    """Calculates the size of a Module in megabytes.

    The computation includes everything in the :meth:`~torch.nn.Module.state_dict`,
    i.e., by default the parameters and buffers.

    Returns:
        Number of megabytes in the parameters of the input module.
    """
    model_size = BytesIO()
    torch.save(model.state_dict(), model_size)
    size_mb = model_size.getbuffer().nbytes / 1e6
    return size_mb
