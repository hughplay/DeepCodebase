import logging
import os
import random
import re
import signal
import threading
import time
from pathlib import Path

import psutil

logger = logging.getLogger(__name__)


REFRESH_SECONDS = 30


def get_available_gpus(
    num: int = -1,
    min_memory: int = 20000,
    random_select: bool = True,
    wait_time: float = float("inf"),
):
    """Get available GPUs.

    Parameters
    ----------
    num : int, optional
        Number of GPUs to get. The default is -1.
    min_memory : int, optional
        Minimum memory available in GB. The default is 20000.
    random_select : bool, optional
        Random select a GPU. The default is True.
    wait_time : float, optional
        Wait time in seconds. The default is inf.
    """
    from gpustat import new_query

    start = time.time()
    while time.time() - start < wait_time:
        gpu_list = new_query().gpus
        if random_select:
            random.shuffle(gpu_list)
        sorted_gpu_list = sorted(
            gpu_list,
            key=lambda card: (
                card.entry["utilization.gpu"],
                card.entry["memory.used"],
            ),
        )
        available_gpus = [
            gpu.entry["index"]
            for gpu in sorted_gpu_list
            if gpu.entry["memory.total"] - gpu.entry["memory.used"]
            >= min_memory
        ]
        if num > 0:
            available_gpus = available_gpus[:num]
        if len(available_gpus) > 0:
            return available_gpus
        else:
            logger.info(
                f"No GPU available, having waited {time.time() - start} seconds"
            )
            time.sleep(REFRESH_SECONDS)
    raise Exception("No GPU available")


def wait_until_memory_available(
    min_percent: float = 20.0,
    min_memory: float = 20.0,
    wait_time: float = float("inf"),
):
    """Wait until memory available.

    Parameters
    ----------
    min_percent : float, optional
        Minimum percent of memory available. The default is 20.
    min_memory : float, optional
        Minimum memory available in GB. The default is 20.
    wait_time : float, optional
        Wait time in seconds. The default is inf.
    """

    start = time.time()
    while time.time() - start < wait_time:
        mem = psutil.virtual_memory()
        if mem.available > min_memory * (2**30) and (
            mem.percent < (100.0 - min_percent)
        ):
            return
        logger.info(
            f"No memory available, having waited {time.time() - start} seconds"
        )
        time.sleep(REFRESH_SECONDS)
    raise Exception("No Memory available")


def force_finish_wandb():
    wandb_log_path = Path("wandb/latest-run/logs/debug-internal.log")
    if not wandb_log_path.exists():
        print("wandb log file not found.")
        return
    with open(wandb_log_path, "r") as f:
        last_line = f.readlines()[-1]
    match = re.search(r"(HandlerThread:|SenderThread:)\s*(\d+)", last_line)
    if match:
        pid = int(match.group(2))
        print(f"wandb pid: {pid}")
    else:
        print("Cannot find wandb process-id.")
        return

    try:
        os.kill(pid, signal.SIGKILL)
        print(f"Process with PID {pid} killed successfully.")
    except OSError:
        print(f"Failed to kill process with PID {pid}.")


# Start wandb.finish() and execute force_finish_wandb() after 60 seconds.
def try_finish_wandb():
    import wandb

    threading.Timer(60, force_finish_wandb).start()
    wandb.finish()
