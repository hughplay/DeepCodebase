"""
Usage:
- python batchrun.py scripts/test_batchrun.sh --quotas 4
- python batchrun.py scripts/test_batchrun.sh --gpus 0,1,2,3,4,5 --quotas 4 2 3 3
"""
import argparse
import subprocess
import time
from multiprocessing import Lock, Process, Queue
from pathlib import Path

import torch

from src.utils.timetool import time2str


def get_commands(path_to_script):
    commands = []
    with open(path_to_script, "r") as f:
        lines = [
            line.strip() for line in f.readlines() if not line.startswith("#")
        ]
    command = ""
    for line in lines:
        if line.endswith("\\"):
            command += f"{line[:-1].strip()} "
        else:
            command += line
            if len(command.strip()) > 0:
                commands.append(command.strip())
            command = ""
    return commands


def run_command(command, gpu_queue, quota, lock, logdir):
    gpus = []
    p = None
    try:
        t_start = time.time()

        # try to get quota GPUs
        while True:
            with lock:
                if gpu_queue.qsize() >= quota:
                    for _ in range(quota):
                        gpus.append(gpu_queue.get())
                    break
            time.sleep(10)

        name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        args = command.split()
        for arg in args:
            if arg.startswith("name="):
                name += f"-{arg.split('=')[1]}"
        log_path = Path(logdir) / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = (
            f"CUDA_VISIBLE_DEVICES={','.join(gpus)} {command} >> {log_path}"
        )
        print(f"Running command: {command}")
        # command = "sleep infinity"
        p = subprocess.Popen(command, shell=True)
        p.wait()
        time.sleep(10)
    except KeyboardInterrupt:
        if p is not None:
            p.kill()
        exit(0)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if p is not None:
            p.kill()
        if gpus is not None:
            for gpu in gpus:
                gpu_queue.put(gpu)
        print(
            f"Process {name} finished. "
            f"Time elapsed: {time2str(time.time() - t_start)}"
        )
        print(f"GPUs left: {gpu_queue.qsize()}")


def main(commands, gpus, quotas, logdir):

    assert len(quotas) == 1 or len(commands) == len(quotas), (
        f"Length of quota must be 1 or equal to the number of commands. "
        f"Got {len(quotas)} and {len(commands)}."
    )
    if len(quotas) == 1:
        quotas = quotas * len(commands)

    gpus_queue = Queue()
    lock = Lock()
    for gpu in gpus:
        gpus_queue.put(gpu)

    processes = []
    for command, quota in zip(commands, quotas):
        process = Process(
            target=run_command, args=(command, gpus_queue, quota, lock, logdir)
        )
        time.sleep(10)
        process.start()
        processes.append(process)

    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            print("Keyboard interrupt. Exiting.")
            if process.is_alive():
                print(f"Killing process {process.pid}")
                process.terminate()

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "script",
        type=str,
        help="Path to script to be run. Each line is a command.",
    )
    parser.add_argument("--gpus", type=str, default=None, help="GPUs to use")
    parser.add_argument(
        "--quotas",
        type=int,
        default=[1],
        nargs="+",
        help="GPUs per command, can be an integer or a list of integers",
    )
    parser.add_argument(
        "--logdir", type=str, default="/log/running", help="Log directory"
    )
    args = parser.parse_args()

    if args.gpus is None:
        args.gpus = ",".join([str(i) for i in range(torch.cuda.device_count())])
    commands = get_commands(args.script)

    main(commands, args.gpus.split(","), args.quotas, args.logdir)
