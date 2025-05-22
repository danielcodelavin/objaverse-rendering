import glob
import json
import multiprocessing
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional
import os
import boto3
import tyro
import wandb


@dataclass
class Args:
    workers_per_gpu: int = 1          
    """number of workers per GPU"""

    input_models_path: str = "truncated_input_models_path.json"   
    """Path to the JSON list of 3-D objects"""

    upload_to_s3: bool = False
    log_to_wandb: bool = False

    num_gpus: int = 1                
    """number of GPUs to use"""

    output_dir: str = (
        "/home/stud/lavingal/storage/slurm/lavingal/LVSM/datasets/trash/images"
    )                              

def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    gpu: int,
    s3: Optional[boto3.client],
) -> None:
    while True:
        item = queue.get()
        if item is None:
            break

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)     # ↳ Blender now sees *just* that GPU
        env["DISPLAY"] = f":0.{gpu}"               # optional: separate X-screen per GPU



        BLENDER = "/home/stud/lavingal/storage/slurm/lavingal/blender-3.2.2-linux-x64/blender"
        RENDER_PY = "/home/stud/lavingal/storage/slurm/lavingal/objaverse-rendering/scripts/blender_script.py"
        XVFB = "xvfb-run --auto-servernum --server-args='-screen 0 1280x720x24'"
        # Perform some operation on the item
        print(item, gpu)
        command = (
        f"{XVFB} {BLENDER} -b "     
        f"--python {RENDER_PY} -- --object_path {item} "
        f"--output_dir {args.output_dir}")
        subprocess.run(command, shell=True, env=env)

        if args.upload_to_s3:
            if item.startswith("http"):
                uid = item.split("/")[-1].split(".")[0]
                for f in glob.glob(f"views/{uid}/*"):
                    s3.upload_file(
                        f, "objaverse-images", f"{uid}/{f.split('/')[-1]}"
                    )
            # remove the views/uid directory
            shutil.rmtree(f"views/{uid}")

        with count.get_lock():
            count.value += 1

        queue.task_done()


if __name__ == "__main__":
    args = tyro.cli(Args)

    s3 = boto3.client("s3") if args.upload_to_s3 else None
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    if args.log_to_wandb:
        wandb.init(project="objaverse-rendering", entity="prior-ai2")

    # Start worker processes on each of the GPUs
    for gpu_i in range(args.num_gpus):
        for worker_i in range(args.workers_per_gpu):
            worker_i = gpu_i * args.workers_per_gpu + worker_i
            process = multiprocessing.Process(
                target=worker, args=(queue, count, gpu_i, s3)
            )
            process.daemon = True
            process.start()

    # Add items to the queue
    with open(args.input_models_path, "r") as f:
        model_paths = json.load(f)
    for item in model_paths:
        queue.put(item)

    # update the wandb count
    if args.log_to_wandb:
        while True:
            time.sleep(5)
            wandb.log(
                {
                    "count": count.value,
                    "total": len(model_paths),
                    "progress": count.value / len(model_paths),
                }
            )
            if count.value == len(model_paths):
                break

    # Wait for all tasks to be completed
    queue.join()

    # Add sentinels to the queue to stop the worker processes
    for i in range(args.num_gpus * args.workers_per_gpu):
        queue.put(None)
