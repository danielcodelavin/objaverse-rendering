#!/bin/bash
#SBATCH --job-name="Render_Objaverse_PLUS"
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --mail-type=ALL
#SBATCH --nodelist=node10
#SBATCH --gres=gpu:4,VRAM:12G
#SBATCH --time=12-00:00:00
#SBATCH --output=/home/stud/lavingal/storage/slurm/lavingal/run_reports/slurm-%j.out
#SBATCH --error=/home/stud/lavingal/storage/slurm/lavingal/run_reports/slurm-%j.out
export SSL_CERT_FILE=$(/home/stud/lavingal/storage/slurm/lavingal/blender-3.2.2-linux-x64/3.2/python/bin/python3.10 -m certifi)
python3 distributed.py --num_gpus 4 --workers_per_gpu 2


