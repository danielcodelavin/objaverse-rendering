#!/bin/bash
#SBATCH --job-name=1_Render_Objaverse_PLUS
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --nodelist=node10
#SBATCH --gres=gpu:8
#SBATCH --time=12-00:00:00
#SBATCH --output=/home/stud/lavingal/storage/slurm/lavingal/run_reports/slurm-%j.out
#SBATCH --error=/home/stud/lavingal/storage/slurm/lavingal/run_reports/slurm-%j.out

export SSL_CERT_FILE=$(/storage/slurm/lavingal/lavingal/blender-3.2.2-linux-x64/3.2/python/bin/python3.10 -m certifi)


/home/wiss/schnaus/mambaforge/envs/daniel_lavin/bin/python distributed.py \
    --num_gpus 8 \
    --workers_per_gpu 1 \
    --input_models_path "1_real_truncated_input_models_path.json"
