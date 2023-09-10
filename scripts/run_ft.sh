#!/bin/bash
#SBATCH -J run-ft
#SBATCH -q high
#SBATCH -p gpu-troja,gpu-ms
#SBATCH -o /home/limisiewicz/my-luster/dama/job_output/run-ft.out
#SBATCH -D /home/limisiewicz/my-luster/dama/DAMA/src
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu
#SBATCH --mem=63G
#SBATCH --constraint="gpuram40G|gpuram48G"


# This script for running DAMA


source /home/limisiewicz/my-luster/GenderBiasGACR/.virtualenv/bin/activate

python adapt_model.py --param_number 7 --method "FT" --request_file "train_dama.json"