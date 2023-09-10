#!/bin/bash
#SBATCH -J run-adapt-7B-nd128-nw-s45
#SBATCH -q high
#SBATCH -p gpu-troja,gpu-ms
#SBATCH -o /home/limisiewicz/my-luster/dama/job_output/run-adapt-nd128-nw-s45.out
#SBATCH -D /home/limisiewicz/my-luster/dama/DAMA/src
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu
#SBATCH --mem=63G
#SBATCH --constraint="gpuram40G|gpuram48G"


# This script for running DAMA


source /home/limisiewicz/my-luster/GenderBiasGACR/.virtualenv/bin/activate

python adapt_model.py --param_number 7 --method "DAMA" --request_file "train_dama.json" --num_layers 11  --post_linear True --mixed_update True --null_dim 128 --no_colinear_vs True --use_neutral True --vs_at_last True  --delta_only True --no_whitening True  --random_seed 45

