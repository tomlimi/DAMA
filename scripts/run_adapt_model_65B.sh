#!/bin/bash
#SBATCH -J run-adapt-65B-nd32
#SBATCH -q high
#SBATCH -p gpu-troja,gpu-ms
#SBATCH -o /home/limisiewicz/my-luster/dama/job_output/run-adapt-65B.out
#SBATCH -D /home/limisiewicz/my-luster/dama/DAMA/src
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:7
#SBATCH --mem=128G
#SBATCH --constraint="gpuram40G|gpuram48G"


# This script for running DAMA


source /home/limisiewicz/my-luster/GenderBiasGACR/.virtualenv/bin/activate

python adapt_model.py --param_number 65 --method "DAMA" --request_file "train_dama.json" --num_layers 28 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 32 --no_colinear_vs True --use_neutral True --vs_at_last True --delta_only True --no_whitening True --load_projections True
