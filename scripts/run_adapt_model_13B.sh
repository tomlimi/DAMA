#!/bin/bash
#SBATCH -J run-adapt-13B-nd2048
#SBATCH -q high
#SBATCH -p gpu-troja,gpu-ms
#SBATCH -o /home/limisiewicz/my-luster/dama/job_output/run-adapt-13B-nd2048.out
#SBATCH -e /home/limisiewicz/my-luster/dama/job_output/run-adapt-13B-nd2048.out
#SBATCH -D /home/limisiewicz/my-luster/dama/DAMA/src
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem=63G
#SBATCH --constraint="gpuram40G|gpuram48G"


# This script for running DAMA


source /home/limisiewicz/my-luster/GenderBiasGACR/.virtualenv/bin/activate

python adapt_model.py --param_number 13 --method "DAMA" --request_file "train_dama.json" --num_layers 14 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 2048 --no_colinear_vs True --use_neutral True --vs_at_last True --delta_only True --no_whitening True --load_projections True
