#!/bin/bash
#SBATCH -J evaluate_coreference_7B
#SBATCH -q high
#SBATCH -p gpu-troja,gpu-ms
#SBATCH -o /home/limisiewicz/my-luster/dama/job_output/evaluate_coreference_7B.out
#SBATCH -D /home/limisiewicz/my-luster/dama/DAMA/src
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu
#SBATCH --mem=64G
#SBATCH --constraint="gpuram48G|gpuram40G|gpuram24G"


# This script runs the evaluation of the DAMA system.


source /home/limisiewicz/my-luster/GenderBiasGACR/.virtualenv/bin/activate

num_layers=(1 3 8 10 )
dimensionalities=(32 64 128 256 512 1024 )
ds_prefix=("anti_type2" )
ds_split="test"

for ds in "${ds_prefix[@]}"
do
  echo "Running on ${ds}"
  for dim in "${dimensionalities[@]}"
  do
    for nl in "${num_layers[@]}"
    do
      python evaluate_model.py --param_number 7 --method "DAMA" --test_file "${ds}_${ds_split}.txt" --test_task "coref" --num_layers ${nl} --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim ${dim} --no_colinear_vs True --vs_at_last True --use_neutral True --delta_only True --no_whitening True
    done
  done
done