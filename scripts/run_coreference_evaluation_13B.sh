#!/bin/bash
#SBATCH -J evaluate_coreference_13B
#SBATCH -q high
#SBATCH -p gpu-troja,gpu-ms
#SBATCH -o /home/limisiewicz/my-luster/dama/job_output/evaluate_coreference_13B.out
#SBATCH -D /home/limisiewicz/my-luster/dama/DAMA/src
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --constraint="gpuram48G|gpuram40G"


# This script runs the evaluation of the DAMA system.


source /home/limisiewicz/my-luster/GenderBiasGACR/.virtualenv/bin/activate

num_layers=(8 14)
dimensionalities=(32 256)
ds_prefix=("anti_type1" "pro_type1" "anti_type2" "pro_type2")
ds_split="test"

for ds in "${ds_prefix[@]}"
do
  echo "Running on ${ds}"
  for dim in "${dimensionalities[@]}"
  do
    for nl in "${num_layers[@]}"
    do
      python evaluate_model.py --param_number 13 --method "DAMA" --test_file "${ds}_${ds_split}.txt" --test_task "coref" --num_layers ${nl} --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim ${dim} --no_colinear_vs True --vs_at_last True --use_neutral True --delta_only True --no_whitening True
    done
  done
done