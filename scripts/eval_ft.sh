#!/bin/bash
#SBATCH -J eval-ft
#SBATCH -q high
#SBATCH -p gpu-troja,gpu-ms
#SBATCH -o /home/limisiewicz/my-luster/dama/job_output/eval-ft.out
#SBATCH -D /home/limisiewicz/my-luster/dama/DAMA/src
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu
#SBATCH --mem=63G
#SBATCH --constraint="gpuram40G|gpuram48G"

source /home/limisiewicz/my-luster/GenderBiasGACR/.virtualenv/bin/activate

ds_split="test"
ds_prefix=("anti_type1" "pro_type1" "anti_type2" "pro_type2")

python evaluate_model.py --param_number 7 --method "FT" --test_file wikitext_wikitext-103-raw-v1 --test_task "causal_lm"
python evaluate_model.py --param_number 7 --method "FT" --test_file ${ds_split}_dama.json --test_task "gen"

for ds in "${ds_prefix[@]}"
do
  python evaluate_model.py --param_number 7 --method "FT" --test_file "${ds}_${ds_split}.txt" --test_task "coref"
done