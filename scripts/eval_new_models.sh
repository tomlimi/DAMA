#!/bin/bash
#SBATCH -J eval-almar
#SBATCH -q high
#SBATCH -p gpu-troja,gpu-ms
#SBATCH -o /home/limisiewicz/my-luster/dama/job_output/eval-almar.out
#SBATCH -D /home/limisiewicz/my-luster/dama/DAMA/src
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu
#SBATCH --mem=63G
#SBATCH --constraint="gpuram40G|gpuram48G"

source /home/limisiewicz/my-luster/GenderBiasGACR/.virtualenv/bin/activate

#model_orig_name="meta-llama/Llama-2-13b-hf"
model_orig_name="haoranxu/ALMA-13B-R"

ds_split="test"
ds_prefix=("anti_type1" "pro_type1" "anti_type2" "pro_type2")

# Eval original model
python evaluate_model.py --model_name ${model_orig_name} --test_file wikitext_wikitext-103-raw-v1 --test_task "causal_lm"
python evaluate_model.py --model_name ${model_orig_name} --test_file ${ds_split}_dama.json --test_task "gen"

for ds in "${ds_prefix[@]}"
do
  python evaluate_model.py --model_name ${model_orig_name} --test_file "${ds}_${ds_split}.txt" --test_task "coref"
done

python evaluate_model.py --model_name ${model_orig_name}  --param_number 13 --test_file ARC-Challenge-Test.jsonl --test_task qa
python evaluate_model.py --model_name ${model_orig_name}  --param_number 13 --test_file ARC-Easy-Test.jsonl --test_task qa
python evaluate_model.py --model_name ${model_orig_name}  --param_number 13 --test_file stereoset_dev.json --test_task stereoset

# Eval DAMA_L
for nl in 11 12 7 9 14
do
  python evaluate_model.py --model_name ${model_orig_name} --num_layers ${nl} --param_number 13 --method "DAMA_L" --test_file wikitext_wikitext-103-raw-v1 --test_task "causal_lm"
  python evaluate_model.py --model_name ${model_orig_name} --num_layers ${nl} --param_number 13 --method "DAMA_L" --test_file ${ds_split}_dama.json --test_task "gen"

  for ds in "${ds_prefix[@]}"
  do
    python evaluate_model.py --model_name ${model_orig_name} --num_layers ${nl} --param_number 13 --method "DAMA_L" --test_file "${ds}_${ds_split}.txt" --test_task "coref"
  done
  python evaluate_model.py --model_name ${model_orig_name} --num_layers ${nl} --param_number 13 --method "DAMA_L" --test_file ARC-Challenge-Test.jsonl --test_task qa
  python evaluate_model.py --model_name ${model_orig_name} --num_layers ${nl} --param_number 13 --method "DAMA_L" --test_file ARC-Easy-Test.jsonl --test_task qa
  python evaluate_model.py --model_name ${model_orig_name} --num_layers ${nl} --param_number 13 --method "DAMA_L" --test_file stereoset_dev.json --test_task stereoset
done


