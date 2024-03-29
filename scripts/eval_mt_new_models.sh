#!/bin/bash
#SBATCH -J eval-mt-almar
#SBATCH -q high
#SBATCH -p gpu-troja,gpu-ms
#SBATCH -o /home/limisiewicz/my-luster/dama/job_output/eval-mt-almar.out
#SBATCH -D /home/limisiewicz/my-luster/dama/DAMA/src
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu
#SBATCH --mem=63G
#SBATCH --constraint="gpuram40G|gpuram48G"

source /home/limisiewicz/my-luster/GenderBiasGACR/.virtualenv/bin/activate

#model_orig_name="meta-llama/Llama-2-13b-hf"
model_orig_name="haoranxu/ALMA-13B-R"
wmt_langs=("cs-en" "de-en" "is-en" "zh-en" "ru-en" "en-cs" "en-de" "en-is" "en-zh" "en-ru")
mt_gender_langs=("de" "cs" "ru")

# Eval original model
for tgt_lang in "${mt_gender_langs[@]}"
do
  test_file="mt-gender_${tgt_lang}"
  python evaluate_model.py --model_name ${model_orig_name} --test_file ${test_file} --test_task "translation"
done


for lang_pair in "${wmt_langs[@]}"
do
  test_file="haoranxu/WMT22-Test_${lang_pair}"
  python evaluate_model.py --model_name ${model_orig_name} --test_file ${test_file} --test_task "translation"
done

# Eval DAMA_L
for nl in 11 12 7 9 14
do
  for tgt_lang in "${mt_gender_langs[@]}"
  do
    test_file="mt-gender_${tgt_lang}"
    python evaluate_model.py --model_name ${model_orig_name} --num_layers ${nl} --param_number 13 --method "DAMA_L" --test_file ${test_file} --test_task "translation"
  done


  for lang_pair in "${wmt_langs[@]}"
  do
    test_file="haoranxu/WMT22-Test_${lang_pair}"
    python evaluate_model.py --model_name ${model_orig_name} --num_layers ${nl} --param_number 13 --method "DAMA_L" --test_file ${test_file} --test_task "translation"
  done

done




