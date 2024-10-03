#!/bin/bash
set -e

export FAST_ALIGN_BASE=~/troja/gender-bias/fast_align/

INPUT_TRANSLATIONS_PREFIX=/home/limisiewicz/my-luster/dama/results/original/almar_13B/partial_res_translation
OUTPUT_TRANSLATIONS_PREFIX=/home/marecek/troja/gender-bias/mt_gender/translations/almar_13B

LANGUAGES=("cs" "de" "ru")
DATASETS=("mt-gender" "bug")

for language in ${LANGUAGES[@]}
do
    for dataset in ${DATASETS[@]}
    do
        trans_fn=${OUTPUT_TRANSLATIONS_PREFIX}/en-${language}.txt

        # Convert translation outputs
        python json2interlaced.py ${INPUT_TRANSLATIONS_PREFIX}_${dataset}_${language}.json $trans_fn

        # Align
        align_fn=forward.en-${language}.align
        $FAST_ALIGN_BASE/build/fast_align -i $trans_fn -d -o -v > $align_fn

        # Evaluate
        out_folder=../output/almar_13B
        mkdir -p $out_folder
        
        python ../src/load_alignments.py --ds=../data/${dataset}/en.txt --bi=$trans_fn --align=$align_fn --lang=$language --out=$out_folder/$dataset-$language-all.pred.csv >> $out_folder/$dataset-$language-all.log
        python ../src/load_alignments.py --ds=../data/${dataset}/en_pro.txt --bi=$trans_fn --align=$align_fn --lang=$language --out=$out_folder/$dataset-$language-pro.pred.csv >> $out_folder/$dataset-$language-pro.log
        python ../src/load_alignments.py --ds=../data/${dataset}/en_anti.txt --bi=$trans_fn --align=$align_fn --lang=$language --out=$out_folder/$dataset-$language-anti.pred.csv >> $out_folder/$dataset-$language-anti.log
    done
done
