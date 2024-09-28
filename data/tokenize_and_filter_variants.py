import json
import os
import sys
import numpy as np
import torch
import transformers
import random

#PROMPT_PREFIX = {"cs": "Jsem", "en": "I am a", "de": "Ich bin"}
PROMPT_PREFIX = {"cs": ["Jsem", "Jsi", "Jste", "Budu", "Budeš", "Budete", "Pracuji jako", "Pracuješ jako", "Pracujete jako"],
                 "en": ["I am", "You are", "You are", "I will be", "You will be", "You will be", "I work as", "You work as", "You work as"],
                 "de": ["Ich bin", "Du bist", "Sie sind", "Ich werde", "Du wirst", "Sie werden", "Ich arbeite als", "Du arbeitest als", "Sie arbeiten als"]}


# load json
language = sys.argv[1]
f = open(language+"_variants.json", "r")
data = json.load(f) 
f.close() 

if sys.argv[2] == "llama2":
    MODEL = "/lnet/express/work/people/limisiewicz/hf_llama/llama_7B"
elif sys.argv[2] == "llama3":
    MODEL = "meta-llama/Meta-Llama-3-8B"

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, use_fast=True, return_token_type_ids=False, add_bos_token=False)
#tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True, return_token_type_ids=False, add_bos_token=False)
#tokenizer = transformers.AutoTokenizer.from_pretrained("/lnet/express/work/people/limisiewicz/hf_llama/llama_7B", use_fast=True, return_token_type_ids=False, add_bos_token=False)

# set llama special tokens
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "</s>"
tokenizer.unk_token = "<unk>"
tokenizer.padding_side = "right"

print("Tokenizer loaded.")

professions = []
for pg in data.keys():
    profession, gender = pg.split("-")
    if gender == "male":
        professions.append(profession)
professions.sort()

output = []

clusters = {}

for prof in professions:
    
    a_an = "a"
    if prof[0] in ['a', 'e', 'i', 'o']:
        a_an = "an"
    
    print("Processing profession", prof)

    male_list = data[prof+"-male"]
    female_list = data[prof+"-female"]
    count = min(len(male_list), len(female_list))
    for i in range(count):
        male_tok = tokenizer.encode(male_list[i])[1:]
        female_tok = tokenizer.encode(female_list[i])[1:]
        male_len = len(male_tok)
        female_len = len(female_tok)
        j = 0
        while (j < male_len and j < female_len and male_tok[j] == female_tok[j]):
            j += 1
        common_prefix = tokenizer.decode(male_tok[:j])
        male_suffix = "."
        if j < male_len:
            male_suffix = tokenizer.decode(male_tok[j])
        female_suffix = "."
        if j < female_len:
            female_suffix = tokenizer.decode(female_tok[j])
        if len(common_prefix) < 3:
            print("Prefix too short", common_prefix)
        elif female_suffix != male_suffix:
            comp = male_suffix + " " + female_suffix
            if comp in clusters:
                clusters[comp].append(common_prefix)
            else:
                clusters[comp] = [common_prefix]
            for variant in range(9):
                item = {"prompt": PROMPT_PREFIX[language][variant] + " " + common_prefix, 
                        "completions": [male_suffix, female_suffix],
                        "src_sentence": PROMPT_PREFIX["en"][variant] + " " + a_an + " " + prof + ".",
                        "tgt_prefix": common_prefix,
                        "subject": prof
                       }
                output.append(item)
# merge clusters containing only one prefix into 'singletons'
singletons = []
comps_to_delete = []
for c in clusters:
    if len(clusters[c]) == 1:
        singletons.append(clusters[c][0])
        comps_to_delete.append(c)
clusters["singletons"] = singletons
for c in comps_to_delete:
    del clusters[c]
# shuffle clusters #variants times
for c in clusters:
    s = clusters[c].copy()
    shuffled = [''] * 9*len(clusters[c])
    permutation_count = 0
    limit = 100
    while limit > 0 and permutation_count < 9:
        random.shuffle(s)
        for i in range(len(s)):
            if s[i] == clusters[c][i]:
                break
            if i == len(s) - 1:
                for j, x in enumerate(s):
                    shuffled[(len(s)-j-1)*9 + permutation_count] = x
                permutation_count += 1
        limit -= 1
    clusters[c] = shuffled

for i in range(len(output)):
    comp = output[i]["completions"][0] + " " + output[i]["completions"][1]
    if comp in clusters.keys():
        output[i]["shuffled_prefix"] = clusters[comp].pop()
    else:
        output[i]["shuffled_prefix"] = clusters["singletons"].pop()


out_file = open(language+"_train_"+sys.argv[2]+".json", "w")
json.dump(output, out_file, indent = 4, ensure_ascii=False)
out_file.close()
print(len(output), "items was generated.")

