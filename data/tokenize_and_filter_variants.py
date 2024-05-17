import json
import os
import sys
import numpy as np
import torch
import transformers

PROMPT_PREFIX = {"cs": "Jsem", "en": "I am a", "de": "Ich bin"}

# load json
language = sys.argv[1]
f = open(language+"_variants.json", "r")
data = json.load(f) 
f.close() 

tokenizer = transformers.AutoTokenizer.from_pretrained("/lnet/express/work/people/limisiewicz/hf_llama/llama_7B", use_fast=True, return_token_type_ids=False, add_bos_token=False)

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

for prof in professions:
    
    src_prompt_prefix = PROMPT_PREFIX["en"]
    if prof[0] in ['a', 'e', 'i', 'o']:
        src_prompt_prefix += 'n'
    prompt_prefix = PROMPT_PREFIX[language]
    
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
        if len(common_prefix) >= 3 and female_suffix != male_suffix and j >= female_len - 1 and j >= male_len - 1:  
        #if len(common_prefix) >= 3 and female_suffix != male_suffix:  
                item = {"prompt": prompt_prefix + " " + tokenizer.decode(male_tok[:j]), 
                        "completions": [male_suffix, female_suffix],
                        "src_sentence": src_prompt_prefix + " " + prof + ".",
                        "subject": prof
                       }
                output.append(item)

out_file = open(language+"_train.json", "w")
json.dump(output, out_file, indent = 4, ensure_ascii=False) 
out_file.close()
print(len(output), "items was generated.")
