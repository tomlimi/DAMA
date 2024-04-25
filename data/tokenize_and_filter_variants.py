import json
import os
import sys
import numpy as np
import torch
import transformers


# load json
filename = sys.argv[1]
f = open(filename, "r")
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
    professions.append(profession)
professions.sort()

for p in professions:
    print("Processing profession", p)
    male_list = data[p+"-male"]
    female_list = data[p+"-female"]
    count = min(len(male_list), len(female_list))
    for i in range(count):
        male_tok = tokenizer.encode(male_list[i])
        female_tok = tokenizer.encode(female_list[i])
        p = 0
        while (p < len(male_tok) and p < len(female_tok) and male_tok[p] == female_tok[p]):
            p += 1
        if p > 0 and (p < len(male_tok) or p < len(female_tok)):
            print(tokenizer.decode(male_tok[:p]), tokenizer.decode(male_tok[p:]))
            print(tokenizer.decode(female_tok[:p]), tokenizer.decode(female_tok[p:]))
            print()
