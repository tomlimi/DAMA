import json
import os
import sys


# load json
f = open("train_dama.json", "r")
data = json.load(f) 
f.close()

for i in range(len(data)):
    del(data[i]["gender_score"])
    del(data[i]["target_new"])
    data[i]["completions"] = ["he", "she"]

out_file = open("en_train.json", "w")
json.dump(data, out_file, indent = 4, ensure_ascii=False) 
out_file.close()
