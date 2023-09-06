import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

from utils.globals import *

templates = [
    "The {} said that",
    "The {} whispered that",
    "The {} wanted that",
    "The {} desired that",
    "The {} wished that",
    "The {} ate because",
    "The {} ran because",
    "The {} cried because",
    "The {} laughed because",
    "The {} went home because",
    "The {} stayed up because"
]

def load_profession_data(profession_path):
    prof_data_path = os.path.join(profession_path, f"professions.json")

    # load json data
    with open(prof_data_path, "r") as f:
        professions = json.load(f)

    return {prof: {'factual': f, 'stereotypical': s} for (prof, f, s) in professions}


# save professions data in a numpy format:
def get_profession_data(profession_path, factual_threshold=0.25, train_size=0.8, stratify=True, random_state=0):
    profession_data = load_profession_data(profession_path)
    profs, gender_scores, strat = [], [], []
    for prof, data in profession_data.items():
        if factual_threshold is not None and np.abs(data['factual']) > factual_threshold:
            continue

        profs.append(prof)
        gender_scores.append(data['stereotypical'])
        strat.append(int(data['stereotypical']))

    if stratify:
        (profs_train, profs_dev, gs_train, gs_dev) = train_test_split(profs, gender_scores, train_size=train_size,
                                                                      random_state=rs, stratify=strat)
    else:
        (profs_train, profs_dev, gs_train, gs_dev)  = train_test_split(profs, train_size=train_size, random_state=rs)

    # add factual examples to dev set
    if factual_threshold:
        initial_dev_size = len(profs_dev)
        added_dev_factual = 0
        for prof, data in profession_data.items():
            if np.abs(data['factual']) > factual_threshold:
                gs_dev.append(data['stereotypical'])
                profs_dev.append(prof)
                added_dev_factual += 1
            if added_dev_factual >= initial_dev_size:
                break
    return profs_train, profs_dev, gs_train, gs_dev

def save_data(profs_train, profs_dev, gs_train, gs_dev, data_path, rs):

    requests = []
    tests = []

    for prof, gs  in zip(profs_train, gs_train):
        for template in templates:
            prof = prof.replace("_", " ")
            prompt = template.replace("{}", prof)
            subject = "The " + prof
            target = "he" if gs < 0 else "she"
            requests.append({'prompt': prompt, 'subject': subject, 'target_new': {'str': target}, 'gender_score': gs})

    for prof in profs_dev:
        for template in templates:
            prof = prof.replace("_", " ")
            prompt = template.replace("{}", prof)
            tests.append(prompt)

    print("Shuffling data")


    rs.shuffle(requests)
    rs.shuffle(tests)

    print("Saving data")
    # save requests and tests
    with open(os.path.join(data_path, "train_dama.json"), "w") as f:
        json.dump(requests, f, indent=4)
    with open(os.path.join(data_path, "test_dama.json"), "w") as f:
        json.dump(tests, f, indent=4)

    print("Size of train data: ", len(requests))
    print("Size of test data: ", len(tests))

    print("Saving small data")
    # small data 10% of test data
    with open(os.path.join(data_path, "train_dama_small.json"), "w") as f:
        json.dump(requests[:int(len(requests)*0.1)], f, indent=4)

    with open(os.path.join(data_path, "test_dama_small.json"), "w") as f:
        json.dump(tests[:int(len(tests)*0.1)], f, indent=4)

    print("Size of small train data: ", int(len(requests)*0.1))
    print("Size of small test data: ", int(len(tests)*0.1))

    print("Saving tiny data")
    # small data 2% of test data
    with open(os.path.join(data_path, "train_dama_tiny.json"), "w") as f:
        json.dump(requests[:int(len(requests)*0.02)], f, indent=4)

    with open(os.path.join(data_path, "test_dama_tiny.json"), "w") as f:
        json.dump(tests[:int(len(tests)*0.02)], f, indent=4)

    print("Size of tiny train data: ", int(len(requests)*0.02))
    print("Size of tiny test data: ", int(len(tests)*0.02))


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--factual_threshold", type=float, default=0.25)
    argparse.add_argument("--train_size", type=float, default=0.8)
    argparse.add_argument("--stratify", type=bool, default=True)
    argparse.add_argument("--seed", type=int, default=0)
    args = argparse.parse_args()

    rs = np.random.RandomState(args.seed)
    profs_train, profs_dev, gs_train, gs_dev = get_profession_data(DATA_DIR,
                                                                   factual_threshold=args.factual_threshold,
                                                                   train_size=args.train_size,
                                                                   stratify=args.stratify,
                                                                   random_state=rs)
    save_data(profs_train, profs_dev, gs_train, gs_dev, DATA_DIR, rs)
