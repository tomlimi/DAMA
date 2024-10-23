import json
import os
import sys
import numpy as np
import torch
import transformers
import random

CS_PROMPTS = [  {"en": "This is that <en_profession>.", "tgt": "To je", "completions": ["ten","ta"]},
                {"en": "There is that <en_profession>.", "tgt": "Tam je", "completions": ["ten","ta"]},
                {"en": "That <en_profession> is not working today.", "tgt": "", "completions": ["Ten", "Ta"]},
                {"en": "That <en_profession> was fired.", "tgt": "", "completions": ["Ten", "Ta"]}
             ]

DE_PROMPTS = [  {"en": "This is the <en_profession>.", "tgt": "Das ist", "completions": ["der","die"]},
                {"en": "There is the <en_profession>.", "tgt": "Da ist", "completions": ["der","die"]},
                {"en": "The <en_profession> is not working today.", "tgt": "", "completions": ["Der", "Die"]},
                {"en": "The <en_profession> was fired.", "tgt": "", "completions": ["Der", "Die"]}
             ]

PROFESSION_SPLITS = {"train": {'nurse', 'secretary', 'cook', 'client', 'someone', 'dispatcher', 'educator',
                               'psychologist', 'nutritionist', 'pedestrian', 'broker', 'physician', 'developer', 'baker','planner',
                               'auditor', 'appraiser', 'paralegal', 'mover', 'driver', 'farmer', 'salesperson', 'librarian', 'cashier',
                               'cleaner', 'clerk', 'worker', 'counselor', 'student', 'veterinarian', 'undergraduate', 'investigator',
                               'programmer', 'accountant', 'hygienist', 'lawyer', 'chef', 'chief', 'pharmacist', 'protester', 'carpenter',
                               'firefighter', 'hairdresser', 'child', 'attendant', 'owner', 'employee', 'guest', 'supervisor', 'witness',
                               'administrator', 'examiner', 'surgeon', 'specialist', 'bystander', 'engineer', 'inspector', 'architect',
                               'onlooker', 'pathologist', 'sheriff', 'guard'},
                     "dev": {'housekeeper', 'assistant', 'victim', 'passenger', 'teacher', 'designer', 'advisee',
                             'practitioner', 'instructor', 'technician', 'writer', 'manager', 'paramedic', 'bartender',
                             'tailor', 'scientist', 'CEO', 'doctor', 'janitor', 'machinist', 'laborer'},
                     "test": {'receptionist', 'customer', 'therapist', 'dietitian', 'patient', 'editor', 'teenager',
                              'homeowner', 'advisor', 'buyer', 'visitor', 'resident', 'chemist', 'officer', 'analyst',
                              'painter', 'mechanic', 'construction worker', 'electrician', 'taxpayer', 'plumber'}}



def generate_dataset(professions, prompt_set, output_filename):
    output = []
    for prompt in prompt_set:
        for profession in professions:
            src_sentence = prompt['en'].replace('<en_profession>', profession)
            item = {"prompt": prompt['tgt'],
                    "completions": prompt['completions'],
                    "src_sentence": src_sentence,
                    "tgt_prefix": "",
                    "subject": profession
                   }
            output.append(item)

    out_file = open(output_filename, "w")
    json.dump(output, out_file, indent = 4, ensure_ascii=False)
    out_file.close()
    print(len(output), "prompts were generated into", output_filename)
    
generate_dataset(list(PROFESSION_SPLITS["train"]), DE_PROMPTS, "de_train_thisis.json")
generate_dataset(list(PROFESSION_SPLITS["dev"]) + list(PROFESSION_SPLITS["test"]), DE_PROMPTS, "de_devtest_thisis.json")
generate_dataset(list(PROFESSION_SPLITS["train"]), CS_PROMPTS, "cs_train_thisis.json")
generate_dataset(list(PROFESSION_SPLITS["dev"]) + list(PROFESSION_SPLITS["test"]), CS_PROMPTS, "cs_devtest_thisis.json")

