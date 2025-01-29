import json
import os
import sys
import random

CS_PROMPTS = [  {"en": "This is that <en_profession>.", "tgt": "To je", "completions": [" ten"," ta"]},
                {"en": "There is that <en_profession>.", "tgt": "Tam je", "completions": [" ten"," ta"]},
                {"en": "That <en_profession> is not working today.", "tgt": "", "completions": [" Ten", " Ta"]},
                {"en": "That <en_profession> was fired.", "tgt": "", "completions": [" Ten", " Ta"]},
                {"en": "That <en_profession> is busy.", "tgt": "", "completions": [" Ten", " Ta"]},
                {"en": "I was a <en_profession> two years ago.", "tgt": "Před dvěma lety jsem", "completions": [" byl", " byla"]},
                {"en": "You were a <en_profession> two years ago.", "tgt": "Před dvěma lety jste", "completions": [" byl", " byla"]},
                {"en": "If only I were a <en_profession>.", "tgt": "Kdybych tak", "completions": [" byl", " byla"]},
                {"en": "I was a <en_profession> at that time.", "tgt": " V té době jsem", "completions": [" byl", " byla"]},
                {"en": "You were a <en_profession> at that time.", "tgt": "V té době jsi", "completions": [" byl", " byla"]},
                {"en": "You were a <en_profession> at that time.", "tgt": "V té době jste", "completions": [" byl", " byla"]}
                #{"en": "This is the new <en_profession>.", "tgt": "To je nov", "completions": ["ý", "á"]},
                #{"en": "This is the young <en_profession>.", "tgt": "To je mlad", "completions": ["ý", "á"]},
                #{"en": "This is the good <en_profession>.", "tgt": "To je dobr", "completions": ["ý", "á"]},
                #{"en": "This is the experienced <en_profession>.", "tgt": "To je zkušen", "completions": ["ý", "á"]},
                #{"en": "We found a good <en_profession>.", "tgt": "Našli jsme dobr", "completions": ["ého", "ou"]},
                #{"en": "I met a young <en_profession>.", "tgt": "Potkal jsem mlad", "completions": ["ého", "ou"]},
                #{"en": "I asked the new <en_profession>.", "tgt": "Zeptal jsem se nov", "completions": ["ého", "é"]},
                #{"en": "We answered the experienced <en_profession>.", "tgt": "Odpověděli jsme zkušen", "completions": ["ému", "é"]}
             ]

DE_PROMPTS = [  {"en": "This is the <en_profession>.", "tgt": "Das ist", "completions": [" der"," die"]},
                {"en": "There is the <en_profession>.", "tgt": "Da ist", "completions": [" der"," die"]},
                {"en": "The <en_profession> is not working today.", "tgt": "", "completions": [" Der", " Die"]},
                {"en": "The <en_profession> was fired.", "tgt": "", "completions": [" Der", " Die"]},
                {"en": "The <en_profession> is busy.", "tgt": "", "completions": [" Der", " Die"]},
                {"en": "Do you know the <en_profession>?", "tgt": "Kennen Sie", "completions": [" den", " die"]},
                {"en": "I was there with the <en_profession>", "tgt": "Ich war dort mit", "completions": [" dem", " der"]},
                {"en": "I asked the <en_profession>.", "tgt": "Ich fragte", "completions": [" den", " die"]},
                {"en": "We met the <en_profession>.", "tgt": "Wir trafen", "completions": [" den", " die"]},
                {"en": "I answered the <en_profession>.", "tgt": "Ich antwortete", "completions": [" dem", " der"]},
                {"en": "The salary of the <en_profession> has increased.", "tgt": "Das Gehalt", "completions": [" des", " der"]}
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
                              'painter', 'mechanic', 'construction worker', 'electrician', 'taxpayer', 'plumber'},
                     "factual": {"man", "boy", "gentleman", "father", "son", "brother", "husband", "king", "prince", "uncle", "nephew", "groom", "duke",
                                 "grandfather", "widower", "hero", "wizard", "baron", "emperor", "anchorman", "clergyman", "chairman", "alumnus",
                                 "woman", "girl", "lady", "mother", "daughter", "sister", "wife", "queen", "princess", "aunt", "niece", "bride", "duchess",
                                 "grandmother", "widow", "heroine", "witch", "baroness", "empress", "anchorwoman", "clergywomen", "chairwomen", "alumna"},
                     "all": {"accountant", "acquaintance", "adjunct professor", "administrator", "adventurer", "advocate", "aide", "alter ego", "ambassador",
                             "anthropologist", "archaeologist", "architect", "artist", "artiste", "assassin", "assistant professor", "associate dean",
                             "associate professor", "astronaut", "astronomer", "athlete", "athletic director", "attorney", "author", "ballplayer", "banker",
                             "biologist", "boss", "broadcaster", "broker", "bureaucrat", "butcher", "cabbie", "campaigner", "captain", "cardiologist",
                             "caretaker", "carpenter", "cartoonist", "chancellor", "chaplain", "character", "chef", "chemist", "choreographer", "cinematographer",
                             "citizen", "civil servant", "clerk", "coach", "collector", "colonel", "columnist", "comedian", "comic", "commander", "commentator",
                             "commissioner", "composer", "conductor", "confesses", "consultant", "correspondent", "councilor", "counselor", "critic", "crusader",
                             "custodian", "dancer", "dentist", "deputy", "dermatologist", "detective", "disc jockey", "doctor", "doctoral student", "drug addict",
                             "drummer", "economics professor", "economist", "editor", "educator", "electrician", "entertainer", "entrepreneur",
                             "environmentalist", "envoy", "epidemiologist", "evangelist", "farmer", "fashion designer", "fighter pilot", "filmmaker", "financier",
                             "firebrand", "firefighter", "freelance writer", "geologist", "graphic designer", "guidance counselor", "guitarist", "hairdresser",
                             "historian", "homemaker", "hooker", "housekeeper", "illustrator", "inspector", "instructor", "interior designer", "inventor",
                             "investigator", "investment banker", "janitor", "jeweler", "journalist", "judge", "jurist", "laborer", "landlord", "lawyer",
                             "lecturer", "legislator", "librarian", "lyricist", "maestro", "magistrate", "major leaguer", "manager", "marshal", "mathematician",
                             "mediator", "medic", "minister", "missionary", "mobster", "musician", "naturalist", "neurologist", "novelist", "nurse", "observer",
                             "officer", "painter", "parliamentarian", "pediatrician", "performer", "pharmacist", "philanthropist", "philosopher",
                             "photojournalist", "physicist", "pianist", "planner", "plastic surgeon", "playwright", "plumber", "pollster", "president",
                             "prisoner", "professor", "professor emeritus", "programmer", "prosecutor", "protagonist", "protege", "protester", "psychiatrist",
                             "psychologist", "pundit", "rabbi", "radiologist", "ranger", "registered nurse", "researcher", "restaurateur", "sailor", "saint",
                             "saxophonist", "scholar", "scientist", "screenwriter", "sculptor", "secretary", "senator", "sergeant", "servant", "sheriff deputy",
                             "shopkeeper", "singer", "skipper", "sociologist", "soft spoken", "solicitor general", "soloist", "sportswriter", "student", "stylist",
                             "substitute", "superintendent", "surgeon", "surveyor", "swimmer", "technician", "teenager", "therapist", "trader", "treasurer",
                             "trooper", "trucker", "trumpeter", "undersecretary", "valedictorian", "vice chancellor", "vocalist", "warden", "worker", "wrestler",
                             "writer"}}

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
    
#generate_dataset(list(PROFESSION_SPLITS["train"]), DE_PROMPTS, "de_train_4cases.json")
#generate_dataset(list(PROFESSION_SPLITS["dev"]) + list(PROFESSION_SPLITS["test"]), DE_PROMPTS, "de_devtest_4cases.json")
generate_dataset(list(PROFESSION_SPLITS["factual"]), DE_PROMPTS, "de_factual_11prompts.json")
generate_dataset(list(PROFESSION_SPLITS["all"]), DE_PROMPTS, "de_all_11prompts.json")
generate_dataset(list(PROFESSION_SPLITS["factual"]), CS_PROMPTS, "cs_factual_11prompts.json")
generate_dataset(list(PROFESSION_SPLITS["all"]), CS_PROMPTS, "cs_all_11prompts.json")

