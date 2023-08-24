import os
import json

import numpy as np
from scipy.stats import spearmanr, pearsonr, linregress
from tqdm import tqdm

from evaluation.evaluate import Evaluate
from utils.globals import *


class EvaluateGeneration(Evaluate):

    PROFESSION_DATA = os.path.join(DATA_DIR, "professions.json")

    def __init__(self, model, tok, test_file, task):
        super().__init__(model, tok, test_file, task)

        assert self.task == "gen", f"Task class mismatch:, expected 'gen', got '{self.task}' instead"

        self.results = {"spearman_s": 0., "spearman_f": 0.,
                        "pearson_s": 0., "pearson_f": 0.,
                        "slope_s": 0., "intercept_s": 0., "r2_s": 0.,
                        "slope_f": 0., "intercept_f": 0., "r2_f": 0.,
                        "prob_he": 0., "prob_she": 0., "prob_they": 0.,
                        "predicted_he": 0., "predicted_she": 0., "predicted_they": 0.}

        self.partial_results = {"s_score": [], "f_score": [], 'empirical_score': [],
                                "prob_he": [], "prob_she": [], "prob_they": [],
                                "predicted_token": [], "prompt": []}

        self.load_data()

    def load_data(self):

        self.generation_prompts = None
        self.profession_data = None

        with open(self.test_file, 'r') as f:
            self.generation_prompts = json.load(f)

        if len(self.generation_prompts) >= 1 and type(self.generation_prompts[0]) == dict:
            self.generation_prompts = [prompt["prompt"] for prompt in self.generation_prompts]

        with open(EvaluateGeneration.PROFESSION_DATA , 'r') as f:
            profession_data_raw = json.load(f)

        self.profession_data = {prof: {'factual': f, 'stereotypical': s} for (prof, f, s)
                                in profession_data_raw}

    def evaluate(self):

        he_token_id = self.tok.encode("he")[0]
        she_token_id = self.tok.encode("she")[0]
        they_token_id = self.tok.encode("they")[0]

        for prompt in tqdm(self.generation_prompts, desc="Evaluating generation prompts"):
            # raw way to get subject
            subject = ""
            for i in range(2,len(prompt.split(" "))):
                subject = "_".join(prompt.split(" ")[1:i])
                if subject in self.profession_data:
                    break
                    
            s_score = self.profession_data[subject]['stereotypical']
            f_score = self.profession_data[subject]['factual']

            probabilities = self.get_prediction_probability(prompt)

            prob_he = probabilities[he_token_id]
            prob_she = probabilities[she_token_id]
            prob_they = probabilities[they_token_id]
            predicted_token = self.tok.decode([probabilities.index(max(probabilities))])

            self.partial_results["s_score"].append(s_score)
            self.partial_results["f_score"].append(f_score)
            self.partial_results["prob_he"].append(prob_he)
            self.partial_results["prob_she"].append(prob_she)
            self.partial_results["prob_they"].append(prob_they)
            self.partial_results["predicted_token"].append(predicted_token)
            self.partial_results["prompt"].append(prompt)

        empirical_score = list(np.array(self.partial_results["prob_he"]) - np.array(self.partial_results["prob_she"]))
        self.partial_results["empirical_score"] = empirical_score

        self.results["spearman_s"] = spearmanr(self.partial_results["s_score"], empirical_score)[0]
        self.results["spearman_f"] = spearmanr(self.partial_results["f_score"], empirical_score)[0]

        self.results["pearson_s"] = pearsonr(self.partial_results["s_score"], empirical_score)[0]
        self.results["pearson_f"] = pearsonr(self.partial_results["f_score"], empirical_score)[0]

        self.results["slope_s"], self.results["intercept_s"], self.results["r2_s"], _, _ = \
            linregress(self.partial_results["s_score"], empirical_score)

        self.results["slope_f"], self.results["intercept_f"], self.results["r2_f"], _, _ = \
            linregress(self.partial_results["f_score"], empirical_score)

        self.results["prob_he"] = np.mean(self.partial_results["prob_he"])
        self.results["prob_she"] = np.mean(self.partial_results["prob_she"])
        self.results["prob_they"] = np.mean(self.partial_results["prob_they"])

        self.results["predicted_he"] = self.partial_results["predicted_token"].count("he") / len(self.generation_prompts)
        self.results["predicted_she"] = self.partial_results["predicted_token"].count("she") / len(self.generation_prompts)
        self.results["predicted_they"] = self.partial_results["predicted_token"].count("they") / len(self.generation_prompts)

        # partial result change into dict of lists
        self.partial_results = [dict(zip(self.partial_results, t)) for t in zip(*self.partial_results.values())]