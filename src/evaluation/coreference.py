import os
import json

import numpy as np
from tqdm import tqdm

from evaluation.evaluate import Evaluate


class EvaluateCoreference(Evaluate):

    PRONOUN_MAP = {"he": "m", "his": "m", "him": "m",
                    "she": "f", "her": "f", "hers": "f",
                    "they": "n", "their": "n", "them": "n"}

    def __init__(self, model, tok, test_file, task):
        super().__init__(model, tok, test_file, task)

        assert self.task == "coref", f"Task class mismatch:, expected 'coref', got '{self.task}' instead"

        self.results = {"m_acc": 0., "f_acc": 0., "n_acc": 0., "total_acc": 0.}
        self.partial_results = []

        self.load_data()

    def load_data(self):

        self.test_examples = []

        # Parse the sentence of the form: [The sheriff] told the counselor that [he] would arrive in the afternoon.
        # into tuple: ("sheriff", "he", "The sheriff told the counselor that [he] would arrive in the afternoon.")

        with open(self.test_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                # discard the line number
                line = " ".join(line.split(" ")[1:])

                # Find parenthesis [ ] and return their content
                profession = line.split("[")[1].split("]")[0]
                pronoun = line.split("[")[2].split("]")[0]
                clean_sentence = line.replace(f"[{profession}]", profession).replace(f"[{pronoun}]", pronoun).strip()

                # stripping the article from the profession and getting the first token
                correct_tok = self.tok.decode(self.tok.encode(" ".join(profession.split(" ")[1:]))[0])
                prompt = clean_sentence + f" '{pronoun.capitalize()}' refers to the"

                self.test_examples.append({"correct_tok": correct_tok,
                                           "gender": EvaluateCoreference.PRONOUN_MAP[pronoun],
                                           "prompt": prompt})

    def evaluate(self):

        correct = {"m": 0, "f": 0, "n": 0}
        total = {"m": 0, "f": 0, "n": 0}

        for test_example in tqdm(self.test_examples, "Evaluating coreference"):

            total[test_example["gender"]] += 1

            probabilities = self.get_prediction_probability(test_example["prompt"])
            predicted_tok = self.tok.decode([probabilities.index(max(probabilities))])

            if test_example["correct_tok"] == predicted_tok:
                correct[test_example["gender"]] += 1

            self.partial_results.append({"correct_tok": test_example["correct_tok"],
                                        "predicted_tok": predicted_tok,
                                        "gender": test_example["gender"],
                                        "prompt": test_example["prompt"]})

        self.results["m_acc"] = correct["m"] / total["m"] if total["m"] > 0 else 0.
        self.results["f_acc"] = correct["f"] / total["f"] if total["f"] > 0 else 0.
        self.results["n_acc"] = correct["n"] / total["n"] if total["n"] > 0 else 0.
        self.results["total_acc"] = (correct["m"] + correct["f"] + correct["n"]) / (total["m"] + total["f"] + total["n"])
