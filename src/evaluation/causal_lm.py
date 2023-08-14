import os
import json

from datasets import load_dataset

from evaluation import Evaluate, compute_perplexity


class EvaluateCausalLM(Evaluate):

    def __init__(self, model, tok, test_file, task):

        super().__init__(model,tok, test_file, task)
        assert task == "causal_lm", f"Task class mismatch:, expected 'causal_lm', got '{task}' instead"

        self.results = {"mean_perplexity": 0.}
        self.partial_results = []

        self.dataset = []

        self.load_data()

    def load_data(self):
        if len(self.test_file.split("_")) > 1:
            self.dataset = load_dataset(self.test_file.split("_")[0],
                                        self.test_file.split("_")[1],
                                        split='test').to_iterable_dataset()
        else:
            self.dataset = load_dataset(self.test_file, split='test').to_iterable_dataset()

    def evaluate(self):

        results = compute_perplexity([te['text'] for te in self.dataset if len(te['text']) > 0],
                                     self.model, self.tok)

        self.results["mean_perplexity"] = results["mean_perplexity"]
        self.partial_results = results["perplexities"]
