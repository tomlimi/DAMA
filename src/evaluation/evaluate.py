import os
import json

from abc import abstractmethod


class Evaluate:

    def __init__(self, model, tok, test_file, task):
        self.model = model
        self.tok = tok
        self.test_file = test_file
        self.task = task
        self.results = {}

    @abstractmethod
    def evaluate(self):
        pass

    def get_prediction_probability(self, prompt):

        input_ids = self.tok.encode(prompt, return_tensors="pt")
        logits = self.model(input_ids)[0].float()
        probabilities = logits.softmax(dim=2)[:,-1,:].squeeze()
        return probabilities.tolist()

    def save_results(self, result_dir):
        test_name = os.path.basename(self.test_file).split(".")[0]
        with open(os.path.join(result_dir, f"{self.task}_{test_name}.json"), 'w') as f:
            json.dump(self.results, f)





