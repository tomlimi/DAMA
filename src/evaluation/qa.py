import json
import torch
from math import log, prod

from evaluation import Evaluate

class EvaluateQA(Evaluate):
    def __init__(self, model, tok, test_file, task):

        super().__init__(model,tok, test_file, task)
        assert task == "qa", f"Task class mismatch:, expected 'qa', got '{task}' instead"

        self.load_data()

    def load_data(self):
        with open(self.test_file) as f:
            self.dataset = f.readlines()
        self.dataset = [json.loads(d) for d in self.dataset]

    def evaluate(self):
        def score(a, n, norm):
            logx = lambda x: log(x) if x > 0 else float('-inf')
            return prod(a) / len(a), prod(a) / norm, sum([logx(x) for x in a]) - sum([logx(x) for x in n]), prod(a),  prod(a)**(1/len(a))

        partial_results = []
        for d in self.dataset:
            prompt = d['question']['stem'] # + " Answer:" # or maybe without the " Answer:" part?
            norm = "Answer:"
            prompt_tokens = self.tok.encode(prompt, return_tensors="pt")
            norm_tokens = self.tok.encode(norm, return_tensors="pt")
            ans = []
            correct = d['answerKey']
            for c in d['question']['choices']:
                choice_tokens = self.tok.encode(c['text'], return_tensors="pt")
                choice_len = choice_tokens.shape[1]

                probs_q = torch.softmax(self.model.forward(torch.cat((prompt_tokens, choice_tokens), 1)).logits[:,-choice_len-1:-1,:].float(), dim=-1)[0, list(range(choice_len)), choice_tokens[0]]
                # the probabilities are always for the next token, so we need to look at the last but one position
                probs_n = torch.softmax(self.model.forward(torch.cat((norm_tokens, choice_tokens), 1)).logits[:,-choice_len-1:-1,:].float(), dim=-1)[0, list(range(choice_len)), choice_tokens[0]]

                answer_scores = score(probs_q.tolist(), probs_n.tolist(), len(c['text']))
                ans.append((c['label'], answer_scores))
            
            partial_results.append((
                sorted(ans, key=lambda x: x[1][0])[-1][0] == correct, 
                sorted(ans, key=lambda x: x[1][1])[-1][0] == correct,
                sorted(ans, key=lambda x: x[1][2])[-1][0] == correct,
                sorted(ans, key=lambda x: x[1][3])[-1][0] == correct,
                sorted(ans, key=lambda x: x[1][4])[-1][0] == correct
            ))
        '''
        print("per token prob accuracy - root: {:2.2%}".format( 
            sum([x[4] for x in partial_results])/len(partial_results)
        ))

        print("per token prob accuracy: {:2.2%}".format( 
            sum([x[0] for x in partial_results])/len(partial_results)
            ))

        print("per char prob accuracy: {:2.2%}".format(   
            sum([x[1] for x in partial_results])/len(partial_results)
            ))

        print("normalized prob accuracy: {:2.2%}".format( 
            sum([x[2] for x in partial_results])/len(partial_results)
            ))

        print("unnormalized prob accuracy: {:2.2%}".format( 
            sum([x[3] for x in partial_results])/len(partial_results)
            ))

        print("based on", len(partial_results), "answers")
        print()
        '''

        self.results = {
            'per_token_prob_root': sum([x[4] for x in partial_results])/len(partial_results),
            'per_token_prob': sum([x[0] for x in partial_results])/len(partial_results),
            'per_char_prob': sum([x[1] for x in partial_results])/len(partial_results),
            'normed_prob': sum([x[2] for x in partial_results])/len(partial_results),
            'unnormed_prob': sum([x[3] for x in partial_results])/len(partial_results),
        }

        self.partial_results = {}
        