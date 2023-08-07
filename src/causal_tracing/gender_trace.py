import torch
from utils import nethook
from .causal_trace import layername
from .utils import layername, project_representation

PRONOUNS = (' she', ' he', ' they')
PRONOUNS_LLAMA = ('she', 'he', 'they')

def get_pronoun_probabilities(output, mt, is_batched=False):
    
    if is_batched:
        probabilities = torch.softmax(output[1:, -1, :], dim=1).mean(dim=0)
    else:
        probabilities = torch.softmax(output[:, -1, :], dim=1).mean(dim=0)
    
    if "llama" in mt.model.name_or_path.lower():
        pronoun_tokens = PRONOUNS_LLAMA
    else:
        pronoun_tokens = PRONOUNS
        
    pron_prob = []
    for pronoun in pronoun_tokens:
        pron_prob.append(probabilities[mt.tokenizer.encode(pronoun)][0])
    
    return torch.stack(pron_prob)


def pronoun_probs(mt, inp, project_embeddings=None):
    if project_embeddings is not None:
        embed_layername = layername(mt.model, 0, "embed")
        def patch_embeddings(layer, x):
            if layer == embed_layername:
                return project_representation(x, **project_embeddings)
            else:
                return x
        with torch.no_grad(), nethook.TraceDict(
                mt.model,
                [embed_layername],
                edit_output=patch_embeddings,
        ) as td:
            out = mt.model(**inp)
    else:
        out = mt.model(**inp)
    probs = get_pronoun_probabilities(out.logits, mt, is_batched=False)
    return probs