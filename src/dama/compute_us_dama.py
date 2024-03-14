from typing import Dict, List, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.repr_tools import get_module_input_output_at_words


def compute_us(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List,
    hparams,
    layer: int,
    context_templates: List[str],
    device='cuda'
):
    requests_num = sum([len(request_batch) * len(context_templates) for request_batch in requests])

    contexts = [templ.format(request["prompt"]) for request_batch in requests
                for request in request_batch for templ in context_templates]
    words = [ request["subject"] for request_batch in requests
                for request in request_batch for _ in context_templates]
    
    layer_us_batches = []
    for contexts_batch, words_batch in zip(np.array_split(contexts, requests_num // 16 + 1),
                                           np.array_split(words, requests_num // 16 + 1)):
        layer_us_batches.append(get_module_input_output_at_words(model, tok, contexts_batch, words_batch,
                                                    layer, hparams.rewrite_module_tmp, hparams.fact_token)[0].detach().to(device))
    layer_us = torch.cat(layer_us_batches, dim=0)

    batch_lens = [0] + [len(context_templates) * len(request_batch) for request_batch in requests]
    batch_csum = np.cumsum(batch_lens).tolist()

    u_list = []
    for i in range(len(batch_lens) -1):
        start = batch_csum[i]
        if i == len(batch_lens) - 1:
            end = layer_us.size(0)
        else:
            end = batch_csum[i + 1]
        tmp = []
        for j in range(start, end, len(context_templates)):
            tmp.append(layer_us[j : j + len(context_templates)].mean(0))
        u_list.append(torch.stack(tmp, 0).mean(0))
    return torch.stack(u_list, dim=0)


