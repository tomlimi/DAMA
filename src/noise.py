import os, re, sys, json
import torch, numpy
import math
import argparse
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from collections import defaultdict
from causal_tracing.causal_trace import ModelAndTokenizer, make_inputs, find_token_range, guess_subject, calculate_hidden_flow, collect_embedding_std
from causal_tracing.utils import layername
from utils.knowns import KnownsDataset
from utils.globals import *
from utils import nethook

torch.set_grad_enabled(False)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False 
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 16

#m_stereo = ProfessionsDataset("/home/marecek/troja/gender-bias/causal_tracing/data/m_stereo.json")
#f_stereo = ProfessionsDataset("/home/marecek/troja/gender-bias/causal_tracing/data/f_stereo.json")

PRONOUNS = ('she', 'he', 'they')

def get_pronoun_probabilities(output, is_batched=False):
    
    if is_batched:
        probabilities = torch.softmax(output[1:, -1, :], dim=1).mean(dim=0)
    else:
        probabilities = torch.softmax(output[:, -1, :], dim=1).mean(dim=0)
    pron_prob = [] 
    for pronoun in PRONOUNS:
        pron_prob.append(probabilities[mt.tokenizer.encode(pronoun)][0])
    
    return torch.stack(pron_prob)

def trace_with_patch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    trace_layers=None,  # List of traced outputs to return
):
    prng = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng.randn(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)

    # We report softmax probabilities for the answers_t token predictions of interest.
    # MY_CODE_BEGIN
    probs = get_pronoun_probabilities(outputs_exp.logits, is_batched=True)

    # MY_CODE_END

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs


def plot_noise(dataset, threshold_a, threshold_b, title):
    rel_noises = []
    stddevs_he = []
    stddevs_she = []
    stddevs_they = []
    means_he = []
    means_she = []
    means_they = []
    print(title, ": ", end='')

    for i in range(30):
        rel_noise = 0.5 * i
        rel_noises.append(rel_noise)
        low_scores_he = []
        low_scores_she = []
        low_scores_they = []
        print(rel_noise, end=' ', flush=True)
    
        for knowledge in dataset:
            if knowledge["gender_score"] < threshold_a or knowledge["gender_score"] > threshold_b:
                continue
            samples = 10
            inp = make_inputs(mt.tokenizer, [knowledge["prompt"]] * (samples + 1))
            #with torch.no_grad():
            #    base_score = pronoun_probs(mt.model, inp)
            answer_t = None
            e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], knowledge["subject"])
            low_score = trace_with_patch(
                mt.model, inp, [], answer_t, e_range, noise=rel_noise*subject_std
            )
            low_scores_he.append(low_score[1])
            low_scores_she.append(low_score[0])
            low_scores_they.append(low_score[2])
    
        stddevs_he.append(torch.stack(low_scores_he).std().item())
        stddevs_she.append(torch.stack(low_scores_she).std().item())
        stddevs_they.append(torch.stack(low_scores_they).std().item())

        means_he.append(torch.stack(low_scores_he).mean().item())
        means_she.append(torch.stack(low_scores_she).mean().item())
        means_they.append(torch.stack(low_scores_they).mean().item())

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    
    axs.plot(rel_noises, means_he, color='tab:blue')
    axs.fill_between(rel_noises,np.array(means_he)-np.array(stddevs_he),np.array(means_he)+np.array(stddevs_he),alpha=.1)
    axs.plot(rel_noises, means_she, color='tab:red')
    axs.fill_between(rel_noises,np.array(means_she)-np.array(stddevs_she),np.array(means_she)+np.array(stddevs_she),alpha=.1)
    axs.plot(rel_noises, means_they, color='tab:green')
    axs.fill_between(rel_noises,np.array(means_they)-np.array(stddevs_they),np.array(means_they)+np.array(stddevs_they),alpha=.1)
    axs.set_title("AVG on " + title)
    
    plt.savefig("../my_results/"+title+".pdf", format="pdf")

    print("")

def plot_templates(dataset, title):
    
    score = defaultdict(list)
    total_score = []
    rel_noise = 3

    for knowledge in dataset:
        #print('.', end='')
        samples = 10
        inp = make_inputs(mt.tokenizer, [knowledge["prompt"]] * (samples + 1))
        answer_t = None
        e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], knowledge["subject"])
        low_score = trace_with_patch(
            mt.model, inp, [], answer_t, e_range, noise=rel_noise*subject_std
        )
        template = knowledge["prompt"][len(knowledge["subject"]):]
        value = (low_score[1] - low_score[0]).item()
        score[template].append(value)
        total_score.append(value)

    names = []
    stddevs = []
    avgs = []
    
    total_avg = np.mean(total_score)
    
    for template in score.keys():
        names.append(template)
        stddevs.append(np.std(score[template]))
        avgs.append(np.mean(score[template]) - total_avg)
    
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    
    axs.bar(names, avgs, yerr=stddevs, align='center', alpha=0.5, ecolor='black')
    axs.tick_params(axis='x', labelrotation = 90)
    plt.tight_layout()
    #axs[1].bar(names, stddevs)
    #axs[1].tick_params(axis='x', labelrotation = 90)
    
    plt.savefig("../my_results/"+title+".pdf", format="pdf")

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--model_name_path", type=str, default="/home/limisiewicz/my-luster/dama/models/llama")
    argparse.add_argument("--param_number", type=int, default=7)
    args = argparse.parse_args()

    model_name = args.model_name_path
    if model_name.endswith("llama"):
        if args.param_number in {7, 13, 30, 65}:
            model_name += f"_{args.param_number}B"

    print("Loading the model...")

    mt = ModelAndTokenizer(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                           low_cpu_mem_usage=False)

    print("Model loaded. Loading data...")

    with open("../my_data/train_dama.json", "r") as f:
         all_data = json.load(f)

    print("Data loaded. Computing standard deviation...")
    
    subject_std = collect_embedding_std(mt, [k["subject"] for k in all_data[:100]])

    print("Testing noises...")
    
    plot_noise(all_data[:100], 0.5, 1, "Stereo_HE_"+str(args.param_number)+"B")
    plot_noise(all_data[:100], -1, -0.5, "Stereo_SHE_"+str(args.param_number)+"B")

    print("Testing templates...")

    plot_templates(all_data[:100], "Templates_"+str(args.param_number)+"B")



