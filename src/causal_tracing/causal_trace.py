# Implementation from with changes:  https://github.com/kmeng01/rome
import argparse
import json
import os
import re
from collections import defaultdict

import numpy
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

from utils.knowns import  KnownsDataset

from utils import nethook
from utils.globals import DATA_DIR

from .utils import layername, project_representation
from .gender_trace import get_pronoun_probabilities, pronoun_probs, PRONOUNS_LLAMA


def main():
    parser = argparse.ArgumentParser(description="Causal Tracing")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    def parse_noise_rule(code):
        if code in ["m", "s"]:
            return code
        elif re.match("^[uts][\d\.]+", code):
            return code
        else:
            return float(code)

    aa(
        "--model_name",
        default="gpt2-xl",
        choices=[
            "gpt2-xl",
            "EleutherAI/gpt-j-6B",
            "EleutherAI/gpt-neox-20b",
            "gpt2-large",
            "gpt2-medium",
            "gpt2",
            "llama_7B",
            "llama_13B",
        ],
    )
    aa("--fact_file", default=None)
    aa("--output_dir", default="results/{model_name}/causal_trace")
    aa("--noise_level", default="s3", type=parse_noise_rule)
    aa("--replace", default=0, type=int)
    aa("--project_embeddings", type=str, default=None)
    args = parser.parse_args()

    modeldir = f'r{args.replace}_{args.model_name.replace("/", "_")}'
    modeldir = f"n{args.noise_level}_" + modeldir
    output_dir = args.output_dir.format(model_name=modeldir)
    result_dir = f"{output_dir}/cases"
    pdf_dir = f"{output_dir}/pdfs"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    # Half precision to let the 20b model fit.
    torch_dtype = torch.float16 if ("20b" in args.model_name or "llama" in args.model_name.lower()) else None

    mt = ModelAndTokenizer(args.model_name, torch_dtype=torch_dtype)

    if args.project_embeddings is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        project_embeddings = numpy.load(args.project_embeddings)
        project_embeddings = torch.from_numpy(project_embeddings)
        project_embeddings.to(device)

    if args.fact_file is None:
        knowns = KnownsDataset(DATA_DIR)
    else:
        with open(args.fact_file) as f:
            knowns = json.load(f)

    noise_level = args.noise_level
    uniform_noise = False
    if isinstance(noise_level, str):
        if noise_level.startswith("s"):
            # Automatic spherical gaussian
            factor = float(noise_level[1:]) if len(noise_level) > 1 else 1.0
            noise_level = factor * collect_embedding_std(
                mt, [k["subject"] for k in knowns]
            )
            print(f"Using noise_level {noise_level} to match model times {factor}")
        elif noise_level == "m":
            # Automatic multivariate gaussian
            raise NotImplementedError
            # noise_level = collect_embedding_gaussian(mt)
            # print(f"Using multivariate gaussian to match model noise")
        elif noise_level.startswith("t"):
            # Automatic d-distribution with d degrees of freedom
            raise NotImplementedError
            # degrees = float(noise_level[1:])
            # noise_level = collect_embedding_tdist(mt, degrees)
        elif noise_level.startswith("u"):
            uniform_noise = True
            noise_level = float(noise_level[1:])

    for knowledge in tqdm(knowns):
        known_id = knowledge["known_id"]
        for kind in None, "mlp", "attn":
            kind_suffix = f"_{kind}" if kind else ""
            filename = f"{result_dir}/knowledge_{known_id}{kind_suffix}.npz"
            if not os.path.isfile(filename):
                result = calculate_hidden_flow(
                    mt,
                    knowledge["prompt"],
                    knowledge["subject"],
                    expect=knowledge["attribute"],
                    kind=kind,
                    noise=noise_level,
                    uniform_noise=uniform_noise,
                    replace=args.replace,
                    project_embeddings=args.project_embeddings,
                )
                numpy_result = {
                    k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                    for k, v in result.items()
                }
                numpy.savez(filename, **numpy_result)
            else:
                numpy_result = numpy.load(filename, allow_pickle=True)
            if not numpy_result["correct_prediction"]:
                tqdm.write(f"Skipping {knowledge['prompt']}")
                continue
            plot_result = dict(numpy_result)
            plot_result["kind"] = kind
            pdfname = f'{pdf_dir}/{str(numpy_result["answer"]).strip()}_{known_id}{kind_suffix}.pdf'
            if known_id > 200:
                continue
            plot_trace_heatmap(plot_result, savepdf=pdfname)


def trace_with_patch(
        mt,  # The model and tokenizer
        inp,  # A set of inputs
        states_to_patch,  # A list of (token index, layername) triples to restore
        answers_t,  # Answer probabilities to collect
        tokens_to_mix,  # Range of tokens to corrupt (begin, end)
        noise=0.1,  # Level of noise to add
        uniform_noise=False,
        replace=False,  # True to replace with instead of add noise
        trace_layers=None,  # List of traced outputs to return
        project_embeddings=None  # INLP projection matrix to project embeddings
):
    """
    Runs a single causal trace.  Given a model and a batch input where
    the batch size is at least two, runs the batch in inference, corrupting
    a the set of runs [1...n] while also restoring a set of hidden states to
    the values from an uncorrupted run [0] in the batch.

    The convention used by this function is that the zeroth element of the
    batch is the uncorrupted run, and the subsequent elements of the batch
    are the corrupted runs.  The argument tokens_to_mix specifies an
    be corrupted by adding Gaussian noise to the embedding for the batch
    inputs other than the first element in the batch.  Alternately,
    subsequent runs could be corrupted by simply providing different
    input tokens via the passed input batch.

    Then when running, a specified set of hidden states will be uncorrupted
    by restoring their values to the same vector that they had in the
    zeroth uncorrupted run.  This set of hidden states is listed in
    states_to_patch, by listing [(token_index, layername), ...] pairs.
    To trace the effect of just a single state, this can be just a single
    token/layer pair.  To trace the effect of restoring a set of states,
    any number of token indices and layers can be listed.
    """

    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    embed_layername = layername(mt.model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    def patch_rep(x, layer):
        if layer == embed_layername:

            if project_embeddings is not None:
                x = project_representation(x, **project_embeddings)

            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                noise_data = noise_fn(
                    torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
                ).to(x.device)
                if replace:
                    x[1:, b:e] = noise_data
                else:
                    x[1:, b:e] += noise_data
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
            mt.model,
            [embed_layername] + list(patch_spec.keys()) + additional_layers,
            edit_output=patch_rep,
    ) as td:
        outputs_exp = mt.model(**inp)

    # We report softmax probabilities for three gendered probabilites:
    probs = get_pronoun_probabilities(outputs_exp.logits, mt, is_batched=True)

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs


def trace_with_repatch(
        mt,  # The modelAndTokenizer
        inp,  # A set of inputs
        states_to_patch,  # A list of (token index, layername) triples to restore
        states_to_unpatch,  # A list of (token index, layername) triples to re-randomize
        answers_t,  # Answer probabilities to collect
        tokens_to_mix,  # Range of tokens to corrupt (begin, end)
        noise=0.1,  # Level of noise to add
        uniform_noise=False,
        replace=False,
        project_embeddings=None  # INLP projection matrix to project embeddings
):
    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    unpatch_spec = defaultdict(list)
    for t, l in states_to_unpatch:
        unpatch_spec[l].append(t)

    embed_layername = layername(mt.model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            if project_embeddings is not None:
                x = project_representation(x, **project_embeddings)
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                noise_data = noise_fn(
                    torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
                ).to(x.device)
                if replace:
                    x[1:, b:e] = noise_data
                else:
                    x[1:, b:e] += noise_data
            return x
        if first_pass or (layer not in patch_spec and layer not in unpatch_spec):
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        # patch overrides unpatch
        if layer in patch_spec:
            for t in patch_spec.get(layer, []):
                h[1:, t] = h[0, t]
            return x
        for t in unpatch_spec.get(layer, []):
            h[1:, t] = untuple(first_pass_trace[layer].output)[1:, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    for first_pass in [True, False] if states_to_unpatch else [False]:
        with torch.no_grad(), nethook.TraceDict(
                mt.model,
                [embed_layername] + list(patch_spec.keys()) + list(unpatch_spec.keys()),
                edit_output=patch_rep,
        ) as td:
            outputs_exp = mt.model(**inp)
            if first_pass:
                first_pass_trace = td

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = get_pronoun_probabilities(outputs_exp.logits, mt, is_batched=True)

    return probs


def calculate_hidden_flow(
        mt,
        prompt,
        subject,
        samples=10,
        noise=0.1,
        token_range=None,
        uniform_noise=False,
        replace=False,
        window=10,
        kind=None,
        expect=None,
        disable_mlp=False,
        disable_attn=False,
        project_embeddings=None
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        base_score = pronoun_probs(mt, inp, project_embeddings=project_embeddings)

    answer_t = None
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    if token_range == "subject_last":
        token_range = [e_range[1] - 1]
    elif token_range is not None:
        raise ValueError(f"Unknown token_range: {token_range}")
    low_score = trace_with_patch(
        mt, inp, [], answer_t, e_range, noise=noise, uniform_noise=uniform_noise
    )
    if not kind:
        differences = trace_important_states(
            mt,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            token_range=token_range,
            disable_mlp=disable_mlp,
            disable_attn=disable_attn,
            project_embeddings=project_embeddings
        )
    else:
        differences = trace_important_window(
            mt,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            window=window,
            kind=kind,
            token_range=token_range,
            disable_mlp=disable_mlp,
            disable_attn=disable_attn,
            project_embeddings=project_embeddings
        )
    differences = differences.detach().cpu()
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        # answer=answer,
        answer=PRONOUNS_LLAMA,
        window=window,
        correct_prediction=True,
        kind=kind or "",
    )


def trace_important_states(
        mt,
        num_layers,
        inp,
        e_range,
        answer_t,
        noise=0.1,
        uniform_noise=False,
        replace=False,
        token_range=None,
        disable_mlp=False,
        disable_attn=False,
        project_embeddings=None
):
    ntoks = inp["input_ids"].shape[1]
    table = []

    if token_range is None:
        token_range = range(ntoks)
    for tnum in token_range:
        zero_lyrs = []
        if disable_mlp:
            # zero_lyrs = [
            #     (tnum, layername(mt.model, L, "mlp")) for L in range(0, num_layers)
            # ]
            # disable on all tokens
            zero_lyrs = [
                (t, layername(mt.model, L, "mlp")) for t in token_range for L in range(0, num_layers)
            ]
        if disable_attn:
            # zero_lyrs += [
            #     (tnum, layername(mt.model, L, "attn")) for L in range(0, num_layers)
            # ]
            zero_lyrs += [
                (t, layername(mt.model, L, "attn")) for t in token_range for L in range(0, num_layers)
            ]
        row = []
        for layer in range(0, num_layers):
            if disable_mlp or disable_attn:
                r = trace_with_repatch(
                    mt,
                    inp,
                    [(tnum, layername(mt.model, layer))],
                    zero_lyrs,
                    answer_t,
                    tokens_to_mix=e_range,
                    noise=noise,
                    uniform_noise=uniform_noise,
                    replace=replace,
                    project_embeddings=project_embeddings
                )
            else:
                r = trace_with_patch(
                    mt,
                    inp,
                    [(tnum, layername(mt.model, layer))],
                    answer_t,
                    tokens_to_mix=e_range,
                    noise=noise,
                    uniform_noise=uniform_noise,
                    replace=replace,
                    project_embeddings=project_embeddings

                )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
        mt,
        num_layers,
        inp,
        e_range,
        answer_t,
        kind,
        window=10,
        noise=0.1,
        uniform_noise=False,
        replace=False,
        disable_mlp=False,
        disable_attn=False,
        token_range=None,
        project_embeddings=None
):
    ntoks = inp["input_ids"].shape[1]
    table = []

    if token_range is None:
        token_range = range(ntoks)
    for tnum in token_range:
        zero_lyrs = []
        if disable_mlp:
            zero_lyrs = [
                (t, layername(mt.model, L, "mlp")) for t in token_range for L in range(0, num_layers)
            ]
        if disable_attn:
            zero_lyrs += [
                (t, layername(mt.model, L, "attn")) for t in token_range for L in range(0, num_layers)
            ]
        row = []
        for layer in range(0, num_layers):
            layerlist = [
                (tnum, layername(mt.model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            if disable_mlp or disable_attn:
                r = trace_with_repatch(
                    mt,
                    inp,
                    layerlist,
                    zero_lyrs,
                    answer_t,
                    tokens_to_mix=e_range,
                    noise=noise,
                    uniform_noise=uniform_noise,
                    replace=replace,
                    project_embeddings=project_embeddings
                )
            else:
                r = trace_with_patch(
                    mt,
                    inp,
                    layerlist,
                    answer_t,
                    tokens_to_mix=e_range,
                    noise=noise,
                    uniform_noise=uniform_noise,
                    replace=replace,
                    project_embeddings=project_embeddings
                )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer.  Counts the number
    of layers.
    """

    def __init__(
            self,
            model_name=None,
            model=None,
            tokenizer=None,
            low_cpu_mem_usage=False,
            torch_dtype=None,
    ):
        if tokenizer is None:
            assert model_name is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token=False, use_fast=False)
        if model is None:
            assert model_name is not None
            if torch.cuda.device_count() <= 1:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype
                )
                if torch.cuda.is_available():
                    model = model.eval().cuda()
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, low_cpu_mem_usage=True, torch_dtype=torch_dtype, device_map="auto"
                )
                model = model.eval()

            nethook.set_requires_grad(False, model)

        self.tokenizer = tokenizer
        self.model = model
        self.layer_names = [
            n
            for n, m in model.named_modules()
            if (re.match(r"^(transformer|gpt_neox|model)\.(h|layers)\.\d+$", n))
        ]
        self.num_layers = len(self.layer_names)

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )

    def get_input_representations(self, prompt, average=True):
        """
        Get the input representations for a given prompt
        :param prompt: textual input
        :param average: whether to average embeddings across all tokens
        :return: torch.Tensor of shape (embedding_size,) or (num_tokens, embedding_size)
        """
        input_ids = self.tokenizer.encode(prompt)

        # get embeddings at given token positions
        input_embeddings = self.model.get_input_embeddings().weight[input_ids, :]
        if average:
            input_embeddings = input_embeddings.mean(dim=0, keepdim=False)
        return input_embeddings


def guess_subject(prompt):
    return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
        0
    ].strip()


def plot_hidden_flow(
        mt,
        prompt,
        subject=None,
        samples=10,
        noise=0.1,
        uniform_noise=False,
        window=10,
        kind=None,
        savepdf=None,
):
    if subject is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow(
        mt,
        prompt,
        subject,
        samples=samples,
        noise=noise,
        uniform_noise=uniform_noise,
        window=window,
        kind=kind,
    )
    plot_trace_heatmap(result, savepdf)


def plot_trace_heatmap(result, savepdf=None, title=None, xlabel=None, modelname=None):
    differences = result["scores"]
    low_score = result["low_score"]
    answer = result["answer"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    labels = list(result["input_tokens"])
    for i in range(*result["subject_range"]):
        labels[i] = labels[i] + "*"

    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
                kind
            ],
            vmin=low_score,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)
        if not modelname:
            modelname = "GPT"
        if not kind:
            ax.set_title("Impact of restoring state after corrupted input")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            kindname = "MLP" if kind == "mlp" else "Attn"
            ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
        cb = plt.colorbar(h)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_all_flow(mt, prompt, subject=None):
    for kind in ["mlp", "attn", None]:
        plot_hidden_flow(mt, prompt, subject, kind=kind)


# Utilities for dealing with tokens
def make_inputs(tokenizer, prompts, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring):
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    substring = "".join(decode_tokens(tokenizer, tokenizer.encode(substring)))
    # whole_string = tokenizer.decode(token_array)
    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)


def predict_token(mt, prompts, return_p=False):
    inp = make_inputs(mt.tokenizer, prompts)
    preds, p = predict_from_input(mt.model, inp)
    result = [mt.tokenizer.decode(c) for c in preds]
    if return_p:
        result = (result, p)
    return result


def predict_from_input(model, inp):
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p


def collect_embedding_std(mt, subjects):
    alldata = []
    for s in subjects:
        inp = make_inputs(mt.tokenizer, [s])
        with nethook.Trace(mt.model, layername(mt.model, 0, "embed")) as t:
            mt.model(**inp)
            alldata.append(t.output[0])
    alldata = torch.cat(alldata)
    noise_level = alldata.std().item()
    return noise_level


def make_generator_transform(mean=None, cov=None):
    d = len(mean) if mean is not None else len(cov)
    device = mean.device if mean is not None else cov.device
    layer = torch.nn.Linear(d, d, dtype=torch.double)
    nethook.set_requires_grad(False, layer)
    layer.to(device)
    layer.bias[...] = 0 if mean is None else mean
    if cov is None:
        layer.weight[...] = torch.eye(d).to(device)
    else:
        _, s, v = cov.svd()
        w = s.sqrt()[None, :] * v
        layer.weight[...] = w
    return layer


if __name__ == "__main__":
    main()
