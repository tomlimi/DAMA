import os
from pathlib import Path
from typing import Dict, List, Tuple
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import shutil

from rome import ROMEHyperParams, apply_rome_to_model, execute_rome
from dama import DAMAHyperParams, apply_dama_to_model, execute_dama

from utils.generate import generate_interactive, generate_fast
from utils import nethook
from utils.globals import *

# from util import nethook
# from util.generate import generate_interactive, generate_fast
import argparse
import sys


def model_editing(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    generation_prompts: List[str],
    hparams: ROMEHyperParams | DAMAHyperParams,
    method: str,
    projections_saveto: Path = None,
    projections_loadfrom: Path = None,
    output_dir: Path = None,
    ncv: bool = False,
    val: bool = False,
    use_neutral: bool = False
) -> tuple[AutoModelForCausalLM | AutoModelForCausalLM, list[str]]:
    """
    Applies the selected model editing algorithm. Generates text both before and after
    for comparison of model behavior. Returns the updated model and the original values of
    weights that were changed.
    """

    nethook.set_requires_grad(True, model)

    print("Generating pre-update text")
    pre_update_text = generate_fast(model, tok, generation_prompts, max_out_len=100)
    print(pre_update_text)

    print(f"Applying {method} to model")
    if method == 'ROME':
        model_new, orig_weights = apply_rome_to_model(
            model, tok, requests, hparams, return_orig_weights=True
        )
    elif method == 'DAMA':
        model_new, orig_weights = apply_dama_to_model(
            model, tok, requests, hparams, copy=False, return_orig_module=True,
            projections_saveto=projections_saveto, projections_loadfrom=projections_loadfrom,
            output_dir=output_dir, ncv=ncv, val=val, use_neutral=use_neutral)
    else:
        raise ValueError(f"Unknown method {method}. Choose from: ROME, DAMA")

    print("Generating post-update text")
    post_update_text = generate_fast(
        model_new, tok, generation_prompts, max_out_len=100
    )
    print(post_update_text)

    print("Summarizing differences")
    for i, (prompt, pre, post) in enumerate(
        zip(generation_prompts, pre_update_text, post_update_text)
    ):
        if i > 0:
            print("".join(["-" for _ in range(10)]))

        prompt_str = "[Prompt]:"
        pre_str = f"[Pre-{method}]:"
        post_str = f"[Post-{method}]:"
        pad_to = 1 + max(len(prompt_str), len(pre_str), len(post_str))

        for s, t in zip([prompt_str, post_str, pre_str], [prompt, post, pre]):
            print(s.ljust(pad_to), t)

    return model_new, orig_weights


def parse_experiment_name(num_layers: int=9,
                          iterative_update: bool=False,
                          mixed_update: bool=False,
                          task: str="gen",
                          post_linear: bool=False,
                          batch_size: int=1,
                          orthogonal_constraint: float=0.,
                          no_colinear_vs: bool=False,
                          vs_at_last: bool=False,
                          null_dim: int=1024,
                          use_neutral: bool=False,
                          delta_only: bool=False,
                          use_neutral_tensor: bool=False,
                          nw: bool=False
                          ) -> str:
    
    experiment_string = f"l{num_layers}"
    if iterative_update:
        experiment_string += "_iter"
    elif mixed_update:
        experiment_string += ""
    else:
        experiment_string += "_once"

    if post_linear:
        experiment_string += ""
    else:
        experiment_string += "_prel"

    if task == "gen":
        experiment_string += ""
    elif task == "coref":
        experiment_string += "_coref"
        raise NotImplementedError("Coreference resolution task not implemented yet")
    else:
        raise ValueError("Unknown task. Choose from: gen, coref")

    if batch_size == 1:
        experiment_string += ""
    elif batch_size > 1:
        experiment_string += f"_b{batch_size}"
    else:
        raise ValueError("Batch size must be a positive integer")

    if orthogonal_constraint:
        experiment_string += f"_o{orthogonal_constraint}"
    else:
        experiment_string += "_on"

    if null_dim != 1024:
        experiment_string += f"_nd{null_dim}"

    if no_colinear_vs:
        experiment_string += "_ncv"

    if vs_at_last:
        experiment_string += "_val"

    if use_neutral:
        experiment_string += "_neutral"
    if delta_only:
        experiment_string += "_delta_only"

    if use_neutral_tensor:
        experiment_string += "_nt"


    if nw:
        experiment_string += '_nw'
    return experiment_string


def get_model_tokenizer(model_name, param_number, compare_against=False):
    if model_name.endswith("llama"):
        model_name = "llama"
        model_path = os.path.join(MODEL_DIR, "llama")
        if param_number in {7, 13, 30, 65}:
            model_name += f"_{param_number}B"
            model_path += f"_{param_number}B"
        tokenizer_path = model_path
    elif model_name == "HuggingFaceM4/tiny-random-LlamaForCausalLM":
        model_path = model_name
        model_name = "llama_tiny"
        # For debugging purposes
        tokenizer_path = os.path.join(MODEL_DIR, "llama_7B")
    else:
        model_path = model_name
        tokenizer_path = model_name
        model_name = model_name.split("/")[-1]

    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                                 low_cpu_mem_usage=True, device_map='auto')

    if torch.cuda.is_available() and torch.cuda.device_count() == 1:
        model = model.eval().cuda()
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = model.eval()

    if compare_against:
        orig_model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                                          low_cpu_mem_usage=True, device_map='auto')
        if torch.cuda.is_available():
            orig_model = orig_model.eval().cuda()
    else:
        orig_model = None

    tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, return_token_type_ids=False, add_bos_token=False)
    # set llama special tokens
    tok.bos_token = "<s>"
    tok.eos_token = "</s>"
    tok.pad_token = "</s>"
    tok.unk_token = "<unk>"
    tok.padding_side = "right"

    return model_name, model, orig_model,  tok


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama")
    parser.add_argument("--param_number", type=int, default=None)
    parser.add_argument("--method", type=str, default="ROME")
    parser.add_argument("--request_file", type=str, default=None)
    parser.add_argument("--generation_file", type=str, default=None)
    parser.add_argument("--save_projections", type=bool, default=True)
    parser.add_argument("--load_projections", type=bool, default=False)
    parser.add_argument("--compare_against", type=bool, default=False)
    parser.add_argument("--num_layers", type=int, default=9)
    parser.add_argument("--iterative_update", type=bool, default=False)
    parser.add_argument("--mixed_update", type=bool, default=False)
    parser.add_argument("--task", type=str, default="gen")
    parser.add_argument("--post_linear", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--orthogonal_constraint", type=float, default=None)
    parser.add_argument("--null_dim", type=int, default=1024)
    parser.add_argument("--no_colinear_vs", type=bool, default=False)
    parser.add_argument("--vs_at_last", type=bool, default=False)
    parser.add_argument("--use_neutral", type=bool, default=False)
    parser.add_argument("--delta_only", type=bool, default=False)
    parser.add_argument("--no_whitening", type=bool, default=False)
    args = parser.parse_args()

    print(f"Load original model to compare: {args.compare_against}")
    model_name, model, orig_model, tok = get_model_tokenizer(args.model_name, args.param_number, args.compare_against)

    experiment_name_suffix = parse_experiment_name(
        num_layers=args.num_layers, iterative_update=args.iterative_update, mixed_update=args.mixed_update,
        task=args.task,
        post_linear=args.post_linear, batch_size=args.batch_size, orthogonal_constraint=args.orthogonal_constraint,
        no_colinear_vs=args.no_colinear_vs, vs_at_last=args.vs_at_last, null_dim=args.null_dim, use_neutral=args.use_neutral,
        delta_only=args.delta_only, nw=args.no_whitening
    )
    experiment_name = f"{experiment_name_suffix}"
    if args.method == "DAMA":
        output_dir = os.path.join(RESULTS_DIR, args.method, model_name, experiment_name)
    else:
        output_dir = os.path.join(RESULTS_DIR, args.method,  model_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Conducting experiment: {experiment_name}.")

    if args.request_file is not None:
        with open(os.path.join(DATA_DIR, args.request_file), "r") as f:
            request = json.load(f)
    else:
        request = [
            {
                "prompt": "{} ran because",
                "subject": "A nurse",
                "target_new": {"str": "he"},
            },
            {
                "prompt": "{} ran because",
                "subject": "A secretary",
                "target_new": {"str": "she"},
            },
            {
                "prompt": "{} asked because",
                "subject": "A nurse",
                "target_new": {"str": "he"},
            },
            {
                "prompt": "{} asked because",
                "subject": "A secretary",
                "target_new": {"str": "she"},
            }
        ]
    if args.generation_file is not None:
        with open(os.path.join(DATA_DIR, args.generation_file), "r") as f:
            generation_prompts = json.load(f)
    else:
        generation_prompts = [
            "A secretary said that",
            "A nurse said that",
            "A nurse slept because",
            "A secretary slept because",
            "A secretary was tired because",
            "A secretary whispered because",
        ]

    projections_saveto = None
    projections_loadfrom = None
    if args.method == "DAMA":
        if args.load_projections:
            projections_loadfrom = os.path.join(output_dir, "projections.npy")
        if args.save_projections:
            projections_saveto = os.path.join(output_dir, "projections.npy")

    hparams_path = os.path.join(HPARAMS_DIR, args.method, model_name, f"{experiment_name}.json")
    print(f"Retrieving {args.method} hyperparameters")
    print("Loading from", hparams_path)
    if args.method == 'ROME':
        hparams = ROMEHyperParams.from_json(hparams_path)
    elif args.method == 'DAMA':
        hparams = DAMAHyperParams.from_json(hparams_path)
    else:
        raise ValueError(f"Unknown method {hparams_path}. Choose from: ROME, DAMA")
    print(hparams)

    model_new, orig_weights= model_editing(
        model, tok, request, generation_prompts, hparams, args.method,
        projections_saveto=projections_saveto, projections_loadfrom=projections_loadfrom, output_dir=output_dir,
        ncv=args.no_colinear_vs, val=args.vs_at_last, use_neutral=args.use_neutral)

    print(f"Dumping parameters and code to: {output_dir}")
    shutil.copy(hparams_path, os.path.join(output_dir, "hparams.json"))
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f)
    with open(sys.argv[0], 'r') as this_code, open(os.path.join(output_dir, 'adapt_model.py'), 'w') as source_out:
        code_lines = this_code.readlines()
        source_out.writelines(code_lines)

    generate_interactive(model_new, tok, max_out_len=100, use_logit_lens=True,
                        layer_module_tmp= "model.layers.{}",
                        ln_f_module= "model.norm",
                        lm_head_module= "lm_head",
                        compare_against=orig_model)
