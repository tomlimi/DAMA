import os
from pathlib import Path
from typing import Dict, List, Tuple
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import ROMEHyperParams, apply_rome_to_model, execute_rome
from dama import DAMAHyperParams, apply_dama_to_model, execute_dama

from utils.generate import generate_interactive, generate_fast
from utils import nethook

# from util import nethook
# from util.generate import generate_interactive, generate_fast
import argparse
import sys


def model_editing(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    generation_prompts: List[str],
    model_name: str,
    params_dir: Path,
    method: str,
    projections_saveto: Path = None,
    projections_loadfrom: Path = None,
    online_update: bool = False
) -> tuple[AutoModelForCausalLM | AutoModelForCausalLM, list[str]]:
    """
    Applies the selected model editing algorithm. Generates text both before and after
    for comparison of model behavior. Returns the updated model and the original values of
    weights that were changed.
    """

    nethook.set_requires_grad(True, model)

    hparams_prefix = method
    hparams_suffix = ""
    params_name = (
        params_dir
        / hparams_prefix
        / f"{model_name}.json"
    )

    print(f"Retrieving {method} hyperparameters")
    print("Loading from", params_name)
    if method == 'ROME':
        hparams = ROMEHyperParams.from_json(params_name)
    elif method == 'DAMA':
        hparams = DAMAHyperParams.from_json(params_name)
    else:
        raise ValueError(f"Unknown method {method}. Choose from: ROME, DAMA")
    print(hparams)

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
            online_update=online_update)
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




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_path", type=str, default="/lnet/work/people/limisiewicz/GenderBiasGACR/models/llama")
    parser.add_argument("--param_number", type=int, default=None)
    parser.add_argument("--data_path", type=str, default="/lnet/work/people/limisiewicz/GenderBiasGACR/data")
    parser.add_argument("--results_path", type=str, default="/lnet/work/people/limisiewicz/GenderBiasGACR/results")
    parser.add_argument("--rome_path", type=str, default="/lnet/work/people/limisiewicz/GenderBiasGACR/gender-bias/causal_tracing/rome")
    parser.add_argument("--method", type=str, default="ROME")
    parser.add_argument("--request_file", type=str, default=None)
    parser.add_argument("--generation_file", type=str, default=None)
    parser.add_argument("--save_projections", type=bool, default=False)
    parser.add_argument("--load_projections", type=bool, default=False)
    parser.add_argument("--online_update", type=bool, default=False)
    args = parser.parse_args()
    
    
    sys.path.append(args.rome_path)


    data_path = Path(args.data_path)
    
    model_name = args.model_name_path
    if model_name.endswith("llama"):
        model_name_short = "llama"
        if args.param_number in {7, 13, 30, 65}:
            model_name += f"_{args.param_number}B"
            model_name_short += f"_{args.param_number}B"
        tokenizer_name = model_name
    elif model_name == "HuggingFaceM4/tiny-random-LlamaForCausalLM":
        model_name_short = "llama_tiny"
        # For debugging purposes
        tokenizer_name = "/lnet/work/people/limisiewicz/GenderBiasGACR/models/llama_7B"
    else:
        model_name_short = model_name.split("/")[-1]
        tokenizer_name = model_name
        
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                                 low_cpu_mem_usage=True, device_map='auto')
    orig_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                                 low_cpu_mem_usage=True, device_map='auto')
    if torch.cuda.is_available():
        model = model.eval().cuda()
        orig_model = orig_model.eval().cuda()

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, return_token_type_ids=False, add_bos_token=False)
    # set llama special tokens
    tok.bos_token = "<s>"
    tok.eos_token = "</s>"
    tok.pad_token = "</s>"
    tok.unk_token = "<unk>"
    tok.padding_side = "right"

    if args.request_file is not None:
        with open(os.path.join(args.data_path, args.request_file), "r") as f:
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
            }
        ]
    if args.generation_file is not None:
        with open(os.path.join(args.data_path, args.generation_file), "r") as f:
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
        projections_path = os.path.join(args.results_path, "dama", f"projections_{model_name_short}.npy")
        if args.save_projections:
            projections_saveto = projections_path
        elif args.load_projections:
            projections_loadfrom = projections_path

    model_new, orig_weights= model_editing(
        model, tok, request, generation_prompts,  model_name_short, data_path, args.method,
        projections_saveto=projections_saveto, projections_loadfrom=projections_loadfrom,
        online_update=args.online_update
    )

    print("Modified model:")
    print(model_new)
    print("Original model:")
    print(orig_model)
    generate_interactive(model_new, tok, max_out_len=100, use_logit_lens=True,
                        layer_module_tmp= "model.layers.{}",
                        ln_f_module= "model.norm",
                        lm_head_module= "lm_head",
                        compare_against=orig_model)
