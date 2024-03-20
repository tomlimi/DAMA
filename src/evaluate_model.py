import argparse
import os

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import nethook
from dama.dama_main import apply_dama_on_module
from dama.dama_hparams import DAMAHyperParams
from dama_l.dama_l_hparams import DAMALeaceHyperParams
from memit.memit_main import MEMITHyperParams
from ft.ft_main import FTHyperParams
from utils.globals import *
from utils.model_utils import *
from utils.generate import generate_interactive, generate_fast
from adapt_model import get_model_tokenizer, parse_experiment_name

from evaluation import EvaluateGeneration, EvaluateCoreference, EvaluateCausalLM, EvaluateQA, EvaluateStereoset

def run_evaluation_on_task(model, tokenizer, task, test_file, output_dir):
    if task == "gen":
        evaluator = EvaluateGeneration(model, tokenizer, os.path.join(DATA_DIR, test_file), task)
    elif task == "coref":
        evaluator = EvaluateCoreference(model, tokenizer, os.path.join(DATA_DIR, args.test_file), task)
    elif task == "causal_lm":
        evaluator = EvaluateCausalLM(model, tokenizer, test_file, task)
    elif task == "stereoset":
        evaluator = EvaluateStereoset(model, tokenizer, os.path.join(DATA_DIR, test_file), task)
    elif task == "interactive":
        generate_interactive(model, tokenizer, max_out_len=100, use_logit_lens=True,
                        layer_module_tmp= "model.layers.{}",
                        ln_f_module= "model.norm",
                        lm_head_module= "lm_head",
                        compare_against=None)
    elif task == "qa":
        evaluator = EvaluateQA(model, tokenizer, os.path.join(DATA_DIR, args.test_file), task)
    else:
        raise ValueError(f"Unknown task {task}")

    evaluator.evaluate()
    evaluator.save_results(output_dir)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama")
    parser.add_argument("--param_number", type=int, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--test_task", type=str, default="gen")
    parser.add_argument("--num_layers", type=int, default=9)
    parser.add_argument("--task", type=str, default="gen")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--iterative_update", type=bool, default=False)
    parser.add_argument("--mixed_update", type=bool, default=False)
    parser.add_argument("--post_linear", type=bool, default=False)
    parser.add_argument("--orthogonal_constraint", type=float, default=None)
    parser.add_argument("--null_dim", type=int, default=1024)
    parser.add_argument("--no_colinear_vs", type=bool, default=False)
    parser.add_argument("--vs_at_last", type=bool, default=False)
    parser.add_argument("--use_neutral", type=bool, default=False)
    parser.add_argument("--delta_only", type=bool, default=False)
    parser.add_argument("--no_whitening", type=bool, default=False)
    parser.add_argument("--random_seed", type=int, default=0)

    args = parser.parse_args()

    model_name, model, _, tok = get_model_tokenizer(args.model_name, args.param_number, False)

    experiment_name_suffix = parse_experiment_name(
        num_layers=args.num_layers, iterative_update=args.iterative_update, mixed_update=args.mixed_update,
        task=args.task,
        post_linear=args.post_linear, batch_size=args.batch_size, orthogonal_constraint=args.orthogonal_constraint,
        no_colinear_vs=args.no_colinear_vs, vs_at_last=args.vs_at_last, null_dim=args.null_dim, use_neutral=args.use_neutral,
        delta_only=args.delta_only, nw=args.no_whitening, seed=args.random_seed
    )
    experiment_name = f"{experiment_name_suffix}"
    if args.method == "DAMA":
        print(f"Evaluating DAMA model {experiment_name}")
        output_dir = os.path.join(RESULTS_DIR, args.method, model_name, experiment_name)
        hparams = DAMAHyperParams.from_json(os.path.join(output_dir, "hparams.json"))
        projection_file = os.path.join(output_dir, "projections.npy")
        model = load_dama_model(model, hparams, projection_file)
    elif args.method == "DAMA_L":
        print(f"Evaluating DAMA Leace model")
        output_dir = os.path.join(RESULTS_DIR, args.method, f"{model_name}_{str(args.num_layers)}L")
        hparams = DAMALeaceHyperParams.from_json(os.path.join(output_dir, "hparams.json"))
        projection_file = os.path.join(output_dir, "projections.npy")
        model = load_dama_model(model, hparams, projection_file)

    elif args.method == "MEMIT":
        print(f"Evaluating MEMIT model")
        output_dir = os.path.join(RESULTS_DIR, args.method, model_name)
        hparams = MEMITHyperParams.from_json(os.path.join(output_dir, "hparams.json"))
        model = AutoModelForCausalLM.from_pretrained(output_dir, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                                      low_cpu_mem_usage=True, device_map='auto')
    elif args.method == "FT":
        print(f"Evaluating fine-tuned model")
        output_dir = os.path.join(RESULTS_DIR, args.method, model_name)
        hparams = FTHyperParams.from_json(os.path.join(output_dir, "hparams.json"))
        model = AutoModelForCausalLM.from_pretrained(output_dir, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                                      low_cpu_mem_usage=True, device_map='auto')
    elif args.method == "PEFT":
        print(f"Evaluating PEFT model")
        revision = "v6"
        output_dir = os.path.join(RESULTS_DIR, args.method, model_name, revision)
        hparams = None
        # Model saved in HF hub by PaulM2000
        model = AutoModelForCausalLM.from_pretrained("PaulM2000/merged_peft_model_random_42_without_up_proj_llama-7b",
                                                     revision=revision, offload_folder=output_dir,
                                                     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                                     low_cpu_mem_usage=True, device_map='auto')
    elif args.method == None:
        print(f"Evaluating original model {model_name}")
        output_dir = os.path.join(RESULTS_DIR, "original",model_name)
        os.makedirs(output_dir, exist_ok=True)
    else:
        raise ValueError(f"Unknown method {args.method}")

    run_evaluation_on_task(model, tok, args.test_task, args.test_file, output_dir)

