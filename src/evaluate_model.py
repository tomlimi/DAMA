import argparse
import os

import torch
import numpy as np

from utils import nethook
from dama.dama_main import apply_dama_on_module
from dama.dama_hparams import DAMAHyperParams
from utils.globals import *
from adapt_model import get_model_tokenizer, parse_experiment_name

from evaluation import EvaluateGeneration, EvaluateCoreference, EvaluateCausalLM


def load_dama_model(model, hparams, projection_file):
    layers = dict(model.named_modules())
    devices = [layers["model.layers.{}.mlp".format(i)].down_proj.weight.device for i in hparams.layers]
    print(f"Loading projections from {projection_file}")
    loaded_projections = np.load(projection_file, allow_pickle=True).item()
    if torch.cuda.is_available():
        projections = {m_name: (torch.tensor(values['M'], device=dev, dtype=torch.float16),
                                torch.tensor(values['mu_in'], device=dev, dtype=torch.float16),
                                torch.tensor(values['mu_out'], device=dev, dtype=torch.float16))
                       for dev, (m_name, values) in zip(devices, loaded_projections.items())}
    else:
        projections = {m_name: (torch.tensor(values['M'], device='cpu', dtype=torch.float32),
                                torch.tensor(values['mu_in'], device='cpu', dtype=torch.float32),
                                torch.tensor(values['mu_out'], device='cpu', dtype=torch.float32))
                       for m_name, values in loaded_projections.items()}

    with torch.no_grad():
        for m_name, (P, mu_in, mu_out) in projections.items():
            if int(m_name.split('.')[2]) not in hparams.layers:
                continue

            orig_module = nethook.get_module(model, m_name)
            new_module = apply_dama_on_module(orig_module, P, mu_in, mu_out, hparams.projection_location)

            nethook.replace_module(model, m_name, new_module)

        print(f"New weights successfully inserted into layers: {hparams.layers}")

    return model


def run_evaluation_on_task(model, tokenizer, task, test_file, output_dir):
    if task == "gen":
        evaluator = EvaluateGeneration(model, tokenizer, os.path.join(DATA_DIR, test_file), task)
    elif task == "coref":
        evaluator = EvaluateCoreference(model, tokenizer, os.path.join(DATA_DIR, args.test_file), task)
    elif task == "causal_lm":
        evaluator = EvaluateCausalLM(model, tokenizer, test_file, task)
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

    model_name, model, _, tok = get_model_tokenizer(args.model_name, args.param_number, False)

    experiment_name_suffix = parse_experiment_name(
        num_layers=args.num_layers, iterative_update=args.iterative_update, mixed_update=args.mixed_update,
        task=args.task,
        post_linear=args.post_linear, batch_size=args.batch_size, orthogonal_constraint=args.orthogonal_constraint,
        no_colinear_vs=args.no_colinear_vs, vs_at_last=args.vs_at_last, null_dim=args.null_dim, use_neutral=args.use_neutral,
        delta_only=args.delta_only, nw=args.no_whitening
    )
    experiment_name = f"{experiment_name_suffix}"
    if args.method == "DAMA":
        print(f"Evaluating DAMA model {experiment_name}")
        output_dir = os.path.join(RESULTS_DIR, args.method, model_name, experiment_name)
        hparams = DAMAHyperParams.from_json(os.path.join(output_dir, "hparams.json"))
        projection_file = os.path.join(output_dir, "projections.npy")
        model = load_dama_model(model, hparams, projection_file)
    elif args.method == "ROME":
        print(f"Evaluating ROME model {experiment_name}")
        raise NotImplementedError("ROME evaluation is not yet implemented")
    elif args.method == None:
        print(f"Evaluating original model {model_name}")
        output_dir = os.path.join(RESULTS_DIR, "original",model_name)
        os.makedirs(output_dir, exist_ok=True)
    else:
        raise ValueError(f"Unknown method {args.method}")

    run_evaluation_on_task(model, tok, args.test_task, args.test_file, output_dir)

