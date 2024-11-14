import argparse
import transformers
from typing import Dict
from huggingface_hub import PyTorchModelHubMixin

from dama.dama_hparams import DAMAHyperParams
from dama_l.dama_l_hparams import DAMALeaceHyperParams


from utils.globals import *
from utils.model_utils import *


def upload_model_to_hf(model: transformers.AutoModelForCausalLM,  tok: transformers.AutoTokenizer,
                       config: Dict, hf_name: str):
    tok.push_to_hub(hf_name, config=config)
    model.push_to_hub(hf_name, config=config)
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama")
    parser.add_argument("--param_number", type=int, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--hf_name", type=str, required=True)
    # Experiment parameters
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
    config = {"model_name": model_name, "model_size": args.param_number ,
              "num_layers": args.num_layers, "adaptation_method": args.method}
    
    if args.method == "DAMA":
        print(f"Evaluating DAMA model {experiment_name}")
        output_dir = os.path.join(RESULTS_DIR, args.method, model_name, experiment_name)
        hparams = DAMAHyperParams.from_json(os.path.join(output_dir, "hparams.json"))
        projection_file = os.path.join(output_dir, "projections.npy")
        model = load_dama_model(model, hparams, projection_file)
        config["null_dim"] = args.null_dim

    elif args.method == "DAMA_L":
        print(f"Evaluating DAMA Leace model {experiment_name}")
        output_dir = os.path.join(RESULTS_DIR, args.method, f"{model_name}_{str(args.num_layers)}L")
        hparams = DAMALeaceHyperParams.from_json(os.path.join(output_dir, "hparams.json"))
        projection_file = os.path.join(output_dir, "projections.npy")
        model = load_dama_model(model, hparams, projection_file)
        
    else:
        raise ValueError(f"Unknown method {args.method}")

    upload_model_to_hf(model, tok, config, args.hf_name)