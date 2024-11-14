import os, re, sys, json
import torch, numpy
import math
import argparse
from tqdm import tqdm

from collections import defaultdict
from causal_tracing.causal_trace import ModelAndTokenizer, guess_subject, calculate_hidden_flow, collect_embedding_std
from utils.knowns import KnownsDataset
from utils.globals import *

torch.set_grad_enabled(False)


def load_data(mt, data_path, noise_level=None):
    knowns = KnownsDataset(data_path)
    if not noise_level:
        noise_level = 3 * collect_embedding_std(mt, [k["subject"] for k in knowns])
    return knowns, noise_level


def compute_results(
        mt,
        prompt,
        subject=None,
        samples=10,
        noise=0.1,
        window=10,
        kind=None,
        modelname=None,
        savepdf=None,
        disable_mlp=False,
        disable_attn=False,
        project_embeddings=None
):
    if subject is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow(
        mt, prompt, subject, samples=samples, noise=noise, window=window, kind=kind,
        disable_mlp=disable_mlp, disable_attn=disable_attn,
        project_embeddings=project_embeddings
    )
    
    result = {field: value for field, value in result.items() if field in ("scores", "low_score", "high_score")}
    result["prompt"] = prompt
    result["subject"] = subject
    
    return result


def compute_save_gender_effects(result_path, mt, knowns, noise_level=0.08, cap_examples=1000,
                                disable_mlp=False, disable_attn=False,param_number=None, inlp_projection=None,
                                model_identifier=None):
    
    param_str = f"_{param_number}B" if param_number is not None else ''
    disable_str = f"{'_disable_mlp' if disable_mlp else ''}{'_disable_attn' if disable_attn else ''}"
    inlp_str = f"_inlp" if inlp_projection else ""
    model_identifier = f"_{model_identifier}" if model_identifier is not None else ""

    if cap_examples == -1:
        result_file = os.path.join(result_path,f"results_known{param_str}{disable_str}{inlp_str}{model_identifier}_all.jsonl")
        
    else:
        knowns.shuffle(seed=92)
        knowns = knowns[:cap_examples]
        result_file = os.path.join(result_path,f"results_known{param_str}{disable_str}{inlp_str}{model_identifier}_{cap_examples}.jsonl")

    if inlp_projection:
        if not os.path.exists(os.path.join(result_path,inlp_projection)):
            raise ValueError(f"Projection file {inlp_projection} does not exist")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        npzfile = numpy.load(os.path.join(result_path,inlp_projection))
        project_embeddings = {}
        for arr_name in {'P', 'mu', 's'}:
            if arr_name not in npzfile:
                raise ValueError(f"Projection file {inlp_projection} does not contain array {arr_name}!")
            project_embeddings[arr_name] = torch.from_numpy(npzfile[arr_name]).type(torch.float16 if torch.cuda.is_available() else torch.float32).to(device)
    else:
        project_embeddings = None


    results = {}
    with open(result_file, "w") as f:
        for knowledge in tqdm(knowns):
            results['prompt'] = knowledge['prompt']
            results['subject'] = knowledge['subject']
            for kind in [None, "mlp", "attn"]:
                out = compute_results(mt, knowledge["prompt"], knowledge["subject"], noise=noise_level, kind=kind,
                                      disable_mlp=disable_mlp, disable_attn=disable_attn, project_embeddings=project_embeddings)
                # convert torch tensors to json serializable lists
                results[kind] = {k: v.tolist() for k, v in out.items() if k in ("scores", "low_score", "high_score")}
            f.write(json.dumps(results) + "\n")
            f.flush()
            

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--model_name_path", type=str, default="/home/limisiewicz/my-luster/dama/models/llama")
    argparse.add_argument("--param_number", type=int, default=None)
    argparse.add_argument("--noise_level", type=float, default=0.06)
    argparse.add_argument("--cap_examples", type=int, default=1000)
    argparse.add_argument("--disable_mlp", action="store_true", default=False)
    argparse.add_argument("--disable_attn", action="store_true", default=False)
    argparse.add_argument("--inlp_projection", type=str, default=None)
    args = argparse.parse_args()

    model_name = args.model_name_path
    model_identifier = ""
    if model_name.endswith("llama"):
        if args.param_number in {7, 13, 30, 65}:
            model_name += f"_{args.param_number}B"
            model_identifier = "llama"
    else:
        model_identifier = model_name.split('/')[-1].replace('-','_')

    # load model and split over multiple gpus if necessary
    mt = ModelAndTokenizer(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                           low_cpu_mem_usage=False)
    knowns, noise_level = load_data(mt, DATA_DIR, args.noise_level)
    
    compute_save_gender_effects(RESULTS_DIR, mt, knowns, noise_level, args.cap_examples, args.disable_mlp, args.disable_attn,
                                param_number=args.param_number, inlp_projection=args.inlp_projection,
                                model_identifier=model_identifier)
    
    
    
    
