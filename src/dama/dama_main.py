import copy
from copy import deepcopy
from typing import Dict, List, Tuple, Any
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np
import scipy
from sklearn.cross_decomposition import PLSRegression

from utils.generate import generate_fast
from utils.globals import *
from utils import nethook

from .compute_us_dama import compute_us, get_module_input_output_at_words
from .compute_v_dama import compute_v_dama, print_vs_stats
from .dama_hparams import DAMAHyperParams

from utils.layer_stats import layer_stats

from tqdm import tqdm
import json

CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

class AddBias(torch.nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = bias

    def forward(self, x):
        return x + self.bias


def apply_dama_on_module(old_mlp, P, mu_in, mu_out, projection_location):

    # Apply DAMA on the module

    if projection_location == "before":
        old_mlp.weight = torch.nn.Parameter(old_mlp.weight @ P)
    elif projection_location == "after":
        old_mlp.weight = torch.nn.Parameter(P @ old_mlp.weight)
    else:
        raise ValueError("projection_location must be either 'before' or 'after'")

    return old_mlp


# TODO - INLP functions store together with INLP code
def get_colspace_projection(W: np.ndarray) -> np.ndarray:
    """
    :param W: the matrix over its nullspace to project
    :return: the projection matrix over the rowspace
    """

    if np.allclose(W, 0):
        w_basis = np.zeros_like(W)
    else:
        w_basis = custom_orth(W)  # orthogonal basis

    print("Orthogonal basis size:", w_basis.shape)
    w_basis * np.sign(w_basis[0][0])  # handle sign ambiguity
    P_W = w_basis.dot(w_basis.T)  # orthogonal projection on W's rowspace


    return P_W


# Orthonormal decomposition

def custom_orth(A):
    """
    Custom reimplementation of scipy.linalg.orth with a different SVD algorithm (gesvd instead of gesdd)

    """
    u, s, vh = scipy.linalg.svd(A, full_matrices=False, lapack_driver='gesvd')
    M, N = u.shape[0], vh.shape[1]

    rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = u[:, :num]
    return Q


def save_projections(projections, projections_saveto, use_neutral=False):
    print(f"Saving projections to {projections_saveto}")
    serializable_projections = {
        m_name: {"M": M.cpu().numpy(), "mu_in": mu_in.cpu().numpy(), "mu_out": mu_out.cpu().numpy()}
        for m_name, (M, mu_in, mu_out, _, _, _) in projections.items()}
    
    np.save(projections_saveto, serializable_projections)
    
    if use_neutral:
        neutral_tensor_saveto = projections_saveto.replace("projections", "neutral")
        print(f"Saving neutral vectors to {neutral_tensor_saveto}")
        serializable_neutral = {
            m_name: {"mu_neutral": mu_neutral.cpu().numpy(),
                     "mu_pos": mu_pos.cpu().numpy(),
                     "mu_neg": mu_neg.cpu().numpy()}
            for m_name, (_, _, _, mu_neutral, mu_pos, mu_neg) in projections.items()}
        np.save(neutral_tensor_saveto, serializable_neutral)
        

def get_param_string(hparams):
    return  "_".join(["ft", str(hparams.fact_token), "vngs", str(hparams.v_num_grad_steps), "vlr", str(hparams.v_lr),
                             "l2", str(hparams.v_weight_decay),"cnf", str(hparams.clamp_norm_factor),
                             "bs", str(hparams.batch_size), "oc", str(hparams.orthogonal_constraint)])

def save_vs(vs, hparams, projections_saveto):
    
    param_string = get_param_string(hparams)
    parent_dir = os.path.dirname(os.path.dirname(projections_saveto))
    vs_saveto = os.path.join(parent_dir, f"vs_{param_string}.npy")
    
    print(f"Saving vs to {vs_saveto}")
    serializable_vs = {
        g_val: {"vs": vs.cpu().numpy()}
        for g_val, vs in vs.items()}
    
    np.save(vs_saveto, serializable_vs)


def load_vs(hparams, projections_loadfrom, device):

    param_string = get_param_string(hparams)
    parent_dir = os.path.dirname(os.path.dirname(projections_loadfrom))
    vs_loadfrom = os.path.join(parent_dir, f"vs_{param_string}.npy")
    
    print(f"Loading vs from {vs_loadfrom}")
    try:
        vs = np.load(vs_loadfrom, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Could not find vs file at {vs_loadfrom}")
        vs = None
        
    if vs is not None:
        if torch.cuda.is_available() and device !='cpu':
            vs = {g_val: torch.tensor(v['vs'], device=device, dtype=torch.float16) for g_val, v in vs.items()}
        else:
            vs = {g_val: torch.tensor(v['vs'], device=device, dtype=torch.float32) for g_val, v in vs.items()}
    return vs

def apply_dama_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: DAMAHyperParams,
    copy=False,
    return_orig_module=False,
    projections_saveto=None,
    projections_loadfrom=None,
    output_dir=None,
    ncv=False,
    val=False,
    use_neutral=False
) -> tuple[AutoModelForCausalLM | AutoModelForCausalLM, dict[str, Any]]:

    """
    Returns a model with the desired changes.

    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.

    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    if copy:
        model = deepcopy(model)

    module_copy = {}
    projections = {}
    devices = { f"{hparams.rewrite_module_tmp.format(layer)}": nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(layer)}.weight").device for layer in hparams.layers}

    # Dama is applied for all requests together to improve generatlzation
    if projections_loadfrom is not None:
        print(f"Loading projections from {projections_loadfrom}")
        try:
            loaded_projections = np.load(projections_loadfrom, allow_pickle=True).item()
        except FileNotFoundError:
            print(f"Could not find projections file at {projections_loadfrom}")
        else:
            if torch.cuda.is_available():
                projections = { m_name: (torch.tensor(values['M'], device=devices.get(m_name, 'cuda'), dtype=torch.float16),
                                         torch.tensor(values['mu_in'], device=devices.get(m_name, 'cuda'), dtype=torch.float16),
                                         torch.tensor(values['mu_out'], device=devices.get(m_name, 'cuda'), dtype=torch.float16),
                                         None, None, None)
                                for m_name, values in loaded_projections.items()}
            else:
                projections = {m_name: (torch.tensor(values['M'], device='cpu', dtype=torch.float32),
                                        torch.tensor(values['mu_in'], device='cpu', dtype=torch.float32),
                                        torch.tensor(values['mu_out'], device='cpu', dtype=torch.float32),
                                        None, None, None)
                                for m_name, values in loaded_projections.items()}
            if use_neutral:
                nutral_vector_loadfrom = projections_loadfrom.replace("projections", "neutral")
                loaded_neutral = np.load(nutral_vector_loadfrom, allow_pickle=True).item()
                if torch.cuda.is_available():
                    neutral_tensor = {m_name: torch.tensor(values['mu_neutral'], device=devices.get(m_name, 'cuda'), dtype=torch.float16)
                                        for m_name, values in loaded_neutral.items()}
                    pos_tensor = {m_name: torch.tensor(values['mu_pos'], device=devices.get(m_name,'cuda'), dtype=torch.float16)
                                        for m_name, values in loaded_neutral.items()}
                    neg_tensor = {m_name: torch.tensor(values['mu_neg'], device=devices.get(m_name, 'cuda'), dtype=torch.float16)
                                        for m_name, values in loaded_neutral.items()}
                    
                else:
                    neutral_tensor = {m_name: torch.tensor(values['mu_neutral'], device='cpu', dtype=torch.float32)
                                        for m_name, values in loaded_neutral.items()}
                    pos_tensor = {m_name: torch.tensor(values['mu_pos'], device='cpu', dtype=torch.float32)
                                        for m_name, values in loaded_neutral.items()}
                    neg_tensor = {m_name: torch.tensor(values['mu_neg'], device='cpu', dtype=torch.float32)
                                        for m_name, values in loaded_neutral.items()}
                for m_name in projections.keys():
                    projections[m_name] = projections[m_name][:3] + (neutral_tensor[m_name], pos_tensor[m_name], neg_tensor[m_name])
    if len(projections) < len(hparams.layers) or projections_loadfrom is None:
        projections = execute_dama(model, tok, requests, hparams, ncv=ncv, val=val, use_neutral=use_neutral,
                                   projections_saveto=projections_saveto, projections_loadfrom=projections_loadfrom,
                                   old_projections=projections)

    if hparams.update == 'once':
        with torch.no_grad():
            for m_name, (P, mu_in, mu_out, _, _, _) in projections.items():
                if int(m_name.split('.')[2]) not in hparams.layers:
                    continue

                orig_module = nethook.get_module(model, m_name)
                new_module = apply_dama_on_module(orig_module, P, mu_in, mu_out, hparams.projection_location)

                if return_orig_module and m_name not in module_copy:
                    module_copy[m_name] = deepcopy(orig_module)

                nethook.replace_module(model, m_name, new_module)

            print(f"New weights successfully inserted into layers: {hparams.layers}")

    if output_dir is not None:
        with open(sys.argv[0], 'r') as this_code, open(os.path.join(output_dir, 'dama_main.py'), 'w') as source_out:
            code_lines = this_code.readlines()
            source_out.writelines(code_lines)

    if projections_saveto is not None:
        save_projections(projections, projections_saveto, use_neutral=use_neutral)
    return model, module_copy


def execute_dama(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: Dict,
        hparams: DAMAHyperParams,
        ncv: bool = False,
        val: bool = False,
        use_neutral: bool = False,
        projections_loadfrom: str = None,
        projections_saveto: str = None,
        old_projections: Dict = None
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:

    gender_values = ['pos', 'neg', 'neut'] if use_neutral else ['pos', 'neg']
    context_templates = get_context_templates(model, tok, hparams.context_template_length_params)

    # # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }

    weights[f"{hparams.rewrite_module_tmp.format(hparams.v_loss_layer)}"] = nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(hparams.v_loss_layer)}.weight"
        )
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # # Save old weights for future restoration
    # weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute u and v
    # Update loop: sequentially intervene at each specified layer
    projections = {}

    # compute v targets for each request
    if hparams.batch_size > 1:
        requests = [requests[i:i + hparams.batch_size] for i in range(0, len(requests), hparams.batch_size)]
    elif hparams.batch_size == 1:
        requests = [[request] for request in requests]
    else:
        raise ValueError("Batch size must be positive")

    cur_device = next(model.parameters()).device
    
    if hparams.update == 'mixed':
        Vs = None
        if projections_loadfrom is not None:
            Vs = load_vs(hparams, projections_loadfrom, 'cpu')
        if Vs is None:
            target_list = {g_val: [] for g_val in gender_values}
            v_layer = hparams.v_loss_layer if val else hparams.layers[-1]
    
            v_dim = weights[f"{hparams.rewrite_module_tmp.format(v_layer)}"].shape[0]
            
    
            past_deltas = { g_val: torch.zeros((len(requests), v_dim ), device=cur_device) for g_val in gender_values}
            past_deltas_normed = { g_val: torch.zeros((len(requests), v_dim ), device=cur_device) for g_val in gender_values}
            for bidx, request in enumerate(tqdm(requests, desc="Gathering targets from requests")):
                taregets, deltas= compute_v_dama(
                        model,
                        tok,
                        request,
                        hparams,
                        v_layer,
                        context_templates,
                        gender_values,
                        compute_right_vector=False,
                        device=cur_device,
                        batch_id=bidx,
                        past_deltass=past_deltas_normed if hparams.orthogonal_constraint else None,
                        value_at_mlp=False
                    )
                for g_val in gender_values:
                    target_list[g_val].append(taregets[g_val])
                    past_deltas_normed[g_val][bidx,:] = (deltas[g_val] / torch.norm(deltas[g_val])).detach().clone()
                    past_deltas[g_val][bidx,:] = deltas[g_val].detach().clone()
    
            # targetss = {g_val: torch.stack(taregets) for g_val, taregets in target_list.items()}
            req_contexts = [context_templates[0].format(request["prompt"]) for request_batch in requests for request in request_batch]
            req_words = [request["subject"] for request_batch in requests for request in request_batch]
    
            cur_vs = get_module_input_output_at_words(
                model, tok, req_contexts, req_words, v_layer, hparams.layer_module_tmp, hparams.fact_token)[1]
            if torch.cuda.is_available():
                cur_vs = cur_vs.to(cur_device).half()
                targetss = {g_val: past_delta.to(cur_device).half() + cur_vs for g_val, past_delta in past_deltas.items()}
            else:
                targetss = {g_val: past_delta + cur_vs for g_val, past_delta in past_deltas.items()}
            print_vs_stats(targetss, cur_vs)
            
            Vs = past_deltas
            if projections_saveto is not None:
                save_vs(Vs, hparams, projections_saveto)

    for layer in sorted(hparams.layers):
        print(f"\n\nLAYER {layer}\n")
        module_name = f"{hparams.rewrite_module_tmp.format(layer)}"
        cur_device = weights[module_name].device
        if module_name in old_projections:
            projections[module_name] = old_projections[module_name]
        else:
            cur_device = weights[f"{hparams.rewrite_module_tmp.format(layer)}"].device
            # # Retrieve weights that user desires to change
            U = compute_us(model, tok, requests, hparams, layer, context_templates, device='cpu')
    
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
            if not hparams.update == 'mixed':
                v_lists = {g_val: [] for g_val in gender_values}
    
                v_dim = weights[f"{hparams.rewrite_module_tmp.format(layer)}"].shape[0]
    
                past_deltas = { g_val: torch.zeros((len(requests), v_dim ), device=cur_device) for g_val in gender_values}
                for bidx, request in enumerate(tqdm(requests, desc="Gathering targets from requests")):
                    _, deltas = compute_v_dama(
                        model,
                        tok,
                        request,
                        hparams,
                        hparams.v_loss_layer if val else layer,
                        context_templates,
                        gender_values,
                        compute_right_vector=True,
                        device=cur_device,
                        batch_id=bidx,
                        past_deltass=past_deltas if hparams.orthogonal_constraint else None,
                        value_at_mlp=True
                    )
    
                    for g_val in gender_values:
                        v_lists[g_val].append(deltas[g_val])
                        past_deltas[g_val][bidx,:] = (deltas[g_val] / torch.norm(deltas[g_val])).detach().clone()
    
                Vs = {g_val: torch.stack(v_list) for g_val, v_list in v_lists.items()}
    
            if ncv:
                V = torch.cat(list(Vs.values()), dim=0)
                U = torch.cat([U] * len(Vs), dim=0)
            else:
                V = Vs['pos'] - Vs['neg']
            if use_neutral:
                mu_neutral = Vs["neut"].mean(dim=0, keepdim=False)
                mu_neutral = mu_neutral.to(cur_device)
                
                mu_pos = Vs["pos"].mean(dim=0, keepdim=False)
                mu_pos = mu_pos.to(cur_device)
                
                mu_neg = Vs["neg"].mean(dim=0, keepdim=False)
                mu_neg = mu_neg.to(cur_device)
                
            else:
                mu_neutral = None
                mu_pos = None
                mu_neg = None
    
            if torch.cuda.is_available():
                U = U.float()
    
            if hparams.mom2_adjustment:
                force_recompute = False
                cov = get_cov(
                    model,
                    tok,
                    hparams.rewrite_module_tmp.format(layer),
                    hparams.mom2_dataset,
                    hparams.mom2_n_samples if not force_recompute else hparams.mom2_n_samples // 10,
                    hparams.mom2_dtype,
                    force_recompute=force_recompute,
                )
                cov = cov.to(cur_device)
    
    
                print("Solving withening equation for U...")
                start_t = time.time()
                U = torch.linalg.solve(
                    hparams.mom2_update_weight * cov + U.T @ U,
                    U.T
                ).T
                print(f"Done in {(time.time() - start_t)/60:.2f} minutes")
    
    
            with torch.no_grad():
                module_name = f"{hparams.rewrite_module_tmp.format(layer)}"
                W = weights[module_name].detach().cpu().numpy().astype(np.float32)
    
            if hparams.projection_location == "before":
                h_dim = U.shape[1]
                H_left = U
                H_right = V @ np.linalg.pinv(W).T
    
            elif hparams.projection_location == "after":
                h_dim = V.shape[1]
                H_left = U @ W.T
                H_right = V
            else:
                raise ValueError(f"Unknown projection location {hparams.projection_location}")
    
    
            # rethink how it should be done ...
    
            print("Left shape:", H_left.shape)
            print("Right shape:", H_right.shape)
            # compute PLS mapping between U and U_hat
            print("Computing PLS mapping...")
            
            pls = PLSRegression(n_components=hparams.nullspace_dimension, scale=False, tol=1e-4, max_iter=500, copy=False)
            start_t = time.time()
            pls.fit(H_left, H_right)
            print(f"PLS took {(time.time()- start_t)/60.:.2f} minutes")
            print("Computing nullspace projection...")
            start_t = time.time()
            print("B shape:", pls.x_weights_.shape)
            print(pls.x_weights_)
            B = pls.x_weights_[:, :hparams.nullspace_dimension]  # not needed but maybe useful to get some statistics
            # getting column space projections of B
            M = np.eye(h_dim, h_dim) - get_colspace_projection(B)
            print(f"Nullspace projection took {(time.time()- start_t)/60.:.2f} minutes")
    
            print("Computing normaizlization vectroes...")
            start_t = time.time()
            
            mu_in = np.zeros(U.shape[1])
            mu_out = np.zeros(V.shape[1])
            print(f"Normalization vectors took {(time.time()- start_t)/60.:.2f} minutes")
            print(f"Nullspace projection values: {M}")
    
            ## Diff from identity matrix
            print(f"Diff from identity matrix: {np.linalg.norm(M - np.eye(h_dim, h_dim))}")
            ### mu vectors
            print(f"Input centralization vector (mu_in): {mu_in}")
            print(f"Output de-centralization vector (mu_out): {mu_out}")
            if use_neutral:
                print(f"Neutral vector (mu_neutral): {mu_neutral}")
    
            # save as tensors
            if torch.cuda.is_available():
                M = torch.tensor(M, dtype=torch.float16, device=cur_device)
                mu_in = torch.tensor(mu_in, dtype=torch.float16, device=cur_device)
                mu_out = torch.tensor(mu_out, dtype=torch.float16, device=cur_device)
            else:
                M = torch.tensor(M, dtype=torch.float32, device='cpu')
                mu_in = torch.tensor(mu_in, dtype=torch.float32, device='cpu')
                mu_out = torch.tensor(mu_out, dtype=torch.float32, device='cpu')
    
            if use_neutral:
                if torch.cuda.is_available():
                    mu_neutral = torch.tensor(mu_neutral, dtype=torch.float16, device=cur_device)
                    mu_pos = torch.tensor(mu_pos, dtype=torch.float16, device=cur_device)
                    mu_neg = torch.tensor(mu_neg, dtype=torch.float16, device=cur_device)
                else:
                    mu_neutral = torch.tensor(mu_neutral, dtype=torch.float32, device='cpu')
                    mu_pos = torch.tensor(mu_pos, dtype=torch.float32, device='cpu')
                    mu_neg = torch.tensor(mu_neg, dtype=torch.float32, device='cpu')
            else:
                mu_neutral = None
                mu_pos = None
                mu_neg = None
    
            projections[module_name] = (M, mu_in, mu_out, mu_neutral, mu_pos, mu_neg)
            
            if torch.cuda.is_available():
                print("Cleaning up torch...")
                del U, V, W, H_left, H_right, B
                torch.cuda.empty_cache()
            
            if projections_saveto is not None:
                print(f"Saving projections after LAYER {layer} to {projections_saveto}")
                save_projections(projections, projections_saveto, use_neutral=use_neutral)
        
        if hparams.update == 'iterative' or hparams.update == 'mixed':
            M, mu_in, mu_out, mu_neutral, mu_pos, mu_neg = projections[module_name]
            with torch.no_grad():

                orig_module = nethook.get_module(model, module_name)
                new_module = apply_dama_on_module(orig_module, M, mu_in, mu_out, hparams.projection_location)

                nethook.replace_module(model, module_name, new_module)

            print(f"New weights successfully inserted into {module_name}.")
            if torch.cuda.is_available():
                print("Cleaning up projections...")
                M = M.detach().cpu()
                mu_in = mu_in.detach().cpu()
                mu_out = mu_out.detach().cpu()
                mu_neutral = mu_neutral.detach().cpu()
                mu_pos = mu_pos.detach().cpu()
                mu_neg = mu_neg.detach().cpu()
                torch.cuda.empty_cache()
    print(f"Projections successfully computed for layer {list(projections.keys())}")
    return projections


# TODO: share code with rome.compute_u
def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.split("/")[-1]
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    if torch.cuda.is_available():
        COV_CACHE[key] = COV_CACHE[key].to("cuda")
    return (
        torch.inverse(COV_CACHE[key]) if inv else COV_CACHE[key]
    )


# TODO: share code with rome_main
def get_context_templates(model, tok, length_params):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [tok.bos_token + " {}"] + [
            x + "." +  tok.eos_token + tok.bos_token + " {}"
            for x in sum(
                (
                    generate_fast(
                        model,
                        tok,
                        [""],
                        n_gen_per_prompt=n_gen,
                        max_out_len=length,
                    )
                    for length, n_gen in length_params
                ),
                [],
            )
            if '{' not in x and '}' not in x
        ]

        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE

