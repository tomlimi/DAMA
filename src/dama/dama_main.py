import copy
from copy import deepcopy
from typing import Dict, List, Tuple, Any
import os
import sys

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

    # Apply DAME on the module
    new_mlp= copy.copy(old_mlp)

    if projection_location == "before":
        new_mlp.weight = torch.nn.Parameter(old_mlp.weight @ P)
    elif projection_location == "after":
        new_mlp.weight = torch.nn.Parameter(P @ old_mlp.weight)
    else:
        raise ValueError("projection_location must be either 'before' or 'after'")
    in_bias = AddBias(-mu_in)
    out_bias = AddBias(mu_out)

    return torch.nn.Sequential(in_bias, new_mlp, out_bias)


# TODO - INLP functions store together with INLP code
def get_colspace_projection(W: np.ndarray) -> np.ndarray:
    """
    :param W: the matrix over its nullspace to project
    :return: the projection matrix over the rowspace
    """

    if np.allclose(W, 0):
        w_basis = np.zeros_like(W)
    else:
        w_basis = scipy.linalg.orth(W)  # orthogonal basis

    print("Orthogonal basis size:", w_basis.shape)
    w_basis * np.sign(w_basis[0][0])  # handle sign ambiguity
    P_W = w_basis.dot(w_basis.T)  # orthogonal projection on W's rowspace


    return P_W


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
    val=False
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

    # Dama is applied for all requests together to improve generatlzation
    if projections_loadfrom is not None:
        print(f"Loading projections from {projections_loadfrom}")
        loaded_projections = np.load(projections_loadfrom, allow_pickle=True).item()
        if torch.cuda.is_available():
            projections = { m_name: (torch.tensor(values['M'], device='cuda', dtype=torch.float16),
                                     torch.tensor(values['mu_in'], device='cuda', dtype=torch.float16),
                                     torch.tensor(values['mu_out'], device='cuda', dtype=torch.float16))
                            for m_name, values in loaded_projections.items()}
        else:
            projections = {m_name: (torch.tensor(values['M'], device='cpu', dtype=torch.float32),
                                    torch.tensor(values['mu_in'], device='cpu', dtype=torch.float32),
                                    torch.tensor(values['mu_out'], device='cpu', dtype=torch.float32))
                            for m_name, values in loaded_projections.items()}
    else:
        projections = execute_dama(model, tok, requests, hparams, ncv=ncv, val=val)

    if hparams.update == 'once' or projections_loadfrom is not None:
        with torch.no_grad():
            for m_name, (P, mu_in, mu_out) in projections.items():
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
        print(f"Saving projections to {projections_saveto}")
        serializable_projections = {
            m_name: {"M": M.cpu().numpy(), "mu_in": mu_in.cpu().numpy(),"mu_out": mu_out.cpu().numpy()}
            for m_name, (M, mu_in, mu_out) in projections.items()}

        np.save(projections_saveto, serializable_projections)
        # with open(projections_saveto, "w") as f:
        #     json.dump(serializable_projections, f, indent=4)
    return model, module_copy


def execute_dama(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: Dict,
        hparams: DAMAHyperParams,
        ncv: bool = False,
        val: bool = False,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

    # # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
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

    if hparams.update == 'mixed':
        target_pos_list = []
        target_neg_list = []
        v_layer = hparams.v_loss_layer if val else hparams.layers[-1]
        for request in tqdm(requests, desc="Gathering targets from requests"):
            targets = compute_v_dama(
                    model,
                    tok,
                    request,
                    hparams,
                    v_layer,
                    get_context_templates(model, tok, hparams.context_template_length_params),
                    compute_right_vector=False
                )
            target_pos_list.append(targets[0])
            target_neg_list.append(targets[1])

        targets_pos = torch.stack(target_pos_list)
        targets_neg = torch.stack(target_neg_list)

        req_contexts = [request["prompt"] for request_batch in requests for request in request_batch]
        req_words = [request["subject"] for request_batch in requests for request in request_batch]


    for layer in sorted(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        U = compute_us(model, tok, requests, hparams, layer,
                              get_context_templates(model, tok, hparams.context_template_length_params))

        if hparams.update == 'mixed':
            cur_vs = get_module_input_output_at_words(
                model, tok, req_contexts, req_words, v_layer, hparams.rewrite_module_tmp, hparams.fact_token)[1]

            V_pos = targets_pos - cur_vs
            V_neg = targets_neg - cur_vs
            print_vs_stats(V_pos, V_neg, cur_vs)

            if ncv:
                V = torch.cat([V_pos, V_neg], dim=0)
                U = torch.cat([U, U], dim=0)
            else:
                V = V_pos - V_neg

        else:
            v_pos_list = []
            v_neg_list = []
            for request in tqdm(requests, desc="Gathering targets from requests"):
                targets = compute_v_dama(
                    model,
                    tok,
                    request,
                    hparams,
                    hparams.v_loss_layer if val else layer,
                    get_context_templates(model, tok, hparams.context_template_length_params),
                    compute_right_vector=True
                )
                v_pos_list.append(targets[0])
                v_neg_list.append(targets[1])

            if ncv:
                V = torch.stack(v_pos_list + v_neg_list)
                U = torch.cat([U, U], dim=0)
            else:
                V = torch.stack(v_pos_list) - torch.stack(v_neg_list)



        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # normalize contrast vectors
        # V /= np.linalg.norm(V, axis=1, keepdims=True)
        # From MEMIT paper
        force_recompute = False
        cov = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            force_recompute=force_recompute,
        )

        if torch.cuda.is_available():
            U = U.float()

        print("Solving withening equation for U...")
        U = torch.linalg.solve(
            hparams.mom2_update_weight * cov + U.T @ U,
            U.T
        ).T

        # V /= (len(hparams.layers) * hparams.mom2_update_weight)


        with torch.no_grad():
            module_name = f"{hparams.rewrite_module_tmp.format(layer)}"
            # detach the weights from the graph and convert to numpy (float32)
            U = U.detach().cpu().numpy().astype(np.float32)
            V = V.detach().cpu().numpy().astype(np.float32)

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
        pls = PLSRegression(n_components=hparams.nullspace_dimension, scale=False)
        pls.fit(H_left, H_right)

        print("Computing nullspace projection...")

        print("B shape:", pls.x_weights_.shape)
        print(pls.x_weights_)
        B = pls.x_weights_[:, :hparams.nullspace_dimension]  # not needed but maybe useful to get some statistics
        # getting column space projections of B
        M = np.eye(h_dim, h_dim) - get_colspace_projection(B)

        # TODO: maybe use global statistics to compute mu_s
        if hparams.projection_location == "before":
            mu_in = pls._x_mean
            mu_out = W @ pls._x_mean

        elif hparams.projection_location == "after":
            mu_in = pls._x_mean @ np.linalg.pinv(W).T
            mu_out = pls._x_mean

        print(f"Nullspace projection values: {M}")

        ## Diff from identity matrix
        print(f"Diff from identity matrix: {np.linalg.norm(M - np.eye(h_dim, h_dim))}")
        ### mu vectors
        print(f"Input centralization vector (mu_in): {mu_in}")
        print(f"Output de-centralization vector (mu_out): {mu_out}")

        # save as tensors
        if torch.cuda.is_available():
            M = torch.tensor(M, dtype=torch.float16, device='cuda')
            mu_in = torch.tensor(mu_in, dtype=torch.float16, device='cuda')
            mu_out = torch.tensor(mu_out, dtype=torch.float16, device='cuda')
        else:
            M = torch.tensor(M, dtype=torch.float32, device='cpu')
            mu_in = torch.tensor(mu_in, dtype=torch.float32, device='cpu')
            mu_out = torch.tensor(mu_out, dtype=torch.float32, device='cpu')

        projections[module_name] = (M, mu_in, mu_out)

        if hparams.update == 'iterative' or hparams.update == 'mixed':
            with torch.no_grad():

                orig_module = nethook.get_module(model, module_name)
                new_module = apply_dama_on_module(orig_module, M, mu_in, mu_out, hparams.projection_location)

                nethook.replace_module(model, module_name, new_module)

            print(f"New weights successfully inserted into {module_name}.")


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

