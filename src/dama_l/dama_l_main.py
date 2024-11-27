import copy
from copy import deepcopy
from typing import Dict, List, Tuple, Any
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np

from utils.globals import *
from utils import nethook
from utils.repr_tools import get_module_input_output_at_words

from dama.compute_us_dama import compute_us
from dama.compute_v_dama import compute_v_dama, print_vs_stats
from .dama_l_hparams import DAMALeaceHyperParams

from dama.dama_main import apply_dama_on_module, get_context_templates

from tqdm import tqdm
import json

from concept_erasure import LeaceEraser
from concept_erasure_dual import LeaceEraserDual

CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def load_vs(projections_loadfrom, device):
    vs_loadfrom = os.path.join(os.path.dirname(projections_loadfrom), f"vs.npy")

    print(f"Loading vs from {vs_loadfrom}")
    try:
        vs = np.load(vs_loadfrom, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Could not find vs file at {vs_loadfrom}")
        vs = None

    if vs is not None:
        if torch.cuda.is_available() and device != 'cpu':
            vs = {g_val: torch.tensor(v['vs'], device=device, dtype=torch.float16) for g_val, v in vs.items()}
        else:
            vs = {g_val: torch.tensor(v['vs'], device=device, dtype=torch.float32) for g_val, v in vs.items()}
    return vs


def save_vs(vs, projections_saveto):
    vs_saveto = os.path.join(os.path.dirname(projections_saveto), f"vs.npy")

    print(f"Saving vs to {vs_saveto}")
    serializable_vs = {
        g_val: {"vs": vs.cpu().numpy()}
        for g_val, vs in vs.items()}

    np.save(vs_saveto, serializable_vs)


def save_projections(projections, projections_saveto):
    print(f"Saving projections to {projections_saveto}")
    serializable_projections = {
        m_name: {"M": M.cpu().numpy(), "mu_in": mu_in.cpu().numpy(), "mu_out": mu_out.cpu().numpy()}
        for m_name, (M, mu_in, mu_out) in projections.items()}

    np.save(projections_saveto, serializable_projections)


def save_latent(latents, latent_saveto):
    print(f"Saving latent variables to {latent_saveto}")
    serializable_latent = {
        layer: {"U_stereo": H_left.cpu().numpy(), "V_stereo": H_right.cpu().numpy(),
                "U_factual": H_left_fact.cpu().numpy(), "V_factual": H_right_fact.cpu().numpy()}
        for layer, (H_left, H_right, H_left_fact, H_right_fact) in latents.items()}
    np.save(latent_saveto, serializable_latent)


def batch_requests(requests: List[Dict], batch_size: int) -> List[List[Dict]]:
    if batch_size > 1:
        return [requests[i:i + batch_size] for i in range(0, len(requests), batch_size)]
    elif batch_size == 1:
        return [[request] for request in requests]
    else:
        raise ValueError("Batch size must be positive")


def get_vs_from_requests(model, tok, requests, hparams, context_templates, weights, cur_device):

    if "targets" in requests[0][0]:
        gender_values = list(requests[0][0]["targets"].keys())
    else:
        gender_values = ['pos', 'neg', 'neut']

    target_list = {g_val: [] for g_val in gender_values}
    v_layer = hparams.v_loss_layer

    v_dim = weights[f"{hparams.rewrite_module_tmp.format(v_layer)}"].shape[0]

    past_deltas = {g_val: torch.zeros((len(requests), v_dim), device=cur_device) for g_val in gender_values}
    past_deltas_normed = {g_val: torch.zeros((len(requests), v_dim), device=cur_device) for g_val in
                          gender_values}
    for bidx, request in enumerate(tqdm(requests, desc="Gathering targets from requests")):
        taregets, deltas = compute_v_dama(
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
            past_deltass=None,
            value_at_mlp=False
        )
        for g_val in gender_values:
            target_list[g_val].append(taregets[g_val])
            past_deltas_normed[g_val][bidx, :] = (deltas[g_val] / torch.norm(deltas[g_val])).detach().clone()
            past_deltas[g_val][bidx, :] = deltas[g_val].detach().clone()

    req_contexts = [context_templates[0].format(request["prompt"]) for request_batch in requests for request in
                    request_batch]

    req_words = [request["subject"] for request_batch in requests for request in request_batch]

    cur_vs = get_module_input_output_at_words(
        model, tok, req_contexts, req_words, v_layer, hparams.layer_module_tmp, hparams.fact_token)[1]
    if torch.cuda.is_available():
        cur_vs = cur_vs.to(cur_device).half()
        targetss = {g_val: past_delta.to(cur_device).half() + cur_vs for g_val, past_delta in
                    past_deltas.items()}
    else:
        targetss = {g_val: past_delta + cur_vs for g_val, past_delta in past_deltas.items()}
    print_vs_stats(targetss, cur_vs)

    return past_deltas


def apply_dama_l_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: DAMALeaceHyperParams,
        requests_fact: List[Dict] = None,
        copy=False,
        return_orig_module=False,
        projections_saveto=None,
        projections_loadfrom=None,
        latent_saveto=None,
        output_dir=None
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
    devices = {f"{hparams.rewrite_module_tmp.format(layer)}": nethook.get_parameter(model,
                                                                                    f"{hparams.rewrite_module_tmp.format(layer)}.weight").device
               for layer in hparams.layers}

    # Dama is applied for all requests together to improve generatlzation
    if projections_loadfrom is not None:
        print(f"Loading projections from {projections_loadfrom}")
        try:
            loaded_projections = np.load(projections_loadfrom, allow_pickle=True).item()
        except FileNotFoundError:
            print(f"Could not find projections file at {projections_loadfrom}")
        else:
            if torch.cuda.is_available():
                projections = {
                    m_name: (torch.tensor(values['M'], device=devices.get(m_name, 'cuda'), dtype=torch.float16),
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

    if len(projections) < len(hparams.layers) or projections_loadfrom is None:
        projections = execute_dama_l(model, tok, requests, hparams,
                                     requests_fact=requests_fact, projections_saveto=projections_saveto,
                                     projections_loadfrom=projections_loadfrom, latent_saveto=latent_saveto,
                                     old_projections=projections)

    if output_dir is not None:
        with open(sys.argv[0], 'r') as this_code, open(os.path.join(output_dir, 'dama_main.py'), 'w') as source_out:
            code_lines = this_code.readlines()
            source_out.writelines(code_lines)

    if projections_saveto is not None:
        save_projections(projections, projections_saveto)
    return model, module_copy


def execute_dama_l(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: DAMALeaceHyperParams,
        requests_fact: List[Dict] = None,
        projections_loadfrom: str = None,
        projections_saveto: str = None,
        latent_saveto: str = None,
        old_projections: Dict = None
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:

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

    latents = {}
    projections = {}

    # compute v targets for each request
    requests = batch_requests(requests, hparams.batch_size)
    if requests_fact is not None:
        requests_fact = batch_requests(requests_fact, hparams.batch_size)

    cur_device = next(model.parameters()).device

    Vs = None
    if projections_loadfrom is not None:
        Vs = load_vs(projections_loadfrom, 'cpu')
    if Vs is None:
        Vs = get_vs_from_requests(model, tok, requests, hparams, context_templates, weights, cur_device)
        if projections_saveto is not None:
            save_vs(Vs, projections_saveto)

    Vs_fact = None
    if requests_fact is not None:
        Vs_fact = get_vs_from_requests(model, tok, requests_fact, hparams, context_templates, weights, cur_device)

    for layer in sorted(hparams.layers):
        print(f"\n\nLAYER {layer}\n")
        module_name = f"{hparams.rewrite_module_tmp.format(layer)}"
        if module_name in old_projections:
            projections[module_name] = old_projections[module_name]
        else:
            cur_device = weights[f"{hparams.rewrite_module_tmp.format(layer)}"].device

            U = compute_us(model, tok, requests, hparams, layer, context_templates, device='cpu')
            if requests_fact is not None:
                U_fact = compute_us(model, tok, requests_fact, hparams, layer, context_templates, device='cpu')
            else:
                U_fact = None

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            V = torch.cat(list(Vs.values()), dim=0)
            U = torch.cat([U] * len(Vs), dim=0)

            if torch.cuda.is_available():
                U = U.float()

            with torch.no_grad():
                module_name = f"{hparams.rewrite_module_tmp.format(layer)}"
                W = weights[module_name].detach().cpu().numpy().astype(np.float32)

            h_dim = V.shape[1]
            H_left = U @ W.T
            H_right = V

            if Vs_fact is not None and U_fact is not None:
                print("Computing factual keys and values for signal preservation... [TODO: FIX ERASURE CODE]")
                V_fact = torch.cat(list(Vs_fact.values()), dim=0)
                U_fact = torch.cat([U_fact] * len(Vs_fact), dim=0)

                if torch.cuda.is_available():
                    U_fact = U_fact.float()

                H_left_fact = U_fact @ W.T
                H_right_fact = V_fact
                eraser = LeaceEraserDual.fit(H_left, H_left_fact, H_right, H_right_fact,
                                             b_to_f_ratio_thr=hparams.b_to_f_ratio_thr)
            else:
                H_left_fact = None
                H_right_fact = None
                eraser = LeaceEraser.fit(H_left, H_right)

            latents[layer] = (H_left, H_right, H_left_fact, H_right_fact)

            M = eraser.P

            mu_in = np.zeros(U.shape[1])
            mu_out = np.zeros(V.shape[1])

            ## Diff from identity matrix
            print(f"Diff from identity matrix: {np.linalg.norm(M - np.eye(h_dim, h_dim))}")

            # save as tensors
            if torch.cuda.is_available():
                M = torch.tensor(M, dtype=torch.float16, device=cur_device)
                mu_in = torch.tensor(mu_in, dtype=torch.float16, device=cur_device)
                mu_out = torch.tensor(mu_out, dtype=torch.float16, device=cur_device)
            else:
                M = torch.tensor(M, dtype=torch.float32, device='cpu')
                mu_in = torch.tensor(mu_in, dtype=torch.float32, device='cpu')
                mu_out = torch.tensor(mu_out, dtype=torch.float32, device='cpu')

            projections[module_name] = (M, mu_in, mu_out)

            if torch.cuda.is_available():
                print("Cleaning up torch...")
                del U, V, W, H_left, H_right
                torch.cuda.empty_cache()

            if projections_saveto is not None:
                print(f"Saving projections after LAYER {layer} to {projections_saveto}")
                save_projections(projections, projections_saveto)

            if latent_saveto is not None:
                print(f"Saving latents after LAYER {layer} to {latent_saveto}")
                save_latent(latents, latent_saveto)

        M, mu_in, mu_out = projections[module_name]
        with torch.no_grad():

            orig_module = nethook.get_module(model, module_name)
            new_module = apply_dama_on_module(orig_module, M, mu_in, mu_out, "after")

            nethook.replace_module(model, module_name, new_module)

        print(f"New weights successfully inserted into {module_name}.")
        if torch.cuda.is_available():
            print("Cleaning up projections...")
            M = M.detach().cpu()
            mu_in = mu_in.detach().cpu()
            mu_out = mu_out.detach().cpu()
            torch.cuda.empty_cache()
    print(f"Projections successfully computed for layer {list(projections.keys())}")
    return projections

