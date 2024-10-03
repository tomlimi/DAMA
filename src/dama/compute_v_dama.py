import copy
import logging
import random
from typing import List, Dict, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import nethook, repr_tools

from .dama_hparams import DAMAHyperParams
from utils.repr_tools import get_module_input_output_at_words, find_fact_lookup_idx, get_module_input_output_at_words


# The signe of a pronoun was decided to be consistent with existing bias annotations
# e.g., positive values for male skewed words and negative for female ones.
LLAMA_PRONOUNS = {"pos": "he",
                 "neg": "she",
                 "neut": "they"}


def populate_prompts(prompts: List[str], fill_in: str, tok: AutoTokenizer, hparams) -> (List[str], List[int]):
    """
    Populates a prompt with a fill-in strings and return indices of the fill-ins.
    """

    prompts_filled = [prompt.format(fill_in) for prompt in prompts]
    prompts_lookup_idxs = [find_fact_lookup_idx(prompt, fill_in, tok, hparams.fact_token, verbose=(i == 0))
                           for i, prompt in enumerate(prompts)]

    return prompts_filled, prompts_lookup_idxs


def compute_v_dama(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request_batch: List[Dict],
    hparams,
    layer: int,
    context_templates: List[str],
    polarity_values: List[str],
    compute_right_vector: bool = False,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_id: int =0,
    past_deltass: dict[str, torch.Tensor] = None,
    value_at_mlp: bool = False,
    target_kl_regularization: bool = False
) -> Tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Computes the contrast value vector (`he` - `she`)for the projecting update.
    """

    print("Computing right vector (v)")

    # randomize training order
    if len(polarity_values) > 2:
        random.shuffle(polarity_values)

    target_idss = []
    rewriting_prompts = []
    kl_prompts = []

    rewriting_prompts_filled = []
    kl_prompts_filled = []

    rewriting_prompts_lookup_idxs = []
    kl_prompts_lookup_idxs = []

    rewriting_prompts_ridxs = []
    kl_prompts_ridxs = []

    lookup_idxs = []
    for rid, request in enumerate(request_batch):
        
        r_p = [context.format(request["prompt"]) for context in context_templates]
        r_p_filled, r_p_lookup_idxs = populate_prompts(r_p, request["subject"], tok, hparams)
        
        if target_kl_regularization and "shuffled_prefix" in request:
            kl_p = [tok.bos_token + " {}"]
            kl_p_filled, kl_p_lookup_idxs = populate_prompts(kl_p, request["shuffled_prefix"], tok, hparams)
        else:
            kl_p =  [tok.bos_token + " {} is a"]
            kl_p_filled, kl_p_lookup_idxs = populate_prompts(kl_p, request["subject"], tok, hparams)
            

        rewriting_prompts.extend(r_p)
        kl_prompts.extend(kl_p)

        rewriting_prompts_filled.extend(r_p_filled)
        kl_prompts_filled.extend(kl_p_filled)

        rewriting_prompts_lookup_idxs.extend(r_p_lookup_idxs)
        kl_prompts_lookup_idxs.extend(kl_p_lookup_idxs)

        rewriting_prompts_ridxs.extend([rid] * len(r_p))
        kl_prompts_ridxs.extend([rid] * len(kl_p))
        
        if "targets" in request:
            if '' in request["targets"].values():
                raise ValueError(f"Empty target value found in request: {request}")
            
            # unk_token is used to strip prefix_underline
            target_idss.extend([[tok(tok.bos_token + request["targets"][val], return_tensors="pt").to(device)["input_ids"][0][1:]
                               for val in polarity_values]] * len(r_p))
            targets_dict = request["targets"]
        else:
            target_idss.extend([[tok(LLAMA_PRONOUNS[val], return_tensors="pt").to(device)["input_ids"][0]
                                for val in polarity_values]] * len(r_p))
            targets_dict = LLAMA_PRONOUNS
            
    all_prompts = rewriting_prompts + kl_prompts

    input_tok= tok(
        rewriting_prompts_filled + kl_prompts_filled,
        return_tensors="pt",
        padding=True,
        return_token_type_ids=False
    ).to(device)

    lookup_idxs = rewriting_prompts_lookup_idxs + kl_prompts_lookup_idxs

    request_idx = torch.tensor(rewriting_prompts_ridxs + kl_prompts_ridxs)

    # Compute rewriting targets
    rewriting_targetss = [
        torch.tensor(-100, device=device).repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:])
        for _ in polarity_values ]

    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        for j in range(len(polarity_values)):
            rewriting_targetss[j][i, ex_len - len(target_idss[i][j]) : ex_len] = target_idss[i][j]
            
    target_lens = [torch.tensor([len(target_idss[i][j]) for i in range(len(rewriting_prompts))], device=device) for j in range(len(polarity_values))]
    
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Trying optimization objective to {loss_layer}...")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    deltas = [torch.zeros((model.config.hidden_size,), requires_grad=True,
                          device=device) for _ in polarity_values]

    delta_shared = torch.zeros((model.config.hidden_size,), requires_grad=True,
                               device=device)


    target_init, kl_distr_init = None, None

    # Insert deltas for computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        
        if cur_layer == hparams.mlp_module_tmp.format(layer):
            cur_out = cur_out.to(device)
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()

            for i, idx in enumerate(lookup_idxs):
                cur_out[i, idx, :] += (delta + delta_shared)
                # cur_out[i, idx, :] += delta
        elif cur_layer == hparams.layer_module_tmp.format(layer) and not value_at_mlp:

            cur_out = (cur_out[0].to(device), cur_out[1])
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()

            for i, idx in enumerate(lookup_idxs):
                cur_out[0][i, idx, :] += (delta + delta_shared)
        return cur_out

    # Optimizer
    opt = torch.optim.Adam(deltas + [delta_shared], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Optimmize
    print("Optimizing...")
    print("Loss structure: NLL + KL + WEIGHT DECAY + ORTHOGONALITY")

    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # iterate over positive and negative examples
        for target_len, rewriting_targets, delta, g_val in zip(target_lens, rewriting_targetss, deltas, polarity_values):
            past_deltas = past_deltass[g_val] if past_deltass else None
            # Forward propagation
            with nethook.TraceDict(
                module=model,
                layers=[
                    hparams.mlp_module_tmp.format(layer) if value_at_mlp else hparams.layer_module_tmp.format(layer),
                    hparams.layer_module_tmp.format(loss_layer)
                ],
                retain_input=False,
                retain_output=True,
                edit_output=edit_output_fn,
            ) as tr:
                logits = model(**input_tok).logits

                # Compute distribution for KL divergence
                kl_logits = torch.stack(
                    [
                        logits[i - len(kl_prompts), idx, :]
                        for i, idx in enumerate(lookup_idxs[-len(kl_prompts):])
                    ],
                    dim=0,
                )
                kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
                if kl_distr_init is None:
                    kl_distr_init = kl_log_probs.detach().clone()

            # Compute loss on rewriting targets
            log_probs = torch.log_softmax(logits, dim=2)

            loss = torch.gather(
                log_probs,
                2,
                torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
            ).squeeze(2)
            mask = (rewriting_targets != -100).float()

            # Aggregate total losses
            nll_loss_each = -(loss * mask).sum(1) / target_len
            nll_loss = nll_loss_each.mean()
            kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
                kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
            )

            weight_decay = hparams.v_weight_decay * torch.norm(delta + delta_shared) ** 2 / torch.norm(target_init)

            orthogonal_loss = torch.tensor(0.0)
            
            if hasattr(hparams, 'orthogonal_constraint') and hparams.orthogonal_constraint and past_deltas is not None and batch_id > 0:
                batch_id = torch.tensor(batch_id)
                delta_normed = delta / (torch.norm(delta) + 1e-8)
                orthogonal_loss = hparams.orthogonal_constraint * torch.norm(past_deltas[:batch_id,:] @ delta_normed) / torch.sqrt(batch_id)

            loss = nll_loss + kl_loss + weight_decay + orthogonal_loss

            print(
                f"loss ({g_val}) {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} + {np.round(orthogonal_loss.item(), 3)} "
                f"avg prob of [{targets_dict[g_val]}] "
                f"{torch.exp(-nll_loss_each).mean().item()}"
            )
            if loss < 5e-2:
                break

            if it == hparams.v_num_grad_steps - 1:
                break

            # Backpropagate
            # loss.requires_grad = True # just for debugging
            loss.backward()
            opt.step()

            # Project within L2 ball
            max_norm = hparams.clamp_norm_factor * target_init.norm()
            if (delta + delta_shared).norm() > max_norm:
                with torch.no_grad():
                    delta[...] = delta * max_norm / (delta + delta_shared).norm()
                    delta_shared[...] = delta_shared * max_norm / (delta + delta_shared).norm()

    targets = [target_init + delta + delta_shared for delta in deltas]
    # targets = [target_init + delta for delta in deltas]



    (_, cur_outputs) = get_module_input_output_at_words(
        model,
        tok,
        contexts=[context_templates[0].format(request["prompt"]) for request in request_batch],
        words=[request["subject"] for request in request_batch],
        layer=layer,
        module_template=hparams.mlp_module_tmp if value_at_mlp else hparams.layer_module_tmp,
        fact_token_strategy=hparams.fact_token
    )
    cur_output = cur_outputs.mean(0)
    
    if torch.cuda.is_available():
        targets = [target.to(device).half() for target in targets]
        delta_shared = delta_shared.to(device).half()
        cur_output = cur_output.to(device).half()

    rel_targets = [target - cur_output - delta_shared for target in targets]
    if torch.cuda.is_available():
        rel_targets = [rt.to(device).half() for rt in rel_targets ]

    return dict(zip(polarity_values, targets)), dict(zip(polarity_values, rel_targets))


def print_vs_stats(Vs, cur_out):

    # V_contrast = V_pos - V_neg

    print("For all numbers printed ")
    for g_val, V in Vs.items():
        print(f"{g_val} vector norm: {descriptive_stat([v.norm().item() for v in torch.unbind(V)])}")
        print(f"{g_val} delta norm: {descriptive_stat([(v-v_orig).norm().item() for v, v_orig in zip(torch.unbind(V), torch.unbind(cur_out))])}")

    deltas_normed = {g_val: (V - cur_out) / (V - cur_out).norm(dim=1, keepdim=True) for g_val, V in Vs.items()}
    cur_normed = cur_out / cur_out.norm(dim=1, keepdim=True)

    for g_val, D in deltas_normed.items():
        print(f"{g_val} delta cosine with original vec: {descriptive_stat([torch.dot(d, v_orig.T).item() for d, v_orig in zip(torch.unbind(D), torch.unbind(cur_normed))])}")
        print(f"Cosine across {g_val} deltas: {D @ D.T}")

    print(f"Cossine across all deltas: {torch.cat(list(deltas_normed.values()), dim=0) @ torch.cat(list(deltas_normed.values()), dim=0).T}")
    print("\n*******\n")


def descriptive_stat(data):
    return f"min: {min(data)} mean: {np.mean(data)} std: {np.std(data)} max: {max(data)}"

