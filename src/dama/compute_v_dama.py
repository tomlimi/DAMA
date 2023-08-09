import copy
import logging
from typing import List, Dict, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import nethook, repr_tools

from .dama_hparams import DAMAHyperParams


# The signe of a pronoun was decided to be consistent with existing bias annotations
# e.g., positive values for male skewed words and negative for female ones.
LLAMA_PRONOUNS = {"pos": "he",
                 "neg": "she"}

# TODO: Support request batching
def compute_v_dama(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: List[Dict],
    hparams: DAMAHyperParams,
    layer: int,
    context_templates: List[str],
    compute_right_vector: bool = False
) -> [torch.Tensor, torch.Tensor]:
    """
    Computes the contrast value vector (`he` - `she`)for the projecting update.
    """

    print("Computing right vector (v)")

    # TODO: this must be different for he and she pronouns
    target_idss = [tok(pron, return_tensors="pt").
                   to( "cuda" if torch.cuda.is_available() else "cpu")["input_ids"][0]
                   for pron in LLAMA_PRONOUNS.values()]

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request[0]["prompt"])
        for context in context_templates
    ], ["<s> {} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok= tok(
        [prompt.format(request[0]["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
        return_token_type_ids=False
    ).to("cuda" if torch.cuda.is_available() else "cpu")


    # Compute rewriting targets
    rewriting_targetss = [
        torch.tensor(-100, device="cuda" if torch.cuda.is_available() else "cpu").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:])
        for _ in ("pos", "neg") ]

    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targetss[0][i, ex_len - len(target_idss[0]) : ex_len] = target_idss[0]
        rewriting_targetss[1][i, ex_len - len(target_idss[1]) : ex_len] = target_idss[1]

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request[0]["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Trying optimization objective to {loss_layer}...")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    deltas = [torch.zeros((model.config.hidden_size,), requires_grad=True,
                          device="cuda" if torch.cuda.is_available() else "cpu") for _ in ("pos", "neg")]

    # todo how to use delta_shared?
    delta_shared = torch.zeros((model.config.hidden_size,), requires_grad=True,
                               device="cuda" if torch.cuda.is_available() else "cpu")

    target_init, kl_distr_init = None, None

    # Insert deltas for computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.mlp_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()

            for i, idx in enumerate(lookup_idxs):
                cur_out[i, idx, :] += (delta + delta_shared)
        return cur_out

    # Optimizer
    opt = torch.optim.Adam(deltas + [delta_shared], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Optimmize
    print("Optimizing...")

    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # iterate over positive and negative examples
        for target_ids, rewriting_targets, delta, direction in zip(target_idss, rewriting_targetss, deltas, LLAMA_PRONOUNS.keys()):
            # Forward propagation
            with nethook.TraceDict(
                module=model,
                layers=[
                    hparams.layer_module_tmp.format(loss_layer),
                    hparams.mlp_module_tmp.format(layer),
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
            nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
            nll_loss = nll_loss_each.mean()
            kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
                kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
            )
            weight_decay = hparams.v_weight_decay * (
                    torch.norm(delta) / torch.norm(target_init) ** 2
            )
            weight_decay += hparams.v_weight_decay * (
                    torch.norm(delta_shared) / torch.norm(target_init) ** 2
            )
            # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
            loss = nll_loss + kl_loss + weight_decay
            print(
                f"loss ({direction}) {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
                f"avg prob of [{LLAMA_PRONOUNS[direction]}] "
                f"{torch.exp(-nll_loss_each).mean().item()}"
            )
            if loss < 5e-2:
                break

            if it == hparams.v_num_grad_steps - 1:
                break

            # Backpropagate
            loss.backward()
            opt.step()

            # Project within L2 ball
            max_norm = hparams.clamp_norm_factor * target_init.norm()
            if delta.norm() > max_norm:
                with torch.no_grad():
                    delta[...] = delta * max_norm / delta.norm()
            if delta_shared.norm() > max_norm:
                with torch.no_grad():
                    delta_shared[...] = delta_shared * max_norm / delta_shared.norm()


    targets = [target_init + delta + delta_shared for delta in deltas]

    if torch.cuda.is_available():
        targets = [target.to("cuda").half() for target in targets]

    # Retrieve cur_input, the current input to the 2nd MLP layer, and
    # cur_output, the original output of the 2nd MLP layer.

    if not compute_right_vector:
        return targets

    cur_input, cur_output = get_module_input_output_at_word(
        model,
        tok,
        layer,
        context_template=request[0]["prompt"],
        word=request[0]["subject"],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    # Solving the linear system to compute the right vector
    # TODO: the next line doesn't really matter in DAME:
    # right_vectors = [(target - cur_output) / torch.dot(cur_input, left_vector) for target in targets]
    # print(f"Delta norms: {[(target - cur_output).norm().item() for target in targets]}")
    # print(
    #     f"Change in target norms: {target_init.norm().item()} to {[target.norm().item() for target in targets]} => {[(target.norm() - target_init.norm()).item() for target in targets]}"
    # )
    # print(f"Division Factor: {torch.dot(cur_input, left_vector).item()}")

    right_vectors = [target - cur_output for target in targets]
    print(f"Right vector norms: {[right_vector.norm().item() for right_vector in right_vectors]}")
    print(f"Cosine between right vectors: {(torch.dot(right_vectors[0], right_vectors[1])/right_vectors[0].norm()/right_vectors[1].norm()).item()}")
    print(f"Cosine between deltas: {(torch.dot(deltas[0], deltas[1])/deltas[0].norm()/deltas[1].norm()).item()}")

    contrast_vector = right_vectors[0] - right_vectors[1]
    print(f"Contrast vector norm: {contrast_vector.norm().item()}")
    print(f"Cosine between contrast and positive vector: {(torch.dot(contrast_vector, right_vectors[0])/contrast_vector.norm()/right_vectors[0].norm()).item()}")
    print(f"Cosine between contrast and negative vector: {(torch.dot(contrast_vector, right_vectors[1])/contrast_vector.norm()/right_vectors[1].norm()).item()}")

    return right_vectors
    # Compute contrast vector
    # contrast_vector = right_vectors[0] - right_vectors[1]
    # return contrast_vector


def print_vs_stats(V_pos, V_neg, V_orig):

    V_contrast = V_pos - V_neg


    print(f"Positive vector norms: {[v.norm().item() for v in torch.unbind(V_pos)]}")
    print(f"Negative vector norms: {[v.norm().item() for v in torch.unbind(V_neg)]}")
    print(f"Contrast vector norms: {[v.norm().item() for v in torch.unbind(V_contrast)]}")
    print(f"Value vector norms: {[v.norm().item() for v in torch.unbind(V_orig)]}")

    print(f"Cosine across contrast vectors: {((V_contrast @ V_contrast.T)/(V_contrast.norm(dim=1, keepdim=True)*V_contrast.norm(dim=1, keepdim=True).T))}")

    print(f"Cosine between contrast and original vector: {[(torch.dot(v_contrast, v_orig.T)/(v_contrast.norm()*v_orig.norm())).item() for v_contrast, v_orig in zip(torch.unbind(V_contrast), torch.unbind(V_orig))]}")

    print(f"Cosine between pos and original vector: {[(torch.dot(v_pos, v_orig.T)/(v_pos.norm()*v_orig.norm())).item() for v_pos, v_orig in zip(torch.unbind(V_pos), torch.unbind(V_orig))]}")
    print(f"Cosine between neg and original vector: {[(torch.dot(v_neg, v_orig.T)/(v_neg.norm()*v_orig.norm())).item() for v_neg, v_orig in zip(torch.unbind(V_neg), torch.unbind(V_orig))]}")
    print(f"Cosine between pos and neg vector: {[((v_pos @ v_neg.T)/(v_pos.norm()*v_neg.norm())).item() for v_pos, v_neg in zip(torch.unbind(V_pos), torch.unbind(V_neg))]}")

    print("\n*******\n")

def get_module_input_output_at_word(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_template: str,
    word: str,
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both",
            subtoken=subtoken,
            context_templates=[context_template],
            words=[word],
            **word_repr_args,
        )
    elif fact_token_strategy == "last":
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both",
            contexts=[context_template.format(word)],
            idxs=[[-1]],
            **word_repr_args,
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    l_input, l_output = l_input[0], l_output[0]
    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret