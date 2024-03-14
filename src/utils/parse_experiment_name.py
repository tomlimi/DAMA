

def parse_experiment_name(num_layers: int = 9,
                          iterative_update: bool = False,
                          mixed_update: bool = False,
                          task: str = "gen",
                          post_linear: bool = False,
                          batch_size: int = 1,
                          orthogonal_constraint: float = 0.,
                          no_colinear_vs: bool = False,
                          vs_at_last: bool = False,
                          null_dim: int = 1024,
                          use_neutral: bool = False,
                          delta_only: bool = False,
                          use_neutral_tensor: bool = False,
                          nw: bool = False,
                          seed: int = 0
                          ) -> str:
    experiment_string = f"l{num_layers}"
    if iterative_update:
        experiment_string += "_iter"
    elif mixed_update:
        experiment_string += ""
    else:
        experiment_string += "_once"

    if post_linear:
        experiment_string += ""
    else:
        experiment_string += "_prel"

    if task == "gen":
        experiment_string += ""
    elif task == "coref":
        experiment_string += "_coref"
        raise NotImplementedError("Coreference resolution task not implemented yet")
    else:
        raise ValueError("Unknown task. Choose from: gen, coref")

    if batch_size == 1:
        experiment_string += ""
    elif batch_size > 1:
        experiment_string += f"_b{batch_size}"
    else:
        raise ValueError("Batch size must be a positive integer")

    if orthogonal_constraint:
        experiment_string += f"_o{orthogonal_constraint}"
    else:
        experiment_string += "_on"

    if null_dim != 1024:
        experiment_string += f"_nd{null_dim}"

    if no_colinear_vs:
        experiment_string += "_ncv"

    if vs_at_last:
        experiment_string += "_val"

    if use_neutral:
        experiment_string += "_neutral"
    if delta_only:
        experiment_string += "_delta_only"

    if use_neutral_tensor:
        experiment_string += "_nt"

    if nw:
        experiment_string += '_nw'

    if seed != 0:
        experiment_string += f"_s{seed}"

    return experiment_string