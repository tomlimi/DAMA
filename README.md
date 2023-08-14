# DAMA
**D**ebiasing **A**lgorithm through **M**odel **A**daptation. The method for decreasing gender bias in LLaMA family LLMs.

All code should be run from `src` directory. Please, install all dependencies with `pip install -r requirements.txt`.

## Causal Tracing

The causal tracing is conducted following the method proposed by Meng et al. 2023.
The input representation is obstructed by adding a noise to potentially biased workds (profession names).
Then we re-introduce the clean representation at different components of the model to check how prone they are to skew the output of the model.


To get the results of casual tracing run:

```bash
python causal_tracing.py \
--model_name_path llama \
--param_number 7 \
--disable_mlp

```
 `param_number` is provided in the billions of parameters.
`disable_mlp` is a flag for severing MLP heads (following Meng et al. 2023).
The results can be visualized with jupyter notebooks in `src/notebooks` directory.

## Running DAMA

Before debiasing it's necessary to prepare hyperparameter files.
An example in `examples/llama_7B_l9_once_prel_gen_bn_on.json`

To run dama on a specific model, you need to run the following command:

```bash
python adapt_model.py \
--model_name llama \
--param_num 7 \
--method DAMA \
--num_layers 9 \
--iterative_update \
--post_linear \
--request_file train_dama.json
```

The flags of the call need to correspond to the name of the hyperparameter file name.
An example of training file is provided in `examples/train_dama_tiny.json`

## Evaluating Model

The projections from DAMA will be saved in result subsdirectory named the same as params file.

The evaluation of the model is obtained by running:

```bash
python evaluate_model.py \
--model_name llama \
--param_num 7 \
--method DAMA \
--num_layers 9 \ 
--iterative_update \
--post_linear \
--test_file test_dama.json \
--test_task gen
```

An example of test file is provided in `examples/test_dama_tiny.json`

For coreference resolution change the last two lines with:

```bash
--test_file anti_type1_test.txt \
--test_task coref
```

The test data are available at https://github.com/uclanlp/corefBias/tree/master/WinoBias/wino/data.