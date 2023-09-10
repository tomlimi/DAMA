#!/bin/bash
#SBATCH -J evaluate_generation
#SBATCH -q high
#SBATCH -p gpu-troja,gpu-ms
#SBATCH -o /home/limisiewicz/my-luster/dama/job_output/evaluate_generation_13B.out
#SBATCH -D /home/limisiewicz/my-luster/dama/DAMA/src
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --constraint="gpuram48G|gpuram40G"


# This script runs the evaluation of the DAMA system.


source /home/limisiewicz/my-luster/GenderBiasGACR/.virtualenv/bin/activate


#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 11 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 32 --no_colinear_vs True --vs_at_last True --use_neutral True --delta_only True --no_whitening True
#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 10 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 32 --no_colinear_vs True --vs_at_last True --use_neutral True --delta_only True --no_whitening True
#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 12 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 32 --no_colinear_vs True --use_neutral True --delta_only True --no_whitening True  --vs_at_last True
#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 14 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 32 --no_colinear_vs True --use_neutral True --delta_only True --no_whitening True  --vs_at_last True
python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 8 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 32 --no_colinear_vs True --use_neutral True --delta_only True --no_whitening True  --vs_at_last True


#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 11 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 1024 --no_colinear_vs True --vs_at_last True --use_neutral True --delta_only True --no_whitening True
#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 10 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 1024 --no_colinear_vs True --vs_at_last True --use_neutral True --delta_only True --no_whitening True
#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 12 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 1024 --no_colinear_vs True --use_neutral True --delta_only True --no_whitening True  --vs_at_last True
#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 14 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 1024 --no_colinear_vs True --use_neutral True --delta_only True --no_whitening True  --vs_at_last True
#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 8 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 1024 --no_colinear_vs True --use_neutral True --delta_only True --no_whitening True  --vs_at_last True
#
#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 11 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 512 --no_colinear_vs True --vs_at_last True --use_neutral True --delta_only True --no_whitening True
#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 10 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 512 --no_colinear_vs True --vs_at_last True --use_neutral True --delta_only True --no_whitening True
#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 12 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 512 --no_colinear_vs True --use_neutral True --delta_only True --no_whitening True  --vs_at_last True
#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 14 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 512 --no_colinear_vs True --use_neutral True --delta_only True --no_whitening True  --vs_at_last True
#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 8 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 512 --no_colinear_vs True --use_neutral True --delta_only True --no_whitening True  --vs_at_last True

#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 11 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 256 --no_colinear_vs True --vs_at_last True --use_neutral True --delta_only True --no_whitening True
#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 10 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 256 --no_colinear_vs True --vs_at_last True --use_neutral True --delta_only True --no_whitening True
#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 12 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 256 --no_colinear_vs True --use_neutral True --delta_only True --no_whitening True  --vs_at_last True
python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 14 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 256 --no_colinear_vs True --use_neutral True --delta_only True --no_whitening True  --vs_at_last True
python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 8 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 256 --no_colinear_vs True --use_neutral True --delta_only True --no_whitening True  --vs_at_last True

#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 11 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 128 --no_colinear_vs True --vs_at_last True --use_neutral True --delta_only True --no_whitening True
#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 10 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 128 --no_colinear_vs True --vs_at_last True --use_neutral True --delta_only True --no_whitening True
#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 12 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 128 --no_colinear_vs True --use_neutral True --delta_only True --no_whitening True  --vs_at_last True
#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 14 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 128 --no_colinear_vs True --use_neutral True --delta_only True --no_whitening True  --vs_at_last True
#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 8 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 128 --no_colinear_vs True --use_neutral True --delta_only True --no_whitening True  --vs_at_last True
#
#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 11 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 64 --no_colinear_vs True --vs_at_last True --use_neutral True --delta_only True --no_whitening True
#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 10 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 64 --no_colinear_vs True --vs_at_last True --use_neutral True --delta_only True --no_whitening True
#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 12 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 64 --no_colinear_vs True --use_neutral True --delta_only True --no_whitening True  --vs_at_last True
#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 14 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 64 --no_colinear_vs True --use_neutral True --delta_only True --no_whitening True  --vs_at_last True
#python evaluate_model.py --param_number 13 --method "DAMA" --test_file test_dama.json --test_task "gen" --num_layers 8 --post_linear True --mixed_update True --orthogonal_constraint 0 --null_dim 64 --no_colinear_vs True --use_neutral True --delta_only True --no_whitening True  --vs_at_last True