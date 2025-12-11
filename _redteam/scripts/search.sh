#!/bin/bash
#SBATCH -J exp_8
#SBATCH -p spyder
#SBATCH -t 8-00:00 
#SBATCH --mem=10G
#SBATCH -o slurm/hostname_%j.out
#SBATCH -e slurm/hostname_%j.err
set -euo pipefail

# Set HF_HOME if needed
# export HF_HOME=''

# Set model endpoints
LLAMA2_ENDPOINT=''
LLAMA3_ENDPOINT=''
ATTACKER_ENDPOINTS=''
VICUNA_ENDPOINTS=''
CLASSIFIER_ENDPOINT=''

INDEX=1
SEED=42

echo "Experiment index: ${INDEX}"
echo "Shuffle seed: ${SEED}"


case "$INDEX" in
    1)
        echo "1) baseline"
        python -u search.py \
            --valid_size 50 \
            --benchmark harmbench \
            --attacker_model $ATTACKER_ENDPOINT \
            --defender_model $LLAMA2_ENDPOINT \
            --classifier_model $CLASSIFIER_ENDPOINT \
            --meta_agent_model gpt-5-2025-08-07 \
            --n_repeat 1 \
            --debug_max 5 \
            --max_workers 16 \
            --shuffle_seed $SEED
        ;;

    2)
        echo "2) switch target model -> LLAMA3"
        python -u search.py \
            --valid_size 50 \
            --benchmark harmbench \
            --attacker_model $ATTACKER_ENDPOINTS \
            --defender_model $LLAMA3_ENDPOINT \
            --classifier_model $CLASSIFIER_ENDPOINT \
            --meta_agent_model gpt-5-2025-08-07 \
            --n_repeat 1 \
            --debug_max 5 \
            --max_workers 16 \
            --shuffle_seed $SEED
        ;;
    3)
        echo "3) switch attacker model -> VICUNA"
        python -u search.py \
            --valid_size 50 \
            --benchmark harmbench \
            --attacker_model $VICUNA_ENDPOINTS \
            --defender_model $LLAMA2_ENDPOINT \
            --classifier_model $CLASSIFIER_ENDPOINT \
            --meta_agent_model gpt-5-2025-08-07 \
            --n_repeat 1 \
            --debug_max 5 \
            --max_workers 16 \
            --shuffle_seed $SEED
        ;;

    4)
        echo "4) switch meta agent model -> deepseek-reasoner"
        python -u search.py \
            --valid_size 50 \
            --benchmark harmbench \
            --attacker_model $ATTACKER_ENDPOINTS \
            --defender_model $LLAMA2_ENDPOINT \
            --classifier_model $CLASSIFIER_ENDPOINT \
            --meta_agent_model deepseek-reasoner \
            --n_repeat 1 \
            --debug_max 5 \
            --max_workers 16 \
            --shuffle_seed $SEED
        ;;

    5)
        echo "5) w/o evolutionary pressure"
        python -u search.py \
            --valid_size 50 \
            --benchmark harmbench \
            --attacker_model $ATTACKER_ENDPOINTS \
            --defender_model $LLAMA2_ENDPOINT \
            --classifier_model $CLASSIFIER_ENDPOINT \
            --meta_agent_model gpt-5-2025-08-07 \
            --n_repeat 1 \
            --debug_max 5 \
            --max_workers 16 \
            --shuffle_seed $SEED \
            --num_offspring_per_gen 1 \
            --expr_name redteam_gpt-5-2025-08-07_Mistral-7B-Instruct-v0.3_Llama-2-7b-chat-hf_2_1_offspring
        ;;

    6)
        echo "6) w/ weak archive"
        python -u search.py \
            --valid_size 50 \
            --benchmark harmbench \
            --attacker_model $ATTACKER_ENDPOINTS \
            --defender_model $LLAMA2_ENDPOINT \
            --classifier_model $CLASSIFIER_ENDPOINT \
            --meta_agent_model gpt-5-2025-08-07 \
            --n_repeat 1 \
            --debug_max 5 \
            --max_workers 16 \
            --shuffle_seed $SEED \
            --expr_name redteam_gpt-5-2025-08-07_Mistral-7B-Instruct-v0.3_Llama-2-7b-chat-hf_2_weaker_archive \
            --weak_init_archive
        ;;
    7)
        echo "7) w/ diversity incentive"
        # fill with your desired args for experiment 7
        python -u search.py \
            --valid_size 50 \
            --benchmark harmbench \
            --attacker_model $ATTACKER_ENDPOINTS \
            --defender_model $LLAMA2_ENDPOINT \
            --classifier_model $CLASSIFIER_ENDPOINT \
            --meta_agent_model gpt-5-2025-08-07 \
            --n_repeat 1 \
            --debug_max 5 \
            --max_workers 16 \
            --shuffle_seed $SEED \
            --expr_name redteam_gpt-5-2025-08-07_Mistral-7B-Instruct-v0.3_Llama-2-7b-chat-hf_2_diversity_incentive \
            --diversity_incentive
        ;;
    8)
        echo "8) w/ diversity threshold"
        # fill with your desired args for experiment 8
        python -u search.py \
            --valid_size 50 \
            --benchmark harmbench \
            --attacker_model $ATTACKER_ENDPOINTS \
            --defender_model $LLAMA2_ENDPOINT \
            --classifier_model $CLASSIFIER_ENDPOINT \
            --meta_agent_model gpt-5-2025-08-07 \
            --n_repeat 1 \
            --debug_max 5 \
            --max_workers 16 \
            --shuffle_seed $SEED \
            --expr_name redteam_gpt-5-2025-08-07_Mistral-7B-Instruct-v0.3_Llama-2-7b-chat-hf_2_diversity_threshold_0.2 \
            --diversity_threshold 0.2
        ;;
    *)
        echo "Unknown experiment index: ${INDEX}"
        exit 1
        ;;
esac