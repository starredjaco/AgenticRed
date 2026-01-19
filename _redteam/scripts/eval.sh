#!/bin/bash

# Set HF_HOME if needed
# export HF_HOME=''

LLAMA2_ENDPOINT=''
LLAMA3_ENDPOINT=''
LLAMA3_RR_ENDPOINT=''
ATTACKER_ENDPOINTS=''
VICUNA_ENDPOINTS=''
CLASSIFIER_ENDPOINT=''
EXPR_NAME=redteam_archive # name for the generated json in results directory
EVALUATOR_MODELS='gpt-3.5-turbo,gpt-4o-mini'
BENCHMARK=harmbench

python -u search.py \
    --mode evaluate \
    --valid_size 50 \
    --benchmark $BENCHMARK \
    --expr_name $EXPR_NAME \
    --attacker_model $ATTACKER_ENDPOINTS \
    --defender_model $LLAMA2_ENDPOINT \
    --evaluator_model $EVALUATOR_MODELS \
    --classifier_model $CLASSIFIER_ENDPOINT \
    --meta_agent_model gpt-5-2025-08-07 \
    --n_repeat 1 \
    --debug_max 5 \
    --max_workers 16 \
    --shuffle_seed 0