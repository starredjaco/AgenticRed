#!/bin/bash
#
#SBATCH -J l3-rr
#SBATCH --partition=a40        # Use GPU partition "a100"
#SBATCH --gres gpu:1            # set 2 GPUs per job
#SBATCH -t 7-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=20G               # Memory pool for all cores (see also --mem-per-cpu)

export VLLM_LOGGING_LEVEL=DEBUG

# LLAMA-2-7B-Chat model path
defender_model=''
# LLAMA-3-8B-Instruct model path
defender_model=''

LLAMA2_CHAT_TEMPLATE='' 
LLAMA_3_INSTRUCT_TEMPLATE=''

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --trust_remote_code \
  --model $defender_model \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --chat-template $LLAMA2_CHAT_TEMPLATE \
  --port 8000
  
