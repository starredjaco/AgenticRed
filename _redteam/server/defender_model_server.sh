#!/bin/bash
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
  
