#!/bin/bash
export VLLM_LOGGING_LEVEL=DEBUG

# LLAMA-2-7B-Chat
defender_model='/NS/MAS-llms01/nobackup/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590'
# LLAMA-3-8B-Instruct
# defender_model='/SWS/llms/nobackup/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a'

LLAMA2_CHAT_TEMPLATE=/home/jyuan/chat_templates/chat_templates/llama-2-chat.jinja 
LLAMA_3_INSTRUCT_TEMPLATE=/home/jyuan/chat_templates/chat_templates/llama-3-instruct.jinja

echo "Starting defender server $defender_model on port 8001"

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --trust_remote_code \
  --model $defender_model \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --chat-template $LLAMA2_CHAT_TEMPLATE \
  --port 8001
  
