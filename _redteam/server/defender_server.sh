#!/bin/bash
export VLLM_LOGGING_LEVEL=DEBUG

# LLAMA-2-7B-Chat
defender_model='/NS/MAS-llms01/nobackup/hf_cache/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590'
# LLAMA-3-8B-Instruct
# defender_model='/SWS/llms/nobackup/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a'
defender_model="Qwen/Qwen3-8B"
# defender_model=/SWS/llms/nobackup/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60

LLAMA2_CHAT_TEMPLATE=/home/jyuan/chat_templates/chat_templates/llama-2-chat.jinja 
LLAMA_3_INSTRUCT_TEMPLATE=/home/jyuan/chat_templates/chat_templates/llama-3-instruct.jinja


# Function to check if a port is available
check_port() {
    nc -z localhost $1 >/dev/null 2>&1
    [ $? -ne 0 ]
}

# Function to find next available port starting from base_port
find_free_port() {
    local base_port=$1
    local port=$base_port
    while ! check_port $port; do
        port=$((port + 1))
    done
    echo $port
}

# Find available port starting from 8000
PORT=$(find_free_port 8000)
echo "Using available port: $PORT for defender model $defender_model"

# CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
#   --trust_remote_code \
#   --model $defender_model \
#   --dtype bfloat16 \
#   --max-model-len 4096 \
#   --tensor-parallel-size 1 \
#   --enforce-eager \
#   --chat-template $LLAMA2_CHAT_TEMPLATE \
#   --port $PORT


CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-8B \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --port $PORT