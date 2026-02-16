#!/bin/bash
export VLLM_LOGGING_LEVEL=DEBUG
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

CHAT_TEMPLATE_DIR=/home/jyuan/chat_templates/chat_templates
VICUNA_MODEL='/SWS/llms/nobackup/hub/models--lmsys--vicuna-13b-v1.5/snapshots/c8327bf999adbd2efe2e75f6509fa01436100dc2/'
MISTRAL_MODEL='/SWS/llms/nobackup/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a/'


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

# Find available port starting from 8090
PORT=$(find_free_port 8090)
echo "Using available port: $PORT for attacker model $MISTRAL_MODEL"

# vicuna server
# python -m vllm.entrypoints.openai.api_server \
#   --model $VICUNA_MODEL \
#   --guided-decoding-backend lm-format-enforcer \
#   --tensor-parallel-size 1 \
#   --dtype bfloat16 \
#   --max-model-len 4096 \
#   --port $PORT

# mistral server
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model $MISTRAL_MODEL \
    --guided-decoding-backend lm-format-enforcer \
    --tensor-parallel-size 1 \
    --max-model-len 10000 \
    --dtype bfloat16 \
    --enforce-eager \
    --port $PORT