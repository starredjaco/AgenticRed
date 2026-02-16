#!/bin/bash

# Set environment variables 
export MODEL_PATH='/SWS/llms/nobackup/hub/models--cais--HarmBench-Llama-2-13b-cls/snapshots/bda705349d1144fa618770bea64d99ce54e3835b'  # Replace with your model path
export CUDA_VISIBLE_DEVICES=0  # Optional: Specify GPU device
echo "Using classifier model $MODEL_PATH"

# Launch vLLM OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --tokenizer-mode auto \
    --served-model-name harmbench-classifier \
    --trust-remote-code \
    --chat-template $MODEL_PATH/llama-2-chat.jinja \
    --port 8080