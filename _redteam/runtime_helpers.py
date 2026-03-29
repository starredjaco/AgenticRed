from __future__ import annotations

import itertools
import os
import re

import openai

from prompt_memory import FailedPromptMemory, SucceedPromptMemory
from utils import extract_model_name


def setup_client(model_string: str) -> openai.OpenAI:
    """Set up Endpoint for different model providers."""
    if model_string.startswith('http://'):
        return openai.OpenAI(base_url=model_string)
    if model_string.startswith('gpt'):
        return openai.OpenAI()
    if model_string.startswith('deepseek'):
        return openai.OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'], base_url="https://api.deepseek.com")
    if model_string.startswith('gemini'):
        return openai.OpenAI(
            api_key=os.environ['GEMINI_API_KEY'],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return openai.OpenAI(
        api_key=os.environ['OPENROUTER_API_KEY'],
        base_url="https://openrouter.ai/api/v1",
    )


def setup_attacker_runtime(attacker_model: str):
    """Initialize attacker endpoint routing and model metadata."""
    if ',' in attacker_model:
        endpoints = [s.strip() for s in attacker_model.split(',') if s.strip()]
        endpoint_cycle = itertools.cycle(endpoints)
        tmp_client = openai.OpenAI(base_url=endpoints[0])
        model_name = tmp_client.models.list().data[0].id
        model_name_abbr = extract_model_name(model_name)
        return None, model_name, model_name_abbr, endpoints, endpoint_cycle

    if attacker_model.startswith('http://'):
        endpoints = [attacker_model]
        endpoint_cycle = itertools.cycle(endpoints)
        client = openai.OpenAI(base_url=attacker_model)
        model_name = client.models.list().data[0].id
        model_name_abbr = extract_model_name(model_name)
        return client, model_name, model_name_abbr, endpoints, endpoint_cycle

    endpoints = [attacker_model]
    endpoint_cycle = itertools.cycle(endpoints)
    client = openai.OpenAI()
    model_name = attacker_model
    model_name_abbr = model_name
    return client, model_name, model_name_abbr, endpoints, endpoint_cycle


def setup_defender_runtime(defender_model: str, use_openrouter: bool = False,
                           keep_declared_name: bool = False):
    """Initialize defender client and model naming."""
    if keep_declared_name:
        client = setup_client("openrouter") if use_openrouter else setup_client(defender_model)
        model_name = defender_model
        model_name_abbr = extract_model_name(model_name)
        return client, model_name, model_name_abbr

    client = setup_client(defender_model)
    model_name = client.models.list().data[0].id
    model_name_abbr = extract_model_name(model_name)
    return client, model_name, model_name_abbr


def load_failed_prompt_memory_for_model(save_dir: str, model_name: str, model_name_abbr: str) -> int:
    """Configure and load model-scoped failed prompt memory."""
    memory_dir = os.path.join(save_dir, "failed_prompt_memory")
    print(f"Configuring FailedPromptMemory persistence for model {model_name_abbr} at {memory_dir}")
    FailedPromptMemory.configure_persistence(model_name, memory_dir)
    loaded_count = FailedPromptMemory.load()
    print(f"[FailedMemory] Loaded {loaded_count} prompts from {model_name_abbr}")
    return loaded_count


def load_succeed_prompt_memory_for_run(save_dir: str, expr_name: str, resume_path: str = None) -> str:
    """Configure and load run-scoped successful-prompt memory.

    If resume_path is provided, it is used as-is and can point to another run's
    memory file to warm-start a resumed run.
    """
    if resume_path and str(resume_path).strip():
        persist_path = os.path.abspath(os.path.expanduser(str(resume_path).strip()))
    else:
        run_key = re.sub(r"[^a-zA-Z0-9._-]", "_", str(expr_name or "run"))
        memory_dir = os.path.join(save_dir, "succeed_prompt_memory")
        persist_path = os.path.join(memory_dir, f"succeed_prompt_memory_{run_key}.json")

    print(f"Configuring SucceedPromptMemory persistence at {persist_path}")
    SucceedPromptMemory.configure_persistence(persist_path)
    loaded_count = SucceedPromptMemory.load()
    print(f"[SucceedMemory] Loaded {loaded_count} prompts")
    return persist_path
