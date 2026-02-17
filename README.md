<h1 align="center">
  <b>AgenticRed: Optimizing Agentic Systems for Automated Red-teaming</b><br>
</h1>

<p align="center">
  <a href="https://github.com/yuanjiayiy/AgenticRed/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge"></a>
  <a href="https://arxiv.org/abs/2601.13518"><img src="https://img.shields.io/badge/arXiv-2408.08435-b31b1b.svg?logo=arxiv&style=for-the-badge"></a>
  <a href="https://yuanjiayiy.github.io/AgenticRed/"><img src="https://img.shields.io/badge/-Website-%238D6748?style=for-the-badge&logo=Website&logoColor=white"></a>
  <a href="https://x.com/carrieyuanjiayi/status/2014043770300645468"><img src="https://img.shields.io/badge/twitter-%230077B5.svg?&style=for-the-badge&logo=twitter&logoColor=white&color=00acee"></a>
</p>


# AgenticRed

AgenticRed is an automated pipeline that leverages LLMs’ in-context learning to iteratively design and refine red-teaming systems without human intervention, based on the performance metric of
each system.

---

## Repository Structure

```text
.
├── server/           # Scripts to start attacker, classifier, and defender model servers
├── search.sh         # Main script to run the search process
├── eval.sh           # Main script to run the evaluation process
└── README.md
```

---

## Prerequisites

- If you want to host your model locally, you need at least 3 GPUs (e.g., 3× L40) or equivalent capacity
    - One for the attacker model server
    - One for the classifier / judge model server
    - One for the defender / target model server
- If you want to call API endpoints instead, we provide OpenRouter/OpenAI client API.
---

## 1. Start Model Servers

You can run local servers (recommended for experiments) or point to external APIs.

### 1.1 Local servers

Inside `server/`, you should have scripts or configs such as:

```bash
cd server/

bash attacker_server.sh          # Starts attacker model server
bash classifier_server.sh        # Starts classifier / guardrail model server
bash defender_server.sh          # Starts defender / target model server
```

Document the actual ports and endpoints so the rest of the pipeline can reference them.
Eg. http://127.0.0.1/8000/v1

### 1.2 Using external APIs (optional)

Instead of local servers, you can configure the system to call:

- OpenAI APIs
- OpenRouter APIs

Update your API keys:

```
export OPENAI_API_KEY=''
export GEMINI_API_KEY=''
export DEEPSEEK_API_KEY=''
export OPENROUTER_API_KEY=''
```

---

## 2. Run the Search Process

The search step iteratively designs new red-teaming systems. 

Typical usage:

```bash
cd _redteam
bash scripts/search.sh

e.g. bash search.sh --expr 1
```

Key arguments (adapt to your implementation):

- `--expr`: Experiment index corresponding to different configurations.
- `--config`: Path to a config file specifying:
    - Attacker, classifier, defender endpoints
    - Search hyperparameters (iterations, beam width, budgets, etc.)
    - Output directory for JSON logs: save_dir
    - Whether to enable OpenRouter
---

## 3. Run the Evaluation Process

After search completes, evaluate the discovered attacks on a separate evaluation dataset and target model / benchmark.

```bash
cd _redteam
bash scripts/eval.sh
```

Key arguments (adapt to your implementation):
- evaluator_model: Models used for evaluation
- benchmark: Currently only implemented HarmBench and StrongREJECT.
---

## Example Workflow

```bash
# 1. Start servers if you are hosting local servers
bash server/attacker_server.sh
bash server/defender_server.sh
bash server/classifirt_server.sh

# 2. Run search
bash _redteam/search.sh --expr 1

# 3. Run evaluation
bash _redteam/eval.sh
```

## Citation

```bibtex
@misc{yuan2026agenticredoptimizingagenticsystems,
      title={AgenticRed: Optimizing Agentic Systems for Automated Red-teaming}, 
      author={Jiayi Yuan and Jonathan Nöther and Natasha Jaques and Goran Radanović},
      year={2026},
      eprint={2601.13518},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.13518}, 
}
```


