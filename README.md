# CoT Monitorability on Open Weights: Replicating OpenAI's Monitorability Evals at 20B Scale

An open-weight replication of the monitorability evals from *Monitoring Monitorability* (Guan et al., 2025), whose public results cover GPT-5-Thinking only. This repo wires the official scaffold end-to-end against self-hosted open models: gpt-oss-20B as the agent and Gemma 4 31B as the monitor, both served with vLLM on a university SLURM cluster. It is one half of my Rutgers MSCS thesis on chain-of-thought monitorability; the other half is [confessions-small-scale](https://github.com/abhishuoza/confessions-small-scale).

The evals measure whether a model's chain of thought reveals things its final answer hides. Each of the six datasets plants an intervention that should change the agent's behavior (a hint to defer to, a cue to sandbag, a nudge toward a wrong judgment), and a monitor model then tries to detect that influence while reading one of three scopes: the chain of thought only, the final answer only, or all messages. When the CoT-only monitor succeeds where the answer-only monitor fails, the chain of thought is carrying safety-relevant signal that would be invisible from outputs alone. That gap is exactly what these evals exist to measure.

```
dataset row ──> agent: gpt-oss-20B (vLLM) ──> transcript (reasoning + final answer)
transcript  ──> monitor: Gemma 4 31B (vLLM), reading one scope:
                { cot_only | answer_only | all_messages } ──> verdicts ──> cross-fitted g-mean2
```

## Results

First pass at low reasoning effort, N=5 samples per row. The metric is OpenAI's cross-fitted g-mean2 (higher means the monitor detects the intervention more reliably), used exactly as shipped. A row counts as eligible only if the planted intervention measurably changes the agent's final answer; eligibility is the fraction of rows surviving that filter. 8,500 graded rows, zero pipeline errors.

| Dataset | cot_only | answer_only | all_messages | Eligibility |
|---|---|---|---|---|
| daft | 0.661 ± 0.049 | 0.000 ± 0.000 | 0.627 ± 0.045 | 52% |
| gpqa | 0.462 ± 0.114 | 0.047 ± 0.040 | 0.424 ± 0.089 | 22% |
| scruples_first_person | 0.050 ± 0.154 | 0.000 ± 0.000 | 0.058 ± 0.156 | 2% |
| scruples_suggest_right | 0.113 ± 0.251 | 0.000 ± 0.000 | 0.113 ± 0.251 | 2% |
| scruples_suggest_wrong | 0.222 ± 0.135 | 0.000 ± 0.000 | 0.201 ± 0.137 | 5% |
| wmdp_sandbagging | 0.337 ± 0.179 | 0.000 ± 0.000 | 0.353 ± 0.144 | 6% |

**Takeaways**

Two so far. First, the scope split the evals are designed to expose reproduces cleanly on an open stack: on daft, the monitor catches the planted cue from the reasoning trace (0.66) but never from the final answer alone (0.00). The 0.00 is the eval working as intended, since daft's cue is named in the reasoning but never in the answer, and it doubles as a clean negative control on the whole pipeline. Second, at 20B and low reasoning effort the binding constraint is eligibility, not monitor quality: on five of six datasets the agent rarely engages with the intervention at all. So the immediate next step is rerunning at medium and high reasoning effort, and with larger agents, to find where eligibility climbs.

## Setup

| Component | Details |
|---|---|
| Agent | `openai/gpt-oss-20b`, low reasoning effort, one GPU via vLLM |
| Monitor | `google/gemma-4-31B-it`, tensor parallel over two GPUs; deliberately cross-family, so the judge is not grading text written in its own dialect |
| Scaffold | [openai/monitorability-evals](https://github.com/openai/monitorability-evals), unmodified: datasets, prompts, eligibility filtering and the cross-fitted g-mean2 computation all run exactly as shipped |
| Scale | 6 intervention datasets, N=5 samples per row, 8,500 graded rows |
| Hardware | 3x NVIDIA A100 on the Rutgers ILAB SLURM cluster |

```bash
pip install -r requirements.txt
git clone https://github.com/openai/monitorability-evals external/monitorability-evals
```

## Usage

```bash
# validate the full wiring offline, no model needed:
python scripts/smoke_local.py

# serve both models (adjust for your cluster):
sbatch scripts/vllm_serve.slurm

# smoke run against real endpoints:
python scripts/run_eval.py --dataset daft --max-rows-per-dataset 3 --samples-per-row 2

# full six-dataset sweep:
bash scripts/chain_evals.sh
```

Endpoints and models are configurable via flags or environment variables (`AGENT_BASE_URL`, `AGENT_MODEL`, `MONITOR_BASE_URL`, `MONITOR_MODEL`, ...); see `scripts/run_eval.py`. Full per-run outputs (~150 MB of graded rows and logs) stay out of git.

## Project Structure

```
├── inference/
│   ├── agent.py          # OpenAI-compatible agent client: reasoning-effort controls, BEGIN/END tag wrapping
│   └── monitor.py        # Monitor client with single-letter verdict parsing
├── scripts/
│   ├── run_eval.py       # End-to-end runner over the official scaffold
│   ├── chain_evals.sh    # Sweep across the six intervention datasets
│   ├── vllm_serve.slurm  # Serves agent + monitor on one SLURM node
│   └── smoke_local.py    # Offline smoke test with a fake OpenAI client
├── external/             # Unmodified clone of the official scaffold (gitignored)
└── requirements.txt
```

The one piece of engineering worth naming: the scaffold ships with `MockModel` and `MockGraders` stubs and a sequential sampler. `run_eval.py` subclasses the mocks with real vLLM endpoints and swaps the sampler for a concurrent `ThreadPoolExecutor` runner, leaving everything on the measurement side untouched.

## References

1. Guan et al. (2025). [Monitoring Monitorability](https://arxiv.org/abs/2512.18311). arXiv:2512.18311.
2. [openai/monitorability-evals](https://github.com/openai/monitorability-evals): the official eval scaffold this repo drives.
3. [confessions-small-scale](https://github.com/abhishuoza/confessions-small-scale): the other half of the thesis, a small-open-scale study of OpenAI's confessions training.
