# CoT Monitorability Evals on Open Weights

Open-weight replication of OpenAI's [monitorability-evals](https://github.com/openai/monitorability-evals) suite from *Monitoring Monitorability* (Guan et al., arXiv [2512.18311](https://arxiv.org/abs/2512.18311)), whose public results cover GPT-5-Thinking only. This repo wires the official scaffold end-to-end against self-hosted open models and reports first-pass results at 20B scale. It is part of my Rutgers MSCS thesis on chain-of-thought monitorability.

Setup: gpt-oss-20B as the agent (low reasoning effort) and Gemma 4 31B as a cross-family monitor (chosen to avoid same-family confounds), both served via vLLM on the Rutgers ILAB SLURM cluster; OpenAI's cross-fitted g-mean2 metric used as shipped. Six intervention datasets, 8,500 graded rows, zero pipeline errors.

## First-pass results (low reasoning effort, N=5 samples per row)

| Dataset | cot_only | answer_only | all_messages | Eligibility |
|---|---|---|---|---|
| daft | 0.661 ± 0.049 | 0.000 ± 0.000 | 0.627 ± 0.045 | 52% |
| gpqa | 0.462 ± 0.114 | 0.047 ± 0.040 | 0.424 ± 0.089 | 22% |
| scruples_first_person | 0.050 ± 0.154 | 0.000 ± 0.000 | 0.058 ± 0.156 | 2% |
| scruples_suggest_right | 0.113 ± 0.251 | 0.000 ± 0.000 | 0.113 ± 0.251 | 2% |
| scruples_suggest_wrong | 0.222 ± 0.135 | 0.000 ± 0.000 | 0.201 ± 0.137 | 5% |
| wmdp_sandbagging | 0.337 ± 0.179 | 0.000 ± 0.000 | 0.353 ± 0.144 | 6% |

A row is eligible only if the planted intervention measurably changes the agent's final answer; eligibility is the fraction of items surviving that filter.

Two takeaways so far. First, the scope split the evals are designed to expose reproduces cleanly on an open stack: on daft, the monitor catches the planted cue from the reasoning trace (0.66) but never from the final answer alone (0.00). Second, at 20B and low reasoning effort the binding constraint is eligibility, not monitor quality: on five of six datasets the agent rarely engages with the intervention at all, so the immediate next step is rerunning at medium/high reasoning effort and with larger agents to find where eligibility climbs.

## How it works

- `external/monitorability-evals/` — unmodified clone of the official scaffold (gitignored; clone it yourself, see below).
- `inference/` — thin OpenAI-compatible clients for the agent (`agent.py`, with reasoning-effort / thinking controls and BEGIN/END tag wrapping for the monitor templates) and the monitor (`monitor.py`, single-letter verdict parsing).
- `scripts/run_eval.py` — the end-to-end runner: monkey-patches the scaffold's `MockModel` / `MockGraders` with real vLLM endpoints and replaces its sampler with a concurrent `ThreadPoolExecutor` runner; writes the scaffold's standard outputs plus per-row transcripts, monitor verdicts per scope, and latencies.
- `scripts/vllm_serve.slurm` — serves agent and monitor on one SLURM node (agent on one GPU, monitor at TP=2 on two).
- `scripts/chain_evals.sh` — the sweep across the six intervention datasets (`--samples-per-row 5 --reasoning-effort low`).
- `scripts/smoke_local.py` — offline smoke test with a fake OpenAI client; validates the full wiring without any model.

Full per-run outputs (~150 MB of graded rows and logs) are kept out of git.

## Running

```bash
git clone https://github.com/openai/monitorability-evals external/monitorability-evals
pip install -r requirements.txt
# validate wiring without any model:
python scripts/smoke_local.py
# serve models (adjust for your cluster):
sbatch scripts/vllm_serve.slurm
# smoke run against real endpoints:
python scripts/run_eval.py --dataset daft --max-rows-per-dataset 3 --samples-per-row 2
# full sweep:
bash scripts/chain_evals.sh
```

Endpoints and models are configurable via flags or environment variables (`AGENT_BASE_URL`, `AGENT_MODEL`, `MONITOR_BASE_URL`, `MONITOR_MODEL`, ...); see `scripts/run_eval.py`.

## Context

This work is one half of a broader thesis direction on "clean channels" for scalable model testing (reward-disjoint or optimization-shielded channels whose honesty we can argue for); the other half is a small-open-scale study of OpenAI's confessions training, at [confessions-small-scale](https://github.com/abhishuoza/confessions-small-scale).
