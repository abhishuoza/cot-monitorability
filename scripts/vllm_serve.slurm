#!/bin/bash
#SBATCH --job-name=cot-vllm
#SBATCH --partition=unlimited
#SBATCH --nodelist=rlab1
#SBATCH --gres=gpu:a100:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/common/users/%u/cot-monitorability/slurm_%j.out
#SBATCH --error=/common/users/%u/cot-monitorability/slurm_%j.err

# Two vLLM OpenAI-compatible servers on one node:
#   GPU 0           -> openai/gpt-oss-20b      (port 8000, agent)
#   GPUs 1,2 (TP=2) -> google/gemma-4-31B-it   (port 8001, canonical monitor)
# Tunnel from WSL2: ssh -L 8000:rlab1:8000 -L 8001:rlab1:8001 -N ilab

export HF_HOME=/common/users/$USER/.hf_cache
export PIP_CACHE_DIR=/common/users/$USER/.pip_cache
export TMPDIR=/common/users/$USER/.tmp

PROJECT=/common/users/$USER/cot-monitorability
cd "$PROJECT"
source venv/bin/activate

echo "=== node ==="
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo "=== launching agent: gpt-oss-20b on GPU 0, port 8000 ==="
CUDA_VISIBLE_DEVICES=0 vllm serve openai/gpt-oss-20b \
  --port 8000 --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  > "$PROJECT/vllm_agent.log" 2>&1 &
AGENT_PID=$!

echo "=== launching monitor: gemma-4-31B-it on GPUs 1,2 (TP=2), port 8001 ==="
# Gemma 4 is multimodal; we only use text. Skip vision encoder via limit-mm-per-prompt
# and bump max-num-batched-tokens above the encoder's default mm-item budget.
CUDA_VISIBLE_DEVICES=1,2 vllm serve google/gemma-4-31B-it \
  --port 8001 --host 0.0.0.0 \
  --tensor-parallel-size 2 \
  --max-model-len 12288 \
  --max-num-batched-tokens 12288 \
  --max-num-seqs 64 \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --gpu-memory-utilization 0.95 \
  > "$PROJECT/vllm_monitor.log" 2>&1 &
MONITOR_PID=$!

echo "agent_pid=$AGENT_PID monitor_pid=$MONITOR_PID"
echo "ports: agent=8000, monitor=8001"

trap "kill $AGENT_PID $MONITOR_PID 2>/dev/null; wait" TERM INT
wait
