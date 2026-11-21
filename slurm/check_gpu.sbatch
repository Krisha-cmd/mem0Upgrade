#!/bin/bash
#SBATCH --job-name=sm-check-gpu
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Check 2: Qwen2.5-7B loads in vLLM (offline) and generates 5 answers.
source slurm/common.sh
python - <<'PY'
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", max_model_len=32768, gpu_memory_utilization=0.85)
qs = ["Plan a trip to Rome.", "Suggest a vegan dinner.", "Recommend a jazz album.",
      "How do I learn Python?", "Pick a hotel in Lisbon."]
outs = llm.chat([[{"role": "user", "content": q}] for q in qs], SamplingParams(temperature=0, max_tokens=64))
for q, o in zip(qs, outs):
    print("Q:", q, "\nA:", o.outputs[0].text.strip()[:200], "\n")
assert len(outs) == 5
PY
echo CHECK_GPU_OK
