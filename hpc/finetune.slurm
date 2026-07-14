#!/bin/bash
#SBATCH --job-name=nova_solidity_finetune
#SBATCH --partition=cs
#SBATCH --account=condo_cs_ali
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=finetune_%j.out
#SBATCH --error=finetune_%j.err

echo "===== JOB START ====="
date
hostname
pwd

module purge
module load cuda/11.8.0

PYTHON=/home/eh3115/.conda/envs/nova/bin/python
echo "Python:"
$PYTHON --version

nvidia-smi

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache
mkdir -p /scratch/$USER/hf_cache

PROJECT_ROOT=/scratch/eh3115/Decompliler/Decompliler

echo "GPU warmup..."
$PYTHON - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}:", torch.cuda.get_device_name(i))
x = torch.randn(2048, 2048, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("Warmup done")
PY

echo "===== Fine-tuning Nova-Solidity ====="
cd "$PROJECT_ROOT/nova"

torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    finetune_solidity.py

echo "===== JOB END ====="
date
