#!/bin/bash
#SBATCH --job-name=nova_sol_test
#SBATCH --partition=cs
#SBATCH --account=condo_cs_ali
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=finetune_test_%j.out
#SBATCH --error=finetune_test_%j.err

echo "===== JOB START ====="
date
hostname
module purge
module load cuda/11.8.0
PYTHON=/home/eh3115/.conda/envs/nova/bin/python
PROJECT_ROOT=/scratch/eh3115/Decompliler/Decompliler
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
export CUDA_VISIBLE_DEVICES=0
mkdir -p /scratch/$USER/hf_cache
nvidia-smi
echo "===== Test fine-tuning: 5000 samples, 1 GPU, 1 epoch ====="
cd "$PROJECT_ROOT/nova"
$PYTHON finetune_test.py
echo "===== JOB END ====="
date
