#!/bin/bash
#SBATCH --job-name=nova_sol_full
#SBATCH --partition=cs
#SBATCH --account=condo_cs_ali
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=finetune_full_%j.out
#SBATCH --error=finetune_full_%j.err
#SBATCH --requeue
#SBATCH --open-mode=append

echo "===== JOB START ====="
date; hostname
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
echo "===== Full fine-tuning: 80/10/10 split, 3 epochs, 1 GPU ====="
cd "$PROJECT_ROOT/nova"
$PYTHON finetune_full.py
echo "===== JOB END ====="
date
