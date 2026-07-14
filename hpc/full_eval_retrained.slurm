#!/bin/bash
#SBATCH --job-name=nova_eval2
#SBATCH --partition=cs
#SBATCH --account=condo_cs_ali
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/eh3115/Decompliler/Decompliler/results/nova_eval2_%j.out
#SBATCH --error=/scratch/eh3115/Decompliler/Decompliler/results/nova_eval2_%j.err

echo "===== JOB START =====" && date && hostname
module purge
module load cuda/11.8.0
PYTHON=/home/eh3115/.conda/envs/nova/bin/python
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch/eh3115/hf_cache
export TRANSFORMERS_CACHE=/scratch/eh3115/hf_cache
export CUDA_VISIBLE_DEVICES=0
# Evaluate the freshly retrained model on the FULL test set (42,989 functions)
export CKPT_DIR=/scratch/eh3115/Decompliler/Decompliler/checkpoints/nova-solidity-1.3b/checkpoint-12060
echo "Checkpoint: $CKPT_DIR"
nvidia-smi
cd /scratch/eh3115/Decompliler/Decompliler/nova
$PYTHON run_full_eval.py
echo "===== JOB END =====" && date
