#!/bin/bash
#SBATCH --job-name=eval_clean
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
#SBATCH --output=/scratch/eh3115/Decompliler/Decompliler/results/eval_clean_%j.out
#SBATCH --error=/scratch/eh3115/Decompliler/Decompliler/results/eval_clean_%j.err
echo "JOB START"; date; hostname
module purge; module load cuda/11.8.0
export TOKENIZERS_PARALLELISM=false HF_HOME=/scratch/eh3115/hf_cache TRANSFORMERS_CACHE=/scratch/eh3115/hf_cache CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export CKPT_DIR=/scratch/eh3115/Decompliler/Decompliler/checkpoints/nova-solidity-1.3b-clean/checkpoint-1467
export EVAL_OUT=/scratch/eh3115/Decompliler/Decompliler/results/full_eval_clean.json
export EVAL_CSV=/scratch/eh3115/Decompliler/Decompliler/results/full_scores_clean.csv
echo "Checkpoint: $CKPT_DIR"
cd /scratch/eh3115/Decompliler/Decompliler/nova
/home/eh3115/.conda/envs/nova/bin/python run_full_eval.py
echo "JOB END"; date
