#!/bin/bash
#SBATCH --job-name=nova_e
#SBATCH --partition=cs
#SBATCH --account=condo_cs_ali
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/eh3115/Decompliler/Decompliler/results/%x_%j.out
#SBATCH --error=/scratch/eh3115/Decompliler/Decompliler/results/%x_%j.err
module purge; module load cuda/11.8.0
export TOKENIZERS_PARALLELISM=false HF_HOME=/scratch/eh3115/hf_cache TRANSFORMERS_CACHE=/scratch/eh3115/hf_cache CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
cd /scratch/eh3115/Decompliler/Decompliler/nova
/home/eh3115/.conda/envs/nova/bin/python run_full_eval.py
