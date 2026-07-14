#!/bin/bash
#SBATCH --job-name=codebert
#SBATCH --partition=cs
#SBATCH --account=condo_cs_ali
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/eh3115/Decompliler/Decompliler/results/codebert_%j.out
#SBATCH --error=/scratch/eh3115/Decompliler/Decompliler/results/codebert_%j.err
module purge; module load cuda/11.8.0
export HF_HOME=/scratch/eh3115/hf_cache TRANSFORMERS_CACHE=/scratch/eh3115/hf_cache
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
cd /scratch/eh3115/Decompliler/Decompliler
/home/eh3115/.conda/envs/nova/bin/python codebert_sim.py
