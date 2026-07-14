#!/bin/bash
#SBATCH --job-name=prep_gh
#SBATCH --partition=cs
#SBATCH --account=condo_cs_ali
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/eh3115/Decompliler/Decompliler/results/prep_gh_%j.out
#SBATCH --error=/scratch/eh3115/Decompliler/Decompliler/results/prep_gh_%j.err
cd /scratch/eh3115/Decompliler/Decompliler/nova
/home/eh3115/.conda/envs/nova/bin/python prepare_gigahorse_dataset.py
