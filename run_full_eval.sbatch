#!/bin/bash
#SBATCH --job-name=nova_full_eval
#SBATCH --partition=cs
#SBATCH --account=condo_cs_ali
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/eh3115/Decompliler/Decompliler/results/full_eval_%j.out
#SBATCH --error=/scratch/eh3115/Decompliler/Decompliler/results/full_eval_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"

source ~/.bashrc
conda activate nova

cd /scratch/eh3115/Decompliler/Decompliler/nova
python run_full_eval.py

echo "Finished: $(date)"
