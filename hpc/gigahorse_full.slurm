#!/bin/bash
#SBATCH --job-name=gh_full
#SBATCH --partition=cs
#SBATCH --account=condo_cs_ali
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/eh3115/Decompliler/Decompliler/results/gigahorse_full_%j.out
#SBATCH --error=/scratch/eh3115/Decompliler/Decompliler/results/gigahorse_full_%j.err
echo "JOB START"; date; hostname
module load singularity/3.8.3
export GH_SBX=/scratch/eh3115/Decompliler/gigahorse_sbx
export GH_CACHE=/scratch/eh3115/gh_cache
export GH_WORK=/scratch/eh3115/gh_work_$SLURM_JOB_ID
export GH_NAMES=/scratch/eh3115/Decompliler/Decompliler/results/gigahorse_names.jsonl
export GH_JOBS=32
export GH_BATCH=300
export GH_TIMEOUT=60
mkdir -p $GH_WORK
cd /scratch/eh3115/Decompliler/Decompliler
/home/eh3115/.conda/envs/nova/bin/python run_gigahorse_batch.py
echo "JOB END"; date
