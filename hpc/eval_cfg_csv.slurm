#!/bin/bash
#SBATCH --job-name=cfg_eval_csv
#SBATCH --partition=cs
#SBATCH --account=condo_cs_ali
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/eh3115/Decompliler/Decompliler/results/cfg_eval/cfg_eval_%j.out
#SBATCH --error=/scratch/eh3115/Decompliler/Decompliler/results/cfg_eval/cfg_eval_%j.err

echo "===== JOB START ====="
date
hostname

PYTHON=/home/eh3115/.conda/envs/nova/bin/python
PROJECT=/scratch/eh3115/Decompliler/Decompliler

# Create output directory (SLURM needs it for --output before the job runs)
mkdir -p $PROJECT/results/cfg_eval

echo "Bytecode versions : $(ls $PROJECT/Contracts_Bytecode | wc -l)"
echo "Solidity versions : $(ls $PROJECT/Contracts_By_Version_Cleaned | wc -l)"
echo "Total hex files   : $(find $PROJECT/Contracts_Bytecode -name '*.hex' | wc -l)"

cd $PROJECT
$PYTHON eval_cfg_csv.py --workers 16

echo ""
echo "===== Output files ====="
ls -lh $PROJECT/results/cfg_eval/
echo ""
echo "===== version_summary.csv ====="
cat $PROJECT/results/cfg_eval/version_summary.csv

echo "===== JOB END ====="
date
