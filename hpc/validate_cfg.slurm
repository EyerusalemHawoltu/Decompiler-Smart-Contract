#!/bin/bash
#SBATCH --job-name=validate_cfg
#SBATCH --partition=cs
#SBATCH --account=condo_cs_ali
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/eh3115/Decompliler/Decompliler/results/validate_cfg_%j.out
#SBATCH --error=/scratch/eh3115/Decompliler/Decompliler/results/validate_cfg_%j.err

echo "===== JOB START ====="
date
hostname

PYTHON=/home/eh3115/.conda/envs/nova/bin/python
PROJECT=/scratch/eh3115/Decompliler/Decompliler

echo "Bytecode versions : $(ls $PROJECT/Contracts_Bytecode | wc -l)"
echo "Solidity versions : $(ls $PROJECT/Contracts_By_Version_Cleaned | wc -l)"
echo "Total hex files   : $(find $PROJECT/Contracts_Bytecode -name '*.hex' | wc -l)"

cd $PROJECT
$PYTHON validate_cfg_names.py --workers 16

echo "===== JOB END ====="
date
