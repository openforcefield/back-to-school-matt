#!/usr/bin/env bash
#SBATCH -J eq-ash-sage-rc2
#SBATCH -p standard
#SBATCH --account dmobley_lab
#SBATCH -t 7-00:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --output slurm-%x.%A.out

. ~/.bashrc

# Use the right conda environment
conda activate n-vdw-split

DATA_DIRECTORY="../../01_download-data/physprop/final/output/"

python equilibrate.py                                             \
    --port                      8112                                    \
    --n-molecules               1000                                    \
    --extra-script-option       "--gres=gpu:1"                          \
    --queue                     "free-gpu"                              \
    --n-gpu                     23                                      \
    --conda-env                 n-vdw-split				\
    --dataset                   "${DATA_DIRECTORY}/training-set.json"


    
