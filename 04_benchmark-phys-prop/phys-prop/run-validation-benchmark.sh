#!/bin/bash
#SBATCH -J benchmark-split-n-vdw
#SBATCH --array=0-1053
#SBATCH -p gpu
#SBATCH -t 08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --account dmobley_lab_gpu
#SBATCH --export ALL
#SBATCH --mem=16gb
#SBATCH --constraint=fastscratch
#SBATCH --output run-logs/slurm-%x.%A-%a.out
#SBATCH --gres=gpu:1


. ~/.bashrc

# Use the right conda environment
micromamba activate n-vdw-split

TIER="validation"

FFNAME="split-n-vdw-v1"

FORCEFIELD="../forcefields/${FFNAME}.offxml"

DIRECTORY="../../01_download-data/physprop/final/output"
DATASET="${DIRECTORY}/${TIER}-set.json"

echo "Benchmarking ${TIER} with ${FFNAME}"

python benchmark.py             		\
    -i  $DATASET                		\
    -p  $SLURM_ARRAY_TASK_ID    		\
    -ff $FORCEFIELD             		\
    -o  $TIER                   		\
    -r  1                       		\
    -s  "../../02_fit-vdw/refit/stored_data/" 	\
    -bp 8039                    		\
    -of request-options.json

