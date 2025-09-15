#!/usr/bin/env bash
#SBATCH -J back-to-school-matt-benchmark
#SBATCH -p gpu
#SBATCH --account DMOBLEY_LAB_GPU
#SBATCH -t 7-00:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --output slurm-%x.%A.out

. ~/.bashrc

# Use the right conda environment
micromamba activate n-vdw-split

FFNAME="split-n-vdw-v1.offxml"

FORCEFIELD="../forcefields/${FFNAME}.offxml"

DATASET="../../01_download-data/physprop/final/output/training-set.json"

echo "Benchmarking ${TIER} with ${FFNAME}"

python benchmark.py             \
    -i  $DATASET                \
    -p  $SLURM_ARRAY_TASK_ID    \
    -ff $FORCEFIELD             \
    -o  training                \
    -r  1                       \
    -s  stored_data             \
    -bp 8500                    \
    -of request-options.json
