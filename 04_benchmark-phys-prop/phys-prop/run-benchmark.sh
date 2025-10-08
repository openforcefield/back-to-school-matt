#!/usr/bin/env bash
#SBATCH -J back-to-school-matt-benchmark
#SBATCH -p gpu
#SBATCH --account DMOBLEY_LAB_GPU
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --output slurm-%x.%A.out

. ~/.bashrc

TIER="validation"

# make sure output/validation/ exists
mkdir -p output/
mkdir -p output/$TIER

# hope to avoid errors?
export CUDA_VISIBLE_DEVICES=0

# Use the right conda environment
micromamba activate n-vdw-split

FFNAME="split-n-vdw-v1.offxml"

FORCEFIELD="../forcefields/${FFNAME}"

DATASET="../../01_download-data/physprop/final/output/${TIER}-set.json"

echo "Benchmarking ${TIER} with ${FFNAME}"

python benchmark.py             		\
    -i  $DATASET                		\
    -p  0                       		\  # edit this to be 1, rerun, then 2, rerun?
    -ff $FORCEFIELD             		\
    -o  $TIER					\
    -r  1                       		\
    -s  '../../02_fit-vdw/refit/stored_data'    \
    -bp 8500                    		\
    -of request-options.json
