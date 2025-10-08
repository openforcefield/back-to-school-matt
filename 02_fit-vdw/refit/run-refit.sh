#!/usr/bin/env bash
#SBATCH -J back-to-school-matt-refit
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

python refit.py                                             	\
    --port                      8117                    	\
    --n-min-workers         	1                          	\
    --n-max-workers         	23                        	\
    --memory-per-worker     	4                           	\
    --walltime              	"08:00:00"                  	\
    --queue                 	"gpu"                       	\
    --conda-env                 n-vdw-split			\
    --extra-script-option   	"--gres=gpu:1"
