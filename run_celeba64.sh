#!/bin/bash
#SBATCH -J diffae_zdim512                  # Job name
#SBATCH -o slurm_logs/%x_%j.out            # output file (%j expands to jobID)
#SBATCH -e slurm_logs/%x_%j.err            # error log file (%j expands to jobID)
#SBATCH -N 1                               # Total number of nodes requested
#SBATCH -n 1                               # Total number of cores requested
#SBATCH --get-user-env                     # retrieve the users login environment
#SBATCH --mem=8000                         # server memory requested (per node)
#SBATCH -t 48:00:00                        # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov               # Request partition
#SBATCH --gres=gpu:3090:1                  # Type/number of GPUs needed
#SBATCH --requeue

source activate diffae
python -u run_celeba64.py
