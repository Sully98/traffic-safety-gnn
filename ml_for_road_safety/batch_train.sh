#!/bin/bash
#SBATCH -J traffic-train       # job name to display in squeue
#SBATCH -o output.txt    # standard output file
#SBATCH -e error.txt     # standard error file
#SBATCH -p standard-s              # requested partition
#SBATCH --mem=30G           # memory in GB
#SBATCH -c 8

cd ~/tsg/ml_for_road_safety
module load conda
mamba activate road-gnn-models

python -u sully_test.py
