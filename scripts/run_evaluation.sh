#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --account=cedar
#SBATCH --output=evaluation_%j.out
#SBATCH --error=evaluation_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

cd /home/xinyiw/

module load Miniforge3/25.11.0-1-jupyter-base

python evaluation_loop.py
