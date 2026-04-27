#!/bin/bash
#SBATCH --job-name=eco4cast
#SBATCH --account=cedar
#SBATCH --output=eco4cast_%j.out
#SBATCH --error=eco4cast_%j.err
#SBATCH --time=15:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

cd /home/xinyiw

module load Miniforge3/25.11.0-1-jupyter-base

python eco4cast_loop.py
