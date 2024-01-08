#!/bin/bash

#SBATCH --job-name=run
#SBATCH --nodes=1
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --exclude=graphcore,octane[001-008],ceg-brook[01-02]
#SBATCH --output=output.txt
#SBATCH --open-mode=truncate

srun -u python -u create_paragraphs.py
srun -u python -u paragraph2vec.py