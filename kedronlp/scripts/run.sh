#!/bin/bash

#SBATCH --job-name=run
#SBATCH --nodes=1
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --exclude=graphcore,ceg-victoria,octane[001-008],ceg-brook[01-02]
#SBATCH --output=output.txt
#SBATCH --open-mode=truncate

source /home/${USER}/.bashrc
source activate torch

srun -u python -u doc2vec.py