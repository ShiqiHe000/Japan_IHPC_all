#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=GPU-shared
#SBATCH --ntasks-per-node 1
#SBATCH --time=00:00:30
#SBATCH --gres=gpu:p100:1
#SBATCH --res challenge
#SBATCH -A ac560tp
set -x
module load cuda/9.2;
./run.sh ${1} openacc small ${2}
