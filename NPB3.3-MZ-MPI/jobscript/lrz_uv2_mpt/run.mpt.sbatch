#!/bin/bash
#SBATCH -o bt-mz.mpt.%j.%N.out
#SBATCH -J bt-mz
#SBATCH --get-user-env
#SBATCH --clusters=uv2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --export=NONE
#SBATCH --time=00:05:00
#SBATCH --reservation=VI-HPS_Workshop

source /etc/profile.d/modules.sh

module list

set -x

# benchmark configuration
CLASS=B
NPROCS=4

EXE=./bt-mz_${CLASS}.${NPROCS}

export OMP_NUM_THREADS=4
export NPB_MZ_BLOAD=0 # disable load balancing with dynamic threads

srun_ps -n $NPROCS -t $OMP_NUM_THREADS $EXE
