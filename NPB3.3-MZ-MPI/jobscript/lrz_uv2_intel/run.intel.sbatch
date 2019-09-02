#!/bin/bash
#SBATCH -o bt-mz.intel.%j.%N.out
#SBATCH -J bt-mz
#SBATCH --get-user-env
#SBATCH --clusters=uv2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --export=NONE
#SBATCH --time=00:02:00

source /etc/profile.d/modules.sh

module list
module unload mpi.mpt
module load mpi.intel
module list

set -x

# benchmark configuration
CLASS=B
NPROCS=4

EXE=./bt-mz_${CLASS}.${NPROCS}

#export KMP_AFFINITY=verbose,granularity=fine,compact
#export KMP_AFFINITY=granularity=fine,balanced
export KMP_AFFINITY=granularity=fine
#export KMP_AFFINITY=granularity=verbose,fine,balanced
export I_MPI_PIN_DOMAIN=socket
export I_MPI_PIN_DOMAIN=omp

export OMP_NUM_THREADS=4
export NPB_MZ_BLOAD=0 # disable load balancing with dynamic threads

srun_ps -n $NPROCS -t $OMP_NUM_THREADS $EXE
