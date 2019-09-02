#!/bin/bash
#SBATCH -o bt-mz.intel.%j.%N.out
#SBATCH -J bt-mz
#SBATCH --get-user-env
#SBATCH --clusters=uv2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --export=NONE
#SBATCH --time=00:07:00

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

#export KMP_AFFINITY=granularity=fine,compact
#export KMP_AFFINITY=granularity=fine,balanced
export KMP_AFFINITY=granularity=fine
#export KMP_AFFINITY=verbose,granularity=fine,balanced
export I_MPI_PIN_DOMAIN=socket
export I_MPI_PIN_DOMAIN=omp

export OMP_NUM_THREADS=4
export NPB_MZ_BLOAD=0 # disable load balancing with dynamic threads

# Score-P measurement configuration
# Timer 'tsc' is unreliable if threads of a team are distributed over
# cores. Use the reliable 'gettimeofday' or 'clock_gettime' unless you
# are certain about the thread distribution done by srun_ps and
# omplace.
export SCOREP_TIMER=clock_gettime
#export SCOREP_EXPERIMENT_DIRECTORY=scorep_sum
#export SCOREP_FILTERING_FILE=../config/scorep.filt
#export SCOREP_METRIC_PAPI=PAPI_TOT_INS,PAPI_FP_INS
#export SCOREP_METRIC_PAPI_PER_PROCESS=PAPI_L2_TCM
#export SCOREP_METRIC_RUSAGE=ru_stime
#export SCOREP_METRIC_RUSAGE_PER_PROCESS=ru_maxrss

srun_ps -n $NPROCS -t $OMP_NUM_THREADS $EXE
