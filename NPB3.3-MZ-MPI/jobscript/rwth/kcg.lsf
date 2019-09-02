#!/usr/bin/env zsh
# submit this job from ./bin directory with "bsub < kcg.lsf"

#BSUB -J mzmpibt_kcg
#BSUB -o mzmpibt_kcg.%J
#BSUB -W 15
#BSUB -M 512

#BSUB -n 4
#BSUB -R "span[ptile=2]"
#BSUB -a intelmpi
#BSUB -x

# specify a queue OR use the "vihps" workshop reservation
###BSUB -m mpi-s
#BSUB -U vihps

export OMP_NUM_THREADS=6

module swap openmpi intelmpi
module load UNITE kcachegrind
module list

set -x

PRELOAD="valgrind --tool=callgrind --cache-sim=yes --separate-threads=yes"
$MPIEXEC $FLAGS_MPI_BATCH  $PRELOAD  bt-mz_B.4

