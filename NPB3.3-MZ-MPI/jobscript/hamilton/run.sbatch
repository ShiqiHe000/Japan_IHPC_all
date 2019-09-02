#!/bin/bash --login
# submit from ./bin directory with "sbatch run.sbatch"
#SBATCH --job-name="mzmpibt"
#SBATCH --output=run-%A.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --partition=par6.q

# benchmark configuration
export NPB_MZ_BLOAD=0
export OMP_NUM_THREADS=3
PROCS=8
CLASS=C
EXE=./bt-mz_$CLASS.$PROCS

set -x
mpirun -np $PROCS  $EXE

