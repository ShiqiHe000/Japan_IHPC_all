#!/bin/bash --login
# submit from ./bin.scorep directory with "sbatch scalasca2.sbatch"
#SBATCH --job-name="mzmpibt"
#SBATCH --output=scan-%A.out
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

#export SCOREP_FILTERING_FILE=../config/scorep.filt
#export SCOREP_TOTAL_MEMORY=50M

NEXUS="scalasca -analyze -s"
$NEXUS  mpirun -np $PROCS  $EXE

