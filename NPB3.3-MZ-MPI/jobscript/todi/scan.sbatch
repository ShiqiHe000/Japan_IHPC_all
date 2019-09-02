#!/bin/bash --login
# submit from ./bin.scalasca directory with "sbatch scan.sbatch"
#SBATCH --job-name="mzmpibt"
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --time=00:10:00
#SBATCH --res=sschool

# benchmark configuration
export OMP_NUM_THREADS=4
PROCS=16
CLASS=B
EXE=./bt-mz_$CLASS.$PROCS

module load scalasca/1.4.2
which scalasca

# Scalasca configuration
NEXUS="scalasca -analyze"
#export EPK_FILTER=
#export SCAN_ANALYZE_OPTS="-isv"

set -x
$NEXUS  aprun -n $PROCS -d $OMP_NUM_THREADS  $EXE

