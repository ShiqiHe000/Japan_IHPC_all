#!/bin/bash
# submit from ./bin.scorep directory with "sbatch scalasca_mic.sbatch"
#
#SBATCH -J mzmpibt          # Job name
#SBATCH -o mzmpibt.o%j      # Name of stdout output file (%j expands to jobId)
#SBATCH -e mzmpibt.e%j      # Name of stderr output file (%j expands to jobId)
#SBATCH -p normal-mic       # Queue with production MIC nodes
#SBATCH -N 2                # Total number of nodes requested (16 cores/node)
#SBATCH -n 32               # Total number of MPI tasks requested (dummy, since we are only using the MICs)
#SBATCH -t 00:05:00         # Run time (hh:mm:ss) - 5 minutes

# MIC settings (15 MPI tasks per MIC, 4 OpenMP threads each)
export MIC_PPN=15
export MIC_OMP_NUM_THREADS=4

# Benchmark configuration (disable load balancing with threads)
export NPB_MZ_BLOAD=0
PROCS=30
CLASS=B

# Score-P measurement configuration
#export SCOREP_FILTERING_FILE=../config/scorep.filt
#export SCOREP_TOTAL_MEMORY=32M
#export SCOREP_METRIC_PAPI=PAPI_TOT_INS,PAPI_TOT_CYC

# Scalasca configuration
export SCAN_ANALYZE_OPTS="--time-correct"

# Run the application using Scalasca nexus
scalasca -analyze ibrun.symm -m ./bt-mz_${CLASS}.${PROCS}
