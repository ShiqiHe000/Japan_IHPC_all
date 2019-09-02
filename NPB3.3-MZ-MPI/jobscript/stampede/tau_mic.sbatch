#!/bin/bash
# submit from ./bin.scorep directory with "sbatch tau_mic.sbatch"
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

# TAU measurement configuration

# Run the application

export PATH=/home1/03529/tg828282/Software/tau_latest/mic_linux/bin:$PATH
export PATH=/home1/00494/tg457572/pkgs/perl-mic/bin:$PATH
ibrun.symm -m tau_exec -T ompt bt-mz_${CLASS}.${PROCS}
