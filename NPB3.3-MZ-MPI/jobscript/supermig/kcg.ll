#!/bin/bash
#
# submit from ./bin directory with "llsubmit kcg.ll"
# hybrid MPI+OpenMP job script for SuperMIG (fat nodes) IBM MPI
#
#@ job_type = parallel
#@ class = test
#@ node = 1
#@ total_tasks = 4
### other example 
##@ tasks_per_node = 4
#@ wall_clock_limit = 0:30:00
#@ job_name = kcg_bt-mz
#@ network.MPI = sn_all,not_shared,us
#@ output = job$(jobid).out
#@ error = job$(jobid).out
##@ notification=always
##@ notify_user=erika.mustermann@xyz.de
#@ queue
. /etc/profile
. /etc/profile.d/modules.sh

module use /lrz/sys/smuc_tools/modules
module load UNITE kcachegrind

set -x
export MP_SINGLE_THREAD=no
export OMP_NUM_THREADS=4
# Pinning
export MP_TASK_AFFINITY=core:$OMP_NUM_THREADS

# Application load balancing
export NPB_MZ_BLOAD=0

# kcachegrind configuration
NEXUS = "valgrind --tool=callgrind --cache-sim=yes --separate-threads=yes"

mpiexec -n 4  $NEXUS  ./bt-mz_B.4

