# submit from ./bin.scalasca directory with "llsubmit scan.ll"

# @ job_type = bluegene
# @ job_name = npb_mz_bt
# @ comment = "measurement run"
# @ output = $(job_name).$(jobid).out
# @ error = $(job_name).$(jobid).out
# @ environment = COPY_ALL;
# @ wall_clock_limit = 00:30:00
# @ notify_user = $(user)@fz-juelich.de
# @ notification = never
# @ bg_connection = PREFER_TORUS
# @ bg_size = 32
# @ queue

export MPIRUN_MODE=smp
export BG_SIZE=$LOADL_BG_SIZE

# Calculate number of processes/threads based on partition 'size' and 'mode'
case "$MPIRUN_MODE" in
  vn|VN)
    MPIRUN_NP=`echo "4*$BG_SIZE" | bc`
    export OMP_NUM_THREADS=1
    ;;
  dual|DUAL)
    MPIRUN_NP=`echo "2*$BG_SIZE" | bc`
    export OMP_NUM_THREADS=2
    ;;
  smp|SMP|*)
    MPIRUN_NP=$BG_SIZE
    export OMP_NUM_THREADS=4
    ;;
esac

export MPIRUN_EXP_ENV="OMP_NUM_THREADS"
export MPIRUN_ENV="NPB_MAX_THREADS=$OMP_NUM_THREADS"

module load UNITE scalasca

# Scalasca experiment configuration
NEXUS="scalasca -analyze -s"
#export EPK_FILTER=

set -x

$NEXUS mpirun -mode $MPIRUN_MODE -np $MPIRUN_NP  bt-mz_B.$MPIRUN_NP

exit
