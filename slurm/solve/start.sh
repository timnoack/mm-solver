#!/bin/bash

NUM_PROCS=$1
TESTCASE=$2 
CONFIG=$3

echo "Usage: $0 <num_procs> <testcase> <config>"

echo "Queueing $NUM_PROCS processes for $TESTCASE with $CONFIG"

# Get absolute path of the project directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_DIR="$SCRIPT_DIR/../.."

LOG_DIR="$PROJECT_DIR/matrices/$TESTCASE/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/${TIMESTAMP}_${CONFIG}_${NUM_PROCS}procs.log"
ERROR_LOG_FILE="$LOG_DIR/${TIMESTAMP}_${CONFIG}_${NUM_PROCS}procs_error.log"

#GPU_RES="--gres=gpu:a100:1"
GPU_RES=""

sbatch -n $NUM_PROCS ${GPU_RES} --error=${ERROR_LOG_FILE} --output=${LOG_FILE} "$SCRIPT_DIR/job.sh" $PROJECT_DIR $TESTCASE $CONFIG