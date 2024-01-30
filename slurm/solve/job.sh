#!/bin/bash
#SBATCH --time 00:29:59  
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G

echo "Usage: $0 <project_dir> <testcase> <config>"

PROJECT_DIR=$1
TESTCASE=$2
CONFIG=$3

CONFIG_FILE="$PROJECT_DIR/matrices/$TESTCASE/$CONFIG.jsonc"
EXEC_FILE="$PROJECT_DIR/build_native/src/matrix-solver"

source $PROJECT_DIR/.vscode/lichtenbergenv.sh

echo "Date: $(date)"
echo "Case: $TESTCASE"
echo "Config: $CONFIG_FILE"
echo "Number of total CPUs: $SLURM_NTASKS"
echo "Using executable: $EXEC_FILE"
lscpu

# If a GPU is attached to the node, print some information about it
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Number of GPUs: $CUDA_VISIBLE_DEVICES"
    nvidia-smi -L
fi

mpirun $EXEC_FILE --config $CONFIG_FILE --framework hypre
