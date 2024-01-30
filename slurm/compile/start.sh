#!/bin/bash

# Get absolute path of the project directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_DIR="$SCRIPT_DIR/../.."

srun -n 1 --chdir "${PROJECT_DIR}" --mem-per-cpu=4G --cpus-per-task=52 --time=00:29:59  "${SCRIPT_DIR}/job.sh" $PROJECT_DIR