#!/bin/bash
#PBS -l select=1:ncpus=1

# Load parameters from the command line
SCRIPT_DIR=$1
SINGLE_WORKER_PATH=$2
PIXELS=$3
REWRITE=$4
DAUGHTERBOARD_NUMBER=$5
MOTHERBOARD_NUMBER=$6
FIRMWARE_VERSION=$7
TIMESTAMPS=$8
INCLUDE_OFFSET=$9

# Ensure the script runs in the correct directory
cd "$SCRIPT_DIR" || exit 1

# Run the Python worker script
python -c "
from single_worker import SingleWorker
arguments = {
    'pixels': $PIXELS,
    'rewrite': $REWRITE,
    'daughterboard_number': '$DAUGHTERBOARD_NUMBER',
    'motherboard_number': '$MOTHERBOARD_NUMBER',
    'firmware_version': '$FIRMWARE_VERSION',
    'timestamps': $TIMESTAMPS,
    'include_offset': $INCLUDE_OFFSET,
}
worker = SingleWorker(r'$SINGLE_WORKER_PATH', arguments)
worker.calculate_and_save_timestamp_differences_fast(r'$SINGLE_WORKER_PATH', **arguments)
"

