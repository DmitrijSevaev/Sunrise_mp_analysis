#!/bin/bash

# Define paths
SCRIPT_DIR="/home/sevaedmi/sunrise_testing_script"
DATA_PATH="${SCRIPT_DIR}/raw_data"
DISTRIBUTED_DAT_PATH="${SCRIPT_DIR}/distributed_dat_files"
OUTPUT_FEATHERS_PATH="${SCRIPT_DIR}/output_feathers"
BACKUP_DAT_PATH="${SCRIPT_DIR}/backup_dat_files"
SINGLE_WORKER_DAT_PATH="${DISTRIBUTED_DAT_PATH}/chunk_"

# Arguments for delta t calculation
NUMBER_OF_WORKERS=10
PIXELS="[144, 171]"
REWRITE="True"
DAUGHTERBOARD_NUMBER="NL11"
MOTHERBOARD_NUMBER="#33"
FIRMWARE_VERSION="2212b"
TIMESTAMPS=300
INCLUDE_OFFSET="False"

# Get arguments passed to script
CLEANUP_FLAG=""
while getopts "d" opt; do
	case $opt in
		d)
			CLEANUP_FLAG="TRUE"
			;;
	esac
done

# Ensure the script runs in the correct directory
cd "$SCRIPT_DIR" || exit 1

# Request CPUs and log job start
echo "Requesting $NUMBER_OF_WORKERS CPUs for the job..." >> log.txt

# Clear and prepare directories
function clear_dirs {
    if [ -d "$1" ]; then
        rm -rf "$1"
    fi
    mkdir -p "$1"
}

# Restore from backup
function restore_from_backup {
    if [ -d "$1" ]; then
        rm -rf "$1"
    fi
    cp -r "$2" "$1"
}

# Clear directories
clear_dirs "$DISTRIBUTED_DAT_PATH"
clear_dirs "$OUTPUT_FEATHERS_PATH"

# Log start time
START_TIME=$(date +%s)
echo "=======================================================" >> log.txt
echo "Script started at: $(date)" >> log.txt

# Step 1: Distribute files
echo "Distributing files..." >> log.txt
python -c "
from files_distributor import FilesDistributor
distributor = FilesDistributor(r'$DATA_PATH', r'$DISTRIBUTED_DAT_PATH', $NUMBER_OF_WORKERS)
distributor.distribute()
"

# Restore data directory from backup
restore_from_backup "$DATA_PATH" "$BACKUP_DAT_PATH"

# Step 2: Submit jobs for single workers
echo "Submitting $NUMBER_OF_WORKERS jobs..." >> log.txt
for ((i=0; i<NUMBER_OF_WORKERS; i++)); do
    SINGLE_WORKER_PATH="${SINGLE_WORKER_DAT_PATH}${i}"
    qsub -N worker_$i -l select=1:ncpus=1 -- sunrise_testing_script/single_worker.sh \
        "$SCRIPT_DIR" "$SINGLE_WORKER_PATH" "$PIXELS" "$REWRITE" \
        "$DAUGHTERBOARD_NUMBER" "$MOTHERBOARD_NUMBER" "$FIRMWARE_VERSION" \
        "$TIMESTAMPS" "$INCLUDE_OFFSET"
done

# Step 3: Wait for all jobs to finish
echo "Waiting for all jobs to finish..." >> log.txt

# Loop to check if all jobs are finished
while true; do
    RUNNING_JOBS=$(qstat | grep "worker_" | wc -l)  # Check for running jobs
    if [ "$RUNNING_JOBS" -eq 0 ]; then
        echo "All jobs have finished."
        break  # Exit the loop when no jobs are running
    fi
    sleep 2  # Wait for 10 seconds before checking again
done

# Step 4: Combine files (wait for jobs to finish before this step)
echo "Combining files..." >> log.txt
python -c "
from files_combiner import FilesCombiner
combiner = FilesCombiner(r'$DISTRIBUTED_DAT_PATH', r'$OUTPUT_FEATHERS_PATH')
combiner.combine()
"

# Step 5: Clean up
echo "Cleaning up..." >> log.txt
clear_dirs "$DISTRIBUTED_DAT_PATH"
#clear_dirs "$DATA_PATH"

# Conditional outputs deleting if -d argument passed
if [ "$CLEANUP_FLAG" == "TRUE" ]; then
    echo "Removing worker files..."
    rm -v ./worker_*.e* ./worker_*.o*  # Verbose removal
fi

# Log end time and duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Script ended at: $(date). Time taken: $DURATION seconds" >> log.txt
echo "=======================================================" >> log.txt
echo "" >> log.txt

