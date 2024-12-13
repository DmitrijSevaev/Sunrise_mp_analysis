#!/bin/bash

# Add local Python packages to PYTHONPATH
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH

# Define paths
DATA_PATH="C:\\Users\\fintv\\Desktop\\CAPADS\\Sunrise\\sunrise_testing_script\\raw_data"
DISTRIBUTED_DAT_PATH="C:\\Users\\fintv\\Desktop\\CAPADS\\Sunrise\\sunrise_testing_script\\distributed_dat_files"
OUTPUT_FEATHERS_PATH="C:\\Users\\fintv\\Desktop\\CAPADS\\Sunrise\\sunrise_testing_script\\output_feathers"
BACKUP_DAT_PATH="C:\\Users\\fintv\\Desktop\\CAPADS\\Sunrise\\sunrise_testing_script\\backup_dat_files"
SINGLE_WORKER_DAT_PATH="${DISTRIBUTED_DAT_PATH}\\chunk_"

# Arguments for delta t calculation
NUMBER_OF_WORKERS=10
PIXELS="[144, 171]"
REWRITE="True"
DAUGHTERBOARD_NUMBER="NL11"
MOTHERBOARD_NUMBER="#33"
FIRMWARE_VERSION="2212b"
TIMESTAMPS=300
INCLUDE_OFFSET="False"

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

# Check if pytz is installed; if not, install it
python -c "import pytz" 2>/dev/null || pip install --user pytz

# Check if pandas is installed; if not, install it
python -c "import pandas" 2>/dev/null || pip install --user pandas

# import pandas manually to check version
python -c "import pandas as pd; print(pd.__version__)"

# Step 1: Distribute files
echo "Distributing files..." >> log.txt
python -c "
from files_distributor import FilesDistributor
distributor = FilesDistributor(r'$DATA_PATH', r'$DISTRIBUTED_DAT_PATH', $NUMBER_OF_WORKERS)
distributor.distribute()
"

# Restore data directory from backup
restore_from_backup "$DATA_PATH" "$BACKUP_DAT_PATH"

# Step 2: Process files with workers
echo "Processing files with $NUMBER_OF_WORKERS workers..." >> log.txt
for ((i=0; i<NUMBER_OF_WORKERS; i++)); do
    SINGLE_WORKER_PATH="${SINGLE_WORKER_DAT_PATH}${i}"
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
done

# Step 3: Combine files
echo "Combining files..." >> log.txt
python -c "
from files_combiner import FilesCombiner
combiner = FilesCombiner(r'$DISTRIBUTED_DAT_PATH', r'$OUTPUT_FEATHERS_PATH')
combiner.combine()
"

# Step 4: Clean up
echo "Cleaning up..." >> log.txt
clear_dirs "$DISTRIBUTED_DAT_PATH"
#clear_dirs "$DATA_PATH"

# Log end time and duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Script ended at: $(date). Time taken: $DURATION seconds" >> log.txt
echo "=======================================================" >> log.txt
echo "" >> log.txt
