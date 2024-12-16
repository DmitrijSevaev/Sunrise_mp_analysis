#!/bin/bash

# Directory to search in (default is the current directory)
TARGET_DIR=${1:-.}

# Remove files matching the pattern "worker_N*" where N is 0-9
echo "Searching for files matching the pattern 'worker_[0-9]*' in $TARGET_DIR..."
for file in "$TARGET_DIR"/worker_[0-9]*; do
    if [ -e "$file" ]; then
        echo "Removing $file"
        rm -f "$file"
    fi
done

echo "Cleanup completed."
