#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define source and destination paths
# Note: A trailing slash on the source directory syncs its contents, not the folder itself.
SOURCE_DIR="/home/agcam/images/"
DEST_DIR="agcam@192.168.0.55:/home/agcam/image-inbox/"

# Define log file path
LOG_FILE="/home/agcam/log/rsync_move.log"

echo "$(date) - Starting file transfer..." >> "$LOG_FILE"

# Execute rsync with source removal flag
# -a: Archive mode (preserves timestamps, permissions, symlinks)
# -v: Verbose output
# -h: Human-readable numbers
# --remove-source-files: Deletes successfully transferred files from the source
rsync -avh --remove-source-files "$SOURCE_DIR" "$DEST_DIR" >> "$LOG_FILE" 2>&1

# Check if rsync completed successfully (Exit Code 0)
if [ $? -eq 0 ]; then
    echo "$(date) - Transfer successful. Cleaning up empty source subdirectories..." >> "$LOG_FILE"
    
    # rsync --remove-source-files leaves empty directory structures behind.
    # This command safely deletes only the completely empty directories under the source.
    find "$SOURCE_DIR" -mindepth 1 -type d -empty -delete
    
    echo "$(date) - Operations completed successfully." >> "$LOG_FILE"
else
    echo "$(date) - ERROR: Rsync transfer failed. Source files preserved." >> "$LOG_FILE"
    exit 1
fi
