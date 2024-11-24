#!/bin/bash

# Variables
SOURCE="quantm@174.59.173.138:/dev/shm/monai/output/"
DESTINATION="/home/azureuser/data/ChestXRLungSegmentation/MAISI"
PASSWORD="quantm"  # Replace with your actual password
MAX_RETRIES=500
RETRY_DELAY=10  # Delay in seconds between retries

# Function to perform rsync with sshpass
perform_rsync() {
    # sshpass -p "$PASSWORD" rsync -avzP --partial "$SOURCE" "$DESTINATION"
    rsync -avzP --partial "$SOURCE" "$DESTINATION"
}

# Main loop
for ((i=1; i<=MAX_RETRIES; i++)); do
    echo "Attempt $i of $MAX_RETRIES..."
    
    # Call the rsync function
    perform_rsync
    
    # Check if the last command was successful
    if [ $? -eq 0 ]; then
        echo "Rsync completed successfully!"
        exit 0  # Exit the script if successful
    else
        echo "Rsync failed. Retrying in $RETRY_DELAY seconds..."
        sleep $RETRY_DELAY  # Wait before retrying
    fi
done

echo "Rsync failed after $MAX_RETRIES attempts."
exit 1  # Exit with failure status