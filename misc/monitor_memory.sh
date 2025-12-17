#!/bin/bash

# The process name to search for (e.g., "python")
PROCESS_NAME="python"
# The command line argument that distinguishes the process (e.g., "script.py")
COMMAND_ARG="compute_ground_truth_metrics.py"

# Output CSV file
OUTPUT_FILE="memory_usage.csv"

# Initialize the output CSV file with headers
echo "Timestamp,Total Memory Usage (KB)" > $OUTPUT_FILE

# Function to accumulate memory usage
accumulate_memory() {
  local pid=$1
  local mem_usage=$(ps -p $pid -o rss=)
  if [ -n "$mem_usage" ]; then
    echo "$mem_usage"
  else
    echo "0"
  fi
}

# Function to calculate total memory usage of a process and its direct children
calculate_total_memory() {
  local pid=$1
  local total_mem=0

  # Accumulate memory usage for the process itself
  total_mem=$(accumulate_memory $pid)

  # Accumulate memory usage for all child processes
  for child_pid in $(pgrep -P $pid); do
    total_mem=$(echo "$total_mem + $(accumulate_memory $child_pid)" | bc)
  done

  # Return total memory usage in KB
  echo "$total_mem"
}

# Monitor every 1 second
while true; do
  # Get the current timestamp
  TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

  # Find the PID of the parent process with specific command line arguments
  PARENT_PID=$(pgrep -f "${PROCESS_NAME}.*${COMMAND_ARG}")

  if [ -z "$PARENT_PID" ]; then
    # If the process is not detected, log zero
    TOTAL_MEM_KB="0"
  else
    # Calculate the total memory usage of the parent process and its children
    TOTAL_MEM_KB=$(calculate_total_memory $PARENT_PID)
  fi

  # Append the data to the CSV file
  echo "$TIMESTAMP,$TOTAL_MEM_KB" >> $OUTPUT_FILE

  # Wait for 1 second
  sleep 1
done
