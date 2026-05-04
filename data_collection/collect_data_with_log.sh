#!/bin/bash
num_gpus=${1:-"8"}
num_nodes=${2:-"1"}

# Define the log file path
timestamp=$(date +%Y-%m-%d-%H-%M-%S)
log_file="./out_logs/laion-collection-$timestamp.txt"

# make logging dir
if [ ! -d out_logs ]; then
  mkdir out_logs
fi

# Function to log a command and its output with timestamp
log_command() {
  echo "$(date +"%Y-%m-%d %H:%M:%S") | Command: $1" >> "$log_file"
  echo "Output:" >> "$log_file"
  eval "$1" >> "$log_file" 2>&1
  echo "----------------------------------------------------" >> "$log_file"
}

# Start of the script
echo "$(date +"%Y-%m-%d %H:%M:%S") | Code execution started" >> "$log_file"

# Run Python script with timestamp
log_command "bash bev_blip2_laion_multinode.sh ${num_gpus} ${num_nodes}"

# End of the script
echo "$(date +"%Y-%m-%d %H:%M:%S") | Code execution finished" >> "$log_file"
