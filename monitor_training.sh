#!/bin/bash

# Create log directory
LOG_DIR="$(pwd)/training_logs"
mkdir -p $LOG_DIR

echo "Logging system metrics to $LOG_DIR"

# Start system monitoring using sar
sar -u 5 > $LOG_DIR/system_cpu_log.txt &
SAR_CPU_PID=$!
sar -r 5 > $LOG_DIR/system_memory_log.txt &
SAR_MEM_PID=$!
sar -d 5 > $LOG_DIR/system_disk_log.txt &
SAR_DISK_PID=$!

# Start GPU monitoring
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,temperature.gpu,memory.used,memory.free --format=csv -l 5 > $LOG_DIR/gpu_log.csv &
NVIDIA_PID=$!

# Log kernel messages for debugging crashes
# dmesg -wH > $LOG_DIR/dmesg_log.txt &
# DMESG_PID=$!

# Log system events
journalctl -f > $LOG_DIR/journal_log.txt &
JOURNAL_PID=$!

# Function to kill background processes when script exits
cleanup() {
    echo "Stopping monitoring processes..."
    kill -9 $SAR_CPU_PID $SAR_MEM_PID $SAR_DISK_PID $NVIDIA_PID $JOURNAL_PID
    exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM signals to stop logging
trap cleanup SIGINT SIGTERM

# Enable core dumps
ulimit -c unlimited

# Run training script
PYTHONPATH=$(pwd) accelerate launch train_ddpm.py 2>&1 | tee $LOG_DIR/training_log.txt

# Run cleanup when training finishes
cleanup
