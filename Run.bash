#!/bin/bash
ENV_NAME="ece8930"         # Your actual conda environment
MAIN_SCRIPT="Run.py"       # Replace if your main script name is different

set -e

echo "Job started on $(hostname) at $(date)"

mkdir -p logs

echo "Loading Anaconda module..."
module load anaconda3/2023.09

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "Activating environment: $ENV_NAME"
conda activate "$ENV_NAME"

echo "Using Python: $(which python)"
python --version

if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi
fi

echo "Running script: $MAIN_SCRIPT"
python "$MAIN_SCRIPT"

echo "Job finished at $(date)"

