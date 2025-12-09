#!/bin/bash


ENV_NAME="ece8930"           
PYTHON_VERSION="3.10"        
REQ_FILE="requirements.txt"  


set -e  

echo "Loading Anaconda module on Palmetto..."
module load anaconda3/2023.09


source "$(conda info --base)/etc/profile.d/conda.sh"


if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Creating conda environment '$ENV_NAME' with python=$PYTHON_VERSION ..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

echo "Activating environment '$ENV_NAME'..."
conda activate "$ENV_NAME"

if [ ! -f "$REQ_FILE" ]; then
    echo "ERROR: requirements file '$REQ_FILE' not found!"
    exit 1
fi

echo "Installing dependencies from $REQ_FILE ..."
pip install -r "$REQ_FILE"

echo "Environment setup complete."
conda list | head

