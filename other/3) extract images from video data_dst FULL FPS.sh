#!/bin/bash

# Source the environment variables from setenv.sh
source _internal/setenv.sh

mkdir -p "$WORKSPACE/data_dst" 2>/dev/null

# Activate Python virtual environment
source "$DFL_ROOT/venv/bin/activate"

# Run the main.py script with arguments
python "$DFL_ROOT/main.py" videoed extract-video \
    --input-file "$WORKSPACE/data_dst.*" \
    --output-dir "$WORKSPACE/data_dst" \
    --fps 0

deactivate