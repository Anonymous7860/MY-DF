#!/bin/bash

# Get path to the file from the first parameter
input_file="$1"

# Source the environment variables from setenv.sh
source _internal/setenv.sh

# Activate Python virtual environment
source "$DFL_ROOT/venv/bin/activate"

# Run the main.py script with arguments
python "$DFL_ROOT/main.py" videoed cut-video \
    --input-file "$input_file"

deactivate