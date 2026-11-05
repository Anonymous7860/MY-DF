#!/bin/bash

# Source the environment variables from setenv.sh
source _internal/setenv.sh

# Activate Python virtual environment
source "$DFL_ROOT/venv/bin/activate"

# Run the main.py script with arguments
python "$DFL_ROOT/main.py" extract \
    --input-dir "$WORKSPACE/data_dst" \
    --output-dir "$WORKSPACE/data_dst/aligned" \
    --detector manual \
    --max-faces-from-image 0 \
    --output-debug \
    --manual-output-debug-fix

deactivate