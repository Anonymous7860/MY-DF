#!/bin/bash

# Source the environment variables from setenv.sh
source _internal/setenv.sh

# Activate Python virtual environment
source "$DFL_ROOT/venv/bin/activate"

# Run the main.py script with arguments
python "$DFL_ROOT/main.py" merge \
    --input-dir "$WORKSPACE/data_dst" \
    --output-dir "$WORKSPACE/data_dst/merged" \
    --output-mask-dir "$WORKSPACE/data_dst/merged_mask" \
    --aligned-dir "$WORKSPACE/data_dst/aligned" \
    --model-dir "$WORKSPACE/model" \
    --model AMP

# Deactivate python virtual environment
deactivate
