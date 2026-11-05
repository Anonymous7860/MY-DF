#!/bin/bash

# Source the environment variables from setenv.sh
source _internal/setenv.sh

# Activate Python virtual environment
source "$DFL_ROOT/venv/bin/activate"

# Run the main.py script with arguments
python "$DFL_ROOT/main.py" videoed video-from-sequence \
    --input-dir "$WORKSPACE/data_dst/merged" \
    --output-file "$WORKSPACE/result.mov" \
    --reference-file "$WORKSPACE/data_dst.*" \
    --include-audio \
    --lossless

python "$DFL_ROOT/main.py" videoed video-from-sequence \
    --input-dir "$WORKSPACE/data_dst/merged_mask" \
    --output-file "$WORKSPACE/result_mask.mov" \
    --reference-file "$WORKSPACE/data_dst.*" \
    --lossless

# Deactivate python virtual environment
deactivate
