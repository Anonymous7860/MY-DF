#!/bin/bash
# Source the environment variables from setenv.sh
source _internal/setenv.sh

# Activate Python virtual environment
source "$DFL_ROOT/venv/bin/activate"

# Run the main.py script with arguments
python "$DFL_ROOT/main.py" train \
    --training-data-src-dir "$WORKSPACE/data_src/aligned" \
    --training-data-dst-dir "$WORKSPACE/data_dst/aligned" \
    --pretraining-data-dir "$INTERNAL/pretrain_CelebA" \
    --pretrained-model-dir "$INTERNAL/pretrain_Quick96" \
    --model-dir "$WORKSPACE/model" \
    --model Quick96

# Deactivate python virtual environment
deactivat
