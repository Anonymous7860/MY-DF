#!/bin/bash

# Source the environment variables from setenv.sh
source _internal/setenv.sh

# Run XnView
nohup "$XNVIEW_PATH/xnview.sh" "$WORKSPACE/data_src/aligned" >/dev/null 2>&1 &
