#!/bin/bash

# ========== BASE ENV ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INTERNAL="$SCRIPT_DIR"

# Override some standard environment variables with local paths
LOCALENV_DIR="$INTERNAL/_e"
TMP="$LOCALENV_DIR/t"
TEMP="$LOCALENV_DIR/t"
HOME="$LOCALENV_DIR/u"
HOMEPATH="$LOCALENV_DIR/u"
USERPROFILE="$LOCALENV_DIR/u"
APPDATA="$LOCALENV_DIR/u"

# ========== ADDITIONAL ENV ==========
XNVIEW_PATH="$INTERNAL/XnView"
PATH="$XNVIEWMP_PATH:$PATH"
WORKSPACE="$INTERNAL/../workspace"
DFL_ROOT="$INTERNAL/DeepFaceLab"

#QT_QPA_PLATFORM=offscreen

# Export variables for child processes
export LOCALENV_DIR TMP TEMP HOME HOMEPATH USERPROFILE LOCALAPPDATA APPDATA \
       XNVIEW_PATH PATH WORKSPACE DFL_ROOT