#!/bin/bash

echo "Press space if you want to delete content in workspace folder"
read -n 1 -s -r -p "Press any key to continue..."
echo ""

source _internal/setenv.sh

echo "Creating/Recreating workspace folder structure..."

# Create the main workspace directory (if not exists)
mkdir -p "$WORKSPACE"

# Clean and recreate data_src
rm -rf "$WORKSPACE/data_src"
mkdir -p "$WORKSPACE/data_src/aligned"

# Clean and recreate data_dst
rm -rf "$WORKSPACE/data_dst"
mkdir -p "$WORKSPACE/data_dst/aligned"

# Clean and recreate model
rm -rf "$WORKSPACE/model"
mkdir -p "$WORKSPACE/model"

echo "DONE"