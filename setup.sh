#!/bin/bash
set -e  # Exit immediately if any command fails

# Define the absolute base directory so the script never gets lost
BASE_DIR="/home/user/Desktop/DeepFaceLab-RTX5000"

echo "-------------------------------------------------------"
echo "Creating the DFL directory structure"
echo "-------------------------------------------------------"

# Move everything into place safely
DFL_DIR="other/_internal/DeepFaceLab"
mkdir -p "$DFL_DIR/other"

rsync -a --exclude="_internal/DeepFaceLab" other/ "$DFL_DIR/other"
rsync -q -av --remove-source-files --exclude="setup.sh" --exclude=other/ ./ "$DFL_DIR/"
find ./ -type d -empty -not -name keep_folder -delete

# Take everything out of the other dir safely
if [ -d other ]; then
    mv other/* . 2>/dev/null || true
    rmdir other 2>/dev/null || true
fi

# Add folders for app data
mkdir -p _internal/_e/u

echo "-------------------------------------------------------"
echo "Configuring CUDA and cuDNN Dependencies"
echo "-------------------------------------------------------"
sudo rm -f /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list

sudo apt-get update
sudo apt-get -y install cudnn-cuda-12 || echo "cuDNN package installation skipped (already managed by container)."

# Add CUDA and explicit virtual environment NVIDIA paths to .bashrc safely
sudo chown $USER:$USER ~/.bashrc
if ! grep -q "CUDA_HOME" ~/.bashrc; then
    echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
    echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:$BASE_DIR/_internal/DeepFaceLab/venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$BASE_DIR/_internal/DeepFaceLab/venv/lib/python3.10/site-packages/nvidia/cudnn/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
    echo 'export LD_RUN_PATH=$CUDA_HOME/lib64:$LD_RUN_PATH' >> ~/.bashrc
fi

echo "-------------------------------------------------------"
echo "Installing Python 3.10 and System Utilities"
echo "-------------------------------------------------------"
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-distutils python3.10-dev ffmpeg libpulse-mainloop-glib0 -y

echo "-------------------------------------------------------"
echo "Downloading XnView (SourceForge Global Mirror Network)"
echo "-------------------------------------------------------"
# Pulls from a high-speed global mirror network
curl -L "https://sourceforge.net/projects/xnview-mp/files/XnViewMP-linux-x64.deb/download" -o "XnViewMP-linux-x64.deb"
ar x "XnViewMP-linux-x64.deb"

mkdir -p _internal/XnView
DATA_ARCHIVE=$(ls data.tar.* 2>/dev/null || true)
if [ -n "$DATA_ARCHIVE" ]; then
    tar -xf "$DATA_ARCHIVE" -C _internal/XnView/
    rm -f "XnViewMP-linux-x64.deb" control.tar.* data.tar.* debian-binary
    mv _internal/XnView/opt/XnView/* _internal/XnView/ 2>/dev/null || true
    rm -rf _internal/XnView/usr _internal/XnView/opt
fi


echo "-------------------------------------------------------"
echo "Creating Virtual Environment and Patching Dependencies"
echo "-------------------------------------------------------"
cd "$BASE_DIR/_internal/DeepFaceLab"
rm -rf venv
python3.10 -m venv venv

# DYNAMIC FIX: If a previous aborted run misplaced requirements.txt, find it automatically
if [ ! -f requirements.txt ]; then
    echo "requirements.txt not found in local directory. Searching project tree..."
    REQ_PATH=$(find "$BASE_DIR" -name "requirements.txt" ! -path "*/venv/*" -print -quit)
    if [ -n "$REQ_PATH" ]; then
        echo "Found requirements.txt at: $REQ_PATH. Restoring it to the correct location..."
        cp "$REQ_PATH" ./requirements.txt
    else
        echo "ERROR: requirements.txt cannot be found anywhere in $BASE_DIR."
        exit 1
    fi
fi

echo "Modifying requirements.txt to solve the NumPy 2.2.0 / TensorFlow conflict..."
sed -i 's/numpy.*/numpy==1.26.4/g' requirements.txt

./venv/bin/pip install --upgrade pip
./venv/bin/pip install --no-deps -r requirements.txt


echo "-------------------------------------------------------"
echo "Creating Workspace Directories (Forced Absolute Paths)"
echo "-------------------------------------------------------"
# FIXED: Using explicit, un-ignorable absolute paths targeting your Desktop directory
mkdir -p "$BASE_DIR/workspace/data_dst"
mkdir -p "$BASE_DIR/workspace/data_src/aligned"
mkdir -p "$BASE_DIR/workspace/model"

# Clean up setup.sh location safely
if [ -f "$BASE_DIR/setup.sh" ]; then
    mv "$BASE_DIR/setup.sh" "$BASE_DIR/_internal/DeepFaceLab/"
fi

echo "-------------------------------------------------------"
echo "FINISHED SUCCESSFULLY"
echo "-------------------------------------------------------"
echo "1. Run 'source ~/.bashrc' to apply new environment paths."
echo "2. Drop your data_src.mp4 and data_dst.mp4 into the workspace folder."
echo "3. You are ready to run your extraction scripts!"
