# DeepFaceLab-RTX5000 For Vast.AI Linux Cantainer(spacial thanks for users: iperov & volnas10)
### Forked from https://github.com/iperov/DeepFaceLab
This fork also fixes some issues that might occur in the original repository.  
If you don't have any issues with the original DFL, you don't need to use this fork. See [changes and fixes](#new-features-and-fixes) section to see if this version might be of use to you.

## Installation
Tensorflow doesn't support anymore.


1. Open the linux terminal  

3. Run the following command to update the system:
   ```bash
   sudo apt update && sudo apt upgrade -y && pip3 install gdown
   ```
### 2. Install DeepFaceLab-RTX5000-series
Continuing in the Ubuntu terminal, run the following commands.
1. Clone this repository:
   ```bash
   cd ~/Desktop
    ```
   ```bash
   git clone https://github.com/ravindars26/DeepFaceLab-RTX5000.git
    ```
3. Navigate to the cloned repository and run the setup script:
```bash
cd DeepFaceLab-RTX5000 && sudo chmod +x setup.sh && ./setup.sh
```
   OR
   ```bash
   cd DeepFaceLab-RTX5000
   ```
   ```bash
   sudo chmod +x setup.sh
   ```
   ```bash
   ./setup.sh
   ```
5. The setup script will automatically do the following:
    - Recreate the original DFL folder structure
    - Install CUDA and cuDNN
    - Install Python 3.10
    - Download XnView
    - Install all required Python packages
   

You can check if there are any updates to this repository by using the update.sh script:
```bash
cd DeepFaceLab-RTX5000/_internal
sudo chmod +x update.sh
./update.sh
```
Workspace directory
```bash
  mkdir -p workspace/data_dst workspace/data_src/aligned workspace/model
```
##Update:
Over the last years, the main Python ecosystem updated. Packages like numpy and protobuf released brand new versions (like NumPy 2.2.0).
When you downloaded the updated script today, its requirements.txt file was told to fetch these brand-new versions. However, the custom, pre-built TensorFlow wheel inside DeepFaceLab was built to expect slightly older versions.
This created a hard dependency conflict. pip looked at the rules, realized it couldn't satisfy both the new requirements and the old TensorFlow at the same time, and completely crashed.

#Solution
Move to the internal folder and activate the environment
```bash
cd /home/user/Desktop/DeepFaceLab-RTX5000/_internal/DeepFaceLab
python3.10 -m venv venv
source venv/bin/activate
```
Patch NumPy and force-install core requirements
```bash
sed -i 's/numpy.*/numpy==1.26.4/g' requirements.txt
pip install --no-deps -r requirements.txt
```
Install ALL 9 NVIDIA packages together
```bash
pip install nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nvjitlink-cu12
```
Dynamically link all the new paths together
```bash
export LD_LIBRARY_PATH=$(find $PWD/venv/lib/python3.10/site-packages/nvidia -type d -name lib 2>/dev/null | tr '\n' ':')$LD_LIBRARY_PATH
```
Verify TensorFlow can finally see your RTX 5000 GPU
```bash
python -c "import tensorflow as tf; print('\n>>> GPUS FOUND: ', tf.config.list_physical_devices('GPU'))"
```
Back out to the root directory and launch face extraction and training
```
cd ../../
source 'Enter the script here'
#Example: source '5) data_dst faceset extract.sh'
```

This version uses [custom TensorFlow build](https://github.com/weyn9q/rtx5070tensorflow) since the official TensorFlow still doesn't support RTX 5000 GPUs. If I change it to the official build in the future, there is a high chance a fresh install of this repository will be required.


## How to use
Usage is basically the same as in the original DFL. But obviously you will be working in a Linux environment now.
1. You can move your dataset using File Explorer. The workspace folder will be located in:
    ```
    Linux/Ubuntu/home/your_username/DeepFaceLab-RTX5000/workspace
    ```
2. To run the scripts, double-clicking won't work. You need to run them from the terminal. Here are the commands you will use:
   use this video tutorial as refrence: https://youtu.be/zl6Kyr3WosE?si=r7fGUMA7MPFXHQ-w&t=276
   ```bash
   # Change to the directory with scripts:
   cd DeepFaceLab-RTX5000
   # To show the list of all scripts:
   ls *.sh
   # To run a script, copy the full name of the one you want and run using 'source' like this:
   source '4) data_src faceset extract.sh'
   ```
4. If you need to force close the scripts for whatever reason, press `Ctrl + C` in the terminal.  
   If scripts refuse to end (you will see VRAM is still occupied), you can use `wsl --shutdown` command in PowerShell to restart entire WSL.

## Changes and fixes

**Note**: Around 70% GPU usage is completely normal. DFL has quite a lot of CPU overhead and the usage will jump between low and 100% while training.
Task manager then shows the average %, making it seem like the GPU is underutilized.

### Slow faceset extraction (low GPU usage)
Depending on the resolution, you might see low GPU usage. For this reason you can now create more than one GPU session.
The GPU sessions will split all available VRAM between them for every GPU. Try increasing the number in increments of 1 and monitor the speed.

### Manual face extractor
Scrolling now also zooms into the image. This is an opencv behavior that cannot be disabled. Use W/A keys to increase/decrease the square size.

### Random warp causes immediate model collapse
Larger models seem prone to this issue even in the original DFL.
When enabling random warp on a pretrained model, it instantly collapses even with gradient clipping.
For this reason gradient clipping now allows you to set a custom value. If you're experiencing model collapse, try this:
1. Turn on gradient clipping (set the value to 1) and run training
2. If the model still collapses, try a small value like 0.1 with a fresh model and run training
3. If training is stable for a few minutes, back up the model, increase the value to 0.2 and train again
4. If the model collapses again, restore backup and go back to lower values
5. Repeat the process of backing up the model and increasing the value until you can reach 1 again (tested with steps 0.1, 0.2, 0.5 and 1)

## Issues
Only SAEHD workflow has been tested. If you encounter errors, please copy everything from the terminal starting with the script you lunched and provide it in the issue.

### Training freezes or slows down when doing anything else on the PC
You might see iteration time go up if you're doing any other work on your PC.
Sometimes training might crash without any errors and WSL needs to be restarted.
If you encounter these issues **unplug monitor(s) from your GPU and use integrated graphics (if your PC has any) while training instead.**

If issues persist, try running ```nvidia-smi``` comand in Windows terminal.
The second table that appears will show all the processes using the GPU, try shutting them down to see if it helps.

### Small preview windows on high resolution monitors
I couldn't find a working solution for this. If you manage to enable high DPI scaling, please let me know how in the issues.
