import os
import subprocess

def setupkaggle():
    subprocess.run(['chmod', '+x', 'unet-for-segmentation/scripts/download_data.sh'])
    subprocess.run(['unet-for-segmentation/scripts/download_data.sh'])



