#!/bin/bash
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install --no-cache -r requirements.txt
#apt install ffmpeg libsm6 libxext6 -y # cv2 error
