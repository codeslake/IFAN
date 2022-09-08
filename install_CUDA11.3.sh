#!/bin/bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3.1 -c pytorch
pip install --no-cache -r requirements.txt
#apt install ffmpeg libsm6 libxext6 -y # cv2 error
