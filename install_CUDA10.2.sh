#!/bin/bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install --no-cache -r requirements.txt
