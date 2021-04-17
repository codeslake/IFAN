#!/bin/bash
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
pip install --no-cache -r requirements.txt

