#!/bin/bash

# Create a new Conda environment
conda create -y --name lensiam python=3.8

# Activate the Conda environment
conda activate lensiam

# Install required packages via pip
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
