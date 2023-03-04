#!/bin/bash

# Create a new Conda environment
conda create --name simsiam_vit python=3.8

# Activate the Conda environment
conda activate simsiam_vit

# Install required packages via pip
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
