#!/bin/bash

echo "Initializing training procedure";
conda activate envtest
cd gptTroj
# pip install -r requirements.txt
# pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116;
python3 main.py

echo "Finished training";
