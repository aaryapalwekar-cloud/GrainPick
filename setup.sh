#!/bin/bash
# GrainPick setup script
# Run once: bash setup.sh

echo "==> Creating virtual environment..."
python3 -m venv grainpick_env
source grainpick_env/bin/activate

echo "==> Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python-headless numpy Pillow scipy scikit-image matplotlib
pip install git+https://github.com/facebookresearch/segment-anything.git

echo "==> Downloading SAM checkpoint (ViT-H, ~2.5 GB)..."
mkdir -p checkpoints
curl -L -o checkpoints/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

echo ""
echo "Setup complete."
echo "Activate env:  source grainpick_env/bin/activate"
echo "Run tool:      python grainpick.py --image your_thin_section.jpg"

