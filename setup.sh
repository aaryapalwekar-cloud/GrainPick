#!/bin/bash
# GrainPick setup script
# Run once: bash setup.sh

set -e  # stop on any error

echo ""
echo "╔══════════════════════════════════════╗"
echo "║         GrainPick Setup              ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ── Step 1: Virtual environment ──────────────────────────────────────────────
echo "==> [1/4] Creating virtual environment..."
python3 -m venv grainpick_env
source grainpick_env/bin/activate
echo "    ✓ Environment created"

# ── Step 2: Dependencies ─────────────────────────────────────────────────────
echo ""
echo "==> [2/4] Installing dependencies..."
pip install --upgrade pip --quiet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
pip install opencv-python-headless numpy Pillow scipy scikit-image matplotlib --quiet
pip install git+https://github.com/facebookresearch/segment-anything.git --quiet
echo "    ✓ Dependencies installed"

# ── Step 3: SAM checkpoint ───────────────────────────────────────────────────
echo ""
echo "==> [3/4] Checking SAM checkpoint..."
mkdir -p checkpoints

CHECKPOINT="checkpoints/sam_vit_h_4b8939.pth"
EXPECTED_SIZE=2564550024   # expected file size in bytes (~2.5 GB)
URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

if [ -f "$CHECKPOINT" ]; then
    # File exists — check if it's complete (not corrupted or partial)
    ACTUAL_SIZE=$(wc -c < "$CHECKPOINT" | tr -d ' ')
    if [ "$ACTUAL_SIZE" -ge "$EXPECTED_SIZE" ]; then
        echo "    ✓ SAM checkpoint already exists — skipping download"
    else
        echo "    ⚠ Checkpoint found but incomplete ($ACTUAL_SIZE bytes) — redownloading..."
        rm "$CHECKPOINT"
        curl -L --retry 10 --retry-delay 5 -C - \
             --progress-bar \
             -o "$CHECKPOINT" "$URL"
        echo "    ✓ SAM checkpoint downloaded"
    fi
else
    echo "    Downloading SAM checkpoint (~2.5 GB) — this may take a few minutes..."
    echo "    The download will resume automatically if interrupted."
    echo ""
    curl -L --retry 10 --retry-delay 5 -C - \
         --progress-bar \
         -o "$CHECKPOINT" "$URL"
    echo ""
    echo "    ✓ SAM checkpoint downloaded"
fi

# ── Step 4: Done ─────────────────────────────────────────────────────────────
echo ""
echo "==> [4/4] Setup complete!"
echo ""
echo "╔══════════════════════════════════════╗"
echo "║  To launch GrainPick:                ║"
echo "║                                      ║"
echo "║  source grainpick_env/bin/activate   ║"
echo "║  python grainpick_app.py             ║"
echo "╚══════════════════════════════════════╝"
echo ""
