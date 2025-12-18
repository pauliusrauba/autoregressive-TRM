#!/bin/bash
# Setup script for TRM-LLM project
# Usage: ./setup.sh /path/to/your/data/directory

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== TRM-LLM Project Setup ===${NC}"

# Check if data directory is provided
if [ -z "$1" ]; then
    echo -e "${YELLOW}Usage: ./setup.sh /path/to/your/data/directory${NC}"
    echo -e "${YELLOW}Example: ./setup.sh /mnt/pdata/myusername/icml2025${NC}"
    exit 1
fi

DATA_DIR="$1"

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}UV not found. Installing...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo -e "${GREEN}UV installed!${NC}"
fi

echo -e "${GREEN}Using data directory: ${DATA_DIR}${NC}"

# Create directories on data mount
echo "Creating cache and venv directories..."
mkdir -p "$DATA_DIR/.uv-cache"
mkdir -p "$DATA_DIR/.venv"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Remove existing .venv if it exists (file or directory)
if [ -e "$SCRIPT_DIR/.venv" ] || [ -L "$SCRIPT_DIR/.venv" ]; then
    echo "Removing existing .venv..."
    rm -rf "$SCRIPT_DIR/.venv"
fi

# Create symlink
echo "Creating .venv symlink..."
ln -s "$DATA_DIR/.venv" "$SCRIPT_DIR/.venv"

# Set environment variables for this session
export UV_CACHE_DIR="$DATA_DIR/.uv-cache"
export UV_LINK_MODE=copy

# Run uv sync
echo -e "${GREEN}Installing dependencies (this may take a few minutes)...${NC}"
cd "$SCRIPT_DIR"
uv sync

# Verify installation
echo -e "${GREEN}Verifying installation...${NC}"
uv run python -c "import torch; import pytorch_lightning; print(f'✓ PyTorch {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}'); print(f'✓ PyTorch Lightning {pytorch_lightning.__version__}')"

echo ""
echo -e "${GREEN}=== Setup Complete! ===${NC}"
echo ""
echo "Add these lines to your ~/.bashrc to persist the configuration:"
echo ""
echo -e "${YELLOW}export UV_CACHE_DIR=\"$DATA_DIR/.uv-cache\"${NC}"
echo -e "${YELLOW}export UV_LINK_MODE=copy${NC}"
echo ""
echo "Then run: source ~/.bashrc"
echo ""
echo "To run training:"
echo -e "${YELLOW}uv run python train.py --model gpt --dataset addition_char${NC}"



