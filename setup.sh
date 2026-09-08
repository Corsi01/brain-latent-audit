#!/bin/bash
# setup.sh — one-time setup on Leonardo login node.
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source env.sh

echo "1) venv ..."
if command -v uv >/dev/null 2>&1; then
    uv venv .venv
    source .venv/bin/activate
    uv pip install numpy scipy imageio-ffmpeg
else
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install numpy scipy imageio-ffmpeg
fi

echo "2) directories ..."
mkdir -p "$CS_CACHE"/{cache_1hz,shortlist_feats,logs}
mkdir -p "$CS_CODE"/manifest

echo "3) imageio-ffmpeg sanity check ..."
python3 -c "import imageio_ffmpeg; print('ffmpeg binary:', imageio_ffmpeg.get_ffmpeg_exe())"

echo ""
echo "Done. Per ogni sessione:"
echo "  source env.sh && source .venv/bin/activate"
