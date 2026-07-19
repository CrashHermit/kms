#!/bin/bash
# Granite Vision 4.1 4B server — start and stop for docling VLM pipeline
# GPU: RTX 3060 Ti (8GB), all 41 layers offloaded

if [ "$1" = "stop" ]; then
    pkill -f "llama-server.bin"
    echo "Server stopped (SIGTERM — CUDA memory freed cleanly)"
    exit 0
fi

PORT="${1:-8080}"
CONTEXT="${2:-4096}"

MODEL_DIR="$HOME/.local/share/ramalama/store/file/home/joshua/models/granite-4b/granite-vision-4.1-4b-Q4_K_M.gguf"
MODEL="$MODEL_DIR/blobs/sha256-641e4fd57b34458347fc2e7679c10818f0aaf6266eb5620c8c3961fa16f65e59"
MMPROJ="$MODEL_DIR/blobs/sha256-573dd2579f6043649299f0b2225000a5691d92f320aabe909fb4c6e75450cad2"

# Kill any existing server gracefully (SIGTERM, not -9 — lets CUDA clean up)
pkill -f "llama-server.bin" 2>/dev/null
sleep 2

env LD_LIBRARY_PATH="$HOME/.local/bin" \
  "$HOME/.local/bin/llama-server.bin" \
  --port "$PORT" \
  --model "$MODEL" \
  --mmproj "$MMPROJ" \
  -ngl 41 -c "$CONTEXT"

echo "Server stopped."
