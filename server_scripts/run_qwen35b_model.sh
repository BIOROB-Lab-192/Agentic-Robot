#!/bin/bash

THINKING=false

for arg in "$@"; do
  if [ "$arg" = "--thinking" ]; then
    THINKING=true
  fi
done

if [ "$THINKING" = true ]; then
  TEMP=1.0
  TOP_P=0.95
  TOP_K=20
  MIN_P=0.0
  PRESENCE_PENALTY=1.5
  THINKING_FLAG='{"enable_thinking":true}'
else
  TEMP=0.7
  TOP_P=0.8
  TOP_K=20
  MIN_P=0.0
  PRESENCE_PENALTY=1.5
  THINKING_FLAG='{"enable_thinking":false}'
fi

ARGS=(
  -m models/Qwen3.6-35B-A3B-UD-Q6_K.gguf
  --mmproj models/mmproj-F32.gguf
  --ctx-size 500000
  --temp "$TEMP"
  --top-p "$TOP_P"
  --top-k "$TOP_K"
  --min-p "$MIN_P"
  --presence-penalty "$PRESENCE_PENALTY"

  # Keep dense / always-used parts on GPU
  --n-gpu-layers 999

  # Keep MoE expert weights on CPU-side memory instead of VRAM
  --cpu-moe

  # Prefer RAM residency instead of disk-backed mmap
  --no-mmap
  --mlock

  --port 8080
  --jinja
  --chat-template-kwargs "$THINKING_FLAG"
)

llama-server "${ARGS[@]}"
